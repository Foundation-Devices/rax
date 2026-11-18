//! The code generator for an AArch64 host.
//!
//! The guest registers r0 to r14 are in the host registers x0 to x14 and the guest flags
//! are the host flags.

use std::collections::HashMap;
use std::mem::offset_of;

use rax::arm::Armv7Cpu;
use rax::arm::Psr;
use rax::arm::cp15::Cp15State;
use rax::arm::decoder::ShiftType;
use rax::arm::vfp;

use super::{
    C, CPSR, Context, DREGS_OFFSET, FPSCR, FpControls, JUMP_BITS, Jump, Links, PAGE_SIZE, REGS,
    Stubs, TLB_ADDEND, TLB_BITS, TLB_DIRECT_PAGE, TLB_ENTRY_SHIFT, Unit, V, always_jumps,
    is_arithmetic, live_flags,
};
use crate::a64::{self, Asm, Label, Reg, ZR};
use crate::{
    Alu, Block, COND_ALWAYS, Carry, Extend, FPSCR_MODES, FpAccumulate, FpBinary, FpConvert,
    FpUnary, Index, Op, PC, SP, SimdLane, SimdLogic, Step, USER_MODE, Width,
};

/// True when the code generator has code for the FP or SIMD operation.
pub(super) fn has_fp_code(_op: Op) -> bool {
    true
}

/// More than the largest code of one instruction, in bytes.
pub(super) const INSTRUCTION_CODE: usize = 1024;

// Host registers. The guest register N is the host register N, for N from 0 to 14.
const T0: Reg = 16;
const T1: Reg = 17;
const T2: Reg = 15;
const T3: Reg = 19;
/// The guest flags during a call.
const SAVED_FLAGS: Reg = 20;
const STATE: Reg = 21;
const CONTEXT: Reg = 22;
const READS: Reg = 23;
const WRITES: Reg = 24;
const JUMPS: Reg = 25;
const BUDGET: Reg = 26;
/// The address of the D registers of the guest.
const DREGS: Reg = 27;
/// Not 0 when FP code ran since the entry: the FPSR of the host has its exceptions.
const FP_USED: Reg = 28;
// Host FP and SIMD registers with no guest value.
const V0: Reg = 0;
const V1: Reg = 1;
const V2: Reg = 2;
const LINK: Reg = 30;

const GUEST_SP: Reg = 13;
const GUEST_LR: Reg = 14;

/// The bits of the FPSCR that the host has in its FPSR: QC and the cumulative exceptions.
const FPSR_STATUS: u32 = 0x0800_009f;

/// A load or a store with a call for the accesses with no direct entry.
struct SlowAccess {
    /// The start of the code with the call, and the place after the instruction.
    start: Label,
    done: Label,
    index: usize,
    load: bool,
    width: Width,
    rt: Reg,
    rn: Reg,
    mode: Index,
    offset: i32,
    address: Reg,
}

/// The carry flag after an operation that writes N and Z from its result.
#[derive(Clone, Copy, PartialEq)]
enum CarryOut {
    Keep,
    Value(bool),
    /// Bit 0 of T2.
    Register,
}

struct Compiler<'a> {
    asm: Asm,
    block: &'a Block,
    /// The virtual address of the page of the block, and its bytes.
    page: u32,
    code: &'a [u8],
    fp: FpControls,
    stubs: Stubs,
    /// The number of the block in the words after a call of `leave_at_instruction`.
    number: u32,
    /// For each instruction, the code that leaves before it.
    leaves: Vec<Option<Label>>,
    slow: Vec<SlowAccess>,
    /// The code of the guest addresses in the physical page of the block.
    entries: &'a HashMap<u32, usize>,
    /// The branches to guest addresses in the page with no code: the address and the
    /// index of the word of the branch.
    links: Links,
    /// A branch to `leave_to_address`, in the range of each conditional branch.
    no_jump: Label,
}

impl Compiler<'_> {
    /// The label of the code that leaves before the instruction `index`.
    fn leave_at(&mut self, index: usize) -> Label {
        if self.leaves[index].is_none() {
            self.leaves[index] = Some(self.asm.label());
        }
        self.leaves[index].unwrap()
    }

    fn give_back(&mut self, instructions: usize) {
        if instructions != 0 {
            self.asm
                .imm12(a64::ADD_IMM | a64::X, BUDGET, BUDGET, instructions as u32);
        }
    }

    /// Goes to the code of the guest address in T0 (T bit in bit 0), after the
    /// instruction `index`.
    fn jump(&mut self, index: usize) {
        self.give_back(self.block.len() - index - 1);
        self.lookup();
    }

    /// Goes to the code that the table of the jumps has for the guest address in T0.
    fn lookup(&mut self) {
        let a = &mut self.asm;
        a.bitfield(a64::UBFM | a64::X | 1 << 22, T1, T0, 1, JUMP_BITS);
        a.shifted(a64::ADD | a64::X, T1, JUMPS, T1, a64::LSL, 4);
        a.mem(a64::LDR, T2, T1, offset_of!(Jump, key));
        a.rrr(a64::EOR, T2, T2, T0);
        a.cbnz(T2, self.no_jump);
        a.mem(a64::LDRX, T1, T1, offset_of!(Jump, code));
        a.br(T1);
    }

    /// Goes to the code of the guest address `key` (T bit in bit 0), after the
    /// instruction `index`.
    fn jump_to(&mut self, index: usize, key: u32) {
        // An address in the page of the block has the code of this physical page while
        // the block runs: a branch can go to it with no look in the table.
        if key & !(PAGE_SIZE - 1) == self.page {
            self.give_back(self.block.len() - index - 1);
            let own = self.page.wrapping_add(self.block.offset(0)) | u32::from(self.block.thumb);
            if key == own {
                self.asm.b_to(self.asm.base());
                return;
            }
            if let Some(&entry) = self.entries.get(&key) {
                self.asm.b_to(entry);
                return;
            }
            // A branch to the next word, until the address has code.
            self.links.push((key, self.asm.words.len()));
            self.asm.emit(a64::B | 1);
            self.asm.mov_imm(T0, key);
            self.lookup();
            return;
        }
        self.asm.mov_imm(T0, key);
        self.jump(index);
    }

    /// Writes N and Z from `result`, and C as `carry` says. V stays.
    fn nz_flags(&mut self, result: Reg, carry: CarryOut, live: u8) {
        let a = &mut self.asm;
        let new_carry = carry != CarryOut::Keep && live & C != 0;
        let kept = live & if carry == CarryOut::Keep { C | V } else { V };
        if kept == 0 {
            match carry {
                // SUBS with 0 sets C and clears V; ANDS clears the two.
                CarryOut::Value(true) if new_carry => return a.imm12(a64::SUBS_IMM, ZR, result, 0),
                CarryOut::Register if new_carry => {}
                _ => return a.rrr(a64::ANDS, ZR, result, result),
            }
        }
        if kept != 0 {
            a.mrs(T0, a64::NZCV);
        }
        a.rrr(a64::ANDS, ZR, result, result);
        a.mrs(T1, a64::NZCV);
        if kept != 0 {
            a.logical(a64::AND_IMM, T0, T0, u32::from(kept) << 28);
            a.rrr(a64::ORR, T1, T1, T0);
        }
        match carry {
            CarryOut::Value(true) if new_carry => {
                a.logical(a64::ORR_IMM, T1, T1, u32::from(C) << 28);
            }
            CarryOut::Register if new_carry => a.shifted(a64::ORR, T1, T1, T2, a64::LSL, 29),
            _ => {}
        }
        a.msr(a64::NZCV, T1);
    }

    #[allow(clippy::too_many_arguments)]
    fn alu_imm(
        &mut self,
        kind: Alu,
        flags: bool,
        carry: Carry,
        rd: Reg,
        rn: Reg,
        imm: u32,
        live: u8,
    ) {
        let set = flags && live != 0;
        if kind.is_compare() && !set {
            return;
        }
        let a = &mut self.asm;
        let rd = if kind.is_compare() && is_arithmetic(kind) {
            ZR
        } else {
            rd
        };
        match kind {
            Alu::Add | Alu::Cmn | Alu::Sub | Alu::Cmp => {
                let (with_imm, with_register) = match (matches!(kind, Alu::Sub | Alu::Cmp), set) {
                    (false, false) => (a64::ADD_IMM, a64::ADD),
                    (false, true) => (a64::ADDS_IMM, a64::ADDS),
                    (true, false) => (a64::SUB_IMM, a64::SUB),
                    (true, true) => (a64::SUBS_IMM, a64::SUBS),
                };
                if imm < 0x1000 {
                    a.imm12(with_imm, rd, rn, imm);
                } else if imm & 0xfff == 0 && imm < 0x100_0000 {
                    a.imm12(with_imm | 1 << 22, rd, rn, imm >> 12);
                } else {
                    a.mov_imm(T1, imm);
                    a.rrr(with_register, rd, rn, T1);
                }
            }
            Alu::Rsb => {
                let subtract = if set { a64::SUBS } else { a64::SUB };
                if imm == 0 {
                    a.rrr(subtract, rd, ZR, rn);
                } else {
                    a.mov_imm(T1, imm);
                    a.rrr(subtract, rd, T1, rn);
                }
            }
            Alu::Adc | Alu::Sbc => {
                a.mov_imm(T1, imm);
                a.rrr(with_carry(kind, set), rd, rn, T1);
            }
            _ => {
                let value = if matches!(kind, Alu::Bic | Alu::Orn | Alu::Mvn) {
                    !imm
                } else {
                    imm
                };
                let result = if kind.is_compare() { T3 } else { rd };
                let (with_imm, with_register) = match kind {
                    Alu::Mov | Alu::Mvn => (0, 0),
                    Alu::And | Alu::Bic | Alu::Tst => (a64::AND_IMM, a64::AND),
                    Alu::Eor | Alu::Teq => (a64::EOR_IMM, a64::EOR),
                    _ => (a64::ORR_IMM, a64::ORR),
                };
                if with_imm == 0 {
                    a.mov_imm(result, value);
                } else if !a.logical(with_imm, result, rn, value) {
                    a.mov_imm(T1, value);
                    a.rrr(with_register, result, rn, T1);
                }
                if set {
                    let carry = match carry {
                        Carry::Keep => CarryOut::Keep,
                        Carry::Clear => CarryOut::Value(false),
                        Carry::Set => CarryOut::Value(true),
                    };
                    self.nz_flags(result, carry, live);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn alu_reg(
        &mut self,
        kind: Alu,
        flags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: ShiftType,
        amount: u32,
        live: u8,
    ) {
        let set = flags && live != 0;
        if kind.is_compare() && !set {
            return;
        }
        let a = &mut self.asm;
        let arithmetic = is_arithmetic(kind);
        let mut carry = CarryOut::Keep;
        if !arithmetic && set && live & C != 0 {
            // The bit of Rm that the shift moves out last.
            let bit = match shift {
                ShiftType::LSL if amount == 0 => None,
                ShiftType::LSL => Some(32 - amount),
                ShiftType::RRX => Some(0),
                _ => Some(amount - 1),
            };
            if let Some(bit) = bit {
                a.ubfx(T2, rm, bit, 1);
                carry = CarryOut::Register;
            }
        }
        let (mut operand, mut host_shift, mut host_amount) = match shift {
            ShiftType::LSL => (rm, a64::LSL, amount),
            ShiftType::LSR if amount == 32 => {
                a.mov_imm(T1, 0);
                (T1, a64::LSL, 0)
            }
            ShiftType::LSR => (rm, a64::LSR, amount),
            ShiftType::ASR => (rm, a64::ASR, amount.min(31)),
            ShiftType::ROR => (rm, a64::ROR, amount),
            ShiftType::RRX => {
                a.cset(T0, a64::CS);
                a.extr(T1, T0, rm, 1);
                (T1, a64::LSL, 0)
            }
        };
        // ADD and SUB have no rotation, and the others take only a register.
        let plain = matches!(kind, Alu::Rsb | Alu::Adc | Alu::Sbc)
            || (arithmetic && host_shift == a64::ROR);
        if plain && host_amount != 0 {
            a.shifted(a64::ORR, T1, ZR, operand, host_shift, host_amount);
            (operand, host_shift, host_amount) = (T1, a64::LSL, 0);
        }
        match kind {
            Alu::Add | Alu::Cmn | Alu::Sub | Alu::Cmp => {
                let op = match (matches!(kind, Alu::Sub | Alu::Cmp), set) {
                    (false, false) => a64::ADD,
                    (false, true) => a64::ADDS,
                    (true, false) => a64::SUB,
                    (true, true) => a64::SUBS,
                };
                let rd = if kind.is_compare() { ZR } else { rd };
                a.shifted(op, rd, rn, operand, host_shift, host_amount);
            }
            Alu::Rsb => a.rrr(if set { a64::SUBS } else { a64::SUB }, rd, operand, rn),
            Alu::Adc | Alu::Sbc => a.rrr(with_carry(kind, set), rd, rn, operand),
            _ => {
                let result = if kind.is_compare() { T3 } else { rd };
                let (op, first) = match kind {
                    Alu::And | Alu::Tst => (a64::AND, rn),
                    Alu::Eor | Alu::Teq => (a64::EOR, rn),
                    Alu::Orr => (a64::ORR, rn),
                    Alu::Bic => (a64::BIC, rn),
                    Alu::Orn => (a64::ORN, rn),
                    Alu::Mov => (a64::ORR, ZR),
                    _ => (a64::ORN, ZR),
                };
                a.shifted(op, result, first, operand, host_shift, host_amount);
                if set {
                    self.nz_flags(result, carry, live);
                }
            }
        }
    }

    /// A shift by the low byte of Rm. The code has only the amounts below 32, and with
    /// the flags only the amounts that are not 0.
    #[allow(clippy::too_many_arguments)]
    fn shift_reg(
        &mut self,
        index: usize,
        shift: ShiftType,
        flags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        live: u8,
    ) {
        let leave = self.leave_at(index);
        let a = &mut self.asm;
        let op = match shift {
            ShiftType::LSL => a64::LSLV,
            ShiftType::LSR => a64::LSRV,
            ShiftType::ASR => a64::ASRV,
            _ => a64::RORV,
        };
        if !(flags && live != 0) {
            if shift != ShiftType::ROR {
                a.logical(a64::AND_IMM, T1, rm, 0xe0);
                a.cbnz(T1, leave);
            }
            return a.rrr(op, rd, rn, rm);
        }
        a.logical(a64::AND_IMM, T1, rm, 0xff);
        a.cbz(T1, leave);
        a.lsr(T0, T1, 5);
        a.cbnz(T0, leave);
        a.imm12(a64::SUB_IMM, T0, T1, 1);
        match shift {
            ShiftType::LSL => {
                a.rrr(a64::LSLV, T2, rn, T0);
                a.lsr(T2, T2, 31);
            }
            ShiftType::LSR | ShiftType::ASR => {
                a.rrr(a64::LSRV, T2, rn, T0);
                a.logical(a64::AND_IMM, T2, T2, 1);
            }
            _ => {}
        }
        a.rrr(op, rd, rn, T1);
        if shift == ShiftType::ROR {
            a.lsr(T2, rd, 31);
        }
        self.nz_flags(rd, CarryOut::Register, live);
    }

    /// Leaves before the instruction `index` when the `size` bytes at the guest address
    /// in `address` are not all in one page with a direct entry. Else T1 gets the
    /// address in the host of the guest address 0.
    fn direct(&mut self, index: usize, address: Reg, size: u32, table: Reg) {
        let leave = self.leave_at(index);
        self.direct_or(leave, address, size, table);
    }

    /// The same, with a branch to `other` in the place of the exit.
    fn direct_or(&mut self, other: Label, address: Reg, size: u32, table: Reg) {
        let leave = other;
        let a = &mut self.asm;
        a.ubfx(T1, address, 12, TLB_BITS);
        a.shifted(a64::ADD | a64::X, T1, table, T1, a64::LSL, TLB_ENTRY_SHIFT);
        a.mem(a64::LDR, T3, T1, TLB_DIRECT_PAGE);
        let mut last = address;
        if size > 1 {
            a.imm12(a64::ADD_IMM, T2, address, size - 1);
            last = T2;
        }
        a.logical(a64::AND_IMM, T2, last, !(PAGE_SIZE - 1));
        a.rrr(a64::EOR, T2, T2, T3);
        a.cbnz(T2, leave);
        a.mem(a64::LDRX, T1, T1, TLB_ADDEND);
    }

    /// Leaves before the instruction `index` when the address is not a multiple of 4.
    fn aligned(&mut self, index: usize, address: Reg) {
        let leave = self.leave_at(index);
        self.asm.logical(a64::AND_IMM, T2, address, 3);
        self.asm.cbnz(T2, leave);
    }

    /// The address of an access with an offset: `rn`, or T0.
    fn address(&mut self, rn: Reg, mode: Index, offset: i32) -> Reg {
        if mode == Index::Post || offset == 0 {
            return rn;
        }
        self.asm.add_imm(T0, rn, offset as u32, T1);
        T0
    }

    /// The new value of the base register, after the access at `address`.
    fn writeback(&mut self, rn: Reg, mode: Index, offset: i32, address: Reg) {
        match mode {
            Index::Offset => {}
            Index::Pre if address != rn => self.asm.mov(rn, address),
            Index::Pre => {}
            Index::Post => self.asm.add_imm(rn, rn, offset as u32, T2),
        }
    }

    /// A load or a store of one register at the guest address in `address`: the access
    /// with a direct entry here, each other access with a call (`slow_access`).
    #[allow(clippy::too_many_arguments)]
    fn access(
        &mut self,
        index: usize,
        load: bool,
        width: Width,
        rt: Reg,
        rn: Reg,
        mode: Index,
        offset: i32,
        address: Reg,
    ) {
        let (op, size) = if load {
            load_of(width)
        } else {
            store_of(width)
        };
        let (start, done) = (self.asm.label(), self.asm.label());
        self.direct_or(start, address, size, if load { READS } else { WRITES });
        self.asm.mem_uxtw(op, rt, T1, address);
        self.writeback(rn, mode, offset, address);
        self.asm.bind(done);
        self.slow.push(SlowAccess {
            start,
            done,
            index,
            load,
            width,
            rt,
            rn,
            mode,
            offset,
            address,
        });
    }

    /// The code of an access with a call.
    fn slow_access(&mut self, access: &SlowAccess) {
        let (retry, stop) = (self.leave_at(access.index), self.leave_at(access.index + 1));
        let rest = (self.block.len() - access.index) as u32;
        let a = &mut self.asm;
        a.bind(access.start);
        if access.address != T0 {
            a.mov(T0, access.address);
        }
        if !access.load {
            a.mov(T1, access.rt);
        }
        let size = if access.load {
            load_of(access.width).1
        } else {
            store_of(access.width).1
        };
        a.mov_imm(T2, size | rest << 8);
        a.bl_to(if access.load {
            self.stubs.call_load
        } else {
            self.stubs.call_store
        });
        a.tbnz(T1, 1, retry);
        if access.load {
            match access.width {
                Width::SignedByte => a.bitfield(a64::SBFM, access.rt, T0, 0, 7),
                Width::SignedHalf => a.bitfield(a64::SBFM, access.rt, T0, 0, 15),
                _ => a.mov(access.rt, T0),
            }
        }
        // T0 has no address now: the new base is from the old one.
        if access.mode != Index::Offset {
            a.add_imm(access.rn, access.rn, access.offset as u32, T2);
        }
        a.cbnz(T1, stop);
        a.b(access.done);
    }

    /// Loads or stores words in sequence at the guest address in `address`. The
    /// address is a multiple of 4, or the code leaves before the instruction `index`.
    fn words(&mut self, index: usize, load: bool, registers: &[Reg], address: Reg) {
        self.aligned(index, address);
        self.direct(
            index,
            address,
            registers.len() as u32 * 4,
            if load { READS } else { WRITES },
        );
        let a = &mut self.asm;
        a.add_uxtw(T1, T1, address);
        for (pair, registers) in registers.chunks(2).enumerate() {
            match *registers {
                [first, second] => a.pair(load, first, second, T1, pair * 8),
                [one] => a.mem(if load { a64::LDR } else { a64::STR }, one, T1, pair * 8),
                _ => unreachable!(),
            }
        }
    }

    /// True when the FP instruction `index` can run with no check of the FP controls.
    fn fp_admitted(&self, index: usize) -> bool {
        self.block.is_admitted(index, self.fp.controls)
    }

    /// True when the host does the FP arithmetic as the guest: its controls are those
    /// of a guest with round to nearest, no flush to zero and no default NaN.
    fn fp_arithmetic(&self) -> bool {
        self.fp.fpscr & FPSCR_MODES == 0
    }

    /// A load or a store of the guest register S`number`, or D`number`, or the Q
    /// register at D`number`.
    fn fp_reg(&mut self, op: a64::Mem, host: Reg, number: u8) {
        let single = op == a64::LDRS || op == a64::STRS || op == a64::LDR || op == a64::STR;
        let offset = if single {
            usize::from(number) * 4
        } else {
            usize::from(number) * 8
        };
        self.asm.mem(op, host, DREGS, offset);
    }

    /// Loads or stores `count` D registers from D`d` at the guest address in `address`.
    #[allow(clippy::too_many_arguments)]
    fn fp_block(&mut self, index: usize, load: bool, d: u8, count: u8, address: Reg) {
        self.direct(
            index,
            address,
            u32::from(count) * 8,
            if load { READS } else { WRITES },
        );
        self.asm.add_uxtw(T1, T1, address);
        for register in 0..count {
            let (guest, host) = (usize::from(d + register) * 8, usize::from(register) * 8);
            if load {
                self.asm.mem(a64::LDRX, T2, T1, host);
                self.asm.mem(a64::STRX, T2, DREGS, guest);
            } else {
                self.asm.mem(a64::LDRX, T2, DREGS, guest);
                self.asm.mem(a64::STRX, T2, T1, host);
            }
        }
    }

    /// Leaves before the instruction `index` when the address is not a multiple of
    /// `align`, a power of 2.
    fn aligned_to(&mut self, index: usize, address: Reg, align: u32) {
        if align > 1 {
            let leave = self.leave_at(index);
            self.asm.logical(a64::AND_IMM, T2, address, align - 1);
            self.asm.cbnz(T2, leave);
        }
    }

    fn simd_lane(&mut self, op: Op) {
        let Op::SimdLane {
            kind,
            size,
            quad,
            unsigned,
            d,
            n,
            m,
            imm,
        } = op
        else {
            unreachable!();
        };
        let element = 8 << size;
        let source_quad = if kind.is_long() {
            false
        } else {
            quad || kind.is_narrow()
        };
        let result_quad = if kind.is_narrow() {
            false
        } else {
            quad || kind.is_long()
        };
        let (load, store) = (
            if source_quad { a64::LDRQ } else { a64::LDRD },
            if result_quad { a64::STRQ } else { a64::STRD },
        );
        if !kind.has_one_operand() {
            self.fp_reg(load, V0, n);
        }
        self.fp_reg(load, V1, m);
        let mut result = V0;
        if matches!(kind, SimdLane::Mla | SimdLane::Mls) {
            self.fp_reg(load, V2, d);
            result = V2;
        }
        let a = &mut self.asm;
        let sized = u32::from(size) << 22;
        // The host has the elements of 64 bits only in registers of 128 bits. The high
        // half of the operands is 0 after a load of 64 bits.
        let quad = if quad || element == 64 { a64::QUAD } else { 0 };
        let sign = if unsigned { a64::UNSIGNED } else { 0 };
        let shift = |amount: u32| (element * 2 - amount) << 16;
        match kind {
            SimdLane::Cgt => a.rrr(a64::V_CMGT | quad | sign | sized, V0, V0, V1),
            SimdLane::Cge => a.rrr(a64::V_CMGE | quad | sign | sized, V0, V0, V1),
            SimdLane::Ceq => a.rrr(a64::V_CMEQ | quad | sized, V0, V0, V1),
            SimdLane::Clt0 => a.rr(a64::V_CMLT_ZERO | quad | sized, V0, V1),
            SimdLane::Add => a.rrr(a64::V_ADD | quad | sized, V0, V0, V1),
            SimdLane::Sub => a.rrr(a64::V_SUB | quad | sized, V0, V0, V1),
            SimdLane::Mul => a.rrr(a64::V_MUL | quad | sized, V0, V0, V1),
            SimdLane::Mla => a.rrr(a64::V_MLA | quad | sized, V2, V0, V1),
            SimdLane::Mls => a.rrr(a64::V_MLS | quad | sized, V2, V0, V1),
            SimdLane::Qadd => a.rrr(a64::V_QADD | quad | sign | sized, V0, V0, V1),
            SimdLane::Qsub => a.rrr(a64::V_QSUB | quad | sign | sized, V0, V0, V1),
            // The host takes the value first and the amounts second.
            SimdLane::Shl => a.rrr(a64::V_SHL | quad | sign | sized, V0, V1, V0),
            SimdLane::Shr => a.rr(a64::V_SHR | quad | sign | shift(imm.into()), V0, V1),
            SimdLane::Rshr => a.rr(a64::V_RSHR | quad | sign | shift(imm.into()), V0, V1),
            SimdLane::Mull => a.rrr(a64::V_MULL | sign | sized, V0, V0, V1),
            SimdLane::Addl => a.rrr(a64::V_ADDL | sign | sized, V0, V0, V1),
            SimdLane::Movl => a.rr(a64::V_SHLL | sign | element << 16, V0, V1),
            SimdLane::Rshrn => a.rr(a64::V_RSHRN | shift(imm.into()), V0, V1),
            SimdLane::Qmovn => a.rr(a64::V_QXTN | sign | sized, V0, V1),
        }
        self.fp_reg(store, result, d);
    }

    /// The code of the FP or SIMD instruction `index`. Returns false, with no code,
    /// when the operation has no compiled form.
    fn fp_op(&mut self, index: usize, step: Step, live: u8) -> bool {
        let pc = self.page.wrapping_add(u32::from(step.offset));
        if !self.fp_admitted(index) {
            return false;
        }
        let arithmetic = self.fp_arithmetic();
        let a = &mut self.asm;
        let precision = |double: bool| if double { a64::DOUBLE } else { 0 };
        let (load_of, store_of) = (
            |double: bool| if double { a64::LDRD } else { a64::LDRS },
            |double: bool| if double { a64::STRD } else { a64::STRS },
        );
        match step.op {
            Op::FpBinary {
                kind,
                double,
                d,
                n,
                m,
            } if arithmetic => {
                self.fp_reg(load_of(double), V0, n);
                self.fp_reg(load_of(double), V1, m);
                let op = match kind {
                    FpBinary::Add => a64::FADD,
                    FpBinary::Sub => a64::FSUB,
                    FpBinary::Mul | FpBinary::Nmul => a64::FMUL,
                    FpBinary::Div => a64::FDIV,
                };
                self.asm.rrr(op | precision(double), V0, V0, V1);
                if kind == FpBinary::Nmul {
                    self.asm.rr(a64::FNEG | precision(double), V0, V0);
                }
                self.fp_reg(store_of(double), V0, d);
            }
            Op::FpAccumulate {
                kind,
                double,
                d,
                n,
                m,
            } if arithmetic => {
                self.fp_reg(load_of(double), V0, n);
                self.fp_reg(load_of(double), V1, m);
                self.fp_reg(load_of(double), V2, d);
                let a = &mut self.asm;
                a.rrr(a64::FMUL | precision(double), V0, V0, V1);
                if matches!(kind, FpAccumulate::Mls | FpAccumulate::Nmla) {
                    a.rr(a64::FNEG | precision(double), V0, V0);
                }
                if matches!(kind, FpAccumulate::Nmla | FpAccumulate::Nmls) {
                    a.rr(a64::FNEG | precision(double), V2, V2);
                }
                a.rrr(a64::FADD | precision(double), V0, V2, V0);
                self.fp_reg(store_of(double), V0, d);
            }
            Op::FpUnary {
                kind: FpUnary::Sqrt,
                ..
            } if !arithmetic => return false,
            Op::FpUnary { kind, double, d, m } => {
                self.fp_reg(load_of(double), V0, m);
                match kind {
                    FpUnary::Mov => {}
                    FpUnary::Abs => self.asm.rr(a64::FABS | precision(double), V0, V0),
                    FpUnary::Neg => self.asm.rr(a64::FNEG | precision(double), V0, V0),
                    FpUnary::Sqrt => self.asm.rr(a64::FSQRT | precision(double), V0, V0),
                }
                self.fp_reg(store_of(double), V0, d);
            }
            Op::FpCompare {
                all_nans_signal,
                with_zero,
                double,
                d,
                m,
            } if arithmetic => {
                self.fp_reg(load_of(double), V0, d);
                let mut op = a64::FCMP | precision(double);
                if all_nans_signal {
                    op |= a64::FCMP_SIGNAL;
                }
                if with_zero {
                    op |= a64::FCMP_ZERO;
                } else {
                    self.fp_reg(load_of(double), V1, m);
                }
                let a = &mut self.asm;
                // The compare writes the host flags, which are the guest flags.
                if live != 0 {
                    a.mrs(T0, a64::NZCV);
                }
                a.rrr(op, 0, V0, if with_zero { 0 } else { V1 });
                a.mrs(T1, a64::NZCV);
                if live != 0 {
                    a.msr(a64::NZCV, T0);
                }
                a.mem(a64::LDR, T2, STATE, FPSCR);
                a.logical(a64::AND_IMM, T2, T2, 0x0fff_ffff);
                a.rrr(a64::ORR, T2, T2, T1);
                a.mem(a64::STR, T2, STATE, FPSCR);
            }
            Op::FpImmediate {
                double: false,
                d,
                imm,
            } => {
                a.mov_imm(T0, vfp::vfp_expand_imm_f32(imm));
                self.fp_reg(a64::STR, T0, d);
            }
            Op::FpImmediate {
                double: true,
                d,
                imm,
            } => {
                a.mov_imm64(T0, vfp::vfp_expand_imm_f64(imm));
                self.fp_reg(a64::STRX, T0, d);
            }
            Op::FpConvert { kind, d, m } if arithmetic => {
                let (to_float, double, unsigned) = match kind {
                    FpConvert::SignedToF32 => (true, false, false),
                    FpConvert::UnsignedToF32 => (true, false, true),
                    FpConvert::SignedToF64 => (true, true, false),
                    FpConvert::UnsignedToF64 => (true, true, true),
                    FpConvert::F32ToSigned => (false, false, false),
                    FpConvert::F32ToUnsigned => (false, false, true),
                    FpConvert::F64ToSigned => (false, true, false),
                    FpConvert::F64ToUnsigned => (false, true, true),
                };
                if to_float {
                    self.fp_reg(a64::LDR, T0, m);
                    let op = if unsigned { a64::UCVTF } else { a64::SCVTF };
                    self.asm.rr(op | precision(double), V0, T0);
                    self.fp_reg(store_of(double), V0, d);
                } else {
                    self.fp_reg(load_of(double), V0, m);
                    let op = if unsigned { a64::FCVTZU } else { a64::FCVTZS };
                    self.asm.rr(op | precision(double), T0, V0);
                    self.fp_reg(a64::STR, T0, d);
                }
            }
            Op::FpFromCore { s, rt } => self.fp_reg(a64::STR, rt, s),
            Op::FpToCore { rt, s } => self.fp_reg(a64::LDR, rt, s),
            Op::FpFromCorePair {
                double: false,
                v,
                rt,
                rt2,
            } => {
                self.fp_reg(a64::STR, rt, v);
                self.fp_reg(a64::STR, rt2, v + 1);
            }
            Op::FpFromCorePair {
                double: true,
                v,
                rt,
                rt2,
            } => {
                a.shifted(a64::ORR | a64::X, T0, rt, rt2, a64::LSL, 32);
                self.fp_reg(a64::STRX, T0, v);
            }
            Op::FpToCorePair {
                double: false,
                rt,
                rt2,
                v,
            } => {
                self.fp_reg(a64::LDR, rt, v);
                self.fp_reg(a64::LDR, rt2, v + 1);
            }
            Op::FpToCorePair {
                double: true,
                rt,
                rt2,
                v,
            } => {
                self.fp_reg(a64::LDRX, T0, v);
                self.split(rt, rt2);
            }
            Op::FpLoad {
                double,
                d,
                rn,
                offset,
            } => {
                let address = self.address(rn, Index::Offset, offset);
                self.aligned(index, address);
                self.direct(index, address, if double { 8 } else { 4 }, READS);
                self.asm
                    .mem_uxtw(if double { a64::LDRX } else { a64::LDR }, T0, T1, address);
                self.fp_reg(if double { a64::STRX } else { a64::STR }, T0, d);
            }
            Op::FpLoadLiteral { double, d, offset } => {
                let address = (pc.wrapping_add(4) & !3).wrapping_add(offset as u32);
                let size = if double { 8 } else { 4 };
                let in_page = address.wrapping_sub(self.page);
                if address & 3 != 0 {
                    return false;
                }
                if in_page <= PAGE_SIZE - size {
                    // A write to the page removes the block, thus the value stays.
                    let bytes = &self.code[in_page as usize..];
                    if double {
                        a.mov_imm64(T0, u64::from_le_bytes(bytes[..8].try_into().unwrap()));
                    } else {
                        a.mov_imm(T0, u32::from_le_bytes(bytes[..4].try_into().unwrap()));
                    }
                } else {
                    a.mov_imm(T0, address);
                    self.direct(index, T0, size, READS);
                    self.asm
                        .mem_uxtw(if double { a64::LDRX } else { a64::LDR }, T0, T1, T0);
                }
                self.fp_reg(if double { a64::STRX } else { a64::STR }, T0, d);
            }
            Op::FpStore {
                double,
                d,
                rn,
                offset,
            } => {
                let address = self.address(rn, Index::Offset, offset);
                self.aligned(index, address);
                self.direct(index, address, if double { 8 } else { 4 }, WRITES);
                self.fp_reg(if double { a64::LDRX } else { a64::LDR }, T2, d);
                self.asm
                    .mem_uxtw(if double { a64::STRX } else { a64::STR }, T2, T1, address);
            }
            Op::FpFlags => {
                a.mem(a64::LDR, T0, STATE, FPSCR);
                a.logical(a64::AND_IMM, T0, T0, 0xf000_0000);
                a.msr(a64::NZCV, T0);
            }
            Op::SimdLogic {
                kind,
                quad,
                d,
                n,
                m,
            } => {
                let (load, store, quad) = if quad {
                    (a64::LDRQ, a64::STRQ, a64::QUAD)
                } else {
                    (a64::LDRD, a64::STRD, 0)
                };
                self.fp_reg(load, V0, n);
                self.fp_reg(load, V1, m);
                let (op, with_destination) = match kind {
                    SimdLogic::Not => (a64::V_NOT, false),
                    SimdLogic::And => (a64::V_AND, false),
                    SimdLogic::Bic => (a64::V_BIC, false),
                    SimdLogic::Orr => (a64::V_ORR, false),
                    SimdLogic::Orn => (a64::V_ORN, false),
                    SimdLogic::Eor => (a64::V_EOR, false),
                    SimdLogic::Bsl => (a64::V_BSL, true),
                    SimdLogic::Bit => (a64::V_BIT, true),
                    SimdLogic::Bif => (a64::V_BIF, true),
                };
                if with_destination {
                    self.fp_reg(load, V2, d);
                    self.asm.rrr(op | quad, V2, V0, V1);
                    self.fp_reg(store, V2, d);
                } else if kind == SimdLogic::Not {
                    self.asm.rr(op | quad, V0, V1);
                    self.fp_reg(store, V0, d);
                } else {
                    self.asm.rrr(op | quad, V0, V0, V1);
                    self.fp_reg(store, V0, d);
                }
            }
            Op::SimdLane { .. } => self.simd_lane(step.op),
            Op::FpElementFromCore {
                d,
                offset,
                size,
                rt,
            } => {
                let op = [a64::STRB, a64::STRH, a64::STR][usize::from(size)];
                a.mem(op, rt, DREGS, usize::from(d) * 8 + usize::from(offset));
            }
            Op::FpElementToCore {
                rt,
                d,
                offset,
                size,
                signed,
            } => {
                let op = match (size, signed) {
                    (0, false) => a64::LDRB,
                    (0, true) => a64::LDRSB,
                    (1, false) => a64::LDRH,
                    (1, true) => a64::LDRSH,
                    _ => a64::LDR,
                };
                a.mem(op, rt, DREGS, usize::from(d) * 8 + usize::from(offset));
            }
            Op::SimdDuplicate {
                quad,
                d,
                size,
                from_core,
                source,
                offset,
            } => {
                let mut element = source;
                if !from_core {
                    let op = [a64::LDRB, a64::LDRH, a64::LDR][usize::from(size)];
                    a.mem(op, T0, DREGS, usize::from(source) * 8 + usize::from(offset));
                    element = T0;
                }
                a.rr(
                    a64::V_DUP | if quad { a64::QUAD } else { 0 } | 0x1_0000 << size,
                    V0,
                    element,
                );
                self.fp_reg(if quad { a64::STRQ } else { a64::STRD }, V0, d);
            }
            Op::SimdStructure {
                load,
                d,
                registers,
                spacing,
                size,
                rn,
                rm,
                align,
            } => {
                self.aligned_to(index, rn, align.into());
                self.direct(
                    index,
                    rn,
                    u32::from(registers) * 8,
                    if load { READS } else { WRITES },
                );
                self.asm.add_uxtw(T1, T1, rn);
                // The host has the same order of the elements, with registers in sequence.
                let op = (if registers == 4 {
                    a64::V_ST4
                } else {
                    a64::V_ST3
                }) | u32::from(size) << 10;
                if load {
                    self.asm.rr(op | a64::V_STRUCTURE_LOAD, V0, T1);
                }
                for number in 0..registers {
                    self.fp_reg(
                        if load { a64::STRD } else { a64::LDRD },
                        V0 + number,
                        d + number * spacing,
                    );
                }
                if !load {
                    self.asm.rr(op, V0, T1);
                }
                match rm {
                    PC => {}
                    SP => self
                        .asm
                        .imm12(a64::ADD_IMM, rn, rn, u32::from(registers) * 8),
                    _ => self.asm.rrr(a64::ADD, rn, rn, rm),
                }
            }
            Op::SimdFloatCompare {
                kind,
                quad,
                d,
                n,
                m,
            } => {
                let (load, store) = if quad {
                    (a64::LDRQ, a64::STRQ)
                } else {
                    (a64::LDRD, a64::STRD)
                };
                self.fp_reg(load, V0, n);
                self.fp_reg(load, V1, m);
                let leave = self.leave_at(index);
                let a = &mut self.asm;
                // V4: the elements of an operand that are a NaN or a denormal number.
                // The 4 elements of the host registers are in the test: the high ones
                // are 0 after a load of a D register.
                let (smallest, infinity, magnitude, first, second, bad) = (6, 7, 2, 3, 5, 4);
                let words = a64::QUAD | 2 << 22;
                a.mov_imm(T0, 0x0080_0000);
                a.rr(a64::V_DUP_WORD, smallest, T0);
                a.mov_imm(T0, 0x7f80_0000);
                a.rr(a64::V_DUP_WORD, infinity, T0);
                for (operand, target) in [(V0, bad), (V1, first)] {
                    a.rr(a64::V_FABS, magnitude, operand);
                    a.rrr(
                        a64::V_CMGT | a64::UNSIGNED | words,
                        target,
                        smallest,
                        magnitude,
                    );
                    a.rrr(a64::V_CMTST | words, second, magnitude, magnitude);
                    a.rrr(a64::V_AND | a64::QUAD, target, target, second);
                    a.rrr(
                        a64::V_CMGT | a64::UNSIGNED | words,
                        second,
                        magnitude,
                        infinity,
                    );
                    a.rrr(a64::V_ORR | a64::QUAD, target, target, second);
                }
                a.rrr(a64::V_ORR | a64::QUAD, bad, bad, first);
                a.rr(a64::V_UMAXV_WORDS, bad, bad);
                a.rr(a64::FMOV_TO_CORE, T0, bad);
                a.cbnz(T0, leave);
                let op = match kind {
                    SimdLane::Ceq => a64::V_FCMEQ,
                    SimdLane::Cge => a64::V_FCMGE,
                    _ => a64::V_FCMGT,
                };
                a.rrr(op | a64::QUAD, V0, V0, V1);
                self.fp_reg(store, V0, d);
            }
            Op::SimdImmediate { quad, d, constant } => {
                a.mov_imm64(T0, self.block.constants[usize::from(constant)]);
                self.fp_reg(a64::STRX, T0, d);
                if quad {
                    self.fp_reg(a64::STRX, T0, d + 1);
                }
            }
            Op::SimdLoad {
                d,
                count,
                rn,
                rm,
                align,
            }
            | Op::SimdStore {
                d,
                count,
                rn,
                rm,
                align,
            } => {
                self.aligned_to(index, rn, align.into());
                self.fp_block(index, matches!(step.op, Op::SimdLoad { .. }), d, count, rn);
                match rm {
                    PC => {}
                    SP => self.asm.imm12(a64::ADD_IMM, rn, rn, u32::from(count) * 8),
                    _ => self.asm.rrr(a64::ADD, rn, rn, rm),
                }
            }
            Op::FpLoadMultiple {
                d,
                count,
                rn,
                decrement,
                writeback,
            }
            | Op::FpStoreMultiple {
                d,
                count,
                rn,
                decrement,
                writeback,
            } => {
                let size = u32::from(count) * 8;
                let mut start = rn;
                if decrement {
                    a.imm12(a64::SUB_IMM, T0, rn, size);
                    start = T0;
                }
                self.aligned(index, start);
                self.fp_block(
                    index,
                    matches!(step.op, Op::FpLoadMultiple { .. }),
                    d,
                    count,
                    start,
                );
                match (writeback, decrement) {
                    (false, _) => {}
                    (true, true) => self.asm.mov(rn, T0),
                    (true, false) => self.asm.imm12(a64::ADD_IMM, rn, rn, size),
                }
            }
            _ => return false,
        }
        true
    }

    /// Leaves before the instruction `index` in User mode.
    fn privileged(&mut self, index: usize) {
        let leave = self.leave_at(index);
        let a = &mut self.asm;
        a.add_imm_x(T1, STATE, CPSR + offset_of!(Psr, mode));
        a.mem(a64::LDRB, T0, T1, 0);
        a.imm12(a64::SUB_IMM, T0, T0, USER_MODE.into());
        a.cbz(T0, leave);
    }

    /// Writes the 64-bit number in T0 to the guest registers `lo` and `hi`.
    fn split(&mut self, lo: Reg, hi: Reg) {
        self.asm.mov(lo, T0);
        self.asm.lsr_x(hi, T0, 32);
    }

    /// The code of the instruction `index`. Returns false, with no code, when the
    /// operation has no compiled form.
    fn op(&mut self, index: usize, step: Step, live: u8) -> bool {
        let pc = self.page.wrapping_add(u32::from(step.offset));
        let thumb = u32::from(self.block.thumb);
        let a = &mut self.asm;
        match step.op {
            Op::Nop | Op::It { .. } => {}
            Op::AluImm {
                kind,
                flags,
                carry,
                rd,
                rn,
                imm,
            } => self.alu_imm(kind, flags, carry, rd, rn, imm, live),
            Op::AluReg {
                kind,
                flags,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                self.alu_reg(kind, flags, rd, rn, rm, shift, amount.into(), live);
            }
            Op::ShiftReg {
                shift,
                flags,
                rd,
                rn,
                rm,
            } => self.shift_reg(index, shift, flags, rd, rn, rm, live),
            Op::Movt { rd, imm } => a.movk_high(rd, imm.into()),
            Op::Adr { rd, offset } => {
                a.mov_imm(rd, (pc.wrapping_add(4) & !3).wrapping_add(offset as u32))
            }
            Op::AddPc { rd } => {
                a.mov_imm(T1, pc.wrapping_add(4));
                a.rrr(a64::ADD, rd, rd, T1);
            }
            Op::Extend {
                kind,
                rd,
                rm,
                rotation,
            } => {
                let mut source = rm;
                if rotation != 0 {
                    a.extr(T1, rm, rm, rotation.into());
                    source = T1;
                }
                let (op, last_bit) = match kind {
                    Extend::Sxth => (a64::SBFM, 15),
                    Extend::Sxtb => (a64::SBFM, 7),
                    Extend::Uxth => (a64::UBFM, 15),
                    Extend::Uxtb => (a64::UBFM, 7),
                };
                a.bitfield(op, rd, source, 0, last_bit);
            }
            Op::Clz { rd, rm } => a.rr(a64::CLZ, rd, rm),
            Op::Rbit { rd, rm } => a.rr(a64::RBIT, rd, rm),
            Op::Rev { rd, rm } => a.rr(a64::REV, rd, rm),
            Op::Ubfx { rd, rn, lsb, width } => a.ubfx(rd, rn, lsb.into(), width.into()),
            Op::Sbfx { rd, rn, lsb, width } => {
                a.bitfield(a64::SBFM, rd, rn, lsb.into(), u32::from(lsb + width) - 1);
            }
            Op::Bfi { rd, rn, lsb, width } => {
                a.bitfield(
                    a64::BFM,
                    rd,
                    rn,
                    (32 - u32::from(lsb)) % 32,
                    u32::from(width) - 1,
                );
            }
            Op::Bfc { rd, lsb, width } => {
                a.bitfield(
                    a64::BFM,
                    rd,
                    ZR,
                    (32 - u32::from(lsb)) % 32,
                    u32::from(width) - 1,
                );
            }
            Op::Mul { flags, rd, rn, rm } => {
                a.multiply(a64::MADD, rd, rn, rm, ZR);
                if flags && live != 0 {
                    self.nz_flags(rd, CarryOut::Keep, live);
                }
            }
            Op::Mla { rd, rn, rm, ra } => a.multiply(a64::MADD, rd, rn, rm, ra),
            Op::Mls { rd, rn, rm, ra } => a.multiply(a64::MSUB, rd, rn, rm, ra),
            Op::Umull { lo, hi, rn, rm } => {
                a.multiply(a64::UMADDL, T0, rn, rm, ZR);
                self.split(lo, hi);
            }
            Op::Umlal { lo, hi, rn, rm } => {
                a.shifted(a64::ORR | a64::X, T1, lo, hi, a64::LSL, 32);
                a.multiply(a64::UMADDL, T0, rn, rm, T1);
                self.split(lo, hi);
            }
            Op::Umaal { lo, hi, rn, rm } => {
                a.multiply(a64::UMADDL, T0, rn, rm, lo);
                a.rrr(a64::ADD | a64::X, T0, T0, hi);
                self.split(lo, hi);
            }
            Op::Smull { lo, hi, rn, rm } => {
                a.multiply(a64::SMADDL, T0, rn, rm, ZR);
                self.split(lo, hi);
            }
            Op::Udiv { rd, rn, rm } => a.rrr(a64::UDIV, rd, rn, rm),
            Op::Sdiv { rd, rn, rm } => a.rrr(a64::SDIV, rd, rn, rm),
            Op::Load {
                width,
                rt,
                rn,
                index: mode,
                offset,
            } => {
                let address = self.address(rn, mode, offset);
                self.access(index, true, width, rt, rn, mode, offset, address);
            }
            Op::Store {
                width,
                rt,
                rn,
                index: mode,
                offset,
            } => {
                let address = self.address(rn, mode, offset);
                self.access(index, false, width, rt, rn, mode, offset, address);
            }
            Op::LoadReg {
                width,
                rt,
                rn,
                rm,
                shift,
            } => {
                a.shifted(a64::ADD, T0, rn, rm, a64::LSL, shift.into());
                self.access(index, true, width, rt, rn, Index::Offset, 0, T0);
            }
            Op::StoreReg {
                width,
                rt,
                rn,
                rm,
                shift,
            } => {
                a.shifted(a64::ADD, T0, rn, rm, a64::LSL, shift.into());
                self.access(index, false, width, rt, rn, Index::Offset, 0, T0);
            }
            Op::LoadLiteral { width, rt, offset } => {
                let address = (pc.wrapping_add(4) & !3).wrapping_add(offset as u32);
                let (op, size) = load_of(width);
                let in_page = address.wrapping_sub(self.page);
                if in_page <= PAGE_SIZE - size {
                    // A write to the page removes the block, thus the value stays.
                    let bytes = &self.code[in_page as usize..];
                    let value = match width {
                        Width::Byte => bytes[0].into(),
                        Width::SignedByte => bytes[0] as i8 as u32,
                        Width::Half => u16::from_le_bytes([bytes[0], bytes[1]]).into(),
                        Width::SignedHalf => i16::from_le_bytes([bytes[0], bytes[1]]) as u32,
                        Width::Word => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    };
                    a.mov_imm(rt, value);
                } else {
                    a.mov_imm(T0, address);
                    self.direct(index, T0, size, READS);
                    self.asm.mem_uxtw(op, rt, T1, T0);
                }
            }
            Op::LoadPair {
                rt,
                rt2,
                rn,
                index: mode,
                offset,
            } => {
                let address = self.address(rn, mode, offset);
                self.words(index, true, &[rt, rt2], address);
                self.writeback(rn, mode, offset, address);
            }
            Op::StorePair {
                rt,
                rt2,
                rn,
                index: mode,
                offset,
            } => {
                let address = self.address(rn, mode, offset);
                self.words(index, false, &[rt, rt2], address);
                self.writeback(rn, mode, offset, address);
            }
            Op::LoadMultiple {
                rn,
                mask,
                writeback,
            }
            | Op::StoreMultiple {
                rn,
                mask,
                writeback,
            } => {
                let registers: Vec<Reg> = (0..15)
                    .filter(|register| mask & 1 << register != 0)
                    .collect();
                self.words(
                    index,
                    matches!(step.op, Op::LoadMultiple { .. }),
                    &registers,
                    rn,
                );
                if writeback {
                    self.asm.add_imm(rn, rn, registers.len() as u32 * 4, T2);
                }
            }
            Op::Push { mask } => {
                let registers: Vec<Reg> = (0..15)
                    .filter(|register| mask & 1 << register != 0)
                    .collect();
                a.imm12(a64::SUB_IMM, T0, GUEST_SP, registers.len() as u32 * 4);
                self.words(index, false, &registers, T0);
                self.asm.mov(GUEST_SP, T0);
            }
            Op::Pop { mask } => {
                // The value of the PC goes to T0.
                let registers: Vec<Reg> = (0..16)
                    .filter(|register| mask & 1 << register != 0)
                    .map(|register| if register == PC { T0 } else { register })
                    .collect();
                self.words(index, true, &registers, GUEST_SP);
                self.asm
                    .imm12(a64::ADD_IMM, GUEST_SP, GUEST_SP, registers.len() as u32 * 4);
                if mask & 1 << PC != 0 {
                    self.jump(index);
                }
            }
            Op::TableBranch { rn, rm, halfword } => {
                let shift = u32::from(halfword);
                if rn == PC {
                    a.mov_imm(T0, pc.wrapping_add(4));
                    a.shifted(a64::ADD, T0, T0, rm, a64::LSL, shift);
                } else {
                    a.shifted(a64::ADD, T0, rn, rm, a64::LSL, shift);
                }
                self.direct(index, T0, 1 << shift, READS);
                let a = &mut self.asm;
                a.mem_uxtw(if halfword { a64::LDRH } else { a64::LDRB }, T0, T1, T0);
                a.mov_imm(T1, pc.wrapping_add(4) | 1);
                a.shifted(a64::ADD, T0, T1, T0, a64::LSL, 1);
                self.jump(index);
            }
            Op::Branch { offset } => self.jump_to(
                index,
                pc.wrapping_add(4).wrapping_add(offset as u32) | thumb,
            ),
            Op::BranchLink { offset } => {
                a.mov_imm(GUEST_LR, pc.wrapping_add(4) | 1);
                self.jump_to(index, pc.wrapping_add(4).wrapping_add(offset as u32) | 1);
            }
            Op::BranchLinkArm { offset } => {
                a.mov_imm(GUEST_LR, pc.wrapping_add(4));
                self.jump_to(index, pc.wrapping_add(4).wrapping_add(offset as u32));
            }
            Op::BranchLinkToArm { offset } => {
                a.mov_imm(GUEST_LR, pc.wrapping_add(4) | 1);
                self.jump_to(index, (pc.wrapping_add(4) & !3).wrapping_add(offset as u32));
            }
            Op::CompareBranch {
                rn,
                nonzero,
                offset,
            } => {
                let next = a.label();
                if nonzero {
                    a.cbz(rn, next);
                } else {
                    a.cbnz(rn, next);
                }
                self.jump_to(index, pc.wrapping_add(4 + u32::from(offset)) | 1);
                self.asm.bind(next);
            }
            Op::BranchExchange { rm } => {
                a.mov(T0, rm);
                self.jump(index);
            }
            Op::BranchLinkExchange { rm } => {
                a.mov(T0, rm);
                a.mov_imm(GUEST_LR, pc.wrapping_add(2) | 1);
                self.jump(index);
            }
            Op::CacheOperation => self.privileged(index),
            Op::ThreadRegister { rt, register } => {
                if register == 2 {
                    self.privileged(index);
                }
                let field = [
                    offset_of!(Cp15State, tpidrurw),
                    offset_of!(Cp15State, tpidruro),
                    offset_of!(Cp15State, tpidrprw),
                ];
                self.asm.add_imm_x(
                    T1,
                    STATE,
                    offset_of!(Armv7Cpu, cp15) + field[usize::from(register)],
                );
                self.asm.mem(a64::LDR, rt, T1, 0);
            }
            Op::MulHalves {
                rd,
                rn,
                rm,
                n_high,
                m_high,
            } => {
                for (target, source, high) in [(T0, rn, n_high), (T1, rm, m_high)] {
                    let (first_bit, last_bit) = if high { (16, 31) } else { (0, 15) };
                    a.bitfield(a64::SBFM, target, source, first_bit, last_bit);
                }
                a.multiply(a64::MADD, rd, T0, T1, ZR);
            }
            op if op.is_fp() => return self.fp_op(index, step, live),
            _ => return false,
        }
        true
    }

    /// True when the instruction `index` has a compiled form in each state of the guest.
    /// Without that, the code can stop before the instruction.
    fn always_has_code(&self, index: usize) -> bool {
        let op = self.block.steps[index].op;
        op != Op::Slow && !op.is_fp()
    }

    /// Returns the code and its branches to addresses of the page with no code (the
    /// address and the index of the word), or None when the first instruction has no
    /// compiled form.
    fn compile(mut self) -> Option<(Vec<u32>, Links)> {
        let block = self.block;
        let steps = &block.steps;
        let live = live_flags(steps, |index| self.always_has_code(index));
        let over_budget = self.asm.label();
        self.asm
            .imm12(a64::SUB_IMM | a64::X, BUDGET, BUDGET, steps.len() as u32);
        self.asm.tbnz(BUDGET, 63, over_budget);
        if (0..steps.len()).any(|index| steps[index].op.is_fp() && self.fp_admitted(index)) {
            // The FP code is for one state of the FP controls.
            let leave = self.leave_at(0);
            let a = &mut self.asm;
            a.mem(a64::LDRX, T0, CONTEXT, offset_of!(Context, fp_controls));
            a.mov_imm64(T1, self.fp.controls);
            a.rrr(a64::EOR | a64::X, T0, T0, T1);
            a.cbnz_x(T0, leave);
            // A write of the FPSR is slow in the host: only the first FP block of a
            // run makes it empty.
            let used = a.label();
            a.cbnz_x(FP_USED, used);
            a.msr(a64::FPSR, ZR);
            a.mov_imm(FP_USED, 1);
            a.bind(used);
        }
        let mut compiled = steps.len();
        for (index, step) in steps.iter().enumerate() {
            let next = self.asm.label();
            // The condition 15 is always true, as 14.
            if step.cond < COND_ALWAYS {
                self.asm.b_cond(u32::from(step.cond ^ 1), next);
            }
            if !self.op(index, *step, live[index]) {
                // The interpreter runs the instruction. The code after it is for a
                // condition that is false.
                let leave = self.leave_at(index);
                self.asm.b(leave);
                if step.cond >= COND_ALWAYS {
                    compiled = index;
                    break;
                }
            }
            self.asm.bind(next);
        }
        if compiled == 0 {
            return None;
        }
        // The end of the code in sequence: the instruction with no compiled form, or the
        // address after the block.
        let thumb = u32::from(block.thumb);
        let last = steps[compiled - 1];
        if compiled < steps.len() {
            // The code ended with the exit before the instruction `compiled`.
        } else if block.end_it_state != 0 {
            let leave = self.leave_at(compiled);
            self.asm.b(leave);
        } else if last.cond != COND_ALWAYS || !always_jumps(last.op) {
            self.jump_to(
                compiled - 1,
                self.page.wrapping_add(block.end_offset.into()) | thumb,
            );
        }
        self.asm.bind(over_budget);
        self.give_back(steps.len());
        self.asm
            .mov_imm(T0, self.page.wrapping_add(block.offset(0)) | thumb);
        self.asm.bind(self.no_jump);
        self.asm.b_to(self.stubs.leave_to_address);
        for access in std::mem::take(&mut self.slow) {
            self.slow_access(&access);
        }
        for index in 0..self.leaves.len() {
            if let Some(label) = self.leaves[index] {
                self.asm.bind(label);
                self.give_back(steps.len() - index);
                self.asm.bl_to(self.stubs.leave_at_instruction);
                self.asm.emit(self.number);
                self.asm.emit(index as u32);
            }
        }
        self.asm.finish();
        Some((self.asm.words, self.links))
    }
}

/// The host load of a guest load, and the size of the access.
fn load_of(width: Width) -> (a64::Mem, u32) {
    match width {
        Width::Byte => (a64::LDRB, 1),
        Width::SignedByte => (a64::LDRSB, 1),
        Width::Half => (a64::LDRH, 2),
        Width::SignedHalf => (a64::LDRSH, 2),
        Width::Word => (a64::LDR, 4),
    }
}

fn store_of(width: Width) -> (a64::Mem, u32) {
    match width {
        Width::Byte | Width::SignedByte => (a64::STRB, 1),
        Width::Half | Width::SignedHalf => (a64::STRH, 2),
        Width::Word => (a64::STR, 4),
    }
}

fn with_carry(kind: Alu, set: bool) -> u32 {
    match (kind, set) {
        (Alu::Adc, false) => a64::ADC,
        (Alu::Adc, true) => a64::ADCS,
        (_, false) => a64::SBC,
        (_, true) => a64::SBCS,
    }
}

/// Adds the exceptions in the FPSR of the host to the FPSCR of the guest.
fn fpsr_to_fpscr(a: &mut Asm, first: Reg, second: Reg) {
    a.mrs(first, a64::FPSR);
    a.mov_imm(second, FPSR_STATUS);
    a.rrr(a64::AND, first, first, second);
    a.mem(a64::LDR, second, STATE, FPSCR);
    a.rrr(a64::ORR, second, second, first);
    a.mem(a64::STR, second, STATE, FPSCR);
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// Returns the code of the block and its branches to addresses of the page with no code
/// (the address and the offset of the branch), or None when the first instruction has no
/// compiled form.
pub(super) fn compile(unit: Unit) -> Option<(Vec<u8>, Links)> {
    let mut asm = Asm::new(unit.entry);
    let compiler = Compiler {
        entries: unit.entries,
        links: Vec::new(),
        no_jump: asm.label(),
        asm,
        block: unit.block,
        page: unit.page,
        code: unit.code,
        fp: unit.fp,
        stubs: unit.stubs,
        number: unit.number,
        leaves: vec![None; unit.block.len() + 1],
        slow: Vec::new(),
    };
    let (words, links) = compiler.compile()?;
    let links = links.into_iter().map(|(key, word)| (key, word * 4));
    Some((bytes(&words), links.collect()))
}

/// The address and the bytes of a branch to `target` at `site`.
pub(super) fn link(site: usize, target: usize) -> (usize, [u8; 4]) {
    let word = a64::B | ((target as i64 - site as i64) / 4) as u32 & 0x03ff_ffff;
    (site, word.to_le_bytes())
}

/// The code that goes from the caller to compiled code and back.
pub(super) fn stubs(base: usize) -> (Vec<u8>, Stubs) {
    const FRAME: [(Reg, Reg); 5] = [(19, 20), (21, 22), (23, 24), (25, 26), (27, 28)];
    let mut a = Asm::new(base);
    let enter = a.here();
    // stp x29, x30, [sp, #-96]!
    a.emit(0xa9ba_7bfd);
    for (index, (first, second)) in FRAME.into_iter().enumerate() {
        a.pair_x(false, first, second, a64::SP, 16 + index * 16);
    }
    a.mov_x(CONTEXT, 0);
    a.mov_x(T1, 1);
    a.mem(a64::LDRX, STATE, CONTEXT, offset_of!(Context, cpu));
    a.mem(a64::LDRX, JUMPS, CONTEXT, offset_of!(Context, jumps));
    a.mem(a64::LDRX, READS, CONTEXT, offset_of!(Context, reads));
    a.mem(a64::LDRX, WRITES, CONTEXT, offset_of!(Context, writes));
    a.mem(a64::LDRX, BUDGET, CONTEXT, offset_of!(Context, budget));
    a.add_imm_x(DREGS, STATE, DREGS_OFFSET);
    a.mov_imm(FP_USED, 0);
    a.add_imm_x(T2, STATE, CPSR);
    a.mem(a64::LDRB, T0, T2, offset_of!(Psr, n));
    a.lsl(T0, T0, 31);
    for (offset, bit) in [
        (offset_of!(Psr, z), 30),
        (offset_of!(Psr, c), 29),
        (offset_of!(Psr, v), 28),
    ] {
        a.mem(a64::LDRB, T3, T2, offset);
        a.shifted(a64::ORR, T0, T0, T3, a64::LSL, bit);
    }
    a.msr(a64::NZCV, T0);
    a.add_imm_x(T0, STATE, REGS);
    for register in (0..14).step_by(2) {
        a.pair(true, register, register + 1, T0, usize::from(register) * 4);
    }
    a.mem(a64::LDR, GUEST_LR, T0, 56);
    a.br(T1);

    let leave_to_address = a.here();
    let leave = a.label();
    a.add_imm_x(T1, STATE, REGS);
    a.logical(a64::AND_IMM, T2, T0, !1);
    a.mem(a64::STR, T2, T1, 60);
    a.add_imm_x(T1, STATE, CPSR);
    a.logical(a64::AND_IMM, T2, T0, 1);
    a.mem(a64::STRB, T2, T1, offset_of!(Psr, t));
    a.mem(a64::STRX, ZR, CONTEXT, offset_of!(Context, site));
    a.b(leave);

    let leave_at_instruction = a.here();
    a.mem(a64::STRX, LINK, CONTEXT, offset_of!(Context, site));
    a.bind(leave);
    a.add_imm_x(T1, STATE, REGS);
    for register in (0..14).step_by(2) {
        a.pair(false, register, register + 1, T1, usize::from(register) * 4);
    }
    a.mem(a64::STR, GUEST_LR, T1, 56);
    a.mrs(T0, a64::NZCV);
    a.add_imm_x(T1, STATE, CPSR);
    for (offset, bit) in [
        (offset_of!(Psr, n), 31),
        (offset_of!(Psr, z), 30),
        (offset_of!(Psr, c), 29),
        (offset_of!(Psr, v), 28),
    ] {
        a.ubfx(T2, T0, bit, 1);
        a.mem(a64::STRB, T2, T1, offset);
    }
    a.mem(a64::STRX, BUDGET, CONTEXT, offset_of!(Context, budget));
    let no_fp = a.label();
    a.cbz(FP_USED, no_fp);
    fpsr_to_fpscr(&mut a, T0, T2);
    a.bind(no_fp);
    for (index, (first, second)) in FRAME.into_iter().enumerate() {
        a.pair_x(true, first, second, a64::SP, 16 + index * 16);
    }
    // ldp x29, x30, [sp], #96
    a.emit(0xa8c6_7bfd);
    a.ret();

    let mut call = |store: bool| {
        let start = a.here();
        a.mem(a64::STRX, BUDGET, CONTEXT, offset_of!(Context, budget));
        a.add_imm_x(T3, STATE, REGS);
        for register in (0..14).step_by(2) {
            a.pair(false, register, register + 1, T3, usize::from(register) * 4);
        }
        a.mem(a64::STR, GUEST_LR, T3, 56);
        // The registers 19 and 20 stay in a call, and the guest registers are free now.
        a.mrs(SAVED_FLAGS, a64::NZCV);
        a.mov_x(T3, LINK);
        let no_fp = a.label();
        a.cbz(FP_USED, no_fp);
        fpsr_to_fpscr(&mut a, 8, 9);
        a.bind(no_fp);
        a.mov_x(0, CONTEXT);
        a.mov(1, T0);
        if store {
            a.mov(2, T1);
            a.mov(3, T2);
            a.mem(a64::LDRX, 8, CONTEXT, offset_of!(Context, store));
        } else {
            a.mov(2, T2);
            a.mem(a64::LDRX, 8, CONTEXT, offset_of!(Context, load));
        }
        a.blr(8);
        a.mov(T0, 0);
        a.lsr_x(T1, 0, 32);
        // The function can change the FPSR of the host.
        let no_fp = a.label();
        a.cbz(FP_USED, no_fp);
        a.msr(a64::FPSR, ZR);
        a.bind(no_fp);
        a.add_imm_x(T2, STATE, REGS);
        for register in (0..14).step_by(2) {
            a.pair(true, register, register + 1, T2, usize::from(register) * 4);
        }
        a.mem(a64::LDR, GUEST_LR, T2, 56);
        a.msr(a64::NZCV, SAVED_FLAGS);
        a.mem(a64::LDRX, BUDGET, CONTEXT, offset_of!(Context, budget));
        a.br(T3);
        start
    };
    let (call_load, call_store) = (call(false), call(true));
    a.finish();
    (
        bytes(&a.words),
        Stubs {
            enter,
            leave_to_address,
            leave_at_instruction,
            call_load,
            call_store,
        },
    )
}
