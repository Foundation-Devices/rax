//! The code generator for an x86-64 host with the System V calling convention.
//!
//! Nine guest registers are in host registers (`HOST`) and the others are in the CPU state.
//! The guest flags are in RAX: N, Z and C in the bits 7, 6 and 0 of AH, where LAHF and
//! SAHF have SF, ZF and CF, and V in AL. The flags of the host are a copy of a part of
//! them only from the instruction that made them to the next instruction that changes
//! the flags of the host (`Compiler::host_flags`).
//!
//! The FP arithmetic is that of SSE, only for a result that is sure to be the result of
//! the guest with its flags: see `Compiler::fp_guard`.

use std::collections::HashMap;
use std::mem::offset_of;

use rax::arm::Armv7Cpu;
use rax::arm::Psr;
use rax::arm::cp15::Cp15State;
use rax::arm::decoder::ShiftType;
use rax::arm::vfp;

use super::{
    ALL_FLAGS, C, CPSR, Context, DREGS_OFFSET, FPSCR, FpControls, JUMP_BITS, Jump, Links, N,
    PAGE_SIZE, REGS, Stubs, TLB_ADDEND, TLB_BITS, TLB_DIRECT_PAGE, TLB_ENTRY_SHIFT, Unit, V, Z,
    always_jumps, condition_flags, is_arithmetic, live_flags,
};
use crate::x64::{
    self, Asm, Byte, Label, Long, Quad, R8, R9, R10, R11, R12, R13, R14, R15, RAX, RBP, RBX, RCX,
    RDI, RDX, RSI, RSP, Reg, Rm, Shift, Sse, Unary, Word, indexed, mem,
};
use crate::{
    Alu, Block, COND_ALWAYS, Carry, Extend, FPSCR_MODES, FpAccumulate, FpBinary, FpConvert,
    FpUnary, Index, Op, PC, SP, SimdLane, SimdLogic, Step, USER_MODE, Width,
};

/// More than the largest code of one instruction, in bytes.
pub(super) const INSTRUCTION_CODE: usize = 1024;

const FLAGS: Reg = RAX;
const T0: Reg = RBX;
const T1: Reg = RDX;
/// CL has the amount of a shift by a register.
const T2: Reg = RCX;
const STATE: Reg = RBP;
/// The instructions that can run. A register, or else each change of the count in a
/// block waits for the store of the change before it.
const BUDGET: Reg = R14;
/// The host register of each guest register that has one: r0 to r6, r8 and the SP are the
/// registers of the most operands of the compiled blocks of a KeyOS boot.
const HOST: [Option<Reg>; 15] = [
    Some(RSI),
    Some(RDI),
    Some(R8),
    Some(R9),
    Some(R10),
    Some(R11),
    Some(R12),
    None,
    Some(R13),
    None,
    None,
    None,
    None,
    Some(R15),
    None,
];
const GUEST_SP: u8 = 13;
const GUEST_LR: u8 = 14;

// The stack of the compiled code, from RSP. RSP is a multiple of 16 there.
const CONTEXT: usize = 0;
const READS: usize = 8;
const WRITES: usize = 16;
const JUMPS: usize = 24;
/// The guest flags during a call.
const SAVED_FLAGS: usize = 32;
/// 32 bits that are not 0 when FP arithmetic ran since the entry: the MXCSR has its
/// exceptions.
const FP_USED: usize = 40;
/// 32 bits for a read or a write of the MXCSR.
const MXCSR: usize = 44;
/// With the return address and the six saved registers, a multiple of 16.
const FRAME: u32 = 56;

/// Round to nearest, no flush to zero, no trap, no exception.
pub(super) const MXCSR_DEFAULT: u32 = 0x1f80;
/// Division by zero, overflow and inexact in the FPSCR: the bits of the MXCSR one place
/// to the right.
const FPSCR_FROM_MXCSR: u32 = 0x16;
// Host FP registers with no guest value.
const X0: Reg = 0;
const X1: Reg = 1;
const X2: Reg = 2;

/// The bit of C in RAX.
const FLAGS_C: u32 = 8;

/// A load or a store with a call for the accesses with no direct entry.
struct SlowAccess {
    /// The start of the code with the call, and the place after the instruction.
    start: Label,
    done: Label,
    index: usize,
    load: bool,
    width: Width,
    rt: u8,
    rn: u8,
    mode: Index,
    offset: i32,
    address: Reg,
}

/// A product or a quotient that is 0: the code that looks at the operands.
struct ZeroResult {
    start: Label,
    done: Label,
    index: usize,
    double: bool,
    /// The register with the second operand of a product; a quotient has none.
    second: Option<Reg>,
    /// The guest register of the first operand.
    first: u8,
}

/// The results of an FP operation that are 0 with an underflow.
#[derive(Clone, Copy, PartialEq)]
enum Zero {
    /// A sum, a difference or a square root that is 0 has no rounding.
    Exact,
    Product,
    Quotient,
}

/// The carry flag after an operation that writes N and Z from its result.
#[derive(Clone, Copy, PartialEq)]
enum CarryOut {
    Keep,
    Value(bool),
    /// CL, which is 0 or 1.
    Register,
}

/// The second operand of an operation.
#[derive(Clone, Copy, PartialEq)]
enum Source {
    Operand(Rm),
    Immediate(u32),
}

fn guest(register: u8) -> Rm {
    match HOST[usize::from(register)] {
        Some(host) => Rm::Reg(host),
        None => mem(STATE, REGS + usize::from(register) * 4),
    }
}

/// The guest register S`number`, and D`number`, in the CPU state.
fn sreg(number: u8) -> Rm {
    mem(STATE, DREGS_OFFSET + usize::from(number) * 4)
}

fn dreg(number: u8) -> Rm {
    mem(STATE, DREGS_OFFSET + usize::from(number) * 8)
}

/// The guest register S`number`, or D`number` for `double`.
fn fp_reg(double: bool, number: u8) -> Rm {
    if double { dreg(number) } else { sreg(number) }
}

/// True when the operation is FP arithmetic of the host.
fn is_fp_arithmetic(op: Op) -> bool {
    matches!(
        op,
        Op::FpBinary { .. }
            | Op::FpAccumulate { .. }
            | Op::FpUnary {
                kind: FpUnary::Sqrt,
                ..
            }
            | Op::FpCompare { .. }
            | Op::FpConvert { .. }
    )
}

/// True when SSE2 has the operation for the elements of 8 << `size` bits.
fn has_lane_code(kind: SimdLane, size: u8, unsigned: bool) -> bool {
    match kind {
        SimdLane::Add | SimdLane::Sub => true,
        SimdLane::Ceq | SimdLane::Cgt => size < 3,
        SimdLane::Shr => unsigned || size == 1 || size == 2,
        _ => false,
    }
}

/// True when the code generator has code for the FP or SIMD operation.
pub(super) fn has_fp_code(op: Op) -> bool {
    match op {
        op if is_fp_arithmetic(op) => true,
        Op::SimdLane {
            kind,
            size,
            unsigned,
            ..
        } => has_lane_code(kind, size, unsigned),
        Op::SimdFloatCompare { .. } => true,
        Op::FpUnary { .. }
        | Op::FpImmediate { .. }
        | Op::FpFromCore { .. }
        | Op::FpToCore { .. }
        | Op::FpFromCorePair { .. }
        | Op::FpToCorePair { .. }
        | Op::FpLoad { .. }
        | Op::FpLoadLiteral { .. }
        | Op::FpStore { .. }
        | Op::FpFlags
        | Op::SimdLogic { .. }
        | Op::SimdImmediate { .. }
        | Op::SimdLoad { .. }
        | Op::SimdStore { .. }
        | Op::FpElementFromCore { .. }
        | Op::FpElementToCore { .. }
        | Op::SimdDuplicate { .. }
        | Op::FpLoadMultiple { .. }
        | Op::FpStoreMultiple { .. } => true,
        _ => false,
    }
}

fn low_bits(width: u8) -> u32 {
    u32::MAX >> (32 - u32::from(width))
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
    zeros: Vec<ZeroResult>,
    /// The code of the guest addresses in the physical page of the block.
    entries: &'a HashMap<u32, usize>,
    /// The branches to guest addresses in the page with no code: the address and the
    /// offset of the branch.
    links: Links,
    /// A branch to `leave_to_address`.
    no_jump: Label,
    /// The guest flags that the flags of the host have now, when `asm.flags_changed`
    /// is not set.
    host_flags: u8,
    /// CF is the inverse of C in `host_flags`.
    borrow: bool,
    /// The count of the changes of `host_flags`.
    flag_changes: u32,
}

impl Compiler<'_> {
    /// The label of the code that leaves before the instruction `index`.
    fn leave_at(&mut self, index: usize) -> Label {
        if self.leaves[index].is_none() {
            self.leaves[index] = Some(self.asm.label());
        }
        self.leaves[index].unwrap()
    }

    /// The register with the value of the guest register: `temp` when it has no host
    /// register.
    fn read(&mut self, register: u8, temp: Reg) -> Reg {
        match guest(register) {
            Rm::Reg(host) => host,
            state => {
                self.asm.mov(Long, temp, state);
                temp
            }
        }
    }

    fn load(&mut self, target: Reg, register: u8) {
        if guest(register) != Rm::Reg(target) {
            self.asm.mov(Long, target, guest(register));
        }
    }

    fn write(&mut self, register: u8, value: Reg) {
        match guest(register) {
            Rm::Reg(host) if host == value => {}
            Rm::Reg(host) => self.asm.mov(Long, host, Rm::Reg(value)),
            state => self.asm.mov_to(Long, state, value),
        }
    }

    fn write_imm(&mut self, register: u8, value: u32) {
        match guest(register) {
            Rm::Reg(host) => self.asm.mov_imm(host, value),
            state => self.asm.mov_imm_to(Long, state, value),
        }
    }

    /// The register where the code makes the new value of the guest register.
    fn target(register: u8) -> Reg {
        HOST[usize::from(register)].unwrap_or(T0)
    }

    /// The guest flags that the flags of the host have now.
    fn host_flags(&mut self) -> u8 {
        if self.asm.flags_changed {
            self.host_flags = 0;
        }
        self.host_flags
    }

    /// The last instruction gave these guest flags to the flags of the host.
    fn set_host_flags(&mut self, flags: u8, borrow: bool) {
        self.host_flags = flags;
        self.borrow = borrow;
        self.flag_changes += 1;
        self.asm.flags_changed = false;
    }

    /// The condition of the host that is true when the guest condition is true.
    fn condition(&mut self, cond: u8) -> u8 {
        let needed = condition_flags(cond);
        if self.host_flags() & needed != needed {
            // OF gets V: 1 - (-127) is an overflow, 0 - (-127) is not.
            self.asm.cmp_al(0x81);
            self.asm.sahf();
            self.set_host_flags(ALL_FLAGS, false);
        }
        match cond {
            0 => x64::E,
            1 => x64::NE,
            2 | 3 => (if self.borrow { x64::AE } else { x64::B }) ^ (cond & 1),
            4 => x64::S,
            5 => x64::NS,
            6 => x64::O,
            7 => x64::NO,
            8 | 9 => {
                // The host has "above" only for a borrow.
                if !self.borrow {
                    let flags = self.host_flags;
                    self.asm.cmc();
                    self.set_host_flags(flags, true);
                }
                if cond == 8 { x64::A } else { x64::BE }
            }
            10 => x64::GE,
            11 => x64::L,
            12 => x64::G,
            _ => x64::LE,
        }
    }

    /// CF gets C, or its inverse for `borrow`.
    fn carry_to_host(&mut self, borrow: bool) {
        if self.host_flags() & C == 0 {
            self.asm.bt(Rm::Reg(FLAGS), FLAGS_C);
            self.set_host_flags(C, false);
        }
        if self.borrow != borrow {
            let flags = self.host_flags;
            self.asm.cmc();
            self.set_host_flags(flags, borrow);
        }
    }

    /// The guest flags get the flags of the host after an addition, or after a
    /// subtraction (`borrow`).
    fn save_arithmetic(&mut self, live: u8, mut borrow: bool) {
        if borrow && live & C != 0 {
            self.asm.cmc();
            borrow = false;
        }
        if live & (N | Z | C) != 0 {
            self.asm.lahf();
        }
        if live & V != 0 {
            self.asm.setcc(x64::O, Rm::Reg(FLAGS));
        }
        self.set_host_flags(ALL_FLAGS, borrow);
    }

    /// Before the instruction of the host that gives N and Z of a logical operation:
    /// T2 gets the C that the operation keeps.
    fn keep_carry(&mut self, carry: CarryOut, live: u8) {
        if carry == CarryOut::Keep && live & C != 0 {
            self.asm.mov(Long, T2, Rm::Reg(FLAGS));
            self.asm
                .alu_imm(x64::Alu::And, Long, Rm::Reg(T2), 1 << FLAGS_C);
        }
    }

    /// After that instruction, which also made CF 0: writes N and Z, and C as `carry`
    /// says. V stays.
    fn save_logical(&mut self, carry: CarryOut, live: u8) {
        if live & (N | Z | C) == 0 {
            return;
        }
        let a = &mut self.asm;
        match carry {
            CarryOut::Keep => {
                a.lahf();
                if live & C != 0 {
                    // An addition that keeps the flags of the host: the bit of C is 0.
                    a.lea(Long, FLAGS, indexed(FLAGS, T2, 0));
                }
                self.set_host_flags(N | Z, false);
            }
            CarryOut::Value(carry) => {
                if carry {
                    a.stc();
                }
                a.lahf();
                self.set_host_flags(N | Z | C, false);
            }
            CarryOut::Register => {
                a.lahf();
                a.or_ah_cl();
            }
        }
    }

    /// `rd = rn op source`, with the flags of the host from the operation.
    fn binary(&mut self, op: x64::Alu, rd: u8, rn: u8, source: Source) {
        if rd == rn && HOST[usize::from(rd)].is_none() {
            match source {
                Source::Immediate(imm) => return self.asm.alu_imm(op, Long, guest(rd), imm),
                Source::Operand(Rm::Reg(register)) => {
                    return self.asm.alu_to(op, Long, guest(rd), register);
                }
                Source::Operand(_) => {}
            }
        }
        // The host register of rd, when the operation does not read it after its write.
        let work = match HOST[usize::from(rd)] {
            Some(host) if rd == rn || source != Source::Operand(Rm::Reg(host)) => host,
            _ => T0,
        };
        self.load(work, rn);
        match source {
            Source::Operand(operand) => self.asm.alu(op, Long, work, operand),
            Source::Immediate(imm) => self.asm.alu_imm(op, Long, Rm::Reg(work), imm),
        }
        self.write(rd, work);
    }

    /// The flags of the host from `rn - source`.
    fn compare(&mut self, rn: u8, source: Source) {
        let a = &mut self.asm;
        match (guest(rn), source) {
            (first, Source::Immediate(imm)) => a.alu_imm(x64::Alu::Cmp, Long, first, imm),
            (Rm::Reg(first), Source::Operand(second)) => a.alu(x64::Alu::Cmp, Long, first, second),
            (first, Source::Operand(Rm::Reg(second))) => {
                a.alu_to(x64::Alu::Cmp, Long, first, second)
            }
            (first, Source::Operand(second)) => {
                a.mov(Long, T0, first);
                a.alu(x64::Alu::Cmp, Long, T0, second);
            }
        }
    }

    /// The flags of the host from `rn & source`.
    fn test(&mut self, rn: u8, source: Source) {
        let a = &mut self.asm;
        match (guest(rn), source) {
            (first, Source::Immediate(imm)) => a.test_imm(first, imm),
            (first, Source::Operand(Rm::Reg(second))) => a.test(Long, first, second),
            (Rm::Reg(first), Source::Operand(second)) => a.test(Long, second, first),
            (first, Source::Operand(second)) => {
                a.mov(Long, T0, first);
                a.test(Long, second, T0);
            }
        }
    }

    /// T0 gets `rn op source`, with the flags of the host from the operation.
    fn in_temp(&mut self, op: x64::Alu, rn: u8, source: Source) {
        self.load(T0, rn);
        match source {
            Source::Operand(operand) => self.asm.alu(op, Long, T0, operand),
            Source::Immediate(imm) => self.asm.alu_imm(op, Long, Rm::Reg(T0), imm),
        }
    }

    /// The operation of the guest with the operand `source`. `carry` is the C of a
    /// logical operation that sets the flags.
    #[allow(clippy::too_many_arguments)]
    fn alu(
        &mut self,
        kind: Alu,
        set: bool,
        carry: CarryOut,
        rd: u8,
        rn: u8,
        source: Source,
        live: u8,
    ) {
        match kind {
            Alu::Add | Alu::Sub => {
                let subtract = kind == Alu::Sub;
                match (HOST[usize::from(rd)], guest(rn), source) {
                    // An addition that keeps the flags of the host.
                    (Some(rd), Rm::Reg(rn), Source::Immediate(imm)) if !set && rd != rn => {
                        let imm = if subtract { imm.wrapping_neg() } else { imm };
                        self.asm.lea(Long, rd, mem(rn, imm as i32 as usize));
                    }
                    _ => {
                        let op = if subtract {
                            x64::Alu::Sub
                        } else {
                            x64::Alu::Add
                        };
                        self.binary(op, rd, rn, source);
                    }
                }
                if set {
                    self.save_arithmetic(live, subtract);
                }
            }
            Alu::Cmp => {
                self.compare(rn, source);
                self.save_arithmetic(live, true);
            }
            Alu::Cmn => {
                self.in_temp(x64::Alu::Add, rn, source);
                self.save_arithmetic(live, false);
            }
            Alu::Rsb => {
                if source == Source::Immediate(0) {
                    let work = Self::target(rd);
                    self.load(work, rn);
                    self.asm.unary(Unary::Neg, Long, Rm::Reg(work));
                    self.write(rd, work);
                } else {
                    match source {
                        Source::Operand(operand) => self.asm.mov(Long, T0, operand),
                        Source::Immediate(imm) => self.asm.mov_imm(T0, imm),
                    }
                    self.asm.alu(x64::Alu::Sub, Long, T0, guest(rn));
                    self.write(rd, T0);
                }
                if set {
                    self.save_arithmetic(live, true);
                }
            }
            Alu::Adc | Alu::Sbc => {
                // SBB subtracts CF, and the guest subtracts the inverse of C.
                let subtract = kind == Alu::Sbc;
                self.carry_to_host(subtract);
                let op = if subtract {
                    x64::Alu::Sbb
                } else {
                    x64::Alu::Adc
                };
                self.binary(op, rd, rn, source);
                if set {
                    self.save_arithmetic(live, subtract);
                }
            }
            Alu::And | Alu::Eor | Alu::Orr | Alu::Bic | Alu::Orn => {
                let mut source = source;
                if matches!(kind, Alu::Bic | Alu::Orn) {
                    source = match source {
                        Source::Immediate(imm) => Source::Immediate(!imm),
                        Source::Operand(operand) => {
                            if operand != Rm::Reg(T1) {
                                self.asm.mov(Long, T1, operand);
                            }
                            self.asm.unary(Unary::Not, Long, Rm::Reg(T1));
                            Source::Operand(Rm::Reg(T1))
                        }
                    };
                }
                let op = match kind {
                    Alu::And | Alu::Bic => x64::Alu::And,
                    Alu::Eor => x64::Alu::Xor,
                    _ => x64::Alu::Or,
                };
                if set {
                    self.keep_carry(carry, live);
                }
                self.binary(op, rd, rn, source);
                if set {
                    self.save_logical(carry, live);
                }
            }
            Alu::Mov | Alu::Mvn => {
                let invert = kind == Alu::Mvn;
                match source {
                    Source::Immediate(imm) if !set => {
                        return self.write_imm(rd, if invert { !imm } else { imm });
                    }
                    Source::Operand(Rm::Reg(register)) if !set && !invert => {
                        return self.write(rd, register);
                    }
                    _ => {}
                }
                let work = Self::target(rd);
                match source {
                    Source::Immediate(imm) => {
                        self.asm.mov_imm(work, if invert { !imm } else { imm })
                    }
                    Source::Operand(operand) => {
                        if operand != Rm::Reg(work) {
                            self.asm.mov(Long, work, operand);
                        }
                        if invert {
                            self.asm.unary(Unary::Not, Long, Rm::Reg(work));
                        }
                    }
                }
                self.write(rd, work);
                if set {
                    self.keep_carry(carry, live);
                    self.asm.test(Long, Rm::Reg(work), work);
                    self.save_logical(carry, live);
                }
            }
            Alu::Tst => {
                self.keep_carry(carry, live);
                self.test(rn, source);
                self.save_logical(carry, live);
            }
            Alu::Teq => {
                self.keep_carry(carry, live);
                self.in_temp(x64::Alu::Xor, rn, source);
                self.save_logical(carry, live);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn alu_imm(
        &mut self,
        kind: Alu,
        flags: bool,
        carry: Carry,
        rd: u8,
        rn: u8,
        imm: u32,
        live: u8,
    ) {
        let set = flags && live != 0;
        if kind.is_compare() && !set {
            return;
        }
        let carry = match carry {
            Carry::Keep => CarryOut::Keep,
            Carry::Clear => CarryOut::Value(false),
            Carry::Set => CarryOut::Value(true),
        };
        self.alu(kind, set, carry, rd, rn, Source::Immediate(imm), live);
    }

    #[allow(clippy::too_many_arguments)]
    fn alu_reg(
        &mut self,
        kind: Alu,
        flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        shift: ShiftType,
        amount: u32,
        live: u8,
    ) {
        let set = flags && live != 0;
        if kind.is_compare() && !set {
            return;
        }
        if shift == ShiftType::LSL && amount == 0 {
            let source = Source::Operand(guest(rm));
            return self.alu(kind, set, CarryOut::Keep, rd, rn, source, live);
        }
        let host_shift = match shift {
            ShiftType::LSL => Shift::Shl,
            ShiftType::LSR => Shift::Shr,
            ShiftType::ASR => Shift::Sar,
            ShiftType::ROR => Shift::Ror,
            ShiftType::RRX => Shift::Rcr,
        };
        let moves = matches!(shift, ShiftType::LSL | ShiftType::LSR | ShiftType::ASR);
        if kind == Alu::Mov && moves && amount < 32 {
            // The shift of the host gives N, Z and C as the guest wants them.
            let work = Self::target(rd);
            self.load(work, rm);
            self.asm.shift(host_shift, Long, Rm::Reg(work), amount);
            self.write(rd, work);
            if set {
                if live & (N | Z | C) != 0 {
                    self.asm.lahf();
                }
                self.set_host_flags(N | Z | C, false);
            }
            return;
        }
        let mut carry = CarryOut::Keep;
        if !is_arithmetic(kind) && set && live & C != 0 {
            // The bit of Rm that the shift moves out last.
            let bit = match shift {
                ShiftType::LSL => 32 - amount,
                ShiftType::RRX => 0,
                _ => amount - 1,
            };
            self.asm.bt(guest(rm), bit);
            self.asm.setcc(x64::B, Rm::Reg(T2));
            carry = CarryOut::Register;
        }
        if shift == ShiftType::LSR && amount == 32 {
            self.asm.mov_imm(T1, 0);
        } else {
            if shift == ShiftType::RRX {
                self.carry_to_host(false);
            }
            self.load(T1, rm);
            let amount = match shift {
                ShiftType::RRX => 1,
                ShiftType::ASR => amount.min(31),
                _ => amount,
            };
            self.asm.shift(host_shift, Long, Rm::Reg(T1), amount);
        }
        let source = Source::Operand(Rm::Reg(T1));
        self.alu(kind, set, carry, rd, rn, source, live);
    }

    /// A shift by the low byte of Rm. The code has only the amounts below 32, and with
    /// the flags only the amounts that are not 0.
    #[allow(clippy::too_many_arguments)]
    fn shift_reg(
        &mut self,
        index: usize,
        shift: ShiftType,
        flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        live: u8,
    ) {
        let leave = self.leave_at(index);
        let set = flags && live != 0;
        self.load(T2, rm);
        let a = &mut self.asm;
        if set {
            a.extend(false, Byte, T1, Rm::Reg(T2));
            a.lea(Long, T1, mem(T1, -1_i32 as usize));
            a.alu_imm(x64::Alu::Cmp, Long, Rm::Reg(T1), 31);
            a.jcc(x64::AE, leave);
        } else if shift != ShiftType::ROR {
            a.test_imm(Rm::Reg(T2), 0xe0);
            a.jcc(x64::NE, leave);
        }
        let op = match shift {
            ShiftType::LSL => Shift::Shl,
            ShiftType::LSR => Shift::Shr,
            ShiftType::ASR => Shift::Sar,
            _ => Shift::Ror,
        };
        let work = Self::target(rd);
        self.load(work, rn);
        self.asm.shift_cl(op, Rm::Reg(work));
        self.write(rd, work);
        if !set {
            return;
        }
        if shift == ShiftType::ROR {
            // The rotation gives only CF.
            self.asm.setcc(x64::B, Rm::Reg(T2));
            self.asm.test(Long, Rm::Reg(work), work);
            self.save_logical(CarryOut::Register, live);
        } else {
            if live & (N | Z | C) != 0 {
                self.asm.lahf();
            }
            self.set_host_flags(N | Z | C, false);
        }
    }

    fn give_back(&mut self, instructions: usize) {
        if instructions != 0 {
            self.asm
                .alu_imm(x64::Alu::Add, Quad, Rm::Reg(BUDGET), instructions as u32);
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
        a.mov(Long, T1, Rm::Reg(T0));
        a.shift(Shift::Shr, Long, Rm::Reg(T1), 1);
        a.alu_imm(x64::Alu::And, Long, Rm::Reg(T1), (1 << JUMP_BITS) - 1);
        a.shift(Shift::Shl, Long, Rm::Reg(T1), 4);
        a.alu(x64::Alu::Add, Quad, T1, mem(RSP, JUMPS));
        a.alu(x64::Alu::Cmp, Long, T0, mem(T1, offset_of!(Jump, key)));
        a.jcc(x64::NE, self.no_jump);
        a.jmp_rm(mem(T1, offset_of!(Jump, code)));
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
                self.asm.jmp_to(self.asm.base());
                return;
            }
            if let Some(&entry) = self.entries.get(&key) {
                self.asm.jmp_to(entry);
                return;
            }
            // A branch to the next instruction, until the address has code.
            self.links.push((key, self.asm.bytes.len()));
            self.asm.emit(&[0xe9, 0, 0, 0, 0]);
            self.asm.mov_imm(T0, key);
            self.lookup();
            return;
        }
        self.asm.mov_imm(T0, key);
        self.jump(index);
    }

    /// Leaves before the instruction `index` when the `size` bytes at the guest address
    /// in `address` are not all in one page with a direct entry. Else T1 gets the
    /// address in the host of the guest address 0.
    fn direct(&mut self, index: usize, address: Reg, size: u32, table: usize) {
        let leave = self.leave_at(index);
        self.direct_or(leave, address, size, table);
    }

    /// The same, with a branch to `other` in the place of the exit.
    fn direct_or(&mut self, other: Label, address: Reg, size: u32, table: usize) {
        let a = &mut self.asm;
        a.mov(Long, T1, Rm::Reg(address));
        a.shift(Shift::Shr, Long, Rm::Reg(T1), 12);
        a.alu_imm(x64::Alu::And, Long, Rm::Reg(T1), (1 << TLB_BITS) - 1);
        a.shift(Shift::Shl, Long, Rm::Reg(T1), TLB_ENTRY_SHIFT);
        a.alu(x64::Alu::Add, Quad, T1, mem(RSP, table));
        if size > 1 {
            a.lea(Long, T2, mem(address, size as usize - 1));
        } else {
            a.mov(Long, T2, Rm::Reg(address));
        }
        a.alu_imm(x64::Alu::And, Long, Rm::Reg(T2), !(PAGE_SIZE - 1));
        a.alu(x64::Alu::Cmp, Long, T2, mem(T1, TLB_DIRECT_PAGE));
        a.jcc(x64::NE, other);
        a.mov(Quad, T1, mem(T1, TLB_ADDEND));
    }

    /// Leaves before the instruction `index` when the address is not a multiple of 4.
    fn aligned(&mut self, index: usize, address: Reg) {
        let leave = self.leave_at(index);
        self.asm.test_imm(Rm::Reg(address), 3);
        self.asm.jcc(x64::NE, leave);
    }

    /// The register with the address of an access with an offset: T0 when the address
    /// is not the value of a host register.
    fn address(&mut self, rn: u8, mode: Index, offset: i32) -> Reg {
        let base = self.read(rn, T0);
        if mode == Index::Post || offset == 0 {
            return base;
        }
        self.asm.lea(Long, T0, mem(base, offset as usize));
        T0
    }

    /// The new value of the base register, after the access at `address`.
    fn writeback(&mut self, rn: u8, mode: Index, offset: i32, address: Reg) {
        match mode {
            Index::Offset => {}
            Index::Pre => self.write(rn, address),
            Index::Post => self
                .asm
                .alu_imm(x64::Alu::Add, Long, guest(rn), offset as u32),
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
        rt: u8,
        rn: u8,
        mode: Index,
        offset: i32,
        address: Reg,
    ) {
        let (start, done) = (self.asm.label(), self.asm.label());
        let table = if load { READS } else { WRITES };
        self.direct_or(start, address, size_of(width), table);
        let place = indexed(T1, address, 0);
        if load {
            let target = HOST[usize::from(rt)].unwrap_or(T2);
            self.load_from(width, target, place);
            self.write(rt, target);
        } else {
            let value = self.read(rt, T2);
            let size = match size_of(width) {
                1 => Byte,
                2 => Word,
                _ => Long,
            };
            self.asm.mov_to(size, place, value);
        }
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

    fn load_from(&mut self, width: Width, target: Reg, place: Rm) {
        let a = &mut self.asm;
        match width {
            Width::Byte => a.extend(false, Byte, target, place),
            Width::SignedByte => a.extend(true, Byte, target, place),
            Width::Half => a.extend(false, Word, target, place),
            Width::SignedHalf => a.extend(true, Word, target, place),
            Width::Word => a.mov(Long, target, place),
        }
    }

    /// The code of an access with a call.
    fn slow_access(&mut self, access: &SlowAccess) {
        let (retry, stop) = (self.leave_at(access.index), self.leave_at(access.index + 1));
        let rest = (self.block.len() - access.index) as u32;
        self.asm.bind(access.start);
        if access.address != T0 {
            self.asm.mov(Long, T0, Rm::Reg(access.address));
        }
        if !access.load {
            self.load(T1, access.rt);
        }
        self.asm.mov_imm(T2, size_of(access.width) | rest << 8);
        self.asm.call_to(if access.load {
            self.stubs.call_load
        } else {
            self.stubs.call_store
        });
        self.asm.test_imm(Rm::Reg(T1), 2);
        self.asm.jcc(x64::NE, retry);
        if access.load {
            let target = HOST[usize::from(access.rt)].unwrap_or(T2);
            match access.width {
                Width::SignedByte => self.asm.extend(true, Byte, target, Rm::Reg(T0)),
                Width::SignedHalf => self.asm.extend(true, Word, target, Rm::Reg(T0)),
                _ => self.asm.mov(Long, target, Rm::Reg(T0)),
            }
            self.write(access.rt, target);
        }
        // T0 has no address now: the new base is from the old one.
        if access.mode != Index::Offset {
            self.asm
                .alu_imm(x64::Alu::Add, Long, guest(access.rn), access.offset as u32);
        }
        self.asm.test(Long, Rm::Reg(T1), T1);
        self.asm.jcc(x64::NE, stop);
        self.asm.jmp(access.done);
    }

    /// Loads or stores words in sequence at the guest address in `address`. The value
    /// of the register `PC` goes to T0. The address is a multiple of 4, or the code
    /// leaves before the instruction `index`.
    fn words(&mut self, index: usize, load: bool, registers: &[u8], address: Reg) {
        self.aligned(index, address);
        let table = if load { READS } else { WRITES };
        self.direct(index, address, registers.len() as u32 * 4, table);
        // A load can change the register of the address.
        self.asm.alu(x64::Alu::Add, Quad, T1, Rm::Reg(address));
        for (word, &register) in registers.iter().enumerate() {
            let place = mem(T1, word * 4);
            if !load {
                let value = self.read(register, T2);
                self.asm.mov_to(Long, place, value);
            } else if register == PC {
                self.asm.mov(Long, T0, place);
            } else {
                let target = HOST[usize::from(register)].unwrap_or(T2);
                self.asm.mov(Long, target, place);
                self.write(register, target);
            }
        }
    }

    /// Leaves before the instruction `index` in User mode.
    fn privileged(&mut self, index: usize) {
        let leave = self.leave_at(index);
        let mode = mem(STATE, CPSR + offset_of!(Psr, mode));
        self.asm
            .alu_imm(x64::Alu::Cmp, Byte, mode, USER_MODE.into());
        self.asm.jcc(x64::E, leave);
    }

    /// T0 gets the product of 64 bits of two guest registers.
    fn long_product(&mut self, signed: bool, rn: u8, rm: u8) {
        if signed {
            self.asm.movsxd(T0, guest(rn));
            self.asm.movsxd(T1, guest(rm));
        } else {
            self.asm.mov(Long, T0, guest(rn));
            self.asm.mov(Long, T1, guest(rm));
        }
        self.asm.imul(Quad, T0, Rm::Reg(T1));
    }

    /// Writes the 64-bit number in T0 to the guest registers `lo` and `hi`.
    fn split(&mut self, lo: u8, hi: u8) {
        self.write(lo, T0);
        self.asm.shift(Shift::Shr, Quad, Rm::Reg(T0), 32);
        self.write(hi, T0);
    }

    fn divide(&mut self, signed: bool, rd: u8, rn: u8, rm: u8) {
        let (by_zero, done) = (self.asm.label(), self.asm.label());
        self.load(T2, rm);
        // The division of the host is for EDX:EAX.
        self.asm.mov(Long, T0, Rm::Reg(FLAGS));
        self.load(RAX, rn);
        let a = &mut self.asm;
        a.test(Long, Rm::Reg(T2), T2);
        a.jcc(x64::E, by_zero);
        if signed {
            // The host has an exception for the smallest number divided by -1.
            let negate = a.label();
            a.alu_imm(x64::Alu::Cmp, Long, Rm::Reg(T2), u32::MAX);
            a.jcc(x64::E, negate);
            a.cdq();
            a.unary(Unary::Idiv, Long, Rm::Reg(T2));
            a.jmp(done);
            a.bind(negate);
            a.unary(Unary::Neg, Long, Rm::Reg(RAX));
            a.jmp(done);
        } else {
            a.mov_imm(RDX, 0);
            a.unary(Unary::Div, Long, Rm::Reg(T2));
            a.jmp(done);
        }
        a.bind(by_zero);
        a.mov_imm(RAX, 0);
        a.bind(done);
        self.write(rd, RAX);
        self.asm.mov(Long, FLAGS, Rm::Reg(T0));
    }

    /// The code of the instruction `index`. Returns false, with no code, when the
    /// operation has no compiled form.
    fn op(&mut self, index: usize, step: Step, live: u8) -> bool {
        let pc = self.page.wrapping_add(u32::from(step.offset));
        let thumb = u32::from(self.block.thumb);
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
            Op::Movt { rd, imm } => {
                self.asm.alu_imm(x64::Alu::And, Long, guest(rd), 0xffff);
                self.asm
                    .alu_imm(x64::Alu::Or, Long, guest(rd), u32::from(imm) << 16);
            }
            Op::Adr { rd, offset } => {
                self.write_imm(rd, (pc.wrapping_add(4) & !3).wrapping_add(offset as u32));
            }
            Op::AddPc { rd } => {
                self.asm
                    .alu_imm(x64::Alu::Add, Long, guest(rd), pc.wrapping_add(4));
            }
            Op::Extend {
                kind,
                rd,
                rm,
                rotation,
            } => {
                let (signed, from) = match kind {
                    Extend::Sxth => (true, Word),
                    Extend::Sxtb => (true, Byte),
                    Extend::Uxth => (false, Word),
                    Extend::Uxtb => (false, Byte),
                };
                let work = Self::target(rd);
                if rotation == 0 {
                    self.asm.extend(signed, from, work, guest(rm));
                } else {
                    self.load(T1, rm);
                    self.asm
                        .shift(Shift::Ror, Long, Rm::Reg(T1), rotation.into());
                    self.asm.extend(signed, from, work, Rm::Reg(T1));
                }
                self.write(rd, work);
            }
            Op::Clz { rd, rm } => {
                // 31 minus the number of the highest bit, which is -1 for the value 0.
                let a = &mut self.asm;
                a.bsr(T2, guest(rm));
                a.mov_imm(T1, u32::MAX);
                a.cmov(x64::E, T2, Rm::Reg(T1));
                let work = Self::target(rd);
                a.mov_imm(work, 31);
                a.alu(x64::Alu::Sub, Long, work, Rm::Reg(T2));
                self.write(rd, work);
            }
            Op::Rbit { rd, rm } => {
                self.load(T0, rm);
                let a = &mut self.asm;
                // The bits change places in groups of 1, 2 and 4, then the bytes.
                for (amount, mask) in [(1, 0x5555_5555), (2, 0x3333_3333), (4, 0x0f0f_0f0f)] {
                    a.mov(Long, T1, Rm::Reg(T0));
                    a.shift(Shift::Shr, Long, Rm::Reg(T1), amount);
                    a.alu_imm(x64::Alu::And, Long, Rm::Reg(T1), mask);
                    a.alu_imm(x64::Alu::And, Long, Rm::Reg(T0), mask);
                    a.shift(Shift::Shl, Long, Rm::Reg(T0), amount);
                    a.alu(x64::Alu::Or, Long, T0, Rm::Reg(T1));
                }
                a.bswap(T0);
                self.write(rd, T0);
            }
            Op::Rev { rd, rm } => {
                let work = Self::target(rd);
                self.load(work, rm);
                self.asm.bswap(work);
                self.write(rd, work);
            }
            Op::Ubfx { rd, rn, lsb, width } => {
                let work = Self::target(rd);
                self.load(work, rn);
                if lsb != 0 {
                    self.asm.shift(Shift::Shr, Long, Rm::Reg(work), lsb.into());
                }
                if width < 32 {
                    self.asm
                        .alu_imm(x64::Alu::And, Long, Rm::Reg(work), low_bits(width));
                }
                self.write(rd, work);
            }
            Op::Sbfx { rd, rn, lsb, width } => {
                let work = Self::target(rd);
                self.load(work, rn);
                let (left, right) = (32 - u32::from(lsb + width), 32 - u32::from(width));
                if left != 0 {
                    self.asm.shift(Shift::Shl, Long, Rm::Reg(work), left);
                }
                if right != 0 {
                    self.asm.shift(Shift::Sar, Long, Rm::Reg(work), right);
                }
                self.write(rd, work);
            }
            Op::Bfi { rd, rn, lsb, width } => {
                self.load(T0, rn);
                let a = &mut self.asm;
                a.alu_imm(x64::Alu::And, Long, Rm::Reg(T0), low_bits(width));
                if lsb != 0 {
                    a.shift(Shift::Shl, Long, Rm::Reg(T0), lsb.into());
                }
                a.alu_imm(x64::Alu::And, Long, guest(rd), !(low_bits(width) << lsb));
                match guest(rd) {
                    Rm::Reg(host) => a.alu(x64::Alu::Or, Long, host, Rm::Reg(T0)),
                    state => a.alu_to(x64::Alu::Or, Long, state, T0),
                }
            }
            Op::Bfc { rd, lsb, width } => {
                self.asm
                    .alu_imm(x64::Alu::And, Long, guest(rd), !(low_bits(width) << lsb));
            }
            Op::Mul { flags, rd, rn, rm } => {
                self.load(T0, rn);
                self.asm.imul(Long, T0, guest(rm));
                self.write(rd, T0);
                if flags && live != 0 {
                    self.keep_carry(CarryOut::Keep, live);
                    self.asm.test(Long, Rm::Reg(T0), T0);
                    self.save_logical(CarryOut::Keep, live);
                }
            }
            Op::Mla { rd, rn, rm, ra } => {
                self.load(T0, rn);
                self.asm.imul(Long, T0, guest(rm));
                self.asm.alu(x64::Alu::Add, Long, T0, guest(ra));
                self.write(rd, T0);
            }
            Op::Mls { rd, rn, rm, ra } => {
                self.load(T0, rn);
                self.asm.imul(Long, T0, guest(rm));
                self.load(T1, ra);
                self.asm.alu(x64::Alu::Sub, Long, T1, Rm::Reg(T0));
                self.write(rd, T1);
            }
            Op::Umull { lo, hi, rn, rm } => {
                self.long_product(false, rn, rm);
                self.split(lo, hi);
            }
            Op::Umlal { lo, hi, rn, rm } => {
                self.long_product(false, rn, rm);
                self.load(T1, hi);
                self.asm.shift(Shift::Shl, Quad, Rm::Reg(T1), 32);
                self.load(T2, lo);
                self.asm.alu(x64::Alu::Or, Quad, T1, Rm::Reg(T2));
                self.asm.alu(x64::Alu::Add, Quad, T0, Rm::Reg(T1));
                self.split(lo, hi);
            }
            Op::Umaal { lo, hi, rn, rm } => {
                self.long_product(false, rn, rm);
                for register in [lo, hi] {
                    self.load(T1, register);
                    self.asm.alu(x64::Alu::Add, Quad, T0, Rm::Reg(T1));
                }
                self.split(lo, hi);
            }
            Op::Smull { lo, hi, rn, rm } => {
                self.long_product(true, rn, rm);
                self.split(lo, hi);
            }
            Op::Udiv { rd, rn, rm } => self.divide(false, rd, rn, rm),
            Op::Sdiv { rd, rn, rm } => self.divide(true, rd, rn, rm),
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
                self.register_address(rn, rm, shift);
                self.access(index, true, width, rt, rn, Index::Offset, 0, T0);
            }
            Op::StoreReg {
                width,
                rt,
                rn,
                rm,
                shift,
            } => {
                self.register_address(rn, rm, shift);
                self.access(index, false, width, rt, rn, Index::Offset, 0, T0);
            }
            Op::LoadLiteral { width, rt, offset } => {
                let address = (pc.wrapping_add(4) & !3).wrapping_add(offset as u32);
                let size = size_of(width);
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
                    self.write_imm(rt, value);
                } else {
                    self.asm.mov_imm(T0, address);
                    self.direct(index, T0, size, READS);
                    let target = HOST[usize::from(rt)].unwrap_or(T2);
                    self.load_from(width, target, indexed(T1, T0, 0));
                    self.write(rt, target);
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
                let registers: Vec<u8> = (0..15)
                    .filter(|register| mask & 1 << register != 0)
                    .collect();
                let address = self.read(rn, T0);
                let load = matches!(step.op, Op::LoadMultiple { .. });
                self.words(index, load, &registers, address);
                if writeback {
                    let size = registers.len() as u32 * 4;
                    self.asm.alu_imm(x64::Alu::Add, Long, guest(rn), size);
                }
            }
            Op::Push { mask } => {
                let registers: Vec<u8> = (0..15)
                    .filter(|register| mask & 1 << register != 0)
                    .collect();
                let size = registers.len() as i32 * 4;
                let sp = self.read(GUEST_SP, T0);
                self.asm.lea(Long, T0, mem(sp, -size as usize));
                self.words(index, false, &registers, T0);
                self.write(GUEST_SP, T0);
            }
            Op::Pop { mask } => {
                let registers: Vec<u8> = (0..16)
                    .filter(|register| mask & 1 << register != 0)
                    .collect();
                let sp = self.read(GUEST_SP, T0);
                self.words(index, true, &registers, sp);
                let size = registers.len() as u32 * 4;
                self.asm.alu_imm(x64::Alu::Add, Long, guest(GUEST_SP), size);
                if mask & 1 << PC != 0 {
                    self.jump(index);
                }
            }
            Op::TableBranch { rn, rm, halfword } => {
                self.load(T0, rm);
                if halfword {
                    self.asm.shift(Shift::Shl, Long, Rm::Reg(T0), 1);
                }
                if rn == PC {
                    self.asm
                        .alu_imm(x64::Alu::Add, Long, Rm::Reg(T0), pc.wrapping_add(4));
                } else {
                    self.asm.alu(x64::Alu::Add, Long, T0, guest(rn));
                }
                self.direct(index, T0, 1 << u32::from(halfword), READS);
                let from = if halfword { Word } else { Byte };
                self.asm.extend(false, from, T0, indexed(T1, T0, 0));
                self.asm.shift(Shift::Shl, Long, Rm::Reg(T0), 1);
                self.asm
                    .alu_imm(x64::Alu::Add, Long, Rm::Reg(T0), pc.wrapping_add(4) | 1);
                self.jump(index);
            }
            Op::Branch { offset } => self.jump_to(
                index,
                pc.wrapping_add(4).wrapping_add(offset as u32) | thumb,
            ),
            Op::BranchLink { offset } => {
                self.write_imm(GUEST_LR, pc.wrapping_add(4) | 1);
                self.jump_to(index, pc.wrapping_add(4).wrapping_add(offset as u32) | 1);
            }
            Op::BranchLinkArm { offset } => {
                self.write_imm(GUEST_LR, pc.wrapping_add(4));
                self.jump_to(index, pc.wrapping_add(4).wrapping_add(offset as u32));
            }
            Op::BranchLinkToArm { offset } => {
                self.write_imm(GUEST_LR, pc.wrapping_add(4) | 1);
                self.jump_to(index, (pc.wrapping_add(4) & !3).wrapping_add(offset as u32));
            }
            Op::CompareBranch {
                rn,
                nonzero,
                offset,
            } => {
                let next = self.asm.label();
                self.asm.alu_imm(x64::Alu::Cmp, Long, guest(rn), 0);
                self.asm.jcc(if nonzero { x64::E } else { x64::NE }, next);
                self.jump_to(index, pc.wrapping_add(4 + u32::from(offset)) | 1);
                self.asm.bind(next);
            }
            Op::BranchExchange { rm } => {
                self.load(T0, rm);
                self.jump(index);
            }
            Op::BranchLinkExchange { rm } => {
                self.load(T0, rm);
                self.write_imm(GUEST_LR, pc.wrapping_add(2) | 1);
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
                let state = offset_of!(Armv7Cpu, cp15) + field[usize::from(register)];
                let target = Self::target(rt);
                self.asm.mov(Long, target, mem(STATE, state));
                self.write(rt, target);
            }
            Op::MulHalves {
                rd,
                rn,
                rm,
                n_high,
                m_high,
            } => {
                for (target, source, high) in [(T0, rn, n_high), (T1, rm, m_high)] {
                    if high {
                        self.load(target, source);
                        self.asm.shift(Shift::Sar, Long, Rm::Reg(target), 16);
                    } else {
                        self.asm.extend(true, Word, target, guest(source));
                    }
                }
                self.asm.imul(Long, T0, Rm::Reg(T1));
                self.write(rd, T0);
            }
            op if op.is_fp() => return self.fp_op(index, step),
            _ => return false,
        }
        true
    }

    /// True when the FP instruction `index` can run with no check of the FP controls.
    fn fp_admitted(&self, index: usize) -> bool {
        self.block.is_admitted(index, self.fp.controls)
    }

    /// Loads or stores `count` D registers from D`d` at the guest address in `address`.
    fn fp_block(&mut self, index: usize, load: bool, d: u8, count: u8, address: Reg) {
        let table = if load { READS } else { WRITES };
        self.direct(index, address, u32::from(count) * 8, table);
        let a = &mut self.asm;
        a.alu(x64::Alu::Add, Quad, T1, Rm::Reg(address));
        for register in 0..count {
            let (state, place) = (dreg(d + register), mem(T1, usize::from(register) * 8));
            let (from, to) = if load { (place, state) } else { (state, place) };
            a.mov(Quad, T2, from);
            a.mov_to(Quad, to, T2);
        }
    }

    /// True when the host does the FP arithmetic as the guest: the controls of the guest
    /// are round to nearest, no flush to zero and no default NaN, as those of the host.
    fn fp_arithmetic(&self) -> bool {
        self.fp.fpscr & FPSCR_MODES == 0
    }

    /// Leaves before the instruction `index` when the result of the FP arithmetic in
    /// `register` is not sure to be the result of the guest with its flags. It is sure
    /// when it is a number with an exponent that is not the largest or one of the two
    /// smallest: then the operation was not invalid and had no underflow, and the host
    /// and the guest have the same rule for each other flag. They differ for a result at
    /// the limit of the normal numbers, where the guest looks for an underflow before it
    /// rounds and the host after. A result that is 0 is sure when it has no rounding
    /// (`zero`, with `first` the guest register of the first operand and X1 the second).
    ///
    /// The exceptions of a result that is not sure stay in the MXCSR. The division by
    /// zero, the overflow and the inexact result are also those of the guest for this
    /// operation, and the code reads no other one.
    fn fp_guard(&mut self, index: usize, double: bool, register: Reg, zero: Zero, first: u8) {
        let unsure = self.leave_at(index);
        let a = &mut self.asm;
        let (is_zero, done) = (a.label(), a.label());
        let (size, exponent, exponents) = if double {
            (Quad, 53, 2047)
        } else {
            (Long, 24, 255)
        };
        a.movd_from_xmm(size, T0, register);
        a.shift(Shift::Shl, size, Rm::Reg(T0), 1);
        a.jcc(x64::E, if zero == Zero::Exact { done } else { is_zero });
        a.shift(Shift::Shr, size, Rm::Reg(T0), exponent);
        a.lea(Long, T0, mem(T0, -2_i32 as usize));
        a.alu_imm(x64::Alu::Cmp, Long, Rm::Reg(T0), exponents - 2);
        a.jcc(x64::AE, unsure);
        a.bind(done);
        if zero != Zero::Exact {
            self.zeros.push(ZeroResult {
                start: is_zero,
                done,
                index,
                double,
                second: (zero == Zero::Product).then_some(X1),
                first,
            });
        }
    }

    /// The code for a product or a quotient that is 0. It has no rounding when an
    /// operand of the product, or the first operand of the quotient, is 0. Else it has
    /// an underflow, or the second operand of the quotient is an infinity.
    fn zero_result(&mut self, zero: &ZeroResult) {
        let unsure = self.leave_at(zero.index);
        let size = if zero.double { Quad } else { Long };
        let a = &mut self.asm;
        a.bind(zero.start);
        if let Some(second) = zero.second {
            a.movd_from_xmm(size, T0, second);
            a.shift(Shift::Shl, size, Rm::Reg(T0), 1);
            a.jcc(x64::E, zero.done);
        }
        a.mov(size, T0, fp_reg(zero.double, zero.first));
        a.shift(Shift::Shl, size, Rm::Reg(T0), 1);
        a.jcc(x64::E, zero.done);
        a.jmp(unsure);
    }

    /// The code of the FP arithmetic of the instruction `index`.
    fn fp_arithmetic_op(&mut self, index: usize, op: Op) {
        match op {
            Op::FpBinary {
                kind,
                double,
                d,
                n,
                m,
            } => {
                self.asm.fp_load(double, X0, fp_reg(double, n));
                self.asm.fp_load(double, X1, fp_reg(double, m));
                let (op, zero) = match kind {
                    FpBinary::Add => (Sse::Add, Zero::Exact),
                    FpBinary::Sub => (Sse::Sub, Zero::Exact),
                    FpBinary::Mul | FpBinary::Nmul => (Sse::Mul, Zero::Product),
                    FpBinary::Div => (Sse::Div, Zero::Quotient),
                };
                self.asm.fp(op, double, X0, X1);
                self.fp_guard(index, double, X0, zero, n);
                self.asm.fp_store(double, fp_reg(double, d), X0);
                if kind == FpBinary::Nmul {
                    self.negate(double, d);
                }
            }
            Op::FpAccumulate {
                kind,
                double,
                d,
                n,
                m,
            } => {
                self.asm.fp_load(double, X0, fp_reg(double, n));
                self.asm.fp_load(double, X1, fp_reg(double, m));
                self.asm.fp(Sse::Mul, double, X0, X1);
                self.fp_guard(index, double, X0, Zero::Product, n);
                let a = &mut self.asm;
                // The result of the guest is (d or -d) + (product or -product).
                let result = match kind {
                    FpAccumulate::Mla => {
                        a.fp_load(double, X1, fp_reg(double, d));
                        a.fp(Sse::Add, double, X0, X1);
                        X0
                    }
                    FpAccumulate::Mls => {
                        a.fp_load(double, X1, fp_reg(double, d));
                        a.fp(Sse::Sub, double, X1, X0);
                        X1
                    }
                    FpAccumulate::Nmls => {
                        a.fp_load(double, X1, fp_reg(double, d));
                        a.fp(Sse::Sub, double, X0, X1);
                        X0
                    }
                    FpAccumulate::Nmla => {
                        if double {
                            a.mov(Quad, T0, dreg(d));
                            a.mov_imm64(T1, 1 << 63);
                            a.alu(x64::Alu::Xor, Quad, T0, Rm::Reg(T1));
                            a.movd_to_xmm(Quad, X1, T0);
                        } else {
                            a.mov(Long, T0, sreg(d));
                            a.alu_imm(x64::Alu::Xor, Long, Rm::Reg(T0), 1 << 31);
                            a.movd_to_xmm(Long, X1, T0);
                        }
                        a.fp(Sse::Sub, double, X1, X0);
                        X1
                    }
                };
                self.fp_guard(index, double, result, Zero::Exact, n);
                self.asm.fp_store(double, fp_reg(double, d), result);
            }
            Op::FpUnary { double, d, m, .. } => {
                self.asm.fp_load(double, X0, fp_reg(double, m));
                self.asm.fp(Sse::Sqrt, double, X0, X0);
                self.fp_guard(index, double, X0, Zero::Exact, m);
                self.asm.fp_store(double, fp_reg(double, d), X0);
            }
            Op::FpCompare {
                with_zero,
                double,
                d,
                m,
                ..
            } => {
                let unsure = self.leave_at(index);
                let a = &mut self.asm;
                a.fp_load(double, X0, fp_reg(double, d));
                if with_zero {
                    a.mov_imm(T0, 0);
                    a.movd_to_xmm(Quad, X1, T0);
                } else {
                    a.fp_load(double, X1, fp_reg(double, m));
                }
                a.ucomis(double, X0, X1);
                // A NaN: the flags of the guest are for the kind of the NaN and of the
                // compare.
                a.jcc(x64::P, unsure);
                // N for less, Z and C for equal, C for greater.
                a.mov_imm(T0, 0x2000_0000);
                a.mov_imm(T1, 0x6000_0000);
                a.cmov(x64::E, T0, Rm::Reg(T1));
                a.mov_imm(T1, 0x8000_0000);
                a.cmov(x64::B, T0, Rm::Reg(T1));
                a.mov(Long, T1, mem(STATE, FPSCR));
                a.alu_imm(x64::Alu::And, Long, Rm::Reg(T1), 0x0fff_ffff);
                a.alu(x64::Alu::Or, Long, T1, Rm::Reg(T0));
                a.mov_to(Long, mem(STATE, FPSCR), T1);
            }
            Op::FpConvert { kind, d, m } => {
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
                let unsure = self.leave_at(index);
                let a = &mut self.asm;
                match (to_float, unsigned) {
                    (true, false) => {
                        a.cvtsi2s(double, Long, X0, sreg(m));
                        a.fp_store(double, fp_reg(double, d), X0);
                    }
                    (true, true) => {
                        a.mov(Long, T0, sreg(m));
                        a.cvtsi2s(double, Quad, X0, Rm::Reg(T0));
                        a.fp_store(double, fp_reg(double, d), X0);
                    }
                    // The guest has the nearest limit for a number out of its range,
                    // with the invalid operation and never the inexact result. The
                    // host has no such result.
                    (false, false) => {
                        // The host gives this number, and no inexact result, for each
                        // number out of the range and for a NaN.
                        const OUT_OF_RANGE: u32 = 1 << 31;
                        a.fp_load(double, X0, fp_reg(double, m));
                        a.cvtts2si(double, Long, T0, X0);
                        a.alu_imm(x64::Alu::Cmp, Long, Rm::Reg(T0), OUT_OF_RANGE);
                        a.jcc(x64::E, unsure);
                        a.mov_to(Long, sreg(d), T0);
                    }
                    (false, true) => {
                        // The host has signed numbers only, and a number out of the
                        // range of 32 bits can be inexact in 64 bits: the conversion is
                        // for the numbers from +0 to below 2^32, which are below this
                        // limit as unsigned integers. A negative number is above it.
                        let (high, limit) = if double {
                            (sreg(m * 2 + 1), 0x41f0_0000)
                        } else {
                            (sreg(m), 0x4f80_0000)
                        };
                        a.alu_imm(x64::Alu::Cmp, Long, high, limit);
                        a.jcc(x64::AE, unsure);
                        a.fp_load(double, X0, fp_reg(double, m));
                        a.cvtts2si(double, Quad, T0, X0);
                        a.mov_to(Long, sreg(d), T0);
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    /// Changes the sign of the guest register S`number`, or D`number`.
    fn negate(&mut self, double: bool, number: u8) {
        let sign = if double {
            sreg(number * 2 + 1)
        } else {
            sreg(number)
        };
        self.asm.alu_imm(x64::Alu::Xor, Long, sign, 1 << 31);
    }

    /// A load of the guest register D`number`, or of the Q register at D`number`.
    fn simd_load(&mut self, quad: bool, host: Reg, number: u8) {
        if quad {
            self.asm.movdqu_load(host, dreg(number));
        } else {
            self.asm.movq_load(host, dreg(number));
        }
    }

    fn simd_store(&mut self, quad: bool, number: u8, host: Reg) {
        if quad {
            self.asm.movdqu_store(dreg(number), host);
        } else {
            self.asm.movq_store(dreg(number), host);
        }
    }

    /// Each element of 32 bits of the XMM register gets `value`.
    fn simd_words(&mut self, host: Reg, value: u32) {
        self.asm.mov_imm(T0, value);
        self.asm.movd_to_xmm(Long, host, T0);
        self.asm.pshufd(host, host, 0);
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
        if kind != SimdLane::Shr {
            self.simd_load(quad, X0, n);
        }
        self.simd_load(quad, X1, m);
        let result = match kind {
            SimdLane::Add => {
                let opcode = if size == 3 {
                    x64::PADDQ
                } else {
                    x64::PADD + size
                };
                self.asm.packed(opcode, X0, X1);
                X0
            }
            SimdLane::Sub => {
                self.asm.packed(x64::PSUB + size, X0, X1);
                X0
            }
            SimdLane::Ceq => {
                self.asm.packed(x64::PCMPEQ + size, X0, X1);
                X0
            }
            SimdLane::Cgt => {
                if unsigned {
                    // The host compares signed numbers: without the sign bits, the
                    // order is that of the unsigned numbers.
                    let signs = [0x8080_8080, 0x8000_8000, 0x8000_0000][usize::from(size)];
                    self.simd_words(X2, signs);
                    self.asm.packed(x64::PXOR, X0, X2);
                    self.asm.packed(x64::PXOR, X1, X2);
                }
                self.asm.packed(x64::PCMPGT + size, X0, X1);
                X0
            }
            SimdLane::Shr if size == 0 => {
                // The host has no shift of bytes: the bits of the next byte go away.
                self.asm.packed_shift(1, false, X1, imm);
                self.simd_words(X2, (0xff >> imm) * 0x0101_0101);
                self.asm.packed(x64::PAND, X1, X2);
                X1
            }
            SimdLane::Shr => {
                self.asm.packed_shift(size, !unsigned, X1, imm);
                X1
            }
            _ => unreachable!(),
        };
        self.simd_store(quad, d, result);
    }

    /// A compare of FP numbers of 32 bits. The guest takes a denormal number as 0 here:
    /// the code leaves before the instruction `index` for one, and for a NaN.
    fn simd_float_compare(
        &mut self,
        index: usize,
        kind: SimdLane,
        quad: bool,
        d: u8,
        n: u8,
        m: u8,
    ) {
        const X3: Reg = 3;
        const MAGNITUDE: Reg = 4;
        const SMALLEST: Reg = 5;
        const LARGEST: Reg = 6;
        const ZERO: Reg = 7;
        const BAD: Reg = 8;
        const X9: Reg = 9;
        let leave = self.leave_at(index);
        // The 4 elements of the host registers are in the test: the high ones are 0
        // after a load of a D register.
        self.simd_load(quad, X0, n);
        self.simd_load(quad, X1, m);
        self.simd_words(MAGNITUDE, 0x7fff_ffff);
        self.simd_words(SMALLEST, 0x0080_0000_u32.wrapping_neg());
        self.simd_words(LARGEST, 0x7f80_0000 - 0x0080_0000);
        let a = &mut self.asm;
        a.packed(x64::PXOR, ZERO, ZERO);
        a.packed(x64::PXOR, BAD, BAD);
        for operand in [X0, X1] {
            a.packed(x64::MOVDQA, X2, operand);
            a.packed(x64::PAND, X2, MAGNITUDE);
            // The magnitude minus the smallest normal number: more than `LARGEST` for a
            // NaN, and negative for a denormal number and for 0.
            a.packed(x64::MOVDQA, X3, X2);
            a.packed(x64::PADD + 2, X3, SMALLEST);
            a.packed(x64::MOVDQA, X9, X3);
            a.packed(x64::PCMPGT + 2, X9, LARGEST);
            a.packed(x64::POR, BAD, X9);
            a.packed_shift(2, true, X3, 31);
            a.packed(x64::PCMPEQ + 2, X2, ZERO);
            a.packed(x64::PANDN, X2, X3);
            a.packed(x64::POR, BAD, X2);
        }
        a.pmovmskb(T0, BAD);
        a.test(Long, Rm::Reg(T0), T0);
        a.jcc(x64::NE, leave);
        // The host has "less": greater is less with the other order.
        let result = match kind {
            SimdLane::Ceq => {
                a.cmpps(X0, X1, 0);
                X0
            }
            SimdLane::Cge => {
                a.cmpps(X1, X0, 2);
                X1
            }
            _ => {
                a.cmpps(X1, X0, 1);
                X1
            }
        };
        self.simd_store(quad, d, result);
    }

    /// The new value of the base register of a SIMD load or store of `size` bytes.
    fn simd_writeback(&mut self, rn: u8, rm: u8, size: u32) {
        match rm {
            PC => {}
            SP => self.asm.alu_imm(x64::Alu::Add, Long, guest(rn), size),
            _ => self.binary(x64::Alu::Add, rn, rn, Source::Operand(guest(rm))),
        }
    }

    /// The code of the FP or SIMD instruction `index`. Returns false, with no code, when
    /// the operation has no compiled form.
    fn fp_op(&mut self, index: usize, step: Step) -> bool {
        let pc = self.page.wrapping_add(u32::from(step.offset));
        if !has_fp_code(step.op) || !self.fp_admitted(index) {
            return false;
        }
        if is_fp_arithmetic(step.op) {
            if self.fp_arithmetic() {
                self.fp_arithmetic_op(index, step.op);
            }
            return self.fp_arithmetic();
        }
        match step.op {
            Op::FpUnary {
                kind,
                double: false,
                d,
                m,
            } => {
                let a = &mut self.asm;
                a.mov(Long, T0, sreg(m));
                match kind {
                    FpUnary::Abs => a.alu_imm(x64::Alu::And, Long, Rm::Reg(T0), !(1 << 31)),
                    FpUnary::Neg => a.alu_imm(x64::Alu::Xor, Long, Rm::Reg(T0), 1 << 31),
                    _ => {}
                }
                a.mov_to(Long, sreg(d), T0);
            }
            Op::FpUnary {
                kind,
                double: true,
                d,
                m,
            } => {
                let a = &mut self.asm;
                a.mov(Quad, T0, dreg(m));
                if kind != FpUnary::Mov {
                    a.mov_imm64(T1, 1 << 63);
                    if kind == FpUnary::Abs {
                        a.unary(Unary::Not, Quad, Rm::Reg(T1));
                        a.alu(x64::Alu::And, Quad, T0, Rm::Reg(T1));
                    } else {
                        a.alu(x64::Alu::Xor, Quad, T0, Rm::Reg(T1));
                    }
                }
                a.mov_to(Quad, dreg(d), T0);
            }
            Op::FpImmediate {
                double: false,
                d,
                imm,
            } => {
                self.asm
                    .mov_imm_to(Long, sreg(d), vfp::vfp_expand_imm_f32(imm));
            }
            Op::FpImmediate {
                double: true,
                d,
                imm,
            } => {
                self.asm.mov_imm64(T0, vfp::vfp_expand_imm_f64(imm));
                self.asm.mov_to(Quad, dreg(d), T0);
            }
            Op::FpFromCore { s, rt } => {
                let value = self.read(rt, T0);
                self.asm.mov_to(Long, sreg(s), value);
            }
            Op::FpToCore { rt, s } => {
                let target = Self::target(rt);
                self.asm.mov(Long, target, sreg(s));
                self.write(rt, target);
            }
            Op::FpFromCorePair {
                double, v, rt, rt2, ..
            } => {
                // A D register is two S registers in the state.
                let s = if double { v * 2 } else { v };
                for (register, s) in [(rt, s), (rt2, s + 1)] {
                    let value = self.read(register, T0);
                    self.asm.mov_to(Long, sreg(s), value);
                }
            }
            Op::FpToCorePair {
                double, rt, rt2, v, ..
            } => {
                let s = if double { v * 2 } else { v };
                for (register, s) in [(rt, s), (rt2, s + 1)] {
                    let target = Self::target(register);
                    self.asm.mov(Long, target, sreg(s));
                    self.write(register, target);
                }
            }
            Op::FpLoad {
                double,
                d,
                rn,
                offset,
            } => {
                let address = self.address(rn, Index::Offset, offset);
                self.aligned(index, address);
                let (size, state) = if double {
                    (Quad, dreg(d))
                } else {
                    (Long, sreg(d))
                };
                self.direct(index, address, if double { 8 } else { 4 }, READS);
                self.asm.mov(size, T2, indexed(T1, address, 0));
                self.asm.mov_to(size, state, T2);
            }
            Op::FpLoadLiteral { double, d, offset } => {
                let address = (pc.wrapping_add(4) & !3).wrapping_add(offset as u32);
                let bytes = if double { 8 } else { 4 };
                let in_page = address.wrapping_sub(self.page);
                if address & 3 != 0 {
                    return false;
                }
                let (size, state) = if double {
                    (Quad, dreg(d))
                } else {
                    (Long, sreg(d))
                };
                if in_page <= PAGE_SIZE - bytes {
                    // A write to the page removes the block, thus the value stays.
                    let bytes = &self.code[in_page as usize..];
                    if double {
                        self.asm
                            .mov_imm64(T2, u64::from_le_bytes(bytes[..8].try_into().unwrap()));
                    } else {
                        self.asm
                            .mov_imm(T2, u32::from_le_bytes(bytes[..4].try_into().unwrap()));
                    }
                } else {
                    self.asm.mov_imm(T0, address);
                    self.direct(index, T0, bytes, READS);
                    self.asm.mov(size, T2, indexed(T1, T0, 0));
                }
                self.asm.mov_to(size, state, T2);
            }
            Op::FpStore {
                double,
                d,
                rn,
                offset,
            } => {
                let address = self.address(rn, Index::Offset, offset);
                self.aligned(index, address);
                let (size, state) = if double {
                    (Quad, dreg(d))
                } else {
                    (Long, sreg(d))
                };
                self.direct(index, address, if double { 8 } else { 4 }, WRITES);
                self.asm.mov(size, T2, state);
                self.asm.mov_to(size, indexed(T1, address, 0), T2);
            }
            Op::FpFlags => {
                // N, Z, C and V are the bits 31 to 28 of the FPSCR.
                let a = &mut self.asm;
                a.mov(Long, T1, mem(STATE, FPSCR));
                a.shift(Shift::Shr, Long, Rm::Reg(T1), 28);
                a.mov(Long, FLAGS, Rm::Reg(T1));
                a.alu_imm(x64::Alu::And, Long, Rm::Reg(FLAGS), 1);
                a.shift(Shift::Shr, Long, Rm::Reg(T1), 1);
                a.mov(Long, T2, Rm::Reg(T1));
                a.alu_imm(x64::Alu::And, Long, Rm::Reg(T2), 1);
                a.shift(Shift::Shr, Long, Rm::Reg(T1), 1);
                a.shift(Shift::Shl, Long, Rm::Reg(T1), 6);
                a.alu(x64::Alu::Or, Long, T1, Rm::Reg(T2));
                a.shift(Shift::Shl, Long, Rm::Reg(T1), 8);
                a.alu(x64::Alu::Or, Long, FLAGS, Rm::Reg(T1));
            }
            Op::SimdLogic {
                kind,
                quad,
                d,
                n,
                m,
            } => {
                let a = &mut self.asm;
                for half in 0..=u8::from(quad) {
                    let (d, n, m) = (dreg(d + half), dreg(n + half), dreg(m + half));
                    a.mov(Quad, T0, n);
                    a.mov(Quad, T1, m);
                    match kind {
                        SimdLogic::Not => {
                            a.mov(Quad, T0, Rm::Reg(T1));
                            a.unary(Unary::Not, Quad, Rm::Reg(T0));
                        }
                        SimdLogic::And => a.alu(x64::Alu::And, Quad, T0, Rm::Reg(T1)),
                        SimdLogic::Orr => a.alu(x64::Alu::Or, Quad, T0, Rm::Reg(T1)),
                        SimdLogic::Eor => a.alu(x64::Alu::Xor, Quad, T0, Rm::Reg(T1)),
                        SimdLogic::Bic | SimdLogic::Orn => {
                            a.unary(Unary::Not, Quad, Rm::Reg(T1));
                            let op = if kind == SimdLogic::Bic {
                                x64::Alu::And
                            } else {
                                x64::Alu::Or
                            };
                            a.alu(op, Quad, T0, Rm::Reg(T1));
                        }
                        // Each one takes the bits of `first` where `mask` is 1 and the
                        // bits of `second` where it is 0: second ^ ((first ^ second) & mask).
                        SimdLogic::Bsl | SimdLogic::Bit | SimdLogic::Bif => {
                            a.mov(Quad, T2, d);
                            let (first, second, mask) = match kind {
                                SimdLogic::Bsl => (T0, T1, T2),
                                SimdLogic::Bit => (T0, T2, T1),
                                _ => (T2, T0, T1),
                            };
                            a.alu(x64::Alu::Xor, Quad, first, Rm::Reg(second));
                            a.alu(x64::Alu::And, Quad, first, Rm::Reg(mask));
                            a.alu(x64::Alu::Xor, Quad, first, Rm::Reg(second));
                            if first != T0 {
                                a.mov(Quad, T0, Rm::Reg(first));
                            }
                        }
                    }
                    a.mov_to(Quad, d, T0);
                }
            }
            Op::SimdLane { .. } => self.simd_lane(step.op),
            Op::SimdFloatCompare {
                kind,
                quad,
                d,
                n,
                m,
            } => self.simd_float_compare(index, kind, quad, d, n, m),
            Op::SimdImmediate { quad, d, constant } => {
                self.asm
                    .mov_imm64(T0, self.block.constants[usize::from(constant)]);
                for half in 0..=u8::from(quad) {
                    self.asm.mov_to(Quad, dreg(d + half), T0);
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
                let address = self.read(rn, T0);
                if align > 1 {
                    let leave = self.leave_at(index);
                    self.asm.test_imm(Rm::Reg(address), u32::from(align) - 1);
                    self.asm.jcc(x64::NE, leave);
                }
                let load = matches!(step.op, Op::SimdLoad { .. });
                self.fp_block(index, load, d, count, address);
                self.simd_writeback(rn, rm, u32::from(count) * 8);
            }
            Op::FpElementFromCore {
                d,
                offset,
                size,
                rt,
            } => {
                let value = self.read(rt, T0);
                let place = mem(
                    STATE,
                    DREGS_OFFSET + usize::from(d) * 8 + usize::from(offset),
                );
                self.asm
                    .mov_to([Byte, Word, Long][usize::from(size)], place, value);
            }
            Op::FpElementToCore {
                rt,
                d,
                offset,
                size,
                signed,
            } => {
                let place = mem(
                    STATE,
                    DREGS_OFFSET + usize::from(d) * 8 + usize::from(offset),
                );
                let target = Self::target(rt);
                match size {
                    0 => self.asm.extend(signed, Byte, target, place),
                    1 => self.asm.extend(signed, Word, target, place),
                    _ => self.asm.mov(Long, target, place),
                }
                self.write(rt, target);
            }
            Op::SimdDuplicate {
                quad,
                d,
                size,
                from_core,
                source,
                offset,
            } => {
                let element = if from_core {
                    guest(source)
                } else {
                    mem(
                        STATE,
                        DREGS_OFFSET + usize::from(source) * 8 + usize::from(offset),
                    )
                };
                let a = &mut self.asm;
                // The product with a 1 at each place of an element.
                let ones: u64 = match size {
                    0 => {
                        a.extend(false, Byte, T0, element);
                        0x0101_0101_0101_0101
                    }
                    1 => {
                        a.extend(false, Word, T0, element);
                        0x0001_0001_0001_0001
                    }
                    _ => {
                        a.mov(Long, T0, element);
                        0x0000_0001_0000_0001
                    }
                };
                a.mov_imm64(T1, ones);
                a.imul(Quad, T0, Rm::Reg(T1));
                for half in 0..=u8::from(quad) {
                    a.mov_to(Quad, dreg(d + half), T0);
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
                let size = i32::from(count) * 8;
                let mut start = self.read(rn, T0);
                if decrement {
                    self.asm.lea(Long, T0, mem(start, -size as usize));
                    start = T0;
                }
                self.aligned(index, start);
                let load = matches!(step.op, Op::FpLoadMultiple { .. });
                self.fp_block(index, load, d, count, start);
                match (writeback, decrement) {
                    (false, _) => {}
                    (true, true) => self.write(rn, T0),
                    (true, false) => {
                        self.asm
                            .alu_imm(x64::Alu::Add, Long, guest(rn), size as u32);
                    }
                }
            }
            _ => unreachable!(),
        }
        true
    }

    /// T0 gets the address `rn + (rm << shift)`.
    fn register_address(&mut self, rn: u8, rm: u8, shift: u8) {
        self.load(T0, rm);
        if shift != 0 {
            self.asm.shift(Shift::Shl, Long, Rm::Reg(T0), shift.into());
        }
        self.asm.alu(x64::Alu::Add, Long, T0, guest(rn));
    }

    /// True when the instruction `index` has a compiled form in each state of the guest.
    /// Without that, the code can stop before the instruction.
    fn always_has_code(&self, index: usize) -> bool {
        let op = self.block.steps[index].op;
        op != Op::Slow && !op.is_fp()
    }

    /// Returns the code and its branches to addresses of the page with no code, or None
    /// when the first instruction has no compiled form.
    fn compile(mut self) -> Option<(Vec<u8>, Links)> {
        let block = self.block;
        let steps = &block.steps;
        let live = live_flags(steps, |index| self.always_has_code(index));
        let over_budget = self.asm.label();
        self.asm
            .alu_imm(x64::Alu::Sub, Quad, Rm::Reg(BUDGET), steps.len() as u32);
        self.asm.jcc(x64::S, over_budget);
        if (0..steps.len()).any(|index| has_fp_code(steps[index].op) && self.fp_admitted(index)) {
            // The FP code is for one state of the FP controls.
            let leave = self.leave_at(0);
            let a = &mut self.asm;
            a.mov(Quad, T0, mem(RSP, CONTEXT));
            a.mov_imm64(T1, self.fp.controls);
            a.alu(
                x64::Alu::Cmp,
                Quad,
                T1,
                mem(T0, offset_of!(Context, fp_controls)),
            );
            a.jcc(x64::NE, leave);
        }
        let arithmetic =
            |index: usize| is_fp_arithmetic(steps[index].op) && self.fp_admitted(index);
        if self.fp_arithmetic() && (0..steps.len()).any(arithmetic) {
            // The exceptions in the MXCSR are those of the code of the caller: only the
            // first block with FP arithmetic of a run makes it empty.
            let a = &mut self.asm;
            let used = a.label();
            a.alu_imm(x64::Alu::Cmp, Long, mem(RSP, FP_USED), 0);
            a.jcc(x64::NE, used);
            a.mov_imm_to(Long, mem(RSP, MXCSR), MXCSR_DEFAULT);
            a.ldmxcsr(mem(RSP, MXCSR));
            a.mov_imm_to(Long, mem(RSP, FP_USED), 1);
            a.bind(used);
        }
        let mut compiled = steps.len();
        for (index, step) in steps.iter().enumerate() {
            let next = self.asm.label();
            // The condition 15 is always true, as 14.
            let conditional = step.cond < COND_ALWAYS;
            if conditional {
                let condition = self.condition(step.cond);
                self.asm.jcc(condition ^ 1, next);
            }
            let before = (self.host_flags(), self.flag_changes);
            if !self.op(index, *step, live[index]) {
                // The interpreter runs the instruction. The code after it is for a
                // condition that is false.
                let leave = self.leave_at(index);
                self.asm.jmp(leave);
                if step.cond >= COND_ALWAYS {
                    compiled = index;
                    break;
                }
            }
            if conditional {
                // The flags of the host are known after the instruction only when they
                // are the same with and without it.
                let same = !self.asm.flags_changed && self.flag_changes == before.1;
                self.asm.bind(next);
                self.asm.flags_changed = !(same && before.0 != 0);
            }
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
            self.asm.jmp(leave);
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
        self.asm.jmp_to(self.stubs.leave_to_address);
        for access in std::mem::take(&mut self.slow) {
            self.slow_access(&access);
        }
        for zero in std::mem::take(&mut self.zeros) {
            self.zero_result(&zero);
        }
        for index in 0..self.leaves.len() {
            if let Some(label) = self.leaves[index] {
                self.asm.bind(label);
                self.give_back(steps.len() - index);
                self.asm.call_to(self.stubs.leave_at_instruction);
                self.asm.emit_u32(self.number);
                self.asm.emit_u32(index as u32);
            }
        }
        self.asm.finish();
        Some((self.asm.bytes, self.links))
    }
}

/// The size of an access in bytes.
fn size_of(width: Width) -> u32 {
    match width {
        Width::Byte | Width::SignedByte => 1,
        Width::Half | Width::SignedHalf => 2,
        Width::Word => 4,
    }
}

/// Returns the code of the block and its branches to addresses of the page with no code
/// (the address and the offset of the branch), or None when the first instruction has no
/// compiled form.
pub(super) fn compile(unit: Unit) -> Option<(Vec<u8>, Links)> {
    let mut asm = Asm::new(unit.entry);
    let compiler = Compiler {
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
        zeros: Vec::new(),
        entries: unit.entries,
        links: Vec::new(),
        host_flags: 0,
        borrow: false,
        flag_changes: 0,
    };
    compiler.compile()
}

/// The address and the bytes of the offset of a branch to `target` at `site`.
pub(super) fn link(site: usize, target: usize) -> (usize, [u8; 4]) {
    let offset = target as i64 - (site as i64 + 5);
    (site + 1, (offset as i32).to_le_bytes())
}

/// Adds the exceptions in the MXCSR to the FPSCR of the guest, when FP arithmetic ran.
/// `above` is the count of the bytes on the stack above the frame.
fn mxcsr_to_fpscr(a: &mut Asm, above: usize, temp: Reg) {
    let no_fp = a.label();
    a.alu_imm(x64::Alu::Cmp, Long, mem(RSP, above + FP_USED), 0);
    a.jcc(x64::E, no_fp);
    a.stmxcsr(mem(RSP, above + MXCSR));
    a.mov(Long, temp, mem(RSP, above + MXCSR));
    a.shift(Shift::Shr, Long, Rm::Reg(temp), 1);
    a.alu_imm(x64::Alu::And, Long, Rm::Reg(temp), FPSCR_FROM_MXCSR);
    a.alu_to(x64::Alu::Or, Long, mem(STATE, FPSCR), temp);
    a.bind(no_fp);
}

fn guest_registers() -> impl Iterator<Item = (Reg, Rm)> {
    (0..15)
        .filter_map(|register| HOST[register].map(|host| (host, mem(STATE, REGS + register * 4))))
}

/// The code that goes from the caller to compiled code and back.
pub(super) fn stubs(base: usize) -> (Vec<u8>, Stubs) {
    const SAVED: [Reg; 6] = [RBP, RBX, R12, R13, R14, R15];
    const PSR_FLAGS: [(usize, u32); 4] = [
        (offset_of!(Psr, n), 15),
        (offset_of!(Psr, z), 14),
        (offset_of!(Psr, c), FLAGS_C),
        (offset_of!(Psr, v), 0),
    ];
    let mut a = Asm::new(base);
    let enter = a.here();
    for register in SAVED {
        a.push(register);
    }
    a.alu_imm(x64::Alu::Sub, Quad, Rm::Reg(RSP), FRAME);
    a.mov_to(Quad, mem(RSP, CONTEXT), RDI);
    for (field, slot) in [
        (offset_of!(Context, reads), READS),
        (offset_of!(Context, writes), WRITES),
        (offset_of!(Context, jumps), JUMPS),
    ] {
        a.mov(Quad, RAX, mem(RDI, field));
        a.mov_to(Quad, mem(RSP, slot), RAX);
    }
    a.mov(Quad, BUDGET, mem(RDI, offset_of!(Context, budget)));
    a.mov_imm_to(Long, mem(RSP, FP_USED), 0);
    a.mov(Quad, STATE, mem(RDI, offset_of!(Context, cpu)));
    a.mov(Quad, T1, Rm::Reg(RSI));
    a.mov_imm(FLAGS, 0);
    let mut last = 15;
    for (field, bit) in PSR_FLAGS {
        a.shift(Shift::Shl, Long, Rm::Reg(FLAGS), (last - bit).max(1));
        a.extend(false, Byte, T2, mem(STATE, CPSR + field));
        a.alu(x64::Alu::Or, Long, FLAGS, Rm::Reg(T2));
        last = bit;
    }
    for (host, state) in guest_registers() {
        a.mov(Long, host, state);
    }
    a.jmp_rm(Rm::Reg(T1));

    let leave_to_address = a.here();
    let leave = a.label();
    a.mov(Long, T1, Rm::Reg(T0));
    a.alu_imm(x64::Alu::And, Long, Rm::Reg(T1), !1);
    a.mov_to(Long, mem(STATE, REGS + 60), T1);
    a.alu_imm(x64::Alu::And, Long, Rm::Reg(T0), 1);
    a.mov_to(Byte, mem(STATE, CPSR + offset_of!(Psr, t)), T0);
    a.mov(Quad, T2, mem(RSP, CONTEXT));
    a.mov_imm_to(Quad, mem(T2, offset_of!(Context, site)), 0);
    a.jmp(leave);

    // The return address of the call is the place of the two words.
    let leave_at_instruction = a.here();
    a.pop(T1);
    a.mov(Quad, T2, mem(RSP, CONTEXT));
    a.mov_to(Quad, mem(T2, offset_of!(Context, site)), T1);
    a.bind(leave);
    for (host, state) in guest_registers() {
        a.mov_to(Long, state, host);
    }
    for (field, bit) in PSR_FLAGS {
        a.mov(Long, T1, Rm::Reg(FLAGS));
        if bit != 0 {
            a.shift(Shift::Shr, Long, Rm::Reg(T1), bit);
        }
        a.alu_imm(x64::Alu::And, Long, Rm::Reg(T1), 1);
        a.mov_to(Byte, mem(STATE, CPSR + field), T1);
    }
    a.mov_to(Quad, mem(T2, offset_of!(Context, budget)), BUDGET);
    mxcsr_to_fpscr(&mut a, 0, T1);
    a.alu_imm(x64::Alu::Add, Quad, Rm::Reg(RSP), FRAME);
    for register in SAVED.into_iter().rev() {
        a.pop(register);
    }
    a.ret();

    // The return address of the call of the stub is on the stack.
    let frame = |slot: usize| mem(RSP, slot + 8);
    let mut call = |store: bool| {
        let start = a.here();
        for (host, state) in guest_registers() {
            a.mov_to(Long, state, host);
        }
        a.mov_to(Quad, frame(SAVED_FLAGS), FLAGS);
        a.mov(Quad, RDI, frame(CONTEXT));
        a.mov_to(Quad, mem(RDI, offset_of!(Context, budget)), BUDGET);
        mxcsr_to_fpscr(&mut a, 8, RAX);
        a.mov(Long, RSI, Rm::Reg(T0));
        // T1 and T2 are the registers of the third and the fourth argument.
        let function = if store {
            offset_of!(Context, store)
        } else {
            a.mov(Long, RDX, Rm::Reg(T2));
            offset_of!(Context, load)
        };
        // RSP is a multiple of 16 at a call.
        a.alu_imm(x64::Alu::Sub, Quad, Rm::Reg(RSP), 8);
        a.call_rm(mem(RDI, function));
        a.alu_imm(x64::Alu::Add, Quad, Rm::Reg(RSP), 8);
        // The function can change the exceptions in the MXCSR.
        let no_fp = a.label();
        a.alu_imm(x64::Alu::Cmp, Long, frame(FP_USED), 0);
        a.jcc(x64::E, no_fp);
        a.mov_imm_to(Long, frame(MXCSR), MXCSR_DEFAULT);
        a.ldmxcsr(frame(MXCSR));
        a.bind(no_fp);
        a.mov(Long, T0, Rm::Reg(RAX));
        a.mov(Quad, T1, Rm::Reg(RAX));
        a.shift(Shift::Shr, Quad, Rm::Reg(T1), 32);
        for (host, state) in guest_registers() {
            a.mov(Long, host, state);
        }
        a.mov(Quad, T2, frame(CONTEXT));
        a.mov(Quad, BUDGET, mem(T2, offset_of!(Context, budget)));
        a.mov(Quad, FLAGS, frame(SAVED_FLAGS));
        a.ret();
        start
    };
    let (call_load, call_store) = (call(false), call(true));
    a.finish();
    (
        a.bytes,
        Stubs {
            enter,
            leave_to_address,
            leave_at_instruction,
            call_load,
            call_store,
        },
    )
}
