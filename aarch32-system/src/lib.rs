//! Blocks of decoded instructions, with a fast interpreter for the common Thumb ones.
//!
//! A block holds the instructions of one page from an entry address. The fast
//! interpreter runs the instructions that it knows and stops at each other
//! one: the caller runs that one with the rax executor.

use std::cell::Cell;

use rax::arm::decoder::{ShiftType, ThumbDecoder};
use rax::arm::execution::{add_with_carry, decode_imm_shift};
use rax::arm::vfp::{self, Fpscr, VfpState};
use rax::arm::{Armv7Cpu, DecodedInsn, Decoder, ExecutionState, Mnemonic, Psr};

mod a64;
pub mod cache;
pub mod jit;
pub mod tlb;
#[cfg(test)]
mod tests;

pub const MAX_INSTRUCTIONS: usize = 64;
pub const PAGE_SIZE: usize = 4096;
const COND_ALWAYS: u8 = 14;
const SP: u8 = 13;
const PC: u8 = 15;

/// A fault of a slow access. The memory keeps the details.
#[derive(Debug)]
pub struct Fault;

pub trait Memory {
    /// An access with no side effect. Returns None when the access needs the slow path.
    fn load<const N: usize>(&self, address: u32) -> Option<[u8; N]>;
    /// Returns false when the access needs the slow path. Nothing is written then.
    fn store<const N: usize>(&mut self, address: u32, bytes: [u8; N]) -> bool;
    /// An access with all its effects. `done` is the number of instructions that the
    /// run completed before the instruction of the access.
    fn load_slow<const N: usize>(&mut self, address: u32, done: usize) -> Result<[u8; N], Fault>;
    fn store_slow<const N: usize>(&mut self, address: u32, bytes: [u8; N], done: usize) -> Result<(), Fault>;
}

/// Code that runs before each instruction of [`Block::run`].
pub trait Hook {
    /// False removes the hook and the register updates that it needs from the interpreter.
    const ACTIVE: bool;
    /// The PC and the IT state of `cpu` are those of the instruction `index`.
    fn before(&mut self, cpu: &Armv7Cpu, block: &Block, index: usize);
}

pub struct NoHook;

impl Hook for NoHook {
    const ACTIVE: bool = false;
    fn before(&mut self, _cpu: &Armv7Cpu, _block: &Block, _index: usize) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// The instruction at this index did not start. The caller must run it.
    Slow(usize),
    /// A memory access of the instruction at this index gave a fault. The registers are
    /// those before the instruction.
    Fault(usize),
    /// The limit, the end of the block or a branch.
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Alu {
    And,
    Eor,
    Sub,
    Rsb,
    Add,
    Adc,
    Sbc,
    Orr,
    Mov,
    Bic,
    Mvn,
    Orn,
    Tst,
    Teq,
    Cmp,
    Cmn,
}

impl Alu {
    fn is_compare(self) -> bool {
        matches!(self, Alu::Tst | Alu::Teq | Alu::Cmp | Alu::Cmn)
    }

    fn has_first_operand(self) -> bool {
        !matches!(self, Alu::Mov | Alu::Mvn)
    }
}

/// The carry flag after a logical operation with an immediate operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carry {
    Keep,
    Clear,
    Set,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    Byte,
    Half,
    Word,
    SignedByte,
    SignedHalf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Index {
    Offset,
    Pre,
    Post,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Extend {
    Sxth,
    Sxtb,
    Uxth,
    Uxtb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FpBinary {
    Add,
    Sub,
    Mul,
    Div,
    Nmul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FpAccumulate {
    Mla,
    Mls,
    Nmla,
    Nmls,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FpUnary {
    Mov,
    Abs,
    Neg,
    Sqrt,
}

/// Conversions between a 32-bit integer in a single-precision register and a
/// floating-point number. The ones to an integer round to zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FpConvert {
    SignedToF32,
    UnsignedToF32,
    SignedToF64,
    UnsignedToF64,
    F32ToSigned,
    F32ToUnsigned,
    F64ToSigned,
    F64ToUnsigned,
}

/// The Advanced SIMD operations on all the bits of D or Q registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdLogic {
    And,
    Bic,
    Orr,
    Orn,
    Eor,
    Bsl,
    Bit,
    Bif,
    /// VMVN of a register: the operand is `m`.
    Not,
}

/// The Advanced SIMD operations on the elements of D or Q registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdLane {
    // The result has the registers and the elements of the operands.
    Cgt,
    Cge,
    Ceq,
    /// Less than zero.
    Clt0,
    Add,
    Sub,
    Mul,
    Mla,
    Mls,
    Qadd,
    Qsub,
    /// Shifts to the right by the immediate value, with no rounding and with it.
    Shr,
    Rshr,
    // The result is a Q register with elements of two times the size.
    Mull,
    Addl,
    Movl,
    // The operand is a Q register with elements of two times the size.
    Rshrn,
    Qmovn,
}

impl SimdLane {
    fn is_long(self) -> bool {
        matches!(self, SimdLane::Mull | SimdLane::Addl | SimdLane::Movl)
    }

    fn is_narrow(self) -> bool {
        matches!(self, SimdLane::Rshrn | SimdLane::Qmovn)
    }

    /// The operations with the operand `m` only: their encodings have no register `n`.
    fn has_one_operand(self) -> bool {
        matches!(
            self,
            SimdLane::Clt0 | SimdLane::Shr | SimdLane::Rshr | SimdLane::Movl | SimdLane::Rshrn | SimdLane::Qmovn
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// The caller runs the instruction.
    Slow,
    Nop,
    AluImm { kind: Alu, flags: bool, carry: Carry, rd: u8, rn: u8, imm: u32 },
    AluReg { kind: Alu, flags: bool, rd: u8, rn: u8, rm: u8, shift: ShiftType, amount: u8 },
    ShiftReg { shift: ShiftType, flags: bool, rd: u8, rn: u8, rm: u8 },
    Movt { rd: u8, imm: u16 },
    /// The base is the PC of the instruction, word aligned, plus 4.
    Adr { rd: u8, offset: i32 },
    /// ADD Rd, PC: the PC of the instruction plus 4.
    AddPc { rd: u8 },
    Extend { kind: Extend, rd: u8, rm: u8, rotation: u8 },
    Clz { rd: u8, rm: u8 },
    Rbit { rd: u8, rm: u8 },
    Rev { rd: u8, rm: u8 },
    Ubfx { rd: u8, rn: u8, lsb: u8, width: u8 },
    Sbfx { rd: u8, rn: u8, lsb: u8, width: u8 },
    Bfi { rd: u8, rn: u8, lsb: u8, width: u8 },
    Bfc { rd: u8, lsb: u8, width: u8 },
    Mul { flags: bool, rd: u8, rn: u8, rm: u8 },
    Mla { rd: u8, rn: u8, rm: u8, ra: u8 },
    Mls { rd: u8, rn: u8, rm: u8, ra: u8 },
    Umull { lo: u8, hi: u8, rn: u8, rm: u8 },
    Umlal { lo: u8, hi: u8, rn: u8, rm: u8 },
    Umaal { lo: u8, hi: u8, rn: u8, rm: u8 },
    Smull { lo: u8, hi: u8, rn: u8, rm: u8 },
    Udiv { rd: u8, rn: u8, rm: u8 },
    Sdiv { rd: u8, rn: u8, rm: u8 },
    Load { width: Width, rt: u8, rn: u8, index: Index, offset: i32 },
    LoadReg { width: Width, rt: u8, rn: u8, rm: u8, shift: u8 },
    LoadLiteral { width: Width, rt: u8, offset: i32 },
    Store { width: Width, rt: u8, rn: u8, index: Index, offset: i32 },
    StoreReg { width: Width, rt: u8, rn: u8, rm: u8, shift: u8 },
    LoadPair { rt: u8, rt2: u8, rn: u8, index: Index, offset: i32 },
    StorePair { rt: u8, rt2: u8, rn: u8, index: Index, offset: i32 },
    /// LDMIA and STMIA.
    LoadMultiple { rn: u8, mask: u16, writeback: bool },
    StoreMultiple { rn: u8, mask: u16, writeback: bool },
    Push { mask: u16 },
    Pop { mask: u16 },
    /// The offsets of the branches are from the PC of the instruction plus 4.
    Branch { offset: i32 },
    BranchCond { cond: u8, offset: i32 },
    BranchLink { offset: i32 },
    /// The A32 BL: the offset is from the PC of the instruction plus 4, as for the others.
    BranchLinkArm { offset: i32 },
    CompareBranch { rn: u8, nonzero: bool, offset: u8 },
    BranchExchange { rm: u8 },
    BranchLinkExchange { rm: u8 },
    It { state: u8 },
    // The VFP operations. A register number is that of an S register, or of a D
    // register when `double` is set.
    FpBinary { kind: FpBinary, double: bool, d: u8, n: u8, m: u8 },
    FpAccumulate { kind: FpAccumulate, double: bool, d: u8, n: u8, m: u8 },
    FpUnary { kind: FpUnary, double: bool, d: u8, m: u8 },
    FpCompare { all_nans_signal: bool, with_zero: bool, double: bool, d: u8, m: u8 },
    FpImmediate { double: bool, d: u8, imm: u8 },
    FpConvert { kind: FpConvert, d: u8, m: u8 },
    FpFromCore { s: u8, rt: u8 },
    FpToCore { rt: u8, s: u8 },
    /// `v` is a D register, or the first of two S registers.
    FpFromCorePair { double: bool, v: u8, rt: u8, rt2: u8 },
    FpToCorePair { double: bool, rt: u8, rt2: u8, v: u8 },
    FpLoad { double: bool, d: u8, rn: u8, offset: i32 },
    /// The base is the PC of the instruction, word aligned, plus 4.
    FpLoadLiteral { double: bool, d: u8, offset: i32 },
    FpStore { double: bool, d: u8, rn: u8, offset: i32 },
    /// VMRS APSR_nzcv, FPSCR.
    FpFlags,
    /// `d`, `n` and `m` are D registers: the first of two for a Q register.
    SimdLogic { kind: SimdLogic, quad: bool, d: u8, n: u8, m: u8 },
    /// `size` is the log2 of the bytes of the smaller element. `quad` is for the
    /// operations with operands and result of the same size.
    SimdLane { kind: SimdLane, size: u8, quad: bool, unsigned: bool, d: u8, n: u8, m: u8, imm: u8 },
    /// VLD1 and VST1 of `count` D registers with no alignment. `rm` is 15 for no
    /// writeback and 13 for a writeback of the size of the transfer.
    /// `align` is the alignment in bytes that the encoding asks for.
    SimdLoad { d: u8, count: u8, rn: u8, rm: u8, align: u8 },
    SimdStore { d: u8, count: u8, rn: u8, rm: u8, align: u8 },
    /// VMOV of an immediate: the value is at `constant` in the constants of the block.
    SimdImmediate { quad: bool, d: u8, constant: u16 },
    /// VLDM, VSTM, VPUSH and VPOP of `count` D registers. `decrement` is the form that
    /// goes down from the base before the access; it always has a writeback.
    FpLoadMultiple { d: u8, count: u8, rn: u8, decrement: bool, writeback: bool },
    FpStoreMultiple { d: u8, count: u8, rn: u8, decrement: bool, writeback: bool },
    /// TBB and TBH. `rn` is 15 for a table after the instruction.
    TableBranch { rn: u8, rm: u8, halfword: bool },
    /// BLX to A32 code: the offset is from the PC of the instruction plus 4, word aligned.
    BranchLinkToArm { offset: i32 },
}

impl Op {
    fn is_fp(self) -> bool {
        matches!(
            self,
            Op::FpBinary { .. }
                | Op::FpAccumulate { .. }
                | Op::FpUnary { .. }
                | Op::FpCompare { .. }
                | Op::FpImmediate { .. }
                | Op::FpConvert { .. }
                | Op::FpFromCore { .. }
                | Op::FpToCore { .. }
                | Op::FpFromCorePair { .. }
                | Op::FpToCorePair { .. }
                | Op::FpLoad { .. }
                | Op::FpLoadLiteral { .. }
                | Op::FpStore { .. }
                | Op::FpFlags
                | Op::SimdLogic { .. }
                | Op::SimdLane { .. }
                | Op::SimdLoad { .. }
                | Op::SimdStore { .. }
                | Op::SimdImmediate { .. }
                | Op::FpLoadMultiple { .. }
                | Op::FpStoreMultiple { .. }
        )
    }

    fn can_branch(self) -> bool {
        matches!(
            self,
            Op::Branch { .. }
                | Op::BranchCond { .. }
                | Op::BranchLink { .. }
                | Op::BranchLinkArm { .. }
                | Op::BranchLinkToArm { .. }
                | Op::TableBranch { .. }
                | Op::CompareBranch { .. }
                | Op::BranchExchange { .. }
                | Op::BranchLinkExchange { .. }
        ) || matches!(self, Op::Pop { mask } if mask & 1 << PC != 0)
    }
}

#[derive(Clone, Copy)]
struct Step {
    op: Op,
    cond: u8,
    /// ITSTATE before the instruction.
    it_state: u8,
    /// The offset of the instruction in its page.
    offset: u16,
}

pub struct Block {
    steps: Vec<Step>,
    /// For each instruction, the `fp_controls` number for which the executor completed it,
    /// or 0. The FP access checks of the executor pass again for this number.
    admitted: Vec<Cell<u64>>,
    /// The instructions for the executor: an Advanced SIMD one has its A32 encoding, or
    /// else the executor makes this copy at each run.
    executor_insns: Vec<DecodedInsn>,
    /// The values of the SimdImmediate operations.
    constants: Vec<u64>,
    insns: Vec<DecodedInsn>,
    end_offset: u16,
    end_it_state: u8,
    thumb: bool,
    tier: Cell<jit::Tier>,
    /// The compilations of the block for more FP instructions that the executor accepted.
    fp_compilations: Cell<u8>,
}

fn bits(value: u32, low: u32, count: u32) -> u32 {
    (value >> low) & ((1 << count) - 1)
}

fn reg(value: u32, low: u32) -> u8 {
    bits(value, low, 4) as u8
}

fn sign_extend(value: u32, width: u32) -> i32 {
    ((value << (32 - width)) as i32) >> (32 - width)
}

fn any_pc(registers: &[u8]) -> bool {
    registers.contains(&PC)
}

/// True when a register is the SP or the PC: most T32 encodings have no defined result then.
fn any_sp_or_pc(registers: &[u8]) -> bool {
    registers.iter().any(|&register| register == SP || register == PC)
}

/// The data-processing operations that take the SP as an operand or as the result.
fn sp_is_valid(kind: Alu) -> bool {
    matches!(kind, Alu::Add | Alu::Sub | Alu::Cmp | Alu::Cmn | Alu::Mov)
}

fn alu_reg(kind: Alu, flags: bool, rd: u8, rn: u8, rm: u8) -> Op {
    Op::AluReg { kind, flags, rd, rn, rm, shift: ShiftType::LSL, amount: 0 }
}

fn alu_imm(kind: Alu, flags: bool, rd: u8, rn: u8, imm: u32) -> Op {
    Op::AluImm { kind, flags, carry: Carry::Keep, rd, rn, imm }
}

fn thumb16(hw: u32, in_it: bool) -> Op {
    let flags = !in_it;
    let low = |shift: u32| bits(hw, shift, 3) as u8;
    let imm8 = hw & 0xff;
    let imm5 = bits(hw, 6, 5);
    match hw >> 11 {
        0b00000..=0b00010 => {
            let (shift, amount) = decode_imm_shift((hw >> 11) as u8, imm5 as u8);
            Op::AluReg { kind: Alu::Mov, flags, rd: low(0), rn: 0, rm: low(3), shift, amount }
        }
        0b00011 => {
            let kind = if hw & 0x200 != 0 { Alu::Sub } else { Alu::Add };
            if hw & 0x400 != 0 {
                alu_imm(kind, flags, low(0), low(3), bits(hw, 6, 3))
            } else {
                alu_reg(kind, flags, low(0), low(3), low(6))
            }
        }
        0b00100 => alu_imm(Alu::Mov, flags, low(8), 0, imm8),
        0b00101 => alu_imm(Alu::Cmp, true, 0, low(8), imm8),
        0b00110 => alu_imm(Alu::Add, flags, low(8), low(8), imm8),
        0b00111 => alu_imm(Alu::Sub, flags, low(8), low(8), imm8),
        0b01000 if hw & 0x400 == 0 => {
            let (rdn, rm) = (low(0), low(3));
            let shift_reg = |shift| Op::ShiftReg { shift, flags, rd: rdn, rn: rdn, rm };
            match bits(hw, 6, 4) {
                0 => alu_reg(Alu::And, flags, rdn, rdn, rm),
                1 => alu_reg(Alu::Eor, flags, rdn, rdn, rm),
                2 => shift_reg(ShiftType::LSL),
                3 => shift_reg(ShiftType::LSR),
                4 => shift_reg(ShiftType::ASR),
                5 => alu_reg(Alu::Adc, flags, rdn, rdn, rm),
                6 => alu_reg(Alu::Sbc, flags, rdn, rdn, rm),
                7 => shift_reg(ShiftType::ROR),
                8 => alu_reg(Alu::Tst, true, 0, rdn, rm),
                9 => alu_imm(Alu::Rsb, flags, rdn, rm, 0),
                10 => alu_reg(Alu::Cmp, true, 0, rdn, rm),
                11 => alu_reg(Alu::Cmn, true, 0, rdn, rm),
                12 => alu_reg(Alu::Orr, flags, rdn, rdn, rm),
                13 => Op::Mul { flags, rd: rdn, rn: rm, rm: rdn },
                14 => alu_reg(Alu::Bic, flags, rdn, rdn, rm),
                _ => alu_reg(Alu::Mvn, flags, rdn, 0, rm),
            }
        }
        0b01000 => {
            let rm = reg(hw, 3);
            let rdn = (bits(hw, 0, 3) | bits(hw, 7, 1) << 3) as u8;
            match bits(hw, 8, 2) {
                3 if rm == PC || hw & 7 != 0 => Op::Slow,
                3 if hw & 0x80 != 0 => Op::BranchLinkExchange { rm },
                3 => Op::BranchExchange { rm },
                0 if rm == PC && rdn != PC => Op::AddPc { rd: rdn },
                _ if any_pc(&[rm, rdn]) => Op::Slow,
                0 => alu_reg(Alu::Add, false, rdn, rdn, rm),
                1 => alu_reg(Alu::Cmp, true, 0, rdn, rm),
                _ => alu_reg(Alu::Mov, false, rdn, 0, rm),
            }
        }
        0b01001 => Op::LoadLiteral { width: Width::Word, rt: low(8), offset: (imm8 * 4) as i32 },
        0b01010 | 0b01011 => {
            let (rt, rn, rm) = (low(0), low(3), low(6));
            let load = |width| Op::LoadReg { width, rt, rn, rm, shift: 0 };
            let store = |width| Op::StoreReg { width, rt, rn, rm, shift: 0 };
            match bits(hw, 9, 3) {
                0 => store(Width::Word),
                1 => store(Width::Half),
                2 => store(Width::Byte),
                3 => load(Width::SignedByte),
                4 => load(Width::Word),
                5 => load(Width::Half),
                6 => load(Width::Byte),
                _ => load(Width::SignedHalf),
            }
        }
        0b01100..=0b10001 => {
            let (width, scale) = match hw >> 12 {
                0b0110 => (Width::Word, 4),
                0b0111 => (Width::Byte, 1),
                _ => (Width::Half, 2),
            };
            let (rt, rn, offset) = (low(0), low(3), (imm5 * scale) as i32);
            if hw & 0x800 != 0 {
                Op::Load { width, rt, rn, index: Index::Offset, offset }
            } else {
                Op::Store { width, rt, rn, index: Index::Offset, offset }
            }
        }
        0b10010 => {
            Op::Store { width: Width::Word, rt: low(8), rn: 13, index: Index::Offset, offset: (imm8 * 4) as i32 }
        }
        0b10011 => {
            Op::Load { width: Width::Word, rt: low(8), rn: 13, index: Index::Offset, offset: (imm8 * 4) as i32 }
        }
        0b10100 => Op::Adr { rd: low(8), offset: (imm8 * 4) as i32 },
        0b10101 => alu_imm(Alu::Add, false, low(8), 13, imm8 * 4),
        0b10110 | 0b10111 => match bits(hw, 8, 4) {
            0 => {
                let kind = if hw & 0x80 != 0 { Alu::Sub } else { Alu::Add };
                alu_imm(kind, false, 13, 13, (hw & 0x7f) * 4)
            }
            1 | 3 | 9 | 11 => Op::CompareBranch {
                rn: low(0),
                nonzero: hw & 0x800 != 0,
                offset: (bits(hw, 9, 1) << 6 | bits(hw, 3, 5) << 1) as u8,
            },
            2 => {
                let kind = match bits(hw, 6, 2) {
                    0 => Extend::Sxth,
                    1 => Extend::Sxtb,
                    2 => Extend::Uxth,
                    _ => Extend::Uxtb,
                };
                Op::Extend { kind, rd: low(0), rm: low(3), rotation: 0 }
            }
            4 | 5 if hw & 0x1ff != 0 => Op::Push { mask: (imm8 | bits(hw, 8, 1) << 14) as u16 },
            12 | 13 if hw & 0x1ff != 0 => Op::Pop { mask: (imm8 | bits(hw, 8, 1) << 15) as u16 },
            10 if bits(hw, 6, 2) == 0 => Op::Rev { rd: low(0), rm: low(3) },
            15 if hw & 0xf != 0 => Op::It { state: imm8 as u8 },
            15 if imm8 == 0 => Op::Nop,
            _ => Op::Slow,
        },
        0b11000 => multiple(false, low(8), imm8 as u16, true),
        0b11001 => multiple(true, low(8), imm8 as u16, true),
        0b11010 | 0b11011 if bits(hw, 8, 4) < 14 => {
            Op::BranchCond { cond: reg(hw, 8), offset: sign_extend(imm8, 8) * 2 }
        }
        0b11100 => Op::Branch { offset: sign_extend(hw & 0x7ff, 11) * 2 },
        _ => Op::Slow,
    }
}

/// LDMIA or STMIA with the registers `mask`, in the forms with a defined result.
fn multiple(load: bool, rn: u8, mask: u16, writeback: bool) -> Op {
    if mask == 0 || mask & (1 << PC | 1 << SP) != 0 || rn == PC {
        return Op::Slow;
    }
    if load {
        // The loaded value of the base register stays.
        Op::LoadMultiple { rn, mask, writeback: writeback && mask & 1 << rn == 0 }
    } else if writeback && mask & 1 << rn != 0 {
        Op::Slow
    } else {
        Op::StoreMultiple { rn, mask, writeback }
    }
}

/// The offsets from the PC are for the PC of the instruction plus 4, as in Thumb
/// state: this adds the other 4 of the A32 PC.
const ARM_PC_OFFSET: i32 = 4;

/// The index mode and the offset of the A32 loads and stores, from the P, U and W bits.
fn arm_index(raw: u32, imm: u32) -> Option<(Index, i32)> {
    let offset = if raw & 0x0080_0000 != 0 { imm as i32 } else { -(imm as i32) };
    match (raw & 0x0100_0000 != 0, raw & 0x0020_0000 != 0) {
        (true, false) => Some((Index::Offset, offset)),
        (true, true) => Some((Index::Pre, offset)),
        (false, false) => Some((Index::Post, offset)),
        // The unprivileged access (LDRT, STRT).
        (false, true) => None,
    }
}

fn arm_data_processing(raw: u32) -> Op {
    let immediate = raw & 0x0200_0000 != 0;
    let opcode = bits(raw, 21, 4);
    let flags = raw & 0x0010_0000 != 0;
    let (rn, rd, rm) = (reg(raw, 16), reg(raw, 12), reg(raw, 0));
    if !flags && (8..12).contains(&opcode) {
        let imm16 = bits(raw, 16, 4) << 12 | raw & 0xfff;
        return match opcode {
            _ if !immediate || rd == PC => Op::Slow,
            8 => alu_imm(Alu::Mov, false, rd, 0, imm16),
            10 => Op::Movt { rd, imm: imm16 as u16 },
            _ => Op::Slow,
        };
    }
    let kind = match opcode {
        0 => Alu::And,
        1 => Alu::Eor,
        2 => Alu::Sub,
        3 => Alu::Rsb,
        4 => Alu::Add,
        5 => Alu::Adc,
        6 => Alu::Sbc,
        8 => Alu::Tst,
        9 => Alu::Teq,
        10 => Alu::Cmp,
        11 => Alu::Cmn,
        12 => Alu::Orr,
        13 => Alu::Mov,
        14 => Alu::Bic,
        15 => Alu::Mvn,
        _ => return Op::Slow,
    };
    if !kind.is_compare() && rd == PC {
        return Op::Slow;
    }
    if immediate {
        let rotation = bits(raw, 8, 4) * 2;
        let imm = (raw & 0xff).rotate_right(rotation);
        if rn == PC && kind.has_first_operand() {
            return match kind {
                Alu::Add if !flags => Op::Adr { rd, offset: imm as i32 + ARM_PC_OFFSET },
                Alu::Sub if !flags => Op::Adr { rd, offset: ARM_PC_OFFSET - imm as i32 },
                _ => Op::Slow,
            };
        }
        let carry = match (rotation, imm >> 31) {
            (0, _) => Carry::Keep,
            (_, 0) => Carry::Clear,
            _ => Carry::Set,
        };
        return Op::AluImm { kind, flags, carry, rd, rn, imm };
    }
    // Bit 4 selects a shift by a register, or a multiply or an extra load with bit 7.
    if raw & 0x10 != 0 || rm == PC || (rn == PC && kind.has_first_operand()) {
        return Op::Slow;
    }
    let (shift, amount) = decode_imm_shift(bits(raw, 5, 2) as u8, bits(raw, 7, 5) as u8);
    Op::AluReg { kind, flags, rd, rn, rm, shift, amount }
}

fn arm(raw: u32) -> Op {
    if raw >> 28 == 0xf {
        return Op::Slow;
    }
    let (rn, rt, rm) = (reg(raw, 16), reg(raw, 12), reg(raw, 0));
    let load = raw & 0x0010_0000 != 0;
    let (p, u, w) = (raw & 0x0100_0000 != 0, raw & 0x0080_0000 != 0, raw & 0x0020_0000 != 0);
    match bits(raw, 25, 3) {
        0b000 if raw & 0x0ff0_00f0 == 0x0000_0090 => {
            let (rd, rs) = (rn, reg(raw, 8));
            if any_pc(&[rd, rm, rs]) { Op::Slow } else { Op::Mul { flags: false, rd, rn: rm, rm: rs } }
        }
        0b000 if raw & 0x0ff0_00f0 == 0x0020_0090 => {
            let (rd, ra, rs) = (rn, rt, reg(raw, 8));
            if any_pc(&[rd, ra, rm, rs]) { Op::Slow } else { Op::Mla { rd, rn: rm, rm: rs, ra } }
        }
        0b000 if raw & 0x0f10_00f0 == 0x0000_0090 => {
            let (hi, lo, rs) = (rn, rt, reg(raw, 8));
            if any_pc(&[hi, lo, rm, rs]) || hi == lo {
                return Op::Slow;
            }
            match bits(raw, 21, 3) {
                0b010 => Op::Umaal { lo, hi, rn: rm, rm: rs },
                0b100 => Op::Umull { lo, hi, rn: rm, rm: rs },
                0b101 => Op::Umlal { lo, hi, rn: rm, rm: rs },
                0b110 => Op::Smull { lo, hi, rn: rm, rm: rs },
                _ => Op::Slow,
            }
        }
        0b000 if raw & 0x0fff_fff0 == 0x012f_ff10 && rm != PC => Op::BranchExchange { rm },
        0b000 if raw & 0x0fff_0ff0 == 0x016f_0f10 && !any_pc(&[rt, rm]) => Op::Clz { rd: rt, rm },
        0b011 if raw & 0x0f8f_03f0 == 0x068f_0070 && !any_pc(&[rt, rm]) => {
            let kind = match bits(raw, 20, 3) {
                0b010 => Extend::Sxtb,
                0b011 => Extend::Sxth,
                0b110 => Extend::Uxtb,
                0b111 => Extend::Uxth,
                _ => return Op::Slow,
            };
            Op::Extend { kind, rd: rt, rm, rotation: (bits(raw, 10, 2) * 8) as u8 }
        }
        0b000 if raw & 0x0040_0090 == 0x0040_0090 => {
            // The halfword and signed loads and stores with an immediate offset.
            let width = match (bits(raw, 5, 2), load) {
                (0b01, _) => Width::Half,
                (0b10, true) => Width::SignedByte,
                (0b11, true) => Width::SignedHalf,
                _ => return Op::Slow,
            };
            let Some((index, offset)) = arm_index(raw, bits(raw, 8, 4) << 4 | raw & 0xf) else {
                return Op::Slow;
            };
            if any_pc(&[rn, rt]) {
                return Op::Slow;
            }
            thumb32_single(load, width, rt, rn, index, offset)
        }
        0b000 | 0b001 if raw & 0x0e00_0090 != 0x0000_0090 => arm_data_processing(raw),
        0b010 => {
            let width = if raw & 0x0040_0000 != 0 { Width::Byte } else { Width::Word };
            let Some((index, offset)) = arm_index(raw, raw & 0xfff) else {
                return Op::Slow;
            };
            if rt == PC {
                Op::Slow
            } else if rn != PC {
                thumb32_single(load, width, rt, rn, index, offset)
            } else if load && index == Index::Offset {
                Op::LoadLiteral { width, rt, offset: offset + ARM_PC_OFFSET }
            } else {
                Op::Slow
            }
        }
        0b011 if raw & 0x10 == 0 && p && u && !w && bits(raw, 5, 2) == 0 && !any_pc(&[rn, rt, rm]) => {
            let width = if raw & 0x0040_0000 != 0 { Width::Byte } else { Width::Word };
            let shift = bits(raw, 7, 5) as u8;
            if load { Op::LoadReg { width, rt, rn, rm, shift } } else { Op::StoreReg { width, rt, rn, rm, shift } }
        }
        0b100 if raw & 0x0040_0000 == 0 => {
            let mask = raw as u16;
            match (p, u, load) {
                (true, false, false) if rn == SP && w && mask.count_ones() >= 2 && mask >> 13 & 0b101 == 0 => {
                    Op::Push { mask }
                }
                (false, true, true) if rn == SP && w && mask.count_ones() >= 2 && mask & 1 << SP == 0 => {
                    Op::Pop { mask }
                }
                (false, true, _) => multiple(load, rn, mask, w),
                _ => Op::Slow,
            }
        }
        0b101 => {
            let offset = sign_extend(raw & 0x00ff_ffff, 24) * 4 + ARM_PC_OFFSET;
            if raw & 0x0100_0000 != 0 { Op::BranchLinkArm { offset } } else { Op::Branch { offset } }
        }
        _ => Op::Slow,
    }
}

/// ThumbExpandImm_C. Returns None for the encodings with no defined result.
fn thumb_expand_imm(imm12: u32) -> Option<(u32, Carry)> {
    let imm8 = imm12 & 0xff;
    if imm12 >> 10 != 0 {
        let value = (0x80 | imm8 & 0x7f).rotate_right(imm12 >> 7);
        let carry = if value >> 31 != 0 { Carry::Set } else { Carry::Clear };
        return Some((value, carry));
    }
    let value = match imm12 >> 8 {
        0 => imm8,
        _ if imm8 == 0 => return None,
        1 => imm8 << 16 | imm8,
        2 => imm8 << 24 | imm8 << 8,
        _ => imm8 * 0x0101_0101,
    };
    Some((value, Carry::Keep))
}

/// The operation of a T32 data-processing instruction, from its opcode field.
fn thumb32_alu(opcode: u32, flags: bool, rd: u8, rn: u8) -> Option<Alu> {
    Some(match opcode {
        0 if rd == PC && flags => Alu::Tst,
        0 => Alu::And,
        1 => Alu::Bic,
        2 if rn == PC => Alu::Mov,
        2 => Alu::Orr,
        3 if rn == PC => Alu::Mvn,
        3 => Alu::Orn,
        4 if rd == PC && flags => Alu::Teq,
        4 => Alu::Eor,
        8 if rd == PC && flags => Alu::Cmn,
        8 => Alu::Add,
        10 => Alu::Adc,
        11 => Alu::Sbc,
        13 if rd == PC && flags => Alu::Cmp,
        13 => Alu::Sub,
        14 => Alu::Rsb,
        _ => return None,
    })
}

/// A load or a store of one register, in the T32 encodings with an immediate offset.
fn thumb32_single(load: bool, width: Width, rt: u8, rn: u8, index: Index, offset: i32) -> Op {
    if index != Index::Offset && rn == rt {
        return Op::Slow;
    }
    if load {
        Op::Load { width, rt, rn, index, offset }
    } else {
        Op::Store { width, rt, rn, index, offset }
    }
}

/// The index mode and the offset of the T32 encodings with P, U and W bits.
fn thumb32_index(p: bool, u: bool, w: bool, imm: u32) -> Option<(Index, i32)> {
    let offset = if u { imm as i32 } else { -(imm as i32) };
    match (p, w) {
        (true, false) => Some((Index::Offset, offset)),
        (true, true) => Some((Index::Pre, offset)),
        (false, true) => Some((Index::Post, offset)),
        (false, false) => None,
    }
}

fn thumb32(raw: u32) -> Op {
    let (hw1, hw2) = (raw >> 16, raw & 0xffff);
    let rn = reg(hw1, 0);
    let rd = reg(hw2, 8);
    let rm = reg(hw2, 0);
    let rt = reg(hw2, 12);
    let flags = hw1 & 0x10 != 0;
    let imm3_imm8 = bits(hw2, 12, 3) << 8 | hw2 & 0xff;
    let imm12 = bits(hw1, 10, 1) << 11 | imm3_imm8;

    if hw1 >> 11 == 0b11110 && hw2 & 0x8000 == 0 {
        if hw1 & 0x200 == 0 {
            let Some(kind) = thumb32_alu(bits(hw1, 5, 4), flags, rd, rn) else {
                return Op::Slow;
            };
            let Some((imm, carry)) = thumb_expand_imm(imm12) else {
                return Op::Slow;
            };
            if (!kind.is_compare() && rd == PC) || (kind.has_first_operand() && rn == PC) {
                return Op::Slow;
            }
            if !sp_is_valid(kind) && (rd == SP || rn == SP) {
                return Op::Slow;
            }
            return Op::AluImm { kind, flags, carry, rd, rn, imm };
        }
        let add_or_sub = matches!(bits(hw1, 4, 5), 0b00000 | 0b01010);
        if rd == PC || (!add_or_sub && (rd == SP || rn == SP)) {
            return Op::Slow;
        }
        let lsb = (bits(hw2, 12, 3) << 2 | bits(hw2, 6, 2)) as u8;
        let low5 = (hw2 & 0x1f) as u8;
        return match bits(hw1, 4, 5) {
            0b00000 if rn == PC => Op::Adr { rd, offset: imm12 as i32 },
            0b00000 => alu_imm(Alu::Add, false, rd, rn, imm12),
            0b00100 => alu_imm(Alu::Mov, false, rd, 0, u32::from(rn) << 12 | imm12),
            0b01010 if rn == PC => Op::Adr { rd, offset: -(imm12 as i32) },
            0b01010 => alu_imm(Alu::Sub, false, rd, rn, imm12),
            0b01100 => Op::Movt { rd, imm: (u32::from(rn) << 12 | imm12) as u16 },
            0b10100 if rn != PC && lsb + low5 < 32 => Op::Sbfx { rd, rn, lsb, width: low5 + 1 },
            0b11100 if rn != PC && lsb + low5 < 32 => Op::Ubfx { rd, rn, lsb, width: low5 + 1 },
            0b10110 if low5 < lsb => Op::Slow,
            0b10110 if rn == PC => Op::Bfc { rd, lsb, width: low5 - lsb + 1 },
            0b10110 => Op::Bfi { rd, rn, lsb, width: low5 - lsb + 1 },
            _ => Op::Slow,
        };
    }

    if hw1 >> 11 == 0b11110 {
        let (s, j1, j2) = (bits(hw1, 10, 1), bits(hw2, 13, 1), bits(hw2, 11, 1));
        let imm11 = hw2 & 0x7ff;
        let far = || {
            let (i1, i2) = (!(j1 ^ s) & 1, !(j2 ^ s) & 1);
            sign_extend(s << 24 | i1 << 23 | i2 << 22 | bits(hw1, 0, 10) << 12 | imm11 << 1, 25)
        };
        return match (hw2 & 0x4000 != 0, hw2 & 0x1000 != 0) {
            (false, false) if bits(hw1, 6, 4) < 14 => {
                let imm = s << 20 | j2 << 19 | j1 << 18 | bits(hw1, 0, 6) << 12 | imm11 << 1;
                Op::BranchCond { cond: reg(hw1, 6), offset: sign_extend(imm, 21) }
            }
            (false, true) => Op::Branch { offset: far() },
            (true, true) => Op::BranchLink { offset: far() },
            (true, false) if hw2 & 1 == 0 => Op::BranchLinkToArm { offset: far() },
            _ => Op::Slow,
        };
    }

    if hw1 >> 9 == 0b1110100 && hw1 & 0x40 == 0 {
        // Only the forms that the assembler writes as PUSH and POP of two registers or more.
        let (wback, load) = (hw1 & 0x20 != 0, hw1 & 0x10 != 0);
        let mask = hw2 as u16;
        if mask.count_ones() < 2 || mask & 1 << 13 != 0 {
            return Op::Slow;
        }
        if rn != 13 {
            // LDMIA and STMIA. A base register in the list has no defined result with a writeback.
            let defined = bits(hw1, 7, 2) == 0b01 && !(wback && mask & 1 << rn != 0);
            return if defined { multiple(load, rn, mask, wback) } else { Op::Slow };
        }
        if !wback {
            return Op::Slow;
        }
        return match (bits(hw1, 7, 2), load) {
            (0b10, false) if mask & 1 << PC == 0 => Op::Push { mask },
            (0b01, true) if mask >> 14 != 0b11 => Op::Pop { mask },
            _ => Op::Slow,
        };
    }

    if hw1 >> 9 == 0b1110100 {
        let (p, u, w) = (hw1 & 0x100 != 0, hw1 & 0x80 != 0, hw1 & 0x20 != 0);
        let load = hw1 & 0x10 != 0;
        if raw & 0xfff0_ffe0 == 0xe8d0_f000 {
            return if any_sp_or_pc(&[rm]) || rn == SP {
                Op::Slow
            } else {
                Op::TableBranch { rn, rm, halfword: hw2 & 0x10 != 0 }
            };
        }
        let Some((index, offset)) = thumb32_index(p, u, w, (hw2 & 0xff) * 4) else {
            return Op::Slow;
        };
        let rt2 = rd;
        if rn == PC || any_sp_or_pc(&[rt, rt2]) || (load && rt == rt2) {
            return Op::Slow;
        }
        if index != Index::Offset && (rn == rt || rn == rt2) {
            return Op::Slow;
        }
        return if load {
            Op::LoadPair { rt, rt2, rn, index, offset }
        } else {
            Op::StorePair { rt, rt2, rn, index, offset }
        };
    }

    if hw1 >> 9 == 0b1110101 {
        if hw2 & 0x8000 != 0 {
            return Op::Slow;
        }
        let Some(kind) = thumb32_alu(bits(hw1, 5, 4), flags, rd, rn) else {
            return Op::Slow;
        };
        if (!kind.is_compare() && rd == PC) || (kind.has_first_operand() && rn == PC) {
            return Op::Slow;
        }
        if any_sp_or_pc(&[rm]) || (!sp_is_valid(kind) && (rd == SP || rn == SP)) {
            return Op::Slow;
        }
        let imm5 = bits(hw2, 12, 3) << 2 | bits(hw2, 6, 2);
        let (shift, amount) = decode_imm_shift(bits(hw2, 4, 2) as u8, imm5 as u8);
        return Op::AluReg { kind, flags, rd, rn, rm, shift, amount };
    }

    if hw1 >> 9 == 0b1111100 && !(hw1 & 0x100 != 0 && hw1 & 0x10 == 0) {
        let load = hw1 & 0x10 != 0;
        let signed = hw1 & 0x100 != 0;
        let width = match (bits(hw1, 5, 2), signed) {
            (0, false) => Width::Byte,
            (1, false) => Width::Half,
            (2, false) => Width::Word,
            (0, true) => Width::SignedByte,
            (1, true) => Width::SignedHalf,
            _ => return Op::Slow,
        };
        if rt == PC || (rt == SP && width != Width::Word) {
            return Op::Slow;
        }
        let positive = hw1 & 0x80 != 0;
        if rn == PC {
            if !load {
                return Op::Slow;
            }
            let imm = (hw2 & 0xfff) as i32;
            return Op::LoadLiteral { width, rt, offset: if positive { imm } else { -imm } };
        }
        if positive {
            return thumb32_single(load, width, rt, rn, Index::Offset, (hw2 & 0xfff) as i32);
        }
        if hw2 & 0x800 != 0 {
            let (p, u, w) = (hw2 & 0x400 != 0, hw2 & 0x200 != 0, hw2 & 0x100 != 0);
            // P and U with no W is the unprivileged access (LDRT, STRT).
            if p && u && !w {
                return Op::Slow;
            }
            let Some((index, offset)) = thumb32_index(p, u, w, hw2 & 0xff) else {
                return Op::Slow;
            };
            return thumb32_single(load, width, rt, rn, index, offset);
        }
        if hw2 & 0xfc0 != 0 || any_sp_or_pc(&[rm]) {
            return Op::Slow;
        }
        let shift = bits(hw2, 4, 2) as u8;
        return if load {
            Op::LoadReg { width, rt, rn, rm, shift }
        } else {
            Op::StoreReg { width, rt, rn, rm, shift }
        };
    }

    if hw1 >> 8 == 0b1111_1010 && hw2 >> 12 == 0xf {
        if any_sp_or_pc(&[rd, rm]) {
            return Op::Slow;
        }
        if hw1 & 0x80 == 0 {
            if hw2 & 0xf0 == 0 && !any_sp_or_pc(&[rn]) {
                let shift = ShiftType::from_bits(bits(hw1, 5, 2) as u8);
                return Op::ShiftReg { shift, flags, rd, rn, rm };
            }
            if hw2 & 0xc0 != 0x80 || rn != PC {
                return Op::Slow;
            }
            let kind = match bits(hw1, 4, 3) {
                0 => Extend::Sxth,
                1 => Extend::Uxth,
                4 => Extend::Sxtb,
                5 => Extend::Uxtb,
                _ => return Op::Slow,
            };
            return Op::Extend { kind, rd, rm, rotation: (bits(hw2, 4, 2) * 8) as u8 };
        }
        if hw1 & 0x40 != 0 || bits(hw2, 6, 2) != 0b10 || rn != rm {
            return Op::Slow;
        }
        return match (bits(hw1, 4, 2), bits(hw2, 4, 2)) {
            (1, 0) => Op::Rev { rd, rm },
            (1, 2) => Op::Rbit { rd, rm },
            (3, 0) => Op::Clz { rd, rm },
            _ => Op::Slow,
        };
    }

    if hw1 >> 7 == 0b1_1111_0110 {
        let ra = rt;
        if any_sp_or_pc(&[rd, rn, rm]) || ra == SP {
            return Op::Slow;
        }
        return match (bits(hw1, 4, 3), bits(hw2, 4, 4)) {
            (0, 0) if ra == PC => Op::Mul { flags: false, rd, rn, rm },
            (0, 0) => Op::Mla { rd, rn, rm, ra },
            (0, 1) if ra != PC => Op::Mls { rd, rn, rm, ra },
            _ => Op::Slow,
        };
    }

    if hw1 >> 7 == 0b1_1111_0111 {
        let (lo, hi) = (rt, rd);
        if any_sp_or_pc(&[rd, rn, rm]) || lo == SP {
            return Op::Slow;
        }
        return match (bits(hw1, 4, 3), bits(hw2, 4, 4)) {
            (0, 0) if lo != PC && lo != hi => Op::Smull { lo, hi, rn, rm },
            (2, 0) if lo != PC && lo != hi => Op::Umull { lo, hi, rn, rm },
            (1, 0xf) if lo == PC => Op::Sdiv { rd, rn, rm },
            (3, 0xf) if lo == PC => Op::Udiv { rd, rn, rm },
            _ => Op::Slow,
        };
    }

    Op::Slow
}

fn executor_insn(insn: &DecodedInsn) -> DecodedInsn {
    let mut insn = insn.clone();
    if insn.state.is_thumb() && insn.size == 4 {
        insn.raw = ThumbDecoder::a32_equivalent(insn.raw).unwrap_or(insn.raw);
    }
    insn
}

/// VLD1 or VST1 of full registers, in its T32 encoding.
fn thumb_simd_memory(insn: &DecodedInsn) -> Op {
    let raw = insn.raw;
    let count = match bits(raw, 8, 4) {
        0b0111 => 1,
        0b1010 => 2,
        0b0110 => 3,
        0b0010 => 4,
        _ => return Op::Slow,
    };
    let d = (bits(raw, 22, 1) << 4 | bits(raw, 12, 4)) as u8;
    let (rn, rm) = (reg(raw, 16), reg(raw, 0));
    let align = match bits(raw, 4, 2) {
        0 => 1,
        bits => 4 << bits,
    };
    if raw & 0xff90_0000 != 0xf900_0000 || d + count > 32 || rn == PC {
        return Op::Slow;
    }
    match insn.mnemonic {
        Mnemonic::VLD1 if raw & 0x20_0000 != 0 => Op::SimdLoad { d, count, rn, rm, align },
        Mnemonic::VST1 if raw & 0x20_0000 == 0 => Op::SimdStore { d, count, rn, rm, align },
        _ => Op::Slow,
    }
}

/// AdvSIMDExpandImm for the encodings that are VMOV. None for the other ones.
fn simd_expand_immediate(op: bool, cmode: u32, imm8: u64) -> Option<u64> {
    let words = |imm32: u64| imm32 << 32 | imm32;
    let halfwords = |imm16: u64| imm16 * 0x0001_0001_0001_0001;
    Some(match (cmode, op) {
        (0b0000 | 0b0010 | 0b0100 | 0b0110, false) => words(imm8 << (cmode * 4)),
        (0b1000 | 0b1010, false) => halfwords(imm8 << ((cmode & 2) * 4)),
        (0b1100, false) => words(imm8 << 8 | 0xff),
        (0b1101, false) => words(imm8 << 16 | 0xffff),
        (0b1110, false) => imm8 * 0x0101_0101_0101_0101,
        (0b1110, true) => (0..8).filter(|bit| imm8 >> bit & 1 != 0).fold(0, |value, bit| value | 0xff << (bit * 8)),
        (0b1111, false) => words(vfp::vfp_expand_imm_f32(imm8 as u8).into()),
        _ => return None,
    })
}

/// An Advanced SIMD operation on integer elements. `raw` is the A32 encoding.
fn simd_lane(insn: &DecodedInsn, raw: u32, quad: bool, d: u8, n: u8, m: u8) -> Option<Op> {
    let unsigned = raw & 0x0100_0000 != 0;
    let size = bits(raw, 20, 2) as u8;
    let opcode = bits(raw, 8, 4);
    let lane = |kind: SimdLane, size: u8, unsigned: bool, imm: u8| {
        let odd = |register: u8| register & 1 != 0;
        let n = if kind.has_one_operand() { m } else { n };
        let bad = (quad && !kind.is_long() && !kind.is_narrow() && (odd(d) || odd(n) || odd(m)))
            || (kind.is_long() && odd(d))
            || (kind.is_narrow() && odd(m));
        (!bad).then_some(Op::SimdLane { kind, size, quad, unsigned, d, n, m, imm })
    };
    if raw & 0xfe80_0000 == 0xf200_0000 {
        // Three registers of the same length.
        let kind = match (insn.mnemonic, opcode, raw & 0x10 != 0, unsigned) {
            (Mnemonic::VQADD, 0b0000, true, _) => SimdLane::Qadd,
            (Mnemonic::VQSUB, 0b0010, true, _) => SimdLane::Qsub,
            (Mnemonic::VCGT, 0b0011, false, _) => SimdLane::Cgt,
            (Mnemonic::VCGE, 0b0011, true, _) => SimdLane::Cge,
            (Mnemonic::VADD, 0b1000, false, false) => SimdLane::Add,
            (Mnemonic::VSUB, 0b1000, false, true) => SimdLane::Sub,
            (Mnemonic::VCEQ, 0b1000, true, true) => SimdLane::Ceq,
            (Mnemonic::VMLA, 0b1001, false, false) if size != 3 => SimdLane::Mla,
            (Mnemonic::VMLS, 0b1001, false, true) if size != 3 => SimdLane::Mls,
            (Mnemonic::VMUL, 0b1001, true, false) if size != 3 => SimdLane::Mul,
            _ => return None,
        };
        // Add, Sub, Ceq, Mul, Mla and Mls give the same bits for signed and unsigned.
        return lane(kind, size, unsigned, 0);
    }
    if raw & 0xfe80_0010 == 0xf280_0010 && raw & 0x00b8_0080 != 0 {
        // Two registers and a shift amount in imm6, with the element size in L:imm6.
        let imm6 = bits(raw, 16, 6);
        let long_element = raw & 0x80 != 0;
        let (size, base) = match imm6 {
            _ if long_element => (3, 64),
            32.. => (2, 64),
            16.. => (1, 32),
            _ => (0, 16),
        };
        return match (insn.mnemonic, opcode) {
            (Mnemonic::VSHR, 0b0000) => lane(SimdLane::Shr, size, unsigned, (base - imm6) as u8),
            (Mnemonic::VRSHR, 0b0010) => lane(SimdLane::Rshr, size, unsigned, (base - imm6) as u8),
            // The element sizes are those of the result: half of the ones of the operand.
            (Mnemonic::VRSHRN, 0b1000) if !long_element && !unsigned && raw & 0x40 != 0 => {
                lane(SimdLane::Rshrn, size, false, (base - imm6) as u8)
            }
            (Mnemonic::VMOVL, 0b1010) if !long_element && raw & 0x40 == 0 && matches!(imm6, 8 | 16 | 32) => {
                lane(SimdLane::Movl, size, unsigned, 0)
            }
            _ => None,
        };
    }
    if raw & 0xfe80_0050 == 0xf280_0000 && size != 3 {
        // Three registers of different lengths.
        return match (insn.mnemonic, opcode) {
            (Mnemonic::VADDL, 0b0000) => lane(SimdLane::Addl, size, unsigned, 0),
            (Mnemonic::VMULL, 0b1100) => lane(SimdLane::Mull, size, unsigned, 0),
            _ => None,
        };
    }
    if raw & 0xffb0_0810 == 0xf3b0_0000 {
        // Two registers, miscellaneous: the size is in bits 19:18.
        let size = bits(raw, 18, 2) as u8;
        return match (insn.mnemonic, bits(raw, 16, 2), bits(raw, 7, 4)) {
            (Mnemonic::VCLT, 0b01, 0b0100) if size != 3 => lane(SimdLane::Clt0, size, false, 0),
            (Mnemonic::VQMOVN, 0b10, 0b0101) if size != 3 => {
                // Bit 6 selects the unsigned operand and result.
                lane(SimdLane::Qmovn, size, raw & 0x40 != 0, 0)
            }
            _ => None,
        };
    }
    None
}

fn sign_extend_element(value: u64, bits: u32) -> i128 {
    i128::from((value << (64 - bits)) as i64 >> (64 - bits))
}

/// Runs a SimdLane operation. Returns true when a result did not fit (QC).
fn run_simd_lane(fp: &mut VfpState, op: Op) -> bool {
    let Op::SimdLane { kind, size, quad, unsigned, d, n, m, imm } = op else {
        unreachable!();
    };
    let small = 8u32 << size;
    let (source_bits, result_bits) = match () {
        _ if kind.is_long() => (small, small * 2),
        _ if kind.is_narrow() => (small * 2, small),
        _ => (small, small),
    };
    let register = |fp: &VfpState, number: u8, two: bool| {
        let high = if two { fp.read_d_bits(number + 1) } else { 0 };
        u128::from(high) << 64 | u128::from(fp.read_d_bits(number))
    };
    let source_quad = if kind.is_long() { false } else { quad || kind.is_narrow() };
    let result_quad = if kind.is_narrow() { false } else { quad || kind.is_long() };
    let (a, b, old) = (register(fp, n, source_quad), register(fp, m, source_quad), register(fp, d, result_quad));
    let element = |value: u128, bits: u32, index: u32| (value >> (bits * index)) as u64 & (u64::MAX >> (64 - bits));
    let extend = |value: u64, bits: u32| if unsigned { i128::from(value) } else { sign_extend_element(value, bits) };
    let (low, high) = if unsigned {
        (0, (1i128 << result_bits) - 1)
    } else {
        (-(1i128 << (result_bits - 1)), (1i128 << (result_bits - 1)) - 1)
    };
    let count = if source_quad || result_quad { 128 } else { 64 } / source_bits.max(result_bits);
    let mut result = 0u128;
    let mut saturated = false;
    for index in 0..count {
        let (x, y) = (extend(element(a, source_bits, index), source_bits), extend(element(b, source_bits, index), source_bits));
        let z = i128::from(element(old, result_bits, index));
        let all_ones = |condition: bool| if condition { -1 } else { 0 };
        let mut saturate = |value: i128| {
            saturated |= value < low || value > high;
            value.clamp(low, high)
        };
        let value = match kind {
            SimdLane::Cgt => all_ones(x > y),
            SimdLane::Cge => all_ones(x >= y),
            SimdLane::Ceq => all_ones(x == y),
            SimdLane::Clt0 => all_ones(y < 0),
            SimdLane::Add | SimdLane::Addl => x + y,
            SimdLane::Sub => x - y,
            SimdLane::Mul | SimdLane::Mull => x.wrapping_mul(y),
            SimdLane::Mla => z.wrapping_add(x.wrapping_mul(y)),
            SimdLane::Mls => z.wrapping_sub(x.wrapping_mul(y)),
            SimdLane::Qadd => saturate(x + y),
            SimdLane::Qsub => saturate(x - y),
            SimdLane::Shr => y >> imm,
            SimdLane::Rshr | SimdLane::Rshrn => (y + (1 << (imm - 1))) >> imm,
            SimdLane::Movl => y,
            SimdLane::Qmovn => saturate(y),
        };
        result |= (value as u128 & (u128::MAX >> (128 - result_bits))) << (result_bits * index);
    }
    fp.write_d_bits(d, result as u64);
    if result_quad {
        fp.write_d_bits(d + 1, (result >> 64) as u64);
    }
    saturated
}

/// An Advanced SIMD instruction with a fast operation: a logic operation on three
/// registers or VMOV of an immediate. `raw` is the A32 encoding.
fn simd(insn: &DecodedInsn, raw: u32, constants: &mut Vec<u64>) -> Op {
    let quad = raw & 0x40 != 0;
    let d = (bits(raw, 22, 1) << 4 | bits(raw, 12, 4)) as u8;
    let n = (bits(raw, 7, 1) << 4 | bits(raw, 16, 4)) as u8;
    let m = (bits(raw, 5, 1) << 4 | bits(raw, 0, 4)) as u8;
    if insn.mnemonic == Mnemonic::VMOV && raw & 0xfeb8_0090 == 0xf280_0010 {
        let imm8 = bits(raw, 24, 1) << 7 | bits(raw, 16, 3) << 4 | raw & 0xf;
        let Some(value) = simd_expand_immediate(raw & 0x20 != 0, bits(raw, 8, 4), imm8.into()) else {
            return Op::Slow;
        };
        if quad && d & 1 != 0 {
            return Op::Slow;
        }
        constants.push(value);
        return Op::SimdImmediate { quad, d, constant: (constants.len() - 1) as u16 };
    }
    if let Some(op) = simd_lane(insn, raw, quad, d, n, m) {
        return op;
    }
    if insn.mnemonic == Mnemonic::VMVN && raw & 0xffb3_0f90 == 0xf3b0_0580 && !(quad && (d | m) & 1 != 0) {
        return Op::SimdLogic { kind: SimdLogic::Not, quad, d, n: m, m };
    }
    let kind = match insn.mnemonic {
        Mnemonic::VAND => SimdLogic::And,
        Mnemonic::VBIC => SimdLogic::Bic,
        Mnemonic::VORR => SimdLogic::Orr,
        Mnemonic::VORN => SimdLogic::Orn,
        Mnemonic::VEOR => SimdLogic::Eor,
        Mnemonic::VBSL => SimdLogic::Bsl,
        Mnemonic::VBIT => SimdLogic::Bit,
        Mnemonic::VBIF => SimdLogic::Bif,
        _ => return Op::Slow,
    };
    if raw & 0xfe80_0f10 != 0xf200_0110 || (quad && (d | n | m) & 1 != 0) {
        return Op::Slow;
    }
    Op::SimdLogic { kind, quad, d, n, m }
}

/// A VFP instruction in a T32 coprocessor encoding. The fields are those that the
/// rax executor reads.
fn thumb_vfp(insn: &DecodedInsn) -> Op {
    let raw = insn.raw;
    let double = match bits(raw, 8, 4) {
        0b1010 => false,
        0b1011 => true,
        _ => return Op::Slow,
    };
    let (vd, vn, vm) = (bits(raw, 12, 4), bits(raw, 16, 4), bits(raw, 0, 4));
    let (d_bit, n_bit, m_bit) = (bits(raw, 22, 1), bits(raw, 7, 1), bits(raw, 5, 1));
    let single = |v: u32, bit: u32| (v << 1 | bit) as u8;
    let register = |v: u32, bit: u32| if double { (bit << 4 | v) as u8 } else { single(v, bit) };
    let (d, n, m) = (register(vd, d_bit), register(vn, n_bit), register(vm, m_bit));
    let (rt, rt2) = (vd as u8, vn as u8);
    let data_processing = raw >> 24 == 0xee && raw & 0x10 == 0;
    let binary = |kind| Op::FpBinary { kind, double, d, n, m };
    let accumulate = |kind| Op::FpAccumulate { kind, double, d, n, m };
    let unary = |kind| Op::FpUnary { kind, double, d, m };
    let convert = |kind, d, m| Op::FpConvert { kind, d, m };
    let (d_single, m_single) = (single(vd, d_bit), single(vm, m_bit));
    match insn.mnemonic {
        Mnemonic::VMRS if raw == 0xeef1_fa10 => Op::FpFlags,
        Mnemonic::VLDM | Mnemonic::VPOP | Mnemonic::VSTM | Mnemonic::VPUSH if double => {
            let words = raw & 0xff;
            let count = (words / 2) as u8;
            let (p, u, writeback) = (raw & 0x0100_0000 != 0, raw & 0x0080_0000 != 0, raw & 0x0020_0000 != 0);
            let decrement = match (p, u) {
                (false, true) => false,
                (true, false) if writeback => true,
                _ => return Op::Slow,
            };
            if raw >> 25 != 0b111_0110 || words % 2 != 0 || count == 0 || count > 16 || d + count > 32 || rt2 == PC {
                return Op::Slow;
            }
            if raw & 0x0010_0000 != 0 {
                Op::FpLoadMultiple { d, count, rn: rt2, decrement, writeback }
            } else {
                Op::FpStoreMultiple { d, count, rn: rt2, decrement, writeback }
            }
        }
        Mnemonic::VLDR | Mnemonic::VSTR if raw >> 24 == 0xed && raw & 0x20_0000 == 0 => {
            let imm = (raw & 0xff) as i32 * 4;
            let offset = if raw & 0x80_0000 != 0 { imm } else { -imm };
            if rt2 == PC && insn.mnemonic == Mnemonic::VLDR {
                Op::FpLoadLiteral { double, d, offset }
            } else if rt2 == PC {
                Op::Slow
            } else if insn.mnemonic == Mnemonic::VLDR {
                Op::FpLoad { double, d, rn: rt2, offset }
            } else {
                Op::FpStore { double, d, rn: rt2, offset }
            }
        }
        Mnemonic::VMOV if raw >> 21 == 0xec4 >> 1 && raw & 0xd0 == 0x10 => {
            let v = if double { (m_bit << 4 | vm) as u8 } else { m_single };
            if any_pc(&[rt, rt2]) || (!double && v == 31) {
                Op::Slow
            } else if raw & 0x10_0000 == 0 {
                Op::FpFromCorePair { double, v, rt, rt2 }
            } else if rt != rt2 {
                Op::FpToCorePair { double, rt, rt2, v }
            } else {
                Op::Slow
            }
        }
        Mnemonic::VMOV if raw >> 21 == 0xee0 >> 1 && raw & 0x7f == 0x10 && !double && rt != PC => {
            let s = single(vn, n_bit);
            if raw & 0x10_0000 == 0 { Op::FpFromCore { s, rt } } else { Op::FpToCore { rt, s } }
        }
        _ if !data_processing => Op::Slow,
        Mnemonic::VMOV if raw & 0x00b0_00c0 == 0x00b0_0000 => {
            Op::FpImmediate { double, d, imm: (vn << 4 | vm) as u8 }
        }
        Mnemonic::VMOV if raw & 0x00bf_00c0 == 0x00b0_0040 => unary(FpUnary::Mov),
        Mnemonic::VABS => unary(FpUnary::Abs),
        Mnemonic::VNEG => unary(FpUnary::Neg),
        Mnemonic::VSQRT => unary(FpUnary::Sqrt),
        Mnemonic::VADD => binary(FpBinary::Add),
        Mnemonic::VSUB => binary(FpBinary::Sub),
        Mnemonic::VMUL => binary(FpBinary::Mul),
        Mnemonic::VDIV => binary(FpBinary::Div),
        Mnemonic::VNMUL => binary(FpBinary::Nmul),
        Mnemonic::VMLA => accumulate(FpAccumulate::Mla),
        Mnemonic::VMLS => accumulate(FpAccumulate::Mls),
        Mnemonic::VNMLA => accumulate(FpAccumulate::Nmla),
        Mnemonic::VNMLS => accumulate(FpAccumulate::Nmls),
        Mnemonic::VCMP | Mnemonic::VCMPE => Op::FpCompare {
            all_nans_signal: insn.mnemonic == Mnemonic::VCMPE,
            with_zero: vn == 5,
            double,
            d,
            m,
        },
        Mnemonic::VCVT_F32_S32 => convert(FpConvert::SignedToF32, d_single, m_single),
        Mnemonic::VCVT_F32_U32 => convert(FpConvert::UnsignedToF32, d_single, m_single),
        Mnemonic::VCVT_F64_S32 => convert(FpConvert::SignedToF64, d, m_single),
        Mnemonic::VCVT_F64_U32 => convert(FpConvert::UnsignedToF64, d, m_single),
        Mnemonic::VCVT_S32_F32 => convert(FpConvert::F32ToSigned, d_single, m_single),
        Mnemonic::VCVT_U32_F32 => convert(FpConvert::F32ToUnsigned, d_single, m_single),
        Mnemonic::VCVT_S32_F64 => convert(FpConvert::F64ToSigned, d_single, m),
        Mnemonic::VCVT_U32_F64 => convert(FpConvert::F64ToUnsigned, d_single, m),
        _ => Op::Slow,
    }
}

/// Default NaN, flush to zero and the rounding mode.
const FPSCR_MODES: u32 = 0x03c0_0000;
const FPSCR_INEXACT: u32 = 1 << 4;

/// `a + b` or `a * b` with the arithmetic of the host, for the cases where the result
/// and the flags of the soft-float code of rax are sure: round to nearest, no other
/// mode, and operands and result that are zero or normal numbers. None for the others.
fn host_f32(multiply: bool, a: f32, b: f32, fpscr: &mut Fpscr) -> Option<f32> {
    let plain = |value: f32| value == 0.0 || value.is_normal();
    if fpscr.bits() & FPSCR_MODES != 0 || !plain(a) || !plain(b) {
        return None;
    }
    let exact = if multiply {
        f64::from(a) * f64::from(b)
    } else {
        // The sum of two numbers with near exponents has no more bits than an f64.
        let exponent = |value: f32| value.to_bits() >> 23 & 0xff;
        if a != 0.0 && b != 0.0 && exponent(a).abs_diff(exponent(b)) > 25 {
            return None;
        }
        f64::from(a) + f64::from(b)
    };
    let result = exact as f32;
    if !result.is_normal() && exact != 0.0 {
        return None;
    }
    if f64::from(result) != exact {
        *fpscr = Fpscr::from_bits(fpscr.bits() | FPSCR_INEXACT);
    }
    Some(result)
}

/// The flags of a compare of two numbers that are not NaN, with no flush to zero. False
/// for the other cases: nothing changed.
fn host_compare(order: Option<std::cmp::Ordering>, fpscr: &mut Fpscr) -> bool {
    let flags = match order {
        _ if fpscr.bits() & FPSCR_MODES != 0 => return false,
        Some(std::cmp::Ordering::Equal) => 0x6,
        Some(std::cmp::Ordering::Less) => 0x8,
        Some(std::cmp::Ordering::Greater) => 0x2,
        None => return false,
    };
    *fpscr = Fpscr::from_bits(fpscr.bits() & 0x0fff_ffff | flags << 28);
    true
}

const FPSCR_LENGTH_AND_STRIDE: u32 = 0x0037_0000;
const FPSCR_TRAP_ENABLES: u32 = 0x0000_9f00;
/// The condition flags, QC and the cumulative exception flags.
const FPSCR_STATUS: u32 = 0xf800_009f;

/// A number for the state of the controls that decide how the executor runs a VFP
/// instruction. 0 when short vectors or trapped exceptions are on: the fast
/// interpreter has no operation for these.
pub fn fp_controls(cpu: &Armv7Cpu) -> u64 {
    let fpscr = cpu.vfp.fpscr.bits();
    if fpscr & (FPSCR_LENGTH_AND_STRIDE | FPSCR_TRAP_ENABLES) != 0 {
        return 0;
    }
    let cp15 = &cpu.cp15;
    let words = [
        fpscr & !FPSCR_STATUS,
        cpu.vfp.fpexc,
        cp15.cpacr.bits(),
        cp15.nsacr,
        cp15.hcptr,
        cp15.hcr,
        cp15.scr,
        u32::from(cpu.cpsr.mode) | u32::from(cpu.is_secure()) << 8,
    ];
    words.iter().fold(1, |key, &word| (key ^ u64::from(word)).wrapping_mul(0x100_0000_01b3)) | 1
}

/// Shift_C of the architecture, with an amount of 0 to 255. RRX takes no amount.
fn shift_c(value: u32, shift: ShiftType, amount: u32, carry: bool) -> (u32, bool) {
    if amount == 0 {
        return (value, carry);
    }
    match shift {
        ShiftType::LSL if amount > 32 => (0, false),
        ShiftType::LSL if amount == 32 => (0, value & 1 != 0),
        ShiftType::LSL => (value << amount, value >> (32 - amount) & 1 != 0),
        ShiftType::LSR if amount > 32 => (0, false),
        ShiftType::LSR if amount == 32 => (0, value >> 31 != 0),
        ShiftType::LSR => (value >> amount, value >> (amount - 1) & 1 != 0),
        ShiftType::ASR if amount >= 32 => ((value as i32 >> 31) as u32, value >> 31 != 0),
        ShiftType::ASR => ((value as i32 >> amount) as u32, value >> (amount - 1) & 1 != 0),
        ShiftType::ROR => {
            let result = value.rotate_right(amount % 32);
            (result, result >> 31 != 0)
        }
        ShiftType::RRX => (value >> 1 | u32::from(carry) << 31, value & 1 != 0),
    }
}

fn condition_passed(cond: u8, psr: &Psr) -> bool {
    let result = match cond >> 1 {
        0 => psr.z,
        1 => psr.c,
        2 => psr.n,
        3 => psr.v,
        4 => psr.c && !psr.z,
        5 => psr.n == psr.v,
        6 => !psr.z && psr.n == psr.v,
        _ => true,
    };
    result != (cond & 1 != 0 && cond != 15)
}

/// The memory accesses of one instruction.
struct Access<'a, M> {
    memory: &'a mut M,
    /// The instructions that the run completed before this one.
    done: usize,
    /// Set when an access took the slow path: the run must end after the instruction,
    /// because a peripheral can change the timers and the interrupts.
    slow: &'a mut bool,
}

impl<M: Memory> Access<'_, M> {
    #[inline(always)]
    fn load_bytes<const N: usize>(&mut self, address: u32) -> Result<[u8; N], Fault> {
        if let Some(bytes) = self.memory.load(address) {
            return Ok(bytes);
        }
        *self.slow = true;
        self.memory.load_slow(address, self.done)
    }

    #[inline(always)]
    fn store_bytes<const N: usize>(&mut self, address: u32, bytes: [u8; N]) -> Result<(), Fault> {
        if self.memory.store(address, bytes) {
            return Ok(());
        }
        *self.slow = true;
        self.memory.store_slow(address, bytes, self.done)
    }

    #[inline(always)]
    fn load(&mut self, width: Width, address: u32) -> Result<u32, Fault> {
        Ok(match width {
            Width::Byte => u32::from(self.load_bytes::<1>(address)?[0]),
            Width::SignedByte => self.load_bytes::<1>(address)?[0] as i8 as u32,
            Width::Half => u32::from(u16::from_le_bytes(self.load_bytes(address)?)),
            Width::SignedHalf => i16::from_le_bytes(self.load_bytes(address)?) as u32,
            Width::Word => u32::from_le_bytes(self.load_bytes(address)?),
        })
    }

    #[inline(always)]
    fn store(&mut self, width: Width, address: u32, value: u32) -> Result<(), Fault> {
        match width {
            Width::Byte | Width::SignedByte => self.store_bytes(address, [value as u8]),
            Width::Half | Width::SignedHalf => self.store_bytes(address, (value as u16).to_le_bytes()),
            Width::Word => self.store_bytes(address, value.to_le_bytes()),
        }
    }
}

/// The address of the access and the value of the base register after it.
fn indexed(base: u32, index: Index, offset: i32) -> (u32, u32) {
    let with_offset = base.wrapping_add(offset as u32);
    match index {
        Index::Offset => (with_offset, base),
        Index::Pre => (with_offset, with_offset),
        Index::Post => (base, with_offset),
    }
}

#[inline(always)]
fn alu(cpu: &mut Armv7Cpu, kind: Alu, flags: bool, rd: u8, a: u32, b: u32, shifter_carry: bool) {
    let carry_in = u32::from(cpu.cpsr.c);
    let logical = |result| (result, shifter_carry, cpu.cpsr.v);
    let (result, carry, overflow) = match kind {
        Alu::And | Alu::Tst => logical(a & b),
        Alu::Eor | Alu::Teq => logical(a ^ b),
        Alu::Orr => logical(a | b),
        Alu::Bic => logical(a & !b),
        Alu::Orn => logical(a | !b),
        Alu::Mov => logical(b),
        Alu::Mvn => logical(!b),
        Alu::Add | Alu::Cmn => add_with_carry(a, b, 0),
        Alu::Adc => add_with_carry(a, b, carry_in),
        Alu::Sub | Alu::Cmp => add_with_carry(a, !b, 1),
        Alu::Sbc => add_with_carry(a, !b, carry_in),
        Alu::Rsb => add_with_carry(!a, b, 1),
    };
    if flags {
        cpu.cpsr.n = result >> 31 != 0;
        cpu.cpsr.z = result == 0;
        cpu.cpsr.c = carry;
        cpu.cpsr.v = overflow;
    }
    if !kind.is_compare() {
        cpu.regs[usize::from(rd)] = result;
    }
}

fn set_nz(cpu: &mut Armv7Cpu, result: u32) {
    cpu.cpsr.n = result >> 31 != 0;
    cpu.cpsr.z = result == 0;
}

impl Block {
    /// Decodes `code`, the bytes from `offset` in a page to the end of the page or less,
    /// to an instruction that always leaves the block, the end of the bytes or an
    /// instruction with no decode. The block is empty when the first instruction has no
    /// decode or goes across the end of the bytes.
    pub fn translate(code: &[u8], offset: usize, thumb: bool) -> Block {
        let decoder = Decoder::new(if thumb { ExecutionState::Thumb } else { ExecutionState::Aarch32 });
        let mut steps = Vec::new();
        let mut insns = Vec::new();
        let mut constants = Vec::new();
        let mut psr = Psr::default();
        let mut at = offset;
        while steps.len() < MAX_INSTRUCTIONS && at - offset + 2 <= code.len() {
            let wide = !thumb || code[at - offset + 1] >> 3 >= 0x1d;
            let length = if wide { 4 } else { 2 };
            let Some(bytes) = code.get(at - offset..at - offset + length) else {
                break;
            };
            let Ok(insn) = decoder.decode(bytes) else {
                break;
            };
            let in_it = psr.in_it_block();
            let mut op = match (thumb, wide) {
                _ if matches!(insn.mnemonic, Mnemonic::UNDEFINED | Mnemonic::UNKNOWN) => Op::Slow,
                // An A32 PC that is not a multiple of 4 has no fast operation.
                (false, _) if at % 4 != 0 => Op::Slow,
                _ if matches!(insn.mnemonic, Mnemonic::DMB | Mnemonic::DSB | Mnemonic::ISB) => Op::Nop,
                (false, _) if insn.raw >> 25 == 0b111_1001 => simd(&insn, insn.raw, &mut constants),
                (false, _) => arm(insn.raw),
                (true, false) => thumb16(insn.raw, in_it),
                (true, true) if matches!(insn.raw >> 24, 0xec..=0xee) => thumb_vfp(&insn),
                (true, true) if matches!(insn.raw >> 24, 0xef | 0xff) => {
                    simd(&insn, ThumbDecoder::a32_equivalent(insn.raw).unwrap_or(insn.raw), &mut constants)
                }
                (true, true) if insn.raw & 0xff10_0000 == 0xf900_0000 => thumb_simd_memory(&insn),
                (true, true) => thumb32(insn.raw),
            };
            // A branch is the last instruction of its IT block, and a conditional branch
            // or a second IT is not in an IT block: the other uses have no defined result.
            let last_in_it = psr.last_in_it_block();
            if in_it && (matches!(op, Op::BranchCond { .. } | Op::It { .. }) || (op.can_branch() && !last_in_it)) {
                op = Op::Slow;
            }
            // The executor makes the decision for the instructions that it runs.
            let mut cond = if op == Op::Slow {
                COND_ALWAYS
            } else if !thumb {
                (insn.raw >> 28) as u8
            } else if in_it {
                psr.it_condition()
            } else {
                COND_ALWAYS
            };
            let it_state = psr.it_state;
            match op {
                Op::It { state } => psr.it_state = state,
                Op::BranchCond { cond: branch_cond, offset } => {
                    cond = branch_cond;
                    op = Op::Branch { offset };
                    psr.advance_it_state();
                }
                _ => psr.advance_it_state(),
            }
            steps.push(Step { op, cond, it_state, offset: at as u16 });
            insns.push(insn);
            at += length;
            let leaves = matches!(op, Op::Branch { .. } | Op::BranchLink { .. } | Op::BranchLinkArm { .. } | Op::BranchLinkToArm { .. } | Op::TableBranch { .. } | Op::BranchExchange { .. } | Op::BranchLinkExchange { .. })
                || matches!(op, Op::Pop { mask } if mask & 1 << PC != 0);
            if leaves && cond == COND_ALWAYS {
                break;
            }
        }
        Block {
            admitted: vec![Cell::new(0); steps.len()],
            constants,
            executor_insns: insns.iter().map(executor_insn).collect(),
            steps,
            insns,
            end_offset: at as u16,
            end_it_state: psr.it_state,
            thumb,
            tier: Cell::new(jit::Tier::Cold { runs: 0 }),
            fp_compilations: Cell::new(0),
        }
    }

    /// Removes the instructions from `index` to the end of the block.
    pub fn truncate(&mut self, index: usize) {
        self.end_offset = self.offset(index) as u16;
        self.end_it_state = self.it_state(index);
        self.steps.truncate(index);
        self.insns.truncate(index);
        self.executor_insns.truncate(index);
    }

    /// Tells the block that the executor completed the instruction `index` with no
    /// exception, in the state `controls` of the FP controls before and after it.
    pub fn set_admitted(&self, index: usize, controls: u64) {
        self.admitted[index].set(controls);
    }

    /// True when the FP access checks of the executor pass for the instruction `index`
    /// in the state `controls` of the FP controls.
    pub fn is_admitted(&self, index: usize, controls: u64) -> bool {
        controls != 0 && self.admitted[index].get() == controls
    }

    /// The instruction `index` in the form that the executor runs with the least work.
    pub fn executor_insn(&self, index: usize) -> &DecodedInsn {
        &self.executor_insns[index]
    }

    pub fn insns(&self) -> &[DecodedInsn] {
        &self.insns
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// True when the interpreter has an operation for the instruction `index`.
    pub fn has_fast_operation(&self, index: usize) -> bool {
        self.steps[index].op != Op::Slow
    }

    pub fn is_thumb(&self) -> bool {
        self.thumb
    }

    /// The offset in the page of the instruction `index`, or of the end of the block.
    pub fn offset(&self, index: usize) -> u32 {
        self.steps.get(index).map_or(self.end_offset, |step| step.offset).into()
    }

    /// ITSTATE before the instruction `index`, or after the last instruction.
    pub fn it_state(&self, index: usize) -> u8 {
        self.steps.get(index).map_or(self.end_it_state, |step| step.it_state)
    }

    /// Runs at most `limit` instructions from the instruction `start`. `page` is the
    /// virtual address of the page of the block. Returns the number of instructions
    /// that ran. The PC and the IT state of `cpu` are correct on return.
    pub fn run<M: Memory, H: Hook>(
        &self,
        start: usize,
        limit: usize,
        page: u32,
        cpu: &mut Armv7Cpu,
        memory: &mut M,
        hook: &mut H,
    ) -> (usize, Exit) {
        let end = self.steps.len().min(start.saturating_add(limit));
        let mut index = start;
        macro_rules! r {
            ($register:expr) => {
                cpu.regs[usize::from($register) & 15]
            };
        }
        let mut slow_access = false;
        let mut fp_state = None;
        macro_rules! access {
            () => {
                Access { memory: &mut *memory, done: index - start, slow: &mut slow_access }
            };
        }
        let exit = 'run: loop {
            let Some(step) = self.steps[..end].get(index) else {
                break Exit::Done;
            };
            let pc = page.wrapping_add(u32::from(step.offset));
            if H::ACTIVE {
                cpu.regs[15] = pc;
                cpu.cpsr.it_state = step.it_state;
                hook.before(cpu, self, index);
            }
            if step.cond != COND_ALWAYS && !condition_passed(step.cond, &cpu.cpsr) {
                index += 1;
                continue;
            }
            if step.op.is_fp() {
                // The controls do not change in a run: only the executor writes them.
                let controls = *fp_state.get_or_insert_with(|| fp_controls(cpu));
                if !self.is_admitted(index, controls) {
                    break Exit::Slow(index);
                }
            }
            let target = match step.op {
                Op::Slow => break Exit::Slow(index),
                Op::FpBinary { kind, double: false, d, n, m } => {
                    let fp = &mut cpu.vfp;
                    let (a, b) = (fp.read_s(n), fp.read_s(m));
                    let host = match kind {
                        FpBinary::Add => host_f32(false, a, b, &mut fp.fpscr),
                        FpBinary::Sub => host_f32(false, a, -b, &mut fp.fpscr),
                        FpBinary::Mul => host_f32(true, a, b, &mut fp.fpscr),
                        _ => None,
                    };
                    let result = match kind {
                        _ if host.is_some() => host.unwrap(),
                        FpBinary::Add => vfp::vadd_f32(a, b, &mut fp.fpscr),
                        FpBinary::Sub => vfp::vsub_f32(a, b, &mut fp.fpscr),
                        FpBinary::Mul => vfp::vmul_f32(a, b, &mut fp.fpscr),
                        FpBinary::Div => vfp::vdiv_f32(a, b, &mut fp.fpscr),
                        FpBinary::Nmul => vfp::vnmul_f32(a, b, &mut fp.fpscr),
                    };
                    fp.write_s(d, result);
                    None
                }
                Op::FpBinary { kind, double: true, d, n, m } => {
                    let fp = &mut cpu.vfp;
                    let (a, b) = (fp.read_d(n), fp.read_d(m));
                    let result = match kind {
                        FpBinary::Add => vfp::vadd_f64(a, b, &mut fp.fpscr),
                        FpBinary::Sub => vfp::vsub_f64(a, b, &mut fp.fpscr),
                        FpBinary::Mul => vfp::vmul_f64(a, b, &mut fp.fpscr),
                        FpBinary::Div => vfp::vdiv_f64(a, b, &mut fp.fpscr),
                        FpBinary::Nmul => vfp::vnmul_f64(a, b, &mut fp.fpscr),
                    };
                    fp.write_d(d, result);
                    None
                }
                Op::FpAccumulate { kind, double: false, d, n, m } => {
                    let fp = &mut cpu.vfp;
                    let (acc, a, b) = (fp.read_s(d), fp.read_s(n), fp.read_s(m));
                    let saved = fp.fpscr.bits();
                    let host = (kind == FpAccumulate::Mla)
                        .then(|| host_f32(true, a, b, &mut fp.fpscr))
                        .flatten()
                        .and_then(|product| host_f32(false, acc, product, &mut fp.fpscr));
                    if host.is_none() {
                        // The soft-float code starts again from the flags before.
                        fp.fpscr = Fpscr::from_bits(saved);
                    }
                    let result = match kind {
                        _ if host.is_some() => host.unwrap(),
                        FpAccumulate::Mla => vfp::vmla_f32(acc, a, b, &mut fp.fpscr),
                        FpAccumulate::Mls => vfp::vmls_f32(acc, a, b, &mut fp.fpscr),
                        FpAccumulate::Nmla => vfp::vnmla_f32(acc, a, b, &mut fp.fpscr),
                        FpAccumulate::Nmls => vfp::vnmls_f32(acc, a, b, &mut fp.fpscr),
                    };
                    fp.write_s(d, result);
                    None
                }
                Op::FpAccumulate { kind, double: true, d, n, m } => {
                    let fp = &mut cpu.vfp;
                    let (acc, a, b) = (fp.read_d(d), fp.read_d(n), fp.read_d(m));
                    let result = match kind {
                        FpAccumulate::Mla => vfp::vmla_f64(acc, a, b, &mut fp.fpscr),
                        FpAccumulate::Mls => vfp::vmls_f64(acc, a, b, &mut fp.fpscr),
                        FpAccumulate::Nmla => vfp::vnmla_f64(acc, a, b, &mut fp.fpscr),
                        FpAccumulate::Nmls => vfp::vnmls_f64(acc, a, b, &mut fp.fpscr),
                    };
                    fp.write_d(d, result);
                    None
                }
                Op::FpUnary { kind: FpUnary::Mov, double: false, d, m } => {
                    let fp = &mut cpu.vfp;
                    fp.write_s_bits(d, fp.read_s_bits(m));
                    None
                }
                Op::FpUnary { kind: FpUnary::Mov, double: true, d, m } => {
                    let fp = &mut cpu.vfp;
                    fp.write_d_bits(d, fp.read_d_bits(m));
                    None
                }
                Op::FpUnary { kind, double: false, d, m } => {
                    let fp = &mut cpu.vfp;
                    let value = fp.read_s(m);
                    let result = match kind {
                        FpUnary::Abs => vfp::vabs_f32(value),
                        FpUnary::Neg => vfp::vneg_f32(value),
                        _ => vfp::vsqrt_f32(value, &mut fp.fpscr),
                    };
                    fp.write_s(d, result);
                    None
                }
                Op::FpUnary { kind, double: true, d, m } => {
                    let fp = &mut cpu.vfp;
                    let value = fp.read_d(m);
                    let result = match kind {
                        FpUnary::Abs => vfp::vabs_f64(value),
                        FpUnary::Neg => vfp::vneg_f64(value),
                        _ => vfp::vsqrt_f64(value, &mut fp.fpscr),
                    };
                    fp.write_d(d, result);
                    None
                }
                Op::FpCompare { all_nans_signal, with_zero, double: false, d, m } => {
                    let fp = &mut cpu.vfp;
                    let other = if with_zero { 0.0 } else { fp.read_s(m) };
                    if !host_compare(fp.read_s(d).partial_cmp(&other), &mut fp.fpscr) {
                        vfp::vcmp_f32_with_exception(fp.read_s(d), other, all_nans_signal, &mut fp.fpscr);
                    }
                    None
                }
                Op::FpCompare { all_nans_signal, with_zero, double: true, d, m } => {
                    let fp = &mut cpu.vfp;
                    let other = if with_zero { 0.0 } else { fp.read_d(m) };
                    if !host_compare(fp.read_d(d).partial_cmp(&other), &mut fp.fpscr) {
                        vfp::vcmp_f64_with_exception(fp.read_d(d), other, all_nans_signal, &mut fp.fpscr);
                    }
                    None
                }
                Op::FpImmediate { double: false, d, imm } => {
                    let fp = &mut cpu.vfp;
                    fp.write_s_bits(d, vfp::vfp_expand_imm_f32(imm));
                    None
                }
                Op::FpImmediate { double: true, d, imm } => {
                    let fp = &mut cpu.vfp;
                    fp.write_d_bits(d, vfp::vfp_expand_imm_f64(imm));
                    None
                }
                Op::FpConvert { kind, d, m } => {
                    let fp = &mut cpu.vfp;
                    let integer = fp.read_s_bits(m);
                    let fpscr: &mut Fpscr = &mut fp.fpscr;
                    match kind {
                        FpConvert::SignedToF32 => {
                            let result = vfp::vcvt_f32_s32_with_fpscr(integer as i32, fpscr);
                            fp.write_s(d, result);
                        }
                        FpConvert::UnsignedToF32 => {
                            let result = vfp::vcvt_f32_u32_with_fpscr(integer, fpscr);
                            fp.write_s(d, result);
                        }
                        FpConvert::SignedToF64 => {
                            let result = vfp::vcvt_f64_s32_with_fpscr(integer as i32, fpscr);
                            fp.write_d(d, result);
                        }
                        FpConvert::UnsignedToF64 => {
                            let result = vfp::vcvt_f64_u32_with_fpscr(integer, fpscr);
                            fp.write_d(d, result);
                        }
                        FpConvert::F32ToSigned => {
                            let result = vfp::vcvt_s32_f32(fp.read_s(m), &mut fp.fpscr);
                            fp.write_s_bits(d, result as u32);
                        }
                        FpConvert::F32ToUnsigned => {
                            let result = vfp::vcvt_u32_f32(fp.read_s(m), &mut fp.fpscr);
                            fp.write_s_bits(d, result);
                        }
                        FpConvert::F64ToSigned => {
                            let result = vfp::vcvt_s32_f64(fp.read_d(m), &mut fp.fpscr);
                            fp.write_s_bits(d, result as u32);
                        }
                        FpConvert::F64ToUnsigned => {
                            let result = vfp::vcvt_u32_f64(fp.read_d(m), &mut fp.fpscr);
                            fp.write_s_bits(d, result);
                        }
                    }
                    None
                }
                Op::FpFromCore { s, rt } => {
                    cpu.vfp.write_s_bits(s, r!(rt));
                    None
                }
                Op::FpToCore { rt, s } => {
                    r!(rt) = cpu.vfp.read_s_bits(s);
                    None
                }
                Op::FpFromCorePair { double: false, v, rt, rt2 } => {
                    cpu.vfp.write_s_bits(v, r!(rt));
                    cpu.vfp.write_s_bits(v + 1, r!(rt2));
                    None
                }
                Op::FpFromCorePair { double: true, v, rt, rt2 } => {
                    cpu.vfp.write_d_bits(v, u64::from(r!(rt)) | u64::from(r!(rt2)) << 32);
                    None
                }
                Op::FpToCorePair { double: false, rt, rt2, v } => {
                    r!(rt) = cpu.vfp.read_s_bits(v);
                    r!(rt2) = cpu.vfp.read_s_bits(v + 1);
                    None
                }
                Op::FpToCorePair { double: true, rt, rt2, v } => {
                    let value = cpu.vfp.read_d_bits(v);
                    r!(rt) = value as u32;
                    r!(rt2) = (value >> 32) as u32;
                    None
                }
                Op::FpLoad { double, d, rn: _, offset } | Op::FpLoadLiteral { double, d, offset } => {
                    let base = match step.op {
                        Op::FpLoad { rn, .. } => r!(rn),
                        _ => pc.wrapping_add(4) & !3,
                    };
                    let address = base.wrapping_add(offset as u32);
                    if address & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    let Ok(low) = access!().load(Width::Word, address) else {
                        break Exit::Fault(index);
                    };
                    if !double {
                        cpu.vfp.write_s_bits(d, low);
                    } else {
                        let Ok(high) = access!().load(Width::Word, address.wrapping_add(4)) else {
                            break Exit::Fault(index);
                        };
                        cpu.vfp.write_d_bits(d, u64::from(high) << 32 | u64::from(low));
                    }
                    None
                }
                Op::FpStore { double, d, rn, offset } => {
                    let address = r!(rn).wrapping_add(offset as u32);
                    if address & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    let value = if double { cpu.vfp.read_d_bits(d) } else { u64::from(cpu.vfp.read_s_bits(d)) };
                    if access!().store(Width::Word, address, value as u32).is_err() {
                        break Exit::Fault(index);
                    }
                    if double && access!().store(Width::Word, address.wrapping_add(4), (value >> 32) as u32).is_err() {
                        break Exit::Fault(index);
                    }
                    None
                }
                Op::SimdLogic { kind, quad, d, n, m } => {
                    for half in 0..=u8::from(quad) {
                        let fp = &mut cpu.vfp;
                        let (a, b, c) = (fp.read_d_bits(d + half), fp.read_d_bits(n + half), fp.read_d_bits(m + half));
                        let result = match kind {
                            SimdLogic::And => b & c,
                            SimdLogic::Bic => b & !c,
                            SimdLogic::Orr => b | c,
                            SimdLogic::Orn => b | !c,
                            SimdLogic::Eor => b ^ c,
                            SimdLogic::Bsl => b & a | c & !a,
                            SimdLogic::Bit => b & c | a & !c,
                            SimdLogic::Bif => a & c | b & !c,
                            SimdLogic::Not => !c,
                        };
                        fp.write_d_bits(d + half, result);
                    }
                    None
                }
                Op::SimdLoad { d, count, rn, rm, align } => {
                    // Only direct loads: the executor changes a part of the registers
                    // when a later byte gives a fault.
                    let base = r!(rn);
                    if base % u32::from(align) != 0 {
                        break Exit::Slow(index);
                    }
                    let mut values = [0; 4];
                    for (register, value) in values.iter_mut().enumerate().take(usize::from(count)) {
                        let Some(bytes) = memory.load(base.wrapping_add(register as u32 * 8)) else {
                            break 'run Exit::Slow(index);
                        };
                        *value = u64::from_le_bytes(bytes);
                    }
                    for register in 0..count {
                        cpu.vfp.write_d_bits(d + register, values[usize::from(register)]);
                    }
                    match rm {
                        PC => {}
                        SP => r!(rn) = base.wrapping_add(u32::from(count) * 8),
                        _ => r!(rn) = base.wrapping_add(r!(rm)),
                    }
                    None
                }
                Op::SimdStore { d, count, rn, rm, align } => {
                    // A store that is not direct leaves the instruction to the executor,
                    // which stores the same bytes again.
                    let base = r!(rn);
                    if base % u32::from(align) != 0 {
                        break Exit::Slow(index);
                    }
                    for register in 0..count {
                        let bytes = cpu.vfp.read_d_bits(d + register).to_le_bytes();
                        if !memory.store(base.wrapping_add(u32::from(register) * 8), bytes) {
                            break 'run Exit::Slow(index);
                        }
                    }
                    match rm {
                        PC => {}
                        SP => r!(rn) = base.wrapping_add(u32::from(count) * 8),
                        _ => r!(rn) = base.wrapping_add(r!(rm)),
                    }
                    None
                }
                Op::SimdLane { .. } => {
                    if run_simd_lane(&mut cpu.vfp, step.op) {
                        cpu.vfp.fpscr.set_qc(true);
                    }
                    None
                }
                Op::SimdImmediate { quad, d, constant } => {
                    let value = self.constants[usize::from(constant)];
                    cpu.vfp.write_d_bits(d, value);
                    if quad {
                        cpu.vfp.write_d_bits(d + 1, value);
                    }
                    None
                }
                Op::FpLoadMultiple { d, count, rn, decrement, writeback } => {
                    // Only direct loads, as for SimdLoad.
                    let base = r!(rn);
                    let size = u32::from(count) * 8;
                    let start = if decrement { base.wrapping_sub(size) } else { base };
                    if start & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    let mut values = [0; 16];
                    for (register, value) in values.iter_mut().enumerate().take(usize::from(count)) {
                        let Some(bytes) = memory.load(start.wrapping_add(register as u32 * 8)) else {
                            break 'run Exit::Slow(index);
                        };
                        *value = u64::from_le_bytes(bytes);
                    }
                    for register in 0..count {
                        cpu.vfp.write_d_bits(d + register, values[usize::from(register)]);
                    }
                    if writeback {
                        r!(rn) = if decrement { start } else { base.wrapping_add(size) };
                    }
                    None
                }
                Op::FpStoreMultiple { d, count, rn, decrement, writeback } => {
                    let base = r!(rn);
                    let size = u32::from(count) * 8;
                    let start = if decrement { base.wrapping_sub(size) } else { base };
                    if start & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    for register in 0..count {
                        let bytes = cpu.vfp.read_d_bits(d + register).to_le_bytes();
                        if !memory.store(start.wrapping_add(u32::from(register) * 8), bytes) {
                            break 'run Exit::Slow(index);
                        }
                    }
                    if writeback {
                        r!(rn) = if decrement { start } else { base.wrapping_add(size) };
                    }
                    None
                }
                Op::FpFlags => {
                    let flags = cpu.vfp.fpscr.bits();
                    cpu.cpsr.n = flags >> 31 != 0;
                    cpu.cpsr.z = flags >> 30 & 1 != 0;
                    cpu.cpsr.c = flags >> 29 & 1 != 0;
                    cpu.cpsr.v = flags >> 28 & 1 != 0;
                    None
                }
                Op::Nop | Op::It { .. } => None,
                Op::AluImm { kind, flags, carry, rd, rn, imm } => {
                    let carry = match carry {
                        Carry::Keep => cpu.cpsr.c,
                        Carry::Clear => false,
                        Carry::Set => true,
                    };
                    alu(cpu, kind, flags, rd, r!(rn), imm, carry);
                    None
                }
                Op::AluReg { kind, flags, rd, rn, rm, shift, amount } => {
                    let (operand, carry) = shift_c(r!(rm), shift, u32::from(amount), cpu.cpsr.c);
                    alu(cpu, kind, flags, rd, r!(rn), operand, carry);
                    None
                }
                Op::ShiftReg { shift, flags, rd, rn, rm } => {
                    let (result, carry) = shift_c(r!(rn), shift, r!(rm) & 0xff, cpu.cpsr.c);
                    alu(cpu, Alu::Mov, flags, rd, 0, result, carry);
                    None
                }
                Op::Movt { rd, imm } => {
                    r!(rd) = r!(rd) & 0xffff | u32::from(imm) << 16;
                    None
                }
                Op::AddPc { rd } => {
                    r!(rd) = r!(rd).wrapping_add(pc.wrapping_add(4));
                    None
                }
                Op::Adr { rd, offset } => {
                    r!(rd) = (pc.wrapping_add(4) & !3).wrapping_add(offset as u32);
                    None
                }
                Op::Extend { kind, rd, rm, rotation } => {
                    let value = r!(rm).rotate_right(u32::from(rotation));
                    r!(rd) = match kind {
                        Extend::Sxth => value as i16 as u32,
                        Extend::Sxtb => value as i8 as u32,
                        Extend::Uxth => value & 0xffff,
                        Extend::Uxtb => value & 0xff,
                    };
                    None
                }
                Op::Clz { rd, rm } => {
                    r!(rd) = r!(rm).leading_zeros();
                    None
                }
                Op::Rbit { rd, rm } => {
                    r!(rd) = r!(rm).reverse_bits();
                    None
                }
                Op::Rev { rd, rm } => {
                    r!(rd) = r!(rm).swap_bytes();
                    None
                }
                Op::Ubfx { rd, rn, lsb, width } => {
                    r!(rd) = r!(rn) << (32 - lsb - width) >> (32 - width);
                    None
                }
                Op::Sbfx { rd, rn, lsb, width } => {
                    r!(rd) = ((r!(rn) << (32 - lsb - width)) as i32 >> (32 - width)) as u32;
                    None
                }
                Op::Bfi { rd, rn, lsb, width } => {
                    let mask = (u32::MAX >> (32 - width)) << lsb;
                    r!(rd) = r!(rd) & !mask | r!(rn) << lsb & mask;
                    None
                }
                Op::Bfc { rd, lsb, width } => {
                    r!(rd) &= !((u32::MAX >> (32 - width)) << lsb);
                    None
                }
                Op::Mul { flags, rd, rn, rm } => {
                    let result = r!(rn).wrapping_mul(r!(rm));
                    if flags {
                        set_nz(cpu, result);
                    }
                    r!(rd) = result;
                    None
                }
                Op::Mla { rd, rn, rm, ra } => {
                    r!(rd) = r!(rn).wrapping_mul(r!(rm)).wrapping_add(r!(ra));
                    None
                }
                Op::Mls { rd, rn, rm, ra } => {
                    r!(rd) = r!(ra).wrapping_sub(r!(rn).wrapping_mul(r!(rm)));
                    None
                }
                Op::Umull { lo, hi, rn, rm } => {
                    let product = u64::from(r!(rn)) * u64::from(r!(rm));
                    r!(lo) = product as u32;
                    r!(hi) = (product >> 32) as u32;
                    None
                }
                Op::Umlal { lo, hi, rn, rm } => {
                    let sum = (u64::from(r!(rn)) * u64::from(r!(rm)))
                        .wrapping_add(u64::from(r!(hi)) << 32 | u64::from(r!(lo)));
                    r!(lo) = sum as u32;
                    r!(hi) = (sum >> 32) as u32;
                    None
                }
                Op::Umaal { lo, hi, rn, rm } => {
                    let sum = u64::from(r!(rn)) * u64::from(r!(rm)) + u64::from(r!(lo)) + u64::from(r!(hi));
                    r!(lo) = sum as u32;
                    r!(hi) = (sum >> 32) as u32;
                    None
                }
                Op::Smull { lo, hi, rn, rm } => {
                    let product = i64::from(r!(rn) as i32) * i64::from(r!(rm) as i32);
                    r!(lo) = product as u32;
                    r!(hi) = (product >> 32) as u32;
                    None
                }
                Op::Udiv { rd, rn, rm } => {
                    r!(rd) = r!(rn).checked_div(r!(rm)).unwrap_or(0);
                    None
                }
                Op::Sdiv { rd, rn, rm } => {
                    let divisor = r!(rm) as i32;
                    r!(rd) = if divisor == 0 { 0 } else { (r!(rn) as i32).wrapping_div(divisor) as u32 };
                    None
                }
                Op::Load { width, rt, rn, index: mode, offset } => {
                    let (address, base) = indexed(r!(rn), mode, offset);
                    let Ok(value) = access!().load(width, address) else {
                        break Exit::Fault(index);
                    };
                    r!(rn) = base;
                    r!(rt) = value;
                    None
                }
                Op::LoadReg { width, rt, rn, rm, shift } => {
                    let address = r!(rn).wrapping_add(r!(rm) << shift);
                    let Ok(value) = access!().load(width, address) else {
                        break Exit::Fault(index);
                    };
                    r!(rt) = value;
                    None
                }
                Op::LoadLiteral { width, rt, offset } => {
                    let address = (pc.wrapping_add(4) & !3).wrapping_add(offset as u32);
                    let Ok(value) = access!().load(width, address) else {
                        break Exit::Fault(index);
                    };
                    r!(rt) = value;
                    None
                }
                Op::Store { width, rt, rn, index: mode, offset } => {
                    let (address, base) = indexed(r!(rn), mode, offset);
                    if access!().store(width, address, r!(rt)).is_err() {
                        break Exit::Fault(index);
                    }
                    r!(rn) = base;
                    None
                }
                Op::StoreReg { width, rt, rn, rm, shift } => {
                    let address = r!(rn).wrapping_add(r!(rm) << shift);
                    if access!().store(width, address, r!(rt)).is_err() {
                        break Exit::Fault(index);
                    }
                    None
                }
                Op::LoadPair { rt, rt2, rn, index: mode, offset } => {
                    let (address, base) = indexed(r!(rn), mode, offset);
                    if address & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    let Ok(first) = access!().load(Width::Word, address) else {
                        break Exit::Fault(index);
                    };
                    let Ok(second) = access!().load(Width::Word, address.wrapping_add(4)) else {
                        break Exit::Fault(index);
                    };
                    r!(rn) = base;
                    r!(rt) = first;
                    r!(rt2) = second;
                    None
                }
                Op::StorePair { rt, rt2, rn, index: mode, offset } => {
                    let (address, base) = indexed(r!(rn), mode, offset);
                    if address & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    if access!().store(Width::Word, address, r!(rt)).is_err()
                        || access!().store(Width::Word, address.wrapping_add(4), r!(rt2)).is_err()
                    {
                        break Exit::Fault(index);
                    }
                    r!(rn) = base;
                    None
                }
                Op::LoadMultiple { rn, mask, writeback } => {
                    let mut address = r!(rn);
                    if address & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    let mut values = [0; 16];
                    for (register, value) in values.iter_mut().enumerate() {
                        if mask & 1 << register != 0 {
                            let Ok(word) = access!().load(Width::Word, address) else {
                                break 'run Exit::Fault(index);
                            };
                            *value = word;
                            address = address.wrapping_add(4);
                        }
                    }
                    for register in 0..15 {
                        if mask & 1 << register != 0 {
                            cpu.regs[register] = values[register];
                        }
                    }
                    if writeback {
                        r!(rn) = address;
                    }
                    None
                }
                Op::StoreMultiple { rn, mask, writeback } => {
                    let mut address = r!(rn);
                    if address & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    for register in 0..15 {
                        if mask & 1 << register != 0 {
                            if access!().store(Width::Word, address, cpu.regs[register]).is_err() {
                                break 'run Exit::Fault(index);
                            }
                            address = address.wrapping_add(4);
                        }
                    }
                    if writeback {
                        r!(rn) = address;
                    }
                    None
                }
                Op::Push { mask } => {
                    let low = r!(13u8).wrapping_sub(mask.count_ones() * 4);
                    if low & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    let mut address = low;
                    for register in 0..15 {
                        if mask & 1 << register != 0 {
                            if access!().store(Width::Word, address, cpu.regs[register]).is_err() {
                                break 'run Exit::Fault(index);
                            }
                            address = address.wrapping_add(4);
                        }
                    }
                    r!(13u8) = low;
                    None
                }
                Op::Pop { mask } => {
                    let mut address = r!(13u8);
                    if address & 3 != 0 {
                        break Exit::Slow(index);
                    }
                    let mut values = [0; 16];
                    for (register, value) in values.iter_mut().enumerate() {
                        if mask & 1 << register != 0 {
                            let Ok(word) = access!().load(Width::Word, address) else {
                                break 'run Exit::Fault(index);
                            };
                            *value = word;
                            address = address.wrapping_add(4);
                        }
                    }
                    for register in 0..15 {
                        if mask & 1 << register != 0 {
                            cpu.regs[register] = values[register];
                        }
                    }
                    r!(13u8) = address;
                    if mask & 1 << PC == 0 {
                        None
                    } else {
                        let target = values[15];
                        cpu.cpsr.t = target & 1 != 0;
                        Some(target & !1)
                    }
                }
                Op::Branch { offset } => Some(pc.wrapping_add(4).wrapping_add(offset as u32)),
                // The translation puts the condition in the step.
                Op::BranchCond { .. } => unreachable!(),
                Op::BranchLink { offset } => {
                    r!(14u8) = pc.wrapping_add(4) | 1;
                    Some(pc.wrapping_add(4).wrapping_add(offset as u32))
                }
                Op::BranchLinkArm { offset } => {
                    r!(14u8) = pc.wrapping_add(4);
                    Some(pc.wrapping_add(4).wrapping_add(offset as u32))
                }
                Op::BranchLinkToArm { offset } => {
                    r!(14u8) = pc.wrapping_add(4) | 1;
                    cpu.cpsr.t = false;
                    Some((pc.wrapping_add(4) & !3).wrapping_add(offset as u32))
                }
                Op::TableBranch { rn, rm, halfword } => {
                    let base = if rn == PC { pc.wrapping_add(4) } else { r!(rn) };
                    let (width, address) = if halfword {
                        (Width::Half, base.wrapping_add(r!(rm) << 1))
                    } else {
                        (Width::Byte, base.wrapping_add(r!(rm)))
                    };
                    let Ok(entry) = access!().load(width, address) else {
                        break Exit::Fault(index);
                    };
                    Some(pc.wrapping_add(4).wrapping_add(entry * 2))
                }
                Op::CompareBranch { rn, nonzero, offset } => {
                    ((r!(rn) != 0) == nonzero).then(|| pc.wrapping_add(4 + u32::from(offset)))
                }
                Op::BranchExchange { rm } => {
                    let target = r!(rm);
                    cpu.cpsr.t = target & 1 != 0;
                    Some(target & !1)
                }
                Op::BranchLinkExchange { rm } => {
                    let target = r!(rm);
                    r!(14u8) = pc.wrapping_add(2) | 1;
                    cpu.cpsr.t = target & 1 != 0;
                    Some(target & !1)
                }
            };
            index += 1;
            if let Some(target) = target {
                cpu.regs[15] = target;
                cpu.cpsr.it_state = 0;
                return (index - start, Exit::Done);
            }
            if slow_access {
                break Exit::Done;
            }
        };
        cpu.regs[15] = page.wrapping_add(self.offset(index));
        cpu.cpsr.it_state = self.it_state(index);
        (index - start, exit)
    }
}
