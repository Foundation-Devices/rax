//! An assembler for the AArch64 instructions that the JIT emits.
//!
//! A register is its number. In the shifted-register and bit-field forms 31 is the zero
//! register; in the immediate forms of ADD and SUB and as a base address it is the SP.

pub type Reg = u8;
pub const ZR: Reg = 31;
pub const SP: Reg = 31;

pub const LSL: u32 = 0;
pub const LSR: u32 = 1;
pub const ASR: u32 = 2;
pub const ROR: u32 = 3;

// Shifted register, 32 bits.
pub const ADD: u32 = 0x0b00_0000;
pub const ADDS: u32 = 0x2b00_0000;
pub const SUB: u32 = 0x4b00_0000;
pub const SUBS: u32 = 0x6b00_0000;
pub const AND: u32 = 0x0a00_0000;
pub const BIC: u32 = 0x0a20_0000;
pub const ORR: u32 = 0x2a00_0000;
pub const ORN: u32 = 0x2a20_0000;
pub const EOR: u32 = 0x4a00_0000;
pub const ANDS: u32 = 0x6a00_0000;
/// Makes a 32-bit register operation a 64-bit one.
pub const X: u32 = 0x8000_0000;

// Three registers, 32 bits.
pub const ADC: u32 = 0x1a00_0000;
pub const ADCS: u32 = 0x3a00_0000;
pub const SBC: u32 = 0x5a00_0000;
pub const SBCS: u32 = 0x7a00_0000;
pub const UDIV: u32 = 0x1ac0_0800;
pub const SDIV: u32 = 0x1ac0_0c00;
pub const LSLV: u32 = 0x1ac0_2000;
pub const LSRV: u32 = 0x1ac0_2400;
pub const ASRV: u32 = 0x1ac0_2800;
pub const RORV: u32 = 0x1ac0_2c00;

// Two registers, 32 bits.
pub const CLZ: u32 = 0x5ac0_1000;
pub const RBIT: u32 = 0x5ac0_0000;
pub const REV: u32 = 0x5ac0_0800;

// Immediate with 12 bits, 32 bits.
pub const ADD_IMM: u32 = 0x1100_0000;
pub const ADDS_IMM: u32 = 0x3100_0000;
pub const SUB_IMM: u32 = 0x5100_0000;
pub const SUBS_IMM: u32 = 0x7100_0000;

// Logical immediate, 32 bits.
pub const AND_IMM: u32 = 0x1200_0000;
pub const ORR_IMM: u32 = 0x3200_0000;
pub const EOR_IMM: u32 = 0x5200_0000;

// Bit fields, 32 bits.
pub const SBFM: u32 = 0x1300_0000;
pub const BFM: u32 = 0x3300_0000;
pub const UBFM: u32 = 0x5300_0000;

// Multiply with a third operand: Rd = Ra +- Rn * Rm.
pub const MADD: u32 = 0x1b00_0000;
pub const MSUB: u32 = 0x1b00_8000;
pub const UMADDL: u32 = 0x9ba0_0000;
pub const SMADDL: u32 = 0x9b20_0000;

// Scalar floating point on S registers. `DOUBLE` makes the operation one on D registers.
pub const FADD: u32 = 0x1e20_2800;
pub const FSUB: u32 = 0x1e20_3800;
pub const FMUL: u32 = 0x1e20_0800;
pub const FDIV: u32 = 0x1e20_1800;
pub const FABS: u32 = 0x1e20_c000;
pub const FNEG: u32 = 0x1e21_4000;
pub const FSQRT: u32 = 0x1e21_c000;
/// The compare of two registers. `FCMP_ZERO` makes it one with 0, `FCMP_SIGNAL` makes
/// each NaN an invalid operation.
pub const FCMP: u32 = 0x1e20_2000;
pub const FCMP_ZERO: u32 = 0x8;
pub const FCMP_SIGNAL: u32 = 0x10;
// Conversions between a 32-bit core register and an S register.
pub const SCVTF: u32 = 0x1e22_0000;
pub const UCVTF: u32 = 0x1e23_0000;
pub const FCVTZS: u32 = 0x1e38_0000;
pub const FCVTZU: u32 = 0x1e39_0000;
pub const DOUBLE: u32 = 1 << 22;

// Advanced SIMD. `QUAD` makes an operation on 64 bits one on 128 bits, `UNSIGNED`
// selects the second operation of a pair, and the element size goes to bit 22.
pub const QUAD: u32 = 1 << 30;
pub const UNSIGNED: u32 = 1 << 29;
pub const V_AND: u32 = 0x0e20_1c00;
pub const V_BIC: u32 = 0x0e60_1c00;
pub const V_ORR: u32 = 0x0ea0_1c00;
pub const V_ORN: u32 = 0x0ee0_1c00;
pub const V_EOR: u32 = 0x2e20_1c00;
pub const V_BSL: u32 = 0x2e60_1c00;
pub const V_BIT: u32 = 0x2ea0_1c00;
pub const V_BIF: u32 = 0x2ee0_1c00;
/// ST3 and ST4 of registers in sequence at the address in a register, with no offset.
/// The element size goes to bit 10.
pub const V_ST3: u32 = 0x0c00_4000;
pub const V_ST4: u32 = 0x0c00_0000;
/// Makes them LD3 and LD4.
pub const V_STRUCTURE_LOAD: u32 = 1 << 22;
/// The low bits of a core register to each element. Bit 16 shifted by the log2 of the
/// element size selects the size.
pub const V_DUP: u32 = 0x0e00_0c00;
/// A 32-bit core register to the 4 elements of a 128-bit register.
pub const V_DUP_WORD: u32 = 0x4e04_0c00;
/// The largest of the 4 elements of 32 bits, to the low element.
pub const V_UMAXV_WORDS: u32 = 0x6eb0_a800;
/// The low 32 bits of an FP register to a core register.
pub const FMOV_TO_CORE: u32 = 0x1e26_0000;
/// FABS of 4 single-precision elements.
pub const V_FABS: u32 = 0x4ea0_f800;
pub const V_CMTST: u32 = 0x0e20_8c00;
// Compares of single-precision elements.
pub const V_FCMEQ: u32 = 0x0e20_e400;
pub const V_FCMGE: u32 = 0x2e20_e400;
pub const V_FCMGT: u32 = 0x2ea0_e400;
/// NOT of one register.
pub const V_NOT: u32 = 0x2e20_5800;
/// SQADD and UQADD.
pub const V_QADD: u32 = 0x0e20_0c00;
pub const V_QSUB: u32 = 0x0e20_2c00;
/// SSHL and USHL: a shift by the signed low byte of each element of the second operand.
pub const V_SHL: u32 = 0x0e20_4400;
/// CMGT and CMHI.
pub const V_CMGT: u32 = 0x0e20_3400;
/// CMGE and CMHS.
pub const V_CMGE: u32 = 0x0e20_3c00;
pub const V_CMEQ: u32 = 0x2e20_8c00;
pub const V_CMLT_ZERO: u32 = 0x0e20_a800;
pub const V_ADD: u32 = 0x0e20_8400;
pub const V_SUB: u32 = 0x2e20_8400;
pub const V_MUL: u32 = 0x0e20_9c00;
pub const V_MLA: u32 = 0x0e20_9400;
pub const V_MLS: u32 = 0x2e20_9400;
/// SMULL and UMULL, SADDL and UADDL: the result has elements of two times the size.
pub const V_MULL: u32 = 0x0e20_c000;
pub const V_ADDL: u32 = 0x0e20_0000;
/// SQXTN and UQXTN: the result has elements of half the size.
pub const V_QXTN: u32 = 0x0e21_4800;
// Shifts by an immediate: the fields immh:immb go to bit 16.
/// SSHR and USHR.
pub const V_SHR: u32 = 0x0f00_0400;
/// SRSHR and URSHR.
pub const V_RSHR: u32 = 0x0f00_2400;
pub const V_RSHRN: u32 = 0x0f00_8c00;
/// SSHLL and USHLL.
pub const V_SHLL: u32 = 0x0f00_a400;

/// Loads and stores of core registers: the bits 31:30 and 23:22 of the encodings.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Mem(u32);
pub const LDRB: Mem = Mem(0x0040_0000);
pub const LDRSB: Mem = Mem(0x00c0_0000);
pub const LDRH: Mem = Mem(0x4040_0000);
pub const LDRSH: Mem = Mem(0x40c0_0000);
pub const LDR: Mem = Mem(0x8040_0000);
pub const LDRX: Mem = Mem(0xc040_0000);
pub const STRB: Mem = Mem(0x0000_0000);
pub const STRH: Mem = Mem(0x4000_0000);
pub const STR: Mem = Mem(0x8000_0000);
pub const STRX: Mem = Mem(0xc000_0000);
// The same for the FP and SIMD registers S, D and Q.
pub const LDRS: Mem = Mem(0x8440_0000);
pub const STRS: Mem = Mem(0x8400_0000);
pub const LDRD: Mem = Mem(0xc440_0000);
pub const STRD: Mem = Mem(0xc400_0000);
pub const LDRQ: Mem = Mem(0x04c0_0000);
pub const STRQ: Mem = Mem(0x0480_0000);

impl Mem {
    /// The log2 of the size of the access in bytes.
    fn scale(self) -> u32 {
        if self == LDRQ || self == STRQ {
            4
        } else {
            self.0 >> 30
        }
    }
}

/// A branch: the offset in words goes to the low bits.
pub const B: u32 = 0x1400_0000;

/// The condition "carry set".
pub const CS: u32 = 2;

// System registers, as the operand of MRS and MSR.
pub const NZCV: u32 = 0x000b_4200;
pub const FPSR: u32 = 0x000b_4420;

#[derive(Clone, Copy)]
pub struct Label(usize);

/// The fields of a 32-bit logical immediate, or None when the value has no encoding.
pub fn logical_imm(value: u32) -> Option<u32> {
    if value == 0 || value == u32::MAX {
        return None;
    }
    for size in [2u32, 4, 8, 16, 32] {
        let mask = u32::MAX >> (32 - size);
        let element = value & mask;
        if (0..32 / size).any(|index| value >> (index * size) & mask != element) {
            continue;
        }
        let ones = element.count_ones();
        let run = u32::MAX >> (32 - ones);
        let rotate_left =
            |by: u32| (element << by | element.checked_shr(size - by).unwrap_or(0)) & mask;
        if let Some(immr) = (0..size).find(|&by| rotate_left(by) == run) {
            let imms = (!(size * 2 - 1) & 0x3f) | (ones - 1);
            return Some(immr << 16 | imms << 10);
        }
    }
    None
}

pub struct Asm {
    /// The address of the first word.
    base: usize,
    pub words: Vec<u32>,
    labels: Vec<Option<usize>>,
    /// The branches to labels: the index of the word and the label.
    fixups: Vec<(usize, usize)>,
}

impl Asm {
    pub fn new(base: usize) -> Self {
        Asm {
            base,
            words: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
        }
    }

    pub fn base(&self) -> usize {
        self.base
    }

    /// The address of the next word.
    pub fn here(&self) -> usize {
        self.base + self.words.len() * 4
    }

    pub fn emit(&mut self, word: u32) {
        self.words.push(word);
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label.0] = Some(self.words.len());
    }

    /// Writes the offsets of the branches to labels. Each label must have a place.
    pub fn finish(&mut self) {
        for &(at, label) in &self.fixups {
            let offset = (self.labels[label].unwrap() as i64 - at as i64) as u32;
            let word = &mut self.words[at];
            if *word & 0x7c00_0000 == 0x1400_0000 {
                *word |= offset & 0x03ff_ffff;
            } else if *word & 0x7e00_0000 == 0x3600_0000 {
                *word |= (offset & 0x3fff) << 5;
            } else {
                *word |= (offset & 0x7_ffff) << 5;
            }
        }
        self.fixups.clear();
    }

    fn branch_to(&mut self, word: u32, label: Label) {
        self.fixups.push((self.words.len(), label.0));
        self.emit(word);
    }

    fn offset_to(&self, address: usize) -> u32 {
        let offset = (address as i64 - self.here() as i64) / 4;
        assert!((-(1 << 25)..1 << 25).contains(&offset));
        offset as u32 & 0x03ff_ffff
    }

    pub fn rrr(&mut self, op: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(op | u32::from(rm) << 16 | u32::from(rn) << 5 | u32::from(rd));
    }

    /// A shifted-register operation: `rm` shifted by `amount`.
    pub fn shifted(&mut self, op: u32, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) {
        self.rrr(op | shift << 22 | amount << 10, rd, rn, rm);
    }

    pub fn rr(&mut self, op: u32, rd: Reg, rn: Reg) {
        self.emit(op | u32::from(rn) << 5 | u32::from(rd));
    }

    pub fn mov(&mut self, rd: Reg, rm: Reg) {
        self.rrr(ORR, rd, ZR, rm);
    }

    pub fn mov_x(&mut self, rd: Reg, rm: Reg) {
        self.rrr(ORR | X, rd, ZR, rm);
    }

    /// ADD or SUB of an immediate of 12 bits, or of 12 bits shifted by 12 with `1 << 22`.
    pub fn imm12(&mut self, op: u32, rd: Reg, rn: Reg, imm: u32) {
        self.emit(op | imm << 10 | u32::from(rn) << 5 | u32::from(rd));
    }

    /// A logical operation with an immediate. Returns false when the value has no encoding.
    pub fn logical(&mut self, op: u32, rd: Reg, rn: Reg, value: u32) -> bool {
        let Some(fields) = logical_imm(value) else {
            return false;
        };
        self.emit(op | fields | u32::from(rn) << 5 | u32::from(rd));
        true
    }

    pub fn mov_imm(&mut self, rd: Reg, value: u32) {
        let rd32 = u32::from(rd);
        let (low, high) = (value & 0xffff, value >> 16);
        if high == 0 {
            self.emit(0x5280_0000 | low << 5 | rd32);
        } else if low == 0 {
            self.emit(0x52a0_0000 | high << 5 | rd32);
        } else if !value <= 0xffff {
            self.emit(0x1280_0000 | !value << 5 | rd32);
        } else if !self.logical(ORR_IMM, rd, ZR, value) {
            self.emit(0x5280_0000 | low << 5 | rd32);
            self.emit(0x72a0_0000 | high << 5 | rd32);
        }
    }

    pub fn mov_imm64(&mut self, rd: Reg, value: u64) {
        self.emit(0xd280_0000 | (value as u32 & 0xffff) << 5 | u32::from(rd));
        for part in 1..4 {
            let bits = (value >> (part * 16)) as u32 & 0xffff;
            if bits != 0 {
                self.emit(0xf280_0000 | part << 21 | bits << 5 | u32::from(rd));
            }
        }
    }

    /// MOVK of the high half of a 32-bit register.
    pub fn movk_high(&mut self, rd: Reg, imm: u32) {
        self.emit(0x72a0_0000 | imm << 5 | u32::from(rd));
    }

    /// `rd = rn + value` with 32-bit registers and no flags. `rd` and `rn` are not `temp`.
    pub fn add_imm(&mut self, rd: Reg, rn: Reg, value: u32, temp: Reg) {
        let negative = value.wrapping_neg();
        if value < 0x1000 {
            self.imm12(ADD_IMM, rd, rn, value);
        } else if negative < 0x1000 {
            self.imm12(SUB_IMM, rd, rn, negative);
        } else {
            self.mov_imm(temp, value);
            self.rrr(ADD, rd, rn, temp);
        }
    }

    /// `rd = rn + value` with 64-bit registers, for a value below 2^24.
    pub fn add_imm_x(&mut self, rd: Reg, rn: Reg, value: usize) {
        assert!(value < 1 << 24);
        let (low, high) = (value as u32 & 0xfff, value as u32 >> 12);
        self.imm12(ADD_IMM | X, rd, rn, low);
        if high != 0 {
            self.imm12(ADD_IMM | X | 1 << 22, rd, rd, high);
        }
    }

    /// `rd = rn + rm`, with `rm` a 32-bit number with no sign.
    pub fn add_uxtw(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.rrr(0x8b20_4000, rd, rn, rm);
    }

    pub fn bitfield(&mut self, op: u32, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        self.emit(op | immr << 16 | imms << 10 | u32::from(rn) << 5 | u32::from(rd));
    }

    pub fn ubfx(&mut self, rd: Reg, rn: Reg, lsb: u32, width: u32) {
        self.bitfield(UBFM, rd, rn, lsb, lsb + width - 1);
    }

    pub fn lsl(&mut self, rd: Reg, rn: Reg, amount: u32) {
        self.bitfield(UBFM, rd, rn, (32 - amount) % 32, 31 - amount);
    }

    pub fn lsr(&mut self, rd: Reg, rn: Reg, amount: u32) {
        self.bitfield(UBFM, rd, rn, amount, 31);
    }

    pub fn lsr_x(&mut self, rd: Reg, rn: Reg, amount: u32) {
        self.bitfield(UBFM | X | 1 << 22, rd, rn, amount, 63);
    }

    /// `rd = (rn:rm) >> lsb`, 32 bits. With `rn == rm` it is a rotation to the right.
    pub fn extr(&mut self, rd: Reg, rn: Reg, rm: Reg, lsb: u32) {
        self.rrr(0x1380_0000 | lsb << 10, rd, rn, rm);
    }

    /// `rd = ra + rn * rm` or one of the other multiplies with a third operand.
    pub fn multiply(&mut self, op: u32, rd: Reg, rn: Reg, rm: Reg, ra: Reg) {
        self.rrr(op | u32::from(ra) << 10, rd, rn, rm);
    }

    /// `rd = 1` when the condition is true, else 0.
    pub fn cset(&mut self, rd: Reg, cond: u32) {
        self.emit(0x1a9f_07e0 | (cond ^ 1) << 12 | u32::from(rd));
    }

    pub fn mrs(&mut self, rt: Reg, register: u32) {
        self.emit(0xd530_0000 | register | u32::from(rt));
    }

    pub fn msr(&mut self, register: u32, rt: Reg) {
        self.emit(0xd510_0000 | register | u32::from(rt));
    }

    /// A load or a store at `rn + offset`. The offset is a multiple of the access size.
    pub fn mem(&mut self, op: Mem, rt: Reg, rn: Reg, offset: usize) {
        let scaled = (offset >> op.scale()) as u32;
        assert!(offset & ((1 << op.scale()) - 1) == 0 && scaled < 0x1000);
        self.emit(0x3900_0000 | op.0 | scaled << 10 | u32::from(rn) << 5 | u32::from(rt));
    }

    /// A load or a store at `rn + rm`, with `rm` a 32-bit number with no sign.
    pub fn mem_uxtw(&mut self, op: Mem, rt: Reg, rn: Reg, rm: Reg) {
        self.rrr(0x3820_4800 | op.0, rt, rn, rm);
    }

    /// LDP or STP of two 32-bit registers at `rn + offset`.
    pub fn pair(&mut self, load: bool, rt: Reg, rt2: Reg, rn: Reg, offset: usize) {
        let op = if load { 0x2940_0000 } else { 0x2900_0000 };
        assert!(offset.is_multiple_of(4) && offset < 256);
        self.emit(
            op | (offset as u32 / 4) << 15
                | u32::from(rt2) << 10
                | u32::from(rn) << 5
                | u32::from(rt),
        );
    }

    /// LDP or STP of two 64-bit registers at `rn + offset`.
    pub fn pair_x(&mut self, load: bool, rt: Reg, rt2: Reg, rn: Reg, offset: usize) {
        let op = if load { 0xa940_0000 } else { 0xa900_0000 };
        assert!(offset.is_multiple_of(8) && offset < 512);
        self.emit(
            op | (offset as u32 / 8) << 15
                | u32::from(rt2) << 10
                | u32::from(rn) << 5
                | u32::from(rt),
        );
    }

    pub fn b(&mut self, label: Label) {
        self.branch_to(0x1400_0000, label);
    }

    pub fn b_to(&mut self, address: usize) {
        let offset = self.offset_to(address);
        self.emit(0x1400_0000 | offset);
    }

    pub fn bl_to(&mut self, address: usize) {
        let offset = self.offset_to(address);
        self.emit(0x9400_0000 | offset);
    }

    pub fn b_cond(&mut self, cond: u32, label: Label) {
        self.branch_to(0x5400_0000 | cond, label);
    }

    /// CBZ of a 32-bit register.
    pub fn cbz(&mut self, rt: Reg, label: Label) {
        self.branch_to(0x3400_0000 | u32::from(rt), label);
    }

    pub fn cbnz(&mut self, rt: Reg, label: Label) {
        self.branch_to(0x3500_0000 | u32::from(rt), label);
    }

    pub fn cbnz_x(&mut self, rt: Reg, label: Label) {
        self.branch_to(0xb500_0000 | u32::from(rt), label);
    }

    pub fn tbnz(&mut self, rt: Reg, bit: u32, label: Label) {
        self.branch_to(
            0x3700_0000 | (bit >> 5) << 31 | (bit & 31) << 19 | u32::from(rt),
            label,
        );
    }

    pub fn br(&mut self, rn: Reg) {
        self.emit(0xd61f_0000 | u32::from(rn) << 5);
    }

    pub fn blr(&mut self, rn: Reg) {
        self.emit(0xd63f_0000 | u32::from(rn) << 5);
    }

    pub fn ret(&mut self) {
        self.emit(0xd65f_03c0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(emit: impl FnOnce(&mut Asm)) -> Vec<u32> {
        let mut asm = Asm::new(0x1000);
        emit(&mut asm);
        asm.finish();
        asm.words
    }

    #[test]
    fn instructions_have_the_words_of_the_host_assembler() {
        // The text, the words that clang 17 gives for it, and the words of the assembler.
        let table: &[(&str, &[u32], Vec<u32>)] = &[
            (
                "add w3, w4, w5",
                &[0x0b05_0083],
                words(|a| a.rrr(ADD, 3, 4, 5)),
            ),
            (
                "adds w3, w4, w5",
                &[0x2b05_0083],
                words(|a| a.rrr(ADDS, 3, 4, 5)),
            ),
            (
                "sub w3, w4, w5",
                &[0x4b05_0083],
                words(|a| a.rrr(SUB, 3, 4, 5)),
            ),
            (
                "subs w3, w4, w5",
                &[0x6b05_0083],
                words(|a| a.rrr(SUBS, 3, 4, 5)),
            ),
            (
                "and w3, w4, w5",
                &[0x0a05_0083],
                words(|a| a.rrr(AND, 3, 4, 5)),
            ),
            (
                "bic w3, w4, w5",
                &[0x0a25_0083],
                words(|a| a.rrr(BIC, 3, 4, 5)),
            ),
            (
                "orr w3, w4, w5",
                &[0x2a05_0083],
                words(|a| a.rrr(ORR, 3, 4, 5)),
            ),
            (
                "orn w3, w4, w5",
                &[0x2a25_0083],
                words(|a| a.rrr(ORN, 3, 4, 5)),
            ),
            (
                "eor w3, w4, w5",
                &[0x4a05_0083],
                words(|a| a.rrr(EOR, 3, 4, 5)),
            ),
            (
                "ands w3, w4, w5",
                &[0x6a05_0083],
                words(|a| a.rrr(ANDS, 3, 4, 5)),
            ),
            (
                "add w3, w4, w5, lsl #7",
                &[0x0b05_1c83],
                words(|a| a.shifted(ADD, 3, 4, 5, LSL, 7)),
            ),
            (
                "sub w3, w4, w5, asr #3",
                &[0x4b85_0c83],
                words(|a| a.shifted(SUB, 3, 4, 5, ASR, 3)),
            ),
            (
                "subs w3, w4, w5, lsr #31",
                &[0x6b45_7c83],
                words(|a| a.shifted(SUBS, 3, 4, 5, LSR, 31)),
            ),
            (
                "and w3, w4, w5, ror #9",
                &[0x0ac5_2483],
                words(|a| a.shifted(AND, 3, 4, 5, ROR, 9)),
            ),
            (
                "orr w17, wzr, w5, lsr #1",
                &[0x2a45_07f1],
                words(|a| a.shifted(ORR, 17, ZR, 5, LSR, 1)),
            ),
            (
                "orn w3, wzr, w5, asr #31",
                &[0x2aa5_7fe3],
                words(|a| a.shifted(ORN, 3, ZR, 5, ASR, 31)),
            ),
            (
                "add x17, x25, x17, lsl #4",
                &[0x8b11_1331],
                words(|a| a.shifted(ADD | X, 17, 25, 17, LSL, 4)),
            ),
            (
                "eor x16, x16, x17",
                &[0xca11_0210],
                words(|a| a.rrr(EOR | X, 16, 16, 17)),
            ),
            (
                "tst w4, w4",
                &[0x6a04_009f],
                words(|a| a.rrr(ANDS, ZR, 4, 4)),
            ),
            (
                "cmp w4, w5",
                &[0x6b05_009f],
                words(|a| a.rrr(SUBS, ZR, 4, 5)),
            ),
            (
                "cmn w4, w5, lsl #2",
                &[0x2b05_089f],
                words(|a| a.shifted(ADDS, ZR, 4, 5, LSL, 2)),
            ),
            (
                "neg w3, w4",
                &[0x4b04_03e3],
                words(|a| a.rrr(SUB, 3, ZR, 4)),
            ),
            (
                "adc w3, w4, w5",
                &[0x1a05_0083],
                words(|a| a.rrr(ADC, 3, 4, 5)),
            ),
            (
                "adcs w3, w4, w5",
                &[0x3a05_0083],
                words(|a| a.rrr(ADCS, 3, 4, 5)),
            ),
            (
                "sbc w3, w4, w5",
                &[0x5a05_0083],
                words(|a| a.rrr(SBC, 3, 4, 5)),
            ),
            (
                "sbcs w3, w4, w5",
                &[0x7a05_0083],
                words(|a| a.rrr(SBCS, 3, 4, 5)),
            ),
            (
                "udiv w3, w4, w5",
                &[0x1ac5_0883],
                words(|a| a.rrr(UDIV, 3, 4, 5)),
            ),
            (
                "sdiv w3, w4, w5",
                &[0x1ac5_0c83],
                words(|a| a.rrr(SDIV, 3, 4, 5)),
            ),
            (
                "lsl w3, w4, w5",
                &[0x1ac5_2083],
                words(|a| a.rrr(LSLV, 3, 4, 5)),
            ),
            (
                "lsr w3, w4, w5",
                &[0x1ac5_2483],
                words(|a| a.rrr(LSRV, 3, 4, 5)),
            ),
            (
                "asr w3, w4, w5",
                &[0x1ac5_2883],
                words(|a| a.rrr(ASRV, 3, 4, 5)),
            ),
            (
                "ror w3, w4, w5",
                &[0x1ac5_2c83],
                words(|a| a.rrr(RORV, 3, 4, 5)),
            ),
            ("clz w3, w4", &[0x5ac0_1083], words(|a| a.rr(CLZ, 3, 4))),
            ("rbit w3, w4", &[0x5ac0_0083], words(|a| a.rr(RBIT, 3, 4))),
            ("rev w3, w4", &[0x5ac0_0883], words(|a| a.rr(REV, 3, 4))),
            (
                "add w3, w4, #0xabc",
                &[0x112a_f083],
                words(|a| a.imm12(ADD_IMM, 3, 4, 0xabc)),
            ),
            (
                "adds w3, w4, #1",
                &[0x3100_0483],
                words(|a| a.imm12(ADDS_IMM, 3, 4, 1)),
            ),
            (
                "sub w3, w4, #0xfff",
                &[0x513f_fc83],
                words(|a| a.imm12(SUB_IMM, 3, 4, 0xfff)),
            ),
            (
                "subs w3, w4, #0",
                &[0x7100_0083],
                words(|a| a.imm12(SUBS_IMM, 3, 4, 0)),
            ),
            (
                "add w3, w4, #0x12, lsl #12",
                &[0x1140_4883],
                words(|a| a.imm12(ADD_IMM | 1 << 22, 3, 4, 0x12)),
            ),
            (
                "add x26, x26, #5",
                &[0x9100_175a],
                words(|a| a.imm12(ADD_IMM | X, 26, 26, 5)),
            ),
            (
                "sub x26, x26, #64",
                &[0xd101_035a],
                words(|a| a.imm12(SUB_IMM | X, 26, 26, 64)),
            ),
            (
                "cmp w4, #7",
                &[0x7100_1c9f],
                words(|a| a.imm12(SUBS_IMM, ZR, 4, 7)),
            ),
            (
                "and w3, w4, #0xff",
                &[0x1200_1c83],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0xff))),
            ),
            (
                "and w3, w4, #0xe0",
                &[0x121b_0883],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0xe0))),
            ),
            (
                "and w3, w4, #0x3",
                &[0x1200_0483],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0x3))),
            ),
            (
                "and w3, w4, #0x1",
                &[0x1200_0083],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0x1))),
            ),
            (
                "and w3, w4, #0xfffffffe",
                &[0x121f_7883],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0xfffffffe))),
            ),
            (
                "and w3, w4, #0xfffff000",
                &[0x1214_4c83],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0xfffff000))),
            ),
            (
                "and w3, w4, #0x20000000",
                &[0x1203_0083],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0x20000000))),
            ),
            (
                "and w3, w4, #0x10000000",
                &[0x1204_0083],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0x10000000))),
            ),
            (
                "and w3, w4, #0xff00ff00",
                &[0x1208_9c83],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0xff00ff00))),
            ),
            (
                "and w3, w4, #0x7ffffffe",
                &[0x121f_7483],
                words(|a| assert!(a.logical(AND_IMM, 3, 4, 0x7ffffffe))),
            ),
            (
                "orr w3, w4, #0x20000000",
                &[0x3203_0083],
                words(|a| assert!(a.logical(ORR_IMM, 3, 4, 0x2000_0000))),
            ),
            (
                "eor w3, w4, #0x55555555",
                &[0x5200_f083],
                words(|a| assert!(a.logical(EOR_IMM, 3, 4, 0x5555_5555))),
            ),
            (
                "sxtb w3, w4",
                &[0x1300_1c83],
                words(|a| a.bitfield(SBFM, 3, 4, 0, 7)),
            ),
            (
                "sxth w3, w4",
                &[0x1300_3c83],
                words(|a| a.bitfield(SBFM, 3, 4, 0, 15)),
            ),
            (
                "sbfx w3, w4, #5, #6",
                &[0x1305_2883],
                words(|a| a.bitfield(SBFM, 3, 4, 5, 10)),
            ),
            (
                "asr w3, w4, #31",
                &[0x131f_7c83],
                words(|a| a.bitfield(SBFM, 3, 4, 31, 31)),
            ),
            (
                "bfi w3, w4, #8, #4",
                &[0x3318_0c83],
                words(|a| a.bitfield(BFM, 3, 4, 24, 3)),
            ),
            (
                "bfxil w3, w4, #0, #16",
                &[0x3300_3c83],
                words(|a| a.bitfield(BFM, 3, 4, 0, 15)),
            ),
            (
                "ubfx w3, w4, #5, #6",
                &[0x5305_2883],
                words(|a| a.ubfx(3, 4, 5, 6)),
            ),
            (
                "ubfx w3, w4, #31, #1",
                &[0x531f_7c83],
                words(|a| a.ubfx(3, 4, 31, 1)),
            ),
            (
                "ubfx w17, w16, #12, #8",
                &[0x530c_4e11],
                words(|a| a.ubfx(17, 16, 12, 8)),
            ),
            (
                "ubfx x17, x16, #1, #12",
                &[0xd341_3211],
                words(|a| a.bitfield(UBFM | X | 1 << 22, 17, 16, 1, 12)),
            ),
            ("lsl w3, w4, #7", &[0x5319_6083], words(|a| a.lsl(3, 4, 7))),
            (
                "lsl w3, w4, #31",
                &[0x5301_0083],
                words(|a| a.lsl(3, 4, 31)),
            ),
            ("lsl w3, w4, #0", &[0x5300_7c83], words(|a| a.lsl(3, 4, 0))),
            ("lsr w3, w4, #7", &[0x5307_7c83], words(|a| a.lsr(3, 4, 7))),
            (
                "lsr w3, w4, #31",
                &[0x531f_7c83],
                words(|a| a.lsr(3, 4, 31)),
            ),
            (
                "lsr x17, x0, #32",
                &[0xd360_fc11],
                words(|a| a.lsr_x(17, 0, 32)),
            ),
            (
                "extr w3, w4, w5, #7",
                &[0x1385_1c83],
                words(|a| a.extr(3, 4, 5, 7)),
            ),
            (
                "ror w3, w4, #31",
                &[0x1384_7c83],
                words(|a| a.extr(3, 4, 4, 31)),
            ),
            (
                "madd w3, w4, w5, w6",
                &[0x1b05_1883],
                words(|a| a.multiply(MADD, 3, 4, 5, 6)),
            ),
            (
                "msub w3, w4, w5, w6",
                &[0x1b05_9883],
                words(|a| a.multiply(MSUB, 3, 4, 5, 6)),
            ),
            (
                "mul w3, w4, w5",
                &[0x1b05_7c83],
                words(|a| a.multiply(MADD, 3, 4, 5, ZR)),
            ),
            (
                "umaddl x3, w4, w5, x6",
                &[0x9ba5_1883],
                words(|a| a.multiply(UMADDL, 3, 4, 5, 6)),
            ),
            (
                "smaddl x3, w4, w5, x6",
                &[0x9b25_1883],
                words(|a| a.multiply(SMADDL, 3, 4, 5, 6)),
            ),
            (
                "umull x3, w4, w5",
                &[0x9ba5_7c83],
                words(|a| a.multiply(UMADDL, 3, 4, 5, ZR)),
            ),
            (
                "smull x3, w4, w5",
                &[0x9b25_7c83],
                words(|a| a.multiply(SMADDL, 3, 4, 5, ZR)),
            ),
            (
                "fadd s0, s1, s2",
                &[0x1e22_2820],
                words(|a| a.rrr(FADD, 0, 1, 2)),
            ),
            (
                "fadd d0, d1, d2",
                &[0x1e62_2820],
                words(|a| a.rrr(FADD | DOUBLE, 0, 1, 2)),
            ),
            (
                "fsub s0, s1, s2",
                &[0x1e22_3820],
                words(|a| a.rrr(FSUB, 0, 1, 2)),
            ),
            (
                "fsub d0, d1, d2",
                &[0x1e62_3820],
                words(|a| a.rrr(FSUB | DOUBLE, 0, 1, 2)),
            ),
            (
                "fmul s0, s1, s2",
                &[0x1e22_0820],
                words(|a| a.rrr(FMUL, 0, 1, 2)),
            ),
            (
                "fmul d0, d1, d2",
                &[0x1e62_0820],
                words(|a| a.rrr(FMUL | DOUBLE, 0, 1, 2)),
            ),
            (
                "fdiv s0, s1, s2",
                &[0x1e22_1820],
                words(|a| a.rrr(FDIV, 0, 1, 2)),
            ),
            (
                "fdiv d0, d1, d2",
                &[0x1e62_1820],
                words(|a| a.rrr(FDIV | DOUBLE, 0, 1, 2)),
            ),
            ("fabs s0, s1", &[0x1e20_c020], words(|a| a.rr(FABS, 0, 1))),
            (
                "fabs d0, d1",
                &[0x1e60_c020],
                words(|a| a.rr(FABS | DOUBLE, 0, 1)),
            ),
            ("fneg s0, s1", &[0x1e21_4020], words(|a| a.rr(FNEG, 0, 1))),
            (
                "fneg d0, d1",
                &[0x1e61_4020],
                words(|a| a.rr(FNEG | DOUBLE, 0, 1)),
            ),
            ("fsqrt s0, s1", &[0x1e21_c020], words(|a| a.rr(FSQRT, 0, 1))),
            (
                "fsqrt d0, d1",
                &[0x1e61_c020],
                words(|a| a.rr(FSQRT | DOUBLE, 0, 1)),
            ),
            (
                "fcmp s1, s2",
                &[0x1e22_2020],
                words(|a| a.rrr(FCMP, 0, 1, 2)),
            ),
            (
                "fcmp d1, d2",
                &[0x1e62_2020],
                words(|a| a.rrr(FCMP | DOUBLE, 0, 1, 2)),
            ),
            (
                "fcmpe s1, s2",
                &[0x1e22_2030],
                words(|a| a.rrr(FCMP | FCMP_SIGNAL, 0, 1, 2)),
            ),
            (
                "fcmp s1, #0.0",
                &[0x1e20_2028],
                words(|a| a.rrr(FCMP | FCMP_ZERO, 0, 1, 0)),
            ),
            (
                "fcmpe d1, #0.0",
                &[0x1e60_2038],
                words(|a| a.rrr(FCMP | FCMP_ZERO | FCMP_SIGNAL | DOUBLE, 0, 1, 0)),
            ),
            ("scvtf s0, w3", &[0x1e22_0060], words(|a| a.rr(SCVTF, 0, 3))),
            ("ucvtf s0, w3", &[0x1e23_0060], words(|a| a.rr(UCVTF, 0, 3))),
            (
                "scvtf d0, w3",
                &[0x1e62_0060],
                words(|a| a.rr(SCVTF | DOUBLE, 0, 3)),
            ),
            (
                "ucvtf d0, w3",
                &[0x1e63_0060],
                words(|a| a.rr(UCVTF | DOUBLE, 0, 3)),
            ),
            (
                "fcvtzs w3, s0",
                &[0x1e38_0003],
                words(|a| a.rr(FCVTZS, 3, 0)),
            ),
            (
                "fcvtzu w3, s0",
                &[0x1e39_0003],
                words(|a| a.rr(FCVTZU, 3, 0)),
            ),
            (
                "fcvtzs w3, d0",
                &[0x1e78_0003],
                words(|a| a.rr(FCVTZS | DOUBLE, 3, 0)),
            ),
            (
                "fcvtzu w3, d0",
                &[0x1e79_0003],
                words(|a| a.rr(FCVTZU | DOUBLE, 3, 0)),
            ),
            (
                "and v0.8b, v1.8b, v2.8b",
                &[0x0e22_1c20],
                words(|a| a.rrr(V_AND, 0, 1, 2)),
            ),
            (
                "and v0.16b, v1.16b, v2.16b",
                &[0x4e22_1c20],
                words(|a| a.rrr(V_AND | QUAD, 0, 1, 2)),
            ),
            (
                "bic v0.8b, v1.8b, v2.8b",
                &[0x0e62_1c20],
                words(|a| a.rrr(V_BIC, 0, 1, 2)),
            ),
            (
                "bic v0.16b, v1.16b, v2.16b",
                &[0x4e62_1c20],
                words(|a| a.rrr(V_BIC | QUAD, 0, 1, 2)),
            ),
            (
                "orr v0.8b, v1.8b, v2.8b",
                &[0x0ea2_1c20],
                words(|a| a.rrr(V_ORR, 0, 1, 2)),
            ),
            (
                "orr v0.16b, v1.16b, v2.16b",
                &[0x4ea2_1c20],
                words(|a| a.rrr(V_ORR | QUAD, 0, 1, 2)),
            ),
            (
                "orn v0.8b, v1.8b, v2.8b",
                &[0x0ee2_1c20],
                words(|a| a.rrr(V_ORN, 0, 1, 2)),
            ),
            (
                "orn v0.16b, v1.16b, v2.16b",
                &[0x4ee2_1c20],
                words(|a| a.rrr(V_ORN | QUAD, 0, 1, 2)),
            ),
            (
                "eor v0.8b, v1.8b, v2.8b",
                &[0x2e22_1c20],
                words(|a| a.rrr(V_EOR, 0, 1, 2)),
            ),
            (
                "eor v0.16b, v1.16b, v2.16b",
                &[0x6e22_1c20],
                words(|a| a.rrr(V_EOR | QUAD, 0, 1, 2)),
            ),
            (
                "bsl v0.8b, v1.8b, v2.8b",
                &[0x2e62_1c20],
                words(|a| a.rrr(V_BSL, 0, 1, 2)),
            ),
            (
                "bsl v0.16b, v1.16b, v2.16b",
                &[0x6e62_1c20],
                words(|a| a.rrr(V_BSL | QUAD, 0, 1, 2)),
            ),
            (
                "bit v0.8b, v1.8b, v2.8b",
                &[0x2ea2_1c20],
                words(|a| a.rrr(V_BIT, 0, 1, 2)),
            ),
            (
                "bit v0.16b, v1.16b, v2.16b",
                &[0x6ea2_1c20],
                words(|a| a.rrr(V_BIT | QUAD, 0, 1, 2)),
            ),
            (
                "bif v0.8b, v1.8b, v2.8b",
                &[0x2ee2_1c20],
                words(|a| a.rrr(V_BIF, 0, 1, 2)),
            ),
            (
                "bif v0.16b, v1.16b, v2.16b",
                &[0x6ee2_1c20],
                words(|a| a.rrr(V_BIF | QUAD, 0, 1, 2)),
            ),
            (
                "st3 {v0.8b, v1.8b, v2.8b}, [x17]",
                &[0x0c00_4220],
                words(|a| a.rr(V_ST3, 0, 17)),
            ),
            (
                "st3 {v0.4h, v1.4h, v2.4h}, [x17]",
                &[0x0c00_4620],
                words(|a| a.rr(V_ST3 | 1 << 10, 0, 17)),
            ),
            (
                "st4 {v0.2s, v1.2s, v2.2s, v3.2s}, [x17]",
                &[0x0c00_0a20],
                words(|a| a.rr(V_ST4 | 2 << 10, 0, 17)),
            ),
            (
                "ld3 {v0.8b, v1.8b, v2.8b}, [x17]",
                &[0x0c40_4220],
                words(|a| a.rr(V_ST3 | V_STRUCTURE_LOAD, 0, 17)),
            ),
            (
                "ld4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x17]",
                &[0x0c40_0620],
                words(|a| a.rr(V_ST4 | V_STRUCTURE_LOAD | 1 << 10, 0, 17)),
            ),
            (
                "dup v0.8b, w3",
                &[0x0e01_0c60],
                words(|a| a.rr(V_DUP | 1 << 16, 0, 3)),
            ),
            (
                "dup v0.4h, w3",
                &[0x0e02_0c60],
                words(|a| a.rr(V_DUP | 2 << 16, 0, 3)),
            ),
            (
                "dup v0.2s, w3",
                &[0x0e04_0c60],
                words(|a| a.rr(V_DUP | 4 << 16, 0, 3)),
            ),
            (
                "dup v0.16b, w3",
                &[0x4e01_0c60],
                words(|a| a.rr(V_DUP | QUAD | 1 << 16, 0, 3)),
            ),
            (
                "dup v0.4s, w3",
                &[0x4e04_0c60],
                words(|a| a.rr(V_DUP_WORD, 0, 3)),
            ),
            (
                "umaxv s0, v1.4s",
                &[0x6eb0_a820],
                words(|a| a.rr(V_UMAXV_WORDS, 0, 1)),
            ),
            (
                "fmov w3, s0",
                &[0x1e26_0003],
                words(|a| a.rr(FMOV_TO_CORE, 3, 0)),
            ),
            (
                "fabs v0.4s, v1.4s",
                &[0x4ea0_f820],
                words(|a| a.rr(V_FABS, 0, 1)),
            ),
            (
                "cmtst v0.8b, v1.8b, v2.8b",
                &[0x0e22_8c20],
                words(|a| a.rrr(V_CMTST, 0, 1, 2)),
            ),
            (
                "cmtst v0.4s, v1.4s, v2.4s",
                &[0x4ea2_8c20],
                words(|a| a.rrr(V_CMTST | QUAD | 2 << 22, 0, 1, 2)),
            ),
            (
                "fcmeq v0.2s, v1.2s, v2.2s",
                &[0x0e22_e420],
                words(|a| a.rrr(V_FCMEQ, 0, 1, 2)),
            ),
            (
                "fcmge v0.4s, v1.4s, v2.4s",
                &[0x6e22_e420],
                words(|a| a.rrr(V_FCMGE | QUAD, 0, 1, 2)),
            ),
            (
                "fcmgt v0.2s, v1.2s, v2.2s",
                &[0x2ea2_e420],
                words(|a| a.rrr(V_FCMGT, 0, 1, 2)),
            ),
            (
                "mvn v0.8b, v1.8b",
                &[0x2e20_5820],
                words(|a| a.rr(V_NOT, 0, 1)),
            ),
            (
                "mvn v0.16b, v1.16b",
                &[0x6e20_5820],
                words(|a| a.rr(V_NOT | QUAD, 0, 1)),
            ),
            (
                "sqadd v0.8b, v1.8b, v2.8b",
                &[0x0e22_0c20],
                words(|a| a.rrr(V_QADD, 0, 1, 2)),
            ),
            (
                "uqadd v0.8h, v1.8h, v2.8h",
                &[0x6e62_0c20],
                words(|a| a.rrr(V_QADD | UNSIGNED | QUAD | 1 << 22, 0, 1, 2)),
            ),
            (
                "sqadd v0.2d, v1.2d, v2.2d",
                &[0x4ee2_0c20],
                words(|a| a.rrr(V_QADD | QUAD | 3 << 22, 0, 1, 2)),
            ),
            (
                "sqsub v0.8b, v1.8b, v2.8b",
                &[0x0e22_2c20],
                words(|a| a.rrr(V_QSUB, 0, 1, 2)),
            ),
            (
                "uqsub v0.8h, v1.8h, v2.8h",
                &[0x6e62_2c20],
                words(|a| a.rrr(V_QSUB | UNSIGNED | QUAD | 1 << 22, 0, 1, 2)),
            ),
            (
                "sqsub v0.2d, v1.2d, v2.2d",
                &[0x4ee2_2c20],
                words(|a| a.rrr(V_QSUB | QUAD | 3 << 22, 0, 1, 2)),
            ),
            (
                "sshl v0.8b, v1.8b, v2.8b",
                &[0x0e22_4420],
                words(|a| a.rrr(V_SHL, 0, 1, 2)),
            ),
            (
                "ushl v0.8h, v1.8h, v2.8h",
                &[0x6e62_4420],
                words(|a| a.rrr(V_SHL | UNSIGNED | QUAD | 1 << 22, 0, 1, 2)),
            ),
            (
                "sshl v0.2d, v1.2d, v2.2d",
                &[0x4ee2_4420],
                words(|a| a.rrr(V_SHL | QUAD | 3 << 22, 0, 1, 2)),
            ),
            (
                "cmgt v0.8b, v1.8b, v2.8b",
                &[0x0e22_3420],
                words(|a| a.rrr(V_CMGT, 0, 1, 2)),
            ),
            (
                "cmhi v0.8h, v1.8h, v2.8h",
                &[0x6e62_3420],
                words(|a| a.rrr(V_CMGT | UNSIGNED | QUAD | 1 << 22, 0, 1, 2)),
            ),
            (
                "cmgt v0.2d, v1.2d, v2.2d",
                &[0x4ee2_3420],
                words(|a| a.rrr(V_CMGT | QUAD | 3 << 22, 0, 1, 2)),
            ),
            (
                "cmge v0.8b, v1.8b, v2.8b",
                &[0x0e22_3c20],
                words(|a| a.rrr(V_CMGE, 0, 1, 2)),
            ),
            (
                "cmhs v0.8h, v1.8h, v2.8h",
                &[0x6e62_3c20],
                words(|a| a.rrr(V_CMGE | UNSIGNED | QUAD | 1 << 22, 0, 1, 2)),
            ),
            (
                "cmge v0.2d, v1.2d, v2.2d",
                &[0x4ee2_3c20],
                words(|a| a.rrr(V_CMGE | QUAD | 3 << 22, 0, 1, 2)),
            ),
            (
                "cmeq v0.8b, v1.8b, v2.8b",
                &[0x2e22_8c20],
                words(|a| a.rrr(V_CMEQ, 0, 1, 2)),
            ),
            (
                "cmeq v0.4s, v1.4s, v2.4s",
                &[0x6ea2_8c20],
                words(|a| a.rrr(V_CMEQ | QUAD | 2 << 22, 0, 1, 2)),
            ),
            (
                "cmlt v0.8b, v1.8b, #0",
                &[0x0e20_a820],
                words(|a| a.rr(V_CMLT_ZERO, 0, 1)),
            ),
            (
                "cmlt v0.8h, v1.8h, #0",
                &[0x4e60_a820],
                words(|a| a.rr(V_CMLT_ZERO | QUAD | 1 << 22, 0, 1)),
            ),
            (
                "add v0.8b, v1.8b, v2.8b",
                &[0x0e22_8420],
                words(|a| a.rrr(V_ADD, 0, 1, 2)),
            ),
            (
                "add v0.4s, v1.4s, v2.4s",
                &[0x4ea2_8420],
                words(|a| a.rrr(V_ADD | QUAD | 2 << 22, 0, 1, 2)),
            ),
            (
                "sub v0.8b, v1.8b, v2.8b",
                &[0x2e22_8420],
                words(|a| a.rrr(V_SUB, 0, 1, 2)),
            ),
            (
                "sub v0.4s, v1.4s, v2.4s",
                &[0x6ea2_8420],
                words(|a| a.rrr(V_SUB | QUAD | 2 << 22, 0, 1, 2)),
            ),
            (
                "mul v0.8b, v1.8b, v2.8b",
                &[0x0e22_9c20],
                words(|a| a.rrr(V_MUL, 0, 1, 2)),
            ),
            (
                "mul v0.4s, v1.4s, v2.4s",
                &[0x4ea2_9c20],
                words(|a| a.rrr(V_MUL | QUAD | 2 << 22, 0, 1, 2)),
            ),
            (
                "mla v0.8b, v1.8b, v2.8b",
                &[0x0e22_9420],
                words(|a| a.rrr(V_MLA, 0, 1, 2)),
            ),
            (
                "mla v0.4s, v1.4s, v2.4s",
                &[0x4ea2_9420],
                words(|a| a.rrr(V_MLA | QUAD | 2 << 22, 0, 1, 2)),
            ),
            (
                "mls v0.8b, v1.8b, v2.8b",
                &[0x2e22_9420],
                words(|a| a.rrr(V_MLS, 0, 1, 2)),
            ),
            (
                "mls v0.4s, v1.4s, v2.4s",
                &[0x6ea2_9420],
                words(|a| a.rrr(V_MLS | QUAD | 2 << 22, 0, 1, 2)),
            ),
            (
                "add v0.2d, v1.2d, v2.2d",
                &[0x4ee2_8420],
                words(|a| a.rrr(V_ADD | QUAD | 3 << 22, 0, 1, 2)),
            ),
            (
                "smull v0.8h, v1.8b, v2.8b",
                &[0x0e22_c020],
                words(|a| a.rrr(V_MULL, 0, 1, 2)),
            ),
            (
                "umull v0.2d, v1.2s, v2.2s",
                &[0x2ea2_c020],
                words(|a| a.rrr(V_MULL | UNSIGNED | 2 << 22, 0, 1, 2)),
            ),
            (
                "saddl v0.4s, v1.4h, v2.4h",
                &[0x0e62_0020],
                words(|a| a.rrr(V_ADDL | 1 << 22, 0, 1, 2)),
            ),
            (
                "uaddl v0.8h, v1.8b, v2.8b",
                &[0x2e22_0020],
                words(|a| a.rrr(V_ADDL | UNSIGNED, 0, 1, 2)),
            ),
            (
                "sqxtn v0.8b, v1.8h",
                &[0x0e21_4820],
                words(|a| a.rr(V_QXTN, 0, 1)),
            ),
            (
                "uqxtn v0.2s, v1.2d",
                &[0x2ea1_4820],
                words(|a| a.rr(V_QXTN | UNSIGNED | 2 << 22, 0, 1)),
            ),
            (
                "sshr v0.8b, v1.8b, #3",
                &[0x0f0d_0420],
                words(|a| a.rr(V_SHR | 13 << 16, 0, 1)),
            ),
            (
                "ushr v0.4s, v1.4s, #31",
                &[0x6f21_0420],
                words(|a| a.rr(V_SHR | UNSIGNED | QUAD | 33 << 16, 0, 1)),
            ),
            (
                "sshr v0.2d, v1.2d, #64",
                &[0x4f40_0420],
                words(|a| a.rr(V_SHR | QUAD | 64 << 16, 0, 1)),
            ),
            (
                "srshr v0.4h, v1.4h, #16",
                &[0x0f10_2420],
                words(|a| a.rr(V_RSHR | 16 << 16, 0, 1)),
            ),
            (
                "urshr v0.16b, v1.16b, #1",
                &[0x6f0f_2420],
                words(|a| a.rr(V_RSHR | UNSIGNED | QUAD | 15 << 16, 0, 1)),
            ),
            (
                "rshrn v0.8b, v1.8h, #3",
                &[0x0f0d_8c20],
                words(|a| a.rr(V_RSHRN | 13 << 16, 0, 1)),
            ),
            (
                "rshrn v0.2s, v1.2d, #32",
                &[0x0f20_8c20],
                words(|a| a.rr(V_RSHRN | 32 << 16, 0, 1)),
            ),
            (
                "sshll v0.8h, v1.8b, #0",
                &[0x0f08_a420],
                words(|a| a.rr(V_SHLL | 8 << 16, 0, 1)),
            ),
            (
                "ushll v0.2d, v1.2s, #0",
                &[0x2f20_a420],
                words(|a| a.rr(V_SHLL | UNSIGNED | 32 << 16, 0, 1)),
            ),
            (
                "ldrb w3, [x4, #5]",
                &[0x3940_1483],
                words(|a| a.mem(LDRB, 3, 4, 5)),
            ),
            (
                "ldrsb w3, [x4, #5]",
                &[0x39c0_1483],
                words(|a| a.mem(LDRSB, 3, 4, 5)),
            ),
            (
                "ldrh w3, [x4, #6]",
                &[0x7940_0c83],
                words(|a| a.mem(LDRH, 3, 4, 6)),
            ),
            (
                "ldrsh w3, [x4, #6]",
                &[0x79c0_0c83],
                words(|a| a.mem(LDRSH, 3, 4, 6)),
            ),
            (
                "ldr w3, [x4, #8]",
                &[0xb940_0883],
                words(|a| a.mem(LDR, 3, 4, 8)),
            ),
            (
                "ldr x3, [x4, #16]",
                &[0xf940_0883],
                words(|a| a.mem(LDRX, 3, 4, 16)),
            ),
            (
                "strb w3, [x4, #4095]",
                &[0x393f_fc83],
                words(|a| a.mem(STRB, 3, 4, 4095)),
            ),
            (
                "strh w3, [x4, #8190]",
                &[0x793f_fc83],
                words(|a| a.mem(STRH, 3, 4, 8190)),
            ),
            (
                "str w3, [x4, #60]",
                &[0xb900_3c83],
                words(|a| a.mem(STR, 3, 4, 60)),
            ),
            (
                "str x3, [x4, #32760]",
                &[0xf93f_fc83],
                words(|a| a.mem(STRX, 3, 4, 32760)),
            ),
            (
                "ldr s3, [x4, #8]",
                &[0xbd40_0883],
                words(|a| a.mem(LDRS, 3, 4, 8)),
            ),
            (
                "str s3, [x4, #124]",
                &[0xbd00_7c83],
                words(|a| a.mem(STRS, 3, 4, 124)),
            ),
            (
                "ldr d3, [x4, #16]",
                &[0xfd40_0883],
                words(|a| a.mem(LDRD, 3, 4, 16)),
            ),
            (
                "str d3, [x4, #248]",
                &[0xfd00_7c83],
                words(|a| a.mem(STRD, 3, 4, 248)),
            ),
            (
                "ldr q3, [x4, #32]",
                &[0x3dc0_0883],
                words(|a| a.mem(LDRQ, 3, 4, 32)),
            ),
            (
                "str q3, [x4, #496]",
                &[0x3d80_7c83],
                words(|a| a.mem(STRQ, 3, 4, 496)),
            ),
            (
                "ldr w3, [x4]",
                &[0xb940_0083],
                words(|a| a.mem(LDR, 3, 4, 0)),
            ),
            (
                "ldr x19, [sp, #16]",
                &[0xf940_0bf3],
                words(|a| a.mem(LDRX, 19, SP, 16)),
            ),
            (
                "ldrb w3, [x17, w5, uxtw]",
                &[0x3865_4a23],
                words(|a| a.mem_uxtw(LDRB, 3, 17, 5)),
            ),
            (
                "ldrsb w3, [x17, w5, uxtw]",
                &[0x38e5_4a23],
                words(|a| a.mem_uxtw(LDRSB, 3, 17, 5)),
            ),
            (
                "ldrh w3, [x17, w5, uxtw]",
                &[0x7865_4a23],
                words(|a| a.mem_uxtw(LDRH, 3, 17, 5)),
            ),
            (
                "ldrsh w3, [x17, w5, uxtw]",
                &[0x78e5_4a23],
                words(|a| a.mem_uxtw(LDRSH, 3, 17, 5)),
            ),
            (
                "ldr w3, [x17, w5, uxtw]",
                &[0xb865_4a23],
                words(|a| a.mem_uxtw(LDR, 3, 17, 5)),
            ),
            (
                "strb w3, [x17, w5, uxtw]",
                &[0x3825_4a23],
                words(|a| a.mem_uxtw(STRB, 3, 17, 5)),
            ),
            (
                "strh w3, [x17, w5, uxtw]",
                &[0x7825_4a23],
                words(|a| a.mem_uxtw(STRH, 3, 17, 5)),
            ),
            (
                "str w3, [x17, w5, uxtw]",
                &[0xb825_4a23],
                words(|a| a.mem_uxtw(STR, 3, 17, 5)),
            ),
            (
                "ldp w3, w4, [x5, #8]",
                &[0x2941_10a3],
                words(|a| a.pair(true, 3, 4, 5, 8)),
            ),
            (
                "stp w3, w4, [x5, #252]",
                &[0x291f_90a3],
                words(|a| a.pair(false, 3, 4, 5, 252)),
            ),
            (
                "ldp w0, w1, [x16]",
                &[0x2940_0600],
                words(|a| a.pair(true, 0, 1, 16, 0)),
            ),
            (
                "ldp x19, x20, [sp, #16]",
                &[0xa941_53f3],
                words(|a| a.pair_x(true, 19, 20, SP, 16)),
            ),
            (
                "stp x27, x28, [sp, #80]",
                &[0xa905_73fb],
                words(|a| a.pair_x(false, 27, 28, SP, 80)),
            ),
            ("mov w3, w4", &[0x2a04_03e3], words(|a| a.mov(3, 4))),
            ("mov x3, x4", &[0xaa04_03e3], words(|a| a.mov_x(3, 4))),
            ("mov w3, #0", &[0x5280_0003], words(|a| a.mov_imm(3, 0))),
            (
                "mov w3, #0x1234",
                &[0x5282_4683],
                words(|a| a.mov_imm(3, 0x1234)),
            ),
            (
                "mov w3, #0x12340000",
                &[0x52a2_4683],
                words(|a| a.mov_imm(3, 0x1234_0000)),
            ),
            (
                "mov w3, #0xffffff00",
                &[0x1280_1fe3],
                words(|a| a.mov_imm(3, 0xffff_ff00)),
            ),
            (
                "mov w3, #0xffffffff",
                &[0x1280_0003],
                words(|a| a.mov_imm(3, 0xffff_ffff)),
            ),
            (
                "mov w3, #0xff00ff",
                &[0x3200_9fe3],
                words(|a| a.mov_imm(3, 0x00ff_00ff)),
            ),
            (
                "mov w3, #0x5678; movk w3, #0x1234, lsl #16",
                &[0x528a_cf03, 0x72a2_4683],
                words(|a| a.mov_imm(3, 0x1234_5678)),
            ),
            (
                "mov x3, #0xdef0; movk x3, #0x9abc, lsl #16; movk x3, #0x5678, lsl #32; movk x3, #0x1234, lsl #48",
                &[0xd29b_de03, 0xf2b3_5783, 0xf2ca_cf03, 0xf2e2_4683],
                words(|a| a.mov_imm64(3, 0x1234_5678_9abc_def0)),
            ),
            (
                "mov x3, #2; movk x3, #1, lsl #48",
                &[0xd280_0043, 0xf2e0_0023],
                words(|a| a.mov_imm64(3, 0x0001_0000_0000_0002)),
            ),
            (
                "movz x3, #0; movk x3, #7, lsl #32",
                &[0xd280_0003, 0xf2c0_00e3],
                words(|a| a.mov_imm64(3, 0x0000_0007_0000_0000)),
            ),
            (
                "movk w3, #0xabcd, lsl #16",
                &[0x72b5_79a3],
                words(|a| a.movk_high(3, 0xabcd)),
            ),
            (
                "add w3, w4, #5",
                &[0x1100_1483],
                words(|a| a.add_imm(3, 4, 5, 16)),
            ),
            (
                "sub w3, w4, #5",
                &[0x5100_1483],
                words(|a| a.add_imm(3, 4, 5u32.wrapping_neg(), 16)),
            ),
            (
                "mov w16, #0x2345; movk w16, #1, lsl #16; add w3, w4, w16",
                &[0x5284_68b0, 0x72a0_0030, 0x0b10_0083],
                words(|a| a.add_imm(3, 4, 0x1_2345, 16)),
            ),
            (
                "add x3, x4, #8",
                &[0x9100_2083],
                words(|a| a.add_imm_x(3, 4, 8)),
            ),
            (
                "add x3, x4, #0x456; add x3, x3, #0x123, lsl #12",
                &[0x9111_5883, 0x9144_8c63],
                words(|a| a.add_imm_x(3, 4, 0x12_3456)),
            ),
            (
                "add x17, x17, w5, uxtw",
                &[0x8b25_4231],
                words(|a| a.add_uxtw(17, 17, 5)),
            ),
            ("cset w3, hs", &[0x1a9f_37e3], words(|a| a.cset(3, CS))),
            ("mrs x3, nzcv", &[0xd53b_4203], words(|a| a.mrs(3, NZCV))),
            ("msr nzcv, x3", &[0xd51b_4203], words(|a| a.msr(NZCV, 3))),
            ("mrs x3, fpsr", &[0xd53b_4423], words(|a| a.mrs(3, FPSR))),
            ("msr fpsr, xzr", &[0xd51b_443f], words(|a| a.msr(FPSR, ZR))),
            ("br x17", &[0xd61f_0220], words(|a| a.br(17))),
            ("blr x8", &[0xd63f_0100], words(|a| a.blr(8))),
            ("ret", &[0xd65f_03c0], words(|a| a.ret())),
            ("b . + 4", &[0x1400_0001], words(|a| a.emit(B | 1))),
        ];
        for (text, expected, emitted) in table {
            assert_eq!(emitted, expected, "{text}");
        }
    }

    #[test]
    fn branches_have_the_offsets_of_the_host_assembler() {
        // b 1f; b.hs 1f; b.ne 1f; cbz w3, 1f; cbnz w3, 1f; cbnz x3, 1f; tbnz x3, #63, 1f;
        // tbnz w4, #1, 1f; 1: ret; b 1b; b.hs 1b; cbz w5, 1b; tbnz w28, #0, 1b;
        // b . + 0x40; bl . - 0x38; b . - 0x3c
        let expected = [
            0x1400_0008,
            0x5400_00e2,
            0x5400_00c1,
            0x3400_00a3,
            0x3500_0083,
            0xb500_0063,
            0xb7f8_0043,
            0x3708_0024,
            0xd65f_03c0,
            0x17ff_ffff,
            0x54ff_ffc2,
            0x34ff_ffa5,
            0x3707_ff9c,
            0x1400_0010,
            0x97ff_fff2,
            0x17ff_fff1,
        ];
        let emitted = words(|a| {
            let (forward, back) = (a.label(), a.label());
            a.b(forward);
            a.b_cond(CS, forward);
            a.b_cond(1, forward);
            a.cbz(3, forward);
            a.cbnz(3, forward);
            a.cbnz_x(3, forward);
            a.tbnz(3, 63, forward);
            a.tbnz(4, 1, forward);
            a.bind(forward);
            a.bind(back);
            a.ret();
            a.b(back);
            a.b_cond(CS, back);
            a.cbz(5, back);
            a.tbnz(28, 0, back);
            a.b_to(a.here() + 0x40);
            a.bl_to(a.here() - 0x38);
            a.b_to(a.base());
        });
        assert_eq!(emitted, expected);
    }

    #[test]
    fn each_logical_immediate_gives_its_value_again() {
        // DecodeBitMasks of the architecture, for 32 bits.
        let decode = |fields: u32| {
            let (immr, imms) = (fields >> 16 & 0x3f, fields >> 10 & 0x3f);
            let size = 1u32 << (31 - (!imms & 0x3f).leading_zeros());
            let (ones, rotation) = ((imms & (size - 1)) + 1, immr & (size - 1));
            let run = u32::MAX >> (32 - ones);
            let mask = u32::MAX >> (32 - size);
            let element = (run >> rotation | run << (size - rotation) % size) & mask;
            (0..32 / size).fold(0, |value, index| value | element << (index * size))
        };
        let mut encoded = 0;
        for size in [2u32, 4, 8, 16, 32] {
            for ones in 1..size {
                for rotation in 0..size {
                    let run = u32::MAX >> (32 - ones);
                    let mask = u32::MAX >> (32 - size);
                    let element = (run >> rotation | run << (size - rotation) % size) & mask;
                    let value =
                        (0..32 / size).fold(0, |value, index| value | element << (index * size));
                    let fields = logical_imm(value).unwrap_or_else(|| panic!("{value:#x}"));
                    assert_eq!(decode(fields), value, "{value:#x} as {fields:#x}");
                    encoded += 1;
                }
            }
        }
        assert_eq!(encoded, 2 + 12 + 56 + 240 + 992);
        for value in [0, u32::MAX, 0x1234, 5, 0xffff_fffa, 0x0101_0102] {
            assert_eq!(logical_imm(value), None, "{value:#x}");
        }
    }

    #[test]
    fn logical_immediates_have_the_encodings_of_the_architecture() {
        // (value, N:immr:imms) from the assembler of LLVM.
        assert_eq!(logical_imm(0x3000_0000), Some(4 << 16 | 1 << 10));
        assert_eq!(logical_imm(0xff), Some(7 << 10));
        assert_eq!(logical_imm(0xffff_fffe), Some(31 << 16 | 30 << 10));
        assert_eq!(logical_imm(0x5555_5555), Some(0x3c << 10));
        assert_eq!(logical_imm(0x00ff_00ff), Some(0x27 << 10));
        assert_eq!(logical_imm(0x1234), None);
    }
}
