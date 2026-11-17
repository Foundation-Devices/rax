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
        if self == LDRQ || self == STRQ { 4 } else { self.0 >> 30 }
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
        let rotate_left = |by: u32| (element << by | element.checked_shr(size - by).unwrap_or(0)) & mask;
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
        Asm { base, words: Vec::new(), labels: Vec::new(), fixups: Vec::new() }
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

    fn to_label(&mut self, word: u32, label: Label) {
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
        assert!(offset % 4 == 0 && offset < 256);
        self.emit(op | (offset as u32 / 4) << 15 | u32::from(rt2) << 10 | u32::from(rn) << 5 | u32::from(rt));
    }

    /// LDP or STP of two 64-bit registers at `rn + offset`.
    pub fn pair_x(&mut self, load: bool, rt: Reg, rt2: Reg, rn: Reg, offset: usize) {
        let op = if load { 0xa940_0000 } else { 0xa900_0000 };
        assert!(offset % 8 == 0 && offset < 512);
        self.emit(op | (offset as u32 / 8) << 15 | u32::from(rt2) << 10 | u32::from(rn) << 5 | u32::from(rt));
    }

    pub fn b(&mut self, label: Label) {
        self.to_label(0x1400_0000, label);
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
        self.to_label(0x5400_0000 | cond, label);
    }

    /// CBZ of a 32-bit register.
    pub fn cbz(&mut self, rt: Reg, label: Label) {
        self.to_label(0x3400_0000 | u32::from(rt), label);
    }

    pub fn cbnz(&mut self, rt: Reg, label: Label) {
        self.to_label(0x3500_0000 | u32::from(rt), label);
    }

    pub fn cbnz_x(&mut self, rt: Reg, label: Label) {
        self.to_label(0xb500_0000 | u32::from(rt), label);
    }

    pub fn tbnz(&mut self, rt: Reg, bit: u32, label: Label) {
        self.to_label(0x3700_0000 | (bit >> 5) << 31 | (bit & 31) << 19 | u32::from(rt), label);
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
