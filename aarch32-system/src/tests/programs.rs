//! Random programs with loops, calls, memory accesses and writes to their own code, in
//! Thumb and in A32. Each register has one use, thus a program stays in its code.

use super::Random;
use super::machine::RAM_BASE;
use super::suite::{
    ARM_STOP, CALL_PAGE, CODE, Case, DATA, DEVICE_PAGE, Insn, Program, THUMB_STOP, t16, wide,
};

const AL: u32 = 14;
// The registers of a program.
/// r0 to r3 and r12 take the results.
const SCRATCH: [u32; 5] = [0, 1, 2, 3, 12];
/// A pointer to data that stays.
const BASE: u32 = 4;
/// A pointer to data that moves with each writeback.
const WALK: u32 = 5;
const COUNTER: u32 = 6;
const DEVICE: u32 = 7;
/// A pointer to memory with no direct entry.
const CALL: u32 = 8;
/// The address of the instruction that the program changes, and the changed one.
const SITE: u32 = 9;
const PATCH: u32 = 10;
/// A pointer to data, not a multiple of 4.
const ODD: u32 = 11;
const SP: u32 = 13;
const LR: u32 = 14;

#[derive(Clone, Copy, PartialEq)]
enum Fix {
    /// Thumb B with a condition, 16 bits and 32 bits, B, BL, CBZ; A32 B and BL.
    ThumbCond16,
    ThumbCond32,
    Thumb16,
    Thumb32,
    Cbz,
    Arm,
}

#[derive(Clone, Copy)]
pub struct Label(usize);

pub struct Builder {
    pub thumb: bool,
    at: u32,
    insns: Vec<Insn>,
    labels: Vec<Option<u32>>,
    fixups: Vec<(usize, usize, Fix)>,
    /// The address of an instruction that a store of the program can change.
    pub site: Option<u32>,
}

fn thumb_branch32(base: u32, offset: i32) -> u32 {
    let offset = offset as u32;
    let s = offset >> 24 & 1;
    let j1 = (!(offset >> 23) & 1) ^ s;
    let j2 = (!(offset >> 22) & 1) ^ s;
    base | s << 26 | (offset >> 12 & 0x3ff) << 16 | j1 << 13 | j2 << 11 | offset >> 1 & 0x7ff
}

impl Builder {
    pub fn new(thumb: bool, at: u32) -> Builder {
        Builder {
            thumb,
            at,
            insns: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
            site: None,
        }
    }

    pub fn here(&self) -> u32 {
        self.at
            + self
                .insns
                .iter()
                .map(|insn| u32::from(insn.size))
                .sum::<u32>()
    }

    fn address(&self, index: usize) -> u32 {
        self.at
            + self.insns[..index]
                .iter()
                .map(|insn| u32::from(insn.size))
                .sum::<u32>()
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label.0] = Some(self.here());
    }

    pub fn emit(&mut self, insn: Insn) {
        assert!(self.thumb || insn.size == 4);
        self.insns.push(insn);
    }

    /// A 32-bit instruction: A32 with the condition, or the T32 word.
    fn w(&mut self, cond: u32, arm: u32, thumb: u32) {
        self.emit(wide(if self.thumb { thumb } else { cond << 28 | arm }));
    }

    fn branch(&mut self, raw: u32, size: u8, label: Label, fix: Fix) {
        self.fixups.push((self.insns.len(), label.0, fix));
        self.emit(Insn { raw, size });
    }

    /// B to the label, with a condition or 14.
    pub fn b(&mut self, cond: u32, label: Label, short: bool) {
        match (self.thumb, cond == AL, short) {
            (false, ..) => self.branch(cond << 28 | 0x0a00_0000, 4, label, Fix::Arm),
            (true, true, true) => self.branch(0xe000, 2, label, Fix::Thumb16),
            (true, true, false) => self.branch(0xf000_9000, 4, label, Fix::Thumb32),
            (true, false, true) => self.branch(0xd000 | cond << 8, 2, label, Fix::ThumbCond16),
            (true, false, false) => {
                self.branch(0xf000_8000 | cond << 22, 4, label, Fix::ThumbCond32)
            }
        }
    }

    /// BL, or BLX for a target in the other instruction set.
    pub fn call(&mut self, target: u32, target_thumb: bool) {
        let pc = self.here() + if self.thumb { 4 } else { 8 };
        let raw = match (self.thumb, target_thumb) {
            (true, true) => thumb_branch32(0xf000_d000, target.wrapping_sub(pc) as i32),
            (true, false) => thumb_branch32(0xf000_c000, target.wrapping_sub(pc & !3) as i32) & !1,
            (false, false) => 0xeb00_0000 | (target.wrapping_sub(pc) >> 2 & 0x00ff_ffff),
            (false, true) => {
                let offset = target.wrapping_sub(pc);
                0xfa00_0000 | (offset >> 1 & 1) << 24 | (offset >> 2 & 0x00ff_ffff)
            }
        };
        self.emit(wide(raw));
    }

    pub fn finish(mut self) -> Program {
        for &(index, label, fix) in &self.fixups {
            let pc = self.address(index) + if self.thumb { 4 } else { 8 };
            let offset = self.labels[label].unwrap().wrapping_sub(pc) as i32;
            let raw = &mut self.insns[index].raw;
            match fix {
                Fix::Arm => *raw |= (offset >> 2) as u32 & 0x00ff_ffff,
                Fix::Thumb16 => {
                    assert!((-2048..2048).contains(&offset));
                    *raw |= (offset >> 1) as u32 & 0x7ff;
                }
                Fix::ThumbCond16 => {
                    assert!((-256..256).contains(&offset));
                    *raw |= (offset >> 1) as u32 & 0xff;
                }
                Fix::Thumb32 => *raw = thumb_branch32(*raw, offset),
                Fix::ThumbCond32 => {
                    let offset = offset as u32;
                    *raw |= (offset >> 20 & 1) << 26
                        | (offset >> 12 & 0x3f) << 16
                        | (offset >> 18 & 1) << 13
                        | (offset >> 19 & 1) << 11
                        | offset >> 1 & 0x7ff;
                }
                Fix::Cbz => {
                    assert!((0..128).contains(&offset));
                    let offset = offset as u32;
                    *raw |= (offset >> 6 & 1) << 9 | (offset >> 1 & 0x1f) << 3;
                }
            }
        }
        Program {
            thumb: self.thumb,
            at: self.at,
            insns: self.insns,
        }
    }
}

fn pick(random: &mut Random, from: &[u32]) -> u32 {
    from[random.below(from.len() as u32) as usize]
}

fn scratch(random: &mut Random) -> u32 {
    pick(random, &SCRATCH)
}

/// A register for an operand that the instruction only reads.
fn source(random: &mut Random) -> u32 {
    pick(random, &[0, 1, 2, 3, 12, BASE, WALK, COUNTER, ODD, SP, LR])
}

/// The A32 opcode and the T32 opcode of the operations with the two encodings.
const ALU: [(u32, u32); 8] = [
    (0, 0),
    (1, 4),
    (2, 13),
    (3, 14),
    (4, 8),
    (5, 10),
    (6, 11),
    (12, 2),
];

/// One instruction with no memory access and no branch. `cond` is for A32 only.
fn compute(b: &mut Builder, random: &mut Random, cond: u32) {
    let (rd, rn, rm) = (scratch(random), source(random), source(random));
    // The T32 encodings take no SP as an operand in most places.
    let (tn, tm) = (rn % 13, rm % 13);
    let s = random.below(2);
    let imm8 = random.next() & 0xff;
    match random.below(22) {
        0..=3 => {
            let (arm, thumb) = ALU[random.below(8) as usize];
            let rotation = random.below(16);
            // A T32 immediate: a byte, or a byte with its top bit at one of 24 places.
            let imm12 = if random.below(2) == 0 {
                imm8
            } else {
                (8 + random.below(24)) << 7 | imm8 & 0x7f
            };
            b.w(
                cond,
                0x0200_0000 | arm << 21 | s << 20 | rn << 16 | rd << 12 | rotation << 8 | imm8,
                0xf000_0000
                    | (imm12 >> 11) << 26
                    | thumb << 21
                    | s << 20
                    | tn << 16
                    | (imm12 >> 8 & 7) << 12
                    | rd << 8
                    | imm12 & 0xff,
            );
        }
        4..=7 => {
            let (arm, thumb) = ALU[random.below(8) as usize];
            let (kind, amount) = (random.below(4), random.below(32));
            b.w(
                cond,
                arm << 21 | s << 20 | rn << 16 | rd << 12 | amount << 7 | kind << 5 | tm,
                0xea00_0000
                    | thumb << 21
                    | s << 20
                    | tn << 16
                    | (amount >> 2) << 12
                    | rd << 8
                    | (amount & 3) << 6
                    | kind << 4
                    | tm,
            );
        }
        8 if b.thumb => {
            // The 16-bit forms: they set the flags outside an IT block.
            let (rd, rm) = (random.below(4), random.below(8));
            let raw = match random.below(6) {
                0 => 0x3000 | rd << 8 | imm8,
                1 => 0x3800 | rd << 8 | imm8,
                2 => 0x2000 | rd << 8 | imm8,
                3 => 0x1800 | random.below(8) << 6 | rm << 3 | rd,
                4 => 0x4000 | pick(random, &[0, 1, 5, 6, 12, 14, 15]) << 6 | rm << 3 | rd,
                _ => random.below(3) << 11 | random.below(32) << 6 | rm << 3 | rd,
            };
            b.emit(t16(raw));
        }
        8 => {
            // A shift by a register: the executor runs it.
            let kind = random.below(4);
            b.w(
                cond,
                random.below(16) << 21
                    | 1 << 20
                    | rn << 16
                    | rd << 12
                    | COUNTER << 8
                    | kind << 5
                    | 0x10
                    | tm,
                0,
            );
        }
        9 => {
            let imm16 = random.next() & 0xffff;
            let top = random.below(2);
            b.w(
                cond,
                0x0300_0000 | top << 22 | (imm16 >> 12) << 16 | rd << 12 | imm16 & 0xfff,
                0xf240_0000
                    | top << 23
                    | (imm16 >> 11 & 1) << 26
                    | (imm16 >> 12) << 16
                    | (imm16 >> 8 & 7) << 12
                    | rd << 8
                    | imm16 & 0xff,
            );
        }
        10 => {
            // CMP and CMN with an immediate, TST with a register.
            let kind = random.below(3);
            let (arm, thumb) = [(10, 13), (11, 8), (8, 0)][kind as usize];
            if kind == 2 {
                b.w(
                    cond,
                    arm << 21 | 1 << 20 | rn << 16 | tm,
                    0xea10_0f00 | thumb << 21 | tn << 16 | tm,
                );
            } else {
                b.w(
                    cond,
                    0x0200_0000 | arm << 21 | 1 << 20 | rn << 16 | imm8,
                    0xf010_0f00 | thumb << 21 | tn << 16 | imm8,
                );
            }
        }
        11 => b.w(
            cond,
            s << 20 | rd << 16 | tn << 8 | 0x90 | tm,
            0xfb00_f000 | tn << 16 | rd << 8 | tm,
        ),
        12 => {
            let ra = scratch(random);
            b.w(
                cond,
                0x0020_0090 | rd << 16 | ra << 12 | tn << 8 | tm,
                0xfb00_0000 | tn << 16 | ra << 12 | rd << 8 | tm,
            );
        }
        13 => {
            let (lo, hi) = if rd == 0 { (0, 1) } else { (rd, 0) };
            let kind = random.below(4);
            let arm = [0x0080_0090, 0x00c0_0090, 0x00a0_0090, 0x0040_0090][kind as usize];
            let thumb = [0xfba0_0000, 0xfb80_0000, 0xfbe0_0000, 0xfbe0_0060][kind as usize];
            b.w(
                cond,
                arm | hi << 16 | lo << 12 | tn << 8 | tm,
                thumb | tn << 16 | lo << 12 | hi << 8 | tm,
            );
        }
        14 => {
            let signed = random.below(2);
            b.w(
                cond,
                0x0710_f010 | (1 - signed) << 21 | rd << 16 | tm << 8 | tn,
                0xfb90_f0f0 | (1 - signed) << 21 | tn << 16 | rd << 8 | tm,
            );
        }
        15 => {
            let lsb = random.below(32);
            let width = 1 + random.below(32 - lsb);
            let signed = random.below(2);
            b.w(
                cond,
                0x07a0_0050 | (1 - signed) << 22 | (width - 1) << 16 | rd << 12 | lsb << 7 | tn,
                0xf340_0000
                    | (1 - signed) << 23
                    | tn << 16
                    | (lsb >> 2) << 12
                    | rd << 8
                    | (lsb & 3) << 6
                    | (width - 1),
            );
        }
        16 => {
            // BFI, or BFC with the register 15.
            let lsb = random.below(32);
            let msb = lsb + random.below(32 - lsb);
            let tn = if random.below(3) == 0 { 15 } else { tn };
            b.w(
                cond,
                0x07c0_0010 | msb << 16 | rd << 12 | lsb << 7 | tn,
                0xf360_0000 | tn << 16 | (lsb >> 2) << 12 | rd << 8 | (lsb & 3) << 6 | msb,
            );
        }
        17 => match random.below(3) {
            0 => b.w(
                cond,
                0x016f_0f10 | rd << 12 | tm,
                0xfab0_f080 | tm << 16 | rd << 8 | tm,
            ),
            1 => b.w(
                cond,
                0x06bf_0f30 | rd << 12 | tm,
                0xfa90_f080 | tm << 16 | rd << 8 | tm,
            ),
            _ => b.w(
                cond,
                0x06ff_0f30 | rd << 12 | tm,
                0xfa90_f0a0 | tm << 16 | rd << 8 | tm,
            ),
        },
        18 => {
            // SXTB, SXTH, UXTB and UXTH with a rotation.
            let kind = random.below(4);
            let rotation = random.below(4);
            let arm = [0x06af_0070, 0x06bf_0070, 0x06ef_0070, 0x06ff_0070][kind as usize];
            let thumb = [0xfa4f_f080, 0xfa0f_f080, 0xfa5f_f080, 0xfa1f_f080][kind as usize];
            b.w(
                cond,
                arm | rd << 12 | rotation << 10 | tm,
                thumb | rd << 8 | rotation << 4 | tm,
            );
        }
        19 => match random.below(3) {
            // The executor runs these: MRS, QADD and MSR of the flags.
            0 => b.w(cond, 0x010f_0000 | rd << 12, 0xf3ef_8000 | rd << 8),
            1 => b.w(
                cond,
                0x0100_0050 | tn << 16 | rd << 12 | tm,
                0xfa80_f080 | tn << 16 | rd << 8 | tm,
            ),
            _ => b.w(cond, 0x0128_f000 | tm, 0xf380_8800 | tm << 16),
        },
        _ => float(b, random, cond),
    }
}

/// One VFP instruction on the registers 0 to 3 of each size.
fn float(b: &mut Builder, random: &mut Random, cond: u32) {
    let (d, n, m) = (random.below(4), random.below(4), random.below(4));
    let double = random.below(2);
    let size = double << 8;
    // A single-precision register has its low bit in an other place.
    let reg = |number: u32, high: u32, low: u32| {
        if double == 1 {
            number << high
        } else {
            (number >> 1) << high | (number & 1) << low
        }
    };
    let (vd, vn, vm) = (reg(d, 12, 22), reg(n, 16, 7), reg(m, 0, 5));
    let both = |b: &mut Builder, raw: u32| b.w(cond, raw & 0x0fff_ffff, raw);
    match random.below(13) {
        0 => both(b, 0xee30_0a00 | size | vd | vn | vm),
        1 => both(b, 0xee30_0a40 | size | vd | vn | vm),
        2 => both(b, 0xee20_0a00 | size | vd | vn | vm),
        3 => both(b, 0xee80_0a00 | size | vd | vn | vm),
        4 => both(b, 0xee00_0a00 | random.below(2) << 6 | size | vd | vn | vm),
        5 => both(b, 0xeeb0_0a40 | random.below(2) << 7 | size | vd | vm),
        6 => both(b, 0xeeb4_0a40 | random.below(2) << 7 | size | vd | vm),
        12 => both(b, 0xeef1_fa10),
        7 => both(
            b,
            0xed10_0a00 | random.below(2) << 23 | size | vd | BASE << 16 | random.below(64),
        ),
        8 => both(
            b,
            0xed00_0a00 | random.below(2) << 23 | size | vd | BASE << 16 | random.below(64),
        ),
        9 => both(
            b,
            0xee00_0a10
                | random.below(2) << 20
                | (n >> 1) << 16
                | (n & 1) << 7
                | random.below(4) << 12,
        ),
        // Conversions between an integer in a single-precision register and a number.
        10 => both(
            b,
            0xeeb8_0a40 | random.below(2) << 7 | size | vd | (m >> 1) | (m & 1) << 5,
        ),
        _ => both(
            b,
            0xeebc_0ac0 | random.below(2) << 16 | size | (d >> 1) << 12 | (d & 1) << 22 | vm,
        ),
    }
}

/// One load or store of a core register, or of a few. `cond` is for A32 only.
fn memory(b: &mut Builder, random: &mut Random, cond: u32) {
    let load = random.below(2);
    let rt = if load == 1 {
        scratch(random)
    } else {
        source(random)
    };
    // The T32 encodings take no SP as the register of the value.
    let rt = rt % 13;
    let width = random.below(5);
    let imm = random.below(256);
    match random.below(14) {
        0..=2 => {
            // An offset from a pointer that stays: positive in 12 bits, or negative.
            let base = pick(random, &[BASE, BASE, ODD, CALL, SP]);
            let up = random.below(4) != 0;
            let offset = if up { random.below(0x400) } else { imm };
            single(b, cond, load, width, rt, base, offset, 1, u32::from(up), 0);
        }
        3 | 4 => {
            // With a writeback.
            let pre = random.below(2);
            let offset = random.below(9) * 4;
            // A store of the base with a writeback has no defined value.
            let rt = if rt == WALK { 0 } else { rt };
            single(
                b,
                cond,
                load,
                width,
                rt,
                WALK,
                offset,
                pre,
                random.below(2),
                1,
            );
        }
        5 | 6 => {
            // A register offset: the counter of the loop, with a shift in the forms
            // that have one.
            let shift = random.below(4);
            let base = pick(random, &[BASE, ODD, CALL]);
            let width = if load == 0 { width.min(2) } else { width };
            let thumb = [
                0xf800_0000u32,
                0xf820_0000,
                0xf840_0000,
                0xf900_0000,
                0xf920_0000,
            ][width as usize];
            let arm = match width {
                0 | 2 => 0x0780_0000 | u32::from(width == 0) << 22 | shift << 7,
                _ => 0x0180_0090 | [0, 0x20, 0, 0x40, 0x60][width as usize],
            };
            b.w(
                cond,
                arm | load << 20 | base << 16 | rt << 12 | COUNTER,
                thumb | load << 20 | base << 16 | rt << 12 | shift << 4 | COUNTER,
            );
        }
        7 if b.thumb => {
            // The 16-bit forms, with the base in r4 or r5 and the value in r0 to r3.
            let rt = random.below(4);
            let base = pick(random, &[BASE, WALK]);
            let raw = match random.below(4) {
                0 => 0x6000 | load << 11 | random.below(32) << 6 | base << 3 | rt,
                1 => 0x7000 | load << 11 | random.below(32) << 6 | base << 3 | rt,
                2 => 0x8000 | load << 11 | random.below(32) << 6 | base << 3 | rt,
                _ => 0x9000 | load << 11 | rt << 8 | random.below(64),
            };
            b.emit(t16(raw));
        }
        7 | 8 => {
            // PUSH and POP of results. A32 PUSH with one register is not a multiple.
            let mask = (1 + random.below(15)) | (random.below(2) << 12);
            let mask = if mask.count_ones() < 2 {
                mask | 0x1001
            } else {
                mask
            };
            if load == 1 {
                b.w(cond, 0x08bd_0000 | mask, 0xe8bd_0000 | mask);
            } else {
                b.w(
                    cond,
                    0x092d_0000 | mask | 0x4000,
                    0xe92d_0000 | mask | 0x4000,
                );
            }
        }
        9 | 10 => {
            // LDMIA and STMIA with the pointer that moves, and with the one that stays.
            let mask = (1 + random.below(15)) | (random.below(2) << 12) | 1;
            let mask = if mask.count_ones() < 2 {
                mask | 0x1000
            } else {
                mask
            };
            let (base, writeback) = if random.below(3) == 0 {
                (BASE, 0)
            } else {
                (WALK, 1)
            };
            b.w(
                cond,
                0x0880_0000 | writeback << 21 | load << 20 | base << 16 | mask,
                0xe880_0000 | writeback << 21 | load << 20 | base << 16 | mask,
            );
        }
        11 => {
            // LDRD and STRD with an offset: the executor runs the A32 ones.
            let rt = random.below(2) * 2;
            b.w(
                cond,
                0x01c0_00d0
                    | (1 - load) << 5
                    | BASE << 16
                    | rt << 12
                    | (imm & 0xf0) << 4
                    | imm & 0xc,
                0xe9c0_0000 | load << 20 | BASE << 16 | rt << 12 | (rt + 1) << 8 | imm,
            );
        }
        12 => {
            // The device: a read gives the count of instructions, a write moves the
            // next event.
            b.w(
                cond,
                0x0580_0000 | load << 20 | DEVICE << 16 | rt << 12,
                0xf8c0_0000 | load << 20 | DEVICE << 16 | rt << 12,
            );
        }
        _ => {
            // LDREX and STREX at the pointer that stays.
            if load == 1 {
                b.w(
                    cond,
                    0x0190_0f9f | BASE << 16 | rt << 12,
                    0xe850_0f00 | BASE << 16 | rt << 12,
                );
            } else {
                let rd = scratch(random);
                let rt = if rt == rd { (rd + 1) % 4 } else { rt };
                b.w(
                    cond,
                    0x0180_0f90 | BASE << 16 | rd << 12 | rt,
                    0xe840_0000 | BASE << 16 | rt << 12 | rd << 8,
                );
            }
        }
    }
}

/// LDR, STR and the forms of other sizes with an immediate offset. `width` is byte,
/// halfword, word, signed byte, signed halfword; a store has no signed form.
#[allow(clippy::too_many_arguments)]
fn single(
    b: &mut Builder,
    cond: u32,
    load: u32,
    width: u32,
    rt: u32,
    base: u32,
    offset: u32,
    p: u32,
    u: u32,
    w: u32,
) {
    let width = if load == 0 { width.min(2) } else { width };
    // A32 has no W bit for the writeback after the access.
    let puw = p << 24 | u << 23 | (p & w) << 21;
    let arm = match width {
        0 | 2 => {
            0x0400_0000
                | puw
                | u32::from(width == 0) << 22
                | load << 20
                | base << 16
                | rt << 12
                | offset
        }
        _ => {
            let offset = offset & 0xff;
            let kind = [0, 0xb0, 0, 0xd0, 0xf0][width as usize];
            0x0040_0000
                | puw
                | load << 20
                | base << 16
                | rt << 12
                | (offset >> 4) << 8
                | kind
                | offset & 0xf
        }
    };
    let kind = [
        0xf800_0000u32,
        0xf820_0000,
        0xf840_0000,
        0xf900_0000,
        0xf920_0000,
    ][width as usize];
    let thumb = if p == 1 && u == 1 && w == 0 {
        kind | 0x0080_0000 | load << 20 | base << 16 | rt << 12 | offset & 0xfff
    } else {
        kind | load << 20
            | base << 16
            | rt << 12
            | 0x800
            | p << 10
            | u << 9
            | w << 8
            | offset & 0xff
    };
    b.w(cond, arm, thumb);
}

fn condition(random: &mut Random) -> u32 {
    random.below(14)
}

/// An A32 condition for one instruction: most have none.
fn arm_condition(b: &Builder, random: &mut Random) -> u32 {
    if b.thumb || random.below(3) != 0 {
        AL
    } else {
        condition(random)
    }
}

fn one(b: &mut Builder, random: &mut Random) {
    let cond = arm_condition(b, random);
    if random.below(5) < 2 {
        memory(b, random, cond);
    } else {
        compute(b, random, cond);
    }
}

/// What a program can have, from the simple parts to the ones with other code.
pub struct Options {
    /// The other code that a call can go to: the address and its instruction set.
    pub calls: Vec<(u32, bool)>,
    /// Stores that change an instruction of the program.
    pub patches: bool,
    /// Instructions for a machine with vectors: SVC, UDF, an access with an abort,
    /// changes of the mode and of the interrupt mask.
    pub system: bool,
}

/// One instruction that gives an exception or changes the mode. The handlers of
/// `install_vectors` come back after it.
fn system(b: &mut Builder, random: &mut Random) {
    match random.below(8) {
        0 | 1 => {
            let number = random.below(256);
            if b.thumb {
                b.emit(t16(0xdf00 | number));
            } else {
                b.emit(wide(0xef00_0000 | number));
            }
        }
        2 if b.thumb => b.emit(t16(0xde00 | random.below(256))),
        2 => b.emit(wide(0xe7f0_00f0)),
        3 => {
            // An access with a writeback before it, from the pointer near the end of
            // its page: an offset of 128 or more is in the page with no memory, and
            // the abort keeps the pointer.
            let load = random.below(2);
            let rt = scratch(random);
            let offset = 0x80 + random.below(0x80);
            single(b, AL, load, random.below(5), rt, CALL, offset, 1, 1, 1);
        }
        4 | 5 => {
            // CPS to an other mode: each mode has its stack, and FIQ its r8 to r12.
            let mode = pick(random, &[0x11, 0x12, 0x13, 0x17, 0x1b, 0x1f]);
            if b.thumb {
                b.emit(wide(0xf3af_8100 | mode));
            } else {
                b.emit(wide(0xf102_0000 | mode));
            }
        }
        6 => {
            // CPSID i and CPSIE i.
            let disable = random.below(2);
            if b.thumb {
                b.emit(t16(0xb662 | disable << 4));
            } else {
                b.emit(wide(0xf108_0080 | disable << 18));
            }
        }
        _ => {
            // The SPSR of the mode to a result, or a result to its flags.
            let rd = scratch(random);
            if random.below(2) == 0 {
                b.w(AL, 0x014f_0000 | rd << 12, 0xf3ff_8000 | rd << 8);
            } else {
                b.w(AL, 0x0168_f000 | rd, 0xf390_8800 | rd << 16);
            }
        }
    }
}

/// The address of the vectors of `install_vectors`.
pub const VECTORS: u32 = RAM_BASE;
/// What the handlers write: counts and return addresses.
pub const LOG: u32 = RAM_BASE + 0x1000;
const HANDLER_STACKS: u32 = RAM_BASE + 0x2000;

/// Puts A32 vectors and handlers in the first page. Each handler keeps the registers
/// of the program, writes what it saw to the log and goes back to the program: after
/// an SVC or an undefined instruction, after the instruction of a data abort (a 32-bit
/// one), and to the instruction that an interrupt request came before. A prefetch
/// abort stops the machine. `log` is the address of the log.
pub fn install_vectors(case: &mut Case, log: u32) {
    let word =
        |at: u32, target: u32| 0xea00_0000 | (target.wrapping_sub(at + 8) >> 2 & 0x00ff_ffff);
    let (undefined, svc, prefetch, data, irq) = (0x40, 0x48, 0x80, 0xa0, 0xe0);
    let stop = [0xe320_f003, 0xeaff_fffe];
    let ram = &mut case.ram;
    for (vector, target) in [
        (0x00, prefetch),
        (0x04, undefined),
        (0x08, svc),
        (0x0c, prefetch),
        (0x10, data),
        (0x14, prefetch),
        (0x18, irq),
        (0x1c, prefetch),
    ] {
        ram.write_words(
            VECTORS + vector,
            &[word(VECTORS + vector, VECTORS + target)],
        );
    }
    // The literal with the address of the log is at 0x200.
    let literal = |at: u32| 0xe59f_0000 | (0x200 - (at + 8));
    // MOVS pc, lr.
    ram.write_words(VECTORS + undefined, &[0xe1b0_f00e]);
    ram.write_words(
        VECTORS + svc,
        &[
            0xe92d_0003,
            literal(svc + 4),
            0xe590_1000,
            0xe281_1001,
            0xe580_1000,
            0xe580_e004,
            0xe8bd_0003,
            0xe1b0_f00e,
        ],
    );
    ram.write_words(VECTORS + prefetch, &stop);
    // The return is SUBS pc, lr, #4.
    ram.write_words(
        VECTORS + data,
        &[
            0xe92d_0003,
            literal(data + 4),
            0xe590_1008,
            0xe281_1001,
            0xe580_1008,
            0xe580_e00c,
            0xe8bd_0003,
            0xe25e_f004,
        ],
    );
    // The log keeps the return address and the SPSR of the last 32 requests. The
    // return is LDM with the PC and the SPSR.
    ram.write_words(
        VECTORS + irq,
        &[
            0xe24e_e004,
            0xe92d_4003,
            literal(irq + 8),
            0xe590_1010,
            0xe281_1001,
            0xe580_1010,
            0xe201_101f,
            0xe080_1181,
            0xe581_e020,
            0xe14f_e000,
            0xe581_e024,
            0xe8fd_8003,
        ],
    );
    ram.write_words(VECTORS + 0x200, &[log]);
    let cpu = &mut case.cpu;
    cpu.cp15.vbar = VECTORS;
    let stack = |number: u32| HANDLER_STACKS + 0x400 * number;
    // Each bank has the pointers that the registers of a program need.
    let regs = cpu.regs;
    cpu.regs_usr = [stack(1), regs[14]];
    cpu.regs_irq = [stack(2), regs[14]];
    cpu.regs_svc = [stack(3), regs[14]];
    cpu.regs_abt = [stack(4), regs[14]];
    cpu.regs_und = [stack(5), regs[14]];
    cpu.regs_fiq = [
        regs[8].wrapping_add(0x40),
        regs[9],
        regs[10],
        regs[11].wrapping_add(0x41),
        regs[12] ^ 0x5a5a,
        stack(6),
        regs[14],
    ];
    cpu.regs_usr_high = [regs[8], regs[9], regs[10], regs[11], regs[12]];
}

/// A part of a program with a branch over instructions, an IT block, a literal, a
/// table branch, a call or a store to the code.
fn part(b: &mut Builder, random: &mut Random, options: &Options) {
    match random.below(16) {
        0 | 1 => {
            let over = b.label();
            b.b(condition(random), over, random.below(2) == 0);
            for _ in 0..1 + random.below(3) {
                one(b, random);
            }
            b.bind(over);
        }
        2 | 3 if b.thumb => {
            // An IT block: each instruction in it is one that sets no flags there.
            let count = 1 + random.below(4);
            let cond = condition(random);
            let mut mask = 1 << (4 - count);
            for index in 1..count {
                let then = random.below(2);
                mask |= (cond & 1 ^ (1 - then)) << (4 - index);
            }
            b.emit(t16(0xbf00 | cond << 4 | mask));
            for _ in 0..count {
                one(b, random);
            }
        }
        4 if b.thumb => {
            // CBZ or CBNZ of the counter or of a result, over instructions.
            let over = b.label();
            let rn = pick(random, &[COUNTER, 0, 1, 2]);
            b.branch(0xb100 | random.below(2) << 11 | rn, 2, over, Fix::Cbz);
            for _ in 0..1 + random.below(3) {
                one(b, random);
            }
            b.bind(over);
        }
        5 => {
            // A load of a literal after a branch over it, or the address of it.
            let rt = scratch(random);
            let over = b.label();
            if b.thumb {
                let literal = (b.here() + 6 + 3) & !3;
                let offset = literal - ((b.here() + 4) & !3);
                if random.below(3) == 0 {
                    b.emit(wide(0xf20f_0000 | rt << 8 | offset));
                } else {
                    b.emit(wide(0xf8df_0000 | rt << 12 | offset));
                }
                b.b(AL, over, true);
                if b.here() & 2 != 0 {
                    b.emit(t16(0xbf00));
                }
                b.emit(t16(random.next() & 0xffff));
                b.emit(t16(random.next() & 0xffff));
            } else {
                if random.below(3) == 0 {
                    b.emit(wide(0xe28f_0000 | rt << 12));
                } else {
                    b.emit(wide(0xe59f_0000 | rt << 12));
                }
                b.b(AL, over, true);
                b.emit(wide(random.next() << 16 | random.next() & 0xffff));
            }
            b.bind(over);
        }
        6 if b.thumb => {
            // TBB with a table of 4 entries after it, or TBH with 2.
            let halfword = random.below(2);
            let entries = if halfword == 1 { 2 } else { 4 };
            b.emit(wide(0xf000_0000 | (entries - 1)));
            let table_branch = b.insns.len();
            b.emit(wide(0xe8df_f000 | halfword << 4));
            let table = b.here();
            b.emit(t16(0));
            b.emit(t16(0));
            let mut offsets = Vec::new();
            for _ in 0..entries {
                offsets.push((b.here() - table) / 2);
                one(b, random);
            }
            let raw = if halfword == 1 {
                [offsets[0], offsets[1]]
            } else {
                [offsets[0] | offsets[1] << 8, offsets[2] | offsets[3] << 8]
            };
            b.insns[table_branch + 1].raw = raw[0];
            b.insns[table_branch + 2].raw = raw[1];
        }
        7 | 8 if !options.calls.is_empty() => {
            let (target, thumb) = options.calls[random.below(options.calls.len() as u32) as usize];
            if random.below(3) == 0 {
                // BLX of a register with the address.
                let value = target | u32::from(thumb);
                for (top, half) in [(0, value & 0xffff), (1, value >> 16)] {
                    b.w(
                        AL,
                        0x0300_0000 | top << 22 | (half >> 12) << 16 | LR << 12 | half & 0xfff,
                        0xf240_0000
                            | top << 23
                            | (half >> 11 & 1) << 26
                            | (half >> 12) << 16
                            | (half >> 8 & 7) << 12
                            | LR << 8
                            | half & 0xff,
                    );
                }
                if b.thumb {
                    b.emit(t16(0x4780 | LR << 3));
                } else {
                    b.emit(wide(0xe12f_ff30 | LR));
                }
            } else {
                b.call(target, thumb);
            }
        }
        9 if options.patches => {
            // A store of the counter, or of the other instruction, to the instruction
            // that changes. The instruction can be the next one.
            let byte = random.below(2);
            let rt = if byte == 1 { COUNTER } else { PATCH };
            if b.thumb {
                b.emit(wide(0xf8a0_0000 & !(byte << 21) | SITE << 16 | rt << 12));
            } else {
                b.emit(wide(0xe580_0000 | byte << 22 | SITE << 16 | rt << 12));
            }
            if b.site.is_none() {
                site(b, random);
            }
        }
        10 if options.patches && b.site.is_none() => site(b, random),
        11 if options.system => system(b, random),
        _ => one(b, random),
    }
}

/// The instruction that the program changes: an ADD of a byte to r0.
fn site(b: &mut Builder, random: &mut Random) {
    for _ in 0..random.below(3) {
        one(b, random);
    }
    b.site = Some(b.here());
    if b.thumb {
        b.emit(t16(0x3000 | random.below(256)));
    } else {
        b.emit(wide(0xe280_0000 | random.below(256)));
    }
}

/// Code for a call: a few instructions and a return.
pub fn subroutine(random: &mut Random, thumb: bool, at: u32) -> Program {
    let mut b = Builder::new(thumb, at);
    let options = Options {
        calls: Vec::new(),
        patches: false,
        system: false,
    };
    let saves = random.below(3) == 0;
    if saves {
        b.w(AL, 0x092d_4001, 0xe92d_4001);
    }
    for _ in 0..random.below(5) {
        part(&mut b, random, &options);
    }
    match (saves, thumb) {
        (true, _) => b.w(AL, 0x08bd_8001, 0xe8bd_8001),
        (false, true) => b.emit(t16(0x4700 | LR << 3)),
        // BX, or a MOV to the PC that the executor runs.
        (false, false) if random.below(2) == 0 => b.emit(wide(0xe12f_ff1e)),
        (false, false) => b.emit(wide(0xe1a0_f00e)),
    }
    b.finish()
}

/// A program of straight parts and loops, with a stop at its end.
pub fn main_program(
    random: &mut Random,
    thumb: bool,
    at: u32,
    options: &Options,
) -> (Program, Option<u32>) {
    let mut b = Builder::new(thumb, at);
    for _ in 0..1 + random.below(4) {
        for _ in 0..random.below(5) {
            part(&mut b, random, options);
        }
        // A loop: the counter goes down to 0.
        let count = 1 + random.below(14);
        b.w(
            AL,
            0x03a0_0000 | COUNTER << 12 | count,
            0xf04f_0000 | COUNTER << 8 | count,
        );
        let top = b.label();
        b.bind(top);
        for _ in 0..1 + random.below(7) {
            part(&mut b, random, options);
        }
        if thumb && random.below(2) == 0 {
            b.emit(t16(0x3801 | COUNTER << 8));
        } else {
            b.w(
                AL,
                0x0250_0001 | COUNTER << 16 | COUNTER << 12,
                0xf1b0_0001 | COUNTER << 16 | COUNTER << 8,
            );
        }
        let short = thumb && b.here() + 4 - b.labels[top.0].unwrap() < 250;
        b.b(1, top, short);
    }
    for _ in 0..random.below(4) {
        part(&mut b, random, options);
    }
    if thumb {
        b.emit(t16(u32::from(THUMB_STOP)));
    } else {
        b.emit(wide(ARM_STOP));
    }
    let site = b.site;
    (b.finish(), site)
}

/// The registers of a program at its start.
pub fn enter(case: &mut Case, random: &mut Random, program: &Program, site: Option<u32>) {
    let cpu = &mut case.cpu;
    for register in SCRATCH {
        cpu.regs[register as usize] = if random.below(2) == 0 {
            random.next()
        } else {
            random.below(256)
        };
    }
    cpu.regs[COUNTER as usize] = random.below(16);
    cpu.regs[BASE as usize] = DATA + 0x800 + random.below(0x200) * 4;
    cpu.regs[WALK as usize] = DATA + 0x2000 + random.below(0x100) * 4;
    cpu.regs[DEVICE as usize] = DEVICE_PAGE + random.below(16) * 4;
    cpu.regs[CALL as usize] = CALL_PAGE + 0x400 + random.below(0x100) * 4;
    cpu.regs[ODD as usize] = DATA + 0x1000 + random.below(0x1000);
    cpu.regs[SP as usize] = DATA + 0x5000 + random.below(0x100) * 4;
    cpu.regs[LR as usize] = program.end() - if program.thumb { 1 } else { 4 };
    if let Some(site) = site {
        cpu.regs[SITE as usize] = site;
        // An other ADD to r0, or an instruction with no effect on the registers of
        // the program.
        cpu.regs[PATCH as usize] = match (program.thumb, random.below(3)) {
            (true, 0) => 0xbf00,
            (true, _) => 0x3000 | random.below(256),
            (false, 0) => 0xe1a0_0000,
            (false, _) => 0xe280_0000 | random.below(256),
        };
    }
    case.enter(program);
}

/// The second code page: the code for the calls is there.
pub const CALLED: u32 = CODE + 0x1000;
