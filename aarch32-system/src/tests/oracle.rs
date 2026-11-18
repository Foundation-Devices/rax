//! Integer instructions against a model that follows the pseudocode of the ARM
//! architecture manual. The three engines run one instruction with random operands,
//! and the registers and the flags after it are those of the model.

use super::Random;
use super::suite::{CODE, Case, Program, Trio, cases, random_cpu, wide};

#[derive(Clone, Debug, PartialEq)]
struct State {
    r: [u32; 16],
    n: bool,
    z: bool,
    c: bool,
    v: bool,
    q: bool,
    ge: u32,
}

type Apply = Box<dyn Fn(&mut State)>;
/// The A32 encoding with no condition, the T32 encoding, and what the instruction does.
type Instruction = (Option<u32>, Option<u32>, Apply);

fn distinct<const N: usize>(random: &mut Random) -> [u32; N] {
    let mut registers = [0; N];
    for index in 0..N {
        registers[index] = loop {
            let register = random.below(13);
            if !registers[..index].contains(&register) {
                break register;
            }
        };
    }
    registers
}

/// Shift_C with a type of 0 to 3 (LSL, LSR, ASR, ROR) or 4 (RRX).
fn shift_c(value: u32, kind: u32, amount: u32, carry: bool) -> (u32, bool) {
    if amount == 0 {
        return (value, carry);
    }
    let wide = u64::from(value);
    match kind {
        0 if amount > 32 => (0, false),
        0 => ((wide << amount) as u32, wide << amount >> 32 & 1 != 0),
        1 if amount > 32 => (0, false),
        1 => ((wide >> amount) as u32, wide >> (amount - 1) & 1 != 0),
        2 => {
            let amount = amount.min(32);
            let signed = i64::from(value as i32);
            ((signed >> amount) as u32, signed >> (amount - 1) & 1 != 0)
        }
        3 => {
            let result = value.rotate_right(amount % 32);
            (result, result >> 31 != 0)
        }
        _ => (value >> 1 | u32::from(carry) << 31, value & 1 != 0),
    }
}

/// DecodeImmShift: the type and the amount of a shift by an immediate.
fn immediate_shift(kind: u32, imm5: u32) -> (u32, u32) {
    match (kind, imm5) {
        (1 | 2, 0) => (kind, 32),
        (3, 0) => (4, 1),
        _ => (kind, imm5),
    }
}

fn add_with_carry(x: u32, y: u32, carry: bool) -> (u32, bool, bool) {
    let unsigned = u64::from(x) + u64::from(y) + u64::from(carry);
    let signed = i64::from(x as i32) + i64::from(y as i32) + i64::from(carry);
    let result = unsigned as u32;
    (
        result,
        unsigned >> 32 != 0,
        i64::from(result as i32) != signed,
    )
}

/// The data-processing operation with the A32 opcode, on `a` and the shifter operand
/// `b` with its carry. Returns the result, when the instruction writes one.
fn data_processing(
    state: &mut State,
    opcode: u32,
    set: bool,
    a: u32,
    b: u32,
    carry: bool,
) -> Option<u32> {
    let logical = |result: u32| (result, carry, state.v);
    let (result, c, v) = match opcode {
        0 | 8 => logical(a & b),
        1 | 9 => logical(a ^ b),
        2 | 10 => add_with_carry(a, !b, true),
        3 => add_with_carry(!a, b, true),
        4 | 11 => add_with_carry(a, b, false),
        5 => add_with_carry(a, b, state.c),
        6 => add_with_carry(a, !b, state.c),
        7 => add_with_carry(!a, b, state.c),
        12 => logical(a | b),
        13 => logical(b),
        14 => logical(a & !b),
        15 => logical(!b),
        // ORN, which only T32 has.
        _ => logical(a | !b),
    };
    if set {
        (state.n, state.z, state.c, state.v) = (result >> 31 != 0, result == 0, c, v);
    }
    (!(8..12).contains(&opcode)).then_some(result)
}

/// The T32 opcode of an A32 data-processing opcode, with Rn and Rd of the T32 forms
/// that are compares and moves. None for RSC.
fn thumb_data_processing(opcode: u32, rd: u32, rn: u32) -> Option<(u32, u32, u32)> {
    Some(match opcode {
        0 => (0, rd, rn),
        1 => (4, rd, rn),
        2 => (13, rd, rn),
        3 => (14, rd, rn),
        4 => (8, rd, rn),
        5 => (10, rd, rn),
        6 => (11, rd, rn),
        7 => return None,
        8 => (0, 15, rn),
        9 => (4, 15, rn),
        10 => (13, 15, rn),
        11 => (8, 15, rn),
        12 => (2, rd, rn),
        13 => (2, rd, 15),
        14 => (1, rd, rn),
        15 => (3, rd, 15),
        _ => (3, rd, rn),
    })
}

fn is_compare(opcode: u32) -> bool {
    (8..12).contains(&opcode)
}

fn alu_immediate(random: &mut Random) -> Instruction {
    let [rd, rn] = distinct(random);
    let opcode = random.below(17);
    let set = is_compare(opcode) || random.below(2) == 0;
    let s = u32::from(set);
    if opcode < 16 && random.below(2) == 0 {
        // A32: a byte with a rotation.
        let (rotation, imm8) = (random.below(16), random.next() & 0xff);
        let raw = 0x0200_0000 | opcode << 21 | s << 20 | rn << 16 | rd << 12 | rotation << 8 | imm8;
        let value = imm8.rotate_right(rotation * 2);
        return (
            Some(raw),
            None,
            Box::new(move |state| {
                let carry = if rotation == 0 {
                    state.c
                } else {
                    value >> 31 != 0
                };
                if let Some(result) = {
                    let first = state.r[rn as usize];
                    data_processing(state, opcode, set, first, value, carry)
                } {
                    state.r[rd as usize] = result;
                }
            }),
        );
    }
    // T32: ThumbExpandImm_C. A byte, a byte at two or four places, or a byte with its
    // top bit set and a rotation.
    let opcode = if opcode == 7 { 16 } else { opcode };
    let imm8 = random.next() & 0xff;
    let imm12 = match random.below(4) {
        0 => imm8,
        1 => (1 + random.below(3)) << 8 | imm8.max(1),
        _ => (8 + random.below(24)) << 7 | imm8 & 0x7f,
    };
    let (value, rotated) = match imm12 >> 8 {
        0 => (imm12, false),
        1 => ((imm12 & 0xff) * 0x0001_0001, false),
        2 => ((imm12 & 0xff) * 0x0100_0100, false),
        3 => ((imm12 & 0xff) * 0x0101_0101, false),
        _ => ((0x80 | imm12 & 0x7f).rotate_right(imm12 >> 7), true),
    };
    let (thumb, td, tn) = thumb_data_processing(opcode, rd, rn).unwrap();
    let raw = 0xf000_0000
        | (imm12 >> 11) << 26
        | thumb << 21
        | s << 20
        | tn << 16
        | (imm12 >> 8 & 7) << 12
        | td << 8
        | imm12 & 0xff;
    (
        None,
        Some(raw),
        Box::new(move |state| {
            let carry = if rotated { value >> 31 != 0 } else { state.c };
            if let Some(result) = {
                let first = state.r[rn as usize];
                data_processing(state, opcode, set, first, value, carry)
            } {
                state.r[rd as usize] = result;
            }
        }),
    )
}

fn alu_shifted_register(random: &mut Random) -> Instruction {
    let [rd, rn, rm] = distinct(random);
    let rm = if random.below(4) == 0 { rn } else { rm };
    let opcode = random.below(17);
    let set = is_compare(opcode) || random.below(2) == 0;
    let s = u32::from(set);
    let (kind, imm5) = (
        random.below(4),
        [0, 1, 31, random.below(32)][random.below(4) as usize],
    );
    let arm = (opcode < 16)
        .then(|| opcode << 21 | s << 20 | rn << 16 | rd << 12 | imm5 << 7 | kind << 5 | rm);
    let thumb = thumb_data_processing(opcode, rd, rn).map(|(thumb, td, tn)| {
        0xea00_0000
            | thumb << 21
            | s << 20
            | tn << 16
            | (imm5 >> 2) << 12
            | td << 8
            | (imm5 & 3) << 6
            | kind << 4
            | rm
    });
    (
        arm,
        thumb,
        Box::new(move |state| {
            let (kind, amount) = immediate_shift(kind, imm5);
            let (value, carry) = shift_c(state.r[rm as usize], kind, amount, state.c);
            if let Some(result) = {
                let first = state.r[rn as usize];
                data_processing(state, opcode, set, first, value, carry)
            } {
                state.r[rd as usize] = result;
            }
        }),
    )
}

/// A32 data processing with a shift by a register, and the T32 shifts by a register.
fn alu_register_shift(random: &mut Random) -> Instruction {
    let [rd, rn, rm, rs] = distinct(random);
    let kind = random.below(4);
    let s = random.below(2);
    if random.below(3) == 0 {
        let raw = 0xfa00_f000 | kind << 21 | s << 20 | rn << 16 | rd << 8 | rm;
        return (
            None,
            Some(raw),
            Box::new(move |state| {
                let amount = state.r[rm as usize] & 0xff;
                let (value, carry) = shift_c(state.r[rn as usize], kind, amount, state.c);
                state.r[rd as usize] = data_processing(state, 13, s == 1, 0, value, carry).unwrap();
            }),
        );
    }
    let opcode = random.below(16);
    let set = is_compare(opcode) || s == 1;
    let raw =
        opcode << 21 | u32::from(set) << 20 | rn << 16 | rd << 12 | rs << 8 | kind << 5 | 0x10 | rm;
    (
        Some(raw),
        None,
        Box::new(move |state| {
            let amount = state.r[rs as usize] & 0xff;
            let (value, carry) = shift_c(state.r[rm as usize], kind, amount, state.c);
            if let Some(result) = {
                let first = state.r[rn as usize];
                data_processing(state, opcode, set, first, value, carry)
            } {
                state.r[rd as usize] = result;
            }
        }),
    )
}

fn move_wide(random: &mut Random) -> Instruction {
    let rd = random.below(13);
    let imm16 = random.next() & 0xffff;
    let top = random.below(2);
    (
        Some(0x0300_0000 | top << 22 | (imm16 >> 12) << 16 | rd << 12 | imm16 & 0xfff),
        Some(
            0xf240_0000
                | top << 23
                | (imm16 >> 11 & 1) << 26
                | (imm16 >> 12) << 16
                | (imm16 >> 8 & 7) << 12
                | rd << 8
                | imm16 & 0xff,
        ),
        Box::new(move |state| {
            let old = state.r[rd as usize];
            state.r[rd as usize] = if top == 1 {
                old & 0xffff | imm16 << 16
            } else {
                imm16
            };
        }),
    )
}

fn set_nz64(state: &mut State, result: u64) {
    (state.n, state.z) = (result >> 63 != 0, result == 0);
}

fn multiply(random: &mut Random) -> Instruction {
    let [rd, rn, rm, ra] = distinct(random);
    let (lo, hi) = (rd, ra);
    let s = random.below(2);
    let r = |state: &State, register: u32| state.r[register as usize];
    let pair = move |state: &mut State, value: u64| {
        state.r[lo as usize] = value as u32;
        state.r[hi as usize] = (value >> 32) as u32;
    };
    let long_arm = |base: u32| Some(base | s << 20 | hi << 16 | lo << 12 | rm << 8 | 0x90 | rn);
    let long_thumb = |base: u32| (s == 0).then_some(base | rn << 16 | lo << 12 | hi << 8 | rm);
    let accumulator = move |state: &State| u64::from(r(state, hi)) << 32 | u64::from(r(state, lo));
    match random.below(8) {
        0 => (
            Some(s << 20 | rd << 16 | rm << 8 | 0x90 | rn),
            (s == 0).then_some(0xfb00_f000 | rn << 16 | rd << 8 | rm),
            Box::new(move |state| {
                let result = r(state, rn).wrapping_mul(r(state, rm));
                state.r[rd as usize] = result;
                if s == 1 {
                    (state.n, state.z) = (result >> 31 != 0, result == 0);
                }
            }),
        ),
        1 => (
            Some(0x0020_0000 | s << 20 | rd << 16 | ra << 12 | rm << 8 | 0x90 | rn),
            (s == 0).then_some(0xfb00_0000 | rn << 16 | ra << 12 | rd << 8 | rm),
            Box::new(move |state| {
                let result = r(state, rn)
                    .wrapping_mul(r(state, rm))
                    .wrapping_add(r(state, ra));
                state.r[rd as usize] = result;
                if s == 1 {
                    (state.n, state.z) = (result >> 31 != 0, result == 0);
                }
            }),
        ),
        2 => (
            Some(0x0060_0000 | rd << 16 | ra << 12 | rm << 8 | 0x90 | rn),
            Some(0xfb00_0010 | rn << 16 | ra << 12 | rd << 8 | rm),
            Box::new(move |state| {
                state.r[rd as usize] =
                    r(state, ra).wrapping_sub(r(state, rn).wrapping_mul(r(state, rm)));
            }),
        ),
        3 => (
            long_arm(0x0080_0000),
            long_thumb(0xfba0_0000),
            Box::new(move |state| {
                let result = u64::from(r(state, rn)) * u64::from(r(state, rm));
                pair(state, result);
                if s == 1 {
                    set_nz64(state, result);
                }
            }),
        ),
        4 => (
            long_arm(0x00a0_0000),
            long_thumb(0xfbe0_0000),
            Box::new(move |state| {
                let product = u64::from(r(state, rn)) * u64::from(r(state, rm));
                let result = product.wrapping_add(accumulator(state));
                pair(state, result);
                if s == 1 {
                    set_nz64(state, result);
                }
            }),
        ),
        5 => (
            long_arm(0x00c0_0000),
            long_thumb(0xfb80_0000),
            Box::new(move |state| {
                let result =
                    (i64::from(r(state, rn) as i32) * i64::from(r(state, rm) as i32)) as u64;
                pair(state, result);
                if s == 1 {
                    set_nz64(state, result);
                }
            }),
        ),
        6 => (
            long_arm(0x00e0_0000),
            long_thumb(0xfbc0_0000),
            Box::new(move |state| {
                let product =
                    (i64::from(r(state, rn) as i32) * i64::from(r(state, rm) as i32)) as u64;
                let result = product.wrapping_add(accumulator(state));
                pair(state, result);
                if s == 1 {
                    set_nz64(state, result);
                }
            }),
        ),
        _ => (
            Some(0x0040_0000 | hi << 16 | lo << 12 | rm << 8 | 0x90 | rn),
            Some(0xfbe0_0060 | rn << 16 | lo << 12 | hi << 8 | rm),
            Box::new(move |state| {
                let result = u64::from(r(state, rn)) * u64::from(r(state, rm))
                    + u64::from(r(state, hi))
                    + u64::from(r(state, lo));
                pair(state, result);
            }),
        ),
    }
}

fn half(value: u32, high: u32) -> i64 {
    i64::from((if high == 1 { value >> 16 } else { value }) as i16)
}

/// Writes the low 32 bits of `value`, and sets Q when the value is not in 32 bits.
fn write_with_q(state: &mut State, rd: u32, value: i128) {
    state.r[rd as usize] = value as u32;
    if value != i128::from(value as i32) {
        state.q = true;
    }
}

fn halfword_multiply(random: &mut Random) -> Instruction {
    let [rd, rn, rm, ra] = distinct(random);
    let (lo, hi) = (rd, ra);
    let (x, y) = (random.below(2), random.below(2));
    let r = |state: &State, register: u32| state.r[register as usize];
    let signed = |state: &State, register: u32| i128::from(state.r[register as usize] as i32);
    let accumulator =
        move |state: &State| (u64::from(r(state, hi)) << 32 | u64::from(r(state, lo))) as i64;
    let pair = move |state: &mut State, value: i128| {
        state.r[lo as usize] = value as u32;
        state.r[hi as usize] = (value >> 32) as u32;
    };
    // The two products of the halves, with the halves of Rm in the other order for `x`.
    let products = move |state: &State| {
        let m = r(state, rm).rotate_right(16 * x);
        let n = r(state, rn);
        (
            i128::from(half(n, 0) * half(m, 0)),
            i128::from(half(n, 1) * half(m, 1)),
        )
    };
    match random.below(14) {
        0 => (
            Some(0x0100_0080 | rd << 16 | ra << 12 | rm << 8 | y << 6 | x << 5 | rn),
            Some(0xfb10_0000 | rn << 16 | ra << 12 | rd << 8 | x << 5 | y << 4 | rm),
            Box::new(move |state| {
                let product = half(r(state, rn), x) * half(r(state, rm), y);
                write_with_q(state, rd, i128::from(product) + signed(state, ra));
            }),
        ),
        1 => (
            Some(0x0160_0080 | rd << 16 | rm << 8 | y << 6 | x << 5 | rn),
            Some(0xfb10_f000 | rn << 16 | rd << 8 | x << 5 | y << 4 | rm),
            Box::new(move |state| {
                state.r[rd as usize] = (half(r(state, rn), x) * half(r(state, rm), y)) as u32;
            }),
        ),
        2 => (
            Some(0x0120_0080 | rd << 16 | ra << 12 | rm << 8 | y << 6 | rn),
            Some(0xfb30_0000 | rn << 16 | ra << 12 | rd << 8 | y << 4 | rm),
            Box::new(move |state| {
                let product = signed(state, rn) * i128::from(half(r(state, rm), y));
                write_with_q(state, rd, (product + (signed(state, ra) << 16)) >> 16);
            }),
        ),
        3 => (
            Some(0x0120_00a0 | rd << 16 | rm << 8 | y << 6 | rn),
            Some(0xfb30_f000 | rn << 16 | rd << 8 | y << 4 | rm),
            Box::new(move |state| {
                let product = signed(state, rn) * i128::from(half(r(state, rm), y));
                state.r[rd as usize] = (product >> 16) as u32;
            }),
        ),
        4 => (
            Some(0x0140_0080 | hi << 16 | lo << 12 | rm << 8 | y << 6 | x << 5 | rn),
            Some(0xfbc0_0080 | rn << 16 | lo << 12 | hi << 8 | x << 5 | y << 4 | rm),
            Box::new(move |state| {
                let product = half(r(state, rn), x) * half(r(state, rm), y);
                pair(state, i128::from(accumulator(state).wrapping_add(product)));
            }),
        ),
        5 => (
            Some(0x0700_0010 | rd << 16 | ra << 12 | rm << 8 | x << 5 | rn),
            Some(0xfb20_0000 | rn << 16 | ra << 12 | rd << 8 | x << 4 | rm),
            Box::new(move |state| {
                let (low, high) = products(state);
                write_with_q(state, rd, low + high + signed(state, ra));
            }),
        ),
        6 => (
            Some(0x0700_f010 | rd << 16 | rm << 8 | x << 5 | rn),
            Some(0xfb20_f000 | rn << 16 | rd << 8 | x << 4 | rm),
            Box::new(move |state| {
                let (low, high) = products(state);
                write_with_q(state, rd, low + high);
            }),
        ),
        7 => (
            Some(0x0700_0050 | rd << 16 | ra << 12 | rm << 8 | x << 5 | rn),
            Some(0xfb40_0000 | rn << 16 | ra << 12 | rd << 8 | x << 4 | rm),
            Box::new(move |state| {
                let (low, high) = products(state);
                write_with_q(state, rd, low - high + signed(state, ra));
            }),
        ),
        8 => (
            Some(0x0700_f050 | rd << 16 | rm << 8 | x << 5 | rn),
            Some(0xfb40_f000 | rn << 16 | rd << 8 | x << 4 | rm),
            Box::new(move |state| {
                let (low, high) = products(state);
                state.r[rd as usize] = (low - high) as u32;
            }),
        ),
        9 => (
            Some(0x0740_0010 | hi << 16 | lo << 12 | rm << 8 | x << 5 | rn),
            Some(0xfbc0_00c0 | rn << 16 | lo << 12 | hi << 8 | x << 4 | rm),
            Box::new(move |state| {
                let (low, high) = products(state);
                pair(state, i128::from(accumulator(state)) + low + high);
            }),
        ),
        10 => (
            Some(0x0740_0050 | hi << 16 | lo << 12 | rm << 8 | x << 5 | rn),
            Some(0xfbd0_00c0 | rn << 16 | lo << 12 | hi << 8 | x << 4 | rm),
            Box::new(move |state| {
                let (low, high) = products(state);
                pair(state, i128::from(accumulator(state)) + low - high);
            }),
        ),
        // The most significant word of a product, with rounding for `x`.
        11 => (
            Some(0x0750_0010 | rd << 16 | ra << 12 | rm << 8 | x << 5 | rn),
            Some(0xfb50_0000 | rn << 16 | ra << 12 | rd << 8 | x << 4 | rm),
            Box::new(move |state| {
                let sum = (signed(state, ra) << 32) + signed(state, rn) * signed(state, rm);
                state.r[rd as usize] = ((sum + (i128::from(x) << 31)) >> 32) as u32;
            }),
        ),
        12 => (
            Some(0x0750_f010 | rd << 16 | rm << 8 | x << 5 | rn),
            Some(0xfb50_f000 | rn << 16 | rd << 8 | x << 4 | rm),
            Box::new(move |state| {
                let product = signed(state, rn) * signed(state, rm);
                state.r[rd as usize] = ((product + (i128::from(x) << 31)) >> 32) as u32;
            }),
        ),
        _ => (
            Some(0x0750_00d0 | rd << 16 | ra << 12 | rm << 8 | x << 5 | rn),
            Some(0xfb60_0000 | rn << 16 | ra << 12 | rd << 8 | x << 4 | rm),
            Box::new(move |state| {
                let difference = (signed(state, ra) << 32) - signed(state, rn) * signed(state, rm);
                state.r[rd as usize] = ((difference + (i128::from(x) << 31)) >> 32) as u32;
            }),
        ),
    }
}

/// SignedSatQ: the value in `bits` bits, and true when it was out of the range.
fn signed_saturate(value: i64, bits: u32) -> (i64, bool) {
    let (low, high) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
    (value.clamp(low, high), value < low || value > high)
}

fn unsigned_saturate(value: i64, bits: u32) -> (i64, bool) {
    let high = (1i64 << bits) - 1;
    (value.clamp(0, high), value < 0 || value > high)
}

fn saturating(random: &mut Random) -> Instruction {
    let [rd, rn, rm] = distinct(random);
    let signed = |state: &State, register: u32| i64::from(state.r[register as usize] as i32);
    match random.below(4) {
        0 => {
            // QADD, QSUB, QDADD and QDSUB.
            let (double, subtract) = (random.below(2), random.below(2));
            (
                Some(0x0100_0050 | double << 22 | subtract << 21 | rn << 16 | rd << 12 | rm),
                Some(0xfa80_f080 | rn << 16 | rd << 8 | subtract << 5 | double << 4 | rm),
                Box::new(move |state| {
                    let mut second = signed(state, rn);
                    if double == 1 {
                        let (doubled, saturated) = signed_saturate(2 * second, 32);
                        second = doubled;
                        state.q |= saturated;
                    }
                    let first = signed(state, rm);
                    let (result, saturated) = signed_saturate(
                        if subtract == 1 {
                            first - second
                        } else {
                            first + second
                        },
                        32,
                    );
                    state.q |= saturated;
                    state.r[rd as usize] = result as u32;
                }),
            )
        }
        1 | 2 => {
            // SSAT and USAT of a shifted register.
            let unsigned = random.below(2);
            let sat = random.below(32);
            let (arithmetic, imm5) = (random.below(2), random.below(32));
            // T32 has no ASR by 32: the encoding is that of SSAT16.
            let thumb = (arithmetic == 0 || imm5 != 0).then_some(
                0xf300_0000
                    | unsigned << 23
                    | arithmetic << 21
                    | rn << 16
                    | (imm5 >> 2) << 12
                    | rd << 8
                    | (imm5 & 3) << 6
                    | sat,
            );
            (
                Some(
                    0x06a0_0010
                        | unsigned << 22
                        | sat << 16
                        | rd << 12
                        | imm5 << 7
                        | arithmetic << 6
                        | rn,
                ),
                thumb,
                Box::new(move |state| {
                    let value = state.r[rn as usize];
                    let operand = match (arithmetic, imm5) {
                        (0, _) => value << imm5,
                        (_, 0) => (value as i32 >> 31) as u32,
                        _ => (value as i32 >> imm5) as u32,
                    };
                    let (result, saturated) = if unsigned == 1 {
                        unsigned_saturate(i64::from(operand as i32), sat)
                    } else {
                        signed_saturate(i64::from(operand as i32), sat + 1)
                    };
                    state.q |= saturated;
                    state.r[rd as usize] = result as u32;
                }),
            )
        }
        _ => {
            // SSAT16 and USAT16.
            let unsigned = random.below(2);
            let sat = random.below(16);
            (
                Some(0x06a0_0f30 | unsigned << 22 | sat << 16 | rd << 12 | rn),
                Some(0xf320_0000 | unsigned << 23 | rn << 16 | rd << 8 | sat),
                Box::new(move |state| {
                    let value = state.r[rn as usize];
                    let mut result = 0;
                    for part in 0..2 {
                        let operand = i64::from((value >> (16 * part)) as i16);
                        let (one, saturated) = if unsigned == 1 {
                            unsigned_saturate(operand, sat)
                        } else {
                            signed_saturate(operand, sat + 1)
                        };
                        state.q |= saturated;
                        result |= (one as u32 & 0xffff) << (16 * part);
                    }
                    state.r[rd as usize] = result;
                }),
            )
        }
    }
}

/// The parallel additions and subtractions. `prefix` is S, Q, SH, U, UQ, UH and
/// `operation` is ADD16, ASX, SAX, SUB16, ADD8, SUB8.
fn parallel(random: &mut Random) -> Instruction {
    let [rd, rn, rm] = distinct(random);
    let (prefix, operation) = (random.below(6), random.below(6));
    let arm_prefix = [
        0x0610_0000,
        0x0620_0000,
        0x0630_0000,
        0x0650_0000,
        0x0660_0000,
        0x0670_0000,
    ];
    let arm_operation = [0x10, 0x30, 0x50, 0x70, 0x90, 0xf0];
    let thumb_prefix = [0x00, 0x10, 0x20, 0x40, 0x50, 0x60];
    let thumb_operation = [
        0xfa90_f000,
        0xfaa0_f000,
        0xfae0_f000,
        0xfad0_f000,
        0xfa80_f000,
        0xfac0_f000,
    ];
    (
        Some(
            arm_prefix[prefix as usize]
                | rn << 16
                | rd << 12
                | 0xf00
                | arm_operation[operation as usize]
                | rm,
        ),
        Some(
            thumb_operation[operation as usize]
                | rn << 16
                | rd << 8
                | thumb_prefix[prefix as usize]
                | rm,
        ),
        Box::new(move |state| {
            let (n, m) = (state.r[rn as usize], state.r[rm as usize]);
            let unsigned = prefix >= 3;
            let bits = if operation < 4 { 16 } else { 8 };
            let lanes = 32 / bits;
            let lane = |value: u32, index: u32| {
                let part = value >> (index * bits) & ((1 << bits) - 1);
                match (unsigned, bits) {
                    (true, _) => i64::from(part),
                    (false, 16) => i64::from(part as i16),
                    (false, _) => i64::from(part as i8),
                }
            };
            let mut result = 0;
            let mut ge = 0;
            for index in 0..lanes {
                // The operands of the lane, and true for an addition.
                let (a, b, add) = match (operation, index) {
                    (0 | 4, _) => (lane(n, index), lane(m, index), true),
                    (3 | 5, _) => (lane(n, index), lane(m, index), false),
                    (1, 0) => (lane(n, 0), lane(m, 1), false),
                    (1, _) => (lane(n, 1), lane(m, 0), true),
                    (_, 0) => (lane(n, 0), lane(m, 1), true),
                    _ => (lane(n, 1), lane(m, 0), false),
                };
                let value = if add { a + b } else { a - b };
                let holds = match (unsigned, add) {
                    (false, _) => value >= 0,
                    (true, true) => value >= 1 << bits,
                    (true, false) => value >= 0,
                };
                if holds {
                    ge |= ((1 << (4 / lanes)) - 1) << (index * 4 / lanes);
                }
                let one = match prefix {
                    1 => signed_saturate(value, bits).0,
                    4 => unsigned_saturate(value, bits).0,
                    2 | 5 => value >> 1,
                    _ => value,
                };
                result |= (one as u32 & ((1 << bits) - 1)) << (index * bits);
            }
            state.r[rd as usize] = result;
            if prefix == 0 || prefix == 3 {
                state.ge = ge;
            }
        }),
    )
}

fn bits_and_bytes(random: &mut Random) -> Instruction {
    let [rd, rn, rm, ra] = distinct(random);
    let r = |state: &State, register: u32| state.r[register as usize];
    let lsb = random.below(32);
    let width = 1 + random.below(32 - lsb);
    let msb = lsb + width - 1;
    let mask = (u32::MAX >> (32 - width)) << lsb;
    let rotation = random.below(4);
    let thumb_field = |base: u32, rn: u32, low: u32| {
        base | rn << 16 | (lsb >> 2) << 12 | rd << 8 | (lsb & 3) << 6 | low
    };
    match random.below(16) {
        0 => (
            Some(0x07e0_0050 | (width - 1) << 16 | rd << 12 | lsb << 7 | rn),
            Some(thumb_field(0xf3c0_0000, rn, width - 1)),
            Box::new(move |state| state.r[rd as usize] = (r(state, rn) & mask) >> lsb),
        ),
        1 => (
            Some(0x07a0_0050 | (width - 1) << 16 | rd << 12 | lsb << 7 | rn),
            Some(thumb_field(0xf340_0000, rn, width - 1)),
            Box::new(move |state| {
                state.r[rd as usize] =
                    ((r(state, rn) << (32 - lsb - width)) as i32 >> (32 - width)) as u32;
            }),
        ),
        2 => (
            Some(0x07c0_0010 | msb << 16 | rd << 12 | lsb << 7 | rn),
            Some(thumb_field(0xf360_0000, rn, msb)),
            Box::new(move |state| {
                state.r[rd as usize] = r(state, rd) & !mask | r(state, rn) << lsb & mask;
            }),
        ),
        3 => (
            Some(0x07c0_001f | msb << 16 | rd << 12 | lsb << 7),
            Some(thumb_field(0xf360_0000, 15, msb)),
            Box::new(move |state| state.r[rd as usize] &= !mask),
        ),
        4 => (
            Some(0x016f_0f10 | rd << 12 | rm),
            Some(0xfab0_f080 | rm << 16 | rd << 8 | rm),
            Box::new(move |state| state.r[rd as usize] = r(state, rm).leading_zeros()),
        ),
        5 => (
            Some(0x06ff_0f30 | rd << 12 | rm),
            Some(0xfa90_f0a0 | rm << 16 | rd << 8 | rm),
            Box::new(move |state| state.r[rd as usize] = r(state, rm).reverse_bits()),
        ),
        6 => (
            Some(0x06bf_0f30 | rd << 12 | rm),
            Some(0xfa90_f080 | rm << 16 | rd << 8 | rm),
            Box::new(move |state| state.r[rd as usize] = r(state, rm).swap_bytes()),
        ),
        7 => (
            Some(0x06bf_0fb0 | rd << 12 | rm),
            Some(0xfa90_f090 | rm << 16 | rd << 8 | rm),
            Box::new(move |state| {
                let value = r(state, rm);
                state.r[rd as usize] = (value & 0xff00_ff00) >> 8 | (value & 0x00ff_00ff) << 8;
            }),
        ),
        8 => (
            Some(0x06ff_0fb0 | rd << 12 | rm),
            Some(0xfa90_f0b0 | rm << 16 | rd << 8 | rm),
            Box::new(move |state| {
                state.r[rd as usize] = (r(state, rm) as u16).swap_bytes() as i16 as u32;
            }),
        ),
        9 | 10 => {
            // The extends, with an addition for the forms with Rn.
            let kind = random.below(6);
            let rn = if random.below(2) == 0 { 15 } else { rn };
            let arm = [
                0x0680_0070,
                0x06a0_0070,
                0x06b0_0070,
                0x06c0_0070,
                0x06e0_0070,
                0x06f0_0070,
            ];
            let thumb = [
                0xfa20_f080,
                0xfa40_f080,
                0xfa00_f080,
                0xfa30_f080,
                0xfa50_f080,
                0xfa10_f080,
            ];
            (
                Some(arm[kind as usize] | rn << 16 | rd << 12 | rotation << 10 | rm),
                Some(thumb[kind as usize] | rn << 16 | rd << 8 | rotation << 4 | rm),
                Box::new(move |state| {
                    let rotated = r(state, rm).rotate_right(8 * rotation);
                    let base = if rn == 15 { 0 } else { r(state, rn) };
                    state.r[rd as usize] = match kind {
                        0 => {
                            let low = (base as u16).wrapping_add(rotated as i8 as u16);
                            let high =
                                ((base >> 16) as u16).wrapping_add((rotated >> 16) as i8 as u16);
                            u32::from(high) << 16 | u32::from(low)
                        }
                        1 => base.wrapping_add(rotated as i8 as u32),
                        2 => base.wrapping_add(rotated as i16 as u32),
                        3 => {
                            let low = (base as u16).wrapping_add(rotated as u8 as u16);
                            let high =
                                ((base >> 16) as u16).wrapping_add((rotated >> 16) as u8 as u16);
                            u32::from(high) << 16 | u32::from(low)
                        }
                        4 => base.wrapping_add(rotated & 0xff),
                        _ => base.wrapping_add(rotated & 0xffff),
                    };
                }),
            )
        }
        11 => (
            Some(0x0680_0fb0 | rn << 16 | rd << 12 | rm),
            Some(0xfaa0_f080 | rn << 16 | rd << 8 | rm),
            Box::new(move |state| {
                let mask = (0..4).fold(0, |mask, byte| {
                    mask | if state.ge >> byte & 1 != 0 {
                        0xff << (byte * 8)
                    } else {
                        0
                    }
                });
                state.r[rd as usize] = r(state, rn) & mask | r(state, rm) & !mask;
            }),
        ),
        12 | 13 => {
            // USAD8, and USADA8 with Ra.
            let accumulate = random.below(2) == 0;
            let ra = if accumulate { ra } else { 15 };
            (
                Some(0x0780_0010 | rd << 16 | ra << 12 | rm << 8 | rn),
                Some(0xfb70_0000 | rn << 16 | ra << 12 | rd << 8 | rm),
                Box::new(move |state| {
                    let (n, m) = (r(state, rn).to_le_bytes(), r(state, rm).to_le_bytes());
                    let sum: u32 = (0..4)
                        .map(|byte| u32::from(n[byte].abs_diff(m[byte])))
                        .sum();
                    let base = if accumulate { r(state, ra) } else { 0 };
                    state.r[rd as usize] = base.wrapping_add(sum);
                }),
            )
        }
        _ => {
            let signed = random.below(2);
            (
                Some(0x0710_f010 | (1 - signed) << 21 | rd << 16 | rm << 8 | rn),
                Some(0xfb90_f0f0 | (1 - signed) << 21 | rn << 16 | rd << 8 | rm),
                Box::new(move |state| {
                    let (n, m) = (r(state, rn), r(state, rm));
                    state.r[rd as usize] = match (signed, m) {
                        (_, 0) => 0,
                        (1, _) => (n as i32).wrapping_div(m as i32) as u32,
                        _ => n / m,
                    };
                }),
            )
        }
    }
}

const CLASSES: [(&str, fn(&mut Random) -> Instruction); 8] = [
    ("alu_immediate", alu_immediate),
    ("alu_shifted_register", alu_shifted_register),
    ("alu_register_shift", alu_register_shift),
    ("move_wide", move_wide),
    ("multiply", multiply),
    ("halfword_multiply", halfword_multiply),
    ("saturating", saturating),
    ("parallel", parallel),
];

const VALUES: [u32; 12] = [
    0,
    1,
    2,
    0x7f,
    0x80,
    0xff,
    0x7fff,
    0x8000,
    0xffff,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
];

fn check(
    trio: &mut Trio,
    random: &mut Random,
    name: &str,
    generate: fn(&mut Random) -> Instruction,
) -> Result<(), String> {
    let (arm, thumb, apply) = generate(random);
    let (raw, is_thumb) = match (arm, thumb) {
        (Some(arm), Some(thumb)) if random.below(2) == 0 => {
            let _ = thumb;
            (0xe000_0000 | arm, false)
        }
        (_, Some(thumb)) => (thumb, true),
        (Some(arm), None) => (0xe000_0000 | arm, false),
        (None, None) => unreachable!(),
    };
    let mut case = Case::new(random, is_thumb);
    case.cpu = random_cpu(random);
    for register in &mut case.cpu.regs[..15] {
        *register = match random.below(8) {
            0..=2 => VALUES[random.below(VALUES.len() as u32) as usize],
            3 => random.below(256),
            4 => (random.next() as i32 >> random.below(32)) as u32,
            _ => random.next(),
        };
    }
    let program = Program {
        thumb: is_thumb,
        at: CODE + random.below(64) * 4,
        insns: vec![wide(raw)],
    };
    case.load(&program);
    case.enter(&program);
    case.until = 1;
    let cpu = &case.cpu;
    let mut state = State {
        r: cpu.regs,
        n: cpu.cpsr.n,
        z: cpu.cpsr.z,
        c: cpu.cpsr.c,
        v: cpu.cpsr.v,
        q: cpu.cpsr.q,
        ge: cpu.cpsr.ge.into(),
    };
    apply(&mut state);
    state.r[15] = program.end();
    let outcome = trio.run(&case)?;
    let cpsr = outcome.cpsr;
    let engines = State {
        r: outcome.regs,
        n: cpsr >> 31 != 0,
        z: cpsr >> 30 & 1 != 0,
        c: cpsr >> 29 & 1 != 0,
        v: cpsr >> 28 & 1 != 0,
        q: cpsr >> 27 & 1 != 0,
        ge: cpsr >> 16 & 0xf,
    };
    if engines == state && outcome.stop.is_none() {
        return Ok(());
    }
    Err(format!(
        "{name}: {} {:?} {:08x} : {raw:08x} from {:x?} cpsr {:#x}\n  the engines: {:x?} {:?}\n  the model:   {:x?}",
        if is_thumb { "T32" } else { "A32" },
        rax::arm::Decoder::new(if is_thumb {
            rax::arm::ExecutionState::Thumb
        } else {
            rax::arm::ExecutionState::Aarch32
        })
        .decode(&program.bytes())
        .map(|insn| insn.mnemonic),
        raw & 0xfff0_00c0,
        cpu.regs,
        cpu.cpsr.to_u32(),
        engines,
        outcome.stop,
        state
    ))
}

#[test]
fn integer_instructions_give_the_results_of_the_architecture() {
    let mut trio = Trio::new();
    cases(
        "integer_instructions_give_the_results_of_the_architecture",
        0x0a11_ce5e_ed00_0006,
        120_000,
        |random| {
            let (name, generate) = CLASSES[random.below(CLASSES.len() as u32) as usize];
            check(&mut trio, random, name, generate)?;
            check(&mut trio, random, "bits_and_bytes", bits_and_bytes)
        },
    );
}
