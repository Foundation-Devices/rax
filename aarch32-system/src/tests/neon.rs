//! Advanced SIMD instructions on integer elements against a model that follows the
//! pseudocode of the ARM architecture manual: three registers of the same length, the
//! shifts by an immediate value, and the operations with two lengths.

use rax::arm::vfp::Fpscr;

use super::Random;
use super::suite::{CODE, Case, Program, Trio, cases, random_cpu, t16, wide};

const QC: u32 = 1 << 27;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Hadd,
    Qadd,
    Rhadd,
    Logic,
    Hsub,
    Qsub,
    Cgt,
    Cge,
    Shl,
    Qshl,
    Rshl,
    Qrshl,
    Max,
    Min,
    Abd,
    Aba,
    Add,
    Sub,
    Tst,
    Ceq,
    Mla,
    Mls,
    Mul,
    Pmax,
    Pmin,
    Padd,
    Qdmulh,
    Qrdmulh,
}

/// The operation, its opcode field with the bit 4 below it, the U bit when it is a
/// part of the opcode, and the largest size field.
const KINDS: [(Kind, u32, Option<u32>, u32); 28] = [
    (Kind::Hadd, 0b0000_0, None, 2),
    (Kind::Qadd, 0b0000_1, None, 3),
    (Kind::Rhadd, 0b0001_0, None, 2),
    (Kind::Logic, 0b0001_1, None, 3),
    (Kind::Hsub, 0b0010_0, None, 2),
    (Kind::Qsub, 0b0010_1, None, 3),
    (Kind::Cgt, 0b0011_0, None, 2),
    (Kind::Cge, 0b0011_1, None, 2),
    (Kind::Shl, 0b0100_0, None, 3),
    (Kind::Qshl, 0b0100_1, None, 3),
    (Kind::Rshl, 0b0101_0, None, 3),
    (Kind::Qrshl, 0b0101_1, None, 3),
    (Kind::Max, 0b0110_0, None, 2),
    (Kind::Min, 0b0110_1, None, 2),
    (Kind::Abd, 0b0111_0, None, 2),
    (Kind::Aba, 0b0111_1, None, 2),
    (Kind::Add, 0b1000_0, Some(0), 3),
    (Kind::Sub, 0b1000_0, Some(1), 3),
    (Kind::Tst, 0b1000_1, Some(0), 2),
    (Kind::Ceq, 0b1000_1, Some(1), 2),
    (Kind::Mla, 0b1001_0, Some(0), 2),
    (Kind::Mls, 0b1001_0, Some(1), 2),
    (Kind::Mul, 0b1001_1, Some(0), 2),
    (Kind::Pmax, 0b1010_0, None, 2),
    (Kind::Pmin, 0b1010_1, None, 2),
    (Kind::Padd, 0b1011_1, Some(0), 2),
    (Kind::Qdmulh, 0b1011_0, Some(0), 2),
    (Kind::Qrdmulh, 0b1011_0, Some(1), 2),
];

/// The value in `bits` bits, and true when it was out of the range.
fn saturate(value: i128, bits: u32, unsigned: bool) -> (i128, bool) {
    let (low, high) = if unsigned {
        (0, (1i128 << bits) - 1)
    } else {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    };
    (value.clamp(low, high), value < low || value > high)
}

/// VSHL, VQSHL, VRSHL and VQRSHL of one element: a shift to the left by the signed low
/// byte of `by`, to the right when it is negative.
fn shift(
    value: i128,
    by: i128,
    bits: u32,
    unsigned: bool,
    round: bool,
    saturating: bool,
) -> (i128, bool) {
    let by = i128::from(by as i8);
    if by < 0 {
        let right = (-by) as u32;
        // A shift by more than the element and its rounding bit gives 0, or all the
        // sign bits with no rounding.
        if right > bits + 1 {
            return (if round || value >= 0 { 0 } else { -1 }, false);
        }
        let rounding = if round { 1i128 << (right - 1) } else { 0 };
        return ((value + rounding) >> right, false);
    }
    let left = by as u32;
    if !saturating {
        return (if left >= bits { 0 } else { value << left }, false);
    }
    if left >= bits {
        return saturate(
            if value == 0 { 0 } else { value.signum() << 100 },
            bits,
            unsigned,
        );
    }
    saturate(value << left, bits, unsigned)
}

struct Operation {
    kind: Kind,
    unsigned: bool,
    /// The log2 of the bytes of an element, or the number of the logic operation.
    size: u32,
    quad: bool,
    d: usize,
    n: usize,
    m: usize,
}

impl Operation {
    /// The new value of the result register, and true for a saturation.
    fn apply(&self, d: u128, n: u128, m: u128) -> (u128, bool) {
        if self.kind == Kind::Logic {
            let value = match (self.unsigned, self.size) {
                (false, 0) => n & m,
                (false, 1) => n & !m,
                (false, 2) => n | m,
                (false, _) => n | !m,
                (true, 0) => n ^ m,
                (true, 1) => n & d | m & !d,
                (true, 2) => n & m | d & !m,
                (true, _) => d & m | n & !m,
            };
            return (value, false);
        }
        let bits = 8 << self.size;
        let count = if self.quad { 128 } else { 64 } / bits;
        let mask = (1u128 << bits) - 1;
        let lane = |vector: u128, index: u32| {
            let raw = vector >> (index * bits) & mask;
            if self.unsigned || raw >> (bits - 1) == 0 {
                raw as i128
            } else {
                raw as i128 - (1i128 << bits)
            }
        };
        let ones = |holds: bool| if holds { -1 } else { 0 };
        let mut result = 0u128;
        let mut saturated = false;
        for index in 0..count {
            // The pairwise operations take two elements of one operand.
            let (a, b) = match self.kind {
                Kind::Pmax | Kind::Pmin | Kind::Padd if index < count / 2 => {
                    (lane(n, 2 * index), lane(n, 2 * index + 1))
                }
                Kind::Pmax | Kind::Pmin | Kind::Padd => {
                    (lane(m, 2 * index - count), lane(m, 2 * index - count + 1))
                }
                _ => (lane(n, index), lane(m, index)),
            };
            let c = lane(d, index);
            let (value, out_of_range) = match self.kind {
                Kind::Hadd => ((a + b) >> 1, false),
                Kind::Rhadd => ((a + b + 1) >> 1, false),
                Kind::Hsub => ((a - b) >> 1, false),
                Kind::Qadd => saturate(a + b, bits, self.unsigned),
                Kind::Qsub => saturate(a - b, bits, self.unsigned),
                Kind::Cgt => (ones(a > b), false),
                Kind::Cge => (ones(a >= b), false),
                // The value is the second operand, the amount the first one.
                Kind::Shl => shift(b, a, bits, self.unsigned, false, false),
                Kind::Qshl => shift(b, a, bits, self.unsigned, false, true),
                Kind::Rshl => shift(b, a, bits, self.unsigned, true, false),
                Kind::Qrshl => shift(b, a, bits, self.unsigned, true, true),
                Kind::Max | Kind::Pmax => (a.max(b), false),
                Kind::Min | Kind::Pmin => (a.min(b), false),
                Kind::Abd => ((a - b).abs(), false),
                Kind::Aba => (c + (a - b).abs(), false),
                Kind::Add | Kind::Padd => (a + b, false),
                Kind::Sub => (a - b, false),
                Kind::Tst => (ones(a & b != 0), false),
                Kind::Ceq => (ones(a == b), false),
                Kind::Mla => (c + a * b, false),
                Kind::Mls => (c - a * b, false),
                Kind::Mul => (a * b, false),
                Kind::Qdmulh => saturate((2 * a * b) >> bits, bits, false),
                Kind::Qrdmulh => saturate((2 * a * b + (1 << (bits - 1))) >> bits, bits, false),
                Kind::Logic => unreachable!(),
            };
            saturated |= out_of_range;
            result |= (value as u128 & mask) << (index * bits);
        }
        (result, saturated)
    }
}

/// The elements of `count` elements of `bits` bits of a vector, with their sign or not.
fn elements(vector: u128, bits: u32, count: u32, unsigned: bool) -> Vec<i128> {
    let mask = if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    (0..count)
        .map(|index| {
            let raw = vector >> (index * bits) & mask;
            if unsigned || raw >> (bits - 1) == 0 {
                raw as i128
            } else {
                raw as i128 - (1i128 << bits)
            }
        })
        .collect()
}

fn vector_of(values: &[i128], bits: u32) -> u128 {
    let mask = (1u128 << bits) - 1;
    values
        .iter()
        .enumerate()
        .fold(0, |vector, (index, &value)| {
            vector | (value as u128 & mask) << (index as u32 * bits)
        })
}

fn field(number: usize, high: u32, low: u32) -> u32 {
    (number as u32 & 15) << high | (number as u32 >> 4) << low
}

/// What one instruction does to the D registers: the name of the case, the A32 word,
/// and the change, which returns true for a saturation.
type Instruction = (String, u32, Box<dyn Fn(&mut [u64; 32]) -> bool>);

fn quad(d: &[u64; 32], number: usize) -> u128 {
    u128::from(d[number + 1]) << 64 | u128::from(d[number])
}

fn set_quad(d: &mut [u64; 32], number: usize, value: u128) {
    d[number] = value as u64;
    d[number + 1] = (value >> 64) as u64;
}

fn three_registers(random: &mut Random) -> Instruction {
    let (kind, opcode, fixed_u, sizes) = KINDS[random.below(KINDS.len() as u32) as usize];
    let pairwise = matches!(kind, Kind::Pmax | Kind::Pmin | Kind::Padd);
    let is_quad = !pairwise && random.below(2) == 0;
    let u = fixed_u.unwrap_or_else(|| random.below(2));
    let size = match kind {
        Kind::Qdmulh | Kind::Qrdmulh => 1 + random.below(2),
        _ => random.below(sizes + 1),
    };
    let register =
        |random: &mut Random| (random.below(32) & if is_quad { !1 } else { !0 }) as usize;
    let operation = Operation {
        kind,
        // The operations with U in the opcode are the same with and with no sign,
        // but the two multiplies with a saturation, which have a sign.
        unsigned: fixed_u.map_or(u == 1, |_| !matches!(kind, Kind::Qdmulh | Kind::Qrdmulh)),
        size,
        quad: is_quad,
        d: register(random),
        n: register(random),
        m: register(random),
    };
    let arm = 0xf200_0000
        | u << 24
        | size << 20
        | (opcode >> 1) << 8
        | u32::from(is_quad) << 6
        | (opcode & 1) << 4
        | field(operation.d, 12, 22)
        | field(operation.n, 16, 7)
        | field(operation.m, 0, 5);
    (
        format!(
            "{kind:?} unsigned {} size {size} quad {is_quad}",
            operation.unsigned
        ),
        arm,
        Box::new(move |d| {
            let vector = |d: &[u64; 32], number: usize| {
                if is_quad {
                    quad(d, number)
                } else {
                    d[number].into()
                }
            };
            let (result, saturated) = operation.apply(
                vector(d, operation.d),
                vector(d, operation.n),
                vector(d, operation.m),
            );
            d[operation.d] = result as u64;
            if is_quad {
                d[operation.d + 1] = (result >> 64) as u64;
            }
            saturated
        }),
    )
}

/// The shifts by an immediate value: `1111 001U 1D imm6 Vd opcode L Q M 1 Vm`.
fn shift_immediate(random: &mut Random) -> Instruction {
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Shift {
        Shr,
        Sra,
        Rshr,
        Rsra,
        Sri,
        Shl,
        Sli,
        Qshlu,
        Qshl,
        Shrn,
        Rshrn,
        Qshrun,
        Qrshrun,
        Qshrn,
        Qrshrn,
        Shll,
    }
    // The operation, its opcode, the U bit when it is a part of the opcode, and the
    // bit 6 for the operations with a result of an other length.
    const SHIFTS: [(Shift, u32, Option<u32>, Option<u32>); 16] = [
        (Shift::Shr, 0b0000, None, None),
        (Shift::Sra, 0b0001, None, None),
        (Shift::Rshr, 0b0010, None, None),
        (Shift::Rsra, 0b0011, None, None),
        (Shift::Sri, 0b0100, Some(1), None),
        (Shift::Shl, 0b0101, Some(0), None),
        (Shift::Sli, 0b0101, Some(1), None),
        (Shift::Qshlu, 0b0110, Some(1), None),
        (Shift::Qshl, 0b0111, None, None),
        (Shift::Shrn, 0b1000, Some(0), Some(0)),
        (Shift::Rshrn, 0b1000, Some(0), Some(1)),
        (Shift::Qshrun, 0b1000, Some(1), Some(0)),
        (Shift::Qrshrun, 0b1000, Some(1), Some(1)),
        (Shift::Qshrn, 0b1001, None, Some(0)),
        (Shift::Qrshrn, 0b1001, None, Some(1)),
        (Shift::Shll, 0b1010, None, Some(0)),
    ];
    let (kind, opcode, fixed_u, fixed_bit6) = SHIFTS[random.below(16) as usize];
    let u = fixed_u.unwrap_or_else(|| random.below(2));
    let narrow = matches!(
        kind,
        Shift::Shrn | Shift::Rshrn | Shift::Qshrun | Shift::Qrshrun | Shift::Qshrn | Shift::Qrshrn
    );
    let long = kind == Shift::Shll;
    let left = matches!(
        kind,
        Shift::Shl | Shift::Sli | Shift::Qshlu | Shift::Qshl | Shift::Shll
    );
    // The size of the smaller element. The operations with two lengths have none of 64.
    let bits = 8 << random.below(if narrow || long { 3 } else { 4 });
    let amount = if left {
        random.below(bits)
    } else {
        1 + random.below(bits)
    };
    // L:imm6 has the size as its top set bit, and the amount below it.
    let l_imm6 = if left {
        bits + amount
    } else {
        2 * bits - amount
    };
    let is_quad = fixed_bit6.is_none() && random.below(2) == 0;
    let bit6 = fixed_bit6.unwrap_or(u32::from(is_quad));
    let register =
        |random: &mut Random, even: bool| (random.below(32) & if even { !1 } else { !0 }) as usize;
    let d = register(random, is_quad || long);
    let m = register(random, is_quad || narrow);
    let arm = 0xf280_0010
        | u << 24
        | (l_imm6 & 0x3f) << 16
        | opcode << 8
        | (l_imm6 >> 6) << 7
        | bit6 << 6
        | field(d, 12, 22)
        | field(m, 0, 5);
    let unsigned = match kind {
        Shift::Qshlu | Shift::Qshrun | Shift::Qrshrun => false,
        _ => u == 1,
    };
    (
        format!("{kind:?} unsigned {unsigned} bits {bits} quad {is_quad}"),
        arm,
        Box::new(move |regs| {
            let (source_bits, count) = match (narrow, is_quad) {
                (true, _) => (2 * bits, 64 / bits),
                (false, true) => (bits, 128 / bits),
                (false, false) => (bits, 64 / bits),
            };
            let source = if narrow || is_quad {
                quad(regs, m)
            } else {
                regs[m].into()
            };
            let old = if long || is_quad {
                quad(regs, d)
            } else {
                regs[d].into()
            };
            let result_bits = if long { 2 * bits } else { bits };
            let values = elements(source, source_bits, count, unsigned);
            let old_values = elements(old, result_bits, count, true);
            let mut saturated = false;
            let rounding = 1i128 << (amount.max(1) - 1);
            let ones = (1i128 << bits) - 1;
            let results: Vec<i128> = values
                .iter()
                .zip(&old_values)
                .map(|(&value, &old)| {
                    let (result, out_of_range) = match kind {
                        Shift::Shr => (value >> amount, false),
                        Shift::Sra => (old + (value >> amount), false),
                        Shift::Rshr => ((value + rounding) >> amount, false),
                        Shift::Rsra => (old + ((value + rounding) >> amount), false),
                        // The bits of the shifted value go in, the other bits stay.
                        Shift::Sri => {
                            let mask = ones >> amount;
                            (old & !mask | value >> amount & mask, false)
                        }
                        Shift::Sli => {
                            let mask = ones << amount & ones;
                            (old & !mask | value << amount & mask, false)
                        }
                        Shift::Shl | Shift::Shll => (value << amount, false),
                        Shift::Qshlu => saturate(value << amount, bits, true),
                        Shift::Qshl => saturate(value << amount, bits, unsigned),
                        Shift::Shrn => (value >> amount, false),
                        Shift::Rshrn => ((value + rounding) >> amount, false),
                        Shift::Qshrun => saturate(value >> amount, bits, true),
                        Shift::Qrshrun => saturate((value + rounding) >> amount, bits, true),
                        Shift::Qshrn => saturate(value >> amount, bits, unsigned),
                        Shift::Qrshrn => saturate((value + rounding) >> amount, bits, unsigned),
                    };
                    saturated |= out_of_range;
                    result
                })
                .collect();
            let result = vector_of(&results, result_bits);
            if long || is_quad {
                set_quad(regs, d, result);
            } else {
                regs[d] = result as u64;
            }
            saturated
        }),
    )
}

/// The operations with operands and a result of two lengths:
/// `1111 001U 1D size Vn Vd opcode N 0 M 0 Vm`.
fn two_lengths(random: &mut Random) -> Instruction {
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Long {
        Addl,
        Addw,
        Subl,
        Subw,
        Addhn,
        Raddhn,
        Abal,
        Subhn,
        Rsubhn,
        Abdl,
        Mlal,
        Mlsl,
        Mull,
        Qdmlal,
        Qdmlsl,
        Qdmull,
    }
    const LONGS: [(Long, u32, Option<u32>); 16] = [
        (Long::Addl, 0b0000, None),
        (Long::Addw, 0b0001, None),
        (Long::Subl, 0b0010, None),
        (Long::Subw, 0b0011, None),
        (Long::Addhn, 0b0100, Some(0)),
        (Long::Raddhn, 0b0100, Some(1)),
        (Long::Abal, 0b0101, None),
        (Long::Subhn, 0b0110, Some(0)),
        (Long::Rsubhn, 0b0110, Some(1)),
        (Long::Abdl, 0b0111, None),
        (Long::Mlal, 0b1000, None),
        (Long::Mlsl, 0b1010, None),
        (Long::Mull, 0b1100, None),
        (Long::Qdmlal, 0b1001, Some(0)),
        (Long::Qdmlsl, 0b1011, Some(0)),
        (Long::Qdmull, 0b1101, Some(0)),
    ];
    let (kind, opcode, fixed_u) = LONGS[random.below(16) as usize];
    let u = fixed_u.unwrap_or_else(|| random.below(2));
    let doubling = matches!(kind, Long::Qdmlal | Long::Qdmlsl | Long::Qdmull);
    let size = if doubling {
        1 + random.below(2)
    } else {
        random.below(3)
    };
    let bits = 8 << size;
    let narrow = matches!(
        kind,
        Long::Addhn | Long::Raddhn | Long::Subhn | Long::Rsubhn
    );
    let wide_n = narrow || matches!(kind, Long::Addw | Long::Subw);
    let register =
        |random: &mut Random, even: bool| (random.below(32) & if even { !1 } else { !0 }) as usize;
    let d = register(random, !narrow);
    let n = register(random, wide_n);
    let m = register(random, narrow);
    let arm = 0xf280_0000
        | u << 24
        | size << 20
        | opcode << 8
        | field(d, 12, 22)
        | field(n, 16, 7)
        | field(m, 0, 5);
    let unsigned = !doubling && u == 1;
    (
        format!("{kind:?} unsigned {unsigned} size {size}"),
        arm,
        Box::new(move |regs| {
            let count = 64 / bits;
            let operand = |regs: &[u64; 32], number: usize, wide: bool| {
                if wide {
                    elements(quad(regs, number), 2 * bits, count, unsigned)
                } else {
                    elements(regs[number].into(), bits, count, unsigned)
                }
            };
            let (a, b) = (operand(regs, n, wide_n), operand(regs, m, narrow));
            let old = if narrow {
                vec![0; count as usize]
            } else {
                operand(regs, d, true)
            };
            let mut saturated = false;
            let rounding = 1i128 << (bits - 1);
            let results: Vec<i128> = (0..count as usize)
                .map(|index| {
                    let (a, b, old) = (a[index], b[index], old[index]);
                    let mut doubled = || {
                        let (product, out_of_range) = saturate(2 * a * b, 2 * bits, false);
                        saturated |= out_of_range;
                        product
                    };
                    let (result, out_of_range) = match kind {
                        Long::Addl | Long::Addw => (a + b, false),
                        Long::Subl | Long::Subw => (a - b, false),
                        Long::Addhn => ((a + b) >> bits, false),
                        Long::Raddhn => ((a + b + rounding) >> bits, false),
                        Long::Subhn => ((a - b) >> bits, false),
                        Long::Rsubhn => ((a - b + rounding) >> bits, false),
                        Long::Abal => (old + (a - b).abs(), false),
                        Long::Abdl => ((a - b).abs(), false),
                        Long::Mlal => (old + a * b, false),
                        Long::Mlsl => (old - a * b, false),
                        Long::Mull => (a * b, false),
                        Long::Qdmlal => saturate(old + doubled(), 2 * bits, false),
                        Long::Qdmlsl => saturate(old - doubled(), 2 * bits, false),
                        Long::Qdmull => (doubled(), false),
                    };
                    saturated |= out_of_range;
                    result
                })
                .collect();
            if narrow {
                regs[d] = vector_of(&results, bits) as u64;
            } else {
                set_quad(regs, d, vector_of(&results, 2 * bits));
            }
            saturated
        }),
    )
}

fn check(trio: &mut Trio, random: &mut Random) -> Result<(), String> {
    let (name, arm, apply) = match random.below(3) {
        0 => three_registers(random),
        1 => shift_immediate(random),
        _ => two_lengths(random),
    };
    let thumb = random.below(2) == 0;
    let raw = if thumb {
        0xef00_0000 | (arm >> 24 & 1) << 28 | arm & 0x00ff_ffff
    } else {
        arm
    };
    let mut case = Case::new(random, thumb);
    case.cpu = random_cpu(random);
    for register in &mut case.cpu.vfp.dregs {
        // Elements at the limits of each size, small ones, and random ones.
        *register = match random.below(4) {
            0 => {
                let limits = [
                    0u64,
                    1,
                    0x7f,
                    0x80,
                    0xff,
                    0x7fff,
                    0x8000,
                    0xffff,
                    0x7fff_ffff,
                    0x8000_0000,
                ];
                (0..8).fold(0, |value, _| {
                    value << 8 << random.below(4) * 8 | limits[random.below(10) as usize]
                })
            }
            1 => (0..8).fold(0, |value, _| {
                value << 8 | u64::from(random.below(67).wrapping_sub(33) as u8)
            }),
            _ => u64::from(random.next()) << 32 | u64::from(random.next()),
        };
    }
    case.cpu.vfp.fpscr = Fpscr::from_bits(random.next() & (0xf000_0000 | QC));
    let at = CODE + random.below(64) * 4;
    // The instruction, then a branch back to it: the second run is one of the fast
    // interpreter and of the compiled code, when they have the operation.
    let program = Program {
        thumb,
        at,
        insns: vec![
            wide(raw),
            if thumb {
                t16(0xe7fc)
            } else {
                wide(0xeaff_fffd)
            },
        ],
    };
    case.load(&program);
    case.enter(&program);
    case.until = 3;
    let mut d = case.cpu.vfp.dregs;
    let mut fpscr = case.cpu.vfp.fpscr.bits();
    for _ in 0..2 {
        if apply(&mut d) {
            fpscr |= QC;
        }
    }
    let outcome = trio.run(&case)?;
    if outcome.dregs == d && outcome.fpscr == fpscr && outcome.stop.is_none() {
        return Ok(());
    }
    let different: Vec<_> = (0..32)
        .filter(|&index| outcome.dregs[index] != d[index])
        .map(|index| {
            format!(
                "d{index}: {:016x}, the model {:016x}",
                outcome.dregs[index], d[index]
            )
        })
        .collect();
    Err(format!(
        "{name} : {} {raw:08x} from fpscr {:#x} dregs {:x?}\n  the engines: fpscr {:#x} {:?}, the model: fpscr {fpscr:#x}\n  {different:?}",
        if thumb { "T32" } else { "A32" },
        case.cpu.vfp.fpscr.bits(),
        case.cpu.vfp.dregs,
        outcome.fpscr,
        outcome.stop,
    ))
}

#[test]
fn simd_integer_instructions_give_the_results_of_the_architecture() {
    let mut trio = Trio::new();
    cases(
        "simd_integer_instructions_give_the_results_of_the_architecture",
        0x0a11_ce5e_ed00_000c,
        60_000,
        |random| check(&mut trio, random),
    );
}
