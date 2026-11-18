//! Advanced SIMD instructions with three registers of the same length, on integer
//! elements, against a model that follows the pseudocode of the ARM architecture manual.

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
fn shift(value: i128, by: i128, bits: u32, unsigned: bool, round: bool, saturating: bool) -> (i128, bool) {
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
        return saturate(if value == 0 { 0 } else { value.signum() << 100 }, bits, unsigned);
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

fn check(trio: &mut Trio, random: &mut Random) -> Result<(), String> {
    let (kind, opcode, fixed_u, sizes) = KINDS[random.below(KINDS.len() as u32) as usize];
    let pairwise = matches!(kind, Kind::Pmax | Kind::Pmin | Kind::Padd);
    let quad = !pairwise && random.below(2) == 0;
    let u = fixed_u.unwrap_or_else(|| random.below(2));
    let size = match kind {
        Kind::Qdmulh | Kind::Qrdmulh => 1 + random.below(2),
        _ => random.below(sizes + 1),
    };
    let register = |random: &mut Random| (random.below(32) & if quad { !1 } else { !0 }) as usize;
    let operation = Operation {
        kind,
        // The operations with U in the opcode are the same with and with no sign,
        // but the two multiplies with a saturation, which have a sign.
        unsigned: fixed_u.map_or(u == 1, |_| !matches!(kind, Kind::Qdmulh | Kind::Qrdmulh)),
        size,
        quad,
        d: register(random),
        n: register(random),
        m: register(random),
    };
    let field = |number: usize, high: u32, low: u32| (number as u32 & 15) << high | (number as u32 >> 4) << low;
    let arm = 0xf200_0000
        | u << 24
        | size << 20
        | (opcode >> 1) << 8
        | u32::from(quad) << 6
        | (opcode & 1) << 4
        | field(operation.d, 12, 22)
        | field(operation.n, 16, 7)
        | field(operation.m, 0, 5);
    let thumb = random.below(2) == 0;
    let raw = if thumb {
        0xef00_0000 | u << 28 | arm & 0x00ff_ffff
    } else {
        arm
    };
    let mut case = Case::new(random, thumb);
    case.cpu = random_cpu(random);
    for register in &mut case.cpu.vfp.dregs {
        // Elements at the limits of each size, small ones, and random ones.
        *register = match random.below(4) {
            0 => {
                let limits = [0u64, 1, 0x7f, 0x80, 0xff, 0x7fff, 0x8000, 0xffff, 0x7fff_ffff, 0x8000_0000];
                (0..8).fold(0, |value, _| {
                    value << 8 << random.below(4) * 8 | limits[random.below(10) as usize]
                })
            }
            1 => (0..8).fold(0, |value, _| value << 8 | u64::from(random.below(67).wrapping_sub(33) as u8)),
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
        insns: vec![wide(raw), if thumb { t16(0xe7fc) } else { wide(0xeaff_fffd) }],
    };
    case.load(&program);
    case.enter(&program);
    case.until = 3;
    let mut d = case.cpu.vfp.dregs;
    let mut fpscr = case.cpu.vfp.fpscr.bits();
    let vector = |d: &[u64; 32], number: usize| {
        let high = if quad { d[number + 1] } else { 0 };
        u128::from(high) << 64 | u128::from(d[number])
    };
    for _ in 0..2 {
        let (result, saturated) = operation.apply(
            vector(&d, operation.d),
            vector(&d, operation.n),
            vector(&d, operation.m),
        );
        d[operation.d] = result as u64;
        if quad {
            d[operation.d + 1] = (result >> 64) as u64;
        }
        if saturated {
            fpscr |= QC;
        }
    }
    let outcome = trio.run(&case)?;
    if outcome.dregs == d && outcome.fpscr == fpscr && outcome.stop.is_none() {
        return Ok(());
    }
    let different: Vec<_> = (0..32)
        .filter(|&index| outcome.dregs[index] != d[index])
        .map(|index| format!("d{index}: {:016x}, the model {:016x}", outcome.dregs[index], d[index]))
        .collect();
    Err(format!(
        "{kind:?} unsigned {} size {size} quad {quad} : {} {raw:08x} d{} = d{}, d{} from fpscr {:#x} dregs {:x?}\n  the engines: fpscr {:#x} {:?}, the model: fpscr {fpscr:#x}\n  {different:?}",
        operation.unsigned,
        if thumb { "T32" } else { "A32" },
        operation.d,
        operation.n,
        operation.m,
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
