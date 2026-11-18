//! VFP instructions against the FP unit of an AArch64 host. The host has the
//! arithmetic of the ARM architecture: the same rounding modes, flush to zero, default
//! NaN, NaN operands and cumulative exception flags as a VFP.

use std::arch::asm;

use rax::arm::vfp::Fpscr;

use super::Random;
use super::suite::{CODE, Case, Program, Trio, cases, random_cpu, wide};

/// The rounding mode, flush to zero and default NaN: the same bits in FPSCR and FPCR.
const MODES: u32 = 0x03c0_0000;
/// The cumulative exception flags: the same bits in FPSCR and FPSR.
const FLAGS: u32 = 0x9f;

/// Runs host instructions on d0 to d2 with the modes of the guest. Returns the bits of
/// d0, then the exception flags, then the NZCV flags.
macro_rules! host {
    ($state:expr, $a:expr, $b:expr, $c:expr, $($code:literal),+) => {{
        let (result, flags, nzcv): (u64, u64, u64);
        // SAFETY: the code changes only the registers of the list, the flags, and
        // the FPCR for its own time.
        unsafe {
            asm!(
                "mrs {saved}, fpcr",
                "msr fpcr, {modes}",
                "msr fpsr, xzr",
                "fmov d0, {a}",
                "fmov d1, {b}",
                "fmov d2, {c}",
                $($code,)+
                "fmov {result}, d0",
                "mrs {flags}, fpsr",
                "mrs {nzcv}, nzcv",
                "msr fpcr, {saved}",
                saved = out(reg) _,
                modes = in(reg) u64::from($state.fpscr & MODES),
                a = in(reg) $a,
                b = in(reg) $b,
                c = in(reg) $c,
                result = out(reg) result,
                flags = out(reg) flags,
                nzcv = out(reg) nzcv,
                out("x9") _,
                out("v0") _,
                out("v1") _,
                out("v2") _,
                out("v3") _,
                options(nomem, nostack),
            );
        }
        $state.fpscr |= flags as u32 & FLAGS;
        (result, nzcv as u32 & 0xf000_0000)
    }};
}

#[derive(Clone, Debug, PartialEq)]
struct State {
    d: [u64; 32],
    fpscr: u32,
}

impl State {
    fn s(&self, number: u32) -> u64 {
        self.d[number as usize / 2] >> (32 * (number & 1)) & 0xffff_ffff
    }

    fn set_s(&mut self, number: u32, value: u64) {
        let shift = 32 * (number & 1);
        let register = &mut self.d[number as usize / 2];
        *register = *register & !(0xffff_ffff << shift) | (value & 0xffff_ffff) << shift;
    }

    /// A register of the size of the instruction.
    fn get(&self, double: bool, number: u32) -> u64 {
        if double {
            self.d[number as usize]
        } else {
            self.s(number)
        }
    }

    fn set(&mut self, double: bool, number: u32, value: u64) {
        if double {
            self.d[number as usize] = value;
        } else {
            self.set_s(number, value);
        }
    }
}

type Apply = Box<dyn Fn(&mut State)>;

/// The fields of a register in the places of Vd, Vn and Vm.
fn fields(double: bool, d: u32, n: u32, m: u32) -> u32 {
    let field = |number: u32, high: u32, low: u32| {
        if double {
            (number & 15) << high | (number >> 4) << low
        } else {
            (number >> 1) << high | (number & 1) << low
        }
    };
    field(d, 12, 22) | field(n, 16, 7) | field(m, 0, 5)
}

fn single_field(number: u32, high: u32, low: u32) -> u32 {
    (number >> 1) << high | (number & 1) << low
}

/// The operations with three registers. The names are those of the VFP instructions.
fn arithmetic(random: &mut Random) -> (u32, Apply) {
    let double = random.below(2) == 0;
    let (d, n, m) = (random.below(32), random.below(32), random.below(32));
    let size = u32::from(double) << 8;
    let kind = random.below(13);
    let base = [
        0xee30_0a00,
        0xee30_0a40,
        0xee20_0a00,
        0xee20_0a40,
        0xee80_0a00,
        0xee00_0a00,
        0xee00_0a40,
        0xee10_0a40,
        0xee10_0a00,
        0xeea0_0a00,
        0xeea0_0a40,
        0xee90_0a40,
        0xee90_0a00,
    ][kind as usize];
    (
        base | size | fields(double, d, n, m),
        Box::new(move |state| {
            let (a, b, c) = (
                state.get(double, d),
                state.get(double, n),
                state.get(double, m),
            );
            let (result, _) = match (kind, double) {
                (0, true) => host!(state, a, b, c, "fadd d0, d1, d2"),
                (0, false) => host!(state, a, b, c, "fadd s0, s1, s2"),
                (1, true) => host!(state, a, b, c, "fsub d0, d1, d2"),
                (1, false) => host!(state, a, b, c, "fsub s0, s1, s2"),
                (2, true) => host!(state, a, b, c, "fmul d0, d1, d2"),
                (2, false) => host!(state, a, b, c, "fmul s0, s1, s2"),
                (3, true) => host!(state, a, b, c, "fnmul d0, d1, d2"),
                (3, false) => host!(state, a, b, c, "fnmul s0, s1, s2"),
                (4, true) => host!(state, a, b, c, "fdiv d0, d1, d2"),
                (4, false) => host!(state, a, b, c, "fdiv s0, s1, s2"),
                // VMLA, VMLS, VNMLA and VNMLS round the product, then the sum.
                (5, true) => host!(state, a, b, c, "fmul d3, d1, d2", "fadd d0, d0, d3"),
                (5, false) => host!(state, a, b, c, "fmul s3, s1, s2", "fadd s0, s0, s3"),
                (6, true) => host!(state, a, b, c, "fmul d3, d1, d2", "fneg d3, d3", "fadd d0, d0, d3"),
                (6, false) => host!(state, a, b, c, "fmul s3, s1, s2", "fneg s3, s3", "fadd s0, s0, s3"),
                (7, true) => host!(
                    state, a, b, c, "fmul d3, d1, d2", "fneg d3, d3", "fneg d0, d0", "fadd d0, d0, d3"
                ),
                (7, false) => host!(
                    state, a, b, c, "fmul s3, s1, s2", "fneg s3, s3", "fneg s0, s0", "fadd s0, s0, s3"
                ),
                (8, true) => host!(state, a, b, c, "fmul d3, d1, d2", "fneg d0, d0", "fadd d0, d0, d3"),
                (8, false) => host!(state, a, b, c, "fmul s3, s1, s2", "fneg s0, s0", "fadd s0, s0, s3"),
                // VFMA, VFMS, VFNMA and VFNMS round one time.
                (9, true) => host!(state, a, b, c, "fmadd d0, d1, d2, d0"),
                (9, false) => host!(state, a, b, c, "fmadd s0, s1, s2, s0"),
                (10, true) => host!(state, a, b, c, "fmsub d0, d1, d2, d0"),
                (10, false) => host!(state, a, b, c, "fmsub s0, s1, s2, s0"),
                (11, true) => host!(state, a, b, c, "fnmadd d0, d1, d2, d0"),
                (11, false) => host!(state, a, b, c, "fnmadd s0, s1, s2, s0"),
                (_, true) => host!(state, a, b, c, "fnmsub d0, d1, d2, d0"),
                (_, false) => host!(state, a, b, c, "fnmsub s0, s1, s2, s0"),
            };
            state.set(double, d, result);
        }),
    )
}

/// VABS, VNEG, VSQRT, VMOV, the compares, and the conversions.
fn one_operand(random: &mut Random) -> (u32, Apply) {
    let double = random.below(2) == 0;
    let (d, m) = (random.below(32), random.below(32));
    let size = u32::from(double) << 8;
    let registers = fields(double, d, 0, m);
    let kind = random.below(12);
    // The single-precision register of an integer operand or result.
    let (sd, sm) = (random.below(32), random.below(32));
    let fraction = 1 + random.below(32);
    let raw = match kind {
        0 => 0xeeb0_0ac0 | size | registers,
        1 => 0xeeb1_0a40 | size | registers,
        2 => 0xeeb1_0ac0 | size | registers,
        3 => 0xeeb0_0a40 | size | registers,
        // VCMP and VCMPE with a register, then with zero.
        4 | 5 => 0xeeb4_0a40 | (kind & 1) << 7 | size | registers,
        6 | 7 => 0xeeb5_0a40 | (kind & 1) << 7 | size | fields(double, d, 0, 0),
        // To an integer: unsigned or signed, with the mode of the FPSCR or to zero.
        8 => {
            0xeebc_0a40
                | random.below(2) << 16
                | random.below(2) << 7
                | size
                | single_field(sd, 12, 22)
                | fields(double, 0, 0, m)
        }
        // From an integer: unsigned or signed.
        9 => 0xeeb8_0a40 | random.below(2) << 7 | size | fields(double, d, 0, 0) | single_field(sm, 0, 5),
        // Between the two sizes: `size` is that of the operand.
        10 => {
            0xeeb7_0ac0
                | size
                | if double {
                    single_field(sd, 12, 22) | fields(true, 0, 0, m)
                } else {
                    fields(true, d, 0, 0) | single_field(m, 0, 5)
                }
        }
        // Between a number and a fixed-point number of 32 bits in the same register.
        _ => {
            let immediate = 32 - fraction;
            0xeeba_0ac0
                | random.below(2) << 18
                | random.below(2) << 16
                | size
                | fields(double, d, 0, 0)
                | (immediate & 1) << 5
                | immediate >> 1
        }
    };
    (
        raw,
        Box::new(move |state| {
            let (a, b) = (state.get(double, d), state.get(double, m));
            let unsigned_or_quiet = raw >> 16 & 1 == 0;
            let bit7 = raw >> 7 & 1 != 0;
            match kind {
                0..=3 => {
                    let (result, _) = match (kind, double) {
                        (0, true) => host!(state, b, 0u64, 0u64, "fabs d0, d0"),
                        (0, false) => host!(state, b, 0u64, 0u64, "fabs s0, s0"),
                        (1, true) => host!(state, b, 0u64, 0u64, "fneg d0, d0"),
                        (1, false) => host!(state, b, 0u64, 0u64, "fneg s0, s0"),
                        (2, true) => host!(state, b, 0u64, 0u64, "fsqrt d0, d0"),
                        (2, false) => host!(state, b, 0u64, 0u64, "fsqrt s0, s0"),
                        _ => (b, 0),
                    };
                    state.set(double, d, result);
                }
                4..=7 => {
                    let other = if kind >= 6 { 0 } else { b };
                    let (_, nzcv) = match (bit7, double) {
                        (false, true) => host!(state, a, other, 0u64, "fcmp d0, d1"),
                        (false, false) => host!(state, a, other, 0u64, "fcmp s0, s1"),
                        (true, true) => host!(state, a, other, 0u64, "fcmpe d0, d1"),
                        (true, false) => host!(state, a, other, 0u64, "fcmpe s0, s1"),
                    };
                    state.fpscr = state.fpscr & 0x0fff_ffff | nzcv;
                }
                8 => {
                    // With the mode of the FPSCR: FRINTX rounds so, and the conversion
                    // of its result is exact.
                    let before = state.fpscr;
                    state.fpscr &= !FLAGS;
                    let (result, _) = match (bit7, unsigned_or_quiet, double) {
                        (true, true, true) => host!(state, b, 0u64, 0u64, "fcvtzu w9, d0", "fmov s0, w9"),
                        (true, true, false) => host!(state, b, 0u64, 0u64, "fcvtzu w9, s0", "fmov s0, w9"),
                        (true, false, true) => host!(state, b, 0u64, 0u64, "fcvtzs w9, d0", "fmov s0, w9"),
                        (true, false, false) => host!(state, b, 0u64, 0u64, "fcvtzs w9, s0", "fmov s0, w9"),
                        (false, true, true) => {
                            host!(state, b, 0u64, 0u64, "frintx d0, d0", "fcvtzu w9, d0", "fmov s0, w9")
                        }
                        (false, true, false) => {
                            host!(state, b, 0u64, 0u64, "frintx s0, s0", "fcvtzu w9, s0", "fmov s0, w9")
                        }
                        (false, false, true) => {
                            host!(state, b, 0u64, 0u64, "frintx d0, d0", "fcvtzs w9, d0", "fmov s0, w9")
                        }
                        (false, false, false) => {
                            host!(state, b, 0u64, 0u64, "frintx s0, s0", "fcvtzs w9, s0", "fmov s0, w9")
                        }
                    };
                    // A result out of the range is an invalid operation and not an
                    // inexact one, but FRINTX saw an inexact one before.
                    if state.fpscr & 1 != 0 {
                        state.fpscr &= !0x10;
                    }
                    state.fpscr |= before & FLAGS;
                    state.set_s(sd, result);
                }
                9 => {
                    let integer = state.s(sm);
                    let (result, _) = match (bit7, double) {
                        (true, true) => host!(state, integer, 0u64, 0u64, "fmov w9, s0", "scvtf d0, w9"),
                        (true, false) => host!(state, integer, 0u64, 0u64, "fmov w9, s0", "scvtf s0, w9"),
                        (false, true) => host!(state, integer, 0u64, 0u64, "fmov w9, s0", "ucvtf d0, w9"),
                        (false, false) => host!(state, integer, 0u64, 0u64, "fmov w9, s0", "ucvtf s0, w9"),
                    };
                    state.set(double, d, result);
                }
                10 => {
                    if double {
                        let (result, _) = host!(state, b, 0u64, 0u64, "fcvt s0, d0");
                        state.set_s(sd, result);
                    } else {
                        let (result, _) = host!(state, b, 0u64, 0u64, "fcvt d0, s0");
                        state.d[d as usize] = result;
                    }
                }
                _ => {
                    let to_fixed = raw >> 18 & 1 != 0;
                    let signed = raw >> 16 & 1 == 0;
                    let before = state.fpscr;
                    state.fpscr &= !FLAGS;
                    if !to_fixed {
                        // To a number: round to nearest, with each mode of the FPSCR.
                        state.fpscr &= !0x00c0_0000;
                    }
                    // The host has one instruction for each count of fraction bits. A
                    // product with a power of two is exact, thus it gives the same
                    // result with one rounding.
                    let up = (f64::from(fraction)).exp2().to_bits();
                    let down = (-f64::from(fraction)).exp2().to_bits();
                    let down_single = u64::from((-(fraction as f32)).exp2().to_bits());
                    let (result, _) = match (to_fixed, signed, double) {
                        (false, true, true) => {
                            host!(state, a, down, 0u64, "fmov w9, s0", "scvtf d0, w9", "fmul d0, d0, d1")
                        }
                        (false, false, true) => {
                            host!(state, a, down, 0u64, "fmov w9, s0", "ucvtf d0, w9", "fmul d0, d0, d1")
                        }
                        (false, true, false) => {
                            host!(state, a, down_single, 0u64, "fmov w9, s0", "scvtf s0, w9", "fmul s0, s0, s1")
                        }
                        (false, false, false) => {
                            host!(state, a, down_single, 0u64, "fmov w9, s0", "ucvtf s0, w9", "fmul s0, s0, s1")
                        }
                        (true, true, true) => {
                            host!(state, a, up, 0u64, "fmul d0, d0, d1", "fcvtzs w9, d0", "fmov s0, w9")
                        }
                        (true, false, true) => {
                            host!(state, a, up, 0u64, "fmul d0, d0, d1", "fcvtzu w9, d0", "fmov s0, w9")
                        }
                        (true, true, false) => host!(
                            state, a, up, 0u64, "fcvt d0, s0", "fmul d0, d0, d1", "fcvtzs w9, d0", "fmov s0, w9"
                        ),
                        (true, false, false) => host!(
                            state, a, up, 0u64, "fcvt d0, s0", "fmul d0, d0, d1", "fcvtzu w9, d0", "fmov s0, w9"
                        ),
                    };
                    // A product out of the range of the host is an invalid operation
                    // of the conversion, and no overflow.
                    if state.fpscr & 1 != 0 {
                        state.fpscr &= !0x14;
                    }
                    state.fpscr |= before & (FLAGS | 0x00c0_0000);
                    let result = match (to_fixed, double, signed) {
                        (true, true, true) => result as u32 as i32 as i64 as u64,
                        _ => result,
                    };
                    state.set(double, d, result);
                }
            }
        }),
    )
}

const SINGLES: [u32; 14] = [
    0,
    0x8000_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7fa0_0001,
    0xffc0_1234,
    0x0000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x7f7f_ffff,
    0x4f00_0000,
];

const DOUBLES: [u64; 14] = [
    0,
    0x8000_0000_0000_0000,
    0x3ff0_0000_0000_0000,
    0xbff0_0000_0000_0000,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0000,
    0x7ff4_0000_0000_0001,
    0xfff8_0000_0000_1234,
    0x0000_0000_0000_0001,
    0x000f_ffff_ffff_ffff,
    0x0010_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
    0x41e0_0000_0000_0000,
];

fn random_single(random: &mut Random) -> u32 {
    match random.below(8) {
        0 | 1 => SINGLES[random.below(SINGLES.len() as u32) as usize],
        2 => (random.next() as i32 >> random.below(32)) as f32 as u32,
        3 => ((random.next() as i32 >> random.below(32)) as f32).to_bits(),
        // Numbers near each other: sums and differences with few bits.
        4 => (1.0 + random.below(64) as f32 / 16.0).to_bits() | random.below(2) << 31,
        // Small and large exponents: underflows and overflows.
        5 => random.next() & 0x807f_ffff | random.below(40) << 23,
        6 => random.next() & 0x807f_ffff | (215 + random.below(40)) << 23,
        _ => random.next(),
    }
}

fn random_double(random: &mut Random) -> u64 {
    let bits = u64::from(random.next()) << 32 | u64::from(random.next());
    match random.below(8) {
        0 | 1 => DOUBLES[random.below(DOUBLES.len() as u32) as usize],
        2 => f64::from(random.next() as i32 >> random.below(32)).to_bits(),
        3 => f64::from(random_single(random) as f32).to_bits(),
        4 => (1.0 + f64::from(random.below(64)) / 16.0).to_bits() | u64::from(random.below(2)) << 63,
        5 => bits & 0x800f_ffff_ffff_ffff | u64::from(random.below(80)) << 52,
        6 => bits & 0x800f_ffff_ffff_ffff | u64::from(1967 + random.below(80)) << 52,
        _ => bits,
    }
}

fn check(trio: &mut Trio, random: &mut Random) -> Result<(), String> {
    let (raw, apply) = if random.below(2) == 0 {
        arithmetic(random)
    } else {
        one_operand(random)
    };
    let thumb = random.below(2) == 0;
    let mut case = Case::new(random, thumb);
    case.cpu = random_cpu(random);
    for register in &mut case.cpu.vfp.dregs {
        *register = if random.below(2) == 0 {
            random_double(random)
        } else {
            u64::from(random_single(random)) << 32 | u64::from(random_single(random))
        };
    }
    // Most cases have round to nearest, as most code has.
    let modes = if random.below(2) == 0 { 0 } else { random.next() & MODES };
    case.cpu.vfp.fpscr = Fpscr::from_bits(modes | random.next() & (0xf000_0000 | FLAGS));
    // The instruction, then a branch back to it: the second run of the instruction
    // is one of the fast interpreter and of the compiled code.
    let at = CODE + random.below(64) * 4;
    let program = Program {
        thumb,
        at,
        insns: vec![
            wide(raw),
            if thumb { super::suite::t16(0xe7fc) } else { wide(0xeaff_fffd) },
        ],
    };
    case.load(&program);
    case.enter(&program);
    case.until = 3;
    let mut state = State {
        d: case.cpu.vfp.dregs,
        fpscr: case.cpu.vfp.fpscr.bits(),
    };
    apply(&mut state);
    apply(&mut state);
    let outcome = trio.run(&case)?;
    let engines = State {
        d: outcome.dregs,
        fpscr: outcome.fpscr,
    };
    if engines == state && outcome.stop.is_none() && outcome.regs[15] == at + 4 {
        return Ok(());
    }
    let different: Vec<_> = (0..32)
        .filter(|&index| engines.d[index] != state.d[index])
        .map(|index| format!("d{index}: {:016x}, the host {:016x}", engines.d[index], state.d[index]))
        .collect();
    Err(format!(
        "{:?} {:08x} : {} {raw:08x} from fpscr {:#x} dregs {:x?}\n  the engines: fpscr {:#x} {:?}\n  the host: fpscr {:#x}\n  {different:?}",
        rax::arm::Decoder::new(rax::arm::ExecutionState::Aarch32)
            .decode(&(raw & 0x0fff_ffff | 0xe000_0000).to_le_bytes())
            .map(|insn| insn.mnemonic),
        raw & 0xffb0_0fd0,
        if thumb { "T32" } else { "A32" },
        case.cpu.vfp.fpscr.bits(),
        case.cpu.vfp.dregs,
        engines.fpscr,
        outcome.stop,
        state.fpscr,
    ))
}

#[test]
fn vfp_instructions_give_the_results_of_the_host() {
    let mut trio = Trio::new();
    cases(
        "vfp_instructions_give_the_results_of_the_host",
        0x0a11_ce5e_ed00_0007,
        100_000,
        |random| check(&mut trio, random),
    );
}
