//! The cases of the faults that these tests found in the executor, each with the
//! result that the ARM architecture manual gives.

use super::Random;
use super::machine::{Outcome, Stop};
use super::suite::{CODE, Case, Insn, Program, Trio, t16, wide};

pub fn run(trio: &mut Trio, case: &Case) -> Outcome {
    trio.run(case).unwrap_or_else(|difference| panic!("{difference}"))
}

pub fn thumb_program(insns: Vec<Insn>) -> Program {
    Program {
        thumb: true,
        at: CODE + 0x100,
        insns,
    }
}

#[test]
fn t32_smlal_of_halves_adds_the_product_to_the_two_registers() {
    let mut trio = Trio::new();
    // SMLALTT r12, r8, r5, r9; SMLALBB r1, r2, r3, r4.
    let program = thumb_program(vec![wide(0xfbc5_c8b9), wide(0xfbc3_1284)]);
    let mut case = Case::new(&mut Random(15), true);
    let regs = &mut case.cpu.regs;
    (regs[5], regs[9], regs[8], regs[12]) = (0x7fff_1234, 0x8000_5678, 1, 0xffff_ffff);
    (regs[3], regs[4], regs[2], regs[1]) = (0x1234_0003, 0x5678_fffe, 0, 5);
    case.load(&program);
    case.enter(&program);
    case.until = 2;
    let outcome = run(&mut trio, &case);
    // 0x1_ffff_ffff - 0x7fff * 0x8000, and 5 - 3 * 2.
    assert_eq!((outcome.regs[8], outcome.regs[12]), (1, 0xc000_7fff));
    assert_eq!((outcome.regs[2], outcome.regs[1]), (0xffff_ffff, 0xffff_ffff));
    assert_eq!(outcome.stop, None);
}

#[test]
fn bkpt_runs_in_an_it_block_with_a_false_condition() {
    let mut trio = Trio::new();
    // IT EQ; BKPT; MOVS r1, #1.
    let program = thumb_program(vec![t16(0xbf08), t16(0xbe00), t16(0x2101)]);
    let mut case = Case::new(&mut Random(13), true);
    case.cpu.regs[1] = 0;
    case.cpu.cpsr.z = false;
    case.load(&program);
    case.enter(&program);
    case.until = 4;
    let outcome = run(&mut trio, &case);
    assert_eq!(outcome.stop, Some(Stop::Exception("Breakpoint(0)".into())));
    assert_eq!((outcome.regs[1], outcome.regs[15]), (0, program.at + 2));
}
