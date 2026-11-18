//! The cases of the faults that these tests found in the executor, each with the
//! result that the ARM architecture manual gives.

use super::Random;
use super::machine::{Outcome, RAM_BASE, Stop};
use super::programs::VECTORS;
use super::suite::{CODE, Case, DATA, Insn, Program, Trio, t16, wide};

pub fn run(trio: &mut Trio, case: &Case) -> Outcome {
    trio.run(case)
        .unwrap_or_else(|difference| panic!("{difference}"))
}

pub fn thumb_program(insns: Vec<Insn>) -> Program {
    Program {
        thumb: true,
        at: CODE + 0x100,
        insns,
    }
}

pub fn arm_program(words: &[u32]) -> Program {
    Program {
        thumb: false,
        at: CODE + 0x100,
        insns: words.iter().copied().map(wide).collect(),
    }
}

/// WFI at each vector: the machine stops with the state of the entry.
pub fn vectors_that_stop(case: &mut Case) {
    for vector in (0..0x20).step_by(4) {
        case.ram.write_words(VECTORS + vector, &[0xe320_f003]);
    }
    case.cpu.cp15.vbar = VECTORS;
    case.take_exceptions = true;
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
    assert_eq!(
        (outcome.regs[2], outcome.regs[1]),
        (0xffff_ffff, 0xffff_ffff)
    );
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

#[test]
fn svc_in_an_it_block_returns_with_the_it_bits_of_the_next_instruction() {
    let mut trio = Trio::new();
    // ITE EQ; SVC 0; MOVS r2, #1 (not run: NE); MOVS r3, #1.
    let program = thumb_program(vec![
        t16(0xbf0c),
        t16(0xdf00),
        t16(0x2201),
        t16(0x2301),
        t16(0xbe00),
    ]);
    let mut case = Case::new(&mut Random(14), true);
    case.only_starts = false;
    vectors_that_stop(&mut case);
    // The SVC handler: MOVS pc, lr.
    case.ram.write_words(VECTORS + 8, &[0xe1b0_f00e]);
    case.cpu.regs[2..4].fill(0);
    case.cpu.cpsr.z = true;
    case.load(&program);
    case.enter(&program);
    case.until = 20;
    let outcome = run(&mut trio, &case);
    assert_eq!(outcome.regs[2..4], [0, 1]);
    assert_eq!(outcome.stop, Some(Stop::Halt));
}

#[test]
fn a32_register_offset_with_rrx_shifts_the_carry_in() {
    let mut trio = Trio::new();
    // STR r0, [r1, -r2, RRX]; LDR r3, [r4], r2, RRX.
    let program = arm_program(&[0xe701_0062, 0xe694_3062]);
    let mut case = Case::new(&mut Random(16), false);
    let regs = &mut case.cpu.regs;
    // The offset is 0x8000_0020 with the carry: the store is at DATA + 0x100.
    (regs[0], regs[2], regs[4]) = (0x1234_5678, 0x41, DATA + 0x200);
    regs[1] = (DATA + 0x100).wrapping_add(0x8000_0020);
    case.cpu.cpsr.c = true;
    case.ram.write_words(DATA + 0x200, &[0x9abc_def0]);
    case.load(&program);
    case.enter(&program);
    case.until = 2;
    let outcome = run(&mut trio, &case);
    let stored = (DATA + 0x100 - RAM_BASE) as usize;
    assert_eq!(
        outcome.ram[stored..stored + 4],
        0x1234_5678_u32.to_le_bytes()
    );
    assert_eq!(outcome.regs[3], 0x9abc_def0);
    assert_eq!(outcome.regs[4], (DATA + 0x200).wrapping_add(0x8000_0020));
    assert_eq!(outcome.stop, None);
}

#[test]
fn vhsub_of_unsigned_elements_keeps_the_sign_of_the_difference() {
    let mut trio = Trio::new();
    // VHSUB.U8 d0, d1, d2.
    let program = arm_program(&[0xf301_0202]);
    let mut case = Case::new(&mut Random(17), false);
    case.cpu.vfp.dregs[1] = 0x1000_0500_80ff_0100;
    case.cpu.vfp.dregs[2] = 0x2001_05ff_7f00_0201;
    case.load(&program);
    case.enter(&program);
    case.until = 1;
    let outcome = run(&mut trio, &case);
    assert_eq!(outcome.dregs[0], 0xf8ff_0080_007f_ffff);
    assert_eq!(outcome.stop, None);
}

#[test]
fn vqdmlal_saturates_the_doubled_product_before_the_sum() {
    let mut trio = Trio::new();
    // VQDMLAL.S16 q0, d2, d3.
    let program = arm_program(&[0xf292_0903]);
    let mut case = Case::new(&mut Random(18), false);
    let d = &mut case.cpu.vfp.dregs;
    // -1 + 2 * 0x8000 * 0x8000, and 5 + 2 * 3 * 2.
    (d[0], d[1], d[2], d[3]) = (0x0000_0005_ffff_ffff, 0, 0x0003_8000, 0x0002_8000);
    case.cpu.vfp.fpscr = rax::arm::vfp::Fpscr::from_bits(0);
    case.load(&program);
    case.enter(&program);
    case.until = 1;
    let outcome = run(&mut trio, &case);
    assert_eq!(outcome.dregs[..2], [0x0000_0011_7fff_fffe, 0]);
    // QC.
    assert_eq!(outcome.fpscr, 1 << 27);
    assert_eq!(outcome.stop, None);
}

#[test]
fn t32_srs_stores_lr_and_the_spsr_on_the_stack_of_its_mode() {
    let mut trio = Trio::new();
    // SRSDB sp!, #19 two times; SRSIA sp!, #31.
    let program = thumb_program(vec![
        wide(0xe82d_c013),
        wide(0xe82d_c013),
        wide(0xe9ad_c01f),
    ]);
    let mut case = Case::new(&mut Random(19), true);
    let (svc_stack, user_stack) = (DATA + 0x200, DATA + 0x300);
    let spsr = rax::arm::Psr::from_u32(0x6000_0030);
    case.cpu.cpsr.mode = 0x12;
    case.cpu.regs[14] = 0x1234_5679;
    case.cpu.spsr_irq = spsr.clone();
    case.cpu.regs_svc[0] = svc_stack;
    case.cpu.regs_usr[0] = user_stack;
    case.load(&program);
    case.enter(&program);
    case.until = 3;
    let outcome = run(&mut trio, &case);
    let saved: Vec<u8> = [0x1234_5679, spsr.to_u32()]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    for at in [svc_stack - 16, svc_stack - 8, user_stack] {
        let at = (at - RAM_BASE) as usize;
        assert_eq!(outcome.ram[at..at + 8], saved);
    }
    assert_eq!(outcome.stop, None);
}

#[test]
fn t32_exception_return_in_an_it_block_takes_the_it_state_of_the_spsr() {
    let mut trio = Trio::new();
    // IT EQ; SUBS pc, lr, #4.
    let program = thumb_program(vec![t16(0xbf08), wide(0xf3de_8f04)]);
    let mut case = Case::new(&mut Random(20), true);
    // Thumb, System mode, the IT state of ITTE with one instruction done.
    let spsr = rax::arm::Psr::from_u32(0x2600_a03f);
    case.cpu.cpsr.mode = 0x12;
    case.cpu.cpsr.z = true;
    case.cpu.regs[14] = CODE + 0x806;
    case.cpu.spsr_irq = spsr.clone();
    case.load(&program);
    case.enter(&program);
    case.until = 2;
    let outcome = run(&mut trio, &case);
    assert_eq!(
        (outcome.cpsr, outcome.regs[15]),
        (spsr.to_u32(), CODE + 0x802)
    );
    assert_eq!(outcome.stop, None);
}
