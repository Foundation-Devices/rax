//! The engine with programs of many blocks: loops, calls, events at each place, and
//! code that the program changes.

use super::machine::{Engine, Kind, Machine};
use super::programs::{
    CALLED, LOG, Options, VECTORS, enter, install_vectors, main_program, subroutine,
};
use super::suite::{
    CALL_PAGE, CODE, Case, DATA, Program, Trio, cases, ram_page, random_cpu, random_periods, wide,
};

#[derive(Clone, Copy, PartialEq)]
enum Programs {
    Plain,
    /// With stores to their own instructions.
    Patches,
    /// With vectors, exceptions, interrupt requests at the events and mode changes.
    System,
}

fn random_program_case(random: &mut super::Random, kind: Programs) -> Case {
    let thumb = random.below(2) == 0;
    let mut case = Case::new(random, thumb);
    case.only_starts = false;
    case.cpu = random_cpu(random);
    case.eager = random.below(4) != 0;
    if kind != Programs::Plain {
        case.ram.kinds[ram_page(CODE)] = Kind::Direct;
        case.ram.kinds[ram_page(CALLED)] = Kind::Direct;
    }
    let mut options = Options {
        calls: Vec::new(),
        patches: kind != Programs::Plain,
        system: kind == Programs::System,
    };
    let mut at = CALLED + 0x400 + random.below(64) * 4;
    for _ in 0..random.below(4) {
        let sub_thumb = if random.below(4) == 0 { !thumb } else { thumb };
        let program = subroutine(random, sub_thumb, at);
        case.fill_code(program.end(), 16, sub_thumb);
        case.load(&program);
        options.calls.push((at, sub_thumb));
        at = (program.end() + 16 + random.below(8) * 4) & !3;
    }
    // Some programs start near the end of the first page and go on in the second one.
    let at = CODE
        + match random.below(4) {
            0 => 0xf00 + random.below(60) * 4,
            _ => random.below(0x200) * 4,
        };
    let (program, site) = main_program(random, thumb, at, &options);
    case.load(&program);
    enter(&mut case, random, &program, site);
    case.periods = random_periods(random);
    case.until = 100 + u64::from(random.below(2500));
    if kind == Programs::System {
        // Some logs are in the page of the handlers: each write removes their code.
        let log = if random.below(8) == 0 {
            VECTORS + 0x400
        } else {
            LOG
        };
        // Near the end of its page: the accesses of the program after it give aborts.
        case.cpu.regs[8] = CALL_PAGE + 0xf80 + random.below(32) * 4;
        install_vectors(&mut case, log);
        case.take_exceptions = true;
        case.irq_events = random.below(4) != 0;
        case.cpu.cpsr.i = random.below(8) == 0;
        if random.below(3) == 0 {
            // User mode: the banks have the registers of the mode before.
            let cpu = &mut case.cpu;
            cpu.regs_svc = [cpu.regs_svc[0], cpu.regs[14]];
            cpu.cpsr.mode = 0x10;
        }
    }
    case
}

#[test]
fn random_programs_run_the_same_with_each_engine() {
    let mut trio = Trio::new();
    cases(
        "random_programs_run_the_same_with_each_engine",
        0x0a11_ce5e_ed00_0001,
        6_000,
        |random| {
            let case = random_program_case(random, Programs::Plain);
            trio.run(&case).map(|_| ())
        },
    );
    eprintln!(
        "{:?} {:?} {:?}",
        trio.ends,
        trio.compiled_share(),
        trio.totals
    );
    assert!(trio.compiled_share().is_none_or(|share| share > 0.3));
}

#[test]
fn programs_that_change_their_code_run_the_same_with_each_engine() {
    let mut trio = Trio::new();
    cases(
        "programs_that_change_their_code_run_the_same_with_each_engine",
        0x0a11_ce5e_ed00_0002,
        6_000,
        |random| {
            let case = random_program_case(random, Programs::Patches);
            trio.run(&case).map(|_| ())
        },
    );
    eprintln!(
        "{:?} {:?} {:?}",
        trio.ends,
        trio.compiled_share(),
        trio.totals
    );
}

#[test]
fn programs_with_exceptions_and_interrupts_run_the_same_with_each_engine() {
    let mut trio = Trio::new();
    cases(
        "programs_with_exceptions_and_interrupts_run_the_same_with_each_engine",
        0x0a11_ce5e_ed00_0003,
        6_000,
        |random| {
            let case = random_program_case(random, Programs::System);
            trio.run(&case).map(|_| ())
        },
    );
    eprintln!(
        "{:?} {:?} {:?}",
        trio.ends,
        trio.compiled_share(),
        trio.totals
    );
}

#[test]
fn an_event_at_each_place_of_the_blocks_keeps_the_state_exact() {
    let mut trio = Trio::new();
    for seed in 1..=6u64 {
        for period in 1..=70 {
            let mut random = super::Random(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let kind = [Programs::Plain, Programs::Patches, Programs::System][seed as usize % 3];
            let mut case = random_program_case(&mut random, kind);
            case.periods = vec![period];
            case.eager = true;
            if let Err(difference) = trio.run(&case) {
                panic!("seed {seed}, period {period}: {difference}");
            }
        }
    }
}

/// A32 code for a loop of `r6` rounds: each round adds 1 to r0.
fn counting_loop(case: &mut Case) {
    // ADD r0, r0, #1; SUBS r6, r6, #1; BNE to the ADD.
    let insns = [0xe280_0001, 0xe256_6001, 0x1aff_fffc].map(wide).to_vec();
    let program = Program {
        thumb: false,
        at: CODE,
        insns,
    };
    case.load(&program);
    case.enter(&program);
}

#[test]
fn a_block_is_compiled_at_its_ninth_run() {
    let Some(mut machine) = Machine::new(Engine::Compiled) else {
        return;
    };
    let mut case = Case::new(&mut super::Random(1), false);
    counting_loop(&mut case);
    case.cpu.regs[6] = 20;
    machine.eager = false;
    machine.reset(&case.ram, &case.cpu);
    machine.run(1000);
    // 8 runs of the block with the interpreter, then one run of the code, which goes
    // back to its start for the 12 other rounds. The last run is for the stop.
    assert_eq!(machine.jit.as_ref().unwrap().stats.compiled_blocks, 1);
    let counts = machine.counts;
    assert_eq!(
        (
            counts.block_runs,
            counts.compiled_runs,
            counts.compiled_instructions
        ),
        (9, 1, 12 * 3)
    );
    assert_eq!((machine.instructions, machine.cpu.regs[0]), (60, 20));
}

#[test]
fn two_addresses_with_one_place_in_the_jump_table() {
    // The table of the jumps has 4096 places for the halfwords: addresses with a
    // distance of 0x2000 have the same place.
    const FAR: u32 = CODE + 0x2000;
    let mut trio = Trio::new();
    let mut case = Case::new(&mut super::Random(2), false);
    case.only_starts = false;
    case.fill_code(FAR, 0x1000, false);
    let branch =
        |at: u32, target: u32| 0xea00_0000 | (target.wrapping_sub(at + 8) >> 2 & 0x00ff_ffff);
    // MOV r6, #40; TST r6, #1; MOVNE r1, r4; MOVEQ r1, r5; BX r1.
    case.ram.write_words(
        CODE,
        &[
            0xe3a0_6028,
            0xe316_0001,
            0x11a0_1004,
            0x01a0_1005,
            0xe12f_ff11,
        ],
    );
    // ADD r0, r0, #1 at one address and ADD r0, r0, #256 at the other, then a branch
    // to SUBS r6, r6, #1; BNE to the TST; BKPT.
    case.ram.write_words(
        CODE + 0x200,
        &[0xe280_0001, branch(CODE + 0x204, CODE + 0x40)],
    );
    case.ram.write_words(
        FAR + 0x200,
        &[0xe280_0c01, branch(FAR + 0x204, CODE + 0x40)],
    );
    case.ram
        .write_words(CODE + 0x40, &[0xe256_6001, 0x1aff_ffee, 0xe120_0070]);
    case.cpu.regs[0] = 0;
    case.cpu.regs[4] = CODE + 0x200;
    case.cpu.regs[5] = FAR + 0x200;
    case.cpu.regs[15] = CODE;
    case.until = 1000;
    let outcome = trio
        .run(&case)
        .unwrap_or_else(|difference| panic!("{difference}"));
    assert_eq!(
        (outcome.regs[0], outcome.instructions),
        (20 * 0x101, 1 + 40 * 8)
    );
    assert!(trio.compiled_share().is_none_or(|share| share > 0.9));
}

#[test]
fn programs_run_the_same_when_the_code_buffer_gets_full() {
    let mut trio = Trio::new();
    // A block of 64 loads and stores has much code. Each reset of the machine
    // removes the code of the page, and the next run compiles the block again.
    let mut filler = Case::new(&mut super::Random(3), false);
    let insns = (0..64)
        .map(|index| wide(0xe584_0000 | (index & 1) << 20 | index << 12 & 0x3000 | (index * 4)))
        .collect();
    let program = Program {
        thumb: false,
        at: CODE,
        insns,
    };
    filler.load(&program);
    filler.enter(&program);
    filler.cpu.regs[4] = DATA + 0x1000;
    filler.until = 64;
    let Some(machine) = trio.compiled() else {
        return;
    };
    machine.eager = true;
    let first = {
        machine.reset(&filler.ram, &filler.cpu);
        machine.run(filler.until);
        machine.outcome()
    };
    // The buffer of the compiler has 64 MiB.
    const NEARLY_FULL: usize = (64 << 20) - (256 << 10);
    while machine.jit.as_ref().unwrap().stats.code_bytes < NEARLY_FULL {
        machine.reset(&filler.ram, &filler.cpu);
        machine.run(filler.until);
        assert_eq!(machine.counts.compiled_instructions, 64);
    }
    assert_eq!(first.difference(&machine.outcome()), None);
    assert_eq!(machine.jit.as_ref().unwrap().stats.flushes, 0);
    cases(
        "programs_run_the_same_when_the_code_buffer_gets_full",
        0x0a11_ce5e_ed00_0004,
        400,
        |random| {
            let case = random_program_case(random, Programs::Patches);
            trio.run(&case).map(|_| ())
        },
    );
    let machine = trio.compiled().unwrap();
    assert!(machine.jit.as_ref().unwrap().stats.flushes >= 1);
    machine.periods.clear();
    machine.reset(&filler.ram, &filler.cpu);
    machine.run(filler.until);
    assert_eq!(first.difference(&machine.outcome()), None);
}

#[test]
#[ignore = "shows the order of calls that runs an old instruction after a code write"]
fn machine_that_keeps_the_left_block_after_a_code_write() {
    let mut trio = Trio::new();
    if let Some(machine) = trio.compiled() {
        machine.keep_left_after_code_write = true;
    }
    cases(
        "machine_that_keeps_the_left_block_after_a_code_write",
        0x0a11_ce5e_ed00_0002,
        6_000,
        |random| {
            let case = random_program_case(random, Programs::Patches);
            trio.run(&case).map(|_| ())
        },
    );
}
