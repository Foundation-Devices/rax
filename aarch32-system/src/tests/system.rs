//! Conditions, IT blocks, exception entries and returns, and the registers of each
//! mode, against the rules of the ARM architecture manual.

use rax::arm::Psr;

use super::Random;
use super::corrections::{arm_program, run, thumb_program, vectors_that_stop};
use super::machine::Stop;
use super::programs::VECTORS;
use super::suite::{CODE, Case, DATA, FAULT_PAGE, Program, Trio, cases, random_cpu, t16, wide};

/// ConditionPassed, from the flags in the bits 3:0 as N, Z, C, V.
fn condition_passed(cond: u32, flags: u32) -> bool {
    let (n, z, c, v) = (
        flags & 8 != 0,
        flags & 4 != 0,
        flags & 2 != 0,
        flags & 1 != 0,
    );
    let result = match cond >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    if cond & 1 != 0 && cond != 15 {
        !result
    } else {
        result
    }
}

fn set_flags(case: &mut Case, flags: u32) {
    let cpsr = &mut case.cpu.cpsr;
    (cpsr.n, cpsr.z, cpsr.c, cpsr.v) = (
        flags & 8 != 0,
        flags & 4 != 0,
        flags & 2 != 0,
        flags & 1 != 0,
    );
}

#[test]
fn each_condition_with_each_value_of_the_flags() {
    let mut trio = Trio::new();
    let mut random = Random(11);
    for cond in 0..15 {
        for flags in 0..16 {
            let passed = u32::from(condition_passed(cond, flags));
            // A32: MOV r0, #1 (compiled), MRS r1, apsr (the executor), a branch over
            // MOV r2, #1, each with the condition; then two instructions with none.
            let program = arm_program(&[
                cond << 28 | 0x03a0_0001,
                cond << 28 | 0x010f_1000,
                cond << 28 | 0x0a00_0000,
                0xe3a0_2001,
                0xe3a0_3001,
            ]);
            let mut case = Case::new(&mut random, false);
            case.cpu.regs[..4].fill(0);
            set_flags(&mut case, flags);
            case.load(&program);
            case.enter(&program);
            case.until = 8;
            let outcome = run(&mut trio, &case);
            let apsr = if passed == 1 { flags << 28 | 0xd3 } else { 0 };
            assert_eq!(
                outcome.regs[..4],
                [passed, apsr, 1 - passed, 1],
                "A32 {cond} {flags:#x}"
            );
            assert_eq!(outcome.cpsr >> 28, flags);

            // Thumb: a branch with the condition over MOVS r2, #1, with 16 and 32 bits.
            if cond == 14 {
                continue;
            }
            for wide_branch in [false, true] {
                // The target is the instruction after the next one.
                let branch = if wide_branch {
                    wide(0xf000_8002 | cond << 22)
                } else {
                    t16(0xd001 | cond << 8)
                };
                // MOV.W r2, #1 and MOV.W r3, #1 keep the flags.
                let program = thumb_program(vec![branch, wide(0xf04f_0201), wide(0xf04f_0301)]);
                let mut case = Case::new(&mut random, true);
                case.cpu.regs[..4].fill(0);
                set_flags(&mut case, flags);
                case.load(&program);
                case.enter(&program);
                case.until = 8;
                let outcome = run(&mut trio, &case);
                assert_eq!(
                    outcome.regs[2..4],
                    [1 - passed, 1],
                    "Thumb {cond} {flags:#x}"
                );
            }
        }
    }
}

#[test]
fn it_blocks_of_each_shape_with_each_value_of_the_flags() {
    let mut trio = Trio::new();
    let mut random = Random(12);
    for first in 0..14 {
        for mask in 1..16u32 {
            let count = 4 - mask.trailing_zeros();
            for flags in 0..16 {
                // IT, then ADD.W of 1 to r1, r2, r3 and r4, which keeps the flags, then
                // an instruction with no condition. The second one of some blocks is
                // LDR r2, [r5], which the engines run in an other way.
                let load = (first + mask + flags) % 3 == 0;
                let mut insns = vec![t16(0xbf00 | first << 4 | mask)];
                for register in 1..=4 {
                    insns.push(wide(if register == 2 && load {
                        0xf8d5_2000
                    } else {
                        0xf100_0001 | register << 16 | register << 8
                    }));
                }
                let program = thumb_program(insns);
                let mut case = Case::new(&mut random, true);
                case.cpu.regs[1..5].fill(0);
                case.cpu.regs[5] = DATA + 0x100;
                case.ram.write_words(DATA + 0x100, &[1]);
                set_flags(&mut case, flags);
                case.load(&program);
                case.enter(&program);
                case.until = 5;
                let outcome = run(&mut trio, &case);
                // The condition of the instruction `index`: the first one, with its low
                // bit from the mask for the next ones.
                let expected: Vec<u32> = (0..4)
                    .map(|index| {
                        let cond = match index {
                            0 => first,
                            _ if index < count => first & !1 | mask >> (4 - index) & 1,
                            _ => 14,
                        };
                        u32::from(condition_passed(cond, flags))
                    })
                    .collect();
                assert_eq!(
                    outcome.regs[1..5],
                    expected,
                    "IT {first} {mask:#x} flags {flags:#x}"
                );
                assert_eq!((outcome.cpsr >> 28, outcome.stop), (flags, None));
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    Undefined,
    Svc,
    Breakpoint,
    DataAbort,
    PrefetchAbort,
    Irq,
}

/// The index of the SPSR of a mode in `Outcome::banked`.
fn spsr_index(mode: u32) -> usize {
    22 + match mode {
        0x11 => 0,
        0x12 => 1,
        0x13 => 2,
        0x17 => 3,
        _ => 4,
    }
}

fn exception_entry(trio: &mut Trio, random: &mut Random) -> Result<(), String> {
    let entry = [
        Entry::Undefined,
        Entry::Svc,
        Entry::Breakpoint,
        Entry::DataAbort,
        Entry::PrefetchAbort,
        Entry::Irq,
    ][random.below(6) as usize];
    let thumb = random.below(2) == 0;
    let wide_form = random.below(2) == 0;
    // In Thumb code, some instructions are in an IT block with a true condition.
    let in_it = thumb && random.below(2) == 0;
    let mut case = Case::new(random, thumb);
    case.only_starts = false;
    case.cpu = random_cpu(random);
    vectors_that_stop(&mut case);
    let cpu = &mut case.cpu;
    cpu.cpsr.mode = [0x10, 0x13, 0x1f, 0x12][random.below(4) as usize];
    cpu.cpsr.i = false;
    cpu.cpsr.z = true;
    cpu.regs[1] = FAULT_PAGE + random.below(64) * 4;
    // The instruction of the exception: its bytes, and its length.
    let nop = if thumb {
        t16(0xbf00)
    } else {
        wide(0xe1a0_0000)
    };
    let cause = match (entry, thumb) {
        (Entry::Undefined, false) => wide(0xe7f0_00f0),
        (Entry::Undefined, true) if wide_form => wide(0xf7f0_a000),
        (Entry::Undefined, true) => t16(0xde00),
        (Entry::Svc, false) => wide(0xef00_0000 | random.below(256)),
        (Entry::Svc, true) => t16(0xdf00 | random.below(256)),
        (Entry::Breakpoint, false) => wide(0xe120_0070),
        (Entry::Breakpoint, true) => t16(0xbe00),
        // LDR r0, [r1] in the page with no memory.
        (Entry::DataAbort, false) => wide(0xe591_0000),
        (Entry::DataAbort, true) if wide_form => wide(0xf8d1_0000),
        (Entry::DataAbort, true) => t16(0x6808),
        // BX r1, to the page with no memory.
        (Entry::PrefetchAbort, false) => wide(0xe12f_ff11),
        (Entry::PrefetchAbort, true) => t16(0x4708),
        (Entry::Irq, _) => nop,
    };
    let target_thumb = random.below(2);
    if entry == Entry::PrefetchAbort {
        cpu.regs[1] |= target_thumb;
    }
    let before = random.below(3);
    let mut insns = vec![nop; before as usize];
    if in_it {
        // IT EQ.
        insns.push(t16(0xbf08));
    }
    let cause_address = CODE + 0x100 + insns.iter().map(|insn| u32::from(insn.size)).sum::<u32>();
    insns.push(cause);
    insns.push(nop);
    let program = Program {
        thumb,
        at: CODE + 0x100,
        insns,
    };
    case.load(&program);
    case.enter(&program);
    case.until = 20;
    if entry == Entry::Irq {
        // The request comes before the instruction `cause`.
        case.irq_events = true;
        case.periods = vec![u64::from(before) + u64::from(in_it), 1000];
    }
    let old = case.cpu.cpsr.clone();
    let outcome = trio.run(&case)?;

    // The IT bits when the exception comes: those of the instruction in the block.
    let it_state = if in_it { 0x08 } else { 0 };
    let (mode, vector, link, saved_it, masks_aborts) = match entry {
        Entry::Undefined => (
            0x1b,
            0x04,
            cause_address + if thumb { 2 } else { 4 },
            it_state,
            false,
        ),
        // The saved IT bits of an SVC are those of the next instruction.
        Entry::Svc => (0x13, 0x08, cause_address + u32::from(cause.size), 0, false),
        Entry::Breakpoint => (0x17, 0x0c, cause_address + 4, it_state, true),
        Entry::DataAbort => (0x17, 0x10, cause_address + 8, it_state, true),
        Entry::PrefetchAbort => (0x17, 0x0c, (case.cpu.regs[1] & !1) + 4, 0, true),
        Entry::Irq => (0x12, 0x18, cause_address + 4, it_state, true),
    };
    let mut saved = old.clone();
    saved.it_state = saved_it;
    if entry == Entry::PrefetchAbort {
        saved.t = target_thumb == 1;
    }
    let mut cpsr = Psr {
        mode,
        t: false,
        it_state: 0,
        i: true,
        e: false,
        ..old.clone()
    };
    cpsr.a |= masks_aborts;
    let expected = (
        cpsr.to_u32(),
        link,
        saved.to_u32(),
        VECTORS + vector + 4,
        Some(Stop::Halt),
    );
    let engines = (
        outcome.cpsr,
        outcome.regs[14],
        outcome.banked[spsr_index(mode.into())],
        outcome.regs[15],
        outcome.stop.clone(),
    );
    if engines == expected && outcome.exceptions.len() == 1 {
        return Ok(());
    }
    Err(format!(
        "{entry:?} : thumb {thumb} wide {wide_form} in IT {in_it} from mode {:#x}: cpsr, lr, spsr, pc\n  the engines: {engines:x?} {:?}\n  the manual:  {expected:x?}",
        old.mode, outcome.exceptions
    ))
}

#[test]
fn exception_entries_have_the_state_of_the_architecture() {
    let mut trio = Trio::new();
    cases(
        "exception_entries_have_the_state_of_the_architecture",
        0x0a11_ce5e_ed00_0009,
        4_000,
        |random| exception_entry(&mut trio, random),
    );
}

const MODES: [u32; 7] = [0x10, 0x11, 0x12, 0x13, 0x17, 0x1b, 0x1f];

/// The number of the bank of r13 and r14 of a mode: User and System have one bank.
fn bank(mode: u32) -> usize {
    match mode {
        0x11 => 1,
        0x12 => 2,
        0x13 => 3,
        0x17 => 4,
        0x1b => 5,
        _ => 0,
    }
}

/// A CPU in `mode` with these values of r13 and r14 for each bank, and these values of
/// r8 to r12 for FIQ mode and for the other modes.
fn banked_cpu(case: &mut Case, mode: u32, banks: &[[u32; 2]; 6], high: &[[u32; 5]; 2]) {
    let cpu = &mut case.cpu;
    cpu.cpsr.mode = mode as u8;
    cpu.regs_usr = banks[0];
    cpu.regs_fiq[5..].copy_from_slice(&banks[1]);
    cpu.regs_irq = banks[2];
    cpu.regs_svc = banks[3];
    cpu.regs_abt = banks[4];
    cpu.regs_und = banks[5];
    cpu.regs[13..15].copy_from_slice(&banks[bank(mode)]);
    cpu.regs_usr_high = high[0];
    cpu.regs_fiq[..5].copy_from_slice(&high[1]);
    cpu.regs[8..13].copy_from_slice(&high[usize::from(mode == 0x11)]);
}

fn mode_changes(trio: &mut Trio, random: &mut Random) -> Result<(), String> {
    let thumb = random.below(2) == 0;
    let start = MODES[1 + random.below(6) as usize];
    let (second, third) = (
        MODES[random.below(7) as usize],
        MODES[1 + random.below(6) as usize],
    );
    let mut banks = [[0; 2]; 6];
    let mut high = [[0; 5]; 2];
    banks.as_flattened_mut().fill_with(|| random.next());
    high.as_flattened_mut().fill_with(|| random.next());
    let mut case = Case::new(random, thumb);
    case.cpu = random_cpu(random);
    banked_cpu(&mut case, start, &banks, &high);
    let new = [random.next(), random.next()];
    case.cpu.regs[2..4].copy_from_slice(&new);
    // CPS to a mode, or MSR of the mode from r4 or r5.
    case.cpu.regs[4] = case.cpu.cpsr.to_u32() & !0x1f | second;
    case.cpu.regs[5] = case.cpu.cpsr.to_u32() & !0x1f | third;
    let with_msr = random.below(2) == 0;
    let change = |mode: u32, register: u32| match (thumb, with_msr) {
        (false, false) => wide(0xf102_0000 | mode),
        (false, true) => wide(0xe121_f000 | register),
        (true, false) => wide(0xf3af_8100 | mode),
        (true, true) => wide(0xf380_8100 | register << 16),
    };
    // MOV between a low register and r13 or r14.
    let read = |low: u32, register: u32| {
        if thumb {
            t16(0x4600 | register << 3 | low)
        } else {
            wide(0xe1a0_0000 | low << 12 | register)
        }
    };
    let write = |register: u32, low: u32| {
        if thumb {
            t16(0x4680 | low << 3 | (register & 7))
        } else {
            wide(0xe1a0_0000 | register << 12 | low)
        }
    };
    let program = Program {
        thumb,
        at: CODE + 0x100,
        insns: vec![
            change(second, 4),
            read(0, 13),
            read(1, 14),
            write(13, 2),
            write(14, 3),
            change(third, 5),
            read(6, 13),
            read(7, 14),
        ],
    };
    case.load(&program);
    case.enter(&program);
    case.until = 8;
    let outcome = trio.run(&case)?;

    // A CPS or an MSR of the mode in User mode changes nothing.
    let (second, third) = if second == 0x10 {
        (0x10, 0x10)
    } else {
        (second, third)
    };
    let mut expected = banks;
    let first_read = expected[bank(second)];
    expected[bank(second)] = new;
    let second_read = expected[bank(third)];
    let registers = [first_read[0], first_read[1], second_read[0], second_read[1]];
    let engines = [
        outcome.regs[0],
        outcome.regs[1],
        outcome.regs[6],
        outcome.regs[7],
    ];
    let high_now = high[usize::from(third == 0x11)];
    if engines == registers && outcome.regs[8..13] == high_now && outcome.cpsr & 0x1f == third {
        return Ok(());
    }
    Err(format!(
        "modes {start:#x}, {second:#x}, {third:#x} : thumb {thumb}, MSR {with_msr}: r13 and r14 of the two modes\n  the engines: {engines:x?} r8 to r12 {:x?} cpsr {:#x}\n  the manual:  {registers:x?} {high_now:x?}",
        &outcome.regs[8..13],
        outcome.cpsr
    ))
}

#[test]
fn each_mode_has_its_registers() {
    let mut trio = Trio::new();
    cases(
        "each_mode_has_its_registers",
        0x0a11_ce5e_ed00_000a,
        4_000,
        |random| mode_changes(&mut trio, random),
    );
}

/// MSR of the CPSR and of the SPSR with each field mask, MRS, and the returns from an
/// exception.
fn status_registers(trio: &mut Trio, random: &mut Random) -> Result<(), String> {
    let thumb = random.below(2) == 0;
    let mut case = Case::new(random, thumb);
    case.cpu = random_cpu(random);
    let mode = MODES[random.below(7) as usize];
    case.cpu.cpsr.mode = mode as u8;
    case.cpu.cpsr.i = random.below(2) == 0;
    case.cpu.cpsr.f = random.below(2) == 0;
    case.cpu.cpsr.a = random.below(2) == 0;
    case.cpu.cpsr.t = thumb;
    let old = case.cpu.cpsr.to_u32();
    let fields = 1 + random.below(15);
    // The value keeps the mode (the test of the modes has the changes) and the
    // order of the bytes of the data.
    let value = random.next() & !0x21f | mode;
    case.cpu.regs[0] = value;
    let spsr = random.next();
    let has_spsr = mode != 0x10 && mode != 0x1f;
    if has_spsr {
        *case.cpu.get_current_spsr_mut().unwrap() = Psr::from_u32(spsr);
    }
    let kind = random.below(if has_spsr { 4 } else { 2 });
    let insn = match (kind, thumb) {
        // MRS r1, CPSR.
        (0, false) => 0xe10f_1000,
        (0, true) => 0xf3ef_8100,
        // MSR CPSR_<fields>, r0.
        (1, false) => 0xe120_f000 | fields << 16,
        (1, true) => 0xf380_8000 | fields << 8,
        // MRS r1, SPSR.
        (2, false) => 0xe14f_1000,
        (2, true) => 0xf3ff_8100,
        // MSR SPSR_<fields>, r0.
        (_, false) => 0xe160_f000 | fields << 16,
        (_, true) => 0xf390_8000 | fields << 8,
    };
    let program = Program {
        thumb,
        at: CODE + 0x100,
        insns: vec![wide(insn)],
    };
    case.load(&program);
    case.enter(&program);
    case.until = 1;
    let outcome = trio.run(&case)?;

    // The bits of each field of the mask. ARMv7 has no bit at 23:20, and no engine
    // has Jazelle (bit 24).
    let field_mask = (0..4).fold(0, |mask, field| {
        mask | if fields >> field & 1 != 0 {
            0xff << (8 * field)
        } else {
            0
        }
    }) & !0x01f0_0000;
    let privileged = mode != 0x10;
    // An MSR of the CPSR writes the flags, Q and GE, E, and with a privilege A, I, F
    // and the mode: not the execution state bits.
    let writable = 0xf80f_0200 | if privileged { 0x1df } else { 0 };
    // In User mode, only the flags, Q and GE of an MRS have a defined value.
    let visible = if privileged { 0xf8ff_03df } else { 0xf80f_0000 };
    let (r1, cpsr, new_spsr) = match kind {
        // An MRS reads the execution state bits as zero.
        0 => (old & visible, old, spsr),
        1 => {
            let mask = field_mask & writable;
            (case.cpu.regs[1], old & !mask | value & mask, spsr)
        }
        2 => (Psr::from_u32(spsr).to_u32(), old, spsr),
        _ => (
            case.cpu.regs[1],
            old,
            spsr & !field_mask | value & field_mask,
        ),
    };
    let saved = if has_spsr {
        outcome.banked[spsr_index(mode)]
    } else {
        Psr::from_u32(new_spsr).to_u32()
    };
    let expected = (r1, cpsr, Psr::from_u32(new_spsr).to_u32());
    let read = if kind == 0 {
        outcome.regs[1] & visible
    } else {
        outcome.regs[1]
    };
    let engines = (read, outcome.cpsr, saved);
    if engines == expected && outcome.stop.is_none() {
        return Ok(());
    }
    Err(format!(
        "{insn:08x} : thumb {thumb} in mode {mode:#x}, r0 {value:#x}, cpsr {old:#x}, spsr {spsr:#x}: r1, cpsr, spsr\n  the engines: {engines:x?} {:?}\n  the manual:  {expected:x?}",
        outcome.stop
    ))
}

#[test]
fn status_registers_have_the_fields_of_the_architecture() {
    let mut trio = Trio::new();
    cases(
        "status_registers_have_the_fields_of_the_architecture",
        0x0a11_ce5e_ed00_000b,
        6_000,
        |random| status_registers(&mut trio, random),
    );
}

/// The returns from an exception: MOVS and SUBS to the PC, LDM with the PC and the
/// SPSR, and RFE. The CPSR is the saved one with its T and IT bits, and r13 and r14
/// are those of its mode.
fn exception_return(trio: &mut Trio, random: &mut Random, thumb: bool) -> Result<(), String> {
    let handler = MODES[1 + random.below(5) as usize];
    let target = MODES[random.below(7) as usize];
    let target_thumb = random.below(2) == 0;
    let mut banks = [[0; 2]; 6];
    let mut high = [[0; 5]; 2];
    banks.as_flattened_mut().fill_with(|| random.next());
    high.as_flattened_mut().fill_with(|| random.next());
    let stack = DATA + 0x400 + random.below(64) * 4;
    banks[bank(handler)][0] = stack;
    let address = CODE + 0x800 + random.below(64) * 4 + if target_thumb { 2 } else { 0 };
    let offset = random.below(3) * 4;
    banks[bank(handler)][1] = address + offset;
    let mut case = Case::new(random, thumb);
    case.cpu = random_cpu(random);
    banked_cpu(&mut case, handler, &banks, &high);
    // The saved CPSR: each flag and mask, and IT bits only for Thumb code.
    let it_state = if target_thumb {
        [0, 0x08, 0x1c, 0xa6][random.below(4) as usize]
    } else {
        0
    };
    let saved = Psr {
        mode: target as u8,
        t: target_thumb,
        it_state,
        e: false,
        ..Psr::from_u32(random.next() & 0xf80f_01c0)
    };
    *case.cpu.get_current_spsr_mut().unwrap() = saved.clone();
    // LDM and RFE take the PC, and RFE the CPSR, from the stack.
    case.ram.write_words(stack, &[address, saved.to_u32()]);
    let kind = random.below(if thumb { 2 } else { 4 });
    let insn = match (kind, thumb) {
        // SUBS pc, lr, #offset, and MOVS pc, lr for no offset.
        (0, false) if offset == 0 => 0xe1b0_f00e,
        (0, false) => 0xe25e_f000 | offset,
        (0, true) => 0xf3de_8f00 | offset,
        // RFEIA sp!.
        (1, false) => 0xf8bd_0a00,
        (1, true) => 0xe9bd_c000,
        // LDMIA sp!, {pc}^ and LDMIB sp, {pc}^.
        (2, false) => 0xe8fd_8000,
        (_, _) => 0xe9dd_8000,
    };
    if kind == 3 {
        case.ram.write_words(stack + 4, &[address]);
    }
    let program = Program {
        thumb,
        at: CODE + 0x100,
        insns: vec![wide(insn)],
    };
    case.load(&program);
    case.enter(&program);
    case.until = 1;
    let outcome = trio.run(&case)?;

    let mut expected = banks;
    // The stack pointer of the handler after RFE and LDM with a writeback.
    if kind == 1 || kind == 2 {
        expected[bank(handler)][0] = stack + if kind == 1 { 8 } else { 4 };
    }
    let registers = expected[bank(target)];
    let high_now = high[usize::from(target == 0x11)];
    let engines = (
        outcome.cpsr,
        outcome.regs[15],
        [outcome.regs[13], outcome.regs[14]],
    );
    let manual = (saved.to_u32(), address, registers);
    if engines == manual && outcome.regs[8..13] == high_now && outcome.stop.is_none() {
        return Ok(());
    }
    Err(format!(
        "{insn:08x} : thumb {thumb} from mode {handler:#x} to {target:#x}: cpsr, pc, r13 and r14\n  the engines: {engines:x?} {:?}\n  the manual:  {manual:x?}",
        outcome.stop
    ))
}

#[test]
fn exception_returns_restore_the_state_of_the_architecture() {
    let mut trio = Trio::new();
    cases(
        "exception_returns_restore_the_state_of_the_architecture",
        0x0a11_ce5e_ed00_000d,
        6_000,
        |random| exception_return(&mut trio, random, false),
    );
}

#[test]
fn t32_exception_returns_restore_the_state_of_the_architecture() {
    let mut trio = Trio::new();
    cases(
        "t32_exception_returns_restore_the_state_of_the_architecture",
        0x0a11_ce5e_ed00_000e,
        2_000,
        |random| exception_return(&mut trio, random, true),
    );
}
