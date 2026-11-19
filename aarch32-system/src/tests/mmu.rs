//! Programs that change the translation while they run: two address spaces with
//! other code at the same address, a new entry of a translation table with a TLB
//! flush or an invalidation of the page, a second address of the same code, and pages
//! with no write permission.

use std::collections::HashMap;

use rax::arm::cp15::Sctlr;
use rax::arm::mmu_v6::{V6Access, V6MmuConfig};

use super::Random;
use super::machine::RAM_BASE;
use super::suite::{ARM_STOP, CODE, Case, DATA, Trio, cases, random_periods};
use crate::tlb::{Page, PhysicalMemory, Tlb};

const PAGE: u32 = 0x1000;
/// The physical page of the code of the second address space.
const OTHER_CODE: u32 = CODE + 2 * PAGE;
const FIRST_LEVEL: [u32; 2] = [RAM_BASE + 20 * PAGE, RAM_BASE + 24 * PAGE];
const SECOND_LEVEL: [u32; 2] = [RAM_BASE + 28 * PAGE, RAM_BASE + 29 * PAGE];
/// A second virtual address of the code page of each address space.
const ALIAS: u32 = RAM_BASE + 0x40 * PAGE;
/// A virtual address of a data page: no write in the first address space.
const GUARDED: u32 = RAM_BASE + 0x50 * PAGE;
/// A small page with all accesses for each privilege.
const SMALL_PAGE: u32 = 0x32;
const READ_ONLY: u32 = 0x200;
/// The addresses of the instruction after each one, at the alias and at the code page.
const TO_ALIAS: u32 = DATA + 0x100;
const TO_CODE: u32 = DATA + 0x300;

// The registers of a program.
const SPACES: [u32; 2] = [2, 3];
const POINTER: u32 = 4;
const GUARDED_POINTER: u32 = 5;
const ALIAS_TABLE: u32 = 7;
/// The addresses of the entry of the code page in the table of each address space.
const ENTRIES: [u32; 2] = [8, 11];
/// The entries for the two physical code pages.
const PAGES: [u32; 2] = [9, 10];
const CODE_TABLE: u32 = 12;

fn pick(random: &mut Random, from: [u32; 2]) -> u32 {
    from[random.below(2) as usize]
}

/// The instructions of one code page. Each instruction has 4 bytes, thus a change of
/// the translation goes on at the same place of the other page.
fn image(random: &mut Random, slots: usize, user: bool) -> Vec<u32> {
    let mut words = Vec::new();
    while words.len() < slots {
        let room = slots - words.len();
        let offset = random.below(64) * 4;
        match random.below(if user { 8 } else { 16 }) {
            0 | 1 => words.push(0xe280_0000 | random.below(256)),
            2 => words.push(0xe020_0060 | (1 + random.below(31)) << 7),
            3 => words.push(0xe580_0000 | POINTER << 16 | offset),
            4 if room >= 2 => {
                words.push(0xe590_1000 | POINTER << 16 | offset);
                words.push(0xe080_0001);
            }
            // An access to the page that the first address space can only read: a
            // store there ends the case with an abort.
            5 => {
                let load = u32::from(random.below(8) != 0);
                words.push(0xe580_0000 | load << 20 | GUARDED_POINTER << 16 | offset);
            }
            // A load to the PC: the next instruction at the other virtual address.
            6 | 7 => {
                let table = pick(random, [ALIAS_TABLE, CODE_TABLE]);
                words.push(0xe590_f000 | table << 16 | (words.len() as u32 * 4));
            }
            // MCR to TTBR0.
            8 | 9 => words.push(0xee02_0f10 | pick(random, SPACES) << 12),
            // A new entry for the code page in a table, then MCR to TLBIALL.
            10 | 11 if room >= 2 => {
                words.push(0xe580_0000 | pick(random, ENTRIES) << 16 | pick(random, PAGES) << 12);
                words.push(0xee08_0f17);
            }
            // The same with TLBIMVAA of an address in the code page, from the table.
            13 | 14 if room >= 3 => {
                words.push(0xe580_0000 | pick(random, ENTRIES) << 16 | pick(random, PAGES) << 12);
                words.push(0xe590_1000 | CODE_TABLE << 16 | (words.len() as u32 * 4));
                words.push(0xee08_1f77);
            }
            // DACR with domain 0 as a client, or as a manager with no permission check.
            12 if room >= 2 => {
                words.push(0xe3a0_1001 | random.below(2) << 1);
                words.push(0xee03_1f10);
            }
            _ => words.push(0xe020_0060 | (1 + random.below(31)) << 7),
        }
    }
    // SUBS r6, r6, #1; BNE to the first instruction; BKPT.
    words.push(0xe256_6001);
    words.push(0x1a00_0000 | (0u32.wrapping_sub(slots as u32 + 3) & 0x00ff_ffff));
    words.push(ARM_STOP);
    words
}

fn translation_case(random: &mut Random) -> Case {
    let mut case = Case::new(random, false);
    case.only_starts = false;
    case.fill_code(OTHER_CODE, PAGE, false);
    let user = random.below(4) == 0;
    let slots = 6 + random.below(24) as usize;
    let start = random.below(64) * 4;
    for page in [CODE, OTHER_CODE] {
        case.ram
            .write_words(page + start, &image(random, slots, user));
    }
    for slot in 0..slots as u32 + 1 {
        let next = start + (slot + 1) * 4;
        case.ram.write_words(TO_ALIAS + slot * 4, &[ALIAS + next]);
        case.ram.write_words(TO_CODE + slot * 4, &[CODE + next]);
    }
    for space in 0..2 {
        let (first, second) = (FIRST_LEVEL[space], SECOND_LEVEL[space]);
        case.ram
            .write_words(first + (RAM_BASE >> 20) * 4, &[second | 1]);
        for page in 0..32 {
            case.ram
                .write_words(second + page * 4, &[(RAM_BASE + page * PAGE) | SMALL_PAGE]);
        }
        let code = [CODE, OTHER_CODE][space];
        case.ram
            .write_words(second + (CODE - RAM_BASE) / PAGE * 4, &[code | SMALL_PAGE]);
        case.ram
            .write_words(second + (ALIAS - RAM_BASE) / PAGE * 4, &[code | SMALL_PAGE]);
        let guard = if space == 0 { READ_ONLY } else { 0 };
        case.ram.write_words(
            second + (GUARDED - RAM_BASE) / PAGE * 4,
            &[(DATA + PAGE) | SMALL_PAGE | guard],
        );
    }
    let cpu = &mut case.cpu;
    cpu.cp15.sctlr = Sctlr::from_bits(cpu.cp15.sctlr.bits() | 1);
    cpu.cp15.ttbr0 = FIRST_LEVEL[0].into();
    cpu.cp15.ttbcr = 0;
    cpu.cp15.dacr = 1;
    cpu.regs[0] = random.next();
    cpu.regs[SPACES[0] as usize] = FIRST_LEVEL[0];
    cpu.regs[SPACES[1] as usize] = FIRST_LEVEL[1];
    cpu.regs[POINTER as usize] = DATA + 2 * PAGE + random.below(64) * 4;
    cpu.regs[GUARDED_POINTER as usize] = GUARDED + random.below(64) * 4;
    cpu.regs[6] = 1 + random.below(40);
    cpu.regs[ALIAS_TABLE as usize] = TO_ALIAS;
    cpu.regs[CODE_TABLE as usize] = TO_CODE;
    for space in 0..2 {
        cpu.regs[ENTRIES[space] as usize] = SECOND_LEVEL[space] + (CODE - RAM_BASE) / PAGE * 4;
        cpu.regs[PAGES[space] as usize] = [CODE, OTHER_CODE][space] | SMALL_PAGE;
    }
    cpu.regs[15] = if random.below(4) == 0 { ALIAS } else { CODE } + start;
    if user {
        cpu.cpsr.mode = 0x10;
    }
    case.periods = random_periods(random);
    case.eager = random.below(4) != 0;
    case.until = 200 + u64::from(random.below(1500));
    case
}

#[test]
fn programs_that_change_the_translation_run_the_same_with_each_engine() {
    let mut trio = Trio::new();
    cases(
        "programs_that_change_the_translation_run_the_same_with_each_engine",
        0x0a11_ce5e_ed00_0005,
        5_000,
        |random| {
            let case = translation_case(random);
            trio.run(&case).map(|_| ())
        },
    );
    eprintln!(
        "{:?} {:?} {:?}",
        trio.ends,
        trio.compiled_share(),
        trio.totals
    );
    // Most cases run to the stop of a program or to the last instruction of the case.
    let ends = |name: &str| trio.ends.get(name).copied().unwrap_or(0);
    assert!(
        ends("Breakpoint") + ends("limit") > 3_500,
        "{:?}",
        trio.ends
    );
    assert!(trio.compiled_share().is_none_or(|share| share > 0.1));
}

struct Tables(HashMap<u32, u32>);

impl PhysicalMemory for Tables {
    fn table_word(&self, pa: u32) -> Option<u32> {
        self.0.get(&pa).copied()
    }

    fn page(&self, _physical_page: u32, _access: V6Access) -> Page {
        Page::Memory
    }
}

#[test]
fn an_invalidation_of_a_page_keeps_the_mapping_only_for_a_small_page_with_no_fetch() {
    const SECTION: u32 = 0x4000_0000;
    let (first, second) = (FIRST_LEVEL[0], SECOND_LEVEL[0]);
    let entry = |va: u32| second + (va - RAM_BASE) / PAGE * 4;
    let mut tables = Tables(HashMap::from([
        (first + (RAM_BASE >> 20) * 4, second | 1),
        // A section with all accesses.
        (first + (SECTION >> 20) * 4, 0x8000_0c02),
        (entry(CODE), CODE | SMALL_PAGE),
        (entry(DATA), DATA | SMALL_PAGE),
        (entry(GUARDED), GUARDED | SMALL_PAGE),
    ]));
    let mut tlb = Tlb::default();
    tlb.set_mmu(V6MmuConfig {
        enabled: true,
        ttbr0: first,
        dacr: 1,
        ..V6MmuConfig::default()
    });
    let read =
        |tlb: &Tlb, tables: &Tables, va| tlb.translate(tables, va, V6Access::Read, true).unwrap();
    tlb.translate(&tables, CODE, V6Access::Execute, true)
        .unwrap();
    assert_eq!(read(&tlb, &tables, DATA + 4), DATA + 4);
    assert_eq!(read(&tlb, &tables, GUARDED), GUARDED);
    let mapping = tlb.mapping();

    // The TLB keeps the old entry of a table until the invalidation of its page.
    tables.0.insert(entry(DATA), ALIAS | SMALL_PAGE);
    tables.0.insert(entry(GUARDED), ALIAS | SMALL_PAGE);
    assert_eq!(read(&tlb, &tables, DATA + 4), DATA + 4);
    tlb.flush_page(DATA + 8);
    assert_eq!(read(&tlb, &tables, DATA + 4), ALIAS + 4);
    assert_eq!(read(&tlb, &tables, GUARDED), GUARDED);
    assert_eq!((tlb.mapping(), tlb.flushes), (mapping, 0));

    tlb.flush_page(CODE);
    assert_ne!(tlb.mapping(), mapping);
    assert_eq!(read(&tlb, &tables, GUARDED), ALIAS);

    // The entries of a section are for more pages than the one of the invalidation.
    let mapping = tlb.mapping();
    assert_eq!(read(&tlb, &tables, SECTION + 0x5000), 0x8000_5000);
    tables.0.insert(first + (SECTION >> 20) * 4, 0x9000_0c02);
    tlb.flush_page(SECTION + 0x9000);
    assert_ne!(tlb.mapping(), mapping);
    assert_eq!(read(&tlb, &tables, SECTION + 0x5000), 0x9000_5000);
}
