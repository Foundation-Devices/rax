//! What the test files share: the three engines side by side, the random cases with
//! their seeds, and programs in the memory of the test machine.

use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;
use std::time::{Duration, Instant};

use rax::arm::Armv7Cpu;
use rax::arm::vfp::Fpscr;

use super::Random;
use super::machine::{Counts, Engine, Kind, Machine, Outcome, RAM_BASE, RAM_PAGES, Ram, Stop};
use crate::PAGE_SIZE;

const PAGE: u32 = PAGE_SIZE as u32;
/// Pages with direct entries, for the data of a program.
pub const DATA: u32 = RAM_BASE + 4 * PAGE;
pub const DATA_PAGES: u32 = 8;
pub const CALL_PAGE: u32 = RAM_BASE + 12 * PAGE;
pub const FAULT_PAGE: u32 = RAM_BASE + 13 * PAGE;
pub const DEVICE_PAGE: u32 = RAM_BASE + 14 * PAGE;
/// One more data page, after the device.
pub const LAST_DATA_PAGE: u32 = RAM_BASE + 15 * PAGE;
/// Two pages for the code of a program.
pub const CODE: u32 = RAM_BASE + 16 * PAGE;

// BKPT: the architecture runs it in each IT state, and it has its own vector.
pub const THUMB_STOP: u16 = 0xbebe;
pub const ARM_STOP: u32 = 0xe120_0070;

fn mix(seed: u64, iteration: u64) -> u64 {
    let mut value = seed ^ iteration.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ value >> 30).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ value >> 27).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ value >> 31) | 1
}

fn number(name: &str) -> Option<u64> {
    let text = std::env::var(name).ok()?;
    let parsed = match text.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => text.parse(),
    };
    Some(parsed.unwrap_or_else(|_| panic!("{name}={text} is not a number")))
}

/// Prints how to run the case again when the case stops with a panic.
struct Replay<'a> {
    name: &'a str,
    case_seed: u64,
    seed: u64,
    iteration: u64,
}

impl Drop for Replay<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "{}: case seed {:#x} (seed {:#x}, iteration {})\n\
                 again: RAX_TEST_CASE={:#x} cargo test --release -p rax-aarch32-system {}",
                self.name, self.case_seed, self.seed, self.iteration, self.case_seed, self.name
            );
        }
    }
}

/// Runs `count` random cases. Each case has its own seed, from the seed of the test
/// and the number of the case. The environment changes the run:
/// - `RAX_TEST_CASE`: only the case of this seed.
/// - `RAX_TEST_SEED`: an other seed for the test.
/// - `RAX_SOAK_SECONDS`: cases for this time, not for `count`.
/// - `RAX_TEST_COLLECT`: no stop at a failure. The end has the count of the failures
///   by the text before " : " in their message, with the first case of each.
pub fn cases(
    name: &str,
    seed: u64,
    count: u64,
    mut case: impl FnMut(&mut Random) -> Result<(), String>,
) {
    let collect = std::env::var_os("RAX_TEST_COLLECT").is_some();
    let mut failures: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut run = |case_seed, seed, iteration| {
        let replay = Replay {
            name,
            case_seed,
            seed,
            iteration,
        };
        let result = case(&mut Random(case_seed));
        std::mem::forget(replay);
        let Err(message) = result else {
            return;
        };
        let again = format!(
            "{name}: case seed {case_seed:#x} (seed {seed:#x}, iteration {iteration})\n\
             again: RAX_TEST_CASE={case_seed:#x} cargo test --release -p rax-aarch32-system {name}"
        );
        if !collect {
            panic!("{message}\n{again}");
        }
        let kind = message.split(" : ").next().unwrap().to_string();
        failures
            .entry(kind)
            .or_insert((0, format!("{message}\n{again}")))
            .0 += 1;
    };
    if let Some(case_seed) = number("RAX_TEST_CASE") {
        run(case_seed, 0, 0);
    } else {
        let seed = number("RAX_TEST_SEED").unwrap_or(seed);
        let soak = number("RAX_SOAK_SECONDS").map(Duration::from_secs);
        let start = Instant::now();
        let mut iteration = 0;
        while soak.map_or(iteration < count, |time| start.elapsed() < time) {
            run(mix(seed, iteration), seed, iteration);
            iteration += 1;
        }
        if soak.is_some() {
            eprintln!("{name}: {iteration} cases, seed {seed:#x}");
        }
    }
    for (kind, (count, message)) in &failures {
        eprintln!("{count} failures of this kind: {kind}\n{message}\n");
    }
    assert!(failures.is_empty(), "{} kinds of failures", failures.len());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Insn {
    pub raw: u32,
    pub size: u8,
}

pub fn t16(raw: u32) -> Insn {
    Insn { raw, size: 2 }
}

pub fn wide(raw: u32) -> Insn {
    Insn { raw, size: 4 }
}

/// Instructions in sequence, with a stop after the last one.
#[derive(Clone, Debug)]
pub struct Program {
    pub thumb: bool,
    pub at: u32,
    pub insns: Vec<Insn>,
}

impl Program {
    pub fn end(&self) -> u32 {
        self.at
            + self
                .insns
                .iter()
                .map(|insn| u32::from(insn.size))
                .sum::<u32>()
    }

    pub fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for insn in &self.insns {
            match (self.thumb, insn.size) {
                (true, 2) => bytes.extend((insn.raw as u16).to_le_bytes()),
                (true, _) => {
                    bytes.extend(((insn.raw >> 16) as u16).to_le_bytes());
                    bytes.extend((insn.raw as u16).to_le_bytes());
                }
                (false, _) => bytes.extend(insn.raw.to_le_bytes()),
            }
        }
        bytes
    }
}

/// One run for the three engines.
pub struct Case {
    pub ram: Ram,
    pub cpu: Armv7Cpu,
    pub until: u64,
    starts: HashSet<u32>,
    /// False: the PC can go to each address.
    pub only_starts: bool,
    pub periods: Vec<u64>,
    pub take_exceptions: bool,
    pub irq_events: bool,
    /// False: a block is compiled after some runs, as in a real machine.
    pub eager: bool,
    programs: Vec<Program>,
}

impl Case {
    /// A memory with random data, and the code pages full of stops for `thumb` code.
    pub fn new(random: &mut Random, thumb: bool) -> Case {
        let mut ram = Ram::new();
        let data = (DATA - RAM_BASE) as usize..(CODE - RAM_BASE) as usize;
        for chunk in ram.bytes[data].chunks_mut(4) {
            chunk.copy_from_slice(&random.next().to_le_bytes());
        }
        ram.kinds[((CALL_PAGE - RAM_BASE) / PAGE) as usize] = Kind::Call;
        ram.kinds[((FAULT_PAGE - RAM_BASE) / PAGE) as usize] = Kind::Fault;
        ram.kinds[((DEVICE_PAGE - RAM_BASE) / PAGE) as usize] = Kind::Device;
        // A write to the program would make other instructions than the ones of `starts`.
        ram.kinds[((CODE - RAM_BASE) / PAGE) as usize] = Kind::ReadOnly;
        ram.kinds[((CODE - RAM_BASE) / PAGE) as usize + 1] = Kind::ReadOnly;
        let mut case = Case {
            ram,
            cpu: Armv7Cpu::new(),
            until: 0,
            starts: HashSet::new(),
            only_starts: true,
            periods: Vec::new(),
            take_exceptions: false,
            irq_events: false,
            eager: true,
            programs: Vec::new(),
        };
        case.fill_code(CODE, 2 * PAGE, thumb);
        case
    }

    /// Fills a range with stops.
    pub fn fill_code(&mut self, at: u32, length: u32, thumb: bool) {
        for address in (at..at + length).step_by(4) {
            if thumb {
                self.ram.write_halfwords(address, &[THUMB_STOP, THUMB_STOP]);
            } else {
                self.ram.write_words(address, &[ARM_STOP]);
            }
        }
    }

    /// Puts the program in the memory. Its instructions and the stop after them are
    /// the places where the PC can go.
    pub fn load(&mut self, program: &Program) {
        self.ram.write(program.at, &program.bytes());
        let mut address = program.at;
        for insn in &program.insns {
            self.starts.insert(address | u32::from(program.thumb));
            address += u32::from(insn.size);
        }
        self.starts.insert(address | u32::from(program.thumb));
        self.programs.push(program.clone());
    }

    /// The CPU at the first instruction of the program.
    pub fn enter(&mut self, program: &Program) {
        self.cpu.regs[15] = program.at;
        self.cpu.cpsr.t = program.thumb;
    }

    fn describe(&self) -> String {
        let mut text = String::new();
        for program in &self.programs {
            let insns: Vec<String> = program
                .insns
                .iter()
                .map(|insn| format!("{:0width$x}", insn.raw, width = usize::from(insn.size) * 2))
                .collect();
            text += &format!(
                "{} code at {:#x}: {}\n  bytes {:02x?}\n",
                if program.thumb { "Thumb" } else { "A32" },
                program.at,
                insns.join(" "),
                program.bytes()
            );
        }
        let cpu = &self.cpu;
        text + &format!(
            "from regs {:x?} cpsr {:#x}\n  dregs {:x?}\n  fpscr {:#x}, {} instructions, events each {:?}",
            cpu.regs,
            cpu.cpsr.to_u32(),
            cpu.vfp.dregs,
            cpu.vfp.fpscr.bits(),
            self.until,
            self.periods
        )
    }
}

/// The reference, the interpreter and, when the host has one, the compiler.
pub struct Trio {
    pub machines: Vec<Machine>,
    /// The sums of the counts of each machine, for all the cases.
    pub totals: Vec<Counts>,
    /// The instructions of the reference, for all the cases.
    pub instructions: u64,
    /// The cases by the kind of their end.
    pub ends: BTreeMap<String, u64>,
}

impl Trio {
    pub fn new() -> Trio {
        let engines = [Engine::Reference, Engine::Interpreter, Engine::Compiled];
        let machines: Vec<_> = engines.into_iter().filter_map(Machine::new).collect();
        Trio {
            totals: vec![Counts::default(); machines.len()],
            machines,
            instructions: 0,
            ends: BTreeMap::new(),
        }
    }

    /// The part of the instructions that ran as compiled code, or None with no compiler.
    pub fn compiled_share(&self) -> Option<f64> {
        let index = self
            .machines
            .iter()
            .position(|machine| machine.engine == Engine::Compiled)?;
        Some(self.totals[index].compiled_instructions as f64 / self.instructions.max(1) as f64)
    }

    pub fn compiled(&mut self) -> Option<&mut Machine> {
        self.machines
            .iter_mut()
            .find(|machine| machine.engine == Engine::Compiled)
    }

    /// Runs the case with each engine. The error is the difference from the reference.
    pub fn run(&mut self, case: &Case) -> Result<Outcome, String> {
        let starts = case.only_starts.then(|| Rc::new(case.starts.clone()));
        let mut outcomes = Vec::new();
        for machine in &mut self.machines {
            machine.periods.clone_from(&case.periods);
            machine.take_exceptions = case.take_exceptions;
            machine.irq_events = case.irq_events;
            machine.eager = case.eager;
            machine.starts.clone_from(&starts);
            machine.reset(&case.ram, &case.cpu);
            machine.run(case.until);
            outcomes.push(machine.outcome());
        }
        for (total, machine) in self.totals.iter_mut().zip(&self.machines) {
            total.add(&machine.counts);
        }
        self.instructions += outcomes[0].instructions;
        let end = match &outcomes[0].stop {
            None => "limit",
            Some(Stop::Halt) => "halt",
            Some(Stop::LeftProgram) => "left",
            Some(Stop::Exception(detail)) => detail.split([' ', '(']).next().unwrap(),
        };
        *self.ends.entry(end.to_string()).or_default() += 1;

        for (machine, outcome) in self.machines.iter().zip(&outcomes).skip(1) {
            if let Some(difference) = outcomes[0].difference(outcome) {
                return Err(format!(
                    "the reference, then {:?}:{difference}\n{}",
                    machine.engine,
                    case.describe()
                ));
            }
        }
        Ok(outcomes.swap_remove(0))
    }
}

const SPECIAL_DOUBLES: [u64; 12] = [
    0,
    0x8000_0000_0000_0000,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0001,
    0x7ff4_0000_0000_0000,
    0x7fc0_0001_7f80_0000,
    0x7fa0_0000_ff80_0000,
    0x0000_0000_0000_0001,
    0x000f_ffff_ffff_ffff,
    0x0010_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
];

/// A pointer for a load or a store: most are in the data pages, some are near the
/// pages with a call, a fault or the device.
pub fn random_pointer(random: &mut Random) -> u32 {
    match random.below(32) {
        0..=17 => DATA + PAGE + random.below((DATA_PAGES - 2) * PAGE / 4) * 4,
        18..=21 => DATA + PAGE + random.below((DATA_PAGES - 2) * PAGE),
        22 | 23 => CALL_PAGE - 32 + random.below(64),
        24 | 25 => CALL_PAGE + PAGE / 2 + random.below(16) * 4,
        26 => FAULT_PAGE - 24 + random.below(48),
        27 => DEVICE_PAGE + random.below(64) * 4,
        28 => LAST_DATA_PAGE - 16 + random.below(32),
        29 => LAST_DATA_PAGE + random.below(PAGE / 4) * 4,
        _ => CODE + random.below(2 * PAGE),
    }
}

/// Registers, flags and FP state for a program of instructions with random operands.
pub fn random_cpu(random: &mut Random) -> Armv7Cpu {
    let mut cpu = Armv7Cpu::new();
    for register in &mut cpu.regs {
        *register = match random.below(16) {
            0..=6 => random_pointer(random),
            7..=9 => random.below(64),
            10 => [
                0,
                1,
                0x7fff_ffff,
                0x8000_0000,
                u32::MAX,
                0xffff,
                0x8000,
                0xff,
            ][random.below(8) as usize],
            _ => random.next(),
        };
    }
    cpu.cpsr.n = random.below(2) == 0;
    cpu.cpsr.z = random.below(2) == 0;
    cpu.cpsr.c = random.below(2) == 0;
    cpu.cpsr.v = random.below(2) == 0;
    cpu.cpsr.q = random.below(2) == 0;
    cpu.cpsr.ge = random.below(16) as u8;
    for register in &mut cpu.vfp.dregs {
        *register = match random.below(4) {
            0 => SPECIAL_DOUBLES[random.below(SPECIAL_DOUBLES.len() as u32) as usize],
            1 => f64::from(random.next() as i32 >> random.below(32)).to_bits(),
            2 => {
                u64::from((random.below(1 << 20) as f32 / 16.0).to_bits()) << 32
                    | u64::from((random.next() as f32).to_bits())
            }
            _ => u64::from(random.next()) << 32 | u64::from(random.next()),
        };
    }
    // The rounding mode, flush-to-zero and default NaN, and the flags of the last compare.
    cpu.vfp.fpscr = Fpscr::from_bits(if random.below(3) == 0 {
        random.next() & 0xf3c0_0000
    } else {
        random.next() & 0xf000_0000
    });
    cpu
}

/// The periods of the events of a case: none, or a few short ones.
pub fn random_periods(random: &mut Random) -> Vec<u64> {
    match random.below(4) {
        0 => Vec::new(),
        1 => vec![1 + u64::from(random.below(6))],
        _ => (0..1 + random.below(4))
            .map(|_| 1 + u64::from(random.below(70)))
            .collect(),
    }
}

pub fn ram_page(address: u32) -> usize {
    assert!((address - RAM_BASE) / PAGE < RAM_PAGES as u32);
    ((address - RAM_BASE) / PAGE) as usize
}
