//! A small machine around the engine. It makes the calls of a real machine in the same
//! order, and it can run the same program with the executor only.

use std::cell::Cell;
use std::collections::HashSet;
use std::rc::Rc;

use rax::arm::execution::{ArmMemory, MemoryError};
use rax::arm::instructions::ExclusiveMonitor;
use rax::arm::mmu_v6::{V6Access, V6MmuConfig};
use rax::arm::{
    Armv7Cpu, DecodedInsn, Decoder, ExceptionType, ExecResult, ExecutionState, Executor, Mnemonic,
};

use crate::cache::BlockCache;
use crate::jit::{ACCESS_RETRY, ACCESS_STOP, Context, FpControls, Jit, Tier};
use crate::tlb::{Page, PhysicalMemory, Tlb};
use crate::{Block, Exit, Fault, MAX_INSTRUCTIONS, Memory, NoHook, PAGE_SIZE, fp_controls};

pub const RAM_BASE: u32 = 0x2000_0000;
pub const RAM_PAGES: usize = 32;
const PAGE_MASK: u32 = PAGE_SIZE as u32 - 1;
const FS_EXTERNAL_ABORT: u32 = 0b01000;
const FS_ALIGNMENT: u32 = 0b00001;
const DFSR_WNR: u32 = 1 << 11;
const USER_MODE: u8 = 0x10;

/// How the accesses go to a physical page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Memory with direct entries in the TLB.
    Direct,
    /// Memory, but each access needs a call.
    Call,
    /// Memory with direct reads. Each write gives an external abort.
    ReadOnly,
    /// Each access gives an external abort.
    Fault,
    /// A read gives the count of the instructions that completed before it. A write
    /// puts the next event `1 + (value & 63)` instructions after the ones that completed.
    Device,
}

#[derive(Clone)]
pub struct Ram {
    pub bytes: Vec<u8>,
    pub kinds: [Kind; RAM_PAGES],
    code: [bool; RAM_PAGES],
    changed_code: Vec<usize>,
}

impl Ram {
    pub fn new() -> Ram {
        Ram {
            bytes: vec![0; RAM_PAGES * PAGE_SIZE],
            kinds: [Kind::Direct; RAM_PAGES],
            code: [false; RAM_PAGES],
            changed_code: Vec::new(),
        }
    }

    fn offset(pa: u32) -> Option<usize> {
        let offset = pa.wrapping_sub(RAM_BASE) as usize;
        (offset < RAM_PAGES * PAGE_SIZE).then_some(offset)
    }

    fn kind(&self, pa: u32) -> Kind {
        Ram::offset(pa).map_or(Kind::Fault, |offset| self.kinds[offset / PAGE_SIZE])
    }

    pub fn write(&mut self, address: u32, bytes: &[u8]) {
        let offset = Ram::offset(address).unwrap();
        self.bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    pub fn write_words(&mut self, address: u32, words: &[u32]) {
        for (index, word) in words.iter().enumerate() {
            self.write(address + index as u32 * 4, &word.to_le_bytes());
        }
    }

    pub fn word(&self, address: u32) -> u32 {
        let offset = Ram::offset(address).unwrap();
        u32::from_le_bytes(self.bytes[offset..offset + 4].try_into().unwrap())
    }

    pub fn write_halfwords(&mut self, address: u32, halfwords: &[u16]) {
        for (index, halfword) in halfwords.iter().enumerate() {
            self.write(address + index as u32 * 2, &halfword.to_le_bytes());
        }
    }
}

struct Pages<'a>(&'a Ram);

impl PhysicalMemory for Pages<'_> {
    fn table_word(&self, pa: u32) -> Option<u32> {
        let offset = Ram::offset(pa)?;
        Some(u32::from_le_bytes(
            self.0.bytes.get(offset..offset + 4)?.try_into().unwrap(),
        ))
    }

    fn page(&self, physical_page: u32, access: V6Access) -> Page {
        let Some(offset) = Ram::offset(physical_page) else {
            return Page::Other;
        };
        let page = offset / PAGE_SIZE;
        match self.0.kinds[page] {
            Kind::Fault | Kind::Device => Page::Other,
            Kind::ReadOnly if access == V6Access::Write => Page::Other,
            Kind::Call => Page::Memory,
            // A write to a code page must reach the machine.
            Kind::Direct if access == V6Access::Write && self.0.code[page] => Page::Memory,
            Kind::Direct | Kind::ReadOnly => Page::Direct {
                place: offset,
                host: self.0.bytes.as_ptr() as usize + offset,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusFault {
    pub address: u32,
    pub status: u32,
    pub write: bool,
}

pub struct Bus {
    pub ram: Ram,
    pub tlb: Tlb,
    pub privileged: bool,
    pub monitor: ExclusiveMonitor,
    fault: Cell<Option<BusFault>>,
    device_access: Cell<bool>,
    /// The instructions that completed before the one that runs now.
    now: u64,
    run_start: u64,
    /// The count of instructions of the next event.
    pub deadline: u64,
    /// The calls of `load_slow` and `store_slow`.
    pub slow_calls: u64,
}

impl Bus {
    fn code_offset(&self, va: u32) -> Option<usize> {
        let page = self
            .tlb
            .code_place(&Pages(&self.ram), va, self.privileged)?;
        Some(page | (va & PAGE_MASK) as usize)
    }

    fn fail(&self, address: u32, status: u32, access: V6Access) -> MemoryError {
        self.fault.set(Some(BusFault {
            address,
            status,
            write: access == V6Access::Write,
        }));
        MemoryError::BusError(address)
    }

    fn translate(&self, va: u32, access: V6Access) -> Result<u32, MemoryError> {
        self.tlb
            .translate(&Pages(&self.ram), va, access, self.privileged)
            .map_err(|fault| self.fail(va, fault.status, access))
    }

    fn read<const N: usize>(&self, va: u32, access: V6Access) -> Result<[u8; N], MemoryError> {
        if (va & PAGE_MASK) as usize + N > PAGE_SIZE {
            let mut bytes = [0; N];
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = self.read::<1>(va.wrapping_add(index as u32), access)?[0];
            }
            return Ok(bytes);
        }
        let pa = self.translate(va, access)?;
        match self.ram.kind(pa) {
            Kind::Fault => Err(self.fail(va, FS_EXTERNAL_ABORT, access)),
            Kind::Device if access == V6Access::Execute => {
                Err(self.fail(va, FS_EXTERNAL_ABORT, access))
            }
            Kind::Device => {
                self.device_access.set(true);
                Ok(self.now.to_le_bytes()[..N].try_into().unwrap())
            }
            Kind::Direct | Kind::Call | Kind::ReadOnly => {
                let offset = Ram::offset(pa).unwrap();
                Ok(self.ram.bytes[offset..offset + N].try_into().unwrap())
            }
        }
    }

    fn write<const N: usize>(&mut self, va: u32, bytes: [u8; N]) -> Result<(), MemoryError> {
        if (va & PAGE_MASK) as usize + N > PAGE_SIZE {
            for (index, byte) in bytes.into_iter().enumerate() {
                self.write::<1>(va.wrapping_add(index as u32), [byte])?;
            }
            return Ok(());
        }
        let pa = self.translate(va, V6Access::Write)?;
        match self.ram.kind(pa) {
            Kind::Fault | Kind::ReadOnly => Err(self.fail(va, FS_EXTERNAL_ABORT, V6Access::Write)),
            Kind::Device => {
                self.device_access.set(true);
                self.deadline = self.now + 1 + u64::from(bytes[0] & 63);
                Ok(())
            }
            Kind::Direct | Kind::Call => {
                let offset = Ram::offset(pa).unwrap();
                let page = offset / PAGE_SIZE;
                if std::mem::take(&mut self.ram.code[page]) {
                    self.ram.changed_code.push(page * PAGE_SIZE);
                }
                self.ram.bytes[offset..offset + N].copy_from_slice(&bytes);
                Ok(())
            }
        }
    }

    fn direct_offset(&self, va: u32, access: V6Access, length: usize) -> Option<usize> {
        self.tlb
            .direct(va, access, self.privileged, length)
            .map(|host| host - self.ram.bytes.as_ptr() as usize)
    }
}

impl Memory for Bus {
    fn load<const N: usize>(&self, address: u32) -> Option<[u8; N]> {
        let offset = self.direct_offset(address, V6Access::Read, N)?;
        Some(self.ram.bytes[offset..offset + N].try_into().unwrap())
    }

    fn store<const N: usize>(&mut self, address: u32, bytes: [u8; N]) -> bool {
        let Some(offset) = self.direct_offset(address, V6Access::Write, N) else {
            return false;
        };
        assert!(!self.ram.code[offset / PAGE_SIZE], "direct store to code");
        self.ram.bytes[offset..offset + N].copy_from_slice(&bytes);
        true
    }

    fn load_slow<const N: usize>(&mut self, address: u32, done: usize) -> Result<[u8; N], Fault> {
        self.now = self.run_start + done as u64;
        self.slow_calls += 1;
        self.read(address, V6Access::Read).map_err(|_| Fault)
    }

    fn store_slow<const N: usize>(
        &mut self,
        address: u32,
        bytes: [u8; N],
        done: usize,
    ) -> Result<(), Fault> {
        self.now = self.run_start + done as u64;
        self.slow_calls += 1;
        self.write(address, bytes).map_err(|_| Fault)
    }

    fn exclusive_monitor(&mut self) -> &mut ExclusiveMonitor {
        &mut self.monitor
    }
}

impl ArmMemory for Bus {
    fn read_word(&self, addr: u32) -> Result<u32, MemoryError> {
        self.read(addr, V6Access::Read).map(u32::from_le_bytes)
    }
    fn write_word(&mut self, addr: u32, value: u32) -> Result<(), MemoryError> {
        self.write(addr, value.to_le_bytes())
    }
    fn read_halfword(&self, addr: u32) -> Result<u16, MemoryError> {
        self.read(addr, V6Access::Read).map(u16::from_le_bytes)
    }
    fn write_halfword(&mut self, addr: u32, value: u16) -> Result<(), MemoryError> {
        self.write(addr, value.to_le_bytes())
    }
    fn read_byte(&self, addr: u32) -> Result<u8, MemoryError> {
        self.read::<1>(addr, V6Access::Read).map(|bytes| bytes[0])
    }
    fn write_byte(&mut self, addr: u32, value: u8) -> Result<(), MemoryError> {
        self.write(addr, [value])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// The executor, one instruction at a time.
    Reference,
    /// The fast interpreter, with the executor for the instructions that it leaves.
    Interpreter,
    /// The compiled code, with the fast interpreter where the code leaves.
    Compiled,
}

/// Why a run ended before its last instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    /// An exception, with the registers as before its instruction.
    Exception(String),
    Halt,
    /// The PC is not a start of the program.
    LeftProgram,
}

/// What a run gives: all of it is the same with each engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub instructions: u64,
    pub regs: [u32; 16],
    pub cpsr: u32,
    pub banked: Vec<u32>,
    pub dregs: [u64; 32],
    pub fpscr: u32,
    /// DFSR, DFAR, IFSR and IFAR.
    pub aborts: [u32; 4],
    pub monitor: (Option<u32>, u8),
    pub stop: Option<Stop>,
    /// The count of instructions and the name of each exception that the guest took.
    pub exceptions: Vec<(u64, String)>,
    /// The count of instructions and a hash of the registers at each event.
    pub events: Vec<(u64, u64)>,
    pub ram: Vec<u8>,
}

impl Outcome {
    /// The first parts that are not the same, as text.
    pub fn difference(&self, other: &Outcome) -> Option<String> {
        let mut text = String::new();
        macro_rules! field {
            ($name:ident) => {
                if self.$name != other.$name {
                    text += &format!(
                        "\n  {}: {:x?}\n  {}  {:x?}",
                        stringify!($name),
                        self.$name,
                        " ".repeat(stringify!($name).len()),
                        other.$name
                    );
                }
            };
        }
        field!(instructions);
        field!(regs);
        field!(cpsr);
        field!(banked);
        field!(dregs);
        field!(fpscr);
        field!(aborts);
        field!(monitor);
        field!(stop);
        field!(exceptions);
        field!(events);
        if self.ram != other.ram {
            let at = (0..self.ram.len())
                .find(|&at| self.ram[at] != other.ram[at])
                .unwrap();
            let end = (at + 16).min(self.ram.len());
            text += &format!(
                "\n  ram at {:#x}: {:02x?}\n  {:02x?}",
                RAM_BASE as usize + at,
                &self.ram[at..end],
                &other.ram[at..end]
            );
        }
        (!text.is_empty()).then_some(text)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    pub steps: u64,
    pub block_runs: u64,
    pub slow_instructions: u64,
    pub compiled_runs: u64,
    pub compiled_instructions: u64,
    pub compiled_leaves: u64,
    pub translated_blocks: u64,
    pub changed_code_pages: u64,
}

impl Counts {
    pub fn add(&mut self, other: &Counts) {
        self.steps += other.steps;
        self.block_runs += other.block_runs;
        self.slow_instructions += other.slow_instructions;
        self.compiled_runs += other.compiled_runs;
        self.compiled_instructions += other.compiled_instructions;
        self.compiled_leaves += other.compiled_leaves;
        self.translated_blocks += other.translated_blocks;
        self.changed_code_pages += other.changed_code_pages;
    }
}

#[repr(C)]
struct CompiledRun {
    context: Context,
    bus: *mut Bus,
    start: u64,
    room: i64,
    until: i64,
}

fn slow_access(
    context: *mut Context,
    info: u32,
    access: impl FnOnce(&mut Bus) -> Result<u32, MemoryError>,
) -> u64 {
    // SAFETY: the context is the first field of the run of `Machine::run_compiled`,
    // which uses its bus only after the code returns.
    let (run, bus) = unsafe {
        let run = &mut *context.cast::<CompiledRun>();
        let bus = &mut *run.bus;
        (run, bus)
    };
    let rest = i64::from(info >> 8);
    let before = run.room - run.context.budget - rest;
    assert!(
        before >= 0 && rest >= 1,
        "budget of a call: {before} {rest}"
    );
    bus.now = run.start + before as u64;
    bus.slow_calls += 1;
    bus.device_access.take();
    let Ok(value) = access(bus) else {
        bus.fault.take();
        return ACCESS_RETRY;
    };
    let value = u64::from(value);
    let device = bus.device_access.take();
    if !bus.ram.changed_code.is_empty() {
        return value | ACCESS_STOP;
    }
    if !device {
        return value;
    }
    let after = before + 1;
    let allowed = bus
        .deadline
        .saturating_sub(run.start + after as u64)
        .min((run.until - after) as u64) as i64;
    let budget = allowed - (rest - 1);
    if budget < 0 {
        return value | ACCESS_STOP;
    }
    run.context.budget = budget;
    run.room = after + allowed;
    value
}

extern "C" fn compiled_load(context: *mut Context, address: u32, info: u32) -> u64 {
    slow_access(context, info, |bus| match info & 0xff {
        1 => bus.read_byte(address).map(u32::from),
        2 => bus.read_halfword(address).map(u32::from),
        _ => bus.read_word(address),
    })
}

extern "C" fn compiled_store(context: *mut Context, address: u32, value: u32, info: u32) -> u64 {
    slow_access(context, info, |bus| {
        match info & 0xff {
            1 => bus.write_byte(address, value as u8),
            2 => bus.write_halfword(address, value as u16),
            _ => bus.write_word(address, value),
        }
        .map(|()| 0)
    })
}

#[derive(PartialEq)]
enum Flow {
    Next,
    Jump,
    Exception,
}

pub struct Machine {
    pub cpu: Armv7Cpu,
    pub bus: Bus,
    pub engine: Engine,
    decoder: Decoder,
    pub blocks: BlockCache,
    pub jit: Option<Jit>,
    pub instructions: u64,
    left: Option<(Rc<Block>, usize)>,
    /// The instructions from one event to the next, in a cycle.
    pub periods: Vec<u64>,
    events: Vec<(u64, u64)>,
    /// False ends the run at the first exception.
    pub take_exceptions: bool,
    exceptions: Vec<(u64, String)>,
    pub stop: Option<Stop>,
    /// The addresses (T bit in bit 0) where an instruction of the program starts.
    pub starts: Option<Rc<HashSet<u32>>>,
    /// Each event is an interrupt request, for the time when the guest takes it.
    pub irq_events: bool,
    irq_pending: bool,
    /// Compiles each block at its first run.
    pub eager: bool,
    /// Runs the instruction where compiled code left from its block also after a write
    /// to the code.
    pub keep_left_after_code_write: bool,
    /// Prints the state at each entry of an engine.
    pub trace: bool,
    pub counts: Counts,
}

impl Machine {
    /// None for the compiled engine on a host with no compiler.
    pub fn new(engine: Engine) -> Option<Machine> {
        let jit = if engine == Engine::Compiled {
            Some(Jit::new()?)
        } else {
            None
        };
        Some(Machine {
            cpu: Armv7Cpu::new(),
            bus: Bus {
                ram: Ram::new(),
                tlb: Tlb::default(),
                privileged: true,
                monitor: ExclusiveMonitor::new(),
                fault: Cell::new(None),
                device_access: Cell::new(false),
                now: 0,
                run_start: 0,
                deadline: u64::MAX,
                slow_calls: 0,
            },
            engine,
            decoder: Decoder::new_aarch32(),
            blocks: BlockCache::default(),
            jit,
            instructions: 0,
            left: None,
            periods: Vec::new(),
            events: Vec::new(),
            take_exceptions: false,
            exceptions: Vec::new(),
            stop: None,
            starts: None,
            irq_events: false,
            irq_pending: false,
            eager: true,
            keep_left_after_code_write: false,
            trace: std::env::var_os("RAX_TEST_TRACE").is_some(),
            counts: Counts::default(),
        })
    }

    /// A new start with this memory and this CPU. The block cache and the compiler keep
    /// what they have: the code of the pages is removed as after a write of the guest.
    pub fn reset(&mut self, ram: &Ram, cpu: &Armv7Cpu) {
        let code_pages: Vec<usize> = (0..RAM_PAGES).map(|page| page * PAGE_SIZE).collect();
        if let Some(jit) = &mut self.jit {
            code_pages.iter().for_each(|&page| jit.remove_page(page));
        }
        self.blocks.remove_pages(code_pages);
        // The cells stay at their place: the compiled code has no pointer to them, but
        // a TLB entry has.
        self.bus.ram.bytes.copy_from_slice(&ram.bytes);
        self.bus.ram.kinds = ram.kinds;
        self.bus.ram.code = [false; RAM_PAGES];
        self.bus.ram.changed_code.clear();
        self.bus.tlb = Tlb::default();
        self.bus.monitor = ExclusiveMonitor::new();
        self.bus.fault.set(None);
        self.bus.deadline = u64::MAX;
        self.bus.slow_calls = 0;
        self.cpu = cpu.clone();
        self.instructions = 0;
        self.left = None;
        self.irq_pending = false;
        self.events.clear();
        self.exceptions.clear();
        self.stop = None;
        self.counts = Counts::default();
        self.sync_mmu(0);
        if let Some(&first) = self.periods.first() {
            self.bus.deadline = first;
        }
    }

    pub fn outcome(&self) -> Outcome {
        let cpu = &self.cpu;
        let mut banked = Vec::new();
        banked.extend(cpu.regs_usr);
        banked.extend(cpu.regs_fiq);
        banked.extend(cpu.regs_usr_high);
        banked.extend(cpu.regs_irq);
        banked.extend(cpu.regs_svc);
        banked.extend(cpu.regs_abt);
        banked.extend(cpu.regs_und);
        for spsr in [
            &cpu.spsr_fiq,
            &cpu.spsr_irq,
            &cpu.spsr_svc,
            &cpu.spsr_abt,
            &cpu.spsr_und,
        ] {
            banked.push(spsr.to_u32());
        }
        let cp15 = &cpu.cp15;
        Outcome {
            instructions: self.instructions,
            regs: cpu.regs,
            cpsr: cpu.cpsr.to_u32(),
            banked,
            dregs: cpu.vfp.dregs,
            fpscr: cpu.vfp.fpscr.bits(),
            aborts: [cp15.dfsr, cp15.dfar, cp15.ifsr, cp15.ifar],
            monitor: (self.bus.monitor.address, self.bus.monitor.size),
            stop: self.stop.clone(),
            exceptions: self.exceptions.clone(),
            events: self.events.clone(),
            ram: self.bus.ram.bytes.clone(),
        }
    }

    fn sync_mmu(&mut self, mcr: u32) {
        let cp15 = &self.cpu.cp15;
        self.bus.tlb.set_mmu(V6MmuConfig {
            enabled: cp15.sctlr.m(),
            ttbr0: cp15.ttbr0 as u32,
            ttbr1: cp15.ttbr1 as u32,
            ttbcr_n: cp15.ttbcr & 0x7,
            dacr: cp15.dacr,
            afe: cp15.sctlr.afe(),
        });
        if mcr >> 8 & 0xf == 15 && mcr >> 16 & 0xf == 8 {
            self.bus.tlb.flush();
        }
    }

    /// The exception, or the end of the run when the machine takes none.
    fn raise(&mut self, exception: ExceptionType, detail: String) {
        if !self.take_exceptions {
            self.stop = Some(Stop::Exception(detail));
            return;
        }
        self.exceptions.push((self.instructions, detail));
        let vbar = self.cpu.cp15.vector_base();
        let mut executor = Executor::with_vbar(&mut self.cpu, &mut self.bus, vbar);
        executor.take_exception(exception);
        self.bus.monitor.clear();
    }

    fn prefetch_abort(&mut self) {
        let fault = self.bus.fault.take().unwrap();
        self.cpu.cp15.ifsr = fault.status;
        self.cpu.cp15.ifar = fault.address;
        self.raise(
            ExceptionType::PrefetchAbort(fault.address),
            format!("prefetch abort {fault:x?}"),
        );
    }

    fn data_abort(&mut self) {
        let fault = self.bus.fault.take().unwrap();
        self.cpu.cp15.dfsr = fault.status | if fault.write { DFSR_WNR } else { 0 };
        self.cpu.cp15.dfar = fault.address;
        self.raise(
            ExceptionType::DataAbort(fault.address),
            format!("data abort {fault:x?}"),
        );
    }

    /// The event at the deadline: it keeps what the guest can see at this instruction.
    fn event(&mut self) {
        let cpu = &self.cpu;
        let hash = cpu
            .regs
            .iter()
            .chain(&[cpu.cpsr.to_u32()])
            .fold(1u64, |hash, &word| {
                (hash ^ u64::from(word)).wrapping_mul(0x100_0000_01b3)
            });
        // With no periods, only a write to the device makes an event.
        self.bus.deadline = match self.periods.len() {
            0 => u64::MAX,
            length => self.instructions + self.periods[self.events.len() % length],
        };
        self.events.push((self.instructions, hash));
        self.irq_pending |= self.irq_events;
    }

    /// Runs to the instruction count `until`, or to a stop.
    pub fn run(&mut self, until: u64) {
        while self.instructions < until && self.stop.is_none() {
            if self.instructions >= self.bus.deadline {
                self.event();
            }
            let pc = self.cpu.regs[15] | u32::from(self.cpu.cpsr.t);
            if self.trace {
                eprintln!(
                    "{:?} {} pc {pc:#x} it {:#x} cpsr {:#x} left {:?}",
                    self.engine,
                    self.instructions,
                    self.cpu.cpsr.it_state,
                    self.cpu.cpsr.to_u32(),
                    self.left.as_ref().map(|(_, index)| index)
                );
            }
            if self
                .starts
                .as_ref()
                .is_some_and(|starts| !starts.contains(&pc))
            {
                self.stop = Some(Stop::LeftProgram);
                break;
            }
            let room = (self.bus.deadline - self.instructions).min(until - self.instructions);
            let limit = room.min(MAX_INSTRUCTIONS as u64) as usize;
            let mut left = self.left.take();
            if self.irq_pending && !self.cpu.cpsr.i {
                self.irq_pending = false;
                self.raise(ExceptionType::Irq, "irq".into());
                continue;
            }
            if !self.keep_left_after_code_write && !self.bus.ram.changed_code.is_empty() {
                left = None;
            }
            if self.engine == Engine::Reference || self.cpu.cpsr.it_state != 0 {
                self.step();
                continue;
            }
            self.bus.privileged = self.cpu.cpsr.mode != USER_MODE;
            if let Some((block, index)) = left.filter(|(block, index)| *index < block.len()) {
                self.run_block(&block, index, 1);
                if let Some(jit) = &mut self.jit {
                    jit.left_at(&block, index);
                }
                continue;
            }
            match self.block() {
                Some(block) if self.run_compiled(&block, until) => {}
                Some(block) => self.run_block(&block, 0, limit),
                None => self.step(),
            }
        }
    }

    fn block(&mut self) -> Option<Rc<Block>> {
        if !self.bus.ram.changed_code.is_empty() {
            let pages = std::mem::take(&mut self.bus.ram.changed_code);
            self.counts.changed_code_pages += pages.len() as u64;
            if let Some(jit) = &mut self.jit {
                pages.iter().for_each(|&page| jit.remove_page(page));
            }
            self.blocks.remove_pages(pages);
        }
        let thumb = self.cpu.cpsr.t;
        let pc = self.cpu.regs[15];
        let address = pc | u32::from(thumb);
        let (privileged, mapping) = (self.bus.privileged, self.bus.tlb.mapping());
        if let Some(block) = self.blocks.linked(address, privileged, mapping) {
            return Some(block);
        }
        let offset = self.bus.code_offset(pc)?;
        let page = offset & !(PAGE_SIZE - 1);
        let (ram, counts) = (&self.bus.ram, &mut self.counts);
        let block = self.blocks.block_at(offset, thumb, || {
            let length = (MAX_INSTRUCTIONS * 4).min(page + PAGE_SIZE - offset);
            counts.translated_blocks += 1;
            Block::translate(&ram.bytes[offset..offset + length], offset - page, thumb)
        });
        if !std::mem::replace(&mut self.bus.ram.code[page / PAGE_SIZE], true) {
            self.bus.tlb.protect_code(page);
        }
        let block = block?;
        self.blocks.link(address, privileged, mapping, &block);
        Some(block)
    }

    fn run_compiled(&mut self, block: &Rc<Block>, until: u64) -> bool {
        let pc = self.cpu.regs[15];
        let page = pc & !PAGE_MASK;
        let Some(jit) = self.jit.as_mut() else {
            return false;
        };
        let room = (self.bus.deadline - self.instructions).min(until - self.instructions);
        if room < block.len() as u64 {
            return false;
        }
        if self.eager && matches!(block.tier.get(), Tier::Cold { .. }) {
            block.tier.set(Tier::Cold { runs: u32::MAX });
        }
        let bus = &self.bus;
        let source = || {
            let physical = bus.code_offset(page).unwrap();
            (
                physical,
                bus.ram.bytes[physical..physical + PAGE_SIZE]
                    .try_into()
                    .unwrap(),
            )
        };
        let controls = fp_controls(&self.cpu);
        let fp = FpControls {
            controls,
            fpscr: self.cpu.vfp.fpscr.bits(),
        };
        let Some(code) = jit.code(block, page, fp, source) else {
            return false;
        };
        let (space, flushes) = self.bus.tlb.mapping();
        let jumps = jit.jumps(
            space * 2 + usize::from(self.bus.privileged),
            (flushes, self.blocks.removals()),
        );
        let (reads, writes) = self.bus.tlb.direct_tables(self.bus.privileged);
        let start = self.instructions;
        let mut run = CompiledRun {
            context: Context {
                cpu: &mut self.cpu,
                jumps,
                reads,
                writes,
                budget: room as i64,
                site: std::ptr::null(),
                fp_controls: controls,
                load: compiled_load,
                store: compiled_store,
            },
            bus: &mut self.bus,
            start,
            room: room as i64,
            until: (until - self.instructions) as i64,
        };
        // SAFETY: `block` is the block at the PC in the address space and the privilege
        // of the tables, the stamp changes with each change of the translations and of
        // the code, and the direct entries give cells of the RAM.
        self.left = unsafe {
            jit.set_jump(jumps, pc | u32::from(block.is_thumb()), code);
            jit.run(&mut run.context, code)
        };
        assert!(run.context.budget >= 0, "budget {}", run.context.budget);
        let count = (run.room - run.context.budget) as u64;
        self.instructions += count;
        self.counts.compiled_runs += 1;
        self.counts.compiled_instructions += count;
        self.counts.compiled_leaves += u64::from(self.left.is_some());
        true
    }

    fn run_block(&mut self, block: &Block, mut index: usize, mut limit: usize) {
        let page = self.cpu.regs[15] & !PAGE_MASK;
        self.counts.block_runs += 1;
        loop {
            self.bus.run_start = self.instructions;
            let (count, exit) = block.run(
                index,
                limit,
                page,
                &mut self.cpu,
                &mut self.bus,
                &mut NoHook,
            );
            assert!(
                count <= limit,
                "{count} instructions with a limit of {limit}"
            );
            self.instructions += count as u64;
            limit -= count;
            let slow = match exit {
                Exit::Done => return,
                Exit::Fault(_) => return self.data_abort(),
                Exit::Slow(slow) => slow,
            };
            assert_eq!(slow, index + count);
            let cpsr = &self.cpu.cpsr;
            let interrupts_and_mode = (cpsr.i, cpsr.f, cpsr.mode);
            self.bus.device_access.take();
            self.counts.slow_instructions += 1;
            let controls = fp_controls(&self.cpu);
            let admitted = block.is_admitted(slow, controls);
            let flow = self.execute(block.executor_insn(slow), admitted);
            if !admitted && flow != Flow::Exception && fp_controls(&self.cpu) == controls {
                block.set_admitted(slow, controls);
            }
            limit -= 1;
            index = slow + 1;
            let cpsr = &self.cpu.cpsr;
            let same_conditions = interrupts_and_mode == (cpsr.i, cpsr.f, cpsr.mode)
                && cpsr.it_state == block.it_state(index)
                && !self.bus.device_access.take()
                && self.bus.ram.changed_code.is_empty();
            let compiles =
                self.jit.is_some() && cpsr.it_state == 0 && !block.has_fast_operation(slow);
            if flow != Flow::Next
                || limit == 0
                || index == block.len()
                || !same_conditions
                || compiles
            {
                return;
            }
        }
    }

    fn step(&mut self) {
        self.counts.steps += 1;
        self.bus.privileged = self.cpu.cpsr.mode != USER_MODE;
        let pc = self.cpu.regs[15];
        let thumb = self.cpu.cpsr.t;
        let mut bytes = [0u8; 4];
        let mut length = 4;
        if thumb {
            let Ok(first) = self.bus.read::<2>(pc, V6Access::Execute) else {
                return self.prefetch_abort();
            };
            bytes[..2].copy_from_slice(&first);
            if first[1] >> 3 >= 0x1d {
                let Ok(second) = self.bus.read::<2>(pc.wrapping_add(2), V6Access::Execute) else {
                    return self.prefetch_abort();
                };
                bytes[2..].copy_from_slice(&second);
            } else {
                length = 2;
            }
        } else {
            let Ok(word) = self.bus.read::<4>(pc, V6Access::Execute) else {
                return self.prefetch_abort();
            };
            bytes = word;
        }
        self.decoder.set_state(if thumb {
            ExecutionState::Thumb
        } else {
            ExecutionState::Aarch32
        });
        let Ok(insn) = self.decoder.decode(&bytes[..length]) else {
            return self.raise(
                ExceptionType::UndefinedInstruction,
                "undefined (decode)".into(),
            );
        };
        self.execute(&insn, false);
    }

    /// Runs the instruction at the PC with the executor.
    fn execute(&mut self, insn: &DecodedInsn, admitted: bool) -> Flow {
        let pc = self.cpu.regs[15];
        let next_pc = pc.wrapping_add(u32::from(insn.size));
        let vbar = self.cpu.cp15.vector_base();
        let advance_it = self.cpu.cpsr.t && self.cpu.cpsr.in_it_block();
        let snapshot = (self.cpu.regs, self.cpu.cpsr.clone());
        if self.trace {
            eprintln!(
                "  execute {:x} {:?} at {pc:#x} it {:#x}",
                insn.raw, insn.mnemonic, self.cpu.cpsr.it_state
            );
        }
        self.bus.now = self.instructions;
        let monitor = std::mem::take(&mut self.bus.monitor);
        let mut executor = Executor::with_vbar(&mut self.cpu, &mut self.bus, vbar);
        executor.exclusive_monitor = monitor;
        let result = if admitted {
            executor.execute_admitted_again(insn)
        } else {
            executor.execute(insn)
        };
        self.bus.monitor = std::mem::take(&mut executor.exclusive_monitor);

        let mut flow = Flow::Jump;
        match result {
            ExecResult::Continue => {
                self.cpu.regs[15] = next_pc;
                if advance_it {
                    self.cpu.cpsr.advance_it_state();
                }
                if insn.mnemonic == Mnemonic::MCR {
                    self.sync_mmu(insn.raw);
                } else {
                    flow = Flow::Next;
                }
            }
            ExecResult::Branch(target) => {
                if insn.mnemonic != Mnemonic::RFE && target & 1 != 0 {
                    self.cpu.cpsr.t = true;
                }
                self.cpu.regs[15] = target & !1;
                if advance_it {
                    self.cpu.cpsr.advance_it_state();
                }
            }
            ExecResult::Halt => {
                self.cpu.regs[15] = next_pc;
                if advance_it {
                    self.cpu.cpsr.advance_it_state();
                }
                self.stop = Some(Stop::Halt);
            }
            ExecResult::Undefined
            | ExecResult::Exception(
                ExceptionType::UndefinedInstruction | ExceptionType::FloatingPoint(_),
            ) => {
                (self.cpu.regs, self.cpu.cpsr) = snapshot;
                self.raise(
                    ExceptionType::UndefinedInstruction,
                    format!("undefined ({:?})", insn.mnemonic),
                );
                return Flow::Exception;
            }
            ExecResult::Exception(exception) => {
                let detail = format!("{exception:x?}");
                if !self.take_exceptions {
                    (self.cpu.regs, self.cpu.cpsr) = snapshot;
                    self.raise(exception, detail);
                    return Flow::Exception;
                }
                self.raise(exception, detail);
                flow = Flow::Exception;
            }
            ExecResult::FpAccessTrap { .. } => panic!("FP access trap at {pc:#x}"),
            ExecResult::MemoryFault(error) => {
                (self.cpu.regs, self.cpu.cpsr) = snapshot;
                // The executor makes the alignment faults with no call of the memory.
                if let MemoryError::Unaligned(address) = error {
                    self.bus.fault.set(Some(BusFault {
                        address,
                        status: FS_ALIGNMENT,
                        write: false,
                    }));
                }
                assert!(self.bus.fault.get().is_some(), "{error} with no fault");
                self.data_abort();
                return Flow::Exception;
            }
        }
        self.instructions += 1;
        flow
    }
}
