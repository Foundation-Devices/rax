use rax::arm::execution::{ArmMemory, MemoryError};
use rax::arm::instructions::ExclusiveMonitor;
use rax::arm::vfp::Fpscr;
use rax::arm::{ExecResult, Executor};

use std::collections::HashSet;
use std::rc::Rc;

use super::jit::{ACCESS_RETRY, ACCESS_STOP, Context, FpControls, Jit, Tier};
use super::*;

const MEMORY_BASE: u32 = 0x2000_0000;
const MEMORY_SIZE: usize = 0x8000;
const CODE_PAGE: u32 = MEMORY_BASE + 0x7000;
const PAGE_SIZE: usize = 0x1000;

mod blocks;
mod corrections;
mod engine;
#[cfg(target_arch = "aarch64")]
mod fp;
mod machine;
mod memory;
mod mmu;
mod neon;
mod oracle;
mod programs;
mod suite;
mod system;

#[derive(Clone)]
struct FlatMemory(Vec<u8>, ExclusiveMonitor);

impl PartialEq for FlatMemory {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && (self.1.address, self.1.size) == (other.1.address, other.1.size)
    }
}

impl FlatMemory {
    fn range(&self, address: u32, length: usize) -> Option<std::ops::Range<usize>> {
        let start = address.wrapping_sub(MEMORY_BASE) as usize;
        (start < MEMORY_SIZE - length).then_some(start..start + length)
    }

    fn read<const N: usize>(&self, address: u32) -> Result<[u8; N], MemoryError> {
        self.load(address).ok_or(MemoryError::OutOfBounds(address))
    }

    fn write<const N: usize>(&mut self, address: u32, bytes: [u8; N]) -> Result<(), MemoryError> {
        if self.store(address, bytes) {
            Ok(())
        } else {
            Err(MemoryError::OutOfBounds(address))
        }
    }
}

impl Memory for FlatMemory {
    fn load<const N: usize>(&self, address: u32) -> Option<[u8; N]> {
        Some(self.0[self.range(address, N)?].try_into().unwrap())
    }

    fn store<const N: usize>(&mut self, address: u32, bytes: [u8; N]) -> bool {
        let Some(range) = self.range(address, N) else {
            return false;
        };
        self.0[range].copy_from_slice(&bytes);
        true
    }

    fn load_slow<const N: usize>(&mut self, _address: u32, _done: usize) -> Result<[u8; N], Fault> {
        Err(Fault)
    }

    fn store_slow<const N: usize>(
        &mut self,
        _address: u32,
        _bytes: [u8; N],
        _done: usize,
    ) -> Result<(), Fault> {
        Err(Fault)
    }

    fn exclusive_monitor(&mut self) -> &mut ExclusiveMonitor {
        &mut self.1
    }
}

impl ArmMemory for FlatMemory {
    fn read_word(&self, addr: u32) -> Result<u32, MemoryError> {
        self.read(addr).map(u32::from_le_bytes)
    }
    fn write_word(&mut self, addr: u32, value: u32) -> Result<(), MemoryError> {
        self.write(addr, value.to_le_bytes())
    }
    fn read_halfword(&self, addr: u32) -> Result<u16, MemoryError> {
        self.read(addr).map(u16::from_le_bytes)
    }
    fn write_halfword(&mut self, addr: u32, value: u16) -> Result<(), MemoryError> {
        self.write(addr, value.to_le_bytes())
    }
    fn read_byte(&self, addr: u32) -> Result<u8, MemoryError> {
        self.read::<1>(addr).map(|bytes| bytes[0])
    }
    fn write_byte(&mut self, addr: u32, value: u8) -> Result<(), MemoryError> {
        self.write(addr, [value])
    }
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, limit: u32) -> u32 {
        self.next() % limit
    }
}

/// One instruction with the Executor, as a machine that has no fast interpreter runs it.
/// Returns false when the instruction does not end with a new PC.
fn reference_step(cpu: &mut Armv7Cpu, memory: &mut FlatMemory, insn: &DecodedInsn) -> bool {
    executor_step(cpu, memory, insn, false)
}

/// `admitted` selects the entry of the executor with no FP access check.
fn executor_step(
    cpu: &mut Armv7Cpu,
    memory: &mut FlatMemory,
    insn: &DecodedInsn,
    admitted: bool,
) -> bool {
    let pc = cpu.regs[15];
    let advance_it = cpu.cpsr.in_it_block();
    // The monitor of the memory, as a machine gives it to the executor.
    let monitor = std::mem::take(&mut memory.1);
    let mut executor = Executor::new(cpu, memory);
    executor.exclusive_monitor = monitor;
    let result = if admitted {
        executor.execute_admitted_again(insn)
    } else {
        executor.execute(insn)
    };
    memory.1 = std::mem::take(&mut executor.exclusive_monitor);
    match result {
        ExecResult::Continue => cpu.regs[15] = pc.wrapping_add(u32::from(insn.size)),
        ExecResult::Branch(target) => {
            cpu.cpsr.t |= target & 1 != 0;
            cpu.regs[15] = target & !1;
        }
        _ => return false,
    }
    if advance_it {
        cpu.cpsr.advance_it_state();
    }
    true
}

/// T32 encodings of the groups that have fast operations: the fixed bits, the random
/// bits, and true when the two copies of Rm must be equal.
const WIDE_TEMPLATES: [(u32, u32, bool); 60] = [
    (0xf000_0000, 0x07ff_ffff, false),
    (0xf000_0000, 0x07ff_ffff, false),
    (0xe800_0000, 0x01ff_ffff, false),
    (0xea00_0000, 0x01ff_ffff, false),
    (0xea00_0000, 0x01ff_ffff, false),
    (0xf800_0000, 0x01ff_ffff, false),
    (0xf800_0000, 0x00ff_ffff, false),
    (0xfa00_f000, 0x00ff_0fff, false),
    (0xfa90_f080, 0x002f_0f3f, true),
    (0xfb00_0000, 0x00ff_ffff, false),
    (0xfb90_f0f0, 0x002f_0f0f, false),
    (0xe92d_0000, 0x0000_ffff, false),
    (0xe8bd_0000, 0x0000_ffff, false),
    (0xf800_0000, 0x00ff_ffff, false),
    // VFP: data processing, moves to and from the core registers, loads and stores.
    (0xee00_0a00, 0x00ff_f1ef, false),
    (0xee00_0a00, 0x00ff_f1ef, false),
    (0xeeb0_0a00, 0x004f_f1ef, false),
    (0xeeb0_0a00, 0x004f_f1ef, false),
    (0xee00_0a10, 0x00ff_f1ef, false),
    (0xee00_0a10, 0x001f_f080, false),
    (0xec40_0a10, 0x001f_f1ef, false),
    (0xed00_0a00, 0x00ff_f1ff, false),
    (0xeef1_fa10, 0x0000_0000, false),
    (0xed1f_0a00, 0x00c0_f1ff, false),
    // Advanced SIMD: the executor runs these.
    (0xef00_0000, 0x10ff_ffff, false),
    (0xef00_0110, 0x107f_f0ef, false),
    (0xf900_0a00, 0x006f_f0cf, false),
    (0xf900_0200, 0x006f_f5cf, false),
    (0xf900_0a00, 0x006f_f0ff, false),
    // Advanced SIMD on integer elements: three registers, shifts, long, narrow, compare.
    (0xef00_0010, 0x107f_f2ef, false),
    (0xef00_0300, 0x107f_f0ff, false),
    (0xef00_0800, 0x107f_f1ff, false),
    (0xef80_0010, 0x107f_f2ef, false),
    (0xef80_0810, 0x103f_f26f, false),
    (0xef80_0000, 0x103f_fcaf, false),
    (0xffb0_0000, 0x004f_f7ef, false),
    (0xef80_0010, 0x1047_ff6f, false),
    (0xec00_0b00, 0x01ff_f01e, false),
    (0xed2d_0b00, 0x0040_f01e, false),
    (0xecbd_0b00, 0x0040_f01e, false),
    (0xe8d0_f000, 0x000f_001f, false),
    (0xf000_c000, 0x07ff_2ffe, false),
    (0xf3bf_8f40, 0x0000_003f, false),
    (0xf900_0000, 0x00ef_ffff, false),
    (0xffb0_0580, 0x0040_f06f, false),
    // VLD3, VST3, VLD4, VST4, and the compares of single-precision elements.
    (0xf900_0000, 0x0064_f1ff, false),
    (0xf900_0400, 0x0064_f1df, false),
    (0xef00_0e00, 0x106f_f0ef, false),
    // CP15 cache operation and thread registers, SMULxy, VMOV of one element, VDUP.
    (0xee07_0f36, 0x0000_f000, false),
    (0xee1d_0f10, 0x0000_f0e0, false),
    (0xfb10_f000, 0x000f_0f3f, false),
    (0xee00_0b10, 0x00ff_f0e0, false),
    (0xee80_0b10, 0x006f_f0a0, false),
    (0xffb0_0c00, 0x004f_f06f, false),
    // LDREX, STREX, LDREXB and LDREXH, STREXB and STREXH, CLREX.
    (0xe850_0f00, 0x000f_f0ff, false),
    (0xe840_0000, 0x000f_ffff, false),
    (0xe8d0_0f4f, 0x000f_f010, false),
    (0xe8c0_0f40, 0x000f_f01f, false),
    (0xf3bf_8f2f, 0x0000_0000, false),
    // VSHL by a register.
    (0xef00_0400, 0x107f_f0ef, false),
];

/// A32 encodings of the groups that have fast operations: the fixed bits and the random bits.
const ARM_TEMPLATES: [(u32, u32); 18] = [
    (0xf200_0110, 0x017f_f0ef),
    (0xf57f_f040, 0x0000_003f),
    (0xe000_0000, 0x01ff_ffef),
    (0xe000_0000, 0x01ff_ffef),
    (0xe200_0000, 0x01ff_ffff),
    (0xe200_0000, 0x01ff_ffff),
    (0x0000_0000, 0xf3ff_ffff),
    (0xe400_0000, 0x03ff_ffff),
    (0xe400_0000, 0x03ff_ffff),
    (0xe000_0090, 0x00ef_ff0f),
    (0xe040_0090, 0x01bf_ff6f),
    (0xe800_0000, 0x01bf_ffff),
    (0xe800_0000, 0x01bf_ffff),
    (0x0a00_0000, 0xf1ff_ffff),
    (0xe12f_ff10, 0x0000_000f),
    (0xe300_0000, 0x004f_ffff),
    (0xe16f_0f10, 0x0000_f00f),
    (0xe6af_0070, 0x0050_fc0f),
];

fn random_arm_code(random: &mut Random, code: &mut [u8]) -> usize {
    let start = random.below(4) as usize * 4;
    for index in 0..1 + random.below(4) as usize {
        let (fixed, mask) = ARM_TEMPLATES[random.below(ARM_TEMPLATES.len() as u32) as usize];
        let raw = fixed | (random.next() << 16 | random.next() & 0xffff) & mask;
        code[start + index * 4..][..4].copy_from_slice(&raw.to_le_bytes());
    }
    start
}

fn random_code(random: &mut Random, code: &mut [u8]) -> usize {
    let mut at = random.below(8) as usize * 2;
    let start = at;
    let mut halfword = |value: u32| {
        code[at..at + 2].copy_from_slice(&(value as u16).to_le_bytes());
        at += 2;
    };
    if random.below(4) == 0 {
        halfword(0xbf00 | random.below(14) << 4 | (1 + random.below(15)));
    }
    for _ in 0..1 + random.below(4) {
        if random.below(2) == 0 {
            let value = random.next() & 0xffff;
            // A random halfword of the 32-bit groups would take the next one as its second half.
            halfword(if value >> 11 >= 0x1d {
                value & 0x7fff
            } else {
                value
            });
            continue;
        }
        let (fixed, mask, same_rm) =
            WIDE_TEMPLATES[random.below(WIDE_TEMPLATES.len() as u32) as usize];
        let mut raw = fixed | random.next() << 16 & mask | random.next() & mask & 0xffff;
        if same_rm {
            raw = raw & !0xf_0000 | (raw & 0xf) << 16;
        }
        halfword(raw >> 16);
        halfword(raw);
    }
    start
}

fn random_cpu(random: &mut Random, entry: u32) -> Armv7Cpu {
    let mut cpu = Armv7Cpu::new();
    for register in &mut cpu.regs {
        *register = match random.below(8) {
            0..=2 => MEMORY_BASE + 0x1000 + random.below(0x1400) * 4,
            3 => MEMORY_BASE + 0x1000 + random.below(0x5000),
            4 | 5 => random.below(64),
            _ => random.next(),
        };
    }
    cpu.regs[15] = entry;
    cpu.cpsr.t = true;
    cpu.cpsr.n = random.below(2) == 0;
    cpu.cpsr.z = random.below(2) == 0;
    cpu.cpsr.c = random.below(2) == 0;
    cpu.cpsr.v = random.below(2) == 0;
    const SPECIAL: [u64; 8] = [
        0,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x7ff4_0000_0000_0000,
        0x7fc0_0001_7f80_0000,
        0x7fa0_0000_ff80_0000,
        0x0000_0000_0000_0001,
    ];
    for register in &mut cpu.vfp.dregs {
        *register = match random.below(4) {
            0 => SPECIAL[random.below(8) as usize],
            1 => f64::from(random.next() as i32 >> random.below(32)).to_bits(),
            2 => {
                u64::from((random.below(1 << 20) as f32 / 16.0).to_bits()) << 32
                    | u64::from((random.next() as f32).to_bits())
            }
            _ => u64::from(random.next()) << 32 | u64::from(random.next()),
        };
    }
    // The rounding mode, flush-to-zero and default NaN, and the flags of the last compare.
    cpu.vfp.fpscr = Fpscr::from_bits(random.next() & 0xf3c0_0000);
    cpu
}

/// Runs the block as the machine does. Returns the number of instructions that ran and
/// the ones that the fast interpreter ran alone, or None when the executor stops at an instruction.
fn run_block(
    block: &Block,
    limit: usize,
    cpu: &mut Armv7Cpu,
    memory: &mut FlatMemory,
) -> Option<(usize, Vec<usize>)> {
    let (mut index, mut total) = (0, 0);
    let mut fast = Vec::new();
    let thumb = cpu.cpsr.t;
    let in_sequence = |cpu: &Armv7Cpu, index| {
        cpu.cpsr.t == thumb
            && cpu.regs[15] == CODE_PAGE + block.offset(index)
            && cpu.cpsr.it_state == block.it_state(index)
    };
    while index < block.len() {
        let (count, exit) = block.run(index, limit, CODE_PAGE, cpu, memory, &mut NoHook);
        total += count;
        if limit == 1 && count == 1 {
            fast.push(index);
        }
        index += count;
        match exit {
            Exit::Done => {}
            Exit::Fault(fault) => {
                assert!(fault == index && in_sequence(cpu, index));
                assert!(!reference_step(cpu, memory, &block.insns()[fault]));
                return None;
            }
            Exit::Slow(slow) => {
                assert!(slow == index && in_sequence(cpu, index));
                let controls = fp_controls(cpu);
                let admitted = block.is_admitted(slow, controls);
                if !executor_step(cpu, memory, block.executor_insn(slow), admitted) {
                    return None;
                }
                if fp_controls(cpu) == controls {
                    block.set_admitted(slow, controls);
                }
                total += 1;
                index += 1;
            }
        }
        if !in_sequence(cpu, index) {
            break;
        }
    }
    Some((total, fast))
}

fn operation_name(op: Op) -> String {
    let name = format!("{op:?}");
    let mut name = name.split([' ', '{']).next().unwrap().to_string();
    if let Op::SimdLane { kind, .. } = op {
        name += &format!("{kind:?}");
    }
    name
}

/// The compiler and what the compiled code of the test gets.
struct Compiled {
    jit: Jit,
    /// A translation table with direct entries for a part of the pages before the code page.
    table: Vec<[u64; 4]>,
    /// The names of the operations that ran as compiled code.
    covered: HashSet<String>,
}

/// One run of compiled code, for the load and the store that the code calls.
#[repr(C)]
struct CompiledRun {
    context: Context,
    memory: *mut FlatMemory,
    /// The code leaves after each call.
    stop: bool,
}

/// The accesses with no direct entry that are in the memory: the code page.
extern "C" fn compiled_load(context: *mut Context, address: u32, info: u32) -> u64 {
    // SAFETY: the context is the first field of the run, and the memory has no other use now.
    let (run, memory) = unsafe {
        let run = &*context.cast::<CompiledRun>();
        (run, &*run.memory)
    };
    let value = match info & 0xff {
        1 => memory.load::<1>(address).map(|bytes| u64::from(bytes[0])),
        2 => memory
            .load(address)
            .map(|bytes| u64::from(u16::from_le_bytes(bytes))),
        _ => memory
            .load(address)
            .map(|bytes| u64::from(u32::from_le_bytes(bytes))),
    };
    value.map_or(ACCESS_RETRY, |value| {
        value | if run.stop { ACCESS_STOP } else { 0 }
    })
}

extern "C" fn compiled_store(context: *mut Context, address: u32, value: u32, info: u32) -> u64 {
    // SAFETY: as in `compiled_load`.
    let (run, memory) = unsafe {
        let run = &*context.cast::<CompiledRun>();
        (run, &mut *run.memory)
    };
    let stored = match info & 0xff {
        1 => memory.store(address, [value as u8]),
        2 => memory.store(address, (value as u16).to_le_bytes()),
        _ => memory.store(address, value.to_le_bytes()),
    };
    match (stored, run.stop) {
        (false, _) => ACCESS_RETRY,
        (true, true) => ACCESS_STOP,
        (true, false) => 0,
    }
}

/// Runs `total` instructions from the PC as the machine does with the compiler: the
/// compiled code of a new block at each entry, and the interpreter for one instruction
/// where the code leaves. `block` is the block of all these instructions. Returns false
/// when an instruction stops.
fn run_compiled(
    compiled: &mut Compiled,
    block: &Block,
    total: usize,
    cpu: &mut Armv7Cpu,
    memory: &mut FlatMemory,
) -> bool {
    let code_offset = (CODE_PAGE - MEMORY_BASE) as usize;
    let mut done = 0;
    while done < total {
        let offset = cpu.regs[15].wrapping_sub(CODE_PAGE);
        let entry = Rc::new(Block::translate(
            &memory.0[code_offset + offset as usize..code_offset + PAGE_SIZE],
            offset as usize,
            cpu.cpsr.t,
        ));
        entry.tier.set(Tier::Cold { runs: u32::MAX });
        let room = total - done;
        let source = || {
            (
                0,
                memory.0[code_offset..code_offset + PAGE_SIZE]
                    .try_into()
                    .unwrap(),
            )
        };
        // The block of the reference run has the FP controls that the executor accepted.
        let fp = FpControls {
            controls: fp_controls(cpu),
            fpscr: cpu.vfp.fpscr.bits(),
        };
        let first = (0..block.len())
            .find(|&index| block.offset(index) == offset)
            .unwrap();
        for (index, admitted) in block.admitted[first..].iter().enumerate().take(entry.len()) {
            entry.set_admitted(index, admitted.get());
        }
        let code = (cpu.cpsr.it_state == 0 && room >= entry.len())
            .then(|| compiled.jit.code(&entry, CODE_PAGE, fp, source))
            .flatten();
        if let Some(code) = code {
            // The accesses to the code page go to the functions of the test, which
            // have the limits of FlatMemory. Each second page before it goes there
            // too, for the accesses of the code with a call.
            for (page, entry) in compiled
                .table
                .iter_mut()
                .enumerate()
                .take(code_offset / PAGE_SIZE)
                .step_by(2)
            {
                let address = MEMORY_BASE + (page * PAGE_SIZE) as u32;
                entry[jit::TLB_DIRECT_PAGE / 8] = u64::from(address) << 32;
                entry[jit::TLB_ADDEND / 8] =
                    (memory.0.as_mut_ptr() as u64).wrapping_sub(MEMORY_BASE.into());
            }
            let table = compiled.table.as_ptr().cast();
            // No code is in the table of the jumps: each jump comes back.
            let jumps = compiled.jit.jumps(0, (0, 0));
            let mut run = CompiledRun {
                context: Context {
                    cpu,
                    jumps,
                    reads: table,
                    writes: table,
                    budget: room as i64,
                    site: std::ptr::null(),
                    fp_controls: fp.controls,
                    load: compiled_load,
                    store: compiled_store,
                },
                memory,
                stop: done % 2 == 0,
            };
            // SAFETY: the code is that of the block at the PC, and the direct entries
            // give the bytes of `memory`.
            unsafe { compiled.jit.run(&mut run.context, code) };
            let count = room - run.context.budget as usize;
            // An instruction with a condition is in the count also when it did not run.
            for step in entry.steps[..count]
                .iter()
                .filter(|step| step.cond == COND_ALWAYS)
            {
                compiled.covered.insert(operation_name(step.op));
            }
            done += count;
            if count != 0 {
                continue;
            }
        }
        // One instruction with the interpreter and the block of the reference run.
        let offset = cpu.regs[15].wrapping_sub(CODE_PAGE);
        let index = (0..block.len())
            .find(|&index| block.offset(index) == offset)
            .unwrap();
        match block.run(index, 1, CODE_PAGE, cpu, memory, &mut NoHook) {
            (_, Exit::Done) => {}
            (_, Exit::Fault(_)) => return false,
            (_, Exit::Slow(_)) => {
                let admitted = block.is_admitted(index, fp_controls(cpu));
                if !executor_step(cpu, memory, block.executor_insn(index), admitted) {
                    return false;
                }
            }
        }
        done += 1;
    }
    true
}

/// Runs random Thumb code with the fast interpreter and with the Executor only.
/// Adds the names of the operations that the fast interpreter ran to `covered`.
fn compare_random_block(
    random: &mut Random,
    memory: &mut FlatMemory,
    covered: &mut HashSet<String>,
    compiled: &mut Option<Compiled>,
) {
    let code_offset = (CODE_PAGE - MEMORY_BASE) as usize;
    let thumb = random.below(4) != 0;
    let code = &mut memory.0[code_offset..code_offset + 64];
    let entry = if thumb {
        random_code(random, code)
    } else {
        random_arm_code(random, code)
    };
    let block = Block::translate(
        &memory.0[code_offset + entry..code_offset + PAGE_SIZE],
        entry,
        thumb,
    );
    let mut start = random_cpu(random, CODE_PAGE + entry as u32);
    start.cpsr.t = thumb;
    let limit = if random.below(2) == 0 { 1 } else { usize::MAX };

    // The first run gives the block the FP controls that the executor accepted.
    run_block(&block, limit, &mut start.clone(), &mut memory.clone());
    let (mut cpu, mut fast_memory) = (start.clone(), memory.clone());
    let Some((total, fast)) = run_block(&block, limit, &mut cpu, &mut fast_memory) else {
        return;
    };

    let (mut reference, mut reference_memory) = (start.clone(), memory.clone());
    for _ in 0..total {
        let offset = reference.regs[15].wrapping_sub(CODE_PAGE);
        let index = (0..block.len())
            .find(|&index| block.offset(index) == offset)
            .unwrap();
        let insn = &block.insns()[index];
        assert!(
            reference_step(&mut reference, &mut reference_memory, insn),
            "the Executor stops at {insn:x?}"
        );
    }
    let context = || {
        let insns: Vec<_> = block
            .insns()
            .iter()
            .map(|insn| format!("{:x} {:?}", insn.raw, insn.mnemonic))
            .collect();
        format!(
            "{insns:?} from {:x?} flags {:x}",
            start.regs,
            start.cpsr.to_u32()
        )
    };
    assert_eq!(cpu.regs, reference.regs, "{}", context());
    assert_eq!(cpu.cpsr.to_u32(), reference.cpsr.to_u32(), "{}", context());
    assert_eq!(cpu.vfp.dregs, reference.vfp.dregs, "{}", context());
    assert_eq!(
        cpu.vfp.fpscr.bits(),
        reference.vfp.fpscr.bits(),
        "{}",
        context()
    );
    assert!(fast_memory == reference_memory, "memory: {}", context());
    for index in fast {
        covered.insert(operation_name(block.steps[index].op));
    }

    // The compiled code against the interpreter. The compiler reads the code again at
    // each entry: a run that wrote to its code has no reference.
    let Some(compiled) = compiled else {
        return;
    };
    if fast_memory.0[code_offset..code_offset + 64] != memory.0[code_offset..code_offset + 64] {
        return;
    }
    // The code page has new code.
    compiled.jit.remove_page(0);
    let (mut jit_cpu, mut jit_memory) = (start.clone(), memory.clone());
    assert!(
        run_compiled(compiled, &block, total, &mut jit_cpu, &mut jit_memory),
        "{}",
        context()
    );
    assert_eq!(jit_cpu.regs, cpu.regs, "compiled: {}", context());
    assert_eq!(
        jit_cpu.cpsr.to_u32(),
        cpu.cpsr.to_u32(),
        "compiled: {}",
        context()
    );
    assert_eq!(jit_cpu.vfp.dregs, cpu.vfp.dregs, "compiled: {}", context());
    assert_eq!(
        jit_cpu.vfp.fpscr.bits(),
        cpu.vfp.fpscr.bits(),
        "compiled: {}",
        context()
    );
    assert!(jit_memory == fast_memory, "compiled, memory: {}", context());
}

#[test]
fn fast_interpreter_agrees_with_executor_on_random_code() {
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let mut memory = FlatMemory(
        (0..MEMORY_SIZE).map(|_| random.next() as u8).collect(),
        ExclusiveMonitor::new(),
    );
    let mut covered = HashSet::new();
    let no_page = [u64::MAX << 32, 0, 0, 0];
    let mut compiled = Jit::new().map(|jit| Compiled {
        jit,
        table: vec![no_page; 1 << jit::TLB_BITS],
        covered: HashSet::new(),
    });
    for _ in 0..1_000_000 {
        compare_random_block(&mut random, &mut memory, &mut covered, &mut compiled);
    }
    let mut covered: Vec<_> = covered.into_iter().collect();
    covered.sort();
    let all = [
        "AddPc",
        "Adr",
        "AluImm",
        "AluReg",
        "Bfc",
        "Bfi",
        "Branch",
        "BranchExchange",
        "BranchLink",
        "BranchLinkArm",
        "BranchLinkExchange",
        "BranchLinkToArm",
        "CacheOperation",
        "ClearExclusive",
        "Clz",
        "CompareBranch",
        "Extend",
        "FpAccumulate",
        "FpBinary",
        "FpCompare",
        "FpConvert",
        "FpElementFromCore",
        "FpElementToCore",
        "FpFlags",
        "FpFromCore",
        "FpFromCorePair",
        "FpImmediate",
        "FpLoad",
        "FpLoadLiteral",
        "FpLoadMultiple",
        "FpStore",
        "FpStoreMultiple",
        "FpToCore",
        "FpToCorePair",
        "FpUnary",
        "It",
        "Load",
        "LoadExclusive",
        "LoadLiteral",
        "LoadMultiple",
        "LoadPair",
        "LoadReg",
        "Mla",
        "Mls",
        "Movt",
        "Mul",
        "MulHalves",
        "Nop",
        "Pop",
        "Push",
        "Rbit",
        "Rev",
        "Sbfx",
        "Sdiv",
        "ShiftReg",
        "SimdDuplicate",
        "SimdFloatCompare",
        "SimdImmediate",
        "SimdLaneAdd",
        "SimdLaneAddl",
        "SimdLaneCeq",
        "SimdLaneCge",
        "SimdLaneCgt",
        "SimdLaneClt0",
        "SimdLaneMla",
        "SimdLaneMls",
        "SimdLaneMovl",
        "SimdLaneMul",
        "SimdLaneMull",
        "SimdLaneQadd",
        "SimdLaneQmovn",
        "SimdLaneQsub",
        "SimdLaneRshr",
        "SimdLaneRshrn",
        "SimdLaneShl",
        "SimdLaneShr",
        "SimdLaneSub",
        "SimdLoad",
        "SimdLogic",
        "SimdStore",
        "SimdStructure",
        "Smull",
        "Store",
        "StoreExclusive",
        "StoreMultiple",
        "StorePair",
        "StoreReg",
        "TableBranch",
        "ThreadRegister",
        "Ubfx",
        "Udiv",
        "Umaal",
        "Umlal",
        "Umull",
    ];
    assert_eq!(covered, all);

    // Each operation with a compiled form ran as compiled code.
    let Some(compiled) = compiled else {
        return;
    };
    let mut covered: Vec<_> = compiled.covered.into_iter().collect();
    covered.sort();
    let all = [
        "AddPc",
        "Adr",
        "AluImm",
        "AluReg",
        "Bfc",
        "Bfi",
        "Branch",
        "BranchExchange",
        "BranchLink",
        "BranchLinkArm",
        "BranchLinkExchange",
        "BranchLinkToArm",
        "CacheOperation",
        "Clz",
        "CompareBranch",
        "Extend",
        "FpAccumulate",
        "FpBinary",
        "FpCompare",
        "FpConvert",
        "FpElementFromCore",
        "FpElementToCore",
        "FpFlags",
        "FpFromCore",
        "FpFromCorePair",
        "FpImmediate",
        "FpLoad",
        "FpLoadLiteral",
        "FpLoadMultiple",
        "FpStore",
        "FpStoreMultiple",
        "FpToCore",
        "FpToCorePair",
        "FpUnary",
        "It",
        "Load",
        "LoadLiteral",
        "LoadMultiple",
        "LoadPair",
        "LoadReg",
        "Mla",
        "Mls",
        "Movt",
        "Mul",
        "MulHalves",
        "Nop",
        "Pop",
        "Push",
        "Rbit",
        "Rev",
        "Sbfx",
        "Sdiv",
        "ShiftReg",
        "SimdDuplicate",
        "SimdFloatCompare",
        "SimdImmediate",
        "SimdLaneAdd",
        "SimdLaneAddl",
        "SimdLaneCeq",
        "SimdLaneCge",
        "SimdLaneCgt",
        "SimdLaneClt0",
        "SimdLaneMla",
        "SimdLaneMls",
        "SimdLaneMovl",
        "SimdLaneMul",
        "SimdLaneMull",
        "SimdLaneQadd",
        "SimdLaneQmovn",
        "SimdLaneQsub",
        "SimdLaneRshr",
        "SimdLaneRshrn",
        "SimdLaneShl",
        "SimdLaneShr",
        "SimdLaneSub",
        "SimdLoad",
        "SimdLogic",
        "SimdStore",
        "SimdStructure",
        "Smull",
        "Store",
        "StoreMultiple",
        "StorePair",
        "StoreReg",
        "TableBranch",
        "ThreadRegister",
        "Ubfx",
        "Udiv",
        "Umaal",
        "Umlal",
        "Umull",
    ];
    assert_eq!(covered, all);
}

#[test]
fn it_block_gives_its_condition_to_the_next_instructions() {
    // ITE EQ; MOVS r0, #1 (no flags in the block); MOVS r0, #2; MOVS r1, #3.
    let mut code = [0u8; 16];
    for (index, halfword) in [0xbf0c_u16, 0x2001, 0x2002, 0x2103].into_iter().enumerate() {
        code[index * 2..index * 2 + 2].copy_from_slice(&halfword.to_le_bytes());
    }
    let block = Block::translate(&code[..8], 0, true);
    assert_eq!(block.len(), 4);
    let mut cpu = Armv7Cpu::new();
    cpu.cpsr.z = false;
    let mut memory = FlatMemory(vec![0; MEMORY_SIZE], ExclusiveMonitor::new());
    assert_eq!(
        block.run(0, 3, 0x8000, &mut cpu, &mut memory, &mut NoHook),
        (3, Exit::Done)
    );
    assert_eq!(
        (cpu.regs[0], cpu.regs[15], cpu.cpsr.it_state),
        (2, 0x8006, 0)
    );
    assert!(!cpu.cpsr.z);
    assert_eq!(
        block.run(0, 2, 0x8000, &mut cpu, &mut memory, &mut NoHook),
        (2, Exit::Done)
    );
    assert_eq!((cpu.regs[15], cpu.cpsr.it_state), (0x8004, 0x18));
}

#[test]
fn strex_stores_one_time_after_ldrex() {
    // LDREX r1, [r0]; STREX r2, r3, [r0]; STREX r4, r3, [r0] (T32).
    let mut code = [0u8; 12];
    let halfwords = [0xe850_u16, 0x1f00, 0xe840, 0x3200, 0xe840, 0x3400];
    for (index, halfword) in halfwords.into_iter().enumerate() {
        code[index * 2..index * 2 + 2].copy_from_slice(&halfword.to_le_bytes());
    }
    let block = Block::translate(&code, 0, true);
    let mut cpu = Armv7Cpu::new();
    cpu.regs[0] = MEMORY_BASE + 0x100;
    cpu.regs[3] = 0x1234_5678;
    let mut memory = FlatMemory(vec![0; MEMORY_SIZE], ExclusiveMonitor::new());
    let run = block.run(0, 3, CODE_PAGE, &mut cpu, &mut memory, &mut NoHook);
    assert_eq!(run, (3, Exit::Done));
    assert_eq!((cpu.regs[1], cpu.regs[2], cpu.regs[4]), (0, 0, 1));
    assert_eq!(memory.0[0x100..0x104], 0x1234_5678_u32.to_le_bytes());
}

#[test]
fn cache_operation_is_fast_only_in_a_privileged_mode() {
    // MCR p15, 0, r0, c7, c6, 1 (DCIMVAC, T32).
    let code = [0x07, 0xee, 0x36, 0x0f];
    let block = Rc::new(Block::translate(&code, 0, true));
    let mut memory = FlatMemory(vec![0; MEMORY_SIZE], ExclusiveMonitor::new());
    let mut jit = Jit::new();
    let table = vec![[u64::MAX << 32, 0, 0, 0]; 1 << jit::TLB_BITS];
    for (mode, count) in [(USER_MODE, 0), (0x13, 1)] {
        let mut cpu = Armv7Cpu::new();
        cpu.cpsr.t = true;
        cpu.cpsr.mode = mode;
        cpu.regs[15] = CODE_PAGE;
        let exit = if count == 0 {
            Exit::Slow(0)
        } else {
            Exit::Done
        };
        let run = block.run(0, 1, CODE_PAGE, &mut cpu.clone(), &mut memory, &mut NoHook);
        assert_eq!(run, (count, exit));

        let Some(jit) = &mut jit else {
            continue;
        };
        block.tier.set(Tier::Cold { runs: u32::MAX });
        let fp = FpControls {
            controls: fp_controls(&cpu),
            fpscr: 0,
        };
        let mut bytes = [0; PAGE_SIZE];
        bytes[..4].copy_from_slice(&code);
        let entry = jit.code(&block, CODE_PAGE, fp, || (0, bytes)).unwrap();
        let mut run = CompiledRun {
            context: Context {
                cpu: &mut cpu,
                jumps: jit.jumps(0, (0, 0)),
                reads: table.as_ptr().cast(),
                writes: table.as_ptr().cast(),
                budget: 1,
                site: std::ptr::null(),
                fp_controls: fp.controls,
                load: compiled_load,
                store: compiled_store,
            },
            memory: &mut memory,
            stop: false,
        };
        // SAFETY: the code is that of the block at the PC; the table has no direct entry.
        let left = unsafe { jit.run(&mut run.context, entry) };
        // In User mode the code leaves before the instruction.
        assert_eq!(left.map(|(_, index)| index), (count == 0).then_some(0));
        assert_eq!(1 - run.context.budget, count as i64);
    }
}

#[test]
fn block_stops_before_an_instruction_across_the_page_end() {
    // MOVS r0, #1 then the first half of a BL.
    let code = [0x01, 0x20, 0x00, 0xf0];
    let block = Block::translate(&code, 0, true);
    assert_eq!((block.len(), block.offset(1)), (1, 2));
    assert!(Block::translate(&code[2..], 2, true).is_empty());
}
