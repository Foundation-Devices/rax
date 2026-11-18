//! A compiler from blocks to the code of the host.
//!
//! This file has what is the same for each host: the blocks with code, the branches
//! between the code of one page, the table of the jumps, the flags that an instruction
//! must keep. A code generator (`aarch64`, `x86_64`) makes the code of one block. The
//! compiled code has only the cases that need no call. Each other case leaves the code before its
//! instruction, with the registers as before it, and the caller runs the instruction with
//! the interpreter.

use std::collections::HashMap;
use std::mem::offset_of;
use std::rc::Rc;

use rax::arm::Armv7Cpu;
use rax::arm::decoder::ShiftType;
use rax::arm::vfp::VfpState;

use super::{Alu, Block, COND_ALWAYS, Carry, Op, PC, Step};

#[cfg_attr(all(target_arch = "x86_64", target_os = "linux"), allow(dead_code))]
mod aarch64;
#[cfg_attr(
    not(all(target_arch = "x86_64", target_os = "linux")),
    allow(dead_code)
)]
mod x86_64;
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
use aarch64 as host;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
use x86_64 as host;

// Flags, as the bits 31:28 of NZCV.
const N: u8 = 8;
const Z: u8 = 4;
const C: u8 = 2;
const V: u8 = 1;
const ALL_FLAGS: u8 = N | Z | C | V;

const CODE_SIZE: usize = 64 << 20;
const JUMP_BITS: u32 = 12;
/// The runs of a block with the interpreter before the compilation.
const HOT_RUNS: u32 = 8;
/// A block with an FP instruction that leaves with each state of the FP controls gets
/// no more than these compilations.
const MAX_FP_COMPILATIONS: u8 = 4;

// The translation tables of the caller. A table has one entry for each value of the
// low bits of the virtual page number. The compiled code reads two fields of an entry.
pub const TLB_BITS: u32 = 8;
/// The log2 of the size of an entry.
pub const TLB_ENTRY_SHIFT: u32 = 5;
/// The offset of a 32-bit field: the virtual page when the accesses to the page can go
/// to the memory of the host with no call, else a number that is not a page address.
pub const TLB_DIRECT_PAGE: usize = 4;
/// The offset of a field of the size of a pointer: the address in the host of the
/// virtual page, minus the virtual address of the page.
pub const TLB_ADDEND: usize = 16;
const PAGE_SIZE: u32 = 4096;

const REGS: usize = offset_of!(Armv7Cpu, regs);
const CPSR: usize = offset_of!(Armv7Cpu, cpsr);
const DREGS_OFFSET: usize = offset_of!(Armv7Cpu, vfp) + offset_of!(VfpState, dregs);
/// The FPSCR is one 32-bit word (`Fpscr`).
const FPSCR: usize = offset_of!(Armv7Cpu, vfp) + offset_of!(VfpState, fpscr);

/// The code of one physical page of the guest.
#[derive(Default)]
struct PageCode {
    /// The code of the guest addresses (T bit in bit 0) of the page.
    entries: HashMap<u32, usize>,
    /// For a guest address with no code: the addresses of the branches of other code
    /// of the page that go to it.
    links: HashMap<u32, Vec<usize>>,
}

/// The compiled code of a guest address, for the lookup in the compiled code.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Jump {
    /// The guest address with the T bit in bit 0.
    key: u32,
    code: usize,
}

struct JumpTable {
    stamp: (u64, u64, u32),
    jumps: Box<[Jump]>,
}

/// What the compiled code gets from its caller and gives back.
#[repr(C)]
pub struct Context {
    pub cpu: *mut Armv7Cpu,
    pub jumps: *const Jump,
    /// The translation tables for the reads and for the writes.
    pub reads: *const u8,
    pub writes: *const u8,
    /// The instructions that can run. On return, the ones that did not run.
    pub budget: i64,
    /// On return: the block and the instruction where the code left, or null when the
    /// code left to a guest address with no compiled code.
    pub site: *const u32,
    /// `fp_controls` of the CPU.
    pub fp_controls: u64,
    /// The load and the store that the code calls for an access with no direct entry.
    /// The arguments are this context, the guest address, the value of a store, and
    /// the size of the access in the bits 7:0 with, in the bits above, the number of
    /// the instructions from this one to the end of its block. `budget` is that of
    /// the place of the call, and the function can change it. The result is the value
    /// of a load with one of the states below.
    pub load: extern "C" fn(*mut Context, u32, u32) -> u64,
    pub store: extern "C" fn(*mut Context, u32, u32, u32) -> u64,
}

/// The access is complete, and the code must leave after the instruction.
pub const ACCESS_STOP: u64 = 1 << 32;
/// No access: the interpreter must run the instruction.
pub const ACCESS_RETRY: u64 = 2 << 32;

/// The state of the FP controls of the guest at the time of a compilation.
#[derive(Clone, Copy)]
pub struct FpControls {
    /// `fp_controls` of the CPU: the code is only for this value.
    pub controls: u64,
    pub fpscr: u32,
}

/// The state of a block for the compiler.
#[derive(Clone, Copy)]
pub enum Tier {
    Cold {
        runs: u32,
    },
    /// The first instruction of the block has no compiled form.
    Declined {
        generation: u32,
    },
    /// Code for the block at the virtual page `page`.
    Code {
        entry: usize,
        page: u32,
        generation: u32,
    },
}

#[derive(Debug, Default)]
pub struct Stats {
    pub compiled_blocks: u64,
    /// Blocks compiled again for a different virtual page.
    pub moved_blocks: u64,
    pub code_bytes: usize,
    pub flushes: u64,
}

#[derive(Clone, Copy)]
struct Stubs {
    enter: usize,
    /// Leaves to the guest address in T0 (T bit in bit 0).
    leave_to_address: usize,
    /// Leaves at the instruction that the two words after the call give.
    leave_at_instruction: usize,
    /// Call `Context::load` and `Context::store`, with the address in T0, the value in
    /// T1 and the size word in T2. The result is in T0, its state in T1.
    call_load: usize,
    call_store: usize,
}

pub struct Jit {
    memory: *mut u8,
    used: usize,
    stubs_end: usize,
    stubs: Stubs,
    /// The blocks with code and the virtual page of the code, by the number that the
    /// code gives when it leaves.
    blocks: Vec<(Rc<Block>, u32)>,
    /// Code of an older generation is not valid.
    generation: u32,
    tables: Vec<Option<JumpTable>>,
    /// The code by the place of its page in the memory of the guest.
    pages: HashMap<usize, PageCode>,
    pub stats: Stats,
}

#[cfg(all(feature = "jit", target_arch = "aarch64", target_os = "macos"))]
unsafe extern "C" {
    fn sys_icache_invalidate(start: *mut libc::c_void, length: usize);
}

/// Writes code.
///
/// # Safety
/// The range is in the mapping of a `Jit`, and no code of the range runs now.
#[cfg(all(feature = "jit", target_arch = "aarch64", target_os = "macos"))]
unsafe fn write_code(start: *mut u8, code: &[u8]) {
    // SAFETY: the range (caller). The mapping takes writes of this thread only between
    // the two calls, and the host keeps old code until the last call.
    unsafe {
        libc::pthread_jit_write_protect_np(0);
        std::ptr::copy_nonoverlapping(code.as_ptr(), start, code.len());
        libc::pthread_jit_write_protect_np(1);
        sys_icache_invalidate(start.cast(), code.len());
    }
}

/// Makes the instruction cache of the host read new code from the memory.
#[cfg(all(feature = "jit", target_arch = "aarch64", target_os = "linux"))]
unsafe fn flush_code(start: usize, end: usize) {
    let cache_type: usize;
    // SAFETY: the cache operations by address have no effect on the content of the
    // memory, and Linux lets a process do them and read CTR_EL0.
    unsafe {
        std::arch::asm!("mrs {}, ctr_el0", out(reg) cache_type);
        // The sizes of the smallest data and instruction cache lines, in bytes.
        let (data_line, instruction_line) =
            (4 << (cache_type >> 16 & 0xf), 4 << (cache_type & 0xf));
        for line in (start & !(data_line - 1)..end).step_by(data_line) {
            std::arch::asm!("dc cvau, {}", in(reg) line);
        }
        std::arch::asm!("dsb ish");
        for line in (start & !(instruction_line - 1)..end).step_by(instruction_line) {
            std::arch::asm!("ic ivau, {}", in(reg) line);
        }
        std::arch::asm!("dsb ish", "isb");
    }
}

/// The pages of the code are not writable: this makes the pages of the range writable
/// for the time of the copy.
#[cfg(all(
    feature = "jit",
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
unsafe fn write_code(start: *mut u8, code: &[u8]) {
    let length = code.len();
    // SAFETY: a call with no effect.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let first = start as usize & !(page - 1);
    let pages = (start as usize + length - first).next_multiple_of(page);
    // SAFETY: the range (caller). The mapping of a `Jit` starts and ends at a page
    // limit, thus these pages are pages of the mapping.
    unsafe {
        let writable = libc::mprotect(first as *mut _, pages, libc::PROT_READ | libc::PROT_WRITE);
        assert_eq!(writable, 0, "mprotect of the code");
        std::ptr::copy_nonoverlapping(code.as_ptr(), start, code.len());
        let executable = libc::mprotect(first as *mut _, pages, libc::PROT_READ | libc::PROT_EXEC);
        assert_eq!(executable, 0, "mprotect of the code");
        // An x86-64 host sees a write to code with no call.
        #[cfg(target_arch = "aarch64")]
        flush_code(start as usize, start as usize + length);
    }
}

/// `Jit::new` gives no compiler on this host.
#[cfg(not(all(
    feature = "jit",
    any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", target_os = "linux")
    )
)))]
unsafe fn write_code(_start: *mut u8, _code: &[u8]) {
    unreachable!();
}

/// The flags that a condition reads.
fn condition_flags(cond: u8) -> u8 {
    match cond >> 1 {
        0 => Z,
        1 => C,
        2 => N,
        3 => V,
        4 => C | Z,
        5 => N | V,
        6 => N | Z | V,
        _ => 0,
    }
}

fn is_arithmetic(kind: Alu) -> bool {
    matches!(
        kind,
        Alu::Add | Alu::Adc | Alu::Sub | Alu::Sbc | Alu::Rsb | Alu::Cmp | Alu::Cmn
    )
}

/// True when the compiled code of the operation never leaves the block.
fn stays(op: Op) -> bool {
    matches!(
        op,
        Op::Nop
            | Op::It { .. }
            | Op::AluImm { .. }
            | Op::AluReg { .. }
            | Op::Movt { .. }
            | Op::Adr { .. }
            | Op::AddPc { .. }
            | Op::Extend { .. }
            | Op::Clz { .. }
            | Op::Rbit { .. }
            | Op::Rev { .. }
            | Op::Ubfx { .. }
            | Op::Sbfx { .. }
            | Op::Bfi { .. }
            | Op::Bfc { .. }
            | Op::Mul { .. }
            | Op::Mla { .. }
            | Op::Mls { .. }
            | Op::Umull { .. }
            | Op::Umlal { .. }
            | Op::Umaal { .. }
            | Op::Smull { .. }
            | Op::Udiv { .. }
            | Op::Sdiv { .. }
            | Op::FpBinary { .. }
            | Op::FpAccumulate { .. }
            | Op::FpUnary { .. }
            | Op::FpCompare { .. }
            | Op::FpImmediate { .. }
            | Op::FpConvert { .. }
            | Op::FpFromCore { .. }
            | Op::FpToCore { .. }
            | Op::FpFromCorePair { .. }
            | Op::FpToCorePair { .. }
            | Op::FpFlags
            | Op::SimdLogic { .. }
            | Op::SimdLane { .. }
            | Op::SimdImmediate { .. }
            | Op::MulHalves { .. }
            | Op::FpElementFromCore { .. }
            | Op::FpElementToCore { .. }
            | Op::SimdDuplicate { .. }
    )
}

/// The flags that the operation reads and the flags that it always writes.
fn flag_effect(op: Op) -> (u8, u8) {
    match op {
        Op::AluImm {
            kind, flags, carry, ..
        } => {
            let reads = if matches!(kind, Alu::Adc | Alu::Sbc) {
                C
            } else {
                0
            };
            let writes = match () {
                _ if !flags => 0,
                _ if is_arithmetic(kind) => ALL_FLAGS,
                _ if carry == Carry::Keep => N | Z,
                _ => N | Z | C,
            };
            (reads, writes)
        }
        Op::AluReg {
            kind,
            flags,
            shift,
            amount,
            ..
        } => {
            let mut reads = if matches!(kind, Alu::Adc | Alu::Sbc) {
                C
            } else {
                0
            };
            if shift == ShiftType::RRX {
                reads |= C;
            }
            let writes = match () {
                _ if !flags => 0,
                _ if is_arithmetic(kind) => ALL_FLAGS,
                _ if shift == ShiftType::LSL && amount == 0 => N | Z,
                _ => N | Z | C,
            };
            (reads, writes)
        }
        Op::Mul { flags: true, .. } => (0, N | Z),
        Op::FpFlags => (0, ALL_FLAGS),
        _ => (0, 0),
    }
}

/// For each instruction, the flags that an instruction after it or the caller can read.
/// The caller reads all the flags at each place where the code can leave.
fn live_flags(steps: &[Step], has_code: impl Fn(usize) -> bool) -> Vec<u8> {
    let mut live = ALL_FLAGS;
    let mut after = vec![0; steps.len()];
    for (index, step) in steps.iter().enumerate().rev() {
        after[index] = live;
        let (reads, writes) = flag_effect(step.op);
        live = match () {
            _ if !stays(step.op) || !has_code(index) => ALL_FLAGS,
            _ if step.cond != COND_ALWAYS => live | reads | condition_flags(step.cond),
            _ => live & !writes | reads,
        };
    }
    after
}

/// Branches to guest addresses with no code: the address (T bit in bit 0) and the offset
/// of the branch in the code.
type Links = Vec<(u32, usize)>;

/// What a code generator gets for one block.
struct Unit<'a> {
    /// The address of the code.
    entry: usize,
    block: &'a Block,
    /// The virtual address of the page of the block, and its bytes.
    page: u32,
    code: &'a [u8],
    fp: FpControls,
    stubs: Stubs,
    /// The number of the block in the words after a call of `leave_at_instruction`.
    number: u32,
    /// The code of the guest addresses in the physical page of the block.
    entries: &'a HashMap<u32, usize>,
}

fn always_jumps(op: Op) -> bool {
    matches!(
        op,
        Op::Branch { .. }
            | Op::BranchLink { .. }
            | Op::BranchLinkArm { .. }
            | Op::BranchLinkToArm { .. }
            | Op::BranchExchange { .. }
            | Op::BranchLinkExchange { .. }
            | Op::TableBranch { .. }
    ) || matches!(op, Op::Pop { mask } if mask & 1 << PC != 0)
}

impl Jit {
    /// None when the crate has no feature `jit`, or the host is not AArch64 macOS,
    /// AArch64 Linux or x86-64 Linux, or gives no memory for code, or has FP controls
    /// that are not the ones of the compiled FP code.
    #[cfg(not(all(
        feature = "jit",
        any(
            all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
            all(target_arch = "x86_64", target_os = "linux")
        )
    )))]
    pub fn new() -> Option<Jit> {
        None
    }

    #[cfg(all(
        feature = "jit",
        any(
            all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
            all(target_arch = "x86_64", target_os = "linux")
        )
    ))]
    pub fn new() -> Option<Jit> {
        #[cfg(target_arch = "aarch64")]
        {
            let fpcr: u64;
            // SAFETY: a read of a register with no effect.
            unsafe { std::arch::asm!("mrs {}, fpcr", out(reg) fpcr) };
            // Round to nearest, no flush to zero, no default NaN, no trap.
            if fpcr != 0 {
                return None;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut mxcsr = 0_u32;
            // SAFETY: a store of a register to this variable.
            unsafe { std::arch::asm!("stmxcsr [{}]", in(reg) &raw mut mxcsr, options(nostack)) };
            // The exceptions are in the low 6 bits.
            if mxcsr & !0x3f != x86_64::MXCSR_DEFAULT {
                return None;
            }
        }
        // macOS: a MAP_JIT mapping, which each thread can write or run (`write_code`).
        #[cfg(target_os = "macos")]
        let (flags, protection) = (
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        );
        #[cfg(target_os = "linux")]
        let (flags, protection) = (
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            libc::PROT_READ | libc::PROT_EXEC,
        );
        // SAFETY: a new mapping, not at a fixed address.
        let memory =
            unsafe { libc::mmap(std::ptr::null_mut(), CODE_SIZE, protection, flags, -1, 0) };
        if memory == libc::MAP_FAILED {
            return None;
        }
        let memory: *mut u8 = memory.cast();
        let (code, stubs) = host::stubs(memory as usize);
        let mut jit = Jit {
            memory,
            used: 0,
            stubs_end: 0,
            stubs,
            blocks: Vec::new(),
            generation: 0,
            tables: Vec::new(),
            pages: HashMap::new(),
            stats: Stats::default(),
        };
        jit.write(&code);
        jit.stubs_end = jit.used;
        Some(jit)
    }

    /// Puts code at the next free place.
    fn write(&mut self, code: &[u8]) {
        assert!(self.used + code.len() <= CODE_SIZE);
        // SAFETY: the range is in the mapping and no code of it runs now.
        unsafe { write_code(self.memory.add(self.used), code) };
        self.used += code.len();
        self.stats.code_bytes = self.used;
    }

    /// Removes all the code and the blocks of the code. No code runs now.
    pub fn flush(&mut self) {
        self.used = self.stubs_end;
        self.blocks.clear();
        self.pages.clear();
        self.generation += 1;
        self.stats.flushes += 1;
    }

    /// The guest wrote to the page at the place `physical` of its memory: the code of
    /// the page has no more use. The caller changes the stamp of `Jit::jumps`.
    pub fn remove_page(&mut self, physical: usize) {
        self.pages.remove(&physical);
    }

    /// Writes a branch to `target` at `site`.
    fn link(&mut self, site: usize, target: usize) {
        let (address, branch) = host::link(site, target);
        // SAFETY: the site is a branch in code of this mapping that does not run now.
        unsafe { write_code(address as *mut u8, &branch) };
    }

    /// The code of the block at the virtual page `page`, or None when the block has none
    /// (now). `source` gives the place of the page in the memory of the guest and the
    /// bytes of the page.
    pub fn code(
        &mut self,
        block: &Rc<Block>,
        page: u32,
        fp: FpControls,
        source: impl FnOnce() -> (usize, [u8; PAGE_SIZE as usize]),
    ) -> Option<usize> {
        match block.tier.get() {
            Tier::Code {
                entry,
                page: code_page,
                generation,
            } if generation == self.generation => {
                if code_page == page {
                    return Some(entry);
                }
                self.stats.moved_blocks += 1;
            }
            Tier::Declined { generation } if generation == self.generation => return None,
            Tier::Cold { runs } if runs < HOT_RUNS => {
                block.tier.set(Tier::Cold { runs: runs + 1 });
                return None;
            }
            _ => {}
        }
        if self.used + block.len() * host::INSTRUCTION_CODE > CODE_SIZE {
            self.flush();
        }
        let entry = self.memory as usize + self.used;
        let (physical, bytes) = source();
        let page_code = self.pages.entry(physical).or_default();
        let unit = Unit {
            entry,
            block,
            page,
            code: &bytes,
            fp,
            stubs: self.stubs,
            number: self.blocks.len() as u32,
            entries: &page_code.entries,
        };
        let Some((code, links)) = host::compile(unit) else {
            block.tier.set(Tier::Declined {
                generation: self.generation,
            });
            return None;
        };
        self.write(&code);
        // The branches of older code to this address, and the ones of this code.
        let key = page.wrapping_add(block.offset(0)) | u32::from(block.thumb);
        let page_code = self.pages.get_mut(&physical).unwrap();
        page_code.entries.insert(key, entry);
        for (target, branch) in links {
            page_code
                .links
                .entry(target)
                .or_default()
                .push(entry + branch);
        }
        for site in page_code.links.remove(&key).unwrap_or_default() {
            self.link(site, entry);
        }
        self.blocks.push((block.clone(), page));
        self.stats.compiled_blocks += 1;
        block.tier.set(Tier::Code {
            entry,
            page,
            generation: self.generation,
        });
        Some(entry)
    }

    /// The compiled code left at the instruction `index` and the interpreter ran it. When
    /// that is an FP instruction with a fast operation, the executor accepts it now, and
    /// the next compilation of the block has code for it.
    #[inline]
    pub fn left_at(&mut self, block: &Block, index: usize) {
        let compilations = block.fp_compilations.get();
        if block.steps[index].op.is_fp()
            && host::has_fp_code(block.steps[index].op)
            && block.admitted[index].get() != 0
            && compilations < MAX_FP_COMPILATIONS
        {
            block.fp_compilations.set(compilations + 1);
            block.tier.set(Tier::Cold { runs: HOT_RUNS });
        }
    }

    /// The table of the compiled code for the guest addresses of one address space.
    /// `stamp` changes when an address can have other code.
    #[inline]
    pub fn jumps(&mut self, space: usize, stamp: (u64, u64)) -> *mut Jump {
        let stamp = (stamp.0, stamp.1, self.generation);
        if self.tables.len() <= space {
            self.tables.resize_with(space + 1, || None);
        }
        // An address with no code goes to the code that leaves to an address.
        let empty = Jump {
            key: 0,
            code: self.stubs.leave_to_address,
        };
        let table = self.tables[space].get_or_insert_with(|| JumpTable {
            stamp,
            jumps: vec![empty; 1 << JUMP_BITS].into(),
        });
        if table.stamp != stamp {
            table.stamp = stamp;
            table.jumps.fill(empty);
        }
        table.jumps.as_mut_ptr()
    }

    /// Puts the code of a guest address (T bit in bit 0) in a table of `Jit::jumps`.
    ///
    /// # Safety
    /// `jumps` is from `Jit::jumps`, and `code` is from `Jit::code` for the page of
    /// `address` and for a block that is in the memory of the guest at this address in
    /// the address space of the table.
    #[inline]
    pub unsafe fn set_jump(&mut self, jumps: *mut Jump, address: u32, code: usize) {
        // SAFETY: the table has 1 << JUMP_BITS entries.
        unsafe {
            *jumps.add((address >> 1) as usize & ((1 << JUMP_BITS) - 1)) =
                Jump { key: address, code }
        };
    }

    /// Runs compiled code. Returns the block and the instruction where the code left,
    /// with the PC and the IT state of this instruction, or None when the code left to
    /// the PC with no IT state.
    ///
    /// # Safety
    /// `code` is from `Jit::code` for the page of the PC, and for a block that is in
    /// the memory of the guest at the PC. `context.jumps` is from `Jit::jumps` with a
    /// stamp for the address space of now. `context.reads` and `context.writes` are
    /// tables of 1 << TLB_BITS entries with the fields that the TLB constants give,
    /// and each direct entry gives memory that the guest can read, or write, at the
    /// addresses of its page. `context.cpu` is from a reference with no other use now.
    pub unsafe fn run(&mut self, context: &mut Context, code: usize) -> Option<(Rc<Block>, usize)> {
        // SAFETY: the stub has this signature, and it keeps the registers that the
        // calling convention of the host gives to the callee.
        let enter: extern "C" fn(*mut Context, usize) =
            unsafe { std::mem::transmute(self.stubs.enter) };
        enter(context, code);
        if context.site.is_null() {
            return None;
        }
        // SAFETY: the two words after the call are the numbers of the block and of the
        // instruction. An x86-64 host has them at each address.
        let (number, index) = unsafe {
            (
                context.site.read_unaligned(),
                context.site.add(1).read_unaligned() as usize,
            )
        };
        let (block, page) = self.blocks[number as usize].clone();
        // SAFETY: the caller made the pointer from a reference that no other code uses now.
        let cpu = unsafe { &mut *context.cpu };
        cpu.regs[15] = page.wrapping_add(block.offset(index));
        cpu.cpsr.t = block.thumb;
        cpu.cpsr.it_state = block.it_state(index);
        Some((block, index))
    }
}
