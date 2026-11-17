# rax-aarch32-system

A block execution engine for an ARMv7-A system emulator (Cortex-A class: A32 and Thumb-2,
VFP, Advanced SIMD, the short-descriptor MMU). It runs guest code faster than one
`Executor` call for each instruction, with the same results: the `Executor` of rax stays
the reference, and it runs each instruction that the engine has no fast operation for.

The engine has no machine: no peripheral, no exception entry, no clock. A machine gives
its memory and its time to the engine through the interfaces below.

## Parts

| file | what |
|---|---|
| `src/lib.rs` | `Block`: the decoded instructions of one page from an entry address, to an instruction that always leaves (64 at most). `Block::run`: the fast interpreter for the common Thumb, A32, VFP and Advanced SIMD operations. |
| `src/cache.rs` | `BlockCache`: the blocks by their place in the memory of the machine, links from virtual addresses, removal of the blocks of a written page. |
| `src/tlb.rs` | `Tlb`: a software TLB with one set of entries for each MMU configuration (an address space change keeps the entries). An entry can be direct: the access goes to host memory with no call. |
| `src/jit.rs` | `Jit`: a compiler from hot blocks to AArch64 code. Feature `jit` (default), AArch64 macOS and Linux hosts. |
| `src/a64.rs` | The AArch64 assembler of the compiler. |

## Interface

- `Memory` (trait of the machine): `load`/`store` for an access with no side effect (the
  machine answers from `Tlb::direct`), and `load_slow`/`store_slow` for an access with all
  its effects (translation, peripherals, faults).
- `tlb::PhysicalMemory` (trait of the machine): a word of a translation table, and how the
  accesses go to a physical page (`Page::Direct` with a host address, `Page::Memory`,
  `Page::Other`).
- Instruction budget: `Block::run(start, limit, ...)` runs at most `limit` instructions and
  returns the count; compiled code gets `jit::Context::budget` and gives back what is left.
  The machine makes the budget from its next timer event, so an event comes at the same
  instruction with each engine.
- Exits: `Exit::Slow(index)` (the machine runs the instruction with the `Executor`),
  `Exit::Fault(index)` (a slow access gave a fault; the registers are those before the
  instruction), `Exit::Done`. Compiled code leaves to a guest address, or before one
  instruction (`Jit::run` returns the block and the index): the machine runs that
  instruction with `Block::run` and a limit of 1.
- Invalidation: `BlockCache::remove_pages` and `Jit::remove_page` after a write to a page
  with code; `Tlb::protect_code` when a page gets its first block (a write to it must not
  be direct); `Tlb::set_mmu` and `Tlb::flush` for the CP15 writes. `Tlb::mapping` and
  `BlockCache::removals` make the stamp of `Jit::jumps`.
- FP: `fp_controls(cpu)` is a number for the state of the FP access controls. The machine
  tells a block when the `Executor` completed an FP instruction with this number
  (`Block::set_admitted`); the fast operations and the compiled code run the instruction
  only then, with no access check.
- Slow accesses of compiled code: `jit::Context::load` and `store`, two functions of the
  machine (see the comment of the fields for the budget rules).

`prime-rax` (an emulator of the Passport Prime, not in this repository) is the first user:
its `machine.rs` and `bus.rs` show the calls in order.

## How the compiled code keeps the state exact

Guest r0 to r14 are host x0 to x14 and the guest flags are the host flags. The code has only
the cases that need no call: an access through a direct TLB entry, a shift amount below
32, VFP arithmetic with round to nearest and no flush to zero or default NaN, the FP
controls of the compilation. For each other case it leaves before the instruction, with the
registers as before it. The start of a block subtracts its length from the budget, and
each exit in the block adds back the instructions that did not run.

## Tests

`cargo test --release -p rax-aarch32-system`: 1 million random Thumb and A32 blocks.
The fast interpreter runs against the `Executor` (registers, flags, VFP registers, FPSCR,
memory), and on AArch64 macOS the compiled code runs against the fast interpreter, entered
as a machine does (a new block at each entry, calls for half of the pages). The test fails
when an operation did not run in an engine that has it. The dev profile also passes (it
found two overflows of the `Executor` that only a build with overflow checks sees).

`cargo test --release -p rax-aarch32-system --no-default-features` runs the same test with
no compiler, as on another host. A second test runs a CP15 cache operation in User mode
and in a privileged mode in the two engines: the random blocks have a privileged mode only.

The system-level tests are in `prime-rax`: a boot of an unmodified OS to its first screens,
compared with the one-instruction interpreter (a hash of the registers at each 4096
instructions, UART text, pictures) and with QEMU.

## Limits

- The compiler is for AArch64 macOS (MAP_JIT) and AArch64 Linux (the code pages are
  writable only for the time of a write). Other hosts have the interpreter only.
  Checked: `cargo zigbuild --tests` for the musl and gnu targets of AArch64 and x86-64
  Linux and for `x86_64-pc-windows-gnu`; a run of the two musl test binaries in a Linux
  AArch64 container (the x86-64 one under emulation). The Windows build was not run.
- Compiled code is for one virtual page of one physical page; a second virtual address of
  the same code compiles it again.
- ARMv7-A without LPAE, big-endian data, ThumbEE or Jazelle. Short vectors and trapped FP
  exceptions always go to the `Executor`.
- The instructions with no fast operation are about 50 times slower: see `Op::Slow` in
  `src/lib.rs` for what has one.
