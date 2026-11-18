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
| `src/jit.rs` | `Jit`: a compiler from hot blocks to the code of the host. Feature `jit` (default); AArch64 macOS, AArch64 Linux and x86-64 Linux hosts. This file has the part that is the same for each host: the blocks with code, the branches between the code of one page, the table of the jumps, the flags that an instruction must keep. |
| `src/jit/aarch64.rs`, `src/a64.rs` | The code generator for AArch64 and its assembler. |
| `src/jit/x86_64.rs`, `src/x64.rs` | The code generator for x86-64 (System V calling convention) and its assembler. |

## Interface

- `Memory` (trait of the machine): `load`/`store` for an access with no side effect (the
  machine answers from `Tlb::direct`), `load_slow`/`store_slow` for an access with all
  its effects (translation, peripherals, faults), and `exclusive_monitor`: the monitor of
  LDREX and STREX, which the machine also gives to the `Executor` and clears at each
  exception entry. The compiled code has no exclusive access: it leaves to the interpreter.
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
  instruction with `Block::run` and a limit of 1, but not after a write to a code page
  in the same run: the block is old then, and the machine gets the block at the PC again.
- Invalidation: `BlockCache::remove_pages` and `Jit::remove_page` after a write to a page
  with code; `Tlb::protect_code` when a page gets its first block (a write to it must not
  be direct); `Tlb::set_mmu` and `Tlb::flush` for the CP15 writes. `Tlb::mapping` and
  `BlockCache::removals` make the stamp of `Jit::jumps`.
- Memory limit: when `BlockCache::is_full` (it made `cache::MAX_BLOCKS` blocks), the
  machine calls `BlockCache::clear` and `Jit::flush` before it gets the next block. The
  code buffer of the compiler has 64 MiB and `Jit::code` flushes it when it is full.
- FP: `fp_controls(cpu)` is a number for the state of the FP access controls. The machine
  tells a block when the `Executor` completed an FP instruction with this number
  (`Block::set_admitted`); the fast operations and the compiled code run the instruction
  only then, with no access check.
- Slow accesses of compiled code: `jit::Context::load` and `store`, two functions of the
  machine (see the comment of the fields for the budget rules).

`prime-rax` (an emulator of the Passport Prime, not in this repository) is the first user:
its `machine.rs` and `bus.rs` show the calls in order.

## How the compiled code keeps the state exact

On AArch64, guest r0 to r14 are host x0 to x14 and the guest flags are the host flags. On
x86-64, r0 to r6, r8 and the SP are in host registers and the other registers are in
the CPU state; the guest flags are in RAX in the form of LAHF and SETO (N, Z and C in AH,
with C and not the borrow of x86; V in AL). The flags of the host are a copy of them only
from the instruction that made them to the next instruction that changes the flags of the
host, and the code generator knows which flags the copy has: a condition reads the copy,
or loads the flags from RAX first.

The code has only the cases that need no call: an access through a direct TLB entry, a shift amount below
32, VFP arithmetic with round to nearest and no flush to zero or default NaN, the FP
controls of the compilation. For each other case it leaves before the instruction, with the
registers as before it. On x86-64 the VFP arithmetic is that of SSE, and the code leaves
before the instruction after a result that is not sure to be the result of the guest with
its flags: a NaN, an infinity, a denormal number or one with the smallest exponent, an
invalid operation, an underflow, a conversion out of the range of 32 bits. It finds them
from the result and the operands, with no read of the MXCSR. At such an exit the MXCSR
keeps the exceptions of the instruction, so the cumulative flags of the FPSCR can have
its division by zero, overflow or inexact result before the interpreter runs it and sets
the same flags. Of the Advanced
SIMD operations on elements, x86-64 has code (SSE2) only for VADD, VSUB, VCEQ, VCGT, VCLT
with 0, VSHR, VMUL, VMLA and VMLS of 16-bit elements, and the compares of FP numbers; the interpreter runs the others and VLD3/VLD4/VST3/VST4. The start of a block subtracts its length from the budget, and
each exit in the block adds back the instructions that did not run.

On x86-64 the code of a block has two parts. The code in sequence is at the next free
place of the code memory. The cold code (the calls for an access with no direct entry, the
exits, the look in the jump table of a branch to an address with no code yet) goes down
from the end of the memory: with it between the blocks, the blocks of 95 % of the
instructions of a KeyOS boot are on 1650 pages of 4 KiB, and on 1240 without it.

## Tests

All the commands are for the directory of rax. The tests are in `src/tests.rs`,
`src/tests/` and `src/a64.rs`.

### Quick tier

`cargo test --release -p rax-aarch32-system`: 42 tests in about 30 s on an Apple
M-series host (110 s of CPU time); 41 on an x86-64 host, where `tests/fp.rs` has no model. `--no-default-features` runs the same tests with no
compiler, as on another host. The dev profile passes too (its overflow checks see what
a release build does not): there `RAX_SOAK_SECONDS=40` gives each random test a time
that its count of cases does not.

A test machine (`tests/machine.rs`) makes the calls of a real machine in the same order
and runs each program three times: with the `Executor` only (the reference), with the
fast interpreter, and with the compiled code. The registers, the banks, the VFP state,
the memory, the exceptions and a hash of the registers at each event are the same, or
the test fails. A page of its memory can need a call for each access, give an abort, or
be a device: a read of the device gives the count of the instructions before it, and a
write moves the next event.

| file | what |
|---|---|
| `tests.rs` | 1 million random blocks, fast interpreter and compiled code against the `Executor`; each operation ran in each engine. |
| `tests/blocks.rs` | Short random programs, also across a page end, with pointers to each kind of page and events at random places. Instructions with random bits: each one with a fast operation runs as the `Executor`. |
| `tests/engine.rs` | Programs with loops, calls to the other instruction set, IT blocks, table branches, literals, stores to their own instructions, exceptions with handlers, interrupt requests, CPS to each mode. An event at each place of the blocks, the compilation at the ninth run, two addresses with one place in the jump table, a full code buffer, a block cache that is full after five blocks. |
| `tests/mmu.rs` | Two address spaces with other code at the same address, a new table entry with a TLB flush, a second address of the same code, DACR, a page with no write permission, User mode. |
| `tests/oracle.rs` | Integer instructions (A32 and T32) against a model from the pseudocode of the architecture manual: data processing with each shift, multiplies, saturating and parallel arithmetic, bit fields, extends, divides. |
| `tests/memory.rs` | Loads and stores against a model: each addressing mode with its writeback, LDRD and STRD, LDM and STM in the four orders, PUSH and POP, the 16-bit forms, loads from the PC, loads of the PC, the alignment rules. |
| `tests/fp.rs` | VFP instructions against the FP unit of an AArch64 host: arithmetic, fused and not fused multiply-add, compares, conversions with integers and fixed-point numbers, with each rounding mode, flush to zero, default NaN, NaN and denormal operands, and the FPSCR flags. |
| `tests/neon.rs` | Advanced SIMD instructions on integer elements against a model: 28 operations with three registers of the same length, 16 shifts by an immediate value, 16 operations with two lengths, 26 operations with two registers; each size, D and Q, with QC. |
| `tests/system.rs` | Each condition with each value of the flags, and in compiled code after each kind of instruction that gives flags, IT blocks of each shape, the state at each exception entry and after each return, the registers of each mode, MRS and MSR with each field mask. |
| `tests/corrections.rs` | The cases of the faults that these tests found in the `Executor`. |
| `a64.rs` | Each instruction of the assembler against the word that clang gives for its text; each of the 1302 logical immediates of 32 bits. |
| `x64.rs` | Each instruction of the x86-64 assembler against the bytes that clang gives for its text. |

### Soak tier

`RAX_SOAK_SECONDS=300 cargo test --release -p rax-aarch32-system -- --test-threads 4
--nocapture`: each of the 16 tests with random cases runs cases for this time and prints
their count (20 minutes for the command, 80 minutes of CPU time). With `RAX_TEST_SEED=7`
the tests take an other seed, thus an other run has other cases.

### Ignored tests

`cargo test --release -p rax-aarch32-system -- --ignored` runs one test that fails:
it shows an order of calls that a machine must not use (the reason is in its `#[ignore]`).

### A failure

A failure prints the seed of its case, the bytes of the program, the registers at the
start, and the command that runs the case again:
`RAX_TEST_CASE=0x89ea1da2d2fe8f19 cargo test --release -p rax-aarch32-system <test>`.
`RAX_TEST_TRACE=1` prints each entry of an engine with the PC and the IT state.
`RAX_TEST_COLLECT=1` goes on after a failure and prints the count of the failures by kind.

The system-level tests are in `prime-rax`: a boot of an unmodified OS to its first screens,
compared with the one-instruction interpreter (a hash of the registers at each 4096
instructions, UART text, pictures) and with QEMU.

## Limits

- The compiler is for AArch64 macOS (MAP_JIT), AArch64 Linux and x86-64 Linux (the code
  pages are writable only for the time of a write). Other hosts have the interpreter only.
  Checked: `cargo zigbuild --tests` for the musl and gnu targets of AArch64 and x86-64
  Linux and for `x86_64-pc-windows-gnu`; a run of the two musl test binaries in a Linux
  AArch64 container. The Windows build was not run.
- No x86-64 processor ran the code generator for x86-64: the tests ran in a linux/amd64
  container on Apple Silicon, where Rosetta translates the code. Rosetta has the FP rules
  of AArch64 in some places where a processor has other rules (the sign of the default
  NaN, the time of the underflow test, no flag for a denormal operand). The compiled FP
  code leaves in each such case, but only a run on a processor can show that.
- Compiled code is for one virtual page of one physical page; a second virtual address of
  the same code compiles it again.
- ARMv7-A without LPAE, big-endian data, ThumbEE or Jazelle. Short vectors and trapped FP
  exceptions always go to the `Executor`.
- The instructions with no fast operation are about 50 times slower: see `Op::Slow` in
  `src/lib.rs` for what has one.
