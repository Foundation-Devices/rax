//! A software TLB for the short-descriptor translation of ARMv7-A, with one set of
//! entries for each MMU configuration.
//!
//! An entry also says when an access to its page can go to the memory of the host with
//! no call: the fast interpreter and the compiled code read these direct entries.

use std::cell::Cell;
use std::mem::offset_of;

use rax::arm::mmu_v6::{self, V6Access, V6MmuConfig};

use crate::PAGE_SIZE;
use crate::jit;

const PAGE_MASK: u32 = PAGE_SIZE as u32 - 1;
const ENTRIES: usize = 256;
/// One table for each access type and privilege.
const TABLES: usize = 6;
const CONTEXTS: usize = 64;
/// Not a page address, thus no address has this tag.
const NO_PAGE: u32 = u32::MAX;

/// How the accesses of one type go to a physical page.
pub enum Page {
    /// To memory of the host. `place` is a number of the machine for the page, a
    /// multiple of PAGE_SIZE: the place of its code in `BlockCache`, and the argument of
    /// `Tlb::protect_code`.
    Direct { place: usize, host: usize },
    /// To memory, but each access needs a call (now).
    Memory,
    /// To a peripheral, or to nothing.
    Other,
}

/// The memory of the machine, for the TLB.
pub trait PhysicalMemory {
    /// A word of a translation table, or None for an address with no memory.
    fn table_word(&self, pa: u32) -> Option<u32>;
    fn page(&self, physical_page: u32, access: V6Access) -> Page;
}

/// A translation fault: the fault status with the domain in bits 7:4.
#[derive(Clone, Copy, Debug)]
pub struct Fault {
    pub status: u32,
}

/// The compiled code reads `direct_page` and `addend`.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Entry {
    /// The virtual page, or NO_PAGE.
    page: u32,
    /// The virtual page when the access is direct, or NO_PAGE.
    direct_page: u32,
    physical_page: u32,
    /// `Page::Direct::place`, when `direct_page` is a page.
    place: u32,
    /// The address of the page in the host minus the virtual address of the page, when
    /// `direct_page` is a page.
    addend: usize,
    /// The page is `Page::Other`.
    other: bool,
}

const NO_ENTRY: Entry = Entry {
    page: NO_PAGE,
    direct_page: NO_PAGE,
    physical_page: 0,
    place: 0,
    addend: 0,
    other: false,
};

const _: () = assert!(
    ENTRIES == 1 << jit::TLB_BITS
        && size_of::<Entry>() == 1 << jit::TLB_ENTRY_SHIFT
        && offset_of!(Entry, direct_page) == jit::TLB_DIRECT_PAGE
        && offset_of!(Entry, addend) == jit::TLB_ADDEND
);

/// The translations of one MMU configuration: each address space of the guest has its own.
struct Context {
    mmu: V6MmuConfig,
    /// Entries of an older flush count are not valid.
    flush_count: u64,
    entries: Box<[Cell<Entry>]>,
}

fn new_entries() -> Box<[Cell<Entry>]> {
    vec![Cell::new(NO_ENTRY); ENTRIES * TABLES].into()
}

fn same_mmu(a: &V6MmuConfig, b: &V6MmuConfig) -> bool {
    (a.enabled, a.ttbr0, a.ttbr1, a.ttbcr_n, a.dacr, a.afe)
        == (b.enabled, b.ttbr0, b.ttbr1, b.ttbcr_n, b.dacr, b.afe)
}

fn table_index(va: u32, access: V6Access, privileged: bool) -> usize {
    (access as usize * 2 + usize::from(privileged)) * ENTRIES + (va >> 12) as usize % ENTRIES
}

pub struct Tlb {
    mmu: V6MmuConfig,
    contexts: Vec<Context>,
    context: usize,
    /// The context that the next new MMU configuration replaces, when all are in use.
    next_context: usize,
    flush_count: u64,
    pub flushes: u64,
    /// One bit for each virtual page: a fetch from the page got a translation after the
    /// last change of `flush_count`. The users of `mapping` keep addresses of such pages only.
    fetched: Box<[Cell<u64>]>,
    /// One bit for each 16 MiB of virtual addresses: a translation of after the last change
    /// of `flush_count` is from a section, a supersection or a large page.
    wide: [Cell<u64>; 4],
    /// Translations with a walk of the translation tables.
    pub misses: Cell<u64>,
}

impl Default for Tlb {
    fn default() -> Self {
        let mmu = V6MmuConfig::default();
        Tlb {
            mmu,
            contexts: vec![Context {
                mmu,
                flush_count: 0,
                entries: new_entries(),
            }],
            context: 0,
            next_context: 0,
            flush_count: 0,
            flushes: 0,
            fetched: vec![Cell::new(0); 1 << (32 - 12 - 6)].into(),
            wide: Default::default(),
            misses: Cell::new(0),
        }
    }
}

impl Tlb {
    /// The guest changed a CP15 register of the MMU.
    pub fn set_mmu(&mut self, mmu: V6MmuConfig) {
        if same_mmu(&mmu, &self.mmu) {
            return;
        }
        self.mmu = mmu;
        if let Some(index) = self
            .contexts
            .iter()
            .position(|context| same_mmu(&context.mmu, &mmu))
        {
            self.context = index;
        } else if self.contexts.len() < CONTEXTS {
            self.context = self.contexts.len();
            self.contexts.push(Context {
                mmu,
                flush_count: self.flush_count,
                entries: new_entries(),
            });
        } else {
            self.context = self.next_context;
            self.next_context = (self.next_context + 1) % CONTEXTS;
            self.contexts[self.context].mmu = mmu;
            // The entries are those of a different configuration. The new flush count
            // also ends the use of this context number by the users of `mapping`.
            self.new_flush_count();
        }
        let context = &mut self.contexts[self.context];
        if context.flush_count != self.flush_count {
            context.flush_count = self.flush_count;
            context.entries.fill(Cell::new(NO_ENTRY));
        }
    }

    fn new_flush_count(&mut self) {
        self.flush_count += 1;
        self.fetched.fill(Cell::new(0));
        self.wide = Default::default();
    }

    /// Removes the translations of all address spaces.
    pub fn flush(&mut self) {
        self.flushes += 1;
        self.new_flush_count();
        let context = &mut self.contexts[self.context];
        context.flush_count = self.flush_count;
        context.entries.fill(Cell::new(NO_ENTRY));
    }

    /// Removes the translations of the page of `va` in all address spaces.
    ///
    /// `mapping` stays when no fetch used the page: a guest that moves data pages between
    /// its processes then keeps the translations and the links of its code.
    pub fn flush_page(&mut self, va: u32) {
        let page = va & !PAGE_MASK;
        let (fetched, fetched_bit) = self.fetched_bit(page);
        let (wide, wide_bit) = self.wide_bit(va);
        // A wide mapping has entries for more pages than this one.
        if fetched.get() & fetched_bit != 0 || wide.get() & wide_bit != 0 {
            return self.flush();
        }
        for context in &mut self.contexts {
            // The entries of an older flush count are not valid already.
            if context.flush_count != self.flush_count {
                continue;
            }
            for table in 0..TABLES {
                let entry =
                    context.entries[table * ENTRIES + (va >> 12) as usize % ENTRIES].get_mut();
                if entry.page == page {
                    *entry = NO_ENTRY;
                }
            }
        }
    }

    fn fetched_bit(&self, page: u32) -> (&Cell<u64>, u64) {
        (&self.fetched[(page >> 18) as usize], 1 << (page >> 12 & 63))
    }

    fn wide_bit(&self, va: u32) -> (&Cell<u64>, u64) {
        (&self.wide[(va >> 30) as usize], 1 << (va >> 24 & 63))
    }

    /// The page at `place` holds decoded code: the next write to it must not be direct.
    pub fn protect_code(&mut self, place: usize) {
        let write_tables =
            ENTRIES * 2 * V6Access::Write as usize..ENTRIES * 2 * (V6Access::Write as usize + 1);
        for context in &mut self.contexts {
            for cell in &mut context.entries[write_tables.clone()] {
                let entry = cell.get_mut();
                if entry.direct_page != NO_PAGE && entry.place as usize == place {
                    entry.direct_page = NO_PAGE;
                }
            }
        }
    }

    /// A number for the translations in use: the address space and the count of the flushes.
    /// A virtual address has the same translation while this number and the privilege stay.
    pub fn mapping(&self) -> (usize, u64) {
        (self.context, self.flush_count)
    }

    fn entry(&self, va: u32, access: V6Access, privileged: bool) -> &Cell<Entry> {
        &self.contexts[self.context].entries[table_index(va, access, privileged)]
    }

    /// The physical address of a virtual address.
    pub fn translate(
        &self,
        memory: &impl PhysicalMemory,
        va: u32,
        access: V6Access,
        privileged: bool,
    ) -> Result<u32, Fault> {
        let page = va & !PAGE_MASK;
        let cell = self.entry(va, access, privileged);
        let mut entry = cell.get();
        if entry.page == page && (entry.direct_page == page || entry.other) {
            return Ok(entry.physical_page | va & PAGE_MASK);
        }
        if entry.page != page {
            self.misses.set(self.misses.get() + 1);
            if access == V6Access::Execute {
                let (word, bit) = self.fetched_bit(page);
                word.set(word.get() | bit);
            }
            let pa = if self.mmu.enabled {
                // The walk reads the descriptor of level 1, then the one of level 2.
                let mut descriptors = [None; 2];
                let mut level = 0;
                let walk = |pa: u32| {
                    let word = memory.table_word(pa);
                    descriptors[level] = word;
                    level += 1;
                    word
                };
                match mmu_v6::translate_v6(&self.mmu, va, privileged, access, walk) {
                    Ok(translation) => {
                        let small_page = descriptors[1].is_some_and(|word| word & 0b10 != 0);
                        if !small_page {
                            let (word, bit) = self.wide_bit(va);
                            word.set(word.get() | bit);
                        }
                        translation.pa
                    }
                    Err(fault) => {
                        return Err(Fault {
                            status: fault.fsr | fault.domain << 4,
                        });
                    }
                }
            } else {
                va
            };
            entry = Entry {
                page,
                physical_page: pa & !PAGE_MASK,
                ..NO_ENTRY
            };
        }
        match memory.page(entry.physical_page, access) {
            Page::Direct { place, host } => {
                entry.direct_page = page;
                entry.place = place as u32;
                entry.addend = host.wrapping_sub(page as usize);
            }
            Page::Memory => {}
            Page::Other => entry.other = true,
        }
        cell.set(entry);
        Ok(entry.physical_page | va & PAGE_MASK)
    }

    /// `Page::Direct::place` of the page of the instruction at `va`, or None when the
    /// fetch gives a fault or is not direct.
    pub fn code_place(
        &self,
        memory: &impl PhysicalMemory,
        va: u32,
        privileged: bool,
    ) -> Option<usize> {
        let page = va & !PAGE_MASK;
        if self
            .entry(va, V6Access::Execute, privileged)
            .get()
            .direct_page
            != page
        {
            self.translate(memory, va, V6Access::Execute, privileged)
                .ok()?;
        }
        let entry = self.entry(va, V6Access::Execute, privileged).get();
        (entry.direct_page == page).then_some(entry.place as usize)
    }

    /// The address in the host of an access that stays in one page with a direct entry.
    pub fn direct(
        &self,
        va: u32,
        access: V6Access,
        privileged: bool,
        length: usize,
    ) -> Option<usize> {
        let entry = self.entry(va, access, privileged).get();
        let in_page = (va & PAGE_MASK) as usize;
        (entry.direct_page == va & !PAGE_MASK && in_page + length <= PAGE_SIZE)
            .then(|| entry.addend.wrapping_add(va as usize))
    }

    /// The tables of the address space and the privilege of now for the reads and for
    /// the writes, in the form of `jit::Context`.
    pub fn direct_tables(&self, privileged: bool) -> (*const u8, *const u8) {
        let entries = &self.contexts[self.context].entries;
        let table = |access| {
            entries[table_index(0, access, privileged)..]
                .as_ptr()
                .cast()
        };
        (table(V6Access::Read), table(V6Access::Write))
    }
}
