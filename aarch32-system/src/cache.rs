//! The decoded blocks of the code in the memory of the guest.

use std::collections::HashMap;
use std::rc::Rc;

use crate::{Block, PAGE_SIZE};

const SLOTS: usize = 1 << 18;
const LINKS: usize = 1 << 12;
/// The blocks of an operating system take about 3.5 KiB each: 450 MiB for this count.
pub const MAX_BLOCKS: usize = 1 << 17;

/// The block at a virtual address, for one privilege, one mapping and one removal count.
#[derive(Clone, Default)]
struct Link {
    /// The PC with the T bit in bit 0.
    address: u32,
    privileged: bool,
    mapping: (usize, u64),
    removals: u64,
    block: Option<Rc<Block>>,
}

/// Decoded blocks. The key is the place of the first instruction in the memory of the
/// machine (`tlb::Page::Direct::place` of its page plus the offset in the page), then the
/// T bit.
pub struct BlockCache {
    slots: Vec<Option<(usize, Rc<Block>)>>,
    /// For each page, the slots that got a block of the page.
    page_slots: HashMap<usize, Vec<usize>>,
    /// The blocks of the last virtual addresses: a hit needs no translation of the PC.
    links: Vec<Link>,
    /// The count of the removals of blocks: a link of an older count is not valid.
    removals: u64,
    /// The blocks that the cache made after the last `clear`.
    made: usize,
    /// The count of blocks that makes the cache full.
    pub limit: usize,
}

impl Default for BlockCache {
    fn default() -> Self {
        BlockCache {
            slots: vec![None; SLOTS],
            page_slots: HashMap::new(),
            links: vec![Link::default(); LINKS],
            removals: 0,
            made: 0,
            limit: MAX_BLOCKS,
        }
    }
}

impl BlockCache {
    /// A number that changes at each removal of blocks.
    pub fn removals(&self) -> u64 {
        self.removals
    }

    /// The block that `link` gave for the guest address (T bit in bit 0), the privilege
    /// and the mapping (`Tlb::mapping`), when no block was removed after that.
    #[inline]
    pub fn linked(
        &self,
        address: u32,
        privileged: bool,
        mapping: (usize, u64),
    ) -> Option<Rc<Block>> {
        let link = &self.links[(address >> 1) as usize % LINKS];
        (link.address == address
            && link.privileged == privileged
            && link.mapping == mapping
            && link.removals == self.removals)
            .then(|| link.block.clone())
            .flatten()
    }

    pub fn link(
        &mut self,
        address: u32,
        privileged: bool,
        mapping: (usize, u64),
        block: &Rc<Block>,
    ) {
        self.links[(address >> 1) as usize % LINKS] = Link {
            address,
            privileged,
            mapping,
            removals: self.removals,
            block: Some(block.clone()),
        };
    }

    /// The block at a place in the memory, or None when the block is empty. `translate`
    /// makes the block when the cache has none.
    pub fn block_at(
        &mut self,
        place: usize,
        thumb: bool,
        translate: impl FnOnce() -> Block,
    ) -> Option<Rc<Block>> {
        let key = place << 1 | usize::from(thumb);
        let slot = (place >> 1) % SLOTS;
        if let Some((_, block)) = self.slots[slot]
            .as_ref()
            .filter(|(slot_key, _)| *slot_key == key)
        {
            return (!block.is_empty()).then(|| block.clone());
        }
        let block = Rc::new(translate());
        self.made += 1;
        self.slots[slot] = Some((key, block.clone()));
        self.page_slots
            .entry(place & !(PAGE_SIZE - 1))
            .or_default()
            .push(slot);
        (!block.is_empty()).then_some(block)
    }

    /// True when the cache made `limit` blocks: `clear` is due, or else the memory of the
    /// blocks has no limit.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.made >= self.limit
    }

    /// Removes all the blocks. `Jit::flush` removes the blocks that have code.
    pub fn clear(&mut self) {
        self.removals += 1;
        self.slots.fill(None);
        self.page_slots.clear();
        self.links.fill(Link::default());
        self.made = 0;
    }

    /// Removes the blocks of pages, after a write to them.
    pub fn remove_pages(&mut self, pages: impl IntoIterator<Item = usize>) {
        self.removals += 1;
        for page in pages {
            for slot in self.page_slots.remove(&page).unwrap_or_default() {
                if self.slots[slot]
                    .as_ref()
                    .is_some_and(|(key, _)| (key >> 1) & !(PAGE_SIZE - 1) == page)
                {
                    self.slots[slot] = None;
                }
            }
        }
    }
}
