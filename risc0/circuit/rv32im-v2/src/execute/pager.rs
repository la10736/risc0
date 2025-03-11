// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::BTreeSet;

use anyhow::{bail, Result};
use bit_vec::BitVec;
use derive_more::Debug;
use risc0_binfmt::{MemoryImage2, Page, WordAddr};
use risc0_zkp::core::digest::Digest;

use super::{node_idx, platform::*};

pub const PAGE_WORDS: usize = PAGE_BYTES / WORD_SIZE;

const LOAD_ROOT_CYCLES: u32 = 1;
const RESUME_CYCLES: u32 = 2;
const SUSPEND_CYCLES: u32 = 2;
const STORE_ROOT_CYCLES: u32 = 1;

const POSEIDON_PAGING: u32 = 1;
const POSEIDON_LOAD_IN: u32 = 2;
const POSEIDON_DO_OUT: u32 = 1;
const POSEIDON_EXTERNAL: u32 = 8;
const POSEIDON_INTERNAL: u32 = 1;
const POSEIDON_ENTRY: u32 = 1;
pub(crate) const POSEIDON_BLOCK_WORDS: u32 = 8;
pub(crate) const POSEIDON_PAGE_ROUNDS: u32 = PAGE_WORDS as u32 / POSEIDON_BLOCK_WORDS;

const PAGE_CYCLES: u32 = POSEIDON_PAGING + 10 * POSEIDON_PAGE_ROUNDS + POSEIDON_DO_OUT;

const NODE_CYCLES: u32 =
    POSEIDON_PAGING + POSEIDON_LOAD_IN + POSEIDON_EXTERNAL + POSEIDON_INTERNAL + POSEIDON_DO_OUT;

pub(crate) const RESERVED_PAGING_CYCLES: u32 = LOAD_ROOT_CYCLES
    + POSEIDON_ENTRY
    + POSEIDON_PAGING
    + RESUME_CYCLES
    + SUSPEND_CYCLES
    + POSEIDON_ENTRY
    + POSEIDON_PAGING
    + STORE_ROOT_CYCLES;

const NUM_PAGES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) enum PageState {
    Unloaded,
    Loaded,
    Dirty,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PageTraceEvent {
    PageIn { cycles: u32 },
    PageOut { cycles: u32 },
}

#[derive(Debug)]
pub(crate) struct PageStates {
    states: BitVec,
    indexes: Vec<u32>,
}

impl PageStates {
    pub(crate) fn new(size: usize) -> Self {
        Self {
            states: BitVec::from_elem(size * 2, false),
            indexes: vec![],
        }
    }

    pub(crate) fn set(&mut self, index: u32, value: PageState) {
        let set_before = self.get(index) != PageState::Unloaded;
        match value {
            PageState::Unloaded => unimplemented!(),
            PageState::Loaded => {
                // b01 => Loaded
                self.states.set(index as usize * 2, false);
                self.states.set(index as usize * 2 + 1, true);
            }
            PageState::Dirty => {
                // b10 | b11 => Dirty
                self.states.set(index as usize * 2, true);
            }
        }

        if !set_before {
            self.indexes.push(index);
        }
    }

    #[inline(always)]
    pub(crate) fn get(&self, index: u32) -> PageState {
        if self.states.get(index as usize * 2).unwrap() {
            // b10 | b11 => Dirty
            PageState::Dirty
        } else if self.states.get(index as usize * 2 + 1).unwrap() {
            // b01 => Loaded
            PageState::Loaded
        } else {
            // b00 => Unloaded
            PageState::Unloaded
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, PageState)> + '_ {
        self.indexes.iter().map(|index| (*index, self.get(*index)))
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = u32> + '_ {
        self.indexes.iter().copied()
    }

    pub(crate) fn clear(&mut self) {
        self.states.clear();
        self.indexes.clear();
    }
}

#[test]
fn page_states() {
    use PageState::*;

    let mut s = PageStates::new(10);
    for (a, b) in [(Loaded, Dirty), (Dirty, Loaded)] {
        s.clear();

        for i in 0..5 {
            assert_eq!(s.get(i), Unloaded);

            s.set(i, a);
            assert_eq!(s.get(i), a);

            s.set(i, b);
            assert_eq!(s.get(i), b);

            s.set(i, a);
            assert_eq!(s.get(i), a);
        }

        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            vec![(0, a), (1, a), (2, a), (3, a), (4, a)]
        );

        assert_eq!(s.keys().collect::<Vec<_>>(), (0..5).collect::<Vec<_>>());
    }
}

const NUM_PAGE_STATES: usize = NUM_PAGES * 2;

struct PageTable {
    table: Vec<u32>,
}

impl Default for PageTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PageTable {
    const INVALID_IDX: u32 = 0;

    fn new() -> Self {
        Self {
            table: vec![Self::INVALID_IDX; NUM_PAGES],
        }
    }

    #[inline(always)]
    fn get(&self, index: u32) -> Option<usize> {
        let value = self.table[index as usize] as usize;
        if value > 0 {
            Some(value - 1)
        } else {
            None
        }
    }

    fn set(&mut self, index: u32, value: usize) {
        self.table[index as usize] = (value + 1) as u32
    }

    fn clear(&mut self) {
        if self.table.is_empty() {
            // For the initial case, allocate a new vector
            self.table = vec![Self::INVALID_IDX; NUM_PAGES];
        } else if self.table.len() == NUM_PAGES {
            // For subsequent calls, reuse existing memory with optimized clearing
            // Use chunks to improve cache locality
            const CHUNK_SIZE: usize = 1024;
            for chunk in self.table.chunks_mut(CHUNK_SIZE) {
                for item in chunk {
                    *item = Self::INVALID_IDX;
                }
            }
        } else {
            // If the size has changed, reallocate
            self.table = vec![Self::INVALID_IDX; NUM_PAGES];
        }
    }
}

#[test]
fn page_table() {
    let mut table = PageTable::new();

    for idx in [0, 5, 12] {
        assert_eq!(table.get(idx), None);

        table.set(idx, 13);
        assert_eq!(table.get(idx).unwrap(), 13);
    }

    table.clear();

    for idx in [0, 5, 12] {
        assert_eq!(table.get(idx), None);
    }
}

#[derive(Debug)]
pub(crate) struct PagedMemory {
    pub image: MemoryImage2,
    #[debug(skip)]
    page_table: PageTable,
    #[debug(skip)]
    page_cache: Vec<Page>,
    pub(crate) page_states: PageStates,
    pub cycles: u32,
    user_registers: [u32; REG_MAX],
    machine_registers: [u32; REG_MAX],
    tracing_enabled: bool,
    trace_events: Vec<PageTraceEvent>,
}

impl PagedMemory {
    pub(crate) fn new(mut image: MemoryImage2, tracing_enabled: bool) -> Self {
        let mut machine_registers = [0; REG_MAX];
        let mut user_registers = [0; REG_MAX];
        let page_idx = MACHINE_REGS_ADDR.waddr().page_idx();
        let page = image.get_page(page_idx).unwrap();
        for idx in 0..REG_MAX {
            machine_registers[idx] = page.load(MACHINE_REGS_ADDR.waddr() + idx);
            user_registers[idx] = page.load(USER_REGS_ADDR.waddr() + idx);
        }

        Self {
            image,
            page_table: PageTable::new(),
            page_cache: Vec::new(),
            page_states: PageStates::new(NUM_PAGE_STATES),
            cycles: RESERVED_PAGING_CYCLES,
            user_registers,
            machine_registers,
            tracing_enabled,
            trace_events: vec![],
        }
    }

    pub(crate) fn reset(&mut self) {
        // Instead of allocating a new page table, prefetch and zero the existing one
        if !self.page_table.table.is_empty() {
            // Prefetch the page table into cache line by line to avoid cache misses
            let table_ptr = self.page_table.table.as_ptr();
            let mut offset = 0;

            // Loop through the table in cache-line sized chunks (typically 64 bytes)
            let cache_line_size = 64; // bytes
            let stride = cache_line_size / std::mem::size_of::<u32>();

            while offset < self.page_table.table.len() {
                // Manual prefetch at cache-line boundaries to improve performance
                // Use read-ahead to ensure we're prefetching future lines
                unsafe {
                    // Simple pointer read to trigger prefetch without compiler optimizing it away
                    let _ = std::ptr::read_volatile(table_ptr.add(offset));
                }

                // Zero out the cache line we just prefetched
                let end = std::cmp::min(offset + stride, self.page_table.table.len());
                for i in offset..end {
                    self.page_table.table[i] = PageTable::INVALID_IDX;
                }

                offset += stride;
            }
        } else {
            self.page_table.clear();
        }

        self.page_cache.clear();
        self.page_states.clear();
        self.cycles = RESERVED_PAGING_CYCLES;
    }

    pub(crate) fn page_indexes(&self) -> BTreeSet<u32> {
        self.page_states.keys().collect()
    }

    fn try_load_register(&self, addr: WordAddr) -> Option<u32> {
        if addr >= USER_REGS_ADDR.waddr() && addr < USER_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = addr - USER_REGS_ADDR.waddr();
            Some(self.user_registers[reg_idx.0 as usize])
        } else if addr >= MACHINE_REGS_ADDR.waddr() && addr < MACHINE_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = addr - MACHINE_REGS_ADDR.waddr();
            Some(self.machine_registers[reg_idx.0 as usize])
        } else {
            None
        }
    }

    #[deprecated(note = "Use optimized store() directly instead")]
    #[allow(dead_code)]
    fn try_store_register(&mut self, addr: WordAddr, word: u32) -> bool {
        if addr >= USER_REGS_ADDR.waddr() && addr < USER_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = addr - USER_REGS_ADDR.waddr();
            self.user_registers[reg_idx.0 as usize] = word;
            true
        } else if addr >= MACHINE_REGS_ADDR.waddr() && addr < MACHINE_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = addr - MACHINE_REGS_ADDR.waddr();
            self.machine_registers[reg_idx.0 as usize] = word;
            true
        } else {
            false
        }
    }

    fn peek_ram(&mut self, addr: WordAddr) -> Result<u32> {
        let page_idx = addr.page_idx();
        if let Some(cache_idx) = self.page_table.get(page_idx) {
            // Loaded, get from cache
            Ok(self.page_cache[cache_idx].load(addr))
        } else {
            // Unloaded, peek into image
            Ok(self.image.get_page(page_idx)?.load(addr))
        }
    }

    pub(crate) fn peek(&mut self, addr: WordAddr) -> Result<u32> {
        if addr >= MEMORY_END_ADDR {
            bail!("Invalid peek address: {addr:?}");
        }

        match self.try_load_register(addr) {
            Some(word) => Ok(word),
            None => self.peek_ram(addr),
        }
    }

    pub(crate) fn peek_page(&mut self, page_idx: u32) -> Result<Vec<u8>> {
        if let Some(cache_idx) = self.page_table.get(page_idx) {
            // Loaded, get from cache
            Ok(self.page_cache[cache_idx].data().clone())
        } else {
            // Unloaded, peek into image
            Ok(self.image.get_page(page_idx)?.data().clone())
        }
    }

    pub(crate) fn load(&mut self, addr: WordAddr) -> Result<u32> {
        #[cold]
        fn handle_error(addr: WordAddr) -> Result<u32> {
            bail!("Invalid load address: {addr:?}")
        }

        // Fast path - directly check if address is a register to avoid extra function call overhead
        if addr >= USER_REGS_ADDR.waddr() && addr < USER_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = addr - USER_REGS_ADDR.waddr();
            return Ok(self.user_registers[reg_idx.0 as usize]);
        } else if addr >= MACHINE_REGS_ADDR.waddr() && addr < MACHINE_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = addr - MACHINE_REGS_ADDR.waddr();
            return Ok(self.machine_registers[reg_idx.0 as usize]);
        }

        if addr >= MEMORY_END_ADDR {
            return handle_error(addr);
        }

        // Memory access path
        let page_idx = addr.page_idx();

        // Prefetch page table entry to reduce cache miss latency
        let cache_idx_opt = self.page_table.get(page_idx);

        // If we have the page in cache, fast path
        if let Some(cache_idx) = cache_idx_opt {
            // Prefetch the actual page data
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            unsafe {
                // Cast to byte pointer for prefetching
                let page_ptr = &self.page_cache[cache_idx] as *const _ as *const u8;
                // Lightweight prefetch hint without compiler optimization
                std::ptr::read_volatile(page_ptr);
            }

            return Ok(self.page_cache[cache_idx].load(addr));
        }

        // Slow path - page miss
        let node_idx = node_idx(page_idx);
        self.load_page(page_idx)?;
        self.page_states.set(node_idx, PageState::Loaded);

        // After loading, we know it's now in the cache
        let cache_idx = self.page_table.get(page_idx).unwrap();
        Ok(self.page_cache[cache_idx].load(addr))
    }

    pub(crate) fn load_register(&mut self, base: WordAddr, idx: usize) -> u32 {
        if base == USER_REGS_ADDR.waddr() {
            self.user_registers[idx]
        } else if base == MACHINE_REGS_ADDR.waddr() {
            self.machine_registers[idx]
        } else {
            unimplemented!("unknown register address {base:?}");
        }
    }

    pub(crate) fn store(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        #[cold]
        fn handle_error(addr: WordAddr) -> Result<()> {
            bail!("Invalid store address: {addr:?}")
        }

        // Fast path - directly check if address is a register to avoid function call overhead
        if addr >= USER_REGS_ADDR.waddr() && addr < USER_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = addr - USER_REGS_ADDR.waddr();
            self.user_registers[reg_idx.0 as usize] = word;
            return Ok(());
        } else if addr >= MACHINE_REGS_ADDR.waddr() && addr < MACHINE_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = addr - MACHINE_REGS_ADDR.waddr();
            self.machine_registers[reg_idx.0 as usize] = word;
            return Ok(());
        }

        if addr >= MEMORY_END_ADDR {
            return handle_error(addr);
        }

        // Memory store path
        let page_idx = addr.page_idx();
        let node_idx = node_idx(page_idx);

        // Optimize the page lookup
        let page = if let Some(cache_idx) = self.page_table.get(page_idx) {
            let current_state = self.page_states.get(node_idx);

            // If the page is loaded but not dirty, mark it dirty
            if current_state == PageState::Loaded {
                self.cycles += PAGE_CYCLES;

                // Only trace if needed (rare case)
                if self.tracing_enabled {
                    self.trace_page_out(PAGE_CYCLES);
                }

                self.fixup_costs(node_idx, PageState::Dirty);
                self.page_states.set(node_idx, PageState::Dirty);
            }

            // Get the page from cache
            &mut self.page_cache[cache_idx]
        } else {
            // Page not in cache, load it
            self.load_page(page_idx)?;

            // Since it's a fresh load, mark it dirty
            self.cycles += PAGE_CYCLES;
            if self.tracing_enabled {
                self.trace_page_out(PAGE_CYCLES);
            }

            self.fixup_costs(node_idx, PageState::Dirty);
            self.page_states.set(node_idx, PageState::Dirty);

            // Now we know the page is in the cache
            let cache_idx = self.page_table.get(page_idx).unwrap();
            &mut self.page_cache[cache_idx]
        };

        // Finally store to the page
        page.store(addr, word);
        Ok(())
    }

    pub(crate) fn store_register(&mut self, base: WordAddr, idx: usize, word: u32) {
        if base == USER_REGS_ADDR.waddr() {
            self.user_registers[idx] = word;
        } else if base == MACHINE_REGS_ADDR.waddr() {
            self.machine_registers[idx] = word;
        } else {
            unimplemented!("unknown register address {base:?}");
        }
    }

    #[deprecated(note = "Use optimized store() directly instead")]
    #[allow(dead_code)]
    fn page_for_writing(&mut self, page_idx: u32) -> Result<&mut Page> {
        let node_idx = node_idx(page_idx);
        let mut state = self.page_states.get(node_idx);
        if state == PageState::Unloaded {
            self.load_page(page_idx)?;
            state = PageState::Loaded;
        };

        if state == PageState::Loaded {
            self.cycles += PAGE_CYCLES;
            self.trace_page_out(PAGE_CYCLES);
            self.fixup_costs(node_idx, PageState::Dirty);
            self.page_states.set(node_idx, PageState::Dirty);
        }
        let cache_idx = self.page_table.get(page_idx).unwrap();
        Ok(self.page_cache.get_mut(cache_idx).unwrap())
    }

    #[allow(deprecated)]
    fn write_registers(&mut self) {
        // Copy register values first to avoid borrow conflicts
        let user_registers = self.user_registers;
        let machine_registers = self.machine_registers;
        // This works because we can assume that user and machine register files
        // live in the same page.
        let page_idx = MACHINE_REGS_ADDR.waddr().page_idx();
        let page = self.page_for_writing(page_idx).unwrap();
        for idx in 0..REG_MAX {
            page.store(MACHINE_REGS_ADDR.waddr() + idx, machine_registers[idx]);
            page.store(USER_REGS_ADDR.waddr() + idx, user_registers[idx]);
        }
    }

    #[allow(deprecated)]
    pub(crate) fn commit(&mut self) -> (MemoryImage2, Digest, Digest) {
        // tracing::trace!("commit: {self:#?}");

        self.write_registers();

        let pre_image = self.image.clone();
        let pre_state = self.image.image_id();

        let mut sorted_keys: Vec<_> = self.page_states.keys().collect();
        sorted_keys.sort();

        for node_idx in sorted_keys {
            if node_idx < MEMORY_PAGES as u32 {
                continue;
            }

            let page_state = self.page_states.get(node_idx);
            let page_idx = page_idx(node_idx);
            tracing::trace!("commit: {page_idx:#08x}, state: {page_state:?}");

            // Update dirty pages into the image that accumulates over a session.
            if page_state == PageState::Dirty {
                let cache_idx = self.page_table.get(page_idx).unwrap();
                let page = &self.page_cache[cache_idx];
                self.image.set_page(page_idx, page.clone());
            }
        }
        self.image.update_digests();

        let post_state = self.image.image_id();
        (pre_image, pre_state, post_state)
    }

    fn load_page(&mut self, page_idx: u32) -> Result<()> {
        tracing::trace!("load_page: {page_idx:#08x}");
        let page = self.image.get_page(page_idx)?;
        self.page_table.set(page_idx, self.page_cache.len());
        self.page_cache.push(page);
        self.cycles += PAGE_CYCLES;
        self.trace_page_in(PAGE_CYCLES);
        self.fixup_costs(node_idx(page_idx), PageState::Loaded);
        Ok(())
    }

    fn fixup_costs(&mut self, mut node_idx: u32, goal: PageState) {
        tracing::trace!("fixup: {node_idx:#010x}: {goal:?}");
        while node_idx != 0 {
            let state = self.page_states.get(node_idx);
            if goal > state {
                if node_idx < MEMORY_PAGES as u32 {
                    if state == PageState::Unloaded {
                        // tracing::trace!("fixup: {state:?}: {node_idx:#010x}");
                        self.cycles += NODE_CYCLES;
                        self.trace_page_in(NODE_CYCLES);
                    }
                    if goal == PageState::Dirty {
                        // tracing::trace!("fixup: {goal:?}: {node_idx:#010x}");
                        self.cycles += NODE_CYCLES;
                        self.trace_page_out(NODE_CYCLES);
                    }
                }
                self.page_states.set(node_idx, goal);
            }
            node_idx /= 2;
        }
    }

    pub(crate) fn trace_events(&self) -> &[PageTraceEvent] {
        &self.trace_events
    }

    pub(crate) fn clear_trace_events(&mut self) {
        self.trace_events.clear();
    }

    fn trace_page_in(&mut self, cycles: u32) {
        if self.tracing_enabled {
            self.trace_events.push(PageTraceEvent::PageIn { cycles });
        }
    }

    fn trace_page_out(&mut self, cycles: u32) {
        if self.tracing_enabled {
            self.trace_events.push(PageTraceEvent::PageOut { cycles });
        }
    }
}

pub(crate) fn page_idx(node_idx: u32) -> u32 {
    node_idx - MEMORY_PAGES as u32
}

pub(crate) fn compute_partial_image(
    input_image: MemoryImage2,
    indexes: BTreeSet<u32>,
) -> MemoryImage2 {
    let mut image = MemoryImage2::default();

    for node_idx in &indexes {
        if *node_idx < MEMORY_PAGES as u32 {
            continue;
        }

        let page_idx = page_idx(*node_idx);

        // Copy original state of all pages accessed in this segment.
        image.set_page(page_idx, input_image.get_existing_page(page_idx));
    }

    // Add minimal needed 'uncles'
    for node_idx in &indexes {
        // If this is a leaf, break
        if *node_idx >= MEMORY_PAGES as u32 {
            break;
        }

        let lhs_idx = *node_idx * 2;
        let rhs_idx = *node_idx * 2 + 1;

        // Otherwise, add whichever child digest (if any) is not loaded
        if !indexes.contains(&lhs_idx) {
            image.set_digest(lhs_idx, *input_image.get_existing_digest(lhs_idx));
        }
        if !indexes.contains(&rhs_idx) {
            image.set_digest(rhs_idx, *input_image.get_existing_digest(rhs_idx));
        }
    }

    image.update_digests();

    image
}
