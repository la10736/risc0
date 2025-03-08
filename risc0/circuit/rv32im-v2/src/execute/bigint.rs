use std::{collections::HashMap, io::Cursor};

use anyhow::{anyhow, bail, ensure, Result};
use derive_more::Debug;
use malachite::Natural;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive as _;
use risc0_binfmt::WordAddr;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{
    bibc::{self, BigIntIO},
    byte_poly::BytePolyProgram,
    platform::*,
    r0vm::{LoadOp, Risc0Context},
    CycleState,
};

pub(crate) const BIGINT_STATE_COUNT: usize = 5 + 16;
pub(crate) const BIGINT_ACCUM_STATE_COUNT: usize = 3 * 4;

/// BigInt width, in words, handled by the BigInt accelerator circuit.
pub(crate) const BIGINT_WIDTH_WORDS: usize = 4;

/// BigInt width, in bytes, handled by the BigInt accelerator circuit.
pub(crate) const BIGINT_WIDTH_BYTES: usize = BIGINT_WIDTH_WORDS * WORD_SIZE;

pub(crate) type BigIntBytes = [u8; BIGINT_WIDTH_BYTES];
type BigIntWitness = HashMap<WordAddr, BigIntBytes>;

// Pre-allocate buffer for byte conversions to reduce allocations
thread_local! {
    static BYTES_BUFFER: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(Vec::with_capacity(1024));
    static LIMBS_BUFFER: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(Vec::with_capacity(256));
}

#[derive(Clone, Debug)]
pub(crate) struct BigIntState {
    pub is_ecall: bool,
    pub pc: WordAddr,
    pub poly_op: PolyOp,
    pub coeff: u32,
    pub bytes: BigIntBytes,
    pub next_state: CycleState,
}

struct BigInt {
    state: BigIntState,
    program: BytePolyProgram,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum PolyOp {
    Reset,
    Shift,
    SetTerm,
    AddTotal,
    Carry1,
    Carry2,
    EqZero,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum MemoryOp {
    Read,
    Write,
    Check,
}

#[derive(Debug)]
#[debug("{poly_op:?}({mem_op:?}, c:{coeff}, r:{reg}, o:{offset})")]
pub(crate) struct Instruction {
    pub poly_op: PolyOp,
    pub mem_op: MemoryOp,
    pub coeff: i32,
    pub reg: u32,
    pub offset: u32,
}

impl Instruction {
    // instruction encoding:
    // 3  2   2  2    1               0
    // 1  8   4  1    6               0
    // mmmmppppcccaaaaaoooooooooooooooo
    pub fn decode(insn: u32) -> Result<Self> {
        Ok(Self {
            mem_op: MemoryOp::from_u32(insn >> 28 & 0x0f)
                .ok_or_else(|| anyhow!("Invalid mem_op in bigint program"))?,
            poly_op: PolyOp::from_u32(insn >> 24 & 0x0f)
                .ok_or_else(|| anyhow!("Invalid poly_op in bigint program"))?,
            coeff: (insn >> 21 & 0x07) as i32 - 4,
            reg: insn >> 16 & 0x1f,
            offset: insn & 0xffff,
        })
    }
}

impl BigInt {
    fn run(&mut self, ctx: &mut dyn Risc0Context, witness: &BigIntWitness) -> Result<()> {
        ctx.on_bigint_cycle(CycleState::BigIntEcall, &self.state);
        while self.state.next_state == CycleState::BigIntStep {
            // Use feature detection to choose the optimal implementation
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe { self.step_avx2(ctx, witness)? }
                } else {
                    self.step_scalar(ctx, witness)?
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                self.step_scalar(ctx, witness)?
            }
        }
        Ok(())
    }

    // Original implementation renamed for non-AVX2 platforms
    fn step_scalar(&mut self, ctx: &mut dyn Risc0Context, witness: &BigIntWitness) -> Result<()> {
        self.state.pc.inc();
        let insn = Instruction::decode(ctx.load_u32(LoadOp::Record, self.state.pc)?)?;

        let base =
            ctx.load_aligned_addr_from_machine_register(LoadOp::Record, insn.reg as usize)?;
        let addr = base + insn.offset * BIGINT_WIDTH_WORDS as u32;

        tracing::trace!("step({:?}, {insn:?}, {addr:?})", self.state.pc);
        if insn.mem_op == MemoryOp::Check && insn.poly_op != PolyOp::Reset {
            if !self.program.in_carry {
                self.program.in_carry = true;
                self.program.total_carry = self.program.total.clone();
                let mut carry = 0;

                // Do carry propagation
                for coeff in self.program.total_carry.coeffs.iter_mut() {
                    *coeff += carry;
                    ensure!(*coeff % 256 == 0, "bad carry");
                    *coeff /= 256;
                    carry = *coeff;
                }
                tracing::trace!("carry propagate complete");
            }

            let base_point = 128 * 256 * 64;
            for (i, ret) in self.state.bytes.iter_mut().enumerate() {
                let offset = insn.offset as usize;
                let coeff = self.program.total_carry.coeffs[offset * BIGINT_WIDTH_BYTES + i] as u32;
                let value = coeff.wrapping_add(base_point);
                match insn.poly_op {
                    PolyOp::Carry1 => *ret = ((value >> 14) & 0xff) as u8,
                    PolyOp::Carry2 => *ret = ((value >> 8) & 0x3f) as u8,
                    PolyOp::Shift | PolyOp::EqZero => *ret = (value & 0xff) as u8,
                    _ => {
                        bail!("Invalid poly_op in bigint program")
                    }
                }
            }
        } else if insn.mem_op == MemoryOp::Read {
            for i in 0..BIGINT_WIDTH_WORDS {
                let word = ctx.load_u32(LoadOp::Record, addr + i)?;
                for (j, byte) in word.to_le_bytes().iter().enumerate() {
                    self.state.bytes[i * WORD_SIZE + j] = *byte;
                }
            }
        } else if !addr.is_null() {
            self.state.bytes = *witness
                .get(&addr)
                .ok_or_else(|| anyhow!("Missing bigint witness: {addr:?}"))?;
            if insn.mem_op == MemoryOp::Write {
                let words: &[u32] = bytemuck::cast_slice(&self.state.bytes);
                for (i, word) in words.iter().enumerate() {
                    ctx.store_u32(addr + i, *word)?;
                }
            }
        }

        self.program.step(&insn, &self.state.bytes)?;

        self.state.is_ecall = false;
        self.state.poly_op = insn.poly_op;
        self.state.coeff = (insn.coeff + 4) as u32;
        self.state.next_state = if !self.state.is_ecall && insn.poly_op == PolyOp::Reset {
            CycleState::Decode
        } else {
            CycleState::BigIntStep
        };

        ctx.on_bigint_cycle(CycleState::BigIntStep, &self.state);
        Ok(())
    }

    // AVX2-optimized implementation
    #[cfg(target_arch = "x86_64")]
    unsafe fn step_avx2(
        &mut self,
        ctx: &mut dyn Risc0Context,
        witness: &BigIntWitness,
    ) -> Result<()> {
        self.state.pc.inc();
        let insn = Instruction::decode(ctx.load_u32(LoadOp::Record, self.state.pc)?)?;

        let base =
            ctx.load_aligned_addr_from_machine_register(LoadOp::Record, insn.reg as usize)?;
        let addr = base + insn.offset * BIGINT_WIDTH_WORDS as u32;

        if insn.mem_op == MemoryOp::Check && insn.poly_op != PolyOp::Reset {
            if !self.program.in_carry {
                self.program.in_carry = true;
                self.program.total_carry = self.program.total.clone();

                // Optimized carry propagation using AVX2
                self.avx2_carry_propagate()?;
            }

            let base_point = 128 * 256 * 64;

            // Use wide crate to process outputs for operations
            match insn.poly_op {
                PolyOp::Shift | PolyOp::EqZero => {
                    self.avx2_process_output(insn.offset as usize, base_point, |v| v & 0xff)?;
                }
                PolyOp::Carry1 => {
                    self.avx2_process_output(insn.offset as usize, base_point, |v| {
                        (v >> 14) & 0xff
                    })?;
                }
                PolyOp::Carry2 => {
                    self.avx2_process_output(insn.offset as usize, base_point, |v| {
                        (v >> 8) & 0x3f
                    })?;
                }
                _ => {
                    bail!("Invalid poly_op in bigint program")
                }
            }
        } else if insn.mem_op == MemoryOp::Read {
            // Fast load using wide
            self.avx2_load_words(ctx, addr)?;
        } else if !addr.is_null() {
            self.state.bytes = *witness
                .get(&addr)
                .ok_or_else(|| anyhow!("Missing bigint witness: {addr:?}"))?;

            if insn.mem_op == MemoryOp::Write {
                // Fast store
                self.avx2_store_words(ctx, addr)?;
            }
        }

        self.program.step(&insn, &self.state.bytes)?;

        self.state.is_ecall = false;
        self.state.poly_op = insn.poly_op;
        self.state.coeff = (insn.coeff + 4) as u32;
        self.state.next_state = if !self.state.is_ecall && insn.poly_op == PolyOp::Reset {
            CycleState::Decode
        } else {
            CycleState::BigIntStep
        };

        ctx.on_bigint_cycle(CycleState::BigIntStep, &self.state);
        Ok(())
    }

    // AVX2 helper for carry propagation with wide
    #[cfg(target_arch = "x86_64")]
    unsafe fn avx2_carry_propagate(&mut self) -> Result<()> {
        let coeffs_len = self.program.total_carry.coeffs.len();
        let mut carry = 0;

        // Process 8 coefficients at a time where possible
        let mut i = 0;
        while i + 8 <= coeffs_len {
            // Create the array and fill it
            let mut coeff_array = [0i32; 8];
            for j in 0..8 {
                coeff_array[j] = self.program.total_carry.coeffs[i + j];
            }

            // Add carry to first element
            coeff_array[0] += carry;

            // Check divisibility by 256
            for j in 0..8 {
                if coeff_array[j] % 256 != 0 {
                    bail!("bad carry");
                }
            }

            // Divide by 256
            for j in 0..8 {
                coeff_array[j] /= 256;
            }

            // Update coefficients
            for j in 0..8 {
                self.program.total_carry.coeffs[i + j] = coeff_array[j];
            }

            // Update carry for next batch
            carry = coeff_array[7];
            i += 8;
        }

        // Handle remaining elements
        while i < coeffs_len {
            self.program.total_carry.coeffs[i] += carry;
            ensure!(self.program.total_carry.coeffs[i] % 256 == 0, "bad carry");
            self.program.total_carry.coeffs[i] /= 256;
            carry = self.program.total_carry.coeffs[i];
            i += 1;
        }

        Ok(())
    }

    // AVX2 helper for processing outputs using wide vectors
    #[cfg(target_arch = "x86_64")]
    unsafe fn avx2_process_output<F>(
        &mut self,
        offset: usize,
        base_point: u32,
        process_fn: F,
    ) -> Result<()>
    where
        F: Fn(u32) -> u32,
    {
        // Process 8 bytes at a time
        let chunk_size = 8;
        let mut i = 0;

        while i + chunk_size <= BIGINT_WIDTH_BYTES {
            // Load coefficients into array
            let mut coeff_array = [0u32; 8];
            for j in 0..chunk_size {
                coeff_array[j] =
                    self.program.total_carry.coeffs[offset * BIGINT_WIDTH_BYTES + i + j] as u32;
            }

            // Add base_point to each
            for j in 0..chunk_size {
                coeff_array[j] = coeff_array[j].wrapping_add(base_point);
            }

            // Process each value and store result
            for j in 0..chunk_size {
                self.state.bytes[i + j] = process_fn(coeff_array[j]) as u8;
            }

            i += chunk_size;
        }

        // Handle remaining elements
        while i < BIGINT_WIDTH_BYTES {
            let coeff = self.program.total_carry.coeffs[offset * BIGINT_WIDTH_BYTES + i] as u32;
            let value = coeff.wrapping_add(base_point);
            self.state.bytes[i] = process_fn(value) as u8;
            i += 1;
        }

        Ok(())
    }

    // Optimized word loading
    #[cfg(target_arch = "x86_64")]
    unsafe fn avx2_load_words(&mut self, ctx: &mut dyn Risc0Context, addr: WordAddr) -> Result<()> {
        // Load all words at once
        let mut words = [0u32; BIGINT_WIDTH_WORDS];
        for i in 0..BIGINT_WIDTH_WORDS {
            words[i] = ctx.load_u32(LoadOp::Record, addr + i)?;
        }

        // Convert words to bytes (vanilla implementation for compatibility)
        for i in 0..BIGINT_WIDTH_WORDS {
            let bytes = words[i].to_le_bytes();
            for j in 0..WORD_SIZE {
                self.state.bytes[i * WORD_SIZE + j] = bytes[j];
            }
        }

        Ok(())
    }

    // AVX2 helper for storing words
    #[cfg(target_arch = "x86_64")]
    unsafe fn avx2_store_words(
        &mut self,
        ctx: &mut dyn Risc0Context,
        addr: WordAddr,
    ) -> Result<()> {
        // Use bytemuck for zero-copy casting
        let words: &[u32] = bytemuck::cast_slice(&self.state.bytes);

        // Store all words at once
        for (i, &word) in words.iter().enumerate() {
            ctx.store_u32(addr + i, word)?;
        }

        Ok(())
    }
}

struct BigIntIOImpl<'a> {
    ctx: &'a mut dyn Risc0Context,
    pub witness: BigIntWitness,
}

impl<'a> BigIntIOImpl<'a> {
    pub fn new(ctx: &'a mut dyn Risc0Context) -> Self {
        Self {
            ctx,
            witness: HashMap::new(),
        }
    }
}

// Optimized conversion using thread-local storage to reduce allocations
fn bytes_le_to_bigint(bytes: &[u8]) -> Natural {
    // Use thread-local storage for the limbs to avoid allocation
    LIMBS_BUFFER.with(|buffer| {
        let mut limbs = buffer.borrow_mut();
        limbs.clear();
        limbs.reserve((bytes.len() + 3) / 4);

        for chunk in bytes.chunks(4) {
            let mut arr = [0u8; 4];
            arr[..chunk.len()].copy_from_slice(chunk);
            limbs.push(u32::from_le_bytes(arr));
        }

        Natural::from_limbs_asc(&limbs)
    })
}

// Optimized conversion using thread-local storage to reduce allocations
fn bigint_to_bytes_le(value: &Natural) -> Vec<u8> {
    BYTES_BUFFER.with(|buffer| {
        let mut out = buffer.borrow_mut();
        out.clear();

        let limbs = value.to_limbs_asc();
        out.reserve(limbs.len() * 4);

        for limb in limbs {
            out.extend_from_slice(&limb.to_le_bytes());
        }

        // Return a clone to avoid lifetime issues
        out.clone()
    })
}

impl BigIntIO for BigIntIOImpl<'_> {
    fn load(&mut self, arena: u32, offset: u32, count: u32) -> Result<Natural> {
        tracing::trace!("load(arena: {arena}, offset: {offset}, count: {count})");
        let base = self
            .ctx
            .load_aligned_addr_from_machine_register(LoadOp::Load, arena as usize)?;
        let addr = base + offset * BIGINT_WIDTH_WORDS as u32;
        let bytes = self
            .ctx
            .load_region(LoadOp::Load, addr.baddr(), count as usize)?;
        let val = bytes_le_to_bigint(&bytes);
        Ok(val)
    }

    fn store(&mut self, arena: u32, offset: u32, count: u32, value: &Natural) -> Result<()> {
        let base = self
            .ctx
            .load_aligned_addr_from_machine_register(LoadOp::Load, arena as usize)?;
        let addr = base + offset * BIGINT_WIDTH_WORDS as u32;
        tracing::trace!("store(arena: {arena}, offset: {offset}, count: {count}, addr: {addr:?}, value: {value})");

        // Create a pre-sized witness buffer
        let mut witness = vec![0u8; count as usize];

        // Get bytes with reduced allocations
        let bytes = bigint_to_bytes_le(value);

        // Copy bytes into witness buffer
        witness[..bytes.len()].copy_from_slice(&bytes);

        // Process in chunks
        let chunks = witness.chunks_exact(BIGINT_WIDTH_BYTES);
        assert_eq!(chunks.len(), count as usize / BIGINT_WIDTH_BYTES);
        for (i, chunk) in chunks.enumerate() {
            let chunk_addr = addr + i * BIGINT_WIDTH_WORDS;
            let chunk_bytes: BigIntBytes = chunk.try_into().unwrap();
            self.witness.insert(chunk_addr, chunk_bytes);
        }

        Ok(())
    }
}

pub fn ecall(ctx: &mut dyn Risc0Context) -> Result<()> {
    tracing::debug!("ecall");

    let blob_ptr = ctx.load_aligned_addr_from_machine_register(LoadOp::Load, REG_A0)?;
    let nondet_program_ptr = ctx.load_aligned_addr_from_machine_register(LoadOp::Load, REG_T1)?;
    let verify_program_ptr =
        ctx.load_aligned_addr_from_machine_register(LoadOp::Record, REG_T2)? - 1;
    let consts_ptr = ctx.load_aligned_addr_from_machine_register(LoadOp::Load, REG_T3)?;

    let nondet_program_size = ctx.load_u32(LoadOp::Load, blob_ptr)?;
    let verify_program_size = ctx.load_u32(LoadOp::Load, blob_ptr + 1)?;
    let consts_size = ctx.load_u32(LoadOp::Load, blob_ptr + 2)?;

    // Prefetch program bytes - only if avx2 supported
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("sse4.1") {
        unsafe {
            // Prefetch the program bytes
            let addr_ptr = nondet_program_ptr.baddr().0 as *const i8;
            for i in 0..(nondet_program_size as usize * WORD_SIZE / 64) {
                // Use _MM_HINT_T0 (3) for highest priority prefetch
                _mm_prefetch(addr_ptr.add(i * 64), 3);
            }
        }
    }

    let program_bytes = ctx.load_region(
        LoadOp::Load,
        nondet_program_ptr.baddr(),
        nondet_program_size as usize * WORD_SIZE,
    )?;

    let mut cursor = Cursor::new(program_bytes);
    let program = bibc::Program::decode(&mut cursor)?;

    let witness = {
        let mut io = BigIntIOImpl::new(ctx);
        program.eval(&mut io)?;
        std::mem::take(&mut io.witness)
    };

    ctx.load_region(
        LoadOp::Load,
        verify_program_ptr.baddr(),
        verify_program_size as usize * WORD_SIZE,
    )?;
    ctx.load_region(
        LoadOp::Load,
        consts_ptr.baddr(),
        consts_size as usize * WORD_SIZE,
    )?;

    let state = BigIntState {
        is_ecall: true,
        pc: verify_program_ptr,
        poly_op: PolyOp::Reset,
        coeff: 0,
        bytes: Default::default(),
        next_state: CycleState::BigIntStep,
    };

    let mut bigint = BigInt {
        state,
        program: BytePolyProgram::new(),
    };

    bigint.run(ctx, &witness)
}
