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

use std::cmp::max;

use crate::zirgen::circuit::ExtVal;
use anyhow::{ensure, Result};
use auto_ops::impl_op_ex;
use once_cell::sync::Lazy;
use rayon;
use rayon::prelude::*;
use risc0_zkp::field::Elem as _;
use smallvec::{smallvec, SmallVec};
use wide::{i32x4, i32x8};

use super::bigint::{BigIntState, Instruction, PolyOp, BIGINT_WIDTH_BYTES};

#[derive(Debug)]
pub(crate) struct BytePolyProgram {
    pub(crate) in_carry: bool,
    pub(crate) poly: BytePolynomial,
    pub(crate) term: BytePolynomial,
    pub(crate) total: BytePolynomial,
    pub(crate) total_carry: BytePolynomial,
}

impl BytePolyProgram {
    pub(crate) fn new() -> Self {
        Self {
            in_carry: false,
            poly: BytePolynomial::zero(),
            term: BytePolynomial::one(),
            total: BytePolynomial::zero(),
            total_carry: BytePolynomial::default(),
        }
    }

    pub(crate) fn step(&mut self, insn: &Instruction, witness: &[u8]) -> Result<()> {
        let coeffs = witness.iter().map(|x| *x as i32);
        let delta_poly = BytePolynomial {
            coeffs: SmallVec::from_iter(coeffs),
        };

        let new_poly = &self.poly + &delta_poly;
        match insn.poly_op {
            PolyOp::Reset => {
                self.reset();
            }
            PolyOp::Shift => {
                self.poly = new_poly.shift();
            }
            PolyOp::SetTerm => {
                self.poly = BytePolynomial::zero();
                self.term = new_poly.clone();
            }
            PolyOp::AddTotal => {
                self.total = &self.total + &new_poly * &self.term * insn.coeff;
                self.term = BytePolynomial::one();
                self.poly = BytePolynomial::zero();
            }
            PolyOp::Carry1 => {
                let neg_poly = BytePolynomial {
                    coeffs: SmallVec::from_elem(-128, BIGINT_WIDTH_BYTES),
                };
                self.poly = &self.poly + (&delta_poly + neg_poly) * 0x4000;
            }
            PolyOp::Carry2 => {
                self.poly = &self.poly + &delta_poly * 256;
            }
            PolyOp::EqZero => {
                let bp = BytePolynomial {
                    coeffs: SmallVec::from_slice(&[-256, 1]),
                };
                self.total = &self.total + bp * &new_poly;
                self.total.eqz()?;
                self.reset();
                self.in_carry = false;
            }
        }

        tracing::trace!(
            "delta_poly[0]: {}, new_poly[0]: {}, poly[0]: {}, term[0]: {}, total[0]: {}",
            &delta_poly.coeffs[0],
            new_poly.coeffs[0],
            self.poly.coeffs[0],
            self.term.coeffs[0],
            self.total.coeffs[0],
        );

        Ok(())
    }

    fn reset(&mut self) {
        self.poly = BytePolynomial::zero();
        self.term = BytePolynomial::one();
        self.total = BytePolynomial::zero();
    }

    #[allow(dead_code)]
    pub fn is_zero_in_range_simd(program: &BytePolyProgram, range: std::ops::Range<usize>) -> bool {
        let slice = &program.total_carry.coeffs[range.clone()];
        let len = slice.len();

        // Process in chunks of 8
        let chunks = len / 8;
        for chunk in 0..chunks {
            let start = chunk * 8;

            // Create a SIMD vector
            let vec = i32x8::new([
                slice[start],
                slice[start + 1],
                slice[start + 2],
                slice[start + 3],
                slice[start + 4],
                slice[start + 5],
                slice[start + 6],
                slice[start + 7],
            ]);

            // Create a zero vector
            let zero_vec = i32x8::splat(0);

            // Check if not equal to zero
            if vec != zero_vec {
                return false;
            }
        }

        // Check remaining elements
        let remaining_start = chunks * 8;
        slice[remaining_start..].iter().all(|&x| x == 0)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BytePolynomial {
    pub coeffs: SmallVec<[i32; 64]>,
}

impl BytePolynomial {
    pub(crate) fn one() -> Self {
        Self {
            coeffs: smallvec![1],
        }
    }

    pub(crate) fn zero() -> Self {
        Self {
            coeffs: smallvec![0],
        }
    }

    pub(crate) fn shift(&self) -> Self {
        let mut ret = self.clone();
        ret.coeffs.insert_from_slice(0, &[0; BIGINT_WIDTH_BYTES]);
        ret
    }

    pub(crate) fn eqz(&self) -> Result<()> {
        ensure!(
            self.coeffs.iter().all(|&coeff| coeff == 0),
            "Invalid eqz in bigint program"
        );
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn add_in_place(&mut self, rhs: &BytePolynomial) {
        if self.coeffs.len() < rhs.coeffs.len() {
            self.coeffs.resize(rhs.coeffs.len(), 0);
        }

        if rhs.coeffs.len() >= 8 {
            add_polynomials_simd_8(&mut self.coeffs, &rhs.coeffs).unwrap();
        } else {
            for (i, &val) in rhs.coeffs.iter().enumerate() {
                self.coeffs[i] += val;
            }
        }
    }

    pub(crate) fn mul_const_in_place(&mut self, constant: i32) {
        if self.coeffs.len() >= 8 {
            byte_poly_mul_const_simd(&mut self.coeffs, constant).unwrap();
        } else {
            for coeff in self.coeffs.iter_mut() {
                *coeff *= constant;
            }
        }
    }
}

fn byte_poly_add(lhs: &BytePolynomial, rhs: &BytePolynomial) -> BytePolynomial {
    let max_len = max(lhs.coeffs.len(), rhs.coeffs.len());
    let mut ret = smallvec![0; max_len];

    // Copy lhs coefficients to the result vector first
    if !lhs.coeffs.is_empty() {
        ret[..lhs.coeffs.len()].copy_from_slice(&lhs.coeffs);
    }

    // Then use SIMD to add the second polynomial to the result
    if !rhs.coeffs.is_empty() {
        // Use SIMD for the common part
        if rhs.coeffs.len() >= 8 {
            add_polynomials_simd_8(&mut ret[..rhs.coeffs.len()], &rhs.coeffs).unwrap();
        } else if rhs.coeffs.len() >= 4 {
            add_polynomials_simd_4(&mut ret[..rhs.coeffs.len()], &rhs.coeffs).unwrap();
        } else {
            // For very small polynomials, just use normal addition
            for (i, &coeff) in rhs.coeffs.iter().enumerate() {
                ret[i] += coeff;
            }
        }
    }

    BytePolynomial { coeffs: ret }
}

fn byte_poly_mul(lhs: &BytePolynomial, rhs: &BytePolynomial) -> BytePolynomial {
    // For very small polynomials, use the naive approach as it's faster
    if lhs.coeffs.len() < 8 || rhs.coeffs.len() < 8 {
        return byte_poly_mul_naive(lhs, rhs);
    }

    // For medium-sized polynomials, use sequential Karatsuba
    if lhs.coeffs.len() < 64 || rhs.coeffs.len() < 64 {
        let result_coeffs = karatsuba_multiply(&lhs.coeffs, &rhs.coeffs);
        return BytePolynomial {
            coeffs: result_coeffs,
        };
    }

    // For large polynomials, use parallel Karatsuba
    let result_coeffs = parallel_karatsuba_multiply(&lhs.coeffs, &rhs.coeffs);
    BytePolynomial {
        coeffs: result_coeffs,
    }
}

// Original naive implementation kept for small polynomials
fn byte_poly_mul_naive(lhs: &BytePolynomial, rhs: &BytePolynomial) -> BytePolynomial {
    let mut ret = smallvec![0; lhs.coeffs.len() + rhs.coeffs.len()];
    for (i, lhs) in lhs.coeffs.iter().enumerate() {
        for (j, rhs) in rhs.coeffs.iter().enumerate() {
            ret[i + j] += lhs * rhs;
        }
    }
    BytePolynomial { coeffs: ret }
}

// Karatsuba algorithm for polynomial multiplication: O(n^1.58) complexity
fn karatsuba_multiply(x: &[i32], y: &[i32]) -> SmallVec<[i32; 64]> {
    let n = max(x.len(), y.len());

    // Base case for recursion
    if n <= 16 {
        let mut result = smallvec![0; x.len() + y.len()];
        for (i, &x_i) in x.iter().enumerate() {
            for (j, &y_j) in y.iter().enumerate() {
                result[i + j] += x_i * y_j;
            }
        }
        return result;
    }

    // Split point
    let m = n / 2;

    // Split x and y into lower and upper parts
    let x_low: SmallVec<[i32; 64]> = x.iter().take(std::cmp::min(m, x.len())).copied().collect();
    let x_high: SmallVec<[i32; 64]> = x.iter().skip(std::cmp::min(m, x.len())).copied().collect();

    let y_low: SmallVec<[i32; 64]> = y.iter().take(std::cmp::min(m, y.len())).copied().collect();
    let y_high: SmallVec<[i32; 64]> = y.iter().skip(std::cmp::min(m, y.len())).copied().collect();

    // Karatsuba steps
    let z0 = karatsuba_multiply(&x_low, &y_low);
    let z2 = karatsuba_multiply(&x_high, &y_high);

    // Compute sum of low and high parts
    let mut x_sum = x_low.clone();
    for (i, &val) in x_high.iter().enumerate() {
        if i < x_sum.len() {
            x_sum[i] += val;
        } else {
            x_sum.push(val);
        }
    }

    let mut y_sum = y_low.clone();
    for (i, &val) in y_high.iter().enumerate() {
        if i < y_sum.len() {
            y_sum[i] += val;
        } else {
            y_sum.push(val);
        }
    }

    let z1_full = karatsuba_multiply(&x_sum, &y_sum);

    // z1 = z1_full - z0 - z2
    let mut z1 = z1_full;
    for i in 0..z0.len() {
        z1[i] -= z0[i];
    }
    for i in 0..z2.len() {
        z1[i] -= z2[i];
    }

    // Combine results
    let result_len = x.len() + y.len();
    let mut result = smallvec![0; result_len];

    // Add z0 to result
    for (i, &val) in z0.iter().enumerate() {
        if i < result_len {
            result[i] += val;
        }
    }

    // Add z1 * B^m
    for (i, &val) in z1.iter().enumerate() {
        let idx = i + m;
        if idx < result_len {
            result[idx] += val;
        }
    }

    // Add z2 * B^(2*m)
    for (i, &val) in z2.iter().enumerate() {
        let idx = i + 2 * m;
        if idx < result_len {
            result[idx] += val;
        }
    }

    result
}

fn byte_poly_mul_const(lhs: &BytePolynomial, rhs: i32) -> BytePolynomial {
    let mut ret = lhs.clone();
    ret.mul_const_in_place(rhs);
    ret
}

#[allow(dead_code)]
fn byte_poly_mul_const_simd(coeffs: &mut [i32], constant: i32) -> Result<()> {
    for chunk in 0..(coeffs.len() / 8) {
        let start = chunk * 8;
        let coeffs_vec = i32x8::new([
            coeffs[start],
            coeffs[start + 1],
            coeffs[start + 2],
            coeffs[start + 3],
            coeffs[start + 4],
            coeffs[start + 5],
            coeffs[start + 6],
            coeffs[start + 7],
        ]);

        // SIMD multiplication by constant
        let const_vec = i32x8::splat(constant);
        let result = coeffs_vec * const_vec;

        // Store result back
        let result_array = result.to_array();
        coeffs[start..start + 8].copy_from_slice(&result_array);
    }

    // Handle remaining elements - fix the needless range loop
    let remaining_start = (coeffs.len() / 8) * 8;
    for coeff in &mut coeffs[remaining_start..] {
        *coeff *= constant;
    }

    Ok(())
}

impl_op_ex!(+|a: &BytePolynomial, b: &BytePolynomial| -> BytePolynomial { byte_poly_add(a, b) });
impl_op_ex!(*|a: &BytePolynomial, b: &BytePolynomial| -> BytePolynomial { byte_poly_mul(a, b) });
impl_op_ex!(*|a: &BytePolynomial, b: i32| -> BytePolynomial { byte_poly_mul_const(a, b) });

const MAX_POWERS: usize = BIGINT_WIDTH_BYTES + 1;

#[derive(Clone, Debug)]
pub(crate) struct BigIntAccumState {
    pub poly: ExtVal,
    pub term: ExtVal,
    pub total: ExtVal,
}

impl BigIntAccumState {
    fn new() -> Self {
        Self {
            poly: ExtVal::ZERO,
            term: ExtVal::ONE,
            total: ExtVal::ZERO,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BigIntAccum {
    pub state: BigIntAccumState,
    powers: [ExtVal; MAX_POWERS],
    neg_poly: ExtVal,
}

// Add ExtVal constants
static EXT_VAL_ZERO: Lazy<ExtVal> = Lazy::new(|| ExtVal::ZERO);
static EXT_VAL_ONE: Lazy<ExtVal> = Lazy::new(|| ExtVal::ONE);
static EXT_VAL_0X4000: Lazy<ExtVal> = Lazy::new(|| ExtVal::from_u32(0x4000));
static EXT_VAL_0X100: Lazy<ExtVal> = Lazy::new(|| ExtVal::from_u32(0x100));
static EXT_VAL_4: Lazy<ExtVal> = Lazy::new(|| ExtVal::from_u32(4));
static EXT_VAL_128: Lazy<ExtVal> = Lazy::new(|| ExtVal::from_u32(128));

impl BigIntAccum {
    pub(crate) fn new(mix: ExtVal) -> Self {
        let mut powers = [ExtVal::default(); MAX_POWERS];
        let mut cur = *EXT_VAL_ONE;
        for power in powers.iter_mut() {
            *power = cur;
            cur *= mix;
        }

        let neg_poly = powers
            .iter()
            .take(BIGINT_WIDTH_BYTES)
            .fold(*EXT_VAL_ZERO, |acc, power| acc + *power * *EXT_VAL_128);

        Self {
            state: BigIntAccumState::new(),
            powers,
            neg_poly,
        }
    }

    pub(crate) fn step(&mut self, state: &BigIntState) -> Result<()> {
        // Pre-compute coefficient ExtVals for better SIMD optimization
        let mut delta_poly = *EXT_VAL_ZERO;
        for i in 0..BIGINT_WIDTH_BYTES {
            delta_poly += self.powers[i] * ExtVal::from_u32(state.bytes[i] as u32);
        }

        let new_poly = self.state.poly + delta_poly;

        match state.poly_op {
            PolyOp::Reset => self.reset(),
            PolyOp::Shift => {
                self.state.poly = new_poly * self.powers[BIGINT_WIDTH_BYTES];
            }
            PolyOp::SetTerm => {
                self.state.poly = *EXT_VAL_ZERO;
                self.state.term = new_poly;
            }
            PolyOp::AddTotal => {
                let coeff = ExtVal::from_u32(state.coeff) - *EXT_VAL_4;
                self.state.total += coeff * self.state.term * new_poly;
                self.state.poly = *EXT_VAL_ZERO;
                self.state.term = *EXT_VAL_ONE;
            }
            PolyOp::Carry1 => {
                self.state.poly += (delta_poly - self.neg_poly) * *EXT_VAL_0X4000;
            }
            PolyOp::Carry2 => {
                self.state.poly += delta_poly * *EXT_VAL_0X100;
            }
            PolyOp::EqZero => {
                let carry = self.powers[1] - ExtVal::from_u32(0x100);
                let goal = self.state.total + new_poly * carry;
                ensure!(goal == ExtVal::ZERO, "Invalid eqz in bigint accum");
                self.reset();
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.state = BigIntAccumState::new();
    }
}

fn add_polynomials_simd_8(dest: &mut [i32], src: &[i32]) -> Result<()> {
    for chunk in 0..(dest.len() / 8) {
        let start = chunk * 8;

        // Load 8 elements at once
        let dest_vec = i32x8::new([
            dest[start],
            dest[start + 1],
            dest[start + 2],
            dest[start + 3],
            dest[start + 4],
            dest[start + 5],
            dest[start + 6],
            dest[start + 7],
        ]);

        let src_vec = i32x8::new([
            src[start],
            src[start + 1],
            src[start + 2],
            src[start + 3],
            src[start + 4],
            src[start + 5],
            src[start + 6],
            src[start + 7],
        ]);

        // SIMD addition
        let result = dest_vec + src_vec;

        // Store result
        let result_array = result.to_array();
        dest[start..start + 8].copy_from_slice(&result_array);
    }

    // Handle remaining elements
    let remaining_start = (dest.len() / 8) * 8;
    for i in remaining_start..dest.len() {
        dest[i] += src[i];
    }

    Ok(())
}

fn add_polynomials_simd_4(dest: &mut [i32], src: &[i32]) -> Result<()> {
    for chunk in 0..(dest.len() / 8) {
        let start = chunk * 8;

        // Load 8 elements at once
        let dest_vec = i32x4::new([
            dest[start],
            dest[start + 1],
            dest[start + 2],
            dest[start + 3],
        ]);

        let src_vec = i32x4::new([src[start], src[start + 1], src[start + 2], src[start + 3]]);

        // SIMD addition
        let result = dest_vec + src_vec;

        // Store result
        let result_array = result.to_array();
        dest[start..start + 8].copy_from_slice(&result_array);
    }

    // Handle remaining elements
    let remaining_start = (dest.len() / 4) * 4;
    for i in remaining_start..dest.len() {
        dest[i] += src[i];
    }

    Ok(())
}

// Modify the parallel_karatsuba_multiply function to use more threads
fn parallel_karatsuba_multiply(x: &[i32], y: &[i32]) -> SmallVec<[i32; 64]> {
    let n = max(x.len(), y.len());

    // Lower the threshold for parallelization to increase thread usage
    if n <= 32 {
        // Reduced from 64
        return karatsuba_multiply(x, y);
    }

    // Split point
    let m = n / 2;

    // Split x and y into lower and upper parts - in parallel
    let (x_parts, y_parts) = rayon::join(
        || {
            let low = x
                .iter()
                .take(std::cmp::min(m, x.len()))
                .copied()
                .collect::<SmallVec<[i32; 64]>>();
            let high = x
                .iter()
                .skip(std::cmp::min(m, x.len()))
                .copied()
                .collect::<SmallVec<[i32; 64]>>();
            (low, high)
        },
        || {
            let low = y
                .iter()
                .take(std::cmp::min(m, y.len()))
                .copied()
                .collect::<SmallVec<[i32; 64]>>();
            let high = y
                .iter()
                .skip(std::cmp::min(m, y.len()))
                .copied()
                .collect::<SmallVec<[i32; 64]>>();
            (low, high)
        },
    );

    let x_low = x_parts.0;
    let x_high = x_parts.1;
    let y_low = y_parts.0;
    let y_high = y_parts.1;

    // Compute sum of low and high parts in parallel
    let (x_sum, y_sum) = rayon::join(
        || {
            let mut sum = x_low.clone();
            // Use parallel iterator for large arrays
            if x_high.len() > 64 {
                // Create a new vector with the right size
                if sum.len() < x_high.len() {
                    sum.resize(x_high.len(), 0);
                }

                // Parallel addition using chunks
                sum.par_iter_mut()
                    .zip(x_high.par_iter())
                    .for_each(|(a, &b)| *a += b);
            } else {
                // Sequential for smaller arrays
                for (i, &val) in x_high.iter().enumerate() {
                    if i < sum.len() {
                        sum[i] += val;
                    } else {
                        sum.push(val);
                    }
                }
            }
            sum
        },
        || {
            let mut sum = y_low.clone();
            // Use parallel iterator for large arrays
            if y_high.len() > 64 {
                // Create a new vector with the right size
                if sum.len() < y_high.len() {
                    sum.resize(y_high.len(), 0);
                }

                // Parallel addition using chunks
                sum.par_iter_mut()
                    .zip(y_high.par_iter())
                    .for_each(|(a, &b)| *a += b);
            } else {
                // Sequential for smaller arrays
                for (i, &val) in y_high.iter().enumerate() {
                    if i < sum.len() {
                        sum[i] += val;
                    } else {
                        sum.push(val);
                    }
                }
            }
            sum
        },
    );

    // Use rayon to compute z0, z1, and z2 in parallel
    let ((z0, z2), z1_full) = rayon::join(
        || {
            rayon::join(
                || parallel_karatsuba_multiply(&x_low, &y_low),
                || parallel_karatsuba_multiply(&x_high, &y_high),
            )
        },
        || parallel_karatsuba_multiply(&x_sum, &y_sum),
    );

    // z1 = z1_full - z0 - z2, do this in parallel for large arrays
    let mut z1 = z1_full;

    if z1.len() > 128 {
        // Parallel subtraction for large arrays
        z1.par_iter_mut().enumerate().for_each(|(i, val)| {
            if i < z0.len() {
                *val -= z0[i];
            }
            if i < z2.len() {
                *val -= z2[i];
            }
        });
    } else {
        // Sequential for smaller arrays
        for i in 0..z0.len() {
            if i < z1.len() {
                z1[i] -= z0[i];
            }
        }
        for i in 0..z2.len() {
            if i < z1.len() {
                z1[i] -= z2[i];
            }
        }
    }

    // Combine results
    let result_len = x.len() + y.len();
    let mut result = smallvec![0; result_len];

    // For large results, parallelize the final combination step
    if result_len > 256 {
        // Prepare an array of operations to perform
        let operations: Vec<(usize, i32)> = z0
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .chain(z1.iter().enumerate().map(|(i, &v)| (i + m, v)))
            .chain(z2.iter().enumerate().map(|(i, &v)| (i + 2 * m, v)))
            .filter(|(idx, _)| *idx < result_len)
            .collect();

        // Group operations by index
        let mut index_map: std::collections::HashMap<usize, Vec<i32>> =
            std::collections::HashMap::new();
        for (idx, val) in operations {
            index_map.entry(idx).or_default().push(val);
        }

        // Apply operations in parallel
        result.par_iter_mut().enumerate().for_each(|(i, val)| {
            if let Some(values) = index_map.get(&i) {
                *val += values.iter().sum::<i32>();
            }
        });
    } else {
        // Sequential addition for smaller results
        for (i, &val) in z0.iter().enumerate() {
            if i < result_len {
                result[i] += val;
            }
        }

        for (i, &val) in z1.iter().enumerate() {
            let idx = i + m;
            if idx < result_len {
                result[idx] += val;
            }
        }

        for (i, &val) in z2.iter().enumerate() {
            let idx = i + 2 * m;
            if idx < result_len {
                result[idx] += val;
            }
        }
    }

    result
}
