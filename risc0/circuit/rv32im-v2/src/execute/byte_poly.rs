use std::cmp::max;

use anyhow::{ensure, Result};
use auto_ops::impl_op_ex;
use smallvec::{smallvec, SmallVec};
use wide::i32x8;

use super::bigint::{Instruction, PolyOp, BIGINT_WIDTH_BYTES};

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
            PolyOp::Reset => self.reset(),
            PolyOp::Shift => self.poly = new_poly.shift(),
            PolyOp::SetTerm => {
                self.poly = BytePolynomial::zero();
                self.term = new_poly.clone();
            }
            PolyOp::AddTotal => {
                let product = &new_poly * &self.term * insn.coeff;
                add_polynomials_simd(&mut self.total.coeffs, &product.coeffs)?;
                self.term = BytePolynomial::one();
                self.poly = BytePolynomial::zero();
            }
            PolyOp::Carry1 => {
                let neg_poly = BytePolynomial {
                    coeffs: SmallVec::from_elem(-128, BIGINT_WIDTH_BYTES),
                };
                self.poly = &self.poly + (&delta_poly + neg_poly) * 64 * 256;
            }
            PolyOp::Carry2 => {
                self.poly = &self.poly + &delta_poly * 256;
            }
            PolyOp::EqZero => {
                let bp = BytePolynomial {
                    coeffs: SmallVec::from_slice(&[-256, 1]),
                };
                add_polynomials_simd(&mut self.total.coeffs, &(bp * &new_poly).coeffs)?;
                self.total.eqz()?;
                self.reset();
                self.in_carry = false;
            }
        }

        tracing::trace!(
            "delta_poly[0]: {}, new_poly[0]: {}, poly[0]: {}, term[0]: {}, total[0]: {}",
            delta_poly.coeffs[0],
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
}

impl_op_ex!(+|a: &BytePolynomial, b: &BytePolynomial| -> BytePolynomial {
    let len = max(a.coeffs.len(), b.coeffs.len());
    let mut ret = smallvec![0; len];
    for i in 0..len {
        ret[i] = a.coeffs.get(i).unwrap_or(&0) + b.coeffs.get(i).unwrap_or(&0);
    }
    BytePolynomial { coeffs: ret }
});

impl_op_ex!(
    *|a: &BytePolynomial, b: &BytePolynomial| -> BytePolynomial {
        let mut ret = smallvec![0; a.coeffs.len() + b.coeffs.len() - 1];
        for (i, &ai) in a.coeffs.iter().enumerate() {
            for (j, &bj) in b.coeffs.iter().enumerate() {
                ret[i + j] += ai * bj;
            }
        }
        BytePolynomial { coeffs: ret }
    }
);

impl_op_ex!(*|a: &BytePolynomial, b: i32| -> BytePolynomial {
    BytePolynomial {
        coeffs: a.coeffs.iter().map(|&c| c * b).collect(),
    }
});

fn add_polynomials_simd(dest: &mut [i32], src: &[i32]) -> Result<()> {
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
