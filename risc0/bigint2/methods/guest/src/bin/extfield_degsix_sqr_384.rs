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

#[cfg(feature = "num-bigint-dig")]
extern crate num_bigint_dig as num_bigint;

#[allow(unused)]
use risc0_zkvm::guest::env;

#[cfg(any(feature = "num-bigint-dig", feature = "num-bigint"))]
fn main() {
    use num_bigint::BigUint;
    use risc0_bigint2::ToBigInt2Buffer;

    let (inp_0, inp_1, inp_2, inp_3, inp_4, inp_5,
         prime, primesqr5):
        (BigUint, BigUint, BigUint, BigUint, BigUint, BigUint,
         BigUint, BigUint) = env::read();
    let inp = [inp_0.to_u32_array(), inp_1.to_u32_array(), inp_2.to_u32_array(),
               inp_3.to_u32_array(), inp_4.to_u32_array(), inp_5.to_u32_array()];
    let prime = prime.to_u32_array();
    let primesqr5 = primesqr5.to_u32_array();

    let mut result = [[0u32; risc0_bigint2::field::FIELD_384_WIDTH_WORDS]; risc0_bigint2::field::EXT_DEGREE_6];
    risc0_bigint2::field::extfield_degsix_sqr_384(&inp, &prime, &primesqr5, &mut result);

    let result = (BigUint::from_slice(&result[0]), BigUint::from_slice(&result[1]), BigUint::from_slice(&result[2])
                  BigUint::from_slice(&result[3]), BigUint::from_slice(&result[4]), BigUint::from_slice(&result[5]));

    env::commit(&result);
}

#[cfg(not(any(feature = "num-bigint-dig", feature = "num-bigint")))]
fn main() {
    panic!("No bigint library enabled");
}
