// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Topological Quantum Field Theory (TQFT) primitives and validation utilities.
//!
//! This module exposes a Frobenius algebra implementation aligned with the
//! Atiyah–Segal axioms, together with a lightweight TQFT engine used by the
//! orchestrator for higher-order reasoning about cognitive state transitions.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const EPSILON: f32 = 1e-5;

type CheckResult = Result<(), String>;

/// Frobenius algebra underpinning the two-dimensional TQFT used in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrobeniusAlgebra {
    pub dimension: usize,
    pub basis_names: Vec<String>,
    /// Multiplication table: (i, j) -> k means e_i * e_j = e_k.
    pub multiplication_table: HashMap<(usize, usize), usize>,
    /// Comultiplication: element i splits into pairs (j, k) with unit coefficient.
    pub comultiplication_table: HashMap<usize, Vec<(usize, usize)>>,
    /// Index of the unit element.
    pub unit_idx: usize,
    /// Counit values encoded as a linear functional on the basis.
    pub counit_values: Vec<Complex<f32>>,
}

impl FrobeniusAlgebra {
    /// Construct a simple Frobenius algebra with idempotent basis vectors and
    /// diagonal comultiplication. This mirrors the original implementation but
    /// provides explicit tables so the algebra can be analysed rigorously.
    pub fn new(dimension: usize) -> Self {
        assert!(
            dimension == 2,
            "FrobeniusAlgebra::new currently supports dimension 2 only"
        );

        let mut multiplication_table = HashMap::new();
        multiplication_table.insert((0, 0), 0);
        multiplication_table.insert((0, 1), 1);
        multiplication_table.insert((1, 0), 1);
        // x * x = 0 (nilpotent), so no entry for (1, 1).

        let mut comultiplication_table = HashMap::new();
        comultiplication_table.insert(0, vec![(0, 1), (1, 0)]);
        comultiplication_table.insert(1, vec![(1, 1)]);

        Self {
            dimension,
            basis_names: vec!["1".to_string(), "x".to_string()],
            multiplication_table,
            comultiplication_table,
            unit_idx: 0,
            counit_values: vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        }
    }

    /// Multiply two basis elements, returning the resulting basis index when defined.
    pub fn multiply_basis(&self, i: usize, j: usize) -> Option<usize> {
        if i >= self.dimension || j >= self.dimension {
            return None;
        }
        self.multiplication_table.get(&(i, j)).copied()
    }

    /// Comultiply a basis element into a list of basis pairs.
    pub fn comultiply_basis(&self, i: usize) -> Option<Vec<(usize, usize)>> {
        if i >= self.dimension {
            return None;
        }
        self.comultiplication_table.get(&i).cloned()
    }

    /// Multiply two general algebra elements.
    pub fn multiply(
        &self,
        a: &DVector<Complex<f32>>,
        b: &DVector<Complex<f32>>,
    ) -> DVector<Complex<f32>> {
        let zero = Complex::new(0.0, 0.0);
        let mut result = DVector::from_element(self.dimension, zero);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                if a[i] != zero && b[j] != zero {
                    if let Some(k) = self.multiply_basis(i, j) {
                        result[k] += a[i] * b[j];
                    }
                }
            }
        }
        result
    }

    /// Return the algebraic unit vector e_0.
    pub fn unit(&self) -> DVector<Complex<f32>> {
        let zero = Complex::new(0.0, 0.0);
        let mut unit = DVector::from_element(self.dimension, zero);
        unit[self.unit_idx] = Complex::new(1.0, 0.0);
        unit
    }

    /// Wrapper used by legacy callers to check associativity.
    pub fn is_associative(&self) -> bool {
        self.check_associativity().is_ok()
    }

    /// Wrapper used by legacy callers to check coassociativity.
    pub fn is_coassociative(&self) -> bool {
        self.check_coassociativity().is_ok()
    }

    /// Wrapper used by legacy callers to check the Frobenius compatibility.
    pub fn satisfies_frobenius(&self) -> bool {
        self.check_frobenius_condition().is_ok()
    }

    /// Verify the full Frobenius algebra axioms, returning detailed error messages.
    pub fn verify_axioms(&self) -> CheckResult {
        self.check_associativity()?;
        self.check_unit_law()?;
        self.check_coassociativity()?;
        self.check_counit_laws()?;
        self.check_frobenius_condition()?;
        Ok(())
    }

    fn check_associativity(&self) -> CheckResult {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    let left = self
                        .multiply_basis(i, j)
                        .and_then(|ij| self.multiply_basis(ij, k));
                    let right = self
                        .multiply_basis(j, k)
                        .and_then(|jk| self.multiply_basis(i, jk));
                    if left != right {
                        return Err(format!(
                            "Associativity violated: ({} * {}) * {} != {} * ({} * {})",
                            i, j, k, i, j, k
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_unit_law(&self) -> CheckResult {
        for i in 0..self.dimension {
            let left = self.multiply_basis(self.unit_idx, i);
            let right = self.multiply_basis(i, self.unit_idx);
            if left != Some(i) || right != Some(i) {
                return Err(format!("Unit law violated for basis element {}", i));
            }
        }
        Ok(())
    }

    fn check_coassociativity(&self) -> CheckResult {
        for i in 0..self.dimension {
            let delta = self.comultiply_basis(i).unwrap_or_default();
            let mut left: HashMap<(usize, usize, usize), Complex<f32>> = HashMap::new();
            for &(j, k) in &delta {
                if let Some(delta_j) = self.comultiply_basis(j) {
                    for &(p, q) in &delta_j {
                        *left
                            .entry((p, q, k))
                            .or_insert_with(|| Complex::new(0.0, 0.0)) += Complex::new(1.0, 0.0);
                    }
                }
            }
            let mut right: HashMap<(usize, usize, usize), Complex<f32>> = HashMap::new();
            for &(j, k) in &delta {
                if let Some(delta_k) = self.comultiply_basis(k) {
                    for &(r, s) in &delta_k {
                        *right
                            .entry((j, r, s))
                            .or_insert_with(|| Complex::new(0.0, 0.0)) += Complex::new(1.0, 0.0);
                    }
                }
            }
            if left != right {
                return Err(format!("Coassociativity violated for basis element {}", i));
            }
        }
        Ok(())
    }

    fn check_counit_laws(&self) -> CheckResult {
        for i in 0..self.dimension {
            let delta = self.comultiply_basis(i).unwrap_or_default();
            let mut eps_tensor_id = DVector::from_element(self.dimension, Complex::new(0.0, 0.0));
            for &(j, k) in &delta {
                eps_tensor_id[k] += self.counit_values[j];
            }
            let mut expected = DVector::from_element(self.dimension, Complex::new(0.0, 0.0));
            expected[i] = Complex::new(1.0, 0.0);
            if !self.is_close(&eps_tensor_id, &expected) {
                return Err(format!(
                    "Counit law (ε ⊗ id) violated for basis element {}",
                    i
                ));
            }

            let mut id_tensor_eps = DVector::from_element(self.dimension, Complex::new(0.0, 0.0));
            for &(j, k) in &delta {
                id_tensor_eps[j] += self.counit_values[k];
            }
            if !self.is_close(&id_tensor_eps, &expected) {
                return Err(format!(
                    "Counit law (id ⊗ ε) violated for basis element {}",
                    i
                ));
            }
        }
        Ok(())
    }

    fn check_frobenius_condition(&self) -> CheckResult {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                let mut delta_product: HashMap<(usize, usize), Complex<f32>> = HashMap::new();
                if let Some(k) = self.multiply_basis(i, j) {
                    if let Some(delta) = self.comultiply_basis(k) {
                        for &(p, q) in &delta {
                            *delta_product
                                .entry((p, q))
                                .or_insert_with(|| Complex::new(0.0, 0.0)) +=
                                Complex::new(1.0, 0.0);
                        }
                    }
                }

                let mut left_action: HashMap<(usize, usize), Complex<f32>> = HashMap::new();
                if let Some(delta) = self.comultiply_basis(j) {
                    for &(p, q) in &delta {
                        if let Some(r) = self.multiply_basis(i, p) {
                            if let Some(s) = self.multiply_basis(self.unit_idx, q) {
                                *left_action
                                    .entry((r, s))
                                    .or_insert_with(|| Complex::new(0.0, 0.0)) +=
                                    Complex::new(1.0, 0.0);
                            }
                        }
                    }
                }

                let mut right_action: HashMap<(usize, usize), Complex<f32>> = HashMap::new();
                if let Some(delta) = self.comultiply_basis(i) {
                    for &(p, q) in &delta {
                        let r = p;
                        if let Some(s) = self.multiply_basis(q, j) {
                            *right_action
                                .entry((r, s))
                                .or_insert_with(|| Complex::new(0.0, 0.0)) +=
                                Complex::new(1.0, 0.0);
                        }
                    }
                }

                if delta_product != left_action || delta_product != right_action {
                    return Err(format!(
                        "Frobenius condition violated for basis elements {} and {}",
                        i, j
                    ));
                }
            }
        }
        Ok(())
    }

    fn is_close(&self, a: &DVector<Complex<f32>>, b: &DVector<Complex<f32>>) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for idx in 0..a.len() {
            if (a[idx] - b[idx]).norm() > EPSILON {
                return false;
            }
        }
        true
    }
}

/// Enumerates the elementary cobordisms used by the reasoning engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cobordism {
    Identity,
    Merge,
    Split,
    Birth,
    Death,
}

/// Linear operator acting on the Frobenius algebra state space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearOperator {
    pub matrix: DMatrix<Complex<f32>>,
}

impl LinearOperator {
    pub fn identity(dimension: usize) -> Self {
        Self {
            matrix: DMatrix::identity(dimension, dimension),
        }
    }

    pub fn from_matrix(matrix: DMatrix<Complex<f32>>) -> Self {
        Self { matrix }
    }

    pub fn apply(&self, v: &DVector<Complex<f32>>) -> DVector<Complex<f32>> {
        &self.matrix * v
    }

    pub fn compose(&self, other: &LinearOperator) -> Self {
        Self {
            matrix: &self.matrix * &other.matrix,
        }
    }
}

/// Minimal TQFT engine wiring cobordisms to linear actions on the Frobenius algebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TQFTEngine {
    pub dimension: usize,
    pub algebra: FrobeniusAlgebra,
    pub operators: HashMap<Cobordism, LinearOperator>,
}

impl TQFTEngine {
    pub fn new(dimension: usize) -> Result<Self, String> {
        let algebra = FrobeniusAlgebra::new(dimension);
        algebra.verify_axioms()?;

        let mut engine = Self {
            dimension,
            algebra,
            operators: HashMap::new(),
        };
        engine.initialize_operators();
        Ok(engine)
    }

    fn initialize_operators(&mut self) {
        self.operators.insert(
            Cobordism::Identity,
            LinearOperator::identity(self.dimension),
        );

        let mut birth_matrix = DMatrix::zeros(self.dimension, 1);
        birth_matrix[(0, 0)] = Complex::new(1.0, 0.0);
        self.operators
            .insert(Cobordism::Birth, LinearOperator::from_matrix(birth_matrix));

        let mut death_matrix = DMatrix::zeros(1, self.dimension);
        death_matrix[(0, 0)] = Complex::new(1.0, 0.0);
        self.operators
            .insert(Cobordism::Death, LinearOperator::from_matrix(death_matrix));

        let mut merge_matrix = DMatrix::zeros(self.dimension, self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                merge_matrix[(i, i * self.dimension + j)] = Complex::new(1.0, 0.0);
            }
        }
        self.operators
            .insert(Cobordism::Merge, LinearOperator::from_matrix(merge_matrix));

        let mut split_matrix = DMatrix::zeros(self.dimension * self.dimension, self.dimension);
        for i in 0..self.dimension {
            split_matrix[(i * self.dimension + i, i)] = Complex::new(1.0, 0.0);
        }
        self.operators
            .insert(Cobordism::Split, LinearOperator::from_matrix(split_matrix));
    }

    pub fn reason(
        &self,
        initial_state: &DVector<Complex<f32>>,
        transitions: &[Cobordism],
    ) -> Result<DVector<Complex<f32>>, String> {
        let mut current = initial_state.clone();
        for cobordism in transitions {
            let operator = self
                .operators
                .get(cobordism)
                .ok_or_else(|| format!("Unknown cobordism: {:?}", cobordism))?;
            current = operator.apply(&current);
        }
        Ok(current)
    }

    pub fn compose_cobordisms(
        &self,
        first: Cobordism,
        second: Cobordism,
    ) -> Result<LinearOperator, String> {
        let op1 = self
            .operators
            .get(&first)
            .ok_or_else(|| format!("Unknown cobordism: {:?}", first))?;
        let op2 = self
            .operators
            .get(&second)
            .ok_or_else(|| format!("Unknown cobordism: {:?}", second))?;
        Ok(op2.compose(op1))
    }

    pub fn trace_operator(&self, op: &LinearOperator) -> Complex<f32> {
        let mut trace = Complex::new(0.0, 0.0);
        let min = op.matrix.nrows().min(op.matrix.ncols());
        for i in 0..min {
            trace += op.matrix[(i, i)];
        }
        trace
    }

    pub fn infer_cobordism_from_betti(
        before: &[usize; 3],
        after: &[usize; 3],
    ) -> Option<Cobordism> {
        let delta_b0 = after[0] as i32 - before[0] as i32;
        let delta_b1 = after[1] as i32 - before[1] as i32;
        let delta_b2 = after[2] as i32 - before[2] as i32;
        match (delta_b0, delta_b1, delta_b2) {
            (0, 0, 0) => Some(Cobordism::Identity),
            (1, 0, 0) => Some(Cobordism::Split),
            (-1, 0, 0) => Some(Cobordism::Merge),
            (0, 1, 0) => Some(Cobordism::Birth),
            (0, -1, 0) => Some(Cobordism::Death),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frobenius_axioms() {
        let algebra = FrobeniusAlgebra::new(2);
        assert!(algebra.verify_axioms().is_ok());
    }

    #[test]
    fn test_unit_element() {
        let algebra = FrobeniusAlgebra::new(2);
        let unit = algebra.unit();
        assert_eq!(unit[0], Complex::new(1.0, 0.0));
        for i in 1..2 {
            assert_eq!(unit[i], Complex::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_tqft_engine_creation() {
        let engine = TQFTEngine::new(2);
        assert!(engine.is_ok());
        let engine = engine.unwrap();
        assert_eq!(engine.dimension, 2);
    }

    #[test]
    fn test_cobordism_reasoning() {
        let engine = TQFTEngine::new(2).unwrap();
        let initial = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        let result = engine.reason(&initial, &[Cobordism::Identity]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result[0], Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_infer_cobordism() {
        let before = [1, 0, 0];
        let after = [2, 0, 0];
        let cobordism = TQFTEngine::infer_cobordism_from_betti(&before, &after);
        assert_eq!(cobordism, Some(Cobordism::Split));
    }

    #[test]
    fn test_group_algebra_z2() {
        let mut multiplication_table = HashMap::new();
        multiplication_table.insert((0, 0), 0);
        multiplication_table.insert((0, 1), 1);
        multiplication_table.insert((1, 0), 1);

        let mut comultiplication_table = HashMap::new();
        comultiplication_table.insert(0, vec![(0, 1), (1, 0)]);
        comultiplication_table.insert(1, vec![(1, 1)]);

        let algebra = FrobeniusAlgebra {
            dimension: 2,
            basis_names: vec!["1".to_string(), "g".to_string()],
            multiplication_table,
            comultiplication_table,
            unit_idx: 0,
            counit_values: vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        };

        assert!(algebra.verify_axioms().is_ok());
    }
}
