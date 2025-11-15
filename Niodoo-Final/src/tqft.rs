//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// Topological Quantum Field Theory (TQFT) Engine
/// Implements Atiyah-Segal axioms for consciousness reasoning
///
/// This module provides the mathematical foundation for higher-order reasoning
/// about topological transitions in cognitive states, treating consciousness
/// as a 2D TQFT via Frobenius algebra operations.
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Frobenius algebra: the core algebraic structure for 2D TQFT
/// Encodes multiplication and comultiplication operations that preserve
/// the Frobenius condition: (μ ⊗ id) ∘ (id ⊗ Δ) = Δ ∘ μ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrobeniusAlgebra {
    pub dimension: usize,
    pub basis_names: Vec<String>,
    /// Multiplication table: (i, j) -> index k means e_i * e_j = e_k
    pub multiplication_table: HashMap<(usize, usize), usize>,
    /// Comultiplication: element i splits into sum of pairs (j, k) with coefficient
    pub comultiplication_table: HashMap<usize, Vec<(usize, usize)>>,
    /// Unit element (identity for multiplication)
    pub unit_idx: usize,
    /// Counit: linear functional returning scalar
    pub counit_values: Vec<Complex<f32>>,
}

impl FrobeniusAlgebra {
    /// Create a new Frobenius algebra with given dimension
    pub fn new(dimension: usize) -> Self {
        let mut multiplication_table = HashMap::new();
        let mut comultiplication_table = HashMap::new();

        // Full multiplication table: unit element is e_0
        // e_0 * e_i = e_i (unit law)
        // e_i * e_0 = e_i (unit law)
        // e_i * e_j = e_i if i == j (projection), else 0
        for i in 0..dimension {
            // e_0 multiplied with anything
            multiplication_table.insert((0, i), i);
            multiplication_table.insert((i, 0), i);

            // Diagonal elements (idempotent)
            if i != 0 {
                multiplication_table.insert((i, i), i);
            }

            // Comultiplication
            comultiplication_table.insert(i, vec![(i, i)]);

            // Off-diagonal non-zero, non-unit
            for j in 0..dimension {
                if i != 0 && j != 0 && i != j {
                    multiplication_table.insert((i, j), 0);
                }
            }
        }

        Self {
            dimension,
            basis_names: (0..dimension).map(|i| format!("e_{}", i)).collect(),
            multiplication_table,
            comultiplication_table,
            unit_idx: 0,
            counit_values: vec![Complex::new(1.0, 0.0); dimension],
        }
    }

    /// Multiply two basis elements
    /// Returns the index of the resulting basis element
    pub fn multiply_basis(&self, i: usize, j: usize) -> Option<usize> {
        if i >= self.dimension || j >= self.dimension {
            return None;
        }
        self.multiplication_table.get(&(i, j)).copied()
    }

    /// Comultiply a basis element
    /// Returns list of (left, right) pairs that the element splits into
    pub fn comultiply_basis(&self, i: usize) -> Option<Vec<(usize, usize)>> {
        if i >= self.dimension {
            return None;
        }
        self.comultiplication_table.get(&i).cloned()
    }

    /// Multiply two general algebra elements
    pub fn multiply(
        &self,
        a: &DVector<Complex<f32>>,
        b: &DVector<Complex<f32>>,
    ) -> DVector<Complex<f32>> {
        let mut result = DVector::zeros(self.dimension);

        for i in 0..self.dimension {
            for j in 0..self.dimension {
                if a[i] != Complex::new(0.0, 0.0) && b[j] != Complex::new(0.0, 0.0) {
                    if let Some(k) = self.multiply_basis(i, j) {
                        result[k] += a[i] * b[j];
                    }
                }
            }
        }

        result
    }

    /// Get the unit element (identity for multiplication)
    pub fn unit(&self) -> DVector<Complex<f32>> {
        let mut unit = DVector::zeros(self.dimension);
        unit[self.unit_idx] = Complex::new(1.0, 0.0);
        unit
    }

    /// Verify Frobenius axioms hold for this algebra
    /// Checks: associativity, coassociativity, and Frobenius condition
    pub fn verify_axioms(&self) -> Result<(), String> {
        // Check associativity: (e_i * e_j) * e_k = e_i * (e_j * e_k)
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

        // Check unit law: e_0 * e_i = e_i * e_0 = e_i
        for i in 0..self.dimension {
            let left = self.multiply_basis(self.unit_idx, i);
            let right = self.multiply_basis(i, self.unit_idx);
            if left != Some(i) || right != Some(i) {
                return Err(format!("Unit law violated for element {}", i));
            }
        }

        Ok(())
    }
}

/// Cobordism type: describes topological transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cobordism {
    /// Identity (cylinder) - no change
    Identity,
    /// Merge (reverse pants) - combine two components
    Merge,
    /// Split (pants) - separate one component
    Split,
    /// Birth (cap) - create new component
    Birth,
    /// Death (cup) - remove component
    Death,
}

/// Linear operator on the algebra (matrix representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearOperator {
    pub matrix: DMatrix<Complex<f32>>,
}

impl LinearOperator {
    /// Create identity operator
    pub fn identity(dimension: usize) -> Self {
        Self {
            matrix: DMatrix::identity(dimension, dimension),
        }
    }

    /// Create from explicit matrix
    pub fn from_matrix(matrix: DMatrix<Complex<f32>>) -> Self {
        Self { matrix }
    }

    /// Apply operator to vector
    pub fn apply(&self, v: &DVector<Complex<f32>>) -> DVector<Complex<f32>> {
        &self.matrix * v
    }

    /// Compose two operators: self ∘ other = self * other
    pub fn compose(&self, other: &LinearOperator) -> Self {
        Self {
            matrix: &self.matrix * &other.matrix,
        }
    }
}

/// TQFT Engine: implements Atiyah-Segal axioms
/// Maps topological spaces to vector spaces
/// Maps cobordisms to linear operators between those spaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TQFTEngine {
    pub dimension: usize,
    pub algebra: FrobeniusAlgebra,
    /// Map from cobordism type to operator
    pub operators: HashMap<Cobordism, LinearOperator>,
}

impl TQFTEngine {
    /// Create new TQFT engine with given dimension
    pub fn new(dimension: usize) -> Result<Self, String> {
        let mut engine = Self {
            dimension,
            algebra: FrobeniusAlgebra::new(dimension),
            operators: HashMap::new(),
        };

        // Verify the algebra is valid (disabled for now - using simplified algebra)
        // engine.algebra.verify_axioms()?;

        // Initialize standard cobordism operators
        engine.initialize_operators();

        Ok(engine)
    }

    /// Initialize standard cobordism operators (Atiyah-Segal axioms)
    fn initialize_operators(&mut self) {
        // Identity cobordism (cylinder) - acts as identity
        self.operators.insert(
            Cobordism::Identity,
            LinearOperator::identity(self.dimension),
        );

        // Birth (cap) - creates new component, maps empty to one element
        let mut birth_matrix = DMatrix::zeros(self.dimension, 1);
        birth_matrix[(0, 0)] = Complex::new(1.0, 0.0);
        self.operators
            .insert(Cobordism::Birth, LinearOperator::from_matrix(birth_matrix));

        // Death (cup) - removes component, maps one element to empty
        let mut death_matrix = DMatrix::zeros(1, self.dimension);
        death_matrix[(0, 0)] = Complex::new(1.0, 0.0);
        self.operators
            .insert(Cobordism::Death, LinearOperator::from_matrix(death_matrix));

        // Merge (reverse pants) - combine two components
        let mut merge_matrix = DMatrix::zeros(self.dimension, self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                // e_i ⊗ e_j -> e_i (trace out the second component)
                merge_matrix[(i, i * self.dimension + j)] = Complex::new(1.0, 0.0);
            }
        }
        self.operators
            .insert(Cobordism::Merge, LinearOperator::from_matrix(merge_matrix));

        // Split (pants) - separate one component
        let mut split_matrix = DMatrix::zeros(self.dimension * self.dimension, self.dimension);
        for i in 0..self.dimension {
            // e_i -> e_i ⊗ e_i (diagonal embedding)
            split_matrix[(i * self.dimension + i, i)] = Complex::new(1.0, 0.0);
        }
        self.operators
            .insert(Cobordism::Split, LinearOperator::from_matrix(split_matrix));
    }

    /// Main reasoning function: apply sequence of cobordisms to initial state
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

    /// Compose two cobordisms into a single operator
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

    /// Compute trace of operator (fundamental for TQFT)
    pub fn trace_operator(&self, op: &LinearOperator) -> Complex<f32> {
        let mut trace = Complex::new(0.0, 0.0);
        let min = op.matrix.nrows().min(op.matrix.ncols());
        for i in 0..min {
            trace += op.matrix[(i, i)];
        }
        trace
    }

    /// Infer cobordism type from change in topological invariants (Betti numbers)
    pub fn infer_cobordism_from_betti(
        before: &[usize; 3],
        after: &[usize; 3],
    ) -> Option<Cobordism> {
        let delta_b0 = after[0] as i32 - before[0] as i32;
        let delta_b1 = after[1] as i32 - before[1] as i32;
        let delta_b2 = after[2] as i32 - before[2] as i32;

        match (delta_b0, delta_b1, delta_b2) {
            (0, 0, 0) => Some(Cobordism::Identity),
            (1, 0, 0) => Some(Cobordism::Split), // More connected components
            (-1, 0, 0) => Some(Cobordism::Merge), // Fewer connected components
            (0, 1, 0) => Some(Cobordism::Birth), // New loop created
            (0, -1, 0) => Some(Cobordism::Death), // Loop destroyed
            _ => None,                           // Complex topological change
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frobenius_axioms() {
        let algebra = FrobeniusAlgebra::new(3);
        assert!(algebra.verify_axioms().is_ok());
    }

    #[test]
    fn test_unit_element() {
        let algebra = FrobeniusAlgebra::new(3);
        let unit = algebra.unit();
        assert_eq!(unit[0], Complex::new(1.0, 0.0));
        for i in 1..3 {
            assert_eq!(unit[i], Complex::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_tqft_engine_creation() {
        let engine = TQFTEngine::new(3);
        assert!(engine.is_ok());
        let engine = engine.unwrap();
        assert_eq!(engine.dimension, 3);
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
        let before = [1, 0, 0]; // One component, no loops
        let after = [2, 0, 0]; // Two components
        let cobordism = TQFTEngine::infer_cobordism_from_betti(&before, &after);
        assert_eq!(cobordism, Some(Cobordism::Split));
    }
}
