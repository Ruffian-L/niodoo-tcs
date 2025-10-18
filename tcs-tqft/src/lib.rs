//! Topological quantum field theory primitives. We model a finite
//! dimensional commutative Frobenius algebra which underpins the
//! categorical interpretation referenced in the README.

/// Frobenius algebra storing multiplication and comultiplication
/// structure constants.
#[derive(Debug, Clone)]
pub struct FrobeniusAlgebra {
    pub multiplication: Vec<Vec<f32>>,
    pub comultiplication: Vec<Vec<f32>>,
    pub unit: Vec<f32>,
    pub counit: Vec<f32>,
}

impl FrobeniusAlgebra {
    pub fn new(dim: usize) -> Self {
        let mut multiplication = vec![vec![0.0; dim]; dim];
        let mut comultiplication = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            multiplication[i][i] = 1.0;
            comultiplication[i][i] = 1.0;
        }
        let unit = vec![1.0; dim];
        let counit = vec![1.0; dim];
        Self {
            multiplication,
            comultiplication,
            unit,
            counit,
        }
    }

    pub fn is_associative(&self) -> bool {
        let dim = self.multiplication.len();
        for a in 0..dim {
            for b in 0..dim {
                for c in 0..dim {
                    let lhs = self.mul(self.mul_basis(a, b), c);
                    let rhs = self.mul(a, self.mul_basis(b, c));
                    if (lhs - rhs).abs() > 1e-5 {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn is_coassociative(&self) -> bool {
        let dim = self.comultiplication.len();
        for a in 0..dim {
            for b in 0..dim {
                let lhs = self.comul(self.comul_basis(a), b);
                let rhs = self.comul(a, self.comul_basis(b));
                if (lhs - rhs).abs() > 1e-5 {
                    return false;
                }
            }
        }
        true
    }

    pub fn satisfies_frobenius(&self) -> bool {
        let dim = self.multiplication.len();
        for a in 0..dim {
            for b in 0..dim {
                let lhs = self.comul(self.mul_basis(a, b), a);
                let rhs = self.mul(self.comul_basis(a), b);
                if (lhs - rhs).abs() > 1e-5 {
                    return false;
                }
            }
        }
        true
    }

    fn mul_basis(&self, a: usize, b: usize) -> f32 {
        self.multiplication[a][b]
    }

    fn mul(&self, coeff: f32, idx: usize) -> f32 {
        coeff * self.unit[idx]
    }

    fn comul_basis(&self, idx: usize) -> f32 {
        self.comultiplication[idx][idx]
    }

    fn comul(&self, coeff: f32, idx: usize) -> f32 {
        coeff * self.counit[idx]
    }
}
