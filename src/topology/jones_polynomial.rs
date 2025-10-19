//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use num_complex::Complex;
use petgraph::graph::{Graph, NodeIndex};
use std::collections::HashMap;

/// Simple polynomial type for Jones polynomial computation
#[derive(Debug, Clone)]
struct Polynomial<T> {
    terms: HashMap<i32, T>,
}

impl<T: Clone> Polynomial<T> {
    fn new() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    fn add_monomial(&mut self, coeff: T, power: i32) {
        self.terms.insert(power, coeff);
    }

    fn coefficients(&self) -> impl Iterator<Item = (&i32, &T)> {
        self.terms.iter()
    }
}

/// Knot diagram representation
#[derive(Debug, Clone)]
pub struct KnotDiagram {
    pub crossings: Vec<Crossing>,
    pub gauss_code: Vec<i32>,
    pub pd_code: Vec<[usize; 4]>, // Planar diagram code
}

#[derive(Debug, Clone)]
pub struct Crossing {
    id: usize,
    over_strand: usize,
    under_strand: usize,
    sign: i8, // +1 for positive, -1 for negative
}

impl Crossing {
    pub fn new(id: usize, over: usize, under: usize, sign: i8) -> Self {
        Self {
            id,
            over_strand: over,
            under_strand: under,
            sign,
        }
    }
}

/// Jones polynomial computation
#[derive(Debug, Clone)]
pub struct JonesPolynomial {
    coefficients: HashMap<i32, Complex<f32>>,
    max_degree: i32,
    min_degree: i32,
}

impl JonesPolynomial {
    /// Compute Jones polynomial using Kauffman bracket
    pub fn compute(knot: &KnotDiagram) -> Self {
        // Use state sum model for Kauffman bracket
        let bracket = Self::kauffman_bracket(knot);

        // Normalize to get Jones polynomial
        let writhe = knot.writhe();
        let jones = Self::normalize_bracket(bracket, writhe);

        jones
    }

    /// Kauffman bracket via state summation
    fn kauffman_bracket(knot: &KnotDiagram) -> Polynomial<Complex<f32>> {
        let n = knot.crossings.len();
        let num_states = 1 << n; // 2^n states

        let mut bracket = Polynomial::new();

        // Sum over all states
        for state in 0..num_states {
            let (circles, a_power, a_inv_power) = Self::evaluate_state(knot, state);

            let contribution = Complex::new(
                (-1.0_f32).powi(circles as i32) * 2.0_f32.powi((circles - 1) as i32),
                0.0,
            );

            let power = a_power as i32 - a_inv_power as i32;
            bracket.add_monomial(contribution, power);
        }

        bracket
    }

    /// Evaluate a single state (Kauffman state)
    fn evaluate_state(knot: &KnotDiagram, state: usize) -> (usize, usize, usize) {
        let mut graph: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
        let mut a_power = 0;
        let mut a_inv_power = 0;

        // Build graph based on state
        for (i, crossing) in knot.crossings.iter().enumerate() {
            if (state >> i) & 1 == 0 {
                // A-smoothing
                a_power += 1;
                // Connect NE-NW and SE-SW
                graph.add_edge(
                    NodeIndex::new(crossing.id * 4),
                    NodeIndex::new(crossing.id * 4 + 1),
                    (),
                );
            } else {
                // A^{-1}-smoothing
                a_inv_power += 1;
                // Connect NE-SE and NW-SW
                graph.add_edge(
                    NodeIndex::new(crossing.id * 4),
                    NodeIndex::new(crossing.id * 4 + 2),
                    (),
                );
            }
        }

        // Count connected components (circles)
        let circles = petgraph::algo::connected_components(&graph);

        (circles, a_power, a_inv_power)
    }

    /// Normalize bracket to Jones polynomial
    fn normalize_bracket(bracket: Polynomial<Complex<f32>>, writhe: i32) -> Self {
        // V(t) = (-A^3)^{-writhe} * <K>
        // where A = t^{-1/4}
        let mut coefficients = HashMap::new();
        let mut max_deg = i32::MIN;
        let mut min_deg = i32::MAX;

        for (power, coeff) in bracket.coefficients() {
            let normalized_power = power - 3 * writhe;
            coefficients.insert(normalized_power, *coeff);
            max_deg = max_deg.max(normalized_power);
            min_deg = min_deg.min(normalized_power);
        }

        Self {
            coefficients,
            max_degree: max_deg,
            min_degree: min_deg,
        }
    }

    /// Compute special cases for common knots
    pub fn compute_special_case(knot_type: &KnotType) -> Self {
        match knot_type {
            KnotType::Unknot => {
                let mut jones = JonesPolynomial::new();
                jones.coefficients.insert(0, Complex::new(1.0, 0.0));
                jones.min_degree = 0;
                jones.max_degree = 0;
                jones
            }
            KnotType::Trefoil => {
                // V = t + t³ - t⁴
                let mut jones = JonesPolynomial::new();
                jones.coefficients.insert(1, Complex::new(1.0, 0.0));
                jones.coefficients.insert(3, Complex::new(1.0, 0.0));
                jones.coefficients.insert(4, Complex::new(-1.0, 0.0));
                jones.min_degree = 1;
                jones.max_degree = 4;
                jones
            }
            KnotType::FigureEight => {
                // V = t^{-2} - t^{-1} + 1 - t + t²
                let mut jones = JonesPolynomial::new();
                jones.coefficients.insert(-2, Complex::new(1.0, 0.0));
                jones.coefficients.insert(-1, Complex::new(-1.0, 0.0));
                jones.coefficients.insert(0, Complex::new(1.0, 0.0));
                jones.coefficients.insert(1, Complex::new(-1.0, 0.0));
                jones.coefficients.insert(2, Complex::new(1.0, 0.0));
                jones.min_degree = -2;
                jones.max_degree = 2;
                jones
            }
            _ => Self::compute(&knot_type.to_diagram()),
        }
    }

    pub fn new() -> Self {
        Self {
            coefficients: HashMap::new(),
            max_degree: 0,
            min_degree: 0,
        }
    }
}

impl KnotDiagram {
    pub fn writhe(&self) -> i32 {
        self.crossings.iter().map(|c| c.sign as i32).sum()
    }

    pub fn crossing_number(&self) -> usize {
        self.crossings.len()
    }
}

/// Knot type enumeration for special cases
#[derive(Debug, Clone, PartialEq)]
pub enum KnotType {
    Unknot,
    Trefoil,
    FigureEight,
    Custom,
}

impl KnotType {
    pub fn to_diagram(&self) -> KnotDiagram {
        match self {
            KnotType::Unknot => KnotDiagram {
                crossings: vec![],
                gauss_code: vec![],
                pd_code: vec![],
            },
            KnotType::Trefoil => {
                // Simplified trefoil representation
                KnotDiagram {
                    crossings: vec![
                        Crossing::new(0, 0, 1, 1),
                        Crossing::new(1, 1, 2, 1),
                        Crossing::new(2, 2, 0, 1),
                    ],
                    gauss_code: vec![1, -2, 3, -1, 2, -3],
                    pd_code: vec![[1, 4, 2, 5], [3, 6, 4, 1], [5, 2, 6, 3]],
                }
            }
            KnotType::FigureEight => {
                // Simplified figure-eight representation
                KnotDiagram {
                    crossings: vec![
                        Crossing::new(0, 0, 1, 1),
                        Crossing::new(1, 1, 2, -1),
                        Crossing::new(2, 2, 3, 1),
                        Crossing::new(3, 3, 0, -1),
                    ],
                    gauss_code: vec![1, -2, 3, -4, -1, 2, -3, 4],
                    pd_code: vec![
                        [1, 8, 2, 3],
                        [3, 1, 4, 14],
                        [4, 15, 5, 6],
                        [6, 5, 7, 8],
                        [9, 4, 10, 11],
                        [11, 10, 12, 13],
                        [13, 12, 14, 15],
                        [7, 16, 9, 2],
                    ],
                }
            }
            KnotType::Custom => panic!("Custom knot needs diagram"),
        }
    }
}
