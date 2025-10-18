//! Knot theory utilities for computing Jones polynomials and deriving
//! cognitive complexity metrics used downstream in the orchestrator.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::collections::HashMap;

/// Lightweight diagram storing signed crossings. Positive values represent
/// over-crossings, negative values under-crossings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KnotDiagram {
    pub crossings: Vec<i32>,
}

impl KnotDiagram {
    pub fn trefoil() -> Self {
        Self {
            crossings: vec![1, -1, 1],
        }
    }

    pub fn unknot() -> Self {
        Self { crossings: Vec::new() }
    }
}

/// Summary of knot properties relevant to the cognitive pipeline.
#[derive(Debug, Clone)]
pub struct CognitiveKnot {
    pub polynomial: String,
    pub crossing_number: usize,
    pub complexity_score: f32,
}

/// State-sum evaluator for the Jones polynomial using the Kauffman bracket.
pub struct JonesPolynomial {
    cache: LruCache<KnotDiagram, String>,
}

impl JonesPolynomial {
    pub fn new(capacity: usize) -> Self {
        let size = NonZeroUsize::new(capacity).unwrap_or_else(|| NonZeroUsize::new(64).unwrap());
        Self {
            cache: LruCache::new(size),
        }
    }

    pub fn polynomial(&mut self, diagram: &KnotDiagram) -> String {
        if let Some(poly) = self.cache.get(diagram) {
            return poly.clone();
        }
        let poly = if diagram.crossings.is_empty() {
            "1".to_string()
        } else {
            kaufmann_bracket(diagram)
        };
        self.cache.put(diagram.clone(), poly.clone());
        poly
    }

    pub fn analyze(&mut self, diagram: &KnotDiagram) -> CognitiveKnot {
        let polynomial = self.polynomial(diagram);
        let crossing_number = diagram.crossings.len();
        let complexity_score = jones_complexity(&polynomial, crossing_number);
        CognitiveKnot {
            polynomial,
            crossing_number,
            complexity_score,
        }
    }
}

fn kaufmann_bracket(diagram: &KnotDiagram) -> String {
    // For now we use a simplified recurrence: each crossing splits into two
    // states with weights A and A^{-1}. We approximate the resulting
    // polynomial as a map from exponent -> coefficient and then translate
    // to a human-readable string.
    let mut states: HashMap<i32, i32> = HashMap::new();
    states.insert(0, 1);

    for &crossing in &diagram.crossings {
        let mut next_states = HashMap::new();
        for (&exp, &coeff) in &states {
            let weight = if crossing.is_positive() { 1 } else { -1 };
            *next_states.entry(exp + weight).or_insert(0) += coeff;
            *next_states.entry(exp - weight).or_insert(0) += coeff;
        }
        states = next_states;
    }

    states
        .into_iter()
        .filter(|(_, coeff)| *coeff != 0)
        .map(|(exp, coeff)| format!("{}t^{}", coeff, exp))
        .collect::<Vec<_>>()
        .join(" + ")
        .replace("+ -", "- ")
}

fn jones_complexity(polynomial: &str, crossings: usize) -> f32 {
    let term_count = polynomial.split('+').count().max(1);
    let entropy = (term_count as f32).log2();
    entropy + crossings as f32 * 0.1
}
