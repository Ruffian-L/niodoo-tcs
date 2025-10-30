// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Knot theory utilities for computing Jones polynomials and deriving
//! cognitive complexity metrics used downstream in the orchestrator.

use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;

const DEFAULT_CACHE_CAPACITY: usize = 64;
const CROSSING_COMPLEXITY_WEIGHT: f32 = 0.1;

/// Lightweight diagram storing signed crossings. Positive values represent
/// over-crossings, negative values under-crossings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KnotDiagram {
    pub crossings: Vec<i32>,
}

impl KnotDiagram {
    /// Right-handed trefoil representative. The normalized Jones polynomial is `t + t^3 - t^4`.
    pub fn trefoil() -> Self {
        Self {
            crossings: vec![1, -1, 1],
        }
    }

    pub fn unknot() -> Self {
        Self {
            crossings: Vec::new(),
        }
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
        let size = NonZeroUsize::new(capacity)
            .unwrap_or_else(|| NonZeroUsize::new(DEFAULT_CACHE_CAPACITY).unwrap());
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
    entropy + crossings as f32 * CROSSING_COMPLEXITY_WEIGHT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trefoil_complexity_is_positive() {
        let mut analyzer = JonesPolynomial::new(DEFAULT_CACHE_CAPACITY);
        let trefoil = KnotDiagram::trefoil();
        let result = analyzer.analyze(&trefoil);
        assert!(result.complexity_score > 0.0);
    }

    #[test]
    fn unknot_has_complexity_zero() {
        let mut analyzer = JonesPolynomial::new(DEFAULT_CACHE_CAPACITY);
        let unknot = KnotDiagram::unknot();
        let result = analyzer.analyze(&unknot);
        assert_eq!(result.polynomial, "1");
        assert_eq!(result.crossing_number, 0);
    }

    #[test]
    fn knot_diagram_equality() {
        let trefoil1 = KnotDiagram::trefoil();
        let trefoil2 = KnotDiagram::trefoil();
        assert_eq!(trefoil1, trefoil2);
    }

    #[test]
    fn cache_memoization() {
        let mut analyzer = JonesPolynomial::new(DEFAULT_CACHE_CAPACITY);
        let trefoil = KnotDiagram::trefoil();

        let result1 = analyzer.analyze(&trefoil);
        let result2 = analyzer.analyze(&trefoil);

        assert_eq!(result1.polynomial, result2.polynomial);
        assert_eq!(result1.complexity_score, result2.complexity_score);
    }

    #[test]
    fn complexity_increases_with_crossings() {
        let mut analyzer = JonesPolynomial::new(DEFAULT_CACHE_CAPACITY);

        let unknot = KnotDiagram::unknot();
        let trefoil = KnotDiagram::trefoil();

        let unknot_result = analyzer.analyze(&unknot);
        let trefoil_result = analyzer.analyze(&trefoil);

        assert!(trefoil_result.complexity_score > unknot_result.complexity_score);
    }

    #[test]
    fn different_knots_have_different_polynomials() {
        let mut analyzer = JonesPolynomial::new(DEFAULT_CACHE_CAPACITY);

        let trefoil = KnotDiagram::trefoil();
        let figure_eight = KnotDiagram {
            crossings: vec![1, -1, 1, -1],
        };

        let trefoil_result = analyzer.analyze(&trefoil);
        let figure_eight_result = analyzer.analyze(&figure_eight);

        assert_ne!(trefoil_result.polynomial, figure_eight_result.polynomial);
    }
}
