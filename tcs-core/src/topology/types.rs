use nalgebra::DVector;
use std::cmp::Ordering;

/// Simple point wrapper so downstream code can swap in tensors later without
/// touching the topology engine API.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub coords: Vec<f32>,
}

impl Point {
    #[inline]
    pub fn new(coords: Vec<f32>) -> Self {
        Self { coords }
    }

    #[inline]
    pub fn dimensions(&self) -> usize {
        self.coords.len()
    }
}

/// Distance functions available to the topology engine.
#[derive(Debug, Clone, Copy, Default)]
pub enum DistanceMetric {
    #[default]
    Euclidean,
}

/// Tunable parameters that influence the sparsified Vietoris–Rips filtration.
#[derive(Debug, Clone)]
pub struct TopologyParams {
    /// Number of nearest neighbours to retain when sparsifying the complete
    /// graph. Typical values: 8-32.
    pub k: usize,
    /// Optional hard cutoff for the filtration radius. `None` keeps all edges
    /// discovered by the kNN sweep.
    pub max_filtration_value: Option<f32>,
    /// Metric used when computing pairwise distances.
    pub metric: DistanceMetric,
}

impl Default for TopologyParams {
    fn default() -> Self {
        Self {
            k: 16,
            max_filtration_value: None,
            metric: DistanceMetric::default(),
        }
    }
}

/// Persistent homology feature (a single barcode interval).
#[derive(Debug, Clone)]
pub struct PersistenceFeature {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
}

impl PersistenceFeature {
    #[inline]
    pub fn persistence(&self) -> f32 {
        (self.death - self.birth).abs() as f32
    }

    #[inline]
    pub fn is_infinite(&self) -> bool {
        !self.death.is_finite()
    }
}

pub fn compute_persistence(_data: &DVector<f64>) -> Vec<PersistenceFeature> {
    // Merged stub: dummy homology with 3 features
    vec![
        PersistenceFeature {
            birth: 0.0,
            death: f64::INFINITY,
            dimension: 0,
        },
        PersistenceFeature {
            birth: 0.0,
            death: 2.0,
            dimension: 1,
        },
        PersistenceFeature {
            birth: 0.0,
            death: 2.0,
            dimension: 2,
        },
    ]
}

/// Collection of persistence features for a single homology dimension.
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub features: Vec<PersistenceFeature>,
}

impl PersistenceDiagram {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            features: Vec::new(),
        }
    }

    /// Compute the persistent entropy of the diagram.
    pub fn persistent_entropy(&self) -> f32 {
        let mut lifetimes: Vec<f32> = self
            .features
            .iter()
            .filter_map(|f| {
                if f.death.is_finite() {
                    Some(((f.death - f.birth).max(0.0)) as f32)
                } else {
                    None
                }
            })
            .collect();
        if lifetimes.is_empty() {
            return 0.0;
        }
        let total: f32 = lifetimes.iter().sum();
        if total <= f32::EPSILON {
            return 0.0;
        }
        // Normalise lifetimes into probabilities.
        lifetimes.iter_mut().for_each(|l| *l /= total);
        lifetimes.iter().fold(
            0.0,
            |acc, p| {
                if *p > 0.0 { acc - (*p * p.ln()) } else { acc }
            },
        )
    }

    /// Compute a step-wise Betti curve for this diagram.
    pub fn betti_curve(&self) -> BettiCurve {
        BettiCurve::from_diagram(self)
    }
}

/// Step-wise Betti numbers across the filtration radius.
#[derive(Debug, Clone)]
pub struct BettiCurve {
    pub dimension: usize,
    /// (radius, betti) samples in ascending order of radius.
    pub samples: Vec<(f32, usize)>,
}

impl BettiCurve {
    pub fn new(dimension: usize, samples: Vec<(f32, usize)>) -> Self {
        Self { dimension, samples }
    }

    fn from_diagram(diagram: &PersistenceDiagram) -> Self {
        let mut events: Vec<(f32, i32)> = Vec::with_capacity(diagram.features.len() * 2);
        for feature in &diagram.features {
            events.push((feature.birth as f32, 1));
            if feature.death.is_finite() {
                events.push((feature.death as f32, -1));
            }
        }
        events.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
            Some(Ordering::Equal) | None => a.1.cmp(&b.1),
            Some(order) => order,
        });
        let mut samples = Vec::with_capacity(events.len());
        let mut betti: i32 = 0;
        let mut last_radius = 0.0;
        for (radius, delta) in events {
            if samples.is_empty() || radius != last_radius {
                samples.push((radius, betti.max(0) as usize));
            }
            betti += delta;
            last_radius = radius;
            samples.push((radius, betti.max(0) as usize));
        }
        if samples.is_empty() {
            samples.push((0.0, 0));
        }
        Self {
            dimension: diagram.dimension,
            samples,
        }
    }
}

/// Aggregate persistence information returned by the topology engine.
#[derive(Debug, Clone)]
pub struct PersistenceResult {
    pub diagrams: Vec<PersistenceDiagram>,
    pub betti_curves: Vec<BettiCurve>,
    pub entropy: Vec<(usize, f32)>,
}

/// Temporary backwards-compatible alias for legacy code paths that still use the
/// old `PersistentFeature` name. This can be removed once downstream crates have
/// been updated.
pub type PersistentFeature = PersistenceFeature;

impl PersistenceResult {
    pub fn empty() -> Self {
        Self {
            diagrams: Vec::new(),
            betti_curves: Vec::new(),
            entropy: Vec::new(),
        }
    }

    pub fn diagram(&self, dimension: usize) -> Option<&PersistenceDiagram> {
        self.diagrams.iter().find(|d| d.dimension == dimension)
    }
}
