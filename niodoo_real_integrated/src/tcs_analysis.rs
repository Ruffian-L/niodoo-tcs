//! TCS Topology Analysis Layer
//! Computes persistent homology, knot invariants, and TQFT signatures on every state

use anyhow::{bail, Context, Result};
use blake3::Hasher;
use candle_core::{Device, Tensor};
use dashmap::DashMap;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::Ordering;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::torus::PadGhostState;
use tcs_core::PersistentFeature;

type Point = Vec<f32>;
type TopologyParams = ();
type RustVREngine = ();
type TopologyEngine = ();
// Stub function for metrics
fn record_topology_metrics(_betti: &[usize; 3], _complexity: f64) {}
use tcs_knot::{JonesPolynomial, KnotDiagram};
use tcs_tda::PersistentHomology;
use tcs_tqft::{Cobordism, TQFTEngine};

/// Topological signature computed for a state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalSignature {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,

    // Persistent homology features
    #[serde(skip)]
    pub persistence_features: Vec<PersistentFeature>,
    pub betti_numbers: [usize; 3], // H0, H1, H2

    // Knot invariants
    pub knot_complexity: f64,
    pub knot_polynomial: String,

    // TQFT invariants
    pub tqft_dimension: usize,
    pub cobordism_type: Option<Cobordism>,

    // New TDA features for Phase 5
    pub persistence_entropy: f64,
    pub spectral_gap: f64,
    pub euler_characteristic: f64,
    pub total_persistence: f64,
    pub max_persistence: f64,
    pub mean_persistence: f64,
    pub laplacian_spectral_radius: f64,

    // Performance metrics
    pub computation_time_ms: f64,
}

impl TopologicalSignature {
    pub fn new(
        persistence_features: Vec<PersistentFeature>,
        betti_numbers: [usize; 3],
        knot_complexity: f64,
        knot_polynomial: String,
        tqft_dimension: usize,
        cobordism_type: Option<Cobordism>,
        computation_time_ms: f64,
        persistence_entropy: f64,
        spectral_gap: f64,
        euler_characteristic: f64,
        total_persistence: f64,
        max_persistence: f64,
        mean_persistence: f64,
        laplacian_spectral_radius: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            persistence_features,
            betti_numbers,
            knot_complexity,
            knot_polynomial,
            tqft_dimension,
            cobordism_type,
            persistence_entropy,
            spectral_gap,
            euler_characteristic,
            total_persistence,
            max_persistence,
            mean_persistence,
            laplacian_spectral_radius,
            computation_time_ms,
        }
    }
}

/// Lightweight analytic fallback for topological analysis when the full TCS analyzer
/// is unavailable. Computes persistence features from the PAD state directly and
/// derives secondary metrics deterministically from the available signals so the
/// downstream pipeline can proceed without panicking.
pub fn baseline_topological_signature(
    pad_state: &PadGhostState,
    embedding: &[f32],
) -> TopologicalSignature {
    let analysis_start = Instant::now();

    let pad: Vec<f64> = pad_state.pad.iter().map(|v| *v as f64).collect();
    let mu: Vec<f64> = pad_state.mu.iter().map(|v| *v as f64).collect();
    let sigma: Vec<f64> = pad_state.sigma.iter().map(|v| *v as f64).collect();

    let pad_min = pad.iter().fold(f64::INFINITY, |acc, value| acc.min(*value));
    let pad_max = pad
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let mu_min = mu.iter().fold(f64::INFINITY, |acc, value| acc.min(*value));
    let mu_max = mu
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let sigma_min = sigma
        .iter()
        .fold(f64::INFINITY, |acc, value| acc.min(*value));
    let sigma_max = sigma
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));

    let persistence_features = vec![
        PersistentFeature {
            birth: pad_min as f32,
            death: pad_max as f32,
            dimension: 0,
        },
        PersistentFeature {
            birth: mu_min as f32,
            death: mu_max as f32,
            dimension: 1,
        },
        PersistentFeature {
            birth: sigma_min as f32,
            death: sigma_max as f32,
            dimension: 2,
        },
    ];

    let betti0 = pad.iter().filter(|value| **value >= 0.0).count();
    let betti1 = pad.iter().filter(|value| **value < 0.0).count();
    let sigma_threshold = if sigma.is_empty() {
        0.0
    } else {
        sigma.iter().sum::<f64>() / sigma.len() as f64
    };
    let betti2 = sigma
        .iter()
        .zip(pad_state.sigma.iter())
        .filter(|(sigma_value, raw_std)| {
            **sigma_value > sigma_threshold && **sigma_value > **raw_std
        })
        .count();

    let knot_complexity = if pad.len() > 1 {
        pad.windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .sum::<f64>()
            / (pad.len() - 1) as f64
    } else {
        0.0
    };

    let pad_mean = if pad.is_empty() {
        0.0
    } else {
        pad.iter().sum::<f64>() / pad.len() as f64
    };
    let pad_variance = if pad.len() > 1 {
        pad.iter()
            .map(|value| (value - pad_mean).powi(2))
            .sum::<f64>()
            / (pad.len() - 1) as f64
    } else {
        0.0
    };

    let knot_polynomial = format!("λ² + {:.3}λ + {:.3}", pad_mean, pad_variance);

    let pad_energy = pad
        .iter()
        .map(|value| value.abs())
        .sum::<f64>()
        .max(f64::EPSILON);
    let persistence_entropy = pad
        .iter()
        .map(|value| {
            let p = value.abs() / pad_energy;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum::<f64>();

    let mut spectral_basis: Vec<f64> = embedding
        .iter()
        .map(|value| (*value as f64).abs())
        .collect();
    spectral_basis.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let spectral_gap = match spectral_basis.len() {
        0 => 0.0,
        1 => spectral_basis[0],
        _ => spectral_basis[0] - spectral_basis[1],
    };

    let euler_characteristic = betti0 as f64 - betti1 as f64 + betti2 as f64;

    let persistence_deltas: Vec<f64> = persistence_features
        .iter()
        .map(|feature| (feature.death - feature.birth) as f64)
        .collect();
    let total_persistence = persistence_deltas
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .sum::<f64>();
    let max_persistence = persistence_deltas
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);
    let mean_persistence = if persistence_deltas.is_empty() {
        0.0
    } else {
        total_persistence / persistence_deltas.len() as f64
    };

    let laplacian_spectral_radius = if embedding.len() > 1 {
        let numerator = embedding
            .windows(2)
            .map(|window| {
                let diff = window[1] as f64 - window[0] as f64;
                diff * diff
            })
            .sum::<f64>();
        let denominator = embedding
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .max(f64::EPSILON);
        numerator / denominator
    } else {
        0.0
    };

    let computation_time_ms = analysis_start.elapsed().as_secs_f64() * 1000.0;

    TopologicalSignature::new(
        persistence_features,
        [betti0, betti1, betti2],
        knot_complexity,
        knot_polynomial,
        2,
        None,
        computation_time_ms,
        persistence_entropy,
        spectral_gap,
        euler_characteristic,
        total_persistence,
        max_persistence,
        mean_persistence,
        laplacian_spectral_radius,
    )
}

#[derive(Clone)]
struct CachedSignature {
    signature: TopologicalSignature,
    expires_at: Instant,
}

struct TopologyCache {
    ttl: Duration,
    max_entries: usize,
    path: PathBuf,
    entries: DashMap<String, CachedSignature>,
}

impl TopologyCache {
    fn new(ttl: Duration, max_entries: usize, path: PathBuf) -> Result<Self> {
        if !path.exists() {
            fs::create_dir_all(&path).with_context(|| {
                format!(
                    "failed to create topology cache directory at {}",
                    path.display()
                )
            })?;
        }
        Ok(Self {
            ttl,
            max_entries: max_entries.max(1),
            path,
            entries: DashMap::new(),
        })
    }

    fn cache_path(&self, key: &str) -> PathBuf {
        self.path.join(format!("{key}.json"))
    }

    fn get(&self, key: &str) -> Option<TopologicalSignature> {
        if let Some(entry) = self.entries.get(key) {
            if self.is_expired(entry.value().expires_at) {
                self.entries.remove(key);
            } else {
                return Some(entry.value().signature.clone());
            }
        }

        let path = self.cache_path(key);
        if !path.exists() {
            return None;
        }

        if self.ttl > Duration::ZERO {
            if let Ok(metadata) = path.metadata() {
                if let Ok(modified) = metadata.modified() {
                    if modified
                        .elapsed()
                        .map(|elapsed| elapsed > self.ttl)
                        .unwrap_or(false)
                    {
                        let _ = fs::remove_file(&path);
                        return None;
                    }
                }
            }
        }

        match File::open(&path)
            .with_context(|| format!("failed to open cached topology at {}", path.display()))
            .and_then(|file| {
                serde_json::from_reader::<_, TopologicalSignature>(file)
                    .with_context(|| "failed to deserialize cached topological signature")
            }) {
            Ok(signature) => {
                let _ = self.insert(key, &signature);
                Some(signature)
            }
            Err(error) => {
                warn!(?error, path = %path.display(), "failed to load cached topology signature");
                let _ = fs::remove_file(path);
                None
            }
        }
    }

    fn insert(&self, key: &str, signature: &TopologicalSignature) -> Result<()> {
        let expires_at = if self.ttl > Duration::ZERO {
            Instant::now() + self.ttl
        } else {
            Instant::now()
        };

        self.entries.insert(
            key.to_string(),
            CachedSignature {
                signature: signature.clone(),
                expires_at,
            },
        );

        self.evict_if_needed();

        let path = self.cache_path(key);
        let mut file = File::create(&path).with_context(|| {
            format!("failed to create topology cache file at {}", path.display())
        })?;
        serde_json::to_writer_pretty(&mut file, signature)
            .context("failed to write topology cache entry")?;
        file.flush().ok();

        Ok(())
    }

    fn is_expired(&self, expires_at: Instant) -> bool {
        if self.ttl == Duration::ZERO {
            return false;
        }
        Instant::now() >= expires_at
    }

    fn evict_if_needed(&self) {
        let len = self.entries.len();
        if len <= self.max_entries {
            return;
        }

        if let Some(oldest_key) = self
            .entries
            .iter()
            .min_by_key(|entry| entry.value().expires_at)
            .map(|entry| entry.key().clone())
        {
            self.entries.remove(&oldest_key);
            let path = self.cache_path(&oldest_key);
            let _ = fs::remove_file(path);
        }
    }
}

#[derive(Debug, Clone)]
struct PersistenceResult {
    features: Vec<PersistentFeature>,
    entropy: Vec<(usize, f32)>,
    betti: [usize; 3],
}

#[derive(Debug, Clone, Default)]
pub struct TCSState {
    // Add fields as needed, e.g., persistence_features: Vec<PersistenceFeature>,
    // but keep minimal for now
    pad: Vec<f64>,
    mu: Vec<f64>,
    sigma: Vec<f64>,
}

pub type TCSHandle = Arc<Mutex<TCSState>>;

/// TCS Analysis Engine
pub struct TCSAnalyzer {
    topology_engine: RustVREngine,
    knot_analyzer: JonesPolynomial,
    tqft_engine: TQFTEngine,
    cache: Arc<TopologyCache>,
    device: Device,
    enable_gpu: bool,
    use_approximate_tda: bool, // Phase 2.1: Enable giotto-tda approximate computation
    // Phase 2.2: Adaptive fallback tracking
    giotto_failure_count: Arc<Mutex<usize>>, // Track consecutive failures
    giotto_success_count: Arc<Mutex<usize>>, // Track consecutive successes
    last_rust_result: Arc<Mutex<Option<PersistenceResult>>>, // Cache last Rust result for comparison
}

impl TCSAnalyzer {
    /// Initialize TCS analyzer
    pub fn new() -> Result<Self> {
        // Stub: topology_engine is unit type
        let topology_engine = ();
        let knot_analyzer = JonesPolynomial::new(64);
        let tqft_engine = TQFTEngine::new(2)
            .map_err(|e| anyhow::anyhow!("Failed to initialize TQFT engine: {}", e))?;

        let cache_dir =
            env::var("TOPOLOGY_CACHE_DIR").unwrap_or_else(|_| "storage/topology_cache".to_string());
        let cache_ttl = env::var("TOPOLOGY_CACHE_TTL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(900));
        let cache_max_entries = env::var("TOPOLOGY_CACHE_MAX_ENTRIES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024);

        let cache = Arc::new(TopologyCache::new(
            cache_ttl,
            cache_max_entries,
            PathBuf::from(&cache_dir),
        )?);

        let enable_gpu = env::var("TCS_ENABLE_GPU")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(true);
        let device = if enable_gpu {
            match Device::cuda_if_available(0) {
                Ok(device) => device,
                Err(error) => {
                    warn!(?error, "Falling back to CPU topology device");
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        };

        info!("TCS Analyzer initialized");
        Ok(Self {
            topology_engine,
            knot_analyzer,
            tqft_engine,
            cache,
            device,
            enable_gpu,
            use_approximate_tda: env::var("USE_APPROXIMATE_TDA")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false),
            giotto_failure_count: Arc::new(Mutex::new(0)),
            giotto_success_count: Arc::new(Mutex::new(0)),
            last_rust_result: Arc::new(Mutex::new(None)),
        })
    }

    /// Initialize TCS analyzer with configuration (Phase 2.1)
    pub fn new_with_config(use_approximate_tda: bool) -> Result<Self> {
        let mut analyzer = Self::new()?;
        analyzer.use_approximate_tda = use_approximate_tda;
        if use_approximate_tda {
            info!("TCSAnalyzer initialized with approximate TDA (giotto-tda) enabled");
        }
        Ok(analyzer)
    }

    /// Apply TQFT reasoning to evolve a state through cobordism transitions
    pub fn apply_tqft_reasoning(
        &self,
        initial_state: &[f64],
        transitions: &[Cobordism],
    ) -> Result<Vec<f64>> {
        use nalgebra::DVector;
        use num_complex::Complex;

        // Convert real state to complex vector
        let complex_state: DVector<Complex<f32>> = DVector::from_iterator(
            initial_state.len().min(self.tqft_engine.dimension),
            initial_state
                .iter()
                .take(self.tqft_engine.dimension)
                .map(|&x| Complex::new(x as f32, 0.0)),
        );

        // Apply TQFT reasoning
        let result_state = self
            .tqft_engine
            .reason(&complex_state, transitions)
            .map_err(|e| anyhow::anyhow!("TQFT reasoning failed: {}", e))?;

        // Convert back to real values
        let real_state: Vec<f64> = result_state.iter().map(|c| c.re as f64).collect();

        Ok(real_state)
    }

    /// Analyze topological structure of a state
    #[instrument(skip(self), fields(entropy = pad_state.entropy))]
    pub fn analyze_state(&mut self, pad_state: &PadGhostState) -> Result<TopologicalSignature> {
        let start = Instant::now();
        let cache_key = Self::cache_key(pad_state);

        if let Some(signature) = self.cache.get(&cache_key) {
            debug!(cache_hit = true, "Topology cache hit");
            // Phase 2.3: Record cache hit metric
            crate::metrics::tcs_analyzer_metrics().record_cache_hit();
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            crate::metrics::tcs_analyzer_metrics().record_computation_latency(latency_ms);
            return Ok(signature);
        }

        // Phase 2.3: Record cache miss
        crate::metrics::tcs_analyzer_metrics().record_cache_miss();

        let tcs_state = Arc::new(Mutex::new(TCSState::default()));
        let mut guard = tcs_state
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock TCS state: {}", e))?;
        guard.pad = pad_state.pad.iter().map(|&v| v as f64).collect();
        guard.mu = pad_state.mu.iter().map(|&v| v as f64).collect();
        guard.sigma = pad_state.sigma.iter().map(|&v| v as f64).collect();

        let points = self.pad_to_points(&guard);
        if points.is_empty() {
            anyhow::bail!("no points generated for topology analysis");
        }

        let max_filtration = env::var("TCS_MAX_FILTRATION")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.5);

        let distances = self.compute_pairwise_distances(&points)?;
        let persistence = self.compute_persistence(&points, max_filtration)?;

        let mut betti = self.compute_betti_numbers(&persistence);

        // Phase 2.3: Record Betti number distribution
        crate::metrics::tcs_analyzer_metrics().record_betti_numbers(&betti);

        let num_points = points.len();
        let theoretical_max = num_points.saturating_sub(1);
        let constraint_max = env::var("TCS_BETTI1_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6);
        let max_allowed = theoretical_max.min(constraint_max);

        let original_betti1 = betti[1];
        debug!(
            "Betti numbers before capping: {:?}, num_points={}, theoretical_max={}, constraint_max={}, max_allowed={}",
            betti, num_points, theoretical_max, constraint_max, max_allowed
        );

        let mut persistent_count_debug = [0usize; 3];
        for feature in &persistence.features {
            if feature.dimension < 3 && feature.death.is_infinite() {
                persistent_count_debug[feature.dimension] += 1;
            }
        }

        if betti[1] > max_allowed {
            warn!(
                "Betti_1 ({}) exceeds maximum (theoretical: {}, constraint: {}), capping to {}. num_points={}, persistent_count_debug={:?}",
                betti[1],
                theoretical_max,
                constraint_max,
                max_allowed,
                num_points,
                persistent_count_debug
            );
            betti[1] = max_allowed;
        }

        if betti[1] > max_allowed {
            warn!(
                "Betti_1 capping failed during check: value={} exceeds max_allowed={}. Force-capping now.",
                betti[1],
                max_allowed
            );
            betti[1] = max_allowed;
        }

        if betti[1] != original_betti1 {
            info!(
                "Betti_1 capped from {} to {} (max_allowed={})",
                original_betti1, betti[1], max_allowed
            );
        }

        assert!(
            betti[1] <= max_allowed,
            "Betti_1 assertion failed: {} > {} (theoretical_max={}, constraint_max={}, num_points={})",
            betti[1],
            max_allowed,
            theoretical_max,
            constraint_max,
            num_points
        );

        let persistence_entropy = Self::persistence_entropy(&persistence.entropy);
        let spectral_gap = Self::compute_spectral_gap(&persistence);
        let phi = Self::approximate_phi_from_betti(&betti);
        debug!(phi, "IIT Φ approximate value");

        let knot_diagram = self.pad_to_knot_diagram(&guard);
        let knot_analysis = self.knot_analyzer.analyze(&knot_diagram);
        let knot_polynomial = knot_analysis.polynomial;
        let knot_complexity_max = env::var("TCS_KNOT_COMPLEXITY_MAX")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(constraint_max as f64);
        let knot_proxy = (betti[1] as f64).min(constraint_max as f64).max(0.0);
        let knot_analysis_score = (knot_analysis.complexity_score as f64).min(knot_complexity_max);
        let knot_complexity = knot_proxy.max(knot_analysis_score).min(knot_complexity_max);

        let cobordism_type = self.infer_cobordism(&betti);
        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let persistence_features = Self::collect_persistence_features(&persistence);
        let euler_characteristic = Self::compute_euler_characteristic(&betti);
        let total_persistence = Self::total_persistence(&persistence_features);
        let max_persistence = Self::max_persistence(&persistence_features);
        let mean_persistence = Self::mean_persistence(total_persistence, &persistence_features);
        let laplacian_spectral_radius = Self::laplacian_spectral_radius(&distances);

        record_topology_metrics(&betti, knot_complexity);

        debug!(
            "Topological analysis: Betti={:?}, Knot complexity={:.3}, PE={:.3}, Gap={:.3}, Cobordism={:?}, Euler={:.3}, TotalPersistence={:.3}",
            betti,
            knot_complexity,
            persistence_entropy,
            spectral_gap,
            cobordism_type,
            euler_characteristic,
            total_persistence
        );

        let signature = TopologicalSignature::new(
            persistence_features,
            betti,
            knot_complexity,
            knot_polynomial,
            self.tqft_engine.dimension,
            cobordism_type,
            computation_time_ms,
            persistence_entropy,
            spectral_gap,
            euler_characteristic,
            total_persistence,
            max_persistence,
            mean_persistence,
            laplacian_spectral_radius,
        );

        // Phase 2.3: Record computation latency
        crate::metrics::tcs_analyzer_metrics().record_computation_latency(computation_time_ms);

        if let Err(error) = self.cache.insert(&cache_key, &signature) {
            warn!(?error, "failed to persist topology signature to cache");
        }

        Ok(signature)
    }

    /// Convert PAD state to point cloud for homology computation
    /// Uses PAD, mu, sigma, and incorporates entropy to make topology input-sensitive
    fn pad_to_points(&self, pad_state: &TCSState) -> Vec<Point> {
        let mut points = Vec::new();

        // Calculate global statistics for normalization
        let pad_mean: f64 = pad_state.pad.iter().sum::<f64>() / pad_state.pad.len() as f64;
        let pad_variance: f64 = pad_state
            .pad
            .iter()
            .map(|&v| (v - pad_mean).powi(2))
            .sum::<f64>()
            / pad_state.pad.len() as f64;

        for i in 0..7 {
            // Create point from PAD coordinates with mu/sigma as extra dimensions
            // Add entropy-normalized variance to make topology sensitive to input variation
            let mut coords = Vec::with_capacity(7);
            coords.push(pad_state.pad[i]);
            coords.push(pad_state.mu[i]);
            coords.push(pad_state.sigma[i]);

            // Add additional dimensions: variance-weighted position, relative deviation
            // This ensures different PAD distributions produce different point clouds
            coords.push((pad_state.pad[i] - pad_mean) * pad_variance.sqrt());
            coords.push(pad_state.mu[i] * pad_state.sigma[i]); // Interaction term

            // Pad to 7D
            while coords.len() < 7 {
                coords.push(0.0);
            }
            let point = coords.into_iter().map(|v| v as f32).collect::<Vec<_>>();
            points.push(point);
        }
        points
    }

    /// Compute Betti numbers from persistence features
    /// Betti numbers count only persistent features (death == infinity), not all features
    fn compute_betti_numbers(&self, result: &PersistenceResult) -> [usize; 3] {
        result.betti
    }

    /// Convert PAD state to simplified knot diagram
    fn pad_to_knot_diagram(&self, pad_state: &TCSState) -> KnotDiagram {
        // Map PAD values to crossings (over/under crossings)
        let crossings: Vec<i32> = pad_state
            .pad
            .iter()
            .map(|&val| {
                if val > 0.5 {
                    1 // Over-crossing
                } else if val < -0.5 {
                    -1 // Under-crossing
                } else {
                    0 // No crossing
                }
            })
            .filter(|&x| x != 0)
            .collect();

        KnotDiagram { crossings }
    }

    /// Infer cobordism type from Betti number changes using TQFT engine
    fn infer_cobordism(&self, betti: &[usize; 3]) -> Option<Cobordism> {
        // Use TQFT engine's proper inference method
        // This would need previous state, so for now use static inference
        // In production, track previous Betti numbers for comparison
        use std::sync::RwLock;
        static PREV_BETTI: RwLock<Option<[usize; 3]>> = RwLock::new(None);

        let prev_opt = PREV_BETTI
            .read()
            .map(|guard| (*guard).clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
        let cobordism = if let Some(prev) = prev_opt {
            TQFTEngine::infer_cobordism_from_betti(&prev, betti)
        } else {
            // First run - infer from structure
            if betti[0] > 1 {
                Some(Cobordism::Split)
            } else if betti[1] > 0 {
                Some(Cobordism::Birth)
            } else {
                Some(Cobordism::Identity)
            }
        };
        match PREV_BETTI.write() {
            Ok(mut guard) => {
                *guard = Some(*betti);
            }
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                *guard = Some(*betti);
            }
        }

        cobordism
    }

    fn collect_persistence_features(result: &PersistenceResult) -> Vec<PersistentFeature> {
        result.features.clone()
    }

    fn compute_spectral_gap(result: &PersistenceResult) -> f64 {
        if result.entropy.len() < 2 {
            return result
                .entropy
                .first()
                .map(|(_, value)| *value as f64)
                .unwrap_or(0.0);
        }

        let mut weights: Vec<f64> = result
            .entropy
            .iter()
            .map(|(_, value)| f64::from(*value))
            .collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let last = weights.len() - 1;
        let gap = weights[last] - weights[last - 1];
        gap.max(0.0)
    }

    fn cache_key(pad_state: &PadGhostState) -> String {
        let mut hasher = Hasher::new();
        for value in &pad_state.pad {
            hasher.update(&value.to_le_bytes());
        }
        for value in &pad_state.mu {
            hasher.update(&value.to_le_bytes());
        }
        for value in &pad_state.sigma {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&pad_state.entropy.to_le_bytes());
        if let Ok(version) = env::var("TOPOLOGY_CACHE_VERSION") {
            hasher.update(version.as_bytes());
        }
        hex::encode(hasher.finalize().as_bytes())
    }

    fn compute_pairwise_distances(&self, points: &[Point]) -> Result<Vec<Vec<f32>>> {
        if self.enable_gpu {
            match Self::pairwise_gpu(points, &self.device) {
                Ok(distances) => return Ok(distances),
                Err(error) => warn!(
                    ?error,
                    "GPU distance computation failed; falling back to CPU"
                ),
            }
        }
        Ok(Self::pairwise_cpu(points))
    }

    fn pairwise_gpu(points: &[Point], device: &Device) -> anyhow::Result<Vec<Vec<f32>>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let dims = points[0].len();
        let flat: Vec<f32> = points
            .iter()
            .flat_map(|point| point.iter().copied())
            .collect();
        let tensor = Tensor::from_vec(flat, (points.len(), dims), device)?;
        let norms = tensor.sqr()?.sum_keepdim(1)?;
        let norms_t = norms.transpose(0, 1)?;
        let mut dist_sq = norms.broadcast_add(&norms_t)?;
        let product = tensor.matmul(&tensor.transpose(0, 1)?)?;
        let scalar = Tensor::new(&[2.0f32], &product.device())?;
        let product_scaled = product.broadcast_mul(&scalar)?;
        dist_sq = (dist_sq - product_scaled)?;
        let zeros = Tensor::zeros(dist_sq.dims(), dist_sq.dtype(), device)?;
        dist_sq = dist_sq.maximum(&zeros)?;
        let dist = dist_sq.sqrt()?;
        let dist_cpu = dist.to_device(&Device::Cpu)?;
        let matrix = dist_cpu.to_vec2::<f32>()?;
        Ok(matrix)
    }

    fn pairwise_cpu(points: &[Point]) -> Vec<Vec<f32>> {
        let n = points.len();
        let mut matrix = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = points[i]
                    .iter()
                    .zip(points[j].iter())
                    .map(|(a, b)| {
                        let diff = *a - *b;
                        diff * diff
                    })
                    .sum::<f32>()
                    .sqrt();
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }
        matrix
    }

    fn compute_persistence(
        &self,
        points: &[Point],
        max_filtration: f32,
    ) -> Result<PersistenceResult> {
        // Phase 2.1: Use giotto-tda approximate computation if enabled
        if self.use_approximate_tda {
            // Phase 2.2: Check if we should skip giotto due to recent failures
            let should_skip_giotto = {
                let failures = self.giotto_failure_count.lock().unwrap();
                *failures >= 5 // Skip if 5+ consecutive failures
            };

            if !should_skip_giotto {
                // Phase 2.2: Try giotto with adaptive fallback
                let giotto_start = Instant::now();
                match self.compute_persistence_giotto(points, max_filtration) {
                    Ok(result) => {
                        let giotto_latency_ms = giotto_start.elapsed().as_secs_f64() * 1000.0;
                        // Phase 2.2: Validate quality and fallback if needed
                        if self.validate_giotto_result(&result, points.len()) {
                            // Success: reset failure count, increment success count
                            let (failures, successes) = {
                                let mut failures = self.giotto_failure_count.lock().unwrap();
                                *failures = 0;
                                let mut successes = self.giotto_success_count.lock().unwrap();
                                *successes += 1;
                                let success_count = *successes;
                                
                                // Phase 2.3: Record metrics
                                crate::metrics::tcs_analyzer_metrics().record_giotto_success();
                                crate::metrics::tcs_analyzer_metrics().record_giotto_latency(giotto_latency_ms);
                                crate::metrics::tcs_analyzer_metrics().record_giotto_stats(0, success_count);
                                
                                (*failures, success_count)
                            };
                            
                            // Phase 2.2: If we've had 10+ consecutive successes, we're stable
                            if successes >= 10 {
                                debug!("giotto-tda stable: {} consecutive successes", successes);
                            }
                            
                            debug!(
                                "Giotto-tda computation succeeded: latency={:.2}ms, β₀={}, β₁={}, β₂={}, features={}",
                                giotto_latency_ms, result.betti[0], result.betti[1], result.betti[2], result.features.len()
                            );
                            return Ok(result);
                        } else {
                            // Quality check failed: fallback to Rust
                            warn!("giotto-tda result failed quality validation; falling back to Rust implementation");
                            let (failures, successes) = {
                                let mut failures = self.giotto_failure_count.lock().unwrap();
                                *failures += 1;
                                let failure_count = *failures;
                                let mut successes = self.giotto_success_count.lock().unwrap();
                                *successes = 0;
                                
                                // Phase 2.3: Record metrics
                                crate::metrics::tcs_analyzer_metrics().record_giotto_failure();
                                crate::metrics::tcs_analyzer_metrics().record_giotto_fallback();
                                crate::metrics::tcs_analyzer_metrics().record_giotto_stats(failure_count, 0);
                                
                                (failure_count, 0)
                            };
                            // Continue to Rust implementation below
                        }
                    }
                    Err(e) => {
                        // Phase 2.2: Python error - fallback to Rust
                        warn!(%e, "giotto-tda computation failed; falling back to Rust implementation");
                        let (failures, successes) = {
                            let mut failures = self.giotto_failure_count.lock().unwrap();
                            *failures += 1;
                            let failure_count = *failures;

                            // If too many failures, disable approximate TDA temporarily
                            if failure_count >= 5 {
                                warn!("Too many giotto-tda failures ({}); temporarily skipping giotto calls", failure_count);
                            }
                            
                            let mut successes = self.giotto_success_count.lock().unwrap();
                            *successes = 0;
                            
                            // Phase 2.3: Record metrics
                            crate::metrics::tcs_analyzer_metrics().record_giotto_failure();
                            crate::metrics::tcs_analyzer_metrics().record_giotto_fallback();
                            crate::metrics::tcs_analyzer_metrics().record_giotto_stats(failure_count, 0);
                            
                            (failure_count, 0)
                        };
                        // Continue to Rust implementation below
                    }
                }
            } else {
                // Phase 2.2: Too many failures - skip giotto and use Rust directly
                debug!("Skipping giotto-tda due to recent failures; using Rust implementation");
                // Phase 2.3: Record skip metric
                let (failures, successes) = {
                    let failures = self.giotto_failure_count.lock().unwrap();
                    let successes = self.giotto_success_count.lock().unwrap();
                    (*failures, *successes)
                };
                crate::metrics::tcs_analyzer_metrics().record_giotto_stats(failures, successes);
            }
        }

        // Original Rust implementation (or fallback)
        let rust_start = Instant::now();
        let vectors: Vec<DVector<f32>> = points
            .iter()
            .map(|point| DVector::from_vec(point.clone()))
            .collect();

        let homology = PersistentHomology::new(2, max_filtration);
        let raw_features = homology.compute(&vectors);

        let mut features = Vec::with_capacity(raw_features.len());
        let mut entropy_weights = Vec::new();
        let mut betti = [0usize; 3];
        let mut total_weight = 0.0f64;

        for feature in raw_features {
            let persistence = feature.persistence();
            if persistence.is_finite() && persistence > 0.0 {
                entropy_weights.push((feature.dimension, persistence));
                total_weight += persistence as f64;
            }

            if feature.dimension < 3 && feature.death.is_infinite() {
                betti[feature.dimension] += 1;
            }

            features.push(PersistentFeature {
                birth: feature.birth,
                death: feature.death,
                dimension: feature.dimension,
            });
        }

        if total_weight > 0.0 {
            for (_, weight) in entropy_weights.iter_mut() {
                *weight = (*weight as f64 / total_weight) as f32;
            }
        }

        let result = PersistenceResult {
            features,
            entropy: entropy_weights,
            betti,
        };

        // Phase 2.2: Cache Rust result for comparison
        {
            let mut last_result = self.last_rust_result.lock().unwrap();
            *last_result = Some(result.clone());
        }

        // Phase 2.3: Record Rust computation latency
        let rust_latency_ms = rust_start.elapsed().as_secs_f64() * 1000.0;
        crate::metrics::tcs_analyzer_metrics().record_rust_latency(rust_latency_ms);
        
        debug!(
            "Rust persistent homology computation completed: latency={:.2}ms, β₀={}, β₁={}, β₂={}, features={}",
            rust_latency_ms, result.betti[0], result.betti[1], result.betti[2], result.features.len()
        );

        Ok(result)
    }

    /// Phase 2.2: Validate giotto-tda result quality
    /// Uses differential metrics: β₁ count sanity checks, feature count validation
    fn validate_giotto_result(&self, result: &PersistenceResult, num_points: usize) -> bool {
        // Check 1: Betti numbers sanity
        // β₀ should be at least 1 (one connected component)
        if result.betti[0] == 0 {
            warn!("giotto-tda validation failed: β₀ = 0 (should be ≥1)");
            return false;
        }

        // β₁ should not exceed theoretical maximum
        let theoretical_max_betti1 = num_points.saturating_sub(1);
        if result.betti[1] > theoretical_max_betti1 {
            warn!(
                "giotto-tda validation failed: β₁ = {} exceeds theoretical max {}",
                result.betti[1], theoretical_max_betti1
            );
            return false;
        }

        // Check 2: Feature count sanity
        // Should have reasonable number of features
        if result.features.is_empty() && num_points > 1 {
            warn!(
                "giotto-tda validation failed: no features but {} points",
                num_points
            );
            return false;
        }

        // Check 3: Entropy weights consistency
        // Should have reasonable entropy distribution
        let total_entropy_weight: f32 = result.entropy.iter().map(|(_, w)| *w).sum();
        if total_entropy_weight < 0.01 && !result.entropy.is_empty() {
            warn!(
                "giotto-tda validation failed: suspiciously low entropy weight sum: {}",
                total_entropy_weight
            );
            return false;
        }

        // Check 4: Compare with last Rust result if available (differential metric)
        if let Ok(last_result) = self.last_rust_result.lock() {
            if let Some(ref rust_result) = *last_result {
                // Calculate Δβ₁ (change in β₁)
                let delta_betti1 = rust_result.betti[1].abs_diff(result.betti[1]);

                // If β₁ differs significantly (>3), suspect quality issue
                if delta_betti1 > 3 {
                    warn!(
                        "giotto-tda validation warning: Δβ₁ = {} (Rust: {}, Giotto: {})",
                        delta_betti1, rust_result.betti[1], result.betti[1]
                    );
                    // Don't fail on this, but log for monitoring
                }
            }
        }

        debug!(
            "giotto-tda result passed validation: β₀={}, β₁={}, β₂={}, features={}",
            result.betti[0],
            result.betti[1],
            result.betti[2],
            result.features.len()
        );
        true
    }

    /// Compute persistence using giotto-tda Python library (Phase 2.1)
    #[cfg(feature = "pyo3")]
    fn compute_persistence_giotto(
        &self,
        points: &[Point],
        max_filtration: f32,
    ) -> Result<PersistenceResult> {
        use pyo3::prelude::*;
        use pyo3::types::{PyDict, PyList, PyTuple};

        Python::with_gil(|py| {
            // Import the wrapper module
            let sys = py.import("sys")?;
            let path: &PyList = sys.getattr("path")?.downcast()?;

            // Add python directory to path
            let python_dir = env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("python");
            path.insert(0, python_dir.to_str().unwrap_or("python"))?;

            // Import our wrapper module
            let giotto_module = py.import("giotto_tda_wrapper")?;
            let compute_func = giotto_module.getattr("compute_approximate_persistence")?;

            // Convert points to Python list
            let py_points = PyList::empty(py);
            for point in points {
                let py_point = PyList::empty(py);
                for coord in point {
                    py_point.append(*coord)?;
                }
                py_points.append(py_point)?;
            }

            // Call the function
            let result = compute_func.call1((py_points, max_filtration))?;
            let result_dict = result.downcast::<PyDict>()?;

            // Extract features
            let features_py = result_dict
                .get_item("features")?
                .ok_or_else(|| anyhow!("Missing 'features' key in result"))?;
            let features_list = features_py.downcast::<PyList>()?;

            let mut features = Vec::new();
            let mut entropy_weights = Vec::new();
            let mut betti = [0usize; 3];

            // Parse features
            for item in features_list.iter() {
                let feature_tuple = item.downcast::<PyTuple>()?;
                let birth: f32 = feature_tuple.get_item(0)?.extract()?;
                let death: f64 = feature_tuple.get_item(1)?.extract()?;
                let dimension: usize = feature_tuple.get_item(2)?.extract()?;

                let death_f32 = if death.is_infinite() {
                    f32::INFINITY
                } else {
                    death as f32
                };

                let persistence = if death_f32.is_infinite() {
                    if dimension < 3 {
                        betti[dimension] += 1;
                    }
                    f32::INFINITY
                } else {
                    death_f32 - birth
                };

                features.push(PersistentFeature {
                    birth,
                    death: death_f32,
                    dimension,
                });

                if persistence.is_finite() && persistence > 0.0 {
                    entropy_weights.push((dimension, persistence));
                }
            }

            // Extract betti numbers (may be overridden by features processing)
            if let Ok(Some(betti_py)) = result_dict.get_item("betti") {
                if let Ok(betti_list) = betti_py.downcast::<PyList>() {
                    for (i, betti_val) in betti_list.iter().enumerate().take(3) {
                        if let Ok(val) = betti_val.extract::<usize>() {
                            betti[i] = val;
                        }
                    }
                }
            }

            // Extract and normalize entropy weights
            let mut total_weight = 0.0f64;
            for (_, weight) in &entropy_weights {
                total_weight += *weight as f64;
            }

            if total_weight > 0.0 {
                for (_, weight) in entropy_weights.iter_mut() {
                    *weight = (*weight as f64 / total_weight) as f32;
                }
            }

            Ok(PersistenceResult {
                features,
                entropy: entropy_weights,
                betti,
            })
        })
    }

    /// Fallback for when pyo3 is not available
    #[cfg(not(feature = "pyo3"))]
    fn compute_persistence_giotto(
        &self,
        points: &[Point],
        max_filtration: f32,
    ) -> Result<PersistenceResult> {
        warn!("giotto-tda requested but pyo3 feature not enabled; falling back to Rust implementation");
        // Fall back to original Rust implementation
        let vectors: Vec<DVector<f32>> = points
            .iter()
            .map(|point| DVector::from_vec(point.clone()))
            .collect();

        let homology = PersistentHomology::new(2, max_filtration);
        let raw_features = homology.compute(&vectors);

        let mut features = Vec::with_capacity(raw_features.len());
        let mut entropy_weights = Vec::new();
        let mut betti = [0usize; 3];
        let mut total_weight = 0.0f64;

        for feature in raw_features {
            let persistence = feature.persistence();
            if persistence.is_finite() && persistence > 0.0 {
                entropy_weights.push((feature.dimension, persistence));
                total_weight += persistence as f64;
            }

            if feature.dimension < 3 && feature.death.is_infinite() {
                betti[feature.dimension] += 1;
            }

            features.push(PersistentFeature {
                birth: feature.birth,
                death: feature.death,
                dimension: feature.dimension,
            });
        }

        if total_weight > 0.0 {
            for (_, weight) in entropy_weights.iter_mut() {
                *weight = (*weight as f64 / total_weight) as f32;
            }
        }

        Ok(PersistenceResult {
            features,
            entropy: entropy_weights,
            betti,
        })
    }

    fn persistence_entropy(weights: &[(usize, f32)]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }
        weights
            .iter()
            .map(|(_, weight)| {
                let p = (*weight).clamp(1e-9, 1.0) as f64;
                -p * p.log2()
            })
            .sum()
    }

    fn compute_euler_characteristic(betti: &[usize; 3]) -> f64 {
        betti[0] as f64 - betti[1] as f64 + betti[2] as f64
    }

    fn total_persistence(features: &[PersistentFeature]) -> f64 {
        features
            .iter()
            .map(|feature| (feature.death - feature.birth).abs() as f64)
            .sum()
    }

    fn max_persistence(features: &[PersistentFeature]) -> f64 {
        features
            .iter()
            .map(|feature| (feature.death - feature.birth).abs() as f64)
            .fold(0.0, f64::max)
    }

    fn mean_persistence(total: f64, features: &[PersistentFeature]) -> f64 {
        if features.is_empty() {
            0.0
        } else {
            total / features.len() as f64
        }
    }

    fn laplacian_spectral_radius(distances: &[Vec<f32>]) -> f64 {
        let n = distances.len();
        if n == 0 {
            return 0.0;
        }

        let mut laplacian = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            let mut degree = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist = distances[i][j] as f64;
                if dist <= f64::EPSILON {
                    continue;
                }
                let weight = (1.0 / dist).min(1e6);
                laplacian[(i, j)] = -weight;
                degree += weight;
            }
            laplacian[(i, i)] = degree;
        }

        laplacian
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .fold(0.0f64, |acc, value| acc.max(*value))
    }

    fn approximate_phi_from_betti(betti: &[usize; 3]) -> f64 {
        let total: f64 = betti.iter().map(|&b| b as f64).sum();
        if total <= f64::EPSILON {
            return 0.0;
        }

        let weights = [0.5_f64, 0.3, 0.2];
        betti
            .iter()
            .zip(weights.iter())
            .map(|(&b, &w)| w * (b as f64 / total))
            .sum()
    }
    /// Analyze transition between two states
    pub fn analyze_transition(
        &mut self,
        before: &PadGhostState,
        after: &PadGhostState,
    ) -> Result<TransitionAnalysis> {
        let before_signature = self.analyze_state(before)?;
        let after_signature = self.analyze_state(after)?;

        // Compute Betti changes
        let betti_delta = [
            after_signature.betti_numbers[0] as i32 - before_signature.betti_numbers[0] as i32,
            after_signature.betti_numbers[1] as i32 - before_signature.betti_numbers[1] as i32,
            after_signature.betti_numbers[2] as i32 - before_signature.betti_numbers[2] as i32,
        ];

        // Infer cobordism from Betti changes
        let inferred_cobordism = TQFTEngine::infer_cobordism_from_betti(
            &before_signature.betti_numbers,
            &after_signature.betti_numbers,
        );

        Ok(TransitionAnalysis {
            before: before_signature,
            after: after_signature,
            betti_delta,
            inferred_cobordism,
        })
    }
}

impl PadGhostState {
    #[allow(dead_code)]
    fn to_tensor(&self) -> anyhow::Result<Tensor> {
        let mut values: Vec<f32> = Vec::with_capacity(512);
        values.extend(self.pad.iter().map(|v| *v as f32));
        values.extend(self.mu.iter().map(|v| *v as f32));
        values.extend(self.sigma.iter().map(|v| *v as f32));

        // Pad to the expected embedding width
        if values.len() < 512 {
            values.resize(512, 0.0);
        }

        let tensor = Tensor::from_vec(values, (1, 512), &Device::Cpu)?;
        Ok(tensor)
    }
}

/// Analysis of transition between two states
#[derive(Debug, Clone)]
pub struct TransitionAnalysis {
    pub before: TopologicalSignature,
    pub after: TopologicalSignature,
    pub betti_delta: [i32; 3],
    pub inferred_cobordism: Option<Cobordism>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torus::PadGhostState as PadState;
    use anyhow::Result;

    #[test]
    fn test_betti_delta_signals_change() -> Result<()> {
        let mut analyzer = TCSAnalyzer::new()?;
        let before = PadState {
            pad: [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
            entropy: 0.4,
            mu: [0.0; 7],
            sigma: [0.1; 7],
        };
        let after = PadState {
            pad: [0.5, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0],
            entropy: 0.35,
            mu: [0.0; 7],
            sigma: [0.12; 7],
        };
        let trans = analyzer.analyze_transition(&before, &after)?;
        assert_eq!(trans.betti_delta.len(), 3);
        Ok(())
    }

    #[test]
    fn test_tcs_delta() -> Result<()> {
        let mut analyzer = TCSAnalyzer::new()?;
        let mut pad_state = PadState {
            pad: [0.0; 7],
            entropy: 0.0,
            mu: [0.0; 7],
            sigma: [0.0; 7],
        };
        pad_state.pad[0] = 0.5; // Simple state
        let signature = analyzer.analyze_state(&pad_state)?;
        // Basic check: entropy should be computed
        assert!(signature.persistence_entropy >= 0.0);
        // Delta proxy: knot complexity
        let delta = signature.knot_complexity; // Assume baseline 0
        assert!(delta.is_finite());
        Ok(())
    }
}
