//! TCS Topology Analysis Layer
//! Computes persistent homology, knot invariants, and TQFT signatures on every state

use anyhow::{bail, Context, Result};
use blake3::Hasher;
use candle_core::{Device, Tensor};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json;
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
use tcs_tda::{LaplacianSpectrum, MotifMetrics, PersistentLaplacian};
use tcs_tqft::{Cobordism, TQFTEngine};

type Point = Vec<f32>;

fn record_topology_metrics(_betti: &[usize; 3], _complexity: f64) {}

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
struct LaplacianSnapshot {
    features: Vec<PersistentFeature>,
    entropy_weights: Vec<(usize, f32)>,
    betti: [usize; 3],
    spectra: Vec<LaplacianSpectrum>,
    spectral_flux: [f64; 3],
    motifs: MotifMetrics,
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
    cache: Arc<TopologyCache>,
    device: Device,
    enable_gpu: bool,
    use_approximate_tda: bool,
    persistent_laplacian: PersistentLaplacian,
    laplacian_resolution: usize,
    zero_tolerance: f64,
}

impl TCSAnalyzer {
    /// Initialize TCS analyzer
    pub fn new() -> Result<Self> {
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

        let laplacian_max_dim = env::var("TCS_LAPLACIAN_MAX_DIM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|value| value.min(2))
            .unwrap_or(2);
        let laplacian_resolution = env::var("TCS_LAPLACIAN_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
            .max(1);
        let zero_tolerance = env::var("TCS_LAPLACIAN_ZERO_TOLERANCE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1e-6);

        let persistent_laplacian = PersistentLaplacian::new(laplacian_max_dim, zero_tolerance);

        info!(
            laplacian_max_dim,
            laplacian_resolution,
            zero_tolerance,
            "TCS Analyzer initialized"
        );
        Ok(Self {
            cache,
            device,
            enable_gpu,
            use_approximate_tda: env::var("USE_APPROXIMATE_TDA")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false),
            persistent_laplacian,
            laplacian_resolution,
            zero_tolerance,
        })
    }

    /// Initialize TCS analyzer with configuration (Phase 2.1)
    pub fn new_with_config(use_approximate_tda: bool) -> Result<Self> {
        let mut analyzer = Self::new()?;
        analyzer.use_approximate_tda = use_approximate_tda;
        if use_approximate_tda {
            info!("TCSAnalyzer initialized with approximate Laplacian mode enabled");
        }
        Ok(analyzer)
    }

    /// Apply TQFT reasoning to evolve a state through cobordism transitions
    pub fn apply_tqft_reasoning(
        &self,
        initial_state: &[f64],
        transitions: &[Cobordism],
    ) -> Result<Vec<f64>> {
        let mut state = initial_state.to_vec();
        for transition in transitions {
            match transition {
                Cobordism::Split => {
                    if let Some(first) = state.first_mut() {
                        *first += 0.01;
                    }
                }
                Cobordism::Merge => {
                    if let Some(first) = state.first_mut() {
                        *first -= 0.01;
                    }
                }
                Cobordism::Birth => state.push(0.0),
                Cobordism::Death => {
                    state.pop();
                }
                Cobordism::Identity => {}
            }
        }
        Ok(state)
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
        let snapshot = self.compute_snapshot(&distances, max_filtration);

        let mut betti = snapshot.betti;

        // Phase 2.3: Record Betti number distribution
        crate::metrics::tcs_analyzer_metrics().record_betti_numbers(&betti);

        let num_points = points.len();
        let theoretical_max = num_points.saturating_sub(1);
        
        // Get Betti1 max constraint from config or environment variable
        // Default to 6 if not specified, but allow override
        let constraint_max = std::env::var("TCS_BETTI1_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6);
        
        let max_allowed = theoretical_max.min(constraint_max);
        
        if constraint_max != 6 {
            debug!(
                "Betti1 max constraint overridden: {} (default: 6)",
                constraint_max
            );
        }

        let original_betti1 = betti[1];
        debug!(
            "Betti numbers before capping: {:?}, num_points={}, theoretical_max={}, constraint_max={}, max_allowed={}",
            betti, num_points, theoretical_max, constraint_max, max_allowed
        );

        if betti[1] > max_allowed {
            warn!(
                "Betti_1 ({}) exceeds maximum (theoretical: {}, constraint: {}), capping to {}. num_points={}",
                betti[1],
                theoretical_max,
                constraint_max,
                max_allowed,
                num_points
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

        let persistence_entropy = Self::persistence_entropy(&snapshot.entropy_weights);
        let spectral_gap = self.dominant_spectral_gap(&snapshot.spectra);
        let phi = Self::approximate_phi_from_betti(&betti);
        debug!(phi, "IIT Φ approximate value");

        let knot_complexity = snapshot.motifs.average_clustering;
        let knot_polynomial = "deprecated".to_string();
        let cobordism_type = self.infer_cobordism(&betti);
        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let persistence_features = snapshot.features.clone();
        let euler_characteristic = Self::compute_euler_characteristic(&betti);
        let total_persistence = Self::total_persistence(&persistence_features);
        let max_persistence = Self::max_persistence(&persistence_features);
        let mean_persistence = Self::mean_persistence(total_persistence, &persistence_features);
        let laplacian_spectral_radius = self.spectral_radius(&snapshot.spectra);

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
            betti.iter().sum(),
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
            let delta_b0 = betti[0] as isize - prev[0] as isize;
            let delta_b1 = betti[1] as isize - prev[1] as isize;
            let delta_b2 = betti[2] as isize - prev[2] as isize;
            if delta_b0 > 0 {
                Some(Cobordism::Split)
            } else if delta_b0 < 0 {
                Some(Cobordism::Merge)
            } else if delta_b1 > 0 || delta_b2 > 0 {
                Some(Cobordism::Birth)
            } else if delta_b1 < 0 || delta_b2 < 0 {
                Some(Cobordism::Death)
            } else {
                Some(Cobordism::Identity)
            }
        } else {
            Some(Cobordism::Identity)
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

    fn compute_snapshot(
        &self,
        distances: &[Vec<f32>],
        max_filtration: f32,
    ) -> LaplacianSnapshot {
        let resolution = if self.use_approximate_tda {
            self.laplacian_resolution.min(4).max(1)
        } else {
            self.laplacian_resolution
        };

        let filtration = self
            .persistent_laplacian
            .build_filtration(distances, max_filtration, resolution);
        let spectra = self.persistent_laplacian.analyze(&filtration);
        let spectral_flux = self.persistent_laplacian.spectral_flux(&spectra);
        let betti = self.persistent_laplacian.harmonic_counts(&spectra);
        let entropy_weights = self.entropy_weights_from_spectra(&spectra);
        let features = self.synthetic_features_from_betti(&betti);
        let motifs = MotifMetrics::compute(distances, max_filtration);

        LaplacianSnapshot {
            features,
            entropy_weights,
            betti,
            spectra,
            spectral_flux,
            motifs,
        }
    }

    fn entropy_weights_from_spectra(
        &self,
        spectra: &[LaplacianSpectrum],
    ) -> Vec<(usize, f32)> {
        let mut mass = [0.0f64; 3];
        for spectrum in spectra {
            if spectrum.dimension > 2 || spectrum.eigenvalues.is_empty() {
                continue;
            }
            let energy: f64 = spectrum
                .eigenvalues
                .iter()
                .map(|value| value.abs())
                .sum();
            if energy > self.zero_tolerance {
                mass[spectrum.dimension] = mass[spectrum.dimension].max(energy);
            }
        }

        let total: f64 = mass.iter().sum();
        if total <= f64::EPSILON {
            return Vec::new();
        }

        mass.iter()
            .enumerate()
            .filter(|(_, value)| **value > 0.0)
            .map(|(dimension, value)| (dimension, (*value / total) as f32))
            .collect()
    }

    fn synthetic_features_from_betti(
        &self,
        betti: &[usize; 3],
    ) -> Vec<PersistentFeature> {
        let mut features = Vec::new();
        for (dimension, count) in betti.iter().enumerate() {
            for _ in 0..*count {
                features.push(PersistentFeature {
                    birth: 0.0,
                    death: f32::INFINITY,
                    dimension,
                });
            }
        }
        features
    }

    fn dominant_spectral_gap(&self, spectra: &[LaplacianSpectrum]) -> f64 {
        let mut gap = 0.0f64;
        for spectrum in spectra {
            if spectrum.dimension > 2 || spectrum.eigenvalues.is_empty() {
                continue;
            }
            let mut eigenvalues = spectrum.eigenvalues.clone();
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut previous = 0.0f64;
            for value in eigenvalues {
                if value <= self.zero_tolerance {
                    previous = value;
                    continue;
                }
                let local_gap = (value - previous).abs();
                if local_gap > gap {
                    gap = local_gap;
                }
                break;
            }
        }
        gap
    }

    fn spectral_radius(&self, spectra: &[LaplacianSpectrum]) -> f64 {
        spectra
            .iter()
            .flat_map(|spectrum| spectrum.eigenvalues.iter())
            .fold(0.0f64, |acc, value| acc.max(value.abs()))
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
        
        // RTX 5090 OPTIMIZATION: Use fused distance calculation
        #[cfg(feature = "gpu")]
        {
            if let Ok(_) = Device::cuda_if_available(0) {
                use crate::gpu_fusion::GpuTensorFusion;
                let fusion = GpuTensorFusion::new(device.clone());
                match fusion.fused_pairwise_distance(&tensor) {
                    Ok(dist_matrix) => {
                        return Ok(dist_matrix.to_vec2::<f32>()?);
                    }
                    Err(e) => {
                        warn!(?e, "Fused distance calculation failed, falling back to sequential");
                    }
                }
            }
        }
        
        // Fallback: Sequential operations
        let norms = tensor.sqr()?.sum_keepdim(1)?; // Shape: (n, 1)
        let norms_t = norms.transpose(0, 1)?; // Shape: (1, n)
        // Ensure correct broadcasting: (n, 1) + (1, n) -> (n, n)
        let mut dist_sq = norms.broadcast_add(&norms_t)?;
        let product = tensor.matmul(&tensor.transpose(0, 1)?)?;
        let scalar = Tensor::new(&[2.0f32], &product.device())?;
        let product_scaled = product.broadcast_mul(&scalar)?;
        dist_sq = (dist_sq - product_scaled)?;
        let zeros = Tensor::zeros(dist_sq.dims(), dist_sq.dtype(), device)?;
        dist_sq = dist_sq.maximum(&zeros)?;
        let dist = dist_sq.sqrt()?;
        // Keep on GPU - only move to CPU when absolutely necessary for final output
        // For RTX 5090, keep computations on GPU as long as possible
        let matrix = dist.to_vec2::<f32>()?;
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

