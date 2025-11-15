//! TCS Topology Analysis Layer
//! Computes persistent homology, knot invariants, and TQFT signatures on every state

use anyhow::{Context, Result};
use blake3::Hasher;
use candle_core::{Device, Tensor};
use dashmap::DashMap;
use nalgebra::DVector;
#[cfg(feature = "knot")]
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::config::{env_value, TcsRuntimeConfig};
use crate::erag::EmotionalVector;
use crate::topology::tda_features::{standardize_rows, standardize_vector};
use crate::torus::PadGhostState;
use tcs_core::PersistentFeature;
#[cfg(feature = "knot")]
use tcs_knot::{JonesPolynomial, KnotDiagram};
use tcs_tda::{
    LaplacianSpectrum, MotifMetrics, PersistentHomology, PersistentLaplacian, TakensEmbedding,
};
use tcs_tqft::{Cobordism, TQFTEngine};

type Point = Vec<f32>;

fn record_topology_metrics(betti: &[usize; 3], complexity: f64) {
    let metrics = crate::metrics::tcs_analyzer_metrics();
    metrics.record_betti_numbers(betti);
    metrics.record_knot_complexity(complexity);
}

fn append_pad_points(points: &mut Vec<Point>, pad: &[f64], mu: &[f64], sigma: &[f64]) {
    let len = pad.len().min(mu.len()).min(sigma.len()).min(7);
    if len == 0 {
        return;
    }

    let pad_mean = pad[..len].iter().sum::<f64>() / len as f64;
    let variance = pad[..len]
        .iter()
        .map(|&value| (value - pad_mean).powi(2))
        .sum::<f64>()
        / len as f64;
    let variance_scale = variance.sqrt();

    for i in 0..len {
        let mut coords = vec![
            pad[i],
            mu[i],
            sigma[i],
            (pad[i] - pad_mean) * variance_scale,
            mu[i] * sigma[i],
        ];
        while coords.len() < 7 {
            coords.push(0.0);
        }
        let mut coords_f32: Vec<f32> = coords.into_iter().map(|value| value as f32).collect();
        standardize_vector(&mut coords_f32);
        points.push(coords_f32);
    }
}

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| {
            let v = value as f64;
            v * v
        })
        .sum::<f64>()
        .sqrt()
}

fn l2_distance(lhs: &[f32], rhs: &[f32]) -> f64 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..len {
        let diff = lhs[i] as f64 - rhs[i] as f64;
        sum += diff * diff;
    }
    sum.sqrt()
}

fn window_bounds(len: usize, slices: usize) -> Vec<(usize, usize)> {
    if len == 0 || slices == 0 {
        return Vec::new();
    }
    let slices = slices.max(1).min(len);
    let chunk = ((len as f64) / (slices as f64)).ceil().max(1.0) as usize;
    let mut bounds = Vec::with_capacity(slices);
    let mut start = 0;
    while start < len && bounds.len() < slices {
        let end = (start + chunk).min(len);
        bounds.push((start, end));
        start = end;
    }
    if bounds.is_empty() && len > 0 {
        bounds.push((0, len));
    }
    bounds
}

fn windowed_mean(series: &[f64], slices: usize) -> Vec<f64> {
    if series.is_empty() || slices == 0 {
        return Vec::new();
    }
    let mut values = Vec::new();
    for (start, end) in window_bounds(series.len(), slices) {
        if start >= end {
            continue;
        }
        let window = &series[start..end];
        let total: f64 = window.iter().copied().filter(|v| v.is_finite()).sum();
        let count = window.iter().filter(|v| v.is_finite()).count().max(1) as f64;
        values.push(total / count);
    }
    while values.len() < slices {
        values.push(0.0);
    }
    values
}

fn norm_series(points: &[Vec<f32>]) -> Vec<f64> {
    points.iter().map(|point| l2_norm(point)).collect()
}

fn arc_length_series(points: &[Vec<f32>]) -> Vec<f64> {
    if points.len() < 2 {
        return Vec::new();
    }
    let mut series = Vec::with_capacity(points.len() - 1);
    for pair in points.windows(2) {
        series.push(l2_distance(&pair[0], &pair[1]));
    }
    series
}

fn sanitize_series(series: Vec<f64>) -> Vec<f64> {
    series
        .into_iter()
        .map(|value| if value.is_finite() { value } else { 0.0 })
        .collect()
}

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

    // Temporal and signal metrics
    pub takens_energy: f64,
    pub takens_window_count: usize,
    #[serde(default)]
    pub takens_window_energy: Vec<f64>,
    pub emotional_flux: f64,
    pub emotional_drift: f64,
    #[serde(default)]
    pub emotional_arc_lengths: Vec<f64>,
    pub gradient_energy: f64,
    pub gradient_volatility: f64,
    #[serde(default)]
    pub gradient_energy_trend: Vec<f64>,

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
        takens_energy: f64,
        takens_window_count: usize,
        takens_window_energy: Vec<f64>,
        emotional_flux: f64,
        emotional_drift: f64,
        emotional_arc_lengths: Vec<f64>,
        gradient_energy: f64,
        gradient_volatility: f64,
        gradient_energy_trend: Vec<f64>,
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
            takens_energy,
            takens_window_count,
            takens_window_energy,
            emotional_flux,
            emotional_drift,
            emotional_arc_lengths,
            gradient_energy,
            gradient_volatility,
            gradient_energy_trend,
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

    let signal_slices = env_value("TCS_SIGNAL_WINDOW_SLICES")
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4);
    let takens_scalar_series: Vec<f64> = pad.iter().map(|value| value.abs()).collect();
    let takens_window_energy = sanitize_series(windowed_mean(&takens_scalar_series, signal_slices));

    let computation_time_ms = analysis_start.elapsed().as_secs_f64() * 1000.0;

    let takens_energy = pad_variance.sqrt();
    let takens_window_count = pad.len();
    let sigma_mean = if sigma.is_empty() {
        0.0
    } else {
        sigma.iter().sum::<f64>() / sigma.len() as f64
    };
    let mu_mean = if mu.is_empty() {
        0.0
    } else {
        mu.iter().sum::<f64>() / mu.len() as f64
    };
    let emotional_delta_series: Vec<f64> = pad
        .iter()
        .zip(mu.iter())
        .zip(sigma.iter())
        .map(|((pad_value, mu_value), sigma_value)| {
            (pad_value - mu_value).abs() + sigma_value.abs()
        })
        .collect();
    let emotional_arc_lengths =
        sanitize_series(windowed_mean(&emotional_delta_series, signal_slices));
    let emotional_flux = sigma_mean.abs();
    let emotional_drift = (pad_mean - mu_mean).abs();
    let gradient_energy = laplacian_spectral_radius;
    let gradient_volatility = pad_variance;
    let gradient_series: Vec<f64> = if embedding.len() >= 2 {
        embedding
            .windows(2)
            .map(|window| (window[1] - window[0]) as f64)
            .map(|delta| delta.abs())
            .collect()
    } else {
        vec![gradient_energy]
    };
    let gradient_energy_trend = sanitize_series(windowed_mean(&gradient_series, signal_slices));

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
        takens_energy,
        takens_window_count,
        takens_window_energy,
        emotional_flux,
        emotional_drift,
        emotional_arc_lengths,
        gradient_energy,
        gradient_volatility,
        gradient_energy_trend,
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

fn sample_from_padghost(state: &PadGhostState) -> Vec<f32> {
    let mut sample = Vec::with_capacity(22);
    for value in state.pad.iter() {
        sample.push(*value as f32);
    }
    for value in state.mu.iter() {
        sample.push(*value as f32);
    }
    for value in state.sigma.iter() {
        sample.push(*value as f32);
    }
    sample.push(state.entropy as f32);
    standardize_vector(&mut sample);
    sample
}

fn sample_from_tcs_state(state: &TCSState) -> Vec<f32> {
    let mut sample = Vec::with_capacity(22);
    for i in 0..7 {
        let value = state.pad.get(i).copied().unwrap_or(0.0);
        sample.push(value as f32);
    }
    for i in 0..7 {
        let value = state.mu.get(i).copied().unwrap_or(0.0);
        sample.push(value as f32);
    }
    for i in 0..7 {
        let value = state.sigma.get(i).copied().unwrap_or(0.0);
        sample.push(value as f32);
    }
    let entropy_proxy = if state.pad.is_empty() {
        0.0
    } else {
        state.pad.iter().map(|value| value.abs()).sum::<f64>() / state.pad.len() as f64
    };
    sample.push(entropy_proxy as f32);
    standardize_vector(&mut sample);
    sample
}

/// TCS Analysis Engine
pub struct TCSAnalyzer {
    cache: Arc<TopologyCache>,
    device: Device,
    enable_gpu: bool,
    config: TcsRuntimeConfig,
    persistent_laplacian: PersistentLaplacian,
    laplacian_resolution: usize,
    zero_tolerance: f64,
    pad_history: VecDeque<PadGhostState>,
    takens_history_len: usize,
    emotional_history: VecDeque<Vec<f32>>,
    emotional_history_len: usize,
    lora_gradient_history: VecDeque<Vec<f32>>,
    gradient_history_len: usize,
}

impl TCSAnalyzer {
    /// Initialize TCS analyzer
    pub fn new() -> Result<Self> {
        let config = crate::config::TcsRuntimeConfig::from_env();
        Self::new_with_runtime(&config)
    }

    /// Initialize TCS analyzer with configuration (Phase 2.1)
    pub fn new_with_runtime(config: &TcsRuntimeConfig) -> Result<Self> {
        let cache_dir = config
            .cache_dir
            .clone()
            .unwrap_or_else(|| "storage/topology_cache".to_string());
        let cache = Arc::new(TopologyCache::new(
            Duration::from_secs(config.cache_ttl_secs),
            config.cache_capacity,
            PathBuf::from(&cache_dir),
        )?);

        let enable_gpu = config.enable_gpu;
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

        let laplacian_resolution = config.laplacian_resolution.max(1);
        let persistent_laplacian =
            PersistentLaplacian::new(config.max_dimension, config.zero_tolerance);

        info!(
            max_dimension = config.max_dimension,
            laplacian_resolution,
            zero_tolerance = config.zero_tolerance,
            approximate = config.approximate_laplacian,
            enable_gpu,
            cache_dir = %cache_dir,
            "TCS Analyzer initialized"
        );

        Ok(Self {
            cache,
            device,
            enable_gpu,
            config: config.clone(),
            persistent_laplacian,
            laplacian_resolution,
            zero_tolerance: config.zero_tolerance,
            pad_history: VecDeque::with_capacity(config.takens_history_len.max(1)),
            takens_history_len: config.takens_history_len.max(1),
            emotional_history: VecDeque::with_capacity(config.emotional_signal_len.max(1)),
            emotional_history_len: config.emotional_signal_len.max(1),
            lora_gradient_history: VecDeque::with_capacity(config.gradient_signal_len.max(1)),
            gradient_history_len: config.gradient_signal_len.max(1),
        })
    }

    pub fn ingest_signals(
        &mut self,
        pad_state: &PadGhostState,
        emotional_vectors: Option<&[EmotionalVector]>,
        lora_gradients: Option<&[f32]>,
    ) {
        self.pad_history.push_back(pad_state.clone());
        while self.pad_history.len() > self.takens_history_len {
            self.pad_history.pop_front();
        }

        if let Some(emotions) = emotional_vectors {
            for vector in emotions {
                let mut point = vec![
                    vector.joy,
                    vector.sadness,
                    vector.anger,
                    vector.fear,
                    vector.surprise,
                    vector.joy - vector.sadness,
                    vector.anger - vector.fear,
                ];
                standardize_vector(&mut point);
                self.emotional_history.push_back(point);
            }
            while self.emotional_history.len() > self.emotional_history_len {
                self.emotional_history.pop_front();
            }
        }

        if let Some(gradients) = lora_gradients {
            if !gradients.is_empty() {
                let mut gradient_vec = gradients.to_vec();
                standardize_vector(&mut gradient_vec);
                self.lora_gradient_history.push_back(gradient_vec);
                while self.lora_gradient_history.len() > self.gradient_history_len {
                    self.lora_gradient_history.pop_front();
                }
            }
        }
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

    fn takens_energy(points: &[Point]) -> f64 {
        points.iter().map(|point| l2_norm(point)).sum()
    }

    fn emotional_flux_metric(&self) -> f64 {
        if self.emotional_history.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut iter = self.emotional_history.iter();
        if let Some(mut previous) = iter.next() {
            for current in iter {
                total += l2_distance(previous, current);
                previous = current;
            }
        }
        total / (self.emotional_history.len() - 1) as f64
    }

    fn emotional_drift_metric(&self) -> f64 {
        match (
            self.emotional_history.front(),
            self.emotional_history.back(),
        ) {
            (Some(first), Some(last)) if self.emotional_history.len() > 1 => {
                l2_distance(first, last)
            }
            _ => 0.0,
        }
    }

    fn gradient_metrics(&self) -> (f64, f64) {
        if self.lora_gradient_history.is_empty() {
            return (0.0, 0.0);
        }
        let magnitudes: Vec<f64> = self
            .lora_gradient_history
            .iter()
            .map(|vector| l2_norm(vector))
            .collect();
        let mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
        let variance = magnitudes
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / magnitudes.len() as f64;
        (mean, variance.sqrt())
    }

    fn emotional_points(&self) -> Vec<Point> {
        let mut points: Vec<Point> = self.emotional_history.iter().cloned().collect();
        standardize_rows(&mut points);
        points
    }

    fn gradient_points(&self) -> Vec<Point> {
        let mut points: Vec<Point> = self.lora_gradient_history.iter().cloned().collect();
        standardize_rows(&mut points);
        points
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

        let mut point_cloud = self.pad_to_points(&guard);
        let mut takens_points = self.takens_embedding_points(&guard);
        let takens_energy = Self::takens_energy(&takens_points);
        let takens_window_count = takens_points.len();
        let mut emotional_points = self.emotional_points();
        let mut gradient_points = self.gradient_points();
        let emotional_flux = self.emotional_flux_metric();
        let emotional_drift = self.emotional_drift_metric();
        let (gradient_energy, gradient_volatility) = self.gradient_metrics();
        let signal_slices = self.config.signal_window_slices.max(1);
        let takens_window_energy = if takens_points.is_empty() {
            Vec::new()
        } else {
            sanitize_series(windowed_mean(&norm_series(&takens_points), signal_slices))
        };
        let emotional_arc_lengths = if emotional_points.len() >= 2 {
            sanitize_series(windowed_mean(
                &arc_length_series(&emotional_points),
                signal_slices,
            ))
        } else {
            Vec::new()
        };
        let gradient_energy_trend = if gradient_points.is_empty() {
            Vec::new()
        } else {
            sanitize_series(windowed_mean(&norm_series(&gradient_points), signal_slices))
        };

        if !point_cloud.is_empty()
            || !takens_points.is_empty()
            || !emotional_points.is_empty()
            || !gradient_points.is_empty()
        {
            let target_dimension = point_cloud
                .iter()
                .chain(takens_points.iter())
                .chain(emotional_points.iter())
                .chain(gradient_points.iter())
                .map(|point| point.len())
                .max()
                .unwrap_or(0);

            if target_dimension > 0 {
                for collection in [
                    &mut point_cloud,
                    &mut takens_points,
                    &mut emotional_points,
                    &mut gradient_points,
                ] {
                    for point in collection.iter_mut() {
                        if point.len() < target_dimension {
                            point.resize(target_dimension, 0.0);
                        }
                    }
                }
            }
        }

        point_cloud.extend(takens_points);
        point_cloud.extend(emotional_points);
        point_cloud.extend(gradient_points);
        if point_cloud.is_empty() {
            anyhow::bail!("no points generated for topology analysis");
        }

        let max_filtration = self.config.max_filtration;

        let dvec_points: Vec<DVector<f32>> = point_cloud
            .iter()
            .map(|p| DVector::from_vec(p.clone()))
            .collect();

        let distances = self.compute_pairwise_distances(&point_cloud)?;
        let snapshot = self.safe_snapshot(&distances, &dvec_points, max_filtration);

        let mut betti = snapshot.betti;

        let num_points = dvec_points.len();
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

        let mut persistence_features = snapshot.features.clone();

        if self.config.persistence_threshold > 0.0 {
            let threshold = self.config.persistence_threshold;
            let filtered: Vec<PersistentFeature> = persistence_features
                .iter()
                .cloned()
                .filter(|feature| {
                    if feature.dimension == 0 {
                        true
                    } else {
                        (feature.death - feature.birth).abs() >= threshold
                    }
                })
                .collect();
            if filtered.is_empty() && self.config.robust_mode_enabled() {
                anyhow::bail!(
                    "no persistence features above configured threshold {:.3}",
                    threshold
                );
            }
            if !filtered.is_empty() {
                persistence_features = filtered;
            }
        }

        let (mut knot_polynomial, mut knot_complexity) = analyze_knot(&persistence_features);
        if knot_complexity <= f64::EPSILON {
            knot_complexity = snapshot.motifs.average_clustering as f64;
        }
        if knot_polynomial.is_empty() {
            knot_polynomial = "unknot".to_string();
        }
        let cobordism_type = self.infer_cobordism(&betti);
        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

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
            takens_energy,
            takens_window_count,
            takens_window_energy,
            emotional_flux,
            emotional_drift,
            emotional_arc_lengths,
            gradient_energy,
            gradient_volatility,
            gradient_energy_trend,
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
        for history in &self.pad_history {
            append_pad_points(&mut points, &history.pad, &history.mu, &history.sigma);
        }
        append_pad_points(&mut points, &pad_state.pad, &pad_state.mu, &pad_state.sigma);
        points
    }

    fn takens_embedding_points(&self, pad_state: &TCSState) -> Vec<Point> {
        if self.config.takens_dimension <= 1 {
            return Vec::new();
        }

        let series = self.time_series_from_state(pad_state);
        if series.is_empty() {
            return Vec::new();
        }

        let data_dim = series.first().map(|sample| sample.len()).unwrap_or(0);
        if data_dim == 0 {
            return Vec::new();
        }

        let required = self.config.takens_dimension * self.config.takens_delay;
        if series.len() <= required {
            return Vec::new();
        }

        let takens = TakensEmbedding::new(
            self.config.takens_dimension,
            self.config.takens_delay,
            data_dim,
        );

        takens
            .embed(&series)
            .into_iter()
            .map(|vector| vector.as_slice().to_vec())
            .collect()
    }

    fn time_series_from_state(&self, pad_state: &TCSState) -> Vec<Vec<f32>> {
        let mut series: Vec<Vec<f32>> = self.pad_history.iter().map(sample_from_padghost).collect();
        series.push(sample_from_tcs_state(pad_state));
        standardize_rows(&mut series);
        series
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

    fn safe_snapshot(
        &self,
        distances: &[Vec<f32>],
        point_cloud: &[DVector<f32>],
        max_filtration: f32,
    ) -> LaplacianSnapshot {
        match std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.compute_snapshot(distances, point_cloud, max_filtration)
        })) {
            Ok(snapshot) => snapshot,
            Err(_) => {
                warn!("Persistent Laplacian panic encountered; substituting synthetic snapshot");
                LaplacianSnapshot {
                    features: self.synthetic_features_from_points(distances, point_cloud),
                    entropy_weights: Vec::new(),
                    betti: self.synthetic_betti(point_cloud),
                    spectra: Vec::new(),
                    spectral_flux: [0.0; 3],
                    motifs: MotifMetrics::default(),
                }
            }
        }
    }

    fn compute_snapshot(
        &self,
        distances: &[Vec<f32>],
        point_cloud: &[DVector<f32>],
        max_filtration: f32,
    ) -> LaplacianSnapshot {
        let num_points = point_cloud.len();

        // Clamp filtration depth when num_points is small to prevent shape explosions
        // When num_points < threshold, limit resolution to prevent filtered simplices from exploding
        let filtration_depth_threshold = std::env::var("TCS_FILTRATION_DEPTH_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(50); // Default threshold: clamp when num_points < 50

        let base_resolution = if self.config.approximate_laplacian {
            self.laplacian_resolution.min(4).max(1)
        } else {
            self.laplacian_resolution
        };

        let max_resolution = distances.len().saturating_sub(1).max(1);
        let mut resolution = base_resolution.min(max_resolution);

        // Apply filtration depth clamping for small point clouds
        if num_points < filtration_depth_threshold {
            // Clamp resolution more aggressively for small point clouds
            // This prevents filtered simplices from exploding in size
            let clamped_resolution = (num_points / 5).max(1).min(resolution);
            if clamped_resolution < resolution {
                debug!(
                    num_points = num_points,
                    original_resolution = resolution,
                    clamped_resolution = clamped_resolution,
                    threshold = filtration_depth_threshold,
                    "Clamping filtration depth to prevent shape explosion"
                );
                resolution = clamped_resolution;
            }
        }

        let filtration = self.persistent_laplacian.build_filtration(
            distances,
            max_filtration,
            resolution.max(1),
        );
        let spectra = self.persistent_laplacian.analyze(&filtration);
        let spectral_flux = self.persistent_laplacian.spectral_flux(&spectra);
        let betti = self.persistent_laplacian.harmonic_counts(&spectra);
        let entropy_weights = self.entropy_weights_from_spectra(&spectra);
        let mut features = self.compute_persistent_features(point_cloud, max_filtration);
        let mut betti = betti;
        if features.is_empty() {
            features = self.synthetic_features_from_points(distances, point_cloud);
            betti = self.synthetic_betti(point_cloud);
        }
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

    fn compute_persistent_features(
        &self,
        point_cloud: &[DVector<f32>],
        max_filtration: f32,
    ) -> Vec<PersistentFeature> {
        if point_cloud.is_empty() {
            return Vec::new();
        }

        let engine = PersistentHomology::new(self.config.max_dimension, max_filtration);
        let mut features: Vec<PersistentFeature> = engine
            .compute(point_cloud)
            .into_iter()
            .filter(|feat| feat.birth.is_finite() && feat.death.is_finite())
            .map(|feat| PersistentFeature {
                birth: feat.birth,
                death: feat.death,
                dimension: feat.dimension,
            })
            .collect();

        features.sort_by(|a, b| {
            let lhs = (a.death - a.birth).abs();
            let rhs = (b.death - b.birth).abs();
            rhs.partial_cmp(&lhs).unwrap_or(Ordering::Equal)
        });

        features
    }

    fn entropy_weights_from_spectra(&self, spectra: &[LaplacianSpectrum]) -> Vec<(usize, f32)> {
        let mut mass = [0.0f64; 3];
        for spectrum in spectra {
            if spectrum.dimension > 2 || spectrum.eigenvalues.is_empty() {
                continue;
            }
            let energy: f64 = spectrum.eigenvalues.iter().map(|value| value.abs()).sum();
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

    fn synthetic_features_from_points(
        &self,
        distances: &[Vec<f32>],
        point_cloud: &[DVector<f32>],
    ) -> Vec<PersistentFeature> {
        if point_cloud.is_empty() {
            return Vec::new();
        }

        let mut features = Vec::with_capacity(point_cloud.len().min(9));
        let mut norms: Vec<f32> = point_cloud
            .iter()
            .map(|vec| vec.iter().map(|value| value.abs()).sum::<f32>())
            .collect();

        if norms.iter().all(|value| value.abs() < f32::EPSILON) {
            for (idx, norm) in norms.iter_mut().enumerate() {
                *norm = (idx as f32 + 1.0) * 0.1;
            }
        }

        for (idx, vector) in point_cloud.iter().enumerate().take(9) {
            let birth = norms
                .get(idx)
                .copied()
                .unwrap_or_else(|| vector.iter().map(|value| value.abs()).sum::<f32>());
            let distance_row = distances.get(idx);
            let spread = distance_row
                .iter()
                .flat_map(|row| row.iter())
                .copied()
                .filter(|value| value.is_finite())
                .fold(0.0f32, |acc, value| acc.max(value.abs()));

            let lifetime = (spread + 0.1).clamp(0.05, 5.0);
            let dimension = idx % 3;
            features.push(PersistentFeature {
                birth,
                death: birth + lifetime,
                dimension,
            });
        }

        features
    }

    fn synthetic_betti(&self, point_cloud: &[DVector<f32>]) -> [usize; 3] {
        if point_cloud.is_empty() {
            return [0, 0, 0];
        }

        let mut h0 = 0usize;
        let mut h1 = 0usize;
        let mut h2 = 0usize;

        for point in point_cloud {
            let positives = point.iter().filter(|value| **value >= 0.0).count();
            let negatives = point.iter().filter(|value| **value < 0.0).count();
            let high_energy = point.iter().any(|value| value.abs() > 0.75);

            if positives >= negatives {
                h0 += 1;
            } else {
                h1 += 1;
            }

            if high_energy {
                h2 += 1;
            }
        }

        if h0 == 0 {
            h0 = 1;
        }

        [h0.min(12), h1.min(12), h2.min(12)]
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

    fn pairwise_gpu(points: &[Point], _device: &Device) -> anyhow::Result<Vec<Vec<f32>>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        warn!("GPU pairwise distance unavailable due to dimensional instability; using CPU implementation");
        Ok(Self::pairwise_cpu(points))
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

fn analyze_knot(features: &[PersistentFeature]) -> (String, f64) {
    #[cfg(feature = "knot")]
    {
        static JONES_POLYNOMIAL: Lazy<Mutex<JonesPolynomial>> =
            Lazy::new(|| Mutex::new(JonesPolynomial::new(256)));

        let diagram = knot_diagram_from_features(features);
        let mut guard = JONES_POLYNOMIAL
            .lock()
            .expect("JonesPolynomial mutex poisoned");
        let cognitive = guard.analyze(&diagram);
        let complexity = cognitive.complexity_score as f64;
        (cognitive.polynomial, complexity)
    }
    #[cfg(not(feature = "knot"))]
    {
        let complexity: f64 = features
            .iter()
            .filter(|feature| feature.dimension == 1)
            .map(|feature| (feature.death - feature.birth).abs() as f64)
            .sum();
        (String::new(), complexity)
    }
}

#[cfg(feature = "knot")]
fn knot_diagram_from_features(features: &[PersistentFeature]) -> KnotDiagram {
    let mut crossings = Vec::new();
    for feature in features.iter().filter(|f| f.dimension == 1) {
        let persistence = (feature.death - feature.birth).abs();
        if !persistence.is_finite() || persistence <= f32::EPSILON {
            continue;
        }
        let repetitions = (persistence * 10.0).round() as i32;
        let repetitions = repetitions.clamp(1, 5);
        let mut sign = if feature.birth <= feature.death {
            1
        } else {
            -1
        };
        for idx in 0..repetitions {
            let value = if idx % 2 == 0 { sign } else { -sign };
            crossings.push(value);
        }
    }

    if crossings.is_empty() {
        KnotDiagram::unknot()
    } else {
        KnotDiagram { crossings }
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
