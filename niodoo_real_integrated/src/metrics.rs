//! Phase 5: Enhanced telemetry for optimization metrics
//!
//! Adds comprehensive metrics for tracking optimization performance:
//! - ERAG batch operations (Phase 1)
//! - TCS Analyzer performance (Phase 2)
//! - GPU fitness calculations (Phase 4.3)
//! - CRDT consolidation operations (Phase 4.4)
//! - Curator feedback metrics (Phase 4.2)

use anyhow::{Error, Result};
use once_cell::sync::Lazy;
use prometheus::{
    register_counter, register_gauge, register_histogram, register_histogram_vec, Counter, Encoder,
    Gauge, Histogram, HistogramOpts, HistogramVec, TextEncoder,
};

static METRICS: Lazy<PipelineMetrics> = Lazy::new(|| {
    PipelineMetrics::new().unwrap_or_else(|e| {
        panic!("Failed to initialize Prometheus metrics: {}. This is a critical infrastructure failure.", e);
    })
});

static WEIGHTED_MEMORY_METRICS: Lazy<WeightedMemoryMetrics> = Lazy::new(|| {
    WeightedMemoryMetrics::new().unwrap_or_else(|e| {
        panic!("Failed to initialize weighted memory metrics: {}. This is a critical infrastructure failure.", e);
    })
});

#[derive(Clone)]
pub struct PipelineMetrics {
    entropy_gauge: Gauge,
    latency_histogram: Histogram,
    rouge_gauge: Gauge,
    threats_counter: Counter,
    healings_counter: Counter,
    stage_latency: HistogramVec,
}

impl PipelineMetrics {
    fn new() -> Result<Self> {
        let entropy_gauge = register_gauge!("niodoo_entropy_bits", "Current consciousness entropy")
            .map_err(Error::from)?;
        let latency_histogram = register_histogram!(HistogramOpts::new(
            "niodoo_latency_ms",
            "Pipeline latency in milliseconds",
        )
        .buckets(vec![50.0, 100.0, 150.0, 250.0, 500.0, 1000.0]))
        .map_err(Error::from)?;
        let rouge_gauge = register_gauge!(
            "niodoo_rouge_l",
            "ROUGE-L similarity between baseline and hybrid responses"
        )
        .map_err(Error::from)?;
        let threats_counter =
            register_counter!("niodoo_threat_cycles", "Threat detections").map_err(Error::from)?;
        let healings_counter = register_counter!("niodoo_healing_cycles", "Healing detections")
            .map_err(Error::from)?;
        let stage_latency = register_histogram_vec!(
            "niodoo_stage_latency_ms",
            "Latency per pipeline stage in milliseconds",
            &["stage"],
            vec![5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        )
        .map_err(Error::from)?;

        Ok(Self {
            entropy_gauge,
            latency_histogram,
            rouge_gauge,
            threats_counter,
            healings_counter,
            stage_latency,
        })
    }

    pub fn observe_cycle(
        &self,
        entropy: f64,
        latency_ms: f64,
        rouge: f64,
        is_threat: bool,
        is_healing: bool,
    ) {
        self.entropy_gauge.set(entropy);
        self.latency_histogram.observe(latency_ms);
        self.rouge_gauge.set(rouge);
        if is_threat {
            self.threats_counter.inc();
        }
        if is_healing {
            self.healings_counter.inc();
        }
    }

    pub fn record_stage_latency(&self, stage: &str, latency_ms: f64) {
        self.stage_latency
            .with_label_values(&[stage])
            .observe(latency_ms);
    }

    pub fn gather(&self) -> Result<String> {
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        TextEncoder::new().encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap_or_default())
    }
}

pub fn metrics() -> &'static PipelineMetrics {
    &METRICS
}

/// RCE metrics (β_meta, spectral gap, persistence entropy, spike counter)
#[derive(Clone)]
pub struct RceMetrics {
    beta_meta_current: Gauge,
    beta_meta_peak: Gauge,
    laplacian_spectral_gap: Gauge,
    persistence_entropy: Gauge,
    beta_meta_spikes_total: Counter,
}

impl RceMetrics {
    fn new() -> Result<Self> {
        let beta_meta_current = register_gauge!(
            "niodoo_rce_beta_meta_current",
            "Current β_meta value"
        )
        .map_err(Error::from)?;
        let beta_meta_peak = register_gauge!(
            "niodoo_rce_beta_meta_peak",
            "Peak β_meta in current session"
        )
        .map_err(Error::from)?;
        let laplacian_spectral_gap = register_gauge!(
            "niodoo_rce_laplacian_spectral_gap",
            "Latest Laplacian spectral gap proxy"
        )
        .map_err(Error::from)?;
        let persistence_entropy = register_gauge!(
            "niodoo_rce_persistence_entropy",
            "Latest persistence entropy"
        )
        .map_err(Error::from)?;
        let beta_meta_spikes_total = register_counter!(
            "niodoo_rce_beta_meta_spikes_total",
            "Total β_meta spikes (threshold crossings)"
        )
        .map_err(Error::from)?;

        Ok(Self {
            beta_meta_current,
            beta_meta_peak,
            laplacian_spectral_gap,
            persistence_entropy,
            beta_meta_spikes_total,
        })
    }

    pub fn record_beta_meta(&self, current: f64, peak: f64) {
        self.beta_meta_current.set(current);
        self.beta_meta_peak.set(peak);
    }

    pub fn record_spectral_gap(&self, gap: f64) {
        self.laplacian_spectral_gap.set(gap);
    }

    pub fn record_persistence_entropy(&self, entropy: f64) {
        self.persistence_entropy.set(entropy);
    }

    pub fn inc_spike(&self) {
        self.beta_meta_spikes_total.inc();
    }
}

static RCE_METRICS: Lazy<RceMetrics> = Lazy::new(|| {
    RceMetrics::new().unwrap_or_else(|e| {
        panic!(
            "Failed to initialize RCE metrics: {}. This is a critical infrastructure failure.",
            e
        );
    })
});

pub fn rce_metrics() -> &'static RceMetrics {
    &RCE_METRICS
}

#[derive(Clone)]
pub struct CacheMetrics {
    embedding_hits: Counter,
    embedding_misses: Counter,
    embedding_compression_ratio: Histogram,
    collapse_hits: Counter,
    collapse_misses: Counter,
    collapse_compression_ratio: Histogram,
    prefetch_jobs: Counter,
    prefetch_failures: Counter,
}

impl CacheMetrics {
    fn new() -> Result<Self> {
        let embedding_hits =
            register_counter!("niodoo_embedding_cache_hits_total", "Embedding cache hits")
                .map_err(Error::from)?;
        let embedding_misses = register_counter!(
            "niodoo_embedding_cache_misses_total",
            "Embedding cache misses"
        )
        .map_err(Error::from)?;
        let embedding_compression_ratio = register_histogram!(HistogramOpts::new(
            "niodoo_embedding_cache_compression_ratio",
            "Compression ratio for embedding cache entries"
        )
        .buckets(vec![0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1]))
        .map_err(Error::from)?;

        let collapse_hits =
            register_counter!("niodoo_collapse_cache_hits_total", "Collapse cache hits")
                .map_err(Error::from)?;
        let collapse_misses = register_counter!(
            "niodoo_collapse_cache_misses_total",
            "Collapse cache misses"
        )
        .map_err(Error::from)?;
        let collapse_compression_ratio = register_histogram!(HistogramOpts::new(
            "niodoo_collapse_cache_compression_ratio",
            "Compression ratio for collapse cache entries"
        )
        .buckets(vec![0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1]))
        .map_err(Error::from)?;

        let prefetch_jobs = register_counter!(
            "niodoo_cache_prefetch_jobs_total",
            "Number of cache prefetch jobs scheduled"
        )
        .map_err(Error::from)?;
        let prefetch_failures = register_counter!(
            "niodoo_cache_prefetch_failures_total",
            "Failed cache prefetch attempts"
        )
        .map_err(Error::from)?;

        Ok(Self {
            embedding_hits,
            embedding_misses,
            embedding_compression_ratio,
            collapse_hits,
            collapse_misses,
            collapse_compression_ratio,
            prefetch_jobs,
            prefetch_failures,
        })
    }

    pub fn record_embedding_hit(&self, compression_ratio: Option<f64>) {
        self.embedding_hits.inc();
        if let Some(ratio) = compression_ratio {
            self.observe_embedding_entry(ratio);
        }
    }

    pub fn record_embedding_miss(&self) {
        self.embedding_misses.inc();
    }

    pub fn observe_embedding_entry(&self, ratio: f64) {
        self.embedding_compression_ratio.observe(ratio);
    }

    pub fn record_collapse_hit(&self, compression_ratio: Option<f64>) {
        self.collapse_hits.inc();
        if let Some(ratio) = compression_ratio {
            self.observe_collapse_entry(ratio);
        }
    }

    pub fn record_collapse_miss(&self) {
        self.collapse_misses.inc();
    }

    pub fn observe_collapse_entry(&self, ratio: f64) {
        self.collapse_compression_ratio.observe(ratio);
    }

    pub fn record_prefetch_job(&self) {
        self.prefetch_jobs.inc();
    }

    pub fn record_prefetch_failure(&self) {
        self.prefetch_failures.inc();
    }
}

static CACHE_METRICS: Lazy<CacheMetrics> = Lazy::new(|| {
    CacheMetrics::new().unwrap_or_else(|e| {
        panic!(
            "Failed to initialize cache metrics: {}. This is a critical infrastructure failure.",
            e
        );
    })
});

pub fn cache_metrics() -> &'static CacheMetrics {
    &CACHE_METRICS
}

/// Tokenizer metrics instrumentation
pub struct TokenizerMetrics {
    promoted_per_cycle: Histogram,
    pruned_per_cycle: Histogram,
    promotion_duration_ms: Histogram,
    vocab_size_gauge: Gauge,
    oov_rate_gauge: Gauge,
}

impl TokenizerMetrics {
    fn new() -> Result<Self> {
        let promoted_per_cycle = register_histogram!(HistogramOpts::new(
            "tokenizer_promoted_tokens",
            "Tokens promoted during a promotion cycle",
        )
        .buckets(vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]))
        .map_err(Error::from)?;

        let pruned_per_cycle = register_histogram!(HistogramOpts::new(
            "tokenizer_pruned_tokens",
            "Tokens pruned during a promotion cycle",
        )
        .buckets(vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]))
        .map_err(Error::from)?;

        let promotion_duration_ms = register_histogram!(HistogramOpts::new(
            "tokenizer_promotion_duration_ms",
            "Promotion cycle duration in milliseconds",
        )
        .buckets(vec![5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1000.0]))
        .map_err(Error::from)?;

        let vocab_size_gauge = register_gauge!(
            "tokenizer_vocab_size",
            "Total vocabulary size (base + extended)"
        )
        .map_err(Error::from)?;

        let oov_rate_gauge =
            register_gauge!("tokenizer_oov_rate", "Estimated out-of-vocabulary rate")
                .map_err(Error::from)?;

        Ok(Self {
            promoted_per_cycle,
            pruned_per_cycle,
            promotion_duration_ms,
            vocab_size_gauge,
            oov_rate_gauge,
        })
    }

    pub fn record_promotion(&self, promoted: usize, pruned: usize, duration_ms: f64) {
        self.promoted_per_cycle.observe(promoted as f64);
        self.pruned_per_cycle.observe(pruned as f64);
        self.promotion_duration_ms.observe(duration_ms.max(0.0));
    }

    pub fn record(&self, vocab_size: f64, oov_rate: f64) {
        self.vocab_size_gauge.set(vocab_size.max(0.0));
        self.oov_rate_gauge.set(oov_rate.clamp(0.0, 1.0));
    }
}

static TOKENIZER_METRICS: Lazy<TokenizerMetrics> =
    Lazy::new(|| TokenizerMetrics::new().expect("failed to initialise tokenizer metrics"));

pub fn tokenizer_metrics() -> &'static TokenizerMetrics {
    &TOKENIZER_METRICS
}

/// Weighted Memory Metrics
#[derive(Clone)]
pub struct WeightedMemoryMetrics {
    /// Weight evolution update latency
    weight_update_latency_ms: Histogram,
    /// Discoveries per second
    discoveries_per_second: Gauge,
    /// Current weight evolution score
    weight_evolution_score: Gauge,
    /// Best weight evolution score
    weight_evolution_best_score: Gauge,
    /// Fitness score distribution
    fitness_score_distribution: Histogram,
    /// Topology update count
    topology_updates_counter: Counter,
    /// Consolidation throughput (memories per second)
    consolidation_throughput: Gauge,
    /// Beta 1 connectivity average
    beta_1_connectivity_avg: Gauge,
    /// Consonance score average
    consonance_score_avg: Gauge,
}

impl WeightedMemoryMetrics {
    fn new() -> Result<Self> {
        let weight_update_latency_ms = register_histogram!(HistogramOpts::new(
            "weighted_memory_weight_update_latency_ms",
            "Weight evolution update latency in milliseconds"
        )
        .buckets(vec![10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]))
        .map_err(Error::from)?;

        let discoveries_per_second = register_gauge!(
            "weighted_memory_discoveries_per_second",
            "Discovery throughput (discoveries per second)"
        )
        .map_err(Error::from)?;

        let weight_evolution_score = register_gauge!(
            "weighted_memory_evolution_score",
            "Current weight evolution score"
        )
        .map_err(Error::from)?;

        let weight_evolution_best_score = register_gauge!(
            "weighted_memory_evolution_best_score",
            "Best weight evolution score achieved"
        )
        .map_err(Error::from)?;

        let fitness_score_distribution = register_histogram!(HistogramOpts::new(
            "weighted_memory_fitness_score",
            "Distribution of memory fitness scores"
        )
        .buckets(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        .map_err(Error::from)?;

        let topology_updates_counter = register_counter!(
            "weighted_memory_topology_updates_total",
            "Total number of topology updates"
        )
        .map_err(Error::from)?;

        let consolidation_throughput = register_gauge!(
            "weighted_memory_consolidation_throughput",
            "Memory consolidation throughput (memories per second)"
        )
        .map_err(Error::from)?;

        let beta_1_connectivity_avg = register_gauge!(
            "weighted_memory_beta_1_connectivity_avg",
            "Average Betti β₁ connectivity score"
        )
        .map_err(Error::from)?;

        let consonance_score_avg = register_gauge!(
            "weighted_memory_consonance_score_avg",
            "Average consonance score"
        )
        .map_err(Error::from)?;

        Ok(Self {
            weight_update_latency_ms,
            discoveries_per_second,
            weight_evolution_score,
            weight_evolution_best_score,
            fitness_score_distribution,
            topology_updates_counter,
            consolidation_throughput,
            beta_1_connectivity_avg,
            consonance_score_avg,
        })
    }

    /// Record weight evolution update latency
    pub fn record_weight_update_latency(&self, latency_ms: f64) {
        self.weight_update_latency_ms.observe(latency_ms);
    }

    /// Record discovery throughput
    pub fn record_discovery_throughput(&self, discoveries_per_sec: f64) {
        self.discoveries_per_second.set(discoveries_per_sec);
    }

    /// Record weight evolution scores
    pub fn record_weight_evolution_scores(&self, current_score: f64, best_score: f64) {
        self.weight_evolution_score.set(current_score);
        self.weight_evolution_best_score.set(best_score);
    }

    /// Record fitness score
    pub fn record_fitness_score(&self, fitness: f32) {
        self.fitness_score_distribution.observe(fitness as f64);
    }

    /// Record topology update
    pub fn record_topology_update(&self) {
        self.topology_updates_counter.inc();
    }

    /// Record consolidation throughput
    pub fn record_consolidation_throughput(&self, throughput: f64) {
        self.consolidation_throughput.set(throughput);
    }

    /// Record topological features
    pub fn record_topological_features(&self, beta_1_avg: f32, consonance_avg: f32) {
        self.beta_1_connectivity_avg.set(beta_1_avg as f64);
        self.consonance_score_avg.set(consonance_avg as f64);
    }
}

pub fn weighted_memory_metrics() -> &'static WeightedMemoryMetrics {
    &WEIGHTED_MEMORY_METRICS
}

/// ERAG batch operation metrics (Phase 1.5)
#[derive(Clone)]
pub struct EragBatchMetrics {
    /// Batch size distribution
    batch_size_histogram: Histogram,
    /// Batch flush latency
    batch_flush_latency_ms: Histogram,
    /// Batch throughput (batches per second)
    batch_throughput: Gauge,
    /// Queued points count
    queued_points_gauge: Gauge,
    /// Batch flush count
    batch_flush_count: Counter,
    /// Batch flush failures
    batch_flush_failures: Counter,
    /// Points upserted via batches
    batched_points_total: Counter,
    /// Points upserted immediately (non-batched)
    immediate_points_total: Counter,
}

impl EragBatchMetrics {
    fn new() -> Result<Self> {
        let batch_size_histogram = register_histogram!(HistogramOpts::new(
            "erag_batch_size",
            "ERAG batch size distribution"
        )
        .buckets(vec![1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]))
        .map_err(Error::from)?;

        let batch_flush_latency_ms = register_histogram!(HistogramOpts::new(
            "erag_batch_flush_latency_ms",
            "ERAG batch flush latency in milliseconds"
        )
        .buckets(vec![10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]))
        .map_err(Error::from)?;

        let batch_throughput = register_gauge!(
            "erag_batch_throughput",
            "ERAG batch throughput (batches per second)"
        )
        .map_err(Error::from)?;

        let queued_points_gauge = register_gauge!(
            "erag_queued_points",
            "Current number of points queued for batch upsert"
        )
        .map_err(Error::from)?;

        let batch_flush_count =
            register_counter!("erag_batch_flush_total", "Total number of batch flushes")
                .map_err(Error::from)?;

        let batch_flush_failures = register_counter!(
            "erag_batch_flush_failures_total",
            "Total number of batch flush failures"
        )
        .map_err(Error::from)?;

        let batched_points_total = register_counter!(
            "erag_batched_points_total",
            "Total points upserted via batches"
        )
        .map_err(Error::from)?;

        let immediate_points_total = register_counter!(
            "erag_immediate_points_total",
            "Total points upserted immediately (non-batched)"
        )
        .map_err(Error::from)?;

        Ok(Self {
            batch_size_histogram,
            batch_flush_latency_ms,
            batch_throughput,
            queued_points_gauge,
            batch_flush_count,
            batch_flush_failures,
            batched_points_total,
            immediate_points_total,
        })
    }

    pub fn record_batch_flush(&self, batch_size: usize, latency_ms: f64) {
        self.batch_size_histogram.observe(batch_size as f64);
        self.batch_flush_latency_ms.observe(latency_ms);
        self.batch_flush_count.inc();
        self.batched_points_total.inc_by(batch_size as f64);
    }

    pub fn record_batch_flush_failure(&self) {
        self.batch_flush_failures.inc();
    }

    pub fn record_immediate_upsert(&self) {
        self.immediate_points_total.inc();
    }

    pub fn record_queued_points(&self, count: usize) {
        self.queued_points_gauge.set(count as f64);
    }

    pub fn record_throughput(&self, batches_per_sec: f64) {
        self.batch_throughput.set(batches_per_sec);
    }
}

static ERAG_BATCH_METRICS: Lazy<EragBatchMetrics> = Lazy::new(|| {
    EragBatchMetrics::new().unwrap_or_else(|e| {
        panic!(
            "Failed to initialize ERAG batch metrics: {}. This is a critical infrastructure failure.",
            e
        )
    })
});

pub fn erag_batch_metrics() -> &'static EragBatchMetrics {
    &ERAG_BATCH_METRICS
}

/// TCS Analyzer metrics (Phase 2.3)
#[derive(Clone)]
pub struct TCSAnalyzerMetrics {
    /// TCS computation latency
    tcs_computation_latency_ms: Histogram,
    /// Giotto computation latency
    giotto_latency_ms: Histogram,
    /// Rust computation latency
    rust_latency_ms: Histogram,
    /// Cache hit rate
    cache_hits: Counter,
    /// Cache misses
    cache_misses: Counter,
    /// Giotto success count
    giotto_successes: Counter,
    /// Giotto failures (validation or Python errors)
    giotto_failures: Counter,
    /// Giotto fallbacks (automatic fallback to Rust)
    giotto_fallbacks: Counter,
    /// Current consecutive giotto failures
    giotto_consecutive_failures: Gauge,
    /// Current consecutive giotto successes
    giotto_consecutive_successes: Gauge,
    /// Betti number distribution
    betti_0_distribution: Histogram,
    betti_1_distribution: Histogram,
    betti_2_distribution: Histogram,
}

impl TCSAnalyzerMetrics {
    fn new() -> Result<Self> {
        let tcs_computation_latency_ms = register_histogram!(HistogramOpts::new(
            "tcs_computation_latency_ms",
            "TCS topology computation latency in milliseconds"
        )
        .buckets(vec![10.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0]))
        .map_err(Error::from)?;

        let giotto_latency_ms = register_histogram!(HistogramOpts::new(
            "tcs_giotto_latency_ms",
            "Giotto-tda computation latency in milliseconds"
        )
        .buckets(vec![5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0]))
        .map_err(Error::from)?;

        let rust_latency_ms = register_histogram!(HistogramOpts::new(
            "tcs_rust_latency_ms",
            "Rust persistent homology computation latency in milliseconds"
        )
        .buckets(vec![50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0]))
        .map_err(Error::from)?;

        let cache_hits = register_counter!(
            "tcs_cache_hits_total",
            "Total topology cache hits"
        )
        .map_err(Error::from)?;

        let cache_misses = register_counter!(
            "tcs_cache_misses_total",
            "Total topology cache misses"
        )
        .map_err(Error::from)?;

        let giotto_successes = register_counter!(
            "tcs_giotto_successes_total",
            "Total successful giotto-tda computations"
        )
        .map_err(Error::from)?;

        let giotto_failures = register_counter!(
            "tcs_giotto_failures_total",
            "Total giotto-tda failures (validation or Python errors)"
        )
        .map_err(Error::from)?;

        let giotto_fallbacks = register_counter!(
            "tcs_giotto_fallbacks_total",
            "Total automatic fallbacks from giotto to Rust"
        )
        .map_err(Error::from)?;

        let giotto_consecutive_failures = register_gauge!(
            "tcs_giotto_consecutive_failures",
            "Current consecutive giotto-tda failures"
        )
        .map_err(Error::from)?;

        let giotto_consecutive_successes = register_gauge!(
            "tcs_giotto_consecutive_successes",
            "Current consecutive giotto-tda successes"
        )
        .map_err(Error::from)?;

        let betti_0_distribution = register_histogram!(HistogramOpts::new(
            "tcs_betti_0_distribution",
            "Distribution of Betti β₀ values"
        )
        .buckets(vec![0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]))
        .map_err(Error::from)?;

        let betti_1_distribution = register_histogram!(HistogramOpts::new(
            "tcs_betti_1_distribution",
            "Distribution of Betti β₁ values"
        )
        .buckets(vec![0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 20.0]))
        .map_err(Error::from)?;

        let betti_2_distribution = register_histogram!(HistogramOpts::new(
            "tcs_betti_2_distribution",
            "Distribution of Betti β₂ values"
        )
        .buckets(vec![0.0, 1.0, 2.0, 3.0, 5.0, 10.0]))
        .map_err(Error::from)?;

        Ok(Self {
            tcs_computation_latency_ms,
            giotto_latency_ms,
            rust_latency_ms,
            cache_hits,
            cache_misses,
            giotto_successes,
            giotto_failures,
            giotto_fallbacks,
            giotto_consecutive_failures,
            giotto_consecutive_successes,
            betti_0_distribution,
            betti_1_distribution,
            betti_2_distribution,
        })
    }

    pub fn record_computation_latency(&self, latency_ms: f64) {
        self.tcs_computation_latency_ms.observe(latency_ms);
    }

    pub fn record_giotto_latency(&self, latency_ms: f64) {
        self.giotto_latency_ms.observe(latency_ms);
    }

    pub fn record_rust_latency(&self, latency_ms: f64) {
        self.rust_latency_ms.observe(latency_ms);
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.inc();
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses.inc();
    }

    pub fn record_giotto_success(&self) {
        self.giotto_successes.inc();
    }

    pub fn record_giotto_failure(&self) {
        self.giotto_failures.inc();
    }

    pub fn record_giotto_fallback(&self) {
        self.giotto_fallbacks.inc();
    }

    pub fn record_giotto_stats(&self, failures: usize, successes: usize) {
        self.giotto_consecutive_failures.set(failures as f64);
        self.giotto_consecutive_successes.set(successes as f64);
    }

    pub fn record_betti_numbers(&self, betti: &[usize; 3]) {
        self.betti_0_distribution.observe(betti[0] as f64);
        self.betti_1_distribution.observe(betti[1] as f64);
        self.betti_2_distribution.observe(betti[2] as f64);
    }
}

static TCS_ANALYZER_METRICS: Lazy<TCSAnalyzerMetrics> = Lazy::new(|| {
    TCSAnalyzerMetrics::new().unwrap_or_else(|e| {
        panic!(
            "Failed to initialize TCS analyzer metrics: {}. This is a critical infrastructure failure.",
            e
        )
    })
});

pub fn tcs_analyzer_metrics() -> &'static TCSAnalyzerMetrics {
    &TCS_ANALYZER_METRICS
}

/// Phase 5.2: Curator Feedback Controller metrics
#[derive(Clone)]
pub struct CuratorFeedbackMetrics {
    /// Adaptive threshold value
    adaptive_threshold_gauge: Gauge,
    /// Quality trend (positive = improving, negative = degrading)
    quality_trend_gauge: Gauge,
    /// Recent quality average
    recent_quality_avg: Gauge,
    /// Learned rate (percentage of responses marked as learned)
    learned_rate_gauge: Gauge,
    /// Parameter adjustments applied
    parameter_adjustments_total: Counter,
    /// Temperature adjustments
    temperature_adjustments: Counter,
    /// Top-p adjustments
    top_p_adjustments: Counter,
    /// Retrieval top-k adjustments
    retrieval_top_k_adjustments: Counter,
}

impl CuratorFeedbackMetrics {
    fn new() -> Result<Self> {
        let adaptive_threshold_gauge = register_gauge!(
            "curator_feedback_adaptive_threshold",
            "Current adaptive quality threshold"
        )
        .map_err(Error::from)?;

        let quality_trend_gauge = register_gauge!(
            "curator_feedback_quality_trend",
            "Quality trend (positive = improving, negative = degrading)"
        )
        .map_err(Error::from)?;

        let recent_quality_avg = register_gauge!(
            "curator_feedback_recent_quality_avg",
            "Recent quality average"
        )
        .map_err(Error::from)?;

        let learned_rate_gauge = register_gauge!(
            "curator_feedback_learned_rate",
            "Percentage of responses marked as learned"
        )
        .map_err(Error::from)?;

        let parameter_adjustments_total = register_counter!(
            "curator_feedback_parameter_adjustments_total",
            "Total parameter adjustments applied"
        )
        .map_err(Error::from)?;

        let temperature_adjustments = register_counter!(
            "curator_feedback_temperature_adjustments_total",
            "Total temperature adjustments"
        )
        .map_err(Error::from)?;

        let top_p_adjustments = register_counter!(
            "curator_feedback_top_p_adjustments_total",
            "Total top_p adjustments"
        )
        .map_err(Error::from)?;

        let retrieval_top_k_adjustments = register_counter!(
            "curator_feedback_retrieval_top_k_adjustments_total",
            "Total retrieval_top_k adjustments"
        )
        .map_err(Error::from)?;

        Ok(Self {
            adaptive_threshold_gauge,
            quality_trend_gauge,
            recent_quality_avg,
            learned_rate_gauge,
            parameter_adjustments_total,
            temperature_adjustments,
            top_p_adjustments,
            retrieval_top_k_adjustments,
        })
    }

    pub fn record_feedback(&self, adaptive_threshold: f32, quality_trend: f32, recent_quality: f32, learned_rate: f32) {
        self.adaptive_threshold_gauge.set(adaptive_threshold as f64);
        self.quality_trend_gauge.set(quality_trend as f64);
        self.recent_quality_avg.set(recent_quality as f64);
        self.learned_rate_gauge.set(learned_rate as f64);
    }

    pub fn record_parameter_adjustment(&self, param: &str) {
        self.parameter_adjustments_total.inc();
        match param {
            "temperature" => self.temperature_adjustments.inc(),
            "top_p" => self.top_p_adjustments.inc(),
            "retrieval_top_k" => self.retrieval_top_k_adjustments.inc(),
            _ => {}
        }
    }
}

static CURATOR_FEEDBACK_METRICS: Lazy<CuratorFeedbackMetrics> = Lazy::new(|| {
    CuratorFeedbackMetrics::new().unwrap_or_else(|e| {
        panic!(
            "Failed to initialize curator feedback metrics: {}. This is a critical infrastructure failure.",
            e
        )
    })
});

pub fn curator_feedback_metrics() -> &'static CuratorFeedbackMetrics {
    &CURATOR_FEEDBACK_METRICS
}

/// Phase 5.2: CRDT Consolidation metrics
#[derive(Clone)]
pub struct CrdtConsolidationMetrics {
    /// Merge operations counter
    merge_operations_total: Counter,
    /// Batch merge operations
    batch_merge_operations_total: Counter,
    /// Average batch size
    batch_size_histogram: Histogram,
    /// Merge latency
    merge_latency_ms: Histogram,
    /// Vector clock updates
    vector_clock_updates_total: Counter,
}

impl CrdtConsolidationMetrics {
    fn new() -> Result<Self> {
        let merge_operations_total = register_counter!(
            "crdt_consolidation_merge_operations_total",
            "Total CRDT merge operations"
        )
        .map_err(Error::from)?;

        let batch_merge_operations_total = register_counter!(
            "crdt_consolidation_batch_merge_operations_total",
            "Total batch CRDT merge operations"
        )
        .map_err(Error::from)?;

        let batch_size_histogram = register_histogram!(HistogramOpts::new(
            "crdt_consolidation_batch_size",
            "CRDT consolidation batch size"
        )
        .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]))
        .map_err(Error::from)?;

        let merge_latency_ms = register_histogram!(HistogramOpts::new(
            "crdt_consolidation_merge_latency_ms",
            "CRDT consolidation merge latency in milliseconds"
        )
        .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]))
        .map_err(Error::from)?;

        let vector_clock_updates_total = register_counter!(
            "crdt_consolidation_vector_clock_updates_total",
            "Total vector clock updates"
        )
        .map_err(Error::from)?;

        Ok(Self {
            merge_operations_total,
            batch_merge_operations_total,
            batch_size_histogram,
            merge_latency_ms,
            vector_clock_updates_total,
        })
    }

    pub fn record_merge(&self, latency_ms: f64) {
        self.merge_operations_total.inc();
        self.merge_latency_ms.observe(latency_ms);
    }

    pub fn record_batch_merge(&self, batch_size: usize, latency_ms: f64) {
        self.batch_merge_operations_total.inc();
        self.batch_size_histogram.observe(batch_size as f64);
        self.merge_latency_ms.observe(latency_ms);
    }

    pub fn record_vector_clock_update(&self) {
        self.vector_clock_updates_total.inc();
    }
}

static CRDT_CONSOLIDATION_METRICS: Lazy<CrdtConsolidationMetrics> = Lazy::new(|| {
    CrdtConsolidationMetrics::new().unwrap_or_else(|e| {
        panic!(
            "Failed to initialize CRDT consolidation metrics: {}. This is a critical infrastructure failure.",
            e
        )
    })
});

pub fn crdt_consolidation_metrics() -> &'static CrdtConsolidationMetrics {
    &CRDT_CONSOLIDATION_METRICS
}

/// Phase 5.2: GPU Fitness metrics
#[derive(Clone)]
pub struct GPUFitnessMetrics {
    /// GPU fitness calculations
    gpu_fitness_calculations_total: Counter,
    /// CPU fallback calculations
    cpu_fallback_calculations_total: Counter,
    /// Batch size histogram
    batch_size_histogram: Histogram,
    /// Calculation latency
    calculation_latency_ms: Histogram,
    /// GPU availability gauge
    gpu_available_gauge: Gauge,
}

impl GPUFitnessMetrics {
    fn new() -> Result<Self> {
        let gpu_fitness_calculations_total = register_counter!(
            "gpu_fitness_calculations_total",
            "Total GPU fitness calculations"
        )
        .map_err(Error::from)?;

        let cpu_fallback_calculations_total = register_counter!(
            "gpu_fitness_cpu_fallback_calculations_total",
            "Total CPU fallback calculations (GPU unavailable)"
        )
        .map_err(Error::from)?;

        let batch_size_histogram = register_histogram!(HistogramOpts::new(
            "gpu_fitness_batch_size",
            "GPU fitness calculation batch size"
        )
        .buckets(vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]))
        .map_err(Error::from)?;

        let calculation_latency_ms = register_histogram!(HistogramOpts::new(
            "gpu_fitness_calculation_latency_ms",
            "GPU fitness calculation latency in milliseconds"
        )
        .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]))
        .map_err(Error::from)?;

        let gpu_available_gauge = register_gauge!(
            "gpu_fitness_gpu_available",
            "Whether GPU is available for fitness calculations"
        )
        .map_err(Error::from)?;

        Ok(Self {
            gpu_fitness_calculations_total,
            cpu_fallback_calculations_total,
            batch_size_histogram,
            calculation_latency_ms,
            gpu_available_gauge,
        })
    }

    pub fn record_gpu_calculation(&self, batch_size: usize, latency_ms: f64) {
        self.gpu_fitness_calculations_total.inc();
        self.batch_size_histogram.observe(batch_size as f64);
        self.calculation_latency_ms.observe(latency_ms);
    }

    pub fn record_cpu_fallback(&self, batch_size: usize, latency_ms: f64) {
        self.cpu_fallback_calculations_total.inc();
        self.batch_size_histogram.observe(batch_size as f64);
        self.calculation_latency_ms.observe(latency_ms);
    }

    pub fn set_gpu_available(&self, available: bool) {
        self.gpu_available_gauge.set(if available { 1.0 } else { 0.0 });
    }
}

static GPU_FITNESS_METRICS: Lazy<GPUFitnessMetrics> = Lazy::new(|| {
    GPUFitnessMetrics::new().unwrap_or_else(|e| {
        panic!(
            "Failed to initialize GPU fitness metrics: {}. This is a critical infrastructure failure.",
            e
        )
    })
});

pub fn gpu_fitness_metrics() -> &'static GPUFitnessMetrics {
    &GPU_FITNESS_METRICS
}
