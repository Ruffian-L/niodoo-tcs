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
