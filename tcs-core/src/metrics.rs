//! Prometheus metrics for TCS monitoring.

use std::sync::Arc;

use once_cell::sync::Lazy;
use prometheus::{CounterVec, Gauge, GaugeVec, HistogramOpts, HistogramVec, Opts, Registry};

use crate::topology::PersistenceResult;

#[derive(Debug)]
struct MetricsHandles {
    registry: Arc<Registry>,
    entropy_gauge: GaugeVec,
    persistence_entropy: GaugeVec,
    betti_0: Gauge,
    betti_1: Gauge,
    prompt_counter: CounterVec,
    output_hist: HistogramVec,
    output_var: HistogramVec,
    memory_counter: CounterVec,
    memory_gauge: GaugeVec,
    rag_hist: HistogramVec,
    rag_counter: CounterVec,
    rag_similarity: HistogramVec,
    llm_counter: CounterVec,
    learning_entropy_delta: GaugeVec,
}

impl MetricsHandles {
    fn record_topology(&self, result: &PersistenceResult) {
        for (dimension, value) in &result.entropy {
            self.persistence_entropy
                .with_label_values(&[&dimension.to_string()])
                .set(f64::from(*value));
        }

        // Preserve the legacy entropy gauge by publishing the aggregate entropy value.
        let aggregate_entropy: f32 = result.entropy.iter().map(|(_, v)| *v).sum();
        self.entropy_gauge
            .with_label_values(&["persistence"])
            .set(f64::from(aggregate_entropy));

        let betti0 = betti_value(result, 0);
        let betti1 = betti_value(result, 1);
        self.betti_0.set(betti0 as f64);
        self.betti_1.set(betti1 as f64);
    }
}

static METRICS: Lazy<MetricsHandles> = Lazy::new(|| {
    let registry = Arc::new(Registry::new());

    let entropy_gauge = register_gauge_vec(
        &registry,
        "tcs_entropy",
        "Current persistence entropy",
        &["component"],
    );
    let persistence_entropy = register_gauge_vec(
        &registry,
        "tcs_persistence_entropy",
        "Persistent entropy by homology dimension",
        &["dimension"],
    );
    let betti_0 = register_gauge(&registry, "tcs_betti_0", "Betti-0 (connected components)");
    let betti_1 = register_gauge(&registry, "tcs_betti_1", "Betti-1 (loops)");
    let prompt_counter = register_counter_vec(
        &registry,
        "tcs_prompts_total",
        "Prompt status counts",
        &["type"],
    );
    let output_hist = register_histogram_vec(
        &registry,
        "tcs_output_duration_seconds",
        "Output processing time",
        &["type"],
    );
    let output_var = register_histogram_vec(
        &registry,
        "tcs_output_var",
        "Output variance histogram",
        &["component"],
    );
    let memory_counter = register_counter_vec(
        &registry,
        "tcs_memories_saved_total",
        "Memory save counter",
        &["type"],
    );
    let memory_gauge = register_gauge_vec(
        &registry,
        "tcs_memories_size_bytes",
        "Memory storage size",
        &["type"],
    );
    let rag_hist = register_histogram_vec(
        &registry,
        "tcs_rag_latency_seconds",
        "RAG retrieval latency",
        &["success"],
    );
    let rag_counter = register_counter_vec(
        &registry,
        "tcs_rag_hits_total",
        "RAG retrieval hits",
        &["component"],
    );
    let rag_similarity = register_histogram_vec(
        &registry,
        "tcs_rag_similarity",
        "RAG similarity scores",
        &["component"],
    );
    let llm_counter = register_counter_vec(
        &registry,
        "tcs_llm_prompts_total",
        "LLM prompt counter",
        &["type"],
    );
    let learning_entropy_delta = register_gauge_vec(
        &registry,
        "tcs_learning_entropy_delta",
        "Entropy delta over epochs",
        &["component"],
    );

    MetricsHandles {
        registry,
        entropy_gauge,
        persistence_entropy,
        betti_0,
        betti_1,
        prompt_counter,
        output_hist,
        output_var,
        memory_counter,
        memory_gauge,
        rag_hist,
        rag_counter,
        rag_similarity,
        llm_counter,
        learning_entropy_delta,
    }
});

pub fn init_metrics() {
    Lazy::force(&METRICS);
}

pub fn get_registry() -> Arc<Registry> {
    METRICS.registry.clone()
}

pub fn record_topology_metrics(result: &PersistenceResult) {
    Lazy::force(&METRICS);
    METRICS.record_topology(result);
}

fn betti_value(result: &PersistenceResult, dimension: usize) -> usize {
    result
        .betti_curves
        .iter()
        .find(|curve| curve.dimension == dimension)
        .and_then(|curve| curve.samples.last().map(|(_, value)| *value))
        .unwrap_or_else(|| {
            result
                .diagram(dimension)
                .map(|diagram| {
                    diagram
                        .features
                        .iter()
                        .filter(|feature| feature.is_infinite())
                        .count()
                })
                .unwrap_or(0)
        })
}

fn register_gauge_vec(
    registry: &Arc<Registry>,
    name: &str,
    help: &str,
    labels: &[&str],
) -> GaugeVec {
    let gauge = GaugeVec::new(Opts::new(name, help), labels).expect("failed to create gauge vec");
    registry
        .register(Box::new(gauge.clone()))
        .expect("failed to register gauge vec");
    gauge
}

fn register_gauge(registry: &Arc<Registry>, name: &str, help: &str) -> Gauge {
    let gauge = Gauge::with_opts(Opts::new(name, help)).expect("failed to create gauge");
    registry
        .register(Box::new(gauge.clone()))
        .expect("failed to register gauge");
    gauge
}

fn register_counter_vec(
    registry: &Arc<Registry>,
    name: &str,
    help: &str,
    labels: &[&str],
) -> CounterVec {
    let counter =
        CounterVec::new(Opts::new(name, help), labels).expect("failed to create counter vec");
    registry
        .register(Box::new(counter.clone()))
        .expect("failed to register counter vec");
    counter
}

fn register_histogram_vec(
    registry: &Arc<Registry>,
    name: &str,
    help: &str,
    labels: &[&str],
) -> HistogramVec {
    let opts = HistogramOpts::new(name, help);
    let histogram = HistogramVec::new(opts, labels).expect("failed to create histogram vec");
    registry
        .register(Box::new(histogram.clone()))
        .expect("failed to register histogram vec");
    histogram
}
