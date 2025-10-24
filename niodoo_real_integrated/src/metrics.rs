use anyhow::{Error, Result};
use once_cell::sync::Lazy;
use prometheus::{
    register_counter, register_gauge, register_histogram, Counter, Encoder, Gauge, Histogram,
    HistogramOpts, TextEncoder,
};

static METRICS: Lazy<PipelineMetrics> =
    Lazy::new(|| PipelineMetrics::new().expect("failed to initialise Prometheus metrics"));

#[derive(Clone)]
pub struct PipelineMetrics {
    entropy_gauge: Gauge,
    latency_histogram: Histogram,
    rouge_gauge: Gauge,
    threats_counter: Counter,
    healings_counter: Counter,
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

        Ok(Self {
            entropy_gauge,
            latency_histogram,
            rouge_gauge,
            threats_counter,
            healings_counter,
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

pub fn evaluate_failure(
    rouge: f64,
    entropy_delta: f64,
    curator: f64,
    ucb1: f64,
) -> (String, String) {
    if rouge < 0.5 || entropy_delta > 0.1 || curator < 0.7 {
        (
            "hard".to_string(),
            "Low quality or high uncertainty".to_string(),
        )
    } else if ucb1 < 0.3 {
        ("soft".to_string(), "Low search confidence".to_string())
    } else {
        ("none".to_string(), "".to_string())
    }
}

pub struct FailureSignals;

impl FailureSignals {
    pub fn evaluate(rouge: f64, delta: f64, curator: f64, ucb1: f64) -> (String, String) {
        evaluate_failure(rouge, delta, curator, ucb1)
    }
}
