# Design Document: Silicon Synapse Hardware Monitoring System

## Project Context

### Niodoo-Feeling Project Overview

The Niodoo-Feeling project is an AI research initiative focused on simulating consciousness using geometric, topological, and machine learning concepts. It's primarily written in Rust, with integrations for ML (Candle), visualization (Qt), and Python components. The core demonstrates mathematical models for thought processes, including hyperbolic geometry, continuous attractor networks, and topological data analysis. The project is structured as a Cargo workspace with multiple crates, emphasizing ethical AI and consciousness modeling.

**Key Project Characteristics:**
- **Primary Language:** Rust (with C++/Qt for UI, Python for ML utilities)
- **Core Technologies:** Candle ML framework, Qt/QML visualization, tokio async runtime
- **Architecture:** Modular Cargo workspace with consciousness simulation, brain models, RAG system, geometry modules
- **Current State:** Active development with functional demos, needs production hardening
- **Working Directory:** `/home/ruffian/Desktop/Projects/Niodoo-Feeling`

**Existing Components to Integrate With:**
- `src/consciousness.rs` - Consciousness state management
- `src/brain.rs` - Brain simulation models
- `src/geometry/` - Geometric consciousness models (hyperbolic, topological)
- `src/rag/` - Retrieval-Augmented Generation system
- `candle-feeling-core/` - ML inference core
- `qt-inference-engine/` - Visualization layer

## Overview

The Silicon Synapse is a Rust-native, multi-layered telemetry and observability system designed to monitor the Niodoo-Feeling Gen 1 AI consciousness engine. It establishes a comprehensive monitoring pipeline that captures hardware metrics, inference performance, and model internal states, exposing them through a Prometheus-compatible interface for visualization in Grafana and analysis by anomaly detection algorithms.

The system is architected as a non-invasive observer that integrates with the existing consciousness simulation without introducing architectural dependencies or performance bottlenecks. It follows the principle of "fail-safe" operation: monitoring failures must never cascade to the core inference system.

**Core Design Philosophy:**
- Rust-first implementation for performance and safety (aligns with project's Rust-centric approach)
- Zero-copy metric collection where possible
- Asynchronous telemetry to avoid blocking inference
- Pluggable architecture for extensibility
- Hardware-agnostic abstractions with platform-specific implementations
- Integration with existing Cargo workspace structure


## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Niodoo Consciousness Engine                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Dual Möbius  │  │ EchoMemoria  │  │   Inference  │          │
│  │   Gaussian   │  │    Memory    │  │   Pipeline   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                     │
│                    ┌───────▼────────┐                           │
│                    │  Telemetry Bus │ (async channel)           │
│                    └───────┬────────┘                           │
└────────────────────────────┼──────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │    Silicon Synapse Monitoring System    │
        │                                          │
        │  ┌──────────────────────────────────────────────┐  │
        │  │        Metric Collection Layer            │  │
        │  │  ┌────────┐ ┌────────┐ ┌──────┐ ┌──────┐ │  │
        │  │  │Hardware│ │Inference│ │Model │ │Temporal│ │
        │  │  │Collector│ │Collector│ │Probe │ │Percept.│ │
        │  │  └───┬────┘ └───┬────┘ └──┬───┘ └──┬───┘ │  │
        │  └──────┼──────────┼─────────┼────────┼─────┘  │
        │         │          │         │        │         │
        │         │          │         │    ┌───▼────┐    │
        │         │          │         │    │  Git   │    │
        │         │          │         │    │Monitor │    │
        │         │          │         │    └────────┘    │
        │  ┌──────▼──────────▼─────────▼─────┐  │
        │  │    Metric Aggregation Engine    │  │
        │  │  (time-series buffer, reducer)  │  │
        │  └──────┬──────────────────────────┘  │
        │         │                              │
        │  ┌──────▼──────────────────────────┐  │
        │  │   Baseline & Anomaly Detector   │  │
        │  │  (statistical models, ML algos) │  │
        │  └──────┬──────────────────────────┘  │
        │         │                              │
        │  ┌──────▼──────────────────────────┐  │
        │  │   Prometheus Exporter (HTTP)    │  │
        │  │      /metrics endpoint          │  │
        │  └──────┬──────────────────────────┘  │
        └─────────┼───────────────────────────────┘
                  │
         ┌────────▼────────┐
         │   Prometheus    │ (scrapes every 15s)
         │   Time-Series   │
         │     Database    │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │     Grafana     │
         │   Dashboards    │
         └─────────────────┘
```

### Component Breakdown


#### 1. Telemetry Bus

**Purpose:** Decouples metric production from collection, ensuring the consciousness engine never blocks on monitoring.

**Implementation:**
- Rust `tokio::sync::mpsc` unbounded channel
- Consciousness components send `TelemetryEvent` messages asynchronously
- If channel is full, events are dropped (monitoring never blocks inference)
- Dedicated tokio task consumes events and routes to collectors

**Data Structure:**
```rust
pub enum TelemetryEvent {
    InferenceStart { request_id: Uuid, timestamp: Instant },
    TokenGenerated { request_id: Uuid, token_id: u32, logits: Option<Vec<f32>> },
    InferenceComplete { request_id: Uuid, total_tokens: usize },
    ConsciousnessStateUpdate { state: ConsciousnessState },
    LayerActivation { layer_name: String, activations: Arc<Tensor> },
}
```

#### 2. Hardware Collector

**Purpose:** Interfaces with GPU and system hardware to extract physical metrics.

**Architecture:**
- Trait-based abstraction: `HardwareMonitor` trait
- Platform-specific implementations: `NvidiaMonitor`, `AmdMonitor`, `CpuMonitor`
- Polling-based collection on a separate thread (not async, as FFI calls may block)
- Sampling rate: configurable, default 1Hz for temperature/power, 10Hz for utilization

**Rust Dependencies:**
- `nvml-wrapper` crate for NVIDIA GPU monitoring
- `sysinfo` crate for cross-platform CPU/RAM monitoring
- Custom FFI bindings to ROCm SMI for AMD GPUs

**Key Trait:**
```rust
pub trait HardwareMonitor: Send + Sync {
    fn collect_metrics(&self) -> Result<HardwareMetrics, MonitorError>;
    fn device_name(&self) -> &str;
    fn is_available(&self) -> bool;
}

pub struct HardwareMetrics {
    pub timestamp: SystemTime,
    pub gpu_temp_celsius: Option<f32>,
    pub gpu_power_watts: Option<f32>,
    pub gpu_fan_speed_percent: Option<f32>,
    pub vram_used_bytes: Option<u64>,
    pub vram_total_bytes: Option<u64>,
    pub gpu_utilization_percent: Option<f32>,
    pub cpu_utilization_percent: f32,
    pub ram_used_bytes: u64,
}
```


#### 3. Inference Collector

**Purpose:** Tracks performance characteristics of the inference pipeline.

**Implementation:**
- Hooks into existing inference pipeline via instrumentation points
- Maintains per-request state in a `DashMap<Uuid, RequestMetrics>`
- Calculates TTFT, TPOT, throughput in real-time
- Emits aggregated metrics every second

**Key Structure:**
```rust
pub struct InferenceCollector {
    active_requests: Arc<DashMap<Uuid, RequestState>>,
    throughput_counter: Arc<AtomicU64>,
    metrics_tx: mpsc::Sender<InferenceMetrics>,
}

struct RequestState {
    start_time: Instant,
    first_token_time: Option<Instant>,
    tokens_generated: usize,
    token_times: Vec<Duration>,
}

pub struct InferenceMetrics {
    pub timestamp: SystemTime,
    pub ttft_ms: Option<f32>,
    pub avg_tpot_ms: f32,
    pub throughput_tps: f32,
    pub active_requests: usize,
    pub error_count: u64,
}
```

**Integration Points:**
- `inference_pipeline::start_request()` → emit `InferenceStart`
- `inference_pipeline::generate_token()` → emit `TokenGenerated`
- `inference_pipeline::complete_request()` → emit `InferenceComplete`


#### 4. Model Probe

**Purpose:** Extracts internal model states (softmax entropy, activation patterns) during inference.

**Implementation:**
- Registers forward hooks on specified transformer layers
- Computes scalar metrics from high-dimensional tensors
- Uses `candle` tensor operations for GPU-accelerated reductions
- Configurable: which layers to probe, which metrics to compute

**Rust Dependencies:**
- `candle-core` for tensor operations
- Integration with existing Niodoo model architecture

**Key Components:**
```rust
pub struct ModelProbe {
    hooked_layers: Vec<String>,
    entropy_calculator: EntropyCalculator,
    activation_analyzer: ActivationAnalyzer,
}

pub struct ModelMetrics {
    pub timestamp: SystemTime,
    pub softmax_entropy: Option<f32>,  // 0-1 normalized
    pub activation_sparsity: HashMap<String, f32>,  // per layer
    pub activation_magnitude: HashMap<String, f32>, // L2 norm per layer
}

impl ModelProbe {
    pub fn register_hooks(&self, model: &mut TransformerModel) -> Result<()>;
    pub fn compute_entropy(&self, logits: &Tensor) -> Result<f32>;
    pub fn analyze_activations(&self, layer: &str, tensor: &Tensor) -> Result<(f32, f32)>;
}
```

**Entropy Calculation:**
```rust
// Shannon entropy: H(P) = -Σ p_i * log2(p_i)
// Normalized to [0, 1] by dividing by log2(vocab_size)
fn compute_entropy(logits: &Tensor) -> Result<f32> {
    let probs = softmax(logits, -1)?;
    let log_probs = probs.log()?;
    let entropy = -(probs * log_probs).sum(-1)?;
    let max_entropy = (logits.dim(-1)? as f32).log2();
    Ok(entropy.to_scalar::<f32>()? / max_entropy)
}
```


#### 5. Temporal Perception Collector

**Purpose:** Monitors the AI's ability to estimate task duration, detecting temporal perception distortions that cause user frustration.

**Implementation:**
- Captures AI-generated time estimates as telemetry events
- Monitors Git repository for task completion signals
- Calculates estimation error ratios (estimated / actual)
- Detects temporal anomalies using same framework as hardware/inference anomalies
- Computes calibration factors to correct systematic estimation bias

**Rust Dependencies:**
- `git2` crate for Git repository analysis
- `regex` for commit message parsing
- Integration with existing anomaly detection framework

**Key Components:**
```rust
pub struct TemporalPerceptionCollector {
    git_monitor: GitMonitor,
    active_estimates: Arc<DashMap<TaskId, EstimateRecord>>,
    calibration_model: Option<CalibrationModel>,
}

pub struct EstimateRecord {
    pub task_id: TaskId,
    pub estimated_duration: Duration,
    pub estimate_timestamp: SystemTime,
    pub task_context: TaskContext,
    pub ai_model_version: String,
}

pub struct TaskContext {
    pub code_complexity: Option<f32>,
    pub file_count: usize,
    pub issue_type: String,
    pub hardware_state: HardwareSnapshot,
}

pub struct CompletionRecord {
    pub task_id: TaskId,
    pub actual_duration: Duration,
    pub completion_timestamp: SystemTime,
    pub completion_method: CompletionMethod, // Git commit, PR merge, etc.
}

pub struct TemporalMetrics {
    pub timestamp: SystemTime,
    pub estimation_error_ratio: f32,  // estimated / actual
    pub is_anomaly: bool,
    pub anomaly_severity: Option<Severity>,
    pub calibration_factor: Option<f32>,
}

impl TemporalPerceptionCollector {
    pub fn record_estimate(&self, estimate: EstimateRecord) -> Result<()>;
    pub fn detect_completion(&self, repo_path: &Path) -> Result<Vec<CompletionRecord>>;
    pub fn calculate_error(&self, estimate: &EstimateRecord, completion: &CompletionRecord) -> f32;
    pub fn classify_temporal_anomaly(&self, error_ratio: f32) -> Option<(bool, Severity)>;
    pub fn compute_calibration_factor(&self, historical_errors: &[f32]) -> f32;
}
```

**Git Monitoring Strategy:**
```rust
// Detect task completion via commit analysis
fn detect_completion(repo: &Repository) -> Result<Vec<CompletionRecord>> {
    let commits = repo.head()?.peel_to_commit()?;
    let mut completions = Vec::new();
    
    for commit in commits.iter() {
        let message = commit.message().unwrap_or("");
        
        // Parse task IDs from commit messages
        // Patterns: "fixes #123", "closes TASK-456", "[DONE] feature-xyz"
        if let Some(task_id) = extract_task_id(message) {
            completions.push(CompletionRecord {
                task_id,
                actual_duration: calculate_duration_from_git_history(&task_id, repo)?,
                completion_timestamp: SystemTime::from(commit.time()),
                completion_method: CompletionMethod::GitCommit,
            });
        }
    }
    
    Ok(completions)
}
```

**Temporal Anomaly Classification:**
```rust
// Same anomaly detection philosophy as hardware/inference
fn classify_temporal_anomaly(error_ratio: f32) -> Option<(bool, Severity)> {
    match error_ratio {
        r if r < 0.5 => Some((true, Severity::Low)),      // Underestimate by 2x
        r if r > 2.0 && r < 5.0 => Some((true, Severity::Medium)),  // Overestimate 2-5x
        r if r > 5.0 && r < 10.0 => Some((true, Severity::High)),   // Overestimate 5-10x
        r if r >= 10.0 => Some((true, Severity::Critical)), // Overestimate 10x+ (cognitive failure)
        _ => None, // Within normal range (0.5x - 2x)
    }
}
```

**Calibration Model:**
```rust
// Learn systematic bias and compute correction factors
pub struct CalibrationModel {
    pub version: u32,
    pub sample_count: usize,
    pub mean_error_ratio: f64,
    pub median_error_ratio: f64,
    pub calibration_factor: f64,  // Multiply AI estimates by this
}

impl CalibrationModel {
    pub fn from_historical_data(errors: &[f32]) -> Self {
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        let mut sorted = errors.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        
        // Calibration factor: if AI consistently overestimates by 5x, factor = 0.2
        let calibration_factor = 1.0 / median as f64;
        
        Self {
            version: 1,
            sample_count: errors.len(),
            mean_error_ratio: mean as f64,
            median_error_ratio: median as f64,
            calibration_factor,
        }
    }
    
    pub fn apply_calibration(&self, raw_estimate: Duration) -> Duration {
        Duration::from_secs_f64(raw_estimate.as_secs_f64() * self.calibration_factor)
    }
}
```

#### 6. Metric Aggregation Engine

**Purpose:** Buffers, aligns, and reduces metrics from all collectors into time-windowed aggregates.

**Implementation:**
- Maintains ring buffers for each metric type
- Aligns metrics from different sources by timestamp
- Computes rolling statistics (mean, stddev, percentiles)
- Emits aggregated metrics at configurable intervals (1s, 10s, 1m)

**Key Structure:**
```rust
pub struct AggregationEngine {
    hardware_buffer: RingBuffer<HardwareMetrics>,
    inference_buffer: RingBuffer<InferenceMetrics>,
    model_buffer: RingBuffer<ModelMetrics>,
    window_size: Duration,
    aggregation_interval: Duration,
}

pub struct AggregatedMetrics {
    pub timestamp: SystemTime,
    pub window: Duration,
    pub hardware: HardwareStats,
    pub inference: InferenceStats,
    pub model: ModelStats,
}

pub struct HardwareStats {
    pub gpu_temp_mean: f32,
    pub gpu_temp_max: f32,
    pub gpu_power_mean: f32,
    pub vram_utilization_mean: f32,
    // ... other stats
}
```

**Aggregation Strategy:**
- Hardware metrics: mean, max, min over window
- Inference metrics: mean, p50, p95, p99 latencies
- Model metrics: mean entropy, max sparsity
- All metrics: sample count, data completeness percentage


#### 6. Baseline & Anomaly Detector

**Purpose:** Learns normal operational patterns and detects deviations in real-time.

**Architecture:**
- Two-phase operation: learning mode → detection mode
- Statistical baseline: Gaussian models per metric
- Multivariate baseline: correlation matrix between metrics
- Pluggable anomaly detection algorithms

**Baseline Model:**
```rust
pub struct BaselineModel {
    pub version: u32,
    pub created_at: SystemTime,
    pub sample_count: usize,
    pub univariate_stats: HashMap<String, MetricStats>,
    pub correlation_matrix: CorrelationMatrix,
}

pub struct MetricStats {
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: Percentiles,
}

impl BaselineModel {
    pub fn is_anomalous(&self, metric: &str, value: f64) -> bool {
        // 3-sigma rule: anomaly if |value - mean| > 3 * stddev
        let stats = &self.univariate_stats[metric];
        (value - stats.mean).abs() > 3.0 * stats.stddev
    }
    
    pub fn detect_multivariate_anomaly(&self, metrics: &AggregatedMetrics) -> Option<Anomaly>;
}
```

**Anomaly Detection Algorithms:**
1. **Statistical (3-sigma):** Fast, interpretable, good for univariate
2. **Isolation Forest:** Unsupervised, good for multivariate outliers
3. **Autoencoder:** Neural network-based, learns complex patterns (future enhancement)

**Anomaly Classification:**
```rust
pub enum AnomalyType {
    SecurityThreat,      // Unusual power/compute patterns
    ModelInstability,    // Repetitive loops, high entropy
    PerformanceDegradation, // Increasing latency over time
    EmergentBehavior,    // Novel activation patterns
}

pub struct Anomaly {
    pub id: Uuid,
    pub timestamp: SystemTime,
    pub anomaly_type: AnomalyType,
    pub severity: Severity,
    pub affected_metrics: Vec<String>,
    pub deviation_magnitude: f64,
    pub context: AggregatedMetrics,
}
```


#### 7. Prometheus Exporter

**Purpose:** Exposes all metrics via HTTP in Prometheus text format for scraping.

**Implementation:**
- Axum-based HTTP server on configurable port (default: 9090)
- `/metrics` endpoint returns Prometheus text format
- `/health` endpoint for liveness checks
- Metrics registry maintains current values

**Rust Dependencies:**
- `prometheus` crate for metric types and encoding
- `axum` for HTTP server
- `tokio` for async runtime

**Metric Types:**
```rust
// Gauges for instantaneous values
gpu_temperature_celsius{device="nvidia_0"} 72.5
gpu_power_watts{device="nvidia_0"} 245.3
vram_used_bytes{device="nvidia_0"} 8589934592

// Counters for cumulative values
tokens_generated_total{model="niodoo_v1"} 1523847

// Histograms for distributions
inference_ttft_milliseconds_bucket{le="100"} 45
inference_ttft_milliseconds_bucket{le="500"} 198
inference_ttft_milliseconds_bucket{le="+Inf"} 200
inference_ttft_milliseconds_sum 23456.7
inference_ttft_milliseconds_count 200

// Custom metrics
model_softmax_entropy{layer="output"} 0.342
activation_sparsity{layer="layer_12"} 0.87
```

**Exporter Structure:**
```rust
pub struct PrometheusExporter {
    registry: Registry,
    hardware_gauges: HardwareGauges,
    inference_histograms: InferenceHistograms,
    model_gauges: ModelGauges,
    anomaly_counter: IntCounter,
}

impl PrometheusExporter {
    pub async fn serve(self, addr: SocketAddr) -> Result<()>;
    pub fn update_metrics(&self, metrics: &AggregatedMetrics);
    pub fn record_anomaly(&self, anomaly: &Anomaly);
}
```


## Components and Interfaces

### Core Module Structure

```
src/silicon_synapse/
├── mod.rs                    # Public API, initialization
├── telemetry_bus.rs          # Event channel and routing
├── collectors/
│   ├── mod.rs
│   ├── hardware.rs           # HardwareMonitor trait + impls
│   ├── inference.rs          # InferenceCollector
│   └── model_probe.rs        # ModelProbe, hooks
├── aggregation.rs            # AggregationEngine
├── baseline/
│   ├── mod.rs
│   ├── model.rs              # BaselineModel, persistence
│   └── detector.rs           # Anomaly detection algorithms
├── exporters/
│   ├── mod.rs
│   ├── prometheus.rs         # PrometheusExporter
│   └── json.rs               # JSON API (future)
└── config.rs                 # Configuration structs
```

### Public API

```rust
// Main entry point
pub struct SiliconSynapse {
    config: Config,
    telemetry_bus: TelemetryBus,
    collectors: Vec<Box<dyn Collector>>,
    aggregator: AggregationEngine,
    detector: AnomalyDetector,
    exporter: PrometheusExporter,
}

impl SiliconSynapse {
    pub fn new(config: Config) -> Result<Self>;
    pub async fn start(&mut self) -> Result<()>;
    pub async fn shutdown(&mut self) -> Result<()>;
    pub fn telemetry_sender(&self) -> TelemetrySender;
}

// Integration with consciousness engine
pub trait TelemetryEmitter {
    fn emit(&self, event: TelemetryEvent);
}

// Implement for consciousness components
impl TelemetryEmitter for InferencePipeline {
    fn emit(&self, event: TelemetryEvent) {
        let _ = self.telemetry_tx.try_send(event);
    }
}
```

### Configuration Interface

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub enabled: bool,
    pub hardware: HardwareConfig,
    pub inference: InferenceConfig,
    pub model_probe: ModelProbeConfig,
    pub baseline: BaselineConfig,
    pub exporter: ExporterConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HardwareConfig {
    pub enabled: bool,
    pub sampling_rate_hz: f32,
    pub gpu_monitoring: bool,
    pub cpu_monitoring: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelProbeConfig {
    pub enabled: bool,
    pub compute_entropy: bool,
    pub hooked_layers: Vec<String>,
    pub activation_metrics: Vec<ActivationMetric>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BaselineConfig {
    pub learning_duration_hours: u64,
    pub auto_recalibrate: bool,
    pub anomaly_threshold_sigma: f64,
}
```


## Data Models

### Metric Flow Data Structures

```rust
// Raw telemetry events from consciousness engine
pub enum TelemetryEvent {
    InferenceStart {
        request_id: Uuid,
        timestamp: Instant,
        prompt_length: usize,
    },
    TokenGenerated {
        request_id: Uuid,
        token_id: u32,
        timestamp: Instant,
        logits: Option<Arc<Tensor>>,
    },
    InferenceComplete {
        request_id: Uuid,
        timestamp: Instant,
        total_tokens: usize,
        error: Option<String>,
    },
    ConsciousnessStateUpdate {
        timestamp: Instant,
        state: ConsciousnessState,
    },
    LayerActivation {
        request_id: Uuid,
        layer_name: String,
        timestamp: Instant,
        activations: Arc<Tensor>,
    },
}

// Collected metrics from each layer
pub struct CollectedMetrics {
    pub timestamp: SystemTime,
    pub hardware: Option<HardwareMetrics>,
    pub inference: Option<InferenceMetrics>,
    pub model: Option<ModelMetrics>,
}

// Aggregated metrics for export
pub struct AggregatedMetrics {
    pub timestamp: SystemTime,
    pub window_duration: Duration,
    pub sample_count: usize,
    
    // Hardware stats
    pub gpu_temp_mean: f32,
    pub gpu_temp_max: f32,
    pub gpu_power_mean: f32,
    pub vram_utilization_mean: f32,
    pub gpu_utilization_mean: f32,
    
    // Inference stats
    pub ttft_mean_ms: f32,
    pub ttft_p95_ms: f32,
    pub tpot_mean_ms: f32,
    pub tpot_p95_ms: f32,
    pub throughput_tps: f32,
    pub error_rate: f32,
    
    // Model stats
    pub entropy_mean: f32,
    pub entropy_max: f32,
    pub activation_sparsity: HashMap<String, f32>,
    pub activation_magnitude: HashMap<String, f32>,
}
```

### Baseline and Anomaly Models

```rust
// Persisted baseline model
#[derive(Serialize, Deserialize)]
pub struct BaselineModel {
    pub version: u32,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub sample_count: usize,
    pub metrics: HashMap<String, MetricBaseline>,
    pub correlations: Vec<MetricCorrelation>,
}

#[derive(Serialize, Deserialize)]
pub struct MetricBaseline {
    pub name: String,
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Serialize, Deserialize)]
pub struct MetricCorrelation {
    pub metric_a: String,
    pub metric_b: String,
    pub correlation_coefficient: f64,
    pub expected_ratio: Option<f64>,
}

// Detected anomaly
pub struct Anomaly {
    pub id: Uuid,
    pub timestamp: SystemTime,
    pub anomaly_type: AnomalyType,
    pub severity: Severity,
    pub affected_metrics: Vec<AffectedMetric>,
    pub description: String,
    pub context: Box<AggregatedMetrics>,
}

pub struct AffectedMetric {
    pub name: String,
    pub observed_value: f64,
    pub expected_value: f64,
    pub deviation_sigma: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Low,      // 3-4 sigma
    Medium,   // 4-5 sigma
    High,     // 5-6 sigma
    Critical, // >6 sigma or safety-critical pattern
}
```


## Error Handling

### Error Hierarchy

```rust
#[derive(Debug, thiserror::Error)]
pub enum SiliconSynapseError {
    #[error("Hardware monitoring error: {0}")]
    HardwareMonitor(#[from] HardwareMonitorError),
    
    #[error("Model probe error: {0}")]
    ModelProbe(#[from] ModelProbeError),
    
    #[error("Aggregation error: {0}")]
    Aggregation(String),
    
    #[error("Baseline error: {0}")]
    Baseline(#[from] BaselineError),
    
    #[error("Exporter error: {0}")]
    Exporter(#[from] ExporterError),
    
    #[error("Configuration error: {0}")]
    Config(String),
}

#[derive(Debug, thiserror::Error)]
pub enum HardwareMonitorError {
    #[error("GPU not available")]
    GpuNotAvailable,
    
    #[error("NVML initialization failed: {0}")]
    NvmlInit(String),
    
    #[error("ROCm SMI initialization failed: {0}")]
    RocmInit(String),
    
    #[error("Metric collection failed: {0}")]
    CollectionFailed(String),
}
```

### Error Handling Strategy

**Principle: Fail-Safe, Never Fail-Fatal**

1. **Hardware Monitor Failures:**
   - If GPU monitoring fails to initialize → log warning, continue with CPU-only monitoring
   - If a single metric collection fails → return partial metrics, log error
   - If all hardware monitoring fails → continue with inference/model metrics only

2. **Model Probe Failures:**
   - If hook registration fails → log error, continue without internal state monitoring
   - If tensor operation fails → skip that metric for this sample, log error
   - If entropy calculation fails → return None, continue with activation metrics

3. **Aggregation Failures:**
   - If buffer is empty → skip aggregation for this window
   - If statistical calculation fails → use fallback (e.g., last known value)

4. **Baseline Failures:**
   - If baseline file is corrupted → start fresh baseline learning
   - If anomaly detection fails → log error, continue metric collection

5. **Exporter Failures:**
   - If Prometheus server fails to start → log critical error, continue collecting metrics
   - If metric encoding fails → skip that metric, continue with others

**Logging Strategy:**
- Use `tracing` crate for structured logging
- Error level: failures that prevent a component from functioning
- Warn level: degraded functionality (e.g., partial metrics)
- Info level: normal operations (startup, baseline updates)
- Debug level: detailed metric values, anomaly detection reasoning
- Trace level: individual telemetry events


## Testing Strategy

### Unit Testing

**Hardware Collectors:**
- Mock GPU/CPU interfaces for deterministic testing
- Test metric extraction with known hardware states
- Test error handling (device not available, metric read failure)
- Test cross-platform compatibility (NVIDIA, AMD, CPU-only)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    struct MockGpuMonitor {
        temp: f32,
        power: f32,
        should_fail: bool,
    }
    
    impl HardwareMonitor for MockGpuMonitor {
        fn collect_metrics(&self) -> Result<HardwareMetrics> {
            if self.should_fail {
                return Err(HardwareMonitorError::CollectionFailed("mock".into()));
            }
            Ok(HardwareMetrics {
                gpu_temp_celsius: Some(self.temp),
                gpu_power_watts: Some(self.power),
                // ...
            })
        }
    }
    
    #[test]
    fn test_hardware_collector_success() {
        let monitor = MockGpuMonitor { temp: 70.0, power: 200.0, should_fail: false };
        let metrics = monitor.collect_metrics().unwrap();
        assert_eq!(metrics.gpu_temp_celsius, Some(70.0));
    }
    
    #[test]
    fn test_hardware_collector_failure_handling() {
        let monitor = MockGpuMonitor { temp: 0.0, power: 0.0, should_fail: true };
        assert!(monitor.collect_metrics().is_err());
    }
}
```

**Inference Collector:**
- Test TTFT/TPOT calculation with synthetic request timings
- Test throughput counter accuracy
- Test concurrent request tracking with DashMap
- Test error rate calculation

**Model Probe:**
- Test entropy calculation with known probability distributions
- Test activation sparsity calculation with synthetic tensors
- Test hook registration and cleanup
- Test tensor reduction operations

**Aggregation Engine:**
- Test ring buffer behavior (overflow, windowing)
- Test statistical calculations (mean, stddev, percentiles)
- Test timestamp alignment across metric sources
- Test aggregation interval timing

**Baseline & Anomaly Detector:**
- Test baseline learning with synthetic normal data
- Test anomaly detection with known outliers
- Test 3-sigma threshold calculation
- Test multivariate correlation detection
- Test baseline persistence and loading

### Integration Testing

**End-to-End Metric Flow:**
```rust
#[tokio::test]
async fn test_e2e_metric_flow() {
    let config = Config::default();
    let mut synapse = SiliconSynapse::new(config).unwrap();
    synapse.start().await.unwrap();
    
    let tx = synapse.telemetry_sender();
    
    // Simulate inference
    let request_id = Uuid::new_v4();
    tx.send(TelemetryEvent::InferenceStart {
        request_id,
        timestamp: Instant::now(),
        prompt_length: 100,
    }).await.unwrap();
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    tx.send(TelemetryEvent::TokenGenerated {
        request_id,
        token_id: 42,
        timestamp: Instant::now(),
        logits: None,
    }).await.unwrap();
    
    // Wait for aggregation
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Verify metrics are exported
    let response = reqwest::get("http://localhost:9090/metrics").await.unwrap();
    let body = response.text().await.unwrap();
    assert!(body.contains("inference_ttft_milliseconds"));
}
```

**Prometheus Integration:**
- Test metrics endpoint returns valid Prometheus format
- Test Prometheus can successfully scrape the endpoint
- Test metric labels are correctly applied
- Test histogram buckets are properly configured

**Grafana Integration:**
- Test Grafana can connect to Prometheus data source
- Test PromQL queries return expected data
- Test dashboard panels render correctly
- Test alerting rules trigger on anomalies

### Performance Testing

**Overhead Measurement:**
- Benchmark telemetry event emission latency (target: <1μs)
- Benchmark metric collection overhead (target: <5% CPU)
- Benchmark memory usage with 1000 concurrent requests
- Benchmark Prometheus scrape response time (target: <100ms)

**Load Testing:**
- Test with high inference throughput (1000 req/s)
- Test with long-running requests (1000+ tokens)
- Test with many concurrent requests (100+)
- Test baseline learning with large datasets (1M+ samples)

**Stress Testing:**
- Test behavior when telemetry channel is full
- Test behavior when Prometheus scraping is slow
- Test behavior when disk is full (baseline persistence)
- Test recovery after component failures


## Deployment and Operations

### Configuration File Format

**config/silicon_synapse.toml:**
```toml
[silicon_synapse]
enabled = true

[silicon_synapse.hardware]
enabled = true
sampling_rate_hz = 1.0
gpu_monitoring = true
cpu_monitoring = true

[silicon_synapse.inference]
enabled = true
track_per_request = true
throughput_window_seconds = 1

[silicon_synapse.model_probe]
enabled = true
compute_entropy = true
hooked_layers = ["layer_12", "layer_24", "output"]

[silicon_synapse.model_probe.activation_metrics]
sparsity = true
magnitude = true

[silicon_synapse.baseline]
learning_duration_hours = 24
auto_recalibrate = false
anomaly_threshold_sigma = 3.0
persistence_path = "./data/baseline.json"

[silicon_synapse.anomaly_detection]
enabled = true
algorithms = ["statistical", "isolation_forest"]

[silicon_synapse.exporter]
type = "prometheus"
bind_address = "0.0.0.0:9090"
metrics_path = "/metrics"
```

### Prometheus Configuration

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'niodoo-silicon-synapse'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
    scrape_timeout: 10s
```

### Grafana Dashboard Configuration

**Dashboard Structure:**

1. **System Overview Row:**
   - Single stat panels: Current GPU temp, power, VRAM usage
   - Gauge panels: GPU utilization, CPU utilization
   - Time series: All hardware metrics over time

2. **Inference Performance Row:**
   - Time series: TTFT and TPOT over time
   - Histogram: TPOT distribution
   - Single stat: Current throughput (TPS)
   - Time series: Active requests count

3. **Model Internal State Row:**
   - Time series: Softmax entropy over time
   - Heatmap: Activation sparsity per layer
   - Time series: Activation magnitude per layer

4. **Anomaly Detection Row:**
   - State timeline: System operational state
   - Table: Recent anomalies with severity
   - Time series: Anomaly count by type
   - Annotations: Anomaly events on all panels

**Example PromQL Queries:**
```promql
# GPU temperature
gpu_temperature_celsius{device="nvidia_0"}

# TTFT p95
histogram_quantile(0.95, rate(inference_ttft_milliseconds_bucket[5m]))

# Throughput
rate(tokens_generated_total[1m])

# Anomaly rate
rate(anomalies_detected_total[5m])

# VRAM utilization percentage
(vram_used_bytes / vram_total_bytes) * 100

# Correlation: power vs throughput
gpu_power_watts / rate(tokens_generated_total[1m])
```

### Alerting Rules

**prometheus_alerts.yml:**
```yaml
groups:
  - name: silicon_synapse_alerts
    interval: 30s
    rules:
      - alert: HighGPUTemperature
        expr: gpu_temperature_celsius > 85
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "GPU temperature is critically high"
          
      - alert: InferenceLatencyHigh
        expr: histogram_quantile(0.95, rate(inference_ttft_milliseconds_bucket[5m])) > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile TTFT exceeds 1 second"
          
      - alert: CriticalAnomaly
        expr: rate(anomalies_detected_total{severity="critical"}[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Critical anomaly detected in AI system"
```

### Operational Procedures

**Baseline Initialization:**
1. Deploy system with `baseline.learning_duration_hours = 24`
2. Run representative workload for 24 hours
3. Verify baseline file is created at configured path
4. Review baseline statistics in logs
5. Switch to detection mode

**Baseline Recalibration:**
1. Backup current baseline file
2. Set `auto_recalibrate = true` or manually delete baseline
3. Run new representative workload
4. Compare new baseline with old baseline
5. Validate anomaly detection with known test cases

**Anomaly Investigation:**
1. Check Grafana dashboard for anomaly timeline
2. Identify affected metrics and deviation magnitude
3. Query Prometheus for detailed metric history around anomaly time
4. Correlate with application logs and consciousness state
5. Determine if anomaly is benign or requires action

**Performance Tuning:**
1. Monitor telemetry channel saturation (dropped events)
2. Adjust sampling rates if overhead is too high
3. Reduce number of hooked layers if model probe is expensive
4. Tune aggregation window sizes for desired granularity
5. Optimize Prometheus retention and query performance


## Security Considerations

### Threat Model

**Threats Mitigated:**
1. **Adversarial Prompt Attacks:** Detect unusual computational patterns from malicious prompts
2. **Model Extraction Attacks:** Detect abnormal query patterns attempting to extract model weights
3. **Denial of Service:** Detect resource exhaustion attacks via anomalous power/memory usage
4. **Data Poisoning:** Detect drift in model behavior from poisoned training data
5. **Hardware Trojans:** Detect anomalous power signatures from compromised hardware

**Threats NOT Mitigated:**
- Direct attacks on the monitoring system itself (out of scope)
- Social engineering attacks
- Physical access attacks

### Security Best Practices

**Metrics Endpoint Security:**
- Bind to localhost by default, require explicit configuration for external access
- Implement authentication for Prometheus scraping (basic auth or mTLS)
- Rate limit scrape requests to prevent DoS
- Do not expose sensitive data in metric labels (e.g., prompt content)

**Baseline Model Security:**
- Store baseline files with restricted permissions (0600)
- Validate baseline file integrity on load (checksum)
- Encrypt baseline files if they contain sensitive distribution information
- Version baseline files to detect tampering

**Anomaly Alert Security:**
- Sanitize anomaly descriptions before logging (no PII)
- Implement alert rate limiting to prevent alert flooding
- Use secure channels for critical anomaly notifications (encrypted webhooks)
- Maintain audit log of all anomaly events

**Code Security:**
- Use `#![forbid(unsafe_code)]` where possible
- Validate all external inputs (config files, telemetry events)
- Use bounded channels to prevent memory exhaustion
- Implement timeouts for all blocking operations
- Regular dependency audits with `cargo audit`

### Privacy Considerations

**Data Minimization:**
- Do not log or export prompt content or generated text
- Do not log user identifiers in metrics
- Aggregate metrics before export (no per-request granularity in Prometheus)
- Implement data retention policies (auto-delete old baselines)

**Anonymization:**
- Use request_id (UUID) instead of user_id in telemetry
- Hash any identifiers before including in logs
- Redact sensitive information from error messages


## Performance Optimization

### Telemetry Event Emission

**Zero-Copy Design:**
- Use `Arc<Tensor>` for sharing activation tensors without cloning
- Use `Arc<str>` for shared string data (layer names, etc.)
- Avoid serialization in hot path

**Async Channel Tuning:**
- Use unbounded channel for telemetry bus (never block inference)
- Implement backpressure monitoring (track dropped events)
- Consider ring buffer for high-frequency events

**Batching:**
- Batch multiple telemetry events before processing
- Batch metric updates to Prometheus registry
- Batch baseline updates (don't persist on every sample)

### Hardware Monitoring Optimization

**Polling Strategy:**
- Use separate OS thread for hardware polling (not async)
- Cache GPU device handles (don't reinitialize on each poll)
- Use bulk query APIs where available (query all metrics in one call)
- Implement adaptive sampling (reduce rate when idle)

**NVML Optimization:**
```rust
// Cache device handle
struct NvidiaMonitor {
    device: nvml_wrapper::Device<'static>,
    // ...
}

// Bulk query
impl HardwareMonitor for NvidiaMonitor {
    fn collect_metrics(&self) -> Result<HardwareMetrics> {
        // Single NVML call for multiple metrics
        let utilization = self.device.utilization()?;
        let memory = self.device.memory_info()?;
        let temperature = self.device.temperature(TemperatureSensor::Gpu)?;
        let power = self.device.power_usage()?;
        
        Ok(HardwareMetrics {
            gpu_utilization_percent: Some(utilization.gpu as f32),
            vram_used_bytes: Some(memory.used),
            gpu_temp_celsius: Some(temperature as f32),
            gpu_power_watts: Some(power as f32 / 1000.0),
            // ...
        })
    }
}
```

### Model Probe Optimization

**Selective Hooking:**
- Only hook layers specified in config (not all layers)
- Use forward hooks, not backward hooks (cheaper)
- Detach tensors from computation graph immediately

**Tensor Reduction:**
- Perform reductions on GPU where possible
- Use optimized BLAS operations for norms
- Cache reduction results within aggregation window

**Entropy Calculation:**
```rust
// Optimized entropy calculation
fn compute_entropy_fast(logits: &Tensor) -> Result<f32> {
    // Use log_softmax for numerical stability
    let log_probs = logits.log_softmax(-1)?;
    let probs = log_probs.exp()?;
    
    // Compute entropy in single pass
    let entropy = -(probs * log_probs).sum(-1)?;
    
    // Normalize
    let vocab_size = logits.dim(-1)? as f32;
    Ok(entropy.to_scalar::<f32>()? / vocab_size.log2())
}
```

### Aggregation Optimization

**Lock-Free Data Structures:**
- Use `Arc<AtomicU64>` for counters
- Use `DashMap` for concurrent request tracking
- Use lock-free ring buffers where possible

**SIMD Vectorization:**
- Use SIMD for statistical calculations (mean, stddev)
- Leverage `std::simd` or `packed_simd` crates
- Batch metric calculations for vectorization

**Memory Management:**
- Pre-allocate ring buffers at startup
- Reuse metric objects (object pooling)
- Use `SmallVec` for small collections
- Implement custom allocator for hot paths

### Prometheus Exporter Optimization

**Metric Registry:**
- Use `LocalMetric` for thread-local metrics
- Batch metric updates before encoding
- Cache encoded metric strings between scrapes

**HTTP Server:**
- Use `axum` with `tower` middleware for efficiency
- Enable HTTP keep-alive
- Compress response with gzip if requested
- Implement metric scraping timeout

