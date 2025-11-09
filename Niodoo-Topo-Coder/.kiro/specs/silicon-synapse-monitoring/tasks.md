# Implementation Plan: Silicon Synapse Hardware Monitoring System

- [ ] 1. Set up project structure and core module scaffolding
  - Create `src/silicon_synapse/` module directory structure
  - Define public API in `src/silicon_synapse/mod.rs`
  - Add Silicon Synapse as a workspace member in root `Cargo.toml`
  - Create configuration module with TOML deserialization
  - _Requirements: 10.1, 10.2_

- [ ] 2. Implement telemetry bus and event system
  - [ ] 2.1 Define `TelemetryEvent` enum with all event types
    - Create event variants for inference lifecycle (start, token, complete)
    - Create event variants for consciousness state updates
    - Create event variants for layer activations
    - Implement `Clone` and `Debug` traits for events
    - _Requirements: 1.10, 8.2_
  
  - [ ] 2.2 Implement `TelemetryBus` with async channel
    - Create unbounded mpsc channel using tokio
    - Implement sender/receiver wrapper types
    - Add event routing logic to dispatch to collectors
    - Implement dropped event counter for backpressure monitoring
    - Write unit tests for event routing and channel behavior
    - _Requirements: 1.10, 8.6_

- [ ] 3. Implement hardware monitoring layer
  - [ ] 3.1 Define `HardwareMonitor` trait and data structures
    - Create `HardwareMonitor` trait with `collect_metrics()` method
    - Define `HardwareMetrics` struct with all hardware fields
    - Define `HardwareMonitorError` enum for error handling
    - Implement `Send + Sync` bounds for thread safety
    - _Requirements: 1.1, 1.9_
  
  - [ ] 3.2 Implement NVIDIA GPU monitoring
    - Add `nvml-wrapper` crate dependency
    - Implement `NvidiaMonitor` struct with device handle caching
    - Implement GPU temperature collection via NVML
    - Implement power consumption collection
    - Implement fan speed collection
    - Implement VRAM utilization collection
    - Implement GPU compute utilization collection
    - Add error handling for device not available
    - Write unit tests with mock NVML interface
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6_
  
  - [ ] 3.3 Implement AMD GPU monitoring
    - Research and add ROCm SMI bindings (or use existing crate)
    - Implement `AmdMonitor` struct with ROCm SMI integration
    - Implement equivalent metrics collection for AMD GPUs
    - Add fallback behavior if ROCm is not available
    - Write unit tests for AMD monitoring
    - _Requirements: 1.1, 1.9_
  
  - [ ] 3.4 Implement CPU and system monitoring
    - Add `sysinfo` crate dependency
    - Implement `CpuMonitor` struct
    - Implement per-core and aggregate CPU utilization
    - Implement system RAM usage collection
    - Write unit tests for CPU monitoring
    - _Requirements: 1.7, 1.8_
  
  - [ ] 3.5 Implement hardware collector orchestration
    - Create `HardwareCollector` that manages all monitor implementations
    - Implement polling loop on dedicated OS thread
    - Implement configurable sampling rate
    - Add metrics emission to telemetry bus
    - Implement graceful degradation when monitors fail
    - Write integration tests for multi-monitor collection
    - _Requirements: 1.9, 8.5, 8.6_


- [ ] 4. Implement inference performance monitoring layer
  - [ ] 4.1 Create inference collector data structures
    - Define `InferenceCollector` struct with DashMap for request tracking
    - Define `RequestState` struct to track per-request timing
    - Define `InferenceMetrics` struct for aggregated metrics
    - Implement atomic counters for throughput tracking
    - _Requirements: 2.1, 2.10_
  
  - [ ] 4.2 Implement TTFT calculation
    - Hook into inference start event to record timestamp
    - Hook into first token generation to calculate TTFT
    - Emit TTFT metric to aggregation engine
    - Flag TTFT values exceeding 500ms threshold
    - Write unit tests with synthetic timing data
    - _Requirements: 2.2, 2.7_
  
  - [ ] 4.3 Implement TPOT calculation
    - Track timestamp for each token generation
    - Calculate per-token latency
    - Calculate average TPOT for completed requests
    - Flag TPOT values exceeding 50ms threshold
    - Write unit tests for TPOT calculation accuracy
    - _Requirements: 2.3, 2.4, 2.8_
  
  - [ ] 4.4 Implement throughput tracking
    - Maintain running counter of tokens per second
    - Implement windowed throughput calculation (1s, 10s, 1m)
    - Emit throughput metrics continuously
    - Write unit tests for throughput accuracy under load
    - _Requirements: 2.5, 2.10_
  
  - [ ] 4.5 Implement error tracking
    - Track inference failures and timeouts
    - Record error events with full context
    - Calculate error rate over time windows
    - Write unit tests for error rate calculation
    - _Requirements: 2.9_
  
  - [ ] 4.6 Integrate with existing inference pipeline
    - Identify instrumentation points in `src/` inference code
    - Add telemetry event emission at request start
    - Add telemetry event emission at token generation
    - Add telemetry event emission at request completion
    - Ensure zero-copy event emission (use Arc where needed)
    - Write integration tests with real inference pipeline
    - _Requirements: 2.6, 8.1, 8.6_

- [ ] 5. Implement model internal state probing
  - [ ] 5.1 Create model probe infrastructure
    - Define `ModelProbe` struct with layer hook management
    - Define `ModelMetrics` struct for internal state metrics
    - Create hook registration system for transformer layers
    - Implement hook cleanup on shutdown
    - _Requirements: 3.5, 3.9_
  
  - [ ] 5.2 Implement softmax entropy calculation
    - Enable `output_scores=True` in model generation
    - Capture raw logit tensors from generation
    - Implement softmax transformation using Candle ops
    - Implement Shannon entropy calculation
    - Normalize entropy to 0-1 range
    - Write unit tests with known probability distributions
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [ ] 5.3 Implement activation pattern analysis
    - Register forward hooks on specified transformer layers
    - Capture activation tensors during forward pass
    - Implement activation sparsity calculation (% near-zero)
    - Implement activation magnitude calculation (L2 norm)
    - Reduce tensors to scalar metrics efficiently
    - Write unit tests with synthetic activation tensors
    - _Requirements: 3.6, 3.7, 3.8_
  
  - [ ] 5.4 Integrate with Candle-based models
    - Identify hook points in `candle-feeling-core/` models
    - Implement Candle-compatible hook registration
    - Ensure GPU-accelerated tensor operations
    - Add configuration for which layers to probe
    - Write integration tests with actual model inference
    - _Requirements: 3.5, 8.1, 8.3_
  
  - [ ] 5.5 Implement fail-safe error handling
    - Add try-catch around all tensor operations
    - Return None for failed metric calculations
    - Log errors without crashing inference
    - Write tests for error scenarios (invalid tensors, OOM)
    - _Requirements: 3.9, 8.5_


- [ ] 6. Implement temporal perception monitoring layer
  - [ ] 6.1 Create temporal perception data structures
    - Define `TemporalPerceptionCollector` struct
    - Define `EstimateRecord` struct for AI time estimates
    - Define `CompletionRecord` struct for actual task completion
    - Define `TemporalMetrics` struct for estimation accuracy
    - Define `CalibrationModel` struct for bias correction
    - _Requirements: 11.1, 11.2_
  
  - [ ] 6.2 Implement estimate capture
    - Hook into AI estimation output to capture time estimates
    - Store estimates in DashMap with task identifiers
    - Include task context (complexity, file count, hardware state)
    - Emit estimate events to telemetry bus
    - Write unit tests for estimate recording
    - _Requirements: 11.1, 11.2, 11.9_
  
  - [ ] 6.3 Implement Git monitoring for completion detection
    - Add `git2` crate dependency
    - Implement Git repository monitoring
    - Parse commit messages for task identifiers
    - Detect completion patterns (fixes #, closes, [DONE])
    - Calculate actual duration from Git history
    - Write unit tests with mock Git repository
    - _Requirements: 11.3, 11.4_
  
  - [ ] 6.4 Implement estimation error calculation
    - Match completed tasks with stored estimates
    - Calculate error ratio (estimated / actual)
    - Classify temporal anomalies by severity
    - Flag errors exceeding 2x, 5x, 10x thresholds
    - Write unit tests for error calculation
    - _Requirements: 11.5, 11.6, 11.7, 11.8_
  
  - [ ] 6.5 Implement calibration model
    - Collect historical estimation errors
    - Compute mean and median error ratios
    - Calculate calibration factors
    - Apply calibration to new estimates
    - Persist calibration model to disk
    - Write unit tests for calibration accuracy
    - _Requirements: 11.15, 11.16, 11.18, 11.19_
  
  - [ ] 6.6 Implement temporal baseline learning
    - Establish baseline temporal perception accuracy
    - Define normal estimation error range
    - Detect temporal perception drift over time
    - Trigger recalibration on AI updates
    - Write tests for baseline learning
    - _Requirements: 11.10, 11.11, 11.12, 11.17_
  
  - [ ] 6.7 Integrate with anomaly detection framework
    - Emit temporal anomalies to anomaly detector
    - Use same severity classification as hardware/inference
    - Log temporal anomalies with full context
    - Export temporal metrics to Prometheus
    - Write integration tests with anomaly detector
    - _Requirements: 11.8, 11.9, 11.13_
  
  - [ ] 6.8 Implement Grafana temporal perception dashboard
    - Create "Temporal Perception" dashboard panel
    - Visualize estimation accuracy over time
    - Show estimation error distribution histogram
    - Display calibration factor trends
    - Add alerts for critical temporal anomalies
    - _Requirements: 11.14_
  
  - [ ] 6.9 Optimize temporal monitoring performance
    - Ensure non-blocking Git monitoring
    - Implement efficient task ID matching
    - Minimize overhead (<1% of estimation process)
    - Profile and benchmark temporal collector
    - _Requirements: 11.20_

- [ ] 7. Implement metric aggregation engine
  - [ ] 6.1 Create aggregation data structures
    - Define `AggregationEngine` struct with ring buffers
    - Define `AggregatedMetrics` struct with all statistical fields
    - Implement ring buffer for time-windowed storage
    - Add timestamp alignment logic
    - _Requirements: 2.10, 3.10_
  
  - [ ] 6.2 Implement statistical aggregation functions
    - Implement mean calculation over time windows
    - Implement standard deviation calculation
    - Implement percentile calculation (p50, p95, p99)
    - Implement min/max tracking
    - Write unit tests for statistical accuracy
    - _Requirements: 5.2, 5.3_
  
  - [ ] 6.3 Implement multi-layer metric correlation
    - Align hardware, inference, and model metrics by timestamp
    - Compute correlation coefficients between metric pairs
    - Identify expected metric relationships
    - Write unit tests for correlation calculation
    - _Requirements: 5.3, 5.9_
  
  - [ ] 6.4 Implement windowed aggregation
    - Create 1-second aggregation window
    - Create 10-second aggregation window
    - Create 1-minute aggregation window
    - Emit aggregated metrics at configured intervals
    - Write integration tests for multi-window aggregation
    - _Requirements: 2.10_
  
  - [ ] 6.5 Optimize aggregation performance
    - Use lock-free data structures where possible
    - Implement SIMD vectorization for statistical calculations
    - Pre-allocate buffers to avoid runtime allocation
    - Profile and benchmark aggregation overhead
    - _Requirements: 8.6_

- [ ] 7. Implement baseline learning and persistence
  - [ ] 7.1 Create baseline model data structures
    - Define `BaselineModel` struct with versioning
    - Define `MetricStats` struct for univariate statistics
    - Define `MetricCorrelation` struct for multivariate patterns
    - Implement Serialize/Deserialize for persistence
    - _Requirements: 5.1, 5.4_
  
  - [ ] 7.2 Implement baseline learning algorithm
    - Collect metrics during configurable learning period
    - Compute mean, stddev, percentiles for each metric
    - Compute correlation matrix between metrics
    - Identify normal ranges (mean ± 3σ)
    - Write unit tests with synthetic normal data
    - _Requirements: 5.2, 5.3, 5.8_
  
  - [ ] 7.3 Implement baseline persistence
    - Serialize baseline model to JSON file
    - Implement versioned baseline storage
    - Load baseline from disk on startup
    - Validate baseline integrity (checksum)
    - Handle corrupted baseline files gracefully
    - Write tests for save/load cycle
    - _Requirements: 5.4, 5.5, 5.10_
  
  - [ ] 7.4 Implement baseline recalibration
    - Support manual baseline reset
    - Implement auto-recalibration on workload change detection
    - Backup old baseline before recalibration
    - Write tests for recalibration scenarios
    - _Requirements: 5.7_
  
  - [ ] 7.5 Handle baseline initialization
    - Detect missing baseline on first run
    - Enter learning mode automatically
    - Log learning progress
    - Transition to detection mode after learning completes
    - Write integration tests for cold start
    - _Requirements: 5.6_


- [ ] 8. Implement anomaly detection system
  - [ ] 8.1 Create anomaly detection data structures
    - Define `Anomaly` struct with all required fields
    - Define `AnomalyType` enum (security, instability, degradation, emergent)
    - Define `Severity` enum (low, medium, high, critical)
    - Define `AffectedMetric` struct for deviation details
    - _Requirements: 6.4, 6.5, 6.6_
  
  - [ ] 8.2 Implement univariate anomaly detection
    - Compare each metric against baseline normal range
    - Calculate deviation in standard deviations (sigma)
    - Flag anomalies exceeding 3-sigma threshold
    - Assign severity based on deviation magnitude
    - Write unit tests with known outliers
    - _Requirements: 6.2, 6.4_
  
  - [ ] 8.3 Implement multivariate anomaly detection
    - Detect violations of expected metric correlations
    - Identify unusual metric combinations
    - Use correlation matrix from baseline
    - Write unit tests for multivariate patterns
    - _Requirements: 6.3_
  
  - [ ] 8.4 Implement anomaly classification
    - Classify power spikes without throughput as security threats
    - Classify repetitive loops as model instability
    - Classify increasing latency as performance degradation
    - Classify novel activation patterns as emergent behavior
    - Write unit tests for each classification rule
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [ ] 8.5 Implement anomaly persistence detection
    - Track anomaly duration over consecutive samples
    - Escalate severity for persistent anomalies
    - Log anomaly resolution events
    - Write tests for persistence tracking
    - _Requirements: 6.9, 6.10_
  
  - [ ] 8.6 Implement anomaly event logging
    - Create dedicated anomaly event stream
    - Log all anomalies with full context
    - Include affected metrics and deviation magnitude
    - Implement structured logging with tracing crate
    - Write tests for log output format
    - _Requirements: 6.7, 9.10_
  
  - [ ] 8.7 Implement alert generation
    - Generate alerts for high/critical severity anomalies
    - Include anomaly type, metrics, and context in alerts
    - Support configurable alert thresholds
    - Write tests for alert triggering logic
    - _Requirements: 6.5, 6.6_
  
  - [ ] 8.8 Implement false positive tuning
    - Add configurable threshold adjustment
    - Support per-metric threshold overrides
    - Log false positive rate for tuning
    - Write tests for threshold configuration
    - _Requirements: 6.8_

- [ ] 9. Implement Prometheus exporter
  - [ ] 9.1 Set up Prometheus metric registry
    - Add `prometheus` crate dependency
    - Create metric registry for all metric types
    - Define Gauge metrics for instantaneous values
    - Define Counter metrics for cumulative values
    - Define Histogram metrics for latency distributions
    - _Requirements: 4.3, 4.4, 4.5_
  
  - [ ] 9.2 Implement hardware metric gauges
    - Create gauges for GPU temperature, power, fan speed
    - Create gauges for VRAM utilization
    - Create gauges for GPU compute utilization
    - Create gauges for CPU and RAM metrics
    - Add device labels to all hardware metrics
    - _Requirements: 4.2, 4.6_
  
  - [ ] 9.3 Implement inference metric histograms
    - Create histogram for TTFT with appropriate buckets
    - Create histogram for TPOT with appropriate buckets
    - Create counter for total tokens generated
    - Create gauge for active request count
    - Add model name labels to inference metrics
    - _Requirements: 4.2, 4.5_
  
  - [ ] 9.4 Implement model internal state gauges
    - Create gauge for softmax entropy
    - Create gauges for activation sparsity per layer
    - Create gauges for activation magnitude per layer
    - Add layer name labels to model metrics
    - _Requirements: 4.2, 4.6_
  
  - [ ] 9.5 Implement anomaly counter
    - Create counter for total anomalies detected
    - Add labels for anomaly type and severity
    - Increment counter on each anomaly detection
    - _Requirements: 4.2_
  
  - [ ] 9.6 Create HTTP server with Axum
    - Add `axum` and `tokio` dependencies
    - Implement `/metrics` endpoint returning Prometheus text format
    - Implement `/health` endpoint for liveness checks
    - Configure server to bind on configurable address
    - Write integration tests for HTTP endpoints
    - _Requirements: 4.1, 4.7_
  
  - [ ] 9.7 Implement metric update logic
    - Subscribe to aggregated metrics stream
    - Update Prometheus registry on each aggregation
    - Handle metric update errors gracefully
    - Write tests for metric update flow
    - _Requirements: 4.2_
  
  - [ ] 9.8 Optimize exporter performance
    - Cache encoded metric strings between scrapes
    - Implement metric scraping timeout
    - Enable HTTP keep-alive
    - Profile and benchmark scrape response time
    - _Requirements: 4.10_


- [ ] 10. Implement Grafana dashboard integration
  - [ ] 10.1 Create Prometheus data source configuration
    - Document Prometheus connection setup in Grafana
    - Provide example prometheus.yml scrape config
    - Test Prometheus data source connectivity
    - _Requirements: 7.1_
  
  - [ ] 10.2 Design hardware metrics dashboard row
    - Create time series panel for GPU temperature over time
    - Create gauge panel for current GPU utilization
    - Create time series panel for power consumption
    - Create gauge panel for VRAM utilization percentage
    - Write PromQL queries for all hardware panels
    - _Requirements: 7.2, 7.6_
  
  - [ ] 10.3 Design inference performance dashboard row
    - Create time series panel for TTFT and TPOT
    - Create histogram panel for TPOT distribution
    - Create single stat panel for current throughput
    - Create time series panel for active request count
    - Write PromQL queries for all inference panels
    - _Requirements: 7.3, 7.6_
  
  - [ ] 10.4 Design model internal state dashboard row
    - Create time series panel for softmax entropy
    - Create heatmap panel for activation sparsity per layer
    - Create time series panel for activation magnitude
    - Write PromQL queries for all model panels
    - _Requirements: 7.4, 7.6, 7.7_
  
  - [ ] 10.5 Design anomaly detection dashboard row
    - Create state timeline panel for operational states
    - Create table panel for recent anomalies
    - Create time series panel for anomaly count by type
    - Add anomaly annotations to all time series panels
    - Write PromQL queries for anomaly visualization
    - _Requirements: 7.5, 7.8, 7.9_
  
  - [ ] 10.6 Configure dashboard time range controls
    - Enable time range selector (5m, 1h, 24h, 7d)
    - Configure auto-refresh intervals
    - Set up dashboard variables for filtering
    - _Requirements: 7.10_
  
  - [ ] 10.7 Export and document dashboard JSON
    - Export complete dashboard as JSON
    - Document dashboard import procedure
    - Provide dashboard screenshots in docs
    - _Requirements: 7.1_

- [ ] 11. Implement configuration system
  - [ ] 11.1 Define configuration schema
    - Create `Config` struct with all subsystem configs
    - Create `HardwareConfig`, `InferenceConfig`, `ModelProbeConfig` structs
    - Create `BaselineConfig`, `ExporterConfig` structs
    - Implement Deserialize trait for all config structs
    - _Requirements: 10.1, 10.5_
  
  - [ ] 11.2 Implement TOML configuration loading
    - Add `toml` crate dependency
    - Load config from `config/silicon_synapse.toml`
    - Validate configuration on load
    - Provide clear error messages for invalid config
    - Write unit tests for config parsing
    - _Requirements: 10.1, 10.2_
  
  - [ ] 11.3 Implement configuration defaults
    - Define sensible defaults for all config values
    - Document all configuration options
    - Provide example configuration file
    - _Requirements: 10.10_
  
  - [ ] 11.4 Implement environment-specific overrides
    - Support environment variable overrides
    - Support command-line argument overrides
    - Document override precedence
    - Write tests for override behavior
    - _Requirements: 10.9_
  
  - [ ] 11.5 Implement hot-reload support
    - Watch config file for changes
    - Reload configuration without restart
    - Apply new config to running components
    - Log configuration changes
    - Write tests for hot-reload scenarios
    - _Requirements: 10.7_

- [ ] 12. Implement safety response system
  - [ ] 12.1 Define safety response actions
    - Create `SafetyAction` enum (shutdown, reject, log, alert)
    - Map anomaly types to appropriate actions
    - Implement action execution logic
    - _Requirements: 9.6, 9.7, 9.8, 9.9_
  
  - [ ] 12.2 Implement emergency shutdown trigger
    - Create shutdown signal channel
    - Implement graceful inference pipeline shutdown
    - Log shutdown events with full context
    - Write tests for shutdown scenarios
    - _Requirements: 9.6_
  
  - [ ] 12.3 Implement request rejection logic
    - Integrate with inference pipeline request handler
    - Reject new requests during instability
    - Return appropriate error responses
    - Write tests for rejection behavior
    - _Requirements: 9.7_
  
  - [ ] 12.4 Implement diagnostic logging
    - Capture detailed system state on anomaly
    - Log hardware metrics, inference state, model state
    - Store diagnostic snapshots for post-mortem
    - Write tests for diagnostic capture
    - _Requirements: 9.8_
  
  - [ ] 12.5 Implement operator alerting
    - Create alert notification interface
    - Support webhook notifications for critical anomalies
    - Include full anomaly context in alerts
    - Write tests for alert delivery
    - _Requirements: 9.9_
  
  - [ ] 12.6 Implement audit trail
    - Log all safety actions with timestamps
    - Include triggering anomaly and context
    - Persist audit log to durable storage
    - Write tests for audit log completeness
    - _Requirements: 9.10_


- [ ] 13. Implement extensibility and plugin system
  - [ ] 13.1 Define collector plugin trait
    - Create `Collector` trait for pluggable metric collectors
    - Define lifecycle methods (init, collect, shutdown)
    - Implement dynamic collector registration
    - Write example custom collector
    - _Requirements: 10.3_
  
  - [ ] 13.2 Define detector plugin trait
    - Create `AnomalyDetector` trait for pluggable algorithms
    - Support multiple detector implementations
    - Implement detector chaining (run multiple detectors)
    - Write example custom detector (e.g., Isolation Forest)
    - _Requirements: 10.4_
  
  - [ ] 13.3 Define exporter plugin trait
    - Create `MetricExporter` trait for pluggable exporters
    - Support multiple simultaneous exporters
    - Implement JSON API exporter as example
    - _Requirements: 10.6_
  
  - [ ] 13.4 Implement custom metric registration
    - Allow runtime registration of custom metrics
    - Auto-expose custom metrics via Prometheus endpoint
    - Document custom metric API
    - Write tests for custom metric flow
    - _Requirements: 10.8_

- [ ] 14. Integration with existing Niodoo components
  - [ ] 14.1 Integrate with consciousness state system
    - Hook into `src/consciousness.rs` state updates
    - Emit telemetry events on consciousness state changes
    - Correlate consciousness states with hardware metrics
    - Write integration tests with consciousness module
    - _Requirements: 8.2, 8.3_
  
  - [ ] 14.2 Integrate with Dual Möbius Gaussian models
    - Hook into Möbius computation execution
    - Track computational characteristics of Möbius operations
    - Correlate Möbius states with performance metrics
    - Write integration tests with geometry modules
    - _Requirements: 8.3_
  
  - [ ] 14.3 Integrate with EchoMemoria memory system
    - Hook into memory operation events
    - Track memory retrieval and storage performance
    - Correlate memory operations with hardware load
    - Write integration tests with EchoMemoria
    - _Requirements: 8.4_
  
  - [ ] 14.4 Integrate with RAG system
    - Hook into RAG retrieval and generation pipeline
    - Track RAG-specific performance metrics
    - Monitor RAG knowledge base access patterns
    - Write integration tests with RAG modules
    - _Requirements: 8.4_
  
  - [ ] 14.5 Ensure non-invasive integration
    - Verify monitoring adds <5% overhead to inference
    - Ensure monitoring failures don't crash inference
    - Test graceful degradation scenarios
    - Profile and optimize integration points
    - _Requirements: 8.5, 8.6, 8.9_
  
  - [ ] 14.6 Create unified monitoring API
    - Expose monitoring data to other Niodoo components
    - Provide query interface for current metrics
    - Provide subscription interface for metric streams
    - Document API for component integration
    - _Requirements: 8.10_

- [ ] 15. Testing and validation
  - [ ] 15.1 Write comprehensive unit tests
    - Test all collector implementations
    - Test aggregation engine statistical functions
    - Test baseline learning and anomaly detection
    - Test configuration loading and validation
    - Achieve >80% code coverage
    - _Requirements: All_
  
  - [ ] 15.2 Write integration tests
    - Test end-to-end metric flow from emission to export
    - Test Prometheus scraping integration
    - Test multi-component interaction
    - Test error propagation and recovery
    - _Requirements: All_
  
  - [ ] 15.3 Write performance benchmarks
    - Benchmark telemetry event emission latency
    - Benchmark metric collection overhead
    - Benchmark aggregation engine throughput
    - Benchmark Prometheus scrape response time
    - Verify <5% overhead target
    - _Requirements: 8.6_
  
  - [ ] 15.4 Conduct load testing
    - Test with 1000 requests/second inference load
    - Test with 100+ concurrent requests
    - Test with long-running 1000+ token generations
    - Verify system stability under load
    - _Requirements: All_
  
  - [ ] 15.5 Conduct stress testing
    - Test telemetry channel saturation behavior
    - Test behavior when Prometheus is unavailable
    - Test disk full scenarios for baseline persistence
    - Test recovery after component crashes
    - _Requirements: All_
  
  - [ ] 15.6 Validate anomaly detection accuracy
    - Create synthetic anomaly test cases
    - Measure false positive and false negative rates
    - Tune detection thresholds for optimal accuracy
    - Document detection performance characteristics
    - _Requirements: 6.1-6.10_

- [ ] 16. Documentation and deployment
  - [ ] 16.1 Write API documentation
    - Document all public APIs with rustdoc
    - Provide usage examples for each component
    - Document configuration options
    - Generate and publish API docs
    - _Requirements: 10.10_
  
  - [ ] 16.2 Write integration guide
    - Document how to integrate monitoring into applications
    - Provide step-by-step instrumentation guide
    - Include code examples for common scenarios
    - _Requirements: 8.1_
  
  - [ ] 16.3 Write operations guide
    - Document baseline initialization procedure
    - Document anomaly investigation workflow
    - Document performance tuning guidelines
    - Document troubleshooting common issues
    - _Requirements: All_
  
  - [ ] 16.4 Create deployment artifacts
    - Provide example Prometheus configuration
    - Provide example Grafana dashboard JSON
    - Provide example systemd service files
    - Create Docker container for monitoring stack
    - _Requirements: All_
  
  - [ ] 16.5 Write security documentation
    - Document security best practices
    - Document threat model and mitigations
    - Document privacy considerations
    - Provide security checklist for deployment
    - _Requirements: All_

- [ ] 17. Final integration and validation
  - [ ] 17.1 Integrate into Niodoo workspace
    - Add silicon_synapse to workspace Cargo.toml
    - Update root documentation with monitoring info
    - Add monitoring to CI/CD pipeline
    - _Requirements: 8.1_
  
  - [ ] 17.2 Conduct end-to-end validation
    - Run full Niodoo system with monitoring enabled
    - Verify all metrics are collected and exported
    - Verify Grafana dashboards display correctly
    - Verify anomaly detection triggers appropriately
    - _Requirements: All_
  
  - [ ] 17.3 Performance validation
    - Measure actual overhead on Niodoo inference
    - Verify <5% overhead requirement is met
    - Optimize any performance bottlenecks
    - _Requirements: 8.6_
  
  - [ ] 17.4 Create demo and examples
    - Create demo script showing monitoring in action
    - Create example anomaly scenarios
    - Record demo video for documentation
    - _Requirements: All_
  
  - [ ] 17.5 Final review and sign-off
    - Review all code for quality and consistency
    - Verify all requirements are implemented
    - Conduct security review
    - Get stakeholder approval for production deployment
    - _Requirements: All_


- [ ] 18. Implement Development Velocity Tracking (Algorithmic Clock)
  - [ ] 18.1 Create Git integration for ground truth extraction
    - Add `git2` crate dependency for Git repository access
    - Implement commit history parser
    - Extract commit timestamps, authors, and messages
    - Correlate commits with task identifiers (parse commit messages, branch names)
    - Distinguish AI agent commits from human commits via author analysis
    - Write unit tests for commit parsing and correlation
    - _Requirements: 11.1, 11.2, 11.5_
  
  - [ ] 18.2 Implement project management integration
    - Add Jira REST API client (or GitHub Issues API)
    - Fetch task metadata (ID, type, complexity, story points, assignee)
    - Extract task start and completion timestamps
    - Correlate PM tasks with Git commits
    - Handle missing or incomplete task data gracefully
    - Write integration tests with mock Jira/GitHub APIs
    - _Requirements: 11.3, 11.6_
  
  - [ ] 18.3 Create AI agent performance tracking
    - Capture token consumption per task (from LLM API calls)
    - Track tool invocation count and types
    - Track iteration count (how many attempts to complete task)
    - Track context window usage
    - Store agent-specific metrics in time-series database
    - Write tests for agent metric collection
    - _Requirements: 11.4_
  
  - [ ] 18.4 Implement task duration calculation
    - Compute actual duration from task start to final commit
    - Handle multi-commit tasks (aggregate time)
    - Handle paused/resumed tasks (exclude idle time)
    - Normalize durations to business hours if configured
    - Write tests for duration calculation edge cases
    - _Requirements: 11.6_
  
  - [ ] 18.5 Create unified task completion dataset
    - Define `TaskCompletion` data structure with all features
    - Merge Git, PM, and agent performance data
    - Handle missing features with imputation strategies
    - Export dataset to training-ready format (CSV, Parquet)
    - Write tests for data merging and feature engineering
    - _Requirements: 11.8_

- [ ] 19. Implement AI Capability Benchmarking
  - [ ] 19.1 Create benchmark score ingestion system
    - Define schema for benchmark results (HumanEval, MBPP, SWE-bench)
    - Implement parsers for common benchmark output formats
    - Normalize scores to 0-1 scale for comparability
    - Store benchmark scores with agent version metadata
    - Write tests for score parsing and normalization
    - _Requirements: 11.7_
  
  - [ ] 19.2 Implement Agent Capability Score calculation
    - Aggregate multiple benchmark scores into composite score
    - Weight benchmarks by relevance to project domain
    - Track capability score over time (detect improvements/regressions)
    - Correlate capability scores with actual task performance
    - Write tests for score calculation and weighting
    - _Requirements: 11.7_
  
  - [ ] 19.3 Implement capability drift detection
    - Monitor for significant changes in agent capability scores
    - Detect when new agent versions are deployed
    - Trigger model recalibration on capability drift
    - Log capability change events
    - Write tests for drift detection thresholds
    - _Requirements: 11.13_

- [ ] 20. Implement Probabilistic Estimation Model
  - [ ] 20.1 Create feature engineering pipeline
    - Extract task features (type, complexity, description length)
    - Extract agent features (capability score, recent performance)
    - Extract context features (time of day, team load)
    - Create interaction features (task complexity × agent capability)
    - Implement feature scaling and encoding
    - Write tests for feature extraction and transformation
    - _Requirements: 11.8_
  
  - [ ] 20.2 Implement training data preparation
    - Split data into train/validation/test sets
    - Handle class imbalance (if certain task types are rare)
    - Implement cross-validation strategy
    - Create data loaders for ML framework
    - Write tests for data splitting and loading
    - _Requirements: 11.9_
  
  - [ ] 20.3 Implement Gradient Boosting model
    - Add `lightgbm` or `xgboost` Rust bindings
    - Define model hyperparameters
    - Implement quantile regression for probabilistic predictions
    - Train model to predict P50, P85, P95 completion times
    - Implement model serialization for persistence
    - Write tests for model training and prediction
    - _Requirements: 11.9, 11.10_
  
  - [ ] 20.4 Implement Monte Carlo simulation
    - Sample from predicted distribution to generate scenarios
    - Run 10,000+ simulations per estimation request
    - Aggregate simulation results into probability distribution
    - Calculate confidence intervals from simulation output
    - Write tests for simulation accuracy and performance
    - _Requirements: 11.14_
  
  - [ ] 20.5 Implement estimation API
    - Create REST endpoint for estimation requests
    - Accept task features as input
    - Return probabilistic forecast with confidence intervals
    - Include visualization data (probability density function)
    - Write integration tests for estimation API
    - _Requirements: 11.10_

- [ ] 21. Implement Calibration Engine
  - [ ] 21.1 Create calibration metrics tracking
    - Track predicted vs actual completion times
    - Calculate mean absolute error (MAE) and root mean squared error (RMSE)
    - Calculate confidence interval coverage (% of actuals within predicted intervals)
    - Track calibration metrics over time
    - Write tests for metric calculation
    - _Requirements: 11.12_
  
  - [ ] 21.2 Implement automated model retraining
    - Trigger retraining when new task completions are available
    - Implement incremental learning (update model with new data)
    - Validate new model against holdout test set
    - Deploy new model only if performance improves
    - Write tests for retraining pipeline
    - _Requirements: 11.11_
  
  - [ ] 21.3 Implement estimation bias detection
    - Detect systematic over-estimation or under-estimation
    - Detect bias for specific task types or agents
    - Flag bias as anomaly requiring investigation
    - Log bias detection events
    - Write tests for bias detection algorithms
    - _Requirements: 11.15_
  
  - [ ] 21.4 Create calibration dashboard
    - Visualize predicted vs actual scatter plots
    - Show calibration curve (predicted probability vs observed frequency)
    - Display confidence interval coverage over time
    - Show model performance metrics (MAE, RMSE)
    - Integrate with Grafana dashboards
    - _Requirements: 11.12_

- [ ] 22. Integration and Validation for Algorithmic Clock
  - [ ] 22.1 Integrate with existing monitoring infrastructure
    - Connect velocity tracking to telemetry bus
    - Export estimation metrics to Prometheus
    - Add velocity panels to Grafana dashboards
    - Ensure <5% overhead on development workflow
    - _Requirements: 11.8_
  
  - [ ] 22.2 Validate estimation accuracy
    - Collect 3+ months of historical task data
    - Train initial model on historical data
    - Backtest model predictions against actual outcomes
    - Measure calibration metrics on test set
    - Tune model hyperparameters for optimal accuracy
    - _Requirements: 11.9, 11.12_
  
  - [ ] 22.3 Conduct user acceptance testing
    - Deploy estimation system to pilot project
    - Collect feedback from project managers
    - Compare AI estimates vs human estimates vs actual
    - Iterate on feature engineering based on feedback
    - _Requirements: 11.10_
  
  - [ ] 22.4 Document Algorithmic Clock system
    - Write user guide for project managers
    - Document data sources and feature engineering
    - Document model architecture and training process
    - Provide interpretation guide for probabilistic forecasts
    - _Requirements: All_
  
  - [ ] 22.5 Create estimation report templates
    - Design report showing probability distributions
    - Include confidence intervals and risk analysis
    - Show historical calibration accuracy
    - Provide actionable recommendations
    - _Requirements: 11.14_
