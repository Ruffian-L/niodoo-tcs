# Silicon Synapse Hardware Monitoring System

A comprehensive hardware-grounded AI state monitoring system for the Niodoo-Feeling Gen 1 consciousness engine.

## Overview

Silicon Synapse establishes a multi-layered telemetry pipeline that captures:
- **Hardware metrics**: GPU temperature, power consumption, VRAM usage, CPU utilization
- **Inference performance**: Time To First Token (TTFT), Time Per Output Token (TPOT), throughput
- **Model internal states**: Softmax entropy, activation patterns, attention weights
- **Anomaly detection**: Real-time deviation detection with classification

## Core Philosophy

The primary goal is NOT to anthropomorphize hardware states as "emotions," but rather to create a robust observability framework for AI safety, anomaly detection, and performance optimization. By establishing a rich baseline of normal operational behavior across all system layers, we can detect security threats, model instabilities, emergent behaviors, and performance degradation through their unique physical and computational signatures.

## Architecture

```text
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
         │  ┌──────────────────────────────────┐  │
         │  │     Metric Collection Layer      │  │
         │  │  ┌────────┐ ┌────────┐ ┌──────┐ │  │
         │  │  │Hardware│ │Inference│ │Model │ │  │
         │  │  │Collector│ │Collector│ │Probe │ │  │
         │  │  └───┬────┘ └───┬────┘ └──┬───┘ │  │
         │  └──────┼──────────┼─────────┼─────┘  │
         │         │          │         │         │
         │  ┌──────▼──────────▼─────────▼─────┐  │
         │  │    Metric Aggregation Engine    │  │
         │  │  (time-series buffer, reducer) │  │
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

## Features

- **Hardware Telemetry**: GPU/CPU monitoring with sub-second granularity
- **Inference Performance**: TTFT, TPOT, throughput tracking
- **Model Internal State**: Softmax entropy, activation pattern analysis
- **Baseline Learning**: Automatic normal behavior profiling
- **Anomaly Detection**: Real-time deviation detection with classification
- **Prometheus Integration**: Standard metrics export for observability
- **Non-Invasive Design**: <5% performance overhead, fail-safe operation

## Quick Start

### 1. Configuration

The system comes with a default configuration file at `config/silicon_synapse.toml`. You can customize it as needed:

```toml
enabled = true

[hardware]
enabled = true
collection_interval_ms = 1000

[inference]
enabled = true
track_ttft = true
track_tpot = true
track_throughput = true

[model_probe]
enabled = true
probe_entropy = true
probe_activations = true

[exporter]
enabled = true
exporter_type = "prometheus"
bind_address = "0.0.0.0:9090"
```

### 2. Running the Demo

The easiest way to get started is with the demo example:

```bash
# Run the demo
cargo run --example silicon_synapse_demo

# In another terminal, check metrics
curl http://localhost:9090/metrics

# Check health
curl http://localhost:9090/health
```

The demo will:
- Start the monitoring system
- Simulate 10 inference requests with token generation
- Export metrics to Prometheus format
- Keep running until you press Ctrl+C

### 3. Integration

For integration with your own code:

```rust
use niodoo_consciousness::silicon_synapse::{SiliconSynapse, Config, TelemetryEvent};
use std::time::Instant;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::load("config/silicon_synapse.toml")?;
    
    // Initialize monitoring system
    let mut synapse = SiliconSynapse::new(config).await?;
    synapse.start().await?;
    
    // Get telemetry sender for consciousness engine integration
    let telemetry_tx = synapse.telemetry_sender();
    
    // Emit telemetry events from consciousness engine
    let request_id = Uuid::new_v4();
    telemetry_tx.send(TelemetryEvent::InferenceStart {
        request_id,
        timestamp: Instant::now(),
        prompt_length: 100,
    }).await?;
    
    // ... consciousness processing ...
    
    telemetry_tx.send(TelemetryEvent::InferenceComplete {
        request_id,
        timestamp: Instant::now(),
        total_tokens: 50,
        error: None,
    }).await?;
    
    // Metrics are automatically exported to Prometheus at :9090/metrics
    // Anomalies are detected and logged in real-time
    
    Ok(())
}
```

## API Reference

### Telemetry Events

```rust
pub enum TelemetryEvent {
    /// Inference request started
    InferenceStart {
        request_id: Uuid,
        timestamp: Instant,
        prompt_length: usize,
    },
    
    /// Token generated during inference
    TokenGenerated {
        request_id: Uuid,
        timestamp: Instant,
        token_index: usize,
        token_length: usize,
    },
    
    /// Inference request completed
    InferenceComplete {
        request_id: Uuid,
        timestamp: Instant,
        total_tokens: usize,
        error: Option<String>,
    },
    
    /// Hardware metrics collected
    HardwareMetrics {
        timestamp: Instant,
        gpu_temperature: Option<f32>,
        gpu_power: Option<f32>,
        gpu_utilization: Option<f32>,
        // ... more hardware metrics
    },
    
    /// Model internal state metrics
    ModelMetrics {
        timestamp: Instant,
        layer_index: usize,
        entropy: Option<f32>,
        activation_sparsity: Option<f32>,
        // ... more model metrics
    },
    
    /// Anomaly detected
    AnomalyDetected {
        timestamp: Instant,
        anomaly_type: String,
        severity: String,
        description: String,
        metrics: Vec<(String, f32)>,
    },
}
```

### Configuration Options

| Section | Option | Description | Default |
|---------|--------|-------------|---------|
| `hardware` | `enabled` | Enable hardware monitoring | `true` |
| `hardware` | `collection_interval_ms` | Collection interval | `1000` |
| `inference` | `track_ttft` | Track Time To First Token | `true` |
| `inference` | `track_tpot` | Track Time Per Output Token | `true` |
| `model_probe` | `probe_entropy` | Probe softmax entropy | `true` |
| `model_probe` | `sampling_rate` | Token sampling rate | `0.1` |
| `exporter` | `bind_address` | HTTP server address | `"0.0.0.0:9090"` |
| `exporter` | `metrics_path` | Metrics endpoint | `"/metrics"` |

## Prometheus Metrics

The system exports metrics in Prometheus format at `/metrics`:

```
# Hardware metrics
silicon_synapse_gpu_temperature{type="mean"} 45.2
silicon_synapse_gpu_power{type="mean"} 150.5
silicon_synapse_gpu_utilization{type="mean"} 75.3

# Inference metrics
silicon_synapse_ttft_ms{type="mean"} 125.4
silicon_synapse_tpot_ms{type="mean"} 25.8
silicon_synapse_throughput_tps{type="mean"} 38.7

# Model metrics
silicon_synapse_model_entropy{layer="0",type="mean"} 2.34
silicon_synapse_model_sparsity{layer="0",type="mean"} 0.15
```

## Anomaly Detection

The system automatically detects anomalies in:

- **Hardware**: Temperature spikes, power anomalies, utilization patterns
- **Inference**: Performance degradation, error rate increases
- **Model**: Unusual entropy patterns, activation distributions

Anomalies are classified by type and severity:
- **Security**: Unusual access patterns
- **Instability**: High error rates, crashes
- **Degradation**: Performance issues
- **Emergent**: Unexpected model behavior
- **Hardware**: Physical component issues

## Development

### Building

```bash
cargo build --release
```

### Testing

```bash
cargo test
```

### Running Tests

```bash
cargo test silicon_synapse
```

## Integration with Niodoo Consciousness

Silicon Synapse integrates seamlessly with the existing Niodoo consciousness architecture:

- **Dual Möbius Gaussian models**: Monitors internal state evolution
- **EchoMemoria memory systems**: Tracks memory access patterns
- **Inference pipeline**: Measures performance and quality metrics
- **Consciousness states**: Correlates hardware states with cognitive processes

## Performance Impact

- **CPU overhead**: <2% during normal operation
- **Memory usage**: ~50MB for monitoring infrastructure
- **Network**: Minimal (local Prometheus scraping)
- **Storage**: Configurable retention policies

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change `bind_address` in configuration
2. **Permission errors**: Ensure access to hardware monitoring APIs
3. **High CPU usage**: Reduce `collection_interval_ms`
4. **Memory leaks**: Check telemetry buffer size limits
5. **Demo won't start**: Ensure port 9090 is available
6. **No metrics appearing**: Check that the telemetry sender is working
7. **High memory usage**: Reduce `max_buffer_size` in telemetry config

### Debugging

Enable debug logging:

```rust
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

### Health Checks

```bash
# Check system health
curl http://localhost:9090/health

# Check metrics endpoint
curl http://localhost:9090/metrics

# Check JSON API
curl http://localhost:9090/api/v1/metrics
```

### Example Prometheus Queries

Once metrics are being collected, you can use these PromQL queries:

```promql
# Average GPU temperature over last 5 minutes
avg_over_time(silicon_synapse_gpu_temperature[5m])

# 95th percentile inference latency
histogram_quantile(0.95, silicon_synapse_inference_duration_seconds)

# Token generation rate
rate(silicon_synapse_tokens_generated_total[1m])

# Error rate
rate(silicon_synapse_inference_errors_total[5m]) / rate(silicon_synapse_inference_requests_total[5m])

# GPU utilization trend
predict_linear(silicon_synapse_gpu_utilization[1h], 3600)
```

### Example Grafana Dashboard

Here's a basic Grafana dashboard JSON for monitoring:

```json
{
  "dashboard": {
    "title": "Silicon Synapse Monitoring",
    "panels": [
      {
        "title": "GPU Temperature",
        "type": "graph",
        "targets": [
          {
            "expr": "silicon_synapse_gpu_temperature",
            "legendFormat": "GPU {{instance}}"
          }
        ]
      },
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, silicon_synapse_inference_duration_seconds)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Token Generation Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(silicon_synapse_tokens_generated_total[1m])",
            "legendFormat": "Tokens/sec"
          }
        ]
      }
    ]
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built for the Niodoo-Feeling Gen 1 consciousness engine
- Integrates with Candle ML framework
- Compatible with Prometheus/Grafana observability stack
- Designed for AI safety and performance optimization