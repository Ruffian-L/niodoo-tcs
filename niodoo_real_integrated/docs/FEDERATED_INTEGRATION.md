# Federated Resilience Integration Guide

This document describes how to integrate the federated shard telemetry system into your runtime.

## Overview

The federated resilience system provides:
- **Shard Telemetry**: gRPC service for topology signature collection
- **Flux Metrics**: Fused flux, IQR, and median gap tracking
- **Client Integration**: Connect to telemetry endpoints and fetch signatures
- **Mock Server**: Test harness for local development

## Quick Start

### 1. Start the Mock Telemetry Server

```bash
# Terminal 1: Start mock server
cd niodoo_real_integrated
cargo run --bin shard_telemetry_server -- \
    --shard-id test-shard-1 \
    --addr 0.0.0.0:50051
```

### 2. Run the Federated Runtime Integration

```bash
# Terminal 2: Run integration test
export TELEMETRY_ENDPOINT=http://localhost:50051
cargo run --bin federated_runtime
```

## Architecture

### Components

#### ShardClient (`src/federated.rs`)

The `ShardClient` connects to telemetry endpoints and fetches topology signatures:

```rust
use niodoo_real_integrated::federated::ShardClient;
use std::time::Duration;

// Connect with timeout
let client = ShardClient::connect_with_timeout(
    "http://localhost:50051",
    Duration::from_secs(5)
).await?;

// Fetch signatures
let endpoints = vec!["shard-1".to_string(), "shard-2".to_string()];
let signatures = client.fetch_shard_signatures(&endpoints).await?;
```

#### FluxMetrics

`FluxMetrics` tracks shard health:

```rust
use niodoo_real_integrated::federated::{FluxMetrics, NodalDiagnostics};

let mut diagnostics = NodalDiagnostics::new();
let metrics = diagnostics.merge_shard_metrics(&proto_bytes, &gaps)?;

println!("Fused Flux: {}", metrics.fused_flux);
println!("Median Gap: {}", metrics.median_gap);
println!("IQR: {}", metrics.interquartile_range);
```

#### MCTS Configuration

Tune MCTS search behavior:

```rust
use niodoo_real_integrated::mcts_config::{MctsConfig, MctsProfile};

// Use a predefined profile
let config = MctsProfile::Balanced.to_config();

// Or create custom configuration
let custom_config = MctsConfig {
    max_simulations: 150,
    max_time_ms: 75,
    exploration_c: 1.5,
    reward_shaping: RewardShaping::default(),
    depth_limits: DepthLimits::default(),
};
```

## Metrics Instrumentation

### Prometheus Integration

FluxMetrics are automatically exported to Prometheus:

- `shard_fused_flux`: Weighted average flux across shards
- `shard_count`: Number of active shards
- `shard_iqr`: Interquartile range of spectral gaps
- `shard_median_gap`: Median spectral gap value

### Adding to Monitoring Stack

Add these metrics to your Prometheus configuration:

```yaml
scrape_configs:
  - job_name: 'niodoo-federated'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

Create Grafana dashboard panels for:
- Fused flux over time
- Shard count histogram
- IQR distribution
- Median gap trends

## MCTS Tuning

### Profiles

Three pre-configured profiles are available:

| Profile   | Simulations | Time Budget | Use Case                |
|-----------|-------------|-------------|-------------------------|
| Fast      | 25          | 20ms        | Low-latency queries     |
| Balanced  | 100         | 50ms        | General purpose         |
| Thorough  | 200         | 100ms       | Complex reasoning       |

### Custom Tuning

Adjust these parameters based on your latency budget:

```rust
let config = MctsConfig {
    max_simulations: 100,  // Iteration cap
    max_time_ms: 50,       // Time budget
    exploration_c: 1.414,  // UCB1 exploration constant
    reward_shaping: RewardShaping {
        exploration_bonus: 0.1,  // Encourage exploration
        depth_penalty: 0.05,     // Discourage deep trees
        reward_scale: 1.0,       // Global reward scaling
        discount_factor: 0.9,    // Future reward discount
    },
    depth_limits: DepthLimits {
        max_depth: 10,           // Hard depth limit
        prune_depth: 8,          // Depth to start pruning
        iterative_deepening: false,
    },
};
```

### Adaptive Configuration

Use `AdaptiveMctsConfig` for runtime tuning:

```rust
use niodoo_real_integrated::mcts_config::AdaptiveMctsConfig;

let mut adaptive = AdaptiveMctsConfig::new(MctsConfig::default());

// Configuration adapts based on performance
adaptive.adapt(elapsed_ms, iterations);
let config = adaptive.config();
```

## Testing

### Unit Tests

```bash
cargo test --lib federated
```

### Integration Test

```bash
# Start mock server in background
cargo run --bin shard_telemetry_server &
MOCK_PID=$!

# Run integration test
cargo run --bin federated_runtime

# Cleanup
kill $MOCK_PID
```

### Performance Profiling

```bash
# Profile MCTS search
cargo bench --bench mcts_search

# Measure telemetry latency
cargo run --bin federated_runtime -- --profile
```

## Deployment

### Production Setup

1. **Stand up telemetry endpoints**: Deploy `shard_telemetry_server` to your infrastructure
2. **Configure endpoints**: Set `TELEMETRY_ENDPOINT` environment variable
3. **Enable metrics**: Ensure Prometheus scraping is configured
4. **Tune MCTS**: Adjust `MctsConfig` based on observed latency

### Monitoring Checklist

- [ ] Telemetry endpoints accessible
- [ ] Prometheus metrics exported
- [ ] Grafana dashboards configured
- [ ] MCTS latency within budget
- [ ] Flux metrics trending healthy

## Troubleshooting

### Connection Errors

```
Error: Failed to connect to telemetry server
```

**Solution**: Ensure mock server is running and endpoint is correct:
```bash
curl http://localhost:50051/health
```

### Timeout Errors

```
Error: fetch_shard_signatures exceeded timeout
```

**Solution**: Increase timeout or reduce shard count:
```rust
let client = ShardClient::connect_with_timeout(
    endpoint,
    Duration::from_secs(10)  // Increase timeout
).await?;
```

### Metrics Not Appearing

**Solution**: Verify Prometheus registry initialization:
```rust
use prometheus::Registry;
let registry = Registry::new();
```

## References

- `src/federated.rs`: Client and metrics implementation
- `src/mcts_config.rs`: MCTS configuration
- `src/bin/shard_telemetry_server.rs`: Mock server
- `src/bin/federated_runtime.rs`: Integration example

