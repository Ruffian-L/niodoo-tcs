# Testing and Monitoring Guide

This guide documents the comprehensive testing, benchmarking, and monitoring infrastructure for the Niodoo-TCS system.

## Overview

The system now includes:
- **End-to-end build/test automation**
- **Comprehensive unit tests**
- **Performance benchmarking suite**
- **Unified service management**
- **Continuous health monitoring**

## Quick Start

### Run All Tests
```bash
./e2e_build_test.sh
```

### Run Performance Benchmarks
```bash
./run_performance_benchmarks.sh
```

### Start All Services
```bash
./unified_service_manager.sh start
```

### Monitor Services
```bash
./unified_service_manager.sh monitor
```

## Detailed Documentation

### 1. End-to-End Build and Test (`e2e_build_test.sh`)

This script provides comprehensive validation of the entire system:

#### Features
- Prerequisites checking (Cargo, Rustc, jq)
- External service health checks (vLLM, Qdrant, Ollama)
- Full workspace build (library, binaries, release)
- Unit tests for all TCS components
- Integration tests
- Performance benchmarks
- Service startup validation
- Health check endpoints
- End-to-end integration test

#### Usage
```bash
# Run complete test suite
./e2e_build_test.sh

# View build logs
cat /tmp/build_lib.log
cat /tmp/build_bins.log
cat /tmp/build_release.log

# View test logs
cat /tmp/test_tcs_core.log
cat /tmp/test_tcs_tda.log
cat /tmp/test_tcs_knot.log
cat /tmp/test_tcs_ml.log
```

#### Output
- Build artifacts in `target/`
- Test results in console
- Summary report at end

### 2. Unit Tests

#### tcs-knot Tests
Located in `tcs-knot/src/lib.rs`:
- `trefoil_complexity_is_positive` - Verifies trefoil knot has positive complexity
- `unknot_has_complexity_zero` - Tests unknot has zero complexity
- `knot_diagram_equality` - Verifies diagram equality
- `cache_memoization` - Tests caching behavior
- `complexity_increases_with_crossings` - Complexity scaling
- `different_knots_have_different_polynomials` - Uniqueness verification

#### tcs-ml Tests
Located in `tcs-ml/src/lib.rs`:
- `exploration_agent_is_deterministic_with_seed` - Deterministic randomness
- `exploration_agent_different_seeds_produce_different_actions` - Seed variation
- `motor_brain_new` - Brain initialization
- `motor_brain_process_empty_input` - Empty input handling
- `motor_brain_process_help_request` - Intent detection
- `equivariant_layer_forward` - Equivariant layer computation
- `pairwise_distances_positive` - Distance matrix properties
- `cognitive_knot_serialization` - Serialization tests

#### tcs-pipeline Tests
Located in `tcs-pipeline/src/lib.rs` and `tcs-pipeline/src/config.rs`:
- `custom_consensus_threshold_is_used` - Config application
- `orchestrator_default_construction` - Default setup
- `orchestrator_config_construction` - Custom config
- `orchestrator_ready_state` - Buffer readiness
- `orchestrator_reset_brain_context` - State reset
- `orchestrator_process_empty_input` - Empty input handling
- `orchestrator_process_not_ready` - Not-ready handling
- `config_serialization` - TOML serialization
- `config_deserialization` - TOML parsing
- `config_from_file` - File loading
- `config_invalid_file` - Error handling

#### Running Tests
```bash
# Run all tests
cargo test

# Run specific package tests
cargo test --package tcs-knot
cargo test --package tcs-ml
cargo test --package tcs-pipeline

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test --test test_name
```

### 3. Performance Benchmarking (`run_performance_benchmarks.sh`)

Comprehensive performance analysis suite:

#### Features
- System resource detection (CPU, Memory, GPU)
- Rust benchmark execution
- TCS pipeline performance tests
- Memory profiling
- Performance metrics collection
- CSV report generation

#### Benchmarks Included
- Topological benchmarks
- Consciousness engine benchmarks
- RAG optimization benchmarks
- Sparse Gaussian process benchmarks

#### Usage
```bash
# Run all benchmarks
./run_performance_benchmarks.sh

# View results
cat results/benchmarks/benchmark_report_*.txt
```

#### Output
Reports include:
- System resources (CPU, memory, GPU)
- Benchmark execution times
- Binary sizes
- Memory usage statistics
- Performance trends

### 4. Unified Service Manager (`unified_service_manager.sh`)

Comprehensive service lifecycle management:

#### Services Managed
- **vLLM** - LLM inference server (port 5001)
- **Qdrant** - Vector database (port 6333)
- **Ollama** - Local LLM server (port 11434)
- **Metrics** - Metrics server (port 9093)

#### Commands
```bash
# Start all services
./unified_service_manager.sh start

# Stop all services
./unified_service_manager.sh stop

# Restart all services
./unified_service_manager.sh restart

# Check status
./unified_service_manager.sh status

# Run health checks
./unified_service_manager.sh health

# Monitor and auto-restart
./unified_service_manager.sh monitor

# Start specific service
./unified_service_manager.sh start-service vllm

# Stop specific service
./unified_service_manager.sh stop-service vllm

# Restart specific service
./unified_service_manager.sh restart-service vllm
```

#### Health Checks
Each service has a dedicated health endpoint:
- vLLM: `http://127.0.0.1:5001/v1/models`
- Qdrant: `http://127.0.0.1:6333/health`
- Ollama: `http://127.0.0.1:11434/api/tags`
- Metrics: `http://127.0.0.1:9093/metrics`

#### Monitoring Mode
The monitor mode continuously checks services every 30 seconds and automatically restarts failed services.

### 5. Continuous Health Monitoring (`continuous_health_monitor.sh`)

Real-time health monitoring with metrics recording:

#### Features
- Continuous health checks (every 10 seconds)
- Response time measurement
- CPU and memory usage tracking
- CSV metrics export
- Real-time status display

#### Usage
```bash
# Start monitoring
./continuous_health_monitor.sh

# View metrics
cat results/health_metrics/health_metrics_*.csv
```

#### Metrics Recorded
- Timestamp
- Service name
- Health status
- Response time (ms)
- CPU usage (%)
- Memory usage (%)

## Integration Testing

### Master Consciousness Orchestrator Test
```bash
cargo run --release --bin master_consciousness_orchestrator
```

This runs a comprehensive integration test that:
- Initializes consciousness engine
- Performs health checks on all services
- Tests vLLM integration
- Validates memory systems
- Reports system status

### TCS Pipeline Integration Test
```bash
cd tcs-pipeline
cargo test --test integration
```

## Troubleshooting

### Services Won't Start
1. Check logs: `tail -f /tmp/niodoo_logs/*.log`
2. Check ports: `netstat -tuln | grep -E '5001|6333|11434|9093'`
3. Check PIDs: `./unified_service_manager.sh status`

### Tests Failing
1. Run individual tests: `cargo test --package <package> -- --nocapture`
2. Check compilation: `cargo build --lib`
3. View error logs: `cat /tmp/test_*.log`

### Benchmarks Failing
1. Check system resources: `./run_performance_benchmarks.sh` (it shows resources)
2. Increase available memory
3. Check GPU availability: `nvidia-smi`

### Health Checks Failing
1. Ensure services are running: `./unified_service_manager.sh status`
2. Check service logs
3. Verify network connectivity
4. Restart services: `./unified_service_manager.sh restart`

## Best Practices

### Development Workflow
1. Write unit tests for new features
2. Run `e2e_build_test.sh` before committing
3. Run benchmarks for performance-critical changes
4. Monitor services during development

### Production Deployment
1. Run full test suite
2. Run performance benchmarks
3. Start services with monitoring: `./unified_service_manager.sh monitor`
4. Set up continuous health monitoring

### CI/CD Integration
```bash
# In CI pipeline
./e2e_build_test.sh
./run_performance_benchmarks.sh
./unified_service_manager.sh health
```

## Metrics and Reporting

### Test Coverage
Unit tests cover:
- Core topological operations
- Knot theory computations
- Machine learning agents
- Pipeline orchestration
- Configuration management

### Performance Baselines
Benchmarks establish performance baselines for:
- Topological computations
- Consciousness engine cycles
- Memory management
- Service response times

### Health Monitoring
Continuous monitoring tracks:
- Service availability
- Response latency
- Resource utilization
- Error rates

## Next Steps

### Short Term
- ✅ End-to-end build/test automation
- ✅ Unit tests for critical components
- ✅ Performance benchmarking suite
- ✅ Service startup and health checks
- ✅ Continuous health monitoring

### Future Enhancements
- Automated regression testing
- Load testing suite
- Distributed testing
- Performance profiling with instruments
- Automated alerting on health failures
- Service mesh integration
- Chaos engineering tests

## Support

For issues or questions:
1. Check logs in `/tmp/niodoo_logs/`
2. Review test output in `/tmp/`
3. Run `./unified_service_manager.sh health`
4. Consult service-specific documentation

## References

- [Rust Testing](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Criterion Benchmarking](https://github.com/bheisler/criterion.rs)
- [Prometheus Metrics](https://prometheus.io/docs/introduction/overview/)
- [Service Health Checks](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)

