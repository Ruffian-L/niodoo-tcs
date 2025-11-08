# NIODOO Pipeline - API Documentation

## Generating Documentation

To generate comprehensive API documentation:

```bash
# Generate HTML documentation
cargo doc --no-deps --open

# Generate markdown documentation
cargo doc --no-deps --document-private-items --format json | cargo-doc-to-markdown
```

## Core Modules

### Pipeline (`pipeline`)

Main pipeline orchestrator:

- `Pipeline::initialise()` - Initialize pipeline with configuration
- `Pipeline::process_prompt()` - Process a single prompt through the pipeline
- `Pipeline::shutdown()` - Gracefully shutdown pipeline

### ERAG (`erag`)

Emotional Retrieval-Augmented Generation:

- `EragClient::new()` - Create ERAG client
- `EragClient::collapse_with_limit()` - Retrieve memories with similarity filtering
- `EragClient::store_experience()` - Store experience in memory

### Generation (`generation`)

Text generation via vLLM:

- `GenerationEngine::new_with_config()` - Create generation engine
- `GenerationEngine::generate()` - Generate response from prompt
- `GenerationEngine::warmup()` - Warm up generation engine

### Learning (`learning`)

Reinforcement learning loop:

- `LearningLoop::new()` - Initialize learning loop
- `LearningLoop::update()` - Update learning from experience
- `LearningLoop::choose_action()` - Select action based on state

### Token Manager (`token_manager`)

Dynamic tokenizer management:

- `DynamicTokenizerManager::initialise()` - Initialize tokenizer
- `DynamicTokenizerManager::run_promotion_cycle()` - Run token promotion cycle
- `DynamicTokenizerManager::process()` - Process prompt with tokenizer

### Circuit Breaker (`circuit_breaker`)

Resilience patterns:

- `CircuitBreaker::new()` - Create circuit breaker
- `CircuitBreaker::call()` - Execute function with circuit breaker protection
- `CircuitBreaker::state()` - Get current circuit state

### Health Checks (`health`)

Production health monitoring:

- `HealthRegistry::new()` - Create health registry
- `HealthRegistry::register_component()` - Register component health
- `HealthRegistry::get_health()` - Get overall system health
- `HealthServer::start()` - Start health check HTTP server

### Metrics (`metrics`)

Prometheus metrics:

- `metrics()` - Get global metrics instance
- `PipelineMetrics` - Pipeline performance metrics
- `TokenizerMetrics` - Tokenizer metrics
- `CacheMetrics` - Cache performance metrics

### Configuration (`config`)

Configuration management:

- `RuntimeConfig::load()` - Load configuration from environment/files
- `RuntimeConfig::validate()` - Validate configuration values
- `CliArgs::parse()` - Parse command-line arguments

## Error Handling

All modules use `anyhow::Result<T>` for error handling:

```rust
use anyhow::Result;

pub async fn example_function() -> Result<()> {
    // Operations that can fail
    Ok(())
}
```

## Async Patterns

All I/O operations are async and use `tokio`:

```rust
use tokio;
use anyhow::Result;

pub async fn async_operation() -> Result<()> {
    // Async operations
    Ok(())
}
```

## Metrics

All metrics are exposed via Prometheus at `/metrics`:

- `niodoo_pipeline_latency_seconds` - Request latency histogram
- `niodoo_pipeline_requests_total` - Total requests counter
- `niodoo_pipeline_errors_total` - Error counter
- `niodoo_cache_hits_total` - Cache hits
- `niodoo_tokenizer_promotions_total` - Token promotions
- `niodoo_circuit_breaker_state` - Circuit breaker state

## Health Endpoints

HTTP endpoints for health monitoring:

- `GET /health` - Liveness probe (200 = healthy, 503 = unhealthy)
- `GET /ready` - Readiness probe (200 = ready, 503 = not ready)
- `GET /metrics` - Prometheus metrics

## Configuration

Configuration via environment variables or ConfigMap:

- `QDRANT_URL` - Qdrant service URL
- `VLLM_ENDPOINT` - vLLM service endpoint
- `VLLM_MODEL` - Model identifier
- `CACHE_CAPACITY` - Cache size
- `SIMILARITY_THRESHOLD` - ERAG similarity threshold
- `TOKEN_PROMOTION_INTERVAL` - Promotion cycle interval

## Examples

See `examples/` directory for usage examples:

- `basic_pipeline.rs` - Basic pipeline usage
- `health_check.rs` - Health check integration
- `circuit_breaker.rs` - Circuit breaker usage
- `metrics.rs` - Metrics collection

## Testing

Run tests:

```bash
cargo test
cargo test --release
cargo test --features otel,svc
```

## Documentation

View generated documentation:

```bash
cargo doc --open
```

For detailed module documentation, see generated rustdoc output.
