//! OpenTelemetry Distributed Tracing Integration
//!
//! Provides distributed tracing using OpenTelemetry for production observability.
//! Requires `otel` feature to be enabled.

#[cfg(feature = "otel")]
use anyhow::Result;
#[cfg(feature = "otel")]
use opentelemetry::global;
#[cfg(feature = "otel")]
use opentelemetry::trace::{Span, Tracer};
#[cfg(feature = "otel")]
use opentelemetry_sdk::propagation::TraceContextPropagator;
#[cfg(feature = "otel")]
use tracing::{info, warn};
#[cfg(feature = "otel")]
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Initialize OpenTelemetry tracing
#[cfg(feature = "otel")]
pub fn init_tracing(endpoint: Option<String>) -> Result<()> {
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::TracerProvider;
    use opentelemetry_sdk::Resource;
    use std::time::Duration;

    let otlp_endpoint = endpoint.unwrap_or_else(|| {
        std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:4317".to_string())
    });

    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "niodoo-pipeline".to_string());

    info!(endpoint = %otlp_endpoint, service = %service_name, "Initializing OpenTelemetry tracing");

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(&otlp_endpoint)
                .with_timeout(Duration::from_secs(5)),
        )
        .with_trace_config(opentelemetry_sdk::trace::Config::default().with_resource(
            Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", service_name),
                opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            ]),
        ))
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .map_err(|e| anyhow::anyhow!("Failed to initialize OpenTelemetry: {}", e))?;

    global::set_text_map_propagator(TraceContextPropagator::new());

    tracing_opentelemetry::layer().with_tracer(tracer).init();

    info!("OpenTelemetry tracing initialized successfully");

    Ok(())
}

/// Shutdown OpenTelemetry tracing
#[cfg(feature = "otel")]
pub fn shutdown_tracing() {
    opentelemetry::global::shutdown_tracer_provider();
}

/// Create a span for a pipeline operation
#[cfg(feature = "otel")]
pub fn create_pipeline_span(operation: &str) -> tracing::Span {
    use tracing::trace_span;
    trace_span!("pipeline", operation = operation).with_context(|| {
        opentelemetry::trace::SpanContext::new(
            opentelemetry::trace::TraceId::from_u128(0),
            opentelemetry::trace::SpanId::from_u64(0),
            0,
            false,
            opentelemetry::trace::TraceFlags::default(),
        )
    })
}

#[cfg(not(feature = "otel"))]
pub fn init_tracing(_endpoint: Option<String>) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(not(feature = "otel"))]
pub fn shutdown_tracing() {}

#[cfg(not(feature = "otel"))]
pub fn create_pipeline_span(_operation: &str) -> tracing::Span {
    tracing::span!(tracing::Level::INFO, "pipeline")
}
