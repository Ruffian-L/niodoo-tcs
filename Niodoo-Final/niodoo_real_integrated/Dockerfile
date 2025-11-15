# Multi-stage Dockerfile for NIODOO Real Integrated (Phase 5 Production Ready)
# Build stage
FROM rust:1.75-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY ../tcs-core ./tcs-core
COPY ../tcs-tda ./tcs-tda
COPY ../tcs-knot ./tcs-knot
COPY ../tcs-tqft ./tcs-tqft
COPY ../tcs-ml ./tcs-ml
COPY ../tcs-consensus ./tcs-consensus
COPY ../tcs-pipeline ./tcs-pipeline
COPY ../niodoo-core ./niodoo-core
COPY ../constants_core ./constants_core
COPY . .

# Build in release mode with optimizations
RUN cargo build --release --bin niodoo_real_integrated && \
    strip target/release/niodoo_real_integrated

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 niodoo && \
    mkdir -p /app/logs /app/data && \
    chown -R niodoo:niodoo /app

# Copy binary from builder
COPY --from=builder /build/target/release/niodoo_real_integrated /app/niodoo_real_integrated

# Copy configuration templates (if any)
COPY config/ /app/config/

USER niodoo

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD pgrep -f niodoo_real_integrated || exit 1

ENTRYPOINT ["/app/niodoo_real_integrated"]

