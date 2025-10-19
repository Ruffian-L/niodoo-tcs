//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// Deployment configurations for TCS
/// Includes Docker, Kubernetes, and monitoring setup

pub const DOCKERFILE: &str = r#"
# Multi-stage Docker build for TCS
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Install Rust and system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    libssl-dev \
    pkg-config \
    libopenblas-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source and build
WORKDIR /app
COPY . .
RUN cargo build --release --features cuda

# Runtime stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

COPY --from=builder /app/target/release/tcs /usr/local/bin/
COPY --from=builder /app/config /etc/tcs/

EXPOSE 8080 9090 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["tcs", "run", "--config", "/etc/tcs/config.toml"]
"#;

pub const KUBERNETES_DEPLOYMENT: &str = r#"
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: tcs-nodes
  namespace: tcs-system
spec:
  serviceName: tcs-cluster
  replicas: 5
  selector:
    matchLabels:
      app: tcs
  template:
    metadata:
      labels:
        app: tcs
    spec:
      containers:
      - name: tcs
        image: tcs:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "48Gi"
            cpu: "16"
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        env:
        - name: RUST_LOG
          value: "info"
        - name: TCS_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /etc/tcs
        ports:
        - containerPort: 8080  # HTTP API
        - containerPort: 9090  # Metrics
        - containerPort: 50051 # gRPC
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: tcs-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: tcs-cluster
  namespace: tcs-system
spec:
  clusterIP: None
  selector:
    app: tcs
  ports:
  - name: consensus
    port: 50051
  - name: http
    port: 8080
  - name: metrics
    port: 9090
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: tcs-config
  namespace: tcs-system
data:
  config.toml: |
    [tcs]
    node_id = "${TCS_NODE_ID}"
    consensus_port = 50051
    http_port = 8080
    metrics_port = 9090

    [topology]
    max_persistence_dimension = 2
    embedding_window_size = 1000

    [gpu]
    enabled = true
    memory_pool_size = "8GB"
"#;

pub const PROMETHEUS_CONFIG: &str = r#"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'tcs-nodes'
    static_configs:
      - targets: ['tcs-node-0:9090', 'tcs-node-1:9090', 'tcs-node-2:9090', 'tcs-node-3:9090', 'tcs-node-4:9090']
    scrape_interval: 5s

  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https
"#;

pub const GRAFANA_DASHBOARD: &str = r#"{
  "dashboard": {
    "title": "TCS Monitoring Dashboard",
    "tags": ["tcs", "topology", "cognitive"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "Persistence Computation Rate",
        "type": "graph",
        "targets": [{
          "expr": "rate(tcs_persistence_computations_total[5m])",
          "legendFormat": "Persistence computations/sec"
        }]
      },
      {
        "title": "Knot Detection Rate",
        "type": "graph",
        "targets": [{
          "expr": "rate(tcs_knot_detections_total[5m])",
          "legendFormat": "Knots detected/sec"
        }]
      },
      {
        "title": "Cognitive Events by Type",
        "type": "bargauge",
        "targets": [
          {
            "expr": "tcs_events_h0_total",
            "legendFormat": "H₀ Events"
          },
          {
            "expr": "tcs_events_h1_total",
            "legendFormat": "H₁ Events"
          },
          {
            "expr": "tcs_events_h2_total",
            "legendFormat": "H₂ Events"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [{
          "expr": "tcs_memory_usage_bytes / 1024 / 1024 / 1024",
          "legendFormat": "Memory (GB)"
        }]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [{
          "expr": "tcs_gpu_utilization_percent",
          "legendFormat": "GPU Usage %"
        }]
      },
      {
        "title": "Consensus Success Rate",
        "type": "stat",
        "targets": [{
          "expr": "rate(tcs_consensus_accepts_total[5m]) / rate(tcs_consensus_proposals_total[5m]) * 100",
          "legendFormat": "Consensus Success %"
        }]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}"#;

/// Generate docker-compose configuration for development
pub fn docker_compose_config() -> String {
    format!(r#"
version: '3.8'

services:
  tcs-node-1:
    build: .
    ports:
      - "8081:8080"
      - "9091:9090"
      - "50051:50051"
    environment:
      - TCS_NODE_ID=node-1
      - RUST_LOG=info
    volumes:
      - ./config:/etc/tcs:ro
      - tcs-data-1:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  tcs-node-2:
    build: .
    ports:
      - "8082:8080"
      - "9092:9090"
      - "50052:50051"
    environment:
      - TCS_NODE_ID=node-2
      - RUST_LOG=info
    volumes:
      - ./config:/etc/tcs:ro
      - tcs-data-2:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/dashboards:/var/lib/grafana/dashboards:ro

volumes:
  tcs-data-1:
  tcs-data-2:
  prometheus-data:
  grafana-data:
"#)
}

/// Health check endpoints for monitoring
pub mod health {
    use warp::Filter;

    pub fn health_routes() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
        warp::path("health")
            .map(|| warp::reply::json(&serde_json::json!({"status": "healthy"})))
    }

    pub fn readiness_routes() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
        warp::path("ready")
            .map(|| warp::reply::json(&serde_json::json!({"status": "ready"})))
    }

    pub fn metrics_routes() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
        warp::path("metrics")
            .map(|| {
                // In a real implementation, this would collect metrics from the metrics registry
                warp::reply::json(&serde_json::json!({
                    "persistence_computations": 0,
                    "knot_detections": 0,
                    "memory_usage_bytes": 0,
                    "gpu_utilization_percent": 0.0
                }))
            })
    }
}

/// Configuration for monitoring and alerting
pub struct MonitoringConfig {
    pub prometheus_endpoint: String,
    pub grafana_endpoint: String,
    pub alert_rules: Vec<AlertRule>,
}

pub struct AlertRule {
    pub name: String,
    pub query: String,
    pub threshold: f64,
    pub severity: AlertSeverity,
}

pub enum AlertSeverity {
    Warning,
    Critical,
}

impl MonitoringConfig {
    pub fn default() -> Self {
        Self {
            prometheus_endpoint: "http://localhost:9090".to_string(),
            grafana_endpoint: "http://localhost:3000".to_string(),
            alert_rules: vec![
                AlertRule {
                    name: "High Memory Usage".to_string(),
                    query: "tcs_memory_usage_bytes > 40 * 1024 * 1024 * 1024".to_string(), // 40GB
                    threshold: 40.0 * 1024.0 * 1024.0 * 1024.0,
                    severity: AlertSeverity::Warning,
                },
                AlertRule {
                    name: "GPU Memory Exhaustion".to_string(),
                    query: "tcs_gpu_memory_usage_bytes / tcs_gpu_memory_total_bytes > 0.9".to_string(),
                    threshold: 0.9,
                    severity: AlertSeverity::Critical,
                },
                AlertRule {
                    name: "Consensus Failure".to_string(),
                    query: "rate(tcs_consensus_accepts_total[5m]) / rate(tcs_consensus_proposals_total[5m]) < 0.5".to_string(),
                    threshold: 0.5,
                    severity: AlertSeverity::Critical,
                },
            ],
        }
    }
}