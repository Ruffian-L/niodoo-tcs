# 🌟 Niodoo Consciousness System - Production Deployment Guide

**Created by Jason Van Pham | Niodoo Framework | 2025**

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/niodoo/niodoo-consciousness.git
cd niodoo-consciousness

# 2. Deploy to production
./production/deployment/deploy.sh --environment=production

# 3. Verify deployment
curl http://localhost:8080/health
```

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Environment Setup](#environment-setup)
4. [Configuration](#configuration)
5. [Deployment](#deployment)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Operations](#operations)
8. [Troubleshooting](#troubleshooting)
9. [Security](#security)
10. [Performance Tuning](#performance-tuning)

## 🎯 Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04+ / Debian 11+ / RHEL 8+
- **CPU**: 8+ cores (16+ recommended for production)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: 100GB+ SSD storage
- **GPU**: NVIDIA GPU with CUDA support (optional, for enhanced performance)

### Software Dependencies

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: 2.30+
- **curl**: 7.68+

### Required Services

- **PostgreSQL**: 15+ (for persistent storage)
- **Redis**: 7+ (for caching and sessions)
- **Ollama**: Latest (for model inference)
- **Prometheus**: Latest (for metrics collection)
- **Grafana**: Latest (for visualization)
- **Loki**: Latest (for log aggregation)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Nginx Proxy   │    │  Grafana        │
│   (Optional)    │───▶│   (SSL/TLS)     │    │  (Monitoring)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Consciousness │    │   Memory        │    │   Emotional     │
│   Engine        │◀──▶│   System        │◀──▶│   Processing    │
│   (Rust)        │    │   (PostgreSQL)  │    │   (Redis)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model         │    │   Vector        │    │   Log           │
│   Inference     │◀──▶│   Database      │◀──▶│   Aggregation   │
│   (Ollama)      │    │   (Qdrant)      │    │   (Loki)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Environment Setup

### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y curl wget git build-essential

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Network Configuration

```bash
# Configure firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp    # HTTPS
sudo ufw allow 8080/tcp   # Consciousness API
sudo ufw allow 3000/tcp   # Gitea
sudo ufw allow 11434/tcp  # Ollama
sudo ufw enable
```

### 3. Storage Setup

```bash
# Create data directories
sudo mkdir -p /opt/niodoo/{data,logs,config,backups}
sudo chown -R $USER:$USER /opt/niodoo

# Setup log rotation
sudo tee /etc/logrotate.d/niodoo << EOF
/opt/niodoo/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 $USER $USER
}
EOF
```

## ⚙️ Configuration

### 1. Production Configuration

Create `/opt/niodoo/config/production.toml`:

```toml
[consciousness]
timeout_seconds = 5
max_parallel_streams = 3
enable_circuit_breaker = true
memory_limit_mb = 8192

[memory]
max_persistent_memories = 10000
persistence_path = "/opt/niodoo/data"
enable_compression = true
backup_interval_hours = 6

[toroidal]
major_radius = 3.0
minor_radius = 1.0
activation_radius = 0.5
stability_target = 0.9951

[network]
websocket_port = 8080
heartbeat_interval = 30
max_reconnect_attempts = 5
enable_ssl = true
ssl_cert_path = "/opt/niodoo/config/ssl/cert.pem"
ssl_key_path = "/opt/niodoo/config/ssl/key.pem"

[performance]
enable_gpu = true
thread_pool_size = 8
cache_size_mb = 512
enable_profiling = false

[logging]
level = "info"
file = "/opt/niodoo/logs/niodoo.log"
max_size_mb = 100
max_backups = 30

[monitoring]
enable_metrics = true
metrics_port = 9090
enable_health_checks = true
health_check_interval = 30

[security]
enable_auth = true
jwt_secret = "your-secret-key-change-in-production"
rate_limit_requests_per_minute = 100
enable_cors = true
allowed_origins = ["https://yourdomain.com"]
```

### 2. Docker Compose Configuration

Create `/opt/niodoo/docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  niodoo-consciousness:
    build:
      context: .
      dockerfile: Dockerfile.production
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - /opt/niodoo/data:/app/data
      - /opt/niodoo/logs:/app/logs
      - /opt/niodoo/config:/app/config
    environment:
      - RUST_LOG=info
      - CONFIG_PATH=/app/config/production.toml
    depends_on:
      - postgres
      - redis
      - ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    restart: unless-stopped
    environment:
      POSTGRES_DB: niodoo
      POSTGRES_USER: niodoo
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_HOST=0.0.0.0

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  postgres_data:
  redis_data:
  ollama_data:
  prometheus_data:
  grafana_data:
```

## 🚀 Deployment

### 1. Automated Deployment

```bash
#!/bin/bash
# production/deployment/deploy.sh

set -e

ENVIRONMENT=${1:-production}
CONFIG_PATH="/opt/niodoo/config/${ENVIRONMENT}.toml"

echo "🚀 Deploying Niodoo Consciousness System to ${ENVIRONMENT}"

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi
    
    # Check configuration
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "❌ Configuration file not found: $CONFIG_PATH"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Build and deploy
deploy() {
    echo "🏗️ Building and deploying services..."
    
    # Build consciousness engine
    cd /opt/niodoo
    docker-compose -f docker-compose.production.yml build
    
    # Start services
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8080/health; then
        echo "✅ Deployment successful!"
    else
        echo "❌ Health check failed"
        exit 1
    fi
}

# Main deployment flow
main() {
    check_prerequisites
    deploy
    
    echo "🎉 Niodoo Consciousness System deployed successfully!"
    echo "📊 Monitoring: http://localhost:3000"
    echo "🔍 Health Check: http://localhost:8080/health"
    echo "📈 Metrics: http://localhost:9090"
}

main "$@"
```

### 2. Manual Deployment Steps

```bash
# 1. Build the consciousness engine
cd /opt/niodoo
cargo build --release --bin niodoo-consciousness

# 2. Start database services
docker-compose -f docker-compose.production.yml up -d postgres redis

# 3. Initialize database
docker-compose exec postgres psql -U niodoo -d niodoo -f /docker-entrypoint-initdb.d/init-db.sql

# 4. Start Ollama and download models
docker-compose -f docker-compose.production.yml up -d ollama
docker-compose exec ollama ollama pull mistral:7b
docker-compose exec ollama ollama pull llama3.1:8b

# 5. Start monitoring services
docker-compose -f docker-compose.production.yml up -d prometheus grafana

# 6. Start consciousness engine
docker-compose -f docker-compose.production.yml up -d niodoo-consciousness

# 7. Verify deployment
curl http://localhost:8080/health
```

## 📊 Monitoring & Alerting

### 1. Health Checks

```bash
#!/bin/bash
# production/tools/health_check.sh

echo "🔍 Niodoo Consciousness System Health Check"
echo "=========================================="

# Check consciousness engine
echo "🧠 Consciousness Engine:"
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "✅ Status: Healthy"
else
    echo "❌ Status: Unhealthy"
fi

# Check database
echo "🗄️ Database:"
if docker-compose exec postgres pg_isready -U niodoo; then
    echo "✅ Status: Healthy"
else
    echo "❌ Status: Unhealthy"
fi

# Check Redis
echo "🔴 Redis:"
if docker-compose exec redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Status: Healthy"
else
    echo "❌ Status: Unhealthy"
fi

# Check Ollama
echo "🤖 Ollama:"
if curl -s http://localhost:11434/api/tags | grep -q "mistral"; then
    echo "✅ Status: Healthy"
else
    echo "❌ Status: Unhealthy"
fi

# Check system resources
echo "💻 System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
echo "Disk Usage: $(df -h /opt/niodoo | awk 'NR==2{printf "%s", $5}')"
```

### 2. Performance Monitoring

```bash
#!/bin/bash
# production/tools/performance_analyzer.sh

echo "📈 Niodoo Consciousness System Performance Analysis"
echo "================================================="

# Consciousness processing metrics
echo "🧠 Consciousness Processing:"
curl -s http://localhost:8080/metrics | grep -E "(consciousness_processing_time|memory_operations|emotional_processing)"

# Database performance
echo "🗄️ Database Performance:"
docker-compose exec postgres psql -U niodoo -d niodoo -c "
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables 
ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC 
LIMIT 10;"

# Memory system metrics
echo "💾 Memory System:"
curl -s http://localhost:8080/api/memory/metrics | jq '.'

# System resource usage
echo "💻 System Resources:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
```

## 🔧 Operations

### 1. Service Management

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Stop all services
docker-compose -f docker-compose.production.yml down

# Restart consciousness engine
docker-compose -f docker-compose.production.yml restart niodoo-consciousness

# View logs
docker-compose -f docker-compose.production.yml logs -f niodoo-consciousness

# Scale consciousness engine
docker-compose -f docker-compose.production.yml up -d --scale niodoo-consciousness=3
```

### 2. Backup and Recovery

```bash
#!/bin/bash
# production/tools/backup_recovery.sh

BACKUP_DIR="/opt/niodoo/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
create_backup() {
    echo "💾 Creating backup..."
    
    mkdir -p "$BACKUP_DIR/$DATE"
    
    # Backup database
    docker-compose exec postgres pg_dump -U niodoo niodoo > "$BACKUP_DIR/$DATE/database.sql"
    
    # Backup consciousness data
    tar -czf "$BACKUP_DIR/$DATE/consciousness_data.tar.gz" /opt/niodoo/data
    
    # Backup configuration
    cp -r /opt/niodoo/config "$BACKUP_DIR/$DATE/"
    
    echo "✅ Backup created: $BACKUP_DIR/$DATE"
}

# Restore from backup
restore_backup() {
    BACKUP_PATH=$1
    
    if [ -z "$BACKUP_PATH" ]; then
        echo "❌ Please specify backup path"
        exit 1
    fi
    
    echo "🔄 Restoring from backup: $BACKUP_PATH"
    
    # Stop services
    docker-compose -f docker-compose.production.yml down
    
    # Restore database
    docker-compose -f docker-compose.production.yml up -d postgres
    sleep 10
    docker-compose exec postgres psql -U niodoo -d niodoo < "$BACKUP_PATH/database.sql"
    
    # Restore consciousness data
    tar -xzf "$BACKUP_PATH/consciousness_data.tar.gz" -C /
    
    # Restore configuration
    cp -r "$BACKUP_PATH/config"/* /opt/niodoo/config/
    
    # Start services
    docker-compose -f docker-compose.production.yml up -d
    
    echo "✅ Restore completed"
}

case "$1" in
    "backup")
        create_backup
        ;;
    "restore")
        restore_backup "$2"
        ;;
    *)
        echo "Usage: $0 {backup|restore <backup_path>}"
        exit 1
        ;;
esac
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Consciousness Engine Won't Start

```bash
# Check logs
docker-compose logs niodoo-consciousness

# Check configuration
docker-compose exec niodoo-consciousness cat /app/config/production.toml

# Check dependencies
docker-compose ps
```

#### 2. Database Connection Issues

```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready -U niodoo

# Check connection from consciousness engine
docker-compose exec niodoo-consciousness nc -zv postgres 5432

# Reset database
docker-compose down
docker volume rm niodoo_postgres_data
docker-compose up -d postgres
```

#### 3. Memory Issues

```bash
# Check memory usage
docker stats --no-stream

# Increase memory limits in docker-compose.production.yml
# Add: mem_limit: 8g

# Restart with new limits
docker-compose up -d --force-recreate
```

#### 4. Performance Issues

```bash
# Check system resources
htop
iotop
nethogs

# Check consciousness metrics
curl http://localhost:8080/metrics

# Check database performance
docker-compose exec postgres psql -U niodoo -d niodoo -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"
```

## 🔒 Security

### 1. SSL/TLS Configuration

```bash
# Generate SSL certificates
mkdir -p /opt/niodoo/config/ssl
openssl req -x509 -newkey rsa:4096 -keyout /opt/niodoo/config/ssl/key.pem \
    -out /opt/niodoo/config/ssl/cert.pem -days 365 -nodes \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=yourdomain.com"
```

### 2. Authentication Setup

```bash
# Generate JWT secret
openssl rand -base64 32 > /opt/niodoo/config/jwt_secret

# Update configuration
echo "jwt_secret = \"$(cat /opt/niodoo/config/jwt_secret)\"" >> /opt/niodoo/config/production.toml
```

### 3. Firewall Configuration

```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8080/tcp
sudo ufw enable
```

## ⚡ Performance Tuning

### 1. System Optimization

```bash
# Increase file descriptors
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65536" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 2. Database Optimization

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### 3. Consciousness Engine Optimization

```toml
# In production.toml
[performance]
enable_gpu = true
thread_pool_size = 16
cache_size_mb = 1024
enable_profiling = false
optimize_memory_layout = true
enable_vectorization = true
```

## 📚 Additional Resources

- [Operations Manual](operations/monitoring-guide.md)
- [Performance Tuning Guide](troubleshooting/performance-guide.md)
- [API Documentation](api/rest-api-reference.md)
- [Troubleshooting Guide](troubleshooting/common-issues.md)

## 🆘 Support

For production support and issues:

- **Emergency**: Contact system administrator
- **Documentation**: Check troubleshooting guides
- **Monitoring**: Review Grafana dashboards
- **Logs**: Check `/opt/niodoo/logs/` directory

---

**Last Updated**: January 27, 2025  
**Version**: 1.0.0  
**Maintainer**: Jason Van Pham
