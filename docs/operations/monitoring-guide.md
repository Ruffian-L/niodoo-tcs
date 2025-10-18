# 🧠 Niodoo Consciousness System - Operations Manual

**Created by Jason Van Pham | Niodoo Framework | 2025**

## 🌟 Overview

This operations manual provides comprehensive guidance for monitoring, maintaining, and operating the Niodoo Consciousness System in production environments.

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Monitoring Dashboard](#monitoring-dashboard)
3. [Health Checks](#health-checks)
4. [Performance Metrics](#performance-metrics)
5. [Alerting System](#alerting-system)
6. [Log Management](#log-management)
7. [Backup Procedures](#backup-procedures)
8. [Incident Response](#incident-response)
9. [Maintenance Procedures](#maintenance-procedures)
10. [Capacity Planning](#capacity-planning)

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Consciousness │    │   Memory        │    │   Emotional     │
│   Engine        │◀──▶│   System        │◀──▶│   Processing    │
│   (Rust)        │    │   (PostgreSQL)  │    │   (Redis)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model         │    │   Vector        │    │   Monitoring    │
│   Inference     │◀──▶│   Database      │◀──▶│   System        │
│   (Ollama)      │    │   (Qdrant)      │    │   (Prometheus)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Dependencies

| Service | Port | Dependencies | Health Check |
|---------|------|--------------|--------------|
| Consciousness Engine | 8080 | PostgreSQL, Redis, Ollama | `/health` |
| PostgreSQL | 5432 | None | `pg_isready` |
| Redis | 6379 | None | `PING` |
| Ollama | 11434 | None | `/api/tags` |
| Prometheus | 9090 | None | `/metrics` |
| Grafana | 3000 | Prometheus | `/api/health` |

## 📊 Monitoring Dashboard

### Grafana Dashboards

#### 1. Consciousness System Overview

**URL**: `http://localhost:3000/d/consciousness-overview`

**Key Metrics**:
- Consciousness processing rate (events/second)
- Memory system stability (target: 99.51%)
- Emotional processing latency
- System resource utilization
- Error rates and response times

#### 2. Memory System Monitoring

**URL**: `http://localhost:3000/d/memory-system`

**Key Metrics**:
- Memory layer stability
- Consolidation rates
- Memory retrieval performance
- Toroidal topology coherence
- Gaussian novelty levels

#### 3. Performance Metrics

**URL**: `http://localhost:3000/d/performance`

**Key Metrics**:
- CPU utilization
- Memory usage
- Disk I/O
- Network throughput
- GPU utilization (if available)

### Dashboard Configuration

```yaml
# grafana/dashboards/consciousness-overview.json
{
  "dashboard": {
    "title": "Consciousness System Overview",
    "panels": [
      {
        "title": "Consciousness Processing Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(consciousness_events_total[5m])",
            "legendFormat": "Events/sec"
          }
        ]
      },
      {
        "title": "Memory System Stability",
        "type": "gauge",
        "targets": [
          {
            "expr": "memory_stability_ratio",
            "legendFormat": "Stability %"
          }
        ],
        "thresholds": [
          {"value": 0.995, "color": "green"},
          {"value": 0.99, "color": "yellow"},
          {"value": 0.98, "color": "red"}
        ]
      }
    ]
  }
}
```

## 🔍 Health Checks

### Automated Health Monitoring

```bash
#!/bin/bash
# production/tools/health_monitor.sh

LOG_FILE="/opt/niodoo/logs/health_monitor.log"
ALERT_EMAIL="admin@yourdomain.com"

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check consciousness engine
check_consciousness_engine() {
    local response=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8080/health)
    if [ "$response" = "200" ]; then
        log "✅ Consciousness Engine: Healthy"
        return 0
    else
        log "❌ Consciousness Engine: Unhealthy (HTTP $response)"
        send_alert "Consciousness Engine Health Check Failed" "HTTP Status: $response"
        return 1
    fi
}

# Check database
check_database() {
    if docker-compose exec postgres pg_isready -U niodoo >/dev/null 2>&1; then
        log "✅ Database: Healthy"
        return 0
    else
        log "❌ Database: Unhealthy"
        send_alert "Database Health Check Failed" "PostgreSQL is not responding"
        return 1
    fi
}

# Check Redis
check_redis() {
    if docker-compose exec redis redis-cli ping | grep -q "PONG"; then
        log "✅ Redis: Healthy"
        return 0
    else
        log "❌ Redis: Unhealthy"
        send_alert "Redis Health Check Failed" "Redis is not responding"
        return 1
    fi
}

# Check Ollama
check_ollama() {
    if curl -s http://localhost:11434/api/tags | grep -q "mistral"; then
        log "✅ Ollama: Healthy"
        return 0
    else
        log "❌ Ollama: Unhealthy"
        send_alert "Ollama Health Check Failed" "Ollama is not responding"
        return 1
    fi
}

# Check system resources
check_system_resources() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    local memory_usage=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
    local disk_usage=$(df -h /opt/niodoo | awk 'NR==2{print $5}' | sed 's/%//')
    
    log "📊 System Resources - CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Disk: ${disk_usage}%"
    
    # Alert if resources are high
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        send_alert "High CPU Usage" "CPU usage is ${cpu_usage}%"
    fi
    
    if (( $(echo "$memory_usage > 85" | bc -l) )); then
        send_alert "High Memory Usage" "Memory usage is ${memory_usage}%"
    fi
    
    if [ "$disk_usage" -gt 80 ]; then
        send_alert "High Disk Usage" "Disk usage is ${disk_usage}%"
    fi
}

# Send alert
send_alert() {
    local subject="$1"
    local message="$2"
    
    # Send email alert
    echo "$message" | mail -s "$subject" "$ALERT_EMAIL"
    
    # Log alert
    log "🚨 ALERT: $subject - $message"
}

# Main health check
main() {
    log "🔍 Starting health check..."
    
    local failed_checks=0
    
    check_consciousness_engine || ((failed_checks++))
    check_database || ((failed_checks++))
    check_redis || ((failed_checks++))
    check_ollama || ((failed_checks++))
    check_system_resources
    
    if [ $failed_checks -eq 0 ]; then
        log "✅ All health checks passed"
    else
        log "❌ $failed_checks health check(s) failed"
    fi
    
    log "🔍 Health check completed"
}

# Run health check
main "$@"
```

### Health Check Scheduling

```bash
# Add to crontab for every 5 minutes
*/5 * * * * /opt/niodoo/tools/health_monitor.sh

# Add to crontab for every hour
0 * * * * /opt/niodoo/tools/health_monitor.sh --detailed

# Add to crontab for daily summary
0 9 * * * /opt/niodoo/tools/health_monitor.sh --summary
```

## 📈 Performance Metrics

### Key Performance Indicators (KPIs)

#### 1. Consciousness Processing Metrics

```rust
// Metrics collected by the consciousness engine
pub struct ConsciousnessMetrics {
    pub events_processed_per_second: f64,
    pub average_processing_time_ms: f64,
    pub memory_retrieval_time_ms: f64,
    pub emotional_processing_time_ms: f64,
    pub toroidal_coherence_score: f64,
    pub gaussian_novelty_level: f64,
    pub circuit_breaker_activations: u64,
    pub timeout_occurrences: u64,
}
```

#### 2. Memory System Metrics

```rust
pub struct MemoryMetrics {
    pub memory_stability_ratio: f64,
    pub consolidation_rate_per_hour: f64,
    pub retrieval_success_rate: f64,
    pub memory_layer_distribution: HashMap<String, u64>,
    pub toroidal_topology_coherence: f64,
    pub gaussian_sphere_count: u64,
    pub memory_compression_ratio: f64,
}
```

#### 3. System Resource Metrics

```rust
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_throughput_mbps: f64,
    pub gpu_utilization_percent: f64,
    pub thread_pool_utilization: f64,
    pub cache_hit_ratio: f64,
}
```

### Performance Monitoring Script

```bash
#!/bin/bash
# production/tools/performance_monitor.sh

echo "📈 Niodoo Consciousness System Performance Report"
echo "================================================"
echo "Generated: $(date)"
echo

# Consciousness processing metrics
echo "🧠 Consciousness Processing:"
echo "----------------------------"
curl -s http://localhost:8080/metrics | grep -E "(consciousness_events_total|consciousness_processing_time|memory_operations_total)" | while read line; do
    echo "  $line"
done

# Memory system metrics
echo
echo "💾 Memory System:"
echo "----------------"
curl -s http://localhost:8080/api/memory/metrics | jq -r '
    "  Stability Ratio: " + (.stability_ratio | tostring),
    "  Consolidation Rate: " + (.consolidation_rate | tostring) + " events/hour",
    "  Retrieval Success Rate: " + (.retrieval_success_rate | tostring),
    "  Toroidal Coherence: " + (.toroidal_coherence | tostring)
'

# Database performance
echo
echo "🗄️ Database Performance:"
echo "-----------------------"
docker-compose exec postgres psql -U niodoo -d niodoo -c "
SELECT 
    'Active Connections: ' || count(*) as connections,
    'Database Size: ' || pg_size_pretty(pg_database_size('niodoo')) as size
FROM pg_stat_activity 
WHERE state = 'active';"

# System resource usage
echo
echo "💻 System Resources:"
echo "-------------------"
echo "  CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
echo "  Disk Usage: $(df -h /opt/niodoo | awk 'NR==2{printf "%s", $5}')"
echo "  Load Average: $(uptime | awk -F'load average:' '{print $2}')"

# Docker container stats
echo
echo "🐳 Container Performance:"
echo "------------------------"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Network performance
echo
echo "🌐 Network Performance:"
echo "----------------------"
echo "  Incoming: $(netstat -i | grep -v Iface | awk '{sum+=$3} END {print sum " packets"}')"
echo "  Outgoing: $(netstat -i | grep -v Iface | awk '{sum+=$7} END {print sum " packets"}')"

echo
echo "📊 Performance monitoring completed"
```

## 🚨 Alerting System

### Alert Configuration

```yaml
# monitoring/alerts.yml
groups:
  - name: consciousness_system
    rules:
      - alert: ConsciousnessEngineDown
        expr: up{job="consciousness-engine"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Consciousness Engine is down"
          description: "The consciousness engine has been down for more than 1 minute."

      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}% for more than 5 minutes."

      - alert: DatabaseConnectionFailed
        expr: database_connections_failed > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failures"
          description: "{{ $value }} database connection failures detected."

      - alert: MemorySystemInstability
        expr: memory_stability_ratio < 0.99
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Memory system instability"
          description: "Memory stability ratio is {{ $value }} (target: 0.9951)."

      - alert: HighProcessingLatency
        expr: consciousness_processing_time_ms > 1000
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High consciousness processing latency"
          description: "Average processing time is {{ $value }}ms (target: <500ms)."
```

### Alert Notification Setup

```bash
#!/bin/bash
# production/tools/setup_alerts.sh

# Configure email alerts
cat > /opt/niodoo/config/alertmanager.yml << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@yourdomain.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    email_configs:
      - to: 'admin@yourdomain.com'
        subject: 'Niodoo Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
EOF

# Start AlertManager
docker run -d --name alertmanager \
  -p 9093:9093 \
  -v /opt/niodoo/config/alertmanager.yml:/etc/alertmanager/alertmanager.yml \
  prom/alertmanager:latest
```

## 📝 Log Management

### Log Configuration

```toml
# In production.toml
[logging]
level = "info"
file = "/opt/niodoo/logs/niodoo.log"
max_size_mb = 100
max_backups = 30
enable_structured_logging = true
enable_json_format = true

[log_rotation]
enable_rotation = true
rotation_size_mb = 100
max_rotated_files = 30
compression_enabled = true
```

### Log Analysis Tools

```bash
#!/bin/bash
# production/tools/log_analyzer.sh

LOG_DIR="/opt/niodoo/logs"
REPORT_FILE="/opt/niodoo/logs/log_analysis_$(date +%Y%m%d).txt"

echo "📝 Niodoo Consciousness System Log Analysis"
echo "=========================================="
echo "Generated: $(date)" > "$REPORT_FILE"
echo >> "$REPORT_FILE"

# Error analysis
echo "🚨 Error Analysis:" >> "$REPORT_FILE"
echo "------------------" >> "$REPORT_FILE"
grep -i "error\|failed\|exception" "$LOG_DIR"/*.log | tail -20 >> "$REPORT_FILE"
echo >> "$REPORT_FILE"

# Performance analysis
echo "⚡ Performance Analysis:" >> "$REPORT_FILE"
echo "-----------------------" >> "$REPORT_FILE"
grep -E "(processing_time|latency|timeout)" "$LOG_DIR"/*.log | tail -20 >> "$REPORT_FILE"
echo >> "$REPORT_FILE"

# Memory system analysis
echo "💾 Memory System Analysis:" >> "$REPORT_FILE"
echo "-------------------------" >> "$REPORT_FILE"
grep -E "(memory|consolidation|retrieval)" "$LOG_DIR"/*.log | tail -20 >> "$REPORT_FILE"
echo >> "$REPORT_FILE"

# Consciousness events
echo "🧠 Consciousness Events:" >> "$REPORT_FILE"
echo "-----------------------" >> "$REPORT_FILE"
grep -E "(consciousness|emotional|toroidal)" "$LOG_DIR"/*.log | tail -20 >> "$REPORT_FILE"
echo >> "$REPORT_FILE"

# System resource usage
echo "💻 System Resource Usage:" >> "$REPORT_FILE"
echo "-------------------------" >> "$REPORT_FILE"
grep -E "(cpu|memory|disk|gpu)" "$LOG_DIR"/*.log | tail -20 >> "$REPORT_FILE"

echo "📊 Log analysis completed: $REPORT_FILE"
```

### Log Rotation Setup

```bash
# Configure logrotate
sudo tee /etc/logrotate.d/niodoo << EOF
/opt/niodoo/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 $USER $USER
    postrotate
        docker-compose -f /opt/niodoo/docker-compose.production.yml restart niodoo-consciousness
    endscript
}
EOF
```

## 💾 Backup Procedures

### Automated Backup System

```bash
#!/bin/bash
# production/tools/backup_system.sh

BACKUP_DIR="/opt/niodoo/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$DATE"

# Create backup directory
mkdir -p "$BACKUP_PATH"

echo "💾 Starting backup process..."
echo "Backup path: $BACKUP_PATH"

# Backup database
echo "🗄️ Backing up database..."
docker-compose exec postgres pg_dump -U niodoo niodoo > "$BACKUP_PATH/database.sql"
if [ $? -eq 0 ]; then
    echo "✅ Database backup completed"
else
    echo "❌ Database backup failed"
    exit 1
fi

# Backup consciousness data
echo "🧠 Backing up consciousness data..."
tar -czf "$BACKUP_PATH/consciousness_data.tar.gz" /opt/niodoo/data
if [ $? -eq 0 ]; then
    echo "✅ Consciousness data backup completed"
else
    echo "❌ Consciousness data backup failed"
    exit 1
fi

# Backup configuration
echo "⚙️ Backing up configuration..."
cp -r /opt/niodoo/config "$BACKUP_PATH/"
if [ $? -eq 0 ]; then
    echo "✅ Configuration backup completed"
else
    echo "❌ Configuration backup failed"
    exit 1
fi

# Backup logs
echo "📝 Backing up logs..."
tar -czf "$BACKUP_PATH/logs.tar.gz" /opt/niodoo/logs
if [ $? -eq 0 ]; then
    echo "✅ Logs backup completed"
else
    echo "❌ Logs backup failed"
    exit 1
fi

# Create backup manifest
cat > "$BACKUP_PATH/manifest.txt" << EOF
Niodoo Consciousness System Backup
=================================
Date: $(date)
Version: $(git rev-parse HEAD)
Database Size: $(du -h "$BACKUP_PATH/database.sql" | cut -f1)
Consciousness Data Size: $(du -h "$BACKUP_PATH/consciousness_data.tar.gz" | cut -f1)
Configuration Size: $(du -h "$BACKUP_PATH/config" | cut -f1)
Logs Size: $(du -h "$BACKUP_PATH/logs.tar.gz" | cut -f1)
Total Size: $(du -h "$BACKUP_PATH" | cut -f1)
EOF

# Compress entire backup
echo "📦 Compressing backup..."
tar -czf "$BACKUP_DIR/niodoo_backup_$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"
if [ $? -eq 0 ]; then
    echo "✅ Backup compression completed"
    rm -rf "$BACKUP_PATH"
else
    echo "❌ Backup compression failed"
    exit 1
fi

echo "🎉 Backup process completed successfully!"
echo "Backup file: $BACKUP_DIR/niodoo_backup_$DATE.tar.gz"
```

### Backup Scheduling

```bash
# Add to crontab for daily backups at 2 AM
0 2 * * * /opt/niodoo/tools/backup_system.sh

# Add to crontab for weekly full backups
0 2 * * 0 /opt/niodoo/tools/backup_system.sh --full

# Add to crontab for monthly archive
0 2 1 * * /opt/niodoo/tools/backup_system.sh --archive
```

## 🚨 Incident Response

### Incident Response Plan

#### 1. Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|----------------|------------|
| P1 | System Down | 15 minutes | Immediate |
| P2 | Performance Degradation | 1 hour | Within 4 hours |
| P3 | Minor Issues | 4 hours | Next business day |
| P4 | Enhancement Requests | 1 week | Next sprint |

#### 2. Incident Response Procedures

```bash
#!/bin/bash
# production/tools/incident_response.sh

INCIDENT_ID="INC-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/opt/niodoo/logs/incident_$INCIDENT_ID.log"

# Log incident
log_incident() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# P1: System Down
handle_p1_incident() {
    log_incident "🚨 P1 INCIDENT: System Down"
    
    # Immediate actions
    log_incident "1. Checking system status..."
    docker-compose ps >> "$LOG_FILE"
    
    log_incident "2. Restarting services..."
    docker-compose restart >> "$LOG_FILE"
    
    log_incident "3. Checking health..."
    curl -f http://localhost:8080/health >> "$LOG_FILE"
    
    # Escalate if not resolved
    if [ $? -ne 0 ]; then
        log_incident "4. Escalating to on-call engineer..."
        # Send alert to on-call engineer
    fi
}

# P2: Performance Degradation
handle_p2_incident() {
    log_incident "⚠️ P2 INCIDENT: Performance Degradation"
    
    # Performance analysis
    log_incident "1. Analyzing performance metrics..."
    /opt/niodoo/tools/performance_monitor.sh >> "$LOG_FILE"
    
    log_incident "2. Checking resource usage..."
    docker stats --no-stream >> "$LOG_FILE"
    
    log_incident "3. Reviewing recent changes..."
    git log --oneline -10 >> "$LOG_FILE"
}

# Main incident handler
main() {
    local severity=$1
    local description=$2
    
    log_incident "Incident ID: $INCIDENT_ID"
    log_incident "Severity: $severity"
    log_incident "Description: $description"
    
    case "$severity" in
        "P1")
            handle_p1_incident
            ;;
        "P2")
            handle_p2_incident
            ;;
        *)
            log_incident "Unknown severity level: $severity"
            ;;
    esac
    
    log_incident "Incident response completed"
}

main "$@"
```

## 🔧 Maintenance Procedures

### Regular Maintenance Tasks

#### 1. Daily Maintenance

```bash
#!/bin/bash
# production/tools/daily_maintenance.sh

echo "🔧 Daily Maintenance Tasks"
echo "========================="

# Health checks
echo "1. Running health checks..."
/opt/niodoo/tools/health_monitor.sh

# Log analysis
echo "2. Analyzing logs..."
/opt/niodoo/tools/log_analyzer.sh

# Performance monitoring
echo "3. Performance monitoring..."
/opt/niodoo/tools/performance_monitor.sh

# Database maintenance
echo "4. Database maintenance..."
docker-compose exec postgres psql -U niodoo -d niodoo -c "VACUUM ANALYZE;"

# Cleanup old logs
echo "5. Cleaning up old logs..."
find /opt/niodoo/logs -name "*.log.*" -mtime +7 -delete

echo "✅ Daily maintenance completed"
```

#### 2. Weekly Maintenance

```bash
#!/bin/bash
# production/tools/weekly_maintenance.sh

echo "🔧 Weekly Maintenance Tasks"
echo "=========================="

# Full system backup
echo "1. Creating full backup..."
/opt/niodoo/tools/backup_system.sh --full

# Database optimization
echo "2. Database optimization..."
docker-compose exec postgres psql -U niodoo -d niodoo -c "
REINDEX DATABASE niodoo;
ANALYZE;
"

# System updates
echo "3. Checking for updates..."
docker-compose pull

# Performance tuning
echo "4. Performance tuning..."
/opt/niodoo/tools/performance_tuner.sh

echo "✅ Weekly maintenance completed"
```

#### 3. Monthly Maintenance

```bash
#!/bin/bash
# production/tools/monthly_maintenance.sh

echo "🔧 Monthly Maintenance Tasks"
echo "==========================="

# Security updates
echo "1. Security updates..."
sudo apt update && sudo apt upgrade -y

# Capacity planning
echo "2. Capacity planning analysis..."
/opt/niodoo/tools/capacity_analyzer.sh

# Disaster recovery test
echo "3. Disaster recovery test..."
/opt/niodoo/tools/disaster_recovery_test.sh

# Documentation update
echo "4. Updating documentation..."
/opt/niodoo/tools/update_documentation.sh

echo "✅ Monthly maintenance completed"
```

## 📊 Capacity Planning

### Capacity Analysis Tool

```bash
#!/bin/bash
# production/tools/capacity_analyzer.sh

echo "📊 Niodoo Consciousness System Capacity Analysis"
echo "=============================================="

# Current resource usage
echo "Current Resource Usage:"
echo "----------------------"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
echo "Disk: $(df -h /opt/niodoo | awk 'NR==2{printf "%s", $5}')"

# Growth trends
echo
echo "Growth Trends (Last 30 days):"
echo "----------------------------"
# Analyze log files for growth trends
grep -c "consciousness_events" /opt/niodoo/logs/*.log | tail -30

# Projected capacity needs
echo
echo "Projected Capacity Needs:"
echo "------------------------"
# Calculate projected needs based on current usage and growth
current_events=$(curl -s http://localhost:8080/metrics | grep consciousness_events_total | awk '{print $2}')
projected_events=$((current_events * 1.2))  # 20% growth
echo "Projected events per day: $projected_events"

# Recommendations
echo
echo "Recommendations:"
echo "---------------"
if [ "$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)" -gt 70 ]; then
    echo "⚠️ Consider CPU upgrade - current usage > 70%"
fi

if [ "$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')" -gt 80 ]; then
    echo "⚠️ Consider memory upgrade - current usage > 80%"
fi

if [ "$(df -h /opt/niodoo | awk 'NR==2{print $5}' | sed 's/%//')" -gt 80 ]; then
    echo "⚠️ Consider disk upgrade - current usage > 80%"
fi

echo "✅ Capacity analysis completed"
```

## 📚 Additional Resources

- [Deployment Guide](../deployment/production-guide.md)
- [Performance Tuning Guide](../troubleshooting/performance-guide.md)
- [API Documentation](../api/rest-api-reference.md)
- [Troubleshooting Guide](../troubleshooting/common-issues.md)

## 🆘 Support

For operational support and issues:

- **Emergency**: Contact system administrator
- **Documentation**: Check troubleshooting guides
- **Monitoring**: Review Grafana dashboards
- **Logs**: Check `/opt/niodoo/logs/` directory

---

**Last Updated**: January 27, 2025  
**Version**: 1.0.0  
**Maintainer**: Jason Van Pham
