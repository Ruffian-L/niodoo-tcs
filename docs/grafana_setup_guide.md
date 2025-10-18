# Grafana Dashboard Setup Guide for Silicon Synapse

This guide provides step-by-step instructions for setting up Grafana dashboards to visualize Silicon Synapse monitoring data.

## Prerequisites

- Prometheus server running and scraping Silicon Synapse metrics
- Grafana server installed and running
- Access to Grafana web interface (typically http://localhost:3000)

## Step 1: Configure Prometheus Data Source

1. Open Grafana web interface
2. Navigate to **Configuration** → **Data Sources**
3. Click **Add data source**
4. Select **Prometheus**
5. Configure the data source:
   - **Name**: `Prometheus`
   - **URL**: `http://localhost:9090`
   - **Access**: Server (default)
6. Click **Save & Test**
7. Verify the connection is successful

## Step 2: Import Silicon Synapse Dashboard

1. Navigate to **Dashboards** → **Import**
2. Click **Upload JSON file**
3. Select the `grafana_dashboard.json` file from the config directory
4. Configure the import:
   - **Name**: `Silicon Synapse Hardware Monitoring`
   - **Folder**: `Niodoo Monitoring`
   - **Prometheus**: Select the Prometheus data source created in Step 1
5. Click **Import**

## Step 3: Configure Alerting (Optional)

1. Navigate to **Configuration** → **Alerting**
2. Click **New alert rule**
3. Configure alerts based on the rules in `silicon_synapse_alerts.yml`:
   - **High GPU Temperature**: `gpu_temperature_celsius > 85`
   - **High Inference Latency**: `histogram_quantile(0.95, rate(inference_ttft_milliseconds_bucket[5m])) > 1000`
   - **Critical Anomaly**: `rate(anomalies_detected_total{severity="critical"}[5m]) > 0`

## Step 4: Customize Dashboard

### Panel Configuration

The dashboard includes the following panels:

1. **Hardware Metrics Row**:
   - GPU Temperature (Celsius)
   - GPU Power Consumption (Watts)
   - GPU Utilization (Percentage)
   - VRAM Usage (Bytes)

2. **Inference Performance Row**:
   - Time To First Token (TTFT) - P50 and P95
   - Time Per Output Token (TPOT) - P50 and P95
   - Token Generation Throughput (TPS)
   - Active Inference Requests

3. **Model Internal State Row**:
   - Model Softmax Entropy
   - Activation Sparsity (Percentage)

4. **Anomaly Detection Row**:
   - Anomaly Detection Rate
   - Inference Error Rate

### Time Range Configuration

- **Default**: Last 1 hour
- **Available**: 5m, 1h, 24h, 7d
- **Auto-refresh**: 5 seconds

### Color Thresholds

- **Green**: Normal operation
- **Yellow**: Warning threshold (80% of max)
- **Red**: Critical threshold (90% of max)

## Step 5: PromQL Queries Reference

### Hardware Metrics

```promql
# GPU Temperature
gpu_temperature_celsius{device="nvidia_0"}

# GPU Power Consumption
gpu_power_watts{device="nvidia_0"}

# GPU Utilization
gpu_utilization_percent{device="nvidia_0"}

# VRAM Usage
vram_used_bytes{device="nvidia_0"}
vram_total_bytes{device="nvidia_0"}

# CPU Utilization
cpu_utilization_percent

# RAM Usage
ram_used_bytes
ram_total_bytes
```

### Inference Metrics

```promql
# TTFT Percentiles
histogram_quantile(0.50, rate(inference_ttft_milliseconds_bucket[5m]))
histogram_quantile(0.95, rate(inference_ttft_milliseconds_bucket[5m]))

# TPOT Percentiles
histogram_quantile(0.50, rate(inference_tpot_milliseconds_bucket[5m]))
histogram_quantile(0.95, rate(inference_tpot_milliseconds_bucket[5m]))

# Throughput
rate(tokens_generated_total[1m])

# Active Requests
active_requests

# Error Rate
rate(inference_errors_total[5m])
```

### Model Metrics

```promql
# Softmax Entropy
model_softmax_entropy{layer="output"}

# Activation Sparsity
activation_sparsity{layer="layer_12"}

# Activation Magnitude
activation_magnitude{layer="layer_12"}
```

### Anomaly Metrics

```promql
# Anomaly Detection Rate
rate(anomalies_detected_total[5m])

# Anomaly Count by Type
anomalies_detected_total

# Anomaly Count by Severity
anomalies_detected_total{severity="critical"}
```

## Step 6: Troubleshooting

### Common Issues

1. **No Data in Panels**:
   - Verify Prometheus is scraping Silicon Synapse metrics
   - Check that Silicon Synapse is running and exposing metrics
   - Verify the Prometheus data source URL is correct

2. **Missing Metrics**:
   - Ensure all Silicon Synapse components are enabled in configuration
   - Check that hardware monitoring is working (GPU drivers, etc.)
   - Verify inference pipeline is generating telemetry events

3. **Dashboard Not Loading**:
   - Check Grafana logs for errors
   - Verify the JSON file is valid
   - Ensure all required data sources are configured

### Debugging Steps

1. **Check Prometheus Targets**:
   - Navigate to Prometheus → Status → Targets
   - Verify `niodoo-silicon-synapse` target is UP

2. **Check Silicon Synapse Metrics**:
   - Visit `http://localhost:9090/metrics`
   - Verify metrics are being exposed

3. **Check Grafana Data Source**:
   - Test the Prometheus data source connection
   - Run a simple query like `up` to verify connectivity

## Step 7: Advanced Configuration

### Custom Panels

To add custom panels:

1. Click **Add panel** on the dashboard
2. Select **Time series** visualization
3. Configure the query using PromQL
4. Set appropriate thresholds and colors
5. Save the panel

### Dashboard Variables

To add dashboard variables:

1. Go to **Dashboard Settings** → **Variables**
2. Add variables for:
   - Device selection
   - Model selection
   - Time range
3. Use variables in panel queries: `gpu_temperature_celsius{device="$device"}`

### Alerting Rules

To configure alerting:

1. Go to **Configuration** → **Alerting** → **Alert rules**
2. Create rules based on the provided alert definitions
3. Configure notification channels (email, Slack, etc.)
4. Set appropriate thresholds and evaluation intervals

## Step 8: Performance Optimization

### Dashboard Performance

- Use appropriate time ranges (shorter ranges for real-time monitoring)
- Limit the number of series in queries
- Use recording rules for expensive queries
- Enable query caching where appropriate

### Prometheus Performance

- Configure appropriate scrape intervals
- Use Prometheus recording rules for complex queries
- Monitor Prometheus resource usage
- Consider federation for large deployments

## Support

For issues with the Silicon Synapse monitoring system:

1. Check the Silicon Synapse logs
2. Verify configuration files
3. Test individual components
4. Review Prometheus and Grafana documentation

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Query Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)







