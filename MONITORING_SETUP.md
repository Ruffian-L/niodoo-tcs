# Niodoo 1000-Cycle Test - Live Monitoring Setup

## Status: ✅ RUNNING

All monitoring services are active and tailing your test logs in real-time.

### Services Running
- **Grafana** (Dashboard UI): http://localhost:3000
- **Loki** (Log aggregation): http://localhost:3100
- **Promtail** (Log collection): http://localhost:9080

### Quick Access

1. **Open Grafana**: Navigate to http://localhost:3000
   - Default login: `admin` / `admin`
   - ⚠️ **Change password immediately after first login**

2. **Add Loki Data Source** (one-time setup):
   - Left sidebar → Connections → Data sources → Add data source
   - Select "Loki"
   - URL: `http://localhost:3100`
   - Click "Save & Test" (should show "Data source is working")

3. **Create Live Dashboard**:
   - Sidebar → Dashboards → New → New Dashboard
   - Add panels with these LogQL queries:

### Useful LogQL Queries

**View all logs**:
```
{job="niodoo_test"}
```

**Learning loop updates**:
```
{job="niodoo_test"} |= "learning loop updated"
```

**Filter by ROUGE scores**:
```
{job="niodoo_test"} |= "rouge="
```

**Errors and warnings**:
```
{job="niodoo_test"} | level="WARN" or level="ERROR"
```

**Retry attempts**:
```
{job="niodoo_test"} |= "retry"
```

**Entropy/topology metrics**:
```
{job="niodoo_test"} |= "entropy_delta" or |= "knot="
```

**ROUGE score wins** (high quality outputs):
```
{job="niodoo_test"} |= "rouge=0.99"
```

### Dashboard Panel Examples

#### 1. Log Stream Panel
- Type: Logs
- Query: `{job="niodoo_test"}`
- Set refresh to **5s** for near-live updates

#### 2. Learning Updates Counter
- Type: Stat
- Query: `count_over_time({job="niodoo_test"} |= "learning loop updated" [1m])`
- Shows learning updates per minute

#### 3. Latest Entropy Value
- Type: Stat
- Query: `topk(1, {job="niodoo_test"} |= "entropy=")`
- Displays current entropy

#### 4. ROUGE Score Trends
- Type: Time series
- Query: Extract numeric values from `|= "rouge="` logs
- Shows quality progression over time

#### 5. Error Rate
- Type: Stat
- Query: `count_over_time({job="niodoo_test"} | level="WARN" [5m])`
- Counts warnings/errors

### Current Test Status

Your test log file: `niodoo_real_integrated/logs/cycle_1000_test_live.log`
- Current lines: ~3200+ and growing
- Service started at: 17:01 UTC
- Test duration so far: ~30-45 minutes (early in 2-3 hour window)

### Key Metrics Being Captured

Based on your log structure, Promtail is extracting:
- **Timestamps**: RFC3339 format
- **Log levels**: INFO, WARN, ERROR
- **Target modules**: e.g., `niodoo_real_integrated::learning`, `niodoo_real_integrated::pipeline`
- **Messages**: Full log content including:
  - Learning loop updates with entropy and ROUGE scores
  - Topology metrics (knot complexity, persistence entropy, spectral gap)
  - Retry attempts and tier escalations
  - Parameter adjustments
  - vLLM errors and recoveries

### Command Reference

**Check service status**:
```bash
ps aux | grep -E "(loki|promtail|grafana)" | grep -v grep
```

**View service logs**:
```bash
tail -f /tmp/loki.log      # Loki logs
tail -f /tmp/promtail.log  # Promtail logs
tail -f /tmp/grafana.log   # Grafana logs
```

**Restart services** (if needed):
```bash
# Kill existing processes
pkill loki promtail grafana-server

# Restart Loki
cd /workspace/Niodoo-Final && /usr/local/bin/loki -config.file=/tmp/loki-config.yml > /tmp/loki.log 2>&1 &

# Restart Promtail
cd /workspace/Niodoo-Final && /usr/local/bin/promtail -config.file=/tmp/promtail-config.yml > /tmp/promtail.log 2>&1 &

# Restart Grafana
cd /workspace/Niodoo-Final && grafana-server --homepath=/usr/share/grafana --config=/etc/grafana/grafana.ini web > /tmp/grafana.log 2>&1 &
```

**Query Loki directly** (for debugging):
```bash
# Get available labels
curl http://localhost:3100/loki/api/v1/labels

# Query logs
curl -G "http://localhost:3100/loki/api/v1/query_range" \
  --data-urlencode 'query={job="niodoo_test"}' \
  --data-urlencode 'limit=10'
```

### Remote Access

If accessing from another machine:

**Option 1: SSH Tunnel**
```bash
ssh -L 3000:localhost:3000 user@your-server
```
Then access http://localhost:3000 on your local machine

**Option 2: ngrok (public URL)**
```bash
# Install ngrok
curl -sSL https://ngrok.com/install.sh | bash

# Create tunnel
ngrok http 3000
```
⚠️ Note: This exposes Grafana publicly. Add authentication!

### Notes

- Rust compile warnings are filtered out to reduce noise
- Log ingestion started from the current state (won't show historical logs before service start)
- Services will keep running in background until stopped
- Storage: ~50GB free space available
- All configs are in `/tmp/` directory

### Next Steps

1. Access Grafana at http://localhost:3000
2. Change default password
3. Add Loki data source
4. Create your dashboard with the queries above
5. Watch your test metrics update live!

The test continues running independently; monitoring won't affect it.

