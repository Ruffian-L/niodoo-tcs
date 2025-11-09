# RunPod Startup Testing & Validation Plan

## Overview

This document outlines how to test, validate, and monitor RunPod endpoint startup reliability.

## Test Scenarios

### Scenario 1: Cold Start (Fresh Instance)

**Purpose**: Test startup from scratch with no cached models or CUDA graphs

**Steps**:
1. Start fresh RunPod instance
2. Run bootstrap: `scripts/runpod_bootstrap.sh`
3. Monitor startup sequence
4. Verify all services are healthy

**Expected Results**:
- Qdrant starts in 5-10 seconds
- vLLM model loads in 2-5 minutes
- Main pipeline starts in 10-20 seconds
- Total time: 2-3 minutes

**Success Criteria**:
- All health checks pass
- Services respond to requests
- No errors in logs

---

### Scenario 2: Warm Start (Cached CUDA Graphs)

**Purpose**: Test startup with cached CUDA graphs (faster)

**Steps**:
1. Start RunPod instance (second boot)
2. Run bootstrap: `scripts/runpod_bootstrap.sh`
3. Monitor startup sequence
4. Verify services start faster

**Expected Results**:
- Qdrant starts in 5-10 seconds
- vLLM loads faster (30-60 seconds with cached graphs)
- Main pipeline starts in 10-20 seconds
- Total time: 30-60 seconds

**Success Criteria**:
- Startup time reduced by 50%+
- All health checks pass
- Services respond correctly

---

### Scenario 3: Service Restart

**Purpose**: Test restarting individual services

**Steps**:
1. Start all services
2. Restart Qdrant: `unified_service_manager.sh stop` then `start`
3. Restart vLLM: Stop and start vLLM service
4. Verify services reconnect

**Expected Results**:
- Services restart cleanly
- Dependencies reconnect automatically
- No data loss

**Success Criteria**:
- Services restart without errors
- Connections re-established
- Health checks pass

---

### Scenario 4: Failure Recovery

**Purpose**: Test recovery from service crashes

**Steps**:
1. Start all services
2. Kill Qdrant process: `kill -9 $(cat .service_pids/qdrant.pid)`
3. Verify service manager detects failure
4. Restart service
5. Verify recovery

**Expected Results**:
- Service manager detects failure
- Service restarts automatically (if implemented)
- Health checks recover

**Success Criteria**:
- Failures detected within 30 seconds
- Services recover automatically
- No cascading failures

---

### Scenario 5: Resource Constraints

**Purpose**: Test behavior under resource limits

**Steps**:
1. Start services with limited GPU memory
2. Monitor resource usage
3. Verify graceful degradation

**Expected Results**:
- Services start with available resources
- Clear error messages if resources insufficient
- No crashes from OOM

**Success Criteria**:
- Services handle resource limits gracefully
- Clear error messages
- No unexpected crashes

---

## Validation Checklist

### Pre-Startup Validation

- [ ] Environment variables set correctly
- [ ] Model paths exist and are accessible
- [ ] Required ports are available (6333, 6334, 5001, 5002, 9090)
- [ ] GPU available and accessible
- [ ] Disk space sufficient (models + logs)
- [ ] Python environment activated
- [ ] Rust toolchain available

### Startup Validation

- [ ] Qdrant starts successfully
- [ ] Qdrant health check passes
- [ ] vLLM starts successfully
- [ ] vLLM model loads completely
- [ ] vLLM health check passes (with model verification)
- [ ] Curator vLLM starts (if separate)
- [ ] Main pipeline starts successfully
- [ ] Main pipeline health check passes

### Post-Startup Validation

- [ ] All services respond to requests
- [ ] Qdrant collections accessible
- [ ] vLLM can generate responses
- [ ] Main pipeline processes prompts
- [ ] Health endpoints return correct status
- [ ] Metrics endpoint accessible
- [ ] Logs show no errors

---

## Monitoring & Observability

### Key Metrics to Monitor

1. **Startup Time**
   - Qdrant startup: Target < 30 seconds
   - vLLM model loading: Target < 5 minutes (cold), < 1 minute (warm)
   - Main pipeline startup: Target < 30 seconds
   - Total startup: Target < 6 minutes (cold), < 2 minutes (warm)

2. **Health Check Success Rate**
   - Qdrant: 100%
   - vLLM: 100% (after model loads)
   - Main pipeline: 100%

3. **Service Availability**
   - Uptime: Target > 99%
   - Response time: Target < 1 second (p95)

4. **Resource Usage**
   - GPU memory: Monitor utilization
   - CPU usage: Monitor during startup
   - Disk I/O: Monitor model loading

### Log Locations

- Bootstrap logs: `logs/runpod_bootstrap.log`
- Qdrant logs: `logs/qdrant.log`
- vLLM logs: `logs/vllm.log`
- Curator logs: `logs/vllm_curator.log`
- Main pipeline logs: `logs/niodoo_main.log`
- Service manager logs: Console output

### Health Check Endpoints

- Qdrant: `http://127.0.0.1:6333/health`
- vLLM: `http://127.0.0.1:5001/v1/models`
- Curator vLLM: `http://127.0.0.1:5002/v1/models` (if separate)
- Main Pipeline: `http://127.0.0.1:9090/health`
- Metrics: `http://127.0.0.1:9090/metrics`

---

## Reproducing Failures

### Common Failure Modes

1. **Service Manager Missing**
   - Symptom: "No service manager found" error
   - Reproduce: Run bootstrap without `unified_service_manager.sh`
   - Fix: Ensure service manager exists and is executable

2. **Health Check Timeout**
   - Symptom: Health checks fail even though service is starting
   - Reproduce: Start vLLM and check health immediately
   - Fix: Use extended timeout for vLLM (10 minutes)

3. **Model Not Loaded**
   - Symptom: vLLM responds but can't process requests
   - Reproduce: Check `/v1/models` before model fully loads
   - Fix: Verify model list is non-empty in health check

4. **Port Conflicts**
   - Symptom: Service fails to bind to port
   - Reproduce: Start service when port already in use
   - Fix: Check port availability before starting

5. **Missing Dependencies**
   - Symptom: Service starts but can't connect to dependencies
   - Reproduce: Start service before dependencies are ready
   - Fix: Implement dependency waiting logic

---

## Automated Testing

### Test Script

Create `scripts/test_startup.sh`:

```bash
#!/bin/bash
set -e

echo "Testing RunPod startup sequence..."

# Test 1: Service manager exists
if [[ ! -x "unified_service_manager.sh" ]]; then
    echo "FAIL: unified_service_manager.sh not found or not executable"
    exit 1
fi

# Test 2: Start services
echo "Starting services..."
./unified_service_manager.sh start

# Test 3: Health checks
echo "Running health checks..."
./scripts/runpod_bootstrap.sh --skip-packages --skip-build --skip-model-download --skip-qdrant --skip-ollama

# Test 4: Verify endpoints
curl -f http://127.0.0.1:6333/health || exit 1
curl -f http://127.0.0.1:5001/v1/models || exit 1
curl -f http://127.0.0.1:9090/health || exit 1

echo "SUCCESS: All tests passed"
```

### Continuous Monitoring

Set up monitoring to track:
- Startup success rate
- Startup time percentiles
- Health check failures
- Service crashes
- Resource usage spikes

---

## Success Criteria

### Startup Reliability

- ✅ **Success Rate**: > 95% of startups succeed
- ✅ **Startup Time**: < 6 minutes (cold), < 2 minutes (warm)
- ✅ **Health Checks**: 100% accuracy (no false positives/negatives)

### Service Availability

- ✅ **Uptime**: > 99% after startup
- ✅ **Response Time**: < 1 second (p95)
- ✅ **Error Rate**: < 1%

### Debugging

- ✅ **Error Messages**: Clear and actionable
- ✅ **Logs**: Comprehensive and accessible
- ✅ **Monitoring**: Real-time visibility

---

## Next Steps

1. Implement automated test script
2. Set up continuous monitoring
3. Create alerting for failures
4. Document troubleshooting procedures
5. Create runbook for common issues



