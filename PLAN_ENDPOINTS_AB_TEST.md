# PLAN: Get All Endpoints Online, Smoke Test, and Run Topology A/B Test

## Objective
1. Get ALL endpoints online and verified
2. Smoke test all endpoints comprehensively
3. Run A/B test to prove AI uses topology for understanding
4. Execute REAL tests (no stubs, no fake data)

## Endpoints Required

### External Services
1. **Qdrant** - Port 6333 (HTTP), 6334 (gRPC)
2. **vLLM Generation** (Qwen 3 Coder) - Port 5001
3. **vLLM Curator** (Qwen 2.5 Topology) - Port 5002 (or 5001 if shared)

### NIODOO Services
4. **Main Pipeline Server** - Port 9090
   - `/health` - Health check
   - `/ready` - Readiness probe
   - `/metrics` - Prometheus metrics
5. **RL Server** - Port 8080
   - `/health` - Health check
   - `/rl/evaluate` - Code evaluation

## Execution Plan

### Phase 1: Start All Services
1. Start Qdrant (Docker)
2. Start vLLM Generation (port 5001)
3. Start vLLM Curator (port 5002)
4. Start Main Pipeline Server (port 9090)
5. Start RL Server (port 8080)

### Phase 2: Verify All Endpoints
- Run `scripts/verify_all_endpoints.sh` to check all endpoints are responding
- Verify each endpoint individually with curl
- Check Prometheus metrics are being collected

### Phase 3: Smoke Test All Endpoints
- Test each endpoint with real requests
- Verify response formats
- Check error handling
- Validate metrics collection

### Phase 4: Run Topology A/B Test
- Baseline: `configs/topology_enabled.json` (hybrid mode, RCE enabled, nTokens enabled)
- Treatment: `configs/topology_disabled.json` (baseline mode, RCE disabled, nTokens bypassed)
- Run `scripts/run_topology_ab_test.sh`
- Analyze results for topology impact

### Phase 5: Validate Results
- Check topology_impact field (positive/negative/neutral/inconclusive)
- Verify persistence_entropy_difference (higher = richer structure)
- Verify quality_difference_pct (higher = better understanding)
- Verify beta_meta_difference (RCE breakthrough detection)

## Success Criteria
- ✅ All endpoints online and responding
- ✅ All smoke tests passing
- ✅ A/B test completes successfully
- ✅ Topology impact is measurable and statistically significant
- ✅ Results prove topology understanding (positive impact)


