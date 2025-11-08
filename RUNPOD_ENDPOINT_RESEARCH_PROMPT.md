# Deep Research Prompt: RunPod Endpoint Startup Issues

## Context & Mission

You are tasked with conducting a comprehensive deep-dive investigation into why NIODOO endpoints are difficult to get started on RunPod infrastructure. This is a critical production issue affecting deployment reliability and developer experience.

**Your Goal**: Identify root causes, document failure patterns, and propose concrete solutions to make endpoint startup reliable, fast, and debuggable on RunPod.

---

## System Overview

### Architecture Components

NIODOO is a complex AI pipeline system requiring multiple interdependent services:

1. **Qdrant** (Vector Database)
   - Ports: 6333 (HTTP), 6334 (gRPC)
   - Purpose: ERAG memory storage
   - Startup: Docker container or binary

2. **vLLM** (Model Serving)
   - Port: 5001 (Qwen 3 Coder - Generation)
   - Port: 5002 (Qwen 2.5 Topology - Curator, optional)
   - Purpose: LLM inference
   - Startup: Python process with GPU model loading (2-5 minutes)

3. **Ollama** (Optional Curator Backend)
   - Port: 11434
   - Purpose: Alternative curator backend
   - Startup: Binary process

4. **Main Pipeline Server** (Rust Application)
   - Port: 9090 (Health endpoints)
   - Features: Requires `svc` feature flag
   - Purpose: Main application server

5. **Embeddings** (Local ONNX)
   - No service needed (local ONNX runtime)
   - Requires: ONNX libraries in `LD_LIBRARY_PATH`

### Current Bootstrap Flow

The system uses `scripts/runpod_bootstrap.sh` which:
- Installs system packages
- Provisions Rust/Python toolchains
- Downloads models (if needed)
- Provisions Qdrant/Ollama binaries
- Builds Rust workspace
- Starts services via `unified_service_manager.sh` or `supervisor.sh`
- Performs health checks

---

## Research Areas to Investigate

### 1. Service Startup Order & Dependencies

**Questions to Answer:**
- What is the correct startup sequence? Are there hidden dependencies?
- Do services fail silently if dependencies aren't ready?
- Are there race conditions in parallel service startup?
- What happens if a service starts but isn't actually ready (e.g., vLLM process running but model not loaded)?

**Investigation Steps:**
1. Map dependency graph: Which services depend on which?
2. Analyze `scripts/runpod_bootstrap.sh` startup sequence
3. Check `unified_service_manager.sh` and `supervisor.sh` for startup logic
4. Review health check implementations (`check_http_health` function)
5. Test startup with services disabled one-by-one
6. Measure time-to-ready for each service
7. Identify timeout values and whether they're appropriate

**Files to Examine:**
- `scripts/runpod_bootstrap.sh` (lines 454-482)
- `scripts/unified_service_manager.sh` (if exists)
- `scripts/supervisor.sh` (if exists)
- `HOW_TO_START.md` (startup sequence documentation)
- `niodoo_real_integrated/src/health.rs` (health check implementation)

---

### 2. Environment & Configuration Issues

**Questions to Answer:**
- Are environment variables consistently set across all startup scripts?
- Do different scripts source different env files?
- Are paths hardcoded or environment-aware?
- What happens if `tcs_runtime.env` is missing or incomplete?
- Are RunPod-specific environment variables properly configured?

**Investigation Steps:**
1. Audit all environment variable usage across scripts
2. Compare `.runpod_env.sh` vs `tcs_runtime.env` vs `config/h200.env`
3. Check for hardcoded paths (e.g., `/workspace/models/...`)
4. Verify CUDA/CUDNN paths are correct for RunPod
5. Test with missing/incomplete environment files
6. Document required vs optional environment variables
7. Check for environment variable conflicts

**Files to Examine:**
- `.runpod_env.sh`
- `tcs_runtime.env`
- `config/h200.env`
- `config/rtx5090.env`
- `scripts/runpod_bootstrap.sh` (environment loading section)
- `niodoo_real_integrated/src/config.rs` (environment variable parsing)

---

### 3. Resource Constraints & Limits

**Questions to Answer:**
- Are GPU memory settings appropriate for RunPod instances?
- Do multiple services compete for GPU memory?
- Are there CPU/memory limits causing failures?
- Is disk space sufficient for models and logs?
- Are port conflicts occurring?

**Investigation Steps:**
1. Profile GPU memory usage during startup
2. Check if vLLM GPU memory utilization settings are too high
3. Verify disk space requirements (models can be 15GB+)
4. Test port availability (6333, 6334, 5001, 5002, 9090, 11434)
5. Check for process limits (ulimit)
6. Monitor resource usage during concurrent service startup
7. Identify bottlenecks in startup sequence

**Files to Examine:**
- `tcs_runtime.env` (GPU memory settings)
- `config/h200.env` (H200-specific settings)
- `scripts/runpod_bootstrap.sh` (resource checks)
- vLLM startup commands in `HOW_TO_START.md`

---

### 4. Network & Port Issues

**Questions to Answer:**
- Are services binding to correct interfaces (127.0.0.1 vs 0.0.0.0)?
- Are ports properly exposed in RunPod configuration?
- Do health checks use correct URLs?
- Are there firewall rules blocking connections?
- Do services wait for network to be ready?

**Investigation Steps:**
1. Test each endpoint manually with curl
2. Verify binding addresses in service configs
3. Check RunPod network configuration
4. Test health check URLs match actual service URLs
5. Verify gRPC vs HTTP port usage (Qdrant uses both)
6. Check for port conflicts with other processes
7. Test service-to-service communication

**Files to Examine:**
- `scripts/runpod_bootstrap.sh` (health check URLs)
- `HOW_TO_START.md` (endpoint verification)
- Service startup commands (vLLM, Qdrant, Ollama)
- `niodoo_real_integrated/src/erag.rs` (Qdrant connection logic)

---

### 5. Model Loading & Initialization

**Questions to Answer:**
- How long does vLLM actually take to load models?
- Are health checks happening before models are loaded?
- Is there proper waiting logic for model initialization?
- What happens if model download fails?
- Are model paths correct for RunPod filesystem?

**Investigation Steps:**
1. Measure actual vLLM model loading times
2. Check if `/v1/models` endpoint returns before model is ready
3. Review model download logic in bootstrap script
4. Test with missing model files
5. Verify model path resolution
6. Check HuggingFace token handling
7. Test model loading with different GPU memory settings

**Files to Examine:**
- `scripts/runpod_bootstrap.sh` (model download section)
- `HOW_TO_START.md` (vLLM startup commands)
- vLLM startup logs
- Model path configuration

---

### 6. Error Handling & Logging

**Questions to Answer:**
- Are errors properly logged and visible?
- Do startup scripts fail fast or continue with broken state?
- Are error messages actionable?
- Is there sufficient logging for debugging?
- Do services log to accessible locations?

**Investigation Steps:**
1. Review error handling in bootstrap script
2. Check log file locations and permissions
3. Test error scenarios and verify logging
4. Review service manager error handling
5. Check if errors are swallowed or hidden
6. Verify log rotation and disk space for logs
7. Test error recovery mechanisms

**Files to Examine:**
- `scripts/runpod_bootstrap.sh` (error handling, logging)
- Service log files (`/tmp/vllm_*.log`, etc.)
- `LOG_DIR` and log file management
- Error messages in startup scripts

---

### 7. Build & Compilation Issues

**Questions to Answer:**
- Does Rust build complete successfully?
- Are all dependencies available?
- Are feature flags correctly set?
- Do ONNX libraries link properly?
- Are there compilation errors that are ignored?

**Investigation Steps:**
1. Test clean build from scratch
2. Verify all Rust dependencies are available
3. Check ONNX runtime library linking
4. Test build with different feature flags
5. Verify CUDA/CUDNN linking
6. Check for compilation warnings that might indicate issues
7. Test build on fresh RunPod instance

**Files to Examine:**
- `scripts/runpod_bootstrap.sh` (build section)
- `Cargo.toml` files
- Build logs
- ONNX library paths in `.runpod_env.sh`

---

### 8. Service Manager & Orchestration

**Questions to Answer:**
- How does `unified_service_manager.sh` work?
- Does it handle service restarts properly?
- Are services started in correct order?
- What happens if a service crashes?
- Is there proper process management?

**Investigation Steps:**
1. Analyze service manager implementation
2. Test service startup order
3. Test service restart scenarios
4. Check process management (PID files, etc.)
5. Verify service dependencies are respected
6. Test graceful shutdown
7. Check for zombie processes

**Files to Examine:**
- `scripts/unified_service_manager.sh` (if exists)
- `scripts/supervisor.sh` (if exists)
- Service startup logic in bootstrap script
- Process management code

---

### 9. RunPod-Specific Issues

**Questions to Answer:**
- Are there RunPod-specific constraints or requirements?
- Does RunPod have different filesystem layout?
- Are there network restrictions?
- Does RunPod require different startup commands?
- Are there container/environment differences?

**Investigation Steps:**
1. Compare local vs RunPod environment differences
2. Check RunPod documentation for constraints
3. Test filesystem paths and permissions
4. Verify network configuration
5. Check for container-specific issues
6. Test with RunPod's recommended practices
7. Document RunPod-specific requirements

**Files to Examine:**
- RunPod-specific configuration files
- Environment setup scripts
- Documentation about RunPod deployment

---

### 10. Health Check Reliability

**Questions to Answer:**
- Are health checks accurate?
- Do they check the right endpoints?
- Are timeout values appropriate?
- Do health checks account for slow startup?
- What happens if health check fails?

**Investigation Steps:**
1. Review health check implementation
2. Test health check accuracy
3. Measure actual service ready times
4. Test health check timeouts
5. Verify health check endpoints exist
6. Test health checks during service startup
7. Check for false positives/negatives

**Files to Examine:**
- `scripts/runpod_bootstrap.sh` (`check_http_health` function)
- `niodoo_real_integrated/src/health.rs`
- Health check URLs and endpoints
- Timeout values

---

## Expected Deliverables

After completing this research, provide:

### 1. Root Cause Analysis Report

Document each identified issue with:
- **Symptom**: What happens (error messages, behavior)
- **Root Cause**: Why it happens (technical explanation)
- **Impact**: How it affects startup reliability
- **Frequency**: How often it occurs
- **Evidence**: Logs, code references, test results

### 2. Failure Pattern Documentation

Categorize failures by:
- **Type**: Environment, Network, Resource, Configuration, etc.
- **Severity**: Critical, High, Medium, Low
- **Detectability**: Easy to spot, Hidden, Silent failure
- **Workaround**: Temporary fixes (if any)

### 3. Solution Proposals

For each root cause, propose:
- **Immediate Fix**: Quick workaround or patch
- **Proper Solution**: Long-term architectural fix
- **Prevention**: How to avoid this in future
- **Implementation Effort**: Estimated complexity

### 4. Improved Startup Script

Create an enhanced version that:
- Handles all identified failure modes
- Provides better error messages
- Implements proper retry logic
- Has comprehensive logging
- Validates environment before starting
- Provides clear status updates
- Handles partial failures gracefully

### 5. Testing & Validation Plan

Document how to:
- Reproduce each failure mode
- Test fixes
- Validate improvements
- Monitor startup reliability

### 6. RunPod Deployment Guide

Create a step-by-step guide for:
- RunPod-specific setup
- Required configuration
- Common pitfalls and solutions
- Troubleshooting checklist
- Success criteria

---

## Research Methodology

### Phase 1: Discovery (Days 1-2)
- Review all relevant code and documentation
- Map system architecture and dependencies
- Identify potential failure points
- Create test scenarios

### Phase 2: Investigation (Days 3-5)
- Execute test scenarios
- Reproduce failures
- Collect logs and evidence
- Measure actual behavior vs expected

### Phase 3: Analysis (Days 6-7)
- Analyze root causes
- Categorize failure patterns
- Prioritize issues by impact
- Document findings

### Phase 4: Solution Design (Days 8-9)
- Design fixes for each issue
- Create improved startup scripts
- Write deployment guide
- Create testing plan

### Phase 5: Validation (Day 10)
- Test solutions
- Validate improvements
- Update documentation
- Create final report

---

## Key Files to Review

### Scripts
- `scripts/runpod_bootstrap.sh` - Main bootstrap script
- `scripts/unified_service_manager.sh` - Service orchestration (if exists)
- `scripts/supervisor.sh` - Service supervisor (if exists)
- `.runpod_env.sh` - RunPod environment setup

### Configuration
- `tcs_runtime.env` - Runtime environment variables
- `config/h200.env` - H200 GPU configuration
- `config/rtx5090.env` - RTX 5090 GPU configuration
- `niodoo_real_integrated/src/config.rs` - Application configuration

### Documentation
- `HOW_TO_START.md` - Startup guide
- `AI_SETUP_GUIDE.md` - System architecture guide
- `CHANGELOG.md` - Recent changes and fixes

### Code
- `niodoo_real_integrated/src/health.rs` - Health check implementation
- `niodoo_real_integrated/src/main.rs` - Application entry point
- `niodoo_real_integrated/src/pipeline/core.rs` - Pipeline initialization

---

## Success Criteria

This research is successful if:

1. ✅ **All failure modes are identified** - No unexplained startup failures remain
2. ✅ **Root causes are documented** - Each issue has a clear technical explanation
3. ✅ **Solutions are proposed** - Every issue has a fix path
4. ✅ **Startup reliability improves** - Success rate increases significantly
5. ✅ **Debugging is easier** - Clear error messages and logging
6. ✅ **Documentation is updated** - RunPod deployment is well-documented

---

## Notes

- **Be thorough**: Don't assume anything works. Test everything.
- **Be systematic**: Follow the research areas methodically.
- **Be evidence-based**: Every claim should have logs, code, or test results.
- **Be practical**: Focus on actionable solutions, not theoretical problems.
- **Be user-focused**: Remember the user has zero technical background but is "architectural jesus" - solutions should be robust and well-architected.

---

**Start your investigation now. Document everything. Find the problems. Fix them. Make endpoints start reliably on RunPod.**

