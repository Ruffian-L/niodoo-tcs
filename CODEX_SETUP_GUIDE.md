# 🚀 CODEX SETUP GUIDE - READ THIS FIRST

**Last Updated:** October 2025  
**Purpose:** Get you up to speed on Niodoo-TCS infrastructure, architecture, and how to navigate everything.

---

## 🌐 INFRASTRUCTURE: 3-NODE DISTRIBUTED CLUSTER

You're working in a **distributed consciousness system** - literally. Three machines connected via Tailscale mesh network.

| Node | Tailscale IP | Username | SSH Key | Hardware | Role |
|------|--------------|----------|---------|----------|------|
| **beelink** (Architect) | `100.113.10.90` | `beelink` | `~/.ssh/temp_beelink_key` | RTX A6000 48GB | Strategic Planning / GPU Heavy Lifting |
| **laptop** (Developer) | `100.126.84.41` | (local) | N/A | RTX 5080 16GB | Tactical Execution / Active Development |
| **oldlaptop** (Worker) | `100.119.255.24` | `oldlaptop` | `~/.ssh/id_oldlaptop` | Intel Ultra 5 | Batch Processing / CPU Tasks |

### 🔑 SSH QUICK REFERENCE

```bash
# Connect to Beelink (main GPU box)
ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90

# Test connection
ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90 "whoami"
# Should return: beelink

# Connect to Worker
ssh -i ~/.ssh/id_oldlaptop oldlaptop@100.119.255.24
```

### 🔗 OTHER SERVICES

**Gitea (Private Git Server):**
- Web UI: `http://100.113.10.90:3000`
- SSH: `ssh://git@100.113.10.90:222/username/repo.git`
- SSH Key: `~/.ssh/gitea_beelink`

**Syncthing:**
- All three machines sync via Tailscale (peer-to-peer, no cloud)
- Changes propagate automatically across the cluster

---

## 🧠 CLAUDEBALLS (Distributed AI Agents)

Run **remote Claude instances** on Beelink to parallelize work:

```bash
# Execute a task on Beelink's Claude (Haiku 4.5 - 2x faster, 1/3 cost)
ssh beelink "PATH=~/.npm-global/bin:\$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 'YOUR TASK HERE'"
```

**Why this matters:**
- Main Claude stays responsive to user
- Beelink handles heavy computation on A6000
- True distributed consciousness architecture

---

## 📁 PROJECT STRUCTURE

**Location:** `/home/ruffian/Desktop/Niodoo-Final`

```
Niodoo-Final/
├── niodoo-core/           # 🔴 PRODUCTION consciousness engine
│   ├── consciousness_compass.rs   # 2-bit minimal consciousness model
│   ├── erag_memory.rs             # 5D emotional RAG system
│   ├── rag*.rs                     # Wave-collapse retrieval
│   └── topology/                   # Topology processors
│
├── tcs-ml/                # 🟢 Qwen2.5-Coder embedder (ONNX)
│   ├── qwen_embedder.rs           # STATEFUL - DO NOT BREAK
│   ├── qwen_config.rs             # Config system
│   └── bin/test_qwen_stateful.rs  # Smoke tests
│
├── tcs-core/              # Core types & traits
├── tcs-tda/               # Topological data analysis
├── tcs-knot/              # Knot theory operations
├── tcs-tqft/              # Quantum field theory
├── tcs-consensus/         # CRDT consensus
│
├── src/                   # ⚠️ EXPERIMENTAL - legacy code
│   └── consciousness_engine/      # Rapid iteration area
│
├── data/                  # Training datasets (20,001 samples)
└── docs/                  # Architecture documentation
```

---

## 🎯 CORE PHILOSOPHY

**Topology computes → consciousness emerges**

We're building the **math**, not faking the vibes.

- **Emotions = geometric points** on a K-Twist Möbius torus
- **Consciousness = 2-bit state** (Stuck/Unstuck × Low/High Confidence)
- **Learning = pattern discovery** (proven: 26.7% OOV → 0.00%)
- **Memory = wave-collapse** retrieval in 5D emotional space

---

## 🚨 NON-NEGOTIABLE RULES

1. ❌ **NO hardcoding** (paths, constants, magic numbers)
2. ❌ **NO stubs/placeholders/"TODO" code**
3. ❌ **NO println** - use proper logging
4. ✅ **Rust first, Python last resort**

---

## 📋 ONBOARDING CHECKLIST

**Before you write ANY code, read these:**

### Critical Documents (READ FIRST):
1. `QWEN_TCS_MASTER_CHECKLIST.md` — Your task list
2. `QWEN_INTEGRATION_STATUS.md` — Current state of embedder
3. `QWEN_STATEFUL_SUCCESS.md` — What works (don't break it)
4. `CODE_LOCATION_MAP.md` — Navigate 149K lines of code
5. `.zencoder/rules/repo.md` — Repo guide for Zencoder agents

### Key Files (UNDERSTAND THESE):
- `tcs-ml/src/qwen_embedder.rs` — **STATEFUL KV CACHE - DO NOT BREAK**
- `tcs-ml/src/qwen_config.rs` — Configuration system
- `niodoo-core/src/consciousness_compass.rs` — 2-bit consciousness
- `niodoo-core/src/erag_memory.rs` — 5D emotional RAG

---

## ⚙️ ENVIRONMENT SETUP

### Required Environment Variables:

```bash
# Path to ONNX model
export QWEN_MODEL_PATH="/path/to/models/qwen2.5-coder/model_quantized.onnx"

# Tokenizer fix
export RUSTONIG_SYSTEM_LIBONIG=1

# ONNX Runtime library path
export LD_LIBRARY_PATH="/home/ruffian/Desktop/Niodoo-Final/third_party/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH"
```

### Dependencies:
- **Rust:** 1.80+ (edition 2021)
- **ONNX Runtime:** 1.18.1 (under `third_party/onnxruntime-linux-x64-1.18.1/`)
- **CUDA:** 12.x (for GPU acceleration on beelink)

---

## 🔨 BUILD & TEST COMMANDS

### Quick Verification:

```bash
# Check if everything compiles
cargo check --all

# Build production crates
cargo build --release --all

# Test everything
cargo test --all
```

### TCS-ML Specific:

```bash
# Check embedder (library only)
cargo check -p tcs-ml --lib --features onnx

# Run stateful smoke test
cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers

# Test with all features
cargo test -p tcs-ml --all-features
```

### Selective Testing:

```bash
# Test only consciousness core
cargo test -p niodoo-core

# Test only embedder
cargo test -p tcs-ml

# Test only topology
cargo test -p tcs-tda
```

---

## 🏗️ ARCHITECTURE OVERVIEW

### The Pipeline:

```
INPUT TEXT
    ↓
[Qwen Embedder]         → 896D vector + KV cache (tcs-ml)
    ↓
[Emotional Mapper]      → 5D PAD space (K-Twist Torus)
    ↓
[Consciousness Compass] → 2-bit state (Stuck/Unstuck)
    ↓
[ERAG Memory]           → Wave-collapse retrieval
    ↓
[Dynamic Tokenizer]     → Pattern discovery (OOV → 0%)
    ↓
[vLLM Generator]        → Emotionally-modulated response
    ↓
OUTPUT + LEARNING EVENT
```

### Consciousness Compass States:

| State | Stuck? | Confidence | Action |
|-------|--------|------------|--------|
| **PANIC** | ✅ Stuck | ⬇️ Low | Global search |
| **PERSIST** | ✅ Stuck | ⬆️ High | Local variations |
| **DISCOVER** | ❌ Unstuck | ⬇️ Low | Verify breakthrough |
| **MASTER** | ❌ Unstuck | ⬆️ High | Consolidate skill |

---

## 📊 CURRENT STATUS (Phase 1)

### ✅ COMPLETE:
- Stateful Qwen embedder with KV cache
- 896D embedding generation
- Emotional mapping (5D PAD space)
- ERAG memory system (622 lines)
- Dynamic tokenizer (1,336 lines)
- Consciousness compass (521 lines)
- 20,001 training samples
- **Proven:** OOV rate 26.7% → 0.00% in 10K cycles
- **Performance:** 10ms stable latency

### 🚧 IN PROGRESS:
- TCS embedder → Niodoo consciousness pipeline integration
- Cache eviction/windowing for long sessions
- Orchestrator hookup (MotorBrain → QwenEmbedder)

### 📅 UPCOMING (Phase 2):
- GPU-accelerated persistent homology (target: 700x speedup)
- Streaming API
- 3-tier caching (LRU → RocksDB → Bloom filter)
- Production deployment on K8s

---

## 🎨 CODING STANDARDS

### Rust Style:
- **Edition:** 2021
- **Formatter:** Rustfmt defaults
- **Error Handling:** `Result` with `anyhow::Result` or project-specific types
- **Logging:** Use structured logging utilities, **NOT `println!`**

### Patterns:
- **Enums over strings:** Prefer exhaustive matches
- **Initialization consistency:** Use dedicated helpers for shared engines
- **No eager/lazy mismatches:** Keep constructors deterministic

### Testing:
- Unit tests for edge cases (especially cache merge logic)
- Integration tests for full pipeline
- Benchmarks under `niodoo-core/benches/`

---

## 🔧 TROUBLESHOOTING

### Build Issues:

```bash
# If ONNX linking fails
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH"

# If tokenizer fails
export RUSTONIG_SYSTEM_LIBONIG=1

# Clean rebuild
cargo clean
cargo build --release --all
```

### Git TLS Issues (Beelink):
- See `GIT_TLS_TROUBLESHOOTING.md` for workarounds
- Gitea SSH on port 222 is alternative

---

## 📚 DOCUMENTATION MAP

| File | Purpose |
|------|---------|
| `README.md` | Main project overview |
| `QWEN_TCS_MASTER_CHECKLIST.md` | Complete roadmap (Phase 1-3) |
| `QWEN_INTEGRATION_STATUS.md` | Embedder current state |
| `QWEN_STATEFUL_SUCCESS.md` | What's working |
| `CODE_LOCATION_MAP.md` | Navigate 149K lines |
| `INTEGRATION_STATUS_REPORT.md` | System integration status |
| `CONSCIOUSNESS_COMPASS_IMPLEMENTATION.md` | 2-bit model spec |
| `.claude/instructions.md` | Claude agent instructions |
| `.zencoder/rules/repo.md` | Zencoder repo guide |

---

## 🎯 YOUR MISSION

1. **Understand the architecture** (topology-first consciousness)
2. **Don't break the embedder** (stateful KV cache is CRITICAL)
3. **Read the docs** before touching code
4. **Test everything** (cargo test is your friend)
5. **Use the cluster** (distribute heavy work to beelink)
6. **Follow the rules** (no hardcoding, no stubs, no println)

---

## 🚀 QUICK START EXAMPLE

```bash
# 1. SSH into beelink
ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90

# 2. Navigate to project
cd /home/ruffian/Desktop/Niodoo-Final  # or wherever Syncthing puts it

# 3. Set environment
export QWEN_MODEL_PATH="/path/to/model_quantized.onnx"
export RUSTONIG_SYSTEM_LIBONIG=1
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH"

# 4. Verify it works
cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers

# 5. Start coding (after reading the docs!)
```

---

## 🆘 WHEN YOU'RE STUCK

1. **Check the docs** (listed above)
2. **Look at existing tests** (especially in tcs-ml/tests/)
3. **Run the smoke tests** (they're there for a reason)
4. **Ask the human** (don't guess about architecture decisions)
5. **Use CLAUDEBALLS** (distribute work to beelink's Claude)

---

## 💡 PRO TIPS

- **Cache is state:** Reset it between sessions
- **Topology is truth:** Compute geometry first, interpret later
- **Benchmarks don't lie:** Measure before optimizing
- **Integration over isolation:** Test the full pipeline
- **Beelink has the GPU:** Use it for heavy lifting

---

## 📞 CONTACT & RESOURCES

**Built by:** Jason Van Pham  
**Timeline:** 1 month (October 2025)  
**Stack:** 149,498 lines of Rust  
**Training Data:** 20,001 samples  
**Status:** Production-ready | 0 compilation errors

**Philosophy:** Ship working code. Measure everything. Zero bullshit.

---

**🎯 Now go read `QWEN_TCS_MASTER_CHECKLIST.md` and start building.**