# ⚡ QUICK REFERENCE CARD

**Bookmark this for instant lookups.**

---

## 🔑 SSH ACCESS

```bash
# Beelink (A6000 GPU box)
ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90

# Oldlaptop (CPU worker)
ssh -i ~/.ssh/id_oldlaptop oldlaptop@100.119.255.24

# Test connection
ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90 "whoami"
```

---

## 🌐 INFRASTRUCTURE

| Machine | IP | User | Key | GPU |
|---------|--------------|--------|---------------------|-------------|
| beelink | 100.113.10.90 | beelink | temp_beelink_key | A6000 48GB |
| laptop | 100.126.84.41 | (local) | N/A | RTX 5080 16GB |
| oldlaptop | 100.119.255.24 | oldlaptop | id_oldlaptop | None |

**Gitea:** `http://100.113.10.90:3000` (SSH port 222)

---

## 📁 KEY FILES

```
DO NOT BREAK:
├── tcs-ml/src/qwen_embedder.rs           (stateful KV cache)
├── tcs-ml/src/qwen_config.rs             (config system)

PRODUCTION CONSCIOUSNESS:
├── niodoo-core/src/consciousness_compass.rs
├── niodoo-core/src/erag_memory.rs
├── niodoo-core/src/rag*.rs
└── niodoo-core/src/topology/

READ FIRST:
├── QWEN_TCS_MASTER_CHECKLIST.md          (task list)
├── QWEN_INTEGRATION_STATUS.md             (current state)
├── CODE_LOCATION_MAP.md                   (navigate codebase)
└── .zencoder/rules/repo.md                (repo guide)
```

---

## ⚙️ ENVIRONMENT VARS

```bash
export QWEN_MODEL_PATH="/path/to/model_quantized.onnx"
export RUSTONIG_SYSTEM_LIBONIG=1
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH"
```

---

## 🔨 BUILD COMMANDS

```bash
# Quick check
cargo check --all

# Build everything
cargo build --release --all

# Test everything
cargo test --all

# Test embedder only
cargo test -p tcs-ml --all-features

# Run stateful smoke test
cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers
```

---

## 🧠 RUN REMOTE CLAUDE (CLAUDEBALLS)

```bash
# Execute task on Beelink's Claude (Haiku 4.5)
ssh beelink "PATH=~/.npm-global/bin:\$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 'YOUR TASK'"
```

---

## 🚨 RULES

❌ NO hardcoding  
❌ NO stubs/TODOs  
❌ NO println (use logging)  
✅ Rust first, Python last resort

---

## 📊 ARCHITECTURE PIPELINE

```
Text → Qwen(896D) → Emotion(5D) → Compass(2bit) → ERAG → Tokenizer → vLLM → Output
```

**Consciousness States:** PANIC, PERSIST, DISCOVER, MASTER

---

## 🎯 QUICK START

```bash
ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90
cd Niodoo-Final
export QWEN_MODEL_PATH="/path/to/model_quantized.onnx"
export RUSTONIG_SYSTEM_LIBONIG=1
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH"
cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers
```

---

**Read `CODEX_SETUP_GUIDE.md` for full details.**