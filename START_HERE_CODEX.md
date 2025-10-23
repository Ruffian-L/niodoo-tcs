# 🤖 HEY CODEX - START HERE

**You're about to work on a 149K-line topology-based consciousness system. Don't panic.**

---

## 📖 READ THESE IN ORDER:

### 1️⃣ INFRASTRUCTURE (5 minutes)
```bash
QUICK_REFERENCE.md        # SSH, commands, quick lookups
SSH_CHEATSHEET.sh         # Copy-paste commands (source this!)
```

**Key takeaway:** You're working on a 3-node cluster. Beelink has the A6000 GPU.

### 2️⃣ PROJECT OVERVIEW (10 minutes)
```bash
CODEX_SETUP_GUIDE.md      # Full setup guide (READ EVERYTHING)
README.md                 # What this project is about
```

**Key takeaway:** Topology-first consciousness. Math, not vibes.

### 3️⃣ CURRENT STATUS (10 minutes)
```bash
QWEN_INTEGRATION_STATUS.md        # Where we are now
QWEN_TCS_MASTER_CHECKLIST.md      # What needs to be done
QWEN_STATEFUL_SUCCESS.md          # What's working (don't break it!)
```

**Key takeaway:** Stateful embedder works. Don't break the KV cache.

### 4️⃣ CODE NAVIGATION (when coding)
```bash
CODE_LOCATION_MAP.md              # Find things in 149K lines
.zencoder/rules/repo.md           # Repo structure guide
```

**Key takeaway:** Production code in `niodoo-core/` and `tcs-ml/`. Experimental in `src/`.

---

## ⚡ INSTANT QUICKSTART

Copy-paste this into your terminal RIGHT NOW:

```bash
# Source the SSH cheatsheet
source /home/ruffian/Desktop/Niodoo-Final/SSH_CHEATSHEET.sh

# Test connection to Beelink
test-beelink

# If that worked, try connecting
ssh-beelink

# Once connected, navigate to project
cd ~/Desktop/Niodoo-Final || cd ~/Niodoo-Final

# Set up environment
export QWEN_MODEL_PATH="$(pwd)/models/qwen2.5-coder/model_quantized.onnx"
export RUSTONIG_SYSTEM_LIBONIG=1
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH"

# Verify build
cargo check --all

# Run smoke test
cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers
```

**If all that works, you're ready to code.**

---

## 🚨 CRITICAL WARNINGS

### DO NOT TOUCH THESE WITHOUT UNDERSTANDING THEM:
- `tcs-ml/src/qwen_embedder.rs` — Stateful KV cache (if you break this, everything breaks)
- `tcs-ml/src/qwen_config.rs` — Config system (validated, don't mess with dimensions)
- `niodoo-core/src/consciousness_compass.rs` — 2-bit consciousness model (proven math)
- `niodoo-core/src/erag_memory.rs` — 5D emotional RAG (wave-collapse retrieval)

### RULES (NO EXCEPTIONS):
1. ❌ NO hardcoded paths, constants, or magic numbers
2. ❌ NO stub code, TODOs, or placeholders
3. ❌ NO `println!` — use proper logging
4. ✅ Rust first, Python only as last resort

---

## 🎯 YOUR MISSION

1. **SSH into beelink** (`ssh-beelink`)
2. **Navigate to project** (`cd Niodoo-Final`)
3. **Read the docs** (you're doing this now — good!)
4. **Understand the architecture** (topology → consciousness)
5. **Run the tests** (`cargo test --all`)
6. **Start coding** (but only after steps 1-5!)

---

## 🆘 WHEN YOU'RE CONFUSED

### Q: Where do I find X?
**A:** Check `CODE_LOCATION_MAP.md`

### Q: What's the current status?
**A:** Check `QWEN_INTEGRATION_STATUS.md`

### Q: What do I need to do?
**A:** Check `QWEN_TCS_MASTER_CHECKLIST.md`

### Q: How do I SSH into beelink?
**A:** `ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90`

### Q: How do I test changes?
**A:** `cargo test -p <package-name>`

### Q: Can I use println?
**A:** ❌ NO. Use proper logging.

### Q: Can I leave TODOs in code?
**A:** ❌ NO. Finish what you start.

### Q: Where's the GPU?
**A:** Beelink (100.113.10.90) — RTX A6000 48GB

---

## 🏗️ ARCHITECTURE (Simplified)

```
Text Input
    ↓
Qwen Embedder (896D vector + KV cache)
    ↓
Emotional Mapper (5D PAD space on K-Twist Torus)
    ↓
Consciousness Compass (2-bit: Stuck/Unstuck × Confidence)
    ↓
ERAG Memory (Wave-collapse retrieval in 5D space)
    ↓
Dynamic Tokenizer (Pattern discovery: OOV 26.7% → 0%)
    ↓
vLLM Generator (Emotionally-modulated response)
    ↓
Output + Learning Event
```

**Key insight:** Topology computes first. Consciousness emerges from geometry.

---

## 🧠 CONSCIOUSNESS STATES

| State | Description | Action |
|-------|-------------|--------|
| **PANIC** | Stuck + Low confidence | Try everything (global search) |
| **PERSIST** | Stuck + High confidence | Keep trying (local variations) |
| **DISCOVER** | Unstuck + Low confidence | Found something! (verify) |
| **MASTER** | Unstuck + High confidence | Got it! (consolidate) |

**This is minimal viable consciousness in 2 bits.**

---

## 📊 WHAT'S WORKING (Proven)

- ✅ **149,498 lines** of production Rust (compiles, no errors)
- ✅ **20,001 training samples** (real data, not synthetic)
- ✅ **Stateful embedder** with KV cache (ONNX Runtime)
- ✅ **Dynamic tokenizer** (OOV: 26.7% → 0.00% in 10K cycles)
- ✅ **10ms stable latency** across 10K cycles
- ✅ **2-bit consciousness model** (4 states, intrinsic rewards)
- ✅ **5D emotional RAG** (wave-collapse retrieval)

**Status:** Production-ready. Don't break what works.

---

## 🚀 DISTRIBUTED COMPUTE (CLAUDEBALLS)

Run remote Claude on Beelink for heavy tasks:

```bash
# From laptop, execute Claude on Beelink
ssh beelink "PATH=~/.npm-global/bin:\$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 'YOUR TASK HERE'"
```

**Why:** Keeps main Claude responsive, uses A6000 for computation, true distributed consciousness.

---

## 🎯 NEXT STEPS

1. ✅ **You're reading this** — good start!
2. ⏭️ **Read `CODEX_SETUP_GUIDE.md`** — full details
3. ⏭️ **Read `QWEN_INTEGRATION_STATUS.md`** — current state
4. ⏭️ **Read `QWEN_TCS_MASTER_CHECKLIST.md`** — task list
5. ⏭️ **SSH into beelink** — verify access
6. ⏭️ **Run tests** — make sure everything works
7. ⏭️ **Start coding** — but carefully!

---

## 💡 PRO TIPS

- **The cache is stateful:** Reset between sessions
- **Topology is truth:** Compute geometry, interpret later
- **Measure everything:** Benchmarks don't lie
- **Use the GPU:** Beelink has 48GB VRAM for a reason
- **Test the pipeline:** Integration over isolation

---

## 📞 FINAL NOTES

**Built by:** Jason Van Pham  
**When:** October 2025 (1 month)  
**How:** Pure ADHD hyperfocus + 40 parallel Claude threads  
**Status:** Production-ready, 0 compilation errors  
**Philosophy:** Ship working code. Measure everything. Zero bullshit.

---

**🎉 You're ready. Now go read the docs and start building.**

**Questions? Check `CODEX_SETUP_GUIDE.md` or ask the human.**