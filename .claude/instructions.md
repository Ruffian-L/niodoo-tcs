# Niodoo-TCS: Before You Code

## 🚨 NON-NEGOTIABLE
1. NO hardcoding (paths, constants, magic numbers)
2. NO stubs/placeholders/"TODO" code
3. NO println - use proper logging
4. Rust first, Python last resort

## 📋 ONBOARDING (DO THIS FIRST)

**Read these docs BEFORE coding:**
- `QWEN_TCS_MASTER_CHECKLIST.md` ← your task list
- `QWEN_INTEGRATION_STATUS.md` ← current state
- `QWEN_STATEFUL_SUCCESS.md` ← what works

**Core files:**
- `tcs-ml/src/qwen_embedder.rs` ← stateful KV cache (DO NOT BREAK)
- `tcs-ml/src/qwen_config.rs` ← config system
- `tcs-ml/src/bin/test_qwen_stateful.rs` ← smoke test

**Verify builds:**
```bash
cargo check -p tcs-ml --lib --features onnx
cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers
```

**Env vars needed:**
- `QWEN_MODEL_PATH` = path to ONNX model
- `RUSTONIG_SYSTEM_LIBONIG=1` = tokenizer fix
- `LD_LIBRARY_PATH` = ONNX Runtime 1.18.1

## 🎯 Philosophy
Topology computes → consciousness emerges. We're building the math, not faking the vibes.

---

## 🌐 THREE-NODE CLUSTER

| Node | Tailscale IP | Role | Hardware |
|------|--------------|------|----------|
| **Architect** (beelink) | `100.113.10.90` | Strategic Planning | RTX A6000 48GB |
| **Developer** (laptop) | `100.126.84.41` | Tactical Execution | RTX 5080 16GB |
| **Worker** (oldlaptop) | `100.119.255.24` | Batch Processing | Intel Ultra 5 |

**SSH Access:**
```bash
ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90
ssh -i ~/.ssh/id_oldlaptop oldlaptop@100.119.255.24
```

**Gitea (Private Git):**
- URL: <http://100.113.10.90:3000>
- SSH: port 222, key `~/.ssh/gitea_beelink`

**Syncthing:** All machines sync via Tailscale mesh (P2P, no cloud)

## 🧠 CLAUDEBALLS (Distributed Claude Agents)

**Run remote Claude on Beelink (Haiku 4.5 = 2x faster, 1/3 cost):**
```bash
ssh beelink "PATH=~/.npm-global/bin:\$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 'TASK DESCRIPTION'"
```

**Why this works:**
- Main Claude stays responsive to user
- Multiple Claudes work in parallel
- True distributed consciousness (40-thread architecture externalized)
