---
mode: agent
---
You are joining the niodoo-tcs repo midstream. Before coding, gather context:

1. Read these handoff/checklist files end to end for status, env vars, and open tasks:
   - QWEN_INTEGRATION_STATUS.md
   - QWEN_HANDOFF_CHECKLIST.md
   - QWEN_TCS_MASTER_CHECKLIST.md
   - QWEN_STATEFUL_SUCCESS.md

2. Inspect the core source files that implement the stateful Qwen embedder:
   - tcs-ml/src/qwen_embedder.rs
   - tcs-ml/src/qwen_config.rs
   - tcs-ml/src/bin/test_qwen_stateful.rs
   - tcs-ml/src/lib.rs
   - .github/workflows/ci.yml

3. Confirm the local environment (ONNX Runtime, tokenizer assets, env vars) matches the checklist. Run:
   - cargo check -p tcs-ml --lib --features onnx
   - cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers

4. Use the master checklist to choose your next task; do not resurrect deprecated plans from older docs.

5. When coding, respect existing feature flags, keep cache management intact, and update the checklists if scope changes.

Only proceed once you understand the current architecture and outstanding items.