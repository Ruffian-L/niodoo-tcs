# Niodoo-Topo-Coder Sync Setup

## 🎯 What This Is

This is your **NIODOO-CODE** workspace - the pivot from emotional intelligence to code intelligence. All critical components from the RunPod's Niodoo-Final have been synced here for local development.

## 📁 Directory Structure

```
Niodoo-Topo-Coder/
├── .kiro/                    # Specs and architecture docs (READ THESE FIRST!)
├── tcs-core/                 # Core topological operations
├── tcs-tda/                  # TDA computation (persistent homology)
├── tcs-tqft/                 # TQFT for state evolution
├── tcs-knot/                 # Knot theory for "thought-knots"
├── tcs-rce/                  # RCE metrics (dBetti/dt tracking)
├── tcs-ml/                   # ML components with Qwen integration
├── tcs-pipeline/             # Pipeline management
├── constants_core/           # Shared constants
├── niodoo_real_integrated/   # Main consciousness engine
├── niodoo-ai/                # Python training scripts
├── Niodoo-TCT/               # Topological token compression
├── Cargo.toml               # Workspace configuration
└── sync.sh                   # Bidirectional sync script
```

## 🔄 Sync Commands

The `sync.sh` script provides bidirectional syncing with the RunPod:

```bash
# Download latest from RunPod (pull)
./sync.sh pull

# Upload local changes to RunPod (push)
./sync.sh push

# Check sync status
./sync.sh status
```

### Important Notes:
- **Always pull before you push** to avoid conflicts
- The sync excludes `target/` directories (build artifacts)
- The RunPod folder is: `/workspace/Niodoo-Topo-Coder-Sync`

## 🚀 Next Steps: NIODOO-CODE Implementation

Based on the technical blueprint you provided, here's what needs to be built:

### Phase 1: Parser Pipeline (Day 1 - 8-10 hours)
Create `tcs-parser` crate with:
- tree-sitter integration for AST parsing
- tree-sitter-graph for AST → Graph conversion
- rust-code-analysis for complexity metrics
- Output: JSON with adjacency matrix + metrics

**Key Files to Study:**
- `.kiro/specs/` - All system specs
- `tcs-tda/src/lib.rs` - Existing TDA pipeline
- `tcs-ml/src/qwen_embedder.rs` - ML integration

### Phase 2: TDA Pipeline (Day 2 - 8-10 hours)
Extend `tcs-tda` with:
- Async FFI bridge to Python (pyo3-async-runtimes)
- giotto-tda integration for persistent homology
- Zero-copy ndarray → PyArray conversion

**Key Files to Study:**
- `tcs-rce/src/rce_metrics.rs` - RCE tracking (dBetti/dt)
- `niodoo_real_integrated/python/giotto_tda_wrapper.py` - Python TDA wrapper

### Phase 3: Training Dataset Construction
Use BigQuery GitHub dataset to:
- Extract 50k-100k high-churn code files
- Label with CQS (Code Quality Score) proxy
- Create topology-aware training examples

**Leverage:**
- `niodoo-ai/` - Existing training infrastructure
- `tcs-ml/` - Qwen integration

### Phase 4: QLoRA Fine-tuning
Fine-tune Qwen2.5-Coder with composite loss:
- L_total = L_crossentropy + λ·L_topo
- Preserve emotional adapters (orthogonal fine-tuning)
- Validate on HiBench and DSR-Bench

## 🔧 Development Workflow

1. **Work locally** on your desktop (fast RTX 5080!)
2. **Sync to RunPod** when you need H100 GPU for training
3. **Read `.kiro/` first** before touching any code
4. **Use proper logging** (log crate), NO println!
5. **No hardcoding**, **no stubs**, **no Python** (unless absolutely necessary)

## 📚 Critical Reading Order

1. `.kiro/ARCHITECTURE_ALIGNMENT.md` - System overview
2. `.kiro/specs/bullshit-buster-mvp/design.md` - Core design philosophy
3. `tcs-rce/src/rce_metrics.rs` - The dBetti/dt tracking
4. `niodoo_real_integrated/PRODUCTION_README.md` - Production setup

## 🎓 Key Concepts from Blueprint

- **RCE Tracking**: Real-time dBetti/dt monitoring for state awareness
- **Topological MRI**: Map entire codebase topology (not just text snippets)
- **Thought-Knots**: Complex cyclical dependencies as persistent Betti-1 loops
- **Composite Loss**: L_crossentropy (correctness) + L_topo (structure quality)
- **Language-Agnostic**: tree-sitter works with any language (COBOL, FORTRAN, etc.)

## 🚨 Remember: NIODOO Code Standards

- ✅ Real implementations with proper error handling
- ✅ Mathematical rigor (consciousness topology!)
- ✅ Performance-first (real-time consciousness simulation)
- ✅ Memory safety (Rust ownership model)
- ❌ NO HARD CODING
- ❌ NO PRINTLN/PRINT (use log crate)
- ❌ NO STUBS
- ❌ NO PYTHON SCRIPTS (absolute last resort)
- ❌ NO BULLSHITTING

## 💡 RunPod Connection

If you need to SSH directly to RunPod:

```bash
# Via direct TCP (recommended for file transfers)
ssh root@38.80.152.72 -p 31008 -i ~/.ssh/id_ed25519

# Via RunPod proxy
ssh 8uapfmstv1x5l5-64411d6f@ssh.runpod.io -i ~/.ssh/id_ed25519
```

---

**Created**: 2025-11-06
**Sync Status**: ✅ All critical files synced
**Ready for**: NIODOO-CODE Day 1 implementation
