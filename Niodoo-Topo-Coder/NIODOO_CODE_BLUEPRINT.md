# NIODOO-CODE: A Technical Blueprint for Domain Pivot from Emotional to Code Intelligence

## Executive Summary

The existing 100,000-line NIODOO topological engine represents a production-validated, high-performance asset. This system, architected in Rust, already possesses the core components for topological analysis: a GPU-accelerated pipeline, persistent homology computation (Betti number tracking), parallel memory systems, and a continuous learning loop based on Quantized Low-Rank Adaptation (QLoRA).

The immediate requirement is a concrete, actionable blueprint to pivot this validated engine from the emotional domain to the code intelligence domain. This pivot introduces three primary engineering challenges:

1. A new data ingestion pipeline must be built in Rust to convert raw source code into graph and point-cloud representations that tcs-tda already consumes
2. A novel "ground truth" dataset for code topology must be constructed
3. A fine-tuning strategy must integrate Qwen2.5-Coder with existing topological optimizers

## I. Implementation Roadmap: The NIODOO-Code Engine

### 1.1 Core Rust Implementation Path: From Text to Graph

The existing TDA pipeline is robust. The primary task is building the pre-processing pipeline that feeds it.

**Core data flow:**
1. `Code (String)` → `tree-sitter::Tree`: Parse into AST using tree-sitter
2. `tree-sitter::Tree` → `petgraph::Graph`: Convert to CFG/DFG using tree-sitter-graph
3. `petgraph::Graph` → `ndarray::Array2`: Convert to adjacency matrix
4. `ndarray::Array2` → `FFI Bridge`: Zero-copy pass to Python via pyo3-async-runtimes
5. `FFI Bridge` → `giotto-tda`: Compute persistent homology from adjacency matrix
6. `giotto-tda` → `PersistenceDiagram`: Return topological signature to Rust

### 1.2 Required Rust Crates and Ingestion Pipeline

| Crate | Version | Purpose |
|-------|---------|---------|
| tree-sitter | ~0.22 | Core parser generator |
| tree-sitter-rust | ~0.21 | Rust language grammar |
| tree-sitter-python | ~0.21 | Python language grammar |
| tree-sitter-graph | ~0.12 | **CRITICAL**: AST → Graph converter |
| petgraph | ~0.6 | Standard Rust graph library |
| ndarray | ~0.15 | Already in NIODOO stack |
| rust-code-analysis | ~0.8 | Extract cyclomatic/cognitive complexity |

### 1.3 Code Example 1: tree-sitter AST Parsing (Rust)

```rust
// In tcs-parser/src/lib.rs
use tree_sitter::{Parser, Language};

// Extern the language grammars (must be included in build.rs)
extern "C" { fn tree_sitter_rust() -> Language; }
extern "C" { fn tree_sitter_python() -> Language; }

/// Parses a string of source code into a tree-sitter AST.
pub fn get_ast(code: &str, language_name: &str) -> Result<tree_sitter::Tree, String> {
    let mut parser = Parser::new();
    let language = match language_name {
        "rust" => unsafe { tree_sitter_rust() },
        "python" => unsafe { tree_sitter_python() },
        _ => return Err(format!("Unsupported language: {}", language_name)),
    };

    parser.set_language(language)
       .map_err(|e| format!("Failed to set language: {}", e))?;

    let tree = parser.parse(code, None)
       .ok_or_else(|| "Failed to parse code (timeout or error)".to_string())?;

    Ok(tree)
}
```

### 1.4 Code Example 2: tree-sitter-graph DSL for AST-to-Graph

Example `rust_cfg.ssg` file:

```lisp
; This file defines the rules for building a graph from a Rust AST.

; Create a graph node for every function definition
(function_item
  name: (identifier) @fn.name) @fn
{
  (node @fn)
  (attr (@fn) "kind" = "function")
  (attr (@fn) "name" = (source-text @fn.name))
}

; Create a graph node for every let binding
(let_declaration) @let
{
  (node @let)
  (attr (@let) "kind" = "statement")
}

; Create a graph node for every expression statement
(expression_statement) @expr
{
  (node @expr)
  (attr (@expr) "kind" = "statement")
}

; Link statements sequentially within a block
(block
  [
    (let_declaration) @stmt1
    (expression_statement) @stmt1
  ]
 .
  [
    (let_declaration) @stmt2
    (expression_statement) @stmt2
  ])
{
  (edge @stmt1 -> @stmt2)
  (attr (@stmt1 -> @stmt2) "kind" = "next_statement")
}

; Link if expressions to their consequence and alternative blocks
(if_expression
  consequence: (block. (_) @cons_stmt)
  alternative: (else_clause
                 (block. (_) @alt_stmt))?) @if
{
  (node @if)
  (attr (@if) "kind" = "branch")

  (edge @if -> @cons_stmt)
  (attr (@if -> @cons_stmt) "kind" = "if_true")

  (edge @if -> @alt_stmt)
  (attr (@if -> @alt_stmt) "kind" = "if_false")
}
```

### 1.5 Code Example 3: The Asynchronous FFI Strategy (Rust → Python)

```rust
// In tcs-tda/src/ffi.rs
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use ndarray::Array2;
use numpy::{PyArray2, ToPyArray};

#[derive(Debug, Clone)]
pub struct PersistencePair {
    pub dimension: i32,
    pub birth: f64,
    pub death: f64,
}

/// Computes persistent homology asynchronously by bridging tokio and asyncio.
pub async fn compute_persistence_async(matrix: Array2<f64>) -> PyResult<Vec<PersistencePair>> {

    // 1. Acquire GIL and convert ndarray to PyArray2 (zero-copy)
    let py_matrix = Python::with_gil(|py| {
        matrix.to_pyarray(py).to_owned()
    })?;

    // 2. Offload the blocking Python TDA computation to an asyncio thread
    let result_future = Python::with_gil(|py| {
        let gtda_homology = py.import("gtda.homology")?;
        let asyncio = py.import("asyncio")?;

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("metric", "precomputed")?;
        let vr_persistence = gtda_homology.getattr("VietorisRipsPersistence")?.call((), Some(kwargs))?;

        let fit_transform = vr_persistence.getattr("fit_transform")?;

        let py_future = asyncio.getattr("to_thread")?
           .call1((fit_transform, py_matrix))?;

        pyo3_async_runtimes::tokio::into_future(py_future)
    })?;

    // 4. await the Rust Future
    let result_from_python = result_future.await?;

    // 5. Acquire GIL again to parse the Python result back into a Rust struct
    Python::with_gil(|py| {
        let diagrams: &numpy::PyArray3<f64> = result_from_python.extract(py)?;
        let data = diagrams.readonly();
        let mut rust_diagrams = Vec::new();

        for row in data.as_array().rows() {
             rust_diagrams.push(PersistencePair {
                birth: row[0],
                death: row[1],
                dimension: row[2] as i32,
             });
        }
        Ok(rust_diagrams)
    })
}
```

### 1.6 Code Transferability Analysis (1:1 vs. Modification)

| Component | Transferability | Modification Required |
|-----------|-----------------|----------------------|
| tcs-core (Traits, TopologicalSignature) | 1:1 Transfer | None - domain-agnostic |
| tcs-tda (TDA Computation) | Modify | Producer must be replaced |
| tcs-tqft (Knot Theory, Jones Poly) | Modify | Input changes to code execution/call graph trajectory |
| tcs-optimizer (argmin, Cost Function) | 1:1 Transfer | Domain-agnostic |
| GPU Pipeline & QLoRA Loop | 1:1 Transfer | Domain-agnostic |
| Async FFI Bridge | 1:1 Transfer | Critical 1:1 component |

## II. Training Data Construction Blueprint

### 2.1 Sourcing Quality Code: The BigQuery GitHub Dataset

**Dataset:** `bigquery-public-data.github_repos.contents` + `bigquery-public-data.github_repos.commits`

**Query Strategy:**
1. Filter contents for files ending in `.rs` or `.py` with size < 1MB
2. Join with commits on `repo_name` and `path`
3. Group by file path and count commits (`COUNT(commit) AS churn`)
4. Select the top 50,000-100,000 most-churned files

**Justification:** High churn = business-critical and problematic code

### 2.2 The "Ground Truth" Problem: A Proxy-Based Labeling Strategy

**Labeling Pipeline:**
1. Parse File: Ingest each of the 100k files
2. Calculate Static Metrics using `rust-code-analysis`:
   - `cyclomatic_complexity`
   - `cognitive_complexity`
   - `lloc` (Logical Lines of Code)
3. Get Churn Metric from BigQuery
4. Compute CQS Label:

```
CQS = (w_cc · norm(avg_cyclomatic_complexity) +
       w_cog · norm(avg_cognitive_complexity) +
       w_churn · norm(churn))
```

### 2.3 Training Data Format and Sizing

```rust
// In tcs-data/src/lib.rs
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTopologicalData {
    /// The raw source code of the function or file.
    pub code_str: String,

    /// The adjacency matrix of the CFG/DFG
    pub graph_adj: Vec<f32>,
    pub graph_dim: (usize, usize),

    /// The ground truth persistence diagram
    pub topology_signature: Vec<(f64, f64, i32)>,

    /// The ground truth proxy label
    pub label_cqs: f32,
}
```

**Size:** 50,000 - 100,000 high-quality labeled examples

### 2.4 Strategic Use of Your 100k-line ADHD Codebase

**DO NOT TRAIN ON IT** - Training would cause catastrophic overfitting.

**USE AS VALIDATION SET** - This is "patient zero" for validating RCE tracking (dBetti/dt) on complex, real-world "thought-knots".

## III. Model Selection and Fine-tuning Strategy

### 3.1 Model Selection: Qwen2.5-Coder

**Model:** Qwen2.5-Coder (32B-Instruct)

**Justification:** State-of-the-art open model, outperforms GPT-4o/CodeLlama/StarCoder, supported by candle-transformers

### 3.2 QLoRA Configuration

```yaml
base_model: "Qwen/Qwen2.5-Coder-32B-Instruct"
model_type: "Qwen2"
tokenizer_type: "Qwen2Tokenizer"
load_in_4bit: true
strict: false

datasets:
  - path: "path/to/your/code_topo_data.parquet"
    type: "arrow"

val_set_size: 0.05
adapter: "qlora"
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"

sequence_len: 4096
sample_packing: true
gradient_accumulation_steps: 4
per_device_train_batch_size: 1
num_epochs: 3
learning_rate: 2.0e-5
bf16: true
gradient_checkpointing: true
optim: "paged_adamw_8bit"
```

### 3.3 Implementation: The Composite Loss Function

**Concept:**
```
L_total = L_crossentropy + λ · L_topo
```

- `L_crossentropy`: Standard LLM loss for next-token prediction (correctness)
- `L_topo`: NIODOO loss (structure quality) - Wasserstein distance between predicted and ground-truth topology

**Implementation:** Requires differentiable TDA pipeline using `torch-tda` or `torchph`

### 3.4 Preserving Topological Weights When Pivoting

**Strategy:** Adapter-based Task Orthogonalization

1. Freeze Base Model: Qwen2.5-Coder weights frozen (standard QLoRA)
2. Load & Freeze Emotional Adapters: Mark as `requires_grad=False`
3. Initialize New Code Adapters: Create new QLoRA adapters
4. Orthogonal Initialization: Minimize destructive interference
5. Train: Only update code adapters

### 3.5 Expected Training Time and Compute

- **Dataset:** 100,000 samples
- **Model:** Qwen2.5-Coder-32B
- **Hardware:** 4 x H100 80GB GPUs
- **QLoRA (4-bit) VRAM:** ~20GB for weights + optimizer
- **Training Time:** 24-36 hours for 3 epochs

## IV. Validation, Metrics, and Demonstration

### 4.1 Benchmark Selection: Proving Structural Reasoning

**Primary Benchmark:** HiBench
- Tests hierarchical reasoning capabilities
- Includes "Code Scenario" for code structure understanding
- Tasks: Time/Space Complexity prediction

**Secondary Benchmark:** DSR-Bench
- Evaluates structural reasoning through data structures
- Tests Graph, BST, Heap operations
- Proves abstract algorithmic manipulation

### 4.2 Ablation Study Design: Quantifying Topology's Contribution

- **Group 1 (Control):** Qwen2.5-Coder + QLoRA with `L_crossentropy` only
- **Group 2 (Treatment):** Qwen2.5-Coder + QLoRA + NIODOO with `L_total`
- **Hypothesis:** Group 2 shows >5-10% increase on multi-hop/hierarchical/graph reasoning

### 4.3 The "Killer Demo": Proving Topological Value

**Setup:** Ingest entire 100k-line ADHD codebase

**Action 1 (Cursor):** "Find the most critical architectural flaw"
- Result: RAG-based text snippets, TODO comments, linter errors

**Action 2 (NIODOO-Code):** Same question
- Result: Full topological model, RCE tracking (dBetti/dt)
- Identifies persistent Betti-1 loop (cyclical dependency) across 10 files, 3 modules
- **This is Topological Code MRI** - impossible for RAG

## V. Competitive Analysis and Market Fit

### 5.1 Deep Dive: Competitor Architecture

- **GitHub Copilot:** Text-completion, statistical understanding, local context
- **Cursor:** RAG + tools, context limitation (99% of codebase not retrieved)
- **Jules (Failed):** Lacked environmental awareness, got stuck in loops

### 5.2 The NIODOO Gap: From Tool-Augmentation to Structural Reasoning

**Solves Jules Problem:** RCE tracking (dBetti/dt) provides environmental awareness

**Solves Cursor Problem:** Sees formal mathematical object (simplicial complex), not text snippets - provable global comprehension

### 5.3 "Killer Feature" and Market Fit

**Killer Feature:** "Topological Code MRI for Legacy Enterprise"

**Market Fit:** Legacy/specialized/proprietary codebases (COBOL, FORTRAN, C++, internal DSLs)

**Unique Value:** Language-agnostic tree-sitter - works with any grammar

## VI. Resource Compendium

### Rust Crates (Code)
- tree-sitter: github.com/tree-sitter/tree-sitter
- tree-sitter-graph: github.com/tree-sitter/tree-sitter-graph
- rust-code-analysis: crates.io/crates/rust_code_analysis
- petgraph: crates.io/crates/petgraph

### FFI Crates
- pyo3: crates.io/crates/pyo3
- pyo3-async-runtimes: crates.io/crates/pyo3-async-runtimes

### Python TDA Libs
- giotto-tda: giotto-ai.github.io/gtda-docs
- Gudhi: gudhi.inria.fr
- Ripser++ (GPU): github.com/simonzhang00/ripser-plusplus

### ML Frameworks
- Candle: github.com/huggingface/candle
- Axolotl: github.com/OpenAccess-AI-Collective/axolotl
- torch-tda: github.com/CompTop/torch-tda
- torchph: github.com/c-hofer/torchph

### Datasets
- BigQuery GitHub: cloud.google.com/blog/topics/public-datasets/github-on-bigquery

### Benchmarks
- HiBench: github.com/jzzzzh/HiBench
- DSR-Bench: huggingface.co/collections/vitercik-lab/dsr-bench

## VII. First 48-Hours Action Plan and Risk Assessment

### 7.1 Day 1 Tasks (Today: 8-10 Hours)

**Goal:** Build the complete Rust-native parser and graph-generation CLI

**Setup:**
```bash
cargo new tcs-parser --lib
cd tcs-parser
cargo add tree-sitter tree-sitter-graph petgraph rust-code-analysis clap serde_json ndarray
```

**Task 1 (4 hours):** Implement `get_ast` function (Section 1.3)
- Create CLI that takes file path
- Read code, print S-expression AST
- Add tree-sitter-rust and tree-sitter-python via build.rs

**Task 2 (4 hours):** Create `rust_cfg.ssg` DSL file (Section 1.4)
- Integrate tree-sitter-graph
- Load DSL, execute on AST
- Serialize petgraph to JSON (adjacency list)

**Task 3 (2 hours):** Integrate rust-code-analysis
- Augment JSON output with cyclomatic/cognitive complexity

**Deliverable:** CLI: `tcs-parser analyze --file /path/to/main.rs`
- Outputs JSON with full graph (adjacency matrix) + complexity metrics

### 7.2 Day 2 Tasks (Tomorrow: 8-10 Hours)

**Goal:** Prove full end-to-end TDA pipeline via async FFI bridge

**Setup:**
```bash
cargo add pyo3 pyo3-async-runtimes tokio numpy
pip install giotto-tda numpy
```

**Task 1 (6 hours):** Implement `compute_persistence_async` (Section 1.5)
- Manage tokio runtime, Python GIL, asyncio.to_thread handoff

**Task 2 (4 hours):** Modify Day 1 CLI
- Add `tcs-parser tda --file /path/to/main.rs`
- Call Day 1 logic to get ndarray::Array2
- Pass to `compute_persistence_async`
- Print Vec<PersistencePair> (Betti numbers)

**Deliverable:** Day 2 CLI = MVP
- End-to-end validation: raw code → topological signature in <1s

### 7.3 MVP Definition

**Day 2 CLI is the Minimum Viable Product** - validates entire data pipeline (biggest unknown in domain pivot)

### 7.4 Risk Mitigation

**Risk 1: TDA Latency (>1s)**
- Mitigation 1: Use sparse graph representations (SparseRipsPersistence)
- Mitigation 2: Re-target FFI to Ripser++ (GPU-accelerated)

**Risk 2: "Topological Ground Truth" is Noise**
- Mitigation: Ablation study (Section 4.2)
- If L_topo provides no lift, pivot to feature extractor (concatenate to LLM input embeddings)

---

**CRITICAL REMINDERS:**
- ✅ Real implementations with proper error handling
- ✅ Mathematical rigor (consciousness topology!)
- ✅ Performance-first (real-time consciousness simulation)
- ✅ Memory safety (Rust ownership model)
- ❌ NO HARD CODING
- ❌ NO PRINTLN/PRINT (use log crate)
- ❌ NO STUBS
- ❌ NO PYTHON SCRIPTS (absolute last resort)
- ❌ NO BULLSHITTING
