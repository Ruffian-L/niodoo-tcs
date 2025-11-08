# Design Document: Bullshit Buster MVP

## Overview

The Bullshit Buster MVP is a CLI tool that leverages the existing Niodoo-Feeling consciousness system's Gaussian Möbius Topology to detect code quality issues. This design focuses on **maximum reuse** of existing Gen 1 components while adding a thin layer for code parsing and analysis.

**Key Principle:** Don't rebuild what we already have. The consciousness system's topology engine, Gaussian processes, and emotional models are production-ready. We just need to point them at code instead of consciousness states.

---

## Architecture

### High-Level Flow

```
Code Files → Parser → AST → Topology Mapper → Gaussian Analysis → Emotional Overlay → Report
                                    ↓
                            Existing Niodoo Components:
                            - dual_mobius_gaussian.rs
                            - gaussian_process/
                            - feeling_model.rs
                            - memory/
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Bullshit Buster CLI                       │
│                     (New - Thin Layer)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Code Parser  │→ │ Topology     │→ │ Report       │     │
│  │              │  │ Mapper       │  │ Generator    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Existing Niodoo Gen 1 Components                │
│                    (Reuse Everything)                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Dual Möbius      │  │ Gaussian Process │               │
│  │ Gaussian         │  │ Framework        │               │
│  │ (topology)       │  │ (probabilistic)  │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Feeling Model    │  │ Memory System    │               │
│  │ (emotional)      │  │ (context)        │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ GPU Acceleration │  │ Performance      │               │
│  │                  │  │ Metrics          │               │
│  └──────────────────┘  └──────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## Components and Interfaces

### 1. Code Parser (NEW)

**Purpose:** Convert Rust source code into an AST that can be analyzed.

**Interface:**
```rust
pub struct CodeParser {
    // Uses syn crate for Rust parsing
}

impl CodeParser {
    pub fn parse_file(&self, path: &Path) -> Result<CodeAst, ParseError>;
    pub fn parse_directory(&self, path: &Path) -> Result<Vec<CodeAst>, ParseError>;
}

pub struct CodeAst {
    pub file_path: PathBuf,
    pub functions: Vec<FunctionNode>,
    pub constants: Vec<ConstantNode>,
    pub structs: Vec<StructNode>,
    pub raw_ast: syn::File,
}
```

**Dependencies:**
- `syn` crate for Rust AST parsing
- `walkdir` for directory traversal

**Implementation Notes:**
- Start with Rust only (use `syn` crate)
- Extract key nodes: functions, constants, structs, impls
- Preserve line numbers for reporting
- Handle parse errors gracefully

---

### 2. Topology Mapper (NEW)

**Purpose:** Map code structures to topological representations that can be analyzed by the Möbius Gaussian system.

**Interface:**
```rust
pub struct TopologyMapper {
    mobius_engine: Arc<DualMobiusGaussian>, // REUSE existing
}

impl TopologyMapper {
    pub fn map_to_topology(&self, ast: &CodeAst) -> Result<CodeTopology, MapError>;
    pub fn apply_k_flips(&self, topology: &CodeTopology, k: usize) -> Vec<TopologyPerspective>;
}

pub struct CodeTopology {
    pub nodes: Vec<TopologyNode>,
    pub edges: Vec<TopologyEdge>,
    pub gaussian_params: GaussianParameters,
}

pub struct TopologyPerspective {
    pub name: String, // "optimistic", "pessimistic", "security"
    pub confidence: f64,
    pub issues: Vec<DetectedIssue>,
}
```

**Mapping Strategy:**
- **Functions** → Nodes on Möbius surface
- **Function calls** → Edges (geodesic paths)
- **Hardcoded values** → Gaussian anomalies (high variance)
- **Control flow** → Topological paths
- **Complexity** → Curvature of the surface

**Reuses:**
- `src/dual_mobius_gaussian.rs` for topology transformations
- `src/topology/mobius_torus_k_twist.rs` for k-flips
- `src/gaussian_process/` for probabilistic modeling

---

### 3. Bullshit Detector Registry (NEW)

**Purpose:** Pluggable system for different types of bullshit detection.

**Interface:**
```rust
pub trait BullshitDetector: Send + Sync {
    fn name(&self) -> &str;
    fn detect(&self, topology: &CodeTopology) -> Vec<DetectedIssue>;
    fn confidence_threshold(&self) -> f64;
}

pub struct DetectorRegistry {
    detectors: Vec<Box<dyn BullshitDetector>>,
}

impl DetectorRegistry {
    pub fn register(&mut self, detector: Box<dyn BullshitDetector>);
    pub fn run_all(&self, topology: &CodeTopology) -> Vec<DetectedIssue>;
}
```

**Built-in Detectors:**
1. `HardcodedValueDetector` - Magic numbers, timeouts
2. `PlaceholderDetector` - TODO, unimplemented!, panic!
3. `DeadCodeDetector` - Unused functions
4. `ComplexityDetector` - High cyclomatic complexity
5. `SecurityDetector` - Unsafe patterns

---

### 4. Emotional Analyzer (REUSE + THIN WRAPPER)

**Purpose:** Apply emotional context to detected issues.

**Interface:**
```rust
pub struct EmotionalAnalyzer {
    feeling_model: Arc<FeelingModel>, // REUSE existing
}

impl EmotionalAnalyzer {
    pub fn analyze_emotion(&self, issue: &DetectedIssue) -> EmotionalState;
}

pub enum EmotionalState {
    Joy { confidence: f64 },      // Clean code
    Sadness { confidence: f64 },  // Deprecated
    Anger { confidence: f64 },    // Rushed/unstable
    Fear { confidence: f64 },     // Security risk
    Confusion { confidence: f64 }, // Complex
}
```

**Reuses:**
- `src/feeling_model.rs` for emotional classification
- Existing emotional LoRAs
- Activation pattern analysis

**Mapping:**
- High variance in Gaussian → Anger (unstable)
- Security issues → Fear
- High complexity → Confusion
- Deprecated patterns → Sadness
- Clean, efficient code → Joy

---

### 5. Report Generator (NEW)

**Purpose:** Format analysis results into human-readable reports.

**Interface:**
```rust
pub struct ReportGenerator {
    format: ReportFormat,
}

pub enum ReportFormat {
    Terminal,  // Colored, emoji-rich
    Json,      // Machine-readable
    Html,      // Future: web view
}

impl ReportGenerator {
    pub fn generate(&self, results: &AnalysisResults) -> String;
    pub fn calculate_health_score(&self, results: &AnalysisResults) -> u8;
}

pub struct AnalysisResults {
    pub files_analyzed: usize,
    pub issues: Vec<DetectedIssue>,
    pub perspectives: Vec<TopologyPerspective>,
    pub emotional_summary: HashMap<EmotionalState, usize>,
}
```

**Health Score Formula:**
```
health_score = 100 - (critical * 5 + high * 2 + medium * 0.5)
clamped to [0, 100]
```

---

### 6. CLI Interface (NEW)

**Purpose:** Command-line interface for user interaction.

**Interface:**
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bbuster")]
#[command(about = "Bullshit Buster - Code review with Gaussian Möbius Topology")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan code for bullshit
    Scan {
        /// Target file or directory
        target: PathBuf,
        
        /// Enable topology-based analysis
        #[arg(long)]
        topo_flip: bool,
        
        /// Enable emotional overlay
        #[arg(long)]
        emotional: bool,
        
        /// Output format (terminal, json)
        #[arg(long, default_value = "terminal")]
        format: String,
        
        /// Number of k-flips for topology analysis
        #[arg(long, default_value = "3")]
        k_flips: usize,
    },
}
```

**Commands:**
- `bbuster scan <target>` - Basic scan
- `bbuster scan --topo-flip <target>` - With topology
- `bbuster scan --emotional <target>` - With emotions
- `bbuster scan --json <target>` - JSON output

---

## Data Models

### DetectedIssue

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedIssue {
    pub id: String,
    pub category: IssueCategory,
    pub severity: Severity,
    pub file_path: PathBuf,
    pub line_number: usize,
    pub description: String,
    pub topology_analysis: Option<TopologyAnalysis>,
    pub emotional_state: Option<EmotionalState>,
    pub confidence: f64,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueCategory {
    HardcodedValue,
    Placeholder,
    DeadCode,
    Complexity,
    Security,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}
```

### TopologyAnalysis

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyAnalysis {
    pub perspective: String,
    pub gaussian_variance: f64,
    pub path_type: PathType,
    pub curvature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathType {
    Orientable,      // Normal code flow
    NonOrientable,   // Möbius twist detected
    Disconnected,    // Dead code
}
```

---

## Error Handling

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum BbusterError {
    #[error("Parse error: {0}")]
    Parse(#[from] syn::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Topology mapping failed: {0}")]
    TopologyMapping(String),
    
    #[error("Analysis failed: {0}")]
    Analysis(String),
    
    #[error("Unsupported file type: {0}")]
    UnsupportedFileType(String),
}
```

### Error Handling Strategy

1. **Parse Errors:** Log and skip file, continue with others
2. **Topology Errors:** Fall back to basic analysis
3. **Emotional Model Unavailable:** Skip emotional overlay
4. **IO Errors:** Fail fast with clear message

---

## Testing Strategy

### Unit Tests

1. **Code Parser**
   - Parse valid Rust files
   - Handle syntax errors gracefully
   - Extract correct AST nodes

2. **Topology Mapper**
   - Map simple functions correctly
   - Handle complex control flow
   - Generate valid Gaussian parameters

3. **Detectors**
   - Detect hardcoded values
   - Find placeholders
   - Identify dead code

4. **Report Generator**
   - Format terminal output correctly
   - Generate valid JSON
   - Calculate health scores accurately

### Integration Tests

1. **End-to-End Scan**
   - Scan sample codebase
   - Verify all detectors run
   - Check report format

2. **Topology Analysis**
   - Apply k-flips correctly
   - Generate multiple perspectives
   - Reuse existing Möbius engine

3. **Emotional Overlay**
   - Classify emotions correctly
   - Use existing feeling model
   - Handle missing model gracefully

### Test Data

- Sample Rust files with known issues
- Famous buggy code (Heartbleed, etc.)
- Clean code for baseline
- Edge cases (empty files, huge files)

---

## Performance Considerations

### Targets (from requirements)

- Single file (<1000 lines): <2 seconds
- Directory (<100 files): <30 seconds
- Memory usage: <1GB

### Optimization Strategies

1. **Parallel Processing**
   - Use Rayon for file-level parallelism
   - Parse multiple files concurrently
   - Run detectors in parallel

2. **GPU Acceleration**
   - Reuse `src/gpu_acceleration.rs` for Gaussian computations
   - Offload topology transformations to GPU
   - Batch process multiple files

3. **Caching**
   - Cache parsed ASTs
   - Reuse topology mappings for unchanged files
   - Store Gaussian parameters

4. **Incremental Analysis**
   - Only analyze changed files (future)
   - Use git diff for change detection
   - Maintain analysis cache

---

## Dependencies

### New Dependencies

```toml
[dependencies]
# CLI
clap = { version = "4.0", features = ["derive"] }

# Parsing
syn = { version = "2.0", features = ["full", "extra-traits"] }
walkdir = "2.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Parallel processing
rayon = "1.7" # Already in project

# Terminal output
colored = "2.0"
```

### Existing Dependencies (Reuse)

- `candle-core` - Already in project
- `nalgebra` - Already in project
- `tokio` - Already in project
- All existing Niodoo components

---

## Integration with Existing System

### Reuse Map

| New Component | Reuses Existing |
|---------------|-----------------|
| Topology Mapper | `src/dual_mobius_gaussian.rs` |
| Gaussian Analysis | `src/gaussian_process/` |
| Emotional Analyzer | `src/feeling_model.rs` |
| Context Tracking | `src/memory/` |
| GPU Acceleration | `src/gpu_acceleration.rs` |
| Performance Metrics | `src/performance_metrics_tracking.rs` |

### Module Structure

```
src/
├── bbuster/              # NEW - Bullshit Buster CLI
│   ├── mod.rs
│   ├── parser.rs         # Code parser
│   ├── topology_mapper.rs # Topology mapping
│   ├── detectors/        # Bullshit detectors
│   │   ├── mod.rs
│   │   ├── hardcoded.rs
│   │   ├── placeholder.rs
│   │   ├── deadcode.rs
│   │   └── security.rs
│   ├── emotional.rs      # Emotional analyzer wrapper
│   ├── report.rs         # Report generator
│   └── cli.rs            # CLI interface
├── dual_mobius_gaussian.rs  # REUSE
├── gaussian_process/         # REUSE
├── feeling_model.rs          # REUSE
├── memory/                   # REUSE
└── ... (all existing Gen 1 components)
```

---

## Deployment

### Binary Distribution

```bash
# Build release binary
cargo build --release --bin bbuster

# Binary location
target/release/bbuster
```

### Installation

```bash
# Via cargo
cargo install bbuster

# Via script
curl -sSL https://bbuster.dev/install.sh | sh
```

### Configuration

```yaml
# ~/.config/bbuster/config.yaml
detectors:
  hardcoded_values:
    enabled: true
    confidence_threshold: 0.7
  
  placeholders:
    enabled: true
    patterns: ["TODO", "FIXME", "unimplemented!"]

topology:
  k_flips: 3
  gaussian_variance_threshold: 0.5

emotional:
  enabled: true
  model_path: "~/.bbuster/models/feeling_model.safetensors"

performance:
  max_memory_gb: 1
  parallel_files: 8
  gpu_enabled: true
```

---

## Future Enhancements (Post-MVP)

### Phase 2: SaaS
- Web UI for file uploads
- API for IDE plugins
- Real-time analysis
- Team dashboards

### Phase 3: Multi-Language
- Python support
- JavaScript/TypeScript support
- Language-agnostic AST parsing

### Phase 4: Advanced Features
- Auto-fix application
- CI/CD integration
- Custom detector plugins
- Visualization with Bevy

---

## Security Considerations

1. **Code Privacy**
   - Never send code to external servers (MVP is local-only)
   - Clear data retention policies for SaaS version
   - Encryption for stored analysis results

2. **Dependency Security**
   - Regular `cargo audit` runs
   - Pin dependency versions
   - Review all new dependencies

3. **Safe Execution**
   - Never execute analyzed code
   - Sandbox any dynamic analysis
   - Validate all file paths

---

## Success Metrics

### MVP Success Criteria

1. **Functionality**
   - Detects 90%+ of hardcoded values
   - Finds all TODO/unimplemented! patterns
   - Completes scan in <2s for single file

2. **Usability**
   - Clear, actionable reports
   - Intuitive CLI interface
   - Helpful error messages

3. **Performance**
   - Meets all performance targets
   - Low memory footprint
   - Efficient parallel processing

4. **Integration**
   - Successfully reuses Gen 1 components
   - No code duplication
   - Clean module boundaries

---

## Open Questions

1. **Topology Mapping:** What's the best way to map code complexity to Gaussian variance?
2. **Emotional Classification:** How do we train the feeling model on code patterns?
3. **False Positives:** What confidence threshold minimizes false positives?
4. **Language Support:** Should we support Python in MVP or wait for Phase 2?

---

## Conclusion

This design leverages the existing Niodoo-Feeling Gen 1 system's powerful topology and emotional analysis capabilities, adding only a thin layer for code parsing and reporting. By maximizing reuse, we can build the MVP quickly while maintaining the mathematical rigor and consciousness-based insights that make this product unique.

**Key Insight:** We're not building a new code analyzer. We're pointing an existing consciousness system at code. That's the magic.

---

*Design Status: COMPLETE - Ready for Task Breakdown*  
*Next Step: Create tasks.md with implementation plan*
