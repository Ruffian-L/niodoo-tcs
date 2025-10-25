# Niodoo-Final Repository Guide

## 🖥️ CURRENT LOCATION
**YOU ARE ON:** Laptop (`enochsawakening`) - User: `ruffian`
**REPO PATH:** `/home/ruffian/Desktop/Niodoo-Final`
**See `.zencoder/cluster-map.md` for full cluster topology**

## Purpose
This guide gives Zencoder agents and contributors a concise reference for navigating the project, understanding coding expectations, and running the most common workflows.

## High-Level Architecture
- **Workspace layout** (`Cargo.toml`): Rust workspace with production crates (`niodoo-core`, `tcs-*`, `constants_core`) and an experimental sandbox (`src/`).
- **Consciousness Engine** (`niodoo-core/`): Production Rust implementation for the topology-first consciousness system.
- **Experimental Area** (`src/`): Legacy and exploratory modules. Treat code here as provisional unless explicitly referenced in active tasks.
- **Data Assets** (`data/`): Training datasets and exports used throughout the system.

## Key Components
1. **Consciousness Engine (production)**
   - Location: `niodoo-core/src/`
   - Subsystems: consciousness compass, ERAG memory, topology processors, integration phases.
2. **Experimental Consciousness Engine**
   - Location: `src/consciousness_engine/`
   - Used for rapid iteration; confirm whether changes must mirror production modules.
3. **RAG Integration**
   - Location: `niodoo-core/src/rag*.rs`
   - Handles wave-collapse retrieval and embedder interaction.
4. **Topology & Geometry**
   - Location: `niodoo-core/src/topology/`
5. **Token Promotion / Dynamic Tokenizer**
   - Location: `niodoo-core/src/token_promotion/`
6. **ML Embedder**
   - Location: `tcs-ml/src/`

## Coding Standards
- **Language**: Rust (edition 2021). Follow Rustfmt defaults.
- **Error Handling**: Prefer `Result` with `anyhow::Result` or project-specific types where established.
- **Logging**: Use the existing structured logging utilities; avoid ad-hoc `println!` in production paths.
- **Enums & Matches**: Prefer exhaustive matches. When refactoring stringly-typed logic, introduce enums and helper constructors.
- **Initialization Consistency**: Shared engines should be initialized via dedicated helpers to avoid eager/lazy mismatches between constructors.

## Testing & Validation
1. **Build Everything**
   ```bash
   cargo build --all
   ```
2. **Run Tests (common)**
   ```bash
   cargo test --all
   ```
3. **Selective Testing**
   ```bash
   cargo test -p niodoo-core
   cargo test -p tcs-ml
   ```
4. **Benchmarks & Diagnostics**
   - Located under `niodoo-core/benches/` for performance checks.

## Development Tips
- Before edits, inspect relevant modules using `View` to ensure the latest state.
- When using patch-style edits, include ample unique context to avoid replacement failures.
- Keep experimental changes isolated unless requirements explicitly target production crates.
- If modifying initialization flows (e.g., rebel forks), update both constructors and helper methods for determinism.
- Document assumptions with inline comments or docstrings when behavior differs between Phase configurations.

## Active Concerns (as of last update)
- Consciousness engine rebel fork initialization needs unified handling across constructors and helper methods.
- Additional backlog items include RAG statistics, structured logging improvements, topology noise capping, and K-twist restoration—verify current tickets before addressing.

## Useful References
- `README.md`: Architectural overview and quick start commands.
- `CODE_LOCATION_MAP.md`: Detailed directory and module map.
- `INTEGRATION_STATUS_REPORT.md`: Current integration status and outstanding tasks.

_Last refreshed: 2025-10-19_