# Niodoo-TCS: Topological Cognitive System (Integrated)

Niodoo-TCS integrates pipelines, embeddings, learning, and generation into a cohesive Topological Cognitive System (TCS). This crate focuses on the integrated runtime that links analysis → TCS state → generation.

## Setup
- Prereqs: Rust (rustup), clang, BLAS (OpenBLAS recommended).
- Build workspace:
  ```bash
  rustup toolchain install stable
  rustup default stable
  cargo build --workspace --release
  ```
- Recommended env vars:
  - `NIODOO_MODEL_DIR`: directory for models/artifacts
  - `RUST_LOG`: log level (e.g., `info`, `debug`)
  - `OPENAI_API_KEY` or `HF_HOME` if your configuration uses external models

## Usage
- Run the integrated binary (if provided by a member crate) or your own harness:
  ```bash
  cargo run -p niodoo_real_integrated -- --prompt "hello tcs"
  ```
  Flags may include model paths and limits defined in `config`.

## Architecture
- Pipeline → TCS → Generation:
  - Input enters `pipeline`, is embedded via `embedding`, analyzed via `tcs_analysis`.
  - `tcs_predictor` and `learning` update state; `generation` produces outputs.
  - `token_manager`, `metrics`, and `util` support orchestration.

## Benchmarks
- For long-run stability and performance, use the million cycle test:
  ```bash
  cargo bench -p niodoo_real_integrated -- million_cycle_test
  ```
  Track throughput, token usage, and PAD-like state stability.

## License
This repository is provided under the Prosperity Public License 3.0.0 for noncommercial use. For commercial use, you must obtain a commercial license from the Licensor. Contact: jasonvanpham@niodoo.com.
