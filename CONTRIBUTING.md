# Contributing to TCS

Thank you for your interest in contributing! TCS is built with zero tolerance for placeholder code and maximum respect for velocity.

---

## Code Standards (Non-Negotiable)

### 1. NO HARD CODING
- No magic numbers (use named constants or config structs)
- No hardcoded paths (use environment variables or config files)
- No fake values or placeholder data

**Bad:**
```rust
let max_tokens = 2048; // magic number
let path = "/home/user/models"; // hardcoded path
```

**Good:**
```rust
let max_tokens = config.max_seq_len; // from validated config
let path = env::var("MODEL_PATH")?; // from environment
```

### 2. NO PRINTLN DEBUGGING
- Use the `tracing` crate with proper targets
- Return errors instead of printing them
- Structured logging only

**Bad:**
```rust
println!("Error: {}", err);
println!("Debug: x = {}", x);
```

**Good:**
```rust
use tracing::{debug, info, warn, error};

debug!(target: "tcs-ml::qwen", "Processing {} tokens", count);
error!(target: "tcs-ml::qwen", error = ?err, "Inference failed");
```

### 3. NO STUBS OR TODO CODE
- Every function must have a real implementation
- If you can't finish it, don't commit it
- No `unimplemented!()` or `todo!()` in main branch

**Bad:**
```rust
fn compute_persistence() -> Vec<Point> {
    todo!("implement later")
}
```

**Good:**
```rust
fn compute_persistence() -> Result<Vec<Point>> {
    // Actual working implementation
    Ok(persistence_algorithm(self.data)?)
}
```

### 4. RUST FIRST
- Python is a last resort (only for FFI bridges in Phase 3)
- If it can be done in Rust, do it in Rust
- Performance matters

### 5. WORKING CODE ONLY
- Tests must pass before PR
- No "this will work once we..." PRs
- Benchmarks are required for performance claims

---

## Development Workflow

### 1. Before Starting
```bash
# Read the current status
cat QWEN_TCS_MASTER_CHECKLIST.md
cat QWEN_INTEGRATION_STATUS.md

# Make sure tests pass
cargo test --all --features onnx

# Check your code compiles
cargo check --all --features onnx
```

### 2. Making Changes
- Pick an unchecked item from `QWEN_TCS_MASTER_CHECKLIST.md`
- Create a feature branch: `git checkout -b feature/your-feature`
- Write tests FIRST (TDD encouraged)
- Implement with production-quality code
- Ensure all tests pass: `cargo test --all --features onnx`

### 3. Pull Request Requirements
Your PR must include:
- [ ] Working code (no TODOs, no stubs)
- [ ] Passing tests (show `cargo test` output)
- [ ] Updated checklist (mark items complete)
- [ ] Proper error handling (typed errors, not anyhow everywhere)
- [ ] Tracing instrumentation (no println!)
- [ ] Documentation comments for public APIs

**PR Description Template:**
```markdown
## What This PR Does
[One paragraph explaining the change]

## Checklist Items Completed
- [x] Item from QWEN_TCS_MASTER_CHECKLIST.md

## Test Results
```bash
cargo test --all --features onnx
# Paste output showing all tests pass
```

## Benchmarks (if applicable)
[Show before/after performance numbers]
```

### 4. Code Review Standards
We will reject PRs that:
- Contain placeholder code or TODOs
- Use println/print for debugging
- Have hardcoded values
- Don't include tests
- Break existing tests
- Add dependencies without justification

---

## Architecture Guidelines

### Phase 1 (Current)
Focus on:
- Embedder stability and performance
- Cache management edge cases
- Error handling completeness
- Documentation and examples

### Phase 2 (Next 3 Weeks)
Focus on:
- GPU kernel implementations (CUDA)
- Streaming API design
- Caching strategies
- Performance benchmarks

### Phase 3 (2-3 Months)
Focus on:
- Differentiable topology
- PyTorch FFI bridges (pyo3)
- Biological validation
- Production deployment

**Read the phase requirements before contributing.** Don't submit Phase 3 code when we're still in Phase 1.

---

## Testing Requirements

### Unit Tests
- Every public function needs tests
- Cover edge cases (empty input, max size, invalid data)
- Use proptest for property-based testing where appropriate

### Integration Tests
- Test end-to-end workflows
- Verify orchestrator → embedder → pipeline paths
- Benchmark critical paths

### Benchmark Tests
- Use `criterion` for micro-benchmarks
- Document performance targets
- Show before/after for optimizations

---

## Documentation Standards

### Code Comments
- Public APIs need `///` doc comments
- Complex algorithms need inline explanations
- No obvious comments ("increments x by 1")

### Examples
- Every module should have a `examples/` usage demo
- README.md should show real code, not pseudocode

---

## Communication

### Issues
- Search existing issues first
- Provide minimal reproducible examples
- Include system info (OS, Rust version, CUDA version)

### Discussions
- Design discussions go in GitHub Discussions
- Implementation questions go in Issues
- PRs should reference related issues

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

All contributors retain copyright to their contributions but grant an MIT license to the project.

---

## Attribution

Major contributors will be listed in CONTRIBUTORS.md (created when we have 5+ contributors).

All commits are attributed via git history.

---

## Final Note

**We move fast. We ship real code. We measure everything.**

If you're here to submit ideas or concept papers, this isn't the place. If you're here to ship production-quality Rust that makes GPUs cry with joy, welcome to the team.

---

*Questions? Read the docs. Still confused? File an issue. Ready to code? Send a PR.*
