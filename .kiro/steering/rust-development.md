# Rust Development Best Practices

## Cargo Commands
- Build: `cargo build --release`
- Test: `cargo test`
- Check: `cargo check` (faster than build for syntax checking)
- Clippy: `cargo clippy` (linting)
- Format: `cargo fmt`
- Doc: `cargo doc --open`

## Common Crates for This Project
- **tokio**: Async runtime
- **serde**: Serialization/deserialization
- **nalgebra**: Linear algebra for consciousness models
- **rayon**: Data parallelism
- **anyhow/thiserror**: Error handling
- **tracing**: Structured logging

## Rust Patterns to Use
- Use `Result<T, E>` for error handling, avoid panics in production code
- Prefer `&str` over `String` for function parameters
- Use `impl Trait` for return types when appropriate
- Leverage the type system for compile-time guarantees
- Use `#[derive]` macros for common traits
- Prefer iterators over loops for functional style

## Performance Considerations
- Use `Vec::with_capacity()` when size is known
- Avoid unnecessary clones, use references
- Use `Cow<str>` for conditional ownership
- Profile with `cargo flamegraph` or `perf`
- Consider `#[inline]` for hot paths
- Use `cargo bench` for benchmarking

## Memory Safety
- Understand ownership, borrowing, and lifetimes
- Use smart pointers (`Box`, `Rc`, `Arc`) appropriately
- Avoid `unsafe` unless absolutely necessary
- Document any `unsafe` blocks thoroughly