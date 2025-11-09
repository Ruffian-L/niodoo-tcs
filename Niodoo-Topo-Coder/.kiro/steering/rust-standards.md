---
inclusion: fileMatch
fileMatchPattern: '*.rs'
---

# Rust Development Standards for Niodoo-Feeling

## Code Style
- Use `cargo fmt` for consistent formatting
- Follow Rust naming conventions (snake_case for functions/variables, PascalCase for types)
- Prefer explicit error handling with `Result<T, E>` over panics
- Use `#[derive(Debug)]` for custom types when appropriate

## Performance Guidelines
- Use `Vec<T>` for dynamic arrays, prefer `&[T]` for function parameters
- Consider `Arc<T>` and `Mutex<T>` for shared state in concurrent contexts
- Profile memory usage for consciousness simulation components
- Use `unsafe` blocks only when absolutely necessary and document thoroughly

## EchoMemoria Specific
- Memory operations should be atomic where possible
- Use structured logging with `log` crate for consciousness events
- Implement proper cleanup for memory simulation resources
- Consider using `tokio` for async operations in memory processing

## Testing
- Unit tests for mathematical models (Möbius Gaussian functions)
- Integration tests for consciousness simulation workflows
- Benchmark critical performance paths
- Mock external dependencies in tests