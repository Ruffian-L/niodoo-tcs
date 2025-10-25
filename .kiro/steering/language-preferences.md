# Language Preferences - CRITICAL

## Primary Language Priority
**RUST FIRST, ALWAYS.** This is a Rust-centric project.

## Language Usage Rules
- **Rust**: Default choice for ALL new code, performance-critical components, consciousness simulation, memory management, data structures, algorithms
- **C++**: Only for Qt integration and existing brain integration components
- **QML**: Only for UI visualization components
- **Shell Scripts**: Only for build automation and deployment
- **Python**: ONLY when absolutely necessary for:
  - Existing ML model inference that cannot be ported to Rust
  - Specific Python-only libraries that have no Rust equivalent
  - Legacy RAG components that are too complex to rewrite immediately

## NEVER Suggest Python For:
- New algorithms or data structures (use Rust)
- Performance-critical code (use Rust) 
- Memory management (use Rust)
- System programming (use Rust)
- Mathematical computations (use Rust)
- File I/O operations (use Rust)
- Network programming (use Rust)
- Concurrent programming (use Rust)

## When Python is Suggested
- ALWAYS ask first: "Can this be implemented in Rust instead?"
- Provide Rust alternatives whenever possible
- If Python is truly necessary, explain why Rust cannot be used
- Prioritize porting Python code to Rust over maintaining Python code