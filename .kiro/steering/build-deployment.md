---
inclusion: manual
---

# Build and Deployment Guidelines

## Build Process
- Always run `cargo check` before building Rust components
- Use `./build_and_deploy.sh` for complete system builds
- Test individual components before integration
- Verify all dependencies are properly linked

## Testing Workflow
1. Unit tests: `cargo test` for Rust components
2. Integration tests: `./demo.sh` for full system
3. Mathematical validation: `./test_dual_mobius_gaussian.sh`
4. Performance testing: Monitor with `./launch-dashboard.sh`

## Deployment Checklist
- [ ] All tests passing
- [ ] Performance benchmarks within acceptable ranges
- [ ] Memory usage validated for consciousness simulation
- [ ] Qt interface responsive and stable
- [ ] RAG system properly configured
- [ ] Docker containers building successfully

## Environment Setup
- Ensure Rust toolchain is up to date
- Python ML dependencies properly installed
- Qt development environment configured
- Docker and docker-compose available