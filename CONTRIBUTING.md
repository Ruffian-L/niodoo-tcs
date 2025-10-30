# Contributing to NIODOO-TCS

Thank you for your interest in contributing to NIODOO-TCS!

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:
1. Check existing issues first
2. Create a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass (`cargo test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow Rust formatting (`cargo fmt`)
- Run clippy (`cargo clippy`)
- Add documentation comments
- Write unit tests for new features

### Testing

- All new features must include tests
- Run validation tests: `cargo run --release --bin qwen_comparison_test`
- Run soak tests: `cargo run --release --bin soak_validator`

### Documentation

- Update README.md if adding features
- Add doc comments to public APIs
- Update RELEASE_NOTES.md for significant changes

## Development Setup

See [GETTING_STARTED.md](GETTING_STARTED.md) for setup instructions.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (AGPL 3.0 or Commercial License).

## Questions?

Open an issue or contact: jasonvanpham@niodoo.com

