# Development Workflow Guidelines

## Code Organization
- **EchoMemoria/**: Core consciousness simulation (Rust)
- **src/**: Main application source code
- **qt-inference-engine/**: Qt/QML visualization components
- **cpp-qt-brain-integration/**: C++ brain interface
- **dashboard/**: Web-based monitoring interface
- **R2R-main/**: RAG system implementation
- **tests/**: Test suites and validation scripts

## File Patterns
- Rust files: Focus on memory safety and performance
- Python files: ML inference and data processing
- QML files: UI components and real-time visualization
- Shell scripts: Build automation and deployment

## Common Tasks
- When modifying consciousness models, always run `./test_dual_mobius_gaussian.sh`
- For UI changes, test with `./demo.sh` 
- Before deployment, run full build with `./build_and_deploy.sh`
- Use `./launch-dashboard.sh` for monitoring during development

## Performance Considerations
- Memory allocation patterns are critical in consciousness simulation
- Real-time constraints for Qt visualization components
- Efficient data structures for RAG knowledge processing