# Debugging and Troubleshooting Guide

## Common Issues
- **Consciousness Model Convergence**: Check mathematical parameters in Dual Möbius Gaussian
- **Memory Leaks**: Monitor Rust memory allocation patterns in EchoMemoria
- **Qt Rendering**: Verify QML component performance and GPU usage
- **RAG Performance**: Check knowledge base indexing and retrieval speeds

## Debugging Tools
- Use `cargo test` for Rust component testing
- Qt Creator debugger for QML/C++ issues  
- Python profiler for ML inference bottlenecks
- Dashboard monitoring for real-time system health

## Log Locations
- Consciousness simulation logs: `./logs/consciousness/`
- Qt application logs: `./logs/qt/`
- RAG system logs: `./logs/rag/`
- Build logs: `./build/logs/`

## Performance Monitoring
- Memory usage patterns during consciousness simulation
- Real-time rendering performance in Qt components
- RAG query response times
- Overall system resource utilization