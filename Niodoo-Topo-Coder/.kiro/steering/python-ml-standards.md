---
inclusion: fileMatch
fileMatchPattern: '*.py'
---

# Python ML Standards for Niodoo-Feeling

## RAG System Guidelines
- Use type hints for all function signatures
- Implement proper error handling for ML inference failures
- Cache embeddings and model outputs when appropriate
- Use `asyncio` for concurrent RAG operations

## Memory Management
- Monitor GPU memory usage during inference
- Implement proper cleanup for large tensors
- Use context managers for resource management
- Profile memory usage in consciousness simulation integration

## Integration with Rust
- Use proper serialization (JSON/MessagePack) for Rust-Python communication
- Handle encoding/decoding errors gracefully
- Implement timeout mechanisms for long-running ML operations
- Use structured logging that matches Rust component logs

## Testing
- Unit tests for RAG retrieval accuracy
- Integration tests with EchoMemoria consciousness system
- Performance benchmarks for inference speed
- Mock external ML services in tests