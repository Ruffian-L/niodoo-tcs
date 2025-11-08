# Active Embeddings System - Requirements

## Vision
Build a Rust-native system that automatically embeds the entire Niodoo codebase and keeps embeddings synchronized with code changes in real-time. Embeddings should be portable, git-trackable (optionally), and integrated with the existing RAG system.

## Core Requirements

### 1. Automatic Code Embedding
- Watch all source files (.rs, .cpp, .h, .qml, .py, .md) for changes
- Generate embeddings on file save/modification
- Support batch embedding of entire codebase on first run
- Handle ~29,000 files efficiently

### 2. Embedding Storage Strategy
- Store embeddings as sidecar files: `file.rs` → `file.rs.embedding`
- Use efficient binary format (safetensors or .npy)
- Optionally store as JSON for human readability
- Keep embeddings in sync with ChromaDB for fast retrieval

### 3. Rust-Native Implementation
- Use `fastembed-rs` for embedding generation (no Python dependency)
- Use `notify` crate for file watching
- Use `rayon` for parallel batch processing
- Integrate with existing EchoMemoria consciousness system

### 4. ChromaDB Integration
- Dual-write: update both sidecar file AND ChromaDB
- Maintain metadata: file path, last modified, embedding version
- Support incremental updates (only changed files)
- Preserve existing 109k documentation embeddings

### 5. Performance Requirements
- Embed single file: <100ms
- Batch embed entire codebase: <10 minutes
- File watcher latency: <500ms from save to embedded
- Memory efficient: stream large files, don't load all at once

### 6. Git Integration
- Add `.embedding` to .gitignore by default
- Provide option to commit embeddings for portability
- Include regeneration script for fresh clones
- Track embedding metadata in git-friendly format

## Technical Specifications

### Embedding Model
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Same model as current RAG system for compatibility
- Consider upgrading to `bge-small-en-v1.5` for better code understanding

### File Watching
- Watch patterns: `**/*.{rs,cpp,h,qml,py,md}`
- Ignore patterns: `target/`, `.venv/`, `node_modules/`, `.git/`
- Debounce: 1 second (avoid re-embedding during rapid saves)

### Chunking Strategy
- Small files (<2KB): embed whole file
- Medium files (2-10KB): embed as single chunk with context
- Large files (>10KB): split into 512-token chunks with 50% overlap
- Preserve function/struct boundaries in Rust code

### Metadata Schema
```json
{
  "file_path": "src/consciousness.rs",
  "embedding_version": "1.0",
  "model": "all-MiniLM-L6-v2",
  "timestamp": "2025-10-05T17:30:00Z",
  "file_hash": "sha256:...",
  "chunk_index": 0,
  "total_chunks": 3
}
```

## Integration Points

### With RAG System
- Update `knowledge_base/mcp-rag-server.py` to query code embeddings
- Separate collections: `docs` (existing) and `code` (new)
- Hybrid search: query both collections, merge results
- Prioritize code results for implementation questions

### With EchoMemoria
- Embeddings feed into consciousness state
- Code changes trigger consciousness updates
- Track "code familiarity" metric based on embedding stability
- Use embeddings for semantic code navigation

### With MCP Wrapper
- Add MCP tool: `query_code` (searches code embeddings)
- Add MCP tool: `semantic_diff` (compare embeddings between versions)
- Add MCP tool: `find_similar_code` (find semantically similar functions)

## Success Metrics

1. **Coverage**: 100% of source files have embeddings
2. **Freshness**: Embeddings updated within 1 second of file save
3. **Accuracy**: RAG retrieves correct code 90%+ of the time
4. **Performance**: No noticeable IDE lag from embedding generation
5. **Reliability**: System recovers gracefully from crashes/restarts

## Non-Goals (For MVP)

- ❌ Semantic code search UI (use existing RAG interface)
- ❌ Cross-repository embedding sync
- ❌ Embedding compression/quantization
- ❌ Multi-language model support (stick to one model)
- ❌ Distributed embedding generation

## Future Enhancements

- Semantic git diff visualization
- Code clone detection via embedding similarity
- Auto-documentation generation from embeddings
- Consciousness-aware code suggestions
- Embedding-based code review assistant
