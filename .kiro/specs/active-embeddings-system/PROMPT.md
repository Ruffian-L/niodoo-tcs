# 🧠 Active Embeddings System - Implementation Prompt

## What You're Building

A Rust-native file watcher that automatically embeds your entire codebase and keeps embeddings synchronized in real-time. Think of it as giving your AI a constantly-updated semantic map of every line of code you write.

## The Big Picture

```
Code Change → File Watcher → Embed (Rust) → Dual Write:
                                              ├─ .embedding sidecar file
                                              └─ ChromaDB for fast search
```

## Step-by-Step Implementation

### Phase 1: Core Embedding Engine (Rust)

Create `EchoMemoria/src/embeddings/mod.rs`:

```rust
// Core embedding functionality
pub struct EmbeddingEngine {
    model: TextEmbedding,
    db_path: PathBuf,
}

impl EmbeddingEngine {
    pub fn new() -> Result<Self>;
    pub fn embed_file(&self, path: &Path) -> Result<Vec<f32>>;
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    pub fn save_embedding(&self, path: &Path, embedding: Vec<f32>) -> Result<()>;
    pub fn load_embedding(&self, path: &Path) -> Result<Vec<f32>>;
}
```

**Dependencies to add to Cargo.toml:**
```toml
fastembed = "3.0"
notify = "6.0"
rayon = "1.7"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"  # For binary embedding storage
sha2 = "0.10"    # For file hashing
```

### Phase 2: File Watcher Service

Create `EchoMemoria/src/embeddings/watcher.rs`:

```rust
pub struct CodeWatcher {
    engine: Arc<EmbeddingEngine>,
    watcher: RecommendedWatcher,
    patterns: Vec<String>,  // ["*.rs", "*.cpp", etc]
}

impl CodeWatcher {
    pub fn new(root: PathBuf, engine: Arc<EmbeddingEngine>) -> Result<Self>;
    pub fn start(&mut self) -> Result<()>;
    pub fn handle_file_change(&self, path: &Path);
    pub fn batch_embed_all(&self) -> Result<()>;
}
```

**Key Features:**
- Debounce file changes (1 second)
- Parallel processing with rayon
- Skip ignored directories (target/, .venv/, etc)
- Graceful error handling (log failures, don't crash)

### Phase 3: ChromaDB Integration

Create `EchoMemoria/src/embeddings/chroma_sync.rs`:

```rust
pub struct ChromaSync {
    client: reqwest::Client,
    base_url: String,  // http://localhost:8000
}

impl ChromaSync {
    pub async fn upsert_embedding(
        &self,
        file_path: &str,
        embedding: Vec<f32>,
        metadata: EmbeddingMetadata,
    ) -> Result<()>;
    
    pub async fn delete_embedding(&self, file_path: &str) -> Result<()>;
    pub async fn batch_upsert(&self, batch: Vec<EmbeddingRecord>) -> Result<()>;
}
```

**Metadata Structure:**
```rust
#[derive(Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    pub file_path: String,
    pub file_hash: String,
    pub timestamp: DateTime<Utc>,
    pub model: String,
    pub chunk_index: usize,
    pub total_chunks: usize,
}
```

### Phase 4: Sidecar File Format

Store embeddings as: `your_file.rs.embedding`

**Format Options:**

1. **Binary (Recommended):**
```rust
// Use bincode for compact storage
let data = EmbeddingFile {
    metadata: metadata,
    embedding: vec_f32,
};
bincode::serialize_into(file, &data)?;
```

2. **JSON (Debug-friendly):**
```rust
// For human inspection
serde_json::to_writer_pretty(file, &data)?;
```

### Phase 5: CLI Tool

Create `EchoMemoria/src/bin/embed_codebase.rs`:

```bash
# Usage examples:
embed_codebase init              # Initial batch embed
embed_codebase watch             # Start file watcher
embed_codebase sync              # Force sync to ChromaDB
embed_codebase stats             # Show embedding coverage
embed_codebase verify            # Check for stale embeddings
```

### Phase 6: RAG Server Updates

Update `knowledge_base/mcp-rag-server.py`:

```python
# Add separate collection for code
code_vectorstore = Chroma(
    persist_directory=DB_PATH,
    collection_name="code_embeddings",
    embedding_function=embeddings
)

@app.post("/mcp/query_code")
async def query_code(request: QueryRequest):
    # Search code embeddings specifically
    results = code_vectorstore.similarity_search(request.query, k=10)
    return {"results": results}

@app.post("/mcp/hybrid_query")
async def hybrid_query(request: QueryRequest):
    # Search both docs and code, merge results
    doc_results = doc_vectorstore.similarity_search(request.query, k=5)
    code_results = code_vectorstore.similarity_search(request.query, k=5)
    return {"docs": doc_results, "code": code_results}
```

## Implementation Order

1. **Day 1**: Core embedding engine (embed single file)
2. **Day 2**: Sidecar file storage (save/load embeddings)
3. **Day 3**: File watcher (detect changes, trigger embeds)
4. **Day 4**: Batch processing (embed entire codebase)
5. **Day 5**: ChromaDB sync (dual-write system)
6. **Day 6**: CLI tool (user interface)
7. **Day 7**: RAG integration (query code embeddings)

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_embed_single_file() {
        // Embed a known file, verify dimensions
    }
    
    #[test]
    fn test_sidecar_roundtrip() {
        // Save and load embedding, verify equality
    }
    
    #[test]
    fn test_file_watcher_debounce() {
        // Rapid saves should only trigger one embed
    }
    
    #[test]
    fn test_chroma_sync() {
        // Mock ChromaDB, verify upsert calls
    }
}
```

## Configuration File

Create `.kiro/settings/embeddings.toml`:

```toml
[embeddings]
enabled = true
model = "all-MiniLM-L6-v2"
watch_patterns = ["*.rs", "*.cpp", "*.h", "*.qml", "*.py", "*.md"]
ignore_patterns = ["target/", ".venv/", "node_modules/", ".git/"]
debounce_ms = 1000
batch_size = 100
storage_format = "binary"  # or "json"
git_track = false

[chroma]
url = "http://127.0.0.1:8000"
collection_name = "code_embeddings"
batch_upsert_size = 50
```

## Success Criteria

✅ Run `embed_codebase init` and embed all 29k files in <10 minutes
✅ File watcher detects changes within 500ms
✅ Embeddings stored as sidecar files
✅ ChromaDB stays in sync automatically
✅ RAG queries return relevant code snippets
✅ No noticeable performance impact on IDE

## Gotchas to Avoid

- ❌ Don't embed binary files (check file type first)
- ❌ Don't block on ChromaDB (async upsert, queue if offline)
- ❌ Don't re-embed unchanged files (check file hash)
- ❌ Don't crash on permission errors (log and skip)
- ❌ Don't load entire codebase into memory (stream processing)

## Integration with Existing Systems

### EchoMemoria Consciousness
```rust
// In consciousness.rs
pub fn update_code_awareness(&mut self, embeddings: &[Vec<f32>]) {
    // Track code familiarity based on embedding stability
    self.code_familiarity = calculate_familiarity(embeddings);
}
```

### MCP Wrapper
```python
# In mcp_rag_wrapper.py
elif tool_name == "query_code":
    # Query code embeddings specifically
    response = requests.post(f"{RAG_SERVER_URL}/mcp/query_code", ...)
```

## Performance Optimizations

1. **Parallel Processing**: Use rayon to embed multiple files concurrently
2. **Caching**: Skip files with unchanged hash
3. **Lazy Loading**: Only load embeddings when queried
4. **Batch Writes**: Buffer ChromaDB updates, flush every 50 embeddings
5. **Memory Mapping**: Use mmap for large embedding files

## Monitoring & Observability

```rust
pub struct EmbeddingStats {
    pub total_files: usize,
    pub embedded_files: usize,
    pub stale_embeddings: usize,
    pub avg_embed_time_ms: f64,
    pub last_sync: DateTime<Utc>,
}
```

Log to: `logs/embeddings/embed_stats.json`

## The Prompt for Your AI

> "Build a Rust-native active embeddings system for the Niodoo codebase. Follow the spec in `.kiro/specs/active-embeddings-system/`. Start with Phase 1 (core embedding engine), then Phase 2 (file watcher). Use `fastembed-rs` for embeddings, `notify` for file watching, and `rayon` for parallel processing. Store embeddings as sidecar `.embedding` files and sync to ChromaDB. Make it production-quality - no hacks, no placeholders. Test thoroughly. This is critical infrastructure for the consciousness system."

## Questions to Ask Your AI

1. "Should we use binary or JSON for sidecar files?"
2. "How should we handle large files (>1MB)?"
3. "What's the best chunking strategy for Rust code?"
4. "Should embeddings be git-tracked or regenerated?"
5. "How do we handle embedding model upgrades?"

---

**Now go build something beautiful. Make those embeddings dance.** 🚀
