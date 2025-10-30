# Semantic Code Search (tcs-code-tools)

## Why
Find code by meaning (natural language) across the monorepo using embeddings + Qdrant.

## Setup
```bash
docker run -p 6333:6333 -v $PWD/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_COLLECTION=code_index
```

## Index
```bash
cargo run -p tcs-code-tools --bin code_indexer -- .
# scope to directories
cargo run -p tcs-code-tools --bin code_indexer -- tcs-core/ src/
```

## Query
```bash
cargo run -p tcs-code-tools --bin code_query -- "where are persistent homology features computed?" 10
```

## Notes
- Embeds up to ~16KB per file to keep vector sizes manageable.
- Re-run indexer after large edits; upserts are idempotent.
- Payload stores `path` for easy printing; integrate with your editor jump if desired.

