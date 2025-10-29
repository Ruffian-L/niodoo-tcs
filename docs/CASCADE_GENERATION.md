# Cascade Generation System

The RAG generation system (`src/rag/generation.rs`) implements a sequential fallback cascade across multiple LLM providers.

## Cascade Order

1. **Mock Mode** (`NIODOO_GENERATION_MOCK=1`) - Echoes back the prompt
2. **Claude** (Anthropic) - If `ANTHROPIC_API_KEY` is set
3. **OpenAI GPT** - If `OPENAI_API_KEY` is set  
4. **vLLM** - Local/self-hosted fallback
5. **Error** - If all fallback

## Environment Variables

### Required (Optional - System activates providers based on which keys are present)

```bash
# Claude/Anthropic
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export ANTHROPIC_MODEL="claude-3-opus-20240229"  # Default
export ANTHROPIC_ENDPOINT="https://api.anthropic.com/v1/messages"  # Default

# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"  # Default
export OPENAI_ENDPOINT="https://api.openai.com/v1/chat/completions"  # Default

# vLLM (Local fallback)
export NIODOO_VLLM_ENDPOINT="http://127.0.0.1:5001"  # Default (matches .env)
export NIODOO_VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"  # Default
```

### Optional Configuration

```bash
# Generation timeout (seconds)
export NIODOO_GENERATION_TIMEOUT=30

# Maximum tokens per generation
export NIODOO_GENERATION_MAX_TOKENS=512

# Temperature for generation
export NIODOO_GENERATION_TEMPERATURE=0.6

# RAG context management
export NIODOO_RAG_MAX_CONTEXT=2400
export NIODOO_RAG_TOP_K=5
export NIODOO_RAG_SIMILARITY_THRESHOLD=0.32
export NIODOO_RAG_TOKEN_ADJUSTMENT=0.0065

# Enable mock mode (disabled by default)
export NIODOO_GENERATION_MOCK=1
```

## Usage

The cascade is automatic. Simply ensure your desired API keys are set, and the system will:

1. Try each provider in sequence
2. Log warnings on fallback (`warn!` level)
3. Track the source in the `GenerationOutcome` structure
4. Return the first successful response

## Implementation Notes

- **Retrieval**: Remains purely local (no external dependencies)
- **Embeddings**: Uses local embedding generator (`local_embeddings` module)
- **Fallback**: Graceful degradation - attempts each provider before erroring
- **Timeout**: 30 second HTTP timeout per request
- **Serialization**: Uses standard request/response structs for each provider

## Example Flow

```rust
let mut rag = RagGeneration::new(RagRuntimeConfig::default())?;
let response = rag.generate("Query text", &mut state)?;
// Logs: "🧠 RAG generation complete (confidence 0.XX)" with source annotation
```

## Verification

Run `cargo check -p niodoo-consciousness` to verify compilation with your configuration.

