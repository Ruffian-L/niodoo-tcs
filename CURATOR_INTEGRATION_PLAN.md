# ✅ VALIDATED CURATOR INTEGRATION PLAN

## What Actually Exists ✅

1. **QwenEmbedder** (`tcs-ml/src/lib.rs` line 517)
   - Has: `embed()`, `extract_embeddings()`
   - Does NOT have: `generate()` for text generation

2. **Generation Engine** (`niodoo_real_integrated/src/generation.rs`)
   - Has: `select_centroid_candidate()` (line 284)
   - Has: `ConsistencyVotingResult` (line 17)
   - Has: Cascade fallback (Claude/GPT)

3. **Config System** (`niodoo_real_integrated/src/config.rs`)
   - Can add curator settings

## What Needs to Change ❌

The plan's assumption: `embedder.generate()` doesn't exist!

**Options**:

### Option A: Curator Uses vLLM (Recommended)
Use HTTP to vLLM for generation, like the rest of the pipeline:

```rust
pub async fn judge_responses(
    &mut self,
    prompt: &str,
    candidates: &[String],
) -> Result<CuratorJudgment> {
    // Build judgment prompt
    let judgment_prompt = self.build_judgment_prompt(prompt, candidates);
    
    // Call vLLM via HTTP (like GenerationEngine does)
    let response = self.client
        .post(&format!("{}/v1/chat/completions", self.vllm_endpoint))
        .json(&json!({
            "model": self.model_name,
            "messages": [{"role": "user", "content": judgment_prompt}],
            "max_tokens": 128,
        }))
        .send()
        .await?
        .json::<Value>()
        .await?;
    
    // Parse response...
}
```

### Option B: Curator Uses Embeddings Only
Compare embeddings instead of generating text:

```rust
pub async fn judge_responses(
    &mut self,
    prompt: &str,
    candidates: &[String],
) -> Result<CuratorJudgment> {
    // Get embeddings for each candidate
    let prompt_emb = self.embedder.embed(prompt)?;
    let candidate_embs: Vec<Vec<f32>> = candidates
        .iter()
        .map(|c| self.embedder.embed(c))
        .collect::<Result<_>>()?;
    
    // Find most similar (cosine similarity)
    let best_index = self.find_most_similar(&prompt_emb, &candidate_embs);
    
    Ok(CuratorJudgment {
        winner_index: best_index,
        confidence: 0.8, // Could compute from similarity
        reasoning: "Embedding similarity".to_string(),
        latency_ms: ...,
    })
}
```

### Option C: Skip Curator Generation Entirely
Just use ROUGE centroid (current behavior) and add curator later when we have better infrastructure.

## Modified Integration Approach

**Simplest path**: Option B (embedding-based judgment)

```rust
// curator.rs
pub struct QwenCurator {
    embedder: tcs_ml::QwenEmbedder,
    vllm_endpoint: String, // For future expansion
}

impl QwenCurator {
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let embedder = tcs_ml::QwenEmbedder::new(&model_path)?;
        Ok(Self {
            embedder,
            vllm_endpoint: "http://localhost:8000".to_string(),
        })
    }
    
    pub async fn judge_responses(
        &mut self,
        prompt: &str,
        candidates: &[String],
    ) -> Result<CuratorJudgment> {
        // Embed all candidates
        let candidates_text = candidates.join("\n\n");
        let judgment = self.embedder.embed(&format!("Prompt: {}\nCandidates: {}", prompt, candidates_text))?;
        
        // Simple heuristic: candidate with best embedding match wins
        // (This is a placeholder - proper implementation would compare embeddings)
        let winner_index = 0; // TODO: Actual comparison
        
        Ok(CuratorJudgment {
            winner_index,
            confidence: 0.7,
            reasoning: "Embedding-based judgment".to_string(),
            latency_ms: 0.0,
        })
    }
}
```

## Integration Points

1. **config.rs**: Add `enable_curator: bool`
2. **pipeline.rs**: Add `curator: Option<QwenCurator>` field
3. **generation.rs**: Pass curator to `generate_with_consistency_voting()`
4. **main.rs**: Initialize curator if enabled

## What This Gives You

✅ Isolation: Curator module separate and testable  
✅ Fallback: Falls back to ROUGE if curator fails  
✅ Config-driven: Can disable curator  
✅ Future-proof: Easy to enhance later  

## Recommendation

**Start simple**: Use Option B (embedding-based) or even skip curator for now, just prepare the integration points.

**Phase 2 Goal**: Get TCS topology layer working first, then add curator refinement.

Want me to implement the skeleton code with proper integration points?

