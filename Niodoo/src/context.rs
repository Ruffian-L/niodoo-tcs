use anyhow::Result;
use serde_json::Value;

use crate::erag::{EragService, SearchResult};

const SYSTEM_DIRECTIVE: &str = "You are NIODOO's System 2 responder. Synthesize the memory \
context with the user prompt to produce a grounded, actionable answer. Reference specific \
memory IDs when you rely on them and identify any gaps honestly. Never invent facts that are \
not supported by the memory payloads.";

const RESPONSE_SCHEMA: &str = r#"{
  "answer": "<succinct response grounded in memories>",
  "memory_evidence": [
    {"memory_id": "<id>", "insight": "<what this memory contributed>"}
  ],
  "topology_alignment": "<how the answer fits the current compass quadrant>",
  "next_actions": ["<optional actionable steps>"],
  "confidence": <integer 0-10>
}"#;

pub async fn augment_prompt_with_memory(
    service: &EragService,
    prompt: &str,
    compass: Option<&str>,
) -> Result<(String, Vec<SearchResult>)> {
    let results = service.embed_and_search(prompt, compass).await?;

    if results.is_empty() {
        return Ok((prompt.to_string(), results));
    }

    let mut sections = Vec::new();
    for (idx, res) in results.iter().enumerate() {
        let payload_json = pretty_payload(&res.payload);
        let id = res.id.as_deref().unwrap_or("(unknown)");
        sections.push(format!(
            "Memory {} (id: {}, score: {:.3}):\n{}",
            idx + 1,
            id,
            res.score,
            payload_json
        ));
    }

    let augmented = format!(
        "{directive}\n\nMemory Context:\n{context}\n\nUser Prompt:\n{prompt}\n\nRespond strictly \
        in JSON matching this schema:\n{schema}",
        directive = SYSTEM_DIRECTIVE,
        context = sections.join("\n\n"),
        prompt = prompt,
        schema = RESPONSE_SCHEMA,
    );

    Ok((augmented, results))
}

fn pretty_payload(value: &Value) -> String {
    match serde_json::to_string_pretty(value) {
        Ok(text) => text,
        Err(_) => "<payload unavailable>".to_string(),
    }
}
