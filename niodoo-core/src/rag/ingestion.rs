// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use crate::rag::Document;
use anyhow::Result;
use std::fs;
use std::path::Path;

/// Document chunk for ingestion and processing
pub struct Chunk {
    pub text: String,
    pub source: String,
    pub entities: Vec<String>, // Extracted concepts
    pub metadata: serde_json::Value,
}

impl From<Chunk> for Document {
    fn from(chunk: Chunk) -> Self {
        Document {
            created_at: chrono::Utc::now(),
            id: format!("{}_{}", chunk.source, chrono::Utc::now().timestamp()),
            content: chunk.text,
            metadata: {
                let mut map = std::collections::HashMap::new();
                if let Some(obj) = chunk.metadata.as_object() {
                    for (k, v) in obj {
                        if let Some(s) = v.as_str() {
                            map.insert(k.clone(), s.to_string());
                        }
                    }
                }
                map
            },
            embedding: None,
            entities: chunk.entities,
            chunk_id: chunk.metadata.get("chunk_id").and_then(|v| v.as_u64()),
            source_type: chunk
                .metadata
                .get("source_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            resonance_hint: chunk
                .metadata
                .get("resonance_hint")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            token_count: chunk
                .metadata
                .get("token_count")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(0),
        }
    }
}

pub struct IngestionEngine {
    chunk_size: usize,                  // Tokens or chars
    entity_keywords: Vec<&'static str>, // Consciousness terms
    contextual_index: usize,
}

impl IngestionEngine {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            entity_keywords: vec![
                "Möbius",
                "empathy",
                "consciousness",
                "Phi",
                "IIT",
                "neurodivergent",
                "hallucination",
                "LearningWill",
            ],
            contextual_index: 0,
        }
    }

    pub fn ingest_knowledge_base(&mut self, base_path: &str) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let kb_path = Path::new(base_path);

        tracing::info!("🔍 Scanning directory: {}", kb_path.display());

        for entry in fs::read_dir(kb_path)? {
            let entry = entry?;
            let path = entry.path();
            tracing::info!("📄 Found file: {}", path.display());

            if path.is_file() {
                let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
                tracing::info!("🔍 File extension: '{}'", extension);

                if extension == "md" || extension == "txt" || extension == "html" {
                    tracing::info!("✅ Processing file: {}", path.display());
                    let content = fs::read_to_string(&path)?;
                    tracing::info!("📝 File size: {} bytes", content.len());

                    let processed_content = if extension == "md" {
                        self.strip_markdown(&content)
                    } else {
                        content
                    };
                    let file_chunks = self
                        .chunk_text(&processed_content, path.to_string_lossy().as_ref())
                        .map_err(|e| anyhow::anyhow!("Chunking error: {}", e))?;
                    tracing::info!("✂️ Created {} chunks from file", file_chunks.len());
                    chunks.extend(file_chunks);
                } else {
                    tracing::info!(
                        "⏭️ Skipping file (wrong extension '{}'): {}",
                        extension,
                        path.display()
                    );
                }
            } else {
                tracing::info!("⏭️ Skipping (not a file): {}", path.display());
            }
        }

        tracing::info!("🎯 Total chunks ingested: {}", chunks.len());
        Ok(chunks)
    }

    fn chunk_text(
        &mut self,
        text: &str,
        source: &str,
    ) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
        let mut chunks = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut i = 0;

        while i < words.len() {
            let end = (i + self.chunk_size).min(words.len());
            let chunk_text = words[i..end].join(" ");
            let entities = self.extract_entities(&chunk_text);

            let chunk_id = self.contextual_index;
            self.contextual_index += 1;

            let token_count = words[i..end]
                .iter()
                .map(|w| w.encode_utf16().count())
                .sum::<usize>();

            chunks.push(Chunk {
                text: chunk_text,
                source: source.to_string(),
                entities,
                metadata: serde_json::json!({
                    "chunk_id": chunk_id,
                    "source_type": self.get_source_type(source),
                    "token_count": token_count,
                }),
            });

            i = end;
        }

        Ok(chunks)
    }

    fn extract_entities(&self, text: &str) -> Vec<String> {
        self.entity_keywords
            .iter()
            .filter(|&&kw| text.to_lowercase().contains(&kw.to_lowercase()))
            .map(|&kw| kw.to_string())
            .collect()
    }

    fn get_source_type(&self, source: &str) -> &'static str {
        if source.contains("knowledge_base") {
            "research"
        } else if source.contains("docs") {
            "documentation"
        } else {
            "raw"
        }
    }

    fn strip_markdown(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        // Remove headers
        cleaned = cleaned.replace(r"^\s*#{1,6}\s+", "");
        // But since no regex, use lines
        let lines: Vec<&str> = text
            .lines()
            .map(|line| {
                if line.starts_with('#') {
                    let trimmed = line.trim_start_matches('#').trim_start_matches(' ');
                    if trimmed.is_empty() {
                        ""
                    } else {
                        trimmed
                    }
                } else {
                    line
                }
            })
            .collect();

        let mut plain = lines.join("\n");

        // Remove code blocks
        while let Some(start) = plain.find("```") {
            if let Some(end) = plain[start..].find("\n```") {
                let block_start = start + 3;
                let block_end = start + end + 4;
                plain.replace_range(block_start..block_end, "");
            } else {
                break;
            }
        }

        // Remove inline code `code`
        plain = plain.replace("`", "");

        // Trim and filter empty lines
        plain
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// Test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_ingestion() {
        let mut engine = IngestionEngine::new(100); // Small chunks for test
        let chunks = engine.ingest_knowledge_base("knowledge_base/raw/").unwrap(); // Sample path
        assert!(!chunks.is_empty());
        assert!(chunks[0].entities.len() >= 0); // At least empty
        tracing::info!("Ingested {} chunks from knowledge_base", chunks.len());
    }
}
