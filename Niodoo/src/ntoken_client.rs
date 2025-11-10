//! NToken Client for FastAPI Token Manager Service
//! 
//! Calls the Dynamic Tokenizer FastAPI service to get CRDT-synced promoted tokens.

use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Request to encode text with extended vocabulary
#[derive(Debug, Serialize)]
struct EncodeRequest {
    text: String,
    #[serde(default)]
    with_pieces: bool,
}

/// Response from encode endpoint
#[derive(Debug, Deserialize)]
struct EncodeResponse {
    tokens: Vec<u32>,
    #[serde(default)]
    pieces: Option<Vec<String>>,
    vocab_size: Option<usize>,
    oov_rate: Option<f64>,
}

/// Encode text using the Dynamic Tokenizer FastAPI service
/// 
/// This calls the CRDT-synced tokenizer service that has access to
/// promoted tokens learned across the system.
pub async fn encode_extended(
    endpoint: &str,
    text: &str,
) -> Result<Vec<u32>> {
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .context("failed to build reqwest client for tokenizer service")?;

    let url = if endpoint.ends_with("/encode") || endpoint.ends_with("/encode_extended") {
        endpoint.to_string()
    } else {
        format!("{}/encode_extended", endpoint.trim_end_matches('/'))
    };

    let request = EncodeRequest {
        text: text.to_string(),
        with_pieces: false,
    };

    let resp = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .with_context(|| format!("failed to call tokenizer service at {url}"))?;

    if !resp.status().is_success() {
        anyhow::bail!("tokenizer service responded with status {}", resp.status());
    }

    let payload: EncodeResponse = resp
        .json()
        .await
        .context("failed to parse tokenizer service response")?;

    Ok(payload.tokens)
}

/// Decode token IDs back to strings using the FastAPI service
pub async fn decode_extended(
    endpoint: &str,
    token_ids: &[u32],
) -> Result<Vec<String>> {
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .context("failed to build reqwest client for tokenizer service")?;

    let url = if endpoint.ends_with("/decode") || endpoint.ends_with("/decode_extended") {
        endpoint.to_string()
    } else {
        format!("{}/decode_extended", endpoint.trim_end_matches('/'))
    };

    #[derive(Serialize)]
    struct DecodeRequest {
        tokens: Vec<u32>,
    }

    let request = DecodeRequest {
        tokens: token_ids.to_vec(),
    };

    let resp = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .with_context(|| format!("failed to call tokenizer service at {url}"))?;

    if !resp.status().is_success() {
        anyhow::bail!("tokenizer service responded with status {}", resp.status());
    }

    #[derive(Deserialize)]
    struct DecodeResponse {
        pieces: Vec<String>,
    }

    let payload: DecodeResponse = resp
        .json()
        .await
        .context("failed to parse tokenizer service response")?;

    Ok(payload.pieces)
}

