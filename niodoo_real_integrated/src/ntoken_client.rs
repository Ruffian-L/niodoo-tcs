use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct NTokenFeatures {
    pub h1_count: f64,
    pub h1_total_persistence: f64,
    pub entropy_norm: f64,
    pub sheaf_energy: f64,
    pub sheaf_mean_section_norm: f64,
    pub raw_features: HashMap<String, f64>,
    pub topological_properties: HashMap<String, HashMap<String, f64>>,
}

#[derive(Debug, Serialize)]
struct SentenceRequest {
    sentences: Vec<String>,
    #[serde(default)]
    include_diagrams: bool,
    #[serde(default)]
    include_embedding: bool,
}

#[derive(Debug, Deserialize)]
struct SentenceResponse {
    results: Vec<SentenceResponseItem>,
}

#[derive(Debug, Deserialize)]
struct SentenceResponseItem {
    sentence: String,
    #[serde(default)]
    features: HashMap<String, f64>,
    #[serde(default)]
    sheaf: HashMap<String, f64>,
    #[serde(default)]
    topological_properties: HashMap<String, HashMap<String, f64>>,
}

pub async fn fetch_features(
    endpoint: &str,
    prompt: &str,
    context: Option<&str>,
) -> Result<Option<NTokenFeatures>> {
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .context("failed to build reqwest client for ntoken service")?;

    let url = if endpoint.ends_with("/ntoken") {
        endpoint.to_string()
    } else {
        format!("{}/ntoken", endpoint.trim_end_matches('/'))
    };

    let combined = if let Some(ctx) = context {
        if ctx.trim().is_empty() {
            prompt.to_string()
        } else {
            format!("{prompt}\n\n[Context]\n{ctx}")
        }
    } else {
        prompt.to_string()
    };

    let request = SentenceRequest {
        sentences: vec![combined],
        include_diagrams: false,
        include_embedding: false,
    };

    let resp = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .with_context(|| format!("failed to call ntoken service at {url}"))?;

    if !resp.status().is_success() {
        anyhow::bail!("ntoken service responded with status {}", resp.status());
    }

    let payload: SentenceResponse = resp
        .json()
        .await
        .context("failed to parse ntoken service response")?;

    let Some(item) = payload.results.into_iter().next() else {
        return Ok(None);
    };

    let raw = item.features.clone();
    if raw.is_empty() {
        return Ok(None);
    }

    let sheaf_energy = item
        .sheaf
        .get("energy")
        .copied()
        .or_else(|| raw.get("sheaf_energy").copied())
        .unwrap_or_default();
    let sheaf_mean_section_norm = item
        .sheaf
        .get("mean_section_norm")
        .copied()
        .or_else(|| raw.get("sheaf_mean_section_norm").copied())
        .unwrap_or_default();

    let h1_count = raw.get("h1_count").copied().unwrap_or_else(|| {
        item.topological_properties
            .get("H1")
            .and_then(|map| map.get("count").copied())
            .unwrap_or_default()
    });

    let h1_total_persistence = raw
        .get("h1_total_persistence")
        .copied()
        .or_else(|| {
            item.topological_properties
                .get("H1")
                .and_then(|map| map.get("total_persistence").copied())
        })
        .unwrap_or_default();

    let entropy_norm = raw.get("entropy_norm").copied().unwrap_or_default();

    Ok(Some(NTokenFeatures {
        h1_count,
        h1_total_persistence,
        entropy_norm,
        sheaf_energy,
        sheaf_mean_section_norm,
        raw_features: raw,
        topological_properties: item.topological_properties,
    }))
}

