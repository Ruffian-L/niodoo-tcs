//! Response parser trait system for curator quality assessment
//! Modular parsing strategies with fallback cascading

use anyhow::{anyhow, Result};
use regex::Regex;
use serde_json::Value;
use tracing::{debug, warn};

/// Parser mode configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserMode {
    Json,
    Regex,
    Heuristic,
}

impl Default for ParserMode {
    fn default() -> Self {
        ParserMode::Regex
    }
}

impl ParserMode {
    pub fn from_env() -> Self {
        std::env::var("CURATOR_PARSE_MODE")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "json" => Some(ParserMode::Json),
                "regex" => Some(ParserMode::Regex),
                "heuristic" => Some(ParserMode::Heuristic),
                _ => None,
            })
            .unwrap_or_default()
    }
}

/// Response parser trait for extracting quality scores
pub trait ResponseParser {
    fn parse_score(&self, text: &str) -> Result<f32>;
}

/// JSON-based parser - expects structured output
pub struct JsonParser;

impl ResponseParser for JsonParser {
    fn parse_score(&self, text: &str) -> Result<f32> {
        let json_val: Value =
            serde_json::from_str(text).map_err(|e| anyhow!("JSON parse failed: {}", e))?;

        if let Some(score) = json_val.get("score").and_then(|v| v.as_f64()) {
            return Ok(score as f32);
        }

        if let Some(score) = json_val.get("quality").and_then(|v| v.as_f64()) {
            return Ok(score as f32);
        }

        Err(anyhow!("No score/quality field in JSON"))
    }
}

/// Regex-based parser - extracts numeric values
pub struct RegexParser {
    pattern: Regex,
}

impl RegexParser {
    pub fn new() -> Result<Self> {
        Ok(Self {
            pattern: Regex::new(r"(\d+\.?\d*)")?,
        })
    }
}

impl ResponseParser for RegexParser {
    fn parse_score(&self, text: &str) -> Result<f32> {
        if let Some(cap) = self.pattern.captures(text) {
            if let Ok(score) = cap[1].parse::<f32>() {
                debug!("RegexParser extracted score: {}", score);
                return Ok(score.clamp(0.0, 1.0));
            }
        }
        Err(anyhow!("No numeric value found"))
    }
}

/// Heuristic fallback parser - uses length/entropy heuristics
pub struct HeuristicParser {
    response: String,
    entropy: f64,
    /// Maximum response length considered for scoring
    max_length: usize,
    /// Optimal entropy range (lower bound)
    optimal_entropy_low: f64,
    /// Optimal entropy range (upper bound)
    optimal_entropy_high: f64,
    /// Score for responses in optimal entropy range
    optimal_entropy_score: f32,
    /// Score for responses outside optimal entropy range
    suboptimal_entropy_score: f32,
    /// Weight for length component (entropy gets 1.0 - this)
    length_weight: f32,
}

impl HeuristicParser {
    pub fn new(response: String, entropy: f64) -> Self {
        Self {
            response,
            entropy,
            max_length: 500,
            optimal_entropy_low: 1.5,
            optimal_entropy_high: 2.5,
            optimal_entropy_score: 0.9,
            suboptimal_entropy_score: 0.6,
            length_weight: 0.4,
        }
    }

    pub fn with_config(
        mut self,
        max_length: usize,
        optimal_entropy_low: f64,
        optimal_entropy_high: f64,
        optimal_entropy_score: f32,
        suboptimal_entropy_score: f32,
        length_weight: f32,
    ) -> Self {
        self.max_length = max_length;
        self.optimal_entropy_low = optimal_entropy_low;
        self.optimal_entropy_high = optimal_entropy_high;
        self.optimal_entropy_score = optimal_entropy_score;
        self.suboptimal_entropy_score = suboptimal_entropy_score;
        self.length_weight = length_weight;
        self
    }
}

impl ResponseParser for HeuristicParser {
    fn parse_score(&self, _text: &str) -> Result<f32> {
        // Length-based score (normalized to 0-1)
        let length_score = self.response.len().min(self.max_length) as f32 / self.max_length as f32;

        // Entropy-based score (check if in optimal range)
        let entropy_score = if self.entropy > self.optimal_entropy_low
            && self.entropy < self.optimal_entropy_high
        {
            self.optimal_entropy_score
        } else {
            self.suboptimal_entropy_score
        };

        // Weighted combination
        let entropy_weight = 1.0 - self.length_weight;
        let quality = (length_score * self.length_weight + entropy_score * entropy_weight)
            .max(0.0)
            .min(1.0);

        debug!(
            "HeuristicParser: length={:.3}, entropy={:.3}, score={:.3}",
            length_score, entropy_score, quality
        );
        Ok(quality)
    }
}

/// Cascading parser that tries multiple strategies
pub struct CascadingParser {
    mode: ParserMode,
    heuristic_fallback: Option<HeuristicParser>,
}

impl CascadingParser {
    pub fn new(mode: ParserMode) -> Self {
        Self {
            mode,
            heuristic_fallback: None,
        }
    }

    pub fn with_heuristic_fallback(mut self, response: String, entropy: f64) -> Self {
        self.heuristic_fallback = Some(HeuristicParser::new(response, entropy));
        self
    }

    pub fn parse(&self, text: &str) -> Result<f32> {
        match self.mode {
            ParserMode::Json => {
                debug!("Trying JsonParser");
                match JsonParser.parse_score(text) {
                    Ok(score) => return Ok(score),
                    Err(e) => {
                        warn!("JsonParser failed: {}, falling back", e);
                    }
                }
            }
            ParserMode::Regex => {
                debug!("Trying RegexParser");
                match RegexParser::new()?.parse_score(text) {
                    Ok(score) => return Ok(score),
                    Err(e) => {
                        warn!("RegexParser failed: {}, falling back", e);
                    }
                }
            }
            ParserMode::Heuristic => {
                if let Some(ref heuristic) = self.heuristic_fallback {
                    return heuristic.parse_score(text);
                }
            }
        }

        // Cascading fallback: try other parsers
        debug!("Cascading to alternative parsers");

        // Try JSON even if mode is regex
        if self.mode != ParserMode::Json {
            if let Ok(score) = JsonParser.parse_score(text) {
                return Ok(score);
            }
        }

        // Try regex even if mode is json
        if self.mode != ParserMode::Regex {
            if let Ok(regex_parser) = RegexParser::new() {
                if let Ok(score) = regex_parser.parse_score(text) {
                    return Ok(score);
                }
            }
        }

        // Final fallback to heuristic
        if let Some(ref heuristic) = self.heuristic_fallback {
            warn!("All parsers failed, using heuristic fallback");
            return heuristic.parse_score(text);
        }

        Err(anyhow!("All parsing strategies failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_parser() {
        let json = r#"{"score": 0.85}"#;
        let parser = JsonParser;
        assert_eq!(parser.parse_score(json).unwrap(), 0.85);
    }

    #[test]
    fn test_regex_parser() {
        let text = "Here's a score: 0.8";
        let parser = RegexParser::new().unwrap();
        assert_eq!(parser.parse_score(text).unwrap(), 0.8);
    }

    #[test]
    fn test_heuristic_parser_config() {
        let parser = HeuristicParser::new("test".to_string(), 1.8)
            .with_config(1000, 1.0, 3.0, 0.95, 0.5, 0.3);
        let score = parser.parse_score("").unwrap();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_cascading_parser() {
        let text = "The quality score is 0.75";
        let parser = CascadingParser::new(ParserMode::Regex)
            .with_heuristic_fallback("test".to_string(), 1.8);
        assert_eq!(parser.parse(text).unwrap(), 0.75);
    }
}
