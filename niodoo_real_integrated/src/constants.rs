//! Constants used throughout the NIODOO system
//! 
//! This module centralizes magic numbers and hardcoded values to improve maintainability
//! and make configuration easier.

/// Default timeout for HTTP client requests (in seconds)
pub const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 60;

/// Default timeout for generation requests (in seconds)
pub const DEFAULT_GENERATION_TIMEOUT_SECS: u64 = 60;

/// Default curator quality score when curator is not available
pub const DEFAULT_CURATOR_QUALITY_SCORE: f64 = 0.5;

/// Default knot complexity threshold for predictor triggering
pub const DEFAULT_KNOT_COMPLEXITY_THRESHOLD: f64 = 0.4;

/// Default emotional coherence calculation bounds
pub const EMOTIONAL_COHERENCE_MIN: f64 = 0.0;
pub const EMOTIONAL_COHERENCE_MAX: f64 = 1.0;

/// Default consonance score bounds
pub const CONSONANCE_SCORE_MIN: f64 = 0.0;
pub const CONSONANCE_SCORE_MAX: f64 = 1.0;

/// Default RCE score bounds
pub const RCE_SCORE_MIN: f64 = 0.0;
pub const RCE_SCORE_MAX: f64 = 1.0;

/// Default confidence value for consonance calculations
pub const DEFAULT_CONSONANCE_CONFIDENCE: f64 = 0.9;

/// Default processing time when unavailable (ms)
pub const DEFAULT_PROCESSING_TIME_MS: f64 = 0.0;

/// Default entropy delta when unavailable
pub const DEFAULT_ENTROPY_DELTA: f64 = 0.0;



