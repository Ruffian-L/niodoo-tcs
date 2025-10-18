//! # Time Utilities
//!
//! Safe time operations that never panic, providing fallback values
//! when system time operations fail.
//!
//! ## Purpose
//!
//! This module eliminates dangerous .unwrap() calls on SystemTime operations
//! throughout the consciousness system by providing safe alternatives with
//! sensible defaults.
//!
//! ## Usage
//!
//! ```rust
//! use niodoo_feeling::time_utils::{timestamp_secs_f64, timestamp_secs, timestamp_nanos};
//!
//! // Get current timestamp as f64 seconds (never panics)
//! let ts = timestamp_secs_f64();
//!
//! // Get current timestamp as u64 seconds (never panics)
//! let ts_u64 = timestamp_secs();
//!
//! // Get current timestamp as u128 nanoseconds (never panics)
//! let ts_nano = timestamp_nanos();
//! ```

use std::time::{SystemTime, UNIX_EPOCH};
use tracing::warn;

/// Get current timestamp as seconds since UNIX_EPOCH as f64.
/// Returns 0.0 if system time is unavailable (should never happen on modern systems).
pub fn timestamp_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_else(|e| {
            warn!("SystemTime error: {}. Using fallback timestamp 0.0", e);
            0.0
        })
}

/// Get current timestamp as seconds since UNIX_EPOCH as u64.
/// Returns 0 if system time is unavailable (should never happen on modern systems).
pub fn timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_else(|e| {
            warn!("SystemTime error: {}. Using fallback timestamp 0", e);
            0
        })
}

/// Get current timestamp as nanoseconds since UNIX_EPOCH as u128.
/// Returns 0 if system time is unavailable (should never happen on modern systems).
pub fn timestamp_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_else(|e| {
            warn!("SystemTime error: {}. Using fallback timestamp 0", e);
            0
        })
}

/// Generate a unique ID string based on current timestamp.
/// Format: "{prefix}_{timestamp_secs}"
/// Returns "{prefix}_0" if system time is unavailable.
pub fn timestamp_id(prefix: &str) -> String {
    format!("{}_{}", prefix, timestamp_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_secs_f64() {
        let ts = timestamp_secs_f64();
        // Should be a reasonable timestamp (after year 2000)
        assert!(ts > 946_684_800.0); // Jan 1, 2000
    }

    #[test]
    fn test_timestamp_secs() {
        let ts = timestamp_secs();
        // Should be a reasonable timestamp (after year 2000)
        assert!(ts > 946_684_800); // Jan 1, 2000
    }

    #[test]
    fn test_timestamp_nanos() {
        let ts = timestamp_nanos();
        // Should be a reasonable timestamp (after year 2000)
        assert!(ts > 946_684_800_000_000_000); // Jan 1, 2000 in nanoseconds
    }

    #[test]
    fn test_timestamp_id() {
        let id = timestamp_id("test");
        assert!(id.starts_with("test_"));
        assert!(id.len() > 5);
    }
}
