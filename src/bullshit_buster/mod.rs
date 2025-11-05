//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

pub mod detector;
/// Bullshit Buster - Automated detection of hardcoded values, stubs, and fake implementations
///
/// This module provides the core functionality for Gen 2 of the Niodoo-Feeling project:
/// detecting and analyzing code quality issues using topological analysis.
pub mod legacy;
pub mod scanner;

pub use detector::{BullshitDetector, DetectionResult, FakePattern};
pub use legacy::*;
pub use scanner::{HardcodedValueScanner, ScanConfig, ScanResult};

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BullshitBusterError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Walkdir error: {0}")]
    Walkdir(#[from] walkdir::Error),

    #[error("Invalid file path: {0}")]
    InvalidPath(PathBuf),

    #[error("Scan failed: {0}")]
    ScanFailed(String),
}

/// Result type alias for bullshit buster errors
/// Note: Use BullshitBusterResult to avoid conflicts with other Result aliases
pub type BullshitBusterResult<T> = std::result::Result<T, BullshitBusterError>;
