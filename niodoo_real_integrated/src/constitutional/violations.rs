//! Violation types and severity

use crate::constitutional::constitution::Principle;
use serde::{Deserialize, Serialize};

/// Severity level of a violation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Low severity - warning only
    Low,
    /// Medium severity - requires revision
    Medium,
    /// High severity - blocks execution
    High,
}

/// A violation of a constitutional principle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub principle_id: String,
    pub principle_description: String,
    pub severity: ViolationSeverity,
    pub line_number: Option<usize>,
    pub message: String,
    pub pattern_matched: Option<String>,
}

impl Violation {
    pub fn new(
        principle: &Principle,
        severity: ViolationSeverity,
        message: String,
        line_number: Option<usize>,
        pattern_matched: Option<String>,
    ) -> Self {
        Self {
            principle_id: principle.id.clone(),
            principle_description: principle.description.clone(),
            severity,
            line_number,
            message,
            pattern_matched,
        }
    }
}



