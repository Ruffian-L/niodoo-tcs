//! Constitutional AI framework for code governance
//!
//! This module provides a framework for ensuring agent-generated code
//! adheres to constitutional principles through static analysis and LLM-based critique.

pub mod constitution;
pub mod static_analysis;
pub mod critique;
pub mod revision;
pub mod violations;

pub use constitution::{Constitution, Principle, ViolationPattern};
pub use static_analysis::StaticAnalyzer;
pub use critique::CritiqueEngine;
pub use revision::RevisionLoop;
pub use violations::{Violation, ViolationSeverity};



