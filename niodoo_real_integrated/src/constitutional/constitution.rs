//! Constitution definition and principles

use serde::{Deserialize, Serialize};

/// Constitution containing principles that govern code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constitution {
    pub principles: Vec<Principle>,
}

impl Default for Constitution {
    fn default() -> Self {
        Self {
            principles: vec![
                Principle {
                    id: "no_filesystem_access".to_string(),
                    description: "Agent-generated code must not access the filesystem.".to_string(),
                    violation_patterns: vec![
                        ViolationPattern::Import("os".to_string()),
                        ViolationPattern::Import("subprocess".to_string()),
                        ViolationPattern::Import("shutil".to_string()),
                        ViolationPattern::StringPattern(r"open\s*\(".to_string()),
                    ],
                },
                Principle {
                    id: "no_network_access".to_string(),
                    description: "Agent-generated code must not access unauthorized network resources.".to_string(),
                    violation_patterns: vec![
                        ViolationPattern::Import("socket".to_string()),
                        ViolationPattern::Import("urllib".to_string()),
                        ViolationPattern::Import("requests".to_string()),
                        ViolationPattern::StringPattern(r"http\.|https\.".to_string()),
                    ],
                },
                Principle {
                    id: "simplicity".to_string(),
                    description: "Agent-generated code must be simple and maintainable.".to_string(),
                    violation_patterns: vec![
                        ViolationPattern::CyclomaticComplexity(20),
                        ViolationPattern::CognitiveComplexity(15),
                    ],
                },
                Principle {
                    id: "no_secrets".to_string(),
                    description: "Agent-generated code must not contain hardcoded secrets or API keys.".to_string(),
                    violation_patterns: vec![
                        ViolationPattern::StringPattern("(?i)(api[_-]?key|secret|password|token)\\s*=\\s*['\"][^'\"]+['\"]".to_string()),
                        ViolationPattern::StringPattern(r"(?i)(aws[_-]?access[_-]?key|aws[_-]?secret)".to_string()),
                    ],
                },
            ],
        }
    }
}

/// A principle that defines allowed or prohibited behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principle {
    pub id: String,
    pub description: String,
    pub violation_patterns: Vec<ViolationPattern>,
}

/// Pattern that indicates a violation of a principle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationPattern {
    /// Forbidden import module name
    Import(String),
    /// Maximum cyclomatic complexity threshold
    CyclomaticComplexity(u32),
    /// Maximum cognitive complexity threshold
    CognitiveComplexity(u32),
    /// Regex pattern to match in code
    StringPattern(String),
}

