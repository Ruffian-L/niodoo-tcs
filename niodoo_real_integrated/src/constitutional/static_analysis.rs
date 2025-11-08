//! Static analysis for code violations

use crate::config::CodeLanguage;
use crate::constitutional::constitution::{Constitution, ViolationPattern};
use crate::constitutional::violations::{Violation, ViolationSeverity};
use anyhow::{Context, Result};
use regex::Regex;
use std::collections::HashMap;
use tracing::{info, warn};

/// Static analyzer for detecting constitutional violations
pub struct StaticAnalyzer {
    constitution: Constitution,
    // Cache compiled regex patterns
    pattern_cache: HashMap<String, Regex>,
}

impl StaticAnalyzer {
    /// Create a new static analyzer with the given constitution
    pub fn new(constitution: Constitution) -> Self {
        let mut analyzer = Self {
            constitution,
            pattern_cache: HashMap::new(),
        };
        analyzer.compile_patterns();
        analyzer
    }

    /// Analyze code for violations
    pub fn analyze(&self, code: &str, language: CodeLanguage) -> Result<Vec<Violation>> {
        let mut violations = Vec::new();

        // Parse code to AST (using tcs-parser if available)
        // For now, do simple text-based analysis
        let lines: Vec<&str> = code.lines().collect();

        for principle in &self.constitution.principles {
            for pattern in &principle.violation_patterns {
                match pattern {
                    ViolationPattern::Import(module_name) => {
                        // Check for import statements
                        for (line_num, line) in lines.iter().enumerate() {
                            if self.check_import(line, module_name) {
                                violations.push(Violation::new(
                                    principle,
                                    ViolationSeverity::High,
                                    format!("Forbidden import '{}' detected", module_name),
                                    Some(line_num + 1),
                                    Some(format!("import {}", module_name)),
                                ));
                            }
                        }
                    }
                    ViolationPattern::StringPattern(pattern_str) => {
                        // Check regex pattern
                        if let Some(regex) = self.pattern_cache.get(pattern_str) {
                            for (line_num, line) in lines.iter().enumerate() {
                                if regex.is_match(line) {
                                    violations.push(Violation::new(
                                        principle,
                                        ViolationSeverity::High,
                                        format!("Pattern violation: {}", pattern_str),
                                        Some(line_num + 1),
                                        Some(pattern_str.clone()),
                                    ));
                                }
                            }
                        }
                    }
                    ViolationPattern::CyclomaticComplexity(max_complexity) => {
                        // NOTE: Cyclomatic complexity check not yet implemented.
                        // Future: Use rust-code-analysis or tcs-parser to compute cyclomatic complexity
                        warn!("Cyclomatic complexity check not yet implemented (max: {})", max_complexity);
                    }
                    ViolationPattern::CognitiveComplexity(max_complexity) => {
                        // NOTE: Cognitive complexity check not yet implemented.
                        // Future: Use rust-code-analysis to compute cognitive complexity
                        warn!("Cognitive complexity check not yet implemented (max: {})", max_complexity);
                    }
                }
            }
        }

        info!(violations_count = violations.len(), "Static analysis completed");
        Ok(violations)
    }

    /// Check if a line contains a forbidden import
    fn check_import(&self, line: &str, module_name: &str) -> bool {
        let trimmed = line.trim();
        // Check for various import patterns
        trimmed.starts_with(&format!("import {}", module_name))
            || trimmed.starts_with(&format!("from {} import", module_name))
            || trimmed.contains(&format!("require('{}')", module_name))
            || trimmed.contains(&format!("require(\"{}\")", module_name))
    }

    /// Compile regex patterns and cache them
    fn compile_patterns(&mut self) {
        for principle in &self.constitution.principles {
            for pattern in &principle.violation_patterns {
                if let ViolationPattern::StringPattern(pattern_str) = pattern {
                    if let Ok(regex) = Regex::new(pattern_str) {
                        self.pattern_cache.insert(pattern_str.clone(), regex);
                    } else {
                        warn!(pattern = pattern_str, "Failed to compile regex pattern");
                    }
                }
            }
        }
    }
}

