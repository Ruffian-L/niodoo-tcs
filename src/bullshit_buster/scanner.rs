//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// Hardcoded Value Scanner - Detects magic numbers and hardcoded values in code
///
/// Converted from scripts/hardcoded_value_buster.py to idiomatic Rust
use super::{BullshitBusterError, BullshitBusterResult};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanConfig {
    /// File extensions to scan
    pub code_extensions: HashSet<String>,
    /// Directories to skip
    pub skip_dirs: HashSet<String>,
    /// Whether to generate mathematical replacement suggestions
    pub generate_suggestions: bool,
    /// Output file path for report
    pub output_file: PathBuf,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            code_extensions: vec![
                ".rs".to_string(),
                ".py".to_string(),
                ".js".to_string(),
                ".ts".to_string(),
                ".cpp".to_string(),
                ".c".to_string(),
                ".h".to_string(),
                ".hpp".to_string(),
                ".java".to_string(),
                ".go".to_string(),
            ]
            .into_iter()
            .collect(),
            skip_dirs: vec![
                ".git".to_string(),
                "target".to_string(),
                "node_modules".to_string(),
                "__pycache__".to_string(),
                ".vscode".to_string(),
                "build".to_string(),
            ]
            .into_iter()
            .collect(),
            generate_suggestions: false,
            output_file: PathBuf::from("hardcoded_analysis_report.json"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardcodedValue {
    pub file: PathBuf,
    pub line: usize,
    pub content: String,
    pub value: String,
    pub pattern: String,
    pub severity: Severity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScanResult {
    pub summary: ScanSummary,
    pub hardcoded_values: Vec<HardcodedValue>,
    pub top_offenders: Vec<FileOffender>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScanSummary {
    pub total_files_scanned: usize,
    pub files_with_hardcoded_values: usize,
    pub total_hardcoded_instances: usize,
    pub scan_timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileOffender {
    pub file: PathBuf,
    pub instance_count: usize,
}

pub struct HardcodedValueScanner {
    config: ScanConfig,
    hardcoded_patterns: Vec<Regex>,
    exclusion_patterns: Vec<Regex>,
}

impl HardcodedValueScanner {
    pub fn new(config: ScanConfig) -> BullshitBusterResult<Self> {
        let hardcoded_patterns = Self::compile_hardcoded_patterns()?;
        let exclusion_patterns = Self::compile_exclusion_patterns()?;

        Ok(Self {
            config,
            hardcoded_patterns,
            exclusion_patterns,
        })
    }

    fn compile_hardcoded_patterns() -> BullshitBusterResult<Vec<Regex>> {
        let patterns = vec![
            // Magic numbers in assignments
            r"\b(let|const|var)\s+\w+\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)\s*[;)]",
            // Function parameters with magic numbers
            r"\b\w+\([^)]*([0-9]+\.[0-9]+|[0-9]+)[^)]*\)",
            // Array/list literals with magic numbers
            r"\[([0-9]+\.[0-9]+|[0-9]+)(?:\s*,\s*([0-9]+\.[0-9]+|[0-9]+))*\]",
            // Dictionary/object literals
            r"\{[^}]*([0-9]+\.[0-9]+|[0-9]+)[^}]*\}",
            // Mathematical operations with magic numbers
            r"[\+\-\*\/]\s*([0-9]+\.[0-9]+|[0-9]+)\s*[\+\-\*\/]",
        ];

        patterns
            .iter()
            .map(|p| Regex::new(p).map_err(BullshitBusterError::from))
            .collect()
    }

    fn compile_exclusion_patterns() -> BullshitBusterResult<Vec<Regex>> {
        let patterns = vec![
            r"//.*[0-9]",        // Comments
            r"/\*.*[0-9].*\*/",  // Multi-line comments
            r#"".*[0-9].*""#,    // String literals
            r"'.*[0-9].*'",      // Character literals
            r"#.*[0-9]",         // Python comments
            r"//.*TODO.*[0-9]",  // TODO comments
            r"//.*FIXME.*[0-9]", // FIXME comments
        ];

        patterns
            .iter()
            .map(|p| Regex::new(p).map_err(BullshitBusterError::from))
            .collect()
    }

    fn is_code_file(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| self.config.code_extensions.contains(&format!(".{}", ext)))
            .unwrap_or(false)
    }

    fn should_exclude_line(&self, line: &str) -> bool {
        self.exclusion_patterns
            .iter()
            .any(|pattern| pattern.is_match(line))
    }

    fn is_acceptable_value(value: f64) -> bool {
        // Skip common acceptable values
        (0.0..=1.0).contains(&value) || [2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 1000.0].contains(&value)
    }

    fn calculate_severity(value: &str) -> Severity {
        if let Ok(num_value) = value.parse::<f64>() {
            if num_value.abs() > 1000.0 {
                Severity::High
            } else if num_value.abs() > 100.0 {
                Severity::Medium
            } else {
                Severity::Low
            }
        } else {
            Severity::Medium
        }
    }

    fn find_hardcoded_values_in_file(
        &self,
        file_path: &Path,
    ) -> BullshitBusterResult<Vec<HardcodedValue>> {
        let mut results = Vec::new();

        let content = fs::read_to_string(file_path)?;

        for (line_num, line) in content.lines().enumerate() {
            let line_num = line_num + 1; // 1-indexed

            // Skip excluded lines
            if self.should_exclude_line(line) {
                continue;
            }

            // Check for hardcoded patterns
            for pattern in &self.hardcoded_patterns {
                for captures in pattern.captures_iter(line) {
                    // Extract the numeric value from captures
                    let value = if let Some(cap) = captures.get(1) {
                        cap.as_str()
                    } else if let Some(cap) = captures.get(0) {
                        cap.as_str()
                    } else {
                        continue;
                    };

                    // Skip if it's an acceptable value
                    if let Ok(num_value) = value.parse::<f64>() {
                        if Self::is_acceptable_value(num_value) {
                            continue;
                        }
                    } else {
                        continue;
                    }

                    results.push(HardcodedValue {
                        file: file_path.to_path_buf(),
                        line: line_num,
                        content: line.trim().to_string(),
                        value: value.to_string(),
                        pattern: pattern.as_str().to_string(),
                        severity: Self::calculate_severity(value),
                        suggestion: None,
                    });
                }
            }
        }

        Ok(results)
    }

    pub fn scan_directory(&self, directory: &Path) -> BullshitBusterResult<ScanResult> {
        info!("Scanning directory: {:?}", directory);

        let mut total_files = 0;
        let mut files_with_hardcoded = HashSet::new();
        let mut all_hardcoded_values = Vec::new();

        for entry in WalkDir::new(directory).into_iter().filter_entry(|e| {
            // Skip directories in the skip list
            e.file_type().is_file()
                || !self
                    .config
                    .skip_dirs
                    .contains(e.file_name().to_str().unwrap_or(""))
        }) {
            let entry = entry?;
            let path = entry.path();

            if !entry.file_type().is_file() {
                continue;
            }

            if !self.is_code_file(path) {
                continue;
            }

            total_files += 1;

            match self.find_hardcoded_values_in_file(path) {
                Ok(values) => {
                    if !values.is_empty() {
                        files_with_hardcoded.insert(path.to_path_buf());
                        all_hardcoded_values.extend(values);
                    }
                }
                Err(e) => {
                    warn!("Error scanning file {:?}: {}", path, e);
                }
            }
        }

        // Calculate top offenders
        let mut file_counts: HashMap<PathBuf, usize> = HashMap::new();
        for value in &all_hardcoded_values {
            *file_counts.entry(value.file.clone()).or_insert(0) += 1;
        }

        let mut top_offenders: Vec<FileOffender> = file_counts
            .into_iter()
            .map(|(file, instance_count)| FileOffender {
                file,
                instance_count,
            })
            .collect();
        top_offenders.sort_by(|a, b| b.instance_count.cmp(&a.instance_count));

        let summary = ScanSummary {
            total_files_scanned: total_files,
            files_with_hardcoded_values: files_with_hardcoded.len(),
            total_hardcoded_instances: all_hardcoded_values.len(),
            scan_timestamp: chrono::Utc::now().to_rfc3339(),
        };

        Ok(ScanResult {
            summary,
            hardcoded_values: all_hardcoded_values,
            top_offenders,
        })
    }

    pub fn generate_suggestions(&self, result: &mut ScanResult) {
        let replacer = HardcodedValueReplacer::new();

        for value in &mut result.hardcoded_values {
            value.suggestion = replacer.suggest_replacement(&value.value, &value.content);
        }
    }

    pub fn save_report(&self, result: &ScanResult) -> BullshitBusterResult<()> {
        let json = serde_json::to_string_pretty(result)?;
        fs::write(&self.config.output_file, json)?;
        info!("Report saved to: {:?}", self.config.output_file);
        Ok(())
    }
}

struct HardcodedValueReplacer;

impl HardcodedValueReplacer {
    fn new() -> Self {
        Self
    }

    fn suggest_replacement(&self, value: &str, context: &str) -> Option<String> {
        let num_value = value.parse::<f64>().ok()?;

        // Check if it's a mathematical constant
        if (num_value - std::f64::consts::PI).abs() < 0.001 {
            return Some("Replace with: std::f64::consts::PI".to_string());
        }
        if (num_value - std::f64::consts::E).abs() < 0.001 {
            return Some("Replace with: std::f64::consts::E".to_string());
        }
        if (num_value - std::f64::consts::TAU).abs() < 0.001 {
            return Some("Replace with: std::f64::consts::TAU".to_string());
        }
        if (num_value - 1.618033988749).abs() < 0.001 {
            return Some("Replace with: GOLDEN_RATIO constant".to_string());
        }

        // Suggest derivation-based replacements
        if (0.0..=1.0).contains(&num_value) {
            Some(self.suggest_probability_derivation(value, context))
        } else if num_value.fract() != 0.0 {
            Some(self.suggest_scaling_derivation(value, context))
        } else {
            Some(self.suggest_count_derivation(value, context))
        }
    }

    fn suggest_probability_derivation(&self, _value: &str, _context: &str) -> String {
        "// Derived from: sigmoid activation\n\
         1.0 / (1.0 + (-value).exp())\n\
         // OR: softmax normalization\n\
         value / value.sum()\n\
         // OR: gaussian probability\n\
         (-0.5 * (value / sigma).powi(2)).exp()"
            .to_string()
    }

    fn suggest_scaling_derivation(&self, _value: &str, _context: &str) -> String {
        "// Derived from: min-max normalization\n\
         (value - min_val) / (max_val - min_val)\n\
         // OR: z-score standardization\n\
         (value - mean) / std_dev\n\
         // OR: logarithmic scaling\n\
         value.log10() / max_log"
            .to_string()
    }

    fn suggest_count_derivation(&self, _value: &str, _context: &str) -> String {
        "// Derived from: array length\n\
         data.len()\n\
         // OR: collection size\n\
         collection.len() as f32\n\
         // OR: iteration count\n\
         (0..count).len() as f32"
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acceptable_values() {
        assert!(HardcodedValueScanner::is_acceptable_value(0.5));
        assert!(HardcodedValueScanner::is_acceptable_value(1.0));
        assert!(HardcodedValueScanner::is_acceptable_value(100.0));
        assert!(!HardcodedValueScanner::is_acceptable_value(123.45));
        assert!(!HardcodedValueScanner::is_acceptable_value(5000.0));
    }

    #[test]
    fn test_severity_calculation() {
        assert!(matches!(
            HardcodedValueScanner::calculate_severity("50"),
            Severity::Low
        ));
        assert!(matches!(
            HardcodedValueScanner::calculate_severity("500"),
            Severity::Medium
        ));
        assert!(matches!(
            HardcodedValueScanner::calculate_severity("5000"),
            Severity::High
        ));
    }
}
