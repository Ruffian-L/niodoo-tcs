/// Bullshit Detector - Detects fake/placeholder/stub code in the codebase
///
/// Converted from scripts/fresh_bullshit_scan.py to idiomatic Rust
use super::{BullshitBusterError, BullshitBusterResult};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FakePattern {
    pub category: String,
    pub patterns: Vec<String>,
}

impl FakePattern {
    fn default_patterns() -> Vec<Self> {
        vec![
            Self {
                category: "stub_functions".to_string(),
                patterns: vec![
                    r"fn.*\{[^}]*//.*[Ss]tub".to_string(),
                    r"fn.*\{[^}]*//.*[Pp]laceholder".to_string(),
                    r"fn.*\{[^}]*//.*[Nn]ot.*implemented".to_string(),
                    r"fn.*\{[^}]*//.*TODO".to_string(),
                    r"fn.*\{[^}]*//.*FIXME".to_string(),
                    r"pub fn.*->.*\{[^}]*\}\s*$".to_string(),
                    r"fn.*panic!.*not implemented".to_string(),
                    r"fn.*unimplemented!.*".to_string(),
                    r"fn.*todo!.*".to_string(),
                ],
            },
            Self {
                category: "hardcoded_values".to_string(),
                patterns: vec![
                    r"0\.5|0\.6|0\.7|0\.8|0\.9".to_string(),
                    r"0\.3|0\.4".to_string(),
                    r"let.*=.*0\.[0-9]+.*//.*magic".to_string(),
                    r"let.*=.*0\.[0-9]+.*//.*hardcoded".to_string(),
                    r"const.*=.*0\.[0-9]+".to_string(),
                ],
            },
            Self {
                category: "fake_models".to_string(),
                patterns: vec![
                    r"//.*[Ff]ake.*model".to_string(),
                    r"//.*[Pp]laceholder.*model".to_string(),
                    r"//.*[Dd]ummy.*model".to_string(),
                    r"//.*[Mm]ock.*model".to_string(),
                    r"load_tokenizer.*Err".to_string(),
                    r"load_model.*Err".to_string(),
                    r"model.*loading.*not.*implemented".to_string(),
                ],
            },
            Self {
                category: "qt_stubs".to_string(),
                patterns: vec![
                    r"TODO.*Qt6".to_string(),
                    r"TODO.*cxx-qt".to_string(),
                    r"TODO.*Qt.*integration".to_string(),
                    r"//.*Qt.*bindings".to_string(),
                    r"//.*Qt.*interface".to_string(),
                ],
            },
            Self {
                category: "lora_fakes".to_string(),
                patterns: vec![
                    r"candle-lora.*not.*exist".to_string(),
                    r"LoRA.*not.*implemented".to_string(),
                    r"//.*LoRA.*stub".to_string(),
                    r"//.*LoRA.*placeholder".to_string(),
                ],
            },
            Self {
                category: "error_returns".to_string(),
                patterns: vec![
                    r"Err.*not.*implemented".to_string(),
                    r"Err.*placeholder".to_string(),
                    r"Err.*stub".to_string(),
                    r"Err.*fake".to_string(),
                    r"Err.*TODO".to_string(),
                ],
            },
        ]
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DetectionMatch {
    pub line: usize,
    pub content: String,
    pub pattern: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileDetection {
    pub file: PathBuf,
    pub fake_count: usize,
    pub categories: Vec<String>,
    pub matches: HashMap<String, Vec<DetectionMatch>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DetectionResult {
    pub summary: DetectionSummary,
    pub worst_offenders: Vec<FileDetection>,
    pub category_counts: HashMap<String, usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DetectionSummary {
    pub total_files: usize,
    pub files_with_fake: usize,
    pub fake_instances: usize,
    pub fake_percentage: f64,
    pub scan_timestamp: String,
}

pub struct BullshitDetector {
    patterns: Vec<(String, Vec<Regex>)>,
    file_extensions: HashSet<String>,
    skip_dirs: HashSet<String>,
}

impl BullshitDetector {
    pub fn new() -> BullshitBusterResult<Self> {
        let fake_patterns = FakePattern::default_patterns();
        let mut patterns = Vec::new();

        for pattern_group in fake_patterns {
            let compiled: Result<Vec<Regex>, BullshitBusterError> = pattern_group
                .patterns
                .iter()
                .map(|p| Regex::new(p).map_err(BullshitBusterError::from))
                .collect();

            patterns.push((pattern_group.category, compiled?));
        }

        Ok(Self {
            patterns,
            file_extensions: vec![
                ".rs".to_string(),
                ".py".to_string(),
                ".cpp".to_string(),
                ".h".to_string(),
                ".hpp".to_string(),
                ".qml".to_string(),
            ]
            .into_iter()
            .collect(),
            skip_dirs: vec![
                "target".to_string(),
                "node_modules".to_string(),
                ".git".to_string(),
                "build".to_string(),
                "__pycache__".to_string(),
                ".venv".to_string(),
                "venv".to_string(),
                "transformers_env".to_string(),
                "llama.cpp".to_string(),
                "vendor".to_string(),
                "examples".to_string(),
            ]
            .into_iter()
            .collect(),
        })
    }

    fn is_code_file(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| self.file_extensions.contains(&format!(".{}", ext)))
            .unwrap_or(false)
    }

    fn should_skip_path(&self, path: &Path) -> bool {
        path.components().any(|comp| {
            if let Some(name) = comp.as_os_str().to_str() {
                self.skip_dirs.contains(name)
            } else {
                false
            }
        })
    }

    fn scan_file(&self, file_path: &Path) -> BullshitBusterResult<Option<FileDetection>> {
        let content = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => return Ok(None), // Skip files that can't be read
        };

        let mut matches: HashMap<String, Vec<DetectionMatch>> = HashMap::new();
        let lines: Vec<&str> = content.lines().collect();

        for (category, regexes) in &self.patterns {
            for (i, line) in lines.iter().enumerate() {
                for regex in regexes {
                    if regex.is_match(line) {
                        matches
                            .entry(category.clone())
                            .or_default()
                            .push(DetectionMatch {
                                line: i + 1,
                                content: line.trim().to_string(),
                                pattern: regex.as_str().to_string(),
                            });
                    }
                }
            }
        }

        if matches.is_empty() {
            Ok(None)
        } else {
            let fake_count: usize = matches.values().map(|v| v.len()).sum();
            let categories: Vec<String> = matches.keys().cloned().collect();

            Ok(Some(FileDetection {
                file: file_path.to_path_buf(),
                fake_count,
                categories,
                matches,
            }))
        }
    }

    pub fn scan_directory(&self, directory: &Path) -> BullshitBusterResult<DetectionResult> {
        info!("Scanning directory for fake/stub code: {:?}", directory);

        let mut total_files = 0;
        let mut files_with_fake = 0;
        let mut fake_instances = 0;
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        let mut worst_offenders = Vec::new();

        for entry in WalkDir::new(directory)
            .into_iter()
            .filter_entry(|e| !self.should_skip_path(e.path()))
        {
            let entry = entry?;
            let path = entry.path();

            if !entry.file_type().is_file() {
                continue;
            }

            if !self.is_code_file(path) {
                continue;
            }

            total_files += 1;

            match self.scan_file(path) {
                Ok(Some(detection)) => {
                    files_with_fake += 1;
                    fake_instances += detection.fake_count;

                    for (category, matches) in &detection.matches {
                        *category_counts.entry(category.clone()).or_insert(0) += matches.len();
                    }

                    worst_offenders.push(detection);
                }
                Ok(None) => {}
                Err(e) => {
                    warn!("Error scanning file {:?}: {}", path, e);
                }
            }
        }

        // Sort worst offenders by fake count
        worst_offenders.sort_by(|a, b| b.fake_count.cmp(&a.fake_count));

        let fake_percentage = if total_files > 0 {
            (files_with_fake as f64 / total_files as f64) * 100.0
        } else {
            0.0
        };

        let summary = DetectionSummary {
            total_files,
            files_with_fake,
            fake_instances,
            fake_percentage,
            scan_timestamp: chrono::Utc::now().to_rfc3339(),
        };

        Ok(DetectionResult {
            summary,
            worst_offenders,
            category_counts,
        })
    }

    pub fn save_report(
        &self,
        result: &DetectionResult,
        output_path: &Path,
    ) -> BullshitBusterResult<()> {
        let json = serde_json::to_string_pretty(result)?;
        fs::write(output_path, json)?;
        info!("Detection report saved to: {:?}", output_path);
        Ok(())
    }
}

impl Default for BullshitDetector {
    fn default() -> Self {
        Self::new().expect("Failed to create BullshitDetector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_detect_stub_function() {
        let detector = BullshitDetector::new().expect("Failed to create BullshitDetector in test");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file in test");
        writeln!(
            temp_file,
            "fn foo() {{\n    // TODO: implement this\n    unimplemented!()\n}}"
        )
        .expect("Failed to write to temp file in test");

        let result = detector
            .scan_file(temp_file.path())
            .expect("Failed to scan file in test");
        assert!(result.is_some());

        let detection = result.expect("Detection result should be present");
        assert!(detection.fake_count > 0);
        assert!(detection.categories.contains(&"stub_functions".to_string()));
    }

    #[test]
    fn test_detect_hardcoded_values() {
        let detector = BullshitDetector::new().expect("Failed to create BullshitDetector in test");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file in test");
        writeln!(temp_file, "let value = 0.7; // magic number")
            .expect("Failed to write to temp file in test");

        let result = detector
            .scan_file(temp_file.path())
            .expect("Failed to scan file in test");
        assert!(result.is_some());

        let detection = result.expect("Detection result should be present");
        assert!(detection.fake_count > 0);
        assert!(detection
            .categories
            .contains(&"hardcoded_values".to_string()));
    }

    #[test]
    fn test_clean_code() {
        let detector = BullshitDetector::new().expect("Failed to create BullshitDetector in test");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file in test");
        writeln!(
            temp_file,
            "fn calculate(config: &Config) -> f64 {{\n    config.get_value()\n}}"
        )
        .expect("Failed to write to temp file in test");

        let result = detector
            .scan_file(temp_file.path())
            .expect("Failed to scan file in test");
        assert!(result.is_none());
    }
}
