//! Security policy and execution result types

use crate::config::CodeLanguage;
use std::time::Duration;

/// Security policy for sandboxed execution
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Allowed import modules (whitelist)
    pub allowed_imports: Vec<String>,
    /// Maximum execution time
    pub timeout: Duration,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<u64>,
    /// Maximum code length in characters
    pub max_code_length: usize,
    /// Whether filesystem access is allowed (read-only, scoped to temp dir)
    pub allow_filesystem: bool,
    /// Whether network access is allowed (only to NIODOO services)
    pub allow_network: bool,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            allowed_imports: vec![
                "niodoo".to_string(),
                "numpy".to_string(),
                "typing".to_string(),
                "collections".to_string(),
                "math".to_string(),
                "json".to_string(),
                "datetime".to_string(),
            ],
            timeout: Duration::from_secs(30),
            max_memory_bytes: Some(100 * 1024 * 1024), // 100 MB
            max_code_length: 10000,
            allow_filesystem: false,
            allow_network: false,
        }
    }
}

/// Result of code execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Return value (if applicable)
    pub return_value: Option<String>,
    /// Whether execution succeeded
    pub success: bool,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Error message if execution failed
    pub error: Option<String>,
}

impl ExecutionResult {
    pub fn success(stdout: String, execution_time_ms: f64) -> Self {
        Self {
            stdout,
            stderr: String::new(),
            return_value: None,
            success: true,
            execution_time_ms,
            error: None,
        }
    }

    pub fn failure(error: String, stderr: String, execution_time_ms: f64) -> Self {
        Self {
            stdout: String::new(),
            stderr,
            return_value: None,
            success: false,
            execution_time_ms,
            error: Some(error),
        }
    }
}



