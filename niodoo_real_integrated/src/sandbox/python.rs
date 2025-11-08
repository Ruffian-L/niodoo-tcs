//! Python sandbox for executing Python code securely

use crate::sandbox::security::{ExecutionResult, SecurityPolicy};
use anyhow::{Context, Result};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::time::timeout;
use tracing::{info, warn};

/// Python sandbox for secure code execution
pub struct PythonSandbox {
    security_policy: SecurityPolicy,
    temp_dir: TempDir,
}

impl PythonSandbox {
    /// Create a new Python sandbox
    pub fn new(security_policy: SecurityPolicy) -> Result<Self> {
        let temp_dir = tempfile::tempdir()
            .context("Failed to create temporary directory for Python sandbox")?;

        Ok(Self {
            security_policy,
            temp_dir,
        })
    }

    /// Execute Python code in the sandbox
    /// 
    /// Security layers:
    /// 1. Static analysis (Guardian using rust-code-analysis) - pre-execution safety checks
    /// 2. Runtime hooks (TODO: Add wasmtime for WASM isolation if TS code involved)
    ///    - Post-execution dynamic violation detection (e.g., indirect network calls)
    ///    - Catches violations that static analysis misses (obfuscated calls)
    ///    - Completes the defense-in-depth safety picture
    pub async fn execute(&self, code: &str) -> Result<ExecutionResult> {
        let start = Instant::now();

        // Create a wrapper script that restricts imports
        let wrapper_code = self.create_wrapper_code(code)?;

        // Write code to temporary file
        let code_file = self.temp_dir.path().join("code.py");
        tokio::fs::write(&code_file, wrapper_code)
            .await
            .context("Failed to write code to temporary file")?;

        // Execute Python code with restrictions
        let mut cmd = Command::new("python3");
        cmd.arg(&code_file)
            .current_dir(self.temp_dir.path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Set environment variables to restrict behavior
        cmd.env("PYTHONPATH", self.temp_dir.path())
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("PYTHONUNBUFFERED", "1");

        // Execute with timeout
        let execution_timeout = self.security_policy.timeout;
        match timeout(execution_timeout, tokio::process::Command::from(cmd).output()).await {
            Ok(Ok(output)) => {
                let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                if output.status.success() {
                    info!(execution_time_ms, "Python code executed successfully");
                    Ok(ExecutionResult::success(stdout, execution_time_ms))
                } else {
                    warn!(stderr, "Python code execution failed");
                    Ok(ExecutionResult::failure(
                        format!("Python execution failed with status: {}", output.status),
                        stderr,
                        execution_time_ms,
                    ))
                }
            }
            Ok(Err(e)) => {
                let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                warn!(?e, "Failed to execute Python code");
                Ok(ExecutionResult::failure(
                    format!("Failed to execute Python: {}", e),
                    String::new(),
                    execution_time_ms,
                ))
            }
            Err(_) => {
                let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                warn!(timeout_secs = ?execution_timeout, "Python code execution timed out");
                Ok(ExecutionResult::failure(
                    format!("Execution timed out after {:?}", execution_timeout),
                    String::new(),
                    execution_time_ms,
                ))
            }
        }
    }

    /// Create wrapper code that enforces import restrictions
    fn create_wrapper_code(&self, user_code: &str) -> Result<String> {
        // Build import whitelist check
        let allowed_imports_str = self
            .security_policy
            .allowed_imports
            .iter()
            .map(|s| format!("\"{}\"", s))
            .collect::<Vec<_>>()
            .join(", ");

        let wrapper = format!(
            r#"
import sys
import importlib.util

# Import whitelist
ALLOWED_IMPORTS = {{{allowed_imports_str}}}

# Override __import__ to enforce whitelist
_original_import = __builtins__.__import__

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Allow standard library imports that are safe
    safe_stdlib = {{'sys', 'json', 'math', 'collections', 'typing', 'datetime', 'time'}}
    
    # Extract base module name
    base_name = name.split('.')[0]
    
    # Check if it's in the whitelist or safe stdlib
    if base_name in ALLOWED_IMPORTS or base_name in safe_stdlib:
        # Additional check: block dangerous stdlib modules
        dangerous = {{'os', 'subprocess', 'socket', 'shutil', 'multiprocessing', 'threading'}}
        if base_name in dangerous:
            raise ImportError(f"Import of '{{base_name}}' is not allowed in sandboxed environment")
        return _original_import(name, globals, locals, fromlist, level)
    else:
        raise ImportError(f"Import of '{{name}}' is not allowed. Allowed imports: {{list(ALLOWED_IMPORTS)}}")

__builtins__.__import__ = _restricted_import

# User code
{user_code}
"#
        );

        Ok(wrapper)
    }
}

