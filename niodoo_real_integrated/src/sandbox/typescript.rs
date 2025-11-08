//! TypeScript sandbox for executing TypeScript code securely

use crate::sandbox::security::{ExecutionResult, SecurityPolicy};
use anyhow::{Context, Result};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::time::timeout;
use tracing::{info, warn};

/// TypeScript sandbox for secure code execution
pub struct TypeScriptSandbox {
    security_policy: SecurityPolicy,
    temp_dir: TempDir,
}

impl TypeScriptSandbox {
    /// Create a new TypeScript sandbox
    pub fn new(security_policy: SecurityPolicy) -> Result<Self> {
        let temp_dir = tempfile::tempdir()
            .context("Failed to create temporary directory for TypeScript sandbox")?;

        Ok(Self {
            security_policy,
            temp_dir,
        })
    }

    /// Execute TypeScript code in the sandbox
    pub async fn execute(&self, code: &str) -> Result<ExecutionResult> {
        let start = Instant::now();

        // For now, we'll use Node.js with vm module for sandboxing
        // In production, consider using Deno with permission flags
        let wrapper_code = self.create_wrapper_code(code)?;

        // Write code to temporary file
        let code_file = self.temp_dir.path().join("code.js");
        tokio::fs::write(&code_file, wrapper_code)
            .await
            .context("Failed to write code to temporary file")?;

        // Execute Node.js code
        let mut cmd = Command::new("node");
        cmd.arg(&code_file)
            .current_dir(self.temp_dir.path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Execute with timeout
        let execution_timeout = self.security_policy.timeout;
        match timeout(execution_timeout, tokio::process::Command::from(cmd).output()).await {
            Ok(Ok(output)) => {
                let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                if output.status.success() {
                    info!(execution_time_ms, "TypeScript code executed successfully");
                    Ok(ExecutionResult::success(stdout, execution_time_ms))
                } else {
                    warn!(stderr, "TypeScript code execution failed");
                    Ok(ExecutionResult::failure(
                        format!("TypeScript execution failed with status: {}", output.status),
                        stderr,
                        execution_time_ms,
                    ))
                }
            }
            Ok(Err(e)) => {
                let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                warn!(?e, "Failed to execute TypeScript code");
                Ok(ExecutionResult::failure(
                    format!("Failed to execute TypeScript: {}", e),
                    String::new(),
                    execution_time_ms,
                ))
            }
            Err(_) => {
                let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                warn!(timeout_secs = ?execution_timeout, "TypeScript code execution timed out");
                Ok(ExecutionResult::failure(
                    format!("Execution timed out after {:?}", execution_timeout),
                    String::new(),
                    execution_time_ms,
                ))
            }
        }
    }

    /// Create wrapper code that uses Node.js vm module for sandboxing
    fn create_wrapper_code(&self, user_code: &str) -> Result<String> {
        // Use Node.js vm module to create a sandboxed context
        // Note: This is a basic implementation. For production, consider using Deno
        // with explicit permission flags for better security.
        let wrapper = format!(
            r#"
const vm = require('vm');
const fs = require('fs');

// Create a sandboxed context with restricted globals
const sandbox = {{
    console: console,
    Buffer: Buffer,
    setTimeout: setTimeout,
    setInterval: setInterval,
    clearTimeout: clearTimeout,
    clearInterval: clearInterval,
    // Block dangerous modules
    require: (module) => {{
        const allowed = ['niodoo', 'util', 'events', 'stream', 'path', 'url', 'querystring', 'crypto'];
        const dangerous = ['fs', 'child_process', 'net', 'dgram', 'http', 'https', 'os', 'cluster'];
        
        if (dangerous.includes(module)) {{
            throw new Error(`Module '${{module}}' is not allowed in sandboxed environment`);
        }}
        
        if (allowed.includes(module)) {{
            return require(module);
        }}
        
        throw new Error(`Module '${{module}}' is not allowed. Allowed modules: ${{allowed.join(', ')}}`);
    }}
}};

// User code
const userCode = `
{user_code}
`;

try {{
    const script = new vm.Script(userCode);
    const context = vm.createContext(sandbox);
    const result = script.runInContext(context, {{ timeout: {timeout_ms} }});
    if (result !== undefined) {{
        console.log(JSON.stringify(result));
    }}
}} catch (error) {{
    console.error(error.message);
    process.exit(1);
}}
"#,
            timeout_ms = self.security_policy.timeout.as_millis()
        );

        Ok(wrapper)
    }
}



