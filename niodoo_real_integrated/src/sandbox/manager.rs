//! Sandbox manager for coordinating code execution

use crate::config::CodeLanguage;
use crate::sandbox::python::PythonSandbox;
use crate::sandbox::security::{ExecutionResult, SecurityPolicy};
use crate::sandbox::typescript::TypeScriptSandbox;
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Manages sandboxed code execution environments
pub struct SandboxManager {
    python_sandbox: Arc<Mutex<PythonSandbox>>,
    typescript_sandbox: Arc<Mutex<TypeScriptSandbox>>,
    security_policy: SecurityPolicy,
}

impl SandboxManager {
    /// Create a new sandbox manager with the given security policy
    pub fn new(security_policy: SecurityPolicy) -> Result<Self> {
        Ok(Self {
            python_sandbox: Arc::new(Mutex::new(PythonSandbox::new(security_policy.clone())?)),
            typescript_sandbox: Arc::new(Mutex::new(TypeScriptSandbox::new(security_policy.clone())?)),
            security_policy,
        })
    }

    /// Execute code in the appropriate sandbox based on language
    pub async fn execute(&self, code: &str, language: CodeLanguage) -> Result<ExecutionResult> {
        // Validate code length
        if code.len() > self.security_policy.max_code_length {
            return Ok(ExecutionResult::failure(
                format!(
                    "Code length {} exceeds maximum {}",
                    code.len(),
                    self.security_policy.max_code_length
                ),
                String::new(),
                0.0,
            ));
        }

        match language {
            CodeLanguage::Python => {
                let sandbox = self.python_sandbox.lock().await;
                sandbox.execute(code).await
            }
            CodeLanguage::TypeScript => {
                let sandbox = self.typescript_sandbox.lock().await;
                sandbox.execute(code).await
            }
        }
    }

    /// Get the current security policy
    pub fn security_policy(&self) -> &SecurityPolicy {
        &self.security_policy
    }
}



