//! Sandboxed code execution module
//!
//! This module provides secure execution environments for agent-generated code.
//! It supports both Python and TypeScript execution with strict security restrictions.

pub mod manager;
pub mod python;
pub mod typescript;
pub mod security;

pub use manager::SandboxManager;
pub use python::PythonSandbox;
pub use typescript::TypeScriptSandbox;
pub use security::{SecurityPolicy, ExecutionResult};



