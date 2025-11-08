//! NIODOO Python API - Rust-Orchestrated Hybrid FFI Bridge
//!
//! This module provides Python bindings for NIODOO functionality via pyo3.
//! The architecture follows the "Rust-Orchestrated Hybrid" model where:
//! - Rust handles parsing and orchestration
//! - Python (via FFI) handles TDA computations using mature libraries like giotto-tda

#[cfg(feature = "pyo3")]
pub mod parser;

#[cfg(feature = "pyo3")]
pub mod tcs;

#[cfg(feature = "pyo3")]
pub mod erag;

#[cfg(feature = "pyo3")]
pub mod tqft;

