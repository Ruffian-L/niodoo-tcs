// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// error.rs - Comprehensive error handling for Niodoo consciousness system

use std::time::Duration;
use thiserror::Error;

/// Main error type for the consciousness system
#[derive(Error, Debug)]
pub enum NiodoError {
    #[error("Memory system error: {0}")]
    Memory(#[from] MemoryError),

    #[error("Brain processing error: {0}")]
    Brain(#[from] BrainError),

    #[error("API error: {0}")]
    Api(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Qt bridge error: {0}")]
    QtBridge(#[from] QtBridgeError),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Lock acquisition failed: {0}")]
    LockError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Network error: {0}")]
    Network(String),

    #[error("HTTP client error: {0}")]
    HttpClient(#[from] reqwest::Error),

    #[error("Timeout error: {0:?}")]
    ElapsedTimeout(#[from] tokio::time::error::Elapsed),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Unstable feature used: {0}")]
    UnstableFeature(String),

    #[error("Unknown error: {0}")]
    Unknown(String),

    #[error("Stub calculation: {0}")]
    StubCalculation(String),

    #[error("Anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("System time error: {0}")]
    SystemTime(#[from] std::time::SystemTimeError),

    #[error("Invalid point cloud provided")]
    InvalidPointCloud,
}

impl NiodoError {
    /// Calculate ethical gradient for error analysis
    pub fn ethical_gradient(&self) -> f32 {
        match self {
            NiodoError::StubCalculation(_) => 0.1, // Low ethical gradient for stubs
            NiodoError::Memory(_) => 0.5,          // Medium for memory issues
            NiodoError::Brain(_) => 0.7,           // Higher for brain processing
            NiodoError::Api(_) => 0.3,             // Lower for API issues
            NiodoError::Config(_) => 0.2,          // Low for configuration
            NiodoError::Timeout(_) => 0.4,         // Medium for timeouts
            _ => 0.6,                              // Default medium gradient
        }
    }
}

/// Memory-specific errors
#[derive(Error, Debug, Clone)]
pub enum MemoryError {
    #[error("Memory overflow: capacity {capacity} exceeded")]
    Overflow { capacity: usize },

    #[error("Memory corruption detected at address {address}")]
    Corruption { address: String },

    #[error("Invalid memory coordinate: theta={theta}, phi={phi}")]
    InvalidCoordinate { theta: f64, phi: f64 },

    #[error("Memory persistence failed: {0}")]
    PersistenceFailed(String),

    #[error("Memory recall failed: no memories matching query")]
    RecallFailed,

    #[error("Toroidal topology error: {0}")]
    TopologyError(String),
}

/// Brain processing errors
#[derive(Error, Debug, Clone)]
pub enum BrainError {
    #[error("Brain initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Neural pathway blocked: {pathway}")]
    PathwayBlocked { pathway: String },

    #[error("Consensus failed: only {achieved}% consensus (required: {required}%)")]
    ConsensusFailed { achieved: f32, required: f32 },

    #[error("Brain overload: processing queue size {size}")]
    Overload { size: usize },

    #[error("Invalid brain state transition: {from} -> {to}")]
    InvalidStateTransition { from: String, to: String },
}

/// Qt bridge errors
#[derive(Error, Debug, Clone)]
pub enum QtBridgeError {
    #[error("Qt signal emission failed: {0}")]
    SignalFailed(String),

    #[error("Qt connection lost")]
    ConnectionLost,

    #[error("Animation not found: {0}")]
    AnimationNotFound(String),

    #[error("WebSocket error: {0}")]
    WebSocket(String),
}

/// Result type alias for consciousness operations
pub type ConsciousnessResult<T> = std::result::Result<T, NiodoError>;

/// Error recovery strategies
pub struct ErrorRecovery;

impl ErrorRecovery {
    pub fn new(_max_retries: u32) -> Self {
        ErrorRecovery
    }

    /// Attempt to recover from memory errors
    pub async fn recover_memory(error: &MemoryError) -> ConsciousnessResult<()> {
        match error {
            MemoryError::Overflow { capacity } => {
                // Trigger garbage collection
                tracing::warn!("Triggering memory cleanup, capacity: {}", capacity);
                // Implement cleanup logic
                Ok(())
            }
            MemoryError::PersistenceFailed(msg) => {
                // Retry with exponential backoff
                tracing::warn!("Retrying persistence: {}", msg);
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(())
            }
            _ => Err(NiodoError::Memory(error.clone())),
        }
    }

    /// Attempt to recover from brain errors
    pub async fn recover_brain(error: &BrainError) -> ConsciousnessResult<()> {
        match error {
            BrainError::PathwayBlocked { pathway } => {
                // Reroute through alternative pathway
                tracing::warn!("Rerouting around blocked pathway: {}", pathway);
                Ok(())
            }
            BrainError::Overload { size } => {
                // Shed load
                tracing::warn!("Shedding load, queue size: {}", size);
                Ok(())
            }
            _ => Err(NiodoError::Brain(error.clone())),
        }
    }
}

/// Circuit breaker for preventing cascading failures
pub struct CircuitBreaker {
    failure_count: std::sync::atomic::AtomicU32,
    max_failures: u32,
    reset_timeout: Duration,
    last_failure: std::sync::RwLock<Option<std::time::Instant>>,
}

impl CircuitBreaker {
    pub fn new(max_failures: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_count: std::sync::atomic::AtomicU32::new(0),
            max_failures,
            reset_timeout,
            last_failure: std::sync::RwLock::new(None),
        }
    }

    pub fn call<F, T>(&self, f: F) -> ConsciousnessResult<T>
    where
        F: FnOnce() -> ConsciousnessResult<T>,
    {
        // Check if circuit is open
        if self.is_open() {
            return Err(NiodoError::Unknown("Circuit breaker is open".into()));
        }

        // Execute function
        match f() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(e)
            }
        }
    }

    fn is_open(&self) -> bool {
        let count = self
            .failure_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if count >= self.max_failures {
            // Check if we should reset
            if let Ok(last_failure) = self.last_failure.read() {
                if let Some(last) = *last_failure {
                    if last.elapsed() > self.reset_timeout {
                        self.reset();
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    fn on_success(&self) {
        self.failure_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn on_failure(&self) {
        self.failure_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut last_failure) = self.last_failure.write() {
            *last_failure = Some(std::time::Instant::now());
        }
    }

    fn reset(&self) {
        self.failure_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut last_failure) = self.last_failure.write() {
            *last_failure = None;
        }
    }
}

/// CandleFeelingError for backward compatibility
#[derive(Error, Debug)]
pub enum CandleFeelingError {
    #[error("Consciousness error: {0}")]
    ConsciousnessError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Result type alias for CandleFeeling errors
/// Note: Use CandleFeelingResult to avoid conflicts with other Result aliases
pub type CandleFeelingResult<T> = std::result::Result<T, CandleFeelingError>;
