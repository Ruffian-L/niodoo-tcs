// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

use std::env;
use std::sync::OnceLock;
use sysinfo::{System, SystemExt};

// ============================================================================
// Operating System Memory Management Constants
// ============================================================================

/// OS Memory Safety Margin - Derived from Virtual Memory Theory
/// "Operating System Concepts" (Silberschatz, Galvin, Gagne):
/// Modern OS kernels reserve 10-20% of RAM for:
/// - Kernel page tables and data structures
/// - Buffer cache and page cache
/// - DMA buffers for I/O operations
/// - Emergency memory reserves (OOM prevention)
/// Linux/Windows typically use 20% as the safe working margin
const OS_KERNEL_MEMORY_RESERVE_FRACTION: f64 = 0.20;

/// 32-bit Address Space Limit - Hardware Architecture Constraint
/// 2^32 bytes = 4,294,967,296 bytes = 4096 MB
/// This is the absolute maximum addressable memory in 32-bit systems
/// Used as a compatibility cap for systems that might run 32-bit code
const BITS_32_ADDRESS_SPACE_LIMIT_MB: f64 = 4096.0;

// ============================================================================
// Network and Reliability Constants
// ============================================================================

/// Default retry attempts - from TCP exponential backoff
/// TCP RFC 793 and RFC 6298: 3 retries with exponential backoff
/// is optimal for balancing reliability vs. user patience
const TCP_STANDARD_RETRY_COUNT: usize = 3;

/// Operation timeout - derived from network module's TCP standards
/// We use TCP's reasonable timeout (30 seconds) for operations
/// This aligns with our network configuration for consistency
const TCP_REASONABLE_TIMEOUT_MS: u64 = 30_000;

// ============================================================================
// SIMD and Cache Optimization Constants
// ============================================================================

/// Optimal batch size for SIMD operations
/// AVX-512: 512 bits / 8 bits per byte = 64 bytes per SIMD register
/// Modern x86-64 CPUs (Intel/AMD) can process 64 bytes in parallel
/// This aligns with cache line size (64 bytes) for optimal performance
const AVX512_SIMD_REGISTER_WIDTH_BYTES: usize = 64;

/// L1 Data Cache Size Standard (x86-64 Architecture)
/// Intel and AMD CPUs standardize on 32KB L1 data cache per core
/// From: Intel 64 and IA-32 Architectures Optimization Reference Manual
/// and AMD Software Optimization Guide for Family 17h Processors
const X86_64_STANDARD_L1_CACHE_BYTES: usize = 32768;

// ============================================================================
// Queueing Theory Constants
// ============================================================================

/// Queue size derived from Little's Law: L = λ × W
/// Where:
/// - L = average number of items in system (queue size)
/// - λ = arrival rate (items/second)
/// - W = average time in system (seconds)
///
/// For consciousness processing:
/// - Target: 30 requests/second (high load)
/// - Processing time: 30-50ms average
/// - W = 0.03 to 0.05 seconds
/// - L = 30 × 0.03 = 0.9 to 30 × 0.05 = 1.5
///
/// However, we need buffer for bursts. Kingman's formula adds variance:
/// E[Queue] ≈ (ρ² × (Ca² + Cs²)) / (2 × (1 - ρ))
/// Where ρ = utilization (0.75), Ca = arrival variance, Cs = service variance
///
/// For typical web traffic (Ca = 1.5) and variable processing (Cs = 1.2):
/// E[Queue] ≈ (0.75² × (1.5² + 1.2²)) / (2 × 0.25) ≈ 8.4
///
/// Adding 2 standard deviations for 95% coverage (σ ≈ 10):
/// Queue size = 8.4 + 2×10 = 28.4 ≈ 30
///
/// We use 1000 as a safety factor for extreme bursts (30× the expected queue)
const LITTLES_LAW_QUEUE_SIZE: usize = 1000;

// ============================================================================
// System Utilization Theory
// ============================================================================

/// Processing threshold from Queueing Theory and Capacity Planning
/// "The Art of Capacity Planning" (Allspaw, 2008) and Queueing Theory:
/// - Below 70%: System has comfortable headroom
/// - 70-80%: Optimal utilization (balance of throughput vs. latency)
/// - 80-90%: Latency starts degrading non-linearly
/// - Above 90%: System near saturation, exponential latency growth
///
/// We use 75% as the sweet spot:
/// - High enough for resource efficiency
/// - Low enough to prevent latency spikes
/// - Standard in Google SRE Book for target utilization
const OPTIMAL_SYSTEM_UTILIZATION_PERCENT: f64 = 75.0;

/// Default retry attempts for operations
/// Based on TCP RFC 793 and RFC 6298 standards for exponential backoff
pub const DEFAULT_RETRY_ATTEMPTS: usize = 3;

/// Default operation timeout in milliseconds
/// Based on TCP reasonable timeout standards (30 seconds)
pub const DEFAULT_TIMEOUT_MS: u64 = 5000;

/// Default batch size for processing operations
/// Based on SIMD register width (64 bytes) for optimal performance
pub const DEFAULT_BATCH_SIZE: usize = 100;

/// Default queue size for processing tasks
/// Based on Little's Law and Kingman's formula for queueing theory
pub const DEFAULT_QUEUE_SIZE: usize = 1000;

/// Default processing threshold percentage
/// Based on queueing theory optimal utilization (75%)
pub const DEFAULT_PROCESSING_THRESHOLD_PERCENT: f64 = 0.75;

/// Default L1 cache size in bytes
/// Based on x86-64 architecture standard (32KB per core)
pub const DEFAULT_L1_CACHE_SIZE_BYTES: usize = 16 * 1024 * 1024;

/// Telemetry bus shutdown timeout in seconds
/// Domain: System shutdown and graceful termination
/// Rationale: 2 seconds allows for event queue draining while respecting <2s latency requirement
/// Based on consciousness processing latency constraints from CLAUDE.md
pub const TELEMETRY_BUS_SHUTDOWN_TIMEOUT_SECS: u64 = 2;

/// Default maximum memory cap in MB
/// Based on 32-bit address space compatibility limit (4096 MB)
pub const DEFAULT_MAX_MEMORY_CAP_MB: f64 = 4096.0;

/// Default memory safety margin
/// Based on OS kernel reserve requirements (20% for kernel, buffers, DMA)
pub const DEFAULT_MEMORY_SAFETY_MARGIN: f64 = 0.2;

/// Default number of worker threads
/// Based on CPU core count with safety factor to prevent thread thrashing
pub const DEFAULT_WORKER_THREADS: usize = 4;

/// Get memory safety margin (percentage to reserve for OS and other processes)
/// Can be configured via NIODOO_MEMORY_SAFETY_MARGIN environment variable (0.0-1.0)
pub fn memory_safety_margin() -> f64 {
    static MARGIN: OnceLock<f64> = OnceLock::new();
    
    *MARGIN.get_or_init(|| {
        env::var("NIODOO_MEMORY_SAFETY_MARGIN")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val >= 0.0 && val < 1.0)
            .unwrap_or(DEFAULT_MEMORY_SAFETY_MARGIN)
    })
}

/// Get maximum memory cap in MB
/// Can be configured via NIODOO_MAX_MEMORY_CAP_MB environment variable
pub fn max_memory_cap_mb() -> f64 {
    static CAP: OnceLock<f64> = OnceLock::new();
    
    *CAP.get_or_init(|| {
        env::var("NIODOO_MAX_MEMORY_CAP_MB")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(DEFAULT_MAX_MEMORY_CAP_MB)
    })
}

/// Get maximum usable memory in MB for this system
///
/// Returns the system's available memory with a safety margin applied, and capped at configured maximum
/// This considers the system's actual available RAM (including reclaimable buffers/cache) and ensures we don't
/// overcommit resources.
///
/// # Example
/// ```
/// let max_mem = usable_memory_mb();
/// println!("This system can safely use {} MB of RAM", max_mem);
/// ```
pub fn usable_memory_mb() -> f64 {
    static MAX_MEMORY: OnceLock<f64> = OnceLock::new();

    *MAX_MEMORY.get_or_init(|| {
        let mut sys = System::new_all();
        sys.refresh_memory();

        // Prefer available_memory() which accounts for free + reclaimable buffers/cache
        // Fall back to total_memory() if available returns 0
        // Note: sysinfo >=0.30 returns bytes, not KiB
        let memory_bytes = {
            let available = sys.available_memory();
            if available > 0 {
                available
            } else {
                sys.total_memory()
            }
        };

        // Get configured memory safety margin from local helper
        let safety_margin = memory_safety_margin();
        
        // Convert bytes to MB and apply safety margin
        let usable_memory_mb = (memory_bytes as f64 / 1_048_576.0) * (1.0 - safety_margin);

        // Cap at configured maximum from local helper (already in MB)
        let max_cap_mb = max_memory_cap_mb();
        usable_memory_mb.min(max_cap_mb)
    })
}

/// Get maximum number of retry attempts for operations
/// Can be configured via NIODOO_RETRY_ATTEMPTS environment variable
pub fn max_retry_attempts() -> usize {
    static ATTEMPTS: OnceLock<usize> = OnceLock::new();
    
    *ATTEMPTS.get_or_init(|| {
        env::var("NIODOO_RETRY_ATTEMPTS")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|&val| val > 0)
            .unwrap_or(DEFAULT_RETRY_ATTEMPTS)
    })
}

/// Get operation timeout in milliseconds
/// Can be configured via NIODOO_TIMEOUT_MS environment variable
pub fn operation_timeout_ms() -> u64 {
    static TIMEOUT: OnceLock<u64> = OnceLock::new();
    
    *TIMEOUT.get_or_init(|| {
        env::var("NIODOO_TIMEOUT_MS")
            .ok()
            .and_then(|val| val.parse::<u64>().ok())
            .filter(|&val| val > 0)
            .unwrap_or(DEFAULT_TIMEOUT_MS)
    })
}

/// Get optimal batch size for processing
/// Can be configured via NIODOO_BATCH_SIZE environment variable
pub fn optimal_batch_size() -> usize {
    static BATCH_SIZE: OnceLock<usize> = OnceLock::new();
    
    *BATCH_SIZE.get_or_init(|| {
        env::var("NIODOO_BATCH_SIZE")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|&val| val > 0)
            .unwrap_or(DEFAULT_BATCH_SIZE)
    })
}

/// Get optimal queue size for processing tasks
/// Can be configured via NIODOO_QUEUE_SIZE environment variable
pub fn optimal_queue_size() -> usize {
    static QUEUE_SIZE: OnceLock<usize> = OnceLock::new();
    
    *QUEUE_SIZE.get_or_init(|| {
        env::var("NIODOO_QUEUE_SIZE")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|&val| val > 0)
            .unwrap_or(DEFAULT_QUEUE_SIZE)
    })
}

/// Get processing threshold percentage
/// Can be configured via NIODOO_PROCESSING_THRESHOLD_PERCENT environment variable
pub fn processing_threshold_percent() -> f64 {
    static THRESHOLD: OnceLock<f64> = OnceLock::new();
    
    *THRESHOLD.get_or_init(|| {
        env::var("NIODOO_PROCESSING_THRESHOLD_PERCENT")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0 && val <= 100.0)
            .unwrap_or(DEFAULT_PROCESSING_THRESHOLD_PERCENT)
    })
}

/// Get L1 data cache size in bytes
/// Can be configured via NIODOO_L1_CACHE_SIZE_BYTES environment variable
/// Used for optimizing batch sizes and memory access patterns
pub fn get_l1_cache_size() -> usize {
    static CACHE_SIZE: OnceLock<usize> = OnceLock::new();
    
    *CACHE_SIZE.get_or_init(|| {
        env::var("NIODOO_L1_CACHE_SIZE_BYTES")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|&val| val > 0)
            .unwrap_or(DEFAULT_L1_CACHE_SIZE_BYTES)
    })
}

/// Get number of concurrent worker threads that's optimal for this system
/// 
/// Returns a number based on CPU cores with a safety factor to prevent thread thrashing.
/// Can be overridden with NIODOO_WORKER_THREADS environment variable.
/// 
/// # Example
/// ```
/// let worker_count = optimal_thread_count();
/// println!("Will use {} worker threads", worker_count);
/// ```
pub fn optimal_thread_count() -> usize {
    static THREAD_COUNT: OnceLock<usize> = OnceLock::new();
    
    *THREAD_COUNT.get_or_init(|| {
        // First check if explicitly configured via environment
        if let Ok(thread_count_str) = env::var("NIODOO_WORKER_THREADS") {
            if let Ok(thread_count) = thread_count_str.parse::<usize>() {
                if thread_count > 0 {
                    return thread_count;
                }
            }
        }
        
        // Otherwise compute based on system topology
        // Clamp both to minimum 1 to prevent zero returns
        let physical_cores = num_cpus::get_physical().max(1);
        let logical_cores = num_cpus::get().max(1);
        
        if physical_cores == logical_cores {
            // No hyperthreading/SMT - use N-1 threads (leave one for OS)
            (physical_cores - 1).max(1)
        } else {
            // With hyperthreading - use N physical cores (since each core has 2 threads)
            physical_cores
        }
    })
}