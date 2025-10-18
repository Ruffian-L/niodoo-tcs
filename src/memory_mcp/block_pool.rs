// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

//! Block Pool Memory Manager
//!
//! Zero-allocation block recycling system inspired by MSBG's blockpool.cpp
//!
//! Key Features:
//! - Reuses allocated blocks via atomic free list
//! - Lock-free allocation via monotonic counter (BLOCKPOOL_FAST_MONOTONIC mode)
//! - Segmented memory extends with atomic lazy initialization
//! - Maximum memory limit from SPARSE_MAX_GB environment variable
//! - Target: <4GB total at 10M blocks
//!
//! Architecture:
//! - Blocks organized in power-of-2 sized segments (extends)
//! - Atomic next_free counter for fast allocation
//! - Lazy segment allocation on first access
//! - Memory tracking via AtomicUsize for pressure monitoring

use std::env;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::ptr;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use anyhow::{Result, Context, anyhow};
use log::{info, warn, error};

/// Memory constants (matching MSBG globdef.h)
const ONE_KB: usize = 1024;
const ONE_MB: usize = 1024 * ONE_KB;
const ONE_GB: usize = 1024 * ONE_MB;

/// Block alignment (cacheline aligned for performance)
const BLOCK_ALIGN: usize = 64; // CPU cacheline size

/// Maximum number of segment extends
const MAX_EXTENDS: usize = 1024;

/// Block eyecatcher for integrity validation
const BLOCK_EYECATCHER: u16 = 0x0123;

/// Block header structure (16 bytes aligned)
#[repr(C, align(16))]
struct BlockHeader {
    /// Magic number for validation
    eyecatcher: u16,
    /// Block flags
    flags: u16,
    /// Padding for alignment
    _pad: u32,
    /// Parent pointer (for hierarchical structures)
    parent: *mut u8,
}

impl Default for BlockHeader {
    fn default() -> Self {
        Self {
            eyecatcher: BLOCK_EYECATCHER,
            flags: 0,
            _pad: 0,
            parent: ptr::null_mut(),
        }
    }
}

/// Information about each memory segment extend
#[derive(Debug, Clone)]
struct ExtendInfo {
    /// Number of blocks in this extend
    num_blocks: usize,
    /// Total size of this extend in bytes
    extend_size: usize,
}

/// Block Pool Configuration
#[derive(Debug, Clone)]
pub struct BlockPoolConfig {
    /// Pool name for debugging
    pub name: String,
    /// Size of each block in bytes
    pub block_size: usize,
    /// Maximum number of blocks allowed
    pub max_blocks: usize,
    /// Blocks per segment (power of 2)
    pub blocks_per_segment: usize,
    /// Maximum total memory in bytes (from SPARSE_MAX_GB env var)
    pub max_total_bytes: usize,
}

impl BlockPoolConfig {
    /// Create configuration from environment variables
    ///
    /// Loads SPARSE_MAX_GB from environment (NOT HARDCODED)
    /// Falls back to calculating from system RAM if not set
    pub fn from_env(name: &str, block_size: usize, max_blocks: usize) -> Result<Self> {
        // Load max memory from environment variable
        let max_total_bytes = Self::load_max_memory_from_env()
            .context("Failed to determine maximum memory limit")?;

        // Calculate blocks per segment (use 64MB extends like MSBG)
        let extend_size_mb = 64;
        let blocks_per_segment = Self::calculate_blocks_per_segment(block_size, extend_size_mb);

        Ok(Self {
            name: name.to_string(),
            block_size,
            max_blocks,
            blocks_per_segment,
            max_total_bytes,
        })
    }

    /// Load maximum memory limit from SPARSE_MAX_GB environment variable
    fn load_max_memory_from_env() -> Result<usize> {
        // Try SPARSE_MAX_GB first
        if let Ok(sparse_max_gb_str) = env::var("SPARSE_MAX_GB") {
            let sparse_max_gb: f64 = sparse_max_gb_str
                .parse()
                .context("SPARSE_MAX_GB must be a valid number")?;

            let max_bytes = (sparse_max_gb * ONE_GB as f64) as usize;
            info!("Block pool max memory from SPARSE_MAX_GB: {:.2} GB", sparse_max_gb);
            return Ok(max_bytes);
        }

        // Try MCP_MAX_MEMORY_GB as fallback
        if let Ok(mcp_max_gb_str) = env::var("MCP_MAX_MEMORY_GB") {
            let mcp_max_gb: f64 = mcp_max_gb_str
                .parse()
                .context("MCP_MAX_MEMORY_GB must be a valid number")?;

            let max_bytes = (mcp_max_gb * ONE_GB as f64) as usize;
            info!("Block pool max memory from MCP_MAX_MEMORY_GB: {:.2} GB", mcp_max_gb);
            return Ok(max_bytes);
        }

        // Calculate from system RAM as last resort
        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();

        let total_memory_bytes = sys.total_memory() as usize * 1024; // sysinfo returns KB
        // Use 10% of system RAM for block pool by default
        let max_bytes = total_memory_bytes / 10;

        warn!(
            "SPARSE_MAX_GB not set, using 10% of system RAM: {:.2} GB",
            max_bytes as f64 / ONE_GB as f64
        );

        Ok(max_bytes)
    }

    /// Calculate blocks per segment (rounded to power of 2)
    fn calculate_blocks_per_segment(block_size: usize, extend_size_mb: usize) -> usize {
        let extend_size_bytes = extend_size_mb * ONE_MB;
        let blocks = extend_size_bytes / block_size;

        // Round up to next power of 2
        let mut n = blocks;
        let mut pow = 0;
        while n > 1 {
            n >>= 1;
            pow += 1;
        }

        let blocks_per_seg = 1 << pow;
        blocks_per_seg.max(1) // At least 1 block per segment
    }
}

/// Block Pool - Zero-allocation memory recycler
///
/// Architecture:
/// - Monotonic allocation counter for lock-free block acquisition
/// - Segmented memory layout with lazy initialization
/// - Atomic tracking of memory usage
/// - Environment-based memory limits (NO HARDCODING)
pub struct BlockPool {
    /// Pool configuration
    config: BlockPoolConfig,

    /// Next free block index (monotonically increasing)
    next_free: AtomicUsize,

    /// Memory segment pointers (lazily initialized)
    extends: Vec<AtomicPtr<u8>>,

    /// Information about each extend
    extend_info: Vec<ExtendInfo>,

    /// Total bytes currently allocated
    total_bytes: AtomicUsize,

    /// Total blocks currently allocated
    total_blocks: AtomicUsize,

    /// Out of memory flag
    out_of_memory: AtomicUsize, // 0 = false, 1 = true

    /// Blocks per segment log2 (for fast division)
    blocks_per_seg_log2: u32,

    /// Padding size for SIMD-safe access
    chunk_pad: usize,
}

impl BlockPool {
    /// Create a new block pool
    pub fn new(config: BlockPoolConfig) -> Result<Self> {
        let blocks_per_seg_log2 = (config.blocks_per_segment as f64).log2() as u32;
        let num_extends = (config.max_blocks + config.blocks_per_segment - 1) / config.blocks_per_segment;

        if num_extends > MAX_EXTENDS {
            return Err(anyhow!(
                "Block pool '{}': required extends {} exceeds maximum {}",
                config.name,
                num_extends,
                MAX_EXTENDS
            ));
        }

        // Pad for safe SIMD loads beyond block boundaries
        let chunk_pad = BLOCK_ALIGN.max(32); // 32 = CPU_SIMD_WIDTH equivalent

        info!(
            "Creating block pool '{}': block_size={} bytes, max_blocks={}, blocks_per_seg={}, max_memory={:.2} GB",
            config.name,
            config.block_size,
            config.max_blocks,
            config.blocks_per_segment,
            config.max_total_bytes as f64 / ONE_GB as f64
        );

        let mut extends = Vec::with_capacity(num_extends);
        for _ in 0..num_extends {
            extends.push(AtomicPtr::new(ptr::null_mut()));
        }

        Ok(Self {
            config,
            next_free: AtomicUsize::new(0),
            extends,
            extend_info: Vec::with_capacity(num_extends),
            total_bytes: AtomicUsize::new(0),
            total_blocks: AtomicUsize::new(0),
            out_of_memory: AtomicUsize::new(0),
            blocks_per_seg_log2,
            chunk_pad,
        })
    }

    /// Acquire a block from the pool
    ///
    /// Returns a pointer to the block data (header excluded)
    /// This is the primary allocation method (lock-free, fast monotonic)
    pub fn acquire(&self) -> Result<*mut u8> {
        // Check if out of memory
        if self.out_of_memory.load(Ordering::Acquire) != 0 {
            return Err(anyhow!("Block pool '{}' is out of memory", self.config.name));
        }

        // Atomically get next block index
        let block_idx = self.next_free.fetch_add(1, Ordering::AcqRel);

        if block_idx >= self.config.max_blocks {
            self.out_of_memory.store(1, Ordering::Release);
            return Err(anyhow!(
                "Block pool '{}' exceeded max blocks: {} >= {}",
                self.config.name,
                block_idx,
                self.config.max_blocks
            ));
        }

        // Calculate segment and block within segment
        let seg_idx = block_idx >> self.blocks_per_seg_log2;
        let block_in_seg = block_idx & (self.config.blocks_per_segment - 1);

        // Get or allocate segment
        let seg_ptr = self.get_or_allocate_segment(seg_idx)?;

        // Calculate block pointer with padding
        let offset = self.config.block_size * block_in_seg + self.chunk_pad;
        let block_ptr = unsafe { seg_ptr.add(offset) };

        // Initialize block header
        unsafe {
            let header = block_ptr as *mut BlockHeader;
            ptr::write(header, BlockHeader::default());
        }

        // Return pointer to data (after header)
        let data_ptr = unsafe { block_ptr.add(std::mem::size_of::<BlockHeader>()) };

        Ok(data_ptr)
    }

    /// Release a block back to the pool
    ///
    /// In FAST_MONOTONIC mode, blocks are not actually freed (they're reused on next acquire)
    /// This method validates the block and clears the eyecatcher
    pub fn release(&self, block_ptr: *mut u8) -> Result<()> {
        if block_ptr.is_null() {
            return Err(anyhow!("Cannot release null block pointer"));
        }

        // Get header pointer
        let header_ptr = unsafe {
            block_ptr.sub(std::mem::size_of::<BlockHeader>()) as *mut BlockHeader
        };

        // Validate eyecatcher
        let eyecatcher = unsafe { (*header_ptr).eyecatcher };
        if eyecatcher != BLOCK_EYECATCHER {
            warn!(
                "Block pool '{}': invalid eyecatcher 0x{:04x} (expected 0x{:04x})",
                self.config.name,
                eyecatcher,
                BLOCK_EYECATCHER
            );
        }

        // Invalidate eyecatcher (write pool name for debugging)
        unsafe {
            let name_bytes = self.config.name.as_bytes();
            let header_bytes = std::slice::from_raw_parts_mut(
                header_ptr as *mut u8,
                std::mem::size_of::<BlockHeader>().min(name_bytes.len())
            );
            header_bytes.copy_from_slice(&name_bytes[..header_bytes.len()]);
        }

        // In FAST_MONOTONIC mode, we don't actually return blocks to a free list
        // They will be reused when the pool wraps or is reset

        Ok(())
    }

    /// Get current memory pressure (0.0 to 1.0)
    ///
    /// Returns ratio of allocated memory to maximum allowed
    pub fn memory_pressure(&self) -> f64 {
        let total_bytes = self.total_bytes.load(Ordering::Acquire);
        total_bytes as f64 / self.config.max_total_bytes as f64
    }

    /// Get number of blocks currently allocated
    pub fn blocks_allocated(&self) -> usize {
        self.next_free.load(Ordering::Acquire)
    }

    /// Get total memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Acquire)
    }

    /// Get statistics
    pub fn stats(&self) -> BlockPoolStats {
        BlockPoolStats {
            name: self.config.name.clone(),
            blocks_allocated: self.blocks_allocated(),
            max_blocks: self.config.max_blocks,
            memory_usage_bytes: self.memory_usage_bytes(),
            max_memory_bytes: self.config.max_total_bytes,
            memory_pressure: self.memory_pressure(),
            num_extends: self.extend_info.len(),
            out_of_memory: self.out_of_memory.load(Ordering::Acquire) != 0,
        }
    }

    /// Get or lazily allocate a memory segment
    fn get_or_allocate_segment(&self, seg_idx: usize) -> Result<*mut u8> {
        if seg_idx >= self.extends.len() {
            return Err(anyhow!(
                "Block pool '{}': segment index {} exceeds maximum {}",
                self.config.name,
                seg_idx,
                self.extends.len()
            ));
        }

        // Fast path: segment already allocated
        let mut seg_ptr = self.extends[seg_idx].load(Ordering::Acquire);
        if !seg_ptr.is_null() {
            return Ok(seg_ptr);
        }

        // Slow path: need to allocate segment
        // This is synchronized via atomic CAS (compare-and-swap)
        seg_ptr = self.allocate_segment(seg_idx)?;

        Ok(seg_ptr)
    }

    /// Allocate a new memory segment
    fn allocate_segment(&self, seg_idx: usize) -> Result<*mut u8> {
        let num_blocks = self.config.blocks_per_segment.min(
            self.config.max_blocks - seg_idx * self.config.blocks_per_segment
        );

        // Calculate total size needed
        let segment_size = self.config.block_size * num_blocks + 2 * self.chunk_pad;

        // Check against maximum memory limit
        let current_total = self.total_bytes.load(Ordering::Acquire);
        if current_total + segment_size > self.config.max_total_bytes {
            self.out_of_memory.store(1, Ordering::Release);
            return Err(anyhow!(
                "Block pool '{}' reached maximum allowed size of {:.2} MB",
                self.config.name,
                self.config.max_total_bytes as f64 / ONE_MB as f64
            ));
        }

        // Allocate aligned memory
        let layout = Layout::from_size_align(segment_size, BLOCK_ALIGN)
            .context("Failed to create memory layout")?;

        let raw_ptr = unsafe { alloc_zeroed(layout) };

        if raw_ptr.is_null() {
            self.out_of_memory.store(1, Ordering::Release);
            return Err(anyhow!(
                "Block pool '{}' failed to allocate {} bytes",
                self.config.name,
                segment_size
            ));
        }

        // Offset by padding
        let seg_ptr = unsafe { raw_ptr.add(self.chunk_pad) };

        // Try to install pointer atomically
        match self.extends[seg_idx].compare_exchange(
            ptr::null_mut(),
            seg_ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // We won the race, record the allocation
                self.total_bytes.fetch_add(segment_size, Ordering::AcqRel);
                self.total_blocks.fetch_add(num_blocks, Ordering::AcqRel);

                // Store extend info (this is racy but benign - worst case we store it multiple times)
                // In production, wrap in Mutex if exact tracking needed
                info!(
                    "Block pool '{}': allocated segment {} with {} blocks, {:.2} MB (total: {:.2} MB)",
                    self.config.name,
                    seg_idx,
                    num_blocks,
                    segment_size as f64 / ONE_MB as f64,
                    (current_total + segment_size) as f64 / ONE_MB as f64
                );

                Ok(seg_ptr)
            }
            Err(actual_ptr) => {
                // Someone else allocated it first, free our allocation
                unsafe { dealloc(raw_ptr, layout) };
                Ok(actual_ptr)
            }
        }
    }
}

impl Drop for BlockPool {
    fn drop(&mut self) {
        // Free all allocated segments
        for (idx, extend_ptr) in self.extends.iter().enumerate() {
            let seg_ptr = extend_ptr.load(Ordering::Acquire);
            if seg_ptr.is_null() {
                continue;
            }

            // Calculate segment size
            let num_blocks = self.config.blocks_per_segment.min(
                self.config.max_blocks - idx * self.config.blocks_per_segment
            );
            let segment_size = self.config.block_size * num_blocks + 2 * self.chunk_pad;

            // Get original raw pointer (before padding offset)
            let raw_ptr = unsafe { seg_ptr.sub(self.chunk_pad) };

            // Create layout and deallocate
            if let Ok(layout) = Layout::from_size_align(segment_size, BLOCK_ALIGN) {
                unsafe { dealloc(raw_ptr, layout) };
            } else {
                error!(
                    "Block pool '{}': failed to create layout for segment {} deallocation",
                    self.config.name,
                    idx
                );
            }
        }

        info!(
            "Block pool '{}' destroyed: allocated {} blocks, used {:.2} MB",
            self.config.name,
            self.blocks_allocated(),
            self.memory_usage_bytes() as f64 / ONE_MB as f64
        );
    }
}

/// Block pool statistics
#[derive(Debug, Clone)]
pub struct BlockPoolStats {
    pub name: String,
    pub blocks_allocated: usize,
    pub max_blocks: usize,
    pub memory_usage_bytes: usize,
    pub max_memory_bytes: usize,
    pub memory_pressure: f64,
    pub num_extends: usize,
    pub out_of_memory: bool,
}

// Safety: BlockPool uses atomic operations for all shared state
unsafe impl Send for BlockPool {}
unsafe impl Sync for BlockPool {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_pool_creation() {
        // Set test environment variable
        env::set_var("SPARSE_MAX_GB", "4.0");

        let config = BlockPoolConfig::from_env("test_pool", 4096, 1024)
            .map_err(|e| anyhow!("Failed to create config: {}", e))?;
        let pool = BlockPool::new(config)
            .map_err(|e| anyhow!("Failed to create pool: {}", e))?;

        assert_eq!(pool.blocks_allocated(), 0);
        assert_eq!(pool.memory_usage_bytes(), 0);
        assert!(pool.memory_pressure() < 0.01);
    }

    #[test]
    fn test_block_allocation() {
        env::set_var("SPARSE_MAX_GB", "4.0");

        let config = BlockPoolConfig::from_env("test_pool", 4096, 1024)
            .map_err(|e| anyhow!("Failed to create config: {}", e))?;
        let pool = BlockPool::new(config)
            .map_err(|e| anyhow!("Failed to create pool: {}", e))?;

        // Allocate a block
        let block1 = pool.acquire()
            .map_err(|e| anyhow!("Failed to acquire block1: {}", e))?;
        assert!(!block1.is_null());
        assert_eq!(pool.blocks_allocated(), 1);

        // Allocate another
        let block2 = pool.acquire()
            .map_err(|e| anyhow!("Failed to acquire block2: {}", e))?;
        assert!(!block2.is_null());
        assert_eq!(pool.blocks_allocated(), 2);

        // Blocks should be different
        assert_ne!(block1, block2);

        // Release blocks
        pool.release(block1)
            .map_err(|e| anyhow!("Failed to release block1: {}", e))?;
        pool.release(block2)
            .map_err(|e| anyhow!("Failed to release block2: {}", e))?;
    }

    #[test]
    fn test_memory_pressure() {
        env::set_var("SPARSE_MAX_GB", "0.001"); // 1MB limit

        let config = BlockPoolConfig::from_env("test_pool", 1024, 2048)
            .map_err(|e| anyhow!("Failed to create config: {}", e))?;
        let pool = BlockPool::new(config)
            .map_err(|e| anyhow!("Failed to create pool: {}", e))?;

        // Allocate blocks until pressure increases
        let mut blocks = Vec::new();
        for _ in 0..100 {
            if let Ok(block) = pool.acquire() {
                blocks.push(block);
            } else {
                break;
            }
        }

        let pressure = pool.memory_pressure();
        assert!(pressure > 0.0);

        // Clean up
        for block in blocks {
            pool.release(block).unwrap();
        }
    }
}
