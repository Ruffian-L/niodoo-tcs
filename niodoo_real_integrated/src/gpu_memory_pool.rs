//! GPU Memory Pool for RTX 5090
//!
//! Reuses GPU tensor buffers to minimize allocation overhead
//! and maximize memory efficiency for RTX 5090's 32GB VRAM.

#[cfg(feature = "gpu")]
use candle_core::{Device, DType, Result, Shape, Tensor};
#[cfg(feature = "gpu")]
use std::collections::VecDeque;
#[cfg(feature = "gpu")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "gpu")]
use tracing::{debug, info};

/// GPU memory pool for tensor reuse - RTX 5090 optimized
#[cfg(feature = "gpu")]
pub struct GpuMemoryPool {
    device: Device,
    pools: Arc<Mutex<Vec<VecDeque<Tensor>>>>, // Indexed by size buckets
    max_pool_size: usize,
}

#[cfg(feature = "gpu")]
impl GpuMemoryPool {
    /// Create new GPU memory pool
    pub fn new(device: Device, max_pool_size: usize) -> Self {
        Self {
            device,
            pools: Arc::new(Mutex::new(Vec::new())),
            max_pool_size,
        }
    }

    /// Get or allocate tensor of specified shape
    pub fn get_tensor(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let total_elements: usize = shape.iter().product();
        let pool_idx = Self::size_bucket(total_elements);
        
        let mut pools = self.pools.lock().unwrap();
        
        // Ensure pool exists
        while pools.len() <= pool_idx {
            pools.push(VecDeque::new());
        }
        
        // Try to reuse from pool
        while let Some(mut tensor) = pools[pool_idx].pop_front() {
            if tensor.shape().dims() == shape && tensor.dtype() == dtype {
                // Reuse this tensor - zero it out
                return Ok(tensor.zeros_like()?);
            }
        }
        
        // Allocate new tensor
        Tensor::zeros(shape, dtype, &self.device)
    }

    /// Return tensor to pool for reuse
    pub fn return_tensor(&self, tensor: Tensor) {
        let total_elements: usize = tensor.shape().dims().iter().product();
        let pool_idx = Self::size_bucket(total_elements);
        
        let mut pools = self.pools.lock().unwrap();
        
        // Ensure pool exists
        while pools.len() <= pool_idx {
            pools.push(VecDeque::new());
        }
        
        // Add to pool if not full
        if pools[pool_idx].len() < self.max_pool_size {
            pools[pool_idx].push_back(tensor);
        }
    }

    /// Size bucket for tensor pooling (logarithmic buckets)
    fn size_bucket(size: usize) -> usize {
        if size == 0 {
            return 0;
        }
        (size.ilog2() as usize).min(30) // Max 30 buckets
    }

    /// Clear pool to free GPU memory
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();
        info!("GPU memory pool cleared");
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize) {
        let pools = self.pools.lock().unwrap();
        let total_tensors: usize = pools.iter().map(|p| p.len()).sum();
        let total_elements: usize = pools.iter().flat_map(|p| {
            p.iter().map(|t| t.shape().dims().iter().product::<usize>())
        }).sum();
        (total_tensors, total_elements)
    }
}

#[cfg(feature = "gpu")]
impl Default for GpuMemoryPool {
    fn default() -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        Self::new(device, 100) // Default: 100 tensors per bucket
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_pool() {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let pool = GpuMemoryPool::new(device, 10);
        
        let tensor1 = pool.get_tensor(&[100, 100], DType::F32).unwrap();
        pool.return_tensor(tensor1);
        
        let tensor2 = pool.get_tensor(&[100, 100], DType::F32).unwrap();
        assert_eq!(tensor2.shape().dims(), &[100, 100]);
    }
}

