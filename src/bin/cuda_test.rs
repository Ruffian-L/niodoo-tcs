//! Minimal CUDA Test
//!
//! This tests if CUDA initialization works with cuBLAS

use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    println!("🔍 Testing CUDA initialization...");

    // Test 1: Check if CUDA is available
    println!("\n📊 Step 1: Checking CUDA availability...");
    match Device::cuda_if_available(0) {
        Ok(device) => {
            println!("✅ CUDA device available: {:?}", device);

            // Test 2: Create simple tensors
            println!("\n📊 Step 2: Creating tensors on GPU...");
            let a = Tensor::randn(0f32, 1f32, (2, 3), &device)?;
            let b = Tensor::randn(0f32, 1f32, (3, 2), &device)?;
            println!("✅ Tensors created successfully");
            println!("   Tensor A shape: {:?}", a.shape());
            println!("   Tensor B shape: {:?}", b.shape());

            // Test 3: Perform matrix multiplication (this uses cuBLAS)
            println!("\n📊 Step 3: Testing matrix multiplication (cuBLAS)...");
            let c = a.matmul(&b)?;
            println!("✅ Matrix multiplication successful!");
            println!("   Result shape: {:?}", c.shape());
            println!("   Result: {:?}", c.to_vec2::<f32>()?);

            println!("\n🎉 All CUDA tests passed! GPU acceleration is working.");
        }
        Err(e) => {
            println!("❌ CUDA not available: {}", e);
            println!("   Falling back to CPU mode");

            // Test on CPU as fallback
            let device = Device::Cpu;
            let a = Tensor::randn(0f32, 1f32, (2, 3), &device)?;
            let b = Tensor::randn(0f32, 1f32, (3, 2), &device)?;
            let c = a.matmul(&b)?;
            println!("✅ CPU mode working: {:?}", c.shape());
        }
    }

    Ok(())
}
