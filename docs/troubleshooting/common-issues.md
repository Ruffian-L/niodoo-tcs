# Common Issues and Solutions

**Created by Jason Van Pham | Niodoo Framework | 2025**

## 🚨 Troubleshooting Guide for Niodoo Consciousness Engine

This guide provides solutions to common issues encountered when using the Niodoo Consciousness Engine, including installation problems, runtime errors, performance issues, and configuration problems.

## 📋 Table of Contents

- [Installation Issues](#installation-issues)
- [Build and Compilation Errors](#build-and-compilation-errors)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Memory Management Problems](#memory-management-problems)
- [GPU Acceleration Issues](#gpu-acceleration-issues)
- [Configuration Problems](#configuration-problems)
- [Network and Connectivity Issues](#network-and-connectivity-issues)
- [Debugging Techniques](#debugging-techniques)

## 🛠️ Installation Issues

### Issue: Rust Installation Problems

**Symptoms:**
- `cargo: command not found`
- `rustc: command not found`
- Build fails with "Rust not found" error

**Solutions:**

1. **Install Rust properly:**
   ```bash
   # Download and install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Reload shell environment
   source ~/.cargo/env
   
   # Verify installation
   rustc --version
   cargo --version
   ```

2. **Update Rust if needed:**
   ```bash
   rustup update stable
   rustup default stable
   ```

3. **Install required components:**
   ```bash
   rustup component add rustfmt clippy rust-src
   ```

### Issue: System Dependencies Missing

**Symptoms:**
- Build fails with "package not found" errors
- Linking errors during compilation
- Missing system libraries

**Solutions:**

1. **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install build-essential pkg-config libssl-dev
   sudo apt install cmake git curl
   ```

2. **CentOS/RHEL:**
   ```bash
   sudo yum groupinstall "Development Tools"
   sudo yum install cmake git curl openssl-devel
   ```

3. **macOS:**
   ```bash
   # Install Xcode command line tools
   xcode-select --install
   
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install dependencies
   brew install cmake git curl openssl
   ```

### Issue: Permission Denied Errors

**Symptoms:**
- `Permission denied` when running commands
- Cannot write to target directory
- Access denied errors

**Solutions:**

1. **Fix directory permissions:**
   ```bash
   # Make sure you own the project directory
   sudo chown -R $USER:$USER /path/to/niodoo-feeling
   
   # Set proper permissions
   chmod -R 755 /path/to/niodoo-feeling
   ```

2. **Run with proper user:**
   ```bash
   # Don't use sudo for cargo commands
   cargo build --release
   
   # If you need to install globally, use cargo install
   cargo install --path .
   ```

## 🔨 Build and Compilation Errors

### Issue: Cargo Build Failures

**Symptoms:**
- `cargo build` fails with compilation errors
- Dependency resolution issues
- Linking errors

**Solutions:**

1. **Clean and rebuild:**
   ```bash
   # Clean build artifacts
   cargo clean
   
   # Remove Cargo.lock if needed
   rm Cargo.lock
   
   # Rebuild
   cargo build --release
   ```

2. **Update dependencies:**
   ```bash
   # Update all dependencies
   cargo update
   
   # Check for outdated dependencies
   cargo outdated
   ```

3. **Check Rust version compatibility:**
   ```bash
   # Ensure you're using a compatible Rust version
   rustc --version
   
   # Update if needed
   rustup update stable
   ```

### Issue: Dependency Conflicts

**Symptoms:**
- Version conflicts between dependencies
- `failed to select a version` errors
- Circular dependency issues

**Solutions:**

1. **Resolve version conflicts:**
   ```bash
   # Update Cargo.toml with specific versions
   # Example:
   [dependencies]
   tokio = "1.0"
   serde = "1.0"
   ```

2. **Use cargo tree to analyze dependencies:**
   ```bash
   # Install cargo-tree if not available
   cargo install cargo-tree
   
   # Analyze dependency tree
   cargo tree
   ```

3. **Check for duplicate dependencies:**
   ```bash
   # Look for duplicate dependencies
   cargo tree --duplicates
   ```

### Issue: Feature Flag Problems

**Symptoms:**
- `feature not found` errors
- Optional dependencies not available
- Conditional compilation failures

**Solutions:**

1. **Enable required features:**
   ```bash
   # Build with specific features
   cargo build --features "phase6,gpu,analytics"
   
   # Build with all features
   cargo build --all-features
   ```

2. **Check feature availability:**
   ```bash
   # List available features
   cargo metadata --format-version 1 | jq '.packages[].features'
   ```

3. **Update Cargo.toml:**
   ```toml
   [features]
   default = ["phase6", "gpu"]
   phase6 = []
   gpu = []
   analytics = []
   ```

## ⚡ Runtime Errors

### Issue: Consciousness Engine Initialization Failures

**Symptoms:**
- `Failed to initialize consciousness engine`
- `Brain coordination failed`
- `Memory manager initialization error`

**Solutions:**

1. **Check system resources:**
   ```bash
   # Check available memory
   free -h
   
   # Check disk space
   df -h
   
   # Check CPU usage
   top
   ```

2. **Verify configuration:**
   ```bash
   # Check configuration file
   cat config/consciousness_config.yaml
   
   # Validate configuration
   cargo run --bin config-validator
   ```

3. **Enable debug logging:**
   ```bash
   # Set debug log level
   export RUST_LOG=debug
   
   # Run with verbose output
   cargo run -- --verbose
   ```

### Issue: Brain Processing Timeouts

**Symptoms:**
- `Brain processing timeout`
- `Consensus generation failed`
- `Parallel processing error`

**Solutions:**

1. **Increase timeout values:**
   ```rust
   // In your code
   let timeout_duration = Duration::from_secs(10); // Increase from 5 to 10
   
   // Or in configuration
   brain_timeout: 10000  # milliseconds
   ```

2. **Check system performance:**
   ```bash
   # Monitor CPU usage
   htop
   
   # Check for high load
   uptime
   
   # Monitor memory usage
   free -h
   ```

3. **Optimize brain processing:**
   ```rust
   // Reduce parallel processing load
   let max_concurrent = 2; // Reduce from default
   
   // Use sequential processing for debugging
   let use_parallel = false;
   ```

### Issue: Memory Management Errors

**Symptoms:**
- `Memory allocation failed`
- `Out of memory` errors
- `Memory consolidation failed`

**Solutions:**

1. **Check memory limits:**
   ```bash
   # Check system memory
   free -h
   
   # Check process memory usage
   ps aux | grep niodoo
   
   # Check memory limits
   ulimit -a
   ```

2. **Adjust memory configuration:**
   ```yaml
   # In consciousness_config.yaml
   memory:
     max_spheres: 5000  # Reduce from default
     consolidation_threshold: 0.8
     cleanup_interval: 300  # seconds
   ```

3. **Enable memory monitoring:**
   ```rust
   // Enable memory monitoring
   let memory_monitor = MemoryMonitor::new()
       .with_threshold(0.8)  // 80% memory usage threshold
       .with_cleanup_interval(Duration::from_secs(300));
   ```

## 🚀 Performance Issues

### Issue: Slow Processing Times

**Symptoms:**
- High latency in responses
- Slow memory operations
- Poor throughput

**Solutions:**

1. **Enable GPU acceleration:**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Enable GPU features
   cargo build --features "gpu"
   
   # Set GPU environment variables
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **Optimize configuration:**
   ```yaml
   # In consciousness_config.yaml
   performance:
     gpu_acceleration: true
     memory_optimization: true
     latency_target: 2000  # milliseconds
     batch_size: 32
   ```

3. **Profile performance:**
   ```bash
   # Install profiling tools
   cargo install cargo-flamegraph
   
   # Profile the application
   cargo flamegraph --bin niodoo
   ```

### Issue: High CPU Usage

**Symptoms:**
- CPU usage consistently high
- System becomes unresponsive
- Performance degradation

**Solutions:**

1. **Check for infinite loops:**
   ```rust
   // Add loop counters
   let mut loop_count = 0;
   while condition {
       loop_count += 1;
       if loop_count > 1000 {
           warn!("Potential infinite loop detected");
           break;
       }
   }
   ```

2. **Optimize algorithms:**
   ```rust
   // Use more efficient data structures
   use std::collections::HashMap;
   use std::collections::HashSet;
   
   // Cache expensive calculations
   let mut cache = HashMap::new();
   ```

3. **Reduce processing load:**
   ```yaml
   # In consciousness_config.yaml
   processing:
     max_concurrent_tasks: 4  # Reduce from default
     batch_processing: true
     adaptive_batching: true
   ```

### Issue: Memory Leaks

**Symptoms:**
- Memory usage continuously increases
- System becomes slow over time
- Out of memory errors

**Solutions:**

1. **Enable memory leak detection:**
   ```bash
   # Install memory profiling tools
   cargo install cargo-valgrind
   
   # Run with memory leak detection
   cargo valgrind --bin niodoo
   ```

2. **Check for circular references:**
   ```rust
   // Use weak references where appropriate
   use std::rc::{Rc, Weak};
   
   struct Node {
       parent: Option<Weak<Node>>,
       children: Vec<Rc<Node>>,
   }
   ```

3. **Implement proper cleanup:**
   ```rust
   impl Drop for MemoryManager {
       fn drop(&mut self) {
           // Clean up resources
           self.cleanup_memory();
           self.close_connections();
       }
   }
   ```

## 💾 Memory Management Problems

### Issue: Gaussian Memory Sphere Issues

**Symptoms:**
- `Memory sphere creation failed`
- `Invalid memory position`
- `Memory consolidation error`

**Solutions:**

1. **Validate memory parameters:**
   ```rust
   // Check position validity
   fn validate_position(position: [f32; 3]) -> bool {
       position.iter().all(|&x| x.is_finite() && x.abs() < 1000.0)
   }
   
   // Check emotional valence range
   fn validate_valence(valence: f32) -> bool {
       valence >= -1.0 && valence <= 1.0
   }
   ```

2. **Handle memory consolidation:**
   ```rust
   // Implement safe consolidation
   pub fn consolidate_memories(&mut self) -> Result<()> {
       let mut to_consolidate = Vec::new();
       
       // Find pairs to consolidate
       for (id1, sphere1) in &self.spheres {
           for (id2, sphere2) in &self.spheres {
               if id1 != id2 && self.should_consolidate(sphere1, sphere2) {
                   to_consolidate.push((id1.clone(), id2.clone()));
               }
           }
       }
       
       // Perform consolidation safely
       for (id1, id2) in to_consolidate {
           self.consolidate_pair(&id1, &id2)?;
       }
       
       Ok(())
   }
   ```

3. **Monitor memory usage:**
   ```rust
   // Add memory monitoring
   pub fn monitor_memory_usage(&self) -> MemoryStats {
       MemoryStats {
           total_spheres: self.spheres.len(),
           memory_usage_mb: self.calculate_memory_usage(),
           fragmentation: self.calculate_fragmentation(),
       }
   }
   ```

### Issue: Memory Access Patterns

**Symptoms:**
- Slow memory retrieval
- Inefficient memory access
- Poor memory locality

**Solutions:**

1. **Optimize memory layout:**
   ```rust
   // Use cache-friendly data structures
   #[repr(C)]
   struct MemorySphere {
       position: [f32; 3],
       emotional_valence: f32,
       access_count: u32,
       last_accessed: u64,
   }
   ```

2. **Implement memory prefetching:**
   ```rust
   // Prefetch related memories
   pub fn prefetch_related_memories(&self, memory_id: &str) {
       if let Some(memory) = self.spheres.get(memory_id) {
           for link in &memory.links {
               if let Some(related) = self.spheres.get(&link.target_id) {
                   // Prefetch related memory
                   self.prefetch_memory(related);
               }
           }
       }
   }
   ```

3. **Use memory pools:**
   ```rust
   // Implement memory pooling
   pub struct MemoryPool {
       available_spheres: Vec<GaussianMemorySphere>,
       in_use_spheres: HashMap<String, GaussianMemorySphere>,
   }
   
   impl MemoryPool {
       pub fn allocate(&mut self) -> Option<GaussianMemorySphere> {
           self.available_spheres.pop()
       }
       
       pub fn deallocate(&mut self, sphere: GaussianMemorySphere) {
           self.available_spheres.push(sphere);
       }
   }
   ```

## 🎮 GPU Acceleration Issues

### Issue: CUDA Installation Problems

**Symptoms:**
- `CUDA not found` errors
- GPU acceleration disabled
- CUDA runtime errors

**Solutions:**

1. **Install CUDA properly:**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Install CUDA toolkit
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

2. **Set CUDA environment variables:**
   ```bash
   # Add to ~/.bashrc
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

3. **Verify CUDA installation:**
   ```bash
   # Check CUDA version
   nvcc --version
   
   # Test CUDA compilation
   cd /usr/local/cuda/samples/1_Utilities/deviceQuery
   make
   ./deviceQuery
   ```

### Issue: GPU Memory Issues

**Symptoms:**
- `Out of GPU memory` errors
- GPU memory allocation failures
- CUDA out of memory

**Solutions:**

1. **Monitor GPU memory:**
   ```bash
   # Check GPU memory usage
   nvidia-smi
   
   # Monitor GPU memory in real-time
   watch -n 1 nvidia-smi
   ```

2. **Optimize GPU memory usage:**
   ```rust
   // Implement GPU memory pooling
   pub struct GpuMemoryPool {
       available_memory: Vec<GpuBuffer>,
       memory_limit: usize,
   }
   
   impl GpuMemoryPool {
       pub fn allocate(&mut self, size: usize) -> Result<GpuBuffer> {
           if self.available_memory.len() > 0 {
               Ok(self.available_memory.pop().unwrap())
           } else {
               self.create_new_buffer(size)
           }
       }
   }
   ```

3. **Reduce GPU memory requirements:**
   ```yaml
   # In consciousness_config.yaml
   gpu:
     memory_limit_mb: 2048  # Reduce from default
     batch_size: 16  # Reduce batch size
     enable_memory_pooling: true
   ```

### Issue: GPU Performance Problems

**Symptoms:**
- Slow GPU processing
- GPU utilization low
- CUDA kernel failures

**Solutions:**

1. **Profile GPU performance:**
   ```bash
   # Install NVIDIA profiling tools
   sudo apt install nvidia-profiler
   
   # Profile GPU usage
   nvprof ./niodoo
   ```

2. **Optimize CUDA kernels:**
   ```rust
   // Use efficient CUDA kernels
   pub fn optimize_cuda_kernel(&self) -> Result<()> {
       // Set optimal block and grid sizes
       let block_size = 256;
       let grid_size = (self.data_size + block_size - 1) / block_size;
       
       // Launch optimized kernel
       self.launch_kernel(grid_size, block_size)?;
       
       Ok(())
   }
   ```

3. **Enable GPU optimizations:**
   ```yaml
   # In consciousness_config.yaml
   gpu:
     optimization_level: 3  # Maximum optimization
     enable_tensor_cores: true
     enable_mixed_precision: true
   ```

## ⚙️ Configuration Problems

### Issue: Configuration File Errors

**Symptoms:**
- `Configuration parsing failed`
- `Invalid configuration value`
- `Configuration file not found`

**Solutions:**

1. **Validate configuration format:**
   ```bash
   # Check YAML syntax
   python -c "import yaml; yaml.safe_load(open('config/consciousness_config.yaml'))"
   
   # Use configuration validator
   cargo run --bin config-validator -- config/consciousness_config.yaml
   ```

2. **Check configuration values:**
   ```yaml
   # Example valid configuration
   consciousness:
     emotional_sensitivity: 0.8  # Must be 0.0-1.0
     memory_capacity: 10000      # Must be positive integer
     brain_timeout: 5000         # Must be positive integer (ms)
   
   memory:
     max_spheres: 5000           # Must be positive integer
     consolidation_threshold: 0.8 # Must be 0.0-1.0
   
   gpu:
     acceleration: true          # Must be boolean
     memory_limit_mb: 4096       # Must be positive integer
   ```

3. **Use default configuration:**
   ```bash
   # Generate default configuration
   cargo run --bin config-generator -- --output config/default_config.yaml
   
   # Use default configuration
   cargo run -- --config config/default_config.yaml
   ```

### Issue: Environment Variable Problems

**Symptoms:**
- Environment variables not recognized
- Configuration not loaded
- Default values used instead

**Solutions:**

1. **Set environment variables:**
   ```bash
   # Set required environment variables
   export NIODO_EMOTIONAL_SENSITIVITY=0.8
   export NIODO_MEMORY_CAPACITY=10000
   export NIODO_GPU_ACCELERATION=true
   export RUST_LOG=info
   ```

2. **Check environment variable loading:**
   ```rust
   // In your code
   let emotional_sensitivity = std::env::var("NIODO_EMOTIONAL_SENSITIVITY")
       .unwrap_or_else(|_| "0.8".to_string())
       .parse::<f32>()
       .unwrap_or(0.8);
   ```

3. **Use configuration file instead:**
   ```yaml
   # In consciousness_config.yaml
   consciousness:
     emotional_sensitivity: 0.8
     memory_capacity: 10000
     gpu_acceleration: true
   ```

## 🌐 Network and Connectivity Issues

### Issue: API Connection Problems

**Symptoms:**
- `Connection refused` errors
- `Timeout` errors
- `Network unreachable` errors

**Solutions:**

1. **Check network connectivity:**
   ```bash
   # Test network connectivity
   ping google.com
   
   # Check DNS resolution
   nslookup api.niodoo.com
   
   # Test specific port
   telnet api.niodoo.com 443
   ```

2. **Configure proxy settings:**
   ```bash
   # Set proxy environment variables
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   export NO_PROXY=localhost,127.0.0.1
   ```

3. **Use local configuration:**
   ```yaml
   # In consciousness_config.yaml
   network:
     api_endpoint: "http://localhost:8080"  # Use local endpoint
     timeout: 30000  # Increase timeout
     retry_attempts: 3
   ```

### Issue: WebSocket Connection Issues

**Symptoms:**
- WebSocket connection failures
- Real-time updates not working
- Connection drops frequently

**Solutions:**

1. **Check WebSocket configuration:**
   ```yaml
   # In consciousness_config.yaml
   websocket:
     enabled: true
     url: "ws://localhost:8080/ws"
     reconnect_interval: 5000  # milliseconds
     max_reconnect_attempts: 10
   ```

2. **Implement connection recovery:**
   ```rust
   // Implement WebSocket reconnection
   pub async fn connect_with_retry(&mut self) -> Result<()> {
       let mut attempts = 0;
       let max_attempts = 10;
       
       while attempts < max_attempts {
           match self.connect().await {
               Ok(_) => return Ok(()),
               Err(e) => {
                   attempts += 1;
                   if attempts >= max_attempts {
                       return Err(e);
                   }
                   tokio::time::sleep(Duration::from_millis(5000)).await;
               }
           }
       }
       
       Err(anyhow::anyhow!("Failed to connect after {} attempts", max_attempts))
   }
   ```

3. **Monitor connection health:**
   ```rust
   // Implement connection health monitoring
   pub async fn monitor_connection_health(&self) {
       let mut interval = tokio::time::interval(Duration::from_secs(30));
       
       loop {
           interval.tick().await;
           
           if !self.is_connected().await {
               warn!("WebSocket connection lost, attempting reconnection");
               if let Err(e) = self.reconnect().await {
                   error!("Failed to reconnect: {}", e);
               }
           }
       }
   }
   ```

## 🔍 Debugging Techniques

### Issue: General Debugging

**Symptoms:**
- Unexpected behavior
- Errors without clear cause
- Performance issues

**Solutions:**

1. **Enable comprehensive logging:**
   ```bash
   # Set detailed log levels
   export RUST_LOG=debug
   export RUST_BACKTRACE=1
   
   # Run with debug output
   cargo run -- --verbose --debug
   ```

2. **Use debugging tools:**
   ```bash
   # Install debugging tools
   cargo install cargo-expand
   cargo install cargo-udeps
   
   # Expand macros for debugging
   cargo expand
   
   # Check for unused dependencies
   cargo udeps
   ```

3. **Implement debug assertions:**
   ```rust
   // Add debug assertions
   debug_assert!(emotional_valence >= -1.0 && emotional_valence <= 1.0);
   debug_assert!(position.iter().all(|&x| x.is_finite()));
   debug_assert!(memory_count > 0);
   ```

### Issue: Memory Debugging

**Symptoms:**
- Memory leaks
- Invalid memory access
- Memory corruption

**Solutions:**

1. **Use memory debugging tools:**
   ```bash
   # Install Valgrind
   sudo apt install valgrind
   
   # Run with memory debugging
   valgrind --leak-check=full --show-leak-kinds=all ./target/release/niodoo
   ```

2. **Implement memory guards:**
   ```rust
   // Add memory guards
   pub struct MemoryGuard {
       data: Vec<u8>,
       canary: u64,
   }
   
   impl MemoryGuard {
       pub fn new(size: usize) -> Self {
           Self {
               data: vec![0; size],
               canary: 0xDEADBEEF,
           }
       }
       
       pub fn check_integrity(&self) -> bool {
           self.canary == 0xDEADBEEF
       }
   }
   ```

3. **Monitor memory usage:**
   ```rust
   // Implement memory monitoring
   pub fn monitor_memory_usage(&self) {
       let memory_info = self.get_memory_info();
       if memory_info.usage_percentage > 0.8 {
           warn!("High memory usage: {:.1}%", memory_info.usage_percentage);
       }
   }
   ```

### Issue: Performance Debugging

**Symptoms:**
- Slow performance
- High CPU usage
- Poor scalability

**Solutions:**

1. **Profile performance:**
   ```bash
   # Install profiling tools
   cargo install cargo-flamegraph
   cargo install cargo-criterion
   
   # Generate flamegraph
   cargo flamegraph --bin niodoo
   
   # Run benchmarks
   cargo bench
   ```

2. **Implement performance monitoring:**
   ```rust
   // Add performance monitoring
   pub struct PerformanceMonitor {
       start_time: Instant,
       operation_count: u64,
   }
   
   impl PerformanceMonitor {
       pub fn record_operation(&mut self, duration: Duration) {
           self.operation_count += 1;
           if duration > Duration::from_millis(100) {
               warn!("Slow operation: {:?}", duration);
           }
       }
   }
   ```

3. **Use performance counters:**
   ```rust
   // Implement performance counters
   pub struct PerformanceCounters {
       pub processing_time: Duration,
       pub memory_usage: usize,
       pub cpu_usage: f32,
       pub gpu_usage: f32,
   }
   
   impl PerformanceCounters {
       pub fn log_performance(&self) {
           info!("Performance: CPU: {:.1}%, GPU: {:.1}%, Memory: {}MB, Time: {:?}",
               self.cpu_usage, self.gpu_usage, self.memory_usage / 1024 / 1024, self.processing_time);
       }
   }
   ```

## 📚 Additional Resources

### Documentation
- [Getting Started Guide](../user-guides/getting-started.md) - Basic setup and usage
- [Developer Guide](../user-guides/developer-guide.md) - Advanced development techniques
- [API Reference](../api/) - Comprehensive API documentation
- [Architecture Documentation](../architecture/) - System architecture details

### Community Support
- [GitHub Issues](https://github.com/niodoo/niodoo-feeling/issues) - Report bugs and issues
- [GitHub Discussions](https://github.com/niodoo/niodoo-feeling/discussions) - Community discussions
- [Discord Community](https://discord.gg/niodoo) - Real-time community support
- [Stack Overflow](https://stackoverflow.com/questions/tagged/niodoo) - Q&A platform

### Tools and Utilities
- [Configuration Validator](../../tools/config-validator.rs) - Validate configuration files
- [Performance Profiler](../../tools/performance-profiler.rs) - Performance analysis tool
- [Memory Monitor](../../tools/memory-monitor.rs) - Memory usage monitoring
- [Debug Helper](../../tools/debug-helper.rs) - Debugging utilities

---

*This troubleshooting guide provides solutions to common issues encountered with the Niodoo Consciousness Engine. For additional help, refer to the community resources or create an issue on GitHub.*