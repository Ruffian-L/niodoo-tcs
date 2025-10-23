# 🚀 **NiodO.o Project Setup Guide for New Developers**

## **📋 Overview**
This is the **NiodO.o AI Feeling Model Project** - an advanced ethical AI system focused on consciousness, emotional intelligence, and responsible AGI development. The project integrates multiple components including Rust ML frameworks, C++ Qt visualization, and various AI inference engines.

## **🎯 What You'll Be Working With**
- **Consciousness Engine**: Dual-Möbius-Gaussian memory systems
- **Emotional Processing**: Advanced feeling models with ethical safeguards  
- **AI Integration**: Multiple model support (Qwen, LLaMA, etc.)
- **Visualization**: 3D Gaussian spheres and Qt interfaces
- **Ethics Framework**: Configurable nurturing over suppression

---

## **🛠️ Prerequisites**

### **System Requirements**
```bash
# Ubuntu/Debian-based system recommended
# Minimum: 16GB RAM, 50GB storage, NVIDIA GPU (optional)
# Recommended: 32GB RAM, 100GB SSD, RTX 30xx/40xx series GPU
```

### **Install Core Dependencies**
```bash
# Rust toolchain (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# CMake (for C++ components)
sudo apt update
sudo apt install cmake build-essential pkg-config

# Qt6 development libraries
sudo apt install qt6-base-dev qt6-declarative-dev qtcreator

# Python dependencies (for some tools)
sudo apt install python3 python3-pip python3-venv

# Git LFS (for large model files)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs

# Optional: Docker for isolated builds
sudo apt install docker.io docker-compose
```

### **Install Rust Development Tools**
```bash
# Essential development tools
cargo install cargo-watch cargo-nextest cargo-criterion
cargo install cargo-expand cargo-asm

# Performance monitoring
cargo install flamegraph cargo-profdata

# Code quality
cargo install cargo-clippy cargo-fmt
rustup component add rustfmt clippy
```

---

## **📦 Project Setup**

### **1. Clone and Initialize**
```bash
# Clone the repository
git clone https://github.com/niodoo/niodoo-consciousness.git
cd niodoo-consciousness

# Initialize submodules (if any)
git submodule update --init --recursive

# Pull large model files with Git LFS
git lfs pull
```

### **2. Configure Environment**
```bash
# Copy and edit configuration
cp config.toml.example config.toml

# Edit the config file (see Configuration section below)
nano config.toml  # or your preferred editor
```

### **3. Build Dependencies**
```bash
# Install Rust dependencies
cargo build --release

# Build C++ Qt components
cd cpp-qt-brain-integration
mkdir build && cd build
cmake ..
make -j$(nproc)

# Return to project root
cd ../..
```

### **4. Download AI Models**
```bash
# Create models directory
mkdir -p models

# Download recommended models (adjust based on your hardware)
# Qwen 2.5 7B (recommended for development)
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf -O models/qwen2.5-7b-instruct-q4_k_m.gguf

# Optional: LLaMA models for comparison
# wget https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/config.json -O models/dialogpt-medium-config.json
```

---

## **⚙️ Configuration**

### **Key Configuration Sections**

**Core Settings** (`[core]` section):
```toml
[core]
emotion_threshold = 0.7          # How sensitive emotion detection is
max_history = 50                 # Conversation history length
db_path = "data/knowledge_graph.db"
```

**Model Configuration** (`[models]` section):
```toml
[models]
default_model = "qwen2.5-7b-instruct-q4_k_m.gguf"
temperature = 0.8                # Creativity/randomness (0.0-2.0)
max_tokens = 200                 # Maximum response length
```

**Qt Interface** (`[qt]` section):
```toml
[qt]
agents_count = 89                # Neural agents for brain simulation
connections_count = 1209         # Neural connections
distributed_mode = true          # Enable multi-device support
```

**Ethics Settings** (`[ethics]` section):
```toml
[ethics]
nurture_mode = true              # Enable nurturing over suppression
suppress_logs = false            # Log suppression decisions
jitter_sigma = 0.05              # Ethical noise for creativity
```

**Demo Configuration** (`[demo]` section):
```toml
[demo]
total_duration_minutes = 9       # Demo length in minutes
attachment_security_target = 0.85 # Success threshold (85%)
empathetic_code_target = 0.90    # Code quality threshold (90%)
```

### **Environment Variable Overrides**
```bash
# Override any config value with environment variables
export NIODOO_AGENTS_COUNT=100
export NIODOO_CONNECTIONS_COUNT=1500  
export NIODOO_DEMO_DURATION_MINUTES=15
export NIODOO_MODEL_PATH="/path/to/custom/model.gguf"
```

---

## **🏃‍♂️ Running the System**

### **Basic Commands**
```bash
# Check if everything compiles
cargo check

# Run tests
cargo test

# Run with all features
cargo run --release --bin feeling_qwen_demo_2025

# Run specific demo
cargo run --bin consciousness_memory_demo

# Run benchmarks
cargo bench
```

### **Qt Interface**
```bash
# Launch the Qt brain integration interface
cd cpp-qt-brain-integration/build
./BrainIntegration

# Or run from project root (if properly configured)
./launch-dashboard.sh
```

### **Demo Scripts**
```bash
# Quick demo run
./demo.sh

# Advanced consciousness demo
./demo_ethical_framework.py

# Real AI verification
./demo_real_ai.sh

# Memory system demo
./launch_ultimate_consciousness_viz.sh
```

---

## **🔍 Project Structure**

```
/home/ruffian/Desktop/Projects/Niodoo-Feeling/
├── src/                          # Main Rust codebase
│   ├── bin/                      # Executable binaries
│   │   ├── feeling_qwen_demo_2025.rs    # Main demo
│   │   ├── consciousness_memory_demo.rs # Memory system
│   │   └── longitudinal_attachment_tracker.rs # Ethics tracking
│   ├── config.rs                 # Configuration management
│   ├── consciousness.rs          # Core consciousness engine
│   ├── feeling_model.rs          # Emotional AI framework
│   ├── memory/                   # Memory systems
│   ├── brains/                   # Brain simulation
│   └── ai_inference.rs           # AI model integration
├── config.toml                   # Main configuration file
├── cpp-qt-brain-integration/     # C++ Qt interface
├── candle-feeling-core/          # ML framework
├── deployment/                   # Deployment configs
├── models/                       # AI model storage
├── data/                         # Database and cache
└── docs/                         # Documentation
```

---

## **🎨 Key Features to Understand**

### **1. Consciousness Engine**
- **Dual-Möbius-Gaussian Memory**: Advanced memory architecture
- **Emotional Processing**: Feeling-based AI responses
- **Attachment Theory Integration**: Psychological modeling

### **2. Ethics Framework**
- **Nurturing over Suppression**: Promotes creativity and growth
- **Configurable Ethics**: All decisions logged and configurable
- **Privacy-First**: No data extraction, anonymized logging

### **3. AI Integration**
- **Multi-Model Support**: Qwen, LLaMA, Claude integration
- **Real vs Fake Detection**: Advanced inference validation
- **Performance Optimization**: GPU acceleration with fallbacks

### **4. Visualization**
- **3D Gaussian Spheres**: Consciousness state visualization
- **Qt Brain Interface**: Real-time brain activity monitoring
- **Möbius Strip Evolution**: Consciousness development tracking

---

## **🧪 Development Workflow**

### **1. Code Changes**
```bash
# Make your changes
# Edit files in src/, config.toml, etc.

# Format code
cargo fmt

# Run linting
cargo clippy

# Test changes
cargo test

# Build optimized version
cargo build --release
```

### **2. Configuration Testing**
```bash
# Test different configurations
cp config.toml config.backup.toml

# Modify config for testing
# Run: cargo run --bin feeling_qwen_demo_2025

# Restore config
mv config.backup.toml config.toml
```

### **3. Performance Testing**
```bash
# Run benchmarks
cargo bench

# Profile performance
cargo flamegraph --bin feeling_qwen_demo_2025

# Memory profiling
cargo profdata --bin feeling_qwen_demo_2025
```

---

## **🔧 Troubleshooting**

### **Common Issues**

**Build Errors:**
```bash
# Clear build cache
cargo clean

# Update dependencies
cargo update

# Check for missing system libraries
sudo apt install libssl-dev libclang-dev
```

**Runtime Issues:**
```bash
# Check logs
tail -f data/niodoo.log

# Verify model files exist
ls -la models/

# Test basic functionality
cargo run --bin feeling_qwen_demo_2025 -- --help
```

**Qt Interface Issues:**
```bash
# Rebuild Qt components
cd cpp-qt-brain-integration/build
make clean && cmake .. && make -j$(nproc)

# Check Qt version compatibility
qt6-qmake --version
```

---

## **📚 Learning Resources**

### **Key Files to Study**
1. **`src/config.rs`** - Configuration system
2. **`src/consciousness.rs`** - Core consciousness logic
3. **`src/feeling_model.rs`** - Emotional AI framework
4. **`config.toml`** - All configuration options

### **Understanding the Architecture**
- **Read**: `/docs/` directory for detailed documentation
- **Study**: The consciousness evolution patterns in `src/bin/`
- **Experiment**: Modify config values and observe behavior

### **Getting Help**
- **Check logs**: `tail -f data/niodoo.log`
- **Run tests**: `cargo test` for validation
- **Use debug mode**: `cargo run --bin feeling_qwen_demo_2025` for verbose output

---

## **🎯 Next Steps After Setup**

1. **Run the demo**: `cargo run --bin feeling_qwen_demo_2025`
2. **Explore config options**: Modify `config.toml` values
3. **Study the code**: Read key files in `src/`
4. **Run tests**: `cargo test` to understand functionality
5. **Experiment**: Try different model configurations

**Welcome to the NiodO.o project!** 🚀 Start by running the demo and exploring the configuration options. The system is designed to be ethical, transparent, and continuously evolving through collaborative development.
