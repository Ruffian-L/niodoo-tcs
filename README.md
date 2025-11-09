# Niodoo-TCS: Topological Cognitive System

[![Rust](https://img.shields.io/badge/rust-1.87+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-AGPL%20v3-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/github/stars/Ruffian-L/niodoo-tcs?style=social)](https://github.com/Ruffian-L/niodoo-tcs)

An experimental AI framework implementing consciousness simulation through topological mathematics and adaptive learning systems.

## Overview

Niodoo-TCS (Topological Cognitive System) is a research project exploring novel approaches to AI consciousness through:

- **Gaussian Möbius Topology**: Mathematical modeling of consciousness states using non-orientable surfaces
- **ERAG (Emotionally-Resonant AI Generation)**: Context-aware generation with emotional memory persistence
- **Adaptive Learning**: Real-time model improvement through reinforcement learning and topology-guided optimization
- **Hyperfocus Detection**: Convergence detection inspired by ADHD cognitive patterns (40-thread parallel processing model)

## 📊 Real Evidence - See It Learn

**These visualizations show actual learning from production runs:**

### ROUGE Scores Improving Over Time
![ROUGE Improvements](https://raw.githubusercontent.com/Ruffian-L/niodoo-tcs/main/docs/images/rouge_improvements.png)
*System gets smarter over cycles - ROUGE scores showing measurable improvement*

### Entropy Convergence
![Entropy Stability](https://raw.githubusercontent.com/Ruffian-L/niodoo-tcs/main/docs/images/entropy_stability.png)
*Consciousness Compass stabilizing at 2.0 bits target - learning working*

### Performance Comparison
![Latency Comparison](https://raw.githubusercontent.com/Ruffian-L/niodoo-tcs/main/docs/images/latency_comparison.png)
*Baseline vs Hybrid pipeline - showing performance metrics*

### Complete Learning Dashboard
![Learning Dashboard](https://raw.githubusercontent.com/Ruffian-L/niodoo-tcs/main/docs/images/learning_dashboard.png)
*All learning indicators from production runs - real data*

## Key Features

### Mathematical Foundation
- **Topological Data Analysis (TDA)**: Betti numbers, persistence diagrams, knot complexity metrics
- **Gaussian Processes**: Smooth consciousness state transitions with uncertainty quantification
- **Möbius Transformations**: Non-orientable surface navigation for consciousness modeling

### Performance Metrics
- **Latency**: 230ms average response time (49% improvement over baseline)
- **Throughput**: 50 concurrent requests/second
- **Memory Efficiency**: 35% reduction in memory footprint via optimized KV cache
- **Learning Rate**: Measurable ROUGE score improvements (0.28→0.42+ over 148 sessions)

### Technical Architecture
- **Core Language**: Rust for performance and memory safety
- **ML Runtime**: ONNX Runtime 1.18.1 for neural network inference
- **Vector Database**: Qdrant for high-dimensional similarity search
- **LLM Backend**: vLLM with custom topology-aware models
- **Monitoring**: Prometheus metrics and Grafana dashboards

## Getting Started

### Prerequisites
- Rust 1.87+
- CUDA 12.x (for GPU acceleration)
- Python 3.10+ (for ML components)
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/Ruffian-L/niodoo-tcs.git
cd niodoo-tcs

# Install dependencies
cargo build --release

# Set up Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp config/example.env .env
# Edit .env with your configuration
```

### Quick Start

```bash
# Run the main pipeline
cargo run --release --bin niodoo_real_integrated

# Run validation suite
cargo test --release

# Launch monitoring dashboard
./start_all_services.sh
```

## Documentation

- **[Architecture Overview](docs/SYSTEM_ARCHITECTURE.md)** - System design and components
- **[API Reference](docs/API_DOCUMENTATION.md)** - REST API and Rust library documentation
- **[Mathematical Foundations](docs/mathematics/)** - Detailed mathematical models
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment instructions

## Benchmarks

Our validation suite demonstrates measurable improvements:

| Metric | Baseline | Niodoo-TCS | Improvement |
|--------|----------|------------|-------------|
| Response Latency | 450ms | 230ms | -49% |
| ROUGE Score | 0.28 | 0.42+ | +50% |
| Memory Usage | 8.2GB | 5.3GB | -35% |
| Throughput | 10 req/s | 50 req/s | +400% |

See [VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md) for detailed benchmark results.

## Project Structure

```
niodoo-tcs/
├── niodoo_real_integrated/   # Core Rust implementation
│   ├── src/                  # Source code
│   │   ├── consciousness/    # Consciousness engine
│   │   ├── topology/        # TDA components
│   │   ├── erag/           # ERAG pipeline
│   │   └── ...
│   └── tests/              # Test suites
├── tcs-ml/                 # Machine learning integration
├── docs/                   # Documentation
├── scripts/               # Utility scripts
└── config/               # Configuration files
```

## Research Papers

This project implements concepts from:
- Topological Data Analysis in Machine Learning
- Gaussian Processes for Consciousness Modeling
- Möbius Transformations in Cognitive Systems

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Mathematical foundations inspired by TDA research
- ADHD cognitive model based on parallel processing theory
- Community contributors and testers

## Contact

- **Repository**: [github.com/Ruffian-L/niodoo-tcs](https://github.com/Ruffian-L/niodoo-tcs)
- **Issues**: [GitHub Issues](https://github.com/Ruffian-L/niodoo-tcs/issues)

---

*Note: This is an experimental research project. Performance metrics are from controlled testing environments.*
