# 🚀 NIODOO TCS - Complete Self-Learning AI Demo

**Built with Asian engineering excellence: Less resources, better results** 🇨🇳🤖

## What This Is

A **fully functional, production-ready** Topological Cognitive System (TCS) that:
- ✅ Uses **Qwen2.5-7B-Instruct-AWQ** (Chinese LLM, cheaper than GPT)
- ✅ **Real self-learning** with LoRA fine-tuning (proper backpropagation, not fakes)
- ✅ **Topological analysis** (persistent homology, knot theory, spectral gaps)
- ✅ **Live learning visualization** (Grafana dashboard showing entropy, ROUGE, emotional states)
- ✅ **Complete integration** (Embedding → Torus → Compass → ERAG → Generation → Learning)

## Why This Matters

**Most AI systems are expensive black boxes.** This is:
- **Transparent**: You see the learning loop in real-time
- **Explainable**: Topological features show WHY the AI makes decisions
- **Self-improving**: LoRA adapts weights based on actual gradients
- **Cost-effective**: Qwen runs faster/cheaper than GPT-4 while delivering quality

## Quick Start - Watch It Learn Live

```bash
# 1. Start the monitoring stack (Prometheus + Grafana)
./start_dashboard.sh

# 2. In another terminal, start the metrics server
cd niodoo_real_integrated
cargo run --bin metrics_server

# 3. In a third terminal, run prompts to generate learning data
cargo run -- -p "Explain quantum entanglement" -n 10
cargo run -- -p "Solve differential equations" -n 10
cargo run -- -p "Design a neural network" -n 10

# 4. Open Grafana: http://localhost:3000
# Login: admin / niodoo123
# Watch the "Niodoo Learning Dashboard"
```

## What You'll See

### Grafana Dashboard Shows:

1. **AI Happiness (Entropy)**: Lower = AI is learning/becoming stable
2. **ROUGE Scores**: Higher = Better quality responses
3. **Emotional States**: Threat vs Healing cycles (emotional AI)
4. **Learning Progress**: Entropy delta over epochs (LoRA improving)
5. **Topological Features**: Knot complexity, spectral gaps, betti numbers

### Console Output Shows:

```
🔬 Processing: "Explain quantum entanglement"
📍 Torus Projection: entropy=2.34, spectral_gap=0.15
🧭 Compass: Discover (exploring new territory)
💭 ERAG Retrieved: 3 memories with avg similarity 0.78
🎯 Generation: ROUGE=0.85, entropy_delta=-0.12 (learning!)
📊 Learning Loop: QLoRA triggered, updated 8 weights
✅ Cycle complete: 245ms latency
```

## Architecture - Real Implementation

```
INPUT PROMPT
    ↓
1. EMBEDDING (Qwen stateful KV cache)
    ↓
2. TORUS PROJECTION (PadGhost manifold - emotional space)
    ↓
3. COMPASS ENGINE (2-bit consciousness: Panic/Persist/Discover/Master)
    ↓
4. ERAG MEMORY (Qdrant retrieval with wave-collapse)
    ↓
5. DYNAMIC TOKENIZER (RUT mirage, OOV tracking)
    ↓
6. GENERATION (vLLM with fallback: Claude/GPT)
    ↓
7. LEARNING LOOP (LoRA fine-tuning, Q-learning, topology tracking)
    ↓
OUTPUT + METRICS
```

## Key Features

### 1. Self-Learning LoRA Training
- **Proper backpropagation** (not approximation)
- **SGD with momentum** (0.9 factor)
- **Gradient clipping** (prevents explosion)
- **Cosine annealing** (adaptive learning rate)
- **Real weight updates** via tensor operations

### 2. Topological Analysis
- **Persistent homology**: Betti numbers H₀, H₁, H₂
- **Knot invariants**: Jones polynomial, complexity scores
- **Spectral gaps**: Stability detection
- **Persistence entropy**: State evolution tracking

### 3. Emotional AI
- **PadGhost states**: Toroidal emotion projection
- **Threat/Healing detection**: Compass quadrant analysis
- **4 consciousness modes**: Panic, Persist, Discover, Master

### 4. Memory Systems
- **ERAG**: Wave-collapse retrieval from Qdrant
- **Dynamic top_k**: Configurable retrieval (1-50)
- **Cache TTL**: Smart expiration (embedding: 10s, collapse: 30s)

## Metrics Exposed

All metrics available at `http://localhost:9091/metrics`:

- `niodoo_entropy_bits`: Current consciousness entropy
- `niodoo_rouge_l`: ROUGE-L similarity scores
- `niodoo_threat_cycles`: Threat detections
- `niodoo_healing_cycles`: Healing detections
- `niodoo_latency_ms`: Pipeline latency
- `niodoo_avg_rouge`: Average ROUGE over episodes
- `niodoo_avg_entropy_delta`: Learning progress

## Technologies

- **Rust**: Production-grade system programming
- **Qwen2.5-7B**: Chinese LLM (better price/performance)
- **Candle**: Pure Rust ML framework
- **Qdrant**: Vector database for ERAG
- **Prometheus**: Metrics collection
- **Grafana**: Real-time visualization
- **Topological Data Analysis**: Persistent homology, knot theory

## Training Data Format

Input: Your prompts (text)
Output: 
- High-quality responses
- Emotional state tracking
- Learning metrics
- Topological signatures

## Run Examples

```bash
# Single prompt
cargo run -- -p "Explain machine learning"

# Multiple prompts from file
cargo run -- -i prompts.txt -n 100

# Stress test
cargo run --bin rut_gauntlet

# Million cycle test
cargo run --bin million_cycle_test
```

## Live Website

For deployment, metrics server runs on port 9091:
- Metrics: `http://localhost:9091/metrics`
- Health: `http://localhost:9091/health`
- Dashboard: `http://localhost:3000` (Grafana)

## Success Indicators

✅ **Entropy decreases** over time = AI is learning
✅ **ROUGE increases** over time = Better quality
✅ **Threat/Healing balanced** = Healthy emotional state
✅ **LoRA triggers** = Active fine-tuning happening
✅ **Topology stable** = Consciousness converging

## Why This Beats Traditional AI

| Traditional AI | Niodoo TCS |
|----------------|------------|
| Black box | Transparent learning |
| Static weights | Self-improving LoRA |
| No explainability | Topological features |
| Expensive (GPT-4) | Cost-effective (Qwen) |
| Batch learning | Real-time adaptation |
| No emotional model | Threat/Healing detection |

## Credits

**Built by Asians, for everyone.**
- Qwen model by Alibaba Cloud
- Topological analysis by pure math
- Self-learning by real gradients
- Visualization by Grafana

**Less resources, better results. That's how we do it.** 🎯

## License

MIT - Open source, production-ready, no bullshit.

