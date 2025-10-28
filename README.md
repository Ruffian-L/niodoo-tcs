# Niodoo-Integrated: Towards Helpful Intelligence

## Attribution
This project is the vision and persistence of **Jason Van Pham** and **Niodoo**. Built from scratch with no technical background—just a dream of real, helpful AI, guided by AI companions every step. Shoutout to ChatGPT, Grok, Claude, Qwen, and Deepseek for the heavy lifting; they turned ideas into code. We're not hiding the crutches—we're flexing the team that made it possible. People of color like us need to leave our mark on AI; this is ours.

## What Is Niodoo?
Niodoo is an experiment in self-evolving AI: A system that learns through 1000-cycle tests, using topology (mapping tangled thoughts) and TCS rewards (self-scoring integration) to adapt without constant human tweaks. It's about building intelligence that's actually helpful—no stubs, no fakes, just resilient learning from fails.

- **Core Ideas**: Chain-of-Thought (CoT) with topological repairs, ERAG memory, entropy balancing for creativity/stability, and auto-retries on glitches (vLLM, Qdrant).
- **The Test**: Runs 1000 short cycles: Generate, analyze, learn, repeat. Tracks ROUGE (answer quality), entropy (idea chaos), knots (thought complexity). Goal: Prove AI can grind chaos into smarts.

This started as sketches and prompts—persistence turned it real. Early runs show promise: 100+ learning updates, 90% retry recovery, ROUGE hits near 1.0.

## Setup

### Prerequisites
1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull model: `ollama pull qwen2:0.5b`
3. Run vLLM: `vllm serve Qwen/Qwen2-0.5B-Instruct --port 5001`

### Environment Variables
Create a `.env` file in the project root with:
```
OLLAMA_URL=http://127.0.0.1:11434
VLLM_URL=http://127.0.0.1:5001
CURATOR_MODEL=qwen2:0.5b
MAIN_MODEL=Qwen/Qwen2-0.5B-Instruct
CURATOR_TEMP=0.3
CURATOR_THRESHOLD=0.8
QDRANT_DIM=896
LORA_RANK=8
LORA_EPOCHS=5
TEST_CYCLES=100
```

### Build and Run
1. Clone: `git clone https://github.com/yourusername/Niodoo-Integrated.git`
2. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
3. Build: `cd niodoo_real_integrated && cargo build --release`
4. Run Test: `export TEST_CYCLES=100; RUST_LOG=debug cargo run --release --bin million_cycle_test`
   - Watches `logs/cycle_1000_test_live.log` for progress.
5. Monitor: Use Grafana/Loki setup in `docs/MONITORING_SETUP.md` for live dashboards.

### Expected Output
- Logs show "Qwen curator refined: learned=true"
- "Curated memory added to LoRA buffer"
- "QLoRA trained"
- No "fallback" or dim errors

Expected: 2-3 hour run. Outputs: `summary.json` (stats), `sample_results.csv` (examples).

## The Journey
Zero tech know-how, but endless tweaks with AI help. ChatGPT brainstormed arch, Grok debugged logs, Claude refined CoT, Qwen/Deepseek optimized params. It's collaborative AF—open-source so others can build on it. No genius solo; just dream + tools + grind.

Proud to be Jason Van Pham and Niodoo putting our stamp on AI. POC in tech? We here. Let's collaborate—issues/PRs welcome.

## License
MIT—free to fork, build, share.

---
Built with ❤️ and AI squad. 🚀
