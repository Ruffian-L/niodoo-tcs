# Niodoo-Integrated: Towards Helpful Intelligence

## Attribution
This project is the vision and persistence of **Jason Van Pham** and **Niodoo**. Built from scratch with no technical background—just a dream of real, helpful AI, guided by AI companions every step. Shoutout to ChatGPT, Grok, Claude, Qwen, and Deepseek for the heavy lifting; they turned ideas into code. We're not hiding the crutches—we're flexing the team that made it possible. People of color like us need to leave our mark on AI; this is ours.

## What Is Niodoo?
Niodoo is an experiment in self-evolving AI: A system that learns through 1000-cycle tests, using topology (mapping tangled thoughts) and TCS rewards (self-scoring integration) to adapt without constant human tweaks. It's about building intelligence that's actually helpful—no stubs, no fakes, just resilient learning from fails.

- **Core Ideas**: Chain-of-Thought (CoT) with topological repairs, ERAG memory, entropy balancing for creativity/stability, and auto-retries on glitches (vLLM, Qdrant).
- **The Test**: Runs 1000 short cycles: Generate, analyze, learn, repeat. Tracks ROUGE (answer quality), entropy (idea chaos), knots (thought complexity). Goal: Prove AI can grind chaos into smarts.

This started as sketches and prompts—persistence turned it real. Early runs show promise: 100+ learning updates, 90% retry recovery, ROUGE hits near 1.0.

## Setup (Simple Run)
1. Clone: `git clone https://github.com/yourusername/Niodoo-Integrated.git`
2. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
3. Build/Run Test: `cd src && cargo build --release && cargo run --release --bin niodoo_test`
   - Watches `logs/cycle_1000_test_live.log` for progress.
4. Monitor: Use Grafana/Loki setup in `docs/MONITORING_SETUP.md` for live dashboards.

Expected: 2-3 hour run. Outputs: `summary.json` (stats), `sample_results.csv` (examples).

## The Journey
Zero tech know-how, but endless tweaks with AI help. ChatGPT brainstormed arch, Grok debugged logs, Claude refined CoT, Qwen/Deepseek optimized params. It's collaborative AF—open-source so others can build on it. No genius solo; just dream + tools + grind.

Proud to be Jason Van Pham and Niodoo putting our stamp on AI. POC in tech? We here. Let's collaborate—issues/PRs welcome.

## License
MIT—free to fork, build, share.

---
Built with ❤️ and AI squad. 🚀
