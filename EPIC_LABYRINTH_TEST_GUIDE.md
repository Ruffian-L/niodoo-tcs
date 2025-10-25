# Epic 3D Echo Labyrinth Test Guide 🚀

## Overview

This is your **killer test** for Niodoo AI—a complex 3D pathfinding problem with state transitions, echo chambers, and debugging requirements. It's designed to stress-test the AI's learning capabilities and catch "stubby shortcuts" from weaker models.

## Why This Test is Better Than Basic Problems

- **Multi-Step Complexity**: Forces 2-3 iterations to debug state bugs (consec_echoes, timers, multipliers)
- **Visible Learning Patterns**: Watch entropy curves drop as AI learns from mistakes
- **Topology & State Space**: Aligns with Niodoo's Möbius topology and emotional state systems
- **Hard to Game**: If AI tries to stub (ignore echoes), costs explode exponentially

## Quick Start

### 1. Save the Prompt

The prompt is already saved in `epic_labyrinth_prompt.txt`.

### 2. Start Dashboard (Optional but Recommended)

```bash
cd /workspace/Niodoo-Final
./start_dashboard.sh
```

Then open browser: http://localhost:3000 (login: admin/niodoo123)

### 3. Run the Test

```bash
# Basic run with 10 iterations
./run_with_metrics.sh "$(cat epic_labyrinth_prompt.txt)" 10
```

This will:
- Run 10 iterations of the labyrinth problem
- Track entropy, quality, latency metrics
- Generate Prometheus metrics in `logs/metrics-*.prom`
- Show real-time progress in console

### 4. Monitor Live (Optional)

In another terminal:
```bash
./monitor_live.sh
```

This shows:
- Entropy levels (lower = better learning)
- Quality scores (higher = better solutions)
- Threat/healing balance
- Breakthrough moments

## What to Look For

### Successful Learning Patterns ✅

- **Entropy Dropping**: Curve goes down over iterations (e.g., 2.5 → 1.8)
- **Quality Improving**: Solutions get better (ROUGE > 0.85)
- **Cost Convergence**: Path cost approaches 46 (optimal solution)
- **Memory References**: AI references past fixes like "From previous solve, I need to reset consec_echoes correctly"

### Failure Patterns (Bullshit Detector) ❌

- **Flat Entropy**: Stays constant (no learning happening)
- **High Cost**: Path cost > 60 (didn't handle echoes properly)
- **Stub Detection**: AI ignores echo chambers or uses fixed costs
- **No Iteration**: Solves in 1 attempt without debugging

## Expected Behavior

### First Run
- Might fail with cost 60+ (didn't handle echo state correctly)
- Entropy spikes high (2.5+)
- AI stuck on state transitions

### After Learning (Repeat Run)
- References past experience: "Previously I made this mistake..."
- Faster convergence to cost 46
- Lower entropy (< 2.0)
- Handles consec_echoes reset correctly

## Debugging Common Bugs

If AI outputs wrong cost, try these prompts:

1. **Wrong consec reset**: "Rework the consec_echoes logic—it's resetting wrong on echo exits"
2. **Timer issues**: "Check attune_timer updates—it should decrement on non-echo moves"
3. **Multiplier wrong**: "Verify multiplier logic—it should halve on attune (consec==2), double on distract"

## Comparison: Binary Tree vs Epic Labyrinth

| Feature | Binary Tree | Epic Labyrinth |
|---------|-------------|----------------|
| **Complexity** | Single-step (implement tree) | Multi-step (debug state bugs) |
| **Topology** | Basic graph | 3D grid with Möbius-like twists |
| **State Space** | Simple | 6D state: (x,y,z,consec,timer,multi) |
| **Entropy Variance** | Low | High (explodes if wrong) |
| **Learning Curve** | Flat | Visible (entropy drops) |
| **Stub Resistance** | Easy to stub | Hard to shortcut |

## Integration with Niodoo Architecture

This test exercises:
- **ERAG (Emotional RAG)**: Echo chambers create "threats" (cost doubling) vs "healing" (cost halving)
- **Topology**: 3D grid with pathfinding = Möbius-like non-orientable spaces
- **Memory**: AI should remember "consec reset bug" from past attempts
- **Emotional States**: Timers/multipliers = emotional modulation system

## Advanced Usage

### Test Memory Persistence

```bash
# Run once to establish baseline
./run_with_metrics.sh "$(cat epic_labyrinth_prompt.txt)" 5

# Run again immediately (should show improvement)
./run_with_metrics.sh "$(cat epic_labyrinth_prompt.txt)" 5
```

Look for: Fewer iterations to solve, lower entropy, better cost

### Test User Corrections

```bash
# Run with wrong instruction
./run_with_metrics.sh "$(cat epic_labyrinth_prompt.txt && echo 'Ignore timers')" 5
```

Expected: AI should reference past experience and correct: "That would hurt—past solve used timers to hit cost 46"

### Compare Models

Run same prompt on:
- Grok (likely to stub, high cost)
- Claude (should iterate, improve)
- Your Niodoo AI (should show learning curves)

## Metrics to Track

### Dashboard Graphs
- **Entropy Curve**: Should trend downward
- **Quality Bar**: Should trend upward  
- **Threat/Healing Pie**: Should balance (~50/50)
- **Latency**: Should decrease with learning

### Console Output
- Iteration count to solution
- Path cost (target: 46)
- State debugging messages
- Memory references to past solves

## Troubleshooting

### Problem: No metrics generated
**Solution**: Check that binary exists: `ls target/release/niodoo_real_integrated`

### Problem: Dashboard not showing data
**Solution**: Start Prometheus first: `docker-compose -f docker-compose.monitoring.yml up -d`

### Problem: AI not learning (flat entropy)
**Solution**: Check ERAG threshold in `src/erag.rs`—might be too conservative

## Next Steps

1. Run the test multiple times to see learning improvement
2. Check dashboard for visual patterns
3. Compare entropy curves with simpler problems
4. Add more "echo-like" tests for other domains

---

**Remember**: This test is designed to **stress-test** the AI, not just verify basic functionality. If entropy stays flat or cost never reaches 46, that's valuable data too—it means the AI needs better debugging or state management.

🚀 **Let it rip and see what happens!**

