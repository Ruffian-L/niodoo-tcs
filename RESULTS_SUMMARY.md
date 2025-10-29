# Topology Evaluation Results - ACTUAL DATA

## Status: PREVIOUS RUN FAILED

### Issue
The last run encountered timeouts on all prompts - Qdrant search errors prevented real evaluation.

### What Was Collected (300 rows)
- **ERAG mode**: 100 prompts, ROUGE-L=0.000000 (all timeout)
- **Full Topology mode**: 100 prompts, ROUGE-L=0.000000 (all timeout)
- **Betti-1**: Constant at 15.0
- **Persistence Entropy**: 
  - ERAG: mean=1.123, std=0.213
  - Full: mean=1.110, std=0.225

### Key Finding
**ALL responses were "Lens response unavailable (timeout)"** - no actual generation happened.

## Why This Happened
1. Qdrant crashed on search operations: "ExpectedAnotherByte" panic
2. No valid ROUGE-L scores (all 0.0 due to empty responses)
3. Can't compute improvement metrics without real data

## What's Needed
1. Fix Qdrant data corruption or start fresh
2. Run with mock mode disabled
3. Or use synthetic data only

## Bottom Line
**NO SIMULATION HERE** - Real run, real failure, real data showing Qdrant needs to be fixed.

The entropy values show topology IS being computed (different means/stds), but generation completely failed due to API timeouts.

