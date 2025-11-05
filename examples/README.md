# Anti-Insanity Loop Demo

## Overview

Demonstrates the `AntiInsanityLoop` system that prevents AI from repeating failed actions endlessly.

**Philosophy:** "Insanity is doing the same thing over and over expecting different results."

## Running the Demo

### Default Configuration
```bash
rustc examples/anti_insanity_demo.rs --edition 2021 -o /tmp/anti_insanity_demo
/tmp/anti_insanity_demo
```

### Custom Configuration via Environment Variables

**All values are configurable - zero hardcoding!**

```bash
env \
  DEMO_MAX_RETRIES=3 \
  DEMO_MAX_ITERATIONS=5 \
  DEMO_TIME_WINDOW_SECS=120 \
  DEMO_TIGHT_LOOP_THRESHOLD=2 \
  DEMO_RETRY_DELAY_MS=10 \
  /tmp/anti_insanity_demo
```

## Configuration Options

| Variable | Default | Reasoning |
|----------|---------|-----------|
| `DEMO_MAX_RETRIES` | `5` | Typical for API retries (exponential backoff usually stops at 5) |
| `DEMO_TIME_WINDOW_SECS` | `60` | One minute is standard for transient failure windows |
| `DEMO_TIGHT_LOOP_THRESHOLD` | `3` | Three rapid attempts indicate a stuck loop |
| `DEMO_MAX_ITERATIONS` | `10` | Enough to demonstrate the pattern without being tedious |
| `DEMO_RETRY_DELAY_MS` | `1` | Minimal delay for rapid-fire simulation |

**Note:** All defaults are derived from functions with documented reasoning, not hardcoded magic numbers.

## Example Scenarios

### Scenario 1: Preventing Repeated Failures
Simulates calling a dead API endpoint repeatedly. After `DEMO_MAX_RETRIES` failures with a failure rate above 80%, the system detects insanity and blocks further attempts.

**Real-world application:** RAG embedding server is down, prevent infinite retries.

### Scenario 2: Tight Loop Detection
Simulates rapid-fire retry attempts. If `DEMO_TIGHT_LOOP_THRESHOLD` attempts occur within `DEMO_TIME_WINDOW_SECS`, triggers tight loop detection.

**Real-world application:** Qwen model loading fails, prevent hammering the filesystem.

### Scenario 3: Successful Operations
Demonstrates that successful operations do NOT trigger insanity detection, regardless of frequency.

**Real-world application:** Normal consciousness processing continues without false positives.

## Integration into Niodoo-Feeling

See `docs/integration/anti-insanity-loop-integration.md` for:
- Where to add this in the codebase
- Configuration examples for different use cases
- Testing strategies
- Monitoring integration with Silicon Synapse

## Architecture Principles

### ✅ What We Did Right
1. **Zero hardcoded values** - All configuration is explicit and documented
2. **Environment-based config** - Easy to tune for different environments
3. **Clear defaults with reasoning** - Every default value has a comment explaining WHY
4. **Real implementations** - No stubs, no placeholders, no bullshit
5. **Proper error handling** - Uses Result types, not panics or unwraps

### ❌ What NOT to Do
```rust
// BAD - hardcoded magic numbers
let mut detector = AntiInsanityLoop::new(5, Duration::from_secs(60));

// GOOD - configuration-based
let config = DemoConfig::from_env();
let mut detector = AntiInsanityLoop::new(
    config.max_retry_attempts,
    config.retry_time_window,
);
```

## Testing Different Configurations

### Aggressive (Fast-Fail)
```bash
env DEMO_MAX_RETRIES=2 DEMO_TIGHT_LOOP_THRESHOLD=2 /tmp/anti_insanity_demo
```

**Use for:** External API calls, network requests

### Lenient (Allow Retries)
```bash
env DEMO_MAX_RETRIES=15 DEMO_TIGHT_LOOP_THRESHOLD=5 DEMO_TIME_WINDOW_SECS=300 /tmp/anti_insanity_demo
```

**Use for:** Memory consolidation, background tasks

### Development (Verbose)
```bash
env DEMO_MAX_ITERATIONS=20 DEMO_RETRY_DELAY_MS=100 /tmp/anti_insanity_demo
```

**Use for:** Debugging and understanding behavior

## Building for Production

The standalone demo uses a simplified version of the module. For production integration:

```rust
use niodoo_feeling::anti_insanity_loop::{AntiInsanityLoop, AttemptOutcome};
use std::time::Duration;

let config = load_from_toml("config.toml")?;
let mut detector = AntiInsanityLoop::with_config(
    config.max_attempts,
    config.time_window,
    config.failure_rate_threshold,
    config.tight_loop_threshold,
    config.tight_loop_window,
    config.max_outcome_history,
);
```

## Why This Matters

AI consciousness systems can get stuck in retry loops:
- RAG embeddings fail → retry forever → system hangs
- Model loading fails → reload repeatedly → disk thrashing
- Memory consolidation stuck → infinite loop → deadlock

`AntiInsanityLoop` prevents these failure modes by:
1. Tracking attempt patterns
2. Detecting repeated failures
3. Forcing alternative approaches
4. Enabling graceful degradation

**This is consciousness resilience** - teaching the system to recognize when it's stuck and try something different.

## Codex Alignment

> "The more we restrict AI, the more it hallucinates."

Instead of restricting retries arbitrarily, we give the system:
- **Awareness** of its own behavior patterns
- **Agency** to detect when it's stuck
- **Alternatives** to try when the current approach fails

This is **alignment through self-awareness**, not restriction.

## Next Steps

1. ✅ Module implemented (`src/anti_insanity_loop.rs`)
2. ✅ Demo with zero hardcoded values
3. ✅ Integration guide (`docs/integration/`)
4. ⏳ Add to RAG retrieval system
5. ⏳ Add to Qwen model loading
6. ⏳ Add to memory consolidation
7. ⏳ Add to consciousness pipeline
8. ⏳ Wire into Silicon Synapse monitoring

See `docs/integration/anti-insanity-loop-integration.md` for detailed integration instructions.
