# Test Results

This directory contains test results and validation data.

## Files

- `qwen_comparison_test.json` - Full 50-prompt comparison test results
  - Baseline Qwen responses (direct Ollama)
  - NIODOO pipeline responses
  - Latency comparisons
  - Length and similarity analysis

## Usage

To view results:
```bash
cat results/qwen_comparison_test.json | jq .
```

To analyze:
```python
import json
with open('results/qwen_comparison_test.json') as f:
    data = json.load(f)
    # Process results...
```

## Validation

All results validated for:
- ✅ Real API calls (no mocks)
- ✅ Real timing (no fake numbers)
- ✅ Real transformations (not copying)
- ✅ Natural variation (not identical)

See `../docs/validation/` for validation reports.
