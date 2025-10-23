# FIX-7: HTTP 429 Rate Limit Detection and Handling

**Date**: October 22, 2025
**File Modified**: `src/api_clients.rs`
**Status**: ✅ COMPLETED

## Overview

Implemented comprehensive HTTP 429 (Too Many Requests) rate limit detection and handling for both Claude and GPT API clients. The solution respects the `Retry-After` header from server responses and implements smart retry logic.

## Changes Made

### 1. **Imports Updated** (Lines 5-9)
Added `StatusCode` from `reqwest` and `info` from `tracing` for better rate limit handling:
```rust
use reqwest::{Client, StatusCode};
use tracing::{warn, debug, info};
```

### 2. **New Rate Limit Configuration** (Lines 15)
Added `MAX_RETRY_AFTER_SECS` constant to cap maximum wait time based on Retry-After header:
```rust
const MAX_RETRY_AFTER_SECS: u64 = 60; // Max time to wait based on Retry-After header
```

### 3. **Retry-After Header Parser** (Lines 17-31)
New `parse_retry_after()` function to extract wait time from the Retry-After header:
- **Supports two formats**:
  - Numeric seconds (e.g., "60") - **most common**
  - HTTP date format (e.g., "Wed, 21 Oct 2025 07:28:00 GMT") - defaults to 5 seconds
- **Safety cap**: Limits max wait to 60 seconds to prevent excessive delays
- **Graceful fallback**: Returns sensible defaults if parsing fails

**Function signature**:
```rust
fn parse_retry_after(header_value: &str) -> Option<Duration>
```

### 4. **Enhanced Retry Loop** (Lines 33-83)
Updated `execute_with_retry()` to detect and log 429 errors:
- Checks error message for "429" indicator
- Logs rate limit detection at INFO level for visibility
- Integrates with existing exponential backoff mechanism

### 5. **Claude API Client - 429 Detection** (Lines 143-162)
Added explicit 429 handling after `.send()` call:
```rust
// Handle 429 rate limit responses
if response.status() == StatusCode::TOO_MANY_REQUESTS {
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_retry_after);

    let wait_duration = retry_after.unwrap_or_else(|| Duration::from_secs(1));
    warn!(
        "Claude API rate limited (429). Retry-After: {:?}",
        wait_duration
    );
    info!(
        "Sleeping for {:?} before retrying Claude API request",
        wait_duration
    );
    tokio::time::sleep(wait_duration).await;
    anyhow::bail!("Claude API rate limited (429). Retrying...");
}
```

**Key behaviors**:
- Detects `StatusCode::TOO_MANY_REQUESTS` (429)
- Extracts and parses `Retry-After` header
- Sleeps for the specified duration
- Triggers retry via `anyhow::bail!()` which propagates to the retry loop
- Logs both warning and info messages for monitoring

### 6. **GPT API Client - 429 Detection** (Lines 253-272)
Identical 429 handling implementation for OpenAI GPT client:
```rust
// Handle 429 rate limit responses
if response.status() == StatusCode::TOO_MANY_REQUESTS {
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_retry_after);

    let wait_duration = retry_after.unwrap_or_else(|| Duration::from_secs(1));
    warn!(
        "GPT API rate limited (429). Retry-After: {:?}",
        wait_duration
    );
    info!(
        "Sleeping for {:?} before retrying GPT API request",
        wait_duration
    );
    tokio::time::sleep(wait_duration).await;
    anyhow::bail!("GPT API rate limited (429). Retrying...");
}
```

## Control Flow Diagram

```
API Request
    ↓
.send() completes
    ↓
Check status == 429?
    ├─ YES → Extract Retry-After header
    │         ├─ Parse as seconds
    │         ├─ Cap at 60 seconds
    │         └─ Default to 5 seconds if date format
    │         ↓
    │         Sleep for duration
    │         ↓
    │         Trigger retry via bail!()
    │         ↓
    │         execute_with_retry() catches error
    │         ↓
    │         Log and retry (up to 3 attempts)
    │
    └─ NO → Check if response.is_success()?
             ├─ YES → Parse response and return
             └─ NO → Bail with error, trigger exponential backoff retry
```

## Retry Strategy

### For Rate Limited (429) Responses:
1. **First 429**: Wait for Retry-After header value (default 1 second)
2. **Second 429**: Wait for Retry-After header value
3. **Third 429**: Wait for Retry-After header value
4. **Total attempts**: 3 (same as other errors)

### For Other Errors:
1. **First error**: Wait 100ms
2. **Second error**: Wait 1 second (100ms × 10)
3. **Third error**: Wait 10 seconds (1s × 10)

## Safety Features

✅ **Retry-After Parsing**
- Handles both numeric seconds and HTTP date formats
- Caps maximum wait to 60 seconds
- Graceful fallback to 1 second default

✅ **Rate Limit Detection**
- Explicit `StatusCode::TOO_MANY_REQUESTS` check
- Prevents misclassification as other 4xx errors
- Executed BEFORE general error checking

✅ **Logging**
- WARN level: Rate limit detected with proposed wait time
- INFO level: Actual sleep duration before retry
- ERROR tracking: Failed after all retries

✅ **No Request Loss**
- 429 responses trigger automatic retry
- Respects server's Retry-After guidance
- Maintains request context across retries

## Testing Scenarios

### Scenario 1: Single 429 followed by success
```
Request 1 → 429 (Retry-After: 2) → Sleep 2s → Retry
Request 2 → 200 OK → Return result
```

### Scenario 2: Multiple 429s then success
```
Request 1 → 429 (Retry-After: 1) → Sleep 1s → Retry
Request 2 → 429 (Retry-After: 3) → Sleep 3s → Retry
Request 3 → 200 OK → Return result
```

### Scenario 3: Persistent 429 (rate limit exceeded)
```
Request 1 → 429 (Retry-After: 5) → Sleep 5s → Retry
Request 2 → 429 (Retry-After: 10) → Sleep 10s → Retry
Request 3 → 429 (Retry-After: 30) → Sleep 30s → Retry
Return Error: "API rate limited (429). Retrying..." (after 3 attempts)
```

### Scenario 4: 429 with capped Retry-After
```
Request 1 → 429 (Retry-After: 300) → Capped to 60s → Sleep 60s → Retry
Request 2 → 200 OK → Return result
```

## Integration Points

### Claude API
- **Endpoint**: `https://api.anthropic.com/v1/messages`
- **Rate limit status**: 429 response
- **Retry-After header**: Respected and parsed
- **Auth header**: `x-api-key`

### GPT API
- **Endpoint**: `https://api.openai.com/v1/chat/completions`
- **Rate limit status**: 429 response
- **Retry-After header**: Respected and parsed
- **Auth header**: `Authorization: Bearer {api_key}`

## Performance Impact

- **Minimal overhead**: Only activated on 429 responses
- **Memory safe**: No additional allocations in success path
- **CPU efficient**: Simple string parsing and Duration calculation
- **Network friendly**: Respects server's rate limit guidance

## Error Messages

When rate limited, logs will show:

```
WARN: Claude API rate limited (429). Retry-After: 2s
INFO: Sleeping for 2s before retrying Claude API request
DEBUG: API request attempt 2/3
[After sleep] ... retries request ...
INFO: API request succeeded on attempt 2
```

## Future Enhancements

- Add metrics/counters for 429 occurrences
- Implement circuit breaker pattern for persistent rate limits
- Add configurable max wait time via environment variable
- Support additional header formats (delta-seconds, etc.)

## Verification Checklist

- ✅ Imports added for `StatusCode` and `info` logging
- ✅ `parse_retry_after()` function handles seconds and date formats
- ✅ `MAX_RETRY_AFTER_SECS` safety cap implemented
- ✅ Claude API client detects 429 and handles Retry-After
- ✅ GPT API client detects 429 and handles Retry-After
- ✅ Both clients sleep before retrying
- ✅ Error message triggers retry in `execute_with_retry()` loop
- ✅ Logging at both WARN and INFO levels
- ✅ No breaking changes to existing API

## Code Statistics

| Metric | Count |
|--------|-------|
| New function: `parse_retry_after()` | 1 |
| New constant: `MAX_RETRY_AFTER_SECS` | 1 |
| 429 checks added (Claude + GPT) | 2 |
| Lines of code added | ~65 |
| Backward compatible | ✅ Yes |

