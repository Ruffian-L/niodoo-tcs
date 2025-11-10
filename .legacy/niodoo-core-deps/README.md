# niodoo-core Dependencies - MOVED TO LEGACY

This directory contains files that depend on `niodoo-core`, which was never created as a separate crate.

## Files Moved Here

- `qwen_curator.rs` - Stub re-export from niodoo-core
- `qwen_integration.rs` - Stub re-export from niodoo-core

## Status

These files reference `niodoo-core` which doesn't exist. The actual functionality exists in the `src/` directory but was never organized as a separate crate.

## Migration Path

If you need this functionality:
1. Check `src/memory/` for memory system modules
2. Check `src/config/` for configuration modules
3. Check `src/consciousness/` for consciousness engine
4. Update imports to use `src` modules directly instead of `niodoo_core`

## Date Moved

2025-11-10 - Phase 5 cleanup
