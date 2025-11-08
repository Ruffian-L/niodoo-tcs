# Legacy Module Migration Guide

## Production vs Research Code Paths

This document clarifies which code paths are production-ready vs research/experimental.

### Production Paths (Use These)

- **`niodoo_real_integrated`**: Production-ready integrated pipeline
  - Location: `niodoo_real_integrated/src/`
  - Status: ✅ Production-ready, fully tested
  - Features: Complete pipeline with Phase 2 integration

- **`niodoo-core`**: Core production modules
  - Location: `niodoo-core/src/`
  - Status: ✅ Production-ready
  - Features: Memory systems, token promotion, topology

- **`tcs-*`**: Topological Cognitive System modules
  - Status: ✅ Production-ready
  - Features: Core TCS functionality

### Legacy/Research Paths (Deprecated - Migration in Progress)

- **`src/`**: Legacy monolithic implementation
  - Location: `src/`
  - Status: ⚠️ **DEPRECATED** - Transitional, use `niodoo_real_integrated` instead
  - Migration: Modules being migrated to `niodoo_real_integrated` gradually
  - **Do not use for new development**

### Migration Status

The following modules from `src/` are being migrated:

1. ✅ **Memory Systems**: Migrated to `niodoo-core/src/memory/`
2. ✅ **Token Promotion**: Migrated to `niodoo-core/src/token_promotion/`
3. ✅ **Pipeline**: Migrated to `niodoo_real_integrated/src/pipeline.rs`
4. ✅ **Topology Analysis**: Migrated to `niodoo_real_integrated/src/tcs_analysis.rs`
5. ✅ **ERAG**: Migrated to `niodoo_real_integrated/src/erag.rs`
6. ✅ **Generation**: Migrated to `niodoo_real_integrated/src/generation.rs`
7. ✅ **Learning**: Migrated to `niodoo_real_integrated/src/learning.rs`
8. ✅ **Phase 2 Integration**: Implemented in `niodoo_real_integrated/src/`

### Using Production Code

Always import from production paths:

```rust
// ✅ CORRECT - Production paths
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_core::memory::EmotionalVector;
use niodoo_core::token_promotion::TokenPromotionEngine;

// ❌ WRONG - Legacy paths (deprecated)
// use src::pipeline::Pipeline;  // DON'T USE
```

### Deprecation Timeline

- **2025-Q1**: `src/` modules marked deprecated
- **2025-Q2**: Critical modules migrated to production paths
- **2025-Q3**: Legacy `src/` marked for removal

