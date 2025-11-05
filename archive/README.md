# Archive Directory

This directory contains code that has been archived but kept for reference purposes.

## Archived Files

### Backup Files (*.full)

These are backup copies of files that were created during development:

- `config.rs.full` - Backup of `config.rs` (dated Oct 31 18:56)
- `learning.rs.full` - Backup of `learning.rs` (dated Oct 31 18:56)
- `pipeline.rs.full` - Backup of `pipeline.rs` (dated Oct 31 20:00)

**Reason for Archival**: These are backup files created during refactoring. The current versions are in `niodoo_real_integrated/src/` and are actively used.

**Status**: Keep for reference, but not loaded by the build system.

## Why Archive Instead of Delete?

These files are kept for:
1. Reference during debugging
2. Historical context
3. Potential rollback scenarios
4. Understanding evolution of codebase

## Additional Archived Code

### Alternative Implementations

- `pipeline_v2/` - Alternative pipeline implementation (archived 2025-01-31)
  - Status: Confirmed unused - not imported in lib.rs
  - Contains unique cache prefetching logic that might be useful for reference
  - Location: Was in `niodoo_real_integrated/src/pipeline_v2/`

- `config_v2/` - Alternative config system (archived 2025-01-31)
  - Status: Confirmed unused - not imported in lib.rs  
  - Location: Was in `niodoo_real_integrated/src/config_v2/`

See `DEAD_CODE_ANALYSIS.md` for complete verification results.

### Not Archived (But Documented)

- `pipeline_legacy.rs` - Commented out in lib.rs, kept for reference
- `consciousness_engine/` - Used by other crates (niodoo-core), not dead code
- `cpp-qt-brain-integration/` - Separate C++ project, not Rust code
- `curator_executor/` - Separate system with unique features, keep as-is

