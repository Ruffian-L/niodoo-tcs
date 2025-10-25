# Implementation Plan: Bullshit Buster MVP

## Overview

This implementation plan breaks down the Bullshit Buster MVP into discrete, manageable coding tasks. Each task builds incrementally on previous work, prioritizing early validation of core functionality.

**Key Principle:** Reuse existing Gen 1 components. Only write new code for parsing and CLI interface.

---

## Task List

- [ ] 1. Project setup and module structure
  - Create `src/bbuster/` module directory
  - Add new dependencies to `Cargo.toml` (clap, syn, walkdir, colored)
  - Create module hierarchy (parser, topology_mapper, detectors, emotional, report, cli)
  - Add `bbuster` binary target to `Cargo.toml`
  - _Requirements: All requirements (foundation)_

- [ ] 2. Implement basic code parser
  - [ ] 2.1 Create `CodeParser` struct with `syn` integration
    - Implement `parse_file()` method using `syn::parse_file()`
    - Extract functions, constants, structs from AST
    - Preserve line numbers for reporting
    - Handle parse errors gracefully with `Result<CodeAst, ParseError>`
    - _Requirements: 1.1, 1.2, 8.3_
  
  - [ ] 2.2 Add directory traversal support
    - Implement `parse_directory()` using `walkdir` crate
    - Filter for `.rs` files only
    - Collect all parsed ASTs into `Vec<CodeAst>`
    - Skip files that fail to parse, log warnings
    - _Requirements: 1.3, 8.1, 8.2_
  
  - [ ] 2.3 Create unit tests for parser
    - Test parsing valid Rust file
    - Test handling syntax errors
    - Test directory traversal
    - Test filtering non-Rust files
    - _Requirements: 1.1, 1.2, 1.3_

- [ ] 3. Implement topology mapper (wrapper around existing system)
  - [ ] 3.1 Create `TopologyMapper` struct
    - Initialize with reference to existing `DualMobiusGaussian`
    - Implement `map_to_topology()` to convert AST to topology representation
    - Map functions to nodes, calls to edges
    - Generate `GaussianParameters` for each node
    - _Requirements: 2.1, 2.5, 5.2_
  
  - [ ] 3.2 Implement k-flip perspectives
    - Implement `apply_k_flips()` using existing `mobius_torus_k_twist.rs`
    - Generate "optimistic" perspective (low variance)
    - Generate "pessimistic" perspective (high variance)
    - Return `Vec<TopologyPerspective>` with confidence scores
    - _Requirements: 2.2, 2.5_
  
  - [ ] 3.3 Add topology analysis tests
    - Test mapping simple function to topology
    - Test k-flip generation
    - Verify integration with existing Möbius engine
    - Test Gaussian parameter generation
    - _Requirements: 2.1, 2.2, 2.5_

- [ ] 4. Implement bullshit detector registry
  - [ ] 4.1 Create `BullshitDetector` trait
    - Define trait methods: `name()`, `detect()`, `confidence_threshold()`
    - Create `DetectorRegistry` struct
    - Implement `register()` and `run_all()` methods
    - Use Rayon for parallel detector execution
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [ ] 4.2 Implement `HardcodedValueDetector`
    - Detect numeric literals in code
    - Use Gaussian variance to identify anomalies
    - Generate `DetectedIssue` with confidence score
    - Suggest dynamic derivation as fix
    - _Requirements: 1.4, 2.1, 2.4_
  
  - [ ] 4.3 Implement `PlaceholderDetector`
    - Search for TODO, FIXME, unimplemented!, panic! patterns
    - Mark as high severity
    - Extract surrounding context
    - _Requirements: 1.4, 4.2_
  
  - [ ] 4.4 Implement `DeadCodeDetector`
    - Identify unused functions (no callers)
    - Identify unreachable code paths
    - Use topology disconnected paths
    - _Requirements: 1.4, 4.2_
  
  - [ ] 4.5 Add detector tests
    - Test each detector independently
    - Test registry parallel execution
    - Test detector failure handling
    - Verify confidence scores
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 5. Implement emotional analyzer (wrapper around existing feeling model)
  - [ ] 5.1 Create `EmotionalAnalyzer` struct
    - Initialize with reference to existing `FeelingModel`
    - Implement `analyze_emotion()` method
    - Map Gaussian variance to emotional states
    - High variance → Anger (unstable code)
    - Security issues → Fear
    - High complexity → Confusion
    - _Requirements: 3.1, 3.2, 5.3_
  
  - [ ] 5.2 Add emotional classification logic
    - Use existing feeling model activation patterns
    - Compute confidence scores for each emotion
    - Return `EmotionalState` enum with confidence
    - Handle missing model gracefully (skip emotional overlay)
    - _Requirements: 3.3, 3.4, 8.5_
  
  - [ ] 5.3 Add emotional analyzer tests
    - Test emotion classification for different issue types
    - Test confidence score calculation
    - Test fallback when model unavailable
    - Verify integration with existing feeling model
    - _Requirements: 3.1, 3.2, 3.3, 5.3_

- [ ] 6. Implement report generator
  - [ ] 6.1 Create `ReportGenerator` struct
    - Support `ReportFormat::Terminal` and `ReportFormat::Json`
    - Implement `generate()` method for formatting
    - Use `colored` crate for terminal output
    - Add emoji indicators for visual clarity
    - _Requirements: 4.1, 4.2, 4.4_
  
  - [ ] 6.2 Implement health score calculation
    - Formula: `100 - (critical * 5 + high * 2 + medium * 0.5)`
    - Clamp to [0, 100] range
    - Display as percentage with emoji
    - _Requirements: 4.3_
  
  - [ ] 6.3 Add report formatting tests
    - Test terminal output format
    - Test JSON output format
    - Test health score calculation
    - Verify emoji and color rendering
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 7. Implement CLI interface
  - [ ] 7.1 Create CLI structure with `clap`
    - Define `Cli` struct with `Parser` derive
    - Define `Commands` enum with `Scan` command
    - Add flags: `--topo-flip`, `--emotional`, `--format`, `--k-flips`
    - Implement `--help` output
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [ ] 7.2 Implement scan command handler
    - Parse command-line arguments
    - Validate target path exists
    - Call parser, topology mapper, detectors, emotional analyzer
    - Generate and display report
    - Return appropriate exit code
    - _Requirements: 6.2, 6.3, 6.4, 6.6_
  
  - [ ] 7.3 Add CLI integration tests
    - Test `bbuster scan <file>`
    - Test `bbuster scan --topo-flip <file>`
    - Test `bbuster scan --emotional <file>`
    - Test `bbuster scan --json <file>`
    - Test error handling for invalid paths
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 8. Add performance optimizations
  - [ ] 8.1 Implement parallel file processing
    - Use Rayon to parse files in parallel
    - Use Rayon to run detectors in parallel
    - Batch GPU operations for efficiency
    - _Requirements: 7.3, 5.2_
  
  - [ ] 8.2 Add GPU acceleration integration
    - Use existing `src/gpu_acceleration.rs` for Gaussian computations
    - Offload topology transformations to GPU
    - Batch process multiple files on GPU
    - _Requirements: 7.5, 5.2_
  
  - [ ] 8.3 Add performance benchmarks
    - Benchmark single file scan (<1000 lines)
    - Benchmark directory scan (<100 files)
    - Verify <2s for single file
    - Verify <30s for directory
    - Monitor memory usage (<1GB)
    - _Requirements: 7.1, 7.2, 7.4_

- [ ] 9. Add error handling and logging
  - [ ] 9.1 Implement error types
    - Create `BbusterError` enum with `thiserror`
    - Add variants for parse, IO, topology, analysis errors
    - Implement `From` conversions for common error types
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [ ] 9.2 Add user-friendly error messages
    - Format errors with context and suggestions
    - Display file path and line number for parse errors
    - Suggest fixes for common issues
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [ ] 9.3 Add logging infrastructure
    - Use `tracing` crate for structured logging
    - Log warnings for skipped files
    - Log info for analysis progress
    - Log debug for topology transformations
    - _Requirements: 8.4, 8.5_

- [ ] 10. Create documentation and examples
  - [ ] 10.1 Write README.md
    - Installation instructions
    - Quick start guide
    - Command reference with examples
    - Example outputs with screenshots
    - _Requirements: 10.2_
  
  - [ ] 10.2 Add rustdoc comments
    - Document all public APIs
    - Add module-level documentation
    - Include code examples in docs
    - Explain Gaussian Möbius Topology concept
    - _Requirements: 10.4, 10.5_
  
  - [ ] 10.3 Create example scans
    - Scan famous buggy code (Heartbleed)
    - Create demo video showing topology flips
    - Document interesting findings
    - _Requirements: 10.2_
  
  - [ ] 10.4 Write usage guide
    - Explain each command and flag
    - Show example workflows
    - Explain emotional states
    - Explain topology perspectives
    - _Requirements: 10.1, 10.3_

- [ ] 11. End-to-end integration testing
  - [ ] 11.1 Test complete scan workflow
    - Parse → Topology → Detect → Emotional → Report
    - Verify all components work together
    - Test with real Rust codebases
    - _Requirements: All requirements_
  
  - [ ] 11.2 Test with existing Niodoo codebase
    - Scan `src/` directory
    - Verify topology integration works
    - Verify emotional model integration works
    - Verify GPU acceleration works
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 11.3 Performance validation
    - Run benchmarks on various codebases
    - Verify all performance targets met
    - Profile memory usage
    - Optimize bottlenecks if needed
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 12. Polish and release preparation
  - [ ] 12.1 Fix clippy warnings
    - Run `cargo clippy` and fix all warnings
    - Ensure code follows Rust best practices
    - _Requirements: All requirements_
  
  - [ ] 12.2 Add CI/CD pipeline
    - GitHub Actions for tests
    - Automated builds for releases
    - Clippy and rustfmt checks
    - _Requirements: All requirements_
  
  - [ ] 12.3 Create release artifacts
    - Build release binaries for Linux, macOS, Windows
    - Create installation script
    - Prepare GitHub release notes
    - _Requirements: All requirements_

---

## Estimated Timeline

### Week 1: Core Implementation (Tasks 1-5)
- Day 1-2: Project setup, parser, topology mapper
- Day 3-4: Detector registry and detectors
- Day 5: Emotional analyzer

### Week 2: CLI and Polish (Tasks 6-12)
- Day 1-2: Report generator and CLI interface
- Day 3: Performance optimizations
- Day 4: Error handling and logging
- Day 5: Documentation and examples
- Day 6-7: Integration testing and release prep

**Total Estimated Time:** 2 weeks (10-14 days)

---

## Dependencies Between Tasks

```
1 (Setup)
  ↓
2 (Parser) → 3 (Topology) → 4 (Detectors) → 5 (Emotional) → 6 (Report) → 7 (CLI)
                                                                              ↓
                                                                         8 (Performance)
                                                                              ↓
                                                                         9 (Errors)
                                                                              ↓
                                                                        10 (Docs)
                                                                              ↓
                                                                        11 (Integration)
                                                                              ↓
                                                                        12 (Release)
```

---

## Success Criteria

### MVP Complete When:
- [ ] All tasks marked complete
- [ ] All tests passing
- [ ] Performance targets met (<2s single file, <30s directory)
- [ ] Documentation complete
- [ ] Binary builds successfully
- [ ] Demo scan of Heartbleed code works
- [ ] Integration with existing Gen 1 components verified

---

## Notes

- **Reuse First:** Always check if Gen 1 has what you need before writing new code
- **Test Early:** Write tests as you implement, don't wait until the end
- **Incremental:** Each task should result in working, testable code
- **Performance:** Profile early, optimize bottlenecks as you find them
- **Documentation:** Write docs as you code, not after

---

## Post-MVP Enhancements (Future)

These are NOT part of the MVP but documented for future reference:

- [ ] Web UI for file uploads
- [ ] API for IDE plugins
- [ ] Python language support
- [ ] JavaScript/TypeScript support
- [ ] Auto-fix application
- [ ] CI/CD integration
- [ ] Custom detector plugins
- [ ] Bevy visualization
- [ ] Team collaboration features
- [ ] SaaS deployment

---

*Task Status: READY FOR IMPLEMENTATION*  
*Next Step: Start with Task 1 when Gen 1 is complete*  
*Estimated MVP Time: 2 weeks*
