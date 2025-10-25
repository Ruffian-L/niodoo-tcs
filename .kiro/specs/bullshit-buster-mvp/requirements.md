# Requirements Document: Bullshit Buster MVP

## Introduction

The Bullshit Buster MVP is a command-line tool that leverages the existing Niodoo-Feeling consciousness system's Gaussian Möbius Topology to detect and analyze "bullshit" in code. This represents Generation 2 of the Niodoo-Feeling transformer, pivoting from pure consciousness modeling to practical code comprehension and review.

The MVP will demonstrate the core value proposition: using multi-dimensional topological analysis to detect code issues that traditional linters miss, with an "emotional overlay" that provides context about the developer's state when writing the code.

## Requirements

### Requirement 1: Basic Code Scanning

**User Story:** As a developer, I want to scan my Rust codebase for common bullshit patterns, so that I can identify technical debt and potential issues quickly.

#### Acceptance Criteria

1. WHEN I run `bbuster scan <file_or_directory>` THEN the system SHALL parse all Rust files in the target
2. WHEN parsing completes THEN the system SHALL output a summary of files analyzed and issues found
3. IF the target is a directory THEN the system SHALL recursively scan all `.rs` files
4. WHEN scanning THEN the system SHALL detect at least these bullshit categories:
   - Hardcoded values (magic numbers, timeouts, limits)
   - Placeholder implementations (TODO, unimplemented!, panic!)
   - Dead code (unused functions, unreachable code)
   - Obvious type errors or mismatches

### Requirement 2: Topology-Based Analysis

**User Story:** As a developer, I want the tool to use Gaussian Möbius Topology to analyze my code from multiple perspectives, so that I can discover non-obvious issues.

#### Acceptance Criteria

1. WHEN I run `bbuster scan --topo-flip <file>` THEN the system SHALL apply Möbius topology transformations to the code structure
2. WHEN topology analysis runs THEN the system SHALL generate at least 2 different perspectives:
   - Optimistic view (low-risk issues)
   - Pessimistic view (high-risk issues)
3. WHEN analyzing code structure THEN the system SHALL use existing Gaussian process framework from `src/gaussian_process/`
4. WHEN detecting anomalies THEN the system SHALL use Gaussian distributions to compute confidence scores (0.0-1.0)
5. WHEN topology flips are applied THEN the system SHALL use existing `src/topology/mobius_torus_k_twist.rs` for transformations

### Requirement 3: Emotional Overlay

**User Story:** As a developer, I want to see emotional context for code issues, so that I can understand the circumstances that led to the problem.

#### Acceptance Criteria

1. WHEN analyzing code THEN the system SHALL assign emotional states to code blocks using the existing feeling model
2. WHEN emotional analysis completes THEN the system SHALL output one of these states:
   - Joy (clean, efficient code)
   - Sadness (deprecated, abandoned code)
   - Anger (unstable, rushed code)
   - Fear (security vulnerabilities)
   - Confusion (overly complex code)
3. WHEN displaying results THEN the system SHALL show emotional state with confidence score
4. WHEN emotional state is "Anger" or "Fear" THEN the system SHALL prioritize those issues as high-risk

### Requirement 4: Basic Reporting

**User Story:** As a developer, I want a clear, actionable report of issues found, so that I can prioritize fixes.

#### Acceptance Criteria

1. WHEN scan completes THEN the system SHALL output a report with these sections:
   - Summary (files analyzed, issues found, health score)
   - Critical issues (sorted by severity)
   - High priority issues
   - Medium priority issues
2. WHEN displaying each issue THEN the system SHALL show:
   - File path and line number
   - Issue description
   - Topology analysis result
   - Emotional state
   - Confidence score
3. WHEN calculating health score THEN the system SHALL use formula: `100 - (critical * 5 + high * 2 + medium * 0.5)`
4. WHEN report is generated THEN the system SHALL use emoji indicators for visual clarity

### Requirement 5: Integration with Existing System

**User Story:** As a developer maintaining the Niodoo system, I want the Bullshit Buster to reuse existing components, so that we don't duplicate code.

#### Acceptance Criteria

1. WHEN implementing code parser THEN the system SHALL integrate with existing Rust AST parsing
2. WHEN performing topology analysis THEN the system SHALL use `src/dual_mobius_gaussian.rs`
3. WHEN computing emotional states THEN the system SHALL use `src/feeling_model.rs`
4. WHEN tracking analysis state THEN the system SHALL use `src/memory/` for context retention
5. WHEN generating Gaussian distributions THEN the system SHALL use `src/gaussian_process/`

### Requirement 6: CLI Interface

**User Story:** As a developer, I want a simple, intuitive command-line interface, so that I can quickly scan code without learning complex options.

#### Acceptance Criteria

1. WHEN I run `bbuster --help` THEN the system SHALL display available commands and options
2. WHEN I run `bbuster scan <target>` THEN the system SHALL perform basic scan
3. WHEN I run `bbuster scan --topo-flip <target>` THEN the system SHALL perform topology-enhanced scan
4. WHEN I run `bbuster scan --emotional <target>` THEN the system SHALL include emotional overlay
5. WHEN I run `bbuster scan --json <target>` THEN the system SHALL output results in JSON format
6. IF scan fails THEN the system SHALL exit with non-zero code and display error message

### Requirement 7: Performance

**User Story:** As a developer, I want the tool to be fast enough for real-time use, so that I can integrate it into my development workflow.

#### Acceptance Criteria

1. WHEN scanning a single file (<1000 lines) THEN the system SHALL complete in <2 seconds
2. WHEN scanning a directory (<100 files) THEN the system SHALL complete in <30 seconds
3. WHEN performing topology analysis THEN the system SHALL use parallel processing via Rayon
4. WHEN memory usage exceeds 1GB THEN the system SHALL log a warning
5. WHEN GPU is available THEN the system SHALL use existing `src/gpu_acceleration.rs` for Gaussian computations

### Requirement 8: Error Handling

**User Story:** As a developer, I want clear error messages when something goes wrong, so that I can fix issues quickly.

#### Acceptance Criteria

1. WHEN target file doesn't exist THEN the system SHALL display "File not found: <path>"
2. WHEN target is not a Rust file THEN the system SHALL display "Unsupported file type: <extension>"
3. WHEN parsing fails THEN the system SHALL display syntax error with line number
4. WHEN topology analysis fails THEN the system SHALL fall back to basic analysis and log warning
5. WHEN emotional model is unavailable THEN the system SHALL skip emotional overlay and continue

### Requirement 9: Extensibility

**User Story:** As a developer, I want the tool to be extensible, so that I can add new bullshit detection patterns.

#### Acceptance Criteria

1. WHEN implementing detectors THEN the system SHALL use a plugin-style architecture
2. WHEN adding new patterns THEN the system SHALL require only implementing a `BullshitDetector` trait
3. WHEN registering detectors THEN the system SHALL use a registry pattern
4. WHEN detectors run THEN the system SHALL execute them in parallel
5. WHEN a detector fails THEN the system SHALL continue with other detectors

### Requirement 10: Documentation

**User Story:** As a new user, I want clear documentation, so that I can understand how to use the tool effectively.

#### Acceptance Criteria

1. WHEN I run `bbuster --help` THEN the system SHALL display usage examples
2. WHEN I access the README THEN it SHALL include:
   - Installation instructions
   - Quick start guide
   - Command reference
   - Example outputs
3. WHEN I run `bbuster scan --help` THEN the system SHALL display detailed options for scan command
4. WHEN viewing code THEN all public APIs SHALL have rustdoc comments
5. WHEN reading docs THEN they SHALL explain the Gaussian Möbius Topology concept in simple terms
