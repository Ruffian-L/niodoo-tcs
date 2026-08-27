//! Stub workspace member so `cargo` can load the tree.
//!
//! Phase 5 cleanup (2025-11-10) moved would-be re-exports to
//! `.legacy/niodoo-core-deps/`. This crate exists so
//! `niodoo_real_integrated` can keep `niodoo-core = { path = "../niodoo-core" }`
//! without breaking `cargo check -p tcs-ml`.
//!
//! It is not the consciousness engine. Do not import it as if it were.
