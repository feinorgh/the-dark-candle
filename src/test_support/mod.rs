//! Shared test-support utilities.
//!
//! Compiled only when the `test-support` feature is enabled (e.g.
//! `cargo test --features test-support`). Production builds do not include
//! this module.

pub mod stress;

pub use stress::{InvariantFailure, InvariantSet, PlanetPreset, StressApp};
