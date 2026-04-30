//! Shared test-support utilities.
//!
//! Compiled only when running tests OR when the `test-support` feature is
//! enabled. Production builds do not include this module.

pub mod stress;

pub use stress::{InvariantFailure, InvariantSet, PlanetPreset, StressApp};
