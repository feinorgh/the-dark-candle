//! Headless gameplay stress-test harness.
//!
//! See `docs/superpowers/specs/2026-04-30-gameplay-stress-tests-design.md`.

use bevy::prelude::*;

/// Choice of planet configuration for the stress test.
#[derive(Clone, Copy, Debug)]
pub enum PlanetPreset {
    /// Earth-scale planet (mean radius ≈ 6.37e6 m).
    Earth,
    /// Default `PlanetConfig::default()` small planet (32 km).
    SmallPlanet,
}

/// Bitflags selecting which invariants to assert.
#[derive(Clone, Copy, Debug, Default)]
pub struct InvariantSet(u8);

impl InvariantSet {
    pub const PANICS: Self = Self(0b00001);
    pub const FINITE: Self = Self(0b00010);
    pub const NO_OVERFLOW: Self = Self(0b00100);
    pub const CHUNK_CACHE: Self = Self(0b01000);
    pub const LOAD_RATE: Self = Self(0b10000);

    pub const ALL_BUT_LOAD_RATE: Self = Self(0b01111);

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for InvariantSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// One invariant violation discovered by `assert_invariants`.
#[derive(Debug, Clone)]
pub enum InvariantFailure {
    Panic {
        thread: String,
        location: String,
        message: String,
    },
    NonFiniteTransform {
        entity: u64,
        kind: &'static str,
        value: String,
    },
    F32Overflow {
        what: String,
        value: f64,
    },
    ChunkCache {
        detail: String,
    },
    LoadRateBelowMin {
        observed: f32,
        min: f32,
    },
}

/// Headless gameplay stress harness.
///
/// Built progressively across the plan tasks:
/// - Task 3: `new`, `tick_n`
/// - Task 4: `teleport`
/// - Tasks 5–9: invariant checks
/// - Task 10: `chunk_load_rate`
#[allow(dead_code)]
pub struct StressApp {
    pub(crate) app: App,
    pub(crate) seed: u64, // fields populated across plan tasks 3-8
}
