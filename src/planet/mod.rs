//! Planetary generation and visualization.
//!
//! This module contains the geodesic grid, tectonic simulation, impact events,
//! celestial mechanics, biome/geology assignment, and rendering infrastructure
//! for generating and visualizing complete planetary worlds.
//!
//! Generation code is pure Rust (no ECS scheduling) so it can run both inside
//! the game and from the standalone `worldgen` binary.

pub mod grid;

use grid::IcosahedralGrid;
use serde::{Deserialize, Serialize};

/// Configuration for planetary generation.
///
/// All physical quantities use SI units (meters, seconds, kilograms).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetConfig {
    /// World generation seed.
    pub seed: u64,
    /// Geodesic grid subdivision level (0–10).
    /// Level 0 = 12 cells, level 7 ≈ 164K, level 9 ≈ 2.6M.
    pub grid_level: u32,
    /// Planet radius in meters.
    pub radius_m: f64,
    /// Number of tectonic simulation steps.
    pub tectonic_steps: u32,
    /// Meteorite bombardment intensity (0.0–1.0).
    pub bombardment_intensity: f64,
    /// Probability of a hemisphere-scale giant impact (0.0–1.0).
    pub giant_impact_probability: f64,
}

impl Default for PlanetConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            grid_level: 7,
            radius_m: 6_371_000.0, // Earth-like
            tectonic_steps: 150,
            bombardment_intensity: 0.3,
            giant_impact_probability: 0.1,
        }
    }
}

/// The complete state of a generated planet.
///
/// Populated incrementally by each generation phase:
/// grid → tectonics → impacts → biomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetData {
    /// Generation parameters.
    pub config: PlanetConfig,
    /// The geodesic grid defining cell positions and connectivity.
    pub grid: IcosahedralGrid,
    /// Per-cell elevation in meters (above/below sea level).
    pub elevation: Vec<f64>,
}

impl PlanetData {
    /// Create a new planet with a flat surface (all elevations at zero).
    pub fn new(config: PlanetConfig) -> Self {
        let grid = IcosahedralGrid::new(config.grid_level);
        let n = grid.cell_count();
        Self {
            config,
            grid,
            elevation: vec![0.0; n],
        }
    }
}
