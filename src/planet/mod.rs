//! Planetary generation and visualization.
//!
//! This module contains the geodesic grid, tectonic simulation, impact events,
//! celestial mechanics, biome/geology assignment, and rendering infrastructure
//! for generating and visualizing complete planetary worlds.
//!
//! Generation code is pure Rust (no ECS scheduling) so it can run both inside
//! the game and from the standalone `worldgen` binary.

pub mod biomes;
pub mod celestial;
pub mod geology;
pub mod grid;
pub mod impacts;
pub mod render;
pub mod tectonics;

use celestial::CelestialSystem;
use grid::IcosahedralGrid;
use serde::{Deserialize, Serialize};

// ─── Biome type ───────────────────────────────────────────────────────────────

/// Surface biome classification for a planetary cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BiomeType {
    /// Open ocean (elevation < 0 m).
    #[default]
    Ocean,
    /// Deep ocean trench (elevation < −4000 m).
    DeepOcean,
    /// Permanent polar ice sheet (T < 263 K).
    IceCap,
    /// Cold treeless plain with permafrost (T 263–273 K).
    Tundra,
    /// Coniferous boreal forest / taiga (T 273–283 K, moderate moisture).
    BorealForest,
    /// Cold, dry grassland or steppe (T 273–293 K, low moisture).
    ColdSteppe,
    /// Temperate mixed or deciduous forest (T 283–295 K, moderate moisture).
    TemperateForest,
    /// High-altitude shrubland above tree line.
    Alpine,
    /// Warm semi-arid grassland (T > 293 K, low-moderate moisture).
    TropicalSavanna,
    /// Dense tropical rainforest (T > 295 K, high moisture).
    TropicalRainforest,
    /// Hot desert (T > 293 K, very low moisture).
    HotDesert,
    /// Cold desert (polar or high-altitude, dry).
    ColdDesert,
    /// Low-lying waterlogged land near the coast.
    Wetland,
    /// Coastal tropical mangrove belt.
    Mangrove,
}

// ─── Rock type ────────────────────────────────────────────────────────────────

/// Dominant surface rock type at a planetary cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RockType {
    /// Extrusive volcanic rock; oceanic crust and rift zones.
    #[default]
    Basalt,
    /// Felsic intrusive rock; continental interiors.
    Granite,
    /// Clastic sedimentary rock; arid plains and ancient dunes.
    Sandstone,
    /// Marine carbonate sedimentary rock; shallow-sea deposits.
    Limestone,
    /// Fine-grained marine or lacustrine sedimentary rock.
    Shale,
    /// Contact metamorphic (from limestone); collision zones.
    Marble,
    /// High-grade metamorphic (from sandstone/shale); ancient mountain cores.
    Quartzite,
    /// Volcanic glass; high-silica lava flows.
    Obsidian,
    /// Mantle-derived ultramafic rock; exposed at oceanic rifts.
    Peridotite,
    /// High-grade continental metamorphic; deep crustal roots.
    Gneiss,
}

/// Type of tectonic plate crust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CrustType {
    /// Thick, buoyant, silica-rich continental crust.
    #[default]
    Continental,
    /// Thin, dense, basaltic oceanic crust.
    Oceanic,
}

/// Classification of the tectonic boundary at a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BoundaryType {
    /// Cell is not adjacent to a plate boundary.
    #[default]
    Interior,
    /// Adjacent plates are moving toward each other.
    Convergent,
    /// Adjacent plates are moving apart.
    Divergent,
    /// Adjacent plates are sliding laterally past each other.
    Transform,
}

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
    /// Planet mass in kilograms.
    pub mass_kg: f64,
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
            mass_kg: 5.972e24,     // Earth mass
            tectonic_steps: 150,
            bombardment_intensity: 0.3,
            giant_impact_probability: 0.1,
        }
    }
}

/// The complete state of a generated planet.
///
/// Populated incrementally by each generation phase:
/// grid → tectonics → impacts → celestial → biomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetData {
    /// Generation parameters.
    pub config: PlanetConfig,
    /// The geodesic grid defining cell positions and connectivity.
    pub grid: IcosahedralGrid,
    /// Per-cell elevation in meters (above/below sea level).
    pub elevation: Vec<f64>,
    /// Which tectonic plate each cell belongs to (0-based index).
    pub plate_id: Vec<u8>,
    /// Crust type per cell: Continental or Oceanic.
    pub crust_type: Vec<CrustType>,
    /// Crust thickness per cell in meters. 0 = exposed core/mantle.
    pub crust_depth: Vec<f32>,
    /// Tectonic boundary classification per cell.
    pub boundary_type: Vec<BoundaryType>,
    /// Volcanic activity intensity per cell (0.0–1.0).
    pub volcanic_activity: Vec<f32>,
    /// Accumulated fault stress per cell (0.0–1.0).
    pub fault_stress: Vec<f32>,
    /// Celestial system: star, moons, rings, and orbital mechanics.
    pub celestial: CelestialSystem,
    // ── Phase 5: biome & geology ──────────────────────────────────────────
    /// Mean annual surface temperature per cell (K).
    pub temperature_k: Vec<f32>,
    /// Mean annual precipitation per cell (mm/year).
    pub precipitation_mm: Vec<f32>,
    /// Ocean proximity per cell (0 = landlocked, 1 = at ocean).
    pub ocean_proximity: Vec<f32>,
    /// Biome classification per cell.
    pub biome: Vec<BiomeType>,
    /// Dominant surface rock type per cell.
    pub surface_rock: Vec<RockType>,
    /// Geological age per cell (0 = ancient craton, 1 = freshly-formed).
    pub geological_age: Vec<f32>,
    /// Ore deposit bitmask per cell (see `geology::ORE_*` constants).
    pub ore_deposits: Vec<u16>,
}

impl PlanetData {
    /// Create a new planet with a flat surface and zeroed tectonic data.
    pub fn new(config: PlanetConfig) -> Self {
        let grid = IcosahedralGrid::new(config.grid_level);
        let n = grid.cell_count();
        let celestial = CelestialSystem::generate(config.mass_kg, config.radius_m, config.seed);
        Self {
            config,
            grid,
            elevation: vec![0.0; n],
            plate_id: vec![0; n],
            crust_type: vec![CrustType::default(); n],
            crust_depth: vec![0.0; n],
            boundary_type: vec![BoundaryType::default(); n],
            volcanic_activity: vec![0.0; n],
            fault_stress: vec![0.0; n],
            celestial,
            temperature_k: vec![0.0; n],
            precipitation_mm: vec![0.0; n],
            ocean_proximity: vec![0.0; n],
            biome: vec![BiomeType::default(); n],
            surface_rock: vec![RockType::default(); n],
            geological_age: vec![0.5; n],
            ore_deposits: vec![0; n],
        }
    }
}
