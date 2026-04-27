// Planetary terrain sampler — bridges PlanetData to voxel chunk generation.
//
// PlanetData contains per-geodesic-cell data (elevation, biome, rock type,
// ore deposits, temperature) derived from the tectonic simulation pipeline.
// This module wraps that data and exposes it in the form expected by the
// chunk generation pipeline:
//
//   1. Surface radius sampling via IDW interpolation between geodesic cells,
//      augmented by fractal detail noise (uses `planet::detail` module).
//
//   2. Material assignment: BiomeType → surface MaterialId; RockType →
//      subsurface MaterialId; ore bitmask → ore vein placement.
//
//   3. ChunkBiomeData component — attached to every chunk entity generated
//      via PlanetaryTerrainSampler, carrying the dominant biome, climate,
//      and ore bitmask for that chunk so procgen systems can use it.
//
// The sampler is Send + Sync so it can be used from AsyncComputeTaskPool.

use std::sync::Arc;

use bevy::math::DVec3;
use bevy::prelude::Component;

use crate::planet::detail::{
    TerrainNoise, interpolate_elevation, sample_detailed_elevation, terrain_roughness,
};
use crate::planet::geology;
use crate::planet::grid::{CellId, CellIndex};
use crate::planet::{BiomeType, PlanetData, RockType};
use crate::world::chunk::{CHUNK_SIZE, Chunk};
use crate::world::planet::PlanetConfig;
use crate::world::terrain::terrain_density;
use crate::world::voxel::{MaterialId, Voxel};

// ─── ChunkBiomeData component ─────────────────────────────────────────────────

/// Per-chunk biome metadata derived from planetary generation.
///
/// Spawned as a component on every chunk entity produced by
/// `PlanetaryTerrainSampler`.  Downstream procgen systems (creature
/// spawning, prop decoration, tree planting) read this instead of
/// computing biome from terrain height heuristics.
#[derive(Component, Debug, Clone)]
pub struct ChunkBiomeData {
    /// Dominant planetary biome for this chunk.
    pub planet_biome: BiomeType,
    /// Mean annual surface temperature at this location (K).
    pub temperature_k: f32,
    /// Mean annual precipitation at this location (mm/year).
    pub precipitation_mm: f32,
    /// Dominant surface rock type.
    pub surface_rock: RockType,
    /// Ore deposit bitmask (use `planet::geology::ORE_*` constants).
    pub ore_bitmask: u16,
    /// Ocean proximity: 0.0 = inland, 1.0 = at ocean edge.
    pub ocean_proximity: f32,
}

// ─── Material mappings ────────────────────────────────────────────────────────

/// Map a `BiomeType` to the voxel material used for the top surface layer.
///
/// Returns the material that appears at depth 0 (the very top voxel of solid
/// terrain).  Deeper layers are determined by `rock_type_to_material`.
pub fn biome_to_surface_material(biome: BiomeType, above_sea_level: bool) -> MaterialId {
    if !above_sea_level {
        return MaterialId::DIRT; // seafloor
    }
    match biome {
        BiomeType::IceCap => MaterialId::ICE,
        BiomeType::HotDesert | BiomeType::ColdDesert => MaterialId::SAND,
        BiomeType::Alpine => MaterialId::STONE,
        BiomeType::Tundra => MaterialId::STONE,
        BiomeType::Wetland | BiomeType::Mangrove => MaterialId::DIRT,
        // All forested/grassland biomes get grass
        BiomeType::BorealForest
        | BiomeType::ColdSteppe
        | BiomeType::TemperateForest
        | BiomeType::TropicalSavanna
        | BiomeType::TropicalRainforest => MaterialId::GRASS,
        // Ocean cells are handled by sea-level logic in the sampler, not here.
        BiomeType::Ocean | BiomeType::DeepOcean => MaterialId::DIRT,
    }
}

/// Map a `BiomeType` to the shallow sub-surface layer (1–`soil_depth` m).
///
/// Typically dirt/regolith.
pub fn biome_to_subsoil_material(biome: BiomeType) -> MaterialId {
    match biome {
        BiomeType::HotDesert | BiomeType::ColdDesert => MaterialId::SAND,
        BiomeType::IceCap => MaterialId::STONE,
        _ => MaterialId::DIRT,
    }
}

/// Map a `RockType` to the base voxel material used for bedrock.
///
/// This crate currently ships a limited set of voxel materials (Stone, Iron,
/// Sand, etc.).  All metamorphic and igneous variants map to Stone for now.
/// Add new `MaterialId` constants and `.material.ron` files to expand this.
pub fn rock_type_to_material(rock: RockType) -> MaterialId {
    match rock {
        RockType::Sandstone => MaterialId::SAND,
        // All other rock types map to Stone (base material)
        RockType::Basalt
        | RockType::Granite
        | RockType::Limestone
        | RockType::Shale
        | RockType::Marble
        | RockType::Quartzite
        | RockType::Obsidian
        | RockType::Peridotite
        | RockType::Gneiss => MaterialId::STONE,
    }
}

/// Return the ore `MaterialId` to scatter at a given depth, given the cell's
/// ore bitmask.  Returns `None` if no ore applies at this depth.
///
/// Ore veins are placed deterministically based on voxel world position.
/// Each ore type occupies a different depth range to avoid overlap.
pub fn ore_material_at(
    ore_bitmask: u16,
    depth_below_surface: f64,
    hash: u64,
) -> Option<MaterialId> {
    // Only place ore in a narrow depth band; use a hash threshold for sparsity.
    if hash % 100 >= 3 {
        return None; // ~97% of candidate positions are plain rock
    }

    // Iron: 5–40 m below surface
    if ore_bitmask & geology::ORE_IRON != 0 && (5.0..40.0).contains(&depth_below_surface) {
        return Some(MaterialId::IRON);
    }
    // Other ore types: Stone for now (extend when new MaterialIds added)
    None
}

// ─── PlanetaryTerrainSampler ──────────────────────────────────────────────────

/// Terrain generator that drives voxel chunk generation from `PlanetData`.
///
/// Uses geodesic-cell IDW elevation + fractal detail noise for surface height,
/// biome-aware materials for surface layers, rock-type materials for bedrock,
/// and ore-bitmask placement for resource veins.
///
/// `Arc`-wrapped fields make this `Send + Sync` so it can be cloned into
/// `AsyncComputeTaskPool` tasks.
pub struct PlanetaryTerrainSampler {
    pub planet_data: Arc<PlanetData>,
    pub cell_index: Arc<CellIndex>,
    pub detail_noise: Arc<TerrainNoise>,
    /// World-space `PlanetConfig` (radius, sea level, gravity, layers…).
    pub planet_config: PlanetConfig,
}

impl PlanetaryTerrainSampler {
    /// Construct a sampler.  Call once at startup; construction is O(n_cells).
    pub fn new(planet_data: Arc<PlanetData>, planet_config: PlanetConfig) -> Self {
        let seed = planet_data.config.seed;
        let cell_index = Arc::new(CellIndex::build(&planet_data.grid));
        let detail_noise = Arc::new(TerrainNoise::new(seed));
        Self {
            planet_data,
            cell_index,
            detail_noise,
            planet_config,
        }
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Return the unit-sphere position for a world voxel.
    fn unit_pos(&self, world_pos: DVec3) -> DVec3 {
        world_pos.normalize_or(DVec3::Y)
    }

    /// Return the surface radius (from planet center) at a unit-sphere position.
    ///
    /// Combines IDW interpolation from geodesic cells with fractal noise from
    /// `TerrainNoise` (same path as `detail::sample_detailed_elevation`).
    pub fn surface_radius_at(&self, unit_pos: DVec3) -> (f64, CellId) {
        let cell = self
            .cell_index
            .nearest_cell(&self.planet_data.grid, unit_pos);
        let (elev_m, ci) =
            sample_detailed_elevation(&self.planet_data, &self.detail_noise, unit_pos, cell);
        let surface_r = self.planet_config.mean_radius + elev_m;
        (surface_r, CellId(ci as u32))
    }

    /// Return `(idw_elevation_m, roughness, is_ocean_biome)` at a unit-sphere position.
    ///
    /// Provides the three raster components needed by the GPU heightmap bake:
    /// - `idw_elevation_m`: IDW-interpolated tectonic elevation **without** procedural noise
    /// - `roughness`: biome/boundary/volcanic noise roughness in \[0, 1\]
    /// - `is_ocean_biome`: whether the nearest cell is an ocean or deep-ocean biome
    ///
    /// This separates the smooth IDW component (safe to bilinearly interpolate) from
    /// the high-frequency TerrainNoise (which must be evaluated at the exact column
    /// position to avoid aliasing).
    pub fn idw_roughness_ocean_at(&self, unit_pos: DVec3) -> (f64, f64, bool) {
        let cell = self
            .cell_index
            .nearest_cell(&self.planet_data.grid, unit_pos);
        let ci = cell.index();
        let idw = interpolate_elevation(&self.planet_data, unit_pos, cell);
        let roughness = terrain_roughness(
            self.planet_data.biome[ci],
            idw,
            self.planet_data.volcanic_activity[ci],
            self.planet_data.boundary_type[ci],
        );
        let is_ocean = matches!(
            self.planet_data.biome[ci],
            BiomeType::Ocean | BiomeType::DeepOcean
        );
        (idw, roughness, is_ocean)
    }

    /// Return the dominant biome data for the cell nearest to `unit_pos`.
    fn cell_biome_data(&self, cell: CellId) -> ChunkBiomeData {
        let i = cell.index();
        ChunkBiomeData {
            planet_biome: self.planet_data.biome[i],
            temperature_k: self.planet_data.temperature_k[i],
            precipitation_mm: self.planet_data.precipitation_mm[i],
            surface_rock: self.planet_data.surface_rock[i],
            ore_bitmask: self.planet_data.ore_deposits[i],
            ocean_proximity: self.planet_data.ocean_proximity[i],
        }
    }

    /// Simple deterministic hash for ore vein placement.
    fn voxel_hash(wx: i64, wy: i64, wz: i64, seed: u64) -> u64 {
        let mut h = seed;
        h ^= (wx as u64).wrapping_mul(6364136223846793005);
        h ^= (wy as u64).wrapping_mul(2862933555777941757);
        h ^= (wz as u64).wrapping_mul(3202034522624059733);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Fill a chunk with terrain derived from `PlanetData`.
    ///
    /// Returns the `ChunkBiomeData` for the chunk center — dominated by the
    /// geodesic cell nearest to the chunk's world-space center.
    pub fn generate_chunk(&self, chunk: &mut Chunk) -> ChunkBiomeData {
        let origin = chunk.coord.world_origin();
        let seed = self.planet_data.config.seed;
        let sea_r = self.planet_config.sea_level_radius;
        let soil_depth = self.planet_config.soil_depth;

        // Determine the dominant cell for the chunk center (for ChunkBiomeData).
        let center = DVec3::new(
            origin.x as f64 + CHUNK_SIZE as f64 * 0.5,
            origin.y as f64 + CHUNK_SIZE as f64 * 0.5,
            origin.z as f64 + CHUNK_SIZE as f64 * 0.5,
        );
        let center_unit = self.unit_pos(center);
        let center_cell = self
            .cell_index
            .nearest_cell(&self.planet_data.grid, center_unit);
        let biome_data = self.cell_biome_data(center_cell);

        // Per column (lx, lz) sample surface radius once, then sweep ly.
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let world_x = origin.x as f64 + lx as f64 + 0.5;
                let world_z = origin.z as f64 + lz as f64 + 0.5;

                // Surface sample uses middle-Y of the column to avoid the
                // zero-vector singularity at the planet center.
                let col_mid_y = origin.y as f64 + CHUNK_SIZE as f64 * 0.5;
                let col_pos = DVec3::new(world_x, col_mid_y, world_z);
                let unit = self.unit_pos(col_pos);

                let (surface_r, cell) = self.surface_radius_at(unit);
                let ci = cell.index();
                let biome = self.planet_data.biome[ci];
                let rock = self.planet_data.surface_rock[ci];
                let ore_mask = self.planet_data.ore_deposits[ci];
                let temp_k = self.planet_data.temperature_k[ci];

                let surface_mat = biome_to_surface_material(biome, surface_r >= sea_r);
                let subsoil_mat = biome_to_subsoil_material(biome);
                let bedrock_mat = rock_type_to_material(rock);

                for ly in 0..CHUNK_SIZE {
                    let world_x_int = origin.x + lx as i32;
                    let world_y_int = origin.y + ly as i32;
                    let world_z_int = origin.z + lz as i32;

                    let world_y = world_y_int as f64 + 0.5;
                    // Radial distance: for a column sample we use the actual
                    // (world_x, world_y, world_z) position.
                    let pos = DVec3::new(world_x, world_y, world_z);
                    let r = pos.length();

                    let depth = surface_r - r;

                    let material = if r > surface_r {
                        // Above surface
                        if r < sea_r {
                            MaterialId::WATER
                        } else {
                            MaterialId::AIR
                        }
                    } else if depth < 1.0 {
                        // Top 1 m: surface material
                        surface_mat
                    } else if depth < soil_depth {
                        // Shallow subsurface: sub-soil layer
                        subsoil_mat
                    } else {
                        // Deep: bedrock with ore veins
                        let hash = Self::voxel_hash(
                            world_x_int as i64,
                            world_y_int as i64,
                            world_z_int as i64,
                            seed,
                        );
                        ore_material_at(ore_mask, depth, hash).unwrap_or(bedrock_mat)
                    };

                    // Cave carving: thin out rock in the crust band.
                    // Uses the world-space PlanetConfig cave noise (3D Perlin).
                    // Skip caves near the surface or in water.
                    let final_material = if material != MaterialId::AIR
                        && material != MaterialId::WATER
                        && depth > 4.0
                        && r > self.planet_config.mean_radius - 4_000.0
                        && self.planet_config_cave_sample(world_x, world_y, world_z)
                    {
                        MaterialId::AIR
                    } else {
                        material
                    };

                    let density = if final_material == MaterialId::WATER {
                        terrain_density(sea_r - r)
                    } else if final_material.is_air() && depth < 0.0 {
                        // Cave-carved air deep underground: binary 0.0
                        0.0
                    } else {
                        terrain_density(depth)
                    };

                    let voxel = Voxel {
                        material: final_material,
                        temperature: temp_k,
                        pressure: crate::physics::constants::ATMOSPHERIC_PRESSURE,
                        damage: 0.0,
                        latent_heat_buffer: 0.0,
                        density,
                    };
                    chunk.set(lx, ly, lz, voxel);
                }
            }
        }

        biome_data
    }

    /// Determine the material at a radial position using `PlanetConfig`-based
    /// logic (same rules as `SphericalTerrainGenerator::material_at_radius`).
    ///
    /// Used by the V2 pipeline to assign voxel materials when the planetary
    /// sampler is active. The result respects sea level and soil depth from
    /// `planet_config` but does not use per-cell biome data.
    pub fn material_at_radius(
        &self,
        r: f64,
        surface_r: f64,
        world_x: f64,
        world_y: f64,
        world_z: f64,
    ) -> crate::world::voxel::MaterialId {
        use crate::world::terrain::SphericalTerrainGenerator;
        use std::cell::RefCell;
        thread_local! {
            static MAT_GEN: RefCell<Option<SphericalTerrainGenerator>> = const { RefCell::new(None) };
        }
        MAT_GEN.with(|g| {
            let mut borrow = g.borrow_mut();
            if borrow.is_none() {
                *borrow = Some(SphericalTerrainGenerator::new(self.planet_config.clone()));
            }
            borrow
                .as_ref()
                .unwrap()
                .material_at_radius(r, surface_r, world_x, world_y, world_z)
        })
    }

    /// Sample the cave noise at a world position using the PlanetConfig parameters.
    fn planet_config_cave_sample(&self, wx: f64, wy: f64, wz: f64) -> bool {
        // Reuse the PlanetConfig noise via the existing SphericalTerrainGenerator.
        // We create a thread-local generator seeded from the planet seed.
        use crate::world::terrain::SphericalTerrainGenerator;
        use std::cell::RefCell;
        thread_local! {
            static GEN: RefCell<Option<SphericalTerrainGenerator>> = const { RefCell::new(None) };
        }
        GEN.with(|g| {
            let mut borrow = g.borrow_mut();
            if borrow.is_none() {
                *borrow = Some(SphericalTerrainGenerator::new(self.planet_config.clone()));
            }
            borrow.as_ref().unwrap().is_cave(wx, wy, wz)
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::biomes::run_biomes;
    use crate::planet::geology::run_geology;
    use crate::planet::tectonics::run_tectonics;
    use crate::planet::{PlanetConfig as PlanetGenConfig, PlanetData};
    use crate::world::chunk::ChunkCoord;
    use crate::world::planet::{GeologicalLayer, PlanetConfig, TerrainMode};

    fn make_sampler() -> PlanetaryTerrainSampler {
        let gen_config = PlanetGenConfig {
            seed: 42,
            grid_level: 3, // 642 cells — fast for tests
            ..Default::default()
        };
        let mut data = PlanetData::new(gen_config);
        run_tectonics(&mut data, |_| {});
        run_biomes(&mut data);
        run_geology(&mut data);

        // Small planet so chunks land on the surface.
        let planet_config = PlanetConfig {
            mode: TerrainMode::Planetary,
            mean_radius: 6_371_000.0,
            sea_level_radius: 6_371_000.0,
            height_scale: 5_000.0,
            soil_depth: 4.0,
            layers: vec![GeologicalLayer {
                name: "crust".into(),
                inner_radius: 0.0,
                outer_radius: 6_371_000.0,
                material: "Stone".into(),
            }],
            ..Default::default()
        };

        PlanetaryTerrainSampler::new(Arc::new(data), planet_config)
    }

    #[test]
    fn generate_chunk_returns_biome_data() {
        let sampler = make_sampler();
        // Place chunk at a point guaranteed on the surface.
        let r = sampler.planet_config.mean_radius as i32;
        let coord = ChunkCoord::new(r / 32, 0, 0);
        let mut chunk = Chunk::new_empty(coord);
        let bd = sampler.generate_chunk(&mut chunk);
        // BiomeData must have a valid temperature.
        assert!(
            bd.temperature_k > 100.0 && bd.temperature_k < 500.0,
            "Unexpected temperature: {}",
            bd.temperature_k
        );
    }

    #[test]
    fn biome_to_surface_material_ice_cap() {
        assert_eq!(
            biome_to_surface_material(BiomeType::IceCap, true),
            MaterialId::ICE
        );
    }

    #[test]
    fn biome_to_surface_material_desert() {
        assert_eq!(
            biome_to_surface_material(BiomeType::HotDesert, true),
            MaterialId::SAND
        );
    }

    #[test]
    fn biome_to_surface_material_forest() {
        assert_eq!(
            biome_to_surface_material(BiomeType::TemperateForest, true),
            MaterialId::GRASS
        );
    }

    #[test]
    fn rock_type_sandstone_gives_sand() {
        assert_eq!(rock_type_to_material(RockType::Sandstone), MaterialId::SAND);
    }

    #[test]
    fn rock_type_basalt_gives_stone() {
        assert_eq!(rock_type_to_material(RockType::Basalt), MaterialId::STONE);
    }
}
