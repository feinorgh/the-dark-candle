// LbmGasPlugin: Bevy ECS integration for the D3Q19 Lattice Boltzmann gas solver.
//
// Manages per-chunk LbmGrids as a resource, runs the LBM simulation on
// FixedUpdate, and syncs results back to chunks. Parallels the AMR fluid
// plugin architecture.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::data::{FluidConfig, MaterialRegistry};
use crate::lighting::{SolarInsolation, TimeOfDay};
use crate::physics::atmosphere::AtmosphereConfig;
use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::chunk_manager::ChunkMap;
use crate::world::planet::PlanetConfig;
use crate::world::voxel::MaterialId;

use super::step;
use super::sync;
use super::types::LbmGrid;

/// Resource: maps chunk coordinates to their LBM gas simulation state.
/// Only chunks containing gas voxels with active dynamics have an entry.
#[derive(Resource, Default)]
pub struct LbmState {
    grids: HashMap<ChunkCoord, LbmGrid>,
}

impl LbmState {
    pub fn get(&self, coord: &ChunkCoord) -> Option<&LbmGrid> {
        self.grids.get(coord)
    }

    pub fn get_mut(&mut self, coord: &ChunkCoord) -> Option<&mut LbmGrid> {
        self.grids.get_mut(coord)
    }

    pub fn insert(&mut self, coord: ChunkCoord, grid: LbmGrid) {
        self.grids.insert(coord, grid);
    }

    pub fn remove(&mut self, coord: &ChunkCoord) -> Option<LbmGrid> {
        self.grids.remove(coord)
    }

    pub fn contains(&self, coord: &ChunkCoord) -> bool {
        self.grids.contains_key(coord)
    }

    pub fn len(&self) -> usize {
        self.grids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }
}

/// Wrapper resource for FluidConfig (shared with AMR fluid plugin).
#[derive(Resource, Default)]
pub struct LbmConfigRes(pub FluidConfig);

/// Tick counter for the LBM gas simulation.
#[derive(Resource, Default)]
pub struct LbmTick(pub u64);

/// Plugin that adds D3Q19 LBM gas simulation to the physics pipeline.
pub struct LbmGasPlugin;

impl Plugin for LbmGasPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LbmState>()
            .init_resource::<LbmConfigRes>()
            .init_resource::<LbmTick>()
            .add_systems(
                FixedUpdate,
                (
                    apply_solar_heating,
                    init_lbm_grids,
                    lbm_gas_step,
                    cleanup_empty_lbm_grids,
                )
                    .chain(),
            );
    }
}

/// Apply solar heating to surface voxels based on sun angle, latitude, and albedo.
/// This drives convection in the LBM gas simulation.
#[allow(clippy::too_many_arguments)]
fn apply_solar_heating(
    mut chunks: Query<&mut Chunk>,
    chunk_map: Option<Res<ChunkMap>>,
    time_of_day: Option<Res<TimeOfDay>>,
    atmosphere_config: Option<Res<AtmosphereConfig>>,
    planet_config: Option<Res<PlanetConfig>>,
    solar_insolation: Option<Res<SolarInsolation>>,
    registry: Option<Res<MaterialRegistry>>,
    time: Res<Time>,
) {
    // Early return if any required resource is missing
    let (
        Some(chunk_map),
        Some(_time_of_day),
        Some(atmosphere),
        Some(_planet),
        Some(insolation),
        Some(registry),
    ) = (
        chunk_map.as_ref(),
        time_of_day.as_ref(),
        atmosphere_config.as_ref(),
        planet_config.as_ref(),
        solar_insolation.as_ref(),
        registry.as_ref(),
    )
    else {
        return;
    };

    // Skip at night
    if insolation.0 <= 0.0 {
        return;
    }

    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    let solar_constant = atmosphere.solar_constant; // W/m²
    let insolation_factor = insolation.0; // 0.0–1.0

    // For each chunk with voxels, find surface voxels and heat them
    for coord in chunk_map.coords() {
        let Some(entity) = chunk_map.get(coord) else {
            continue;
        };
        let Ok(mut chunk) = chunks.get_mut(entity) else {
            continue;
        };

        // Process surface voxels: scan from top down, find first non-air
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                // Find the top-most non-air voxel in this column
                let mut surface_y = None;
                for y in (0..CHUNK_SIZE).rev() {
                    if !chunk.get(x, y, z).is_air() {
                        surface_y = Some(y);
                        break;
                    }
                }

                let Some(sy) = surface_y else {
                    continue; // Empty column
                };

                // Check if there's air above this surface voxel
                // (either y+1 is within chunk and is air, or y+1 is outside chunk = exposed)
                let exposed = if sy + 1 < CHUNK_SIZE {
                    chunk.get(x, sy + 1, z).is_air()
                } else {
                    true // Top of chunk — assume exposed
                };

                if !exposed {
                    continue; // Not a surface
                }

                // Get surface material properties
                let surface_voxel = chunk.get(x, sy, z);
                let Some(mat_data) = registry.get(surface_voxel.material) else {
                    continue;
                };

                let albedo = mat_data.albedo;
                let density = mat_data.density;
                let specific_heat = mat_data.specific_heat_capacity;

                // Heat absorbed: Q = S₀ × insolation_factor × (1 - albedo) [W/m²]
                let q_absorbed = solar_constant * insolation_factor * (1.0 - albedo);

                // Temperature change for surface voxel: ΔT = Q × dt / (ρ × Cₚ × V)
                // where V = 1 m³
                if density > 0.0 && specific_heat > 0.0 {
                    let delta_t_surface = q_absorbed * dt / (density * specific_heat);
                    let voxel = chunk.get_mut(x, sy, z);
                    voxel.temperature += delta_t_surface;

                    // Also heat the air voxel directly above (if present in chunk)
                    if sy + 1 < CHUNK_SIZE {
                        let air_voxel = chunk.get_mut(x, sy + 1, z);
                        if air_voxel.is_air() {
                            // Air gets a fraction of the surface heating (thermal contact)
                            let air_mat = registry.get(air_voxel.material);
                            if let Some(air_data) = air_mat {
                                let air_density = air_data.density;
                                let air_cp = air_data.specific_heat_capacity;
                                if air_density > 0.0 && air_cp > 0.0 {
                                    // Apply ~10% of surface heating to adjacent air
                                    let delta_t_air =
                                        0.1 * q_absorbed * dt / (air_density * air_cp);
                                    air_voxel.temperature += delta_t_air;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Initialize LbmGrids for newly loaded chunks that contain gas voxels.
fn init_lbm_grids(chunks: Query<&Chunk, Added<Chunk>>, mut lbm_state: ResMut<LbmState>) {
    for chunk in chunks.iter() {
        if lbm_state.contains(&chunk.coord) {
            continue;
        }

        // Only create LBM grid if chunk has non-air gas (steam, smoke, etc.)
        // or is adjacent to heat sources. For now, check for steam.
        let has_active_gas = chunk.voxels().iter().any(|v| is_active_gas(v.material));
        if !has_active_gas {
            continue;
        }

        let grid = LbmGrid::from_chunk(chunk, None);
        lbm_state.insert(chunk.coord, grid);
    }
}

/// Run one LBM gas simulation step for all active gas chunks.
#[allow(clippy::too_many_arguments)]
fn lbm_gas_step(
    mut chunks: Query<&mut Chunk>,
    chunk_map: Option<Res<ChunkMap>>,
    config: Res<LbmConfigRes>,
    mut lbm_state: ResMut<LbmState>,
    mut tick: ResMut<LbmTick>,
    time: Res<Time>,
    atmosphere_config: Option<Res<AtmosphereConfig>>,
    planet_config: Option<Res<PlanetConfig>>,
) {
    if !config.0.lbm_enabled {
        return;
    }

    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    let chunk_map = match chunk_map {
        Some(cm) => cm,
        None => return,
    };

    // Gravity in lattice units (small — one LBM step may be multiple seconds)
    // Using a small value since the exact scaling depends on dt_lattice
    let gravity_lattice = [0.0, -0.001, 0.0];
    let rho_ambient = 1.0;

    // Compute Coriolis omega in lattice units
    let coriolis_omega = if let (Some(atm_cfg), Some(planet_cfg)) =
        (atmosphere_config.as_deref(), planet_config.as_deref())
    {
        if atm_cfg.coriolis_enabled {
            // Planetary rotation vector: omega_physical = rotation_rate * rotation_axis
            let omega_physical = [
                (planet_cfg.rotation_rate * planet_cfg.rotation_axis[0]) as f32,
                (planet_cfg.rotation_rate * planet_cfg.rotation_axis[1]) as f32,
                (planet_cfg.rotation_rate * planet_cfg.rotation_axis[2]) as f32,
            ];

            // Convert to lattice units using same scaling as gravity
            // gravity_lattice[1] = -0.001 corresponds to g = 9.80665 m/s²
            // So lattice_scale = 0.001 / 9.80665 ≈ 1.02e-4
            const GRAVITY_MAGNITUDE: f32 = 9.80665;
            let lattice_scale = 0.001 / GRAVITY_MAGNITUDE;

            Some([
                omega_physical[0] * lattice_scale,
                omega_physical[1] * lattice_scale,
                omega_physical[2] * lattice_scale,
            ])
        } else {
            None
        }
    } else {
        None
    };

    let coords: Vec<ChunkCoord> = lbm_state.grids.keys().cloned().collect();
    let n_steps = config.0.lbm_steps_per_tick.max(1);

    for coord in coords {
        let Some(grid) = lbm_state.grids.get_mut(&coord) else {
            continue;
        };

        step::lbm_step_n(
            grid,
            &config.0,
            gravity_lattice,
            rho_ambient,
            coriolis_omega,
            n_steps,
        );

        if let Some(entity) = chunk_map.get(&coord)
            && let Ok(mut chunk) = chunks.get_mut(entity)
        {
            sync::sync_to_chunk(grid, &mut chunk);
        }
    }

    tick.0 += 1;
}

/// Remove LbmGrid entries for chunks that no longer have gas dynamics.
fn cleanup_empty_lbm_grids(mut lbm_state: ResMut<LbmState>) {
    lbm_state.grids.retain(|_, grid| grid.has_gas());
}

/// Check if a material represents a gas with active dynamics.
/// Air alone is passive (ambient); steam, smoke etc. are active.
fn is_active_gas(mat: MaterialId) -> bool {
    mat == MaterialId::STEAM
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbm_state_resource_operations() {
        let mut state = LbmState::default();
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };

        assert!(state.is_empty());
        assert!(!state.contains(&coord));

        state.insert(coord, LbmGrid::new_empty(32));
        assert_eq!(state.len(), 1);
        assert!(state.contains(&coord));

        assert!(state.get(&coord).is_some());
        assert!(state.get_mut(&coord).is_some());

        state.remove(&coord);
        assert!(state.is_empty());
    }

    #[test]
    fn lbm_config_res_has_defaults() {
        let config = LbmConfigRes::default();
        assert!((config.0.lbm_tau - 0.55).abs() < 1e-6);
        assert!((config.0.lbm_smagorinsky_cs - 0.1).abs() < 1e-6);
        assert_eq!(config.0.lbm_steps_per_tick, 1);
        assert!(config.0.lbm_enabled);
    }

    #[test]
    fn is_active_gas_identifies_steam() {
        assert!(is_active_gas(MaterialId::STEAM));
        assert!(!is_active_gas(MaterialId::AIR));
        assert!(!is_active_gas(MaterialId::WATER));
        assert!(!is_active_gas(MaterialId::STONE));
    }

    /// Helper: compute surface heating delta-T for testing
    fn compute_surface_heating(
        solar_constant: f32,
        insolation_factor: f32,
        albedo: f32,
        dt: f32,
        density: f32,
        specific_heat: f32,
    ) -> f32 {
        if density <= 0.0 || specific_heat <= 0.0 {
            return 0.0;
        }
        let q_absorbed = solar_constant * insolation_factor * (1.0 - albedo);
        q_absorbed * dt / (density * specific_heat)
    }

    #[test]
    fn solar_heating_warms_surface_voxel() {
        // Test parameters
        let solar_constant = 1361.0; // W/m²
        let insolation = 1.0; // Peak sun
        let albedo_dark = 0.1;
        let albedo_bright = 0.9;
        let dt = 1.0; // 1 second
        let density_stone = 2700.0; // kg/m³
        let cp_stone = 790.0; // J/(kg·K)

        let delta_t_dark = compute_surface_heating(
            solar_constant,
            insolation,
            albedo_dark,
            dt,
            density_stone,
            cp_stone,
        );
        let delta_t_bright = compute_surface_heating(
            solar_constant,
            insolation,
            albedo_bright,
            dt,
            density_stone,
            cp_stone,
        );

        // Dark surface should warm significantly more than bright surface
        assert!(delta_t_dark > delta_t_bright);
        assert!(delta_t_dark > 0.0);
        assert!(delta_t_bright > 0.0);

        // Check order of magnitude is reasonable
        // Q = 1361 × (1 - 0.9) × 1s = ~136.1 J/m²
        // ΔT = 136.1 / (2700 × 790) ≈ 6.38e-5 K for bright surface
        assert!((delta_t_bright - 6.38e-5).abs() < 1e-5);
    }

    #[test]
    fn solar_heating_respects_albedo() {
        let solar_constant = 1000.0;
        let insolation = 0.8;
        let dt = 10.0;
        let density = 1000.0;
        let cp = 1000.0;

        // Low albedo (dark) absorbs more
        let dt_low_albedo =
            compute_surface_heating(solar_constant, insolation, 0.1, dt, density, cp);
        // High albedo (bright) reflects more, absorbs less
        let dt_high_albedo =
            compute_surface_heating(solar_constant, insolation, 0.9, dt, density, cp);

        assert!(dt_low_albedo > dt_high_albedo);
        // Low albedo absorbs 90%, high albedo absorbs 10%
        // Ratio should be 9:1
        assert!((dt_low_albedo / dt_high_albedo - 9.0).abs() < 0.1);
    }

    #[test]
    fn solar_heating_zero_at_night() {
        let dt = compute_surface_heating(1361.0, 0.0, 0.3, 1.0, 2700.0, 790.0);
        assert_eq!(dt, 0.0);
    }
}
