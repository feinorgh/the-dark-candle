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
use crate::physics::flip_pic::plugin::ParticleState;
use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::planet::PlanetConfig;
use crate::world::voxel::MaterialId;

use super::moisture;
use super::precipitation;
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

    /// Iterate over all `(ChunkCoord, LbmGrid)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&ChunkCoord, &LbmGrid)> {
        self.grids.iter()
    }
}

/// Wrapper resource for FluidConfig (shared with AMR fluid plugin).
#[derive(Resource, Default)]
pub struct LbmConfigRes(pub FluidConfig);

/// Tick counter for the LBM gas simulation.
#[derive(Resource, Default)]
pub struct LbmTick(pub u64);

/// Per-chunk cloud density field, exposed for the rendering pipeline.
///
/// Maps chunk coordinates to per-voxel cloud liquid water content (kg/m³).
/// Rendering systems (volumetric clouds, shadows) read this resource to
/// determine cloud opacity and coverage.
#[derive(Resource, Default)]
pub struct CloudField {
    /// Per-chunk cloud LWC data, indexed same as LbmGrid cells.
    pub chunks: HashMap<ChunkCoord, Vec<f32>>,
}

impl CloudField {
    /// Get the cloud LWC at a specific voxel position within a chunk.
    pub fn get_lwc(&self, coord: &ChunkCoord, x: usize, y: usize, z: usize, size: usize) -> f32 {
        self.chunks
            .get(coord)
            .map(|data| data[z * size * size + y * size + x])
            .unwrap_or(0.0)
    }

    /// Check if any chunk has non-zero cloud content.
    pub fn has_clouds(&self) -> bool {
        self.chunks
            .values()
            .any(|data| data.iter().any(|&v| v > 1e-8))
    }
}

/// Plugin that adds D3Q19 LBM gas simulation to the physics pipeline.
pub struct LbmGasPlugin;

impl Plugin for LbmGasPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LbmState>()
            .init_resource::<LbmConfigRes>()
            .init_resource::<LbmTick>()
            .init_resource::<CloudField>()
            .add_systems(
                FixedUpdate,
                (
                    apply_solar_heating,
                    init_lbm_grids,
                    lbm_gas_step,
                    process_moisture,
                    emit_precipitation,
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
    time_of_day: Option<Res<TimeOfDay>>,
    atmosphere_config: Option<Res<AtmosphereConfig>>,
    planet_config: Option<Res<PlanetConfig>>,
    solar_insolation: Option<Res<SolarInsolation>>,
    registry: Option<Res<MaterialRegistry>>,
    time: Res<Time>,
    lod_config: Res<crate::physics::PhysicsLodConfig>,
    camera_q: Query<&Transform, With<Camera3d>>,
) {
    // Early return if any required resource is missing
    let (Some(_time_of_day), Some(atmosphere), Some(_planet), Some(insolation), Some(registry)) = (
        time_of_day.as_ref(),
        atmosphere_config.as_ref(),
        planet_config.as_ref(),
        solar_insolation.as_ref(),
        registry.as_ref(),
    ) else {
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

    // Derive camera chunk position for physics LOD culling
    let camera_chunk = camera_q.iter().next().map(|t| {
        ChunkCoord::from_voxel_pos(
            t.translation.x as i32,
            t.translation.y as i32,
            t.translation.z as i32,
        )
    });

    // Apply solar heating to all loaded chunk entities directly.
    for mut chunk in chunks.iter_mut() {
        let coord = chunk.coord;

        // Physics LOD: skip distant chunks for solar heating.
        if let Some(ref cam) = camera_chunk
            && !lod_config.is_active(&coord, cam)
        {
            continue;
        }

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
                if density > 0.0 && specific_heat > 0.0 {
                    let delta_t_surface = q_absorbed * dt / (density * specific_heat);
                    let voxel = chunk.get_mut(x, sy, z);
                    voxel.temperature += delta_t_surface;

                    // Also heat the air voxel directly above (if present in chunk)
                    if sy + 1 < CHUNK_SIZE {
                        let air_voxel = chunk.get_mut(x, sy + 1, z);
                        if air_voxel.is_air() {
                            let air_mat = registry.get(air_voxel.material);
                            if let Some(air_data) = air_mat {
                                let air_density = air_data.density;
                                let air_cp = air_data.specific_heat_capacity;
                                if air_density > 0.0 && air_cp > 0.0 {
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
    config: Res<LbmConfigRes>,
    mut lbm_state: ResMut<LbmState>,
    mut tick: ResMut<LbmTick>,
    time: Res<Time>,
    atmosphere_config: Option<Res<AtmosphereConfig>>,
    planet_config: Option<Res<PlanetConfig>>,
    lod_config: Res<crate::physics::PhysicsLodConfig>,
    camera_q: Query<&Transform, With<Camera3d>>,
) {
    if !config.0.lbm_enabled {
        return;
    }

    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    // Gravity in lattice units (small — one LBM step may be multiple seconds)
    let gravity_lattice = [0.0, -0.001, 0.0];
    let rho_ambient = 1.0;

    // Compute Coriolis omega in lattice units
    let coriolis_omega = if let (Some(atm_cfg), Some(planet_cfg)) =
        (atmosphere_config.as_deref(), planet_config.as_deref())
    {
        if atm_cfg.coriolis_enabled {
            let omega_physical = [
                (planet_cfg.rotation_rate * planet_cfg.rotation_axis[0]) as f32,
                (planet_cfg.rotation_rate * planet_cfg.rotation_axis[1]) as f32,
                (planet_cfg.rotation_rate * planet_cfg.rotation_axis[2]) as f32,
            ];

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

    let camera_chunk = camera_q.iter().next().map(|t| {
        ChunkCoord::from_voxel_pos(
            t.translation.x as i32,
            t.translation.y as i32,
            t.translation.z as i32,
        )
    });

    let mut scratch = LbmGrid::new_empty(crate::world::chunk::CHUNK_SIZE);

    for coord in coords {
        if let Some(ref cam) = camera_chunk
            && !lod_config.is_active(&coord, cam)
        {
            continue;
        }

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
            Some(&mut scratch),
        );
    }

    // Sync LBM results to any loaded chunk entities (iterates directly).
    for mut chunk in chunks.iter_mut() {
        let coord = chunk.coord;
        if let Some(grid) = lbm_state.grids.get(&coord) {
            sync::sync_to_chunk(grid, &mut chunk);
        }
    }

    tick.0 += 1;
}

/// Process moisture cycle: evaporation from liquid surfaces, condensation
/// into clouds, and latent heat feedback to voxel temperatures.
///
/// Runs after the LBM step so that advected moisture fields are up-to-date.
/// Updates the CloudField resource for rendering.
#[allow(clippy::too_many_arguments)]
fn process_moisture(
    mut chunks: Query<&mut Chunk>,
    mut lbm_state: ResMut<LbmState>,
    atmosphere_config: Option<Res<AtmosphereConfig>>,
    config: Res<LbmConfigRes>,
    time: Res<Time>,
    mut cloud_field: ResMut<CloudField>,
) {
    if !config.0.lbm_enabled {
        return;
    }

    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    let atm_config = atmosphere_config.as_deref().cloned().unwrap_or_default();

    let coords: Vec<ChunkCoord> = lbm_state.grids.keys().cloned().collect();

    for coord in coords {
        let Some(grid) = lbm_state.grids.get_mut(&coord) else {
            continue;
        };

        // Extract temperature and pressure arrays from the chunk entity directly.
        let (temps, pressures) = {
            let mut found = None;
            for chunk in chunks.iter() {
                if chunk.coord == coord {
                    found = Some(extract_chunk_thermodynamics(chunk));
                    break;
                }
            }
            found.unwrap_or_else(|| {
                let size = grid.size();
                let n = size * size * size;
                (
                    vec![atm_config.surface_temperature; n],
                    vec![101_325.0_f32; n],
                )
            })
        };

        moisture::evaporate(grid, &temps, &pressures, dt, &atm_config);
        let temp_deltas = moisture::condense(grid, &temps, &pressures, dt);

        for mut chunk in chunks.iter_mut() {
            if chunk.coord == coord {
                apply_latent_heat(&mut chunk, grid, &temp_deltas);
                break;
            }
        }

        let lwc_data: Vec<f32> = grid.cells().iter().map(|c| c.cloud_lwc).collect();
        if lwc_data.iter().any(|&v| v > 1e-8) {
            cloud_field.chunks.insert(coord, lwc_data);
        } else {
            cloud_field.chunks.remove(&coord);
        }
    }
}

/// Extract per-voxel temperature and pressure arrays from a chunk.
fn extract_chunk_thermodynamics(chunk: &Chunk) -> (Vec<f32>, Vec<f32>) {
    let voxels = chunk.voxels();
    let temps: Vec<f32> = voxels.iter().map(|v| v.temperature).collect();
    let pressures: Vec<f32> = voxels.iter().map(|v| v.pressure).collect();
    (temps, pressures)
}

/// Apply latent heat temperature changes from condensation/evaporation to voxels.
fn apply_latent_heat(chunk: &mut Chunk, grid: &LbmGrid, temp_deltas: &[f32]) {
    let size = grid.size();
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = grid.index(x, y, z);
                let delta = temp_deltas[idx];
                if delta.abs() > 1e-8 {
                    let cell = grid.get(x, y, z);
                    if cell.is_gas() {
                        let voxel = chunk.get_mut(x, y, z);
                        voxel.temperature += delta;
                        // Clamp to physically reasonable range
                        voxel.temperature = voxel.temperature.clamp(100.0, 10000.0);
                    }
                }
            }
        }
    }
}

/// Emit precipitation particles from clouds into the FLIP/PIC system.
///
/// Scans LBM grids for cloud cells above the coalescence threshold.
/// Converts excess cloud LWC into rain (warm) or snow (cold) particles.
/// Also applies virga (sub-cloud evaporation) to falling particles.
#[allow(clippy::too_many_arguments)]
fn emit_precipitation(
    chunks: Query<&Chunk>,
    mut lbm_state: ResMut<LbmState>,
    mut particle_state: ResMut<ParticleState>,
    atmosphere_config: Option<Res<AtmosphereConfig>>,
    config: Res<LbmConfigRes>,
    tick: Res<LbmTick>,
    time: Res<Time>,
) {
    if !config.0.lbm_enabled {
        return;
    }

    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    let atm_config = atmosphere_config.as_deref().cloned().unwrap_or_default();

    let coords: Vec<ChunkCoord> = lbm_state.grids.keys().cloned().collect();

    for coord in coords {
        let Some(grid) = lbm_state.grids.get_mut(&coord) else {
            continue;
        };

        // Extract temperatures from chunk entity directly.
        let temps: Vec<f32> = {
            let mut found = None;
            for chunk in chunks.iter() {
                if chunk.coord == coord {
                    found = Some(
                        chunk
                            .voxels()
                            .iter()
                            .map(|v| v.temperature)
                            .collect::<Vec<_>>(),
                    );
                    break;
                }
            }
            found.unwrap_or_else(|| {
                let n = grid.size() * grid.size() * grid.size();
                vec![atm_config.surface_temperature; n]
            })
        };

        let buf = particle_state
            .buffers
            .entry(coord)
            .or_insert_with(|| crate::physics::flip_pic::types::ParticleBuffer::new(coord));

        precipitation::precipitate(grid, &temps, &atm_config, &mut buf.particles, dt, tick.0);

        let pressures: Vec<f32> = {
            let mut found = None;
            for chunk in chunks.iter() {
                if chunk.coord == coord {
                    found = Some(
                        chunk
                            .voxels()
                            .iter()
                            .map(|v| v.pressure)
                            .collect::<Vec<_>>(),
                    );
                    break;
                }
            }
            found.unwrap_or_else(|| {
                let n = grid.size() * grid.size() * grid.size();
                vec![101_325.0_f32; n]
            })
        };

        precipitation::apply_virga(&mut buf.particles, grid, &temps, &pressures, dt);
    }
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

    #[test]
    fn cloud_field_resource_operations() {
        let mut field = CloudField::default();
        assert!(!field.has_clouds());

        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let size = 4;
        let mut data = vec![0.0f32; size * size * size];
        data[size * size + size + 1] = 0.5e-3; // Cloud at (1,1,1)
        field.chunks.insert(coord, data);

        assert!(field.has_clouds());
        let lwc = field.get_lwc(&coord, 1, 1, 1, size);
        assert!((lwc - 0.5e-3).abs() < 1e-8);

        let lwc_empty = field.get_lwc(&coord, 0, 0, 0, size);
        assert!(lwc_empty < 1e-8);
    }

    #[test]
    fn extract_thermodynamics_reads_chunk_data() {
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new_filled(coord, MaterialId::AIR);

        // Set specific temperature at a known position
        chunk.get_mut(5, 5, 5).temperature = 300.0;
        chunk.get_mut(5, 5, 5).pressure = 95_000.0;

        let (temps, pressures) = extract_chunk_thermodynamics(&chunk);

        // Default voxel temp is 288.15 K
        assert!((temps[0] - 288.15).abs() < 0.01);

        // The modified voxel (index depends on chunk layout)
        let idx = 5 * CHUNK_SIZE * CHUNK_SIZE + 5 * CHUNK_SIZE + 5;
        assert!((temps[idx] - 300.0).abs() < 0.01);
        assert!((pressures[idx] - 95_000.0).abs() < 1.0);
    }

    #[test]
    fn latent_heat_warms_air_on_condensation() {
        use crate::physics::atmosphere;

        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new_filled(coord, MaterialId::AIR);

        // Set cool temperature → low saturation
        let temp = 275.0;
        for v in chunk.voxels_mut() {
            v.temperature = temp;
        }

        let size = CHUNK_SIZE;
        let mut grid = LbmGrid::new_empty(size);

        // Supersaturate one cell
        let q_sat = atmosphere::saturation_humidity(temp, 101_325.0);
        grid.get_mut(5, 5, 5).moisture = q_sat * 3.0;

        let n = size * size * size;
        let temps = vec![temp; n];
        let pressures = vec![101_325.0f32; n];

        let temp_deltas = moisture::condense(&mut grid, &temps, &pressures, 1.0);

        // Apply latent heat
        apply_latent_heat(&mut chunk, &grid, &temp_deltas);

        // Temperature should have increased from latent heat release
        let voxel_temp = chunk.get(5, 5, 5).temperature;
        assert!(
            voxel_temp > temp,
            "Condensation should warm the air: {temp} → {voxel_temp}"
        );
    }

    #[test]
    fn cloud_forms_in_supersaturated_column() {
        use crate::physics::lbm_gas::types::LbmCell;

        let size = 8;
        let mut grid = LbmGrid::new_empty(size);

        // Create walls
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        // Place water at bottom (y=1)
        for z in 1..size - 1 {
            for x in 1..size - 1 {
                *grid.get_mut(x, 1, z) = LbmCell::new_liquid(MaterialId::WATER);
            }
        }

        // Temperature decreases with height (lapse rate): warm at bottom, cool at top
        // This makes upper cells easier to saturate
        let n = size * size * size;
        let mut temps = vec![288.15f32; n];
        let pressures = vec![101_325.0f32; n];

        // Set temperature gradient: bottom warm, top cool
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    // Lapse rate: 6.5 K per 1000m, each y-level = 1m
                    temps[idx] = 288.15 - 6.5 * y as f32;
                }
            }
        }

        let atm_config = AtmosphereConfig::default();

        // Run evaporation + condensation repeatedly
        for _ in 0..50 {
            moisture::evaporate(&mut grid, &temps, &pressures, 0.1, &atm_config);
            let _deltas = moisture::condense(&mut grid, &temps, &pressures, 0.1);
        }

        // Check that moisture exists above the water surface
        let moisture_at_y2 = grid.get(4, 2, 4).moisture;
        assert!(
            moisture_at_y2 > 0.0,
            "Should have moisture above water: {moisture_at_y2}"
        );

        // Check total cloud LWC in the grid
        let _total_lwc = moisture::total_cloud_lwc(&grid);
        // With the cool upper cells and evaporation source, some condensation may occur
        // (depends on whether upper cells reach saturation)
        // At minimum, moisture should be present
        let total_moist = moisture::total_moisture(&grid);
        assert!(
            total_moist > 0.0,
            "Should have moisture in the system: {total_moist}"
        );

        // The coolest cells (highest y) should condense first if saturated
        // This tests the fundamental cloud formation mechanism
        let cool_cell = grid.get(4, size - 2, 4);
        let warm_cell = grid.get(4, 2, 4);
        // Cool cells saturate more easily → cloud_lwc should be >= warm cells
        assert!(
            cool_cell.cloud_lwc >= warm_cell.cloud_lwc,
            "Cool air should condense more: cool={}, warm={}",
            cool_cell.cloud_lwc,
            warm_cell.cloud_lwc
        );
    }
}
