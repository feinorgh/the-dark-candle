// LbmGasPlugin: Bevy ECS integration for the D3Q19 Lattice Boltzmann gas solver.
//
// Manages per-chunk LbmGrids as a resource, runs the LBM simulation on
// FixedUpdate, and syncs results back to chunks. Parallels the AMR fluid
// plugin architecture.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::data::FluidConfig;
use crate::world::chunk::{Chunk, ChunkCoord};
use crate::world::chunk_manager::ChunkMap;
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
                (init_lbm_grids, lbm_gas_step, cleanup_empty_lbm_grids).chain(),
            );
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
fn lbm_gas_step(
    mut chunks: Query<&mut Chunk>,
    chunk_map: Option<Res<ChunkMap>>,
    config: Res<LbmConfigRes>,
    mut lbm_state: ResMut<LbmState>,
    mut tick: ResMut<LbmTick>,
    time: Res<Time>,
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

    let coords: Vec<ChunkCoord> = lbm_state.grids.keys().cloned().collect();
    let n_steps = config.0.lbm_steps_per_tick.max(1);

    for coord in coords {
        let Some(grid) = lbm_state.grids.get_mut(&coord) else {
            continue;
        };

        step::lbm_step_n(grid, &config.0, gravity_lattice, rho_ambient, n_steps);

        if let Some(entity) = chunk_map.get(&coord) {
            if let Ok(mut chunk) = chunks.get_mut(entity) {
                sync::sync_to_chunk(grid, &mut chunk);
            }
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
}
