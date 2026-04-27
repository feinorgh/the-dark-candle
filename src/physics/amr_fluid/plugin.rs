// AmrFluidPlugin: Bevy ECS integration for the AMR Navier-Stokes fluid solver.
//
// Manages per-chunk FluidGrids as a resource, runs the fluid simulation on
// FixedUpdate, and syncs results back to chunks.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::data::FluidConfig;
use crate::world::chunk::{Chunk, ChunkCoord};
use crate::world::voxel::MaterialId;

use super::injection;
use super::step;
use super::types::FluidGrid;

/// Resource: maps chunk coordinates to their fluid simulation state.
/// Only chunks containing fluid voxels have an entry.
#[derive(Resource, Default)]
pub struct FluidState {
    grids: HashMap<ChunkCoord, FluidGrid>,
}

impl FluidState {
    pub fn get(&self, coord: &ChunkCoord) -> Option<&FluidGrid> {
        self.grids.get(coord)
    }

    pub fn get_mut(&mut self, coord: &ChunkCoord) -> Option<&mut FluidGrid> {
        self.grids.get_mut(coord)
    }

    pub fn insert(&mut self, coord: ChunkCoord, grid: FluidGrid) {
        self.grids.insert(coord, grid);
    }

    pub fn remove(&mut self, coord: &ChunkCoord) -> Option<FluidGrid> {
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

    /// Iterate over all chunk coordinates with active fluid grids.
    pub fn coords(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.grids.keys().copied()
    }
}

#[derive(Resource, Default)]
pub struct FluidConfigRes(pub FluidConfig);

/// Tick counter for the fluid simulation (monotonically increasing).
#[derive(Resource, Default)]
pub struct FluidTick(pub u64);

/// Plugin that adds AMR Navier-Stokes fluid simulation to the physics pipeline.
pub struct AmrFluidPlugin;

impl Plugin for AmrFluidPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FluidState>()
            .init_resource::<FluidConfigRes>()
            .init_resource::<FluidTick>()
            .add_systems(
                FixedUpdate,
                (
                    init_fluid_grids,
                    injection::seed_river_flow,
                    injection::inject_river_sources,
                    amr_fluid_step,
                    cleanup_empty_fluid_grids,
                )
                    .chain(),
            );
    }
}

/// Initialize FluidGrids for newly loaded chunks that contain fluid voxels.
fn init_fluid_grids(chunks: Query<&Chunk, Added<Chunk>>, mut fluid_state: ResMut<FluidState>) {
    for chunk in chunks.iter() {
        if fluid_state.contains(&chunk.coord) {
            continue;
        }

        let has_fluid = chunk.voxels().iter().any(|v| is_fluid_material(v.material));
        if !has_fluid {
            continue;
        }

        let grid = FluidGrid::from_chunk(chunk, None);
        fluid_state.insert(chunk.coord, grid);
    }
}

/// Run one AMR fluid simulation step for all active fluid chunks.
#[allow(clippy::too_many_arguments)]
fn amr_fluid_step(
    config: Res<FluidConfigRes>,
    mut fluid_state: ResMut<FluidState>,
    mut tick: ResMut<FluidTick>,
    time: Res<Time>,
    lod_config: Res<crate::physics::PhysicsLodConfig>,
    camera_q: Query<&Transform, With<Camera3d>>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    let coords: Vec<ChunkCoord> = fluid_state.grids.keys().cloned().collect();

    let camera_chunk = camera_q.iter().next().map(|t| {
        ChunkCoord::from_voxel_pos(
            t.translation.x as i32,
            t.translation.y as i32,
            t.translation.z as i32,
        )
    });

    for coord in coords {
        // Physics LOD: skip chunks outside the active radius.
        if let Some(ref cam) = camera_chunk
            && !lod_config.is_active(&coord, cam)
        {
            continue;
        }

        let Some(grid) = fluid_state.grids.get_mut(&coord) else {
            continue;
        };

        step::fluid_step(grid, None, &config.0, dt);
    }

    tick.0 += 1;
}

/// Remove FluidGrid entries for chunks that no longer contain any fluid.
fn cleanup_empty_fluid_grids(mut fluid_state: ResMut<FluidState>) {
    fluid_state.grids.retain(|_, grid| grid.has_fluid());
}

fn is_fluid_material(mat: MaterialId) -> bool {
    matches!(mat.0, 3 | 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluid_state_resource_operations() {
        let mut state = FluidState::default();
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };

        assert!(state.is_empty());
        assert!(!state.contains(&coord));

        state.insert(coord, FluidGrid::new_empty(32));
        assert_eq!(state.len(), 1);
        assert!(state.contains(&coord));

        assert!(state.get(&coord).is_some());
        assert!(state.get_mut(&coord).is_some());

        state.remove(&coord);
        assert!(state.is_empty());
    }
}
