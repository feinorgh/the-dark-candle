// FLIP/PIC ECS plugin integration.
//
// Manages per-chunk ParticleBuffers as a resource, runs the FLIP simulation on
// FixedUpdate, and cleans up empty buffers. Parallels the LBM gas plugin
// architecture.

use bevy::prelude::*;
use std::collections::HashMap;

use super::step;
use super::types::{AccumulationGrid, ParticleBuffer};
use crate::data::FluidConfig;
use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::chunk_manager::ChunkMap;
use crate::world::voxel::MaterialId;

/// Water boiling point (K).
const WATER_BOILING_POINT: f32 = 373.15;
/// Water triple point (K).
const WATER_TRIPLE_POINT: f32 = 273.16;

/// Resource: maps chunk coordinates to their particle simulation state.
/// Only chunks with active particles or emission sources have an entry.
#[derive(Resource, Default)]
pub struct ParticleState {
    pub buffers: HashMap<ChunkCoord, ParticleBuffer>,
    pub accumulations: HashMap<ChunkCoord, AccumulationGrid>,
}

/// Wrapper resource for FluidConfig (shared with other fluid plugins).
#[derive(Resource, Default)]
pub struct FlipConfigRes(pub FluidConfig);

/// Tick counter for the FLIP/PIC simulation.
#[derive(Resource, Default)]
pub struct FlipTick(pub u64);

/// Plugin that adds FLIP/PIC particle simulation to the physics pipeline.
pub struct FlipPicPlugin;

impl Plugin for FlipPicPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleState>()
            .init_resource::<FlipConfigRes>()
            .init_resource::<FlipTick>()
            .add_systems(
                FixedUpdate,
                (
                    init_particle_buffers,
                    flip_particle_step,
                    cleanup_empty_buffers,
                )
                    .chain(),
            );
    }
}

/// Check whether a chunk has voxels that can emit particles.
fn has_emitting_surfaces(chunk: &Chunk) -> bool {
    chunk.voxels().iter().any(|v| {
        (v.material == MaterialId::WATER && v.temperature > WATER_BOILING_POINT)
            || (v.material == MaterialId::ICE && v.temperature > WATER_TRIPLE_POINT)
    })
}

/// Initialize ParticleBuffers for newly loaded chunks with emitting surfaces.
fn init_particle_buffers(chunks: Query<&Chunk, Added<Chunk>>, mut state: ResMut<ParticleState>) {
    for chunk in chunks.iter() {
        if state.buffers.contains_key(&chunk.coord) {
            continue;
        }

        if !has_emitting_surfaces(chunk) {
            continue;
        }

        state
            .buffers
            .insert(chunk.coord, ParticleBuffer::new(chunk.coord));
        state
            .accumulations
            .insert(chunk.coord, AccumulationGrid::new(CHUNK_SIZE));
    }
}

/// Run FLIP/PIC particle simulation for all active chunks.
fn flip_particle_step(
    mut chunks: Query<&mut Chunk>,
    chunk_map: Option<Res<ChunkMap>>,
    config: Res<FlipConfigRes>,
    mut state: ResMut<ParticleState>,
    mut tick: ResMut<FlipTick>,
    time: Res<Time>,
) {
    if !config.0.flip_enabled {
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

    let n_substeps = config.0.flip_substeps.max(1);

    let coords: Vec<ChunkCoord> = state.buffers.keys().cloned().collect();

    // Destructure to allow simultaneous mutable borrows of both HashMap fields.
    let ParticleState {
        buffers,
        accumulations,
    } = &mut *state;

    for coord in coords {
        let Some(entity) = chunk_map.get(&coord) else {
            continue;
        };
        let Ok(mut chunk) = chunks.get_mut(entity) else {
            continue;
        };

        let (Some(buf), Some(accum)) = (buffers.get_mut(&coord), accumulations.get_mut(&coord))
        else {
            continue;
        };

        step::flip_step_n(
            &mut buf.particles,
            chunk.voxels_mut(),
            accum,
            &config.0,
            dt,
            tick.0,
            n_substeps,
        );
    }

    tick.0 += 1;
}

/// Remove ParticleBuffers that are empty and have no emission sources.
fn cleanup_empty_buffers(
    chunks: Query<&Chunk>,
    chunk_map: Option<Res<ChunkMap>>,
    mut state: ResMut<ParticleState>,
) {
    let chunk_map = match chunk_map {
        Some(cm) => cm,
        None => return,
    };

    state.buffers.retain(|coord, buf| {
        if !buf.is_empty() {
            return true;
        }
        // Check if the chunk still has emitting surfaces.
        if let Some(entity) = chunk_map.get(coord)
            && let Ok(chunk) = chunks.get(entity)
        {
            return has_emitting_surfaces(chunk);
        }
        false
    });

    // Collect coords to remove, then remove them (avoids simultaneous borrow).
    let keep: Vec<ChunkCoord> = state.buffers.keys().cloned().collect();
    state.accumulations.retain(|coord, _| keep.contains(coord));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_registers_resources() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(FlipPicPlugin);

        assert!(app.world().get_resource::<ParticleState>().is_some());
        assert!(app.world().get_resource::<FlipConfigRes>().is_some());
        assert!(app.world().get_resource::<FlipTick>().is_some());
    }

    #[test]
    fn flip_config_res_wraps_fluid_config() {
        let config = FlipConfigRes::default();
        assert!(config.0.flip_enabled);
        assert!((config.0.flip_ratio - 0.97).abs() < 1e-6);
        assert!((config.0.flip_deposit_velocity - 0.5).abs() < 1e-6);
        assert_eq!(config.0.flip_pressure_iterations, 20);
    }

    #[test]
    fn particle_state_is_empty_by_default() {
        let state = ParticleState::default();
        assert!(state.buffers.is_empty());
        assert!(state.accumulations.is_empty());
    }

    #[test]
    fn flip_tick_starts_at_zero() {
        let tick = FlipTick::default();
        assert_eq!(tick.0, 0);
    }
}
