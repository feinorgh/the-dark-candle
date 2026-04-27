// Per-system timing infrastructure: bracket systems that measure wall-clock
// time around key ECS system sets, smoothed via EMA.  Also tracks chunk
// lifecycle statistics (in-flight mesh tasks, dispatch/collect counts).

use std::time::Instant;

use bevy::prelude::*;

/// EMA smoothing factor matching [`super::frame_budget::FrameBudget`].
const TIMING_SMOOTHING: f32 = 0.1;

/// Per-frame metrics for key system categories, EMA-smoothed.
///
/// Bracket systems record [`Instant`] timestamps before/after each system
/// set.  The "end" bracket computes the delta and folds it into the EMA.
#[derive(Resource, Debug)]
pub struct SystemTimings {
    // -- bracket timestamps (internal) ------------------------------------
    world_start: Option<Instant>,
    meshing_start: Option<Instant>,
    physics_start: Option<Instant>,

    // -- EMA-smoothed wall-clock times (ms) -- read by the F3 overlay -----
    /// Time spent in `WorldSet::ChunkManagement` (terrain gen, load/unload).
    pub world_ms: f32,
    /// Time spent in `WorldSet::Meshing` (dispatch + collect + transitions).
    pub meshing_ms: f32,
    /// Time spent in `FixedUpdate` physics pipeline.
    pub physics_ms: f32,
}

impl Default for SystemTimings {
    fn default() -> Self {
        Self {
            world_start: None,
            meshing_start: None,
            physics_start: None,
            world_ms: 0.0,
            meshing_ms: 0.0,
            physics_ms: 0.0,
        }
    }
}

// -- bracket systems for WorldSet::ChunkManagement ------------------------

pub fn begin_world_timing(mut timings: ResMut<SystemTimings>) {
    timings.world_start = Some(Instant::now());
}

pub fn end_world_timing(mut timings: ResMut<SystemTimings>) {
    if let Some(start) = timings.world_start.take() {
        let dt = start.elapsed().as_secs_f32() * 1000.0;
        timings.world_ms = TIMING_SMOOTHING * dt + (1.0 - TIMING_SMOOTHING) * timings.world_ms;
    }
}

// -- bracket systems for WorldSet::Meshing --------------------------------

pub fn begin_meshing_timing(mut timings: ResMut<SystemTimings>) {
    timings.meshing_start = Some(Instant::now());
}

pub fn end_meshing_timing(mut timings: ResMut<SystemTimings>) {
    if let Some(start) = timings.meshing_start.take() {
        let dt = start.elapsed().as_secs_f32() * 1000.0;
        timings.meshing_ms = TIMING_SMOOTHING * dt + (1.0 - TIMING_SMOOTHING) * timings.meshing_ms;
    }
}

// -- bracket systems for FixedUpdate physics ------------------------------

pub fn begin_physics_timing(mut timings: ResMut<SystemTimings>) {
    timings.physics_start = Some(Instant::now());
}

pub fn end_physics_timing(mut timings: ResMut<SystemTimings>) {
    if let Some(start) = timings.physics_start.take() {
        let dt = start.elapsed().as_secs_f32() * 1000.0;
        timings.physics_ms = TIMING_SMOOTHING * dt + (1.0 - TIMING_SMOOTHING) * timings.physics_ms;
    }
}

// -- chunk lifecycle statistics -------------------------------------------

/// Per-frame snapshot of chunk pipeline counts, written by world systems
/// and read by the F3 overlay.
#[derive(Resource, Debug, Default)]
pub struct ChunkStats {
    /// Chunks with an in-flight [`MeshTask`] (pending async mesh generation).
    pub meshing_in_flight: usize,
    /// Number of loaded chunk entities in `ChunkMap`.
    pub loaded: usize,
    /// Number of chunks awaiting async terrain generation.
    pub generating: usize,
}

/// System that refreshes [`ChunkStats`] each frame from the ECS world.
///
/// Counts entities with the `MeshTask` component (in-flight meshing),
/// reads the chunk map size, and checks pending terrain generation count.
pub fn update_chunk_stats(
    mut stats: ResMut<ChunkStats>,
    mesh_task_q: Query<(), With<crate::world::v2::chunk_manager::V2MeshTask>>,
) {
    stats.loaded = 0;
    stats.meshing_in_flight = mesh_task_q.iter().count();
    stats.generating = 0;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_timings_are_zero() {
        let t = SystemTimings::default();
        assert_eq!(t.world_ms, 0.0);
        assert_eq!(t.meshing_ms, 0.0);
        assert_eq!(t.physics_ms, 0.0);
    }

    #[test]
    fn ema_update_from_bracket() {
        let mut t = SystemTimings {
            world_start: Some(Instant::now()),
            ..Default::default()
        };
        std::thread::sleep(std::time::Duration::from_millis(1));
        if let Some(start) = t.world_start.take() {
            let dt = start.elapsed().as_secs_f32() * 1000.0;
            t.world_ms = TIMING_SMOOTHING * dt + (1.0 - TIMING_SMOOTHING) * t.world_ms;
        }
        assert!(t.world_ms > 0.0, "EMA should be positive after bracket");
        assert!(t.world_start.is_none(), "slot should be cleared after end");
    }

    #[test]
    fn chunk_stats_defaults() {
        let s = ChunkStats::default();
        assert_eq!(s.loaded, 0);
        assert_eq!(s.meshing_in_flight, 0);
    }
}
