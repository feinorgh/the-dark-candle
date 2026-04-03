// Cross-model fluid coupling: AMR Navier-Stokes ↔ LBM D3Q19 ↔ FLIP/PIC particles.
//
// Each physics solver owns a per-chunk simulation grid that syncs with the shared
// Chunk voxel array. The coupling layer bridges scenarios that fall between the
// individual sync boundaries:
//
//   1. Virga moisture return (FLIP → LBM): handled in precipitation::apply_virga.
//      Evaporated particle mass is injected back into the LBM moisture field so
//      total atmospheric water is conserved.
//
//   2. AMR sync-from-chunk per step (FLIP/Chemistry → AMR): handled in
//      amr_fluid_step() which calls sync_from_chunk() before each step. FLIP
//      particle deposits and chemistry state transitions (ice melting, water
//      boiling) are reflected in the FluidGrid within the same FixedUpdate tick.
//
//   3. Lazy FluidGrid initialisation (FLIP → AMR): FLIP particles deposit liquid
//      voxels into chunks that previously had no fluid. The init_fluid_grids system
//      in AmrFluidPlugin only fires on Added<Chunk>, so it misses in-place liquid
//      creation. This module provides lazy_init_fluid_grids which periodically
//      scans loaded chunks and creates FluidGrids wherever new liquid is found.
//
// Ordering: CouplingPlugin systems run in FixedUpdate, after the FLIP step, so
// deposited voxels are visible when lazy init scans.

use bevy::prelude::*;

use crate::physics::amr_fluid::plugin::FluidState;
use crate::physics::amr_fluid::sync;
use crate::physics::amr_fluid::types::FluidGrid;
use crate::world::chunk::Chunk;
use crate::world::voxel::MaterialId;

/// How often (in seconds) to scan for new liquid voxels not yet tracked by AMR.
const LAZY_INIT_INTERVAL_SECS: f32 = 1.5;

/// Timer driving the periodic lazy FluidGrid initialisation scan.
#[derive(Resource, Debug)]
pub struct LazyCouplingTimer {
    timer: Timer,
}

impl Default for LazyCouplingTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(LAZY_INIT_INTERVAL_SECS, TimerMode::Repeating),
        }
    }
}

/// Scan loaded chunks for liquid voxels not yet tracked by the AMR fluid solver.
///
/// FLIP particles deposit water or ice voxels into chunks at runtime.
/// `AmrFluidPlugin::init_fluid_grids` only runs on `Added<Chunk>`, so it
/// cannot initialise a FluidGrid for a chunk that already existed when liquid
/// first appeared.  This system bridges that gap by periodically re-scanning.
///
/// Runs every `LAZY_INIT_INTERVAL_SECS` seconds (default 1.5 s) to keep
/// per-frame cost negligible.
fn lazy_init_fluid_grids(
    time: Res<Time>,
    mut timer: ResMut<LazyCouplingTimer>,
    chunks: Query<&Chunk>,
    mut fluid_state: ResMut<FluidState>,
) {
    timer.timer.tick(time.delta());
    if !timer.timer.just_finished() {
        return;
    }

    for chunk in chunks.iter() {
        if fluid_state.contains(&chunk.coord) {
            continue;
        }

        if !has_fluid_voxel(chunk) {
            continue;
        }

        let mut grid = FluidGrid::new_empty(crate::world::chunk::CHUNK_SIZE);
        sync::sync_from_chunk(chunk, &mut grid, None);

        if grid.has_fluid() {
            fluid_state.insert(chunk.coord, grid);
        }
    }
}

/// Returns true if any voxel in the chunk is a trackable fluid material.
fn has_fluid_voxel(chunk: &Chunk) -> bool {
    chunk.voxels().iter().any(|v| is_fluid(v.material))
}

/// Fluid materials tracked by the AMR solver (water=3, lava=10).
fn is_fluid(mat: MaterialId) -> bool {
    matches!(mat.0, 3 | 10)
}

/// Wires cross-model coupling systems into the physics pipeline.
pub struct CouplingPlugin;

impl Plugin for CouplingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LazyCouplingTimer>()
            .add_systems(FixedUpdate, lazy_init_fluid_grids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::ChunkCoord;
    use crate::world::voxel::Voxel;

    #[test]
    fn lazy_coupling_timer_defaults() {
        let timer = LazyCouplingTimer::default();
        let dur = timer.timer.duration().as_secs_f32();
        assert!(
            (dur - LAZY_INIT_INTERVAL_SECS).abs() < f32::EPSILON,
            "expected {LAZY_INIT_INTERVAL_SECS}s, got {dur}"
        );
        assert_eq!(timer.timer.mode(), TimerMode::Repeating);
    }

    #[test]
    fn has_fluid_voxel_detects_water() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        assert!(!has_fluid_voxel(&chunk));

        chunk.set(
            0,
            0,
            0,
            Voxel {
                material: MaterialId::WATER,
                ..Default::default()
            },
        );
        assert!(has_fluid_voxel(&chunk));
    }

    #[test]
    fn has_fluid_voxel_detects_lava() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.set(
            1,
            1,
            1,
            Voxel {
                material: MaterialId::LAVA,
                ..Default::default()
            },
        );
        assert!(has_fluid_voxel(&chunk));
    }

    #[test]
    fn is_fluid_returns_false_for_stone() {
        assert!(!is_fluid(MaterialId::STONE));
        assert!(!is_fluid(MaterialId::AIR));
    }
}
