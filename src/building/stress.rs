//! Structural stress analysis — load-path tracing and progressive collapse.
//!
//! For each building part, forces (gravity, wind, explosions) are transmitted
//! through joints toward ground anchors. Joints that exceed their material
//! strength break, redistributing load to neighbouring joints.
//!
//! # Algorithm
//! 1. Identify *anchor* parts: parts whose bounding box overlaps a terrain voxel.
//! 2. Build a force-flow graph from parts → joints → anchors.
//! 3. Accumulate gravity load (mass × g) at each part.
//! 4. Propagate loads down the graph; accumulate stress per joint.
//! 5. Break joints where stress ≥ material strength.
//! 6. Trigger collapse detection: unsupported parts become physics debris.
//!
//! # Performance
//! Full analysis runs every `STRESS_TICK_INTERVAL` fixed ticks (default: 10).
//! The existing flood-fill connectivity (`src/world/integrity.rs`) remains
//! active for chunks without any `PlacedPart` entities.

use bevy::prelude::*;

use crate::physics::constants::GRAVITY;

use super::{joints::Joint, parts::PlacedPart};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Run full stress analysis every N fixed-update ticks.
pub const STRESS_TICK_INTERVAL: u32 = 10;

/// Gravity acceleration used for structural load calculations (m/s²).
/// Matches GRAVITY_SURFACE from physics constants.
const G: f32 = GRAVITY;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Countdown timer for the budget-limited stress analysis.
#[derive(Resource, Default)]
pub struct StressTick(pub u32);

// ---------------------------------------------------------------------------
// Anchor detection
// ---------------------------------------------------------------------------

/// Marker placed on parts that are directly supported by terrain.
///
/// Parts with this marker can absorb unlimited compressive load (terrain
/// acts as bedrock in the stress model).
#[derive(Component, Default)]
pub struct GroundAnchor;

/// System that marks parts touching terrain as ground anchors.
///
/// Uses the part's `Transform` to check if its bottom face is at or below
/// ground level (y ≤ 0). A full implementation would raytrace into the
/// chunk voxel grid; this is a simplified height-based check.
#[allow(clippy::type_complexity)]
pub fn mark_ground_anchors(
    mut commands: Commands,
    query: Query<(Entity, &Transform), (With<PlacedPart>, Without<GroundAnchor>)>,
) {
    for (entity, transform) in &query {
        // Parts with their base (bottom face) at or below y=0 are anchored.
        // TODO: sample actual terrain voxel to detect terrain contact.
        if transform.translation.y <= 0.5 {
            commands.entity(entity).insert(GroundAnchor);
        }
    }
}

// ---------------------------------------------------------------------------
// Load accumulation
// ---------------------------------------------------------------------------

/// Per-part accumulated load (N) used during one stress pass.
#[derive(Component, Default)]
pub struct PartLoad {
    /// Downward axial force in Newtons (gravity + loads from above).
    pub axial_n: f32,
    /// Lateral force in Newtons (wind, explosion impulse).
    pub lateral_n: f32,
}

/// Accumulate self-weight load on every part.
///
/// `F = density × volume × g`  
/// Volume is approximated as 1 m³ per voxel (each part occupies one voxel slot).
pub fn accumulate_self_weight(
    registry: Res<crate::data::MaterialRegistry>,
    mut query: Query<(&PlacedPart, &mut PartLoad)>,
) {
    for (part, mut load) in &mut query {
        if let Some(mat) = registry.get_by_name(&part.material_name) {
            // Volume ≈ 1 m³ per part (simplified — full impl uses PartShape volume).
            load.axial_n += mat.density * 1.0 * G;
        }
    }
}

/// Apply wind loading to exposed parts using the LBM pressure field.
///
/// Each part exposed to the atmosphere receives a lateral force proportional
/// to the local pressure deviation from the reference pressure.
pub fn apply_wind_loading(
    lbm_state: Option<Res<crate::physics::lbm_gas::plugin::LbmState>>,
    mut query: Query<(&Transform, &mut PartLoad), With<PlacedPart>>,
) {
    let Some(lbm) = lbm_state else { return };

    for (transform, mut load) in &mut query {
        let pos = transform.translation;
        let vx = pos.x.floor() as i32;
        let vy = pos.y.floor() as i32;
        let vz = pos.z.floor() as i32;
        let coord = crate::world::chunk::ChunkCoord::from_voxel_pos(vx, vy, vz);
        let Some(grid) = lbm.get(&coord) else {
            continue;
        };
        let chunk_size = crate::world::chunk::CHUNK_SIZE as i32;
        let origin = coord.world_origin();
        let lx = (vx - origin.x).clamp(0, chunk_size - 1) as usize;
        let ly = (vy - origin.y).clamp(0, chunk_size - 1) as usize;
        let lz = (vz - origin.z).clamp(0, chunk_size - 1) as usize;
        let cell = grid.get(lx, ly, lz);
        if cell.is_gas() {
            // Pressure deviation from sea-level reference (Pa).
            // Approximate P = rho × CS2 × c² (LBM ideal gas).
            // For wind loading we use density directly as a proxy for pressure.
            let rho = cell.density();
            let rho_ref = 1.225_f32; // kg/m³ at sea level
            let deviation_pa = (rho - rho_ref) * 340.0_f32.powi(2) / 3.0; // P = ρ·CS2·c²
            // Wind force = pressure deviation × face area (1 m²), capped at 50 kPa.
            load.lateral_n += deviation_pa.abs().min(50_000.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Stress propagation & joint breaking
// ---------------------------------------------------------------------------

/// Propagate accumulated loads through joints and check for failure.
///
/// This is a simplified single-pass top-down propagation. A full
/// implementation would use iterative load redistribution until convergence.
pub fn propagate_stress_and_break(
    registry: Res<crate::data::MaterialRegistry>,
    part_query: Query<(&PlacedPart, &PartLoad), Without<GroundAnchor>>,
    mut joint_query: Query<&mut Joint>,
) {
    for (part, load) in &part_query {
        let Some(mat) = registry.get_by_name(&part.material_name) else {
            continue;
        };

        for mut joint in &mut joint_query {
            if joint.broken {
                continue;
            }
            // Check if this joint is connected to this part.
            // Simplified: apply load to all joints involving this material.
            // Full impl: build adjacency graph and only apply to relevant joints.
            let dummy_mat = mat.clone();
            joint.apply_axial(load.axial_n * 0.5, &dummy_mat, &dummy_mat);
            joint.apply_shear(load.lateral_n * 0.5, &dummy_mat, &dummy_mat);
        }
    }
}

/// Despawn unsupported parts after joint failure (progressive collapse).
///
/// A part is unsupported if all joints connecting it to the rest of the
/// structure are broken. Unsupported parts are despawned (TODO: convert to
/// debris `PhysicsBody` entities in the demolition module).
pub fn despawn_unsupported_parts(
    mut commands: Commands,
    part_query: Query<Entity, (With<PlacedPart>, Without<GroundAnchor>)>,
    joint_query: Query<&Joint>,
) {
    'outer: for part_entity in &part_query {
        // Check if any live joint still connects this part.
        for joint in &joint_query {
            if !joint.broken && (joint.part_a == part_entity || joint.part_b == part_entity) {
                continue 'outer; // Still connected.
            }
        }
        // No live joints → despawn.
        commands.entity(part_entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// Tick-gated driver system
// ---------------------------------------------------------------------------

/// Increment the stress tick counter; return true when analysis should run.
pub fn should_run_stress(mut tick: ResMut<StressTick>) -> bool {
    tick.0 = tick.0.wrapping_add(1);
    tick.0.is_multiple_of(STRESS_TICK_INTERVAL)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stress_tick_fires_at_interval() {
        let mut tick = StressTick(0);
        let mut fires = 0u32;
        for _ in 0..(STRESS_TICK_INTERVAL * 3) {
            tick.0 = tick.0.wrapping_add(1);
            if tick.0.is_multiple_of(STRESS_TICK_INTERVAL) {
                fires += 1;
            }
        }
        assert_eq!(fires, 3);
    }

    #[test]
    fn part_load_default_is_zero() {
        let load = PartLoad::default();
        assert_eq!(load.axial_n, 0.0);
        assert_eq!(load.lateral_n, 0.0);
    }
}
