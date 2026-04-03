//! Locomotion system: gait cycles, procedural bone animation, and IK foot placement.
//!
//! # Architecture
//! - `GaitData` is a RON-loaded asset describing gait cycles for a species.
//! - `GaitState` tracks the active gait and phase for each entity at runtime.
//! - `update_locomotion` advances the gait phase and places feet on terrain via
//!   downward ray casts into the voxel grid.
//! - `PendingMetabolicCost` accumulates energy debt each tick for the biology
//!   system to drain.

use bevy::prelude::*;
use serde::Deserialize;

use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::chunk_manager::ChunkMap;
use crate::world::raycast;

pub use super::skeleton::Skeleton;

// ---------------------------------------------------------------------------
// RON asset types
// ---------------------------------------------------------------------------

/// A single keyframe target for one bone within a gait phase.
#[derive(Deserialize, Debug, Clone)]
pub struct BoneTarget {
    /// Name of the bone to drive (must match a name in `Skeleton::bone_names`).
    pub bone: String,
    /// Local rotation expressed as Euler angles in radians (pitch, yaw, roll).
    pub local_rotation: Vec3,
}

/// One discrete phase (keyframe) of a gait cycle.
#[derive(Deserialize, Debug, Clone)]
pub struct GaitPhase {
    /// Desired local-space rotation for each driven bone at this phase.
    pub bone_targets: Vec<BoneTarget>,
}

/// A complete named gait cycle (e.g. "walk", "run") for one species.
#[derive(Deserialize, Debug, Clone)]
pub struct GaitCycle {
    /// Unique name used to select this cycle (e.g. `"walk"`, `"run"`).
    pub name: String,
    /// Keyframe sequence. Phases are interpolated in order and wrap around.
    pub phases: Vec<GaitPhase>,
    /// Wall-clock duration of one full cycle in seconds.
    pub cycle_duration_s: f32,
    /// Metabolic energy cost in joules per metre of travel (J/m).
    pub energy_cost_j_per_m: f32,
    /// Minimum locomotion speed (m/s) for which this gait is suitable.
    pub min_speed: f32,
    /// Maximum locomotion speed (m/s) for which this gait is suitable.
    pub max_speed: f32,
}

/// Top-level asset: all gait cycles for one species, loaded from
/// `assets/data/gaits/{species}.gait.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone)]
pub struct GaitData {
    /// Species identifier (e.g. `"biped"`, `"quadruped"`).
    pub species: String,
    /// Ordered list of gait cycles; the locomotion system selects among these
    /// based on `GaitState::speed`.
    pub gaits: Vec<GaitCycle>,
}

// ---------------------------------------------------------------------------
// Runtime components
// ---------------------------------------------------------------------------

/// Which locomotion mode an entity is currently using.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GaitMode {
    Idle,
    Walk,
    Run,
    Crawl,
    Swim,
    Fly,
    Slither,
    Climb,
}

/// ECS component tracking an entity's active gait and animation state.
#[derive(Component, Debug, Clone)]
pub struct GaitState {
    /// Current locomotion mode.
    pub mode: GaitMode,
    /// Normalised phase within the current gait cycle (0.0 – 1.0).
    pub phase: f32,
    /// Horizontal locomotion speed in m/s.
    pub speed: f32,
}

impl Default for GaitState {
    fn default() -> Self {
        Self {
            mode: GaitMode::Idle,
            phase: 0.0,
            speed: 0.0,
        }
    }
}

/// Component holding a handle to the entity's species-specific `GaitData` asset.
#[derive(Component, Debug, Clone)]
pub struct GaitDataHandle(pub Handle<GaitData>);

/// Accumulates metabolic energy expenditure (Joules) each tick.
///
/// The biology/metabolism system should drain this component regularly so that
/// locomotion consumes stamina/calories.
#[derive(Component, Debug, Clone, Default)]
pub struct PendingMetabolicCost(pub f32);

// ---------------------------------------------------------------------------
// Gait selection
// ---------------------------------------------------------------------------

/// Select the most appropriate `GaitMode` for the given speed (m/s).
///
/// Speed thresholds (SI, realistic biped defaults):
/// - < 0.1 m/s  → `Idle`
/// - < 2.5 m/s  → `Walk`  (human walk ≈ 1.4 m/s)
/// - < 6.0 m/s  → `Run`   (easy jog ≈ 3–4 m/s)
/// - ≥ 6.0 m/s  → `Run`   (sprint; falls back to `Run` cycle if no sprint present)
pub fn select_gait_mode(speed: f32) -> GaitMode {
    if speed < 0.1 {
        GaitMode::Idle
    } else if speed < 2.5 {
        GaitMode::Walk
    } else {
        // Speeds ≥ 2.5 m/s use Run; sprinting uses Run if no dedicated sprint cycle.
        GaitMode::Run
    }
}

/// Find the best-matching `GaitCycle` in `gaits` for the given mode and speed.
///
/// Falls back to any `Run` cycle if the requested mode is not present.
fn find_cycle<'a>(gaits: &'a [GaitCycle], mode: &GaitMode) -> Option<&'a GaitCycle> {
    let name = match mode {
        GaitMode::Idle => "idle",
        GaitMode::Walk => "walk",
        GaitMode::Run => "run",
        GaitMode::Crawl => "crawl",
        GaitMode::Swim => "swim",
        GaitMode::Fly => "fly",
        GaitMode::Slither => "slither",
        GaitMode::Climb => "climb",
    };

    gaits
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(name))
        .or_else(|| gaits.iter().find(|c| c.name.eq_ignore_ascii_case("run")))
}

// ---------------------------------------------------------------------------
// Foot placement via downward ray cast
// ---------------------------------------------------------------------------

/// Clearance above the terrain surface at which the foot IK target is placed (m).
const FOOT_CLEARANCE_M: f32 = 0.05;

/// Maximum downward search distance for terrain beneath a foot (m).
const FOOT_SEARCH_DEPTH_M: usize = 4;

/// Cast a short ray straight down from `world_pos` and return the first solid
/// voxel surface it hits, offset upward by `FOOT_CLEARANCE_M`.
///
/// Returns `None` if no chunk is loaded at that position or no terrain was found
/// within `FOOT_SEARCH_DEPTH_M` metres.
fn find_foot_placement(
    world_pos: Vec3,
    chunk_map: &ChunkMap,
    chunks: &Query<&Chunk>,
) -> Option<Vec3> {
    let vx = world_pos.x.floor() as i32;
    let vy = world_pos.y.floor() as i32;
    let vz = world_pos.z.floor() as i32;

    let chunk_coord = ChunkCoord::from_voxel_pos(vx, vy, vz);
    let chunk_entity = chunk_map.get(&chunk_coord)?;
    let chunk = chunks.get(chunk_entity).ok()?;

    let voxels = chunk.voxels();
    let origin = chunk_coord.world_origin();

    // Local coordinates within the chunk for the foot column.
    let lx = (vx - origin.x).rem_euclid(CHUNK_SIZE as i32) as usize;
    let lz = (vz - origin.z).rem_euclid(CHUNK_SIZE as i32) as usize;

    // Local Y for the foot start position.
    let start_ly = (vy - origin.y).rem_euclid(CHUNK_SIZE as i32);

    // Walk downward within the chunk looking for a solid voxel.
    for dy in 0..=(FOOT_SEARCH_DEPTH_M as i32) {
        let ly = start_ly - dy;
        if ly < 0 || ly >= CHUNK_SIZE as i32 {
            break;
        }
        let idx = (ly as usize) * CHUNK_SIZE * CHUNK_SIZE + lz * CHUNK_SIZE + lx;
        let voxel = voxels.get(idx)?;
        if !voxel.is_air() {
            // Surface found: place foot just above this voxel.
            let surface_y = (origin.y + ly) as f32 + 1.0 + FOOT_CLEARANCE_M;
            return Some(Vec3::new(world_pos.x, surface_y, world_pos.z));
        }
    }

    // Also cast via march_grid_ray for a more accurate result when the local
    // column scan fails (e.g. foot is already below terrain).
    let start = [lx, start_ly.max(0) as usize, lz];
    // Direction index 4 in RAY_DIRECTIONS is downward (0, -1, 0).
    const DOWN_DIR: usize = 4;
    if let Some(hit) =
        raycast::march_grid_ray(voxels, CHUNK_SIZE, start, DOWN_DIR, FOOT_SEARCH_DEPTH_M)
    {
        let surface_y = (origin.y + hit.y as i32) as f32 + 1.0 + FOOT_CLEARANCE_M;
        return Some(Vec3::new(world_pos.x, surface_y, world_pos.z));
    }

    None
}

// ---------------------------------------------------------------------------
// Locomotion system
// ---------------------------------------------------------------------------

/// Update gait phases, apply bone angle keyframes, place feet on terrain via IK,
/// and accumulate metabolic energy costs.
///
/// Should run in `FixedUpdate` after physics integration and before rendering.
pub fn update_locomotion(
    time: Res<Time>,
    chunk_map: Res<ChunkMap>,
    gait_assets: Res<Assets<GaitData>>,
    chunks: Query<&Chunk>,
    mut query: Query<(
        &mut GaitState,
        &mut Skeleton,
        Option<&GaitDataHandle>,
        Option<&mut PendingMetabolicCost>,
    )>,
) {
    let dt = time.delta_secs();

    for (mut gait_state, mut skeleton, gait_handle, mut metabolic) in &mut query {
        // ── 1. Update gait mode from speed ──────────────────────────────────
        gait_state.mode = select_gait_mode(gait_state.speed);

        // ── 2. Retrieve gait cycle from asset ───────────────────────────────
        let cycle_opt: Option<&GaitCycle> = gait_handle
            .and_then(|h| gait_assets.get(&h.0))
            .and_then(|data| find_cycle(&data.gaits, &gait_state.mode));

        let cycle_duration = cycle_opt.map(|c| c.cycle_duration_s).unwrap_or(0.6);

        // ── 3. Advance phase ─────────────────────────────────────────────────
        if gait_state.speed > 0.0 {
            gait_state.phase = (gait_state.phase + dt / cycle_duration).rem_euclid(1.0);
        } else {
            // Idle: reset phase toward 0 smoothly.
            gait_state.phase = (gait_state.phase * (1.0 - dt * 4.0)).max(0.0);
        }

        // ── 4. Apply bone angle keyframes ────────────────────────────────────
        if let Some(cycle) = cycle_opt {
            apply_gait_keyframes(&mut skeleton, cycle, gait_state.phase);
        }

        // ── 5. Procedural IK foot placement ─────────────────────────────────
        for i in 0..skeleton.bone_names.len() {
            let name = &skeleton.bone_names[i];
            if name.contains("foot") || name.contains("hoof") || name.contains("paw") {
                let foot_world = skeleton.bone_transforms[i].translation;
                if let Some(target) = find_foot_placement(foot_world, &chunk_map, &chunks) {
                    skeleton.ik_targets[i] = Some(target);
                }
            }
        }

        // ── 6. Metabolic energy expenditure ──────────────────────────────────
        if gait_state.speed > 0.0 {
            let energy_cost_j_per_m = cycle_opt.map(|c| c.energy_cost_j_per_m).unwrap_or(80.0);
            // Distance covered this tick × energy per metre → joules.
            // Divide by 1000 to convert to kJ for the biology system.
            let delta_energy = gait_state.speed * dt * energy_cost_j_per_m / 1000.0;
            if let Some(ref mut cost) = metabolic {
                cost.0 += delta_energy;
            }
        }
    }
}

/// Interpolate between adjacent gait keyframes and write the result into
/// the skeleton's `bone_transforms` rotations.
fn apply_gait_keyframes(skeleton: &mut Skeleton, cycle: &GaitCycle, phase: f32) {
    let n = cycle.phases.len();
    if n == 0 {
        return;
    }

    // Find surrounding keyframe indices.
    let float_idx = phase * n as f32;
    let a_idx = float_idx.floor() as usize % n;
    let b_idx = (a_idx + 1) % n;
    let t = float_idx.fract();

    let phase_a = &cycle.phases[a_idx];
    let phase_b = &cycle.phases[b_idx];

    for target_a in &phase_a.bone_targets {
        let Some(bone_idx) = skeleton.bone_index(&target_a.bone) else {
            continue;
        };
        // Find matching target in the next phase (same bone name).
        let rot_b = phase_b
            .bone_targets
            .iter()
            .find(|t| t.bone == target_a.bone)
            .map(|t| t.local_rotation)
            .unwrap_or(target_a.local_rotation);

        let euler_a = target_a.local_rotation;
        let euler_b = rot_b;
        let euler = euler_a.lerp(euler_b, t);

        skeleton.bone_transforms[bone_idx].rotation =
            Quat::from_euler(EulerRot::XYZ, euler.x, euler.y, euler.z);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_below_threshold() {
        assert_eq!(select_gait_mode(0.0), GaitMode::Idle);
        assert_eq!(select_gait_mode(0.09), GaitMode::Idle);
    }

    #[test]
    fn walk_between_thresholds() {
        assert_eq!(select_gait_mode(0.1), GaitMode::Walk);
        assert_eq!(select_gait_mode(1.4), GaitMode::Walk);
        assert_eq!(select_gait_mode(2.49), GaitMode::Walk);
    }

    #[test]
    fn run_above_walk_threshold() {
        assert_eq!(select_gait_mode(2.5), GaitMode::Run);
        assert_eq!(select_gait_mode(5.9), GaitMode::Run);
        assert_eq!(select_gait_mode(10.0), GaitMode::Run);
    }

    #[test]
    fn gait_phase_wraps_at_one() {
        // Simulate a manual phase increment that goes past 1.0.
        let mut phase: f32 = 0.95;
        let dt = 0.1_f32;
        let cycle_duration = 0.6_f32;
        phase = (phase + dt / cycle_duration).rem_euclid(1.0);
        assert!(phase < 1.0, "phase must be in [0, 1), got {phase}");
        assert!(phase >= 0.0);
    }

    #[test]
    fn phase_stays_in_range_over_many_ticks() {
        let mut phase = 0.0_f32;
        for _ in 0..1000 {
            phase = (phase + 0.016 / 0.6).rem_euclid(1.0);
            assert!((0.0..1.0).contains(&phase));
        }
    }

    #[test]
    fn find_cycle_falls_back_to_run() {
        let gaits = vec![
            GaitCycle {
                name: "walk".into(),
                phases: vec![],
                cycle_duration_s: 0.8,
                energy_cost_j_per_m: 60.0,
                min_speed: 0.1,
                max_speed: 2.5,
            },
            GaitCycle {
                name: "run".into(),
                phases: vec![],
                cycle_duration_s: 0.4,
                energy_cost_j_per_m: 120.0,
                min_speed: 2.5,
                max_speed: 8.0,
            },
        ];
        // Requesting Fly falls back to Run.
        let c = find_cycle(&gaits, &GaitMode::Fly).expect("should fall back to run");
        assert_eq!(c.name, "run");
    }

    #[test]
    fn skeleton_bone_index_found() {
        let sk = Skeleton {
            bone_names: vec!["root".into(), "left_foot".into(), "right_foot".into()],
            bone_transforms: vec![Transform::default(); 3],
            ik_targets: vec![None; 3],
            torques: vec![Vec3::ZERO; 3],
            ..Default::default()
        };
        assert_eq!(sk.bone_index("left_foot"), Some(1));
        assert_eq!(sk.bone_index("missing"), None);
    }
}
