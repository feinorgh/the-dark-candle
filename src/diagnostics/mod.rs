// Diagnostics module: AI-agent-friendly state inspection tools.
//
// Provides two levels of diagnostics:
//
// 1. **Headless grid dumps** (`state_dump::dump_grid_state`) — works with the
//    simulation harness's flat voxel arrays. No Bevy ECS required.
//
// 2. **Live ECS dumps** (`DiagnosticsPlugin`) — Bevy plugin that captures
//    chunk state, camera position, and world resources on demand (F11 key)
//    and writes pretty-printed RON to `diagnostics/` directory.
//
// 3. **Screenshot capture** — captures the rendered frame as a PNG file on
//    demand (F12 key) and saves to `screenshots/` directory.

pub mod frame_budget;
pub mod state_dump;
pub mod system_timings;
pub mod video;
pub mod visualization;

use std::collections::BTreeMap;

use bevy::prelude::*;

use crate::camera::FpsCamera;
use crate::world::chunk::{CHUNK_SIZE, Chunk};
use crate::world::chunk_manager::ChunkMap;

use state_dump::{GridSummary, MaterialStats, RangeStats, StateDump};

/// Bevy plugin that registers on-demand diagnostics systems.
///
/// - **F3**: Toggle performance overlay (FPS, system timings, chunk stats)
/// - **F11**: Dump ECS world state to `diagnostics/<timestamp>.dump.ron`
/// - **F12**: Capture a screenshot to `screenshots/<timestamp>.png`
pub struct DiagnosticsPlugin;

impl Plugin for DiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        use crate::world::WorldSet;
        use system_timings::*;

        app.init_resource::<frame_budget::FrameBudget>()
            .init_resource::<SystemTimings>()
            .init_resource::<ChunkStats>()
            .add_systems(
                Update,
                (
                    ecs_dump_system,
                    screenshot_system,
                    frame_budget::update_frame_budget,
                    frame_budget::toggle_overlay,
                    frame_budget::update_overlay_text,
                ),
            )
            // Bracket systems around WorldSet for per-category timing.
            .add_systems(
                Update,
                (
                    begin_world_timing.before(WorldSet::ChunkManagement),
                    end_world_timing
                        .after(WorldSet::ChunkManagement)
                        .before(WorldSet::Meshing),
                    begin_meshing_timing.before(WorldSet::Meshing),
                    end_meshing_timing.after(WorldSet::Meshing),
                    update_chunk_stats.after(WorldSet::Meshing),
                ),
            )
            // Bracket the FixedUpdate schedule for physics timing.
            .add_systems(FixedFirst, begin_physics_timing)
            .add_systems(FixedLast, end_physics_timing);
    }
}

/// Serializable snapshot of a single chunk's summary.
#[derive(serde::Serialize, Debug, Clone)]
struct ChunkSnapshot {
    coord: [i32; 3],
    dirty: bool,
    solid_count: usize,
    material_histogram: BTreeMap<String, usize>,
    temperature: RangeStats,
    pressure: RangeStats,
}

/// Serializable snapshot of the camera state.
#[derive(serde::Serialize, Debug, Clone)]
struct CameraSnapshot {
    position: [f32; 3],
    pitch_deg: f32,
    yaw_deg: f32,
    speed: f32,
    grounded: bool,
}

/// Full ECS world dump.
#[derive(serde::Serialize, Debug, Clone)]
struct EcsDump {
    loaded_chunk_count: usize,
    camera: Option<CameraSnapshot>,
    chunks: Vec<ChunkSnapshot>,
    world_summary: StateDump,
}

/// System triggered by F11 to dump the live ECS state as RON.
fn ecs_dump_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    chunks: Query<&Chunk>,
    camera_q: Query<(&Transform, &FpsCamera)>,
    chunk_map: Option<Res<ChunkMap>>,
) {
    if !keyboard.just_pressed(KeyCode::F11) {
        return;
    }

    info!("Capturing ECS state dump...");

    // Camera snapshot
    let camera = camera_q.iter().next().map(|(tf, fps)| CameraSnapshot {
        position: [tf.translation.x, tf.translation.y, tf.translation.z],
        pitch_deg: fps.pitch.to_degrees(),
        yaw_deg: fps.yaw.to_degrees(),
        speed: fps.speed,
        grounded: fps.grounded,
    });

    // Aggregate across all loaded chunks
    let mut chunk_snapshots = Vec::new();
    let mut global_histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut global_temp = RangeStats::default();
    let mut global_pressure = RangeStats::default();
    let mut global_damage = RangeStats::default();
    let mut global_per_material: BTreeMap<String, MaterialStats> = BTreeMap::new();
    let mut total_voxels = 0usize;

    for chunk in chunks.iter() {
        let mut chunk_histogram: BTreeMap<String, usize> = BTreeMap::new();
        let mut chunk_temp = RangeStats::default();
        let mut chunk_pressure = RangeStats::default();

        for voxel in chunk.voxels() {
            // Use material ID as name (no registry access in this query)
            let name = format_material_id(voxel.material);

            *chunk_histogram.entry(name.clone()).or_default() += 1;
            *global_histogram.entry(name.clone()).or_default() += 1;

            chunk_temp.accumulate(voxel.temperature);
            chunk_pressure.accumulate(voxel.pressure);
            global_temp.accumulate(voxel.temperature);
            global_pressure.accumulate(voxel.pressure);

            if !voxel.is_air() {
                global_damage.accumulate(voxel.damage);
            }

            let mat = global_per_material
                .entry(name)
                .or_insert_with(|| MaterialStats {
                    count: 0,
                    temperature: RangeStats::default(),
                    pressure: RangeStats::default(),
                });
            mat.count += 1;
            mat.temperature.accumulate(voxel.temperature);
            mat.pressure.accumulate(voxel.pressure);

            total_voxels += 1;
        }

        chunk_temp.finalize();
        chunk_pressure.finalize();

        chunk_snapshots.push(ChunkSnapshot {
            coord: [chunk.coord.x, chunk.coord.y, chunk.coord.z],
            dirty: chunk.is_dirty(),
            solid_count: chunk.solid_count(),
            material_histogram: chunk_histogram,
            temperature: chunk_temp,
            pressure: chunk_pressure,
        });
    }

    global_temp.finalize();
    global_pressure.finalize();
    global_damage.finalize();
    for mat in global_per_material.values_mut() {
        mat.temperature.finalize();
        mat.pressure.finalize();
    }

    // Sort chunks by coordinate for deterministic output
    chunk_snapshots.sort_by_key(|c| (c.coord[0], c.coord[1], c.coord[2]));

    let loaded_count = chunk_map.as_ref().map(|m| m.len()).unwrap_or(0);

    let dump = EcsDump {
        loaded_chunk_count: loaded_count,
        camera,
        chunks: chunk_snapshots,
        world_summary: StateDump {
            grid_size: CHUNK_SIZE,
            total_voxels,
            summary: GridSummary {
                material_histogram: global_histogram,
                temperature: global_temp,
                pressure: global_pressure,
                damage: global_damage,
                per_material: global_per_material,
            },
            simulation_stats: None,
            voxels: None,
        },
    };

    // Write to file
    let config = ron::ser::PrettyConfig::new()
        .depth_limit(8)
        .struct_names(true);
    match ron::ser::to_string_pretty(&dump, config) {
        Ok(ron_text) => {
            if let Err(e) = std::fs::create_dir_all("diagnostics") {
                error!("Failed to create diagnostics/ directory: {e}");
                return;
            }
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let path = format!("diagnostics/{timestamp}.dump.ron");
            match std::fs::write(&path, &ron_text) {
                Ok(()) => info!(
                    "ECS state dump written to {path} ({} bytes)",
                    ron_text.len()
                ),
                Err(e) => error!("Failed to write dump: {e}"),
            }
        }
        Err(e) => error!("Failed to serialize ECS dump: {e}"),
    }
}

/// Format a `MaterialId` as a display name using well-known constants.
fn format_material_id(id: crate::world::voxel::MaterialId) -> String {
    use crate::world::voxel::MaterialId;
    match id {
        MaterialId::AIR => "Air".into(),
        MaterialId::STONE => "Stone".into(),
        MaterialId::DIRT => "Dirt".into(),
        MaterialId::WATER => "Water".into(),
        MaterialId::ICE => "Ice".into(),
        MaterialId::STEAM => "Steam".into(),
        MaterialId::LAVA => "Lava".into(),
        MaterialId::ASH => "Ash".into(),
        other => format!("Material({})", other.0),
    }
}

/// System triggered by F12 to capture a screenshot.
fn screenshot_system(keyboard: Res<ButtonInput<KeyCode>>, mut commands: Commands) {
    if !keyboard.just_pressed(KeyCode::F12) {
        return;
    }

    if let Err(e) = std::fs::create_dir_all("screenshots") {
        error!("Failed to create screenshots/ directory: {e}");
        return;
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = format!("screenshots/{timestamp}.png");

    commands
        .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
        .observe(bevy::render::view::screenshot::save_to_disk(path.clone()));
    info!("Screenshot capture triggered → {path}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_known_material_ids() {
        use crate::world::voxel::MaterialId;
        assert_eq!(format_material_id(MaterialId::AIR), "Air");
        assert_eq!(format_material_id(MaterialId::STONE), "Stone");
        assert_eq!(format_material_id(MaterialId::WATER), "Water");
        assert_eq!(format_material_id(MaterialId(42)), "Material(42)");
    }
}
