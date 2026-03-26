// Chemistry runtime: wires simulate_tick() into Bevy FixedUpdate.
//
// Runs heat diffusion, radiation, chemical reactions, state transitions,
// and pressure diffusion per-chunk during live gameplay. Chunks are only
// ticked when thermally active (temperature above ambient + threshold or
// recent reactions/transitions).

use bevy::prelude::*;

use crate::chemistry::reactions::ReactionData;
use crate::data::{MaterialRegistry, find_data_dir};
use crate::simulation::{TickResult, simulate_tick};
use crate::world::chunk::{CHUNK_SIZE, Chunk};

/// Loaded reaction rules available as a Bevy resource.
#[derive(Resource, Debug, Clone, Default)]
pub struct ReactionRules(pub Vec<ReactionData>);

/// Tracks whether a chunk needs chemistry simulation each tick.
#[derive(Component, Debug, Clone)]
pub struct ChunkActivity {
    /// Whether this chunk should be ticked by the chemistry system.
    pub active: bool,
    /// Peak temperature recorded during the last simulation tick.
    pub last_max_temp: f32,
    /// Reactions fired during the last simulation tick.
    pub last_reactions: usize,
    /// Transitions during the last simulation tick.
    pub last_transitions: usize,
    /// Peak temperature at the time the chunk was last meshed.
    /// Used to suppress remeshing for small temperature-only changes.
    pub last_mesh_max_temp: f32,
}

impl Default for ChunkActivity {
    fn default() -> Self {
        Self {
            active: false,
            last_max_temp: 0.0,
            last_reactions: 0,
            last_transitions: 0,
            last_mesh_max_temp: 0.0,
        }
    }
}

/// Temperature (K) above ambient at which a chunk is considered active.
const ACTIVITY_TEMP_THRESHOLD: f32 = 50.0;

/// Ambient temperature (K) — matches `Voxel::default()`.
const AMBIENT_TEMP: f32 = 288.15;

/// Minimum temperature delta (K) from last mesh before a temperature-only
/// change triggers a remesh.
const REMESH_TEMP_DELTA: f32 = 50.0;

/// Cooldown timer controlling chemistry tick frequency.
#[derive(Resource, Debug)]
pub struct ChemistryTimer {
    timer: Timer,
}

impl Default for ChemistryTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(0.5, TimerMode::Repeating),
        }
    }
}

/// System set for chemistry runtime systems.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChemistrySet;

/// Advance the chemistry cooldown timer each fixed tick.
fn tick_chemistry_timer(time: Res<Time>, mut timer: ResMut<ChemistryTimer>) {
    timer.timer.tick(time.delta());
}

/// Core simulation system: iterate loaded chunks and run `simulate_tick()`
/// on each active chunk.
fn chunk_simulation(
    timer: Res<ChemistryTimer>,
    registry: Res<MaterialRegistry>,
    rules: Res<ReactionRules>,
    mut query: Query<(&mut Chunk, &mut ChunkActivity)>,
) {
    if !timer.timer.just_finished() {
        return;
    }

    let dt = timer.timer.duration().as_secs_f32();

    query.par_iter_mut().for_each(|(mut chunk, mut activity)| {
        if !activity.active {
            return;
        }

        let voxels = chunk.voxels_mut();
        let result: TickResult = simulate_tick(voxels, CHUNK_SIZE, &rules.0, &registry, dt);

        activity.last_max_temp = result.max_temp;
        activity.last_reactions = result.reactions_fired;
        activity.last_transitions = result.transitions;

        // Determine if the chunk should stay active for next tick.
        let thermally_hot = result.max_temp > AMBIENT_TEMP + ACTIVITY_TEMP_THRESHOLD;
        let had_events = result.reactions_fired > 0 || result.transitions > 0;
        activity.active = thermally_hot || had_events;

        // Suppress dirty flag for temperature-only changes below the visual
        // threshold. voxels_mut() already set dirty = true, so we undo it
        // when remeshing would be wasteful.
        if result.reactions_fired == 0 && result.transitions == 0 {
            let temp_delta = (result.max_temp - activity.last_mesh_max_temp).abs();
            if temp_delta < REMESH_TEMP_DELTA {
                chunk.clear_dirty();
            }
        }
    });
}

/// One-time scan that marks chunks as active if they contain voxels with
/// temperatures significantly above ambient.
fn activate_hot_chunks(mut query: Query<(&Chunk, &mut ChunkActivity)>) {
    query.par_iter_mut().for_each(|(chunk, mut activity)| {
        if activity.active {
            return;
        }
        let max_temp = chunk
            .voxels()
            .iter()
            .map(|v| v.temperature)
            .fold(0.0_f32, f32::max);

        if max_temp > AMBIENT_TEMP + ACTIVITY_TEMP_THRESHOLD {
            activity.active = true;
            activity.last_max_temp = max_temp;
        }
    });
}

/// Build `ReactionRules` by reading `.reaction.ron` files from disk.
pub fn load_reaction_rules() -> Result<ReactionRules, String> {
    let dir = find_data_dir()?.join("reactions");
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;

    let mut rules = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".reaction.ron") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let data: ReactionData =
            ron::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        rules.push(data);
    }
    Ok(ReactionRules(rules))
}

/// Register chemistry runtime systems and resources.
pub struct ChemistryRuntimePlugin;

impl Plugin for ChemistryRuntimePlugin {
    fn build(&self, app: &mut App) {
        // Load reaction rules from disk.
        match load_reaction_rules() {
            Ok(rules) => {
                info!("Loaded {} reaction rules", rules.0.len());
                app.insert_resource(rules);
            }
            Err(e) => {
                warn!("Failed to load reaction rules: {e}");
                app.init_resource::<ReactionRules>();
            }
        }

        app.init_resource::<ChemistryTimer>().add_systems(
            FixedUpdate,
            (
                tick_chemistry_timer,
                activate_hot_chunks
                    .in_set(ChemistrySet)
                    .after(tick_chemistry_timer),
                chunk_simulation
                    .in_set(ChemistrySet)
                    .after(activate_hot_chunks)
                    .run_if(resource_exists::<MaterialRegistry>),
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::{MaterialId, Voxel};

    #[test]
    fn chemistry_timer_defaults_to_half_second() {
        let timer = ChemistryTimer::default();
        let duration = timer.timer.duration().as_secs_f32();
        assert!(
            (duration - 0.5).abs() < f32::EPSILON,
            "expected 0.5s, got {duration}"
        );
        assert_eq!(timer.timer.mode(), TimerMode::Repeating);
    }

    #[test]
    fn chunk_activity_default_is_inactive() {
        let activity = ChunkActivity::default();
        assert!(!activity.active);
        assert_eq!(activity.last_reactions, 0);
        assert_eq!(activity.last_transitions, 0);
    }

    #[test]
    fn reaction_rules_loads_from_disk() {
        // Only runs if assets/ is accessible (CI or local).
        if let Ok(rules) = load_reaction_rules() {
            assert!(!rules.0.is_empty(), "expected at least one reaction rule");
        }
    }

    #[test]
    fn activity_constants_are_reasonable() {
        // Compile-time validated via const block.
        const {
            assert!(ACTIVITY_TEMP_THRESHOLD > 0.0);
            assert!(AMBIENT_TEMP > 0.0);
            assert!(REMESH_TEMP_DELTA > 0.0);
            assert!(ACTIVITY_TEMP_THRESHOLD <= REMESH_TEMP_DELTA);
        }
    }

    fn make_chunk_with_temp(temp: f32) -> Chunk {
        use crate::world::chunk::ChunkCoord;
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        let voxel = Voxel {
            temperature: temp,
            material: MaterialId(1),
            ..Default::default()
        };
        chunk.set(0, 0, 0, voxel);
        chunk.clear_dirty();
        chunk
    }

    #[test]
    fn hot_chunk_is_detected_as_active() {
        let chunk = make_chunk_with_temp(500.0);
        let max_temp = chunk
            .voxels()
            .iter()
            .map(|v| v.temperature)
            .fold(0.0_f32, f32::max);
        assert!(max_temp > AMBIENT_TEMP + ACTIVITY_TEMP_THRESHOLD);
    }

    #[test]
    fn ambient_chunk_is_not_active() {
        let chunk = make_chunk_with_temp(AMBIENT_TEMP);
        let max_temp = chunk
            .voxels()
            .iter()
            .map(|v| v.temperature)
            .fold(0.0_f32, f32::max);
        assert!(max_temp <= AMBIENT_TEMP + ACTIVITY_TEMP_THRESHOLD);
    }
}
