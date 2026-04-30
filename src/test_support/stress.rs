//! Headless gameplay stress-test harness.
//!
//! See `docs/superpowers/specs/2026-04-30-gameplay-stress-tests-design.md`.

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;

/// Choice of planet configuration for the stress test.
#[derive(Clone, Copy, Debug)]
pub enum PlanetPreset {
    /// Earth-scale planet (mean radius ≈ 6.37e6 m).
    Earth,
    /// Default `PlanetConfig::default()` small planet (32 km).
    SmallPlanet,
}

/// Bitflags selecting which invariants to assert.
#[derive(Clone, Copy, Debug, Default)]
pub struct InvariantSet(u8);

impl InvariantSet {
    pub const PANICS: Self = Self(0b00001);
    pub const FINITE: Self = Self(0b00010);
    pub const NO_OVERFLOW: Self = Self(0b00100);
    pub const CHUNK_CACHE: Self = Self(0b01000);
    pub const LOAD_RATE: Self = Self(0b10000);

    pub const ALL_BUT_LOAD_RATE: Self = Self(0b01111);

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for InvariantSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// One invariant violation discovered by `assert_invariants`.
#[derive(Debug, Clone)]
pub enum InvariantFailure {
    Panic {
        thread: String,
        location: String,
        message: String,
    },
    NonFiniteTransform {
        entity: u64,
        kind: &'static str,
        value: String,
    },
    F32Overflow {
        what: String,
        value: f64,
    },
    ChunkCache {
        detail: String,
    },
    LoadRateBelowMin {
        observed: f32,
        min: f32,
    },
}

/// Headless gameplay stress harness.
///
/// Built progressively across the plan tasks:
/// - Task 3: `new`, `tick_n`
/// - Task 4: `teleport`
/// - Tasks 5–9: invariant checks
/// - Task 10: `chunk_load_rate`
#[allow(dead_code)]
pub struct StressApp {
    pub(crate) app: App,
    pub(crate) seed: u64, // fields populated across plan tasks 3-8
}

const FIXED_DT_SECS: f64 = 1.0 / 60.0;

impl StressApp {
    /// Build a headless stress-test app with the full non-rendering pipeline.
    pub fn new(seed: u64, preset: PlanetPreset) -> Self {
        use std::sync::Arc;

        use bevy::image::Image;
        use bevy::math::{DVec3, Vec3};
        use bevy::pbr::StandardMaterial;

        use crate::chemistry::ChemistryPlugin;
        use crate::data::DataPlugin;
        use crate::entities::EntityPlugin;
        use crate::floating_origin::{FloatingOriginPlugin, RenderOrigin, WorldPosition};
        use crate::game_state::{GameStatePlugin, SkipWorldCreation};
        use crate::physics::PhysicsPlugin;
        use crate::procgen::ProcgenPlugin;
        use crate::world::chunk_manager::SharedTerrainGen;
        use crate::world::lod::MaterialColorMap;
        use crate::world::planet::PlanetConfig;
        use crate::world::terrain::UnifiedTerrainGenerator;
        use crate::world::v2::chunk_manager::{V2LoadRadius, V2TerrainGen, V2WorldPlugin};

        // Import camera components
        use crate::camera::FpsCamera;

        let planet = match preset {
            PlanetPreset::SmallPlanet => PlanetConfig::default(),
            PlanetPreset::Earth => PlanetConfig {
                mean_radius: 6_371_000.0,
                sea_level_radius: 6_371_000.0,
                ..Default::default()
            },
        };

        // Build terrain generator following the pattern from chunk_manager.rs:1589-1599
        let tgen = {
            let gen_cfg = crate::planet::PlanetConfig {
                seed: planet.seed as u64,
                grid_level: 3,
                ..Default::default()
            };
            let pd = Arc::new(crate::planet::PlanetData::new(gen_cfg));
            Arc::new(UnifiedTerrainGenerator::new(pd, planet.clone()))
        };

        // Store mean_radius before moving planet
        let planet_radius = planet.mean_radius;

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(AssetPlugin::default());

        // Register asset types that DefaultPlugins normally provides but are needed
        // by V2WorldPlugin (meshing) and EntityPlugin (spawning).
        app.init_asset::<Mesh>();
        app.init_asset::<Image>();
        app.init_asset::<StandardMaterial>();

        // Add StatesPlugin - required by GameStatePlugin
        app.add_plugins(bevy::state::app::StatesPlugin);
        // Add TransformPlugin - needed for entity transforms
        app.add_plugins(bevy::transform::TransformPlugin);
        // Add InputPlugin - needed for ButtonInput resources even if unused
        app.add_plugins(bevy::input::InputPlugin);
        // Add GizmosPlugin - required by V2WorldPlugin's debug viz systems
        app.add_plugins(bevy::gizmos::GizmoPlugin);
        app.insert_resource(planet);
        app.insert_resource(SkipWorldCreation); // Skip the UI world creation screen
        app.insert_resource(MaterialColorMap::from_defaults()); // Required by V2WorldPlugin
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            FIXED_DT_SECS,
        )));

        // Insert terrain generator resources before V2WorldPlugin so v2_init_terrain_gen has what it needs
        app.insert_resource(V2TerrainGen(tgen.clone()));
        app.insert_resource(SharedTerrainGen(tgen));
        app.insert_resource(V2LoadRadius {
            horizontal: 2,
            vertical: 1,
        });

        let _ = seed; // Will use seed later when needed

        app.add_plugins((
            FloatingOriginPlugin,
            DataPlugin,
            GameStatePlugin,
            EntityPlugin,
            V2WorldPlugin, // Use V2WorldPlugin directly instead of WorldPlugin to avoid asset loading
            PhysicsPlugin,
            ChemistryPlugin,
            ProcgenPlugin,
        ));

        // Spawn a camera entity with FpsCamera, WorldPosition, and Transform.
        // This is required for V2WorldPlugin's v2_update_chunks system.
        let cam_world = DVec3::new(planet_radius, 0.0, 0.0);
        app.world_mut().spawn((
            FpsCamera::default(),
            WorldPosition::from_dvec3(cam_world),
            Transform::from_translation(Vec3::ZERO),
        ));
        app.insert_resource(RenderOrigin(cam_world));

        Self { app, seed }
    }

    /// Advance the simulation by `n` fixed-update frames.
    pub fn tick_n(&mut self, n: u32) {
        for _ in 0..n {
            self.app.update();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_builds_an_app_and_tick_n_advances_frames() {
        let mut app = StressApp::new(42, PlanetPreset::SmallPlanet);
        // tick_n must not panic on a freshly built app.
        app.tick_n(10);
    }
}
