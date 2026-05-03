//! Headless gameplay stress-test harness.
//!
//! See `docs/superpowers/specs/2026-04-30-gameplay-stress-tests-design.md`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;

#[derive(Debug, Clone)]
pub struct CapturedPanic {
    pub thread: String,
    pub location: String,
    pub message: String,
}

#[derive(Default)]
pub(crate) struct LoadRateTracker {
    samples: std::collections::VecDeque<(f64, usize)>, // (sim_time_secs, loaded_count)
}

impl LoadRateTracker {
    fn push(&mut self, t: f64, loaded: usize) {
        self.samples.push_back((t, loaded));
        // Keep only the last 2 simulated seconds.
        while let Some(&(front_t, _)) = self.samples.front() {
            if t - front_t > 2.0 {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    fn rate(&self) -> f32 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let (t0, n0) = *self.samples.front().unwrap();
        let (t1, n1) = *self.samples.back().unwrap();
        let dt = (t1 - t0) as f32;
        if dt <= 0.0 {
            return 0.0;
        }
        ((n1 as i64 - n0 as i64) as f32 / dt).max(0.0)
    }
}

#[derive(Default)]
pub(crate) struct PanicSlot(Mutex<Option<CapturedPanic>>);

static PANIC_SLOT: std::sync::OnceLock<Arc<PanicSlot>> = std::sync::OnceLock::new();

fn install_panic_hook() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let slot = PANIC_SLOT.get_or_init(|| Arc::new(PanicSlot::default()));
        let slot_clone = Arc::clone(slot);

        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let thread = std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_string();
            let location = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "<unknown>".to_string());
            let message = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());

            if let Ok(mut guard) = slot_clone.0.lock() {
                // Keep the FIRST panic — subsequent panics during unwinding are noise.
                if guard.is_none() {
                    *guard = Some(CapturedPanic {
                        thread,
                        location,
                        message,
                    });
                }
            }

            prev(info);
        }));
    });
}

#[derive(Resource, Default)]
struct PendingTeleport(Option<bevy::math::DVec3>);

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
    pub(crate) load_rate: LoadRateTracker,
    pub(crate) sim_time: f64,
}

const FIXED_DT_SECS: f64 = 1.0 / 60.0;

fn apply_pending_teleport(
    mut pending: ResMut<PendingTeleport>,
    origin: Res<crate::floating_origin::RenderOrigin>,
    mut q: Query<
        (&mut crate::floating_origin::WorldPosition, &mut Transform),
        With<crate::camera::FpsCamera>,
    >,
) {
    let Some(target) = pending.0.take() else {
        return;
    };
    let Ok((mut wp, mut tf)) = q.single_mut() else {
        // Camera not yet spawned; put target back and try next frame.
        pending.0 = Some(target);
        return;
    };
    wp.0 = target;
    tf.translation = wp.render_offset(&origin);
}

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

        install_panic_hook();
        // Clear any leftover panic from a previous test in this process.
        if let Some(slot) = PANIC_SLOT.get()
            && let Ok(mut g) = slot.0.lock()
        {
            *g = None;
        }

        let planet = match preset {
            PlanetPreset::SmallPlanet => PlanetConfig {
                // `seed` is u64 but PlanetConfig.seed is u32; scenario seeds are small
                // values (< 2^32), so the low-32-bit truncation is intentional and safe.
                seed: seed as u32,
                ..PlanetConfig::default()
            },
            PlanetPreset::Earth => PlanetConfig {
                mean_radius: 6_371_000.0,
                sea_level_radius: 6_371_000.0,
                seed: seed as u32, // same truncation as SmallPlanet above
                ..Default::default()
            },
        };

        // `gen_cfg.seed` picks up `planet.seed` which was set from the harness `seed` parameter above.
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

        // Spawn a camera entity with FpsCamera, WorldPosition, Transform, and Camera3d.
        // Camera3d is required for FloatingOriginPlugin's rebasing queries (With<Camera>).
        let cam_world = DVec3::new(planet_radius, 0.0, 0.0);
        app.world_mut().spawn((
            FpsCamera::default(),
            WorldPosition::from_dvec3(cam_world),
            Transform::from_translation(Vec3::ZERO),
            Camera3d::default(),
        ));
        app.insert_resource(RenderOrigin(cam_world));

        // Add pending teleport system
        app.insert_resource(PendingTeleport::default());
        app.add_systems(Update, apply_pending_teleport);

        Self {
            app,
            seed,
            load_rate: LoadRateTracker::default(),
            sim_time: 0.0,
        }
    }

    /// Advance the simulation by `n` fixed-update frames.
    pub fn tick_n(&mut self, n: u32) {
        for _ in 0..n {
            self.app.update();
            self.sim_time += FIXED_DT_SECS;

            let count = self
                .app
                .world()
                .get_resource::<crate::world::v2::chunk_manager::V2ChunkMap>()
                .map(|m| m.loaded_count())
                .unwrap_or(0);
            self.load_rate.push(self.sim_time, count);
        }
    }

    /// Teleport the camera to (lat°, lon°, altitude_m) where altitude_m is
    /// metres above the planet's `sea_level_radius`.
    ///
    /// `lat_deg` is clamped to `[-89.99, +89.99]` to avoid pole singularities.
    /// `lon_deg` is normalized to `[-180, 180)`.  Note that ±180° map to the
    /// same physical direction on the sphere, so no special-casing is needed.
    pub fn teleport(&mut self, lat_deg: f64, lon_deg: f64, altitude_m: f64) {
        use bevy::math::DVec3;

        let lat = lat_deg.clamp(-89.99, 89.99).to_radians();
        // rem_euclid normalizes to [0, 360), then shift to [-180, 180).
        // ±180° give identical trig values (sin(±π)≈0, cos(±π)=−1) so no
        // special case is needed for the antimeridian boundary.
        let lon = (((lon_deg + 180.0).rem_euclid(360.0)) - 180.0).to_radians();

        let cos_lat = lat.cos();
        let dir = DVec3::new(cos_lat * lon.sin(), lat.sin(), cos_lat * lon.cos());

        let radius = self
            .app
            .world()
            .resource::<crate::world::planet::PlanetConfig>()
            .sea_level_radius
            + altitude_m;

        self.app.world_mut().resource_mut::<PendingTeleport>().0 = Some(dir * radius);
    }

    /// Take any panic that was captured by the harness panic hook.
    /// Clears the slot so subsequent calls return `None` until the next panic.
    pub fn take_panic(&self) -> Option<CapturedPanic> {
        PANIC_SLOT
            .get()
            .and_then(|slot| slot.0.lock().ok())
            .and_then(|mut guard| guard.take())
    }

    /// Get the chunk load rate (chunks/second) computed over the last 2 simulated seconds.
    pub fn chunk_load_rate(&self) -> f32 {
        self.load_rate.rate()
    }

    /// Check the requested invariants. Returns the list of failures
    /// (empty = all checks passed for the requested set).
    pub fn assert_invariants(&mut self, which: InvariantSet) -> Vec<InvariantFailure> {
        let mut failures = Vec::new();

        if which.contains(InvariantSet::PANICS)
            && let Some(p) = self.take_panic()
        {
            failures.push(InvariantFailure::Panic {
                thread: p.thread,
                location: p.location,
                message: p.message,
            });
        }

        if which.contains(InvariantSet::FINITE) || which.contains(InvariantSet::NO_OVERFLOW) {
            self.check_transforms(which, &mut failures);
        }

        if which.contains(InvariantSet::CHUNK_CACHE) {
            self.check_chunk_cache(&mut failures);
        }

        // LOAD_RATE handled in Task 8.

        failures
    }

    /// Check the requested invariants with optional minimum load rate.
    /// Returns the list of failures (empty = all checks passed for the requested set).
    pub fn assert_invariants_with_min_rate(
        &mut self,
        which: InvariantSet,
        min_rate: Option<f32>,
    ) -> Vec<InvariantFailure> {
        let mut failures = self.assert_invariants(which);

        if let (Some(min), true) = (min_rate, which.contains(InvariantSet::LOAD_RATE)) {
            let observed = self.chunk_load_rate();
            if observed < min {
                failures.push(InvariantFailure::LoadRateBelowMin { observed, min });
            }
        }

        failures
    }

    fn check_transforms(&mut self, which: InvariantSet, failures: &mut Vec<InvariantFailure>) {
        let world = self.app.world_mut();
        let mut q = world.query::<(Entity, &Transform)>();
        for (entity, tf) in q.iter(world) {
            let id = entity.to_bits();

            if which.contains(InvariantSet::FINITE) {
                if !tf.translation.is_finite() {
                    failures.push(InvariantFailure::NonFiniteTransform {
                        entity: id,
                        kind: "translation",
                        value: format!("{:?}", tf.translation),
                    });
                }
                if !tf.rotation.x.is_finite()
                    || !tf.rotation.y.is_finite()
                    || !tf.rotation.z.is_finite()
                    || !tf.rotation.w.is_finite()
                {
                    failures.push(InvariantFailure::NonFiniteTransform {
                        entity: id,
                        kind: "rotation",
                        value: format!("{:?}", tf.rotation),
                    });
                }
                if !tf.scale.is_finite() {
                    failures.push(InvariantFailure::NonFiniteTransform {
                        entity: id,
                        kind: "scale",
                        value: format!("{:?}", tf.scale),
                    });
                }
            }

            if which.contains(InvariantSet::NO_OVERFLOW) {
                for (axis, v) in [
                    ("translation.x", tf.translation.x),
                    ("translation.y", tf.translation.y),
                    ("translation.z", tf.translation.z),
                ] {
                    if v.is_finite() && v.abs() > F32_OVERFLOW_BOUND {
                        failures.push(InvariantFailure::F32Overflow {
                            what: format!("entity {id} {axis}"),
                            value: v as f64,
                        });
                    }
                }
            }
        }
    }

    fn check_chunk_cache(&mut self, failures: &mut Vec<InvariantFailure>) {
        use crate::world::v2::chunk_manager::V2ChunkMap;
        use std::collections::HashSet;

        let world = self.app.world();
        let Some(map) = world.get_resource::<V2ChunkMap>() else {
            // World plugin not active in this configuration — nothing to check.
            return;
        };

        let planet = world.resource::<crate::world::planet::PlanetConfig>();
        let mean_r = planet.mean_radius;

        let mut seen: HashSet<crate::world::v2::cubed_sphere::CubeSphereCoord> = HashSet::new();
        for (coord, _entity) in map.iter() {
            // Duplicate keys: HashMap can't actually contain duplicates by construction,
            // but a future refactor might expose them; check anyway for defence-in-depth.
            if !seen.insert(*coord) {
                failures.push(InvariantFailure::ChunkCache {
                    detail: format!("duplicate chunk coord: {coord:?}"),
                });
            }

            // Finite world position from the coord.
            let fce = coord.effective_fce(mean_r);
            let (world_pos, _, _) = coord.world_transform_scaled_f64(mean_r, fce);
            if !world_pos.x.is_finite() || !world_pos.y.is_finite() || !world_pos.z.is_finite() {
                failures.push(InvariantFailure::ChunkCache {
                    detail: format!("non-finite world pos for coord {coord:?}: {world_pos:?}"),
                });
            }
        }
    }
}

const F32_OVERFLOW_BOUND: f32 = 1.0e7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_builds_an_app_and_tick_n_advances_frames() {
        let mut app = StressApp::new(42, PlanetPreset::SmallPlanet);
        // tick_n must not panic on a freshly built app.
        app.tick_n(10);
    }

    #[test]
    fn teleport_moves_player_to_requested_lat_lon_alt() {
        use crate::camera::FpsCamera;
        use crate::floating_origin::WorldPosition;
        use bevy::math::DVec3;

        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
        app.tick_n(5); // let any startup systems run

        app.teleport(0.0, 0.0, 100.0); // equator, prime meridian, 100 m above sea level
        app.tick_n(1);

        // Player should now be at radius = sea_level + 100 m, in the +Z direction.
        // lat/lon convention: lon = atan2(x, z), so lon=0 → x=0, z=1 → +Z.
        let world = app.app.world_mut();
        let mut q = world.query_filtered::<&WorldPosition, With<FpsCamera>>();
        let pos = q.iter(world).next().expect("camera entity exists").0;

        let expected_radius = 32_000.0_f64 + 100.0;
        let actual_radius = pos.length();
        assert!(
            (actual_radius - expected_radius).abs() < 1.0,
            "expected radius ≈ {expected_radius}, got {actual_radius}"
        );
        let normalized = pos.normalize();
        assert!(
            (normalized - DVec3::Z).length() < 1.0e-3,
            "expected +Z direction, got {normalized:?}"
        );
    }

    #[test]
    fn panic_hook_captures_panic_in_app_thread() {
        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);

        // Inject a system that panics on first run.
        app.app.add_systems(Update, |mut count: Local<u32>| {
            *count += 1;
            if *count == 2 {
                panic!("synthetic panic for test");
            }
        });

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| app.tick_n(3)));
        assert!(result.is_err(), "expected panic to propagate");

        let captured = app.take_panic().expect("panic was captured");
        assert!(
            captured.message.contains("synthetic panic for test"),
            "captured: {:?}",
            captured
        );
    }

    #[test]
    fn finite_invariant_passes_after_clean_teleport() {
        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
        app.tick_n(30);
        app.teleport(0.0, 0.0, 100.0);
        app.tick_n(5);

        let failures = app.assert_invariants(InvariantSet::FINITE | InvariantSet::NO_OVERFLOW);
        assert!(failures.is_empty(), "unexpected failures: {failures:?}");
    }

    #[test]
    fn finite_invariant_catches_nan_translation() {
        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
        app.tick_n(30);

        // Inject a NaN translation on the camera.
        let world = app.app.world_mut();
        let mut q = world.query_filtered::<&mut Transform, With<crate::camera::FpsCamera>>();
        if let Some(mut tf) = q.iter_mut(world).next() {
            tf.translation.x = f32::NAN;
        }

        let failures = app.assert_invariants(InvariantSet::FINITE);
        assert!(
            failures
                .iter()
                .any(|f| matches!(f, InvariantFailure::NonFiniteTransform { .. })),
            "expected NonFiniteTransform failure, got: {failures:?}"
        );
    }

    #[test]
    fn overflow_invariant_catches_huge_translation() {
        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
        app.tick_n(30);

        let world = app.app.world_mut();
        let mut q = world.query_filtered::<&mut Transform, With<crate::camera::FpsCamera>>();
        if let Some(mut tf) = q.iter_mut(world).next() {
            tf.translation.x = 2.0e7;
        }

        let failures = app.assert_invariants(InvariantSet::NO_OVERFLOW);
        assert!(
            failures
                .iter()
                .any(|f| matches!(f, InvariantFailure::F32Overflow { .. })),
            "expected F32Overflow failure, got: {failures:?}"
        );
    }

    #[test]
    fn chunk_cache_invariant_passes_after_idle() {
        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
        app.tick_n(60); // let some chunks load
        app.teleport(0.0, 0.0, 0.0);
        app.tick_n(60);

        let failures = app.assert_invariants(InvariantSet::CHUNK_CACHE);
        assert!(
            failures.is_empty(),
            "unexpected chunk-cache failures after idle: {failures:?}"
        );
    }

    #[test]
    fn load_rate_reports_zero_on_empty_app_then_positive_after_load() {
        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);

        // Initial rate is undefined / zero.
        let r0 = app.chunk_load_rate();
        assert!(r0 >= 0.0, "rate must be non-negative, got {r0}");

        app.tick_n(120);
        let r1 = app.chunk_load_rate();
        // Expect SOME chunks to have loaded during 2 simulated seconds at the
        // origin of a small planet. If flaky, increase tick count.
        assert!(r1 > 0.0, "expected some chunk loading; got rate = {r1}");
    }
}
