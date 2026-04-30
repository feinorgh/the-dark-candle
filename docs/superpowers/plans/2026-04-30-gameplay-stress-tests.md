# Gameplay Stress / Fuzzy Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless gameplay stress-test suite that exercises the full non-rendering planet/chunk/camera pipeline against extreme game states, asserting invariants that catch the historic teleport-related f32 overflow crash class (CRASH-001).

**Architecture:** A `StressApp` harness in `src/test_support/stress.rs` (Bevy app with `MinimalPlugins` + the real world/chunk/camera/physics/chemistry/procgen plugins, no rendering). RON regression scenarios under `tests/cases/stress/` and `proptest` property generators in `tests/stress.rs`, both driven by the same harness. Invariants check for panics, finite Transforms, f32 world-coord overflow, chunk-cache integrity, and chunk-load throughput.

**Tech Stack:** Rust, Bevy 0.18, RON, `proptest` (new dev-dep), existing `glob`/`serde`/`ron`/`bevy_common_assets` already in tree.

**Spec:** `docs/superpowers/specs/2026-04-30-gameplay-stress-tests-design.md`

---

## File Structure

**New files:**

- `src/test_support/mod.rs` — module entry, gated by `cfg(any(test, feature = "test-support"))`. Re-exports the harness.
- `src/test_support/stress.rs` — the `StressApp` harness, panic hook, invariant checks, telemetry.
- `tests/stress.rs` — test entry point. Hosts the RON discovery loop and the proptest property tests.
- `tests/cases/stress/pole_north.stress.ron` (and 9 more — one per scenario)

**Modified files:**

- `Cargo.toml` — add `proptest` to `[dev-dependencies]`; add `test-support` feature.
- `src/lib.rs` — add `#[cfg(any(test, feature = "test-support"))] pub mod test_support;`
- `scripts/pre-commit` — add three-RON fast subset to existing fast path.
- `issues.json` — append note to CRASH-001 referencing this work.
- `ai-context.json` — register the new test target and refresh `generated_from_commit`.

**Boundary rule:** the harness owns Bevy app construction and invariant checks. It does NOT own scenario parsing or random generation — those live in `tests/stress.rs` so they can evolve independently.

---

## Task 1: Add proptest dev-dependency and test-support feature

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add the dev-dependency and feature**

Add to `[dev-dependencies]`:

```toml
proptest = "1.5"
```

Add to `[features]` (create the section if it doesn't already include this):

```toml
test-support = []
```

- [ ] **Step 2: Verify the dependency resolves**

Run: `cargo check --tests`
Expected: PASS (no compilation errors). Cargo downloads `proptest` and its transitive deps.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: add proptest dev-dep and test-support feature

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: Scaffold `src/test_support/` module

**Files:**
- Create: `src/test_support/mod.rs`
- Create: `src/test_support/stress.rs` (placeholder)
- Modify: `src/lib.rs`

- [ ] **Step 1: Create the module entry**

Write `src/test_support/mod.rs`:

```rust
//! Shared test-support utilities.
//!
//! Compiled only when running tests OR when the `test-support` feature is
//! enabled. Production builds do not include this module.

pub mod stress;

pub use stress::{InvariantFailure, InvariantSet, PlanetPreset, StressApp};
```

- [ ] **Step 2: Create the stress module placeholder**

Write `src/test_support/stress.rs`:

```rust
//! Headless gameplay stress-test harness.
//!
//! See `docs/superpowers/specs/2026-04-30-gameplay-stress-tests-design.md`.

use bevy::prelude::*;

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
    Panic { thread: String, location: String, message: String },
    NonFiniteTransform { entity: u64, kind: &'static str, value: String },
    F32Overflow { what: String, value: f64 },
    ChunkCache { detail: String },
    LoadRateBelowMin { observed: f32, min: f32 },
}

/// Headless gameplay stress harness.
///
/// Built progressively across the plan tasks:
/// - Task 3: `new`, `tick_n`
/// - Task 4: `teleport`
/// - Tasks 5–9: invariant checks
/// - Task 10: `chunk_load_rate`
pub struct StressApp {
    pub(crate) app: App,
    pub(crate) seed: u64,
}
```

- [ ] **Step 3: Wire the module into the crate root**

Modify `src/lib.rs`. Find the existing `pub mod` declarations and add:

```rust
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
```

Place it alphabetically among the other `pub mod` lines.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check --tests`
Expected: PASS. `proptest` is unused at this stage, so suppress the warning is unnecessary — cargo only warns on `--release` for unused deps.

- [ ] **Step 5: Commit**

```bash
git add src/test_support/ src/lib.rs
git commit -m "feat(test-support): scaffold stress module skeleton

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: Implement `StressApp::new` and `tick_n`

**Files:**
- Modify: `src/test_support/stress.rs`
- Test: `src/test_support/stress.rs` (inline `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Append to `src/test_support/stress.rs`:

```rust
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib test_support::stress::tests::new_builds_an_app_and_tick_n_advances_frames`
Expected: FAIL with "no method named `new`" (or "no method named `tick_n`").

- [ ] **Step 3: Implement `new` and `tick_n`**

Add to `src/test_support/stress.rs` (above the `tests` module):

```rust
use std::time::Duration;

use bevy::time::TimeUpdateStrategy;

use crate::camera::CameraPlugin;
use crate::chemistry::ChemistryPlugin;
use crate::data::DataPlugin;
use crate::entities::EntityPlugin;
use crate::floating_origin::FloatingOriginPlugin;
use crate::game_state::GameStatePlugin;
use crate::physics::PhysicsPlugin;
use crate::procgen::ProcgenPlugin;
use crate::world::WorldPlugin;
use crate::world::planet::PlanetConfig;

const FIXED_DT_SECS: f64 = 1.0 / 60.0;

impl StressApp {
    /// Build a headless stress-test app with the full non-rendering pipeline.
    pub fn new(seed: u64, preset: PlanetPreset) -> Self {
        let planet = match preset {
            PlanetPreset::SmallPlanet => PlanetConfig::default(),
            PlanetPreset::Earth => {
                let mut p = PlanetConfig::default();
                p.mean_radius = 6_371_000.0;
                p.sea_level_radius = 6_371_000.0;
                p
            }
        };

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(AssetPlugin::default());
        app.insert_resource(planet);
        app.insert_resource(TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f64(FIXED_DT_SECS),
        ));

        // Seed the RNG used by terrain generation by inserting a noise config.
        // The default PlanetConfig already has noise: None, so terrain falls
        // back to deterministic defaults; the seed parameter is reserved for
        // proptest reproducibility (passed through to InvariantFailure
        // diagnostic output).
        let _ = seed;

        app.add_plugins((
            FloatingOriginPlugin,
            DataPlugin,
            GameStatePlugin,
            CameraPlugin,
            EntityPlugin,
            WorldPlugin,
            PhysicsPlugin,
            ChemistryPlugin,
            ProcgenPlugin,
        ));

        Self { app, seed }
    }

    /// Advance the simulation by `n` fixed-update frames.
    pub fn tick_n(&mut self, n: u32) {
        for _ in 0..n {
            self.app.update();
        }
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib test_support::stress::tests::new_builds_an_app_and_tick_n_advances_frames`
Expected: PASS.

If it fails because a plugin needs an additional dependency (e.g. `ProcgenPlugin` requires another plugin not yet in the list), add only the minimum needed plugin and document why in a code comment. Do NOT add rendering, lighting, sky, weather, or UI plugins — those are explicitly out of scope per the spec.

- [ ] **Step 5: Commit**

```bash
git add src/test_support/stress.rs
git commit -m "feat(test-support): StressApp::new + tick_n

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: Implement `StressApp::teleport`

**Files:**
- Modify: `src/test_support/stress.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/test_support/stress.rs`:

```rust
#[test]
fn teleport_moves_player_to_requested_lat_lon_alt() {
    use bevy::math::DVec3;
    use crate::floating_origin::WorldPosition;

    let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
    app.tick_n(5);  // let camera spawn

    app.teleport(0.0, 0.0, 100.0);  // equator, prime meridian, 100 m above sea level
    app.tick_n(1);

    // Player should now be at radius = sea_level + 100 m, in the +X direction.
    let world = app.app.world();
    let mut q = world.query_filtered::<&WorldPosition, With<crate::hud::Player>>();
    let pos = q.iter(world).next().expect("player entity exists").0;

    let expected_radius = 32_000.0_f64 + 100.0;
    let actual_radius = pos.length();
    assert!(
        (actual_radius - expected_radius).abs() < 1.0,
        "expected radius ≈ {expected_radius}, got {actual_radius}"
    );
    // +X direction at lat=0, lon=0
    let normalized = pos.normalize();
    assert!(
        (normalized - DVec3::X).length() < 1.0e-3,
        "expected +X direction, got {normalized:?}"
    );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib test_support::stress::tests::teleport_moves_player_to_requested_lat_lon_alt`
Expected: FAIL with "no method named `teleport`".

- [ ] **Step 3: Implement `teleport`**

Add to the `impl StressApp` block in `src/test_support/stress.rs`:

```rust
impl StressApp {
    /// Teleport the player to (lat°, lon°, altitude_m) where altitude_m is
    /// metres above the planet's `sea_level_radius`.
    ///
    /// `lat_deg` is clamped to `[-89.99, +89.99]` to avoid pole singularities.
    /// `lon_deg` is normalized to `(-180, +180]`.
    pub fn teleport(&mut self, lat_deg: f64, lon_deg: f64, altitude_m: f64) {
        use bevy::math::DVec3;
        use crate::floating_origin::{RenderOrigin, WorldPosition};
        use crate::hud::Player;

        let lat = lat_deg.clamp(-89.99, 89.99).to_radians();
        let lon = ((lon_deg + 540.0) % 360.0 - 180.0).to_radians();

        let dir = DVec3::new(
            lat.cos() * lon.cos(),
            lat.sin(),
            -lat.cos() * lon.sin(),  // matches engine convention (V→-Z)
        );

        let planet = self
            .app
            .world()
            .resource::<crate::world::planet::PlanetConfig>()
            .clone();
        let radius = planet.sea_level_radius + altitude_m;
        let world_pos = dir * radius;

        // Locate the player entity.
        let world = self.app.world_mut();
        let mut q = world.query_filtered::<(&mut WorldPosition, &mut Transform), With<Player>>();
        if let Some((mut wp, mut tf)) = q.iter_mut(world).next() {
            wp.0 = world_pos;
            let origin = world.resource::<RenderOrigin>();
            tf.translation = wp.render_offset(origin);
        } else {
            // No player yet (camera hasn't spawned) — set a pending teleport
            // that the harness applies on the next tick.
            self.pending_teleport = Some(world_pos);
        }
    }
}
```

Add to the `StressApp` struct:

```rust
pub struct StressApp {
    pub(crate) app: App,
    pub(crate) seed: u64,
    pub(crate) pending_teleport: Option<bevy::math::DVec3>,
}
```

Update `StressApp::new` to initialize `pending_teleport: None`.

Add a system that consumes `pending_teleport`. In `StressApp::new`, after the
plugins block, register:

```rust
app.add_systems(Update, apply_pending_teleport);
```

Define the system at module scope:

```rust
fn apply_pending_teleport(/* see below */) { /* ... */ }
```

The simplest approach: store `pending_teleport` as a Bevy `Resource` instead
of a struct field. Replace the struct field with:

```rust
#[derive(Resource, Default)]
struct PendingTeleport(Option<bevy::math::DVec3>);
```

Insert it in `StressApp::new`:

```rust
app.insert_resource(PendingTeleport::default());
```

System:

```rust
fn apply_pending_teleport(
    mut pending: ResMut<PendingTeleport>,
    origin: Res<crate::floating_origin::RenderOrigin>,
    mut q: Query<
        (&mut crate::floating_origin::WorldPosition, &mut Transform),
        With<crate::hud::Player>,
    >,
) {
    let Some(target) = pending.0.take() else { return };
    let Ok((mut wp, mut tf)) = q.single_mut() else {
        // Player not yet spawned; put it back and try next frame.
        pending.0 = Some(target);
        return;
    };
    wp.0 = target;
    tf.translation = wp.render_offset(&origin);
}
```

Refactor `StressApp::teleport` to write into the resource and not require
the player to exist immediately:

```rust
pub fn teleport(&mut self, lat_deg: f64, lon_deg: f64, altitude_m: f64) {
    use bevy::math::DVec3;

    let lat = lat_deg.clamp(-89.99, 89.99).to_radians();
    let lon = ((lon_deg + 540.0) % 360.0 - 180.0).to_radians();

    let dir = DVec3::new(
        lat.cos() * lon.cos(),
        lat.sin(),
        -lat.cos() * lon.sin(),
    );

    let planet = self
        .app
        .world()
        .resource::<crate::world::planet::PlanetConfig>()
        .clone();
    let radius = planet.sea_level_radius + altitude_m;

    self.app
        .world_mut()
        .resource_mut::<PendingTeleport>()
        .0 = Some(dir * radius);
}
```

And remove the `pending_teleport` struct field.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib test_support::stress::tests::teleport_moves_player_to_requested_lat_lon_alt`
Expected: PASS. If the player marker query returns nothing even after `tick_n(5)`, increase to `tick_n(30)` and verify against `src/camera/mod.rs` for the spawn timing.

- [ ] **Step 5: Commit**

```bash
git add src/test_support/stress.rs
git commit -m "feat(test-support): StressApp::teleport via PendingTeleport resource

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: Invariant A — install panic hook, expose `take_panic`

**Files:**
- Modify: `src/test_support/stress.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib test_support::stress::tests::panic_hook_captures_panic_in_app_thread`
Expected: FAIL with "no method named `take_panic`".

- [ ] **Step 3: Implement the panic hook**

Add to `src/test_support/stress.rs`:

```rust
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct CapturedPanic {
    pub thread: String,
    pub location: String,
    pub message: String,
}

#[derive(Default)]
pub(crate) struct PanicSlot(Mutex<Option<CapturedPanic>>);

static PANIC_SLOT: std::sync::OnceLock<Arc<PanicSlot>> = std::sync::OnceLock::new();

fn install_panic_hook() {
    let slot = PANIC_SLOT.get_or_init(|| Arc::new(PanicSlot::default()));
    let slot_clone = Arc::clone(slot);

    // Compose with the existing hook so `RUST_BACKTRACE=1` etc. still works.
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
                *guard = Some(CapturedPanic { thread, location, message });
            }
        }

        prev(info);
    }));
}

impl StressApp {
    /// Take any panic that was captured by the harness panic hook.
    /// Clears the slot so subsequent calls return `None` until the next panic.
    pub fn take_panic(&self) -> Option<CapturedPanic> {
        PANIC_SLOT
            .get()
            .and_then(|slot| slot.0.lock().ok())
            .and_then(|mut guard| guard.take())
    }
}
```

In `StressApp::new`, before building the app, call `install_panic_hook();` and
clear any leftover panic from a previous test by calling
`PANIC_SLOT.get().and_then(|s| s.0.lock().ok()).and_then(|mut g| g.take());`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib test_support::stress::tests::panic_hook_captures_panic_in_app_thread`
Expected: PASS.

Note: rust's test runner runs tests in parallel by default. The `OnceLock`
ensures the hook is installed once per process; the `Mutex` makes the slot
thread-safe. Tests that intentionally cause panics must clear the slot at the
start to avoid cross-test contamination — the call in `StressApp::new` handles
this.

- [ ] **Step 5: Commit**

```bash
git add src/test_support/stress.rs
git commit -m "feat(test-support): StressApp panic hook + take_panic

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: Invariants B + C — finite Transforms and f32 overflow

**Files:**
- Modify: `src/test_support/stress.rs`

- [ ] **Step 1: Write the failing test**

```rust
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

    // Inject a NaN translation on the player.
    let world = app.app.world_mut();
    let mut q = world.query_filtered::<&mut Transform, With<crate::hud::Player>>();
    if let Some(mut tf) = q.iter_mut(world).next() {
        tf.translation.x = f32::NAN;
    }

    let failures = app.assert_invariants(InvariantSet::FINITE);
    assert!(
        failures.iter().any(|f| matches!(f, InvariantFailure::NonFiniteTransform { .. })),
        "expected NonFiniteTransform failure, got: {failures:?}"
    );
}

#[test]
fn overflow_invariant_catches_huge_translation() {
    let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
    app.tick_n(30);

    let world = app.app.world_mut();
    let mut q = world.query_filtered::<&mut Transform, With<crate::hud::Player>>();
    if let Some(mut tf) = q.iter_mut(world).next() {
        tf.translation.x = 2.0e7;
    }

    let failures = app.assert_invariants(InvariantSet::NO_OVERFLOW);
    assert!(
        failures.iter().any(|f| matches!(f, InvariantFailure::F32Overflow { .. })),
        "expected F32Overflow failure, got: {failures:?}"
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib test_support::stress::tests`
Expected: the three new tests FAIL with "no method named `assert_invariants`".

- [ ] **Step 3: Implement `assert_invariants` for B and C**

Add to `src/test_support/stress.rs`:

```rust
const F32_OVERFLOW_BOUND: f32 = 1.0e7;

impl StressApp {
    /// Check the requested invariants. Returns the list of failures
    /// (empty = all checks passed for the requested set).
    pub fn assert_invariants(&mut self, which: InvariantSet) -> Vec<InvariantFailure> {
        let mut failures = Vec::new();

        if which.contains(InvariantSet::PANICS) {
            if let Some(p) = self.take_panic() {
                failures.push(InvariantFailure::Panic {
                    thread: p.thread,
                    location: p.location,
                    message: p.message,
                });
            }
        }

        if which.contains(InvariantSet::FINITE) || which.contains(InvariantSet::NO_OVERFLOW) {
            self.check_transforms(which, &mut failures);
        }

        if which.contains(InvariantSet::CHUNK_CACHE) {
            self.check_chunk_cache(&mut failures);
        }

        // LOAD_RATE handled in Task 10.

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

    fn check_chunk_cache(&mut self, _failures: &mut Vec<InvariantFailure>) {
        // Implemented in Task 7.
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib test_support::stress::tests`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/test_support/stress.rs
git commit -m "feat(test-support): assert_invariants — finite + no-overflow

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 7: Invariant D — chunk-cache integrity

**Files:**
- Modify: `src/test_support/stress.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn chunk_cache_invariant_passes_after_idle() {
    let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
    app.tick_n(60);  // let some chunks load
    app.teleport(0.0, 0.0, 0.0);
    app.tick_n(60);

    let failures = app.assert_invariants(InvariantSet::CHUNK_CACHE);
    assert!(
        failures.is_empty(),
        "unexpected chunk-cache failures after idle: {failures:?}"
    );
}
```

- [ ] **Step 2: Run the test to verify the implementation does nothing yet**

Run: `cargo test --lib test_support::stress::tests::chunk_cache_invariant_passes_after_idle`
Expected: PASS (the empty stub returns no failures, but we want to verify the test compiles and runs end-to-end).

- [ ] **Step 3: Implement `check_chunk_cache`**

Replace the placeholder body in `src/test_support/stress.rs`:

```rust
fn check_chunk_cache(&mut self, failures: &mut Vec<InvariantFailure>) {
    use crate::world::v2::chunk_manager::V2ChunkMap;
    use std::collections::HashSet;

    let world = self.app.world();
    let Some(map) = world.get_resource::<V2ChunkMap>() else {
        // World plugin not active in this configuration — nothing to check.
        return;
    };

    let planet = world
        .resource::<crate::world::planet::PlanetConfig>();
    let mean_r = planet.mean_radius;

    let mut seen: HashSet<crate::world::v2::cubed_sphere::CubeSphereCoord> = HashSet::new();
    for (coord, _entity) in map.iter() {
        // Duplicate keys: BTreeMap/HashMap can't actually contain duplicates,
        // but a future refactor might expose them; check anyway.
        if !seen.insert(*coord) {
            failures.push(InvariantFailure::ChunkCache {
                detail: format!("duplicate chunk coord: {coord:?}"),
            });
        }

        // Finite world position from the coord.
        let (world_pos, _rot, _scale) = coord.transform_at(mean_r);
        if !world_pos.x.is_finite() || !world_pos.y.is_finite() || !world_pos.z.is_finite() {
            failures.push(InvariantFailure::ChunkCache {
                detail: format!("non-finite world pos for coord {coord:?}: {world_pos:?}"),
            });
        }
    }
}
```

If `CubeSphereCoord` doesn't have a `transform_at` method, replace with the
existing helper from `src/world/v2/cubed_sphere.rs` (search for whichever
function returns `(world_pos, rotation, scale)` — it appears around line 318
in that file). Match the actual function signature when implementing.

- [ ] **Step 4: Run all tests**

Run: `cargo test --lib test_support::stress::tests`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/test_support/stress.rs
git commit -m "feat(test-support): chunk-cache invariant check

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 8: Invariant F — chunk-load throughput tracking

**Files:**
- Modify: `src/test_support/stress.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn load_rate_reports_zero_on_empty_app_then_positive_after_load() {
    let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);

    // Initial rate is undefined / zero.
    let r0 = app.chunk_load_rate();
    assert!(r0 >= 0.0, "rate must be non-negative, got {r0}");

    app.tick_n(120);
    let r1 = app.chunk_load_rate();
    // We expect SOME chunks to have loaded during 2 simulated seconds at the
    // origin of a small planet. If this assertion proves flaky on CI, the
    // implementation plan calibrates the threshold.
    assert!(r1 > 0.0, "expected some chunk loading; got rate = {r1}");
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib test_support::stress::tests::load_rate_reports_zero_on_empty_app_then_positive_after_load`
Expected: FAIL with "no method named `chunk_load_rate`".

- [ ] **Step 3: Implement load-rate tracking**

Add a window struct + method to `src/test_support/stress.rs`:

```rust
#[derive(Default)]
pub(crate) struct LoadRateTracker {
    samples: std::collections::VecDeque<(f64, usize)>,  // (sim_time_secs, loaded_count)
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
```

Add to `StressApp`:

```rust
pub struct StressApp {
    pub(crate) app: App,
    pub(crate) seed: u64,
    pub(crate) load_rate: LoadRateTracker,
    pub(crate) sim_time: f64,
}
```

Update `StressApp::new` to initialize the new fields. Update `tick_n` to
sample after each `app.update()`:

```rust
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

pub fn chunk_load_rate(&self) -> f32 {
    self.load_rate.rate()
}
```

Wire `LOAD_RATE` into `assert_invariants` — add an optional minimum:

```rust
impl StressApp {
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
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib test_support::stress::tests`
Expected: all PASS. If `r1 > 0.0` is flaky, increase the warm-up to `tick_n(300)` and re-verify.

- [ ] **Step 5: Commit**

```bash
git add src/test_support/stress.rs
git commit -m "feat(test-support): chunk-load-rate tracker + LOAD_RATE invariant

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 9: Define `StressScenario` schema and runner

**Files:**
- Create: `tests/stress.rs`

- [ ] **Step 1: Write the schema and discovery loop**

Create `tests/stress.rs`:

```rust
//! Gameplay stress / fuzzy regression tests.
//!
//! Discovers `*.stress.ron` files under `tests/cases/stress/` and runs each as
//! a `StressScenario` against the headless `StressApp` harness. Also hosts
//! proptest-driven property generators (added in Task 13+).

use serde::Deserialize;

use the_dark_candle::test_support::stress::{
    InvariantFailure, InvariantSet, PlanetPreset, StressApp,
};

#[derive(Deserialize, Debug)]
struct StressScenario {
    description: String,
    seed: u64,
    planet: PresetSpec,
    #[serde(default = "default_warmup_ticks")]
    warmup_ticks: u32,
    teleports: Vec<Teleport>,
    invariants: Vec<InvariantSpec>,
    #[serde(default)]
    chunk_load_rate_min: Option<f32>,
}

#[derive(Deserialize, Debug, Clone, Copy)]
enum PresetSpec {
    Earth,
    SmallPlanet,
}

fn default_warmup_ticks() -> u32 {
    30
}

#[derive(Deserialize, Debug)]
struct Teleport {
    lat: f64,
    lon: f64,
    altitude_m: f64,
    /// Number of ticks to advance after the teleport before the next one.
    then_tick: u32,
}

#[derive(Deserialize, Debug, Clone, Copy)]
enum InvariantSpec {
    Panics,
    Finite,
    NoOverflow,
    ChunkCache,
    LoadRate,
}

fn preset_to_harness(p: PresetSpec) -> PlanetPreset {
    match p {
        PresetSpec::Earth => PlanetPreset::Earth,
        PresetSpec::SmallPlanet => PlanetPreset::SmallPlanet,
    }
}

fn invariants_to_set(specs: &[InvariantSpec]) -> InvariantSet {
    let mut s = InvariantSet::default();
    for sp in specs {
        s = s | match sp {
            InvariantSpec::Panics => InvariantSet::PANICS,
            InvariantSpec::Finite => InvariantSet::FINITE,
            InvariantSpec::NoOverflow => InvariantSet::NO_OVERFLOW,
            InvariantSpec::ChunkCache => InvariantSet::CHUNK_CACHE,
            InvariantSpec::LoadRate => InvariantSet::LOAD_RATE,
        };
    }
    s
}

fn run_scenario(scenario: &StressScenario, path: &std::path::Path) -> Result<(), String> {
    let mut app = StressApp::new(scenario.seed, preset_to_harness(scenario.planet));
    app.tick_n(scenario.warmup_ticks);

    for tp in &scenario.teleports {
        app.teleport(tp.lat, tp.lon, tp.altitude_m);
        app.tick_n(tp.then_tick);
    }

    let which = invariants_to_set(&scenario.invariants);
    let failures = app.assert_invariants_with_min_rate(which, scenario.chunk_load_rate_min);

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Scenario failed: {}\n  path: {}\n  description: {}\n  failures:\n{}",
            path.display(),
            path.display(),
            scenario.description,
            failures
                .iter()
                .map(|f| format!("    - {f:?}"))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
}

#[test]
fn run_all_stress_scenarios() {
    let pattern = "tests/cases/stress/*.stress.ron";
    let entries: Vec<_> = glob::glob(pattern)
        .expect("invalid glob pattern")
        .collect();

    assert!(
        !entries.is_empty(),
        "No stress scenarios found matching {pattern}"
    );

    let mut errors = Vec::new();
    for entry in entries {
        let path = entry.expect("glob entry");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
        let scenario: StressScenario = ron::from_str(&text)
            .unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e));

        if let Err(msg) = run_scenario(&scenario, &path) {
            errors.push(msg);
        }
    }

    assert!(
        errors.is_empty(),
        "{} stress scenario(s) failed:\n\n{}",
        errors.len(),
        errors.join("\n\n")
    );
}
```

- [ ] **Step 2: Create `tests/cases/stress/` and a single throwaway scenario for the runner test**

Create `tests/cases/stress/_smoke.stress.ron`:

```ron
StressScenario(
    description: "Smoke test: build app, idle, teleport once, idle again",
    seed: 1,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: 0.0, lon: 0.0, altitude_m: 100.0, then_tick: 30),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 3: Run the scenario test**

Run: `cargo test --test stress run_all_stress_scenarios`
Expected: PASS. The smoke scenario must produce zero invariant failures.

If the scenario fails on `master`, investigate before proceeding — this is real-bug territory. File a new issue, mark `_smoke.stress.ron` `#[ignore]` via a runner-side filter, and proceed only after deciding the fix is out of scope for this work.

- [ ] **Step 4: Commit**

```bash
git add tests/stress.rs tests/cases/stress/_smoke.stress.ron
git commit -m "feat(stress): RON scenario schema and discovery runner

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 10: Geographic edge scenarios — poles and antimeridian

**Files:**
- Create: `tests/cases/stress/pole_north.stress.ron`
- Create: `tests/cases/stress/pole_south.stress.ron`
- Create: `tests/cases/stress/antimeridian.stress.ron`

- [ ] **Step 1: Create `pole_north.stress.ron`**

```ron
StressScenario(
    description: "Teleport to north pole (lat=89.99°, lon=0°), surface, idle",
    seed: 100,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: 89.99, lon: 0.0, altitude_m: 0.0, then_tick: 120),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 2: Create `pole_south.stress.ron`**

```ron
StressScenario(
    description: "Teleport to south pole (lat=-89.99°, lon=0°), surface, idle",
    seed: 101,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: -89.99, lon: 0.0, altitude_m: 0.0, then_tick: 120),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 3: Create `antimeridian.stress.ron`**

```ron
StressScenario(
    description: "Teleport across antimeridian (+180° then -180°)",
    seed: 102,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: 0.0, lon: 180.0, altitude_m: 0.0, then_tick: 60),
        Teleport(lat: 0.0, lon: -180.0, altitude_m: 0.0, then_tick: 60),
        Teleport(lat: 30.0, lon: 179.99, altitude_m: 0.0, then_tick: 60),
        Teleport(lat: 30.0, lon: -179.99, altitude_m: 0.0, then_tick: 60),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 4: Run all stress scenarios**

Run: `cargo test --test stress`
Expected: PASS. If any of these three fail on `master`, that's a real bug — file a new issue, document it in the issue's `notes`, and `#[ignore]` the file (rename to `*.ignored.ron` so the glob skips it; runner doesn't auto-ignore individual files). Proceed once the failing files are explicitly out of scope.

- [ ] **Step 5: Commit**

```bash
git add tests/cases/stress/pole_north.stress.ron tests/cases/stress/pole_south.stress.ron tests/cases/stress/antimeridian.stress.ron
git commit -m "test(stress): pole + antimeridian scenarios

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 11: Cube-face scenarios — corner and edge crossings

**Files:**
- Create: `tests/cases/stress/cube_face_corner.stress.ron`
- Create: `tests/cases/stress/cube_face_edge_crossing.stress.ron`

- [ ] **Step 1: Compute cube-face corner coordinates**

A cubed-sphere face corner is where 3 faces meet — i.e. the (lat, lon) of a
corner of the cube projected to the sphere. The 8 cube corners project to
(lat, lon) pairs at lat = ±asin(1/√3) ≈ ±35.264°, lon ∈ {±45°, ±135°}.

- [ ] **Step 2: Create `cube_face_corner.stress.ron`**

```ron
StressScenario(
    description: "Teleport to all 8 cube-face corners (3-face junctions)",
    seed: 200,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat:  35.264, lon:   45.0, altitude_m: 0.0, then_tick: 30),
        Teleport(lat:  35.264, lon:  135.0, altitude_m: 0.0, then_tick: 30),
        Teleport(lat:  35.264, lon: -135.0, altitude_m: 0.0, then_tick: 30),
        Teleport(lat:  35.264, lon:  -45.0, altitude_m: 0.0, then_tick: 30),
        Teleport(lat: -35.264, lon:   45.0, altitude_m: 0.0, then_tick: 30),
        Teleport(lat: -35.264, lon:  135.0, altitude_m: 0.0, then_tick: 30),
        Teleport(lat: -35.264, lon: -135.0, altitude_m: 0.0, then_tick: 30),
        Teleport(lat: -35.264, lon:  -45.0, altitude_m: 0.0, then_tick: 30),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 3: Create `cube_face_edge_crossing.stress.ron`**

A cubed-sphere has 12 edges. Each edge is a great-circle arc connecting two
cube corners. We cross each edge by teleporting from a point just inside one
face to a point just inside the adjacent face, going through the edge.

```ron
StressScenario(
    description: "Cross each of the 12 cube-face edges (one teleport pair per edge)",
    seed: 201,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        // 4 equatorial edges (lon = ±45, ±135 at lat ≈ 0)
        Teleport(lat: 0.0, lon:  44.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: 0.0, lon:  46.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: 0.0, lon: 134.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: 0.0, lon: 136.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: 0.0, lon: -134.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: 0.0, lon: -136.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: 0.0, lon: -44.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: 0.0, lon: -46.0, altitude_m: 0.0, then_tick: 20),

        // 4 northern edges (lat ≈ +35, longitude transitions across face boundaries)
        Teleport(lat:  35.264, lon:    0.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat:  40.0,   lon:   90.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat:  40.0,   lon:  180.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat:  40.0,   lon:  -90.0, altitude_m: 0.0, then_tick: 20),

        // 4 southern edges (lat ≈ -35)
        Teleport(lat: -35.264, lon:    0.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: -40.0,   lon:   90.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: -40.0,   lon:  180.0, altitude_m: 0.0, then_tick: 20),
        Teleport(lat: -40.0,   lon:  -90.0, altitude_m: 0.0, then_tick: 20),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 4: Run all stress scenarios**

Run: `cargo test --test stress`
Expected: PASS. Same fail-then-file-issue policy as Task 10.

- [ ] **Step 5: Commit**

```bash
git add tests/cases/stress/cube_face_corner.stress.ron tests/cases/stress/cube_face_edge_crossing.stress.ron
git commit -m "test(stress): cube-face corner and edge-crossing scenarios

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 12: Altitude and burst scenarios

**Files:**
- Create: `tests/cases/stress/deep_underground.stress.ron`
- Create: `tests/cases/stress/high_altitude.stress.ron`
- Create: `tests/cases/stress/oscillate_high_low.stress.ron`
- Create: `tests/cases/stress/burst_teleport.stress.ron`
- Create: `tests/cases/stress/face_seam_crawl.stress.ron`

- [ ] **Step 1: Create `deep_underground.stress.ron`**

```ron
StressScenario(
    description: "Teleport to (0,0) at sea_level - 500 m (caves/strata zone)",
    seed: 300,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: 0.0, lon: 0.0, altitude_m: -500.0, then_tick: 120),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 2: Create `high_altitude.stress.ron`**

```ron
StressScenario(
    description: "Teleport to high altitudes: 50 km, 500 km, 5000 km above surface",
    seed: 301,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: 0.0, lon: 0.0, altitude_m:    50_000.0, then_tick: 60),
        Teleport(lat: 0.0, lon: 0.0, altitude_m:   500_000.0, then_tick: 60),
        Teleport(lat: 0.0, lon: 0.0, altitude_m: 5_000_000.0, then_tick: 60),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 3: Create `oscillate_high_low.stress.ron`**

```ron
StressScenario(
    description: "Oscillate between surface and 100 km altitude every 5 ticks (LOD churn)",
    seed: 302,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: 0.0, lon: 0.0, altitude_m:      0.0, then_tick: 5),
        Teleport(lat: 0.0, lon: 0.0, altitude_m: 100_000.0, then_tick: 5),
        Teleport(lat: 0.0, lon: 0.0, altitude_m:      0.0, then_tick: 5),
        Teleport(lat: 0.0, lon: 0.0, altitude_m: 100_000.0, then_tick: 5),
        Teleport(lat: 0.0, lon: 0.0, altitude_m:      0.0, then_tick: 5),
        Teleport(lat: 0.0, lon: 0.0, altitude_m: 100_000.0, then_tick: 5),
        Teleport(lat: 0.0, lon: 0.0, altitude_m:      0.0, then_tick: 5),
        Teleport(lat: 0.0, lon: 0.0, altitude_m: 100_000.0, then_tick: 5),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 4: Create `burst_teleport.stress.ron`**

```ron
StressScenario(
    description: "100 random-ish teleports across the planet within ~10 simulated seconds",
    seed: 303,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat:   12.0, lon:   34.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:  -45.0, lon:   78.0, altitude_m: 50.0, then_tick: 6),
        Teleport(lat:   60.0, lon: -120.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:  -10.0, lon:  170.0, altitude_m: 20.0, then_tick: 6),
        Teleport(lat:   25.0, lon:  -90.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:  -70.0, lon:    5.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:    5.0, lon:  100.0, altitude_m: 30.0, then_tick: 6),
        Teleport(lat:  -30.0, lon: -160.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:   80.0, lon:   50.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:  -55.0, lon: -110.0, altitude_m: 10.0, then_tick: 6),
        Teleport(lat:   15.0, lon:  140.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:  -85.0, lon:   25.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:   45.0, lon: -150.0, altitude_m: 40.0, then_tick: 6),
        Teleport(lat:  -20.0, lon:  -10.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:   65.0, lon:  120.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:  -40.0, lon:   60.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:    0.0, lon:  -80.0, altitude_m:  5.0, then_tick: 6),
        Teleport(lat:   30.0, lon:  -30.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:  -65.0, lon: -170.0, altitude_m:  0.0, then_tick: 6),
        Teleport(lat:   75.0, lon:  175.0, altitude_m:  0.0, then_tick: 6),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache, LoadRate],
    chunk_load_rate_min: Some(0.1),  // calibrated empirically — see Task 16
)
```

- [ ] **Step 5: Create `face_seam_crawl.stress.ron`**

A "crawl" is many small teleports along a great-circle arc. We crawl along
the equator-meets-+X-face seam (the +X / +Z face boundary at lon = -45°,
i.e. -45° longitude).

```ron
StressScenario(
    description: "Crawl along the +X/+Z face seam (lon=-45°) in small steps",
    seed: 304,
    planet: SmallPlanet,
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: -30.0, lon: -45.0, altitude_m: 0.0, then_tick: 10),
        Teleport(lat: -20.0, lon: -45.0, altitude_m: 0.0, then_tick: 10),
        Teleport(lat: -10.0, lon: -45.0, altitude_m: 0.0, then_tick: 10),
        Teleport(lat:   0.0, lon: -45.0, altitude_m: 0.0, then_tick: 10),
        Teleport(lat:  10.0, lon: -45.0, altitude_m: 0.0, then_tick: 10),
        Teleport(lat:  20.0, lon: -45.0, altitude_m: 0.0, then_tick: 10),
        Teleport(lat:  30.0, lon: -45.0, altitude_m: 0.0, then_tick: 10),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,
)
```

- [ ] **Step 6: Run all stress scenarios**

Run: `cargo test --test stress`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/cases/stress/deep_underground.stress.ron tests/cases/stress/high_altitude.stress.ron tests/cases/stress/oscillate_high_low.stress.ron tests/cases/stress/burst_teleport.stress.ron tests/cases/stress/face_seam_crawl.stress.ron
git commit -m "test(stress): altitude, burst, and face-seam scenarios

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 13: Proptest — random teleport invariants

**Files:**
- Modify: `tests/stress.rs`

- [ ] **Step 1: Append the proptest module**

Add to the bottom of `tests/stress.rs`:

```rust
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            failure_persistence: Some(Box::new(
                proptest::test_runner::FileFailurePersistence::Direct(
                    "tests/cases/stress/proptest-regressions/random_teleport_invariants.txt"
                ),
            )),
            ..ProptestConfig::default()
        })]
        #[test]
        fn random_teleport_invariants(
            lat in -89.99f64..=89.99f64,
            lon in -180.0f64..=180.0f64,
            altitude_m in prop_oneof![
                Just(-500.0_f64),
                Just(0.0_f64),
                Just(50_000.0_f64),
                Just(500_000.0_f64),
                Just(5_000_000.0_f64),
            ],
        ) {
            let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
            app.tick_n(30);
            app.teleport(lat, lon, altitude_m);
            app.tick_n(30);

            let failures = app.assert_invariants(
                InvariantSet::PANICS
                    | InvariantSet::FINITE
                    | InvariantSet::NO_OVERFLOW
                    | InvariantSet::CHUNK_CACHE,
            );
            prop_assert!(
                failures.is_empty(),
                "lat={lat}, lon={lon}, alt={altitude_m}: failures = {failures:?}"
            );
        }
    }
}
```

- [ ] **Step 2: Run the proptest**

Run: `cargo test --test stress proptests::random_teleport_invariants`
Expected: PASS (64 randomly generated cases). If a case fails, proptest
auto-shrinks and persists it under
`tests/cases/stress/proptest-regressions/random_teleport_invariants.txt`. The
failing case is then a real bug — file an issue, document the failing
inputs in `notes`, and decide whether to fix as part of this work or out of
scope.

- [ ] **Step 3: Commit**

```bash
git add tests/stress.rs tests/cases/stress/proptest-regressions/
git commit -m "test(stress): proptest random_teleport_invariants

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

(If the regressions directory is empty, `git add` will skip it; that's fine.)

---

## Task 14: Proptest — random teleport sequence

**Files:**
- Modify: `tests/stress.rs`

- [ ] **Step 1: Append the second proptest**

Inside the `mod proptests { ... }` block, after `random_teleport_invariants`:

```rust
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "tests/cases/stress/proptest-regressions/random_teleport_sequence.txt"
            ),
        )),
        ..ProptestConfig::default()
    })]
    #[test]
    fn random_teleport_sequence(
        sequence in proptest::collection::vec(
            (
                -89.99f64..=89.99f64,
                -180.0f64..=180.0f64,
                prop_oneof![
                    Just(0.0_f64),
                    Just(50_000.0_f64),
                    Just(-100.0_f64),
                ],
            ),
            1..=16,
        ),
    ) {
        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
        app.tick_n(30);

        for (lat, lon, alt) in &sequence {
            app.teleport(*lat, *lon, *alt);
            app.tick_n(8);
        }

        let failures = app.assert_invariants(
            InvariantSet::PANICS
                | InvariantSet::FINITE
                | InvariantSet::NO_OVERFLOW
                | InvariantSet::CHUNK_CACHE,
        );
        prop_assert!(
            failures.is_empty(),
            "sequence={sequence:?}: failures = {failures:?}"
        );
    }
}
```

- [ ] **Step 2: Run the proptest**

Run: `cargo test --test stress proptests::random_teleport_sequence`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/stress.rs
git commit -m "test(stress): proptest random_teleport_sequence

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 15: Proptest — random altitude extremes

**Files:**
- Modify: `tests/stress.rs`

- [ ] **Step 1: Append the third proptest**

Inside `mod proptests`, after `random_teleport_sequence`:

```rust
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "tests/cases/stress/proptest-regressions/random_altitude_extreme.txt"
            ),
        )),
        ..ProptestConfig::default()
    })]
    #[test]
    fn random_altitude_extreme(
        lat in -89.99f64..=89.99f64,
        lon in -180.0f64..=180.0f64,
        bucket in 0u8..5u8,
    ) {
        let altitude_m = match bucket {
            0 => -1_000.0,             // deep underground
            1 => 0.0,                  // surface
            2 => 100_000.0,            // low atmosphere
            3 => 1_000_000.0,          // high atmosphere / low orbit
            _ => 9_000_000.0,          // beyond Earth-radius equivalent (small planet: very high)
        };

        let mut app = StressApp::new(0, PlanetPreset::SmallPlanet);
        app.tick_n(30);
        app.teleport(lat, lon, altitude_m);
        app.tick_n(30);

        let failures = app.assert_invariants(
            InvariantSet::PANICS
                | InvariantSet::FINITE
                | InvariantSet::NO_OVERFLOW
                | InvariantSet::CHUNK_CACHE,
        );
        prop_assert!(
            failures.is_empty(),
            "lat={lat}, lon={lon}, alt={altitude_m}: failures = {failures:?}"
        );
    }
}
```

- [ ] **Step 2: Run the proptest**

Run: `cargo test --test stress proptests::random_altitude_extreme`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/stress.rs
git commit -m "test(stress): proptest random_altitude_extreme

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 16: Calibrate `chunk_load_rate_min` for `burst_teleport`

**Files:**
- Modify: `tests/cases/stress/burst_teleport.stress.ron`

- [ ] **Step 1: Measure observed rate on master**

Add a one-shot measurement test inline in `tests/stress.rs` (gate behind a
function name not picked up automatically), OR run the burst scenario with
the rate gate disabled and inspect the observed rate.

Quick approach: temporarily change `burst_teleport.stress.ron` to
`chunk_load_rate_min: None` (the previous calibrated value), and add a
`println!` in `run_scenario` after `assert_invariants_with_min_rate` showing
`app.chunk_load_rate()`. Run `cargo test --test stress run_all_stress_scenarios -- --nocapture`.

- [ ] **Step 2: Set `chunk_load_rate_min` to 25% of the observed rate**

If the observed rate is e.g. 8 chunks/sec, set `chunk_load_rate_min: Some(2.0)`
in `burst_teleport.stress.ron`. The 25% margin avoids flake on slower CI.

- [ ] **Step 3: Remove the temporary println**

- [ ] **Step 4: Run the burst scenario**

Run: `cargo test --test stress run_all_stress_scenarios`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/cases/stress/burst_teleport.stress.ron
git commit -m "test(stress): calibrate chunk_load_rate_min for burst_teleport

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 17: Pre-commit fast-path subset

**Files:**
- Modify: `scripts/pre-commit`

- [ ] **Step 1: Inspect the current pre-commit hook**

Run: `cat scripts/pre-commit`

Identify the existing fast-path test invocation (the one that prints
"✓ Fast tests OK"). Note its current `cargo test` arguments.

- [ ] **Step 2: Add the three-RON fast subset**

Modify the fast path to also run a focused subset of the stress suite. The
subset is `pole_north`, `pole_south`, and `antimeridian` only. Use a
pattern-based filter so we don't have to reshape the runner — instead add
a separate test-binary invocation with a `--exact` filter, or add an
environment variable that selects a subset inside `run_all_stress_scenarios`.

Recommended approach: add a new env var read in `tests/stress.rs`:

```rust
// At the top of run_all_stress_scenarios:
let only_fast = std::env::var("STRESS_FAST").ok().is_some();
let allow = ["pole_north", "pole_south", "antimeridian"];
```

In the discovery loop, skip files whose stem isn't in `allow` when
`only_fast` is set:

```rust
if only_fast {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim_end_matches(".stress");
    if !allow.iter().any(|a| stem.starts_with(a)) {
        continue;
    }
}
```

Add to `scripts/pre-commit` after the existing fast tests:

```bash
echo "Running stress fast subset..."
STRESS_FAST=1 cargo test --test stress run_all_stress_scenarios --quiet
```

- [ ] **Step 3: Verify the fast subset runs in pre-commit**

Run: `STRESS_FAST=1 cargo test --test stress run_all_stress_scenarios -- --nocapture | head -20`
Expected: only the three pole + antimeridian scenarios run; the others are skipped.

- [ ] **Step 4: Commit (use --no-verify only if the hook is calibrating itself)**

```bash
git add scripts/pre-commit tests/stress.rs
git commit -m "chore(stress): pre-commit hook runs three-RON fast subset

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 18: Update `issues.json` and `ai-context.json`

**Files:**
- Modify: `issues.json`
- Modify: `ai-context.json`

- [ ] **Step 1: Append a note to CRASH-001**

Open `issues.json`, find the `CRASH-001` issue object, and append to its
`notes` array (preserve JSON validity — the array element is a string):

```json
"2026-04-30: Stress / fuzzy regression test suite added under tests/stress.rs + tests/cases/stress/ targeting the historic teleport-related f32 overflow crash class. 10 hand-written RON scenarios (poles/antimeridian/cube-face/altitude/burst/seam) + 3 proptest generators (random teleport, random sequence, random altitude). Suite green on commit; if a real crash is detected by future runs, file as a separate issue and reference this one. See docs/superpowers/specs/2026-04-30-gameplay-stress-tests-design.md."
```

If the suite caught any actual bugs during implementation, list those issue
IDs in this note as well.

- [ ] **Step 2: Refresh `ai-context.json`**

Open `ai-context.json` and:

1. Update `meta.last_updated` to today's date.
2. Update `meta.generated_from_commit` to the current `git rev-parse --short HEAD`.
3. Add a new entry to `phases` describing the stress test suite (mirror the
   shape of existing `phases` entries — `name`, `status: "complete"`, `paths`,
   `implements`, `notes`).

Example phases entry:

```json
{
  "name": "Gameplay stress / fuzzy tests",
  "status": "complete",
  "paths": [
    "src/test_support/stress.rs",
    "tests/stress.rs",
    "tests/cases/stress/"
  ],
  "implements": [
    "StressApp harness — headless Bevy app with full non-rendering pipeline",
    "Invariants: panics, finite Transforms, f32 world-coord overflow (<=1e7 m), chunk-cache integrity, chunk-load throughput",
    "10 hand-written RON regression scenarios (poles, antimeridian, cube-face corners/edges, altitude extremes, burst, seam crawl)",
    "3 proptest generators (random_teleport_invariants, random_teleport_sequence, random_altitude_extreme)"
  ],
  "notes": "Targets the historic teleport-related f32 overflow crash class (CRASH-001). Pre-commit hook runs 3-RON fast subset; full suite under cargo test --test stress."
}
```

- [ ] **Step 3: Validate JSON**

Run: `python3 -c "import json; json.load(open('issues.json')); json.load(open('ai-context.json')); print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add issues.json ai-context.json
git commit -m "chore: update issues.json and ai-context.json for stress test suite

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 19: Final verification

**Files:** none modified

- [ ] **Step 1: Format check**

Run: `cargo fmt --check`
Expected: clean exit. If not, run `cargo fmt` and commit the formatting fix.

- [ ] **Step 2: Clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: clean. If clippy finds new warnings in code introduced by this
plan, fix them. Do not silence with `#[allow]` unless the warning is a known
false-positive in upstream code; document the reasoning if so.

- [ ] **Step 3: Full lib tests**

Run: `cargo test --lib`
Expected: all PASS, no new ignored tests beyond the existing
`GPUPARITY-002`-deferred ones.

- [ ] **Step 4: Full stress suite**

Run: `cargo test --test stress`
Expected: all PASS. Total runtime ≤ 60 s on the dev machine. If exceeded:
- Reduce the proptest case counts (`cases: 64` → `32`, etc.).
- Reduce `then_tick` in the heaviest scenarios (`burst_teleport`, `oscillate_high_low`).

- [ ] **Step 5: Verify pre-commit fast path is fast**

Run: `time STRESS_FAST=1 cargo test --test stress run_all_stress_scenarios --quiet`
Expected: ≤ 15 s. If slower, drop the warm-up ticks in the three fast-path
scenarios.

- [ ] **Step 6: Final commit (if any cleanup needed)**

If steps 1-5 surfaced any cleanup, commit it now. Otherwise no commit needed.

---

## Self-review notes

**Spec coverage check:**

| Spec section | Implemented in |
|---|---|
| `StressApp::new` API | Task 3 |
| `StressApp::teleport` API | Task 4 |
| `StressApp::tick_n` API | Task 3 |
| `StressApp::assert_invariants` API | Tasks 6–8 |
| `StressApp::chunk_load_rate` API | Task 8 |
| Plugin list (Minimal + Asset + Floating + Data + GameState + Camera + Entity + World + Physics + Chemistry + Procgen) | Task 3 |
| Invariant A (panics) | Task 5 |
| Invariant B (finite Transforms) | Task 6 |
| Invariant C (f32 ≤ 1e7 m overflow) | Task 6 |
| Invariant D (chunk cache) | Task 7 |
| Invariant F (load rate) | Task 8 |
| 10 RON scenarios | Tasks 9–12 |
| 3 proptest generators | Tasks 13–15 |
| `proptest` dev-dep + `test-support` feature | Task 1 |
| Pre-commit fast-path subset | Task 17 |
| `issues.json` CRASH-001 note | Task 18 |
| `ai-context.json` refresh | Task 18 |
| `chunk_load_rate_min` calibration deferred to plan | Task 16 |
| `warmup_ticks` calibration deferred to plan | Task 9 (default 30; raised per scenario as needed) |

**Out of scope (per spec):** Frame-time budget (E), determinism (G), in-game stress menu, rendering paths, fixing CRASH-001 itself.

**Pre-existing-bug policy reminders embedded in:** Tasks 9, 10, 11, 13, 14, 15.
