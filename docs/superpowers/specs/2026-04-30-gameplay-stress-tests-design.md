# Gameplay Stress / Fuzzy Tests — Design

**Date:** 2026-04-30
**Status:** Spec — pending implementation plan
**Related issues:** [CRASH-001](../../../issues.json)

## Problem

Historically the game has crashed at runtime. The most concrete known crash class
was teleport-related: f32 multiplication and addition overflows in entity world
coordinates and chunk coordinate arithmetic when the player teleported to extreme
points on the planet. CRASH-001 has been open since 2026-04-01 with no reliable
reproducer; recent rendering and chunk-pipeline overhauls may have fixed it, but
there are no regression tests to prove it stays fixed.

We also lack any "fuzzy" stress coverage of the chunk cache and chunk-load
pipeline under burst load, which historically has been a source of stalls,
deadlocks, and cache-thrashing bugs.

## Goal

Add a headless gameplay stress-test suite that:

1. Boots the **full** non-rendering planet/chunk/camera pipeline.
2. Exercises extreme-but-valid game states (geographic edges, altitude extremes,
   burst teleports, rapid LOD churn).
3. Asserts a fixed set of game-state invariants after each tick.
4. Persists failing random cases as repeatable regression scenarios.

The suite is for **regression detection**, not visual correctness. Visual tests
already exist in the `*_visual*.rs` test entry points.

## Non-Goals

- **Frame-time budget assertions.** Flaky on shared CI hardware. Deferred.
- **Determinism (same seed → same final state) checks.** Useful but additive.
  Deferred.
- **In-game interactive stress menu.** Separate UX feature.
- **Rendering / shader-compilation paths.** The harness builds the app without
  `RenderApp`, `winit`, or shader compilation. Visual regressions are out of
  scope here.
- **Fixing CRASH-001 itself.** If the suite catches a real crash on `master`,
  we file a new issue, `#[ignore]` the failing scenario referencing it, and fix
  it in a separate task. The test suite must be green when first committed.

## Architecture

### Layers

```
tests/cases/stress/*.stress.ron          ← hand-written regression scenarios
tests/stress.rs                          ← test entry point + RON discovery + proptest
src/test_support/stress.rs               ← StressApp harness
  └─ uses real WorldV2Plugin, PlanetPlugin, camera/chunk systems
     (no RenderApp, no winit, no shaders)
```

The harness lives **inside `src/`** under a `cfg(any(test, feature = "test-support"))`
gate so it can be reused from multiple test entry points (`tests/stress.rs`, and
potentially future ones) without duplicating the app-building code.

### `StressApp` API

```rust
pub struct StressApp { app: bevy::app::App, /* + telemetry */ }

impl StressApp {
    pub fn new(seed: u64, planet: PlanetPreset) -> Self;
    pub fn teleport(&mut self, lat_deg: f64, lon_deg: f64, altitude_m: f64);
    pub fn tick_n(&mut self, n: u32);
    pub fn assert_invariants(&self, which: InvariantSet) -> Result<(), Vec<InvariantFailure>>;
    pub fn chunk_load_rate(&self) -> f32;  // chunks loaded / second over the last window
}

pub enum PlanetPreset { Earth, SmallPlanet, Custom(PlanetConfig) }

bitflags! {
    pub struct InvariantSet: u8 {
        const PANICS         = 0b00001;  // A
        const FINITE         = 0b00010;  // B
        const NO_OVERFLOW    = 0b00100;  // C
        const CHUNK_CACHE    = 0b01000;  // D
        const LOAD_RATE      = 0b10000;  // F (must be combined with min-rate threshold)
    }
}
```

### Plugins included in the harness app

Mirror the production plugin set from `src/main.rs`, excluding rendering and
windowing:

- `MinimalPlugins` (no window, no winit)
- `AssetPlugin::default()` (terrain data assets need to load)
- `FloatingOriginPlugin`, `DataPlugin`, `GameStatePlugin`
- `CameraPlugin` (camera-band logic — the historic crash site)
- `EntityPlugin`
- `WorldPlugin` (registers `V2WorldPlugin` internally — real chunk manager,
  cubed-sphere, V2 meshing, terrain generator)
- `PhysicsPlugin`, `ChemistryPlugin`
- `ProcgenPlugin` (biome / `BiomePlugin` registration is required for terrain)

**Excluded:** `DefaultPlugins`, `WindowPlugin`, `WireframePlugin`, `LightingPlugin`,
`SkyPlugin`, `WeatherPlugin`, `MapPlugin`, `HudPlugin`, `InteractionPlugin`,
`AudioPlugin`, `DiagnosticsPlugin`, `AgentCapturePlugin`, `PersistencePlugin`,
`BiologyPlugin`, `BuildingPlugin`, `BehaviorPlugin`, `BodiesPlugin`,
`SocialPlugin` — none are needed to exercise the crash class, and several pull
in window-bound resources.

The implementation plan resolves any minor adjustments (e.g. if `ProcgenPlugin`
turns out to require a plugin we excluded, the plan adds the minimum needed
dependency).

## Invariants (V1)

### A — No panics
Any panic on any thread fails the test. The harness installs a panic hook on
construction that captures `(thread_name, location, message)` and re-panics on
the test thread so the failing case is preserved.

### B — No NaN / Inf in `Transform`
For every entity carrying a `Transform`:
- `transform.translation.is_finite()`
- `transform.rotation.is_finite()` (all four components)
- `transform.scale.is_finite()`

`f32::is_finite()` rejects both NaN and ±Inf.

### C — No f32 overflow on world coords
Concrete bound: `|component| ≤ 1.0e7 m` for every entity `Transform.translation`
and for every `ChunkCoord` after conversion to world meters.

Justification: Earth-scale planet has `mean_radius ≈ 6.37e6 m`. f32 has 24-bit
mantissa → ~0.5 m absolute precision at 6.37e6 m. 1e7 gives 1.57× margin past
Earth radius and remains well below the f32 catastrophic-precision range
(~1.7e7 m, where 1 m + 1 m can round to 1 m).

### D — Chunk cache invariants
- Cache size ≤ the configured chunk-cache cap (looked up at runtime from
  whatever resource/config governs it in `V2WorldPlugin`; the implementation
  plan pins the exact field).
- No duplicate `ChunkCoord` keys in the cache.
- Every cached chunk's `ChunkCoord` produces a finite world position.
- No orphan parent references in the LOD graph (every parent referenced by a
  child exists in the cache OR is `None`).

### F — Chunk-load throughput (burst scenarios only)
For scenarios that opt in via `chunk_load_rate_min: Some(rate)`, after the
warm-up window the system must load at least `rate` new chunks per simulated
second. Catches stalls, deadlocks, and infinite-retry loops that don't panic.

The concrete `rate` is calibrated empirically from a baseline run on `master`
during implementation; the spec does not pin a number.

### Deferred to V2
- E — Frame-time budget (flaky)
- G — Determinism (additive)

## Test Content

### Hand-written RON regression scenarios

Ten scenarios under `tests/cases/stress/`, covering the dimensions the user
called out:

| File | What it exercises |
|---|---|
| `pole_north.stress.ron` | Teleport to lat=89.99°, lon=0, surface |
| `pole_south.stress.ron` | Teleport to lat=-89.99°, lon=0, surface |
| `antimeridian.stress.ron` | Teleport to lon=180° and lon=-180° (both sides agree) |
| `cube_face_corner.stress.ron` | Teleport to a cubed-sphere face-corner (3-face junction) |
| `cube_face_edge_crossing.stress.ron` | Teleport sequence crossing each of the 12 cube-face edges in turn |
| `deep_underground.stress.ron` | Teleport to (0,0) at radius = sea_level − 500 m |
| `high_altitude.stress.ron` | Teleport to altitude = 50 km, 500 km, 5000 km above surface |
| `burst_teleport.stress.ron` | 100 random teleports across the planet within 10 simulated seconds |
| `oscillate_high_low.stress.ron` | Teleport between surface and 100 km every tick for N ticks |
| `face_seam_crawl.stress.ron` | Slow camera motion crawling along a cube-face seam |

### Proptest generators (`tests/stress.rs`)

| Generator | Strategy | Cases per run |
|---|---|---|
| `random_teleport_invariants` | uniform `(lat ∈ [-90°,90°], lon ∈ [-180°,180°], altitude ∈ {-500,0,+50k,+500k,+5M} m)` | 256 |
| `random_teleport_sequence` | sequences of length 1..32 of random teleports | 64 |
| `random_altitude_extreme` | biased generator: {underground, surface, low-atm, high-atm, near-orbit} | 128 |

Failures auto-persist to `tests/cases/stress/proptest-regressions/` via
proptest's built-in regression-file mechanism, ensuring discovered cases are
re-run on every subsequent test invocation.

### RON schema

```ron
StressScenario(
    description: "Teleport to north pole and idle",
    seed: 42,
    planet: Earth,                  // | SmallPlanet | Custom(PlanetConfig)
    warmup_ticks: 30,
    teleports: [
        Teleport(lat: 89.99, lon: 0.0, altitude_m: 0.0, then_tick: 60),
    ],
    invariants: [Panics, Finite, NoOverflow, ChunkCache],
    chunk_load_rate_min: None,      // Some(N) opts into invariant F
)
```

The schema field names are normative; the implementation plan may add a small
number of additional optional fields if needed (e.g. fixed `tick_dt`, but only
with explicit justification).

## Dependencies

- **proptest** — new dev-dependency (not currently in `Cargo.toml`). Pin to
  current stable. Required for the property generators. Bevy already pulls in a
  large dep tree, so proptest is incremental cost only on dev builds.
- All other deps already present.

## CI integration

- New test target: `cargo test --test stress`. Runs all RON scenarios + all
  proptest generators.
- **Pre-commit fast path:** the existing pre-commit hook adds a small subset
  (`pole_north`, `pole_south`, `antimeridian` only — three RON scenarios, no
  proptest) so commits stay fast.
- **Full path:** `FULL_TESTS=1` and CI run the entire `cargo test --test stress`.
- Approximate runtime budget: full path ≤ 60 s on the dev machine. Calibrated
  during implementation; if exceeded, proptest case counts are reduced.

## File-touch summary

**New files:**
- `src/test_support/mod.rs`
- `src/test_support/stress.rs`
- `tests/stress.rs`
- `tests/cases/stress/*.stress.ron` (10 files)

**Modified files:**
- `Cargo.toml` — add `proptest` to `[dev-dependencies]`; add `test-support`
  feature gate (or use `cfg(test)` only if cross-test reuse is unnecessary —
  decided during implementation).
- `src/lib.rs` — re-export `test_support` under `cfg(any(test, feature = "test-support"))`.
- `scripts/pre-commit` — add the three-RON fast subset to the existing fast path.
- `issues.json` — append a note to CRASH-001 referencing this spec.
- `ai-context.json` — register the new test target under `phases` /
  `tech_debt` as appropriate.

**No expected production-source changes.** If invariants A–D detect real bugs
during implementation, those bugs are filed as new issues and fixed in
follow-up tasks.

## Open questions deferred to the implementation plan

These are intentionally not pinned in the spec; the plan resolves them with
empirical calibration:

1. Concrete value of `chunk_load_rate_min` per burst scenario.
2. Concrete `warmup_ticks` per scenario type.
3. Whether the `test-support` feature is needed, or `cfg(test)` alone suffices.
4. Exact proptest case counts (the numbers above are starting points; reduced
   if 60 s budget is exceeded).
