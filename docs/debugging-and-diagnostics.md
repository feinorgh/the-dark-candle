# Debugging & Diagnostics Guide

How to debug, inspect, and diagnose issues in The Dark Candle — for both human developers and AI agents working via the CLI.

## Quick Reference

| Task | Command / Action | Output |
|---|---|---|
| Run game | `cargo run --features bevy/dynamic_linking` | Game window (800×600) |
| Run all tests | `cargo test` | Unit + integration test results |
| Run simulation tests | `cargo test --test simulations` | Scenario pass/fail |
| Dump ECS state (in-game) | Press **F11** | `diagnostics/<timestamp>.dump.ron` |
| Capture screenshot | Press **F12** | `screenshots/<timestamp>.png` |
| Produce simulation video | Add `emit_video` to `.simulation.ron` | MP4 video or PNG frames |
| Run visual rendering tests | `cargo test --test visual_rendering` | MP4 videos in `test_output/` |
| Run atmosphere visual tests (CPU) | `cargo test --test atmosphere_visual --release -- --ignored` | MP4 videos in `test_output/` |
| Run atmosphere visual tests (GPU) | `cargo test --test atmosphere_visual_gpu` | MP4 videos in `test_output/` |
| Enable verbose logging | `RUST_LOG=info cargo run --features bevy/dynamic_linking` | Bevy + game logs on stderr |
| Enable debug logging | `RUST_LOG=debug,bevy=info cargo run --features bevy/dynamic_linking` | Game debug logs without Bevy noise |
| See test stderr output | `cargo test -- --nocapture` | Prints `eprintln!` messages from tests |
| Lint check | `cargo clippy` | Clippy warnings/errors |
| Format check | `cargo fmt --check` | Formatting diff |
| Cross-compile (Windows) | `cargo build --target x86_64-pc-windows-gnu --release` | `.exe` in `target/` |

---

## 1. Diagnostics Plugin

The `DiagnosticsPlugin` (registered in `main.rs`) provides two hotkeys during gameplay:

### F11 — ECS State Dump

Captures a full snapshot of the live game world and writes it as pretty-printed [RON](https://github.com/ron-rs/ron) to `diagnostics/<unix-timestamp>.dump.ron`.

**What's captured:**

- **Camera state** — position (x, y, z), pitch/yaw in degrees, speed, grounded flag
- **Per-chunk data** — chunk coordinates, dirty flag, solid voxel count, material histogram, temperature/pressure range stats
- **World summary** — global material histogram, temperature/pressure/damage statistics, per-material breakdowns
- **Loaded chunk count** — how many chunks the chunk manager currently has active

**Example output (abbreviated):**

```ron
EcsDump(
    loaded_chunk_count: 45,
    camera: Some(CameraSnapshot(
        position: [12.3, 97.1, -5.8],
        pitch_deg: -12.5,
        yaw_deg: 45.0,
        speed: 5.0,
        grounded: true,
    )),
    chunks: [
        ChunkSnapshot(
            coord: [0, 2, 0],
            dirty: false,
            solid_count: 18432,
            material_histogram: {
                "Air": 14336,
                "Dirt": 8192,
                "Stone": 10240,
            },
            temperature: RangeStats(min: 287.5, max: 289.0, mean: 288.15, count: 32768),
            pressure: RangeStats(min: 101300.0, max: 101350.0, mean: 101325.0, count: 32768),
        ),
        // ... more chunks ...
    ],
    world_summary: StateDump(
        grid_size: 32,
        total_voxels: 1474560,
        summary: GridSummary(
            material_histogram: { "Air": 900000, "Stone": 350000, "Dirt": 224560 },
            temperature: RangeStats(min: 285.0, max: 310.0, mean: 288.2, count: 1474560),
            // ...
        ),
    ),
)
```

### F12 — Screenshot Capture

Captures the current frame from the primary window and saves it as a PNG file to `screenshots/<unix-timestamp>.png`. Uses Bevy's native screenshot observer API.

**Setup:** No setup required — both directories (`diagnostics/`, `screenshots/`) are created automatically on first use.

---

## 2. Simulation State Dumps

The headless simulation framework (see [simulation-test-system.md](simulation-test-system.md)) supports an optional state dump after the simulation completes.

### Enabling a Dump

Add `emit_dump` to any `.simulation.ron` scenario file:

```ron
SimulationScenario(
    name: "Water freezing",
    // ... other fields ...
    emit_dump: Some("diagnostics/water_freezing.ron"),
)
```

Or use `"stdout"` to print to stderr (useful with `--nocapture`):

```ron
    emit_dump: Some("stdout"),
```

### What's Included

The simulation dump contains:

- **Grid summary** — material histogram (how many voxels of each material), temperature/pressure/damage range statistics
- **Per-material breakdown** — count, temperature range, pressure range for each material type
- **Simulation statistics** — total reactions fired, total phase transitions, peak temperature (K), peak pressure (Pa)

**Example:**

```ron
StateDump(
    grid_size: 8,
    total_voxels: 512,
    summary: GridSummary(
        material_histogram: {
            "Air": 386,
            "Ice": 0,
            "Stone": 98,
            "Water": 28,
        },
        temperature: RangeStats(min: 270.0, max: 300.5, mean: 288.1, count: 512),
        pressure: RangeStats(min: 101300.0, max: 101350.0, mean: 101325.0, count: 512),
        damage: RangeStats(min: 0.0, max: 0.0, mean: 0.0, count: 126),
        per_material: {
            "Water": MaterialStats(
                count: 28,
                temperature: RangeStats(min: 272.0, max: 275.0, mean: 273.2, count: 28),
                pressure: RangeStats(min: 101325.0, max: 101325.0, mean: 101325.0, count: 28),
            ),
            // ... more materials ...
        },
    ),
    simulation_stats: Some(SimulationStatsSnapshot(
        total_reactions: 42,
        total_transitions: 7,
        peak_temperature_k: 1500.0,
        peak_pressure_pa: 200000.0,
    )),
)
```

### Programmatic Usage (Rust)

The dump functions are available as a public API for use in custom tests:

```rust
use the_dark_candle::diagnostics::state_dump::{dump_grid_state, dump_to_ron};

let dump = dump_grid_state(&voxels, grid_size, &registry, Some(&stats), false);
let ron_text = dump_to_ron(&dump).expect("serialization failed");
println!("{ron_text}");
```

Set the fifth argument (`include_voxels`) to `true` to include every non-air voxel with full coordinates, material name, temperature, pressure, damage, and latent heat buffer values. This produces much larger output.

---

## 2b. Simulation Video Visualization

Simulations can produce per-tick video output, encoding each tick as a frame. This is useful for visually verifying heat diffusion, phase transitions, and reaction fronts.

### Enabling Video Output

Add `emit_video` to any `.simulation.ron` scenario file:

```ron
SimulationScenario(
    name: "Water freezing",
    // ... other fields ...
    emit_video: Some(VideoConfig(
        path: "output/water_freezing.mp4",
        view: Slice(axis: Y, depth: 4),
        color_mode: Temperature(min_k: 250.0, max_k: 300.0),
        fps: 30,
        scale: 8,
    )),
)
```

### Video Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | String | (required) | Output video path (`.mp4`) |
| `view` | `Slice { axis, depth }` or `TopDown` | `Slice(axis: Y, depth: 0)` | How to project the 3D grid into 2D |
| `color_mode` | `Material`, `Temperature { min_k, max_k }`, or `Pressure { min_pa, max_pa }` | `Material` | What quantity to visualize |
| `fps` | u32 | 30 | Frames per second in output video |
| `scale` | u32 | 4 | Pixel scale per voxel (e.g. 8× for a 32³ grid → 256×256) |

### View Modes

- **`Slice`**: A 2D cross-section through the grid. `axis` is `X`, `Y`, or `Z`; `depth` is the index along that axis.
- **`TopDown`**: Raycasts down the Y axis; the first opaque (non-air) voxel determines pixel color.

### Color Modes

- **`Material`**: Uses the material's base RGB color from `.material.ron` files. Air renders black.
- **`Temperature`**: Blue → cyan → green → yellow → red heatmap across the `[min_k, max_k]` range.
- **`Pressure`**: Same gradient across `[min_pa, max_pa]`.

### How It Works

1. Before the tick loop, a `FrameEncoder` is created
2. If **ffmpeg** is available on the system, raw RGB frames are piped directly to `ffmpeg` via stdin — no intermediate files
3. If ffmpeg is **not found**, individual PNG frames are saved to `<stem>_frames/` and a command to encode them is printed
4. After the simulation, the encoder is finalized (closes the ffmpeg pipe or prints the PNG frame count)

### Running

```bash
# Run a specific scenario that has emit_video configured
cargo test --test simulations -- water_freezing --nocapture

# The video is written to the configured path
# If ffmpeg is missing, check stderr for the PNG fallback directory
```

### Requirements

- **With ffmpeg** (recommended): Install ffmpeg (`apt install ffmpeg`, `brew install ffmpeg`, or `pacman -S ffmpeg`). The encoder uses `libx264` + `yuv420p` for broad compatibility.
- **Without ffmpeg**: PNG frames are saved; encode manually:
  ```bash
  ffmpeg -framerate 30 -i video_frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4
  ```

---

## 3. Logging

The project uses Bevy's built-in logging (backed by `tracing` / `env_logger`). Control verbosity with the `RUST_LOG` environment variable.

### Log Levels

| Level | Use | Example |
|---|---|---|
| `error` | Failures that prevent an operation | "Failed to write save file" |
| `warn` | Recoverable issues | "No save file found" |
| `info` | Noteworthy events | "ECS state dump written to diagnostics/..." |
| `debug` | Detailed internal state | (available for future use) |
| `trace` | Very detailed per-frame data | (available for future use) |

### Useful Configurations

```bash
# All info-level logs (engine + game)
RUST_LOG=info cargo run --features bevy/dynamic_linking

# Game debug logs, Bevy at info only (avoids render spam)
RUST_LOG=debug,bevy=info cargo run --features bevy/dynamic_linking

# Only game logs, nothing from Bevy
RUST_LOG=off,the_dark_candle=info cargo run --features bevy/dynamic_linking

# Everything (very verbose — useful for Bevy internals)
RUST_LOG=trace cargo run --features bevy/dynamic_linking

# Specific module only
RUST_LOG=the_dark_candle::diagnostics=debug cargo run --features bevy/dynamic_linking
```

### Logging in Tests

Test `eprintln!` output (including simulation convergence messages and state dumps) is captured by default. To see it:

```bash
cargo test -- --nocapture
cargo test --test simulations -- --nocapture
```

---

## 4. Headless ECS Testing

For debugging game systems without a window or renderer, use the `MinimalPlugins` pattern established in `tests/gameplay.rs`:

```rust
use std::time::Duration;
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        // Add only the plugins you need:
        .add_plugins(PhysicsPlugin)
        .init_resource::<ChunkMap>()
        // Deterministic time — each app.update() advances by exactly 1/60 s:
        .insert_resource(TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f64(1.0 / 60.0),
        ));
    app
}

#[test]
fn my_physics_test() {
    let mut app = test_app();

    // Spawn an entity with the components you want to test
    let entity = app.world_mut().spawn((
        Transform::from_xyz(0.0, 100.0, 0.0),
        MyComponent::default(),
    )).id();

    // Run N frames
    for _ in 0..60 {
        app.update();
    }

    // Inspect the result
    let transform = app.world().get::<Transform>(entity).unwrap();
    assert!(transform.translation.y < 100.0, "Entity should have fallen");
}
```

**Key points:**
- `MinimalPlugins` — no renderer, no window, no audio. Just the ECS scheduler and time.
- `TimeUpdateStrategy::ManualDuration` — deterministic frame timing (no wall-clock dependency).
- Add only the plugins relevant to your test to keep it fast and isolated.

---

## 5. Simulation Test Debugging

The full simulation test framework is documented in [simulation-test-system.md](simulation-test-system.md). Here are the most common debugging workflows:

### Running a Specific Scenario

Simulation tests auto-discover all `*.simulation.ron` files. To debug a specific one:

```bash
# Run all simulation scenarios (failures show assertion details)
cargo test --test simulations -- --nocapture

# Filter by name (partial match works)
cargo test --test simulations -- water_freezing --nocapture
```

### Adding a State Dump to a Failing Scenario

Edit the `.simulation.ron` file and add:

```ron
    emit_dump: Some("stdout"),
```

Then run with `--nocapture` to see the full grid state on stderr. This shows you exactly what the grid looks like after the simulation — material counts, temperature ranges, reaction counts — which makes it clear why an assertion is failing.

### Interpreting Assertion Failures

Failures look like:

```
Water freezing: 2 assertion(s) failed after 500 ticks:
  MaterialCountGt("Ice"): expected ≥10, got 3
  RegionAvgTempLt: expected avg temp < 273.0 K, got 275.2 K
```

This tells you:
1. Only 3 voxels froze into ice (expected ≥10) — likely insufficient cooling time or boundary conditions.
2. The region is still at 275.2 K (above the 273.15 K freezing point) — the ambient schedule may need a longer ramp or lower target.

Add `emit_dump: Some("stdout")` to see the full material histogram and per-material temperature breakdown, which usually reveals the root cause.

### Common Debugging Patterns

| Symptom | Likely Cause | What to Check |
|---|---|---|
| "expected ≥N reactions, got 0" | Temperature below ignition point | Dump temperature stats, check `min_temperature` on reaction rules |
| "expected avg temp > X, got Y" | Insufficient heat source or too much diffusion | Check `dt` vs grid size (CFL stability), check `boundary_htc` |
| "MaterialAbsent failed" | Phase transition not triggered | Check material's `melting_point`/`boiling_point` vs actual temperatures |
| Test passes locally, fails in CI | Asset path resolution | Ensure `CARGO_MANIFEST_DIR` is set, check `find_assets_dir()` fallback |

---

## 6. AI Agent Debugging Workflow

This section describes how an AI agent (e.g., GitHub Copilot CLI) should approach debugging tasks.

### Step 1: Understand the Problem

Read the error message or user description. If a test is failing, run it:

```bash
cargo test --test simulations -- --nocapture 2>&1
```

### Step 2: Capture State

For **simulation issues**, add `emit_dump: Some("stdout")` to the scenario and re-run. Parse the RON output to understand material distribution, temperatures, and reaction counts.

For **live game issues**, ask the user to press **F11** and share the `.dump.ron` file. This gives you the full world state without needing to run the game yourself.

For **visual/rendering issues**, ask the user to press **F12** and share the `.png` screenshot. Multimodal models can analyze this directly.

### Step 3: Analyze

The RON dump is structured for machine consumption. Key things to look for:

- **`material_histogram`** — Are the right materials present in the right quantities?
- **`temperature.min / .max / .mean`** — Is the thermal state reasonable? (Ambient is 288.15 K / 15 °C)
- **`pressure.mean`** — Should be near 101325 Pa at sea level.
- **`simulation_stats.total_reactions`** — Did chemistry fire at all?
- **`simulation_stats.total_transitions`** — Did any phase changes occur?
- **`per_material` breakdown** — Drill into a specific material's temperature range to diagnose phase transition issues.

### Step 4: Fix and Verify

Make the code or data change, then re-run:

```bash
cargo clippy && cargo test
```

The pre-commit hook enforces `cargo fmt --check`, `cargo clippy`, and `cargo test`, so all three must pass before committing.

---

## 7. Build & Toolchain Notes

### Linker Configuration

The project uses fast linkers configured in `.cargo/config.toml`:

- **Linux:** `clang` with `lld` (via `-fuse-ld=lld`)
- **Windows cross-compile:** `x86_64-w64-mingw32-gcc` (MinGW)

If you see linker errors on a new system, ensure `lld` and `clang` are installed:

```bash
# Gentoo
emerge sys-devel/lld sys-devel/clang

# Ubuntu/Debian
apt install lld clang

# Fedora
dnf install lld clang
```

### Build Profiles

| Profile | Opt Level (deps) | Opt Level (code) | LTO | Notes |
|---|---|---|---|---|
| `dev` | 3 | 1 | off | Fast incremental builds, optimized deps |
| `release` | — | — | thin | Single codegen unit for best optimization |

### Dynamic Linking (Development)

For fast iteration, use Bevy's dynamic linking feature:

```bash
cargo run --features bevy/dynamic_linking
```

This avoids relinking the entire Bevy engine on each code change. Do **not** use this for release builds.

---

## 8. In-Game Controls Reference

These are useful when manually debugging visual or physics issues:

| Input | Action |
|---|---|
| **W / A / S / D** | Move forward / left / back / right |
| **Mouse** | Look around (click to grab cursor, Esc to release) |
| **Space** | Jump (when grounded) |
| **G** | Toggle gravity (fly mode) |
| **F11** | Dump ECS state to `diagnostics/` |
| **F12** | Capture screenshot to `screenshots/` |

**Camera starts at** `(0, 100, 0)` looking toward `(10, 95, 10)`. Gravity pulls it down to the terrain surface.

**Fly mode** (press G) disables gravity and sets movement speed to 20 m/s, useful for inspecting terrain from above or navigating to a specific chunk quickly.

## 9. Visual Rendering Tests

Integration tests in `tests/visual_rendering.rs` produce MP4 videos for visual evaluation of rendering features. These use a headless software raymarcher (not the GPU pipeline) with the shared DDA raycast module.

**Run all visual tests:**
```bash
cargo test --test visual_rendering -- --nocapture
```

**Output directory:** `test_output/` (created automatically)

| Test | Output | What it shows |
|------|--------|---------------|
| `incandescence_video` | `incandescence.mp4` | Thermal glow colors (800 K → 6000 K) |
| `voxel_grid_video` | `voxel_grid.mp4` | Colored material grid with Lambertian shading |
| `time_of_day_video` | `time_of_day.mp4` | Sun cycle with Rayleigh scattering sky |
| `daynight_terrain_video` | `daynight_terrain.mp4` | Terrain with day/night lighting changes |
| `optics_colored_shadows_video` | `optics_colored_shadows.mp4` | Beer-Lambert absorption through water/glass columns |

Each test renders frames via `render_perspective()` from `src/diagnostics/visualization.rs`, then encodes to MP4 via ffmpeg. The videos are meant for human visual inspection — there are no automated pixel-level assertions.

## 10. Atmosphere Visualization Tests

Two test suites render atmosphere features: a CPU version (`tests/atmosphere_visual.rs`) and a GPU-accelerated version (`tests/atmosphere_visual_gpu.rs`). The GPU version is ~1000× faster and runs as part of `cargo test` by default.

### GPU tests (fast — seconds)

```bash
cargo test --test atmosphere_visual_gpu -- --nocapture
```

| Test | Output | What it shows |
|------|--------|---------------|
| `gpu_sky_panorama_video` | `gpu_sky_panorama.mp4` | Full day/night sky cycle with Rayleigh + Mie scattering |
| `gpu_volumetric_clouds_video` | `gpu_volumetric_clouds.mp4` | Volumetric cloud layer with Beer-Lambert + HG phase function |
| `gpu_atmosphere_showcase_video` | `gpu_atmosphere_showcase.mp4` | Integrative: terrain + clouds + fog + shadows + scattering + day/night cycle |

### CPU tests (slow — minutes to hours, `#[ignore]`d)

```bash
cargo test --test atmosphere_visual --release -- --ignored --nocapture
```

| Test | Output | What it shows |
|------|--------|---------------|
| `sky_panorama_video` | `sky_panorama.mp4` | Rayleigh + Mie sky over full day cycle |
| `volumetric_clouds_video` | `volumetric_clouds.mp4` | Cloud ray-march with silver-lining and dark bases |
| `cloud_shadows_video` | `cloud_shadows.mp4` | Cloud shadow map projected onto terrain |
| `valley_fog_video` | `valley_fog.mp4` | Exponential height fog in a terrain valley |
| `atmosphere_showcase_video` | `atmosphere_showcase.mp4` | Everything combined: terrain, clouds, fog, shadows, scattering |

The CPU tests exist as a reference implementation. For routine work, use the GPU tests.

## 11. GPU Compute Renderer

The `src/gpu/` module provides a headless wgpu compute shader renderer for offscreen visualization. It reproduces the CPU software renderer's output using a single uber-compute-shader dispatched per frame.

**Architecture:**
- `GpuContext` — headless wgpu adapter/device/queue initialization
- `GpuRenderer` — buffer upload (voxels, materials, cloud field, humidity/temp), compute dispatch, RGBA readback to `RgbImage`
- `atmosphere_render.wgsl` — WGSL compute shader implementing: sky scattering, volumetric cloud ray-march, DDA terrain raycast, Lambertian shading, shadow rays, cloud shadows, fog, incandescence, star field, Reinhard tonemapping

**Usage in tests:**
```rust
use the_dark_candle::gpu::{GpuContext, GpuRenderer, GpuRenderParams};

let ctx = GpuContext::new().expect("GPU not available");
let mut renderer = GpuRenderer::new(&ctx, width, height);
renderer.upload_voxels(&ctx, &voxels, chunk_size);
renderer.upload_materials(&ctx, &registry, max_material_id);

let image: image::RgbImage = renderer.render_frame(&ctx, &params);
```

**Buffer layout conventions:**
- Voxels packed as `[material_id: u32, temperature_bits: u32]` (8 bytes/voxel, Z-major index)
- GPU uniform structs use `#[repr(C)]` + `bytemuck::Pod` with 16-byte alignment
- Materials as `vec4<f32>` (r, g, b, transparent_flag) per material ID
