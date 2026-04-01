# Project: The Dark Candle
- **Engine:** Bevy v0.18 (Rust)
- **OS:** Developed on Gentoo Linux (Wayland), cross-compiled to Windows (`x86_64-pc-windows-gnu`).
- **Goal:** A modern, cross-platform game relying heavily on data-driven design and procedural generation.
- **Language:** Rust 1.85 or later (edition 2024)

## ⚠️ CRITICAL: Bevy 0.18 Strict Rules
AI agents often hallucinate Bevy 0.14 or older code. You MUST adhere to modern Bevy 0.18 API structures:
1. **NO BUNDLES:** Bundles are deprecated. Do NOT use `SpriteBundle`, `Camera2dBundle`, `SpatialBundle`, etc.
2. **REQUIRED COMPONENTS:** Spawn base components directly in a tuple. 
   - *Example:* `commands.spawn((Sprite { color: Color::srgb(1., 0., 0.), ..default() }, Transform::from_xyz(0., 0., 0.)));`
3. **CAMERAS:** Spawn cameras directly using the component: `commands.spawn(Camera2d);`
4. **WINDOWS:** `Window` resolution now strictly takes unsigned integers `(u32, u32)`, NOT floats.
5. **STATES (0.18):** `set()` now always triggers `OnEnter`/`OnExit`. Use `set_if_neq()` if you want the old behavior of skipping transitions when the state hasn't changed.
6. **EVENTS (0.18):** `EntityEvent` is immutable by default. Do not attempt to mutate event data directly without `SetEntityEventTarget`.

## Memory Protocol

At the start of every task:
- Read `ai-context.json`
- Check `meta.generated_from_commit` against current HEAD. If they differ by more than 10 commits, flag which sections are likely stale before proceeding.
- Cross-reference the feature list against the files you're about to touch
- If your task overlaps a `complete` feature, comment on the PR explaining why
- **Check `issues.json` for open bugs** — if your task touches affected files, investigate and fix related issues. Update issue `status` and `resolution` as you work.

At the end of every task:
- Update `ai-context.json` with any new features, changed paths, or new env vars
- Include the updated file in your PR
- **If you discover new bugs, add them to `issues.json`** with a unique ID, category, severity, and suspected cause.
- **If you fix a bug, update its status to `resolved`** in `issues.json` with the resolution details and commit SHA.

## Coding Standard
Read `docs/RUST_INSTRUCTIONS.md` and follow the guidelines there.

## Architecture & Data Flow
This game uses a strict Data-Driven ECS architecture. We use the `bevy_common_assets` crate to load text files into Rust structs.
1. **No Hardcoded Data:** Enemy stats, weapon damage, and level properties must be loaded from `.ron` (Rusty Object Notation) files.
2. **Data Structs:** Data containers must derive `serde::Deserialize`, `bevy::asset::Asset`, and `bevy::reflect::TypePath`.
3. **File Locations:** All `.ron` files go in the `assets/data/` directory.
4. **Key modules:** `src/world/` (chunks, meshing, raycast, NoiseStack noise engine, biome integration, scene presets, geological strata, hydraulic erosion, planetary sampling), `src/chemistry/` (heat, reactions, transitions), `src/physics/` (rigid body, gravity, collision, LBM gas dynamics, atmosphere), `src/lighting/` (sun cycle, light maps, sky scattering, volumetric clouds, fog, cloud shadows, GPU atmosphere), `src/map/` (local discovery map, global planet map, fog-of-war, persistence), `src/data/` (material/reaction RON loading), `src/diagnostics/` (ECS dump, screenshots, visualization), `src/gpu/` (headless wgpu compute shader renderer).

### Example: How to Add a New Entity Type
If tasked with creating a new entity (like an Item or Enemy):
1. Define the Rust struct in `src/`.
2. Register the loader in the appropriate plugin (typically `DataPlugin` in `src/data/mod.rs`): `app.add_plugins(RonAssetPlugin::<YourStruct>::new(&["your_suffix.ron"]))`
3. Generate the actual data file in `assets/data/name.your_suffix.ron`.
4. Write a system that listens for the asset load via `Res<Assets<YourStruct>>` and spawns the entity using Bevy 0.18 component tuples.

## ⚠️ CRITICAL: SI Unit System & Real-World Physics

All physical properties in this project use the **International System of Units (SI)** with real-world values. AI agents MUST NOT introduce arbitrary/game-feel constants. Emergent behaviors (terminal velocity, buoyancy effects, drag) arise from the interaction of fundamental forces — they are NOT hardcoded.

### Core Rules

1. **1 voxel = 1 meter.** All spatial units map directly. A 32³ chunk is 32 m × 32 m × 32 m.
2. **No arbitrary physics constants.** Use real values: g = 9.80665 m/s², P₀ = 101325 Pa, etc. All constants live in `src/physics/constants.rs`.
3. **Material properties are real-world SI.** Density in kg/m³, thermal conductivity in W/(m·K), specific heat in J/(kg·K), hardness in Mohs (0–10), viscosity in Pa·s, latent heats in J/kg, heats of combustion in J/kg.
4. **Emergent over hardcoded.** Terminal velocity, buoyancy, drag — these emerge from the force model (gravity + buoyancy + drag + friction). Do NOT add constant caps or magic numbers.
5. **Force model for entities:** Every entity has mass (kg) and a drag profile. Forces are summed per tick: `F_net = F_gravity + F_buoyancy + F_drag + F_friction`. Acceleration = F_net / mass.
6. **Heat transfer uses Fourier's law.** Thermal diffusivity α = k / (ρ × Cₚ). Latent heat must be absorbed before phase transitions complete.
7. **Pressure in Pascals.** Altitude-dependent via the barometric formula. Gas density from ideal gas law.
8. **Reaction energies in J/kg.** Combustion is a sustained process with a burn rate, not an instant conversion.

### SI Unit Reference Table

| Quantity | Unit | Symbol | Example |
|---|---|---|---|
| Length / position | meters | m | 1 voxel = 1 m |
| Mass | kilograms | kg | Iron voxel = 7874 kg |
| Time | seconds | s | Bevy `FixedUpdate` dt |
| Temperature | Kelvin | K | Ambient = 288.15 K |
| Pressure | Pascals | Pa | Sea level = 101325 Pa |
| Force | Newtons | N | Weight = m × g |
| Energy | Joules | J | Latent heat of ice = 334000 J/kg |
| Power | Watts | W | Heat flux = k × A × ΔT / Δx |
| Thermal conductivity | W/(m·K) | k | Iron = 80.2 |
| Specific heat capacity | J/(kg·K) | Cₚ | Water = 4186 |
| Viscosity | Pascal-seconds | Pa·s | Water = 1.0e-3 |
| Density | kg/m³ | ρ | Water = 1000 |
| Refractive index | dimensionless | n | Water = 1.33, glass = 1.52 |
| Absorption coeff. | m⁻¹ | α | Water red channel = 0.45 |
| Speed of light | m/s | c | 299 792 458 |

## Terminal Commands
When asked to provide build or run commands, use the following:
- **Run (Fast execution):** `cargo run --features bevy/dynamic_linking` (or `cargo run --release` for testing performance).
- **Build for Windows:** `cargo build --target x86_64-pc-windows-gnu --release`
- **Test:** `cargo test` (all tests), `cargo test --test simulations` (simulation scenarios only)
- *Note:* We use the `lld` linker on Gentoo for fast compile times.

## Simulation Test Framework
Physics and chemistry are validated by **headless simulation scenarios** defined as `.simulation.ron` files in `tests/cases/simulation/`. These run via `cargo test --test simulations`.

### Key Points for AI Agents
- Scenarios define a voxel grid, material regions, tick parameters, and assertions on physical outcomes.
- The tick loop runs: heat diffusion → chemical reactions → state transitions → pressure → boundary conditions.
- **Do not hardcode physics outcomes.** Assertions verify emergent behavior (e.g. water freezes when cooled, reactions produce expected products).
- Available geometry types: `Fill`, `Shell`, `Layer`, `Sphere`, `Cylinder`, `EveryNth`, `Checkerboard`, `RandomHeightmap`.
- Available assertions: `MaterialCountEq`, `MaterialCountGt`, `MaterialAbsent`, `RegionAvgTempGt`, `RegionAvgTempLt`, `TotalReactionsGt`, `NoReactions`, and more.
- Full documentation: [`docs/simulation-test-system.md`](docs/simulation-test-system.md).
