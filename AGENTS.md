# Project: The Dark Candle
- **Engine:** Bevy v0.18 (Rust)
- **OS:** Developed on Gentoo Linux (Wayland), cross-compiled to Windows (`x86_64-pc-windows-gnu`).
- **Goal:** A modern, cross-platform game relying heavily on data-driven design and procedural generation.

## ⚠️ CRITICAL: Bevy 0.18 Strict Rules
AI agents often hallucinate Bevy 0.14 or older code. You MUST adhere to modern Bevy 0.18 API structures:
1. **NO BUNDLES:** Bundles are deprecated. Do NOT use `SpriteBundle`, `Camera2dBundle`, `SpatialBundle`, etc.
2. **REQUIRED COMPONENTS:** Spawn base components directly in a tuple. 
   - *Example:* `commands.spawn((Sprite { color: Color::srgb(1., 0., 0.), ..default() }, Transform::from_xyz(0., 0., 0.)));`
3. **CAMERAS:** Spawn cameras directly using the component: `commands.spawn(Camera2d);`
4. **WINDOWS:** `Window` resolution now strictly takes unsigned integers `(u32, u32)`, NOT floats.
5. **STATES (0.18):** `set()` now always triggers `OnEnter`/`OnExit`. Use `set_if_neq()` if you want the old behavior of skipping transitions when the state hasn't changed.
6. **EVENTS (0.18):** `EntityEvent` is immutable by default. Do not attempt to mutate event data directly without `SetEntityEventTarget`.

## Architecture & Data Flow
This game uses a strict Data-Driven ECS architecture. We use the `bevy_common_assets` crate to load text files into Rust structs.
1. **No Hardcoded Data:** Enemy stats, weapon damage, and level properties must be loaded from `.ron` (Rusty Object Notation) files.
2. **Data Structs:** Data containers must derive `serde::Deserialize`, `bevy::asset::Asset`, and `bevy::reflect::TypePath`.
3. **File Locations:** All `.ron` files go in the `assets/data/` directory.

### Example: How to Add a New Entity Type
If tasked with creating a new entity (like an Item or Enemy):
1. Define the Rust struct in `src/`.
2. Register the loader in `main.rs`: `app.add_plugins(RonAssetPlugin::<YourStruct>::new(&["your_suffix.ron"]))`
3. Generate the actual data file in `assets/data/name.your_suffix.ron`.
4. Write a system that listens for the asset load via `Res<Assets<YourStruct>>` and spawns the entity using Bevy 0.18 component tuples.

## Terminal Commands
When asked to provide build or run commands, use the following:
- **Run (Fast execution):** `cargo run --features bevy/dynamic_linking` (or `cargo run --release` for testing performance).
- **Build for Windows:** `cargo build --target x86_64-pc-windows-gnu --release`
- *Note:* We use the `lld` linker on Gentoo for fast compile times.
