# Architecture & Data Flow

## 1. The Entity Component System (ECS)
This game strictly adheres to Bevy's ECS model.
- **Components:** Pure data structs. No logic.
- **Systems:** Pure logic functions. They operate on Components via Queries.
- **Resources:** Global singletons (e.g., Score, Time, Asset handles).

## 2. Entry Points

| Binary | Source | Description |
|--------|--------|-------------|
| Main game | `src/main.rs` | CLI parsing, plugin registration, Bevy app startup |
| World generator | `src/bin/worldgen.rs` | Standalone planetary generation pipeline + globe viewer + map export |

## 3. Module Overview

| Module | Path | Purpose |
|--------|------|---------|
| `world` | `src/world/` | Chunks, voxels, terrain generation (NoiseStack, biome integration, scene presets), geological strata & ores, multi-scale caves, hydraulic erosion, Surface Nets meshing, collision, planetary sampling |
| `physics` | `src/physics/` | Gravity, collision, AMR fluid, LBM gas, FLIP/PIC particles |
| `chemistry` | `src/chemistry/` | Materials, heat transfer, reactions, state transitions |
| `building` | `src/building/` | Freeform structural construction: `PartData`/`RecipeData` RON assets, `Joint` stress model, load-path analysis with progressive collapse, player placement (B key, 1 m grid), demolition, tick-based crafting |
| `biology` | `src/biology/` | Metabolism, health, growth, death, plant systems |
| `behavior` | `src/behavior/` | Needs, utility AI, pathfinding, perception |
| `bodies` | `src/bodies/` | Articulated skeleton FK/IK, tissue compound colliders, FABRIK IK, locomotion gaits, player embodiment, per-region injury |
| `social` | `src/social/` | Relationships, factions, reputation, group behaviors |
| `entities` | `src/entities/` | Creature and item spawning, `Inventory` component (weight/volume-limited item stacks) |
| `procgen` | `src/procgen/` | Procedural tree generation, biome integration |
| `lighting` | `src/lighting/` | Sun cycle, light maps, sky scattering, volumetric effects, atmospheric sky (Bevy Atmosphere component), distance fog with time-of-day sync |
| `weather` | `src/weather/` | Cloud particles, wind upload, snow/rain accumulation |
| `camera` | `src/camera/` | First-person camera, fly mode |
| `data` | `src/data/` | RON asset loading, material/reaction registries |
| `persistence` | `src/persistence/` | Save/load system (SAVE_VERSION=4) |
| `diagnostics` | `src/diagnostics/` | ECS dump, screenshots, video encoding |
| `simulation` | `src/simulation/` | Headless tick loop for testing |
| `gpu` | `src/gpu/` | Headless wgpu compute, particle pipeline |
| `planet` | `src/planet/` | Geodesic grid, tectonics, impacts, celestial, biomes, geology, globe renderer, map projections |
| `map` | `src/map/` | In-game map overlay: local discovery map (biome colors, fog-of-war, zoom), global planet map (equirectangular projection, lat/lon marker), discovery persistence |

## 4. The Data-Driven Pipeline (YAML / RON)
To ensure maximum flexibility, all game data is loaded at runtime via `bevy_common_assets`.

### How to create a new Data Type:
If the user requests a new game entity (e.g., "Add a Goblin enemy"):
1. **Create the Rust Struct:** Create a data container in `src/data/` deriving the necessary traits.
   ```rust
   #[derive(serde::Deserialize, bevy::asset::Asset, bevy::reflect::TypePath)]
   pub struct EnemyData {
       pub name: String,
       pub health: f32,
       pub speed: f32,
   }
