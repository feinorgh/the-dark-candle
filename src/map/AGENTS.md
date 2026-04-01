# Map Module

In-game map overlay: M key toggles between `GameState::Playing` and `GameState::Map`.
Two views: local discovery map (visited chunk columns) and global planet map (equirectangular projection).

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | `MapPlugin`: state toggle (M/ESC), system registration |
| `discovery.rs` | `DiscoveredColumns` resource, tracks visited XZ chunk columns with biome data |
| `local_map.rs` | Local discovery map: biome-colored pixels, fog-of-war, 4 zoom levels (1/2/4/8 px/chunk) |
| `global_map.rs` | Global planet map: equirectangular projection via `render_projection()`, player lat/lon marker |
| `ui.rs` | UI layout (overlay, tab bar, coordinate text), Tab switching, scroll zoom, mouse drag pan |

## Key Resources

| Resource | Type | Purpose |
|----------|------|---------|
| `DiscoveredColumns` | `HashMap<[i32; 2], DiscoveredColumn>` | Tracks explored XZ chunk columns with biome + surface_y |
| `MapViewState` | `Resource` | Current view (Local/Global), zoom levels, global pan offset, drag state |

## Controls

| Key | Action |
|-----|--------|
| M | Toggle map overlay (Playing ↔ Map) |
| Tab | Switch Local ↔ Global view |
| ESC | Close map (return to Playing) |
| Scroll | Zoom in/out |
| Mouse drag | Pan (global map when zoomed in) |

## GameState Integration

- `GameState::Map` variant pauses `Time<Virtual>` and unlocks cursor
- Camera module registers `OnEnter(GameState::Map)` → `release_cursor`
- On exit → `GameState::Playing` re-grabs cursor and unpauses time

## Discovery Tracking

- `track_discoveries` system runs during `GameState::Playing`
- Watches `ChunkMap` for changes, queries all `ChunkCoord` + `ChunkBiomeData` entities
- Records highest-Y chunk per XZ column with biome type
- Falls back to `BiomeType::TemperateForest` when no `ChunkBiomeData` present (flat worlds)

## Global Map

- Only available when `PlanetaryData` resource exists (planetary worlds)
- Renders equirectangular projection via `render_projection()` on first open (CPU path, ~100ms for 1024px)
- Result cached as `Handle<Image>` in `Local<GlobalMapCache>`
- Player position converted to lat/lon via `PlanetConfig::lat_lon()`

## Persistence

- `DiscoveredColumns` serialized in `SaveGame.discovered_columns` (SAVE_VERSION=4)
- v3→v4 migration: field defaults to `None` via `#[serde(default)]`
- On load: restores `DiscoveredColumns` resource if present in save data

## Bevy 0.18 Specifics

- Uses `ChildSpawnerCommands` (not `ChildSpawner`) for `with_children` closures
- Mouse input: `Res<AccumulatedMouseMotion>` for pan, `Res<AccumulatedMouseScroll>` for zoom
- `ImageNode::default()` for the map image display node
- `Image::new()` with `TextureFormat::Rgba8UnormSrgb` for dynamic map textures
