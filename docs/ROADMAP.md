# The Dark Candle — Roadmap

A 3D first-person procedural simulation game built on Bevy 0.18. The world is a
chunked voxel grid (32³ per chunk) rendered as smooth meshes via Surface Nets.
Every voxel carries material and physical state, enabling deep simulation of physics,
chemistry, biology, and social systems. Gameplay emerges from the interaction of
independent simulation layers — no scripted events, no hardcoded behaviors.

## Architecture

```
Simulation Stack (each layer reads/writes shared ECS components):

  Layer 3 — Social        relationships, factions, reputation, group behaviors
  Layer 2 — Behavior      needs hierarchy, utility AI, pathfinding, perception
  Layer 1 — Biology       metabolism, health, growth, death/decomposition, plants
  Layer 0 — Physics       gravity, collision, fluids (AMR+LBM+FLIP), pressure, integrity
            Chemistry     materials, heat transfer, reactions, state transitions
            World         voxel grid, chunks, terrain generation, meshing
```

Modules: `camera`, `world`, `physics`, `chemistry`, `biology`, `entities`,
`procgen`, `behavior`, `social`, `data`, `persistence`, `planet`, `map`.

All entity/material/reaction data is loaded from `.ron` files under `assets/data/`.

---

## Completed Phases

### Phase 0 — Restructure & 3D Foundation ✅
Migrated from single-file 2D to modular 3D. 10 Bevy plugins, 3D first-person camera
with WASD/mouse look/fly-mode, placeholder 3D scene.

### Phase 1 — Voxel World & Terrain ✅
Chunked voxel world (32³), cylindrical chunk loading, layered Perlin noise terrain
(continental + detail + cave carving), Surface Nets mesh rendering with per-material
vertex colors. SVO octree subdivision for efficient voxel storage.

### Phase 2 — Materials & Chemistry ✅
12 materials from `.material.ron` files with real-world SI properties (density,
thermal conductivity, specific heat, melting/boiling points). Heat diffusion via
Fourier's law. RON-defined chemical reactions. State transitions (melting, boiling,
freezing, condensation). Fire propagation as emergent behavior.

### Phase 3 — Physics ✅
Entity gravity with force-based model (gravity + buoyancy + drag + friction).
AABB voxel collision. Structural integrity (flood-fill connectivity, collapse).
Gas pressure propagation.

**Fluid simulation** — three-model stack replacing the original CA system:
- **AMR Navier-Stokes** (`amr_fluid/`) — incompressible liquid flow with
  adaptive mesh refinement, pressure projection, surface tracking
- **LBM D3Q19** (`lbm_gas/`) — compressible gas dynamics via lattice Boltzmann,
  BGK collision, streaming, macroscopic field recovery
- **FLIP/PIC** (`flip_pic/`) — particle simulation for precipitation, spray,
  erosion. Staggered MAC grid, trilinear P2G/G2P, Jacobi pressure solve,
  SI drag/gravity, deterministic emission, sub-voxel accumulation

### Phase 4 — Entities & Procedural Generation ✅
`CreatureData` + `ItemData` RON structs. Procedural creature generation (stat
variation, color jitter, deterministic seeding). Procedural item generation
(material properties → emergent weight/durability/damage). 4 biomes with spawn
tables, creature spawning from biome data.

### Phase 5 — Biology ✅
Metabolism (energy from food, starvation). Health (damage types, status effects,
healing). Growth/aging (juvenile → adult → elder). Death → decomposition (corpse
voxels → decay → nutrients). Plant growth (grass spreads to dirt with light/water).

### Phase 6 — Behavior & AI ✅
Needs system (hunger, safety, rest, curiosity, social). Utility-based action
evaluation. 3D voxel pathfinding. Basic behaviors (wander, eat, flee, sleep, follow).
Perception system (sight, hearing, smell).

### Phase 7 — Social Systems ✅
Relationship tracking (trust, familiarity, hostility). Faction system with shared
identity and territory. Reputation from observed actions. Group behaviors
(cooperative hunting, territory defense, trade).

---

## Current State

**All 7 original phases are complete.** The codebase has:
- 148+ source files, ~53,000 lines of Rust (edition 2024)
- 1372+ passing tests (lib) + 14 integration + 3 simulation + 9 visual rendering
- Pre-commit hooks: `cargo fmt` → `cargo clippy -D warnings` → `cargo test`
- CI/CD: GitHub Actions (Linux, Windows, macOS)
- Cross-compilation: `x86_64-pc-windows-gnu`
- Standalone worldgen binary: `cargo run --bin worldgen` (planetary generation + globe + projections)

### Recent work (post-Phase 7)

| Commit | Description |
|--------|-------------|
| `b002808` | In-game map system: local discovery + global planet map (M key toggle) |
| `74e5f7e` | Atmospheric sky rendering: Bevy Atmosphere + DistanceFog + ClearColor |
| `3f2b946` | Terrain Detail milestone: NoiseStack, 8 presets, geology, caves, erosion |
| `06ff4b9` | SVO octree voxel subdivision system |
| `d5881e8` | AMR Navier-Stokes fluid simulation |
| `1e75f32` | D3Q19 LBM gas simulation |
| `79904af` | FLIP/PIC particle simulation |
| `e7e180f` | Remove legacy CA fluid simulation |
| `e420fb8` | Simulation test framework + water freezing scenario |
| `06e0ccd` | ECS state dump + screenshot capture diagnostics |
| `40ba85b` | Debugging and diagnostics documentation |
| `a65d972` | Simulation video visualization pipeline (ffmpeg) |
| `609259a` | Performance: reduce allocations in heat/LBM/chemistry hot loops |
| `ea71ff1` | Named MaterialId constants replacing numeric IDs |
| `0e035d5` | Procedural tree generator + biome integration |
| `c178bc3` | dx-aware radiation in `radiate_chunk` (multiresolution physics) |
| `6e5679a` | Forest fire demo: 8-tree ring with convection proxy |
| *wip* | Power-law plate sizes (Pareto α=1.3) + subduction boundary deformation |
| *wip* | Geological time calibration: TectonicMode (Quick/Normal/Extended), SI plate velocities, acceleration model |
| *wip* | Tectonic time-lapse visualization: snapshot capture, playback controls in globe viewer |

### Planetary World Generation Pipeline ✅

Standalone world generation binary (`cargo run --bin worldgen`) implementing the
geodesic terrain design document's generation pipeline (Sections 3.3, 8.1–8.8).
Produces a complete, physically consistent planet from a seed, with interactive
3D visualization and 2D map export.

**Module:** `src/planet/` (~4,700 LOC, 120+ tests)

| Phase | File | Description |
|-------|------|-------------|
| 1 — Geodesic Grid | `grid.rs` | Icosahedral subdivision with configurable level (10×4^n+2 cells) |
| 2 — Tectonics | `tectonics.rs` | Plate simulation: power-law size distribution, physical velocities (2–10 cm/yr SI), geological time calibration (Quick/Normal/Extended modes), subduction deformation, orogenesis, erosion, time-lapse snapshot capture |
| 3 — Impacts | `impacts.rs` | Asteroid/comet impacts: craters, ejecta, crust thinning |
| 4 — Celestial | `celestial.rs` | Star, moons, rings, Keplerian orbits, tidal forces |
| 5 — Biomes & Geology | `biomes.rs`, `geology.rs` | Climate zones, 14 biome types, rock strata, 7 ore types |
| 6 — Globe Renderer | `render.rs` | Bevy 3D globe with orbital camera, 8 colour modes, screenshot, tectonic time-lapse playback (play/pause, 0.25×–32× speed, frame stepping) |
| 7 — Map Projections | `projections.rs` | Equirectangular, Mollweide, orthographic + rotating animation |

**CLI usage:**
```bash
# Generate and print stats
cargo run --bin worldgen -- --seed 42 --level 4 --stats

# Interactive 3D globe
cargo run --bin worldgen -- --seed 42 --level 4 --globe

# High-fidelity extended simulation over 4.5 Gyr
cargo run --bin worldgen -- --seed 42 --level 4 --tectonic-mode extended --tectonic-age 4.5 --globe

# Export map projection
cargo run --bin worldgen -- --seed 42 --level 4 --projection mollweide --colourmode biome --output world.png

# Rotating animation
cargo run --bin worldgen -- --seed 42 --level 4 --animate rotation.mp4 --colourmode elevation

# Interactive globe with tectonic time-lapse playback
cargo run --bin worldgen -- --seed 42 --level 4 --globe --timelapse
```

Full design: **[geodesic-terrain-design.md](geodesic-terrain-design.md)**

### LOD / Octree Realism Overhaul (latest)

Upgraded the LOD, meshing, and refinement pipeline from disconnected placeholders
to an integrated, camera-aware system:

- **LOD hysteresis** — 12% dead-band on `LodConfig` prevents thrashing at
  distance boundaries. Coarser transitions use standard thresholds; finer
  transitions require crossing a tighter band.
- **Screen-space error metric** — `level_for_screen_error()` projects
  geometric error (in pixels) using FOV + screen height. Selects the coarsest
  LOD whose pixel error stays below a configurable threshold (default 2.0 px).
- **MaterialColorMap resource** — registry-backed color map with hardcoded
  fallback. All mesh generation paths now accept `Option<&MaterialColorMap>`,
  replacing duplicate color tables.
- **LOD-aware meshing** — `ChunkLod` component tracks per-chunk LOD level.
  `dispatch_mesh_tasks` queries the camera, computes LOD with hysteresis, and
  spawns async tasks via `AsyncComputeTaskPool` with stride = 2^level.
  `collect_mesh_results` polls completed tasks and inserts Bevy mesh components.
  Remeshes only on LOD change.
- **Refinement wired up** — `build_refined_octree` now uses `RefinementAnalysis`:
  candidate cells (surface crossings, material boundaries, gradients) are pinned
  at leaf resolution via `tree.set()`, preventing collapse of feature-rich regions.
- **LOD transitions** — `LodTransition` component with Hermite smoothstep
  opacity fade (0.4 s). `tick_lod_transitions` system drives alpha blending
  during LOD switches.

### Phase 9b–9d: Chemistry Runtime & Visual Rendering

Wired the simulation tick loop (heat → reactions → transitions → pressure) into
Bevy's `FixedUpdate` schedule, enabling live chemistry in gameplay. Added visual
rendering features for thermal and atmospheric effects.

| Commit | Description |
|--------|-------------|
| `a5f69ee` | Phase 9c: thermal glow with incandescence colors + bloom |
| `942f788` | Phase 9d: time-of-day lighting with sun cycle |
| `2c2809b` | Visual rendering tests (4 MP4 videos via ffmpeg) |
| `feeca84` | DDA perspective raymarcher with Lambertian shading + shadow rays |

### Phase 12 Tier 1: Optics & Light Phenomena (✅ complete)

Physically-based light transport through the voxel world: material optical
properties, atmospheric scattering, per-channel absorption, and per-voxel
sunlight propagation.

| Commit | Description |
|--------|-------------|
| `a43b717` | Material optical properties + speed_of_light + glass.material.ron |
| `559d334` | Rayleigh scattering sky model (wavelength-dependent 1/λ⁴, 10 tests) |
| `db0a3f8` | Shared arbitrary DDA raymarcher + refactor (13 tests) |
| `83f8318` | Beer-Lambert RGB absorption through transparent voxels |
| `f504290` | Per-voxel sunlight propagation — ChunkLightMap (9 tests) |
| `e9f1dbc` | Visual test: colored shadows through water/glass columns |

**New modules:** `src/lighting/light_map.rs`, `src/lighting/sky.rs`
**New material:** `glass.material.ron` (id=12, n=1.52, transparent)
**New fields on `MaterialData`:** `refractive_index`, `reflectivity`, `absorption_rgb`
**New constant:** `SPEED_OF_LIGHT = 299_792_458.0 m/s`

### Phase 10: Entity Bodies & Organic Physics ✅

Physical embodiment for all living entities — articulated skeletons, soft/rigid
tissue physics, procedural locomotion, injury model, and plant body physics.
Depends on Phases 3–6 (physics, entities, biology, behavior).

---

## Phase 8 — Spherical Planetary Terrain ✅

Spherical planet centered at origin with configurable radius. Surface noise
in spherical coordinates, shell-based chunk loading, radial apparent gravity
(gravitational + centrifugal). Default 32 km radius. Cartesian chunk/LOD
pipeline preserved.

**Planetary terrain connection (completed):** The `--planet` CLI flag runs the
full geodesic pipeline (tectonics → biomes → geology) before the game starts,
then drives per-voxel terrain via `PlanetaryTerrainSampler` — IDW interpolation
from icosahedral cells combined with fractal noise from `TerrainNoise`. Each
chunk entity receives a `ChunkBiomeData` component (planet biome type,
temperature K, precipitation mm, surface rock, ore bitmask), which procgen
systems (`spawn_creatures`, `spawn_items`, `plant_trees`) use for biome
selection in place of the height heuristic.

New in this connection: `CellIndex` (adaptive spatial index in `grid.rs`),
`PlanetaryTerrainSampler`, `ChunkBiomeData`, `PlanetaryData` resource,
`UnifiedTerrainGenerator::Planetary` variant, `--planet-level` and
`--planet-seed` CLI flags, `SphericalPlanet` scene preset.

Full design: **[spherical-terrain.md](spherical-terrain.md)**

---

## Phase 9 — Atmosphere Simulation ✅

Physics-driven weather from LBM gas + FLIP/PIC particles. Humidity transport,
Clausius-Clapeyron cloud formation, Coriolis forcing, solar heating,
precipitation pipeline, volumetric clouds, Rayleigh/Mie scattering, cloud
shadows, exponential fog, GPU compute renderer (1000× speedup).

Sub-phases 9a–9d: radiative heat transfer ✅, chemistry runtime activation,
thermal glow rendering, time-of-day dynamic lighting.

Full design: **[atmosphere-simulation.md](atmosphere-simulation.md)**

---

## Phase 10 — Entity Bodies & Organic Physics ✅

Physical embodiment for all living entities. Data-driven skeletons
(`.skeleton.ron`), soft/rigid tissue layers, procedural IK locomotion (walk,
fly, swim, climb), physical perception (eye FOV cones, ear directionality),
tiered injury model (bruise → fracture → sever), plant body physics with wind
response and felling. 10 implementation steps.

New module: `src/bodies/` (skeleton, tissue, ik, locomotion, player, perception,
injury, plant). New assets: `assets/data/skeletons/`, `assets/data/bodies/`,
`assets/data/gaits/`.

Full design: **[entity-bodies.md](entity-bodies.md)**

---

## Phase 11 — Buildings & Structural Construction ✅

Freeform building from physical materials with data-driven parts (`.part.ron`).
Structural strength properties (tensile, compressive, shear, flexural in Pa),
automatic joint creation, load-path stress analysis with progressive collapse,
12 new construction materials, crafting recipes, player placement/demolition.
Structures interact with full physics stack (gravity, fire, explosions, fluid).

Full design: **[structural-construction.md](structural-construction.md)**

---

## What's Next

With the core simulation stack complete, spherical terrain done (Phase 8,
including planetary terrain connection), atmosphere simulation fully implemented
(Phase 9, all sub-phases 9a–9d), atmospheric sky rendering integrated into
gameplay, in-game map system operational, and structural construction designed
(Phase 11), the project needs integration, polish, and gameplay. These are not
yet planned in detail — each will get a session plan when started.

### Terrain Detail & World Generation Options ✅

Implemented: composable `NoiseStack` noise engine (FBM, ridged fractal, domain
warping, terrain-type selector, micro-detail), 8 scene presets (alpine,
archipelago, desert canyon, rolling plains, volcanic, tundra fjords, plus
valley river and spherical planet), geological depth strata with ore veins,
multi-scale cave systems, continent/ocean masks, biome-terrain integration
(slope/altitude surface materials), and hydraulic erosion (droplet, grid,
combined modes). 8 new geological materials (sandstone, limestone, granite,
basalt, coal, copper ore, gold ore, quartz crystal). World creation UI screen
with preset selector. Extended CLI flags: `--seed`, `--terrain-detail`,
`--height-scale`, `--caves`, `--erosion`, `--hydraulic-erosion`.

Full design: **[terrain-generation.md](terrain-generation.md)**

### Atmospheric Sky Integration ✅

Connected the existing Rayleigh/Mie scattering code (Phase 12 Tier 1, `sky.rs`
+ `scattering.rs`) to the in-game camera via Bevy's built-in `Atmosphere`
component. Previously the sky model was only used in headless visualization
exports — the in-game world appeared flat gray.

- **Atmosphere component** on camera: `Atmosphere::earthlike(medium)` +
  `ScatteringMedium::default()` — drives GPU atmospheric sky rendering
- **ClearColor(Color::BLACK)** — provides background for atmosphere shader
- **DistanceFog** — 500 m visibility, exponential squared falloff
- **update_fog() system** — syncs fog color to sun elevation (warm dawn →
  cool noon → dark night), ambient light reduced (noon 200→80, night 10→5)

### In-Game Map System ✅

Dual-mode map accessible via M key: local discovery map + global planet map.

- **Local discovery map** — biome-colored pixel per discovered chunk column,
  fog-of-war for unexplored, centered on player, 4 zoom levels (1/2/4/8 px/chunk)
- **Global planet map** — equirectangular projection via `render_projection()`,
  player lat/lon marker, zoom/pan (only when `PlanetaryData` available)
- **Tab key** switches local ↔ global; ESC closes
- **Discovery persistence** — `DiscoveredColumns` saved/loaded with game,
  SAVE_VERSION bumped 3→4

**Module:** `src/map/` (discovery.rs, local_map.rs, global_map.rs, ui.rs)

### Coupling & Integration ✅

Cross-model fluid coupling between the three physics solvers:

- **Plugin activation** ✅ — AmrFluidPlugin, LbmGasPlugin, FlipPicPlugin wired
  into PhysicsPlugin::build() for runtime use. Physics validated by the
  [simulation test framework](simulation-test-system.md).
- **Virga moisture return** ✅ — `apply_virga()` in `precipitation.rs` now
  takes `&mut LbmGrid`. Evaporated particle mass is converted from kg to
  specific humidity (kg/kg) using the ideal gas law and injected back into the
  LBM moisture field, conserving total atmospheric water.
- **AMR sync-from-chunk per step** ✅ — `amr_fluid_step()` calls
  `sync::sync_from_chunk()` before each step, picking up FLIP-deposited voxels
  and chemistry state transitions (ice melt, water boiling) within the same
  FixedUpdate tick instead of waiting for the next chunk load.
- **Lazy FluidGrid initialisation** ✅ — `CouplingPlugin` adds a
  `lazy_init_fluid_grids` system (`src/physics/coupling.rs`) that scans loaded
  chunks every 1.5 s and creates FluidGrids for chunks that gained liquid from
  FLIP deposition after initial load.

**New module:** `src/physics/coupling.rs`

### Advanced Physics Systems (planned)

Physics coupling layer (entity ↔ world), rigid body extensions (entity-entity
collision, sleep system), soft body physics, constraints & joints, explosion
mechanics, projectile ballistics, fluid-terrain interaction, vehicle physics,
and acoustics.

Full design: **[advanced-physics.md](advanced-physics.md)**

### Phase 9a–9d: Radiative Heat & Visual Rendering

Sub-phases bridging simulation to visual gameplay: radiative heat transfer
(Stefan-Boltzmann ✅, view factor ✅), chemistry runtime activation, thermal
glow rendering (incandescence + bloom), time-of-day dynamic lighting.

Full design: **[atmosphere-simulation.md](atmosphere-simulation.md)** (sub-phases section)

### Phase 12: Optics & Light Phenomena (Tier 1 ✅)

Physically-based light transport: material optical properties, Rayleigh
scattering sky, DDA raymarcher, Beer-Lambert RGB absorption, per-voxel sunlight.
Tier 2 (refraction/reflection) and Tier 3 (Mie, caustics, dispersion) planned.

Full design: **[optics-light.md](optics-light.md)**

### Multiresolution Fire Propagation

Realistic fire spread through procedurally generated trees, enabled by
multiresolution simulation at sub-meter voxel scales.

**New materials** — 4 kindling-class combustibles with thin-fuel properties:
- Twig (ρ=100 kg/m³, ignition 473 K), DryLeaves (ρ=30, ignition 453 K),
  Bark (ρ=350, ignition 553 K), Charcoal (ρ=250, ignition 623 K)

**New reactions** — 5 combustion/pyrolysis rules:
- Twig→Ash, DryLeaves→Air, Bark→Charcoal, Charcoal→Ash, Wood→Charcoal (pyrolysis)

**dx-aware simulation** — `diffuse_chunk` and `radiate_chunk` accept a `dx`
parameter (voxel edge length in meters). Heat transfer scales correctly:
conduction CFL as dx², radiation ΔT as 1/dx. `simulate_tick_dx()` wraps
the full tick loop with explicit voxel scale.

**Procedural tree generator** — L-system tree builder in `src/procgen/tree.rs`:
- `TreeConfig` RON defines species params (height, radius, branching)
- Trunk (Bark shell + Wood core) → branches → twigs → leaf clusters
- `TreeRegistry` resource, `stamp_tree_into_chunk()`, `plant_trees` ECS system
- Wired into biome system for automatic terrain placement

**Forest fire demo** — 8 trees around a central bonfire (64³ grid, dx=0.5 m).
Convection proxy models buoyant plumes + firebrand transport (not yet in
engine; test-only helper). Result: peak 4044 burning voxels, 26.7% consumed.

**Known limitations:**
- Pure radiation too weak at dx≥0.5 m between distant objects (view factor
  ~0.006 at 3.5 m). Convective heat transport is the dominant real-world
  fire-spread mechanism but is not yet implemented in the engine.
- Interior wood cores pyrolyze to charcoal but can't combust without air
  access, causing permanent stall. Rate-based detection handles this.
- `diffuse_chunk` optical depth for absorption coefficient not yet dx-scaled
  (only matters for radiation through steam/water at non-1m scales).

**Future:** Engine-level convective heat transport (hot gas plumes via LBM
coupling), GPU-accelerated fire at finer resolution, improved tree species.

### Phase 13: Electricity & Magnetism (planned)

Electrical conductivity, resistance networks, Kirchhoff's laws, resistive
heating, magnetic fields, Lorentz force, electromagnetic induction, simplified
wave propagation (FDTD), lightning.

Full design: **[electromagnetism.md](electromagnetism.md)**

### Phase 14: Nuclear Physics & Radiation (planned)

Radioactive decay chains, radiation transport (alpha/beta/gamma/neutron),
ionizing dose model (Gray/Sieverts), biological damage, material activation,
shielding, fission/fusion reactions, criticality.

Full design: **[nuclear-physics.md](nuclear-physics.md)**

### EM, Radiation & Optics — Phasing & Dependencies

These systems build on each other in a natural progression. The recommended
integration order follows the dependency chain:

```
Phase 2 (Materials ✅) ──→ Phase 9a: Radiative Heat Transfer ✅
Phase 3 (Temperature ✅)─┘      │
                                ├──→ Phase 9b–9d: Chemistry Runtime + Visual ✅
Phase 8 (Spherical Planet) ─────┘
Phase 9 (Atmosphere) ───────────┐
                                ├──→ Phase 12: Optics (Tier 1 ✅, Tier 2–3 planned)
Phase 11 (Buildings) ───────────┘
                                ├──→ Phase 13: Electricity & Magnetism
                                └──→ Phase 14: Nuclear Physics & Radiation

Phase 3 (Physics ✅) ───────┐
Phase 4 (Entities ✅) ──────┤
Phase 5 (Biology ✅) ───────┼──→ Phase 10: Entity Bodies & Organic Physics
Phase 6 (Behavior ✅) ──────┘
```

Material property extensions accumulate across phases:
- Phase 9a adds: `absorption_coefficient`, `albedo`
- Phase 12 adds: `refractive_index`, `reflectivity`, `absorption_rgb` (**Tier 1 ✅**)
- Phase 13 adds: `electrical_conductivity`, `magnetic_permeability`
- Phase 14 adds: `atomic_number`, `mass_attenuation_coeff`, `radioactive`

Universal constants can be added to `universal_constants.ron` proactively:
`speed_of_light` (**✅**), `planck_constant`, `boltzmann_constant`, `elementary_charge`,
`vacuum_permittivity`, `vacuum_permeability`, `avogadro`.

### Simulation Video Demos
The video visualization pipeline (`src/diagnostics/video.rs` +
`visualization.rs`) can render per-tick frames from headless simulations and
encode via ffmpeg.

**Implemented** (`tests/physics_visual.rs`):
- **Fire propagation** — wooden cabin ignited at one corner, flames spread
  through walls and beams. Incandescence color mode shows thermal glow
- **Lava phase transition** — stone mountain with internal magma chamber, heat
  diffuses upward melting stone into glowing lava
- **Water boiling** — stone cauldron filled with water, heated from below until
  it transitions to steam. Temperature heatmap color mode
- **Oxyhydrogen detonation** — H₂/O₂ checkerboard in a stone chamber with
  central ignition, chain reaction produces ~3073K white flame
- **Forest fire** — 8 procedural trees in a ring around a central bonfire.
  Multiresolution simulation (dx=0.5 m, 64³ grid). Fire spreads via
  convection proxy modeling buoyant plumes and firebrand transport. Peak
  4044 simultaneous burning voxels, 26.7% fuel consumed. Rate-based stall
  detection stops when interior wood cores become oxygen-starved

**Planned:**
- **Bouncing balls** — 3 rubber spheres in an enclosed cube. Requires entity-vs-
  entity collision (rigid body physics above). Validates restitution, gravity,
  drag, energy conservation
- **Fluid flow** — water filling a basin, visualized with temperature/pressure
  color modes. Validates AMR fluid + video pipeline integration
- **Thermal conduction** — iron bar heated at one end, temperature gradient
  spreading over time. Validates heat diffusion + temperature color mode
- **Ice melting** — ice block on warm stone, melts into water pool. Validates
  latent heat absorption and solid→liquid state transitions
- **Erosion timelapse** — terrain with rain, showing erosion carving channels
  over accelerated geological time
- **Snow accumulation** — terrain during snowfall, particle deposits building
  up on surfaces. Validates weather + accumulation systems
- **Glass optics** — light passing through colored glass prisms, showing
  wavelength-dependent absorption and transparency

### Performance & Scaling

**Completed optimizations:**
- **Heat diffusion** — double-buffer swap (eliminates ~77 MB clone per active
  chunk per tick), stack-allocated neighbor array (eliminates 19.2 M heap
  allocations per chunk), pre-computed thermal property cache
- **Radiation transfer** — half-direction trick (13 symmetric pairs instead of
  26 ray directions), removed per-call HashSet allocation
- **Reaction/pressure snapshots** — lightweight snapshots of only the fields
  read (MaterialId + temperature for reactions, pressure + permeability for
  pressure diffusion) instead of full `Vec<Voxel>` copies
- **LBM streaming** — `stream_into()` double-buffer pattern reuses a
  pre-allocated scratch grid (~2.8 MB saved per step per chunk)
- **LBM collision** — compute equilibrium once in `collide_smagorinsky()`
  instead of twice (strain_rate + collide_bgk were redundant)
- **Solar heating LOD** — physics LOD distance check skips distant chunks
- **Existing:** physics LOD (fluid sims within 3-chunk radius), chemistry
  activity gating (only hot chunks tick), adaptive view distance, mesh
  throttling (8 tasks/frame), shadow throttling, async meshing/terrain

**Remaining opportunities:**
- **GPU compute shaders** — offload P2G/G2P, LBM streaming, heat diffusion
- **Profiling & budgeting** — establish per-frame time budgets for each system
  (F3 overlay shows frame budget, system timings with EMA smoothing)
- **Populate MaterialColorMap from MaterialRegistry** — wire the asset-loaded
  material registry into the color map at startup instead of relying on fallbacks
- **Screen-space error as default LOD strategy** — integrate with camera FOV
  query; currently the distance+hysteresis path is used at runtime
- **Sub-voxel refinement** — enable `SubdivisionConfig.max_depth > 0` to
  interpolate below the 1 m voxel grid at surface crossings and material
  boundaries, using the existing `upsample_voxels` / trilinear interpolation
  pipeline

### Rendering Architecture Overhaul (planned)

The current renderer uses Bevy's default PBR pipeline with one `Mesh3d` +
`StandardMaterial` entity per chunk (~240 draw calls for a typical scene).
Two planned tiers address this:

**Tier A — Decouple Rendering from Simulation (near-term)**

Voxel data remains the simulation substrate (heat, chemistry, LBM gas,
collisions) but the visual representation diverges at distance:

1. **Chunk mesh merging** — combine adjacent chunks at the same LOD level into
   region mega-meshes.  Reduces draw calls from ~240 to ~15–20 while keeping
   the existing Surface Nets algorithm for near-field geometry.
2. **Shared material** — all chunks share a single `StandardMaterial` handle
   (vertex-colored), enabling Bevy's automatic draw-call batching.
3. **Heightmap rendering for distant terrain** — beyond LOD 2 (~256 m), replace
   voxel meshes with a single heightmap mesh per LOD ring (one vertex per
   column, no 3D Surface Nets).  Faster to generate, fewer vertices, one draw
   call per ring.
4. **Near-field voxel meshes preserved** — within LOD 0–1 (~128 m), keep
   Surface Nets so the player sees caves, overhangs, and deformable terrain.

Expected impact: ~12× draw-call reduction, ~40–60 % CPU frame-time reduction.

**Tier B — Custom Voxel Render Phase (long-term)**

Replace Bevy's generic `Mesh3d`/`StandardMaterial` path for terrain with a
purpose-built draw phase:

- Custom `RenderCommand` with instanced indirect draws, bypassing Bevy's
  extract → prepare → sort overhead for chunk geometry.
- Keep Bevy's UI, post-processing (bloom), input, and asset pipeline intact.
- Opens the door to GPU-side frustum/occlusion culling and voxel-specific
  optimisations (greedy face merging, palette texturing).
- Falls under Bevy's "Tier 2" integration model: replace mesh rendering while
  retaining the rest of the engine.

Tier B depends on Tier A being stable.  Both tiers leave the simulation layer
(`src/chemistry/`, `src/physics/`, `src/world/voxel.rs`) completely untouched.

### Gameplay & Content
- **Player interaction** — mining, tool use, UI/HUD
- **World persistence** — save/load chunks to disk
- **Sound** — ambient, material interactions, creature vocalizations
- **Rendering polish** — particle effects, water shading
- **Narrative systems** — quests, lore, emergent storytelling from social layer

---

## Milestone: Valley River Scene (Gaps 1–5 ✅)

Target demonstration scene: procedurally generated valley with flowing river,
scattered props (boulders, rocks, pebbles), and sun-cycle shadow casting. All
5 gaps completed — D8 flow accumulation + valley carving, AMR fluid activation
+ river seeding, prop scattering system, terrain shadow casting. Final scene
integration remaining.

Full design: **[valley-river-milestone.md](valley-river-milestone.md)**

---

## Key Design Decisions

1. **SI units throughout** — all physics uses real-world values (kg, m, s, K, Pa).
   Constants in `src/physics/constants.rs` and `assets/data/`. No magic numbers.
2. **1 voxel = 1 meter** — all spatial units map directly.
3. **Emergent over hardcoded** — terminal velocity, buoyancy, fire spread arise
   from fundamental forces, not caps or special cases.
4. **Data-driven** — RON files for materials, reactions, biomes, creatures, items,
   factions. Code reads data; data drives behavior.
5. **No ECS bundles** — Bevy 0.18 deprecated them. Spawn component tuples.
6. **Deterministic simulation** — FNV-1a hash for jitter, no `rand` crate.
   FixedUpdate for physics. Reproducible for debugging.
7. **Surface Nets** — fewer artifacts than Marching Cubes, good for organic terrain.
8. **Utility AI** — more emergent than behavior trees, scales with complex needs.
9. **Self-contained physics models** — AMR, LBM, FLIP each have their own
   types/step/plugin/octree_bridge. Couple through voxel data, not shared state.
10. **Rust edition 2024** — enables `let` chains, `gen` keyword reservation,
    and other modern Rust features. Minimum Rust version ≥ 1.85.
11. **Spherical planet, Cartesian chunks** — the world is a planet-sized sphere,
    but the chunk grid remains Cartesian. Only the terrain function, chunk loader,
    and gravity know about the sphere. Meshing, octree, LOD, and simulation
    systems are coordinate-agnostic by design.
12. **Real apparent gravity** — local “down” is the resultant of gravitational
    attraction toward the planet center and centrifugal force from rotation. This
    vector defines ground detection, slope physics, buoyancy direction, and fluid
    settling. The geoid (sea level surface) is the equipotential surface
    perpendicular to apparent gravity — an oblate spheroid on a rotating planet.
13. **Structures are voxels with joints** — buildings are not separate entity
    graphs. They are voxels with materials and attachment joints, subject to the
    same physics as terrain. Structural failure emerges from stress exceeding
    material strength — no scripted destruction. Material properties (tensile,
    compressive, shear strength in Pa) are data-driven via RON files.

---

## Open Questions

- **Multiplayer?** Not in scope, but chunk-based architecture doesn’t preclude it.
- **Rendering style?** Low-poly / stylized is easier to ship. Custom voxel render
  phase (Tier B) would decouple this decision from Bevy's PBR defaults.
- **Fluid coupling strategy?** Interface cells? Overlapping domains? TBD.
- **Sound design?** Bevy has built-in audio. Not prioritized yet.
- **Planet scale?** Default is 32 km radius. Earth-scale (6,371 km) requires
  multi-resolution octree leaves for the deep interior — deferred. The standalone
  worldgen pipeline supports arbitrary radius via `--radius-km`.
- **Tectonic simulation fidelity?** Implemented as boundary-only stress model
  with configurable step count (default 100). Full mantle convection deferred.
- **Geodesic grid integration?** The standalone pipeline (`src/planet/`) now drives
  the voxel game directly. The `--planet` flag runs the full tectonic→biome→geology
  pipeline and connects `PlanetData` to chunk generation via `PlanetaryTerrainSampler`
  and `ChunkBiomeData`. Hex chunk layouts and the full geodesic design document
  (Phases 0–5) remain future work for very-large-scale worlds.
- **Geoid precision?** Exact equipotential surface vs. oblate spheroid approximation
  for sea level. Spherical harmonics are overkill for a game — ellipsoid is likely
  sufficient.
- **Structural analysis frequency?** Full stress analysis every tick is expensive.
  Budget-limited (every N ticks, or event-driven when loads change). Acceptable
  latency TBD with profiling.
- **WGSL vs custom shaders?** Bevy's internal renderer is locked to WGSL
  (cross-compiled to SPIR-V via naga at runtime). Custom compute shaders in
  `src/gpu/` could use GLSL or precompiled SPIR-V via wgpu's `ShaderSource`
  enum, but this wouldn't eliminate the Vulkan validation warnings (those
  originate from Bevy's built-in shaders, not ours).
