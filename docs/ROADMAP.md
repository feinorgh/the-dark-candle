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
`procgen`, `behavior`, `social`, `data`, `persistence`.

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
- 100+ source files, ~34,000 lines of Rust (edition 2024)
- 911+ passing tests (lib) + 14 integration + 3 simulation + 5 visual rendering
- Pre-commit hooks: `cargo fmt` → `cargo clippy -D warnings` → `cargo test`
- CI/CD: GitHub Actions (Linux, Windows, macOS)
- Cross-compilation: `x86_64-pc-windows-gnu`

### Recent work (post-Phase 7)

| Commit | Description |
|--------|-------------|
| `06ff4b9` | SVO octree voxel subdivision system |
| `d5881e8` | AMR Navier-Stokes fluid simulation |
| `1e75f32` | D3Q19 LBM gas simulation |
| `79904af` | FLIP/PIC particle simulation |
| `e7e180f` | Remove legacy CA fluid simulation |
| `e420fb8` | Simulation test framework + water freezing scenario |
| `06e0ccd` | ECS state dump + screenshot capture diagnostics |
| `40ba85b` | Debugging and diagnostics documentation |
| `a65d972` | Simulation video visualization pipeline (ffmpeg) |

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

### Phase 10: Entity Bodies & Organic Physics (planned)

Physical embodiment for all living entities — articulated skeletons, soft/rigid
tissue physics, procedural locomotion, injury model, and plant body physics.
Depends on Phases 3–6 (physics, entities, biology, behavior).

---

## Phase 8 — Spherical Planetary Terrain ✅

Spherical planet centered at origin with configurable radius. Surface noise
in spherical coordinates, shell-based chunk loading, radial apparent gravity
(gravitational + centrifugal). Default 32 km radius. Cartesian chunk/LOD
pipeline preserved.

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

## Phase 10 — Entity Bodies & Organic Physics (planned)

Physical embodiment for all living entities. Data-driven skeletons
(`.skeleton.ron`), soft/rigid tissue layers, procedural IK locomotion (walk,
fly, swim, climb), physical perception (eye FOV cones, ear directionality),
tiered injury model (bruise → fracture → sever), plant body physics with wind
response and felling. 10 implementation steps.

Full design: **[entity-bodies.md](entity-bodies.md)**

---

## Phase 11 — Buildings & Structural Construction (planned)

Freeform building from physical materials with data-driven parts (`.part.ron`).
Structural strength properties (tensile, compressive, shear, flexural in Pa),
automatic joint creation, load-path stress analysis with progressive collapse,
12 new construction materials, crafting recipes, player placement/demolition.
Structures interact with full physics stack (gravity, fire, explosions, fluid).

Full design: **[structural-construction.md](structural-construction.md)**

---

## What's Next

With the core simulation stack complete, spherical terrain planned (Phase 8),
atmosphere simulation designed (Phase 9), and structural construction designed
(Phase 11), the project also needs integration, polish, and gameplay. These are
not yet planned in detail — each will get a session plan when started.

**Near-term visual integration (Phase 9b–9d):** The chemistry/heat physics are
fully implemented and tested but not yet running in-game or visible to the
player. Phases 9b (chemistry runtime), 9c (thermal glow), and 9d (time-of-day)
will bridge this gap — see their detailed sections below Phase 9a.

### Terrain Detail & World Generation Options

The terrain generator currently uses only two Perlin noise layers blended at a
fixed 70/30 ratio, producing smooth, repetitive landscapes. This milestone
upgrades to multi-octave FBM, ridged multi-fractal, domain warping, geological
strata, multi-scale caves, and biome-terrain integration. All detail is computed
once during async chunk generation — zero runtime FPS impact.

Nine tasks across four tracks:

1. **Noise engine** — `NoiseStack` with composable FBM, ridged fractal, domain
   warping, terrain-type selector, and micro-detail (T1, T2)
2. **World presets & CLI** — 6+ scene presets (alpine, archipelago, desert,
   plains, volcanic, tundra), extended CLI flags, and a world creation UI
   screen (T3, T4, T5)
3. **Geological depth** — Rock strata by depth (sedimentary/metamorphic/igneous),
   ore veins (coal, copper, iron, gold, crystal), enhanced multi-scale cave
   system with caverns, tunnels, and tube networks (T6, T7)
4. **Biome-terrain integration** — Biome map generation, per-biome terrain
   modifiers (height bias, roughness, erosion rate), slope/altitude surface
   materials (T8, T9)

Full design: **[terrain-generation.md](terrain-generation.md)**

### Coupling & Integration
- **Cross-model fluid coupling** — AMR ↔ LBM mass/heat exchange at liquid-gas
  interfaces; FLIP particles entering/leaving LBM gas fields
- **Plugin activation** — wire AmrFluidPlugin, LbmGasPlugin, FlipPicPlugin into
  PhysicsPlugin::build() for runtime use (currently test-only). Physics are also
  validated by the [simulation test framework](simulation-test-system.md).

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
encode via ffmpeg. Planned demos to exercise and validate both physics and
visualization:

- **Bouncing balls** — 3 rubber spheres in an enclosed cube. Requires entity-vs-
  entity collision (rigid body physics above). Validates restitution, gravity,
  drag, energy conservation
- **Fluid flow** — water filling a basin, visualized with temperature/pressure
  color modes. Validates AMR fluid + video pipeline integration
- **Thermal conduction** — iron bar heated at one end, temperature gradient
  spreading over time. Validates heat diffusion + temperature color mode

### Performance & Scaling
- **Chunk-parallel simulation** — run physics per-chunk on thread pool
- **GPU compute shaders** — offload P2G/G2P, LBM streaming, heat diffusion
- **Profiling & budgeting** — establish per-frame time budgets for each system
- **Populate MaterialColorMap from MaterialRegistry** — wire the asset-loaded
  material registry into the color map at startup instead of relying on fallbacks
- **Screen-space error as default LOD strategy** — integrate with camera FOV
  query; currently the distance+hysteresis path is used at runtime
- **Sub-voxel refinement** — enable `SubdivisionConfig.max_depth > 0` to
  interpolate below the 1 m voxel grid at surface crossings and material
  boundaries, using the existing `upsample_voxels` / trilinear interpolation
  pipeline

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
- **Rendering style?** Low-poly / stylized is easier to ship.
- **Fluid coupling strategy?** Interface cells? Overlapping domains? TBD.
- **Sound design?** Bevy has built-in audio. Not prioritized yet.
- **Planet scale?** Default is 32 km radius. Earth-scale (6,371 km) requires
  multi-resolution octree leaves for the deep interior — deferred.
- **Tectonic simulation fidelity?** Full mantle convection vs. plate-boundary-only
  stress model. TBD when world-gen pipeline is built.
- **Geoid precision?** Exact equipotential surface vs. oblate spheroid approximation
  for sea level. Spherical harmonics are overkill for a game — ellipsoid is likely
  sufficient.
- **Structural analysis frequency?** Full stress analysis every tick is expensive.
  Budget-limited (every N ticks, or event-driven when loads change). Acceptable
  latency TBD with profiling.
