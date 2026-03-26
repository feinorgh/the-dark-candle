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
  `mesh_dirty_chunks` queries the camera, computes LOD with hysteresis, and
  calls `generate_mesh_lod` with stride = 2^level. Remeshes only on LOD change.
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

Transform the flat-plane terrain into a spherical planetary model. The planet is
a sphere centered at world origin `(0, 0, 0)` with configurable radius. Surface
features come from noise sampled in spherical coordinates. Geological layers
(core → mantle → crust) are defined by radial bands. The existing Cartesian
chunk/octree/meshing/LOD pipeline is preserved — only the terrain function, chunk
loader, gravity direction, and altitude-dependent systems change.

### Design

- **Planet center at origin.** All radial math is trivial: `distance = length(pos)`.
- **Keep Cartesian chunks.** The 32³ chunk grid, octree, SVO, Surface Nets meshing,
  and LOD system are coordinate-agnostic. They see local voxel data and don't care
  about planet geometry.
- **Spherical noise sampling.** Convert world position `(x, y, z)` → unit direction
  → `(lat, lon)` via `atan2`. Noise at `(lat, lon)` produces surface radius
  displacement. No pole distortion because noise is sampled on a sphere, not
  projected from a plane.
- **Shell-based chunk loading.** Only load chunks within a radial band around the
  surface (camera altitude ± view depth). Deep interior and outer space chunks are
  never allocated — same ~250 active chunks as the current flat system.
- **Real planetary gravity.** Apparent gravity = gravitational attraction toward
  planet center + centrifugal pseudo-force from rotation. The resultant vector
  defines local "down" — it points along the inward surface normal of the geoid
  (equipotential surface), just like on a real planet. On a non-rotating sphere
  this is purely radial; rotation deflects it toward the equatorial plane and
  bulges the geoid into an oblate spheroid. Slope forces, ground detection, and
  "which way is up" all derive from this single vector — no hardcoded direction.

### Scale (1 voxel = 1 m)

| Planet | Radius | Chunks across diameter | Feasibility |
|--------|--------|------------------------|-------------|
| Small moon | 10 km | ~625 | Immediate — entire surface tractable |
| Default | 32 km | ~2,000 | Immediate — shell loading keeps it light |
| Large | 100 km | ~6,250 | Needs aggressive LOD + streaming |
| Earth-scale | 6,371 km | ~398K | Future — requires multi-res octree leaves |

Default planet: **32 km radius** (~200 km² surface area, comparable to a large
island). Room for mountains, oceans, biomes, and plate tectonics without needing
Earth-scale infrastructure.

### Implementation Steps

1. **`PlanetConfig`** — new `src/world/planet.rs` + `assets/data/planet_config.ron`
   with center, mean radius, sea level radius, layer radii (inner core, outer core,
   mantle, crust thickness), height scale, rotation rate (rad/s), rotation axis,
   surface gravity (m/s²), planet mass (kg). Data-driven via `RonAssetPlugin`.

2. **Spherical terrain generator** — refactor `terrain.rs`. For each voxel:
   compute distance from center, derive `(lat, lon)`, sample surface radius from
   noise, assign material by radial depth band (core → mantle → crust → surface →
   soil → air/water). Cave carving via 3D noise within the crust band.

3. **Shell-aware chunk loading** — modify `chunk_manager.rs` `desired_chunks()`.
   Compute which chunks intersect the surface shell ± view distance. Skip chunks
   whose centers are deeper than `surface_radius - depth_limit` or higher than
   `surface_radius + sky_limit`.

4. **Planetary gravity model** — modify `gravity.rs` `apply_forces()`. Compute
   apparent gravity per entity as the sum of two forces:
   - **Gravitational acceleration**: `g⃗ = -(G·M / r²) · r̂` where `r̂` is the
     unit vector from center to entity. At the surface this equals the configured
     surface gravity (default 9.80665 m/s²); varies with altitude via inverse-square.
   - **Centrifugal pseudo-force**: `a⃗_c = ω² · d⊥` where `ω` is the planet's
     angular velocity and `d⊥` is the entity's perpendicular distance from the
     rotation axis. This deflects "down" toward the equatorial plane and naturally
     produces an oblate geoid.
   The resultant `g⃗_apparent = g⃗_grav + a⃗_centrifugal` defines local "down" for
   all physics — slope forces, ground detection, "which way is up", buoyancy
   direction, fluid settling. No hardcoded -Y anywhere.
   Update `collision.rs` to find ground along the local gravity vector instead of
   a Y-column scan. `PlanetConfig` gains `rotation_rate: f64` (rad/s; Earth ≈
   7.292e-5).

5. **Altitude-dependent systems** — `WorldConstants.sea_level_y` → `sea_level_radius`.
   Barometric formula uses `altitude = distance_from_center - sea_level_radius`.
   Biome height ranges reinterpreted as altitude above/below sea level radius.

6. **Test suite update** — validate spherical shape, shell loading, radial gravity,
   radial altitude calculations. Preserve existing test semantics where applicable.

7. **World-gen extension points** — architecture prep for future phases:
   `WorldGenPhase` enum, `TectonicPlateConfig`, `GeologicalLayer` struct, spherical
   Voronoi placeholder for plate boundaries. No full implementation yet — just the
   hooks that make the next phase possible.

### What stays unchanged

Chunk internals (`Chunk`, `ChunkCoord`, `Voxel`, `MaterialId`), octree (`OctreeNode`,
`VoxelAccess`), meshing, LOD, interpolation, refinement, all fluid simulations
(AMR, LBM, FLIP), chemistry, biology, behavior, social systems, material RON files.

### Future: World Generation Pipeline

Once spherical terrain is in place, subsequent phases can layer on:

- **Tectonic plates** — spherical Voronoi tessellation. Each plate gets a drift
  vector and rotation rate. Convergent boundaries → mountain ranges / subduction.
  Divergent boundaries → rift valleys / mid-ocean ridges. Transform boundaries →
  fault lines.
- **Geological strata** — material layers within the crust defined by depth,
  composition, and tectonic history. Sedimentary layers near the surface, igneous
  intrusions from volcanic activity, metamorphic zones at depth.
- **Erosion** — hydraulic and thermal erosion during world-gen. Rivers carve
  valleys, glaciers scour mountains, wind shapes deserts. Runs as a multi-pass
  simulation before the player enters.
- **Volcanic activity** — magma plumes from mantle, eruption probability based on
  crust thickness and plate stress. Lava flows use existing fluid simulation.
- **Ocean currents & climate** — large-scale fluid simulation determines temperature
  and moisture distribution. Drives biome placement and weather patterns.

---

## Phase 9 — Atmosphere Simulation ✅

A physics-driven atmosphere model that builds on the existing LBM gas simulation,
barometric pressure functions, and FLIP/PIC particle system. Weather, wind, and
precipitation emerge from first-principles thermodynamics on the spherical planet —
not from scripted weather states. Atmospheric conditions feed into a rendering
pipeline for clouds, fog, and dynamic lighting.

### Foundations already in place

| System | Location | Provides |
|--------|----------|----------|
| Barometric formula | `constants.rs` | `P(h) = P₀ × exp(−Mgh/RT)`, `ρ = PM/(RT)` |
| Pressure diffusion | `pressure.rs` | Chunk-local pressure equalization, gradient forces |
| LBM D3Q19 gas | `lbm_gas/` | Compressible gas dynamics, BGK + Smagorinsky turbulence, buoyancy-driven convection, Guo forcing, `ambient_density_at_altitude()` with lapse rate |
| FLIP/PIC particles | `flip_pic/` | Rain/snow/spray emission, advection, sub-voxel accumulation, phase-transition emission (evaporation, melting) |
| Lighting | `world/mod.rs` | Static DirectionalLight (sun) + AmbientLight |

### Design

Weather emerges from the physics. The atmosphere is a continuous gas field
simulated by LBM at macro scale, with FLIP/PIC handling precipitation and spray.
Solar heating, Coriolis force from planet rotation (Phase 8), and moisture
transport produce realistic circulation patterns.

#### 1. Atmospheric circulation & wind

- **Solar heating model.** Sun position (time-of-day + latitude from spherical
  planet) determines surface insolation. Heated surfaces warm adjacent air →
  density drops → buoyancy drives convection via LBM Guo forcing (already
  implemented).
- **Coriolis force.** Planet rotation (from `PlanetConfig.rotation_rate`) deflects
  moving air masses. Applied as Guo forcing in LBM: `F_coriolis = -2mω × v`.
  Produces trade winds, westerlies, and polar easterlies at appropriate latitudes.
- **Pressure fronts.** High/low pressure cells emerge from differential heating
  and Coriolis deflection. The existing `pressure_gradient()` function already
  computes the force that drives air from high to low pressure.
- **Multi-scale wind.** Near the camera: full LBM resolution (chunk-level gas
  dynamics). Distant regions: coarsened LBM or parametric wind field interpolated
  from planetary circulation model. Transition zone blends the two.

#### 2. Moisture transport & cloud formation

- **Humidity field.** Each LBM gas cell gains a moisture scalar (kg water vapor
  per kg air). Advected by LBM velocity field. Evaporation from water/wet surfaces
  adds moisture; precipitation removes it.
- **Saturation & dew point.** Clausius-Clapeyron equation gives saturation vapor
  pressure as a function of temperature: `e_s(T) = e₀ × exp(L/R_v × (1/T₀ - 1/T))`.
  When humidity exceeds saturation → condensation → cloud formation.
- **Cloud formation.** Condensation produces cloud density (kg/m³ liquid water
  content). Rising air cools adiabatically (lapse rate ~6.5 K/km) → clouds form
  at the lifting condensation level. Cloud types emerge from dynamics:
  - Cumulus: strong updrafts, localized convection
  - Stratus: stable layers, widespread lifting
  - Cumulonimbus: deep convection → precipitation
- **Precipitation trigger.** When cloud water content exceeds a coalescence
  threshold → FLIP/PIC particle emission (rain, snow depending on temperature).
  Existing `emit_rain()` / accumulation pipeline handles particle lifecycle.
  Raindrop evaporation below cloud base (virga) emerges from temperature-dependent
  phase transition.

#### 3. Weather phenomena

- **Frontal systems.** Where air masses of different temperature/humidity meet,
  density contrasts create frontal boundaries. Warm fronts (warm air overriding
  cold → gradual lifting → stratus/rain), cold fronts (cold air undercutting warm
  → sharp lifting → cumulonimbus/storms) emerge from the LBM simulation.
- **Storm cells.** Deep convection in unstable atmospheres. Updraft + moisture →
  cumulonimbus → heavy precipitation + strong winds. Intensity scales with
  available convective energy (temperature differential × moisture content).
- **Fog.** Radiation fog: surface cools at night → air at surface reaches dew
  point → condensation at ground level. Advection fog: warm moist air moves over
  cold surface.

#### 4. Climate zones

- **Latitude-driven.** Solar angle varies with latitude on the spherical planet →
  equatorial heating → polar cooling → Hadley/Ferrel/Polar circulation cells
  emerge from LBM + Coriolis.
- **Altitude-driven.** Barometric formula + lapse rate (already in LBM
  `ambient_density_at_altitude()`) → temperature drops with altitude → alpine
  climate, snow lines.
- **Ocean influence.** Proximity to water bodies moderates temperature extremes.
  Ocean surface temperature drives evaporation rates → coastal moisture/humidity.
- **Biome feedback.** Climate parameters (temperature, precipitation, humidity)
  feed into the existing biome selection system (`biomes.rs`
  `biome_matches(height, temperature, moisture)`).

#### 5. Atmospheric rendering

- **Volumetric clouds.** Ray-marched cloud volumes from the moisture/condensation
  field. Cloud density maps to optical thickness. Light scattering: silver lining
  (forward scattering), dark bases (absorption), colored sunsets (Mie/Rayleigh).
  LOD: full ray-march near camera, billboard impostors at distance.
- **Atmospheric scattering.** Rayleigh scattering (blue sky, red sunsets) +
  Mie scattering (haze, sun halos). Implemented as a post-process or sky shader
  that reads sun position and atmospheric density. On the spherical planet, sky
  color varies with altitude (thinner atmosphere = darker sky).
- **Dynamic shadows.** Cloud shadows on terrain via shadow mapping from cloud
  density field projected along sun direction. Overcast conditions reduce
  `DirectionalLight` illuminance and increase `AmbientLight` (diffuse scattering
  through cloud layer).
- **Fog & haze.** Distance fog density driven by humidity and temperature (real
  atmospheric visibility). Exponential fog with altitude-dependent density.
  Morning mist in valleys (cold air pooling + high humidity).
- **Time-of-day.** Sun position orbits based on planet rotation. Dawn/dusk color
  temperature shifts. Night sky with ambient starlight. Moon phases (optional).
  `DirectionalLight` rotation, color, and illuminance update each frame from
  solar angle.

### Implementation Steps

1. **`AtmosphereConfig`** — new `src/physics/atmosphere.rs` +
   `assets/data/atmosphere_config.ron`. Fields: surface temperature gradient,
   lapse rate, tropopause altitude, humidity baseline, Coriolis toggle,
   cloud coalescence threshold, scattering coefficients (Rayleigh/Mie).
   Data-driven via `RonAssetPlugin`.

2. **Humidity transport** — extend `LbmCell` with a moisture scalar. Advect
   moisture with the LBM velocity field (passive scalar transport). Evaporation
   source term at water/wet surfaces. Condensation sink when exceeding saturation
   (Clausius-Clapeyron). Couple with existing `sync_to_chunk` / `sync_from_chunk`.

3. **Coriolis forcing** — add Coriolis pseudo-force to LBM Guo forcing pass
   in `lbm_gas/step.rs`. `F = -2m(ω × v)` where `ω` comes from
   `PlanetConfig.rotation_rate` and rotation axis. Latitude-dependent
   (strongest at poles, zero at equator).

4. **Solar heating model** — compute surface insolation from sun angle × latitude
   × time-of-day. Apply as thermal source term to surface-adjacent air cells.
   Differential heating drives convection cells. Day/night cycle from planet
   rotation.

5. **Cloud formation** — when condensation produces liquid water content above
   threshold, mark cells as cloud. Track cloud density per cell. Trigger FLIP/PIC
   rain/snow emission when cloud water exceeds coalescence threshold. Feed cloud
   field to rendering pipeline.

6. **Precipitation pipeline** — wire cloud condensation → FLIP/PIC emission for
   rain (water particles, T > 273 K) and snow (ice particles, T < 273 K).
   Particles fall under gravity, evaporate in dry air below cloud base,
   accumulate on terrain via existing accumulation system.

7. **Time-of-day & dynamic lighting** — sun position from planet rotation angle.
   Update `DirectionalLight` transform, illuminance, and color temperature each
   frame. Night: reduce to moonlight/starlight levels. Dawn/dusk: warm color
   shift.

8. **Cloud shadow mapping** — project cloud density field along sun direction onto
   terrain. Modulate `DirectionalLight` shadow intensity. Dense clouds →
   reduced direct illumination + increased ambient (diffuse scattering).

9. **Atmospheric scattering shader** — sky dome / post-process pass implementing
   Rayleigh + Mie scattering. Reads sun position, camera altitude, atmospheric
   density profile. Blue sky overhead, red/orange at horizon during sunset,
   thinner atmosphere at altitude.

10. **Fog system** — exponential height fog driven by humidity and temperature
    fields. Morning valley fog from cold air pooling. Visibility distance scales
    with atmospheric moisture content.

### Dependencies

- Steps 1–6 depend on **Phase 8** (spherical planet, radial gravity, planet
  rotation).
- Steps 2–3 depend on the existing LBM gas plugin being wired into runtime
  (`PhysicsPlugin::build()`).
- Steps 7–10 (rendering) can proceed partially in parallel with physics steps.
- Step 5 (clouds) depends on step 2 (humidity) and step 4 (solar heating).
- Step 6 (precipitation) depends on step 5 (clouds).
- Step 8 (cloud shadows) depends on step 5 (clouds) and step 7 (dynamic lighting).

### What stays unchanged

LBM core (collision, streaming, macroscopic recovery), FLIP/PIC core (P2G, G2P,
advection, accumulation), pressure diffusion, barometric formula, material system,
chunk/voxel infrastructure. These are extended (humidity scalar, Coriolis force)
but not rewritten.

### Completion

All 10 implementation steps completed. Commits:

| Commit | Description |
|--------|-------------|
| `0073040` | AtmosphereConfig resource + wire LBM/FLIP into `PhysicsPlugin` gameplay loop |
| `412135a` | Humidity transport (passive scalar advection), Coriolis Guo forcing, solar surface heating |
| `0f20748` | Cloud formation (Clausius-Clapeyron condensation cycle) + atmospheric Rayleigh/Mie scattering |
| `6066811` | Precipitation pipeline (rain/snow FLIP particles, virga) + volumetric cloud ray-marcher |
| `c8122ef` | 14 atmosphere physics integration tests + cloud shadow maps / exponential height fog |
| `a511f06` | 5 CPU atmosphere visualization video tests (sky, clouds, shadows, fog, integrative showcase) |
| `cfea4ce` | GPU-accelerated compute shader renderer (WGSL uber shader, 1000× speedup over CPU) |

**Key modules added:**
- `src/physics/atmosphere.rs` — AtmosphereConfig, Clausius-Clapeyron, dew point
- `src/physics/lbm_gas/moisture.rs` — evaporation, condensation, scalar advection
- `src/physics/lbm_gas/precipitation.rs` — cloud-to-rain/snow emission, virga evaporation
- `src/lighting/scattering.rs` — Rayleigh + Mie CPU ray-marcher, sky LUT
- `src/lighting/clouds.rs` — volumetric cloud ray-march (Beer-Lambert + Henyey-Greenstein)
- `src/lighting/shadows.rs` — cloud shadow projection, exponential height fog
- `src/gpu/` — headless wgpu compute shader renderer (`GpuRenderer`)

**GPU renderer performance** (512×384 output, 30 fps video):

| Test | CPU (release) | GPU | Speedup |
|------|--------------|-----|---------|
| Sky panorama (360 frames) | 349 s | 1.0 s | 342× |
| Volumetric clouds (300 frames) | 931 s | 0.9 s | 1 070× |
| Full showcase (900 frames) | 4 052 s | 2.5 s | 1 608× |

---

## Phase 10 — Entity Bodies & Organic Physics (planned)

Physical embodiment for all living entities — player, creatures, and plants.
Replaces the abstract point-entity model (Phase 4) with articulated bodies
that have mass-distributed skeletal structures, soft/rigid tissue physics,
field of vision, and locomotion driven by anatomy. The player is a regular
creature entity controlled by input rather than AI; no special-case code.

### Foundations already in place

| System | Location | Provides |
|--------|----------|----------|
| Entity physics | `gravity.rs`, `collision.rs` | Force model (gravity + buoyancy + drag + friction), AABB collision |
| Rigid body dynamics | `rigid_body.rs`, `solver.rs` | Angular velocity, moment of inertia, sequential impulse solver |
| Creature data | `procgen/creatures.rs`, `assets/data/` | `CreatureData` RON with stats, color, biome spawning |
| Biology | `biology/` | Metabolism, health, growth/aging, death/decomposition |
| Behavior & perception | `behavior/` | Sight/hearing/smell, pathfinding, needs, utility AI |
| Material properties | `data/mod.rs` | Density, elasticity (Young's modulus), friction, restitution |

### Design

#### 1. Skeletal system

A data-driven skeleton defined per species in `.skeleton.ron` files:

- **Bones** — rigid segments with length (m), mass (kg), and material
  (calciumite for vertebrates, chitin for arthropods, cellulose for plants).
  Each bone has a parent bone (forming a tree), rest pose transform, and
  joint constraints (hinge, ball-and-socket, fixed) with angular limits.
- **Skeleton tree** — root bone (pelvis for bipeds, thorax for insects,
  trunk base for trees). Child bones inherit parent transforms. Forward
  kinematics propagates pose; inverse kinematics solves foot/hand placement.
- **SkeletonData** struct — derives `serde::Deserialize`, `Asset`,
  `TypePath`. Loaded via `RonAssetPlugin<SkeletonData>`. Fields: `bones:
  Vec<BoneData>`, `joints: Vec<JointData>`, `rest_pose: Vec<Transform>`.
- **Skeleton component** — runtime `Skeleton` ECS component holding current
  bone transforms, angular velocities, and accumulated torques.

#### 2. Soft & rigid tissue

Organic bodies are not uniform rigid bodies — they have tissue layers with
distinct mechanical properties:

- **Rigid tissue** (bone, shell, wood) — modeled as rigid body segments
  connected by joints. Uses existing rigid body solver for each segment.
  Material properties from `.material.ron` (density, Young's modulus).
- **Soft tissue** (muscle, fat, skin, bark, leaves) — modeled as
  mass-spring-damper systems anchored to the skeleton. Provides visual
  deformation and collision volume. Spring stiffness and damping from
  tissue material properties.
- **TissueData** — per-species `.body.ron` defines tissue layers per body
  region: `{ region: "torso", layers: [{ tissue: "muscle", thickness: 0.04,
  density: 1060.0 }, { tissue: "fat", thickness: 0.02, density: 920.0 },
  { tissue: "skin", thickness: 0.003, density: 1100.0 }] }`.
- **Collision volumes** — each body region generates a collision capsule
  (or convex hull) from bone length + tissue thickness. Replaces the
  single-AABB model from Phase 3.

#### 3. Locomotion

Movement emerges from skeletal articulation, not velocity teleportation:

- **Gait definitions** — `.gait.ron` files define named animation cycles
  per skeleton: walk, run, crawl, swim, fly, slither. Each gait specifies
  bone target angles per phase, cycle duration, ground contact windows,
  and energy cost (J/m from metabolism).
- **Procedural animation** — IK solvers place feet on terrain surface,
  blend between gaits based on speed/slope/medium. No canned keyframe
  animations — all poses are computed from skeleton constraints + IK targets.
- **Locomotion modes:**
  - *Bipedal/quadrupedal walking* — alternating leg IK with balance
    correction (center of mass over support polygon).
  - *Crawling* — low-clearance gait, belly contact, limbs splayed.
  - *Flying* — wing bones generate lift force proportional to wing area ×
    airspeed² × lift coefficient. Drag from body cross-section.
    Sustained flight requires metabolic energy.
  - *Swimming* — drag-based propulsion in fluid voxels. Fin/limb surface
    area determines thrust.
  - *Slithering* — sinusoidal body wave via sequential bone rotations.
    Friction with ground provides forward force.
  - *Climbing* — IK grip targets on vertical surfaces, weight transfer
    between grip points.
- **Player input mapping** — player input (WASD, jump, crouch) maps to
  gait selection and IK target adjustments on the player's creature
  skeleton. Same system as AI locomotion, different input source.

#### 4. Field of vision & perception bodies

Upgrade the abstract perception system (Phase 6) to use physical geometry:

- **Eye components** — position on skeleton (bone attachment point), FOV
  cone angle, max range. Occlusion via DDA ray-cast against voxel world
  (reuses `src/world/raycast.rs`). Multiple eyes = wider combined FOV.
- **Ear components** — position on skeleton, sensitivity curve, directional
  bias from head orientation.
- **Player camera** — first-person camera attaches to the player entity's
  head bone. Camera FOV = eye FOV. Head-bob from locomotion gait. No
  special player camera system — just a `Camera3d` parented to the head
  bone entity.
- **Smell** — unchanged from Phase 6 (diffusion-based, no body geometry
  needed).

#### 5. Injury & damage model

Tiered physical damage integrated with the skeletal/tissue system:

- **Damage zones** — each body region (head, torso, limb, wing, root, etc.)
  tracks its own hit points derived from tissue mass and material toughness.
- **Injury tiers:**
  - *Bruise/strain* — soft tissue damage. Reduces performance (movement
    speed, grip strength). Heals over time via metabolism.
  - *Fracture* — bone damage. Limb loses structural support — IK solver
    treats fractured bone as a limp/hanging segment. Requires healing time
    proportional to bone mass.
  - *Severing* — catastrophic damage separates a body part. Detached part
    becomes a physics entity (drops with rigid body dynamics). Creature
    loses capabilities associated with that limb permanently (or until
    regeneration, if the species supports it).
- **Damage propagation** — impacts apply force to the collision volume of
  the hit region. Force exceeding tissue toughness creates injury. Armor
  (equipped items with material hardness) absorbs force first.
- **Healing** — biological healing rate from Phase 5 metabolism, scaled by
  injury tier. Fractures heal slowly. Severing doesn't heal without
  regeneration trait.

#### 6. Plant body physics

Trees and large plants as semi-rigid articulated structures:

- **Trunk & branches** — modeled as a skeleton tree. Trunk = root bone,
  branches = child bones. Wood material properties (density, Young's
  modulus, flexural strength from Phase 11 building materials).
- **Root system** — anchor bones extending into terrain voxels. Root
  depth + spread determines wind resistance and nutrient access (ties
  into Phase 5 plant growth).
- **Canopy** — leaf clusters as soft-body masses on branch tips. Wind
  force (from LBM gas field, Phase 9) applies lateral load. Branches
  flex under wind + gravity. Excessive force → branch breakage (uses
  flexural strength).
- **Growth integration** — as plants grow (Phase 5), new bones are added
  to the skeleton. Trunk thickens (bone radius increases), branches
  extend, canopy fills out. Growth rate from metabolism.
- **Felling & damage** — chopping applies damage to trunk bone. When trunk
  HP reaches zero → tree falls as a rigid body chain (bones disconnect
  from root anchor, gravity takes over). Fallen tree becomes harvestable
  material.

### Implementation steps

1. **`SkeletonData` and RON loader** — new `src/bodies/skeleton.rs`.
   `SkeletonData` struct with bones, joints, rest pose. Register
   `RonAssetPlugin<SkeletonData>`. Create skeleton RON files for 2–3
   species (humanoid, quadruped, tree).

2. **`Skeleton` runtime component** — ECS component with current bone
   transforms, angular state. Forward kinematics system in `FixedUpdate`.
   Parent-child transform propagation.

3. **Tissue & collision volumes** — new `src/bodies/tissue.rs`. `BodyData`
   RON with tissue layers per region. Generate per-region collision
   capsules from bone + tissue. Replace single AABB with compound collider.

4. **IK solver** — new `src/bodies/ik.rs`. FABRIK or CCD inverse
   kinematics for limb chains. Foot placement on terrain. Hand/grip
   targeting. Joint constraint enforcement.

5. **Locomotion gaits** — new `src/bodies/locomotion.rs`. `GaitData` RON
   with bone angle targets per phase. Gait state machine (idle → walk →
   run → sprint). Procedural gait blending. Energy cost integration with
   metabolism.

6. **Player embodiment** — player entity spawns with same `Skeleton` +
   `BodyData` as a humanoid creature. Input system maps WASD → gait
   selection → IK targets. `Camera3d` parented to head bone.

7. **Perception body integration** — new `src/bodies/perception.rs`. Eye/ear
   components with skeleton attachment points. FOV occlusion via DDA
   ray-cast. Replace abstract Phase 6 perception radius with physical
   sight cones.

8. **Injury system** — new `src/bodies/injury.rs`. Per-region damage
   tracking. Injury tier logic (bruise → fracture → sever). IK response
   to fractures. Severed limb spawning. Healing rate integration.

9. **Plant bodies** — extend skeleton system for plants. `TreeSkeletonData`
   RON. Wind response system (LBM pressure → branch torque). Growth-driven
   skeleton expansion. Felling mechanics.

10. **Physics integration** — wire articulated body solver into
    `FixedUpdate`. Per-bone collision response via existing narrow phase +
    impulse solver. Mass distribution from tissue layers → moment of
    inertia tensor per bone.

### Dependencies

- Steps 1–3 build on Phase 3 (rigid body physics) and Phase 4 (creature data).
- Step 4 (IK) is self-contained, depends only on step 2 (skeleton).
- Step 5 (locomotion) requires steps 2 + 4 (skeleton + IK).
- Step 6 (player) requires steps 5 (locomotion) + 7 (perception).
- Step 7 extends Phase 6 (behavior/perception) with body geometry.
- Step 8 extends Phase 5 (biology/health) with body-part damage.
- Step 9 depends on steps 1–3 (skeleton + tissue) and Phase 9 (LBM wind).
- Step 10 integrates everything into the physics pipeline.

### What stays unchanged

Creature RON data (extended with skeleton/body references, not replaced).
Biology systems (metabolism, growth — extended with per-limb damage, not
replaced). Behavior AI (action selection unchanged — locomotion replaces
the velocity output). Existing rigid body solver (reused for per-bone
dynamics). Voxel collision (extended from single AABB to compound, not
replaced).

---

## Phase 11 — Buildings & Structural Construction (planned)

A freeform building system where players construct structures from physical
materials — wood, stone, metal, glass, concrete, clay, and more — all defined as
data in RON files. Building parts attach to each other and to the terrain through
joints that transmit forces. Structures interact with the full physics stack:
gravity loads them, wind pushes them, fire burns them, explosions shatter them.
Failure is emergent — buildings collapse when stress exceeds material strength,
not from scripted destruction.

### Foundations already in place

| System | Location | Provides |
|--------|----------|----------|
| Material properties | `data/mod.rs`, `assets/data/materials/` | Density, hardness, Young's modulus, friction, restitution, thermal properties, combustion, phase transitions (12 materials) |
| Structural integrity | `integrity.rs` | Flood-fill connectivity from anchored voxels, unsupported collapse |
| Voxel subdivision | `octree.rs`, `refinement.rs` | SVO with adaptive refinement at damage gradients, material boundaries |
| Chemistry | `chemistry/` | Heat diffusion, combustion reactions, state transitions (wood burns, stone melts, ice melts) |
| Item system | `procgen/items.rs`, `data/mod.rs` | Item templates with material-derived weight/durability, 4 items defined |
| Physics | `gravity.rs`, `pressure.rs`, `collision.rs` | Force-based entity physics, pressure propagation |

### Design

#### 1. Extended material properties

`MaterialData` already has density, hardness, Young's modulus (stiffness),
friction, and restitution. Add structural strength properties (all in Pascals):

| Property | Unit | Description | Example: Wood | Example: Iron |
|----------|------|-------------|---------------|---------------|
| `tensile_strength` | Pa | Max stress before fracture under tension | 40 MPa | 400 MPa |
| `compressive_strength` | Pa | Max stress before crushing | 30 MPa | 250 MPa |
| `shear_strength` | Pa | Max stress before shearing | 8 MPa | 170 MPa |
| `flexural_strength` | Pa | Max bending stress before snapping | 50 MPa | 350 MPa |
| `fracture_toughness` | Pa·√m | Resistance to crack propagation | 10 MPa·√m | 50 MPa·√m |

These are optional fields on `MaterialData` (existing materials get real-world
values added to their `.material.ron` files). Materials without strength values
(air, water, steam) cannot be used as structural elements.

#### 2. Building parts

Building parts are shapes made from a single material, placed by the player.
Each part type is defined in a `.part.ron` file:

- **Block** — 1×1×1 m solid cube. The basic unit.
- **Slab** — 1×1×0.5 m half-height. Floors, shelves.
- **Beam** — 0.25×0.25×N m elongated member. Structural frames.
- **Column** — 0.5×N×0.5 m vertical support. Load-bearing pillars.
- **Wall** — 1×N×0.1 m thin panel. Partitions, facades.
- **Arch** — curved shape with keystone geometry. Bridges, doorways.
- **Stair** — stepped wedge. Vertical traversal.
- **Roof** — angled slab. Water shedding, shelter.

Part definitions specify voxel occupancy (which sub-voxels are filled),
material slot (which material it's made of), and attachment faces (where
other parts can connect). The octree subdivision system represents parts
at sub-voxel resolution — a 0.25 m beam uses depth-2 subdivision within
its host voxel.

#### 3. Attachment & joints

Parts connect at shared faces. Each joint has:

- **Contact area** (m²) — derived from overlapping face geometry.
- **Joint strength** — `min(material_A_strength, material_B_strength) × contact_area`.
  Uses the weakest of tensile/compressive/shear depending on the load direction.
- **Joint type** — rigid (mortar, welding, nails), friction (dry-stacked stone),
  or hinge (door, gate). Type affects which stress modes the joint resists.

Attachment rules:
- Parts snap to a grid aligned with the voxel coordinate system.
- Any two parts with adjacent filled sub-voxels form a joint automatically.
- The player can upgrade joints (apply mortar to stone, nail wood, weld metal)
  to increase strength.
- Terrain voxels act as anchor points (infinite compressive strength, like
  bedrock in the current integrity system).

#### 4. Structural analysis

Upgrade the existing flood-fill integrity system (`integrity.rs`) to a
force-based stress analysis:

- **Load path tracing.** From every part, trace the path gravity forces
  take through joints to the ground. Each joint accumulates the load it
  carries.
- **Stress calculation.** At each joint: `σ = F / A`. Compare against the
  relevant material strength (compressive for columns, tensile for hanging
  loads, shear for lateral forces, flexural for beams).
- **Wind loading.** Exposed surfaces receive force from atmospheric pressure
  gradients (Phase 9 LBM wind field). Tall/wide structures accumulate more
  wind load.
- **Dynamic loads.** Impacts (explosions, projectiles, falling debris)
  apply impulse forces. Joints that exceed their strength break.
- **Progressive collapse.** When a joint breaks, load redistributes to
  neighboring joints. If they also fail → cascade → realistic structural
  failure. Uses the existing damage field on `Voxel` (0.0 = destroyed,
  1.0 = intact) to track degradation.
- **Creep & fatigue** (future). Long-term loads near the strength limit
  gradually degrade joints. Wooden structures rot, metal corrodes (ties
  into chemistry system).

The analysis runs on `FixedUpdate` at a budget-limited frequency (not
every tick — every N ticks or when loads change). The flood-fill fallback
remains for chunks without active structures (performance).

#### 5. Building materials

Expand the material library with construction-specific materials. Each is a
new `.material.ron` file with full SI properties:

| Material | Density | Compressive | Tensile | Notes |
|----------|---------|-------------|---------|-------|
| Oak wood | 600 kg/m³ | 30 MPa | 40 MPa | Burns, biodegrades |
| Pine wood | 500 kg/m³ | 25 MPa | 35 MPa | Lighter, weaker |
| Granite | 2700 kg/m³ | 130 MPa | 7 MPa | Strong in compression, weak in tension |
| Limestone | 2300 kg/m³ | 60 MPa | 4 MPa | Sofite, carvable |
| Brick | 1900 kg/m³ | 20 MPa | 2 MPa | Requires mortar joints |
| Concrete | 2400 kg/m³ | 40 MPa | 3 MPa | Very weak in tension |
| Wrought iron | 7700 kg/m³ | 250 MPa | 350 MPa | Strong in tension |
| Bronze | 8800 kg/m³ | 200 MPa | 300 MPa | Corrosion resistant |
| Copper | 8900 kg/m³ | 70 MPa | 210 MPa | Ductile, conducts heat |
| Glass | 2500 kg/m³ | 1000 MPa | 33 MPa | Brittle, transparent |
| Clay (dried) | 1800 kg/m³ | 3 MPa | 0.5 MPa | Weak, cheap, fire-hardens to brick |
| Thatch | 240 kg/m³ | 0.5 MPa | 1 MPa | Insulating, burns easily |

These interact with existing systems: wood burns (combustion reactions),
stone melts to lava, ice melts to water, metals conduct heat efficiently.

#### 6. Player building mechanics

- **Placement mode.** Player enters build mode, selects part type + material.
  Ghost preview shows placement on the grid. Snap to adjacent parts or terrain.
- **Rotation.** Parts rotate in 90° increments around any axis.
- **Demolition.** Player can remove parts they placed. Removed parts drop as
  items (or break into debris if damaged).
- **Material sourcing.** Building requires material items in inventory. Mining
  terrain yields raw materials (stone, dirt, sand). Crafting converts raw
  materials into construction materials (wood → planks, clay + fire → bricks,
  sand + heat → glass, ore + smelting → metal ingots).
- **Crafting recipes.** Defined in `.recipe.ron` files. Input materials +
  tool requirements + processing (heat, time) → output material/part.

#### 7. Physics interactions

Structures are not separate from the world — they are voxels with materials
and joints, subject to all existing physics:

- **Gravity.** Structures bear their own weight. Overhangs need support.
  Apparent gravity from Phase 8 determines load direction everywhere on the
  sphere.
- **Fire.** Wooden structures burn via existing combustion reactions. Fire
  weakens joints (temperature degrades material strength). Stone/metal
  structures survive fire but conduct heat.
- **Explosions.** Pressure waves from `pressure.rs` apply impulse to
  structural surfaces. Joints near the blast fail → debris.
- **Fluid interaction.** Rising water (AMR) exerts buoyancy and hydrostatic
  pressure on submerged walls. Wind (LBM) applies lateral force.
- **Erosion.** FLIP/PIC particles (rain, sand) erode exposed surfaces over
  time, reducing `Voxel.damage`.

### Implementation Steps

1. **Extended `MaterialData`** — add `tensile_strength`, `compressive_strength`,
   `shear_strength`, `flexural_strength`, `fracture_toughness` (all `Option<f32>`
   in Pa / Pa·√m) to `MaterialData` struct. Update all 12 existing
   `.material.ron` files with real-world values. Validate in tests.

2. **`PartData` and part RON files** — new `src/building/parts.rs`. `PartData`
   struct: name, voxel shape (occupancy mask at subdivision depth),
   attachment faces, material slot. Create `assets/data/parts/*.part.ron`
   for block, slab, beam, column, wall, arch, stair, roof. Register via
   `RonAssetPlugin<PartData>`.

3. **Joint system** — new `src/building/joints.rs`. `Joint` component linking
   two adjacent parts. Computed contact area, type (rigid/friction/hinge),
   current stress, damage accumulation. Joints are created automatically
   when parts are placed adjacent to each other.

4. **Structural stress analysis** — new `src/building/stress.rs`. Replace
   flood-fill integrity with load-path analysis for chunks containing
   building parts. Compute stress per joint from gravity + external loads.
   Break joints exceeding material strength. Progressive collapse via
   load redistribution. Keep flood-fill fallback for non-building chunks.

5. **New construction materials** — add `.material.ron` files for granite,
   limestone, brick, concrete, wrought iron, bronze, copper, glass, dried
   clay, thatch, oak, pine, planks. Full SI properties. New `MaterialId`
   constants where needed.

6. **Crafting recipe system** — new `src/building/crafting.rs` +
   `assets/data/recipes/*.recipe.ron`. `RecipeData`: input materials,
   quantities, tool requirements, processing conditions (temperature,
   duration), output material/part. `RonAssetPlugin<RecipeData>`.

7. **Player placement system** — new `src/building/placement.rs`. Build-mode
   toggle, ghost preview, grid snapping, rotation, placement validation
   (support check, material availability), part spawning with joint creation.

8. **Demolition & debris** — part removal drops items or breaks into physics
   debris (voxel fragments with `PhysicsBody`). Debris inherits material
   and velocity from the destroyed part.

9. **Physics integration** — wire structural analysis into `FixedUpdate`.
   Connect wind loading (Phase 9 LBM pressure field), hydrostatic pressure,
   explosion impulse, fire damage (temperature-dependent strength reduction)
   to the stress system.

10. **Inventory system** — new `src/entities/inventory.rs`. Per-entity item
    storage with weight/volume limits. Material items for building. UI
    integration (future).

### Dependencies

- Steps 1, 5 extend the existing material system (no phase dependency).
- Steps 2–4, 6–8 can begin after step 1 (material properties).
- Step 9 integrates with Phase 8 (radial gravity) and Phase 9 (wind loading).
- Step 7 requires step 10 (inventory) for material consumption.
- Step 4 (stress analysis) benefits from Phase 8 (radial gravity direction)
  but can initially use the current -Y gravity.

### What stays unchanged

Voxel storage, chunk management, meshing pipeline, octree structure (used by
parts for sub-voxel resolution), chemistry system (fire/heat/reactions apply
to building materials automatically), existing material RON files (extended,
not replaced), existing item system (extended with building items).

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

### Coupling & Integration
- **Cross-model fluid coupling** — AMR ↔ LBM mass/heat exchange at liquid-gas
  interfaces; FLIP particles entering/leaving LBM gas fields
- **Plugin activation** — wire AmrFluidPlugin, LbmGasPlugin, FlipPicPlugin into
  PhysicsPlugin::build() for runtime use (currently test-only). Physics are also
  validated by the [simulation test framework](simulation-test-system.md).

### Physics Coupling Layer (Entity ↔ World)
The individual physics engines (rigid body solver, LBM gas, AMR fluid, FLIP
particles, pressure diffusion, heat transfer) are each functional but largely
siloed. A coupling layer is needed so that world-level fields exert forces on
entities and entity actions feed back into world state.

- **Pressure gradient → entity impulse** — sample the pressure field around an
  entity's AABB; the net pressure difference across its surface produces a force
  (F = −∇P × V_displaced). Enables explosion shockwaves to push rigid bodies and
  pressure-driven object movement
- **Wind field → entity drag** — sample LBM velocity at an entity's position;
  compute aerodynamic drag using the entity's `DragProfile` and the *relative*
  velocity (entity velocity minus wind). Enables wind-blown NPCs, flags,
  projectile drift
- **Fluid buoyancy coupling** — the existing buoyancy system uses a generic
  medium density fallback. Couple it to actual AMR fluid voxel state: sample
  fluid density and velocity at the entity's submerged volume. Enables realistic
  floating, sinking, and current-driven drift
- **Particle–entity collisions** — FLIP/PIC particles (rain, spray, debris)
  currently pass through entities. Add narrow-phase tests between particles and
  entity colliders; on hit, transfer momentum (particle → entity impulse) and
  trigger accumulation (wetting, erosion, coating)
- **Collision damage feedback** — the impulse solver already computes contact
  impulse magnitudes. Expose peak impulse per contact pair per frame; when it
  exceeds a material-dependent damage threshold, emit a `DamageEvent`. Enables
  fall damage, impact breakage, and projectile lethality without hardcoded HP
  deductions
- **Heat field → entity temperature** — entities in hot/cold environments should
  gain/lose heat via convection (sample ambient voxel temperature + wind speed
  around the entity). Enables freezing hazards, fire proximity damage, and
  cooking mechanics

Design constraint: all coupling uses existing SI fields — no new magic constants.
Forces emerge from pressure in Pascals, velocity in m/s, temperature in Kelvin.
The coupling layer is a set of systems that *read* world fields and *write*
entity forces (and vice versa), not a new physics engine.

### Rigid Body Physics
The existing entity physics (`PhysicsBody`, `Mass`, `DragProfile`, `Collider`)
handles entity-vs-voxel forces and AABB terrain collision. A full rigid body
system needs:

- **Entity-vs-entity collision** — broad phase (spatial hash / sweep-and-prune)
  + narrow phase (AABB or GJK) between dynamic entities
- **Restitution & friction coefficients** — per-material bounce and slide
  behavior on collision response (impulse-based)
- **Angular dynamics** — `AngularVelocity`, `MomentOfInertia`, `Torque`
  components. Rotational integration in `FixedUpdate`. Coupled with linear
  response at contact points
- **Contact resolution** — sequential impulse solver or position-based correction
  for penetration, stacking, and resting contact
- **Collision shapes** — extend `Collider` beyond AABB: sphere, capsule, convex
  hull for entity-entity narrow phase
- **Spatial partitioning** — uniform grid or dynamic BVH for efficient
  broad-phase entity queries
- **Sleep system** — deactivate rigid bodies whose linear and angular velocities
  stay below a threshold for N consecutive frames. Sleeping bodies skip force
  integration, broad/narrow phase, and solver work. Wake on: external impulse,
  nearby collision, or explicit event. Eliminates residual micro-bounce on
  resting contacts and saves CPU for large entity counts

Design constraint: all collision properties (restitution, friction) derive from
`MaterialData` in RON files. No magic numbers — emergent behavior from SI
material properties.

### Soft Body Physics (Future)
Deformable objects that bend, stretch, compress, or tear under stress — as
opposed to the infinitely-stiff assumption of rigid body dynamics. Candidate
implementation: mass-spring lattice or position-based dynamics (PBD/XPBD).

- **Elastic deformation** — objects return to rest shape when stress < yield
  strength. Governed by Young's modulus (already in `MaterialData`)
- **Plastic deformation & fracture** — permanent shape change above yield
  strength; tearing/snapping above ultimate strength
- **Use cases** — cloth (banners, nets), rope/vines/tethers, organic creatures
  (slimes, tentacles, flesh deformation on impact), vegetation bending in wind
- **Terrain deformation** — soft materials (mud, snow, sand) compressing under
  load rather than discrete voxel placement/removal
- **Coupling with rigid bodies** — soft body nodes exert forces on rigid bodies
  and vice versa (e.g. rope attached to a rigid anchor)

Design constraint: material stiffness, damping, and yield/ultimate strength
derive from `MaterialData` RON files. Deformation emerges from the interaction
of applied forces and material properties — no hardcoded spring constants.

Priority: low. Rigid body physics, fluids (LBM/FLIP), and structural integrity
cover most gameplay needs. Soft bodies become relevant when a core mechanic
requires deformation (rope/grapple, ragdolls, destructible organic entities).

### Constraints & Joints (Future)
Rigid body constraints that restrict relative motion between two entities or
between an entity and a fixed anchor point. Required for mechanical gameplay.

- **Distance constraint** — maintains fixed separation between two anchor points
  (ropes, chains, tethers). Enforce via position-based or impulse-based solver
- **Hinge / revolute joint** — rotation around a single axis (doors, levers,
  cranks, hinged lids)
- **Prismatic / slider joint** — translation along a single axis (pistons,
  sliding doors, drawbridges)
- **Ball-and-socket joint** — rotation around a point with no axis restriction
  (ragdoll shoulders/hips, hanging lanterns)
- **Spring-damper** — elastic constraint with configurable stiffness (N/m) and
  damping (N·s/m). Suspension, bungee cords, shock absorbers
- **Motor** — applies torque or linear force to a joint axis (windmills,
  conveyor belts, powered doors)
- **Breakable constraints** — joints that snap when force exceeds a threshold
  derived from material tensile/shear strength. Enables chain-breaking,
  structural tearing

Design constraint: constraint parameters (stiffness, damping, break force)
derive from `MaterialData` where applicable. Solve within the existing
sequential impulse solver by adding constraint rows alongside contact rows.

### Explosion & Detonation Mechanics (Future)
Explosions as a first-class physics event, bridging pressure diffusion, rigid
body dynamics, and voxel destruction.

- **Detonation source** — an entity or voxel event that injects energy (J) into
  the pressure field at a point. Energy derived from material heat of combustion
  (J/kg) × mass
- **Blast wave propagation** — pressure diffusion (or LBM shock) radiates
  outward. Overpressure decays with distance (inverse-cube for 3D)
- **Structural damage** — overpressure exceeding a voxel's compressive strength
  destroys or fractures it. Cascading destruction via structural integrity
  flood-fill
- **Debris generation** — destroyed voxels spawn rigid body fragments with
  initial velocity from the pressure gradient. Fragment mass = material density ×
  voxel volume
- **Entity blast impulse** — pressure gradient → entity impulse (from the
  coupling layer above). Knockback, ragdoll launch, vehicle flipping
- **Thermal pulse** — detonation injects heat into surrounding voxels. Ignites
  flammable materials via existing combustion reactions

No hardcoded blast radius or damage tables. Destruction is emergent from
pressure magnitude vs. material strength.

### Projectile Ballistics (Future)
Extended force model for high-speed projectiles where aerodynamic effects matter.

- **Magnus force** — spinning projectiles experience lateral force from
  differential air pressure: F = S × (ω × v), where S is a shape-dependent
  coefficient. Enables arrow drift, curveball trajectories
- **Tumbling / angle of attack** — non-spherical projectiles (arrows, javelins)
  have orientation-dependent drag. Misaligned flight increases drag and induces
  torque toward broadside orientation
- **Wind interaction** — projectile drag computed against relative velocity
  (entity − wind field from LBM). Arrows drift in crosswinds
- **Impact model** — on collision, impulse magnitude determines penetration
  depth based on projectile KE vs. target material hardness. Shallow = ricochet,
  deep = embed

Uses existing `DragProfile`, `AngularVelocity`, and collision damage feedback.
No new physics engine — extends the force summation in `apply_forces`.

### Radiative Heat Transfer & Thermal Visualization (Phase 9a)
Supplement conductive (Fourier) heat diffusion with radiative transfer for
long-range thermal effects. Can be implemented before Phase 9 proper since
it only depends on the existing temperature field and material emissivity
(both already in place).

- **Stefan-Boltzmann emission** ✅ — hot surfaces emit thermal radiation at rate
  P = εσAT⁴. The constant σ is already defined in `constants.rs`; emissivity ε
  is already a field on every `MaterialData`. Implemented in `heat.rs` as
  `stefan_boltzmann_flux()`, `effective_emissivity()`, `net_radiative_flux()`.
- **View factor / ray-cast** ✅ — radiative flux between surfaces depends on
  line-of-sight and solid angle. A discrete 3D grid ray march
  (`src/world/raycast.rs`) casts 26 directions from each hot surface voxel.
  Opaque voxels block radiation. Semi-transparent materials (water, ice, steam)
  attenuate via Beer-Lambert law using `absorption_coefficient` ✅. View factor
  uses far-field approximation F ≈ A/(πd²), capped at 0.20 for close pairs.
- **Absorption** ✅ — receiving surfaces absorb radiation proportional to
  emissivity ε via the gray-body effective emissivity formula
  ε_eff = 1/(1/ε₁ + 1/ε₂ − 1). Reflected fraction = (1 − ε) is re-emitted
  diffusely (not yet modeled — deferred to Phase 9b)
- **Chunk-level integration** ✅ — `radiate_chunk()` in `heat.rs` returns
  temperature deltas for a flat `size³` voxel array. Called by `simulate_tick()`
  after conductive diffusion. Pair deduplication via HashSet ensures energy
  conservation. Emission threshold of 500 K limits computation to hot surfaces.
- **Simulation scenarios** ✅ — `radiation_across_air_gap.simulation.ron` and
  `radiation_blocked_by_wall.simulation.ron` validate long-range transfer and
  LOS occlusion
- **Black-body color (Planck's law)** — map voxel temperature to visible
  emission color via Planck spectral radiance. Iron at 1000 K glows dull red;
  at 1800 K bright orange-white. Rendered as emissive mesh color or glow
  post-process
- **Solar insolation** — (Phase 9 integration) sun angle × atmosphere
  transmittance × surface albedo → absorbed heat flux per exposed voxel face.
  Drives diurnal temperature cycling, biome differentiation
- **Use cases** — warming by campfire/lava at distance, forge/kiln radiation,
  solar heating, metal glow, thermal hazards for creatures

New `MaterialData` fields still needed: `albedo: Option<f32>` (0–1) for solar
reflection. `absorption_coefficient` ✅ added (water 100 m⁻¹, ice 50 m⁻¹,
steam 0.5 m⁻¹).

Priority: medium. Core radiation ✅, absorption coefficient ✅. Remaining:
albedo, Planck color, solar insolation.
Depends on: Phase 2 (materials ✅), Phase 3 (temperature field ✅).
Unlocks: Phase 9b (solar optics), thermal visualization.

### Chemistry Runtime Activation (Phase 9b)
Wire the existing simulation pipeline (`simulate_tick`) into the Bevy
`FixedUpdate` schedule so that heat transfer, chemical reactions, and material
state transitions run per-chunk during live gameplay — not just in headless
tests.

Currently, `ChemistryPlugin` only loads reaction data from RON files.
`simulate_tick()` in `src/simulation/mod.rs` integrates conduction, radiation,
reactions, state transitions, and pressure diffusion but is only called by the
test harness. This phase bridges that gap.

- **`ChunkSimulation` system** — new `FixedUpdate` system that iterates loaded
  chunks, calls `simulate_tick()` on each, and marks dirty chunks for remeshing.
  Needs mutable access to `Chunk` voxels (add `voxels_mut()` accessor)
- **Activity tracking** — maintain a `ChunkActivity` component or resource to
  skip simulation on thermally inert chunks (all voxels near ambient, no
  reactions possible). Only chunks containing voxels above a temperature
  threshold or adjacent to active reactions are ticked
- **Throttled execution** — run chemistry at a lower frequency than physics
  (e.g. every 0.5–1.0 s) via a cooldown timer. Full `simulate_tick` on a 32³
  chunk is ~32 K voxels × 6 neighbors — affordable at low frequency but too
  expensive at 60 Hz
- **Cross-chunk boundary** — initial implementation is intra-chunk only (heat
  and reactions don't cross chunk boundaries). Future work: boundary ghost layers
  copied from neighboring chunks before each tick
- **Dirty propagation** — if `TickResult.reactions_fired > 0` or
  `TickResult.transitions > 0`, mark the chunk dirty so the meshing system
  regenerates geometry. Temperature-only changes need a visual threshold (e.g.
  ΔT > 50 K from last mesh) to avoid excessive remeshing
- **Reaction & material loading** — ensure `ReactionData` assets and
  `MaterialRegistry` are available as Bevy resources before the simulation
  system runs. Gate with a `run_if` condition on resource existence

Priority: high. This is the prerequisite for all visual physics feedback.
Depends on: Phase 3 (temperature field ✅), Phase 9a (radiation ✅).
Unlocks: thermal glow, fire visualization, dynamic terrain (melting, freezing).

### Thermal Glow Rendering (Phase 9c)
Make the temperature field visible to the player. Hot voxels glow with
incandescent colors; Bevy's bloom post-process creates a halo effect around
heat sources.

- **Temperature-aware vertex colors** — extend the meshing `material_color`
  function to also accept voxel temperature. Above a glow threshold (~800 K),
  blend the base material color toward an incandescent ramp:
  - 800 K → faint dark red
  - 1200 K → cherry red
  - 1500 K → bright orange
  - 1800 K+ → yellow-white
  Reuse the `heatmap_rgb()` function from `src/diagnostics/visualization.rs`
  (blue→cyan→green→yellow→red) for a debug "thermal vision" toggle, and a
  separate physically-motivated incandescence ramp for normal rendering
- **HDR emissive encoding** — for bloom to work, hot vertex colors must exceed
  1.0 in HDR. Encode emissive intensity as a multiplier on the color channels:
  `color * (1.0 + emissive_factor)` where `emissive_factor` scales with T⁴.
  Alternatively, pack emissive strength in the vertex alpha channel and use a
  custom `Material` impl that reads alpha as emissive weight
- **Bloom post-process** — add `Bloom` component to the camera entity
  (`bevy::core_pipeline::bloom::Bloom`). Tune `intensity`, `threshold`, and
  `composite_mode` so only genuinely hot surfaces trigger bloom (not the sun
  or bright terrain). This is a one-line addition to camera spawn
- **Debug thermal overlay** — bind a key (T) to toggle between normal rendering
  and full thermal-vision mode (all voxels colored by temperature). Useful for
  debugging heat propagation in-game
- **Chunk remesh on temperature change** — tie into the dirty system from
  Phase 9b. Only regenerate mesh when a voxel's temperature crosses a visual
  threshold (e.g. enters or leaves the 800 K+ glow band)

Priority: high. Transforms invisible physics into dramatic visual feedback.
Depends on: Phase 9b (chemistry runtime — need live temperature changes).
Unlocks: fire looks like fire, lava glows, forges radiate visible heat.

### Time-of-Day & Dynamic Lighting (Phase 9d)
Rotate the sun (`DirectionalLight`) through a day-night cycle. Adjusts light
color, intensity, and ambient brightness. Foundation for Phase 9 solar heating.

- **`TimeOfDay` resource** — `f32` in hours (0.0–24.0), advanced each frame by
  `dt × time_scale`. Default cycle: 20 real minutes = 1 game day
  (`time_scale ≈ 72`). Configurable via a `DayNightConfig` RON asset
- **Sun position** — derive `DirectionalLight` rotation from `TimeOfDay`:
  azimuth rotates 360° over 24 h, elevation follows a sinusoidal arc (sunrise
  at 6:00, zenith at 12:00, sunset at 18:00). Below horizon → disable direct
  light
- **Color temperature shift** — dawn/dusk: warm orange (~3500 K color temp →
  Bevy color (1.0, 0.7, 0.4)). Noon: neutral white (~6500 K → (1.0, 1.0,
  0.95)). Night: cool blue moonlight (~10000 K → (0.3, 0.35, 0.5) at very low
  intensity)
- **Ambient light** — scales with sun elevation. Peak brightness at noon,
  minimum at midnight. A small ambient floor (~5% of daytime) prevents total
  blackness at night
- **Shadow updates** — `DirectionalLight` shadow direction follows sun rotation.
  Shadow quality can be reduced at low sun angles (long shadows) for performance
- **Phase 9 solar heating prep** — expose the computed surface insolation factor
  (sun elevation × time-of-day) as a resource for future use by the solar
  heating system. No thermal effect yet, just the geometric calculation

Priority: medium. High visual impact, relatively low effort. Independent of
chemistry runtime — can be implemented in any order relative to 9b/9c.
Depends on: nothing (lighting system already exists).
Unlocks: Phase 9 atmosphere (solar heating), visual atmosphere (sky color),
diurnal gameplay cycles.

### Fluid–Terrain Interaction (Future)
Bridge the AMR fluid simulation with the voxel terrain grid so that liquid
visibly fills, drains, and reshapes the world.

- **Fluid → voxel conversion** — when AMR fluid accumulates ≥ 1 m³ in a cell
  with sufficient dwell time, convert it to a Water (or Lava) terrain voxel.
  Enables flooding, pool formation, lava flows solidifying
- **Voxel → fluid conversion** — when a liquid terrain voxel loses structural
  support or is heated past boiling, convert it back to AMR fluid particles for
  dynamic flow
- **Erosion** — flowing fluid exerts shear stress on adjacent solid voxels
  (τ = μ × dv/dy). When cumulative stress exceeds material cohesion, the voxel
  is destroyed and becomes sediment (FLIP particles). Enables river carving,
  waterfall erosion, wave action on coastlines
- **Weathering** — slow degradation of exposed surfaces from temperature cycling
  (freeze-thaw: water in cracks expands on freezing), rain impact accumulation,
  and wind abrasion (particle impacts from FLIP/PIC). Modeled as a durability
  counter per exposed voxel face
- **Sediment transport & deposition** — eroded particles carried by fluid flow
  (FLIP advection); deposit when flow velocity drops below settling threshold.
  Enables delta formation, silt accumulation, alluvial fans

Design constraint: erosion rates derived from fluid velocity, material hardness,
and cohesion — all from `MaterialData`. No per-material erosion-rate constants.

### Vehicle Physics (Future)
Rigid body entities with wheel constraints and drive systems. Low priority until
gameplay requires rideable mounts or machines.

- **Wheel model** — each wheel is a constraint (hinge joint at axle +
  spring-damper for suspension). Rolling resistance from `PhysicsMaterial`
  friction × normal force
- **Drive torque** — engine/motor applies torque to drive axle joints. Torque
  curve defined in vehicle data (RON)
- **Steering** — front axle hinge limits change with player input. Ackermann
  geometry for multi-axle vehicles
- **Suspension** — spring-damper constraints between chassis and wheel mounts.
  Stiffness and damping from vehicle data
- **Buoyancy for boats** — displaced volume from hull shape (convex hull or
  voxel scan) × fluid density → buoyancy force. Stability from metacentric
  height

Depends on: constraints/joints system, collision shapes beyond AABB.

### Acoustics (Future)
Sound propagation as a physics system rather than purely a rendering/audio
concern. Low priority — relevant when stealth or environmental audio becomes a
gameplay mechanic.

- **Sound pressure waves** — point-source events (explosions, footsteps, speech)
  emit into the pressure field or a dedicated acoustic grid. Propagation speed =
  343 m/s in air (temperature-dependent: c = √(γRT/M))
- **Obstruction & occlusion** — ray cast from source to listener; solid voxels
  attenuate by material density and thickness. Enables muffled sound through
  walls
- **Reflection & reverb** — ray-traced early reflections off nearby surfaces;
  reverb tail from room volume estimate (flood-fill air voxels). Cave echo,
  indoor dampening
- **Doppler effect** — frequency shift based on relative velocity between source
  and listener: f' = f × (c + v_listener) / (c + v_source)
- **AI hearing** — creatures sample the acoustic field at their position; loud
  events above a threshold trigger alert/investigate behaviors

### Optics & Light Phenomena (Phase 12)
Physically-based light transport through the voxel world, enabling glass optics,
underwater caustics, atmospheric color, and material-dependent visual effects.
Builds on the radiative transfer ray-cast infrastructure from Phase 9a.

**Tier 1 — Foundation & Sky (✅ complete):**

- **Material optical properties** — `refractive_index: Option<f32>`,
  `reflectivity: Option<f32>`, `absorption_rgb: Option<[f32; 3]>` on
  `MaterialData`. Updated 8 material files + new glass.material.ron
- **Rayleigh scattering** — `src/lighting/sky.rs`: wavelength-dependent 1/λ⁴
  scattering with RGB β_R coefficients, optical depth integration (16 view + 8
  light samples), Reinhard tone mapping. Produces blue sky, red/orange sunsets
- **Arbitrary DDA raymarcher** — `src/world/raycast.rs`: `dda_march_ray()`,
  `dda_march_ray_attenuated()` (per-channel RGB Beer-Lambert), surface normal
  estimation, shadow testing. Shared infrastructure for rendering + physics
- **Beer-Lambert RGB absorption** — transparent materials attenuate light per
  channel: I = I₀ × e^(−α × d). Water absorbs red (blue tint underwater),
  glass nearly neutral. `MaterialData::light_absorption_rgb()` fallback chain
- **Per-voxel sunlight** — `ChunkLightMap` component stores RGB transmittance
  per voxel. Column-based top-down propagation with Beer-Lambert for transparent
  media. Integrated into meshing pipeline via `apply_light_map()`
- **Speed of light constant** — `SPEED_OF_LIGHT = 299_792_458.0 m/s` in
  `constants.rs` + `universal_constants.ron`

**Tier 2 — Refraction & Reflection (planned):**

- **Refraction (Snell's law)** — light bends at material boundaries proportional
  to the ratio of refractive indices: n₁ sin θ₁ = n₂ sin θ₂. Enables lensing
  through glass blocks, underwater distortion, mirage effects from hot air
  (gradient in n due to temperature-dependent density)
- **Reflection (Fresnel equations)** — partial reflection at every interface;
  reflectance depends on angle and refractive index ratio. At glancing angles
  even water becomes mirror-like (total internal reflection above the critical
  angle)

**Tier 3 — Advanced Phenomena (planned):**

- **Mie scattering** — forward-peaked scattering by particles comparable to
  wavelength (water droplets, dust, ash). Produces halos around sun/moon, white
  clouds, fog glow. Coupled to LBM humidity/particulate density
- **Caustics** — focused light patterns from refraction through curved surfaces
  (underwater ripple patterns, light through glass bottles). Approximate via
  photon mapping or screen-space caustic estimation
- **Dispersion** — wavelength-dependent refractive index separates white light
  into spectral components (prisms, rainbows). Model via 3-channel (RGB)
  refraction with slightly different n per channel

Design constraint: all optical parameters derive from `MaterialData`.
No per-material shader hacks — a single physically-based light transport model
with material-driven parameters.

Priority: medium-high. Optics are central to visual quality and enable unique
gameplay (lens crafting, underwater exploration, light puzzles).
Depends on: Phase 9a (ray-cast infrastructure), Phase 9 (atmosphere, sun
angle), Phase 11 (glass material for structures).

### Electricity & Magnetism (Phase 13)
Full electromagnetic simulation using a simplified Maxwell's equations solver on
the voxel grid. Enables technology progression, electrical hazards, and magnetic
gameplay mechanics.

#### Electrostatics & Current Flow
- **Electrical conductivity** — per-material property `electrical_conductivity:
  Option<f32>` (S/m) in `MaterialData`. Iron = 1.0e7, copper = 5.96e7,
  water = 0.05, stone ≈ 0, air ≈ 0 (insulator). Determines which voxels
  conduct current
- **Resistance network** — connected conductive voxels form a circuit graph.
  Solve for current via Kirchhoff's laws (sparse linear system) or relaxation
  on the voxel grid. Current I = V / R where R = 1 / (σ × A / L)
- **Voltage sources** — batteries (stored charge), generators (mechanical →
  electrical via Faraday's law: EMF = −dΦ_B/dt), piezoelectric crystals
  (pressure → voltage)
- **Resistive heating** — I²R power dissipated as heat into the thermal field.
  Enables electric furnaces, heating elements, short-circuit fires, fuses that
  melt when overloaded

#### Magnetism
- **Magnetic permeability** — per-material `magnetic_permeability: Option<f32>`
  (H/m; vacuum = 4π×10⁻⁷, iron = 6.3×10⁻³). Determines magnetic response
- **Magnetic field** — per-voxel `B: Vec3` (Tesla). Permanent magnets from
  ferromagnetic materials, electromagnets from current-carrying coils
  (Biot-Savart or Ampère's law on the grid)
- **Lorentz force** — charged/magnetized entities experience F = q(v × B).
  Enables magnetic rail transport, compass needles, magnetic locks
- **Electromagnetic induction** — changing B through a conductive loop induces
  EMF (Faraday's law). Generator gameplay, inductive sensors

#### Electromagnetic Waves (simplified)
- **Wave propagation** — EM waves at speed c through the voxel grid (FDTD —
  Finite-Difference Time-Domain — on a coarsened grid for performance).
  Primarily for radio/signal propagation, not visual light (handled by Phase 12
  optics)
- **Absorption & shielding** — conductive materials absorb/reflect EM waves
  (skin depth δ = √(2/(ωμσ))). Faraday cage gameplay, signal blocking through
  metal walls

#### Lightning
- **Atmospheric charge separation** (Phase 9) → leader propagation along
  lowest-resistance voxel path → return stroke. Deposits massive current →
  resistive heating → fire ignition, sand → glass (fulgurite), tree splitting
- **Discharge probability** — builds with charge differential and humidity;
  tall/conductive structures attract strikes

New universal constants needed: `elementary_charge: f64 = 1.602_176_634e-19` C,
`vacuum_permittivity: f64 = 8.854_187_8128e-12` F/m,
`vacuum_permeability: f64 = 1.256_637_062_12e-6` H/m.

Priority: low. Only pursue when a technology/crafting tier requires wiring,
circuits, or electromagnetic machinery.
Depends on: Phase 9 (atmosphere for lightning), Phase 11 (structures for
circuits), Phase 9a (thermal coupling for resistive heating).

### Nuclear Physics & Radiation (Phase 14)
Radioactive decay, nuclear reactions, and ionizing/non-ionizing radiation
transport. Enables late-game content: nuclear materials, radiation hazards,
advanced energy sources.

#### Radioactive Decay
- **Decay modes** — extend `ReactionData` with optional decay fields:
  `decay_half_life: Option<f32>` (seconds), `radiation_type:
  Option<RadiationType>` (Alpha, Beta, Gamma, Neutron). Decay is probabilistic:
  P(decay per tick) = 1 − e^(−λ × dt) where λ = ln(2) / t½
- **Decay chains** — parent isotope decays to daughter product(s), which may
  themselves be radioactive. Model as linked reactions: Uranium → Thorium + α,
  Thorium → Radium + β, etc. Products field:
  `decay_products: Option<Vec<(String, f32)>>` (material name, probability)
- **Mass-energy equivalence** — decay energy Q = Δm × c². New constant:
  `speed_of_light: f64 = 299_792_458.0` m/s (shared with optics). Energy
  released per decay event deposited as heat and radiation

#### Radiation Types & Transport
- **Alpha particles** — heavy (4 amu), highly ionizing, very short range
  (~5 cm in air). Stopped by a single voxel of any solid. Modeled as entity
  spawns or absorbed within the source voxel
- **Beta particles** — electrons/positrons, moderate ionizing power, range ~1 m
  in air, stopped by a few cm of metal. Ray-cast from source; attenuate by
  material density and thickness
- **Gamma rays** — high-energy photons, low ionizing power per interaction but
  very penetrating. Exponential attenuation: I = I₀ × e^(−μ × d) where μ is
  the mass attenuation coefficient (m⁻¹) derived from material density and
  atomic number. Ray-cast through multiple voxels
- **Neutron radiation** — uncharged, penetrates most materials except
  hydrogen-rich ones (water, paraffin). Triggers secondary reactions (neutron
  activation, fission). Range: meters through air, attenuated by light elements
- **Non-ionizing radiation** — thermal infrared (already handled by Phase 9a
  radiative heat), visible light (Phase 12 optics), radio waves (Phase 13 EM).
  No additional system needed — these are subsumed by earlier phases

#### Radiation Effects
- **Ionizing dose** — per-entity cumulative dose in Gray (Gy = J/kg). Absorbed
  energy per unit mass from all incident radiation. Weighted by radiation type
  (quality factor Q: α=20, β=1, γ=1, neutron=5–20) to get equivalent dose in
  Sieverts (Sv)
- **Biological damage** — creatures accumulate dose over time. Threshold effects:
  nausea (1 Sv), radiation sickness (2–6 Sv), lethal (>6 Sv). Chronic low-dose
  effects: mutation chance, cancer probability (stochastic). Integrates with
  Phase 5 (biology/health system)
- **Material activation** — neutron bombardment converts stable materials to
  radioactive isotopes (neutron activation). Extends the reaction framework
- **Shielding** — attenuation by material: lead (high Z, excellent γ shield),
  water/concrete (excellent neutron moderator), any solid (α stopper).
  Effectiveness from `density`, `atomic_number: Option<u8>` (new MaterialData
  field), and thickness

#### Nuclear Reactions
- **Fission** — heavy nucleus splits when struck by neutron. Releases ~200 MeV
  per event (3.2×10⁻¹¹ J) plus 2–3 secondary neutrons → chain reaction.
  Criticality when neutron multiplication factor k ≥ 1. Modeled as cascading
  reactions in the chemistry system with neutron count tracking
- **Fusion** — light nuclei combine at extreme temperature (>10⁷ K). Releases
  energy per the binding energy curve. Only relevant in extreme scenarios
  (stellar simulation, late-game tech). Very low sub-priority
- **Criticality control** — geometry matters: sphere minimizes surface/volume
  ratio → lowest critical mass. The voxel grid naturally supports geometry-
  dependent criticality calculation (count fissile neighbors, track neutron
  economy)

New `MaterialData` fields: `atomic_number: Option<u8>`,
`mass_attenuation_coeff: Option<f32>` (m⁻¹),
`radioactive: Option<RadioactiveProfile>` (half_life, decay_mode, decay_energy).

New universal constants: `speed_of_light` (shared), `planck_constant: f64 =
6.626_070_15e-34` J·s, `boltzmann_constant: f64 = 1.380_649e-23` J/K
(shared with chemistry), `avogadro: f64 = 6.022_140_76e23` mol⁻¹.

Priority: very low. Nuclear physics is late-game content requiring most other
systems to be in place. Radiation transport reuses the ray-cast infrastructure
from Phase 9a/12.
Depends on: Phase 5 (biology for radiation damage), Phase 9a (radiative
transport), Phase 12 (ray-cast optics infrastructure), Phase 13 (EM field model
for neutron interactions).

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

## Milestone: Valley River Scene

A target demonstration scene that exercises terrain generation, fluid dynamics,
static prop scattering, and lighting together for the first time. The scene
shows a procedurally generated valley on the planetary surface with a river
flowing through it, rocky terrain with scattered boulders, stones, and pebbles,
lit by the day-night sun cycle with terrain self-shadowing.

**What the milestone proves:**
- Planetary terrain can produce recognizable landforms (not just noise hills)
- AMR fluid simulation produces visible, flowing water in-game
- The world feels inhabited and textured with scattered natural objects
- Lighting sells the scene with depth via shadows

### Gap 1 — Procedural Valley & Channel Carving *(in progress)*

**Problem:** Terrain is pure Perlin noise — no valleys, ridges, or river
channels. Noise alone cannot produce the directed, branching drainage patterns
that define real landscapes.

**Approach:** D8 flow-accumulation on a coarse heightmap grid, cached on the
terrain generator, with valley carving applied during `generate_chunk()`.

1. **Coarse heightmap sampling** — evaluate `sample_height()` on a 512×512
   grid at 8 m resolution covering a 4096×4096 m region centered at world
   origin. Cost: ~0.3 ms (262 K Perlin evaluations).
2. **Sink filling** — iterative priority-queue flooding (Planchon & Darboux
   2002) eliminates closed depressions where flow would pool artificially.
3. **D8 flow direction** — each cell flows to its steepest downhill neighbor
   (8 directions). Flat cells flow toward the nearest lower cell.
4. **Flow accumulation** — traverse cells in topological order (highest
   first). Each cell passes its accumulation (1 + upstream total) to its
   downstream neighbor. Cells with high accumulation become channels.
5. **Channel carving** — lower the heightmap along high-accumulation paths
   by depth proportional to `ln(flow_count)`. Width widens with
   `√(flow_count)`. Cross-section profile is configurable (V-shaped vs
   U-shaped via cosine/Gaussian falloff).
6. **Erosion texturing** — channel bed → SAND (id 6), valley walls →
   exposed STONE where slope exceeds threshold, valley floor → DIRT.

**Caching:** `FlowMap` stored in `OnceLock` on `TerrainGenerator`. Computed
once on first chunk generation. Fully deterministic from world seed.

**Configuration (`ErosionConfig`, embedded in `TerrainConfig`):**
- `flow_threshold`: 50 (min accumulation to carve)
- `depth_scale`: 3.0 (channel depth = `min(12 m, 3 × ln(flow))`)
- `width_scale`: 2.0 (valley width = `2 × √flow`)
- `valley_shape`: 0.3 (0 = V-shaped, 1 = U-shaped)
- `region_size`: 4096 m, `cell_size`: 8 m

**Integration point:** Inside `TerrainGenerator::generate_chunk()`, after
`sample_height()` returns the base height and before material assignment.
Channels carved below `sea_level` auto-fill with water via existing placement
logic.

**Location:** New `src/world/erosion.rs` module.

**Dependencies:** Terrain system (Phase 1).

**Scope:** Flat terrain mode only (Phase 1). Spherical mode support deferred
to a future pass (requires tangent-plane projection for flow routing).

### Gap 2 — AMR Fluid Visual Surface & Plugin Activation

**Problem:** `AmrFluidPlugin` exists but is not registered in `PhysicsPlugin`.
The AMR simulation runs but its free surface is never extracted into a
renderable mesh. Water voxels are static blocks with no flow.

**Approach:**

1. **Activate `AmrFluidPlugin`** — register it in `PhysicsPlugin::build()`
   alongside `LbmGasPlugin` and `FlipPicPlugin`.
2. **Seed river flow** — when a chunk contains carved river channel voxels
   (from Gap 1), initialize the corresponding `FluidGrid` cells as FLUID
   with a velocity vector pointing downstream (derived from the channel
   gradient).
3. **Free surface extraction** — new system that reads `FluidGrid` SURFACE
   cells and writes their positions into the chunk's voxel data as
   `MaterialId::WATER` with a fluid-fraction field (0.0–1.0) for partial
   fill. The meshing system already handles water material.
4. **Flow-aware vertex animation** (stretch goal) — pass the fluid velocity
   field to the mesh shader as a per-vertex attribute. Animate UV
   coordinates or vertex displacement along the flow direction to give
   visual motion without re-meshing every frame.
5. **Boundary conditions** — upstream chunk edges inject fluid at a
   configurable flow rate; downstream edges drain. River flow is sustained
   by continuous injection, not a finite volume that drains away.

**Location:** Extend `src/physics/amr_fluid/plugin.rs` for activation.
New `src/world/fluid_surface.rs` for extraction. Meshing changes in
`src/world/meshing.rs`.

**Dependencies:** AMR fluid (Phase 3), terrain carving (Gap 1), meshing
(Phase 1).

### Gap 3 — Static Prop System (`PropData` + Scattering) ✅

**Completed** in commit `5245945`. Implemented `PropData` struct with RON
loading, `PropRegistry`, biome `prop_spawns` tables, chunk decoration system
(`NeedsDecoration`/`ChunkProps`), and the `decorate_chunks()` scatter system
in `src/procgen/props.rs`. Six prop RON files created (boulder, rock, pebble,
cobble, log, stick). All four biomes updated with `prop_spawns` tables.

**Files:** `src/procgen/props.rs`, `src/data/mod.rs`, `assets/data/props/*.prop.ron`,
`assets/data/biomes/*.biome.ron`.

### Gap 4 — Terrain Shadow Casting ✅

**Completed** in commits `1e537de` + `4fef32f`. Implemented `SunDirection`
resource, `ShadowConfig` (angle threshold, cone samples, half-angle),
`compute_terrain_shadows()` with DDA ray-cast + golden-spiral cone sampling
for soft penumbra edges, and `update_terrain_shadows()` system with
`LastShadowAngles` caching. Added `shadow: Vec<f32>` field to `ChunkLightMap`
with `apply_light_map()` multiplying `sun_transmittance × shadow_factor`.
Performance fix: `update_chunk_light_maps` skips non-dirty chunks.

**Files:** `src/lighting/shadows.rs`, `src/lighting/light_map.rs`,
`src/lighting/mod.rs`.

### Gap 5 — Prop Spawn ECS Integration ✅

**Completed** as part of Gap 3 (commit `5245945`). The `decorate_chunks()`
system queries chunks with `NeedsDecoration` marker, calls
`plan_chunk_prop_spawns()`, spawns `Prop` entities with `Transform` and
collision shapes, and stores entity handles in `ChunkProps` for cleanup on
chunk despawn. Duplicate decoration prevented by component lifecycle.

### Implementation Order

```
Gap 3 (PropData) ──→ Gap 5 (Spawn ECS) ──────────────────── ✅ Done
Gap 4 (Terrain shadows) ─────────────────────────────────── ✅ Done
Gap 1 (Valley carving) ──→ Gap 2 (Water surface) ──→ Scene integration
         ↑ in progress           ↑ remaining
```

- **Gap 3 + 5** completed: prop data, scattering, and ECS spawning.
- **Gap 4** completed: terrain shadow casting with soft penumbra.
- **Gap 1** in progress: D8 flow accumulation + valley carving. Plan
  finalized, implementation ready.
- **Gap 2** remaining: AMR fluid activation + visual surface extraction.
  Requires Gap 1 (carved channels to seed fluid).
- **Scene integration** is the final step once all gaps are closed.

### Success Criteria

- A valley is visible on the planetary surface with a recognizable
  V/U-shaped cross section and sloped walls.
- Water flows through the valley channel with visible motion (not static
  blue blocks).
- At least 3 prop types (boulder, rock, pebble) are scattered on the
  terrain with density varying by slope and proximity to water.
- The valley walls cast shadows onto the valley floor that move with the
  sun cycle.
- The scene runs at interactive frame rates (≥30 FPS) with at least a
  4×4 chunk view distance.

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
