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
- 90+ source files, ~26,000 lines of Rust (edition 2024)
- 761 passing tests
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

---

## Phase 8 — Spherical Planetary Terrain (planned)

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

## Phase 9 — Atmosphere Simulation (planned)

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

---

## Phase 10 — Buildings & Structural Construction (planned)

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
(Phase 10), the project also needs integration, polish, and gameplay. These are
not yet planned in detail — each will get a session plan when started.

### Coupling & Integration
- **Cross-model fluid coupling** — AMR ↔ LBM mass/heat exchange at liquid-gas
  interfaces; FLIP particles entering/leaving LBM gas fields
- **Plugin activation** — wire AmrFluidPlugin, LbmGasPlugin, FlipPicPlugin into
  PhysicsPlugin::build() for runtime use (currently test-only). Physics are also
  validated by the [simulation test framework](simulation-test-system.md).

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
