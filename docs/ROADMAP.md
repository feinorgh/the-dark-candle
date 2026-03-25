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

- **Stefan-Boltzmann emission** — hot surfaces emit thermal radiation at rate
  P = εσAT⁴. The constant σ is already defined in `constants.rs`; emissivity ε
  is already a field on every `MaterialData`
- **View factor / ray-cast** — radiative flux between surfaces depends on
  line-of-sight and solid angle. Sample via short ray casts between hot emitters
  and nearby voxels/entities. Opaque voxels block radiation; semi-transparent
  materials (glass, water) attenuate by `absorption_coefficient` (new field)
- **Absorption** — receiving surfaces absorb radiation proportional to
  emissivity ε. Reflected fraction = (1 − ε) is re-emitted diffusely
- **Black-body color (Planck's law)** — map voxel temperature to visible
  emission color via Planck spectral radiance. Iron at 1000 K glows dull red;
  at 1800 K bright orange-white. Rendered as emissive mesh color or glow
  post-process
- **Solar insolation** — (Phase 9 integration) sun angle × atmosphere
  transmittance × surface albedo → absorbed heat flux per exposed voxel face.
  Drives diurnal temperature cycling, biome differentiation
- **Use cases** — warming by campfire/lava at distance, forge/kiln radiation,
  solar heating, metal glow, thermal hazards for creatures

New `MaterialData` fields needed: `absorption_coefficient: Option<f32>` (m⁻¹),
`albedo: Option<f32>` (0–1).

Priority: medium. Enables fire/forge gameplay without adjacency.
Depends on: Phase 2 (materials ✅), Phase 3 (temperature field ✅).
Unlocks: Phase 9b (solar optics), thermal visualization.

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

### Optics & Light Phenomena (Phase 11)
Physically-based light transport through the voxel world, enabling glass optics,
underwater caustics, atmospheric color, and material-dependent visual effects.
Builds on the radiative transfer ray-cast infrastructure from Phase 9a.

- **Refraction (Snell's law)** — light bends at material boundaries proportional
  to the ratio of refractive indices: n₁ sin θ₁ = n₂ sin θ₂. New `MaterialData`
  field: `refractive_index: Option<f32>` (dimensionless; air ≈ 1.0003,
  water = 1.33, glass = 1.52, diamond = 2.42). Enables lensing through glass
  blocks, underwater distortion, mirage effects from hot air (gradient in n due
  to temperature-dependent density)
- **Reflection (Fresnel equations)** — partial reflection at every interface;
  reflectance depends on angle and refractive index ratio. At glancing angles
  even water becomes mirror-like (total internal reflection above the critical
  angle). New field: `reflectivity: Option<f32>` (0–1, for metals where Fresnel
  is insufficient)
- **Absorption & extinction (Beer-Lambert law)** — light intensity decays
  exponentially through a medium: I = I₀ × e^(−α × d), where α is the
  absorption coefficient (m⁻¹) and d is path length. Colored glass, murky water,
  fog, and smoke all derive from this. Transparent materials use
  `absorption_coefficient` per RGB channel for wavelength-dependent color
  filtering
- **Rayleigh scattering** — short wavelengths scatter more in atmosphere
  (∝ 1/λ⁴). Produces blue sky, red sunsets, purple twilight. Implemented as
  post-process sky-dome shader driven by sun angle and atmospheric density from
  Phase 9
- **Mie scattering** — forward-peaked scattering by particles comparable to
  wavelength (water droplets, dust, ash). Produces halos around sun/moon, white
  clouds, fog glow. Coupled to LBM humidity/particulate density
- **Caustics** — focused light patterns from refraction through curved surfaces
  (underwater ripple patterns, light through glass bottles). Approximate via
  photon mapping or screen-space caustic estimation
- **Shadows & light propagation** — per-voxel light level from sun + point
  sources, attenuated by opaque/translucent voxels. Extend the existing
  DirectionalLight with a voxel-aware shadow system (shadow maps or ray-traced
  voxel shadows)
- **Dispersion** — wavelength-dependent refractive index separates white light
  into spectral components (prisms, rainbows). Model via 3-channel (RGB)
  refraction with slightly different n per channel

New universal constants needed: `speed_of_light: f64 = 299_792_458.0` m/s
(already needed for nuclear physics).

Design constraint: all optical parameters derive from `MaterialData`.
No per-material shader hacks — a single physically-based light transport model
with material-driven parameters.

Priority: medium-high. Optics are central to visual quality and enable unique
gameplay (lens crafting, underwater exploration, light puzzles).
Depends on: Phase 9a (ray-cast infrastructure), Phase 9 (atmosphere, sun
angle), Phase 10 (glass material for structures).

### Electricity & Magnetism (Phase 12)
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
  Primarily for radio/signal propagation, not visual light (handled by Phase 11
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
Depends on: Phase 9 (atmosphere for lightning), Phase 10 (structures for
circuits), Phase 9a (thermal coupling for resistive heating).

### Nuclear Physics & Radiation (Phase 13)
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
  radiative heat), visible light (Phase 11 optics), radio waves (Phase 12 EM).
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
from Phase 9a/11.
Depends on: Phase 5 (biology for radiation damage), Phase 9a (radiative
transport), Phase 11 (ray-cast optics infrastructure), Phase 12 (EM field model
for neutron interactions).

### EM, Radiation & Optics — Phasing & Dependencies

These systems build on each other in a natural progression. The recommended
integration order follows the dependency chain:

```
Phase 2 (Materials ✅) ──→ Phase 9a: Radiative Heat Transfer
Phase 3 (Temperature ✅)─┘      │
                                ├──→ Phase 9b: Solar Optics (with Phase 9)
Phase 8 (Spherical Planet) ─────┘
Phase 9 (Atmosphere) ───────────┐
                                ├──→ Phase 11: Optics & Light Phenomena
Phase 10 (Buildings) ───────────┘
                                ├──→ Phase 12: Electricity & Magnetism
                                └──→ Phase 13: Nuclear Physics & Radiation
```

Material property extensions accumulate across phases:
- Phase 9a adds: `absorption_coefficient`, `albedo`
- Phase 11 adds: `refractive_index`, `reflectivity`, `transmissivity`
- Phase 12 adds: `electrical_conductivity`, `magnetic_permeability`
- Phase 13 adds: `atomic_number`, `mass_attenuation_coeff`, `radioactive`

Universal constants can be added to `universal_constants.ron` proactively:
`speed_of_light`, `planck_constant`, `boltzmann_constant`, `elementary_charge`,
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
