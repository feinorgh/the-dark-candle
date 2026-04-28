# Geodesic Hexagonal Terrain System — Design Document

> **Status:** Standalone Pipeline Implemented / Full Migration Planned  
> **Date:** 2026-03-30 (updated)  
> **Context:** Exploration of replacing the flat Cartesian voxel grid with a planet-scale geodesic hexagonal system to support realistic weather simulation. The standalone generation pipeline (Sections 3–8) is now **implemented and tested** in `src/planet/`. Full integration with the Cartesian voxel game (Section 9: Migration Phases 0–5) remains future work.

---

## Implementation Status

The standalone planetary generation pipeline is **complete** in `src/planet/`,
runnable via `cargo run --bin worldgen`. It implements sections 3.3, 8.1–8.10 of
this design document as a 7-phase pipeline:

| Phase | Design Section | Module | Description |
|-------|---------------|--------|-------------|
| 1 | §3.3 | `grid.rs` | Icosahedral geodesic grid with configurable subdivision level |
| 2 | §8.2–8.8 | `tectonics.rs` | Plate tectonic simulation (weighted BFS seeding, power-law size distribution, dynamic-boundary subduction, orogenesis, erosion, geological time calibration) |
| 3 | §8.9 | `impacts.rs` | Astronomical impact events (minor/major/catastrophic/giant impacts) |
| 4 | §8.10 | `celestial.rs` | Celestial system generation (star, moons, rings, Keplerian orbits, tidal forces) |
| 5 | — | `biomes.rs`, `geology.rs` | Climate/biome classification + geological layering and ore deposits |
| 6 | — | `render.rs` | Interactive 3D globe viewer (Bevy app with orbital camera, colour modes) |
| 7 | — | `projections.rs` | 2D map projection export (equirectangular, Mollweide, orthographic) + animation |

**Key metrics:**
- ~4,700 LOC across 8 files (including `mod.rs`)
- 120+ unit tests, all passing
- Level 4 (2,562 cells) generates in ~11 ms; level 7 (163,842 cells) in ~2 s
- Full CLI: `cargo run --bin worldgen -- --seed 42 --level 4 --stats --globe`
- Tectonic modes: `--tectonic-mode <quick|normal|extended>`, geological age: `--tectonic-age <Gyr>`

**What is NOT yet implemented** (Section 9: Migration Plan):
- CellGrid trait abstraction (Phase 0)
- Hex-prism chunk storage replacing cubic chunks (Phase 2)
- Physics migration to hex lattice (Phase 3)
- Hex-prism meshing (Phase 4)
- Weather on geodesic grid (Phase 5)

These remain the long-term plan for integrating the geodesic system into the
live game. The standalone pipeline proves the generation algorithms work and
provides visual validation tooling for when the migration begins.

---

## 1. Motivation

The Dark Candle aims for realistic weather phenomena — global wind circulation, pressure systems, fronts, Coriolis-driven storms, seasonal variation. The current flat Cartesian grid works well locally but breaks down at planetary scale:

- No natural Coriolis effect (requires fake correction terms)
- Latitude/longitude have no structural representation
- No horizon curvature, no planetary wrapping
- Anisotropic neighbor distances (diagonal vs cardinal) introduce artifacts in fluid simulation

Climate science has converged on **geodesic hexagonal grids** (icosahedral subdivision with Voronoi dual mesh) as the optimal geometry for atmospheric simulation. This document explores adopting that geometry for the game.

---

## 2. Core Concept: Locally Flat, Globally Curved

The mental model is analogous to curved spacetime in general relativity:

- **Locally**, each chunk is a flat Euclidean space. Physics, collision, pathfinding, and meshing operate on a regular grid within a chunk with no curvature corrections needed.
- **Globally**, chunks are positioned on the surface of a sphere. Each chunk's local "up" points radially outward from the planet center. Adjacent chunks' coordinate frames are slightly rotated relative to each other.

Coordinates are expressed as **(longitude, latitude, elevation)** mapping to **(θ, φ, r)** in spherical terms.

---

## 3. Grid Geometry Options Evaluated

### 3.1 Cartesian Cubes (current system)

**Pros:**
- Trivial O(1) addressing: `voxels[z * SIZE² + y * SIZE + x]`
- Octree subdivision is natural (split each axis in half)
- GPU-friendly (3D textures, compute shaders)
- All existing code works

**Cons:**
- Cannot represent planetary curvature
- Anisotropic neighbor distances (√2 diagonal vs 1.0 cardinal)
- Weather simulation requires artificial correction terms
- No Coriolis without explicit latitude encoding

### 3.2 Cubed Sphere

Six cube faces projected onto a sphere. Each face is a regular 2D grid, with radial stacking for elevation.

**Pros:**
- Preserves regular grid addressing within each face
- Quadtree/octree subdivision works per-face
- Used by NASA (GEOS-5) and NOAA (FV3/GFS) weather models
- ~30% of spatial code needs modification (vs current)

**Cons:**
- ~15% cell area distortion at face corners — requires Jacobian correction in physics
- 6 face seams create numerical noise in advection (weather fronts crossing seams develop artifacts)
- Correction terms add computational cost every physics tick

### 3.3 Geodesic Hexagonal (Icosahedral Subdivision)

Start with a 20-triangle icosahedron, recursively subdivide, take Voronoi dual to get hexagonal cells (plus 12 pentagons at original vertices).

**Pros:**
- Near-uniform cell areas across the entire sphere — no distortion corrections
- 6 equidistant neighbors per cell — isotropic diffusion, LBM, and advection
- Used by the most modern operational weather models (DWD's ICON, NCAR's MPAS)
- Eliminates correction terms — physics code becomes *simpler*
- Natural multi-resolution via subdivision levels

**Cons:**
- ~70% of spatial code needs rewriting
- 12 pentagonal cells (minor nuisance, well-handled in literature)
- Non-trivial GPU buffer layout (no native 3D texture mapping)
- Hex prism meshing is more complex than quad meshing

### 3.4 Recommendation

**Geodesic hexagonal** is the recommended long-term target, primarily because:
1. Weather simulation — the core motivating feature — works naturally without corrections
2. Physics (LBM, heat, pressure) becomes simpler and more accurate
3. The front-loaded implementation cost is offset by simpler, faster simulation code

---

## 4. Computational Cost Analysis

### 4.1 Small-Scale (Gameplay: ~256m radius around player)

| System | Cartesian Cube | Cubed Sphere | Geodesic Hex |
|--------|---------------|--------------|--------------|
| Voxel lookup | O(1) array index | O(1) + face check | O(1) hex index + radial |
| Collision | AABB sweep, trivial | AABB + metric correction | Hex cell test, ~1.3× cost |
| Pathfinding | 8 neighbors/node | 8 neighbors + correction | 6 neighbors/node (fewer) |
| Meshing | Greedy quads, fast | Same + edge vertex warp | Hex prism mesher, ~2× cost |
| Heat diffusion | 6 neighbors | 6 + metric weights | 6 uniform neighbors (cleaner) |
| LBM gas | D3Q19 (19 dirs) | D3Q19 + corrections | Hex lattice (~13 dirs, 30% fewer) |
| Lighting | Column propagation | Same per-face | Hex column, similar cost |

**Local verdict:** Roughly equivalent. Geodesic hex trades slightly more expensive meshing for cheaper physics. Net difference is negligible at gameplay scale.

### 4.2 Large-Scale (Weather, Seasons: Planetary)

| System | Cubed Sphere | Geodesic Hex |
|--------|-------------|--------------|
| Global wind circulation | Works with correction terms per cell | Native — Coriolis emerges from geometry |
| Pressure gradients | ~15% error at face corners, Jacobian corrections | Uniform cell areas, exact gradients |
| Advection (weather fronts) | Seam artifacts at face boundaries | Isotropic, no artifacts |
| Memory for planet | 6 × N² × depth cells | ~same total cells |
| GPU parallelism | 6 Texture3D (one per face), manageable | Custom buffer layout required |
| LOD transitions | Per-face quadtree | Icosahedral subdivision (natural multi-res) |

### 4.3 Estimated Operations per Weather Tick

For a planet with ~10 million surface cells, simulating at 1-minute timesteps:

| Operation | Cubed Sphere | Geodesic Hex |
|-----------|-------------|--------------|
| LBM step (19 vs ~13 directions) | ~190M memory ops | ~130M memory ops |
| Geometric correction terms | ~30M multiplies | 0 |
| Seam/boundary handling | ~6×N edge cells | 12 pentagonal cells |
| **Total per step** | **~220M ops** | **~130M ops** |

**Planetary verdict:** Geodesic hex is ~40% cheaper per weather tick due to fewer LBM directions and zero geometric corrections.

---

## 5. Hybrid Architecture: Fibonacci Lattice + Geodesic Cells

An initial brainstorming idea was to use a **Fibonacci Spherical Lattice** for chunk loading decisions. The refined hybrid approach:

- **Fibonacci lattice** for LOD / streaming decisions: which regions need chunks loaded? Provides perfectly uniform spacing for view-distance checks.
- **Geodesic hexagonal grid** for actual voxel storage: regular addressing, uniform cell areas, physics-friendly.

The lattice points serve as "cameras" that drive which geodesic cells to load, while the geodesic grid provides the spatial structure for simulation.

---

## 6. Impact on Existing Codebase

### 6.1 Systems Audit

| Module | ~LOC | Cartesian Assumption | Migration Impact |
|--------|------|---------------------|------------------|
| `src/world/chunk.rs` | 390 | Cubic 32³ chunks, flat `z*SIZE²+y*SIZE+x` indexing | Full rewrite → hex-prism columns |
| `src/world/chunk_manager.rs` | 700 | Radius-based loading on XZ plane | Rewrite addressing; loading logic adapts |
| `src/world/meshing.rs` | 1,100 | Surface Nets: 8 corners, 12 edges per cell | Full rewrite → hex prism mesher |
| `src/physics/gravity.rs` | 350 | Supports both flat -Y and radial modes | Minor adaptation (radial mode already exists) |
| `src/physics/collision.rs` | 460 | AABB vs axis-aligned voxel grid | Rewrite → hex cell collision |
| `src/physics/lbm_gas/` | 5,066 | D3Q19 lattice, Cartesian streaming | Full rewrite → hex lattice topology |
| `src/chemistry/heat.rs` | 400 | 6-neighbor Cartesian conduction | Simplifies (uniform neighbor distances) |
| `src/lighting/` | 4,155 | Flat voxel indexing, vertical sun, ray-march | Column logic adapts; ray-march rewrite |
| `src/weather/` | 1,308 | Column-based accumulation, grid mapping | Adapts to hex columns |
| `src/behavior/pathfinding.rs` | 450 | 8-neighbor A*, Manhattan heuristic | Adapts to 6-neighbor hex A* (simpler) |

**Total directly affected:** ~14,379 LOC  
**Estimated rewrite:** ~70% of spatial code (~10,000 LOC)  
**Estimated adaptation:** ~30% needs only minor changes (~4,000 LOC)

### 6.2 Systems NOT Affected

These modules are geometry-agnostic and survive unchanged:

- Biology (metabolism, health, growth) — operates on component data, no spatial queries
- Behavior (needs, utility AI, action scoring) — works with abstract targets, not grid geometry
- Social (relationships, reputation) — pure entity-to-entity logic
- Data loading (RON assets, registries) — file I/O, no geometry
- Diagnostics, persistence, UI — no spatial coupling

---

## 7. Planetary Shell Model

### 7.1 Concept

The planet surface is modelled as a **spherical shell** — a variable-depth crust over a hot inner spheroid. The inner spheroid is not an invisible wall but a **thermal/material boundary**: voxels at the core interface are extreme-temperature rock or magma. The physics system handles the rest — heat transfer damages entities, molten rock cannot be mined, toxic gases fill cavities near the boundary.

- **Crust depth** varies geographically: zero at mid-ocean ridges and volcanic vents (exposed magma), a few hundred meters under ocean basins, several kilometers under mountain ranges.
- **Core boundary** sits at radius `R_core`. Voxels at this radius are generated as mantle material (ultramafic rock at ~1500–3000 K). There is no hard collision wall — the environment itself is the barrier.
- **Elevation** is measured relative to a reference "sea level" radius `R_sea`, with crust extending both above (mountains) and below (ocean floor, caves).

### 7.2 Zero-Depth Crust and the Core Boundary

At divergent plate boundaries, volcanic calderas, and mantle plumes, crust depth approaches or reaches zero. In these locations:

- **Surface voxels sit directly on core material** — lava lakes, exposed magma pools, fumarole fields
- **Digging is unnecessary** — the core is already visible/accessible
- **Heat radiation** from core-adjacent voxels creates a natural hazard gradient: survivable at distance, lethal up close
- **Volcanic gases** (SO₂, CO₂) fill the atmosphere near zero-crust zones, requiring protection or avoidance

This creates a spectrum of gameplay environments:

| Crust Depth | Location | Player Experience |
|-------------|----------|-------------------|
| 0 m | Mid-ocean ridge, active volcano | Lava exposed at surface. Extreme heat. Rare minerals visible but dangerous to collect. |
| 10–100 m | Volcanic zone, rift valley | Thin rock layer. Mining quickly reaches extreme temperatures. Hot springs, geysers. |
| 300–1000 m | Ocean floor, plains | Moderate depth. Standard mining progression through sedimentary → igneous layers. |
| 2–8 km | Mountain range, continental interior | Deep mining expeditions. Rich geological variety. Temperature gradient is gradual and manageable. |

### 7.3 Implications for Storage and Persistence

**Memory is bounded:** Unlike an infinite-depth flat world, total vertical extent per column is capped by `max_crust_depth`. On a level-10 geodesic grid (~40M surface cells) with an average 2 km crust, total voxels are enormous but manageable:

- Only columns near the player are loaded (same streaming model as current chunks)
- Unmodified columns are regenerated from the world seed on demand
- Modified columns are persisted as delta patches (only changed voxels saved)
- Octree compression within columns gives ~10:1 compression for typical geology (large homogeneous rock layers)

**Persistence strategy:**

```
World save = {
    world_seed,
    tectonic_output (heightmap + geological metadata),
    modified_columns: HashMap<CellId, ColumnDelta>,
    entity_data (creatures, items, player)
}
```

Regeneration from seed is deterministic: `seed + tectonic_data + noise → identical terrain`. Only player modifications need explicit storage.

### 7.4 Crust Depth as Geological Output

Rather than a fixed crust depth, each hex column's depth is determined by the tectonic generation system:

| Geological Context | Typical Crust Depth | Rock Composition |
|-------------------|--------------------|--------------------|
| Mid-ocean ridge (divergent) | 0 m | Exposed magma / pillow basalt, extreme heat |
| Active volcanic vent | 0–50 m | Thin basalt cap over magma chamber |
| Ocean basin | 0.3–1 km | Basalt, thin sediment |
| Rift valley (divergent) | 0.1–2 km | Fractured basalt, exposed mantle rock |
| Continental shelf | 1–3 km | Sedimentary, limestone |
| Continental interior | 2–4 km | Granite, gneiss, sedimentary layers |
| Volcanic zone (subduction) | 2–5 km | Basalt, andesite, volcanic ash layers |
| Mountain range (convergent) | 4–8 km | Folded sedimentary, metamorphic, granite core |

This directly governs what the player encounters when digging: mountain columns have thick, varied geology with metamorphic layers; ocean floor is thin basalt before hitting the impenetrable core.

---

## 8. Tectonic World Generation

### 8.1 Overview

Plate tectonics runs as a **one-time simulation during world creation** (not at runtime). It produces the large-scale continental structure that noise-based terrain generation then adds local detail to.

The tectonic simulator operates on the coarse geodesic grid (level 7–8, i.e. ~160K–2.6M cells), producing:

- Base heightmap (continental-scale elevation)
- Crust depth per cell
- Geological metadata (rock age, type, mineralization)
- Plate boundary locations and types
- Fault lines and volcanic zones

### 8.2 Generation Pipeline

```
World Seed
    │
    ▼
┌─────────────────────────────┐
│  1. Plate Initialisation    │  Seed → 8–15 plates via weighted BFS flood-fill
│     (~160K cells)           │  Growth weights drawn from Pareto distribution (α=1.3)
│                             │  Each plate: velocity vector, type (continental/oceanic)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. Tectonic Simulation     │  Configurable resolution (Quick/Normal/Extended)
│     (iterative)             │  ~3 Gyr of geological history with physical plate velocities
│                             │  Movement → boundary detection → accumulation → erosion
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. Geological Layering     │  Assign rock types per depth based on tectonic history
│                             │  Place ore/mineral deposits at geologically plausible sites
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. Fine Terrain Generation │  Tectonic heightmap → base elevation per column
│     (per loaded column)     │  Simplex noise adds local detail (hills, cliffs, ravines)
│                             │  Biome assignment from elevation + latitude + moisture
└─────────────────────────────┘
```

### 8.3 Plate Initialisation Detail

Plate centres are seeded using a Fibonacci sphere lattice for near-uniform coverage, then cells are
assigned via **weighted BFS flood-fill**. Each plate's growth weight is sampled from a
**Pareto power-law distribution** (shape α = 1.3, max weight ratio capped at 15×). This naturally
produces a small number of very large plates and many small ones — mirroring Earth's observed
plate size distribution. Larger plates are biased toward oceanic crust (analogous to Earth's
Pacific Plate), while smaller plates skew continental.

### 8.4 Tectonic Simulation Detail

The model uses a **dynamic-boundary** approach: plate assignments are not fixed after
initialisation. Each step, subduction can consume cells at convergent boundaries, shifting
plate extents over time.

Each simulation step:

1. **Recompute boundary normals:** Because subduction shifts plate boundaries each step, normals
   are recomputed at the start of every iteration rather than cached.

2. **Detect boundaries:** For each cell, check if any neighbor belongs to a different plate. Classify the boundary:
   - **Convergent:** Plates moving toward each other → mountain building (orogenesis)
   - **Divergent:** Plates moving apart → rift valleys, new ocean floor
   - **Transform:** Plates sliding past each other → fault lines

3. **Accumulate/erode at boundaries:**
   - Convergent continental–continental: Both sides gain elevation (fold mountains). Crust thickens.
   - Convergent oceanic–continental: Oceanic plate subducts. Volcanic arc on continental side. Ocean trench on oceanic side.
   - Convergent oceanic–oceanic: One subducts. Island arc volcanism.
   - Divergent: Lower elevation at rift. Thin crust. Volcanic activity.
   - Transform: No elevation change. Fault line marked. Earthquake potential stored.

4. **Subduction boundary deformation:** Oceanic cells at convergent boundaries have a ~3% per step
   probability of being consumed by the overriding continental plate. This advances the subduction
   front over time, producing irregular boundaries and further varying plate sizes. Plates that
   fall below 1% of total surface cells are protected from further consumption.

5. **Global erosion pass:** Smooth heightmap slightly each step. River-like erosion from high to low elevation. Sediment accumulation in basins.

6. **Thermal evolution:** Track geological age per cell. Older crust is cooler, thicker, more stable. Younger crust is thinner, more volcanically active.

### 8.5 Geological Time Calibration

The tectonic simulation is calibrated to real-world geological timescales and operates over approximately **3 billion years** of planetary history. All plate velocities and geological rates use strict **SI units** to ensure physically plausible outcomes.

#### 8.5.1 Physical Plate Velocities

Each tectonic plate moves with an **angular velocity** (rad/year) calibrated to Earth-like surface speeds:

- **Surface speed range:** 2–10 cm/year (0.02–0.10 m/year)
- **Angular velocity:** ω = v / R, where R is planetary radius (6,371,000 m by default)
- **Constants:** `MIN_PLATE_SPEED_M_YR = 0.02`, `MAX_PLATE_SPEED_M_YR = 0.10`

This matches observed terrestrial plate velocities (e.g., Pacific Plate ~10 cm/yr, Eurasian Plate ~2 cm/yr). Plate speeds are **not** arbitrary game constants — they emerge from the SI unit system and produce realistic continental drift rates over geological time.

#### 8.5.2 Velocity Evolution Model

Plate velocities are **not static**. Each plate has an `angular_acceleration` (rad/yr²) that causes its velocity to drift over time, modeling the natural variation in plate motion driven by mantle convection changes, slab pull variations, and ridge push fluctuations.

Each simulation step:

1. **Update velocity:** ω_new = ω_old + α × dt_yr
2. **Clamp to physical range:** Ensure ω ∈ [ω_min, ω_max]
3. **Bounce at bounds:** If velocity hits min/max, the acceleration component along the velocity direction is reversed (elastic reflection), causing the plate to decelerate and eventually reverse direction naturally
4. **Perturbation:** Every ~50 Myr, the acceleration vector is randomly perturbed (blended 60/40 with a new random direction) to simulate changes in mantle dynamics

This produces realistic plate behavior: periods of steady motion, gradual slowdowns, directional changes, and occasional reversals — all emergent from the acceleration model rather than hardcoded.

#### 8.5.3 Configurable Time Resolution

The simulation offers **three time resolution modes**, trading speed for detail:

| Mode | dt (Myr/step) | Steps (3 Gyr) | Wall Time | Use Case |
|------|---------------|---------------|-----------|----------|
| **Quick** | 60 | 50 | ~0.2 s | Rapid iteration, testing, procedural generation |
| **Normal** | 15 | 200 | ~0.8 s | Default for gameplay (balanced detail/speed) |
| **Extended** | 5 | 600 | ~2.4 s | High-fidelity worlds, showcase generation |

Players can configure this via `PlanetConfig`:
- `tectonic_mode: TectonicMode` (Quick / Normal / Extended)
- `tectonic_age_gyr: f64` (default 3.0 Gyr)

The number of simulation steps is derived: `steps = (age_gyr × 1000 / dt_myr).round()`.

#### 8.5.4 Time-Scaled Geological Rates

All geological processes scale with the time step to ensure **resolution independence** — the final terrain should look similar regardless of which mode is chosen, just with varying levels of fine detail.

**Reference time step:** dt_ref = 10 Myr. All rates are defined "per million years" and scaled by `dt_scale = dt_myr / 10`.

| Process | Base Rate | Scaling | Physical Unit |
|---------|-----------|---------|---------------|
| **Orogenesis** (continental collision) | 8 m/Myr | × dt_myr | meters per step |
| **Trench formation** (subduction) | 10 m/Myr | × dt_myr | meters per step |
| **Volcanic arc growth** | 6 m/Myr | × dt_myr | meters per step |
| **Rift valley subsidence** | 5 m/Myr | × dt_myr | meters per step |
| **Erosion** (diffusion) | 2% neighbor blend | ^dt_scale | exponential decay |
| **Volcanic heat gain** | gain × dt_scale | — | heat accumulation |
| **Subduction probability** | base_prob × dt_scale | — | chance per step |

**Example:** In Quick mode (dt = 60 Myr), orogenic uplift = 8 × 60 = 480 m/step. In Extended mode (dt = 5 Myr), it's 8 × 5 = 40 m/step. Over 3 Gyr, both produce similar total uplift (~24 km), but Extended mode captures finer variation.

This approach ensures that changing resolution affects computational cost and detail granularity, but **not** the fundamental geological outcomes. The same seed produces recognizably similar continents across all three modes.

#### 8.5.5 Geological Age and Real-World Context

The default **3 Gyr** simulation span corresponds to:

- **3.0 Ga** (billion years ago): Late Archean Eon on Earth — first stable continents forming
- **Present day** (0 Ga): End of simulation

This covers the era of modern-style plate tectonics. Earlier planetary history (Hadean bombardment, magma ocean cooling) is handled separately via initial conditions and the astronomical impact system (§8.8).

The simulation does **not** model:
- Pre-tectonic planetary differentiation (core-mantle separation)
- Mantle plume initiation (assumed as an initial condition)
- Detailed seismic/earthquake mechanics (stored as potential, not simulated)

But it **does** produce:
- Realistic continental assembly/breakup cycles (Wilson cycles)
- Mountain ranges comparable to Himalayas/Andes in height and age
- Ocean basins with age-depth relationships matching mid-ocean ridges
- Cratons (ancient stable continental cores) that resist deformation over billions of years

### 8.6 Geological Layering

After tectonic simulation, each column gets a vertical geological profile:

```
Depth 0m (surface):     Topsoil / sediment / ocean water
Depth 0–50m:            Sedimentary rock (sandstone, limestone, shale)
Depth 50–500m:          Older sedimentary or metamorphic (slate, marble)
Depth 500m–2km:         Igneous basement (granite continental, basalt oceanic)
Depth 2km+:             Deep crust (gneiss, granulite) → core boundary
```

The exact layer thicknesses and compositions vary based on:
- **Tectonic history:** Convergent zones have folded/thrust layers; rift zones have thin, fractured layers
- **Age:** Older crust has more metamorphic rock at depth
- **Volcanic activity:** Intrusions of igneous rock, ore veins along fault boundaries

### 8.7 Resource Placement

Mineral and ore deposits are placed geologically rather than randomly:

| Resource | Geological Context | Placement Rule |
|----------|-------------------|---------------|
| Iron ore | Ancient continental crust | Banded iron formations in old sedimentary layers |
| Gold | Convergent boundaries | Hydrothermal veins along fault lines |
| Copper | Subduction volcanic arcs | Porphyry deposits near volcanic intrusions |
| Coal | Continental basins | Thick sedimentary layers in low-elevation areas |
| Diamonds | Old continental interiors | Deep kimberlite pipes (rare, deep) |
| Gemstones | Metamorphic zones | Near convergent boundaries at medium depth |
| Oil/gas | Sedimentary basins | Trapped under impermeable cap rock layers |

This gives resource distribution a logical, discoverable pattern — players can learn that "gold is found near fault lines" and use geological reasoning to find deposits.

### 8.8 Computational Cost

The computational cost scales linearly with grid size and step count. Example for **Normal mode** (200 steps, dt = 15 Myr) on a level-8 geodesic grid (~2.6M cells):

| Operation | Per Step | Total (200 steps) |
|-----------|----------|-------------------|
| Plate movement (vector field) | ~2.6M vector ops | ~520M ops |
| Velocity evolution (acceleration) | ~10–15 plate updates | ~2K–3K ops |
| Boundary detection (neighbor check) | ~2.6M × 6 neighbors | ~3.1B ops |
| Height accumulation/erosion | ~2.6M adds | ~520M ops |
| Global erosion smoothing | ~2.6M × 6 neighbors | ~3.1B ops |
| **Total** | | **~7.2B ops** |

**Estimated wall time:**
- **Quick mode** (50 steps): ~0.8–2 seconds
- **Normal mode** (200 steps): ~3–10 seconds
- **Extended mode** (600 steps): ~9–30 seconds

All figures on a modern CPU (single-threaded). The simulation is trivially parallelizable across cells (each cell reads only neighbor data). GPU compute could reduce this to <1 second for all modes.

This fits comfortably in a "Generating world..." loading screen.

### 8.9 Astronomical Events During World Generation

In addition to plate tectonics, a planet's geological history is shaped by astronomical events. These can be simulated as **random punctuated events** injected at various points during the tectonic generation timeline, producing dramatic surface features that tectonics alone cannot explain.

#### 8.9.1 Event Types

**Meteorite Impacts**

Small to large impacts create craters, shatter crust, and redistribute material. They are the most common astronomical influence on surface geometry.

| Impact Scale | Crater Diameter | Geological Effect |
|-------------|----------------|-------------------|
| Minor | 1–10 km | Simple bowl crater. Local rock fracturing. Mineral exposure at depth. |
| Major | 10–100 km | Complex crater with central peak. Shattered crust zone. Shock-metamorphosed minerals (e.g., impact diamonds). Ejecta blanket modifies surrounding terrain. |
| Catastrophic | 100–500 km | Multi-ring basin. Regional crust thinning (crust depth → near zero at center). Massive ejecta reshapes terrain for hundreds of km. Triggers volcanic activity in weakened crust. Can redirect tectonic plate motion. |

Generation algorithm per impact:
1. Pick random surface cell + random time in tectonic timeline
2. Apply radial height displacement: excavation at center, rim uplift, ejecta decay with distance
3. Set crust depth to near-zero at crater center (exposes deep/hot rock)
4. Mark crater zone with unique geological metadata (shocked quartz, impact melt, breccia)
5. If catastrophic: perturb nearby plate velocities, trigger volcanic zone at antipodal point

**Planetary Body Collisions (Giant Impacts)**

Extremely rare (0 or 1 per world). A collision with a large body during early planetary formation produces world-defining features:

- **Hemispheric dichotomy:** One hemisphere significantly lower than the other (like Mars' crustal dichotomy). Generated by removing crust across a hemisphere-scale region and lowering the heightmap.
- **Moon formation debris:** If the world has a moon, a giant impact scar could be the in-lore explanation. The impact basin becomes a vast, ancient, partially-filled lowland.
- **Magma ocean remnants:** The impact zone has anomalous geological composition — more mafic rock, thinner crust, higher residual heat even billions of years later.

This is applied *before* tectonic simulation begins, so plate tectonics subsequently reshapes the impact scar over geological time.

**Solar Events**

Solar activity doesn't directly alter surface geometry but influences atmospheric and surface chemistry during generation:

- **Intense early solar wind:** Strips atmosphere from unprotected hemispheres. Affects which regions retain volatile deposits (water ice, organics) in the geological record.
- **Solar flares / coronal mass ejections:** During generation, periodic intense radiation events can be modelled as surface oxidation layers — ancient "rust bands" visible in cliff faces as distinctive red/orange strata.
- **Stellar luminosity evolution:** The star's brightness changes over geological time. Early dim periods favor ice deposits at lower latitudes; later bright periods push ice to poles. This governs where fossil ice deposits exist underground.

#### 8.9.2 Integration with Tectonic Timeline

Astronomical events are injected into the tectonic simulation as discrete interrupts:

```
Tectonic Step 0–10:    [Giant impact, if any — before plates stabilize]
Tectonic Step 10–50:   [Heavy bombardment period — frequent minor/major impacts]
Tectonic Step 50–150:  [Declining bombardment — rare major impacts]
Tectonic Step 150–200: [Quiet period — occasional minor impacts only]
```

The frequency and severity are controlled by world seed parameters:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `bombardment_intensity` | 0.0–1.0 | How many impacts during generation. 0 = pristine, 1 = heavily cratered. |
| `giant_impact_chance` | 0.0–0.3 | Probability of a world-defining giant impact. |
| `solar_activity` | 0.0–1.0 | Intensity of solar weathering effects on surface chemistry. |

#### 8.9.3 Gameplay Consequences

Astronomical events during generation create unique exploration opportunities:

- **Impact craters** are natural points of interest — exposed deep minerals at the center, unusual rock types in the rim, flat crater floors suitable for building
- **Giant impact basins** create entire biome regions — vast lowlands with unique geology unlike anything tectonics produces
- **Ancient bombardment zones** have higher mineral density (impacts fracture rock and expose veins) but unstable terrain (shattered crust, cave-ins)
- **Solar weathering strata** are visible in cliff faces, giving the world a sense of deep history
- **Impact-weakened crust** near old craters has thinner depth to the core — volcanic activity and hot springs are more likely nearby

#### 8.9.4 Computational Cost

Astronomical events add negligible cost to generation:

| Operation | Cost |
|-----------|------|
| Impact crater stamp (per impact) | ~10K–100K cell updates (radial displacement + metadata) |
| Giant impact (if any) | ~1M cell updates (hemisphere-scale) |
| Solar weathering pass | ~2.6M cell reads + conditional writes |
| **Total for ~50 impacts + 1 giant + solar** | **~10M additional ops (<1 second)** |

### 8.10 Celestial System Generation

The world generator doesn't just produce a planet surface — it generates an entire **celestial neighbourhood** from the world seed. Moons, rings, a star, asteroid belts, and background stars are all procedurally determined at world creation time, then simulated at runtime using Keplerian orbital mechanics. These aren't cosmetic — they drive tides, eclipses, night illumination, and sky appearance.

#### 8.10.1 Star (Sun) Generation

Each world orbits a procedurally generated star:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `star_mass` | 0.6–1.8 M☉ | Determines habitable zone distance, luminosity, colour |
| `star_temperature` | 3500–9000 K | Surface colour (red dwarf → blue-white). Drives sky scattering colour. |
| `star_luminosity` | Derived from mass | Determines solar constant at planet distance (W/m²). Affects climate baseline. |
| `star_age` | 1–10 Gyr | Older = more stable. Young = more flare activity. |

The star's spectral characteristics directly feed the existing sky scattering and sun cycle systems — a cooler, redder star produces different sunset colours, different shadow qualities, and a shifted photosynthesis spectrum (affecting biome colours).

#### 8.10.2 Moon Generation

The world seed determines 0–4 moons, each with orbital and physical parameters:

| Parameter | Range | Gameplay Effect |
|-----------|-------|----------------|
| `count` | 0–4 | 0 = no tides, dark nights. 4 = complex tidal patterns, bright nights. |
| `mass` | 0.001–0.02 M_planet | Tidal force strength. Larger moon = stronger tides. |
| `orbital_radius` | 20–80 R_planet | Close = large in sky, fast orbit, strong tides. Far = small, slow, weak tides. |
| `orbital_period` | Derived from radius+mass | Determines lunar cycle length (tidal period, phase cycle). |
| `eccentricity` | 0.0–0.15 | Elliptical orbit → varying apparent size, spring/neap tide variation. |
| `inclination` | 0°–15° | Tilted orbit affects eclipse frequency and latitude visibility. |
| `albedo` | 0.05–0.5 | Surface reflectivity. High = bright nights. Low = dim, reddish moon. |
| `surface_colour` | Derived from composition | Grey (rocky), reddish (iron-rich), white (icy). Visible at close range. |

**Formation history** (flavour, affects surface features visible in sky):
- **Capture moon:** Irregular shape, dark surface, inclined orbit
- **Impact-origin moon** (from giant impact in §8.8): Large, round, close orbit, shares composition with planet crust
- **Co-accretion moon:** Regular orbit, similar composition to planet

Each moon is rendered as a textured sphere in the skybox, with correct phase (illuminated fraction based on star-moon-planet angle), apparent size (angular diameter from orbital distance), and position (from orbital mechanics).

#### 8.10.3 Planetary Rings

Rings are generated with probability ~0.15 per world (rare but dramatic):

| Parameter | Range | Effect |
|-----------|-------|--------|
| `inner_radius` | 1.2–1.8 R_planet | Where ring begins |
| `outer_radius` | 1.8–3.5 R_planet | Where ring ends |
| `opacity` | 0.1–0.9 | Dense rings cast visible shadows on the surface |
| `composition` | Ice, rock, mixed | Colour and brightness of the ring |
| `inclination` | Matches planet axial tilt | Ring plane visible at angle from surface |

**Formation:** Rings are typically generated as a consequence of a moon being inside the Roche limit, or from a catastrophic impact event (§8.8). The world generation log records the origin.

**Gameplay effects:**
- **Ring shadow bands:** Dense rings cast shadow stripes across the planet surface as the planet rotates. These are periodic bands of reduced sunlight — affecting temperature, crop growth, and creature behavior in shadowed latitudes.
- **Night illumination:** Rings reflect starlight, providing diffuse illumination on the night side. Worlds with bright rings have lighter nights.
- **Sky spectacle:** From the surface, rings appear as a luminous arc across the sky, changing angle with latitude. Near the ring plane (equatorial latitudes), they appear edge-on (thin line). At high latitudes, they spread across the sky.

#### 8.10.4 Emergent Astronomical Phenomena

These aren't separately generated — they **emerge** from the orbital mechanics of the generated celestial bodies:

**Tides**

Tidal force from each moon:

```
F_tidal ∝ M_moon / d³
```

Total tide = sum of all moon contributions + star contribution (minor). The tide system modulates sea level at coastal cells:

- **High/low tide cycle:** Period = half the lunar orbital period per moon
- **Spring tides:** When two moons align (or moon + star align) — maximum tidal range
- **Neap tides:** When moons are at 90° — minimum range
- **Tidal range:** From centimeters (small distant moon) to several meters (large close moon)

Tidal data feeds into: coastal flooding, beach exposure (resource gathering at low tide), harbour accessibility, tidal bore events in river mouths.

**Eclipses**

Solar eclipses occur when a moon passes between star and planet. Frequency depends on orbital inclination and moon size:

- **Total solar eclipse:** Moon's angular diameter ≥ star's angular diameter. Dramatic darkness, temperature drop, creature panic behavior, corona visible.
- **Partial solar eclipse:** Partial coverage. Dimming, temperature dip.
- **Annular eclipse:** Moon too far (small apparent size) to fully cover star. Ring of fire effect.
- **Lunar eclipse:** Planet shadow falls on moon. Moon turns reddish (light filtered through atmosphere). Affects night illumination.

Eclipse prediction is deterministic from orbital parameters — events can be forecast in-game (astronomy skill/tool).

**Meteor Showers (Falling Stars)**

Generated from two sources:
- **Asteroid belt debris:** If the system has an asteroid belt, periodic meteor showers occur when the planet's orbit intersects belt debris streams. Predictable, seasonal.
- **Random interplanetary debris:** Sporadic meteors at low frequency.

Visual: bright streaks across the night sky. Rare large ones impact the surface (small crater events from §8.8, runtime variant).

**Night Sky Illumination**

Total night-side brightness is the sum of:

| Source | Contribution | Variability |
|--------|-------------|-------------|
| Moon(s) reflected light | Primary (0–100% depending on phase × albedo) | Lunar cycle |
| Ring-reflected light | Significant if rings are bright/dense | Constant per latitude |
| Star field (background stars) | Baseline dim glow | Constant |
| Zodiacal light | Faint band along ecliptic | Seasonal |
| Aurora (if magnetosphere) | Polar regions, during solar events | Sporadic |

A world with no moons and no rings has genuinely dark nights (dangerous, reduced visibility). A world with a large bright moon and dense rings could have nights almost as navigable as a cloudy day.

**Starry Sky**

A procedural star field generated from the world seed:
- ~2000–5000 visible stars distributed on a celestial sphere
- Brightness follows a realistic luminosity function (many dim, few bright)
- Colour variation: blue-white (hot), yellow (sun-like), orange-red (cool)
- Constellation patterns are unique per world — discoverable by players
- Planets of the system visible as bright "wandering stars" that move against the background over game-weeks
- Host-galaxy band: a brighter strip across the sky (the disk of the
  procedurally generated galaxy the system inhabits — orientation is
  randomised per system, not anchored to our real Milky Way)

The star field is a static skybox texture generated once. Planet/moon positions are updated each frame from orbital mechanics (cheap — just Kepler's equation).

#### 8.10.5 Orbital Mechanics at Runtime

All celestial body positions are computed analytically from Keplerian orbits — no N-body simulation needed:

```
For each body:
    M = mean_anomaly_at_epoch + orbital_speed × game_time
    E = solve_kepler(M, eccentricity)     // iterative, ~5 iterations
    true_anomaly = atan2(...)
    position = orbital_elements_to_cartesian(a, e, i, Ω, ω, true_anomaly)
```

**Cost per frame:** ~50–100 floating-point operations per celestial body. For 1 star + 4 moons + ring data: **< 500 ops/frame**. Negligible.

Tidal force computation per coastal cell is similarly cheap — a few multiplies per moon per affected cell, computed once per game-hour (not per frame).

#### 8.10.6 Generation Parameters

Added to world seed:

| Parameter | Range | Default |
|-----------|-------|---------|
| `moon_count` | 0–4 | Random (weighted: 1 most common) |
| `ring_chance` | 0.0–1.0 | 0.15 |
| `star_type` | "random", "solar", "red_dwarf", "blue" | "random" |
| `asteroid_belt` | true/false | Random (0.3 probability) |
| `meteor_shower_frequency` | 0.0–1.0 | 0.5 |

### 8.11 Runtime Geological and Astronomical Events (Future Possibility)

While tectonic movement and historical impacts are generation-only, the geological and astronomical metadata enables runtime events:

- **Earthquakes:** Fault lines have stored stress. Accumulate over game-time. Release as localized terrain deformation + shaking.
- **Volcanic eruptions:** Volcanic zones have dormancy timers. Eruptions spawn lava source voxels, ash particles, and pyroclastic flow.
- **Erosion:** Runtime water flow slowly erodes terrain (already partially implemented in current erosion system).
- **Landslides:** Steep terrain + rainfall triggers slope failure.
- **Meteor showers:** Small runtime impacts as rare world events. Use the same crater-stamp algorithm at small scale. Could drop rare materials.
- **Solar storms:** Periodic aurora events affecting weather patterns, electromagnetic interference (compass/navigation disruption), and surface radiation in exposed areas.
- **Eclipses:** Emergent from orbital mechanics. Total solar eclipses cause temperature drops, creature behavior shifts, and dramatic visual events.
- **Tidal flooding:** Coastal terrain periodically submerged during spring tides, especially when multiple moons align.

These are all local events using existing voxel manipulation — the tectonic and astronomical data tells them *where* and *how likely* to happen.

### 8.12 Tectonic Time-Lapse Visualization

**Goal:** Visualize plate motion, mountain building, and erosion across geological time.

The tectonic simulation can now capture **snapshots** at regular intervals and play them back interactively in the globe viewer as a time-lapse animation. This allows visual validation of tectonic processes and provides an intuitive understanding of how the world evolved.

**Usage:**

```bash
# Normal mode (default)
cargo run --bin worldgen -- --globe --timelapse

# Extended mode over 4.5 Gyr for maximum detail
cargo run --bin worldgen -- --tectonic-mode extended --tectonic-age 4.5 --globe --timelapse
```

When the `--timelapse` flag is combined with `--globe`, the simulation records intermediate states during the tectonic phase and enters **playback mode** instead of showing the final result statically.

**How Snapshots Are Captured:**

- `run_tectonics_with_history()` (in `tectonics.rs`) runs the standard simulation loop but periodically calls `capture_snapshot()` to record:
  - Per-cell elevation, plate ID, boundary type, crust type, volcanic activity
- Snapshots are stored in a `TectonicHistory` struct with metadata:
  - `dt_myr`: time step in millions of years
  - `total_steps`: total simulation steps
  - `snapshot_interval`: how often snapshots were taken
- Target: ≤ 100 keyframes regardless of simulation length (longer simulations skip more steps between snapshots)
- Memory footprint: ~2.5 MB per snapshot at level 7 (163,842 cells × 15 bytes/cell)

**Playback Controls:**

| Key | Action |
|-----|--------|
| **Space** | Play / Pause animation |
| **Right Arrow** | Step forward one frame |
| **Left Arrow** | Step backward one frame |
| **> (Period)** | Double playback speed (max 32×) |
| **< (Comma)** | Halve playback speed (min 0.25×) |
| **Home** | Jump to first frame (planet formation) |
| **End** | Jump to last frame (final state) |
| **1–8** | Switch colour mode (elevation, biome, plates, etc.) |
| **+/−** | Adjust vertical exaggeration |
| **F12** | Take screenshot |

**Playback Systems (render.rs):**

- `PlaybackState` resource tracks `current_frame`, `playing` (bool), `speed` (0.25× to 32×), and a `Timer` for frame advance
- `playback_controls` system handles keyboard input
- `advance_playback` system uses a `Timer` to automatically step frames when playing (auto-stops at end)
- `apply_snapshot` system overlays snapshot data onto `PlanetData` and triggers mesh rebuild via `TriggerRebuild` event

**Design Notes:**

- Existing `run_tectonics()` is unchanged — time-lapse is an optional extension via `run_tectonics_with_history()`
- Snapshots store only visualization-relevant data (elevation, plate ID, boundary type, crust type, volcanic flag) — not full internal simulation state
- Playback modifies `PlanetData` in-place but does not re-run physics — it's purely a visual overlay
- All existing globe viewer controls (rotation, zoom, colour modes, screenshots) work during playback
- Useful for debugging tectonic algorithms, presenting world generation to players, and educational demonstrations

---

## 9. Phased Migration Plan

### Phase -1: Tectonic World Generator

**Goal:** Implement plate tectonic simulation as a standalone world generation step.

- Define plate initialisation from world seed (weighted BFS flood-fill, Pareto power-law size distribution)
- Dynamic-boundary tectonic simulation: boundary detection, height accumulation, subduction deformation, erosion
- Output: heightmap + crust depth + geological metadata + fault/volcanic zone maps
- Geological layering: assign rock types and mineral deposits per column
- Can be developed and tested independently of any grid migration — operates on abstract cell positions

**Impact:** New module (~3,000–4,000 LOC). Does not touch existing gameplay code.
**Risk:** Low. Pure generation-time simulation. Testable with visualization output.
**Prerequisite for:** Phase 2 (hex chunk storage needs tectonic heightmap as input).

### Phase 0: Abstraction Layer

**Goal:** Decouple all physics/chemistry/pathfinding from grid geometry without changing behavior.

Introduce a `CellGrid` trait:

```rust
pub trait CellGrid {
    type CellId: Copy + Eq + Hash;

    fn get(&self, id: Self::CellId) -> Option<&VoxelData>;
    fn get_mut(&mut self, id: Self::CellId) -> Option<&mut VoxelData>;
    fn neighbors(&self, id: Self::CellId) -> SmallVec<[Self::CellId; 6]>;
    fn distance(&self, a: Self::CellId, b: Self::CellId) -> f32;
    fn cell_volume(&self, id: Self::CellId) -> f32;
    fn cell_area(&self, id: Self::CellId, face: usize) -> f32;
}
```

Current Cartesian chunks implement this trivially (CellId = `(usize, usize, usize)`, distance = Euclidean, volume = 1.0 m³). All physics systems are refactored to use the trait instead of direct array indexing.

**Impact:** ~14,000 LOC touched (mechanical refactor), zero behavior change.  
**Risk:** Low. Fully testable against existing test suite.  
**Value:** Enables side-by-side testing of both backends.

### Phase 1: Geodesic Addressing

**Goal:** Implement the spherical cell index as a standalone module.

- Icosahedral subdivision to configurable resolution
- Voronoi dual mesh generation (hex cells + 12 pentagons)
- Precomputed neighbor lookup tables
- `(lat, lon)` ↔ `CellId` conversion
- `CellId` → world-space `(Vec3, Quaternion)` for chunk placement

**Impact:** New module (~2,000 LOC). Nothing else changes.  
**Risk:** Low. Pure math, extensively unit-testable.

### Phase 2: Hex Chunk Storage

**Goal:** Replace cubic chunks with hex-prism columns.

- Each surface hex cell owns a vertical column of voxels
- Column stored as flat array indexed by elevation layer
- `CellGrid` trait implemented for hex chunks
- Chunk manager adapted to load/unload hex cells by proximity on the sphere

**Impact:** `chunk.rs` rewrite (~400 LOC), `chunk_manager.rs` adaptation (~700 LOC).  
**Risk:** Medium. Affects world generation and entity spawning.

### Phase 3: Physics Migration

**Goal:** Adapt simulation systems to hex lattice.

- **LBM:** D3Q19 → hex-adapted lattice (6 horizontal hex neighbors + vertical). Fewer directions = fewer memory ops per step.
- **Heat diffusion:** 6-neighbor with uniform distances. Fourier's law simplifies (no diagonal correction needed).
- **Pressure:** Direct gradient computation on uniform cells.
- **Gravity:** Already supports radial mode — minimal change.
- **Collision:** Hex cell containment test replaces AABB sweep.

**Impact:** LBM rewrite (~5,000 LOC), heat.rs simplification (~400 LOC), collision.rs rewrite (~460 LOC).  
**Risk:** High. Core simulation changes. Must validate against physics test suite.

### Phase 4: Meshing + Rendering

**Goal:** Render hex-prism terrain.

- New hex-prism mesher: each cell renders as a hexagonal column
- Surface Nets adaptation for smooth terrain within hex cells
- LOD system using icosahedral subdivision levels
- Chunk Transform includes rotation to orient local Y along radial

**Impact:** `meshing.rs` rewrite (~1,100 LOC), new hex mesh generator.  
**Risk:** Medium. Visual validation required.

### Phase 5: Weather + Lighting

**Goal:** Full planetary weather on geodesic grid.

- Weather simulation operates directly on hex grid — no correction terms
- Coriolis effect emerges from cell positions on rotating sphere
- Lighting uses hex columns for sun propagation
- Orbital mechanics drive sun angle per cell latitude/longitude
- Seasonal temperature variation from axial tilt + orbital position

**Impact:** `lighting/` adaptation (~4,000 LOC), `weather/` adaptation (~1,300 LOC).  
**Risk:** Medium. Requires visual and physical validation.

---

## 10. Key Design Decisions (To Be Made)

1. **Planet radius and resolution:** What subdivision level? Level 8 icosahedral gives ~2.6M hex cells (~2.5 km spacing). Level 10 gives ~40M cells (~600m spacing). Higher = more memory, more accurate weather.

2. **Crust depth range:** Minimum is zero (exposed core at divergent boundaries and volcanic vents). Maximum governed by tectonic simulation output, suggested up to ~8 km under mountain ranges. Core boundary is thermal/material, not a collision wall.

3. **Local chunk size within hex cells:** Does each hex cell contain a single voxel column, or a cluster of voxels? A hex cell could contain a mini-grid of, say, 16×16 voxels for local detail while the hex topology handles global curvature.

4. **Dual-resolution approach:** Coarse hex grid for weather/climate (km-scale), fine voxel grid for gameplay (1m-scale) near the player. How do they interface?

5. **GPU strategy:** Custom compute shader buffer layouts for hex grids, or project hex data into regular textures per-face?

6. **Transition period:** Run both backends simultaneously during migration (CellGrid trait enables this) or hard cutover per phase?

7. **Tectonic detail level:** How many plates? How many simulation steps? More steps = more realistic continental shapes but longer generation time. Suggested: 10–12 plates, 100–200 steps.

8. **Geological event frequency:** How often do earthquakes/eruptions trigger at runtime? Pure flavor (rare, dramatic) vs gameplay mechanic (frequent, must be managed)?

---

## 11. References

- **MPAS (Model for Prediction Across Scales):** NCAR's geodesic hex atmospheric model. [https://mpas-dev.github.io/](https://mpas-dev.github.io/)
- **ICON:** DWD/MPI-M icosahedral weather model. Operational since 2015.
- **Fibonacci Sphere Lattice:** Álvaro González, "Measurement of Areas on a Sphere Using Fibonacci and Latitude–Longitude Lattices" (2010).
- **Cubed Sphere:** Ronchi, Iacono, Paolucci, "The Cubed Sphere" (1996). Used in GEOS-5, FV3.
- **Goldberg Polyhedra:** Generalized hex/pent tilings of the sphere.
- **D3Q19 LBM on hex lattices:** Various computational fluid dynamics literature on alternative lattice topologies.
- **Plate Tectonics Simulation:** Cortial et al., "Procedural Tectonic Planets" (2019). Real-time capable tectonic generation for games.

---

## 12. Summary

| Approach | Weather Accuracy | Implementation Cost | Runtime Cost (weather) | Runtime Cost (local) |
|----------|-----------------|--------------------|-----------------------|---------------------|
| Cartesian (current) | Poor at scale | Zero (done) | N/A (can't do planetary) | Baseline |
| Cubed Sphere | Good with corrections | ~30% spatial rewrite | ~220M ops/tick | ~same |
| **Geodesic Hex** | **Excellent, native** | **~70% spatial rewrite** | **~130M ops/tick** | **~same** |

The geodesic hexagonal grid is the recommended long-term architecture if realistic planetary weather is a core feature. The planetary shell model with tectonic world generation provides geologically plausible terrain, bounded vertical extent, and natural resource placement. The `CellGrid` abstraction layer (Phase 0) and tectonic generator (Phase -1) should be implemented first — both are independently valuable and enable incremental migration.
