# 🕯️ The Dark Candle

**A data-driven voxel game with real-world physics, procedural planetary generation, and emergent chemistry.**

Built with [Bevy 0.18](https://bevyengine.org/) · Rust 2024 Edition · MIT / Apache-2.0

---

## World Generation

The Dark Candle generates entire planets from a single seed — geodesic grids,
tectonic plates with physically-driven motion (slab-pull, rifting, suturing,
volcanic hotspots), impact craters, insolation-based climate, biomes, geology,
and ore deposits — then renders them as interactive 3D globes or 2D map
projections with GPU-accelerated terrain detail.

<p align="center">
  <img src="docs/images/elevation_equirect.png" width="700" alt="Elevation map – equirectangular projection" />
  <br/>
  <em>Equirectangular elevation map with hillshading (seed 42, level 6 — 40 962 cells)</em>
</p>

<p align="center">
  <img src="docs/images/biome_equirect.png" width="700" alt="Biome map – equirectangular projection" />
  <br/>
  <em>Biome distribution — 14 climate types from tundra to tropical rainforest</em>
</p>

<p align="center">
  <img src="docs/images/elevation_mollweide.png" width="400" alt="Elevation – Mollweide projection" />
  <img src="docs/images/biome_mollweide.png" width="400" alt="Biome – Mollweide projection" />
  <br/>
  <em>Mollweide equal-area projections — elevation (left) and biome (right)</em>
</p>

<p align="center">
  <img src="docs/images/elevation_ortho.png" width="300" alt="Elevation – orthographic" />
  <img src="docs/images/biome_ortho.png" width="300" alt="Biome – orthographic" />
  <br/>
  <em>Orthographic views</em>
</p>

---

## Features

### 🌍 Planetary Generation
- **Geodesic grid** — icosahedral subdivision (configurable level 0–10, up to ~2.6 M cells)
- **Tectonic simulation** — power-law plate sizes, configurable geological time (Quick/Normal/Extended modes), physical plate velocities (2–10 cm/yr SI), subduction deformation, mountain building
- **Slab-pull force feedback** — physically-driven plate motion from subducting-slab density
- **Malleable plate deformation** — per-cell strain tracking, continental rifting (plate splitting), plate suturing (merging), and boundary-zone orogeny
- **Volcanic terrain shaping** — mantle plume hotspots with shield volcano, dome, and caldera morphologies
- **Sub-stepped plate advection** — forward-scatter transport (≤ 0.5 cell-diameter per sub-step) with same-plate filtering to prevent cross-plate contamination
- **Relief-dependent erosion** — tectonic erosion scales with local relief to preserve landmass
- **Tectonic time-lapse** — step-by-step playback of plate evolution with play/pause, speed control (0.25×–32×), and frame stepping
- **Impact events** — asteroid craters with ejecta blankets and central peaks
- **Celestial system** — procedural star, moons (0–4 with Keplerian orbits), rings, tidal mechanics, eclipse prediction, animated moon orbits with visible orbit trails
- **Climate model** — insolation-based energy balance (Berger 1978 daily integration, ice-albedo feedback, meridional heat transport), altitude lapse rates, ocean proximity
- **14 biome types** — Whittaker classification from temperature and precipitation
- **10 rock types & 7 ore types** — geological age, metamorphism, and hydrothermal deposits
- **GPU-accelerated rendering** — WGSL compute shaders for terrain projection (35× speedup at 4K)
- **Map projections** — equirectangular, Mollweide, orthographic with hillshading
- **Interactive 3D globe** — Bevy renderer with orbital camera, 9 colour modes (elevation, biome, plates, geological age, crust depth, tidal amplitude, rock, temperature, strain), and tectonic time-lapse playback

### 🧱 Voxel World
- **Cubed-sphere chunks** — 6-face cubed-sphere grid with 32³ chunks, greedy meshing, and local tangent orientation (Y = radial up)
- **Octree storage** — sparse voxel octree (SVO) for adaptive multi-resolution subdivision
- **41 material types** — loaded from RON data files: stone, water, iron, lava, wood, glass, plus geological materials (sandstone, limestone, granite, basalt) and ores (coal, copper, gold, quartz crystal), 8 construction materials (oak, pine, brick, concrete, wrought iron, bronze, dried clay, thatch), 3 plasters (white, red, green), and 3 emissive materials (IR emitter, cool/warm LED panels with spectral power distribution)
- **NoiseStack noise engine** — composable multi-octave FBM, ridged fractals, domain warping, terrain-type selector, micro-detail, continent/ocean masks
- **Procedural terrain** — noise-based heightmaps, valley/river carving, hydraulic erosion (droplet, grid, and combined modes)
- **Geological depth** — stratified rock layers (sedimentary/metamorphic/igneous) with depth-based ore veins and multi-scale cave systems (caverns, tunnels, worm tubes with underground lakes and lava)
- **Biome-terrain integration** — slope/altitude-aware surface materials, per-biome terrain modifiers (height bias, roughness, erosion rate)
- **Planetary terrain mode** — `--planet` runs the full geodesic pipeline (tectonics → biomes → geology) and drives voxel surface height, materials, and ore veins from `PlanetData`; `ChunkBiomeData` propagates temperature and precipitation to all procgen systems
- **Tree generation** — L-system-inspired procedural trees with bark, wood, twig, and leaf materials
- **8 scene presets** — alpine, archipelago, desert canyon, rolling plains, volcanic, tundra fjords, valley river, spherical planet
- **World creation UI** — in-game preset selector screen; bypassed by CLI `--scene` flag

### ⚗️ Physics & Chemistry
- **SI units throughout** — 1 voxel = 1 metre, real densities, conductivities, specific heats
- **Heat diffusion** — Fourier's law with thermal conductivity per material
- **Phase transitions** — melting, freezing, boiling, condensation with latent heat
- **Chemical reactions** — combustion, oxidation, thermite, oxyhydrogen — loaded from RON
- **Fire propagation** — temperature-driven ignition with fuel consumption and ash production
- **Radiation** — Stefan-Boltzmann thermal emission and incandescence

### 🌊 Fluid Dynamics
- **Three simulation models:**
  - Adaptive Mesh Refinement Navier-Stokes
  - Lattice Boltzmann Method (D3Q19)
  - FLIP/PIC hybrid particle-grid
- **Viscosity, surface tension, and buoyancy** from real material properties

### 🌤️ Atmosphere & Weather
- **GPU sky dome** — custom Bevy `Material` backed by a WGSL Rayleigh-scattering shader (`sky_dome.wgsl`): 16-sample view path + 8-sample light path per pixel, sun disk, correct twilight gradient from horizon to zenith; renders at clip-space infinity via reverse-Z trick and bypasses the depth prepass with `AlphaMode::Blend`
- **Moon billboards** — one unlit billboard quad spawned per moon, repositioned each frame from Keplerian orbital mechanics (`moon_position()`); scale matches true apparent angular size; hidden below horizon
- **Distance fog** — time-of-day color adaptation (warm dawn → blue noon → dark night) via `update_fog()` system
- **Day/night cycle** — orbital sun position with twilight transitions
- **Volumetric clouds** — 3D noise density with GPU ray marching
- **Weather particles** — rain, snow, and sand with wind advection from LBM field
- **Valley fog** — temperature-inversion fog accumulation in low terrain
- **Cloud shadows** — projected onto terrain from cloud layer

### 💡 Lighting & Optics (Phase 12 — all tiers complete)
- **Sunlight** — directional from orbital position with seasonal variation
- **Ambient occlusion** — voxel-space AO for chunk meshes
- **Thermal glow** — incandescent materials emit light based on temperature
- **Beer-Lambert RGB absorption** — per-channel light attenuation through transparent media (water tints blue, glass near-neutral)
- **Per-voxel sunlight** — `ChunkLightMap` top-down column propagation with material absorption
- **Refraction (Snell's law)** — light bends at refractive boundaries; n from `MaterialData.refractive_index`
- **Fresnel equations** — angle-dependent reflection/transmission at every interface; TIR above critical angle
- **Refractive DDA raymarcher** — `dda_march_ray_refractive` traces bending rays through n-boundaries with TIR bounces
- **Chromatic dispersion** — Cauchy equation n(λ) = A + B/λ²; per-channel (R/G/B) refraction separates white light into a spectrum; borosilicate glass B = 4.61 × 10⁻¹⁵ m², quartz B = 3.40 × 10⁻¹⁵ m²
- **Local Mie scattering** — forward-peaked voxel-scale scattering for steam (β = 50 m⁻¹, g = 0.85) and ash (β = 20 m⁻¹, g = 0.65); Henyey-Greenstein phase function
- **Caustics** — analytical Jacobian factor C = (n₂/n₁)² × (cos θ₁/cos θ₂); stratified photon-beam tracer; Gaussian KDE irradiance estimation

### 🦎 Creatures & AI
- **Biology system** — metabolism, body temperature, hydration, energy
- **Behaviour tree** — seek food, flee threats, idle, wander
- **Social system** — faction membership and relationship tracking
- **Pathfinding** — A* on voxel grid with movement cost per material
- **Food sources** — forageable resources with regrowth timers

### 🏗️ Buildings & Structural Construction
- **Data-driven parts** — 8 part types (block, slab, beam, column, wall, arch, stair, roof) defined in `.part.ron` files; material is a separate field so one part type works with any material
- **Joint stress model** — adjacent parts auto-connect with `Joint` entities; each joint tracks axial and shear stress vs real SI material strengths (Pa)
- **Structural stress analysis** — load-path propagation every 10 ticks; gravity loads accumulate from top-down, LBM wind pressure applies lateral forces, joints that exceed material strength break
- **Progressive collapse** — parts lose all live joints and become unsupported are despawned; debris fragments spawn with scatter velocities
- **Crafting system** — `RecipeData` RON assets define input materials, tool requirement, minimum temperature (K), and duration in ticks; `CraftingQueue` component tracks per-workstation progress
- **Player build mode** — press B to toggle; R rotates 90°; left-click places on 1 m grid with auto joint creation; placement validates support and inventory
- **41 materials** — 33 terrain/chemistry/emissive materials plus 8 construction-specific: oak, pine, brick, concrete, wrought iron, bronze, dried clay, thatch — all with real SI properties
- **Inventory system** — per-entity `Inventory` component with configurable weight (kg) and volume (m³) limits

### 🦎 Entity Bodies & Organic Physics
- **Articulated skeletons** — species-specific `.skeleton.ron` files define bone trees with parent→child hierarchy, rest-pose transforms, hinge/ball-socket/fixed joint constraints
- **Forward kinematics** — pose propagates from root bone down the tree each tick
- **FABRIK inverse kinematics** — `IkChain` solves foot/hand placement on uneven terrain in ≤10 iterations
- **Tissue layers** — `TissueLayer` (skin, muscle, bone, organ) with material-derived density, elasticity, and failure thresholds; compound AABB colliders built from bone extents
- **Locomotion gaits** — `.gait.ron` files define walk/run/sprint/trot/gallop cycle parameters; `LocomotionState` drives limb animation phases
- **Player embodiment** — player is a regular creature with `PlayerBody`; input maps to the same locomotion controller as AI creatures
- **Perception** — `EyeMount` (cone-frustum visibility test) and `EarMount` (range + material attenuation) fire `PerceptionEvent` when conditions are met
- **Per-region injury** — `InjuryRecord` tracks severity (Bruised → Fractured → Severed) per body region; wound effects propagate to locomotion and biology systems
- **Plant bodies** — `PlantBody` with `PlantJoint` spring model for wind sway; felling mechanic when trunk joint fails

- **Save/load** — 4 save slots (1 autosave + 3 manual) in RON format (SAVE_VERSION=4)
- **First-person camera** — WASD + mouse look with configurable sensitivity
- **HUD** — health, temperature, coordinates, diagnostics overlay
- **Hotbar** — material selection for placement
- **Interaction** — voxel placement and removal via raycast
- **In-game map** — M key overlay with local discovery map (biome-colored, fog-of-war, 4 zoom levels) and global planet map (equirectangular projection, lat/lon marker, zoom/pan)

### 🖥️ GPU Compute
- **Headless wgpu pipelines** — no window required for rendering
- **Atmosphere renderer** — sky dome via compute shader
- **Terrain projection** — two-pass elevation + hillshade compute shader
- **Particle system** — GPU-driven particle simulation

---

## Quick Start

### Requirements
- Rust 1.85+ (edition 2024)
- Linux: Wayland or X11 display server
- GPU: Vulkan-capable (for compute shaders and rendering)

### Build & Run

```bash
# Run the game
cargo run --features bevy/dynamic_linking

# Run with release optimisations
cargo run --release

# Run with a specific scene preset and seed
cargo run --release -- --scene alpine --seed 12345

# Tune terrain generation
cargo run --release -- --scene volcanic --terrain-detail 3 --height-scale 80.0 --caves dense --hydraulic-erosion moderate

# Run with planet-driven terrain (generates a geodesic planet first)
cargo run --release -- --planet --planet-seed 42 --planet-level 5
```

### Generate a Planet

```bash
# Generate a planet and print statistics
cargo run --bin worldgen -- --seed 42 --level 6 --stats

# Use a specific tectonic mode and geological age
cargo run --bin worldgen -- --seed 42 --level 6 --tectonic-mode extended --tectonic-age 4.5 --stats

# Export an elevation map
cargo run --release --bin worldgen -- \
  --seed 42 --level 6 \
  --projection equirect --colourmode elevation \
  --width 2048 --output world_elevation.png --gpu

# Export a biome map (Mollweide projection)
cargo run --release --bin worldgen -- \
  --seed 42 --level 6 \
  --projection mollweide --colourmode biome \
  --width 2048 --output world_biome.png --gpu

# Launch the interactive 3D globe viewer
cargo run --release --bin worldgen -- --seed 42 --level 6 --globe

# Launch the globe with tectonic time-lapse playback
cargo run --release --bin worldgen -- --seed 42 --level 6 --globe --timelapse

# Generate a rotating globe animation
cargo run --release --bin worldgen -- \
  --seed 42 --level 6 --animate --width 1024 --gpu
```

**Colour modes:** `elevation`, `biome`, `plates`, `age`, `crust_depth`, `tidal`, `rock`, `temperature`, `strain`

**Projections:** `equirect`, `mollweide`, `orthographic`

**Tectonic modes:** `quick` (50 steps, ~0.2s), `normal` (200 steps, ~0.8s), `extended` (600 steps, ~2.4s) at level 7

### Run Tests

```bash
cargo test --lib                 # 1640+ unit tests
cargo test --test simulations    # Physics simulation scenarios
cargo test --test validate_assets  # Asset loading validation
```

---

## Architecture

The codebase is organised into focused ECS modules:

| Module | Description |
|--------|-------------|
| `world/` | Cubed-sphere chunks (V2), greedy meshing, terrain generation (NoiseStack, biome integration, scene presets), erosion (D8 valley + hydraulic), raycasting, planetary sampling, octree storage |
| `physics/` | Rigid bodies, gravity, collision, LBM gas, FLIP fluid, atmosphere |
| `chemistry/` | Heat transfer, reactions, state transitions, radiation |
| `building/` | Structural construction: part/recipe RON assets, joint stress model, load-path analysis, player placement, demolition, crafting |
| `bodies/` | Articulated skeleton FK/IK, tissue compound colliders, FABRIK IK, locomotion gaits, player embodiment, injury system |
| `planet/` | Geodesic grid, tectonics (advection, deformation, rifting, suturing, slab-pull, volcanic hotspots), impacts, celestial, biomes, geology, rendering |
| `lighting/` | Sun cycle, GPU sky dome (`SkyMaterial` + `sky_dome.wgsl` Rayleigh shader), moon billboards, light maps, volumetric clouds, distance fog, optics (Snell's law, Fresnel, TIR), chromatic dispersion, local Mie scattering, caustics |
| `weather/` | Particle emitters, wind advection, snow/rain accumulation |
| `biology/` | Metabolism, body temperature, hydration, energy systems |
| `behavior/` | Behaviour trees, AI decision-making |
| `social/` | Factions, relationships |
| `entities/` | Creatures, items, `Inventory` component (weight/volume-limited item stacks) |
| `procgen/` | Tree generation, biome decoration |
| `gpu/` | Headless wgpu compute pipelines (atmosphere, terrain, particles) |
| `data/` | RON asset loading for materials, reactions, configs |
| `persistence/` | Save/load system (SAVE_VERSION=4) |
| `diagnostics/` | ECS dump, screenshots, video encoding, visualisation |
| `simulation/` | Headless tick-based simulation runner for tests |
| `camera/` | First-person camera controller |
| `map/` | In-game map overlay: local discovery map + global planet map |

**213 source files · ~84K lines of Rust · 1640+ tests**

### Data-Driven Design

Game data lives in `assets/data/` as RON files:
- **41 materials** — density, thermal conductivity, specific heat, hardness, viscosity, optical properties; 33 terrain/chemistry/emissive materials (including 8 geological, 4 ores, 3 plasters, and 3 emissive with spectral power distribution) + 8 construction materials (oak, pine, brick, concrete, wrought iron, bronze, dried clay, thatch); all with SI structural strength values (tensile, compressive, shear, flexural, fracture toughness)
- **8 chemical reactions** — reactants, products, activation energy, enthalpy
- **8 part types** — block, slab, beam, column, wall, arch, stair, roof (`.part.ron`)
- **5 crafting recipes** — wood_to_planks, clay_to_brick, sand_to_glass, iron_ore_to_ingot, mix_concrete (`.recipe.ron`)
- **1 tree species** — L-system parameters for procedural generation
- **Configs** — atmosphere, fluid, planet, subdivision, universal constants

All physical constants use SI units. No magic numbers — emergent behaviour
arises from the interaction of real material properties and fundamental forces.

---

## Documentation

Detailed design documents live in [`docs/`](docs/):

- [Architecture Overview](docs/architecture.md)
- [Terrain Generation](docs/terrain-generation.md)
- [Geodesic Terrain Design](docs/geodesic-terrain-design.md)
- [Spherical Terrain](docs/spherical-terrain.md)
- [Voxel Subdivision System](docs/voxel-subdivision-system.md)
- [Fluid Simulation](docs/fluid-simulation-system.md)
- [Atmosphere Simulation](docs/atmosphere-simulation.md)
- [Advanced Physics](docs/advanced-physics.md)
- [Electromagnetism](docs/electromagnetism.md)
- [Nuclear Physics](docs/nuclear-physics.md)
- [Entity Bodies](docs/entity-bodies.md)
- [Buildings & Structural Construction](docs/structural-construction.md)
- [Optics & Light](docs/optics-light.md)
- [Simulation Test System](docs/simulation-test-system.md)
- [Showcases](docs/SHOWCASES.md)
- [Debugging & Diagnostics](docs/debugging-and-diagnostics.md)
- [Roadmap](docs/ROADMAP.md)

---

## Project Status

The Dark Candle is in active early development. The planetary generation
pipeline is functional and produces visually compelling worlds. The voxel engine,
physics, chemistry, entity body, and structural construction systems are tested
and working at the simulation level.

Current completed phases (1–12):
- Terrain detail & world generation (8 scene presets, geological depth, hydraulic erosion) ✅
- Physics: LBM gas, FLIP/PIC fluid, AMR fluid coupling, atmosphere ✅
- Atmosphere & weather: orbital sun, sky scattering, volumetric clouds, rain/snow ✅
- Lighting & optics: sunlight, fog, cloud shadows, thermal glow, Snell/Fresnel refraction, chromatic dispersion, local Mie scattering, caustics ✅
- Entity bodies: articulated skeleton, IK, tissue colliders, locomotion, injury ✅
- Buildings & structural construction: parts, joints, stress analysis, crafting ✅

Current focus areas:
- Phase 13: Electricity & Magnetism (conductivity, resistance networks, resistive heating, lightning)
- Phase 14: Nuclear Physics & Radiation (decay chains, ionising dose model, shielding)
- Gameplay systems: crafting UI, build mode preview, inventory UI, NPC spawning & ecology

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
