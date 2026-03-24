# Fluid Simulation System

## Overview

This document describes the fluid simulation architecture for The Dark Candle, designed to
operate on the Sparse Voxel Octree (SVO) subdivision system. It replaces the current cellular
automata fluid model with three complementary simulation methods, each targeting a different
phase of matter and exploiting the octree hierarchy in a different way.

All simulation uses SI units: viscosity in Pa·s, density in kg/m³, pressure in Pa,
velocity in m/s, temperature in K.

## Architecture: Three-Model Approach

### 1. Lattice Boltzmann Method (LBM) — Gases

**Purpose:** Atmosphere, steam, smoke, pressure wave propagation.

**Why LBM:** The stream-and-collide algorithm maps directly to a regular grid of octree leaf
cells. It naturally recovers the Navier-Stokes equations at macroscopic scale without solving
them explicitly, and is the most GPU-friendly fluid method available.

**How it integrates with the SVO:**
- Each octree leaf cell stores a set of distribution functions f_i (D3Q19 or D3Q27 lattice).
- Collision step: relaxation toward equilibrium (BGK or MRT operator) using real gas viscosity.
- Streaming step: distribution functions propagate to face-adjacent neighbor cells.
- Multi-resolution: at octree level boundaries (e.g., 1m cell adjacent to 0.5m cell),
  distribution functions are rescaled using the Filippova-Hänel or Dupuis-Chopard method
  to maintain conservation of mass and momentum across resolution transitions.
- Density recovered from ideal gas law: ρ = P × M / (R × T), consistent with
  `universal_constants.ron` (gas constant R, molar mass M from MaterialData).
- Pressure emerges from the equation of state: P = ρ × c_s² (c_s = lattice speed of sound).

**Barometric formula integration:**
- Background pressure field from altitude-dependent barometric formula provides the
  equilibrium state. LBM simulates deviations from this equilibrium (wind, explosions,
  convection currents).

**GPU offloading:**
- Stream-and-collide is embarrassingly parallel — one compute shader invocation per cell.
- Distribution functions stored in a GPU SSBO or 3D texture array.
- Multi-resolution boundary handling in a separate compute pass.

### 2. Adaptive Mesh Refinement (AMR) Navier-Stokes — Liquids

**Purpose:** Water, lava, and other incompressible or weakly compressible liquids.

**Why AMR N-S:** The octree is inherently an AMR grid. Liquid simulation requires
pressure-velocity coupling (incompressibility constraint) that is best handled by a
projection method on the grid, and the octree depth levels serve as a natural multigrid
hierarchy for the pressure solver.

**How it integrates with the SVO:**
- Velocity field stored per octree leaf cell (Vec3, m/s).
- Advection: semi-Lagrangian backtracing through the octree.
- Diffusion: viscosity-dependent diffusion using real fluid viscosity from MaterialData
  (water = 1.0e-3 Pa·s, lava = 500 Pa·s via dynamic lookup).
- Pressure projection: solve the Poisson equation ∇²p = (ρ/Δt) × ∇·u to enforce
  incompressibility. The octree depth levels form a geometric multigrid:
  - Restriction: fine-level residuals averaged to coarse parent nodes.
  - Prolongation: coarse corrections interpolated back to fine leaves.
  - V-cycle or W-cycle iteration until convergence.
- Free surface tracking: liquid/air interface tracked by the octree itself — cells are
  tagged as LIQUID, AIR, or SURFACE. Adaptive refinement keeps the interface at maximum
  resolution while bulk liquid stays coarse.
- Variable cell size: flux between cells of different sizes uses face-area weighting.
  A 0.5m cell face adjacent to a 1m cell face transfers momentum proportional to the
  shared area.

**Viscosity-dependent behavior:**
- Water (1.0e-3 Pa·s): thin, fast-flowing, splashing.
- Lava (500 Pa·s): thick, slow, creeping flow.
- Viscosity read from MaterialData at runtime — no hardcoded flow rates.

**GPU offloading:**
- Pressure Poisson solve: Jacobi iteration is parallel per level.
- Advection: parallel per cell (read-only backtracing).
- Restriction/prolongation: parallel per level transition.

### 3. FLIP/PIC Particles — Accumulation and Splashing

**Purpose:** Snowfall, rain, spray, evaporation droplets, ash, sand particles, and any
scenario where discrete particles accumulate or detach from surfaces.

**Why FLIP/PIC:** Particles carry velocity without grid diffusion, enabling thin sheets,
droplets, and natural accumulation behavior. The octree provides the pressure grid and
spatial acceleration structure.

**How it integrates with the SVO:**
- Particles exist as lightweight structs: position (Vec3), velocity (Vec3), mass (f32),
  material (MaterialId), temperature (f32).
- Each simulation step:
  1. **Particle-to-grid (P2G):** Transfer particle velocities to octree leaf cells
     (weighted by distance, trilinear for PIC, delta for FLIP).
  2. **Grid solve:** Pressure projection on the octree (shared with AMR N-S solver).
  3. **Grid-to-particle (G2P):** Update particle velocities from the grid.
     FLIP ratio (typically 0.95–0.99) blends PIC (stable) and FLIP (low diffusion).
  4. **Advect particles:** Move by velocity × Δt.
- Octree as spatial hash: particles binned into octree leaves for O(1) neighbor lookup
  during P2G/G2P transfers.

**Accumulation:**
- When a particle's velocity drops below a threshold and it contacts a solid surface,
  it deposits into the voxel grid as sub-voxel material using SVO subdivision.
- Snow accumulates on upward-facing surfaces; ash settles on any surface.
- Accumulation thickness tracked per surface cell; when a sub-voxel layer fills,
  the parent voxel transitions from air to the accumulated material.

**Evaporation:**
- Liquid surface voxels above the boiling point (from MaterialData) emit gas particles.
- Emission rate proportional to surface area and temperature excess above boiling point.
- Emitted particles enter the LBM gas simulation as mass sources.

**Phase transition coupling:**
- Particles carry temperature. When a water droplet particle cools below 273.15 K,
  it freezes on contact (deposits as ice).
- When a snow voxel heats above 273.15 K, it emits water particles (melting).
- Latent heat absorbed/released per MaterialData (latent_heat_fusion, latent_heat_vaporization).

**GPU offloading:**
- P2G/G2P: parallel per particle (atomic adds to grid).
- Advection: parallel per particle.
- Spatial binning: parallel radix sort or compute-shader bin assignment.

## Data Flow Between Models

```
                 ┌──────────────┐
  evaporation    │              │   condensation
  particles ────>│  LBM (Gas)   │<──── particles
                 │              │
                 └──────┬───────┘
                        │ pressure coupling
                        ▼
                 ┌──────────────┐
                 │  AMR N-S     │
  FLIP/PIC <────>│  (Liquid)    │<────> Octree grid
  particles      │              │       (shared pressure solve)
                 └──────┬───────┘
                        │ surface interaction
                        ▼
                 ┌──────────────┐
                 │  FLIP/PIC    │
                 │  (Particles) │───> accumulation into SVO
                 │              │───> erosion from SVO
                 └──────────────┘
```

- **Gas ↔ Liquid:** Evaporation emits FLIP particles that become LBM gas mass sources.
  Condensation creates FLIP droplet particles from supersaturated gas cells.
- **Liquid ↔ Particles:** Splashing at liquid free surfaces spawns FLIP particles.
  Particles re-entering liquid volume are absorbed back into the AMR velocity field.
- **Particles ↔ Voxels:** Accumulation deposits particles as sub-voxel material.
  Erosion (wind, water flow) detaches voxel material as particles.

## Simulation Tick Integration

All three models run in the `FixedUpdate` schedule with consistent Δt:

1. **LBM gas step** (stream + collide)
2. **AMR liquid step** (advect + diffuse + pressure project)
3. **FLIP particle step** (P2G → grid solve → G2P → advect → accumulate/evaporate)
4. **Phase transition check** (temperature-driven material changes)
5. **Adaptive refinement pass** (subdivide/collapse octree based on new fluid state)

Each step reads from the previous step's output, ensuring consistent coupling per tick.

## Constants and Configuration

All physical constants come from existing data files:
- Gravity, atmospheric pressure, sea-level temperature: `world_constants.ron`
- Gas constant, Stefan-Boltzmann: `universal_constants.ron`
- Per-material viscosity, density, boiling/melting points, latent heats: `materials/*.material.ron`

Simulation tuning parameters (new file `assets/data/fluid_config.ron`):
- `lbm_lattice`: D3Q19 or D3Q27
- `lbm_relaxation`: BGK or MRT
- `flip_pic_ratio`: 0.0 (pure PIC) to 1.0 (pure FLIP), default 0.97
- `pressure_solver_iterations`: max multigrid V-cycles, default 50
- `particle_deposit_velocity_threshold`: m/s below which particles accumulate
- `max_particles_per_chunk`: memory budget cap
- `evaporation_rate_scale`: multiplier on emission rate (1.0 = physically accurate)

## Migration Path from Current System

The legacy cellular automata fluid system (`src/physics/fluids.rs`) has been replaced
by three physics-accurate models:

1. ~~Keep CA fluids operational~~ — done (kept operational during development).
2. ~~Implement AMR N-S for water first~~ — done (`src/physics/amr_fluid/`).
3. ~~Add LBM for gases~~ — done (`src/physics/lbm_gas/`).
4. ~~Add FLIP particles~~ — done (`src/physics/flip_pic/`).
5. ~~Remove CA fluids~~ — done. `fluids.rs` and its octree bridge wrapper deleted.

Each model was developed and tested independently. The three-model stack is now the
sole fluid simulation system.
