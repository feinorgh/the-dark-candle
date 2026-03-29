# Simulation Module

Core simulation loop that orchestrates heat transfer, chemical reactions, state transitions, and pressure diffusion on a flat voxel grid.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | `simulate_tick()`, `simulate_tick_dx()`, `simulate_subdivided_region()`, `TickResult`, `SimulationStats` |
| `geometry.rs` | `Region` enum + `apply_regions()` for building voxel grids from RON |
| `assertions.rs` | `Assertion` enum + `evaluate()` for verifying simulation outcomes |
| `scenario.rs` | `SimulationScenario` struct + `run_scenario()` for data-driven tests |

## Dependencies

- **Imports from:** `crate::chemistry::{heat, reactions, state_transitions, fire_propagation}`, `crate::data::{MaterialData, MaterialRegistry, ReactionRule}`, `crate::world::voxel::{MaterialId, Voxel}`
- **Imported by:** `tests/simulations.rs` (scenario runner), `tests/physics_visual.rs` (video demos), `src/chemistry/fire_propagation.rs` (unit tests)

## Key APIs

### `simulate_tick(voxels, size, rules, registry, dt)`

Standard entry point. Runs one full physics tick at 1 m voxel resolution. Calls `simulate_tick_dx()` with `dx = 1.0`.

### `simulate_tick_dx(voxels, size, rules, registry, dt, dx)`

Multiresolution entry point. The `dx` parameter is the voxel edge length in meters. Tick order:

1. **Conduction** — `diffuse_chunk(voxels, size, dt, registry)` (Fourier's law)
2. **Radiation** — `radiate_chunk(voxels, size, dt, dx, registry, σ, threshold, max_steps)` (Stefan-Boltzmann). Temperature change scales as 1/dx.
3. **Reactions** — check all voxel pairs against reaction rules, apply transformations + heat output
4. **Transitions** — phase changes (melt, boil, freeze, condense) with latent heat buffering
5. **Pressure** — gas-phase pressure equalization

Returns `TickResult` with reaction count, transition count, and per-step timing.

### `simulate_subdivided_region(octree, region, rules, registry, dt, depth)`

Flattens an octree region at the given subdivision depth into a higher-resolution flat array, runs `simulate_tick_dx()` at the corresponding `dx`, then writes changes back. Uses existing `octree_to_flat()` / `flat_to_octree()` conversions.

## Data-Driven Scenarios

`.simulation.ron` files in `tests/cases/simulation/` define headless test scenarios. See `docs/simulation-test-system.md` for the full reference.

## Multiresolution Physics

The `dx` parameter affects physics scaling:
- **Conduction**: flux ∝ dx (face area dx² / distance dx), CFL limit ∝ dx²
- **Radiation**: ΔT ∝ 1/dx (radiated power ∝ dx², thermal mass ∝ dx³)
- **Reactions/transitions**: per-voxel, unaffected by dx

Use `simulate_tick()` for standard 1 m chunks. Use `simulate_tick_dx()` for sub-meter simulations (e.g. fire through tree branches at dx=0.5 m).

## Gotchas

- `simulate_tick()` mutates the voxel array in-place. Conduction uses an internal double-buffer; reactions/transitions modify voxels directly.
- Radiation only processes surface voxels above 500 K (emission threshold). Cold surfaces receive but don't emit.
- The tick loop runs conduction → radiation → reactions → transitions → pressure. This order matters: reactions see post-diffusion temperatures.
- At sub-meter dx, radiation is stronger per-voxel but convective heat transport (not yet in engine) is the dominant fire-spread mechanism in reality. See the forest fire test for a convection proxy pattern.
