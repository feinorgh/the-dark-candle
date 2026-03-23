# Chemistry Module

Heat transfer, chemical reactions, material state transitions, and fire propagation.

## Files

| File | Purpose |
|------|---------|
| `heat.rs` | Fourier's-law heat diffusion between neighboring voxels |
| `reactions.rs` | `ReactionData` RON struct, condition matching |
| `state_transitions.rs` | Phase changes (melt/boil/freeze/condense) |
| `fire_propagation.rs` | Integration: heat + reactions + transitions per tick |

## Dependencies

- **Imports from:** `crate::data::{MaterialData, MaterialRegistry, Phase}`, `crate::world::voxel::{MaterialId, Voxel}`
- **Imported by:** (fire_propagation integrates heat + reactions + transitions)

## Key Types

- `MaterialRegistry` — defined in `crate::data`, provides `HashMap<u16, MaterialData>` and name→ID lookup for runtime material resolution
- `ReactionData` — loaded from `.reaction.ron` files: uses material names (Strings) for reactant, catalyst, product, plus temperature range
- `TransitionResult` — what a voxel becomes when heated/cooled past a threshold

## Data Files

- `assets/data/materials/*.material.ron` — physical/chemical properties per material
- `assets/data/reactions/*.reaction.ron` — chemical reaction definitions

## Heat Transfer: Fourier's Law

Heat diffusion uses **Fourier's law** on a discrete voxel grid (dx = 1 m, A = 1 m², V = 1 m³).

### Per-face heat flux

```
Q = k_eff × (T_neighbor − T_self)      [W]
```

where `k_eff = 2 × k₁ × k₂ / (k₁ + k₂)` is the harmonic mean conductivity at the interface.

### Temperature update per voxel per tick

```
ΔT = Σ(Q_neighbors) × dt / (ρ × Cₚ)   [K]
```

Thermal properties (`thermal_conductivity`, `density`, `specific_heat_capacity`) are read from `MaterialData` via `MaterialRegistry` — no hardcoded values.

### API

- `thermal_conductivity(material, registry)` — look up SI conductivity W/(m·K) from the registry.
- `diffuse_temperature(current_temp, self_k, self_rho_cp, neighbors, dt)` — Fourier's law for a single voxel.
- `diffuse_chunk(voxels, size, dt, registry)` — apply one diffusion step to a `size³` voxel array. Takes `&MaterialRegistry` and `dt` (timestep in seconds).
- `conductivity(material)` — **deprecated** legacy hardcoded lookup (normalized 0–1 scale).

### CFL Stability Constraint

```
dt < dx² / (6 × α_max)     where α = k / (ρ × Cₚ)
```

For iron (highest diffusivity, α ≈ 2.27 × 10⁻⁵ m²/s): `dt_max ≈ 7300 s` — comfortably stable at game timesteps (1/60 s).

### Energy Conservation

In a closed system (no external sources/sinks), total thermal energy `Σ(ρ × Cₚ × T)` is conserved across diffusion steps (within floating-point tolerance). This is tested explicitly.

## Patterns

- `check_reaction()` tests if a voxel meets reaction conditions (material name, temperature, neighbor); requires `MaterialRegistry` to resolve names to IDs.
- `check_transition()` tests if a voxel should change phase based on temperature.

## Fire Propagation Tuning

The `heat_output` field in `ReactionData` is a **temperature delta in Kelvin (ΔT)** applied
directly to the reacted voxel. Values should approximate the expected product temperature
relative to the reaction's trigger point:

| Reaction | ΔT (K) | Rationale |
|----------|--------|-----------|
| Wood combustion | 800 | Flame temp ~1300 K − ignition 573 K |
| Grass combustion | 550 | Flame temp ~1100 K − ignition 533 K |
| Ice melting | −10 | Endothermic cooling (latent heat TBD Phase 5) |

> **Note:** The fire-propagation integration tests use deliberately inflated
> `heat_output` values (3000–10000) to compensate for the conduction-only heat
> model (no convection or radiation). 1D line spreading works with 3000; 2D
> surface spreading needs ≥10000 because heat dissipates into more neighbors.

The fire chain reaction depends on heat equilibrium exceeding the ignition point:

```
equilibrium ≈ (heat_output + N × ambient) / (N + 1)
```

- For a 1D wood line with ~2 neighbors: `heat_output = 3000` works.
- For a 2D floor with ~5 neighbors: `heat_output = 10000` needed with SI conduction.
- `ignition_point` for wood is 573 K; ambient is 288.15 K.
- If equilibrium < ignition_point, fire dies out instead of spreading.
- With real SI conduction, integration tests use large `dt` values (e.g. 5000 s) to allow meaningful heat transfer through low-conductivity materials like wood (k = 0.15 W/(m·K)).

> **TODO:** Convert `heat_output` to real SI energy (J/kg) using
> `MaterialData::heat_of_combustion` and a sustained burn-rate model.
> This requires per-voxel burn progress and `MaterialRegistry` access at
> reaction-application time. See `heat_of_combustion` in material data.

## Gotchas

- RON `[f32; 3]` arrays must use tuple syntax: `(0.5, 0.5, 0.5)`, not `[0.5, 0.5, 0.5]`.
- `melted_into`, `boiled_into`, `frozen_into`, `condensed_into` fields are `Option<String>` (material names) with `#[serde(default)]` — they can be omitted from RON files for materials that don't transition.
- Materials with `thermal_conductivity = 0.0` (default) or missing registry entries fall back to default values (k=0.1, ρ=1.0, Cₚ=1000.0) to avoid division by zero.
