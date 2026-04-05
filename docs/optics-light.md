# Phase 12 — Optics & Light Phenomena

Physically-based light transport through the voxel world, enabling glass optics,
underwater caustics, atmospheric color, and material-dependent visual effects.
Builds on the radiative transfer ray-cast infrastructure from Phase 9a.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[atmosphere-simulation.md](atmosphere-simulation.md) (Phase 9, scattering/sky),
[advanced-physics.md](advanced-physics.md) (future physics subsystems).

---

## Tier 1 — Foundation & Sky (✅ complete)

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

## Tier 2 — Refraction & Reflection ✅

- **Refraction (Snell's law)** — light bends at material boundaries proportional
  to the ratio of refractive indices: n₁ sin θ₁ = n₂ sin θ₂. Enables lensing
  through glass blocks, underwater distortion, mirage effects from hot air
  (gradient in n due to temperature-dependent density)
- **Reflection (Fresnel equations)** — partial reflection at every interface;
  reflectance depends on angle and refractive index ratio. At glancing angles
  even water becomes mirror-like (total internal reflection above the critical
  angle)

### Implementation Status (Tier 2 — complete)

| Component | File | Description |
|---|---|---|
| Pure optics math | `src/lighting/optics.rs` | `snell_refract`, `fresnel_reflectance`, `reflect_dir`, `is_total_internal_reflection`, `critical_angle`, temperature-dependent air n, Cauchy dispersion (25 tests) |
| Refractive DDA | `src/world/raycast.rs` | `dda_march_ray_refractive` — traces rays bending at n-boundaries, handles TIR bounces, returns `RefractivePath` with Fresnel transmittance (4 tests) |
| Chunk refraction map | `src/lighting/refraction.rs` | `ChunkRefractionMap` component, `propagate_refraction_from_registry` system, `update_chunk_refraction_maps` wired into `LightingPlugin` (4 tests) |

## Tier 3 — Advanced Phenomena ✅

### Dispersion (Cauchy equation)

White light separates into a spectrum because refractive index varies with
wavelength. This is modeled via the **Cauchy equation**:

```
n(λ) = A + B / λ²
```

The green-channel n (`refractive_index` in `MaterialData`) anchors the
curve; `cauchy_b: Option<f32>` provides the B coefficient in m².

| Material | n (550 nm) | B (m²) | Abbe # |
|---|---|---|---|
| Borosilicate glass | 1.52 | 4.61 × 10⁻¹⁵ | ~65 |
| Fused silica (quartz) | 1.46 | 3.40 × 10⁻¹⁵ | ~68 |

Per-channel n at R=680 nm, G=550 nm, B=440 nm is computed by
`optics::dispersive_n_rgb(base_n, cauchy_b)`.  Blue always has the highest n
(shortest λ → most bending).

### Local Mie Scattering

Voxel-scale particle media (steam, ash) scatter light forward via the
Henyey-Greenstein phase function:

```
p(θ) = (1 − g²) / [4π (1 + g² − 2g cosθ)^1.5]
```

Extinction follows Beer-Lambert: T = exp(−β × d), with β (m⁻¹) and
asymmetry g per material.

| Material | β (m⁻¹) | g |
|---|---|---|
| Steam | 50 | 0.85 |
| Ash | 20 | 0.65 |

### Caustics

The analytical caustic concentration factor at a flat interface is:

```
C = (n₂/n₁)² × (cos θ₁ / cos θ₂)
```

This is the Jacobian of the solid-angle mapping for a refracted photon bundle.
At normal incidence (θ₁ = 0), C = (n₂/n₁)²:

- air → water: C ≈ 1.77
- air → glass: C ≈ 2.31

A photon-beam tracer (`caustics::trace_caustic_beam`) distributes stratified
samples across a cone and casts them through a flat interface, returning
`CausticPhoton { pos, rgb }` structs for kernel density estimation.

### Implementation Status (Tier 3 — complete)

| Component | File | Description |
|---|---|---|
| Dispersion math | `src/lighting/optics.rs` | `cauchy_a_from_n_green`, `dispersive_n_rgb`, `snell_refract_rgb`, `fresnel_reflectance_rgb`, `fresnel_transmittance_rgb` (10 new tests; 35 total) |
| MaterialData cauchy_b | `src/data/mod.rs` | `cauchy_b: Option<f32>` field + `dispersion_n_rgb()` helper; backward-compat via `#[serde(default)]` |
| Dispersive DDA | `src/world/raycast.rs` | `dda_march_ray_dispersive` — 3 independent per-channel ray traces; `DispersivePath` result struct (3 tests) |
| Local Mie module | `src/lighting/mie_local.rs` | `mie_params_for_material`, `mie_transmittance_rgb`, `mie_phase_hg`, `mie_in_scatter_factor` (12 tests) |
| Caustics module | `src/lighting/caustics.rs` | `CausticPhoton`, `trace_caustic_beam`, `caustic_irradiance_at`, `refraction_caustic_factor`, `underwater_irradiance_fraction` (11 tests) |
| Material RON files | `assets/data/materials/` | `glass.material.ron` and `quartz_crystal.material.ron` updated with `cauchy_b` values |
| Simulation scenarios | `tests/cases/simulation/` | `glass_prism_dispersion.simulation.ron` and `underwater_caustics.simulation.ron` |


## Design constraints

All optical parameters derive from `MaterialData`. No per-material shader
hacks — a single physically-based light transport model with material-driven
parameters.

Priority: medium-high. Optics are central to visual quality and enable unique
gameplay (lens crafting, underwater exploration, light puzzles).
Depends on: Phase 9a (ray-cast infrastructure), Phase 9 (atmosphere, sun
angle), Phase 11 (glass material for structures).
