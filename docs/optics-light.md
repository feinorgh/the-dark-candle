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

## Tier 2 — Refraction & Reflection (planned)

- **Refraction (Snell's law)** — light bends at material boundaries proportional
  to the ratio of refractive indices: n₁ sin θ₁ = n₂ sin θ₂. Enables lensing
  through glass blocks, underwater distortion, mirage effects from hot air
  (gradient in n due to temperature-dependent density)
- **Reflection (Fresnel equations)** — partial reflection at every interface;
  reflectance depends on angle and refractive index ratio. At glancing angles
  even water becomes mirror-like (total internal reflection above the critical
  angle)

## Tier 3 — Advanced Phenomena (planned)

- **Mie scattering** — forward-peaked scattering by particles comparable to
  wavelength (water droplets, dust, ash). Produces halos around sun/moon, white
  clouds, fog glow. Coupled to LBM humidity/particulate density
- **Caustics** — focused light patterns from refraction through curved surfaces
  (underwater ripple patterns, light through glass bottles). Approximate via
  photon mapping or screen-space caustic estimation
- **Dispersion** — wavelength-dependent refractive index separates white light
  into spectral components (prisms, rainbows). Model via 3-channel (RGB)
  refraction with slightly different n per channel

## Design constraints

All optical parameters derive from `MaterialData`. No per-material shader
hacks — a single physically-based light transport model with material-driven
parameters.

Priority: medium-high. Optics are central to visual quality and enable unique
gameplay (lens crafting, underwater exploration, light puzzles).
Depends on: Phase 9a (ray-cast infrastructure), Phase 9 (atmosphere, sun
angle), Phase 11 (glass material for structures).
