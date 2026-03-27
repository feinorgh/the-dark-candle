# Phase 9 — Atmosphere Simulation ✅

A physics-driven atmosphere model that builds on the existing LBM gas simulation,
barometric pressure functions, and FLIP/PIC particle system. Weather, wind, and
precipitation emerge from first-principles thermodynamics on the spherical planet —
not from scripted weather states. Atmospheric conditions feed into a rendering
pipeline for clouds, fog, and dynamic lighting.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[spherical-terrain.md](spherical-terrain.md) (Phase 8, planet model),
[fluid-simulation-system.md](fluid-simulation-system.md) (AMR/LBM/FLIP details).

---

## Foundations already in place

| System | Location | Provides |
|--------|----------|----------|
| Barometric formula | `constants.rs` | `P(h) = P₀ × exp(−Mgh/RT)`, `ρ = PM/(RT)` |
| Pressure diffusion | `pressure.rs` | Chunk-local pressure equalization, gradient forces |
| LBM D3Q19 gas | `lbm_gas/` | Compressible gas dynamics, BGK + Smagorinsky turbulence, buoyancy-driven convection, Guo forcing, `ambient_density_at_altitude()` with lapse rate |
| FLIP/PIC particles | `flip_pic/` | Rain/snow/spray emission, advection, sub-voxel accumulation, phase-transition emission (evaporation, melting) |
| Lighting | `world/mod.rs` | Static DirectionalLight (sun) + AmbientLight |

## Design

Weather emerges from the physics. The atmosphere is a continuous gas field
simulated by LBM at macro scale, with FLIP/PIC handling precipitation and spray.
Solar heating, Coriolis force from planet rotation (Phase 8), and moisture
transport produce realistic circulation patterns.

### 1. Atmospheric circulation & wind

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

### 2. Moisture transport & cloud formation

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

### 3. Weather phenomena

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

### 4. Climate zones

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

### 5. Atmospheric rendering

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

## Implementation Steps

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

## Dependencies

- Steps 1–6 depend on **Phase 8** (spherical planet, radial gravity, planet
  rotation).
- Steps 2–3 depend on the existing LBM gas plugin being wired into runtime
  (`PhysicsPlugin::build()`).
- Steps 7–10 (rendering) can proceed partially in parallel with physics steps.
- Step 5 (clouds) depends on step 2 (humidity) and step 4 (solar heating).
- Step 6 (precipitation) depends on step 5 (clouds).
- Step 8 (cloud shadows) depends on step 5 (clouds) and step 7 (dynamic lighting).

## What stays unchanged

LBM core (collision, streaming, macroscopic recovery), FLIP/PIC core (P2G, G2P,
advection, accumulation), pressure diffusion, barometric formula, material system,
chunk/voxel infrastructure. These are extended (humidity scalar, Coriolis force)
but not rewritten.

## Completion

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

## Sub-phases: Chemistry Runtime & Visual Rendering

### Phase 9a: Radiative Heat Transfer & Thermal Visualization

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

### Phase 9b: Chemistry Runtime Activation

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

### Phase 9c: Thermal Glow Rendering

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

### Phase 9d: Time-of-Day & Dynamic Lighting

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
