# Terrain Generation System

Design document for the terrain detail upgrade and world generation options.
Covers the noise engine, scene presets, geological layers, cave systems, and
biome-terrain integration.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[voxel-subdivision-system.md](voxel-subdivision-system.md) (octree/LOD).

---

## Current State

The terrain generator (`src/world/terrain.rs`) supports two modes via
`UnifiedTerrainGenerator`: **Flat** (2D heightmap) and **Spherical** (Phase 8
radial planet). Both share the same noise evaluation and material assignment
logic.

### Noise pipeline (current)

```
sample_height(x, z):
    continent = Perlin(seed).get(x * 0.005, z * 0.005)
    detail    = Perlin(seed+1).get(x * 0.02, z * 0.02)
    return sea_level + (continent * 0.7 + detail * 0.3) * height_scale
```

Two Perlin layers blended at a fixed 70/30 ratio. No multi-octave stacking,
no ridged fractal, no domain warping. The result is smooth, repetitive terrain
with no fine-scale features.

### Material assignment (current)

Per column, top to bottom:

| Depth below surface | Material |
|---------------------|----------|
| Above surface | AIR (or WATER if below sea level) |
| 0–1 m | GRASS (above sea) or DIRT (below) |
| 1–soil_depth (4 m) | DIRT |
| Below soil_depth | STONE (uniform, no strata) |

Cave carving: single 3D Perlin at frequency 0.03, threshold −0.3. Only fires
≥2 m below the surface. No multi-scale caves, no connected networks.

### Erosion (current)

D8 flow accumulation with Planchon–Darboux sink filling. Valley carving via
`carve_valley()` blends V-shaped and U-shaped cross-sections. Enabled per
preset (currently only `valley_river`). Cached in `OnceLock<FlowMap>`.

No particle-based hydraulic erosion. The D8 carving creates large-scale valleys
and drainage channels but doesn't produce fine-scale gullies, sediment fans, or
the natural smoothing that rainfall creates over time.

### Parameters (`TerrainConfig`)

| Field | Default | Description |
|-------|---------|-------------|
| `seed` | 42 | Noise seed |
| `sea_level` | 64 | Water surface Y (flat mode) |
| `height_scale` | 32.0 | Amplitude ±N voxels from sea level |
| `continent_freq` | 0.005 | Low-freq landmass noise (~200 m cycle) |
| `detail_freq` | 0.02 | Mid-freq hill noise (~50 m cycle) |
| `cave_freq` | 0.03 | 3D cave noise (~33 m cycle) |
| `cave_threshold` | −0.3 | Cave carve threshold |
| `soil_depth` | 4 | Dirt layer thickness (voxels) |
| `erosion` | (disabled) | `ErosionConfig` sub-struct |

### Scene presets (current)

Only one preset exists: `valley_river`. Selected via `--scene valley_river`.

### Performance context

Terrain generation runs on `AsyncComputeTaskPool` (up to 8 concurrent chunks).
Each chunk is 32³ = 32,768 voxels. Current cost is dominated by ~34K Perlin
evaluations per flat chunk (~66K for spherical). Generation is fully off the
main thread — **all improvements to terrain detail have zero runtime FPS
impact**. The only cost is initial world load time, which is hidden behind
the loading screen.

LOD meshing (5 levels, stride 1–16) and octree compression already handle
rendering cost regardless of terrain complexity.

---

## Design: Noise Engine Upgrade

### T1. NoiseStack — composable multi-octave noise

Create `src/world/noise.rs` with a `NoiseStack` struct that replaces the
hardcoded 2-layer blend. The stack is built from composable layers, each with
its own noise function, frequency, amplitude, and blend mode.

#### Fractal Brownian Motion (FBM)

Standard multi-octave noise summing progressively finer detail:

```
fbm(x, z, octaves=6, persistence=0.5, lacunarity=2.0):
    value = 0.0
    amplitude = 1.0
    frequency = base_freq
    for i in 0..octaves:
        value += amplitude * Perlin(seed + i).get(x * frequency, z * frequency)
        amplitude *= persistence       // each octave is quieter
        frequency *= lacunarity        // each octave is finer
    return value / normalize           // scale to [-1, 1]
```

Default 6 octaves gives detail down to ~1.5 m features (at `base_freq=0.005`,
the finest octave samples at `0.005 × 2⁵ = 0.16`, giving ~6 m wavelength).
This replaces the current `continent * 0.7 + detail * 0.3`.

#### Ridged multi-fractal

Produces sharp mountain ridges and peaks:

```
ridged(x, z, octaves=5, lacunarity=2.0, gain=2.0):
    value = 0.0
    weight = 1.0
    frequency = base_freq
    for i in 0..octaves:
        signal = 1.0 - abs(Perlin(seed + i).get(x * frequency, z * frequency))
        signal *= signal              // sharpen ridges
        signal *= weight              // weight by previous octave
        weight = clamp(signal * gain, 0, 1)
        value += signal * persistence^i
        frequency *= lacunarity
    return value
```

The `1.0 - abs(noise)` inversion creates ridges where noise crosses zero.
Squaring sharpens them. Gain feedback makes ridges self-amplify.

#### Terrain type selector

A low-frequency noise (0.003) selects between terrain types:

```
selector = Perlin(seed + 100).get(x * 0.003, z * 0.003)

if selector < -0.2:       // 30% — flat/rolling
    height = fbm(x, z) * 0.6
elif selector > 0.3:       // 20% — mountains
    height = ridged(x, z) * 1.5
else:                      // 50% — blended transition
    t = smoothstep(-0.2, 0.3, selector)
    height = lerp(fbm * 0.6, ridged * 1.5, t)
```

This gives the world distinct terrain regions (plains, hills, mountains) with
smooth transitions, all from a single seed.

#### Domain warping

Offset the sample coordinates by another noise field before evaluating height:

```
warp_x = Perlin(seed + 200).get(x * 0.004, z * 0.004) * warp_strength
warp_z = Perlin(seed + 201).get(x * 0.004, z * 0.004) * warp_strength
height = noise_stack.sample(x + warp_x, z + warp_z)
```

Default `warp_strength = 40.0` (voxels). This breaks up the grid-aligned feel
of Perlin noise, producing organic coastlines, meandering valleys, and
irregular mountain shapes. Two passes of warping (warp the warp) increases
the effect further.

#### Micro-detail

A high-frequency (0.1–0.2), low-amplitude (±1–2 voxels) noise layer evaluated
only at the surface (within ±2 voxels of the height). Adds visual roughness
without affecting the broader terrain shape. Negligible cost since it's
per-column, not per-voxel.

#### `NoiseStack` API

```rust
pub struct NoiseStack {
    seed: u32,
    fbm_octaves: u32,        // default 6
    fbm_persistence: f64,    // default 0.5
    fbm_lacunarity: f64,     // default 2.0
    fbm_base_freq: f64,      // default 0.005
    ridged_octaves: u32,     // default 5
    ridged_gain: f64,         // default 2.0
    ridged_base_freq: f64,   // default 0.008
    selector_freq: f64,       // default 0.003
    selector_thresholds: (f64, f64), // default (-0.2, 0.3)
    warp_strength: f64,       // default 40.0
    warp_freq: f64,           // default 0.004
    micro_freq: f64,          // default 0.15
    micro_amplitude: f64,     // default 1.5
    height_scale: f64,        // default 48.0 (increased from 32)
}

impl NoiseStack {
    pub fn sample(&self, x: f64, z: f64) -> f64 { ... }
    pub fn sample_with_detail(&self, x: f64, z: f64) -> f64 { ... }
}
```

Constructed from `TerrainConfig` (extended with noise params) or from a
`NoisePreset` enum. The existing `sample_height()` is replaced by
`noise_stack.sample()`.

#### Performance

| Noise operation | Evaluations per chunk (flat) | Cost vs current |
|-----------------|------------------------------|-----------------|
| Current (2 Perlin) | ~34K | 1.0× |
| FBM 6 octaves | ~6K × 6 = 36K | ~1.1× |
| Ridged 5 octaves | ~6K × 5 = 30K | ~0.9× (only where selector > 0.3) |
| Selector | ~1K | negligible |
| Domain warp | ~2K | negligible |
| Micro-detail | ~1K (surface only) | negligible |
| **Total** | ~70K | **~2× current** |

2× noise cost is ~5–50 ms per chunk (current is ~2.5–25 ms). With 8 async
tasks, the wall-clock difference is negligible during loading.

### T2. Continent shaping & ocean masks

A separate very-low-frequency noise (0.001–0.003) as a continent mask:

```
continent_value = Perlin(seed + 300).get(x * 0.002, z * 0.002)

if continent_value < -0.1:     // deep ocean
    height = sea_level - 30 + (continent_value + 1.0) * 10.0
elif continent_value < 0.05:   // continental shelf / beach
    t = smoothstep(-0.1, 0.05, continent_value)
    height = lerp(ocean_floor, noise_stack.sample(x, z), t)
else:                          // land
    height = noise_stack.sample(x, z)
```

This decouples "where is land" from "how tall is land". The transition zone
creates natural continental shelves, beaches, and coastal cliffs. Ocean floor
gets its own low-amplitude noise for underwater terrain.

---

## Design: World Generation Presets & Options

### T3. Scene presets

Extend `scene_presets.rs` with 6+ new presets. Each constructs a `PlanetConfig`
with a tuned `NoiseStack` configuration.

| Preset | Terrain character | Key noise params |
|--------|-------------------|------------------|
| `valley_river` | Eroded valleys with river channels | (existing — keep as-is, migrate to NoiseStack) |
| `alpine` | Towering ridged peaks, deep valleys, snowline | Ridged dominant (selector threshold −0.5), height_scale 80, micro high |
| `archipelago` | Island chains in open ocean | Continent mask threshold 0.2 (lots of ocean), moderate height_scale |
| `desert_canyon` | Flat mesa tops, deep slot canyons | Plateau noise (clamped FBM), aggressive erosion, cave_threshold −0.2 |
| `rolling_plains` | Gentle hills, wide grasslands | FBM only (selector forces flat), height_scale 16, no ridged |
| `volcanic` | Calderas, lava flows, rugged peaks | Ridged + radial crater noise, high cave density, lava at depth |
| `tundra_fjords` | Glacial U-valleys, flat plateaus | U-shaped erosion (valley_shape 0.8), moderate height |

Each preset also configures erosion, cave density, and optionally overrides
material assignment (e.g., `volcanic` places lava below a depth threshold).

### T4. Extended CLI

Add new `clap` arguments to `Cli`:

```
--scene <name>              Named preset (expanded list)
--seed <u32>                World seed
--terrain-detail <1|2|3>    1=fast (3 octaves), 2=balanced (6), 3=rich (8)
--height-scale <f64>        Override terrain amplitude
--caves <off|sparse|normal|dense>
--erosion <off|light|moderate|heavy>
```

CLI flags override the preset's defaults. For example:
`--scene alpine --seed 12345 --terrain-detail 3 --caves dense`

### T5. In-game world creation screen

New `GameState::WorldCreation` state shown before Loading:

- Preset selector (button row or scrollable list)
- Seed text field with "Random" button
- Terrain detail slider (1–3)
- Cave density selector
- Erosion intensity selector
- Optional: 128×128 heightmap preview (sample NoiseStack on a grid, render
  as a grayscale `Image` in the UI — ~16K noise evaluations, <10 ms)
- "Generate World" button → inserts `PlanetConfig` resource → transitions
  to `GameState::Loading`

---

## Design: Geological Depth

### T6. Depth-based rock layers & ore veins

Replace the uniform "everything below soil = STONE" with stratified geology.

#### Rock strata

Material varies by depth below surface:

| Depth range | Layer | Materials |
|-------------|-------|-----------|
| 0 – soil_depth | Soil | Grass (top), Dirt |
| soil_depth – 20 m | Sedimentary | Sandstone, Limestone (noise-selected) |
| 20 – 60 m | Metamorphic | Slate, Quartzite (noise-selected) |
| 60 m+ | Igneous | Granite, Basalt (noise-selected) |

Within each layer, a low-frequency 3D noise selects between the two material
options, creating natural-looking geological variation.

#### Ore veins

3D Perlin blobs at specific depth ranges:

| Ore | Depth range | Layer | Frequency | Threshold | Rarity |
|-----|-------------|-------|-----------|-----------|--------|
| Coal | 5–30 m | Sedimentary | 0.08 | −0.15 | Common |
| Copper | 15–50 m | Metamorphic | 0.06 | −0.20 | Moderate |
| Iron | 30–80 m | Meta/Igneous | 0.05 | −0.25 | Moderate |
| Gold | 50 m+ | Igneous | 0.04 | −0.35 | Rare |
| Crystal | Cave walls | Any | 0.10 | −0.30 | Rare (caves only) |

Each ore uses its own Perlin noise (seeded from `seed + ore_offset`) and only
fires within its valid depth range. The threshold controls blob size — lower
threshold = smaller, rarer deposits.

#### New materials

6–8 new `.material.ron` files, each with full real-world SI properties:

| Material | Density (kg/m³) | Hardness (Mohs) | k (W/m·K) | Cp (J/kg·K) |
|----------|-----------------|-----------------|-----------|-------------|
| Sandstone | 2,300 | 3.5 | 2.4 | 920 |
| Limestone | 2,500 | 3.0 | 1.3 | 840 |
| Granite | 2,700 | 6.5 | 3.0 | 790 |
| Basalt | 2,900 | 6.0 | 1.7 | 840 |
| Coal | 1,400 | 2.5 | 0.26 | 1,260 |
| Copper (ore) | 4,200 | 3.5 | 1.2 | 530 |
| Gold (ore) | 6,500 | 2.5 | 1.5 | 490 |
| Quartz crystal | 2,650 | 7.0 | 3.0 | 740 |

### T7. Enhanced multi-scale cave system

Replace the single-threshold cave with layered cave types:

#### Cave layers (OR-combined)

| Layer | Type | Frequency | Threshold | Character |
|-------|------|-----------|-----------|-----------|
| Caverns | 3D Perlin | 0.01 | −0.15 | Cathedral-sized chambers |
| Tunnels | 3D Perlin | 0.04 | −0.35 | Narrow connecting passages |
| Tubes | 2D noise pair | 0.025 | −0.25 | Worm-like tubes (Swiss cheese) |

**Tube networks** use a novel approach: two independent 2D noise fields
sampled in perpendicular planes. Where both are below threshold at the same
3D point, a tube exists:

```
tube_xz = Perlin(seed+10).get(x * 0.025, z * 0.025) < -0.25
tube_xy = Perlin(seed+11).get(x * 0.025, y * 0.025) < -0.25
is_tube = tube_xz && tube_xy
```

This creates long, winding tunnels that connect between caverns.

#### Cave features

- **Underground lakes**: Where cavern floor altitude < `sea_level - 5`, fill
  with WATER instead of AIR.
- **Stalactite zones**: Ceiling voxels of large caverns (detected: air below,
  stone above) get a chance to be replaced with a new Calcite material.
- **Crystal deposits**: Rare Quartz material on walls of deep caverns
  (depth > 40 m, noise threshold −0.3).
- **Lava tubes**: At depth > 80 m, tubes may contain LAVA instead of AIR
  (noise threshold −0.2, separate seed).

---

## Design: Biome-Terrain Integration

### T8. Biome map & terrain shaping

#### Biome map generation

A 2D biome map assigns biome IDs to each world column. Computed once and
cached (like `FlowMap`):

1. **Temperature field**: Latitude-based gradient + noise perturbation
2. **Moisture field**: Noise-based + altitude modifier (higher = drier)
3. **Biome selection**: Existing `biome_matches()` with the two fields

The map is sampled at chunk resolution (one biome per 32×32 column block)
with smooth interpolation at boundaries.

#### Per-biome terrain modifiers

Extend `BiomeData` in `.biome.ron` with optional terrain fields:

```ron
(
    name: "desert",
    // ... existing fields ...
    terrain: Some((
        height_bias: -8.0,       // lower base height
        roughness: 0.4,          // less detail amplitude
        erosion_rate: 2.0,       // more erosion (wind-carved)
        subsurface: "sandstone", // geological base material
    )),
)
```

During `generate_chunk()`, the biome's modifiers are applied to the
NoiseStack output:

```
biome = biome_map.get(x, z)
raw_height = noise_stack.sample(x, z)
height = raw_height * biome.roughness + biome.height_bias + sea_level
```

#### Biome boundary blending

At biome boundaries (where adjacent columns have different biome IDs), lerp
the terrain parameters over ~16 voxels using the biome map's fractional
distance to the boundary. This prevents hard terrain seams.

### T9. Surface detail

Enhance the surface material assignment in `generate_chunk()`:

#### Slope-dependent materials

Compute slope as the height difference between adjacent columns. Steep
slopes (gradient > 1.5) → exposed STONE regardless of biome. Moderate slopes
→ DIRT. Flat → biome surface material (GRASS, SAND, etc.).

#### Altitude zones

| Zone | Altitude above sea level | Surface material |
|------|--------------------------|------------------|
| Coastal | −2 to +3 m | Sand |
| Lowland | +3 to +30 m | Biome surface (grass, dirt) |
| Highland | +30 to +60 m | Biome surface with rock patches |
| Alpine | +60 to +80 m | Stone with sparse grass |
| Snow line | +80 m+ | Ice / snow (white material) |

These thresholds are configurable per preset. `alpine` preset would lower
the snow line; `desert_canyon` would replace snow with bare rock.

#### Soil depth variation

Soil depth varies by terrain context:
- Valleys (low slope, low altitude): soil_depth × 1.5 (alluvial deposits)
- Ridges (high slope, high altitude): soil_depth × 0.3 (thin, eroded)
- Normal: soil_depth × 1.0

#### Boulder scattering

3D noise evaluated only at the surface (Y = height ± 1). Where
`Perlin(seed+50).get(x*0.08, z*0.08) > 0.6`, replace the surface material
with STONE to simulate exposed boulders. Only on slopes > 0.5 gradient.

---

## Design: Hydraulic Erosion (Rainfall-Driven)

### T10. Particle-based hydraulic erosion

Add a droplet-simulation erosion pass that runs on the coarse heightmap during
terrain generation (same phase as FlowMap computation — generation-time only,
zero runtime FPS cost). Areas with higher rainfall receive more simulated
droplets, producing denser gullies, smoother slopes, and sediment deposition
in lowlands.

#### How it works

The algorithm simulates thousands of water droplets rolling downhill across the
heightmap. Each droplet picks up sediment where it flows fast (steep slopes,
high velocity) and deposits it where it slows (flat areas, pools). Over many
iterations, this produces:

- **Gullies** — fine branching channels where many droplets converge
- **Smoothed ridges** — exposed summits abraded by repeated flow
- **Sediment fans** — deposited material at the base of slopes
- **Alluvial plains** — fine-grained fill in valley floors
- **Terracing** — natural step patterns on moderate slopes

This complements the existing D8 flow carving, which produces large-scale
valleys. Hydraulic erosion adds the fine-scale detail within and between those
valleys.

#### Algorithm: droplet simulation (Beyer / de Swart variant)

Each droplet is a state tuple `(pos, dir, vel, water, sediment)`:

```
for each droplet:
    1. Spawn at random position on heightmap
       (probability weighted by rainfall intensity map)
    2. Compute gradient at current position (bilinear interpolation)
    3. Update direction: dir = dir * inertia + gradient * (1 - inertia)
       (normalize)
    4. Move: new_pos = pos + dir
    5. Compute height difference: delta_h = h_new - h_old
    6. Compute sediment capacity:
       capacity = max(-delta_h, min_slope) * vel * water * capacity_factor
    7. If sediment > capacity (carrying too much):
       deposit = (sediment - capacity) * deposition_rate
       Add deposit to heightmap at pos (weighted kernel)
       sediment -= deposit
    8. Else (can carry more):
       erode = min((capacity - sediment) * erosion_rate, -delta_h)
       Subtract erode from heightmap at pos (weighted kernel)
       sediment += erode
    9. Update velocity: vel = sqrt(vel² + delta_h * gravity)
   10. Evaporate: water *= (1 - evaporation_rate)
   11. If water < 0.01 or lifetime exceeded or out of bounds: stop
```

#### Erosion radius kernel

Each deposit/erode operation affects a circular area (radius ≈ 3 cells) with
bilinear weight falloff. This prevents single-cell spikes and produces smooth,
natural channels. The kernel is precomputed once per config.

#### Rainfall intensity map

A per-cell `f32` (0.0 = arid, 1.0 = monsoon) controls droplet spawn density.
Generated from composable inputs:

```
rainfall(x, z) = clamp(
    base_moisture                            // from biome (T8) or noise fallback
    + orographic_lift(slope, wind_direction)  // windward slopes get more rain
    + altitude_penalty(height)                // very high = less rain (snow)
    + noise_perturbation(x, z)               // local variation
, 0.0, 1.0)
```

**Before T8 (biome map):** use a standalone noise-based rainfall:
```
rainfall(x, z) = clamp(
    0.5 + 0.4 * Perlin(seed + 200).get(x * 0.003, z * 0.003)
    - 0.3 * max(0, (height - sea_level - 50) / 50)   // high altitude penalty
, 0.0, 1.0)
```

**After T8:** the biome's `moisture_range` and any future `precipitation_rate`
field feed into `base_moisture`, producing biome-aware rainfall patterns.

#### Droplet distribution

Total droplets = `iterations` (config). Each cell's share is proportional to
its rainfall intensity:

```
droplets_for_cell = round(iterations * rainfall[cell] / sum(rainfall))
```

Arid cells (rainfall < 0.05) get zero droplets. High-rainfall cells get
significantly more, concentrating erosion where rain falls.

#### Sediment tracking map

A parallel `Vec<f32>` (same grid as heightmap) accumulates net sediment
deposit per cell. Positive = deposition, negative = erosion. This map is
available during `generate_chunk()` for:

- **Material assignment**: Heavy deposition cells → alluvial Dirt/Clay
  instead of the default surface material
- **Soil depth modulation**: Net deposition thickens the soil layer;
  net erosion thins it or exposes bare rock

#### HydraulicErosionConfig

```rust
pub struct HydraulicErosionConfig {
    pub enabled: bool,                   // Master toggle (default: false)
    pub iterations: u32,                 // Total droplets (default: 100_000)
    pub erosion_radius: f32,             // Deposit/erode kernel radius in cells (default: 3.0)
    pub inertia: f32,                    // Droplet momentum vs gradient (default: 0.05)
    pub sediment_capacity_factor: f32,   // Carrying capacity multiplier (default: 4.0)
    pub min_slope: f32,                  // Floor for capacity calc (default: 0.01)
    pub erosion_rate: f32,               // Fraction of deficit eroded (default: 0.3)
    pub deposition_rate: f32,            // Fraction of excess deposited (default: 0.3)
    pub evaporation_rate: f32,           // Water loss per step (default: 0.01)
    pub gravity: f32,                    // Acceleration factor (default: 4.0)
    pub max_droplet_lifetime: u32,       // Steps before forced stop (default: 30)
    pub rainfall_noise_freq: f64,        // Rainfall pattern frequency (default: 0.003)
    pub rainfall_base: f32,              // Base moisture level (default: 0.5)
    pub altitude_rain_penalty: f32,      // Reduction per metre above threshold (default: 0.006)
    pub altitude_rain_threshold: f32,    // Altitude where penalty starts (default: 50.0 m above sea)
}
```

Embedded in `TerrainConfig` alongside `erosion: ErosionConfig`:

```rust
pub struct TerrainConfig {
    // ... existing fields ...
    pub erosion: ErosionConfig,                      // D8 valley carving
    pub hydraulic_erosion: HydraulicErosionConfig,   // Rainfall-driven droplet erosion
}
```

#### Execution order

Both erosion passes operate on the same coarse heightmap grid (from FlowMap)
and run once during world initialization:

```
1. Sample raw heightmap (noise evaluation)
2. Sink fill (Planchon-Darboux)
3. D8 flow direction + accumulation
4. Valley carving (existing — large-scale channels)
5. ► Hydraulic erosion (NEW — fine-scale rainfall-driven detail)
6. Cache final heightmap + sediment map
7. Per-chunk generation reads cached heightmap
```

Step 5 runs after valley carving so droplets can flow through the carved
channels, depositing sediment in valley floors. The D8 pass creates the river
network; the hydraulic pass adds the gully network and smoothing around it.

#### Performance budget

| Parameter | Value | Cost |
|-----------|-------|------|
| Grid resolution | 512×512 (8 m cells) | — |
| Iterations (droplets) | 100,000 | ~50–150 ms |
| Max lifetime per droplet | 30 steps | 3M step operations worst case |
| Erosion radius | 3 cells | ~28 cells touched per step |

At 100K droplets × 30 steps × ~28 ops = ~84M operations, but most droplets
die early (evaporation, out of bounds, flat terrain). Empirical cost is
~50–150 ms for a 512² grid — comparable to the existing FlowMap computation.
Runs once during world init, fully parallelizable with `rayon` if needed.

#### Preset integration

Each scene preset configures hydraulic erosion intensity:

| Preset | `enabled` | `iterations` | `rainfall_base` | Character |
|--------|-----------|--------------|------------------|-----------|
| `valley_river` | true | 80,000 | 0.6 | Moderate — natural smoothing around rivers |
| `alpine` | true | 120,000 | 0.4 | Aggressive — deep gullies on steep slopes |
| `archipelago` | true | 60,000 | 0.7 | Tropical rain — smooth island terrain |
| `desert_canyon` | false | — | — | No rain — wind erosion not modeled here |
| `rolling_plains` | true | 40,000 | 0.3 | Light — gentle smoothing |
| `volcanic` | true | 100,000 | 0.5 | Moderate — gully erosion on ash slopes |
| `tundra_fjords` | false | — | — | Glacial erosion (different mechanism) |

#### CLI exposure (T4 extension)

```
--hydraulic-erosion <off|light|moderate|heavy>
    off:      disabled
    light:    40K iterations, rainfall_base 0.3
    moderate: 100K iterations, rainfall_base 0.5 (default when enabled)
    heavy:    200K iterations, rainfall_base 0.7
```

#### Testing

- **Unit tests**: Verify droplet moves downhill, deposits on flat, erodes on
  steep. Verify erosion radius kernel sums to 1.0. Verify sediment conservation
  (total eroded ≈ total deposited, within floating-point tolerance).
- **Integration test**: Generate a known slope heightmap (tilted plane), run
  100K droplets. Assert: (a) sediment accumulated at base, (b) heightmap at
  top is lower, (c) total mass conserved.
- **Visual verification**: Run on `valley_river` preset, compare before/after
  heightmap renders. Gullies should be visible branching off the main valley.

#### Files

| File | Action | Description |
|------|--------|-------------|
| `src/world/erosion.rs` | Modify | Add `HydraulicErosionConfig`, `RainfallMap`, `simulate_hydraulic_erosion()`, `SedimentMap` |
| `src/world/terrain.rs` | Modify | Call hydraulic erosion after FlowMap, expose `SedimentMap` for material assignment |
| `src/world/scene_presets.rs` | Modify | Add `HydraulicErosionConfig` to each preset |

---

## Dependency Graph

```
T1 (noise engine) ──→ T2 (continent masks use NoiseStack)
         │
         ├──→ T3 (presets configure NoiseStack params)
         ├──→ T6 (ore noise layers built on noise.rs)
         ├──→ T7 (cave layers built on noise.rs)
         ├──→ T8 (biome terrain modifiers applied to NoiseStack output)
         └──→ T10 (hydraulic erosion runs on improved heightmap)

T3 (presets) ──→ T4 (CLI exposes preset names + overrides)
T4 (CLI) ──→ T5 (world creation UI wraps CLI params)
T6 (geology) ──→ T7 (caves reference rock layer materials)
T8 (biome map) ──→ T9 (surface detail uses biome)
T8 (biome map) ·····→ T10 (optional: biome rainfall data enhances rainfall map)
```

Note: T10 can run before T8 using a noise-based rainfall map. After T8 is
implemented, the rainfall map gains biome-specific moisture data for more
realistic patterns.

## Implementation Order

1. **T1** — Noise engine (foundation for all other tasks)
2. **T3** — Scene presets (immediate user-visible variety)
3. **T4** — Extended CLI (expose controls)
4. **T10** — Hydraulic erosion (rainfall-driven fine-scale detail)
5. **T6** — Rock layers & ores (underground variety)
6. **T7** — Enhanced caves (exploration interest)
7. **T2** — Continent/ocean masks (large-scale geography)
8. **T8** — Biome-terrain integration (biome-aware generation)
9. **T9** — Surface detail (visual polish)
10. **T5** — World creation screen (UI, once all options are stable)

## Files Modified / Created

| File | Action | Description |
|------|--------|-------------|
| `src/world/noise.rs` | **New** | `NoiseStack` with FBM, ridged, selector, warp, micro |
| `src/world/terrain.rs` | Modify | Replace `sample_height()` with `NoiseStack`, extend `generate_chunk()` with strata/surface logic, call hydraulic erosion, read sediment map for material assignment |
| `src/world/erosion.rs` | Modify | Add `HydraulicErosionConfig`, `RainfallMap`, `SedimentMap`, `simulate_hydraulic_erosion()` |
| `src/world/scene_presets.rs` | Modify | Add 6+ presets, each configuring `NoiseStack` + `HydraulicErosionConfig` params |
| `src/world/biome_map.rs` | **New** | Biome map generation and caching |
| `src/world/mod.rs` | Modify | Register `noise` module |
| `src/world_creation.rs` | **New** | World creation UI screen |
| `src/game_state.rs` | Modify | Add `WorldCreation` state |
| `src/main.rs` | Modify | Extended CLI args |
| `src/procgen/biomes.rs` | Modify | Add terrain modifier fields to `BiomeData` |
| `assets/data/materials/*.material.ron` | **New** | 6–8 geological/ore materials |
| `assets/data/biomes/*.biome.ron` | Modify | Add `terrain` section |
