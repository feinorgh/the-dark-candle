# Simulation Test System

A data-driven framework for running repeatable, multi-tick physics and chemistry simulations in a headless voxel grid. Define a scenario in a `.simulation.ron` file, run it with `cargo test`, and assert on the outcome — no game window required.

## Quick Start

1. Create a file in `tests/cases/simulation/` ending with `.simulation.ron`.
2. Run `cargo test --test simulations`.

Every `.simulation.ron` file in that directory is auto-discovered and executed. No Rust code changes are needed.

### Minimal Example

```ron
SimulationScenario(
    name: "My first scenario",
    description: "A sealed stone box with nothing inside. Nothing should happen.",

    grid_size: 6,

    material_source: FromAssets,
    reaction_source: FromAssets,

    regions: [
        Shell(material: "Stone", min: (0, 0, 0), max: (5, 5, 5), thickness: 1),
    ],

    ignition: [],

    ticks: 10,
    dt: 1.0,

    assertions: [
        NoReactions,
    ],
)
```

## Scenario File Reference

A `.simulation.ron` file is a RON-serialized `SimulationScenario` struct. All fields are described below.

### Top-Level Fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | String | yes | — | Human-readable name shown in test output |
| `description` | String | yes | — | What this scenario is testing and why |
| `grid_size` | integer | yes | — | Edge length of the cubic voxel grid (grid is `size³`) |
| `ambient_temperature` | float | no | `288.15` | Initial temperature for all voxels (Kelvin) |
| `ambient_pressure` | float | no | `101325.0` | Initial pressure for all voxels (Pascals) |
| `ambient_schedule` | enum | no | `Constant` | How ambient temperature changes over time (see below) |
| `boundary_htc` | float or None | no | `None` | Convective heat transfer coefficient for air voxels (W/(m²·K)). When set, air voxels are clamped to ambient and a correction flux is applied to adjacent non-air voxels. |
| `material_source` | enum | yes | — | Where to load material definitions (see below) |
| `reaction_source` | enum | yes | — | Where to load reaction rules (see below) |
| `regions` | list | yes | — | Geometry regions applied in order to build the grid |
| `ignition` | list | no | `[]` | Temperature perturbations applied after geometry setup |
| `ticks` | integer | yes | — | Maximum number of simulation ticks to run |
| `dt` | float | yes | — | Timestep per tick in seconds |
| `stop_condition` | enum | no | `FixedTicks` | When to stop (see below) |
| `assertions` | list | yes | — | Checks evaluated after the simulation completes |

> **Tip:** RON supports `//` line comments. Use them liberally — they are stripped at parse time and are a great place to document the science behind your scenario.

### Material and Reaction Sources

Both `material_source` and `reaction_source` accept one of two variants:

```ron
// Load from the project's asset files (most common)
material_source: FromAssets,
reaction_source: FromAssets,

// Self-contained inline definitions (advanced / isolated tests)
material_source: Inline([
    MaterialData(
        id: 99,
        name: "TestFuel",
        default_phase: Solid,
        density: 800.0,
        thermal_conductivity: 0.2,
        specific_heat_capacity: 1200.0,
        // ... other fields have serde defaults
    ),
]),
```

`FromAssets` reads every `*.material.ron` / `*.reaction.ron` file from `assets/data/materials/` and `assets/data/reactions/` respectively. This is the preferred option — it keeps scenarios short, and your test automatically picks up any new materials or reactions added to the game.

Use `Inline(...)` when you need a completely isolated test that must not depend on external data files.

### Ambient Schedule

Controls how the ambient temperature changes over the simulation.

**`Constant`** (default) — Temperature stays at `ambient_temperature` for the entire run.
```ron
ambient_schedule: Constant,
```

**`RampThenHold`** — Linearly ramp from `ambient_temperature` to `end_temperature` over `ramp_seconds`, then hold at `end_temperature` for the rest of the simulation.
```ron
ambient_schedule: RampThenHold(
    end_temperature: 253.15,  // −20°C in Kelvin
    ramp_seconds: 86400.0,    // ramp over 24 hours
),
```

### Convective Boundary (boundary_htc)

At 1 m voxel resolution, pure conduction through air is ~200× too slow compared to real-world convective heat transfer (where h ≈ 10–50 W/(m²·K)). Setting `boundary_htc` corrects this:

1. All air voxels are clamped to the current ambient temperature each tick.
2. Non-air voxels adjacent to air receive a correction flux that boosts the effective heat transfer coefficient from the conduction-only harmonic mean up to the specified `boundary_htc`.

```ron
boundary_htc: Some(25.0),  // 25 W/(m²·K), typical for gentle outdoor air
```

> **Note:** This boundary condition will be replaced by LOD/octree dynamic resolution in the future, which will capture convective effects directly through finer resolution at material interfaces.

### Stop Condition

Controls when the simulation stops.

**`FixedTicks`** (default) — Run for exactly `ticks` iterations, then check assertions.
```ron
stop_condition: FixedTicks,
```

**`UntilAllPass`** — Check assertions every tick. Stop as soon as all pass. `ticks` becomes the maximum cap. Useful for convergence-based scenarios where you don't know how long the process will take.
```ron
stop_condition: UntilAllPass,
```

When `UntilAllPass` converges, the test prints the tick number and equivalent simulated time to stderr.

### Geometry Regions

Regions are applied **in declaration order** — later regions overwrite earlier ones. This enables layered compositions: fill a box, then hollow it, then inject specific materials.

Material names (e.g. `"Stone"`, `"Hydrogen"`) are resolved through the material registry at runtime.

#### Available Region Types

**`Fill`** — Fill a rectangular volume.
```ron
Fill(material: "Hydrogen", min: (1, 1, 1), max: (6, 6, 6))
```

**`Shell`** — Hollow box (only the outer shell of a rectangular volume).
```ron
Shell(material: "Stone", min: (0, 0, 0), max: (7, 7, 7), thickness: 1)
```

**`Single`** — Place exactly one voxel.
```ron
Single(material: "Lava", pos: (4, 4, 4))
```

**`EveryNth`** — Place a material at every Nth position within a volume. Useful for mixing gases in approximate ratios (e.g. oxygen every 3rd voxel in hydrogen ≈ 1:2 ratio).
```ron
EveryNth(material: "Oxygen", min: (1, 1, 1), max: (6, 6, 6), step: 3)
```

**`Checkerboard`** — Alternating pattern where material is placed at positions where `(x + y + z)` is even.
```ron
Checkerboard(material: "Water", min: (1, 1, 1), max: (6, 6, 6))
```

**`Sphere`** — All voxels within a radius of a center point.
```ron
Sphere(material: "Water", center: (8, 8, 8), radius: 5.0)
```

**`Layer`** — Fill an entire horizontal slice at a specific Y level.
```ron
Layer(material: "Sand", y: 3)
```

**`RandomHeightmap`** — Randomized terrain: fills columns from `y_min` to a deterministic random height in `[y_min, y_max]`. Same seed always produces the same terrain.
```ron
RandomHeightmap(
    material: "Stone",
    x_range: (0, 11),
    z_range: (0, 11),
    y_min: 0,
    y_max: 3,
    seed: 42,
)
```

### Ignition (Initial Perturbations)

Ignition entries apply temperature changes **after** geometry is set up. They are optional — scenarios without ignition simply run from ambient conditions.

**`HotSpot`** — Set a single voxel to a specific temperature.
```ron
HotSpot(pos: (3, 3, 3), temperature: 900.0)
```

**`HeatRegion`** — Set an entire rectangular volume to a specific temperature.
```ron
HeatRegion(min: (2, 2, 2), max: (4, 4, 4), temperature: 600.0)
```

### Assertions

Assertions are checked after all ticks complete. A scenario passes only if **every** assertion holds. Failure messages include the actual vs expected values.

#### Material Assertions

| Assertion | Description | Example |
|---|---|---|
| `MaterialCountEq` | Exact count (with tolerance) | `MaterialCountEq(material: "Ash", count: 10, tolerance: 2)` |
| `MaterialCountGt` | At least N voxels of material | `MaterialCountGt(material: "Steam", min_count: 5)` |
| `MaterialCountLt` | At most N voxels of material | `MaterialCountLt(material: "Wood", max_count: 0)` |
| `MaterialAbsent` | Zero voxels of material remain | `MaterialAbsent(material: "Hydrogen")` |

#### Temperature Assertions

| Assertion | Description | Example |
|---|---|---|
| `RegionAvgTempGt` | Average temp in region exceeds threshold | `RegionAvgTempGt(min: (2,2,2), max: (5,5,5), threshold: 500.0)` |
| `RegionAvgTempLt` | Average temp in region is below threshold | `RegionAvgTempLt(min: (1,1,1), max: (4,4,4), threshold: 1000.0)` |
| `VoxelTempGt` | Temperature at a specific voxel exceeds threshold | `VoxelTempGt(pos: (3,3,3), threshold: 800.0)` |
| `MaxTempGt` | Grid-wide peak temperature exceeds threshold | `MaxTempGt(threshold: 1500.0)` |

#### Pressure Assertions

| Assertion | Description | Example |
|---|---|---|
| `RegionAvgPressureGt` | Average pressure in region exceeds threshold | `RegionAvgPressureGt(min: (1,1,1), max: (4,4,4), threshold: 200000.0)` |

#### Reaction Assertions

| Assertion | Description | Example |
|---|---|---|
| `TotalReactionsGt` | Total reactions across all ticks exceeds N | `TotalReactionsGt(min_count: 5)` |
| `NoReactions` | Zero reactions fired (negative test) | `NoReactions` |

## Simulation Tick Loop

Each tick executes these steps in order:

1. **Ambient & boundary conditions** — Compute the current ambient temperature from the schedule. If `boundary_htc` is set, clamp all air voxels to ambient and apply correction flux to adjacent non-air voxels.
2. **Heat diffusion (conduction)** — Fourier's law via `diffuse_chunk`. Thermal energy flows between adjacent voxels based on their conductivity and the timestep `dt`.
3. **Heat transfer (radiation)** — Stefan-Boltzmann radiative exchange via `radiate_chunk`. Hot surface voxels (above 500 K) cast rays in 26 directions to find other surfaces with line-of-sight. Net heat flux flows from hotter to cooler surfaces based on emissivity and view factor. Opaque voxels block radiation. This step is automatic — all scenarios include radiation.
4. **Chemical reactions** — Every voxel is checked against its 6 face-adjacent neighbors. If a reaction rule matches (correct input materials, temperature above threshold), the voxels are transformed and heat is released.
5. **State transitions with latent heat** — Phase changes (melting, boiling, freezing) are checked. If a material defines latent heat (e.g. water's 334,000 J/kg fusion), the temperature is clamped at the phase boundary while energy accumulates in a per-voxel buffer. The transition completes only when the buffer reaches the latent heat threshold. Materials without latent heat transition instantly.
6. **Pressure diffusion** — Gas-phase voxels equalize pressure with their neighbors.

### Choosing `ticks` and `dt`

- **`dt` (timestep)** controls how much simulated time passes per tick. Larger values speed things up but can cause numerical instability. For heat-driven reactions, values in the 100–5000 range work well.
- **`ticks`** is the total number of iterations. More ticks allow the simulation to reach equilibrium. Start with a moderate number (50–200) and increase if assertions fail due to insufficient convergence.

> **Example:** Oxyhydrogen combustion uses `dt: 5000.0` and `ticks: 200`, giving 1,000,000 seconds of simulated time — enough for heat to diffuse through the grid and reactions to propagate.

## Worked Example: Oxyhydrogen Combustion

This scenario models the reaction 2H₂ + O₂ → 2H₂O in a sealed stone chamber.

```ron
// Source: Wikipedia — Oxyhydrogen
//   2 H₂ + O₂ → 2 H₂O
//   Autoignition: 570 °C (843 K)
//   LHV: 241.8 kJ/mol H₂

SimulationScenario(
    name: "Oxyhydrogen combustion in sealed chamber",
    description: "Stoichiometric H₂/O₂ mixture ignited in a stone box.",

    grid_size: 8,
    ambient_temperature: 288.15,
    ambient_pressure: 101325.0,

    material_source: FromAssets,
    reaction_source: FromAssets,

    regions: [
        // Stone walls (hollow box)
        Shell(material: "Stone", min: (0, 0, 0), max: (7, 7, 7), thickness: 1),
        // Fill interior with hydrogen
        Fill(material: "Hydrogen", min: (1, 1, 1), max: (6, 6, 6)),
        // Place oxygen at every 3rd position (≈1:2 O₂:H₂ ratio)
        EveryNth(material: "Oxygen", min: (1, 1, 1), max: (6, 6, 6), step: 3),
    ],

    ignition: [
        // Spark at center, well above autoignition (843 K)
        HotSpot(pos: (3, 3, 3), temperature: 900.0),
    ],

    ticks: 200,
    dt: 5000.0,

    assertions: [
        TotalReactionsGt(min_count: 5),
        MaterialCountGt(material: "Steam", min_count: 5),
        RegionAvgTempGt(min: (2, 2, 2), max: (5, 5, 5), threshold: 300.0),
    ],
)
```

**What's happening:** The grid starts as a hollow stone box filled with hydrogen, with oxygen sprinkled in. A 900 K hot spot at the center exceeds the 843 K autoignition threshold. Heat diffuses outward, triggering chain reactions that convert H₂ + O₂ → Steam. The assertions verify that reactions occurred, steam was produced, and the interior heated up.

**Stoichiometry note:** The reaction system operates on voxel pairs (1:1). To approximate the 2:1 H₂:O₂ stoichiometric ratio, oxygen is placed at every 3rd position using `EveryNth`. This is an intentional simplification — the framework tests emergent behavior from fundamental rules, not exact molecular accounting.

## Worked Example: Radiative Heat Transfer

This pair of scenarios tests Stefan-Boltzmann radiation across an air gap, and validates that opaque walls block it.

### Positive test: radiation across air gap

```ron
SimulationScenario(
    name: "Radiation across air gap",
    description: "A hot stone wall (1200 K) radiates through an air gap to a cold stone target.",

    grid_size: 12,
    ambient_temperature: 288.15,

    material_source: FromAssets,
    reaction_source: FromAssets,

    regions: [
        Shell(material: "Stone", min: (0, 0, 0), max: (11, 11, 11), thickness: 1),
        Layer(material: "Stone", min: (2, 1, 1), max: (3, 10, 10)),   // hot wall
        Layer(material: "Stone", min: (8, 1, 1), max: (8, 10, 10)),   // cold target
    ],

    ignition: [
        RegionHotSpot(min: (2, 1, 1), max: (3, 10, 10), temperature: 1200.0),
    ],

    ticks: 500,
    dt: 1.0,

    assertions: [
        RegionAvgTempGt(min: (8, 5, 5), max: (8, 6, 6), threshold: 295.0),
        NoReactions,
    ],
)
```

**What's happening:** A 2-voxel thick stone wall at 1200 K faces a cold stone slab 4 voxels away across air. Radiation (ε_stone = 0.93) dominates over air conduction at this distance. The cold target warms above 295 K (from ambient 288.15 K) within 500 ticks, confirming long-range heat transfer works.

### Negative test: wall blocks radiation

The companion scenario `radiation_blocked_by_wall.simulation.ron` inserts an opaque stone wall at x=6 between the emitter and target. Assertions verify the target stays *below* 295 K (radiation blocked), while the blocking wall itself warms slightly above 289 K (it absorbs some radiation from the emitter side).

**Design notes:**
- 1200 K is well below stone's 1473 K melting point, avoiding phase transitions that would complicate the test.
- The 500 K emission threshold means only the hot wall participates as a radiation source. The blocking wall (< 500 K) absorbs but never re-radiates to the target.
- Conservative threshold of 295 K (just ~7 K above ambient) ensures the test passes reliably despite floating-point approximations.

## Writing Your Own Scenarios

### Step-by-Step

1. **Pick a phenomenon** you want to test (combustion, heat diffusion, phase change, etc.).
2. **Design the geometry.** What materials go where? Use `Shell` for containment, `Fill`/`EveryNth` for reactants, `Single` for probes.
3. **Set ignition conditions.** Does the reaction need a spark? A heated region? Or will it proceed at ambient?
4. **Choose simulation parameters.** Start with `ticks: 100, dt: 100.0` and adjust.
5. **Write conservative assertions.** Test for qualitative outcomes ("reactions happened", "temperature increased") before adding quantitative thresholds.
6. **Iterate.** Run `cargo test --test simulations` and refine thresholds based on actual output.

### Negative Tests

Always pair positive tests with negative ones. If you test "wood burns when ignited", also test "wood does not burn below ignition temperature". Use `NoReactions` for this:

```ron
SimulationScenario(
    name: "Wood below ignition temperature",
    description: "Wood + Air at room temperature should not combust.",

    grid_size: 4,
    material_source: FromAssets,
    reaction_source: FromAssets,

    regions: [
        Fill(material: "Wood", min: (0, 0, 0), max: (1, 1, 1)),
    ],

    ignition: [],   // no heat applied

    ticks: 50,
    dt: 1.0,

    assertions: [
        NoReactions,
    ],
)
```

### Tips

- **Material names are case-sensitive** and must match the `name` field in the corresponding `.material.ron` file exactly (e.g. `"Stone"`, not `"stone"`).
- **Coordinates are `(x, y, z)` starting at `(0, 0, 0)`.** The grid runs from `(0, 0, 0)` to `(grid_size - 1, grid_size - 1, grid_size - 1)`.
- **Region order matters.** `Fill` then `EveryNth` will overwrite some filled voxels. `EveryNth` then `Fill` will overwrite the placed pattern.
- **All units are SI.** Temperature in Kelvin, pressure in Pascals, time in seconds. See the project's SI unit reference in `AGENTS.md`.
- **Keep grids small** for fast tests. An 8³ grid (512 voxels) is plenty for most chemical scenarios. Use larger grids only when testing spatial propagation over distance.

## Running Tests

```bash
# Run all simulation scenarios
cargo test --test simulations

# Run all tests (simulations + unit tests + other integration tests)
cargo test

# Run with output visible (useful when debugging assertion failures)
cargo test --test simulations -- --nocapture
```

## Adding New Assertion Types

To add a new assertion variant:

1. Add a variant to the `Assertion` enum in `src/simulation/assertions.rs`.
2. Implement its evaluation in the `evaluate()` function in the same file.
3. It becomes immediately usable in RON files — no registration or wiring needed.

The enum derives `serde::Deserialize`, so RON parsing is automatic.

## Adding New Region Types

To add a new geometry builder:

1. Add a variant to the `Region` enum in `src/simulation/geometry.rs`.
2. Handle it in the `apply_one()` function in the same file.
3. Add a unit test.

## Architecture Overview

```
src/simulation/
├── mod.rs           Shared tick loop (simulate_tick, TickResult, SimulationStats)
├── geometry.rs      Region enum + apply_regions()
├── assertions.rs    Assertion enum + evaluate()
└── scenario.rs      SimulationScenario struct + run_scenario()

tests/
├── simulations.rs               Auto-discovery test entrypoint
└── cases/simulation/
    ├── oxyhydrogen_combustion.simulation.ron
    ├── inert_gas_no_reaction.simulation.ron
    ├── water_freezing.simulation.ron
    ├── radiation_across_air_gap.simulation.ron
    └── radiation_blocked_by_wall.simulation.ron
```

The simulation harness (`simulate_tick`) is also used by `src/chemistry/fire_propagation.rs` unit tests, ensuring consistency between inline tests and data-driven scenarios.
