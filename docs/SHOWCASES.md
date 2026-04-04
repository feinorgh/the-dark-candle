# Simulation Showcase Scenarios

Headless simulation scenarios that prove the physics and chemistry engines are
correct. Run with:

```
cargo test --test simulations
```

Each `.simulation.ron` file in `tests/cases/simulation/` is a self-contained
scenario: it defines a voxel grid, initial conditions, an optional ambient
schedule, and assertions that must hold at the end (or on convergence).

---

## Thermal Physics

### `thermal_conduction_gradient`
**Fourier's law — heat flows down the temperature gradient.**

A superheated iron slab (1200 K, k = 80.2 W/(m·K)) sits at the left of a
12³ stone grid. After 50 s the iron stays hot (> 700 K average) while the far
right stone stays below 700 K, proving the temperature gradient emerges from
Fourier conduction without any explicit "spread" logic.

### `thermal_insulation_comparison`
**Material conductivity determines insulation quality.**

Three parallel cores of iron at 1000 K, each separated from outer stone by a
different 1-voxel wall: iron (k = 80.2), thatch (k = 0.06), and air
(k = 0.026). After 50 s, all three iron cores remain hot (> 800 K), verifying
that the conduction model respects real material conductivities.

### `heat_capacity_comparison`
**High ρCp materials change temperature more slowly.**

Two voxel pairs are heated to 1000 K. Sand (low ρCp ≈ 1.6 MJ/(m³·K)) cools
faster than stone (high ρCp ≈ 2.2 MJ/(m³·K)). After 500 ticks the sand region
is cooler than the stone region, verifying the volumetric heat capacity model.

### `stefan_boltzmann_radiation_vacuum`
**T⁴ radiative transfer across an air gap.**

A 1400 K iron sphere sits inside a stone-walled box. Air fills the gap
(k ≈ 0.026 W/(m·K) — negligible conduction). After 1000 ticks × 0.5 s the
outer stone walls have measurably warmed above the 293.15 K ambient, proving
Stefan-Boltzmann radiation (∝ T⁴) carries heat across the effectively-vacuum
gap.

### `radiation_across_air_gap`
**Radiation heats a receiver across a gap.**

Hot iron emits radiation that warms a stone receiver on the opposite side of
an air cavity, verifying the radiate_chunk implementation.

### `radiation_blocked_by_wall`
**Interposed stone attenuates radiation.**

Same geometry as above but with a stone wall inserted between source and
receiver. The receiver warms less (or not at all), verifying that solid voxels
block/absorb radiation before it reaches the far side.

### `thermal_glow_visual`
**MP4 heatmap of a glowing iron sphere.**

Generates `test_output/thermal_glow_heatmap.mp4` — a frame-by-frame
colour-mapped video of temperature as the iron sphere cools via radiation and
conduction. Validates the video-encoder path and provides a visual asset.

---

## Phase Transitions & Latent Heat

### `ice_melting_latent_heat`
**273.15 K temperature plateau while ice absorbs fusion latent heat.**

A 2.5-radius sphere of ice at 253.15 K sits in warm air (283.15 K, htc = 25).
The ice warms to 273.15 K and then *stalls* while absorbing L_f = 334 000 J/kg
of latent heat before finally melting to water. Uses dt = 60 000 s to span the
large thermal mass of 1 m³ voxels.

### `steam_condensation`
**Steam below 373.15 K condenses back to liquid water.**

A sealed stone box full of 450 K steam cools toward ambient (293.15 K) via
conduction through stone walls. Once surface steam drops below the boiling
point (373.15 K) it instantly condenses to water. Eventually all steam
condenses, verifying the `condensed_into` state transition path.

### `lava_cooling_to_stone`
**Molten lava solidifies at 1473 K via Stefan-Boltzmann cooling.**

A lava sphere at 1600 K radiates heat to surrounding stone walls. Once the
lava drops below 1473 K, the `frozen_into: "Stone"` transition fires. All lava
solidifies within ~2 simulated days, verifying the liquid-phase solidification
path.

### `stone_melting_to_lava`
**Stone above 1473 K melts into lava.**

A stone block is force-heated above the melting threshold. Stone voxels
transition to lava via `melted_into`, verifying the solid→liquid path for
geological materials.

### `quartz_melting_to_glass`
**Quartz melts to glass at 1923 K.**

Quartz is heated above 1923 K; all voxels transition to Glass via
`melted_into`. Verifies a second solid→liquid→glass material chain.

### `water_full_phase_cycle`
**Ice → Water → Steam driven by a rising ambient temperature.**

A 4³ ice block at 243.15 K sits in a 6³ open grid with boundary_htc = 50.
Ambient ramps from 243 K to 450 K over 250 000 s. Surface ice melts first;
inner ice melts by conduction through the growing water shell; surface water
eventually boils to steam. The UntilAllPass stop condition exits once
`MaterialAbsent("Ice")` AND `MaterialCountGt("Steam", 1)` both hold
simultaneously, proving all three transitions occurred.

### `water_freezing`
**Water freezes at 273.15 K — the reverse melt path.**

Liquid water at ambient cools below the freezing point; `frozen_into: "Ice"`
fires, verifying the liquid→solid path for water.

---

## Chemical Reactions

### `oxyhydrogen_combustion`
**2 H₂ + O₂ → 2 H₂O — chain reaction in a sealed chamber.**

A checkerboard of H₂ and O₂ voxels fills the interior of a stone box. A
900 K HeatRegion (above the 843 K autoignition point) triggers all H₂ voxels
simultaneously. Over 30 ticks the reactions fire, steam is produced, and the
interior temperature rises well above ambient. Verifies `TotalReactionsGt(5)`,
`MaterialCountGt("Steam", 5)`, and `RegionAvgTempGt(300 K)`.

### `inert_gas_no_reaction`
**N₂ + O₂ at high temperature — no reaction fires.**

Nitrogen (N₂) and oxygen are mixed in a hot chamber. Despite the elevated
temperature, no reaction is defined for this pair, so `NoReactions` passes.
Verifies the reaction engine only fires registered reactions.

### `wood_pyrolysis_charcoal_combustion`
**O₂-dependent vs thermal-only combustion paths.**

A 4³ wood block is heated to 700 K. Surface voxels (adjacent to air) fire
`wood_combustion` (573 K threshold, requires O₂) → Ash. Interior voxels (no
air access) fire `wood_pyrolysis` (473 K threshold, no O₂ required) →
Charcoal. Both products are asserted simultaneously in tick 1, demonstrating
that the reaction engine correctly selects the appropriate pathway based on
neighbour availability.

### `wildfire_propagation`
**Three fuel types combust at their respective ignition temperatures.**

A fuel bed of DryLeaves (453 K), Grass (533 K), and Twig (473 K) is ignited at
700 K. Each material fires its correct combustion reaction against an adjacent
Air voxel. DryLeaves and Grass produce Air (no residue); Twig produces Ash.
Over 100+ reactions, all fuel is consumed in a single tick.

### `multi_layer_fuel_combustion`
**Bark → Charcoal → Ash two-tick chain with air gaps.**

Three fuel layers (Charcoal y=2, Wood y=4, Bark y=6) are separated by air
gaps (y=3, y=5) and ignited simultaneously at 650 K.

- **Tick 1:** Charcoal + Air → Ash; Wood + Air → Ash; Bark + Air → Charcoal
  (bark's combustion product is charcoal, at 1350 K)
- **Tick 2:** New charcoal (from Bark) + Air → Ash

After tick 2: no Bark, no Wood, no Charcoal — only Ash remains. Verifies the
two-tick sequential combustion chain and the importance of air-gap geometry.

---

## Notes for Contributors

- All scenarios use SI units: 1 voxel = 1 m, temperatures in Kelvin.
- Thermal diffusion at 1 m scale is physically slow. Use large `dt` values
  (e.g. 60 000 s) when the scenario involves bulk thermal mass, and
  `UntilAllPass` to exit early once convergence is reached.
- At 1 m SI scale, heat cannot propagate voxel-to-voxel fast enough for
  fire to spread in a physically reasonable tick count. Scenarios that want to
  test *all* fuel combusting use `HeatRegion` ignition rather than a single
  `HotSpot`.
- `NoReactions` assertions confirm that a thermal-only scenario doesn't
  accidentally trigger chemistry.
- See `docs/simulation-test-system.md` for the full assertion and geometry
  reference.
