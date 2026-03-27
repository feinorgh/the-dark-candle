# Phase 11 — Buildings & Structural Construction (planned)

A freeform building system where players construct structures from physical
materials — wood, stone, metal, glass, concrete, clay, and more — all defined as
data in RON files. Building parts attach to each other and to the terrain through
joints that transmit forces. Structures interact with the full physics stack:
gravity loads them, wind pushes them, fire burns them, explosions shatter them.
Failure is emergent — buildings collapse when stress exceeds material strength,
not from scripted destruction.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[entity-bodies.md](entity-bodies.md) (Phase 10, articulated bodies),
[advanced-physics.md](advanced-physics.md) (coupling, constraints, explosions).

---

## Foundations already in place

| System | Location | Provides |
|--------|----------|----------|
| Material properties | `data/mod.rs`, `assets/data/materials/` | Density, hardness, Young's modulus, friction, restitution, thermal properties, combustion, phase transitions (12 materials) |
| Structural integrity | `integrity.rs` | Flood-fill connectivity from anchored voxels, unsupported collapse |
| Voxel subdivision | `octree.rs`, `refinement.rs` | SVO with adaptive refinement at damage gradients, material boundaries |
| Chemistry | `chemistry/` | Heat diffusion, combustion reactions, state transitions (wood burns, stone melts, ice melts) |
| Item system | `procgen/items.rs`, `data/mod.rs` | Item templates with material-derived weight/durability, 4 items defined |
| Physics | `gravity.rs`, `pressure.rs`, `collision.rs` | Force-based entity physics, pressure propagation |

## Design

### 1. Extended material properties

`MaterialData` already has density, hardness, Young's modulus (stiffness),
friction, and restitution. Add structural strength properties (all in Pascals):

| Property | Unit | Description | Example: Wood | Example: Iron |
|----------|------|-------------|---------------|---------------|
| `tensile_strength` | Pa | Max stress before fracture under tension | 40 MPa | 400 MPa |
| `compressive_strength` | Pa | Max stress before crushing | 30 MPa | 250 MPa |
| `shear_strength` | Pa | Max stress before shearing | 8 MPa | 170 MPa |
| `flexural_strength` | Pa | Max bending stress before snapping | 50 MPa | 350 MPa |
| `fracture_toughness` | Pa·√m | Resistance to crack propagation | 10 MPa·√m | 50 MPa·√m |

These are optional fields on `MaterialData` (existing materials get real-world
values added to their `.material.ron` files). Materials without strength values
(air, water, steam) cannot be used as structural elements.

### 2. Building parts

Building parts are shapes made from a single material, placed by the player.
Each part type is defined in a `.part.ron` file:

- **Block** — 1×1×1 m solid cube. The basic unit.
- **Slab** — 1×1×0.5 m half-height. Floors, shelves.
- **Beam** — 0.25×0.25×N m elongated member. Structural frames.
- **Column** — 0.5×N×0.5 m vertical support. Load-bearing pillars.
- **Wall** — 1×N×0.1 m thin panel. Partitions, facades.
- **Arch** — curved shape with keystone geometry. Bridges, doorways.
- **Stair** — stepped wedge. Vertical traversal.
- **Roof** — angled slab. Water shedding, shelter.

Part definitions specify voxel occupancy (which sub-voxels are filled),
material slot (which material it's made of), and attachment faces (where
other parts can connect). The octree subdivision system represents parts
at sub-voxel resolution — a 0.25 m beam uses depth-2 subdivision within
its host voxel.

### 3. Attachment & joints

Parts connect at shared faces. Each joint has:

- **Contact area** (m²) — derived from overlapping face geometry.
- **Joint strength** — `min(material_A_strength, material_B_strength) × contact_area`.
  Uses the weakest of tensile/compressive/shear depending on the load direction.
- **Joint type** — rigid (mortar, welding, nails), friction (dry-stacked stone),
  or hinge (door, gate). Type affects which stress modes the joint resists.

Attachment rules:
- Parts snap to a grid aligned with the voxel coordinate system.
- Any two parts with adjacent filled sub-voxels form a joint automatically.
- The player can upgrade joints (apply mortar to stone, nail wood, weld metal)
  to increase strength.
- Terrain voxels act as anchor points (infinite compressive strength, like
  bedrock in the current integrity system).

### 4. Structural analysis

Upgrade the existing flood-fill integrity system (`integrity.rs`) to a
force-based stress analysis:

- **Load path tracing.** From every part, trace the path gravity forces
  take through joints to the ground. Each joint accumulates the load it
  carries.
- **Stress calculation.** At each joint: `σ = F / A`. Compare against the
  relevant material strength (compressive for columns, tensile for hanging
  loads, shear for lateral forces, flexural for beams).
- **Wind loading.** Exposed surfaces receive force from atmospheric pressure
  gradients (Phase 9 LBM wind field). Tall/wide structures accumulate more
  wind load.
- **Dynamic loads.** Impacts (explosions, projectiles, falling debris)
  apply impulse forces. Joints that exceed their strength break.
- **Progressive collapse.** When a joint breaks, load redistributes to
  neighboring joints. If they also fail → cascade → realistic structural
  failure. Uses the existing damage field on `Voxel` (0.0 = destroyed,
  1.0 = intact) to track degradation.
- **Creep & fatigue** (future). Long-term loads near the strength limit
  gradually degrade joints. Wooden structures rot, metal corrodes (ties
  into chemistry system).

The analysis runs on `FixedUpdate` at a budget-limited frequency (not
every tick — every N ticks or when loads change). The flood-fill fallback
remains for chunks without active structures (performance).

### 5. Building materials

Expand the material library with construction-specific materials. Each is a
new `.material.ron` file with full SI properties:

| Material | Density | Compressive | Tensile | Notes |
|----------|---------|-------------|---------|-------|
| Oak wood | 600 kg/m³ | 30 MPa | 40 MPa | Burns, biodegrades |
| Pine wood | 500 kg/m³ | 25 MPa | 35 MPa | Lighter, weaker |
| Granite | 2700 kg/m³ | 130 MPa | 7 MPa | Strong in compression, weak in tension |
| Limestone | 2300 kg/m³ | 60 MPa | 4 MPa | Sofite, carvable |
| Brick | 1900 kg/m³ | 20 MPa | 2 MPa | Requires mortar joints |
| Concrete | 2400 kg/m³ | 40 MPa | 3 MPa | Very weak in tension |
| Wrought iron | 7700 kg/m³ | 250 MPa | 350 MPa | Strong in tension |
| Bronze | 8800 kg/m³ | 200 MPa | 300 MPa | Corrosion resistant |
| Copper | 8900 kg/m³ | 70 MPa | 210 MPa | Ductile, conducts heat |
| Glass | 2500 kg/m³ | 1000 MPa | 33 MPa | Brittle, transparent |
| Clay (dried) | 1800 kg/m³ | 3 MPa | 0.5 MPa | Weak, cheap, fire-hardens to brick |
| Thatch | 240 kg/m³ | 0.5 MPa | 1 MPa | Insulating, burns easily |

These interact with existing systems: wood burns (combustion reactions),
stone melts to lava, ice melts to water, metals conduct heat efficiently.

### 6. Player building mechanics

- **Placement mode.** Player enters build mode, selects part type + material.
  Ghost preview shows placement on the grid. Snap to adjacent parts or terrain.
- **Rotation.** Parts rotate in 90° increments around any axis.
- **Demolition.** Player can remove parts they placed. Removed parts drop as
  items (or break into debris if damaged).
- **Material sourcing.** Building requires material items in inventory. Mining
  terrain yields raw materials (stone, dirt, sand). Crafting converts raw
  materials into construction materials (wood → planks, clay + fire → bricks,
  sand + heat → glass, ore + smelting → metal ingots).
- **Crafting recipes.** Defined in `.recipe.ron` files. Input materials +
  tool requirements + processing (heat, time) → output material/part.

### 7. Physics interactions

Structures are not separate from the world — they are voxels with materials
and joints, subject to all existing physics:

- **Gravity.** Structures bear their own weight. Overhangs need support.
  Apparent gravity from Phase 8 determines load direction everywhere on the
  sphere.
- **Fire.** Wooden structures burn via existing combustion reactions. Fire
  weakens joints (temperature degrades material strength). Stone/metal
  structures survive fire but conduct heat.
- **Explosions.** Pressure waves from `pressure.rs` apply impulse to
  structural surfaces. Joints near the blast fail → debris.
- **Fluid interaction.** Rising water (AMR) exerts buoyancy and hydrostatic
  pressure on submerged walls. Wind (LBM) applies lateral force.
- **Erosion.** FLIP/PIC particles (rain, sand) erode exposed surfaces over
  time, reducing `Voxel.damage`.

## Implementation Steps

1. **Extended `MaterialData`** — add `tensile_strength`, `compressive_strength`,
   `shear_strength`, `flexural_strength`, `fracture_toughness` (all `Option<f32>`
   in Pa / Pa·√m) to `MaterialData` struct. Update all 12 existing
   `.material.ron` files with real-world values. Validate in tests.

2. **`PartData` and part RON files** — new `src/building/parts.rs`. `PartData`
   struct: name, voxel shape (occupancy mask at subdivision depth),
   attachment faces, material slot. Create `assets/data/parts/*.part.ron`
   for block, slab, beam, column, wall, arch, stair, roof. Register via
   `RonAssetPlugin<PartData>`.

3. **Joint system** — new `src/building/joints.rs`. `Joint` component linking
   two adjacent parts. Computed contact area, type (rigid/friction/hinge),
   current stress, damage accumulation. Joints are created automatically
   when parts are placed adjacent to each other.

4. **Structural stress analysis** — new `src/building/stress.rs`. Replace
   flood-fill integrity with load-path analysis for chunks containing
   building parts. Compute stress per joint from gravity + external loads.
   Break joints exceeding material strength. Progressive collapse via
   load redistribution. Keep flood-fill fallback for non-building chunks.

5. **New construction materials** — add `.material.ron` files for granite,
   limestone, brick, concrete, wrought iron, bronze, copper, glass, dried
   clay, thatch, oak, pine, planks. Full SI properties. New `MaterialId`
   constants where needed.

6. **Crafting recipe system** — new `src/building/crafting.rs` +
   `assets/data/recipes/*.recipe.ron`. `RecipeData`: input materials,
   quantities, tool requirements, processing conditions (temperature,
   duration), output material/part. `RonAssetPlugin<RecipeData>`.

7. **Player placement system** — new `src/building/placement.rs`. Build-mode
   toggle, ghost preview, grid snapping, rotation, placement validation
   (support check, material availability), part spawning with joint creation.

8. **Demolition & debris** — part removal drops items or breaks into physics
   debris (voxel fragments with `PhysicsBody`). Debris inherits material
   and velocity from the destroyed part.

9. **Physics integration** — wire structural analysis into `FixedUpdate`.
   Connect wind loading (Phase 9 LBM pressure field), hydrostatic pressure,
   explosion impulse, fire damage (temperature-dependent strength reduction)
   to the stress system.

10. **Inventory system** — new `src/entities/inventory.rs`. Per-entity item
    storage with weight/volume limits. Material items for building. UI
    integration (future).

## Dependencies

- Steps 1, 5 extend the existing material system (no phase dependency).
- Steps 2–4, 6–8 can begin after step 1 (material properties).
- Step 9 integrates with Phase 8 (radial gravity) and Phase 9 (wind loading).
- Step 7 requires step 10 (inventory) for material consumption.
- Step 4 (stress analysis) benefits from Phase 8 (radial gravity direction)
  but can initially use the current -Y gravity.

## What stays unchanged

Voxel storage, chunk management, meshing pipeline, octree structure (used by
parts for sub-voxel resolution), chemistry system (fire/heat/reactions apply
to building materials automatically), existing material RON files (extended,
not replaced), existing item system (extended with building items).
