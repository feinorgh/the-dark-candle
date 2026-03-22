# The Dark Candle — Architecture & Implementation Plan

## Vision

A 3D first-person procedural simulation game built on Bevy 0.18. The world is a
voxel grid rendered as smooth meshes (Marching Cubes / Surface Nets). Every voxel
carries material and physical state, enabling deep simulation of physics, chemistry,
biology, and social systems. Gameplay emerges from the interaction of independent
simulation layers — no scripted events, no hardcoded behaviors.

**Core principles:**
- **Data-driven:** All entity definitions, material properties, reaction rules, and
  biome parameters live in `.ron` files under `assets/data/`.
- **Layered independence:** Each simulation layer (physics, chemistry, biology, social)
  reads/writes shared ECS components but knows nothing about the layers above it.
- **Non-deterministic emergence:** Seeded RNG per system with entropy injection.
  Reproducible for debugging, chaotic for gameplay.
- **Voxel-native simulation:** The voxel grid *is* the simulation grid. Physics,
  chemistry, and biology operate on voxel data directly.

---

## Current State

**Phases 0 through 5 are complete.** The project has:
- Modular architecture: 10 Bevy plugins across camera, world, physics, chemistry,
  biology, entities, procgen, behavior, social, data
- 3D first-person camera with WASD, mouse look, gravity, jumping, fly-mode toggle
- Chunked voxel world (32³ per chunk) with cylindrical chunk loading around camera
- Layered Perlin noise terrain generation (continental + detail + cave carving)
- Surface Nets mesh rendering with per-material vertex colors
- Data-driven material system: 12 materials loaded from `.material.ron` files
  (air, stone, dirt, water, iron, wood, sand, grass, ice, steam, lava, ash)
- Heat diffusion system (discrete conduction between neighboring voxels)
- Chemical reaction system with RON-defined rules (wood/grass combustion)
- State transitions: melting, boiling, freezing, condensation driven by MaterialData
- MaterialRegistry for runtime MaterialId → MaterialData lookups
- Fire propagation validated: heat + flammable material + air → chain combustion
- Physics: entity gravity, AABB collision, fluid simulation, structural integrity, gas pressure
- CreatureData + ItemData RON structs with loader plugins
- Procedural creature generation (stat variation, color jitter, deterministic seeding)
- Procedural item generation (material properties → emergent weight/durability/damage)
- Biome system: 4 biomes (forest, meadow, cave, tundra) with spawn tables
- Creature spawning: deterministic per-chunk planning from biome data
- Biology: metabolism (energy/starvation), health (damage types, status effects, healing),
  growth (juvenile→adult→elder), death/decomposition (corpse voxels → decay → nutrients),
  plant growth (grass spreads to dirt with light/water)
- CI/CD pipeline (GitHub Actions), pre-commit hooks, 235 tests passing
- Cross-compilation configured for Windows (`x86_64-pc-windows-gnu`)

---

## Target Module Structure

```
src/
├── main.rs                  App entry, plugin registration
├── camera/
│   └── mod.rs               3D camera controller (first-person)
├── world/
│   ├── mod.rs               WorldPlugin, chunk management
│   ├── voxel.rs             Voxel type, MaterialId, VoxelState
│   ├── chunk.rs             Chunk storage, dirty tracking, neighbor access
│   ├── terrain.rs           Terrain generation (noise → voxels)
│   └── meshing.rs           Marching Cubes / Surface Nets mesh generation
├── physics/
│   ├── mod.rs               PhysicsPlugin
│   ├── forces.rs            Gravity, buoyancy, pressure
│   └── collision.rs         Voxel-based collision detection
├── chemistry/
│   ├── mod.rs               ChemistryPlugin
│   ├── material.rs          Material definitions, properties (RON-loaded)
│   ├── reactions.rs         Reaction rules and processing (RON-loaded)
│   └── heat.rs              Heat transfer / state transitions
├── biology/
│   ├── mod.rs               BiologyPlugin
│   ├── metabolism.rs         Hunger, energy, health
│   ├── growth.rs            Growth, reproduction, aging
│   └── decay.rs             Death, decomposition → chemistry
├── entities/
│   ├── mod.rs               EntityPlugin, spawning
│   ├── creature.rs          Creature components and data structs
│   └── item.rs              Item components and data structs
├── procgen/
│   ├── mod.rs               ProcgenPlugin, shared utilities
│   ├── noise.rs             Noise functions (Perlin, Simplex, Worley)
│   ├── biome.rs             Biome definitions and selection (RON-loaded)
│   └── names.rs             Procedural name generation
├── behavior/
│   ├── mod.rs               BehaviorPlugin
│   ├── needs.rs             Needs hierarchy (hunger, safety, curiosity)
│   └── decisions.rs         Decision-making / utility AI
├── social/
│   ├── mod.rs               SocialPlugin
│   ├── relationships.rs     Entity-to-entity relationships
│   └── factions.rs          Group dynamics, reputation
└── data/
    └── mod.rs               All RON-loadable asset structs
```

```
assets/data/
├── materials/               Material definitions (.material.ron)
├── reactions/               Chemical reaction rules (.reaction.ron)
├── biomes/                  Biome parameters (.biome.ron)
├── creatures/               Creature templates (.creature.ron)
├── items/                   Item templates (.item.ron)
└── factions/                Faction definitions (.faction.ron)
```

---

## Simulation Layers

### Layer 0 — World (Voxel Grid & Terrain)

The foundation. A chunked 3D voxel grid where each voxel holds:
- `MaterialId` — what substance this voxel is (stone, water, air, wood, etc.)
- `temperature: f32` — in Kelvin
- `pressure: f32` — ambient pressure
- `damage: f32` — structural integrity loss

Chunks are 32³ voxels. Only dirty chunks re-mesh. Terrain is generated via layered
noise (continent → mountain → cave → ore distribution) driven by biome RON files.

Mesh generation uses Surface Nets (smoother than Marching Cubes, cheaper to compute,
good for organic terrain).

### Layer 1 — Physics & Chemistry

**Physics:** Gravity, fluid flow (cellular automata on voxels), pressure propagation,
structural integrity (unsupported voxels collapse). Operates per-chunk, parallelized.

**Chemistry:** Materials have properties (flammability, melting point, reactivity)
loaded from RON. Reaction rules (also RON) define what happens when materials meet
under conditions (temperature, pressure). Examples:
- Wood + Fire + O₂ → Charcoal + CO₂ + Heat
- Water + Heat(>373K) → Steam
- Iron + Water + O₂ → Rust (slowly)

Heat transfer is a diffusion system across neighboring voxels.

### Layer 2 — Biology

Entities with biological components: metabolism (energy budget from food → actions),
health (damage, healing, disease), growth (aging, size changes), reproduction.

Death feeds back into chemistry: corpses become organic material voxels that decay,
release nutrients, feed plants. The cycle is closed.

### Layer 3 — Behavior & Social

Utility AI: each entity evaluates actions based on weighted needs (hunger, safety,
curiosity, social). No behavior trees, no scripted sequences — just needs + world
state → action selection.

Social layer tracks relationships (trust, hostility, kinship) and faction reputation.
Groups form, splinter, and conflict based on shared resources and history.

---

## Implementation Phases

### Phase 0 — Restructure & 3D Foundation
Migrate from single-file 2D to modular 3D. This is pure scaffolding.

- **0.1** Restructure `main.rs` into module layout (move EnemyData to `data/`,
  create plugin stubs for each module)
- **0.2** Replace Camera2d with a 3D first-person camera + basic input (WASD +
  mouse look)
- **0.3** Add placeholder 3D scene (ground plane, lighting) to verify the 3D
  pipeline works

### Phase 1 — Voxel World & Terrain
The world must exist before anything can live in it.

- **1.1** Define `Voxel`, `MaterialId`, `VoxelState` types
- **1.2** Implement `Chunk` (32³ voxel storage, dirty flag, neighbor references)
- **1.3** Implement chunk manager (load/unload chunks around camera, LOD stub)
- **1.4** Terrain generation: layered noise → voxel fill per chunk
- **1.5** Mesh generation: Surface Nets algorithm, generate `Mesh` from chunk data
- **1.6** Integrate: walk around a procedurally generated 3D voxel world

### Phase 2 — Materials & Chemistry
Give voxels physical meaning.

- **2.1** Define `MaterialData` RON struct (density, melting/boiling points,
  flammability, hardness, color)
- **2.2** Create material RON files (stone, dirt, water, air, wood, iron, etc.)
- **2.3** Material-aware mesh coloring (vertex colors or texture atlas from material)
- **2.4** Heat transfer system (diffusion across voxel neighbors per tick)
- **2.5** State transitions (melting, boiling, freezing based on temperature)
- **2.6** Chemical reaction system: RON-defined rules, condition matching, product
  generation
- **2.7** Fire propagation as an emergent test case (heat + flammable + O₂ → fire)

### Phase 3 — Physics
Make the world physically coherent.

- **3.1** Gravity for entities (not voxels initially — sand/fluid later)
- **3.2** Voxel-based collision for entities (AABB vs voxel grid)
- **3.3** Fluid simulation: cellular automata water flow (simplified Navier-Stokes
  on voxel grid)
- **3.4** Structural integrity: unsupported blocks collapse (flood-fill connectivity
  check)
- **3.5** Pressure system for gases (explosion propagation, ventilation)

### Phase 4 — Entities & Procedural Generation
Populate the world.

- **4.1** Define `CreatureData` RON struct (species, base stats, body plan, diet)
- **4.2** Define `ItemData` RON struct (type, material, properties)
- **4.3** Procedural creature generation from templates + variation
- **4.4** Procedural item generation (material + form → properties)
- **4.5** Biome RON files controlling spawn tables, density, variation
- **4.6** Creature spawning system driven by biome data

### Phase 5 — Biology
Give creatures life.

- **5.1** Metabolism system: energy from food, energy spent on actions
- **5.2** Health system: damage types, healing, disease (interaction with chemistry)
- **5.3** Growth/aging: size changes, stat progression, lifespan
- **5.4** Death → decomposition: entity removal, corpse voxels placed, decay timer
- **5.5** Plant growth: voxels that spread based on light/water/nutrients

### Phase 6 — Behavior & AI
Give creatures agency.

- **6.1** Needs system: hunger, safety, rest, curiosity, social (weighted floats)
- **6.2** Action evaluation: utility scores per available action
- **6.3** Pathfinding on voxel grid (A* or JPS adapted for 3D voxels)
- **6.4** Basic behaviors: wander, eat, flee, sleep, follow
- **6.5** Perception system: what can this entity see/hear/smell?

### Phase 7 — Social Systems
Give creatures relationships.

- **7.1** Relationship tracking: entity↔entity (trust, familiarity, hostility)
- **7.2** Faction system: groups with shared identity, territory, goals
- **7.3** Reputation: actions observed by others modify relationship values
- **7.4** Group behaviors: cooperative hunting, territory defense, trade

---

## Dependencies to Add (by phase)

| Phase | Crate | Purpose |
|-------|-------|---------|
| 1 | `noise` or `fastnoise-lite` | Noise generation for terrain |
| 1 | (custom) | Surface Nets implementation |
| 3 | `bevy_rapier3d` (evaluate) | Rigid-body physics for entities (or hand-roll voxel collision) |
| 6 | (custom or `pathfinding`) | A* / JPS pathfinding |

Bevy 0.18's built-in systems handle rendering, input, asset loading, scheduling, and
ECS. Most simulation logic is custom — this is intentional to keep full control over
the simulation tick.

---

## Key Design Decisions

1. **No ECS bundles** — Bevy 0.18 deprecated them. All entities spawn as component tuples.
2. **RON for all data** — Materials, reactions, biomes, creatures, items, factions.
   YAML as a secondary option (already enabled in Cargo.toml).
3. **Chunk-local simulation** — Physics/chemistry update only active chunks (near
   camera or containing active entities). Distant chunks are frozen.
4. **Fixed timestep for simulation** — Physics and chemistry run on `FixedUpdate` for
   deterministic-per-tick behavior. Non-determinism comes from RNG seeds, not frame
   timing.
5. **Surface Nets over Marching Cubes** — Fewer artifacts, better performance, good
   enough for organic terrain.
6. **Utility AI over behavior trees** — More emergent, less scripted, scales better
   with complex needs.

---

## Open Questions

- **Multiplayer?** Not in initial scope, but chunk-based architecture doesn't preclude it.
- **Rendering style?** Low-poly / stylized is easier to ship. Photorealism is a trap.
- **World persistence?** Save/load chunks to disk? Needed eventually, not in Phase 0–2.
- **Sound?** Bevy has built-in audio. Not prioritized until there's something to hear.
