# The Dark Candle — Roadmap

A 3D first-person procedural simulation game built on Bevy 0.18. The world is a
chunked voxel grid (32³ per chunk) rendered as smooth meshes via Surface Nets.
Every voxel carries material and physical state, enabling deep simulation of physics,
chemistry, biology, and social systems. Gameplay emerges from the interaction of
independent simulation layers — no scripted events, no hardcoded behaviors.

## Architecture

```
Simulation Stack (each layer reads/writes shared ECS components):

  Layer 3 — Social        relationships, factions, reputation, group behaviors
  Layer 2 — Behavior      needs hierarchy, utility AI, pathfinding, perception
  Layer 1 — Biology       metabolism, health, growth, death/decomposition, plants
  Layer 0 — Physics       gravity, collision, fluids (AMR+LBM+FLIP), pressure, integrity
            Chemistry     materials, heat transfer, reactions, state transitions
            World         voxel grid, chunks, terrain generation, meshing
```

Modules: `camera`, `world`, `physics`, `chemistry`, `biology`, `entities`,
`procgen`, `behavior`, `social`, `data`, `persistence`.

All entity/material/reaction data is loaded from `.ron` files under `assets/data/`.

---

## Completed Phases

### Phase 0 — Restructure & 3D Foundation ✅
Migrated from single-file 2D to modular 3D. 10 Bevy plugins, 3D first-person camera
with WASD/mouse look/fly-mode, placeholder 3D scene.

### Phase 1 — Voxel World & Terrain ✅
Chunked voxel world (32³), cylindrical chunk loading, layered Perlin noise terrain
(continental + detail + cave carving), Surface Nets mesh rendering with per-material
vertex colors. SVO octree subdivision for efficient voxel storage.

### Phase 2 — Materials & Chemistry ✅
12 materials from `.material.ron` files with real-world SI properties (density,
thermal conductivity, specific heat, melting/boiling points). Heat diffusion via
Fourier's law. RON-defined chemical reactions. State transitions (melting, boiling,
freezing, condensation). Fire propagation as emergent behavior.

### Phase 3 — Physics ✅
Entity gravity with force-based model (gravity + buoyancy + drag + friction).
AABB voxel collision. Structural integrity (flood-fill connectivity, collapse).
Gas pressure propagation.

**Fluid simulation** — three-model stack replacing the original CA system:
- **AMR Navier-Stokes** (`amr_fluid/`) — incompressible liquid flow with
  adaptive mesh refinement, pressure projection, surface tracking
- **LBM D3Q19** (`lbm_gas/`) — compressible gas dynamics via lattice Boltzmann,
  BGK collision, streaming, macroscopic field recovery
- **FLIP/PIC** (`flip_pic/`) — particle simulation for precipitation, spray,
  erosion. Staggered MAC grid, trilinear P2G/G2P, Jacobi pressure solve,
  SI drag/gravity, deterministic emission, sub-voxel accumulation

### Phase 4 — Entities & Procedural Generation ✅
`CreatureData` + `ItemData` RON structs. Procedural creature generation (stat
variation, color jitter, deterministic seeding). Procedural item generation
(material properties → emergent weight/durability/damage). 4 biomes with spawn
tables, creature spawning from biome data.

### Phase 5 — Biology ✅
Metabolism (energy from food, starvation). Health (damage types, status effects,
healing). Growth/aging (juvenile → adult → elder). Death → decomposition (corpse
voxels → decay → nutrients). Plant growth (grass spreads to dirt with light/water).

### Phase 6 — Behavior & AI ✅
Needs system (hunger, safety, rest, curiosity, social). Utility-based action
evaluation. 3D voxel pathfinding. Basic behaviors (wander, eat, flee, sleep, follow).
Perception system (sight, hearing, smell).

### Phase 7 — Social Systems ✅
Relationship tracking (trust, familiarity, hostility). Faction system with shared
identity and territory. Reputation from observed actions. Group behaviors
(cooperative hunting, territory defense, trade).

---

## Current State

**All 7 original phases are complete.** The codebase has:
- 86 source files, ~24,400 lines of Rust
- 692 passing tests
- Pre-commit hooks: `cargo fmt` → `cargo clippy -D warnings` → `cargo test`
- CI/CD: GitHub Actions (Linux, Windows, macOS)
- Cross-compilation: `x86_64-pc-windows-gnu`

### Recent work (post-Phase 7)

| Commit | Description |
|--------|-------------|
| `06ff4b9` | SVO octree voxel subdivision system |
| `d5881e8` | AMR Navier-Stokes fluid simulation |
| `1e75f32` | D3Q19 LBM gas simulation |
| `79904af` | FLIP/PIC particle simulation |
| `e7e180f` | Remove legacy CA fluid simulation |

---

## What's Next

With the core simulation stack complete, the project needs integration, polish,
and gameplay. These are not yet planned in detail — each will get a session plan
when started.

### Coupling & Integration
- **Cross-model fluid coupling** — AMR ↔ LBM mass/heat exchange at liquid-gas
  interfaces; FLIP particles entering/leaving LBM gas fields
- **Weather system** — drives FLIP rain/snow emission rates, LBM wind patterns
- **Plugin activation** — wire AmrFluidPlugin, LbmGasPlugin, FlipPicPlugin into
  PhysicsPlugin::build() for runtime use (currently test-only)

### Performance & Scaling
- **Chunk-parallel simulation** — run physics per-chunk on thread pool
- **GPU compute shaders** — offload P2G/G2P, LBM streaming, heat diffusion
- **LOD for distant chunks** — skip full simulation, use simplified models
- **Profiling & budgeting** — establish per-frame time budgets for each system

### Gameplay & Content
- **Player interaction** — mining, placing, crafting, inventory
- **World persistence** — save/load chunks to disk
- **Sound** — ambient, material interactions, creature vocalizations
- **Rendering polish** — lighting, shadows, particle effects, water shading
- **More materials & reactions** — expand the material library
- **Narrative systems** — quests, lore, emergent storytelling from social layer

---

## Key Design Decisions

1. **SI units throughout** — all physics uses real-world values (kg, m, s, K, Pa).
   Constants in `src/physics/constants.rs` and `assets/data/`. No magic numbers.
2. **1 voxel = 1 meter** — all spatial units map directly.
3. **Emergent over hardcoded** — terminal velocity, buoyancy, fire spread arise
   from fundamental forces, not caps or special cases.
4. **Data-driven** — RON files for materials, reactions, biomes, creatures, items,
   factions. Code reads data; data drives behavior.
5. **No ECS bundles** — Bevy 0.18 deprecated them. Spawn component tuples.
6. **Deterministic simulation** — FNV-1a hash for jitter, no `rand` crate.
   FixedUpdate for physics. Reproducible for debugging.
7. **Surface Nets** — fewer artifacts than Marching Cubes, good for organic terrain.
8. **Utility AI** — more emergent than behavior trees, scales with complex needs.
9. **Self-contained physics models** — AMR, LBM, FLIP each have their own
   types/step/plugin/octree_bridge. Couple through voxel data, not shared state.

---

## Open Questions

- **Multiplayer?** Not in scope, but chunk-based architecture doesn't preclude it.
- **Rendering style?** Low-poly / stylized is easier to ship.
- **Fluid coupling strategy?** Interface cells? Overlapping domains? TBD.
- **Sound design?** Bevy has built-in audio. Not prioritized yet.
