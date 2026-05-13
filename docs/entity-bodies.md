# Phase 10 — Entity Bodies & Organic Physics ✅

Physical embodiment for all living entities — player, creatures, and plants.
Replaces the abstract point-entity model (Phase 4) with articulated bodies
that have mass-distributed skeletal structures, soft/rigid tissue physics,
field of vision, and locomotion driven by anatomy. The player is a regular
creature entity controlled by input rather than AI; no special-case code.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[structural-construction.md](structural-construction.md) (Phase 11, building materials).

---

## Foundations already in place

| System | Location | Provides |
|--------|----------|----------|
| Entity physics | `gravity.rs`, `collision.rs` | Force model (gravity + buoyancy + drag + friction), AABB collision |
| Rigid body dynamics | `rigid_body.rs`, `solver.rs` | Angular velocity, moment of inertia, sequential impulse solver |
| Creature data | `procgen/creatures.rs`, `assets/data/` | `CreatureData` RON with stats, color, biome spawning |
| Biology | `biology/` | Metabolism, health, growth/aging, death/decomposition |
| Behavior & perception | `behavior/` | Sight/hearing/smell, pathfinding, needs, utility AI |
| Material properties | `data/mod.rs` | Density, elasticity (Young's modulus), friction, restitution |

## Design

### 1. Skeletal system

A data-driven skeleton defined per species in `.skeleton.ron` files:

- **Bones** — rigid segments with length (m), mass (kg), and material
  (calciumite for vertebrates, chitin for arthropods, cellulose for plants).
  Each bone has a parent bone (forming a tree), rest pose transform, and
  joint constraints (hinge, ball-and-socket, fixed) with angular limits.
- **Skeleton tree** — root bone (pelvis for bipeds, thorax for insects,
  trunk base for trees). Child bones inherit parent transforms. Forward
  kinematics propagates pose; inverse kinematics solves foot/hand placement.
- **SkeletonData** struct — derives `serde::Deserialize`, `Asset`,
  `TypePath`. Loaded via `RonAssetPlugin<SkeletonData>`. Fields: `bones:
  Vec<BoneData>`, `joints: Vec<JointData>`, `rest_pose: Vec<Transform>`.
- **Skeleton component** — runtime `Skeleton` ECS component holding current
  bone transforms, angular velocities, and accumulated torques.

### 2. Soft & rigid tissue

Organic bodies are not uniform rigid bodies — they have tissue layers with
distinct mechanical properties:

- **Rigid tissue** (bone, shell, wood) — modeled as rigid body segments
  connected by joints. Uses existing rigid body solver for each segment.
  Material properties from `.material.ron` (density, Young's modulus).
- **Soft tissue** (muscle, fat, skin, bark, leaves) — modeled as
  mass-spring-damper systems anchored to the skeleton. Provides visual
  deformation and collision volume. Spring stiffness and damping from
  tissue material properties.
- **TissueData** — per-species `.body.ron` defines tissue layers per body
  region: `{ region: "torso", layers: [{ tissue: "muscle", thickness: 0.04,
  density: 1060.0 }, { tissue: "fat", thickness: 0.02, density: 920.0 },
  { tissue: "skin", thickness: 0.003, density: 1100.0 }] }`.
- **Collision volumes** — each body region generates a collision capsule
  (or convex hull) from bone length + tissue thickness. Replaces the
  single-AABB model from Phase 3.

### 3. Locomotion

Movement emerges from skeletal articulation, not velocity teleportation:

- **Gait definitions** — `.gait.ron` files define named animation cycles
  per skeleton: walk, run, crawl, swim, fly, slither. Each gait specifies
  bone target angles per phase, cycle duration, ground contact windows,
  and energy cost (J/m from metabolism).
- **Procedural animation** — IK solvers place feet on terrain surface,
  blend between gaits based on speed/slope/medium. No canned keyframe
  animations — all poses are computed from skeleton constraints + IK targets.
- **Locomotion modes:**
  - *Bipedal/quadrupedal walking* — alternating leg IK with balance
    correction (center of mass over support polygon).
  - *Crawling* — low-clearance gait, belly contact, limbs splayed.
  - *Flying* — wing bones generate lift force proportional to wing area ×
    airspeed² × lift coefficient. Drag from body cross-section.
    Sustained flight requires metabolic energy.
  - *Swimming* — drag-based propulsion in fluid voxels. Fin/limb surface
    area determines thrust.
  - *Slithering* — sinusoidal body wave via sequential bone rotations.
    Friction with ground provides forward force.
  - *Climbing* — IK grip targets on vertical surfaces, weight transfer
    between grip points.
- **Player input mapping** — player input (WASD, jump, crouch) maps to
  gait selection and IK target adjustments on the player's creature
  skeleton. Same system as AI locomotion, different input source.

### 4. Field of vision & perception bodies

Upgrade the abstract perception system (Phase 6) to use physical geometry:

- **Eye components** — position on skeleton (bone attachment point), FOV
  cone angle, max range. Occlusion via DDA ray-cast against voxel world
  (reuses `src/world/raycast.rs`). Multiple eyes = wider combined FOV.
- **Ear components** — position on skeleton, sensitivity curve, directional
  bias from head orientation.
- **Player camera** — first-person camera attaches to the player entity's
  head bone. Camera FOV = eye FOV. Head-bob from locomotion gait. No
  special player camera system — just a `Camera3d` parented to the head
  bone entity.
- **Smell** — unchanged from Phase 6 (diffusion-based, no body geometry
  needed).

### 5. Injury & damage model

Tiered physical damage integrated with the skeletal/tissue system:

- **Damage zones** — each body region (head, torso, limb, wing, root, etc.)
  tracks its own hit points derived from tissue mass and material toughness.
- **Injury tiers:**
  - *Bruise/strain* — soft tissue damage. Reduces performance (movement
    speed, grip strength). Heals over time via metabolism.
  - *Fracture* — bone damage. Limb loses structural support — IK solver
    treats fractured bone as a limp/hanging segment. Requires healing time
    proportional to bone mass.
  - *Severing* — catastrophic damage separates a body part. Detached part
    becomes a physics entity (drops with rigid body dynamics). Creature
    loses capabilities associated with that limb permanently (or until
    regeneration, if the species supports it).
- **Damage propagation** — impacts apply force to the collision volume of
  the hit region. Force exceeding tissue toughness creates injury. Armor
  (equipped items with material hardness) absorbs force first.
- **Healing** — biological healing rate from Phase 5 metabolism, scaled by
  injury tier. Fractures heal slowly. Severing doesn't heal without
  regeneration trait.

### 6. Plant body physics

Trees and large plants as semi-rigid articulated structures:

- **Trunk & branches** — modeled as a skeleton tree. Trunk = root bone,
  branches = child bones. Wood material properties (density, Young's
  modulus, flexural strength from Phase 11 building materials).
- **Root system** — anchor bones extending into terrain voxels. Root
  depth + spread determines wind resistance and nutrient access (ties
  into Phase 5 plant growth).
- **Canopy** — leaf clusters as soft-body masses on branch tips. Wind
  force (from LBM gas field, Phase 9) applies lateral load. Branches
  flex under wind + gravity. Excessive force → branch breakage (uses
  flexural strength).
- **Growth integration** — as plants grow (Phase 5), new bones are added
  to the skeleton. Trunk thickens (bone radius increases), branches
  extend, canopy fills out. Growth rate from metabolism.
- **Felling & damage** — chopping applies damage to trunk bone. When trunk
  HP reaches zero → tree falls as a rigid body chain (bones disconnect
  from root anchor, gravity takes over). Fallen tree becomes harvestable
  material.

## Implementation steps

1. **`SkeletonData` and RON loader** — new `src/bodies/skeleton.rs`.
   `SkeletonData` struct with bones, joints, rest pose. Register
   `RonAssetPlugin<SkeletonData>`. Create skeleton RON files for 2–3
   species (humanoid, quadruped, tree).

2. **`Skeleton` runtime component** — ECS component with current bone
   transforms, angular state. Forward kinematics system in `FixedUpdate`.
   Parent-child transform propagation.

3. **Tissue & collision volumes** — new `src/bodies/tissue.rs`. `BodyData`
   RON with tissue layers per region. Generate per-region collision
   capsules from bone + tissue. Replace single AABB with compound collider.

4. **IK solver** — new `src/bodies/ik.rs`. FABRIK or CCD inverse
   kinematics for limb chains. Foot placement on terrain. Hand/grip
   targeting. Joint constraint enforcement.

5. **Locomotion gaits** — new `src/bodies/locomotion.rs`. `GaitData` RON
   with bone angle targets per phase. Gait state machine (idle → walk →
   run → sprint). Procedural gait blending. Energy cost integration with
   metabolism.

6. **Player embodiment** — player entity spawns with same `Skeleton` +
   `BodyData` as a humanoid creature. Input system maps WASD → gait
   selection → IK targets. `Camera3d` parented to head bone.

7. **Perception body integration** — new `src/bodies/perception.rs`. Eye/ear
   components with skeleton attachment points. FOV occlusion via DDA
   ray-cast. Replace abstract Phase 6 perception radius with physical
   sight cones.

8. **Injury system** — new `src/bodies/injury.rs`. Per-region damage
   tracking. Injury tier logic (bruise → fracture → sever). IK response
   to fractures. Severed limb spawning. Healing rate integration.

9. **Plant bodies** — extend skeleton system for plants. `TreeSkeletonData`
   RON. Wind response system (LBM pressure → branch torque). Growth-driven
   skeleton expansion. Felling mechanics.

10. **Physics integration** — wire articulated body solver into
    `FixedUpdate`. Per-bone collision response via existing narrow phase +
    impulse solver. Mass distribution from tissue layers → moment of
    inertia tensor per bone.

## Dependencies

- Steps 1–3 build on Phase 3 (rigid body physics) and Phase 4 (creature data).
- Step 4 (IK) is self-contained, depends only on step 2 (skeleton).
- Step 5 (locomotion) requires steps 2 + 4 (skeleton + IK).
- Step 6 (player) requires steps 5 (locomotion) + 7 (perception).
- Step 7 extends Phase 6 (behavior/perception) with body geometry.
- Step 8 extends Phase 5 (biology/health) with body-part damage.
- Step 9 depends on steps 1–3 (skeleton + tissue) and Phase 9 (LBM wind).
- Step 10 integrates everything into the physics pipeline.

## What stays unchanged

Creature RON data (extended with skeleton/body references, not replaced).
Biology systems (metabolism, growth — extended with per-limb damage, not
replaced). Behavior AI (action selection unchanged — locomotion replaces
the velocity output). Existing rigid body solver (reused for per-bone
dynamics). Voxel collision (extended from single AABB to compound, not
replaced).

---

## Implementation Status ✅

**Completed at commit `bd386b7` (Phase 10).**

### Files created

| File | Description |
|------|-------------|
| `src/bodies/mod.rs` | `BodiesPlugin`, system schedule |
| `src/bodies/skeleton.rs` | `SkeletonData` RON asset, `BoneData`, `SkeletonInstance`, FK propagation |
| `src/bodies/tissue.rs` | `TissueLayer`, `BodyCollider`, compound AABB collider construction |
| `src/bodies/ik.rs` | `IkChain`, FABRIK solver, foot/hand placement |
| `src/bodies/locomotion.rs` | `GaitData` RON asset, `LocomotionState`, `AnimPhase`. Split into `advance_gait_phase` + `apply_skeleton_gait_and_ik` so skeleton-less creatures still advance phase. `ai_gait_from_velocity` copies `PhysicsBody.velocity.xz` → `GaitState.speed`. |
| `src/bodies/perception.rs` | `EyeMount`, `EarMount`, `PerceptionEvent`, range-gated vision/hearing |
| `src/bodies/injury.rs` | `InjuryRecord`, per-region severity (Bruised → Fractured → Severed), wound propagation |
| `src/bodies/plant.rs` | `PlantBody`, `PlantJoint`, wind response, felling mechanics |
| `src/bodies/player.rs` | `PlayerBody` — maps input to locomotion controller. Holds `BipedGaitPath` + `QuadrupedGaitPath` resources. |
| `src/bodies/procedural_body.rs` | `BodyPlan` (Quadruped/Biped/Hexapod/Serpent), `BodyPart`, `body_part_specs()` pure layout, `spawn_procedural_body()` ECS spawner. Cheap multi-cuboid stand-in until creature glTF assets exist. |
| `src/bodies/procedural_body_anim.rs` | `body_part_offset()` pure kinematics, `animate_procedural_body` system. Sine-wave leg swing, torso bob, tail sway driven by `GaitState`. Quadruped trot, biped alternating, hexapod tripod gait. |

### RON data files

- `assets/data/skeletons/humanoid.skeleton.ron` — 14-bone biped skeleton
- `assets/data/skeletons/quadruped.skeleton.ron` — 18-bone quadruped skeleton
- `assets/data/skeletons/tree.skeleton.ron` — 5-level recursive tree trunk
- `assets/data/bodies/humanoid.body.ron` — tissue layers for humanoid
- `assets/data/bodies/quadruped.body.ron` — tissue layers for quadruped
- `assets/data/gaits/biped.gait.ron` — walk/run/sprint gait parameters
- `assets/data/gaits/quadruped.gait.ron` — trot/gallop gait parameters

### Tests

61 unit tests pass (`cargo test --lib -- bodies`).

### Post-Phase 10 addition: procedural multi-part bodies for AI creatures

The original Phase 10 plan assumed all visible creatures would use the
data-driven `Skeleton` + IK pipeline. In practice, AI creatures spawned
by `src/procgen/mod.rs` had no skeleton and rendered as a single
sliding `Cuboid` — physically present but visually inert.

To bridge the gap until glTF/skeletal creature assets exist, a parallel
**procedural body** path was added:

- `src/bodies/procedural_body.rs` — pure `body_part_specs(plan, hitbox)`
  function maps a `BodyPlan` (`Quadruped`, `Biped`, `Hexapod`, `Serpent`)
  + full-extent hitbox into a list of rest poses for child cuboid parts
  (torso, head, legs, optional tail). `spawn_procedural_body()` parents
  one `BodyPart`-tagged child entity per spec under the creature root.
  `BodyPlan::default_for_size(BodySize)` picks Biped for tiny creatures
  and Quadruped for everything else.
- `src/bodies/procedural_body_anim.rs` — pure `body_part_offset()` returns
  the per-tick delta `Transform` for a `BodyPart` given `GaitState`. Legs
  swing forward/back on sine waves with phase offsets selected per plan
  (quadruped trot = FL+BR / FR+BL, hexapod tripod = FL+MR+BL / FR+ML+BR,
  biped alternating). Torso bobs vertically; tail sways yaw. Idle
  creatures animate subtly (`IDLE_AMPLITUDE = 0.12`); amplitude scales
  linearly with speed up to `FULL_SWING_SPEED = 4 m/s`.
- `src/bodies/locomotion.rs` was split: `update_locomotion` previously
  queried `&mut Skeleton`, silently skipping skeleton-less entities.
  It now consists of `advance_gait_phase` (runs on every `GaitState`)
  and `apply_skeleton_gait_and_ik` (`With<Skeleton>` only). Phase
  advances even at zero speed, so idle creatures keep breathing.
- A new `ai_gait_from_velocity` system syncs
  `PhysicsBody.velocity.xz.length()` into `GaitState.speed` for every
  `Creature` except the player (excluded explicitly because the player
  is driven by `FpsCamera.speed`).
- The creature spawn site in `src/procgen/mod.rs` no longer attaches a
  single `Cuboid` mesh. Instead it inserts `GaitState::default()` +
  `GaitDataHandle` (asset loaded from `BipedGaitPath` or
  `QuadrupedGaitPath` based on the chosen plan) and calls
  `spawn_procedural_body()` to spawn the parts.

This is intentionally a temporary stop-gap. When glTF skeletal assets
arrive, `procedural_body*.rs` and the AI creature `BodyPart` children
can be retired in favor of the full `Skeleton` + tissue + IK pipeline
described above. The split between `advance_gait_phase` and
`apply_skeleton_gait_and_ik` keeps both paths additive.

`assets/data/gaits/quadruped.gait.ron` and `biped.gait.ron` are reused
unchanged. Bone-target fields are simply ignored by the procedural
animator (which only consumes `GaitState.phase`/`speed`/`mode`).

