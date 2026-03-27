# Phase 10 — Entity Bodies & Organic Physics (planned)

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
