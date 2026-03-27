# Advanced Physics Systems (planned)

Future physics subsystems that extend the existing force model, rigid body
solver, and fluid simulations. These are planned but not yet designed in full
detail — each will get a detailed design session when implementation begins.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[fluid-simulation-system.md](fluid-simulation-system.md) (AMR/LBM/FLIP),
[structural-construction.md](structural-construction.md) (Phase 11),
[entity-bodies.md](entity-bodies.md) (Phase 10).

---

## Physics Coupling Layer (Entity ↔ World)

The individual physics engines (rigid body solver, LBM gas, AMR fluid, FLIP
particles, pressure diffusion, heat transfer) are each functional but largely
siloed. A coupling layer is needed so that world-level fields exert forces on
entities and entity actions feed back into world state.

- **Pressure gradient → entity impulse** — sample the pressure field around an
  entity's AABB; the net pressure difference across its surface produces a force
  (F = −∇P × V_displaced). Enables explosion shockwaves to push rigid bodies and
  pressure-driven object movement
- **Wind field → entity drag** — sample LBM velocity at an entity's position;
  compute aerodynamic drag using the entity's `DragProfile` and the *relative*
  velocity (entity velocity minus wind). Enables wind-blown NPCs, flags,
  projectile drift
- **Fluid buoyancy coupling** — the existing buoyancy system uses a generic
  medium density fallback. Couple it to actual AMR fluid voxel state: sample
  fluid density and velocity at the entity's submerged volume. Enables realistic
  floating, sinking, and current-driven drift
- **Particle–entity collisions** — FLIP/PIC particles (rain, spray, debris)
  currently pass through entities. Add narrow-phase tests between particles and
  entity colliders; on hit, transfer momentum (particle → entity impulse) and
  trigger accumulation (wetting, erosion, coating)
- **Collision damage feedback** — the impulse solver already computes contact
  impulse magnitudes. Expose peak impulse per contact pair per frame; when it
  exceeds a material-dependent damage threshold, emit a `DamageEvent`. Enables
  fall damage, impact breakage, and projectile lethality without hardcoded HP
  deductions
- **Heat field → entity temperature** — entities in hot/cold environments should
  gain/lose heat via convection (sample ambient voxel temperature + wind speed
  around the entity). Enables freezing hazards, fire proximity damage, and
  cooking mechanics

Design constraint: all coupling uses existing SI fields — no new magic constants.
Forces emerge from pressure in Pascals, velocity in m/s, temperature in Kelvin.
The coupling layer is a set of systems that *read* world fields and *write*
entity forces (and vice versa), not a new physics engine.

---

## Rigid Body Physics

The existing entity physics (`PhysicsBody`, `Mass`, `DragProfile`, `Collider`)
handles entity-vs-voxel forces and AABB terrain collision. A full rigid body
system needs:

- **Entity-vs-entity collision** — broad phase (spatial hash / sweep-and-prune)
  + narrow phase (AABB or GJK) between dynamic entities
- **Restitution & friction coefficients** — per-material bounce and slide
  behavior on collision response (impulse-based)
- **Angular dynamics** — `AngularVelocity`, `MomentOfInertia`, `Torque`
  components. Rotational integration in `FixedUpdate`. Coupled with linear
  response at contact points
- **Contact resolution** — sequential impulse solver or position-based correction
  for penetration, stacking, and resting contact
- **Collision shapes** — extend `Collider` beyond AABB: sphere, capsule, convex
  hull for entity-entity narrow phase
- **Spatial partitioning** — uniform grid or dynamic BVH for efficient
  broad-phase entity queries
- **Sleep system** — deactivate rigid bodies whose linear and angular velocities
  stay below a threshold for N consecutive frames. Sleeping bodies skip force
  integration, broad/narrow phase, and solver work. Wake on: external impulse,
  nearby collision, or explicit event. Eliminates residual micro-bounce on
  resting contacts and saves CPU for large entity counts

Design constraint: all collision properties (restitution, friction) derive from
`MaterialData` in RON files. No magic numbers — emergent behavior from SI
material properties.

---

## Soft Body Physics

Deformable objects that bend, stretch, compress, or tear under stress — as
opposed to the infinitely-stiff assumption of rigid body dynamics. Candidate
implementation: mass-spring lattice or position-based dynamics (PBD/XPBD).

- **Elastic deformation** — objects return to rest shape when stress < yield
  strength. Governed by Young's modulus (already in `MaterialData`)
- **Plastic deformation & fracture** — permanent shape change above yield
  strength; tearing/snapping above ultimate strength
- **Use cases** — cloth (banners, nets), rope/vines/tethers, organic creatures
  (slimes, tentacles, flesh deformation on impact), vegetation bending in wind
- **Terrain deformation** — soft materials (mud, snow, sand) compressing under
  load rather than discrete voxel placement/removal
- **Coupling with rigid bodies** — soft body nodes exert forces on rigid bodies
  and vice versa (e.g. rope attached to a rigid anchor)

Design constraint: material stiffness, damping, and yield/ultimate strength
derive from `MaterialData` RON files. Deformation emerges from the interaction
of applied forces and material properties — no hardcoded spring constants.

Priority: low. Rigid body physics, fluids (LBM/FLIP), and structural integrity
cover most gameplay needs. Soft bodies become relevant when a core mechanic
requires deformation (rope/grapple, ragdolls, destructible organic entities).

---

## Constraints & Joints

Rigid body constraints that restrict relative motion between two entities or
between an entity and a fixed anchor point. Required for mechanical gameplay.

- **Distance constraint** — maintains fixed separation between two anchor points
  (ropes, chains, tethers). Enforce via position-based or impulse-based solver
- **Hinge / revolute joint** — rotation around a single axis (doors, levers,
  cranks, hinged lids)
- **Prismatic / slider joint** — translation along a single axis (pistons,
  sliding doors, drawbridges)
- **Ball-and-socket joint** — rotation around a point with no axis restriction
  (ragdoll shoulders/hips, hanging lanterns)
- **Spring-damper** — elastic constraint with configurable stiffness (N/m) and
  damping (N·s/m). Suspension, bungee cords, shock absorbers
- **Motor** — applies torque or linear force to a joint axis (windmills,
  conveyor belts, powered doors)
- **Breakable constraints** — joints that snap when force exceeds a threshold
  derived from material tensile/shear strength. Enables chain-breaking,
  structural tearing

Design constraint: constraint parameters (stiffness, damping, break force)
derive from `MaterialData` where applicable. Solve within the existing
sequential impulse solver by adding constraint rows alongside contact rows.

---

## Explosion & Detonation Mechanics

Explosions as a first-class physics event, bridging pressure diffusion, rigid
body dynamics, and voxel destruction.

- **Detonation source** — an entity or voxel event that injects energy (J) into
  the pressure field at a point. Energy derived from material heat of combustion
  (J/kg) × mass
- **Blast wave propagation** — pressure diffusion (or LBM shock) radiates
  outward. Overpressure decays with distance (inverse-cube for 3D)
- **Structural damage** — overpressure exceeding a voxel's compressive strength
  destroys or fractures it. Cascading destruction via structural integrity
  flood-fill
- **Debris generation** — destroyed voxels spawn rigid body fragments with
  initial velocity from the pressure gradient. Fragment mass = material density ×
  voxel volume
- **Entity blast impulse** — pressure gradient → entity impulse (from the
  coupling layer above). Knockback, ragdoll launch, vehicle flipping
- **Thermal pulse** — detonation injects heat into surrounding voxels. Ignites
  flammable materials via existing combustion reactions

No hardcoded blast radius or damage tables. Destruction is emergent from
pressure magnitude vs. material strength.

---

## Projectile Ballistics

Extended force model for high-speed projectiles where aerodynamic effects matter.

- **Magnus force** — spinning projectiles experience lateral force from
  differential air pressure: F = S × (ω × v), where S is a shape-dependent
  coefficient. Enables arrow drift, curveball trajectories
- **Tumbling / angle of attack** — non-spherical projectiles (arrows, javelins)
  have orientation-dependent drag. Misaligned flight increases drag and induces
  torque toward broadside orientation
- **Wind interaction** — projectile drag computed against relative velocity
  (entity − wind field from LBM). Arrows drift in crosswinds
- **Impact model** — on collision, impulse magnitude determines penetration
  depth based on projectile KE vs. target material hardness. Shallow = ricochet,
  deep = embed

Uses existing `DragProfile`, `AngularVelocity`, and collision damage feedback.
No new physics engine — extends the force summation in `apply_forces`.

---

## Fluid–Terrain Interaction

Bridge the AMR fluid simulation with the voxel terrain grid so that liquid
visibly fills, drains, and reshapes the world.

- **Fluid → voxel conversion** — when AMR fluid accumulates ≥ 1 m³ in a cell
  with sufficient dwell time, convert it to a Water (or Lava) terrain voxel.
  Enables flooding, pool formation, lava flows solidifying
- **Voxel → fluid conversion** — when a liquid terrain voxel loses structural
  support or is heated past boiling, convert it back to AMR fluid particles for
  dynamic flow
- **Erosion** — flowing fluid exerts shear stress on adjacent solid voxels
  (τ = μ × dv/dy). When cumulative stress exceeds material cohesion, the voxel
  is destroyed and becomes sediment (FLIP particles). Enables river carving,
  waterfall erosion, wave action on coastlines
- **Weathering** — slow degradation of exposed surfaces from temperature cycling
  (freeze-thaw: water in cracks expands on freezing), rain impact accumulation,
  and wind abrasion (particle impacts from FLIP/PIC). Modeled as a durability
  counter per exposed voxel face
- **Sediment transport & deposition** — eroded particles carried by fluid flow
  (FLIP advection); deposit when flow velocity drops below settling threshold.
  Enables delta formation, silt accumulation, alluvial fans

Design constraint: erosion rates derived from fluid velocity, material hardness,
and cohesion — all from `MaterialData`. No per-material erosion-rate constants.

---

## Vehicle Physics

Rigid body entities with wheel constraints and drive systems. Low priority until
gameplay requires rideable mounts or machines.

- **Wheel model** — each wheel is a constraint (hinge joint at axle +
  spring-damper for suspension). Rolling resistance from `PhysicsMaterial`
  friction × normal force
- **Drive torque** — engine/motor applies torque to drive axle joints. Torque
  curve defined in vehicle data (RON)
- **Steering** — front axle hinge limits change with player input. Ackermann
  geometry for multi-axle vehicles
- **Suspension** — spring-damper constraints between chassis and wheel mounts.
  Stiffness and damping from vehicle data
- **Buoyancy for boats** — displaced volume from hull shape (convex hull or
  voxel scan) × fluid density → buoyancy force. Stability from metacentric
  height

Depends on: constraints/joints system, collision shapes beyond AABB.

---

## Acoustics

Sound propagation as a physics system rather than purely a rendering/audio
concern. Low priority — relevant when stealth or environmental audio becomes a
gameplay mechanic.

- **Sound pressure waves** — point-source events (explosions, footsteps, speech)
  emit into the pressure field or a dedicated acoustic grid. Propagation speed =
  343 m/s in air (temperature-dependent: c = √(γRT/M))
- **Obstruction & occlusion** — ray cast from source to listener; solid voxels
  attenuate by material density and thickness. Enables muffled sound through
  walls
- **Reflection & reverb** — ray-traced early reflections off nearby surfaces;
  reverb tail from room volume estimate (flood-fill air voxels). Cave echo,
  indoor dampening
- **Doppler effect** — frequency shift based on relative velocity between source
  and listener: f' = f × (c + v_listener) / (c + v_source)
- **AI hearing** — creatures sample the acoustic field at their position; loud
  events above a threshold trigger alert/investigate behaviors
