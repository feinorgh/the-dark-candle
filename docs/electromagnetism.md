# Phase 13 — Electricity & Magnetism (planned)

Full electromagnetic simulation using a simplified Maxwell's equations solver on
the voxel grid. Enables technology progression, electrical hazards, and magnetic
gameplay mechanics.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[optics-light.md](optics-light.md) (Phase 12, light transport),
[advanced-physics.md](advanced-physics.md) (coupling layer, constraints).

---

## Electrostatics & Current Flow

- **Electrical conductivity** — per-material property `electrical_conductivity:
  Option<f32>` (S/m) in `MaterialData`. Iron = 1.0e7, copper = 5.96e7,
  water = 0.05, stone ≈ 0, air ≈ 0 (insulator). Determines which voxels
  conduct current
- **Resistance network** — connected conductive voxels form a circuit graph.
  Solve for current via Kirchhoff's laws (sparse linear system) or relaxation
  on the voxel grid. Current I = V / R where R = 1 / (σ × A / L)
- **Voltage sources** — batteries (stored charge), generators (mechanical →
  electrical via Faraday's law: EMF = −dΦ_B/dt), piezoelectric crystals
  (pressure → voltage)
- **Resistive heating** — I²R power dissipated as heat into the thermal field.
  Enables electric furnaces, heating elements, short-circuit fires, fuses that
  melt when overloaded

## Magnetism

- **Magnetic permeability** — per-material `magnetic_permeability: Option<f32>`
  (H/m; vacuum = 4π×10⁻⁷, iron = 6.3×10⁻³). Determines magnetic response
- **Magnetic field** — per-voxel `B: Vec3` (Tesla). Permanent magnets from
  ferromagnetic materials, electromagnets from current-carrying coils
  (Biot-Savart or Ampère's law on the grid)
- **Lorentz force** — charged/magnetized entities experience F = q(v × B).
  Enables magnetic rail transport, compass needles, magnetic locks
- **Electromagnetic induction** — changing B through a conductive loop induces
  EMF (Faraday's law). Generator gameplay, inductive sensors

## Electromagnetic Waves (simplified)

- **Wave propagation** — EM waves at speed c through the voxel grid (FDTD —
  Finite-Difference Time-Domain — on a coarsened grid for performance).
  Primarily for radio/signal propagation, not visual light (handled by Phase 12
  optics)
- **Absorption & shielding** — conductive materials absorb/reflect EM waves
  (skin depth δ = √(2/(ωμσ))). Faraday cage gameplay, signal blocking through
  metal walls

## Lightning

- **Atmospheric charge separation** (Phase 9) → leader propagation along
  lowest-resistance voxel path → return stroke. Deposits massive current →
  resistive heating → fire ignition, sand → glass (fulgurite), tree splitting
- **Discharge probability** — builds with charge differential and humidity;
  tall/conductive structures attract strikes

## New constants & material fields

New `MaterialData` fields: `electrical_conductivity: Option<f32>` (S/m),
`magnetic_permeability: Option<f32>` (H/m).

New universal constants: `elementary_charge: f64 = 1.602_176_634e-19` C,
`vacuum_permittivity: f64 = 8.854_187_8128e-12` F/m,
`vacuum_permeability: f64 = 1.256_637_062_12e-6` H/m.

Priority: low. Only pursue when a technology/crafting tier requires wiring,
circuits, or electromagnetic machinery.
Depends on: Phase 9 (atmosphere for lightning), Phase 11 (structures for
circuits), Phase 9a (thermal coupling for resistive heating).
