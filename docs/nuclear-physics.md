# Phase 14 — Nuclear Physics & Radiation (planned)

Radioactive decay, nuclear reactions, and ionizing/non-ionizing radiation
transport. Enables late-game content: nuclear materials, radiation hazards,
advanced energy sources.

See also: [ROADMAP.md](ROADMAP.md) (project-level phasing),
[electromagnetism.md](electromagnetism.md) (Phase 13, EM fields),
[optics-light.md](optics-light.md) (Phase 12, light transport).

---

## Radioactive Decay

- **Decay modes** — extend `ReactionData` with optional decay fields:
  `decay_half_life: Option<f32>` (seconds), `radiation_type:
  Option<RadiationType>` (Alpha, Beta, Gamma, Neutron). Decay is probabilistic:
  P(decay per tick) = 1 − e^(−λ × dt) where λ = ln(2) / t½
- **Decay chains** — parent isotope decays to daughter product(s), which may
  themselves be radioactive. Model as linked reactions: Uranium → Thorium + α,
  Thorium → Radium + β, etc. Products field:
  `decay_products: Option<Vec<(String, f32)>>` (material name, probability)
- **Mass-energy equivalence** — decay energy Q = Δm × c². New constant:
  `speed_of_light: f64 = 299_792_458.0` m/s (shared with optics). Energy
  released per decay event deposited as heat and radiation

## Radiation Types & Transport

- **Alpha particles** — heavy (4 amu), highly ionizing, very short range
  (~5 cm in air). Stopped by a single voxel of any solid. Modeled as entity
  spawns or absorbed within the source voxel
- **Beta particles** — electrons/positrons, moderate ionizing power, range ~1 m
  in air, stopped by a few cm of metal. Ray-cast from source; attenuate by
  material density and thickness
- **Gamma rays** — high-energy photons, low ionizing power per interaction but
  very penetrating. Exponential attenuation: I = I₀ × e^(−μ × d) where μ is
  the mass attenuation coefficient (m⁻¹) derived from material density and
  atomic number. Ray-cast through multiple voxels
- **Neutron radiation** — uncharged, penetrates most materials except
  hydrogen-rich ones (water, paraffin). Triggers secondary reactions (neutron
  activation, fission). Range: meters through air, attenuated by light elements
- **Non-ionizing radiation** — thermal infrared (already handled by Phase 9a
  radiative heat), visible light (Phase 12 optics), radio waves (Phase 13 EM).
  No additional system needed — these are subsumed by earlier phases

## Radiation Effects

- **Ionizing dose** — per-entity cumulative dose in Gray (Gy = J/kg). Absorbed
  energy per unit mass from all incident radiation. Weighted by radiation type
  (quality factor Q: α=20, β=1, γ=1, neutron=5–20) to get equivalent dose in
  Sieverts (Sv)
- **Biological damage** — creatures accumulate dose over time. Threshold effects:
  nausea (1 Sv), radiation sickness (2–6 Sv), lethal (>6 Sv). Chronic low-dose
  effects: mutation chance, cancer probability (stochastic). Integrates with
  Phase 5 (biology/health system)
- **Material activation** — neutron bombardment converts stable materials to
  radioactive isotopes (neutron activation). Extends the reaction framework
- **Shielding** — attenuation by material: lead (high Z, excellent γ shield),
  water/concrete (excellent neutron moderator), any solid (α stopper).
  Effectiveness from `density`, `atomic_number: Option<u8>` (new MaterialData
  field), and thickness

## Nuclear Reactions

- **Fission** — heavy nucleus splits when struck by neutron. Releases ~200 MeV
  per event (3.2×10⁻¹¹ J) plus 2–3 secondary neutrons → chain reaction.
  Criticality when neutron multiplication factor k ≥ 1. Modeled as cascading
  reactions in the chemistry system with neutron count tracking
- **Fusion** — light nuclei combine at extreme temperature (>10⁷ K). Releases
  energy per the binding energy curve. Only relevant in extreme scenarios
  (stellar simulation, late-game tech). Very low sub-priority
- **Criticality control** — geometry matters: sphere minimizes surface/volume
  ratio → lowest critical mass. The voxel grid naturally supports geometry-
  dependent criticality calculation (count fissile neighbors, track neutron
  economy)

## New constants & material fields

New `MaterialData` fields: `atomic_number: Option<u8>`,
`mass_attenuation_coeff: Option<f32>` (m⁻¹),
`radioactive: Option<RadioactiveProfile>` (half_life, decay_mode, decay_energy).

New universal constants: `speed_of_light` (shared), `planck_constant: f64 =
6.626_070_15e-34` J·s, `boltzmann_constant: f64 = 1.380_649e-23` J/K
(shared with chemistry), `avogadro: f64 = 6.022_140_76e23` mol⁻¹.

Priority: very low. Nuclear physics is late-game content requiring most other
systems to be in place. Radiation transport reuses the ray-cast infrastructure
from Phase 9a/12.
Depends on: Phase 5 (biology for radiation damage), Phase 9a (radiative
transport), Phase 12 (ray-cast optics infrastructure), Phase 13 (EM field model
for neutron interactions).
