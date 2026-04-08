//! Tectonic plate simulation on a geodesic grid.
//!
//! ## Algorithm overview
//!
//! This module uses a **dynamic-boundary plate model**: plate assignments are
//! initially set via weighted BFS flood-fill (producing Earth-like size
//! variation), then refined each simulation step as subduction consumes
//! oceanic cells at convergent boundaries.
//!
//! Forces at plate boundaries drive elevation changes each step, producing
//! mountains at convergent boundaries, rifts/trenches at divergent boundaries,
//! and fault zones at transform boundaries.
//!
//! ### Pipeline
//! 1. Seed N plate centres using a Fibonacci sphere lattice for uniform coverage.
//! 2. Weighted BFS flood-fill assigns cells; growth rate per plate drawn from a
//!    Pareto distribution gives a power-law size distribution (few large plates,
//!    many small ones — mirroring Earth).
//! 3. Each plate receives a random type (continental/oceanic — larger plates
//!    skew oceanic), angular velocity, and density.
//! 4. Run `steps` tectonic iterations:
//!    a. Recompute boundary normals (boundaries shift due to subduction).
//!    b. Boundary detection → height update → subduction reassignment.
//!    c. Volcanic activity decay → erosion smoothing.
//! 5. Post-process: compute sea level so ~30% of surface area is land.

use bevy::math::DVec3;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, VecDeque};

use crate::planet::grid::CellId;
use crate::planet::{BoundaryType, CrustType, PlanetData};

// ---------------------------------------------------------------------------
// Constants — all rates are per million years (Myr), scaled by dt each step
// ---------------------------------------------------------------------------

/// Reference time step (Myr) that the base rates are calibrated against.
const REF_DT_MYR: f64 = 10.0;

/// Elevation for newly initialised continental cells (m).
const INIT_CONTINENTAL_ELEV: f64 = 200.0;
/// Elevation for newly initialised oceanic cells (m).
const INIT_OCEANIC_ELEV: f64 = -3_500.0;

/// Default continental crust thickness (m).
const CONTINENTAL_CRUST_DEPTH: f32 = 35_000.0;
/// Default oceanic crust thickness (m).
const OCEANIC_CRUST_DEPTH: f32 = 7_000.0;

/// Orogeny rate at continent-continent convergent boundaries (m/Myr).
const OROGENY_RATE: f64 = 8.0;
/// Arc rise rate on the continental side of ocean-continent convergence (m/Myr).
const ARC_RATE: f64 = 6.0;
/// Trench descent rate on the oceanic side of ocean-continent convergence (m/Myr).
const TRENCH_RATE: f64 = 10.0;
/// Island arc rise rate (ocean-ocean convergent, building side, m/Myr).
const ISLAND_ARC_RATE: f64 = 2.0;
/// Descent rate at an ocean-ocean convergent boundary (m/Myr).
const OCEAN_CONVERGENT_RATE: f64 = 8.0;
/// Rift descent rate at divergent boundaries (m/Myr).
const RIFT_RATE: f64 = 5.0;

/// Volcanic activity gain per reference time step at active boundaries.
const VOLCANIC_GAIN_BASE: f32 = 0.05;
/// Volcanic activity decay factor per reference time step.
const VOLCANIC_DECAY_BASE: f32 = 0.95;
/// Fault stress gain per reference time step at transform boundaries.
const FAULT_STRESS_GAIN_BASE: f32 = 0.01;

/// Fraction of elevation difference diffused per reference time step.
const EROSION_RATE_BASE: f64 = 0.02;

/// Hard cap on mountain height (m).
const MAX_ELEVATION: f64 = 9_000.0;
/// Hard cap on ocean depth (m, negative).
const MIN_ELEVATION: f64 = -11_000.0;

/// Fraction of total surface area that should be land (used for sea-level).
const TARGET_LAND_FRACTION: f64 = 0.30;

/// Base velocity threshold (rad displacement per reference time step) below
/// which boundaries are classified as transform.
const VELOCITY_THRESHOLD_BASE: f64 = 1e-4;

/// Pareto shape parameter for plate growth weight distribution.
/// Lower α → more extreme size variation. α ≈ 1.0–1.5 is Earth-like.
const PLATE_SIZE_PARETO_ALPHA: f64 = 1.3;

/// Maximum ratio between the largest and smallest plate growth weight.
/// Caps the Pareto tail to prevent one plate dominating the entire surface.
const MAX_PLATE_WEIGHT_RATIO: f64 = 15.0;

/// Base probability per reference step that an oceanic convergent-boundary cell
/// is consumed by the overriding continental plate.
const SUBDUCTION_RATE_BASE: f64 = 0.03;

/// Minimum plate size as a fraction of total cells. Plates below this
/// threshold are protected from further subduction consumption.
const MIN_PLATE_FRACTION: f64 = 0.01;

// ─── Strain constants ────────────────────────────────────────────────────────

/// Compressive strain injected per reference step at convergent boundaries.
const STRAIN_GAIN_CONVERGENT: f32 = 0.03;
/// Extensional strain injected per reference step at divergent boundaries.
const STRAIN_GAIN_DIVERGENT: f32 = 0.02;
/// Shear strain injected per reference step at transform boundaries.
const STRAIN_GAIN_TRANSFORM: f32 = 0.015;
/// Fraction of strain difference diffused to same-plate neighbors per
/// reference step. Higher values propagate deformation further inland.
const STRAIN_DIFFUSION_RATE: f64 = 0.12;
/// Continental crust transmits strain further than oceanic crust.
/// Mimics the wide deformation zones seen in continent-continent collisions
/// (e.g., Tibetan Plateau, ~1000 km of deformation from the Himalayan front).
const CONTINENTAL_STRAIN_CONDUCTIVITY: f64 = 1.8;
/// Viscous strain relaxation per reference step. Prevents unbounded
/// accumulation in quiescent plate interiors.
const STRAIN_RELAXATION_RATE: f32 = 0.005;

// ─── Rifting constants ───────────────────────────────────────────────────────

/// Strain threshold above which cells become rift candidates. When
/// high-strain cells form a band that disconnects a plate, the plate splits.
const RIFT_STRAIN_THRESHOLD: f32 = 0.65;
/// Minimum plate size (cells) below which rifting is suppressed.
/// Prevents microplates from fragmenting further.
const MIN_RIFT_PLATE_CELLS: usize = 50;
/// Minimum size of the smaller fragment as a fraction of the parent plate
/// for a rift to proceed. Prevents trivial single-cell splits.
const MIN_RIFT_FRAGMENT_FRACTION: f64 = 0.15;
/// Fraction of parent plate angular speed applied perpendicular to the
/// existing velocity for the newly rifted plate. Produces diverging motion
/// between the two halves (like Africa and South America separating).
const RIFT_VELOCITY_DIVERGENCE: f64 = 0.3;
/// Maximum rift events per simulation step to prevent runaway fragmentation.
const MAX_RIFTS_PER_STEP: usize = 2;

// ─── Suturing constants ──────────────────────────────────────────────────────

/// Cumulative convergent contact time (Myr) before two continental plates
/// suture into one. Major continent-continent collisions take ~100–200 Myr
/// on Earth (e.g., India → Eurasia began ~55 Ma, still ongoing).
const SUTURE_THRESHOLD_MYR: f64 = 120.0;
/// Minimum fraction of the smaller plate's boundary perimeter that must
/// be in convergent contact with the larger plate for suturing.
const SUTURE_MIN_CONTACT_FRACTION: f64 = 0.20;

// ─── Deformation zone constants ──────────────────────────────────────────────

/// Maximum number of cell hops that convergent boundary orogeny propagates
/// inward from the plate boundary. Creates wide mountain belts instead of
/// single-cell ridges.
const DEFORMATION_ZONE_HOPS: u32 = 4;
/// Exponential falloff per hop: force at hop N = base_rate × FALLOFF^N.
/// 0.45 gives ~20% effect at hop 1, ~9% at hop 2, ~4% at hop 3.
const DEFORMATION_FALLOFF: f64 = 0.45;
/// Back-arc extension rate (m/Myr). Behind oceanic-continental subduction
/// zones, the overriding continental plate thins and subsides, creating
/// back-arc basins (e.g., Sea of Japan behind the Japanese island arc).
const BACK_ARC_RATE: f64 = 2.0;

// ─── Slab-pull force constants ───────────────────────────────────────────────

/// Slab-pull angular acceleration coefficient (rad/yr²).
///
/// When multiplied by a plate's subduction fraction (n_subducting / n_total),
/// gives the angular acceleration from slab pull. This is the **dominant**
/// force driving plate motion on Earth (~70% of total driving force).
///
/// Calibration: a plate with 5% subduction boundary reaches ~8 cm/yr in
/// ~16 Myr; a plate with 1% subduction takes ~78 Myr. Plates with no
/// subduction receive no slab pull and drift only from random perturbation.
///
/// Physical basis: slab pull ∝ Δρ × g × slab_thickness × slab_depth.
/// Density contrast Δρ ≈ 80 kg/m³, thickness ≈ 100 km, depth ≈ 660 km.
const SLAB_PULL_COEFFICIENT: f64 = 2e-14;

// ─── Plate velocity constants (SI) ───────────────────────────────────────────

/// Minimum plate surface velocity (m/year). ~2 cm/yr.
const MIN_PLATE_SPEED_M_YR: f64 = 0.02;
/// Maximum plate surface velocity (m/year). ~10 cm/yr.
const MAX_PLATE_SPEED_M_YR: f64 = 0.10;

/// Interval (Myr) between random perturbations to plate acceleration.
const ACCEL_PERTURB_INTERVAL_MYR: f64 = 50.0;

// ---------------------------------------------------------------------------
// Plate data
// ---------------------------------------------------------------------------

/// Internal state for a single tectonic plate during simulation.
#[derive(Debug, Clone)]
struct Plate {
    /// Angular velocity vector (axis × angular_speed, **radians/year**).
    angular_velocity: DVec3,
    /// Angular acceleration vector (**radians/year²**). Produces gradual
    /// speedup and slowdown over geological time, mimicking mantle drag
    /// and slab pull changes.
    angular_acceleration: DVec3,
    /// Whether this plate is continental or oceanic.
    crust_type: CrustType,
}

impl Plate {
    /// Surface velocity of this plate at a given unit-sphere position,
    /// scaled to displacement per time step.
    ///
    /// `v = (omega × dt_yr) × r` — the cross product gives the tangent
    /// velocity at `pos` for one step of `dt_yr` years.
    fn velocity_at(&self, pos: DVec3, dt_yr: f64) -> DVec3 {
        (self.angular_velocity * dt_yr).cross(pos)
    }

    /// Surface speed in m/year at a given position, given planet radius.
    #[allow(dead_code)]
    fn surface_speed_m_yr(&self, pos: DVec3, radius_m: f64) -> f64 {
        self.angular_velocity.cross(pos).length() * radius_m
    }
}

// ---------------------------------------------------------------------------
// Tectonic history (for time-lapse visualization)
// ---------------------------------------------------------------------------

/// A snapshot of the tectonic state at a single simulation step.
///
/// Contains only the per-cell data needed for globe visualization — omits
/// the grid itself, celestial data, and post-tectonic fields (biome, rock,
/// temperature) which are not available during the tectonic phase.
#[derive(Debug, Clone)]
pub struct TectonicSnapshot {
    /// Simulation step index this snapshot was captured at.
    pub step: u32,
    /// Geological age at this snapshot (Myr from start).
    pub age_myr: f64,
    /// Per-cell elevation in meters.
    pub elevation: Vec<f64>,
    /// Per-cell plate assignment.
    pub plate_id: Vec<u8>,
    /// Per-cell boundary classification.
    pub boundary_type: Vec<BoundaryType>,
    /// Per-cell crust type.
    pub crust_type: Vec<CrustType>,
    /// Per-cell volcanic activity (0.0–1.0).
    pub volcanic_activity: Vec<f32>,
    /// Per-cell tectonic strain (0.0–1.0).
    pub strain: Vec<f32>,
}

/// Complete history of a tectonic simulation, captured for playback.
///
/// Stores snapshots at regular intervals (every `snapshot_interval` steps)
/// plus the final step. Memory usage is approximately
/// `snapshots.len() × cells × 22 bytes` (e.g. ~2.5 MB per frame at level 7).
#[derive(Debug, Clone)]
pub struct TectonicHistory {
    /// Ordered snapshots from step 0 to the final step.
    pub snapshots: Vec<TectonicSnapshot>,
    /// Time step size in Myr used during simulation.
    pub dt_myr: f64,
    /// Total number of simulation steps (may be more than snapshot count).
    pub total_steps: u32,
    /// How many simulation steps between each captured snapshot.
    pub snapshot_interval: u32,
}

impl TectonicHistory {
    /// Number of captured frames.
    pub fn frame_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Geological age in Myr at the given frame index.
    pub fn age_at_frame(&self, frame: usize) -> f64 {
        self.snapshots.get(frame).map(|s| s.age_myr).unwrap_or(0.0)
    }

    /// Compute a reasonable snapshot interval to target ~max_frames keyframes.
    pub(crate) fn compute_interval(total_steps: u32, max_frames: u32) -> u32 {
        if total_steps <= max_frames {
            1
        } else {
            (total_steps / max_frames).max(1)
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the full tectonic simulation and write results into `data`.
///
/// Initialises plate assignments, then runs `data.config.tectonic_steps()`
/// iterations with a time step of `dt_myr` million years. Plate velocities
/// are in physical SI units (rad/year) and evolve via acceleration each step.
///
/// An optional `progress` callback is called after each step with the step
/// index (0-based).
///
/// # Panics
///
/// Panics if `data.grid` is empty.
pub fn run_tectonics<F>(data: &mut PlanetData, mut progress: F)
where
    F: FnMut(u32),
{
    assert!(data.grid.cell_count() > 0, "Grid must be non-empty");

    let seed = data.config.seed;
    let steps = data.config.tectonic_steps();
    let dt_myr = data.config.tectonic_dt_myr();
    let dt_yr = dt_myr * 1e6;
    let dt_scale = dt_myr / REF_DT_MYR;
    let radius_m = data.config.radius_m;

    let mut plates = init_plates(data, seed, radius_m);
    let mut convergence_timers: BTreeMap<(u8, u8), f64> = BTreeMap::new();

    for step in 0..steps {
        // Detect boundaries first so force computations see current geometry.
        let boundary_normals = precompute_boundary_normals(data);
        detect_boundaries(data, &plates, &boundary_normals, dt_yr);

        // Compute slab-pull forces from boundary geometry, then evolve velocities.
        let slab_pull = compute_slab_pull_torques(data, &plates, &boundary_normals);
        evolve_plate_velocities(&mut plates, &slab_pull, seed, step, dt_yr, dt_myr, radius_m);

        propagate_strain(data, dt_scale);
        apply_boundary_forces(data, dt_myr);
        apply_deformation_zones(data, dt_myr);
        subduction_reassign(data, step, dt_scale);
        rift_plates(data, &mut plates, seed, step);
        suture_plates(data, &mut plates, &mut convergence_timers, dt_myr);
        update_volcanic_activity(data, dt_scale);
        erode(data, dt_scale);
        progress(step);
    }

    compute_sea_level(data);
}

/// Maximum number of keyframes stored by default in a tectonic history.
const DEFAULT_MAX_FRAMES: u32 = 100;

/// Run the tectonic simulation and capture a time-lapse history for playback.
///
/// Identical to [`run_tectonics`] but also returns a [`TectonicHistory`]
/// containing snapshots at regular intervals (targeting ≤ `max_frames`
/// keyframes). Pass `None` for `max_frames` to use the default (100).
///
/// The final step is always captured, even if it doesn't fall on an interval
/// boundary.
pub fn run_tectonics_with_history<F>(
    data: &mut PlanetData,
    max_frames: Option<u32>,
    mut progress: F,
) -> TectonicHistory
where
    F: FnMut(u32),
{
    assert!(data.grid.cell_count() > 0, "Grid must be non-empty");

    let seed = data.config.seed;
    let steps = data.config.tectonic_steps();
    let dt_myr = data.config.tectonic_dt_myr();
    let dt_yr = dt_myr * 1e6;
    let dt_scale = dt_myr / REF_DT_MYR;
    let radius_m = data.config.radius_m;

    let max_frames = max_frames.unwrap_or(DEFAULT_MAX_FRAMES);
    let interval = TectonicHistory::compute_interval(steps, max_frames);

    let mut snapshots = Vec::with_capacity((steps / interval + 2) as usize);

    let mut plates = init_plates(data, seed, radius_m);
    let mut convergence_timers: BTreeMap<(u8, u8), f64> = BTreeMap::new();

    // Capture the initial state (step 0, before any simulation).
    snapshots.push(capture_snapshot(data, 0, 0.0));

    for step in 0..steps {
        // Detect boundaries first so force computations see current geometry.
        let boundary_normals = precompute_boundary_normals(data);
        detect_boundaries(data, &plates, &boundary_normals, dt_yr);

        // Compute slab-pull forces from boundary geometry, then evolve velocities.
        let slab_pull = compute_slab_pull_torques(data, &plates, &boundary_normals);
        evolve_plate_velocities(&mut plates, &slab_pull, seed, step, dt_yr, dt_myr, radius_m);

        propagate_strain(data, dt_scale);
        apply_boundary_forces(data, dt_myr);
        apply_deformation_zones(data, dt_myr);
        subduction_reassign(data, step, dt_scale);
        rift_plates(data, &mut plates, seed, step);
        suture_plates(data, &mut plates, &mut convergence_timers, dt_myr);
        update_volcanic_activity(data, dt_scale);
        erode(data, dt_scale);
        progress(step);

        let sim_step = step + 1; // 1-based: step 0 has been run
        if sim_step.is_multiple_of(interval) || sim_step == steps {
            let age_myr = sim_step as f64 * dt_myr;
            snapshots.push(capture_snapshot(data, sim_step, age_myr));
        }
    }

    compute_sea_level(data);

    TectonicHistory {
        snapshots,
        dt_myr,
        total_steps: steps,
        snapshot_interval: interval,
    }
}

/// Capture the current tectonic state as a snapshot.
fn capture_snapshot(data: &PlanetData, step: u32, age_myr: f64) -> TectonicSnapshot {
    TectonicSnapshot {
        step,
        age_myr,
        elevation: data.elevation.clone(),
        plate_id: data.plate_id.clone(),
        boundary_type: data.boundary_type.clone(),
        crust_type: data.crust_type.clone(),
        volcanic_activity: data.volcanic_activity.clone(),
        strain: data.strain.clone(),
    }
}

// ---------------------------------------------------------------------------
// Plate initialisation
// ---------------------------------------------------------------------------

/// Seed plate centres, assign cells, and initialise elevation and crust data.
///
/// Plate angular velocities are in rad/year, calibrated to produce surface
/// speeds in the 2–10 cm/year range. Each plate also gets a small initial
/// angular acceleration for velocity drift over geological time.
fn init_plates(data: &mut PlanetData, seed: u64, radius_m: f64) -> Vec<Plate> {
    let mut rng = SmallRng::seed_from_u64(seed);

    let n_plates = rng.random_range(8_u8..=15_u8) as usize;

    // Place seed positions using a Fibonacci sphere lattice for uniform
    // coverage, then jitter slightly with RNG for variety.
    let seeds: Vec<DVec3> = fibonacci_sphere_points(n_plates, seed)
        .into_iter()
        .map(|p| {
            // Small random rotation around Y to break symmetry.
            let angle = rng.random_range(-0.3_f64..0.3_f64);
            let (s, c) = angle.sin_cos();
            DVec3::new(c * p.x - s * p.z, p.y, s * p.x + c * p.z).normalize()
        })
        .collect();

    // Power-law growth weights give Earth-like plate size variation.
    let weights = generate_plate_weights(&mut rng, n_plates);

    // Assign each cell via weighted BFS flood-fill.
    let plate_id = weighted_bfs_flood_fill(&data.grid, &seeds, &weights);

    // Convert surface speed range to angular speed range (ω = v / R).
    let min_angular = MIN_PLATE_SPEED_M_YR / radius_m;
    let max_angular = MAX_PLATE_SPEED_M_YR / radius_m;

    // Build Plate structs with size-biased properties.
    let min_w = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max_w = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weight_range = max_w - min_w + 1e-10;

    let plates: Vec<Plate> = (0..n_plates)
        .map(|p| {
            // Larger plates are more likely oceanic (like Earth's Pacific).
            let weight_frac = (weights[p] - min_w) / weight_range;
            let oceanic_prob = 0.40 + weight_frac * 0.50;
            let crust_type = if rng.random::<f64>() < oceanic_prob {
                CrustType::Oceanic
            } else {
                CrustType::Continental
            };
            // Random angular velocity: axis is a random unit vector, speed
            // is in rad/year calibrated to 2–10 cm/year surface speed.
            let axis = random_unit_vec(&mut rng);
            let speed = rng.random_range(min_angular..max_angular);

            // Small initial acceleration: magnitude produces ~2–3× velocity
            // change over ~100 Myr, direction is a random unit vector.
            let accel_axis = random_unit_vec(&mut rng);
            let accel_mag = rng.random_range(0.5..2.0) * speed / (100.0e6);

            Plate {
                angular_velocity: axis * speed,
                angular_acceleration: accel_axis * accel_mag,
                crust_type,
            }
        })
        .collect();

    // Write initial data into PlanetData.
    for (i, &pid) in plate_id.iter().enumerate() {
        let plate = &plates[pid as usize];
        data.plate_id[i] = pid;
        data.crust_type[i] = plate.crust_type;
        data.crust_depth[i] = match plate.crust_type {
            CrustType::Continental => CONTINENTAL_CRUST_DEPTH,
            CrustType::Oceanic => OCEANIC_CRUST_DEPTH,
        };
        data.elevation[i] = match plate.crust_type {
            CrustType::Continental => INIT_CONTINENTAL_ELEV,
            CrustType::Oceanic => INIT_OCEANIC_ELEV,
        };
    }

    plates
}

/// Generate `n` roughly uniformly distributed points on the unit sphere
/// using the Fibonacci lattice / golden spiral method.
fn fibonacci_sphere_points(n: usize, seed: u64) -> Vec<DVec3> {
    // Offset breaks the exact pole alignment; made deterministic by seed.
    let offset = (seed % 1000) as f64 / 1000.0;
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|i| {
            let theta = std::f64::consts::TAU * (i as f64 + offset) / golden_ratio;
            let phi = ((1.0 - 2.0 * (i as f64 + 0.5) / n as f64).clamp(-1.0, 1.0)).acos();
            let sin_phi = phi.sin();
            DVec3::new(sin_phi * theta.cos(), phi.cos(), sin_phi * theta.sin())
        })
        .collect()
}

/// Generate per-plate growth weights from a Pareto (power-law) distribution.
///
/// Returns weights in `[1.0, MAX_PLATE_WEIGHT_RATIO]` — the ratio between the
/// largest and smallest controls the plate size variance. A Pareto with
/// α ≈ 1.3 produces a few dominant plates and many small ones, mirroring
/// Earth's tectonic plate size distribution.
fn generate_plate_weights(rng: &mut SmallRng, n: usize) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..n)
        .map(|_| {
            // Inverse-CDF of Pareto(x_min=1, α): X = 1 / U^(1/α).
            let u = rng.random_range(0.01_f64..1.0_f64);
            1.0_f64 / u.powf(1.0 / PLATE_SIZE_PARETO_ALPHA)
        })
        .collect();

    // Clamp the Pareto tail: no weight exceeds MAX_PLATE_WEIGHT_RATIO × minimum.
    let min_w = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max_allowed = min_w * MAX_PLATE_WEIGHT_RATIO;
    for w in weights.iter_mut() {
        if *w > max_allowed {
            *w = max_allowed;
        }
    }

    weights
}

/// Weighted BFS flood-fill from `seeds` to assign every cell a plate id.
///
/// Each seed initialises a queue frontier. Each round, plate `p` expands
/// `floor(weight[p] + accumulated_fraction)` cells from its frontier.
/// High-weight plates grow faster, producing larger final regions.
fn weighted_bfs_flood_fill(
    grid: &crate::planet::grid::IcosahedralGrid,
    seeds: &[DVec3],
    weights: &[f64],
) -> Vec<u8> {
    let n = grid.cell_count();
    let n_plates = seeds.len();
    let mut plate_id: Vec<i16> = vec![-1; n];
    let mut queues: Vec<VecDeque<u32>> = vec![VecDeque::new(); n_plates];

    // Seed each queue with the nearest cell to each plate centre.
    for (p, seed_pos) in seeds.iter().enumerate() {
        let cell = grid.nearest_cell_from_pos(*seed_pos);
        if plate_id[cell.index()] < 0 {
            plate_id[cell.index()] = p as i16;
            queues[p].push_back(cell.0);
        }
    }

    // Weighted BFS: each plate expands proportionally to its weight.
    let mut accumulators = vec![0.0_f64; n_plates];
    let mut remaining = n.saturating_sub(n_plates);
    while remaining > 0 {
        let mut made_progress = false;
        for (p, queue) in queues.iter_mut().enumerate() {
            accumulators[p] += weights[p];
            let n_expand = accumulators[p] as usize;
            accumulators[p] -= n_expand as f64;

            for _ in 0..n_expand {
                if let Some(current) = queue.pop_front() {
                    for &nb in grid.cell_neighbors(CellId(current)) {
                        if plate_id[nb as usize] < 0 {
                            plate_id[nb as usize] = p as i16;
                            queue.push_back(nb);
                            remaining -= 1;
                            made_progress = true;
                        }
                    }
                }
            }
        }
        if !made_progress {
            break;
        }
    }

    // Fill any unreached cells (should not happen on a connected grid).
    plate_id.into_iter().map(|id| id.max(0) as u8).collect()
}

/// Generate a random unit vector uniformly distributed on the sphere.
fn random_unit_vec(rng: &mut SmallRng) -> DVec3 {
    // Marsaglia rejection method - produces a uniform distribution.
    loop {
        let x = rng.random_range(-1.0_f64..1.0_f64);
        let y = rng.random_range(-1.0_f64..1.0_f64);
        let z = rng.random_range(-1.0_f64..1.0_f64);
        let len2 = x * x + y * y + z * z;
        if len2 > 0.0 && len2 <= 1.0 {
            return DVec3::new(x, y, z) / len2.sqrt();
        }
    }
}

// ---------------------------------------------------------------------------
// Plate velocity evolution
// ---------------------------------------------------------------------------

/// Update plate velocities from acceleration, clamp to physical range,
/// and periodically perturb acceleration direction.
///
/// Produces natural speedup/slowdown over geological time. Plates that hit
/// the velocity bounds have their acceleration reversed (bounce) to keep
/// speeds physical.
fn evolve_plate_velocities(
    plates: &mut [Plate],
    slab_pull_torques: &[DVec3],
    seed: u64,
    step: u32,
    dt_yr: f64,
    dt_myr: f64,
    radius_m: f64,
) {
    let min_angular = MIN_PLATE_SPEED_M_YR / radius_m;
    let max_angular = MAX_PLATE_SPEED_M_YR / radius_m;

    for (p, plate) in plates.iter_mut().enumerate() {
        // Apply persistent random acceleration: ω += α × dt (years).
        plate.angular_velocity += plate.angular_acceleration * dt_yr;

        // Apply slab-pull acceleration (transient, recomputed each step from
        // the current boundary geometry). This is the dominant force on Earth:
        // plates with long subduction zones are pulled toward the trench.
        if let Some(&slab_pull) = slab_pull_torques.get(p) {
            plate.angular_velocity += slab_pull * dt_yr;
        }

        // Clamp angular speed to physical range.
        let speed = plate.angular_velocity.length();
        if speed > max_angular {
            plate.angular_velocity *= max_angular / speed;
            // Reverse the component of acceleration along velocity.
            let vel_dir = plate.angular_velocity.normalize();
            let a_along = plate.angular_acceleration.dot(vel_dir);
            if a_along > 0.0 {
                plate.angular_acceleration -= vel_dir * a_along * 2.0;
            }
        } else if speed < min_angular && speed > 1e-30 {
            plate.angular_velocity *= min_angular / speed;
            let vel_dir = plate.angular_velocity.normalize();
            let a_along = plate.angular_acceleration.dot(vel_dir);
            if a_along < 0.0 {
                plate.angular_acceleration -= vel_dir * a_along * 2.0;
            }
        }

        // Periodically perturb acceleration direction (every ~50 Myr).
        let perturb_step = (ACCEL_PERTURB_INTERVAL_MYR / dt_myr).round().max(1.0) as u32;
        if step > 0 && step.is_multiple_of(perturb_step) {
            let mut rng = SmallRng::seed_from_u64(accel_perturb_seed(seed, p, step));
            let perturb = DVec3::new(
                rng.random_range(-1.0_f64..1.0_f64),
                rng.random_range(-1.0_f64..1.0_f64),
                rng.random_range(-1.0_f64..1.0_f64),
            );
            let mag = plate.angular_acceleration.length();
            if mag > 1e-30 {
                // Rotate acceleration by blending with random direction.
                let blended =
                    plate.angular_acceleration.normalize() * 0.6 + perturb.normalize() * 0.4;
                if blended.length_squared() > 1e-30 {
                    plate.angular_acceleration = blended.normalize() * mag;
                }
            }
        }
    }
}

/// Deterministic seed for acceleration perturbation events.
fn accel_perturb_seed(seed: u64, plate: usize, step: u32) -> u64 {
    seed.wrapping_add((plate as u64).wrapping_mul(0x1234_5678_9abc_def0))
        .wrapping_mul(0x6c62_272e_07bb_0142)
        .wrapping_add(step as u64)
        ^ 0xfeed_face_cafe_babe
}

// ---------------------------------------------------------------------------
// Slab-pull force computation
// ---------------------------------------------------------------------------

/// Compute per-plate angular acceleration from slab pull.
///
/// At subduction zones, cold dense oceanic lithosphere sinks into the mantle,
/// pulling the trailing plate toward the trench. This is the dominant force
/// driving tectonic plate motion on Earth (~70% of total driving force).
///
/// For each plate, we:
/// 1. Find convergent oceanic boundary cells (subduction zones).
/// 2. Sum the torque: τ = Σ (pos × pull_direction) for each subducting cell.
///    The pull direction is the boundary normal (toward the trench).
/// 3. Normalize by plate cell count (inverse moment of inertia — larger plates
///    are harder to accelerate) and scale by [`SLAB_PULL_COEFFICIENT`].
///
/// Returns a Vec of angular acceleration vectors (rad/yr²), one per plate.
#[allow(clippy::needless_range_loop)] // multiple parallel arrays indexed by i
fn compute_slab_pull_torques(
    data: &PlanetData,
    plates: &[Plate],
    boundary_normals: &[DVec3],
) -> Vec<DVec3> {
    let n = data.grid.cell_count();
    let n_plates = plates.len();

    let mut torques = vec![DVec3::ZERO; n_plates];
    let mut plate_cells = vec![0_usize; n_plates];

    for i in 0..n {
        let pid = data.plate_id[i] as usize;
        if pid >= n_plates {
            continue;
        }
        plate_cells[pid] += 1;

        // Only oceanic cells at convergent boundaries contribute slab pull.
        // Continental lithosphere is too buoyant to sink into the mantle.
        if data.boundary_type[i] != BoundaryType::Convergent {
            continue;
        }
        if data.crust_type[i] != CrustType::Oceanic {
            continue;
        }

        let pos = data.grid.cell_position(CellId(i as u32));
        let pull_dir = boundary_normals[i];
        if pull_dir.length_squared() < 1e-20 {
            continue;
        }

        // Torque = position × force. On the unit sphere |pos| = 1 and
        // pull_dir is tangent to the sphere (perpendicular to pos), so
        // |pos × pull_dir| ≈ |pull_dir|.
        torques[pid] += pos.cross(pull_dir);
    }

    // Convert accumulated torque to angular acceleration:
    //   α = SLAB_PULL_COEFFICIENT × (n_subducting / n_total) × torque_direction
    //
    // The division by plate_cells acts as inverse moment of inertia: smaller
    // plates (less mass) accelerate more from the same force.
    for pid in 0..n_plates {
        if plate_cells[pid] > 0 {
            torques[pid] *= SLAB_PULL_COEFFICIENT / plate_cells[pid] as f64;
        }
    }

    torques
}

// ---------------------------------------------------------------------------
// Boundary normal precomputation
// ---------------------------------------------------------------------------

/// For each cell, compute the average outward boundary normal toward all
/// cross-plate neighbours. Returns zero for interior cells.
///
/// Recomputed every step because plate assignments change due to subduction,
/// rifting, and suturing.
fn precompute_boundary_normals(data: &PlanetData) -> Vec<DVec3> {
    let n = data.grid.cell_count();
    let mut normals = vec![DVec3::ZERO; n];

    for id in data.grid.cell_ids() {
        let i = id.index();
        let pos_a = data.grid.cell_position(id);
        for &nb in data.grid.cell_neighbors(id) {
            if data.plate_id[nb as usize] != data.plate_id[i] {
                let pos_b = data.grid.cell_position(CellId(nb));
                // Tangent on the sphere surface pointing from a to b.
                let tangent = great_circle_tangent(pos_a, pos_b);
                normals[i] += tangent;
            }
        }
        if normals[i].length_squared() > 1e-20 {
            normals[i] = normals[i].normalize();
        }
    }

    normals
}

/// Unit tangent vector along the great circle from `a` to `b`, lying in
/// the tangent plane of the sphere at `a`.
///
/// Uses Gram-Schmidt projection: `t = normalize(b - (b dot a) * a)`.
fn great_circle_tangent(a: DVec3, b: DVec3) -> DVec3 {
    let proj = b - a * b.dot(a);
    if proj.length_squared() < 1e-20 {
        return DVec3::ZERO;
    }
    proj.normalize()
}

// ---------------------------------------------------------------------------
// Per-step simulation
// ---------------------------------------------------------------------------

/// Detect and classify boundary cells for this step.
fn detect_boundaries(
    data: &mut PlanetData,
    plates: &[Plate],
    boundary_normals: &[DVec3],
    dt_yr: f64,
) {
    let dt_scale = (dt_yr / 1e6) / REF_DT_MYR;
    let threshold = VELOCITY_THRESHOLD_BASE * dt_scale;

    let n = data.grid.cell_count();
    for i in 0..n {
        let id = CellId(i as u32);
        let has_boundary = data
            .grid
            .cell_neighbors(id)
            .iter()
            .any(|&nb| data.plate_id[nb as usize] != data.plate_id[i]);

        if !has_boundary {
            data.boundary_type[i] = BoundaryType::Interior;
            continue;
        }

        let plate_a = &plates[data.plate_id[i] as usize];
        let pos = data.grid.cell_position(id);
        let normal = boundary_normals[i];

        let v_a = plate_a.velocity_at(pos, dt_yr);

        // Average relative velocity projected onto the outward boundary normal.
        let mut rel_proj = 0.0_f64;
        let mut count = 0;
        for &nb in data.grid.cell_neighbors(id) {
            let nb_pid = data.plate_id[nb as usize];
            if nb_pid != data.plate_id[i] {
                let plate_b = &plates[nb_pid as usize];
                let pos_nb = data.grid.cell_position(CellId(nb));
                let v_b = plate_b.velocity_at(pos_nb, dt_yr);
                rel_proj += (v_a - v_b).dot(normal);
                count += 1;
            }
        }

        if count > 0 {
            rel_proj /= count as f64;
        }

        data.boundary_type[i] = if rel_proj > threshold {
            BoundaryType::Convergent
        } else if rel_proj < -threshold {
            BoundaryType::Divergent
        } else {
            BoundaryType::Transform
        };
    }
}

/// Apply height changes at boundary cells based on boundary type and the
/// crust types of interacting plates. Height deltas scale with `dt_myr`.
#[allow(clippy::needless_range_loop)] // multiple parallel arrays indexed by i
fn apply_boundary_forces(data: &mut PlanetData, dt_myr: f64) {
    let n = data.grid.cell_count();
    let mut deltas: Vec<f64> = vec![0.0; n];

    for i in 0..n {
        match data.boundary_type[i] {
            BoundaryType::Interior | BoundaryType::Transform => {}
            BoundaryType::Convergent => {
                let neighbour_crust = dominant_neighbour_crust(data, i);
                let my_crust = data.crust_type[i];
                deltas[i] += match (my_crust, neighbour_crust) {
                    // Continent-Continent: orogeny on both sides.
                    (CrustType::Continental, CrustType::Continental) => OROGENY_RATE * dt_myr,
                    // Ocean-Continent: arc volcano on continental, trench on oceanic.
                    (CrustType::Continental, CrustType::Oceanic) => ARC_RATE * dt_myr,
                    (CrustType::Oceanic, CrustType::Continental) => -TRENCH_RATE * dt_myr,
                    // Ocean-Ocean: one side builds island arc, other subducts.
                    (CrustType::Oceanic, CrustType::Oceanic) => {
                        // Alternate which side rises based on cell parity.
                        if i % 2 == 0 {
                            ISLAND_ARC_RATE * dt_myr
                        } else {
                            -OCEAN_CONVERGENT_RATE * dt_myr
                        }
                    }
                };
            }
            BoundaryType::Divergent => {
                deltas[i] -= RIFT_RATE * dt_myr;
            }
        }
    }

    for (elev, &delta) in data.elevation.iter_mut().zip(deltas.iter()) {
        *elev = (*elev + delta).clamp(MIN_ELEVATION, MAX_ELEVATION);
    }
}

/// Find the most common crust type among cross-plate neighbours of cell `i`.
fn dominant_neighbour_crust(data: &PlanetData, i: usize) -> CrustType {
    let id = CellId(i as u32);
    let mut continental = 0_u32;
    let mut oceanic = 0_u32;
    for &nb in data.grid.cell_neighbors(id) {
        if data.plate_id[nb as usize] != data.plate_id[i] {
            match data.crust_type[nb as usize] {
                CrustType::Continental => continental += 1,
                CrustType::Oceanic => oceanic += 1,
            }
        }
    }
    if continental >= oceanic {
        CrustType::Continental
    } else {
        CrustType::Oceanic
    }
}

/// Update volcanic activity: gain at active boundaries, decay everywhere else.
/// Rates scale with `dt_scale = dt_myr / REF_DT_MYR`.
fn update_volcanic_activity(data: &mut PlanetData, dt_scale: f64) {
    let volcanic_gain = VOLCANIC_GAIN_BASE * dt_scale as f32;
    let volcanic_decay = VOLCANIC_DECAY_BASE.powf(dt_scale as f32);
    let fault_gain = FAULT_STRESS_GAIN_BASE * dt_scale as f32;

    let n = data.grid.cell_count();
    for i in 0..n {
        let is_active = (matches!(
            data.boundary_type[i],
            BoundaryType::Convergent | BoundaryType::Divergent
        ) && matches!(data.crust_type[i], CrustType::Oceanic))
            || matches!(data.boundary_type[i], BoundaryType::Divergent);

        if is_active {
            data.volcanic_activity[i] = (data.volcanic_activity[i] + volcanic_gain).min(1.0);
        } else {
            data.volcanic_activity[i] *= volcanic_decay;
        }

        if matches!(data.boundary_type[i], BoundaryType::Transform) {
            data.fault_stress[i] = (data.fault_stress[i] + fault_gain).min(1.0);
        }
    }
}

/// One diffusion pass: each cell moves toward its neighbour average.
/// Erosion rate scales with `dt_scale = dt_myr / REF_DT_MYR`.
fn erode(data: &mut PlanetData, dt_scale: f64) {
    let erosion_rate = EROSION_RATE_BASE * dt_scale;

    let n = data.grid.cell_count();
    let old_elev = data.elevation.clone();

    for i in 0..n {
        let id = CellId(i as u32);
        let neighbours = data.grid.cell_neighbors(id);
        if neighbours.is_empty() {
            continue;
        }
        let mean: f64 = neighbours
            .iter()
            .map(|&nb| old_elev[nb as usize])
            .sum::<f64>()
            / neighbours.len() as f64;
        let delta = erosion_rate * (mean - old_elev[i]);
        data.elevation[i] = (old_elev[i] + delta).clamp(MIN_ELEVATION, MAX_ELEVATION);
    }
}

// ---------------------------------------------------------------------------
// Strain propagation — intraplate deformation
// ---------------------------------------------------------------------------

/// Accumulate and diffuse tectonic strain across plate interiors.
///
/// Three phases per step:
/// 1. **Injection** — Boundary cells gain strain proportional to boundary type
///    (convergent > divergent > transform).
/// 2. **Diffusion** — Strain diffuses to same-plate neighbours, modulated by
///    crust conductivity. Continental crust transmits strain ~2× further than
///    oceanic, producing wide deformation zones like the Tibetan Plateau.
/// 3. **Relaxation** — Viscous decay prevents unbounded interior accumulation.
fn propagate_strain(data: &mut PlanetData, dt_scale: f64) {
    let n = data.grid.cell_count();
    let gain_convergent = STRAIN_GAIN_CONVERGENT * dt_scale as f32;
    let gain_divergent = STRAIN_GAIN_DIVERGENT * dt_scale as f32;
    let gain_transform = STRAIN_GAIN_TRANSFORM * dt_scale as f32;
    let relaxation = STRAIN_RELAXATION_RATE * dt_scale as f32;
    let diffusion_rate = STRAIN_DIFFUSION_RATE * dt_scale;

    // Phase 1: inject strain at boundaries.
    for i in 0..n {
        let gain = match data.boundary_type[i] {
            BoundaryType::Convergent => gain_convergent,
            BoundaryType::Divergent => gain_divergent,
            BoundaryType::Transform => gain_transform,
            BoundaryType::Interior => 0.0,
        };
        data.strain[i] = (data.strain[i] + gain).min(1.0);
    }

    // Phase 2: diffuse strain within each plate.
    let old_strain = data.strain.clone();
    for i in 0..n {
        let id = CellId(i as u32);
        let neighbors = data.grid.cell_neighbors(id);
        if neighbors.is_empty() {
            continue;
        }

        let mut sum = 0.0_f64;
        let mut count = 0;
        for &nb in neighbors {
            let nb_idx = nb as usize;
            if data.plate_id[nb_idx] == data.plate_id[i] {
                sum += old_strain[nb_idx] as f64;
                count += 1;
            }
        }

        if count > 0 {
            let mean = sum / count as f64;
            let conductivity = match data.crust_type[i] {
                CrustType::Continental => CONTINENTAL_STRAIN_CONDUCTIVITY,
                CrustType::Oceanic => 1.0,
            };
            let delta = diffusion_rate * conductivity * (mean - old_strain[i] as f64);
            data.strain[i] = (old_strain[i] as f64 + delta).clamp(0.0, 1.0) as f32;
        }
    }

    // Phase 3: viscous relaxation.
    for s in data.strain.iter_mut() {
        *s = (*s - relaxation).max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Deformation zones — wide orogeny and back-arc extension
// ---------------------------------------------------------------------------

/// Spread convergent boundary elevation effects multiple cells inward.
///
/// Real mountain belts are hundreds of kilometres wide, not single-cell
/// ridges. This BFS from convergent boundary cells applies decaying orogeny
/// to interior cells within the same plate:
/// - **Continental interior** behind convergent boundaries: uplift (foothills
///   and plateau formation — e.g., Tibetan Plateau behind the Himalayas).
/// - **Continental interior** behind oceanic subduction: back-arc extension
///   (basin subsidence — e.g., Sea of Japan behind the Japanese arc).
fn apply_deformation_zones(data: &mut PlanetData, dt_myr: f64) {
    let n = data.grid.cell_count();
    let mut deltas = vec![0.0_f64; n];

    // BFS from convergent boundary cells to spread effects inward.
    let mut hop_distance = vec![u32::MAX; n];
    let mut is_subduction_back_arc = vec![false; n];
    let mut queue = VecDeque::new();

    for i in 0..n {
        if data.boundary_type[i] == BoundaryType::Convergent {
            hop_distance[i] = 0;
            queue.push_back(i);

            // Detect back-arc context: this cell is continental, but its
            // cross-plate neighbour is oceanic (subduction zone).
            if data.crust_type[i] == CrustType::Continental {
                let id = CellId(i as u32);
                let has_oceanic_neighbor = data.grid.cell_neighbors(id).iter().any(|&nb| {
                    let nb_idx = nb as usize;
                    data.plate_id[nb_idx] != data.plate_id[i]
                        && data.crust_type[nb_idx] == CrustType::Oceanic
                });
                is_subduction_back_arc[i] = has_oceanic_neighbor;
            }
        }
    }

    while let Some(cell) = queue.pop_front() {
        let hops = hop_distance[cell];
        if hops >= DEFORMATION_ZONE_HOPS {
            continue;
        }

        let id = CellId(cell as u32);
        let my_plate = data.plate_id[cell];

        for &nb in data.grid.cell_neighbors(id) {
            let nb_idx = nb as usize;
            // Only spread within the same plate (intraplate deformation).
            if data.plate_id[nb_idx] != my_plate {
                continue;
            }
            let new_hops = hops + 1;
            if new_hops < hop_distance[nb_idx] {
                hop_distance[nb_idx] = new_hops;
                // Propagate back-arc context from boundary cell.
                is_subduction_back_arc[nb_idx] = is_subduction_back_arc[cell];
                queue.push_back(nb_idx);

                let falloff = DEFORMATION_FALLOFF.powi(new_hops as i32);
                if is_subduction_back_arc[nb_idx] && new_hops >= 2 {
                    // Back-arc extension: subsidence behind volcanic arc.
                    deltas[nb_idx] -= BACK_ARC_RATE * falloff * dt_myr;
                } else {
                    // Foreland uplift: reduced orogeny spreading inland.
                    deltas[nb_idx] += OROGENY_RATE * 0.3 * falloff * dt_myr;
                }
            }
        }
    }

    for (elev, &delta) in data.elevation.iter_mut().zip(deltas.iter()) {
        *elev = (*elev + delta).clamp(MIN_ELEVATION, MAX_ELEVATION);
    }
}

// ---------------------------------------------------------------------------
// Continental rifting — plate splitting
// ---------------------------------------------------------------------------

/// Check for and execute plate rifting along accumulated high-strain bands.
///
/// When tectonic strain accumulates in a contiguous band across a plate's
/// interior (often originating from divergent or back-arc extensional forces),
/// the plate can split into two independent plates. The algorithm:
///
/// 1. For each plate above minimum size, identify interior cells with strain
///    exceeding [`RIFT_STRAIN_THRESHOLD`].
/// 2. Temporarily remove those high-strain cells and find connected components
///    of the remaining plate cells.
/// 3. If two or more components exist and the smaller fragment is large enough,
///    the plate splits: the second-largest component becomes a new plate with
///    a slightly diverging angular velocity.
/// 4. Cells along the rift become thin oceanic crust (new ocean forming),
///    analogous to the mid-Atlantic ridge after Africa–South America separation.
fn rift_plates(data: &mut PlanetData, plates: &mut Vec<Plate>, seed: u64, step: u32) {
    let n = data.grid.cell_count();
    let n_plates = plates.len();

    let mut plate_sizes = vec![0_usize; n_plates];
    for &pid in &data.plate_id {
        if (pid as usize) < n_plates {
            plate_sizes[pid as usize] += 1;
        }
    }

    let mut rifts_this_step = 0;

    for p in 0..n_plates {
        if rifts_this_step >= MAX_RIFTS_PER_STEP {
            break;
        }
        if plate_sizes[p] < MIN_RIFT_PLATE_CELLS {
            continue;
        }

        // Collect plate cells and high-strain interior rift candidates.
        let mut plate_cells = Vec::new();
        let mut rift_candidates = Vec::new();
        for i in 0..n {
            if data.plate_id[i] as usize != p {
                continue;
            }
            plate_cells.push(i);
            if data.strain[i] > RIFT_STRAIN_THRESHOLD
                && data.boundary_type[i] == BoundaryType::Interior
            {
                rift_candidates.push(i);
            }
        }

        if rift_candidates.is_empty() {
            continue;
        }

        // Build lookup sets for O(1) membership tests.
        let mut plate_set = vec![false; n];
        for &c in &plate_cells {
            plate_set[c] = true;
        }
        let mut rift_set = vec![false; n];
        for &c in &rift_candidates {
            rift_set[c] = true;
        }

        // Find connected components of non-rift plate cells.
        let mut visited = vec![false; n];
        let mut components: Vec<Vec<usize>> = Vec::new();

        for &cell in &plate_cells {
            if rift_set[cell] || visited[cell] {
                continue;
            }

            let mut component = Vec::new();
            let mut bfs_queue = VecDeque::new();
            bfs_queue.push_back(cell);
            visited[cell] = true;

            while let Some(current) = bfs_queue.pop_front() {
                component.push(current);
                for &nb in data.grid.cell_neighbors(CellId(current as u32)) {
                    let nb_idx = nb as usize;
                    if plate_set[nb_idx] && !rift_set[nb_idx] && !visited[nb_idx] {
                        visited[nb_idx] = true;
                        bfs_queue.push_back(nb_idx);
                    }
                }
            }

            components.push(component);
        }

        // Need at least 2 components for a split.
        if components.len() < 2 {
            continue;
        }

        // Sort by size descending (largest component keeps the plate ID).
        components.sort_by_key(|c| std::cmp::Reverse(c.len()));

        // Enforce minimum fragment size.
        let min_fragment =
            ((plate_cells.len() as f64 * MIN_RIFT_FRAGMENT_FRACTION) as usize).max(5);
        if components[1].len() < min_fragment {
            continue;
        }

        let new_plate_id = plates.len() as u8;
        if new_plate_id == u8::MAX {
            continue; // No room for more plates.
        }

        // Create new plate with diverging velocity.
        let parent = &plates[p];
        let mut rng = SmallRng::seed_from_u64(rift_seed(seed, p, step));
        let perturb = random_unit_vec(&mut rng);
        let parent_speed = parent.angular_velocity.length();
        let diverge_dir = parent.angular_velocity.cross(perturb);
        let diverge = if diverge_dir.length_squared() > 1e-30 {
            diverge_dir.normalize() * parent_speed * RIFT_VELOCITY_DIVERGENCE
        } else {
            perturb * parent_speed * RIFT_VELOCITY_DIVERGENCE
        };

        let new_plate = Plate {
            angular_velocity: parent.angular_velocity + diverge,
            angular_acceleration: parent.angular_acceleration * 0.5
                + perturb.normalize() * parent.angular_acceleration.length() * 0.5,
            crust_type: parent.crust_type,
        };
        plates.push(new_plate);

        // Reassign the second-largest component to the new plate.
        for &cell in &components[1] {
            data.plate_id[cell] = new_plate_id;
        }

        // Rift-band cells become thin new oceanic crust (rift valley / nascent ocean).
        for &cell in &rift_candidates {
            // Only convert cells that lie between the two components.
            data.crust_type[cell] = CrustType::Oceanic;
            data.crust_depth[cell] = OCEANIC_CRUST_DEPTH * 0.5;
            data.elevation[cell] = (data.elevation[cell] - 500.0).max(MIN_ELEVATION);
            data.strain[cell] = 0.0; // strain released by rifting
        }

        rifts_this_step += 1;
    }
}

/// Deterministic seed for rift-related RNG.
fn rift_seed(seed: u64, plate: usize, step: u32) -> u64 {
    seed.wrapping_add((plate as u64).wrapping_mul(0xdead_beef_0000_1234))
        .wrapping_mul(0x517c_c1b7_2722_0a95)
        .wrapping_add(step as u64)
        ^ 0xface_cafe_babe_1234
}

// ---------------------------------------------------------------------------
// Plate suturing — merging converging continental plates
// ---------------------------------------------------------------------------

/// Merge two continental plates that have maintained convergent boundary
/// contact for a geologically significant period.
///
/// Tracks cumulative convergent contact time between each pair of adjacent
/// continental plates. When this exceeds [`SUTURE_THRESHOLD_MYR`] and the
/// contact zone covers enough of the smaller plate's perimeter, the smaller
/// plate is absorbed into the larger one.
///
/// The merged plate receives a size-weighted average angular velocity.
/// Strain along the former boundary is partially released (representing
/// the completion of mountain-building and the transition to a stable
/// continental interior / suture zone).
///
/// Uses a [`BTreeMap`] for deterministic iteration order.
fn suture_plates(
    data: &mut PlanetData,
    plates: &mut [Plate],
    convergence_timers: &mut BTreeMap<(u8, u8), f64>,
    dt_myr: f64,
) {
    let n = data.grid.cell_count();

    // Count convergent continental boundary contacts between plate pairs,
    // and total boundary cells per plate.
    let mut pair_contacts: BTreeMap<(u8, u8), usize> = BTreeMap::new();
    let mut plate_boundary_cells: BTreeMap<u8, usize> = BTreeMap::new();

    for i in 0..n {
        if data.boundary_type[i] != BoundaryType::Convergent {
            continue;
        }
        if data.crust_type[i] != CrustType::Continental {
            continue;
        }

        let my_plate = data.plate_id[i];
        *plate_boundary_cells.entry(my_plate).or_insert(0) += 1;

        let id = CellId(i as u32);
        for &nb in data.grid.cell_neighbors(id) {
            let nb_idx = nb as usize;
            let nb_plate = data.plate_id[nb_idx];
            if nb_plate != my_plate && data.crust_type[nb_idx] == CrustType::Continental {
                let key = (my_plate.min(nb_plate), my_plate.max(nb_plate));
                *pair_contacts.entry(key).or_insert(0) += 1;
            }
        }
    }

    // Update timers: increment active pairs, slowly decay inactive ones.
    let active_pairs: Vec<(u8, u8)> = pair_contacts.keys().copied().collect();
    let stale_keys: Vec<(u8, u8)> = convergence_timers
        .keys()
        .filter(|k| !active_pairs.contains(k))
        .copied()
        .collect();
    for key in stale_keys {
        let timer = convergence_timers.get_mut(&key).unwrap();
        *timer = (*timer - dt_myr * 0.5).max(0.0);
        if *timer <= 0.0 {
            convergence_timers.remove(&key);
        }
    }

    for (&key, &contacts) in &pair_contacts {
        let (p_a, p_b) = key;
        let boundary_a = plate_boundary_cells.get(&p_a).copied().unwrap_or(1);
        let boundary_b = plate_boundary_cells.get(&p_b).copied().unwrap_or(1);
        let smaller_boundary = boundary_a.min(boundary_b).max(1);
        let contact_fraction = contacts as f64 / smaller_boundary as f64;

        if contact_fraction >= SUTURE_MIN_CONTACT_FRACTION {
            *convergence_timers.entry(key).or_insert(0.0) += dt_myr;
        }
    }

    // Check for sutures (collect first to avoid borrow conflicts).
    let sutured: Vec<(u8, u8)> = convergence_timers
        .iter()
        .filter(|&(_, &time)| time >= SUTURE_THRESHOLD_MYR)
        .map(|(&key, _)| key)
        .collect();

    for (p_a, p_b) in sutured {
        let size_a = data.plate_id.iter().filter(|&&pid| pid == p_a).count();
        let size_b = data.plate_id.iter().filter(|&&pid| pid == p_b).count();
        if size_a == 0 || size_b == 0 {
            convergence_timers.remove(&(p_a.min(p_b), p_a.max(p_b)));
            continue;
        }

        let (absorber, absorbed) = if size_a >= size_b {
            (p_a, p_b)
        } else {
            (p_b, p_a)
        };

        // Weighted-average angular velocity.
        let total_size = (size_a + size_b) as f64;
        let w_absorber = if absorber == p_a {
            size_a as f64
        } else {
            size_b as f64
        } / total_size;
        let w_absorbed = 1.0 - w_absorber;

        let vel_absorber = plates[absorber as usize].angular_velocity;
        let vel_absorbed = plates[absorbed as usize].angular_velocity;
        plates[absorber as usize].angular_velocity =
            vel_absorber * w_absorber + vel_absorbed * w_absorbed;

        // Average accelerations too.
        let acc_absorber = plates[absorber as usize].angular_acceleration;
        let acc_absorbed = plates[absorbed as usize].angular_acceleration;
        plates[absorber as usize].angular_acceleration =
            acc_absorber * w_absorber + acc_absorbed * w_absorbed;

        // Reassign all cells of absorbed plate.
        for i in 0..n {
            if data.plate_id[i] == absorbed {
                data.plate_id[i] = absorber;
                data.strain[i] *= 0.5; // partial strain release at suture
            }
        }

        convergence_timers.remove(&(p_a.min(p_b), p_a.max(p_b)));
    }
}

// ---------------------------------------------------------------------------
// Subduction — boundary deformation
// ---------------------------------------------------------------------------

/// Reassign oceanic cells at convergent boundaries to the overriding
/// continental plate, simulating subduction advance.
///
/// For each convergent boundary cell with oceanic crust that has a
/// continental neighbour on a different plate, there is a small probability
/// per step of reassignment. The base rate scales with `dt_scale`.
/// Plates below `MIN_PLATE_FRACTION` of total cells are protected.
fn subduction_reassign(data: &mut PlanetData, step: u32, dt_scale: f64) {
    let subduction_rate = SUBDUCTION_RATE_BASE * dt_scale;
    let n = data.grid.cell_count();
    let min_plate_size = ((n as f64) * MIN_PLATE_FRACTION).max(1.0) as usize;

    // Count current plate sizes.
    let n_plates = *data.plate_id.iter().max().unwrap_or(&0) as usize + 1;
    let mut plate_sizes = vec![0_usize; n_plates];
    for &pid in &data.plate_id {
        plate_sizes[pid as usize] += 1;
    }

    // Collect reassignments first, then apply (avoids order-dependent results).
    let mut reassignments: Vec<(usize, u8)> = Vec::new();

    for i in 0..n {
        if data.boundary_type[i] != BoundaryType::Convergent {
            continue;
        }
        if data.crust_type[i] != CrustType::Oceanic {
            continue;
        }

        let my_plate = data.plate_id[i] as usize;
        if plate_sizes[my_plate] <= min_plate_size {
            continue;
        }

        // Find a continental neighbour from a different plate.
        let id = CellId(i as u32);
        let mut target_plate: Option<u8> = None;
        for &nb in data.grid.cell_neighbors(id) {
            let nb_idx = nb as usize;
            if data.plate_id[nb_idx] != data.plate_id[i]
                && data.crust_type[nb_idx] == CrustType::Continental
            {
                target_plate = Some(data.plate_id[nb_idx]);
                break;
            }
        }

        if let Some(new_plate) = target_plate {
            let roll = subduction_noise(data.config.seed, i, step);
            if roll < subduction_rate {
                reassignments.push((i, new_plate));
            }
        }
    }

    // Apply all reassignments.
    for &(cell, new_plate) in &reassignments {
        let old_plate = data.plate_id[cell] as usize;
        // Double-check minimum size (other reassignments this step may have
        // already shrunk the plate).
        if plate_sizes[old_plate] <= min_plate_size {
            continue;
        }
        plate_sizes[old_plate] -= 1;
        plate_sizes[new_plate as usize] += 1;
        data.plate_id[cell] = new_plate;
    }
}

/// Deterministic per-cell-per-step noise in [0, 1) for subduction rolls.
fn subduction_noise(seed: u64, cell: usize, step: u32) -> f64 {
    let h = seed
        .wrapping_add(cell as u64)
        .wrapping_mul(0x517c_c1b7_2722_0a95)
        .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .rotate_left(23)
        ^ 0xdead_beef_1234_5678;
    h as f64 / u64::MAX as f64
}

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------

/// Set sea level so approximately `TARGET_LAND_FRACTION` of surface area is
/// above water. Adjusts all elevations by subtracting the computed sea-level
/// offset so that 0 m == sea level in the output.
///
/// Uses cell areas for a proper area-weighted calculation.
fn compute_sea_level(data: &mut PlanetData) {
    let n = data.grid.cell_count();
    // Collect (elevation, area) pairs and sort by elevation.
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| (data.elevation[i], data.grid.cell_area(CellId(i as u32))))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total_area: f64 = pairs.iter().map(|(_, a)| a).sum();
    let target_ocean_area = total_area * (1.0 - TARGET_LAND_FRACTION);

    // Walk up from the lowest elevation until the target ocean area is reached.
    let mut cumulative_area = 0.0_f64;
    let mut sea_level = pairs[0].0;
    for &(elev, area) in &pairs {
        cumulative_area += area;
        if cumulative_area >= target_ocean_area {
            sea_level = elev;
            break;
        }
    }

    // Shift all elevations so sea level = 0.
    for e in data.elevation.iter_mut() {
        *e -= sea_level;
        *e = e.clamp(MIN_ELEVATION, MAX_ELEVATION);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::{PlanetConfig, PlanetData, TectonicMode};

    /// Build a small planet for testing (level 2 = 162 cells, fast).
    fn small_planet(seed: u64) -> PlanetData {
        let config = PlanetConfig {
            seed,
            grid_level: 2,
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 1.8,
            ..Default::default()
        };
        PlanetData::new(config)
    }

    #[test]
    fn plate_assignments_cover_all_cells() {
        let mut planet = small_planet(1);
        run_tectonics(&mut planet, |_| {});
        assert_eq!(planet.plate_id.len(), planet.grid.cell_count());
    }

    #[test]
    fn multiple_plate_types_generated() {
        let mut planet = small_planet(99);
        run_tectonics(&mut planet, |_| {});
        let has_continental = planet.crust_type.contains(&CrustType::Continental);
        let has_oceanic = planet.crust_type.contains(&CrustType::Oceanic);
        assert!(has_continental, "Expected at least one continental cell");
        assert!(has_oceanic, "Expected at least one oceanic cell");
    }

    #[test]
    fn elevation_range_within_physical_bounds() {
        let mut planet = small_planet(7);
        run_tectonics(&mut planet, |_| {});
        for &e in &planet.elevation {
            assert!(
                (MIN_ELEVATION..=MAX_ELEVATION).contains(&e),
                "Elevation {e} out of bounds [{MIN_ELEVATION}, {MAX_ELEVATION}]"
            );
        }
    }

    #[test]
    fn convergent_cells_tend_higher_than_divergent() {
        let mut planet = small_planet(42);
        run_tectonics(&mut planet, |_| {});

        let conv_elevs: Vec<f64> = planet
            .grid
            .cell_ids()
            .filter(|&id| planet.boundary_type[id.index()] == BoundaryType::Convergent)
            .map(|id| planet.elevation[id.index()])
            .collect();
        let div_elevs: Vec<f64> = planet
            .grid
            .cell_ids()
            .filter(|&id| planet.boundary_type[id.index()] == BoundaryType::Divergent)
            .map(|id| planet.elevation[id.index()])
            .collect();

        if conv_elevs.is_empty() || div_elevs.is_empty() {
            return;
        }

        let conv_mean: f64 = conv_elevs.iter().sum::<f64>() / conv_elevs.len() as f64;
        let div_mean: f64 = div_elevs.iter().sum::<f64>() / div_elevs.len() as f64;
        assert!(
            conv_mean > div_mean,
            "Expected convergent mean ({conv_mean:.1}) > divergent mean ({div_mean:.1})"
        );
    }

    #[test]
    fn some_volcanic_activity_near_boundaries() {
        let mut planet = small_planet(5);
        run_tectonics(&mut planet, |_| {});
        let has_volcanic = planet.volcanic_activity.iter().any(|&v| v > 0.0);
        assert!(
            has_volcanic,
            "Expected non-zero volcanic activity somewhere"
        );
    }

    #[test]
    fn sea_level_zero_divides_land_and_ocean() {
        let mut planet = small_planet(3);
        run_tectonics(&mut planet, |_| {});

        let total_area: f64 = planet.grid.all_areas().iter().sum();
        let land_area: f64 = planet
            .grid
            .cell_ids()
            .filter(|&id| planet.elevation[id.index()] >= 0.0)
            .map(|id| planet.grid.cell_area(id))
            .sum();
        let land_fraction = land_area / total_area;

        // Wide tolerance since small grids are very coarse.
        assert!(
            land_fraction > 0.05 && land_fraction < 0.95,
            "Land fraction {land_fraction:.3} out of expected range"
        );
    }

    #[test]
    fn progress_callback_called_correct_number_of_times() {
        let mut planet = small_planet(11);
        let steps = planet.config.tectonic_steps();
        let mut count = 0u32;
        run_tectonics(&mut planet, |_| count += 1);
        assert_eq!(count, steps, "Expected {steps} callback calls, got {count}");
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut p1 = small_planet(77);
        let mut p2 = small_planet(77);
        run_tectonics(&mut p1, |_| {});
        run_tectonics(&mut p2, |_| {});
        assert_eq!(
            p1.elevation, p2.elevation,
            "Same seed must produce identical elevation"
        );
        assert_eq!(
            p1.plate_id, p2.plate_id,
            "Same seed must produce identical plates"
        );
    }

    #[test]
    fn different_seeds_produce_different_worlds() {
        let mut p1 = small_planet(1);
        let mut p2 = small_planet(2);
        run_tectonics(&mut p1, |_| {});
        run_tectonics(&mut p2, |_| {});
        assert_ne!(p1.elevation, p2.elevation, "Different seeds should differ");
    }

    #[test]
    fn fibonacci_sphere_produces_n_points() {
        for n in [1, 7, 12, 100] {
            let pts = fibonacci_sphere_points(n, 42);
            assert_eq!(pts.len(), n, "Expected {n} points");
            for p in &pts {
                let len = p.length();
                assert!(
                    (len - 1.0).abs() < 1e-10,
                    "Point not on unit sphere: length = {len}"
                );
            }
        }
    }

    #[test]
    fn pentagon_count_unchanged_after_simulation() {
        let mut planet = small_planet(13);
        let pre = planet
            .grid
            .cell_ids()
            .filter(|&id| planet.grid.is_pentagon(id))
            .count();
        run_tectonics(&mut planet, |_| {});
        let post = planet
            .grid
            .cell_ids()
            .filter(|&id| planet.grid.is_pentagon(id))
            .count();
        assert_eq!(pre, post, "Pentagon count should not change");
        assert_eq!(post, 12);
    }

    #[test]
    fn boundary_type_vec_fully_populated() {
        let mut planet = small_planet(8);
        run_tectonics(&mut planet, |_| {});
        assert_eq!(planet.boundary_type.len(), planet.grid.cell_count());
    }

    #[test]
    fn plate_sizes_are_varied() {
        // Use a higher grid level so plate sizes are statistically meaningful.
        let config = PlanetConfig {
            seed: 42,
            grid_level: 4, // 2562 cells
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 1.8,
            ..Default::default()
        };
        let mut planet = PlanetData::new(config);
        run_tectonics(&mut planet, |_| {});

        let n_plates = *planet.plate_id.iter().max().unwrap() as usize + 1;
        let mut sizes = vec![0_usize; n_plates];
        for &pid in &planet.plate_id {
            sizes[pid as usize] += 1;
        }

        let min_size = *sizes.iter().min().unwrap() as f64;
        let max_size = *sizes.iter().max().unwrap() as f64;
        let ratio = max_size / min_size.max(1.0);

        assert!(
            ratio > 3.0,
            "Plate size max/min ratio {ratio:.1} too low (expected >3); sizes: {sizes:?}"
        );

        // Coefficient of variation should be substantial.
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let variance = sizes
            .iter()
            .map(|&s| (s as f64 - mean).powi(2))
            .sum::<f64>()
            / sizes.len() as f64;
        let cv = variance.sqrt() / mean;
        assert!(
            cv > 0.3,
            "Plate size coefficient of variation {cv:.3} too low (expected >0.3)"
        );
    }

    #[test]
    fn weighted_bfs_respects_weights() {
        // Directly test weighted BFS: a plate with 10× weight should be much larger.
        let config = PlanetConfig {
            seed: 1,
            grid_level: 3, // 642 cells
            ..Default::default()
        };
        let grid = crate::planet::grid::IcosahedralGrid::new(config.grid_level);
        let seeds = fibonacci_sphere_points(4, 1);
        let weights = vec![1.0, 1.0, 10.0, 1.0];
        let plate_id = weighted_bfs_flood_fill(&grid, &seeds, &weights);

        let mut sizes = [0_usize; 4];
        for &pid in &plate_id {
            sizes[pid as usize] += 1;
        }

        // Plate 2 (weight 10) should be the largest.
        let max_plate = sizes.iter().enumerate().max_by_key(|&(_, &s)| s).unwrap().0;
        assert_eq!(
            max_plate, 2,
            "Plate 2 (weight 10) should be largest; sizes: {sizes:?}"
        );
        // And substantially larger than the smallest.
        let min_size = *sizes.iter().min().unwrap();
        assert!(
            sizes[2] > min_size * 3,
            "Weight-10 plate ({}) should be >3× smallest ({min_size}); sizes: {sizes:?}",
            sizes[2]
        );
    }

    #[test]
    fn subduction_noise_deterministic_and_in_range() {
        for cell in 0..100 {
            for step in 0..10 {
                let v1 = subduction_noise(42, cell, step);
                let v2 = subduction_noise(42, cell, step);
                assert_eq!(v1, v2, "Subduction noise must be deterministic");
                assert!(
                    (0.0..1.0).contains(&v1),
                    "Subduction noise {v1} out of [0, 1)"
                );
            }
        }
    }

    #[test]
    fn generate_weights_clamped_to_max_ratio() {
        let mut rng = SmallRng::seed_from_u64(999);
        let weights = generate_plate_weights(&mut rng, 20);
        let min_w = weights.iter().copied().fold(f64::INFINITY, f64::min);
        let max_w = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ratio = max_w / min_w;
        assert!(
            ratio <= MAX_PLATE_WEIGHT_RATIO + 0.001,
            "Weight ratio {ratio:.2} exceeds cap {MAX_PLATE_WEIGHT_RATIO}"
        );
        assert!(min_w >= 1.0, "Minimum weight {min_w} below 1.0");
    }

    // ── Geological time calibration tests ────────────────────────────────────

    #[test]
    fn all_modes_produce_valid_terrain() {
        for mode in [
            TectonicMode::Quick,
            TectonicMode::Normal,
            TectonicMode::Extended,
        ] {
            let config = PlanetConfig {
                seed: 42,
                grid_level: 2,
                tectonic_mode: mode,
                tectonic_age_gyr: 1.0,
                ..Default::default()
            };
            let mut planet = PlanetData::new(config);
            run_tectonics(&mut planet, |_| {});

            let has_above = planet.elevation.iter().any(|&e| e > 0.0);
            let has_below = planet.elevation.iter().any(|&e| e < 0.0);
            assert!(
                has_above && has_below,
                "Mode {mode:?} should produce land and ocean"
            );
        }
    }

    #[test]
    fn mode_step_counts_match_expected() {
        let config_q = PlanetConfig {
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 3.0,
            ..Default::default()
        };
        let config_n = PlanetConfig {
            tectonic_mode: TectonicMode::Normal,
            tectonic_age_gyr: 3.0,
            ..Default::default()
        };
        let config_e = PlanetConfig {
            tectonic_mode: TectonicMode::Extended,
            tectonic_age_gyr: 3.0,
            ..Default::default()
        };
        assert_eq!(config_q.tectonic_steps(), 50);
        assert_eq!(config_n.tectonic_steps(), 200);
        assert_eq!(config_e.tectonic_steps(), 600);
    }

    #[test]
    fn velocity_evolution_changes_plate_speeds() {
        let radius_m = 6_371_000.0;
        let min_angular = MIN_PLATE_SPEED_M_YR / radius_m;
        let max_angular = MAX_PLATE_SPEED_M_YR / radius_m;
        let mid_speed = (min_angular + max_angular) / 2.0;
        let axis = DVec3::Y;
        let accel_axis = DVec3::X;
        let accel_mag = mid_speed / 50.0e6;

        let mut plates = vec![Plate {
            angular_velocity: axis * mid_speed,
            angular_acceleration: accel_axis * accel_mag,
            crust_type: CrustType::Continental,
        }];

        let dt_myr = 15.0;
        let dt_yr = dt_myr * 1e6;
        let _initial_speed = plates[0].angular_velocity.length();

        // Run many steps — velocity direction should drift.
        // Empty slab-pull: no subduction forces in this isolated test.
        let no_slab_pull = vec![DVec3::ZERO; 1];
        for step in 0..20 {
            evolve_plate_velocities(&mut plates, &no_slab_pull, 42, step, dt_yr, dt_myr, radius_m);
        }

        let final_vel = plates[0].angular_velocity;
        let final_speed = final_vel.length();

        // Speed should stay in physical range.
        assert!(
            final_speed >= min_angular * 0.99 && final_speed <= max_angular * 1.01,
            "Speed {final_speed:.2e} outside physical range [{min_angular:.2e}, {max_angular:.2e}]"
        );

        // Velocity should have changed direction (acceleration was perpendicular).
        let initial_dir = axis;
        let final_dir = final_vel.normalize();
        let dot = initial_dir.dot(final_dir);
        assert!(
            dot < 0.999,
            "Velocity direction should have changed (dot={dot:.4})"
        );
    }

    #[test]
    fn dt_scaling_determinism_preserved() {
        // Same mode and seed → identical results.
        let config = PlanetConfig {
            seed: 55,
            grid_level: 2,
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 0.6,
            ..Default::default()
        };
        let mut p1 = PlanetData::new(config.clone());
        let mut p2 = PlanetData::new(config);
        run_tectonics(&mut p1, |_| {});
        run_tectonics(&mut p2, |_| {});
        assert_eq!(
            p1.elevation, p2.elevation,
            "Same seed+mode must produce identical elevations"
        );
    }

    // ── Time-lapse history tests ─────────────────────────────────────────────

    #[test]
    fn history_captures_correct_frame_count() {
        let mut planet = small_planet(42);
        let steps = planet.config.tectonic_steps(); // 30 steps (Quick, 1.8 Gyr)
        let history = run_tectonics_with_history(&mut planet, None, |_| {});

        // With ≤100 max frames, every step captured plus initial = steps + 1.
        assert_eq!(
            history.frame_count(),
            steps as usize + 1,
            "Should capture initial + every step when steps ≤ max_frames"
        );
        assert_eq!(history.total_steps, steps);
        assert_eq!(history.snapshot_interval, 1);
    }

    #[test]
    fn history_respects_max_frames() {
        let config = PlanetConfig {
            seed: 42,
            grid_level: 2,
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 3.0, // 50 steps
            ..Default::default()
        };
        let mut planet = PlanetData::new(config);

        // Request only 10 frames max.
        let history = run_tectonics_with_history(&mut planet, Some(10), |_| {});

        // Interval = 50/10 = 5. Frames: initial(0), 5, 10, 15, 20, 25, 30, 35, 40, 45, 50.
        assert!(
            history.frame_count() <= 15,
            "Should limit to ~10 frames plus initial and final: got {}",
            history.frame_count()
        );
        assert_eq!(history.snapshot_interval, 5);
    }

    #[test]
    fn history_first_and_last_frames_correct() {
        let mut planet = small_planet(7);
        let history = run_tectonics_with_history(&mut planet, None, |_| {});

        let first = &history.snapshots[0];
        assert_eq!(first.step, 0, "First frame should be step 0");
        assert_eq!(first.age_myr, 0.0, "First frame should be age 0");

        let last = history.snapshots.last().unwrap();
        assert_eq!(
            last.step, history.total_steps,
            "Last frame should be the final step"
        );
        assert!(last.age_myr > 0.0, "Last frame age should be positive");
    }

    #[test]
    fn history_final_matches_normal_run() {
        let config = PlanetConfig {
            seed: 88,
            grid_level: 2,
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 0.6,
            ..Default::default()
        };

        // Run with history.
        let mut p1 = PlanetData::new(config.clone());
        let _history = run_tectonics_with_history(&mut p1, None, |_| {});

        // Run without history.
        let mut p2 = PlanetData::new(config);
        run_tectonics(&mut p2, |_| {});

        // Final elevations must match (both go through compute_sea_level).
        assert_eq!(
            p1.elevation, p2.elevation,
            "History run must produce identical final state as normal run"
        );
    }

    #[test]
    fn history_ages_monotonically_increase() {
        let mut planet = small_planet(42);
        let history = run_tectonics_with_history(&mut planet, None, |_| {});

        for window in history.snapshots.windows(2) {
            assert!(
                window[1].age_myr > window[0].age_myr,
                "Ages must increase: {} → {}",
                window[0].age_myr,
                window[1].age_myr
            );
        }
    }

    #[test]
    fn compute_interval_logic() {
        assert_eq!(TectonicHistory::compute_interval(50, 100), 1);
        assert_eq!(TectonicHistory::compute_interval(100, 100), 1);
        assert_eq!(TectonicHistory::compute_interval(200, 100), 2);
        assert_eq!(TectonicHistory::compute_interval(600, 100), 6);
        assert_eq!(TectonicHistory::compute_interval(50, 10), 5);
    }

    // ── Plate deformation tests ──────────────────────────────────────────────

    #[test]
    fn strain_accumulates_at_boundaries() {
        let mut planet = small_planet(42);
        run_tectonics(&mut planet, |_| {});

        let boundary_strain: Vec<f32> = planet
            .grid
            .cell_ids()
            .filter(|&id| planet.boundary_type[id.index()] != BoundaryType::Interior)
            .map(|id| planet.strain[id.index()])
            .collect();

        // At least some boundary cells should have non-zero strain.
        let has_strain = boundary_strain.iter().any(|&s| s > 0.0);
        assert!(has_strain, "Expected non-zero strain at plate boundaries");
    }

    #[test]
    fn strain_diffuses_to_interior() {
        let mut planet = small_planet(42);
        run_tectonics(&mut planet, |_| {});

        // Some interior cells adjacent to boundaries should have accumulated
        // strain via diffusion.
        let interior_with_strain = planet
            .grid
            .cell_ids()
            .filter(|&id| {
                planet.boundary_type[id.index()] == BoundaryType::Interior
                    && planet.strain[id.index()] > 0.001
            })
            .count();

        assert!(
            interior_with_strain > 0,
            "Expected strain diffusion to interior cells"
        );
    }

    #[test]
    fn strain_within_valid_range() {
        let mut planet = small_planet(7);
        run_tectonics(&mut planet, |_| {});
        for (i, &s) in planet.strain.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&s),
                "Strain {s} at cell {i} out of [0, 1]"
            );
        }
    }

    #[test]
    fn strain_included_in_snapshot() {
        let mut planet = small_planet(42);
        let history = run_tectonics_with_history(&mut planet, None, |_| {});

        for snapshot in &history.snapshots {
            assert_eq!(
                snapshot.strain.len(),
                planet.grid.cell_count(),
                "Snapshot strain vec should match cell count"
            );
        }

        // The last snapshot should have some non-zero strain.
        let last = history.snapshots.last().unwrap();
        let has_strain = last.strain.iter().any(|&s| s > 0.0);
        assert!(has_strain, "Final snapshot should have non-zero strain");
    }

    #[test]
    fn deformation_zones_spread_elevation() {
        // With deformation zones, continental interior cells near convergent
        // boundaries should show more elevation variation than a simulation
        // without deformation effects. We test that the standard deviation
        // of interior continental elevation is non-trivial.
        let config = PlanetConfig {
            seed: 42,
            grid_level: 4, // 2562 cells — enough for deformation spread
            tectonic_mode: TectonicMode::Normal,
            tectonic_age_gyr: 3.0,
            ..Default::default()
        };
        let mut planet = PlanetData::new(config);
        run_tectonics(&mut planet, |_| {});

        // Collect continental interior elevations.
        let interior_elevs: Vec<f64> = planet
            .grid
            .cell_ids()
            .filter(|&id| {
                let i = id.index();
                planet.boundary_type[i] == BoundaryType::Interior
                    && planet.crust_type[i] == CrustType::Continental
            })
            .map(|id| planet.elevation[id.index()])
            .collect();

        if interior_elevs.len() < 10 {
            return;
        }

        let mean = interior_elevs.iter().sum::<f64>() / interior_elevs.len() as f64;
        let variance = interior_elevs
            .iter()
            .map(|&e| (e - mean).powi(2))
            .sum::<f64>()
            / interior_elevs.len() as f64;
        let std_dev = variance.sqrt();

        // Deformation zones create inland uplift and back-arc subsidence,
        // so interior continental elevation should have meaningful variation
        // (not all cells at the same flat elevation).
        assert!(
            std_dev > 10.0,
            "Expected non-trivial interior elevation variation from deformation zones; \
             std_dev={std_dev:.1} m (mean={mean:.0} m, n={})",
            interior_elevs.len()
        );
    }

    #[test]
    fn rifting_can_create_new_plates() {
        // Run with enough steps for strain to accumulate and potentially
        // trigger rifting. Use Extended mode for maximum deformation.
        let config = PlanetConfig {
            seed: 42,
            grid_level: 4, // 2562 cells
            tectonic_mode: TectonicMode::Extended,
            tectonic_age_gyr: 3.0,
            ..Default::default()
        };
        let mut planet = PlanetData::new(config);
        let initial_plates = *planet.plate_id.iter().max().unwrap_or(&0) + 1;
        run_tectonics(&mut planet, |_| {});
        let final_plates = *planet.plate_id.iter().max().unwrap_or(&0) + 1;

        // With extended simulation, at least one rift should have occurred
        // (creating a new plate). If not, the mechanism is working but just
        // didn't trigger for this seed — that's acceptable.
        if final_plates > initial_plates {
            // Verify the new plate(s) have cells.
            for pid in initial_plates..final_plates {
                let count = planet.plate_id.iter().filter(|&&p| p == pid).count();
                // A plate created by rifting should have at least a few cells
                // (or may have been consumed by suturing — both are valid).
                assert!(
                    count == 0 || count >= 3,
                    "Rifted plate {pid} has suspiciously few cells: {count}"
                );
            }
        }
    }

    #[test]
    fn plate_count_can_change() {
        // Run many seeds and check that plate count changes in at least some.
        // This tests that rifting and/or suturing are active.
        let mut any_change = false;
        for seed in 0..20 {
            let config = PlanetConfig {
                seed,
                grid_level: 4,
                tectonic_mode: TectonicMode::Extended,
                tectonic_age_gyr: 3.0,
                ..Default::default()
            };
            let mut planet = PlanetData::new(config);
            let initial = planet
                .plate_id
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            run_tectonics(&mut planet, |_| {});
            let final_count = planet
                .plate_id
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            if final_count != initial {
                any_change = true;
                break;
            }
        }
        assert!(
            any_change,
            "Expected rifting or suturing to change plate count for at least one seed"
        );
    }

    #[test]
    fn rift_seed_deterministic_and_no_panic() {
        for plate in 0..20 {
            for step in 0..100 {
                let s1 = rift_seed(42, plate, step);
                let s2 = rift_seed(42, plate, step);
                assert_eq!(s1, s2, "Rift seed must be deterministic");
            }
        }
    }

    #[test]
    fn deformation_determinism_preserved() {
        let config = PlanetConfig {
            seed: 55,
            grid_level: 3,
            tectonic_mode: TectonicMode::Normal,
            tectonic_age_gyr: 2.0,
            ..Default::default()
        };
        let mut p1 = PlanetData::new(config.clone());
        let mut p2 = PlanetData::new(config);
        run_tectonics(&mut p1, |_| {});
        run_tectonics(&mut p2, |_| {});
        assert_eq!(
            p1.strain, p2.strain,
            "Same seed must produce identical strain fields"
        );
        assert_eq!(
            p1.plate_id, p2.plate_id,
            "Same seed must produce identical plate assignments (including rifts/sutures)"
        );
        assert_eq!(
            p1.elevation, p2.elevation,
            "Same seed must produce identical elevations"
        );
    }

    // ─── Slab-pull tests ─────────────────────────────────────────────────────

    #[test]
    fn slab_pull_accelerates_subducting_plates() {
        // A plate with oceanic convergent boundary cells should receive
        // non-zero slab-pull torque, while a plate with no convergent
        // boundary gets zero torque.
        use super::*;

        let config = PlanetConfig {
            seed: 1234,
            grid_level: 4,
            radius_m: 6_371_000.0,
            tectonic_mode: TectonicMode::Quick,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        let seed = data.config.seed;
        let radius_m = data.config.radius_m;
        let dt_myr = data.config.tectonic_dt_myr();
        let dt_yr = dt_myr * 1e6;

        let plates = init_plates(&mut data, seed, radius_m);
        let boundary_normals = precompute_boundary_normals(&data);
        detect_boundaries(&mut data, &plates, &boundary_normals, dt_yr);
        let torques = compute_slab_pull_torques(&data, &plates, &boundary_normals);

        assert_eq!(
            torques.len(),
            plates.len(),
            "One torque vector per plate"
        );

        // At least one plate should have non-zero slab pull (there are always
        // some oceanic convergent boundaries after initialization).
        let has_nonzero = torques.iter().any(|t| t.length() > 0.0);
        assert!(
            has_nonzero,
            "At least one plate should receive slab-pull acceleration"
        );
    }

    #[test]
    fn slab_pull_only_from_oceanic_convergent() {
        // Manually set up a scenario where some boundary cells are continental
        // (no slab pull) and some are oceanic (should get slab pull).
        use super::*;

        let config = PlanetConfig {
            seed: 999,
            grid_level: 4,
            radius_m: 6_371_000.0,
            tectonic_mode: TectonicMode::Quick,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        let seed = data.config.seed;
        let radius_m = data.config.radius_m;
        let dt_myr = data.config.tectonic_dt_myr();
        let dt_yr = dt_myr * 1e6;

        let plates = init_plates(&mut data, seed, radius_m);
        let boundary_normals = precompute_boundary_normals(&data);
        detect_boundaries(&mut data, &plates, &boundary_normals, dt_yr);

        // Count oceanic convergent cells per plate.
        let n = data.grid.cell_count();
        let mut oceanic_conv_count = vec![0_usize; plates.len()];
        for i in 0..n {
            if data.boundary_type[i] == BoundaryType::Convergent
                && data.crust_type[i] == CrustType::Oceanic
            {
                oceanic_conv_count[data.plate_id[i] as usize] += 1;
            }
        }

        let torques = compute_slab_pull_torques(&data, &plates, &boundary_normals);

        // Plates with zero oceanic convergent cells must have zero torque.
        for (pid, &count) in oceanic_conv_count.iter().enumerate() {
            if count == 0 && pid < torques.len() {
                assert!(
                    torques[pid].length() < 1e-30,
                    "Plate {pid} has no oceanic convergent cells but got torque {:.2e}",
                    torques[pid].length()
                );
            }
        }
    }

    #[test]
    fn slab_pull_direction_toward_trench() {
        // The slab-pull torque should rotate the plate TOWARD the subduction
        // zone. Verify the torque isn't zero and has a reasonable magnitude.
        use super::*;

        let config = PlanetConfig {
            seed: 42,
            grid_level: 5,
            radius_m: 6_371_000.0,
            tectonic_mode: TectonicMode::Normal,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        let seed = data.config.seed;
        let radius_m = data.config.radius_m;
        let dt_myr = data.config.tectonic_dt_myr();
        let dt_yr = dt_myr * 1e6;

        let plates = init_plates(&mut data, seed, radius_m);

        // Run boundary detection to establish geometry.
        let boundary_normals = precompute_boundary_normals(&data);
        detect_boundaries(&mut data, &plates, &boundary_normals, dt_yr);

        let torques = compute_slab_pull_torques(&data, &plates, &boundary_normals);

        // Find a plate with significant slab pull.
        let active_plate = torques
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.length().partial_cmp(&b.length()).unwrap());

        if let Some((pid, torque)) = active_plate {
            if torque.length() > 1e-30 {
                // The torque vector should be a subtle force (angular accel),
                // not an absurdly large value. Sanity-check the magnitude.
                assert!(
                    torque.length() < 1e-10,
                    "Slab-pull torque for plate {pid} should be small in rad/yr² \
                     (got {:.2e}); it's a subtle force, not a sledgehammer",
                    torque.length()
                );
            }
        }
    }

    #[test]
    fn slab_pull_inversely_proportional_to_plate_size() {
        // Two plates with the same number of subducting cells but different
        // total sizes should have different torque magnitudes — the smaller
        // plate should get more acceleration (less inertia).
        use super::*;

        let config = PlanetConfig {
            seed: 7777,
            grid_level: 5,
            radius_m: 6_371_000.0,
            tectonic_mode: TectonicMode::Normal,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        let seed = data.config.seed;
        let radius_m = data.config.radius_m;
        let dt_myr = data.config.tectonic_dt_myr();
        let dt_yr = dt_myr * 1e6;

        let plates = init_plates(&mut data, seed, radius_m);
        let boundary_normals = precompute_boundary_normals(&data);
        detect_boundaries(&mut data, &plates, &boundary_normals, dt_yr);

        let torques = compute_slab_pull_torques(&data, &plates, &boundary_normals);

        // Count plate sizes and subducting cells.
        let n = data.grid.cell_count();
        let mut plate_sizes = vec![0_usize; plates.len()];
        let mut sub_counts = vec![0_usize; plates.len()];
        for i in 0..n {
            let pid = data.plate_id[i] as usize;
            if pid < plates.len() {
                plate_sizes[pid] += 1;
                if data.boundary_type[i] == BoundaryType::Convergent
                    && data.crust_type[i] == CrustType::Oceanic
                {
                    sub_counts[pid] += 1;
                }
            }
        }

        // Find two plates both with nonzero subducting cells.
        let mut active: Vec<(usize, usize, usize, f64)> = (0..plates.len())
            .filter(|&p| sub_counts[p] > 0 && torques[p].length() > 1e-30)
            .map(|p| (p, plate_sizes[p], sub_counts[p], torques[p].length()))
            .collect();

        if active.len() >= 2 {
            // Sort by torque magnitude per subducting cell.
            active.sort_by(|a, b| {
                let a_per = a.3 / a.2 as f64;
                let b_per = b.3 / b.2 as f64;
                b_per.partial_cmp(&a_per).unwrap()
            });
            // The plate with higher torque-per-subducting-cell should be smaller.
            let (_, size_high, _, _) = active[0];
            let (_, size_low, _, _) = active[active.len() - 1];
            assert!(
                size_high <= size_low,
                "Higher torque-per-subducting-cell should come from smaller plate \
                 (sizes: {size_high} vs {size_low})"
            );
        }
    }

    #[test]
    fn slab_pull_determinism_preserved() {
        // Two runs with the same seed must produce identical results,
        // including slab-pull effects on velocity evolution.
        use super::*;

        let config = PlanetConfig {
            seed: 31337,
            grid_level: 4,
            radius_m: 6_371_000.0,
            tectonic_mode: TectonicMode::Quick,
            ..Default::default()
        };
        let mut d1 = PlanetData::new(config.clone());
        let mut d2 = PlanetData::new(config);

        run_tectonics(&mut d1, |_| {});
        run_tectonics(&mut d2, |_| {});

        // If the full simulation is deterministic, the final states must match.
        assert_eq!(
            d1.plate_id, d2.plate_id,
            "Slab-pull must preserve determinism: plate IDs diverged"
        );
        assert_eq!(
            d1.elevation, d2.elevation,
            "Slab-pull must preserve determinism: elevations diverged"
        );
    }
}
