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
use std::collections::VecDeque;

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

    for step in 0..steps {
        evolve_plate_velocities(&mut plates, seed, step, dt_yr, dt_myr, radius_m);

        // Recompute each step: plate assignments shift due to subduction.
        let boundary_normals = precompute_boundary_normals(data);
        detect_boundaries(data, &plates, &boundary_normals, dt_yr);
        apply_boundary_forces(data, dt_myr);
        subduction_reassign(data, step, dt_scale);
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

    // Capture the initial state (step 0, before any simulation).
    snapshots.push(capture_snapshot(data, 0, 0.0));

    for step in 0..steps {
        evolve_plate_velocities(&mut plates, seed, step, dt_yr, dt_myr, radius_m);

        let boundary_normals = precompute_boundary_normals(data);
        detect_boundaries(data, &plates, &boundary_normals, dt_yr);
        apply_boundary_forces(data, dt_myr);
        subduction_reassign(data, step, dt_scale);
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
    seed: u64,
    step: u32,
    dt_yr: f64,
    dt_myr: f64,
    radius_m: f64,
) {
    let min_angular = MIN_PLATE_SPEED_M_YR / radius_m;
    let max_angular = MAX_PLATE_SPEED_M_YR / radius_m;

    for (p, plate) in plates.iter_mut().enumerate() {
        // Apply acceleration: ω += α × dt (years).
        plate.angular_velocity += plate.angular_acceleration * dt_yr;

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
    seed.wrapping_add(plate as u64 * 0x1234_5678_9abc_def0)
        .wrapping_mul(0x6c62_272e_07bb_0142)
        .wrapping_add(step as u64)
        ^ 0xfeed_face_cafe_babe
}

// ---------------------------------------------------------------------------
// Boundary normal precomputation
// ---------------------------------------------------------------------------

/// For each cell, compute the average outward boundary normal toward all
/// cross-plate neighbours. Returns zero for interior cells.
///
/// Precomputed once since grid and plate assignments are fixed throughout.
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
        for step in 0..20 {
            evolve_plate_velocities(&mut plates, 42, step, dt_yr, dt_myr, radius_m);
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
}
