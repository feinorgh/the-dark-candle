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
// Constants
// ---------------------------------------------------------------------------

/// Elevation for newly initialised continental cells (m).
const INIT_CONTINENTAL_ELEV: f64 = 200.0;
/// Elevation for newly initialised oceanic cells (m).
const INIT_OCEANIC_ELEV: f64 = -3_500.0;

/// Default continental crust thickness (m).
const CONTINENTAL_CRUST_DEPTH: f32 = 35_000.0;
/// Default oceanic crust thickness (m).
const OCEANIC_CRUST_DEPTH: f32 = 7_000.0;

/// Height gained per step at a continent-continent convergent boundary (m).
const OROGENY_RISE: f64 = 80.0;
/// Height gained on the continental side of ocean-continent convergence (m).
const ARC_RISE: f64 = 60.0;
/// Height lost on the oceanic side of ocean-continent convergence (m/step).
const TRENCH_DROP: f64 = 100.0;
/// Height gained at an island arc (ocean-ocean convergent, building side, m).
const ISLAND_ARC_RISE: f64 = 20.0;
/// Height lost at an ocean-ocean convergent boundary (m).
const OCEAN_CONVERGENT_DROP: f64 = 80.0;
/// Height lost per step at a divergent boundary (m).
const RIFT_DROP: f64 = 50.0;

/// How much volcanic activity is added per step at active boundaries.
const VOLCANIC_GAIN: f32 = 0.05;
/// Decay factor applied to volcanic activity each step.
const VOLCANIC_DECAY: f32 = 0.95;
/// Fault stress accumulated per step at transform boundaries.
const FAULT_STRESS_GAIN: f32 = 0.01;

/// Fraction of the elevation difference diffused toward neighbour average.
const EROSION_RATE: f64 = 0.02;

/// Hard cap on mountain height (m).
const MAX_ELEVATION: f64 = 9_000.0;
/// Hard cap on ocean depth (m, negative).
const MIN_ELEVATION: f64 = -11_000.0;

/// Fraction of total surface area that should be land (used for sea-level).
const TARGET_LAND_FRACTION: f64 = 0.30;

/// Velocity threshold (rad/step) below which boundaries are classified as
/// transform rather than convergent/divergent.
const VELOCITY_THRESHOLD: f64 = 1e-4;

/// Pareto shape parameter for plate growth weight distribution.
/// Lower α → more extreme size variation. α ≈ 1.0–1.5 is Earth-like.
const PLATE_SIZE_PARETO_ALPHA: f64 = 1.3;

/// Maximum ratio between the largest and smallest plate growth weight.
/// Caps the Pareto tail to prevent one plate dominating the entire surface.
const MAX_PLATE_WEIGHT_RATIO: f64 = 15.0;

/// Probability per step that an oceanic cell at a convergent boundary is
/// consumed by the overriding continental plate (subduction advance rate).
const SUBDUCTION_RATE: f64 = 0.03;

/// Minimum plate size as a fraction of total cells. Plates below this
/// threshold are protected from further subduction consumption.
const MIN_PLATE_FRACTION: f64 = 0.01;

// ---------------------------------------------------------------------------
// Plate data
// ---------------------------------------------------------------------------

/// Internal state for a single tectonic plate during simulation.
#[derive(Debug, Clone)]
struct Plate {
    /// Angular velocity vector (axis * angular_speed, radians/step).
    angular_velocity: DVec3,
    /// Whether this plate is continental or oceanic.
    crust_type: CrustType,
}

impl Plate {
    /// Surface velocity of this plate at a given unit-sphere position.
    ///
    /// `v = omega x r` (cross product of angular velocity and position).
    fn velocity_at(&self, pos: DVec3) -> DVec3 {
        self.angular_velocity.cross(pos)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the full tectonic simulation and write results into `data`.
///
/// Initialises plate assignments, then runs `data.config.tectonic_steps`
/// iterations. An optional `progress` callback is called after each step
/// with the step index (0-based).
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
    let steps = data.config.tectonic_steps;

    let plates = init_plates(data, seed);

    for step in 0..steps {
        // Recompute each step: plate assignments shift due to subduction.
        let boundary_normals = precompute_boundary_normals(data);
        detect_boundaries(data, &plates, &boundary_normals);
        apply_boundary_forces(data);
        subduction_reassign(data, step);
        update_volcanic_activity(data);
        erode(data);
        progress(step);
    }

    compute_sea_level(data);
}

// ---------------------------------------------------------------------------
// Plate initialisation
// ---------------------------------------------------------------------------

/// Seed plate centres, assign cells, and initialise elevation and crust data.
fn init_plates(data: &mut PlanetData, seed: u64) -> Vec<Plate> {
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
            // Random angular velocity: axis is a random unit vector, speed is
            // small (plates move a fraction of a radian per simulation step).
            let axis = random_unit_vec(&mut rng);
            let speed = rng.random_range(1e-4_f64..5e-3_f64);
            Plate {
                angular_velocity: axis * speed,
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
fn detect_boundaries(data: &mut PlanetData, plates: &[Plate], boundary_normals: &[DVec3]) {
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

        let v_a = plate_a.velocity_at(pos);

        // Average relative velocity projected onto the outward boundary normal.
        let mut rel_proj = 0.0_f64;
        let mut count = 0;
        for &nb in data.grid.cell_neighbors(id) {
            let nb_pid = data.plate_id[nb as usize];
            if nb_pid != data.plate_id[i] {
                let plate_b = &plates[nb_pid as usize];
                let pos_nb = data.grid.cell_position(CellId(nb));
                let v_b = plate_b.velocity_at(pos_nb);
                rel_proj += (v_a - v_b).dot(normal);
                count += 1;
            }
        }

        if count > 0 {
            rel_proj /= count as f64;
        }

        data.boundary_type[i] = if rel_proj > VELOCITY_THRESHOLD {
            BoundaryType::Convergent
        } else if rel_proj < -VELOCITY_THRESHOLD {
            BoundaryType::Divergent
        } else {
            BoundaryType::Transform
        };
    }
}

/// Apply height changes at boundary cells based on boundary type and the
/// crust types of interacting plates.
#[allow(clippy::needless_range_loop)] // multiple parallel arrays indexed by i
fn apply_boundary_forces(data: &mut PlanetData) {
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
                    (CrustType::Continental, CrustType::Continental) => OROGENY_RISE,
                    // Ocean-Continent: arc volcano on continental, trench on oceanic.
                    (CrustType::Continental, CrustType::Oceanic) => ARC_RISE,
                    (CrustType::Oceanic, CrustType::Continental) => -TRENCH_DROP,
                    // Ocean-Ocean: one side builds island arc, other subducts.
                    (CrustType::Oceanic, CrustType::Oceanic) => {
                        // Alternate which side rises based on cell parity.
                        if i % 2 == 0 {
                            ISLAND_ARC_RISE
                        } else {
                            -OCEAN_CONVERGENT_DROP
                        }
                    }
                };
            }
            BoundaryType::Divergent => {
                deltas[i] -= RIFT_DROP;
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
fn update_volcanic_activity(data: &mut PlanetData) {
    let n = data.grid.cell_count();
    for i in 0..n {
        let is_active = (matches!(
            data.boundary_type[i],
            BoundaryType::Convergent | BoundaryType::Divergent
        ) && matches!(data.crust_type[i], CrustType::Oceanic))
            || matches!(data.boundary_type[i], BoundaryType::Divergent);

        if is_active {
            data.volcanic_activity[i] = (data.volcanic_activity[i] + VOLCANIC_GAIN).min(1.0);
        } else {
            data.volcanic_activity[i] *= VOLCANIC_DECAY;
        }

        if matches!(data.boundary_type[i], BoundaryType::Transform) {
            data.fault_stress[i] = (data.fault_stress[i] + FAULT_STRESS_GAIN).min(1.0);
        }
    }
}

/// One diffusion pass: each cell moves `EROSION_RATE * (mean_neighbour - self)`
/// toward its neighbour average.
fn erode(data: &mut PlanetData) {
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
        let delta = EROSION_RATE * (mean - old_elev[i]);
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
/// per step (`SUBDUCTION_RATE`) of reassignment. Plates below
/// `MIN_PLATE_FRACTION` of total cells are protected from further consumption.
fn subduction_reassign(data: &mut PlanetData, step: u32) {
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
            if roll < SUBDUCTION_RATE {
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
    use crate::planet::{PlanetConfig, PlanetData};

    /// Build a small planet for testing (level 2 = 162 cells, fast).
    fn small_planet(seed: u64) -> PlanetData {
        let config = PlanetConfig {
            seed,
            grid_level: 2,
            tectonic_steps: 30,
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
        let steps = planet.config.tectonic_steps;
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
            tectonic_steps: 30,
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
}
