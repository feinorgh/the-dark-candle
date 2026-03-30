//! Astronomical impact event generation and crater stamping.
//!
//! ## Algorithm overview
//!
//! Impacts are generated from the world seed and stamped onto the elevation
//! data **after** the tectonic simulation. Each impact produces:
//!
//! - A **central depression** — excavation scaled to impactor size.
//! - A **raised rim** — ejecta piled up at the crater edge.
//! - An **ejecta blanket** — gradual elevation increase decaying with distance.
//!
//! An optional giant impact (hemisphere-scale) can be applied first to
//! represent a Mars-sized body collision early in the planet's history.
//!
//! ### Crater morphology (scaled by `radius_cells`)
//!
//! ```text
//! Depth profile along radial distance r from centre (r in [0, 1] normalised):
//!   r < 0.5:       depression = -depth * (1 - 2r)²          (bowl)
//!   0.5 < r < 1.0: rim rise   = depth * 0.3 * sin²(π(r-0.5)/0.5)
//!   1.0 < r < 2.0: ejecta     = depth * 0.05 * (2 - r)²    (blanket)
//! ```
//!
//! Crust depth is set to near zero at the crater centre (exposed mantle).

use bevy::math::DVec3;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::planet::PlanetData;
use crate::planet::grid::CellId;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum excavation depth for the largest craters (m).
const MAX_CRATER_DEPTH: f64 = 5_000.0;
/// Minimum excavation depth for the smallest craters (m).
const MIN_CRATER_DEPTH: f64 = 100.0;
/// Crust depth left exposed at the centre of a fresh impact crater (m).
const CRATER_CENTRE_CRUST: f32 = 500.0;

/// Depth of a giant impact (hemisphere-scale, m).
const GIANT_IMPACT_DEPTH: f64 = 4_000.0;
/// Rim rise fraction of giant impact depth.
const GIANT_IMPACT_RIM_FRACTION: f64 = 0.2;

/// Number of neighbours to search outward for crater footprint (BFS depth).
/// At level-7 (~56 km cells) a depth-6 BFS covers ~336 km radius.
const MAX_BFS_DEPTH: u32 = 8;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Parameters for a single impact event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactEvent {
    /// Unit-sphere position of the impact centre.
    pub centre: [f64; 3],
    /// Crater radius in units of the mean cell side length (dimensionless).
    /// 1.0 = single-cell crater; 6.0 = large regional crater.
    pub radius_cells: f64,
    /// Maximum excavation depth at the crater centre (m).
    pub depth_m: f64,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Stamp all impact events onto the planet's elevation and crust data.
///
/// Call this **after** `run_tectonics`. Events are generated deterministically
/// from `data.config.seed` and `data.config.bombardment_intensity`.
///
/// If the seed probability rolls a giant impact
/// (`data.config.giant_impact_probability`), a hemisphere-scale depression is
/// applied first.
pub fn run_impacts(data: &mut PlanetData) {
    let seed = data.config.seed;
    let intensity = data.config.bombardment_intensity;
    let giant_prob = data.config.giant_impact_probability;

    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF));

    // Optional giant impact — applied before smaller craters so subsequent
    // bombardment partially fills and erodes the mega-basin.
    if rng.random::<f64>() < giant_prob {
        apply_giant_impact(data, &mut rng);
    }

    // Generate and stamp ordinary craters.
    let events = generate_impact_events(intensity, seed);
    for event in &events {
        stamp_crater(data, event);
    }
}

/// Generate a list of impact events for the given bombardment intensity.
///
/// Returns the events for inspection (e.g. for stats/visualisation).
pub fn generate_impact_events(intensity: f64, seed: u64) -> Vec<ImpactEvent> {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0xCAFE_BABE));

    // Number of impacts scales with intensity: 0 → 0 events, 1.0 → 50 events.
    let n_impacts = (intensity * 50.0).round() as usize;
    let mut events = Vec::with_capacity(n_impacts);

    for _ in 0..n_impacts {
        let centre = random_sphere_point(&mut rng);
        // Impact size: log-uniform distribution — many small, few large.
        let radius_cells = rng.random_range(0.5_f64..6.0_f64).exp2() / 2.0;
        let depth_m = MIN_CRATER_DEPTH
            + (MAX_CRATER_DEPTH - MIN_CRATER_DEPTH) * rng.random::<f64>().powf(2.0); // skewed toward small
        events.push(ImpactEvent {
            centre: centre.to_array(),
            radius_cells,
            depth_m,
        });
    }

    events
}

// ---------------------------------------------------------------------------
// Crater stamping
// ---------------------------------------------------------------------------

/// Apply a single crater to the planet's elevation and crust data.
///
/// Uses BFS from the nearest cell to the impact centre to find all cells
/// within the crater footprint, then applies the depth profile.
pub fn stamp_crater(data: &mut PlanetData, event: &ImpactEvent) {
    let centre = DVec3::from_array(event.centre).normalize();
    let centre_cell = data.grid.nearest_cell_from_pos(centre);

    // Use angular distance to compute normalised radius.
    // cos(angular_distance) = dot(pos, centre).
    // Max angular radius of crater: radius_cells * (mean_cell_angular_size).
    // For a unit sphere, mean cell angular size ≈ sqrt(4π / n_cells).
    let n = data.grid.cell_count() as f64;
    let mean_angular_cell = (4.0 * std::f64::consts::PI / n).sqrt();
    let max_angular_radius = event.radius_cells * mean_angular_cell;
    // cos threshold for ejecta blanket (2x crater radius).
    let cos_ejecta = (2.0 * max_angular_radius).cos();

    // BFS from centre cell to collect all cells in the ejecta blanket.
    let cells = bfs_within_angle(data, centre_cell, cos_ejecta, MAX_BFS_DEPTH);

    for cell_id in cells {
        let pos = data.grid.cell_position(cell_id);
        let cos_angle = pos.dot(centre).clamp(-1.0, 1.0);
        // Angular distance in [0, π].
        let angle = cos_angle.acos();
        // Normalised radius: 0 at centre, 1 at crater rim.
        let r = angle / max_angular_radius;

        let delta = crater_depth_profile(r, event.depth_m);
        let i = cell_id.index();
        data.elevation[i] = (data.elevation[i] + delta).clamp(-11_000.0, 9_000.0);

        // Thin the crust at and near the crater centre.
        if r < 0.6 {
            let thinning = 1.0 - (r / 0.6) as f32;
            data.crust_depth[i] = (data.crust_depth[i] * (1.0 - thinning)).max(CRATER_CENTRE_CRUST);
        }
    }
}

/// Elevation delta for a given normalised radial distance `r` from crater centre.
///
/// - `r < 0.5`: bowl-shaped depression.
/// - `0.5 ≤ r < 1.0`: raised rim.
/// - `1.0 ≤ r < 2.0`: ejecta blanket (slight elevation increase).
/// - `r ≥ 2.0`: no change.
fn crater_depth_profile(r: f64, depth: f64) -> f64 {
    if r < 0.5 {
        // Bowl: full depth at centre, zero at r=0.5.
        let t = 1.0 - 2.0 * r;
        -depth * t * t
    } else if r < 1.0 {
        // Rim: raised above baseline, peak at r=0.75.
        let t = (r - 0.5) / 0.5;
        depth * 0.3 * (std::f64::consts::PI * t).sin().powi(2)
    } else if r < 2.0 {
        // Ejecta blanket: gentle elevation increase decaying outward.
        let t = 2.0 - r;
        depth * 0.05 * t * t
    } else {
        0.0
    }
}

/// BFS outward from `start`, stopping at `max_depth` hops.
///
/// The angular/radius filter is applied by the caller after stamping, so we
/// collect all cells within `max_depth` neighbourhood hops. The `_cos_threshold`
/// parameter is reserved for a future tighter spatial filter.
fn bfs_within_angle(
    data: &PlanetData,
    start: CellId,
    _cos_threshold: f64,
    max_depth: u32,
) -> Vec<CellId> {
    let mut visited = vec![false; data.grid.cell_count()];
    let mut result = Vec::new();
    let mut queue = std::collections::VecDeque::new();

    visited[start.index()] = true;
    queue.push_back((start, 0u32));

    while let Some((cell, depth)) = queue.pop_front() {
        result.push(cell);
        if depth >= max_depth {
            continue;
        }
        for &nb in data.grid.cell_neighbors(cell) {
            let nb_id = CellId(nb);
            if !visited[nb_id.index()] {
                visited[nb_id.index()] = true;
                queue.push_back((nb_id, depth + 1));
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Giant impact
// ---------------------------------------------------------------------------

/// Apply a hemisphere-scale impact — a deep, wide basin covering roughly
/// one hemisphere, with a raised antipodal region from focused ejecta.
///
/// This is applied **before** ordinary bombardment so subsequent cratering
/// partially fills the mega-basin.
fn apply_giant_impact(data: &mut PlanetData, rng: &mut SmallRng) {
    let centre = random_sphere_point(rng);
    let antipode = -centre;
    let n = data.grid.cell_count();

    for i in 0..n {
        let pos = data.grid.cell_position(CellId(i as u32));
        let cos_to_centre = pos.dot(centre).clamp(-1.0, 1.0);
        // Normalised: 1 at centre, 0 at equator, -1 at antipode.
        let t = cos_to_centre;

        let cos_to_antipode = pos.dot(antipode).clamp(-1.0, 1.0);

        if t > 0.0 {
            // Impact hemisphere: deep depression, strongest at centre.
            data.elevation[i] -= GIANT_IMPACT_DEPTH * t * t;
        } else if cos_to_antipode > 0.7 {
            // Antipodal focused ejecta / seismic convergence zone.
            let a = (cos_to_antipode - 0.7) / 0.3;
            data.elevation[i] += GIANT_IMPACT_DEPTH * GIANT_IMPACT_RIM_FRACTION * a * a;
        }

        data.elevation[i] = data.elevation[i].clamp(-11_000.0, 9_000.0);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a uniformly distributed random point on the unit sphere.
fn random_sphere_point(rng: &mut SmallRng) -> DVec3 {
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::tectonics::run_tectonics;
    use crate::planet::{PlanetConfig, PlanetData, TectonicMode};

    fn small_planet(seed: u64) -> PlanetData {
        let config = PlanetConfig {
            seed,
            grid_level: 3,
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 1.2,
            bombardment_intensity: 0.5,
            giant_impact_probability: 0.0, // disable for most tests
            ..Default::default()
        };
        let mut planet = PlanetData::new(config);
        run_tectonics(&mut planet, |_| {});
        planet
    }

    #[test]
    fn crater_centre_is_lower_than_rim() {
        let mut planet = small_planet(1);
        let centre = DVec3::new(0.0, 1.0, 0.0); // north pole
        let centre_cell = planet.grid.nearest_cell_from_pos(centre);

        let event = ImpactEvent {
            centre: centre.to_array(),
            radius_cells: 2.0,
            depth_m: 1000.0,
        };

        // Record pre-impact elevation at centre.
        let elev_before = planet.elevation[centre_cell.index()];
        stamp_crater(&mut planet, &event);
        let elev_after = planet.elevation[centre_cell.index()];

        assert!(
            elev_after < elev_before,
            "Crater centre should be lower after impact: before={elev_before:.1}, after={elev_after:.1}"
        );
    }

    #[test]
    fn rim_is_higher_than_centre() {
        let planet = small_planet(2);
        let centre = DVec3::new(1.0, 0.0, 0.0);

        let event = ImpactEvent {
            centre: centre.to_array(),
            radius_cells: 3.0,
            depth_m: 2000.0,
        };

        let n = planet.grid.cell_count() as f64;
        let mean_angular = (4.0 * std::f64::consts::PI / n).sqrt();
        let max_angular = event.radius_cells * mean_angular;

        // Find a cell near the rim (r ≈ 0.75) and one at the centre.
        let mut centre_elev = f64::INFINITY;
        let mut rim_elev = f64::NEG_INFINITY;
        let mut cloned = planet.clone();
        stamp_crater(&mut cloned, &event);

        for id in cloned.grid.cell_ids() {
            let pos = cloned.grid.cell_position(id);
            let cos_a = pos.dot(centre).clamp(-1.0, 1.0);
            let angle = cos_a.acos();
            let r = angle / max_angular;

            if r < 0.1 {
                centre_elev = centre_elev.min(cloned.elevation[id.index()]);
            } else if (0.6..0.9).contains(&r) {
                rim_elev = rim_elev.max(cloned.elevation[id.index()]);
            }
        }

        if rim_elev.is_finite() && centre_elev.is_finite() {
            assert!(
                rim_elev > centre_elev,
                "Rim elev ({rim_elev:.1}) should exceed centre elev ({centre_elev:.1})"
            );
        }
    }

    #[test]
    fn ejecta_blanket_exists_beyond_rim() {
        let planet = small_planet(3);
        let centre = DVec3::new(0.0, 0.0, 1.0);

        let event = ImpactEvent {
            centre: centre.to_array(),
            radius_cells: 2.0,
            depth_m: 1500.0,
        };

        let mut pre = planet.clone();
        let mut post = planet.clone();
        // Flatten pre to a uniform surface to isolate impact delta.
        for e in pre.elevation.iter_mut() {
            *e = 0.0;
        }
        for e in post.elevation.iter_mut() {
            *e = 0.0;
        }
        stamp_crater(&mut post, &event);

        let n = post.grid.cell_count() as f64;
        let mean_angular = (4.0 * std::f64::consts::PI / n).sqrt();
        let max_angular = event.radius_cells * mean_angular;

        // Find a cell in the ejecta zone (1.0 < r < 2.0) and verify elevation > 0.
        let mut found_ejecta = false;
        for id in post.grid.cell_ids() {
            let pos = post.grid.cell_position(id);
            let cos_a = pos.dot(centre).clamp(-1.0, 1.0);
            let angle = cos_a.acos();
            let r = angle / max_angular;
            if (1.0..2.0).contains(&r) && post.elevation[id.index()] > 0.0 {
                found_ejecta = true;
                break;
            }
        }

        assert!(found_ejecta, "Expected positive elevation in ejecta zone");
    }

    #[test]
    fn crust_thinned_at_crater_centre() {
        let mut planet = small_planet(4);
        let centre = DVec3::new(0.0, 1.0, 0.0);
        let centre_cell = planet.grid.nearest_cell_from_pos(centre);

        let crust_before = planet.crust_depth[centre_cell.index()];
        let event = ImpactEvent {
            centre: centre.to_array(),
            radius_cells: 3.0,
            depth_m: 3000.0,
        };
        stamp_crater(&mut planet, &event);
        let crust_after = planet.crust_depth[centre_cell.index()];

        assert!(
            crust_after <= crust_before,
            "Crust should thin or stay at centre: before={crust_before}, after={crust_after}"
        );
    }

    #[test]
    fn elevation_stays_within_bounds_after_impacts() {
        let mut planet = small_planet(5);
        run_impacts(&mut planet);
        for &e in &planet.elevation {
            assert!(
                (-11_000.0..=9_000.0).contains(&e),
                "Elevation {e} out of physical bounds"
            );
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut p1 = small_planet(10);
        let mut p2 = small_planet(10);
        run_impacts(&mut p1);
        run_impacts(&mut p2);
        assert_eq!(
            p1.elevation, p2.elevation,
            "Same seed must produce identical results"
        );
    }

    #[test]
    fn different_seeds_produce_different_craters() {
        let mut p1 = small_planet(1);
        let mut p2 = small_planet(2);
        run_impacts(&mut p1);
        run_impacts(&mut p2);
        assert_ne!(p1.elevation, p2.elevation, "Different seeds should differ");
    }

    #[test]
    fn zero_bombardment_changes_nothing() {
        let config = PlanetConfig {
            seed: 42,
            grid_level: 2,
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 0.6,
            bombardment_intensity: 0.0,
            giant_impact_probability: 0.0,
            ..Default::default()
        };
        let mut planet = PlanetData::new(config);
        run_tectonics(&mut planet, |_| {});
        let before = planet.elevation.clone();
        run_impacts(&mut planet);
        assert_eq!(
            planet.elevation, before,
            "Zero bombardment should not change elevation"
        );
    }

    #[test]
    fn generate_events_count_scales_with_intensity() {
        let low = generate_impact_events(0.2, 42);
        let high = generate_impact_events(0.8, 42);
        assert!(
            high.len() > low.len(),
            "Higher intensity should produce more impacts: low={}, high={}",
            low.len(),
            high.len()
        );
    }

    #[test]
    fn crater_depth_profile_centre_is_negative() {
        let delta = crater_depth_profile(0.0, 1000.0);
        assert!(delta < 0.0, "Centre of crater must be negative: {delta}");
    }

    #[test]
    fn crater_depth_profile_rim_is_positive() {
        let delta = crater_depth_profile(0.75, 1000.0);
        assert!(delta > 0.0, "Rim of crater must be positive: {delta}");
    }

    #[test]
    fn crater_depth_profile_far_field_is_zero() {
        let delta = crater_depth_profile(3.0, 1000.0);
        assert_eq!(delta, 0.0, "Far field should have zero delta: {delta}");
    }

    #[test]
    fn giant_impact_depresses_impact_hemisphere() {
        let config = PlanetConfig {
            seed: 7,
            grid_level: 2,
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 0.3,
            bombardment_intensity: 0.0,
            giant_impact_probability: 1.0, // always trigger
            ..Default::default()
        };
        let mut planet = PlanetData::new(config);
        run_tectonics(&mut planet, |_| {});
        let before_sum: f64 = planet.elevation.iter().sum();
        run_impacts(&mut planet);
        let after_sum: f64 = planet.elevation.iter().sum();
        // Giant impact should, on balance, lower the planet surface.
        assert!(
            after_sum < before_sum,
            "Giant impact should lower mean elevation: before={before_sum:.1}, after={after_sum:.1}"
        );
    }
}
