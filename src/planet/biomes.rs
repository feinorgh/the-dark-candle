//! Climate estimation and biome classification for planetary surfaces.
//!
//! A static single-column energy-balance climate model:
//! per-cell mean annual insolation (from stellar luminosity, orbital distance,
//! axial tilt, and latitude) → local energy balance with surface albedo →
//! iterative ice-albedo feedback → meridional heat transport (Laplacian
//! smoothing) → elevation lapse rate → mean annual temperature; then
//! ocean-proximity BFS + Clausius-Clapeyron humidity scaling → mean annual
//! precipitation; simplified Whittaker diagram → biome type.
//!
//! No full atmospheric dynamics are simulated.  The output is a plausible
//! static climate map sufficient for biome and resource distribution.

use super::grid::CellId;
use super::{BiomeType, PlanetData};
use std::collections::VecDeque;

// ─── Climate constants ────────────────────────────────────────────────────────

/// Stefan-Boltzmann constant (W/(m²·K⁴)).
const SIGMA: f64 = 5.670_374_419e-8;
/// Environmental lapse rate (K/m): temperature decreases with altitude.
const LAPSE_RATE: f64 = 6.5e-3;
/// Greenhouse warming above blackbody equilibrium (K).  Earth ≈ 33 K.
const GREENHOUSE_K: f64 = 33.0;
/// Albedo of ice/snow-covered surfaces (fresh snow 0.8, old sea ice 0.5).
const ICE_ALBEDO: f64 = 0.62;
/// Albedo of open ocean.
const OCEAN_ALBEDO: f64 = 0.06;
/// Mean albedo of non-ice land.
const LAND_ALBEDO: f64 = 0.25;
/// Temperature below which the surface is considered frozen and ice albedo
/// applies (K).  Slightly below 273 K to account for seasonal averaging.
const ICE_THRESHOLD_K: f64 = 263.0;
/// Number of iterations to converge the ice-albedo feedback loop.
const ICE_ALBEDO_ITERATIONS: usize = 5;
/// Number of equally-spaced orbital positions sampled to approximate the
/// annual mean insolation integral.
const ORBIT_SAMPLES: usize = 24;
/// Number of Laplacian smoothing passes on the temperature field to
/// approximate meridional heat transport by atmosphere and ocean currents.
const HEAT_TRANSPORT_PASSES: usize = 4;
/// Per-pass blending weight toward the neighbour mean in the Laplacian
/// smoothing (dimensionless, in [0, 1]).
const HEAT_TRANSPORT_ALPHA: f64 = 0.20;
/// Maximum BFS hops used for ocean proximity decay.
const OCEAN_PROX_MAX_DIST: u32 = 20;
/// Maximum annual precipitation in the wettest coastal-tropical cells (mm/year).
const MAX_PRECIP_MM: f32 = 3500.0;
/// Minimum annual precipitation anywhere (mm/year).
const MIN_PRECIP_MM: f32 = 5.0;

// ─── Public entry point ───────────────────────────────────────────────────────

/// Compute climate and assign biomes for all cells.
///
/// Populates `data.temperature_k`, `data.precipitation_mm`,
/// `data.ocean_proximity`, and `data.biome`.
/// Must be called **after** `run_tectonics` and `run_impacts`.
pub fn run_biomes(data: &mut PlanetData) {
    let temperature = cell_temperatures(data);
    let ocean_proximity = ocean_proximity_bfs(data);
    let precipitation = cell_precipitation(data, &temperature, &ocean_proximity);
    let biomes = assign_biomes(data, &temperature, &precipitation, &ocean_proximity);

    let n = data.grid.cell_count();
    data.temperature_k = (0..n).map(|i| temperature[i] as f32).collect();
    data.precipitation_mm = precipitation;
    data.ocean_proximity = ocean_proximity;
    data.biome = biomes;
}

// ─── Insolation model ─────────────────────────────────────────────────────────

/// Solar constant at the planet's orbital distance (W/m²).
///
/// S₀ = L★ / (4π d²)
fn solar_constant(data: &PlanetData) -> f64 {
    let d = data.celestial.planet_orbit_m;
    data.celestial.star.luminosity_w / (4.0 * std::f64::consts::PI * d * d)
}

/// Mean annual insolation at latitude `lat` for obliquity `tilt` (both rad).
///
/// Numerically integrates the Berger (1978) daily insolation formula over one
/// orbit, sampling [`ORBIT_SAMPLES`] equally-spaced orbital longitudes:
///
///   Q_daily = (S₀/π) × (H₀·sin φ·sin δ + cos φ·cos δ·sin H₀)
///
/// where H₀ = arccos(−tan φ · tan δ) is the half-day hour angle (clamped for
/// polar day/night) and δ = ε·sin(λ) is the solar declination at orbital
/// longitude λ.
fn mean_annual_insolation(s0: f64, lat: f64, tilt: f64) -> f64 {
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();

    let mut sum = 0.0;
    for k in 0..ORBIT_SAMPLES {
        let orbital_lon =
            std::f64::consts::TAU * (k as f64 + 0.5) / ORBIT_SAMPLES as f64;
        let declination = tilt * orbital_lon.sin();

        let sin_dec = declination.sin();
        let cos_dec = declination.cos();

        // Half-day hour angle; clamped for polar night (h0=0) / midnight sun (h0=π).
        let cos_h0 = -(sin_lat * sin_dec) / (cos_lat * cos_dec);
        let h0 = if cos_h0 >= 1.0 {
            0.0
        } else if cos_h0 <= -1.0 {
            std::f64::consts::PI
        } else {
            cos_h0.acos()
        };

        let daily = s0 / std::f64::consts::PI
            * (h0 * sin_lat * sin_dec + cos_lat * cos_dec * h0.sin());
        sum += daily.max(0.0);
    }
    sum / ORBIT_SAMPLES as f64
}

// ─── Temperature model ────────────────────────────────────────────────────────

/// Per-cell mean annual temperature (K) from insolation-based energy balance
/// with iterative ice-albedo feedback and meridional heat transport.
///
/// 1. Compute per-cell mean annual insolation from stellar parameters + latitude.
/// 2. Initialise surface albedo (ocean / land).
/// 3. Energy balance: T = (Q·(1−α)/σ)^0.25 + ΔT_greenhouse − lapse_rate × altitude.
/// 4. Where T < [`ICE_THRESHOLD_K`], switch to ice albedo; repeat for convergence.
/// 5. Apply Laplacian smoothing to approximate atmospheric/oceanic heat transport.
fn cell_temperatures(data: &PlanetData) -> Vec<f64> {
    let n = data.grid.cell_count();
    let s0 = solar_constant(data);
    let tilt = data.celestial.axial_tilt_rad;

    // Step 1: per-cell annual mean insolation (W/m²).
    let insolation: Vec<f64> = (0..n)
        .map(|i| {
            let pos = data.grid.cell_position(CellId(i as u32));
            let lat = pos.y.clamp(-1.0, 1.0).asin();
            mean_annual_insolation(s0, lat, tilt)
        })
        .collect();

    // Step 2: initial surface albedo from terrain type.
    let mut albedo: Vec<f64> = (0..n)
        .map(|i| {
            if data.elevation[i] < 0.0 {
                OCEAN_ALBEDO
            } else {
                LAND_ALBEDO
            }
        })
        .collect();

    // Step 3–4: iterative energy balance with ice-albedo feedback.
    let mut temp = vec![0.0_f64; n];
    for _ in 0..ICE_ALBEDO_ITERATIONS {
        for i in 0..n {
            let absorbed = insolation[i] * (1.0 - albedo[i]);
            let t_radiative = if absorbed > 0.0 {
                (absorbed / SIGMA).powf(0.25)
            } else {
                0.0
            };
            let t_surface = t_radiative + GREENHOUSE_K;
            let t_cell = t_surface - LAPSE_RATE * data.elevation[i].max(0.0);
            temp[i] = t_cell;

            // Update albedo for next iteration: frozen → ice, else restore.
            albedo[i] = if t_cell < ICE_THRESHOLD_K {
                ICE_ALBEDO
            } else if data.elevation[i] < 0.0 {
                OCEAN_ALBEDO
            } else {
                LAND_ALBEDO
            };
        }
    }

    // Step 5: Laplacian smoothing (meridional heat transport).
    // Each pass blends each cell toward its neighbour mean, respecting
    // the spherical grid topology.
    for _ in 0..HEAT_TRANSPORT_PASSES {
        let snapshot = temp.clone();
        for i in 0..n {
            let neighbours = data.grid.cell_neighbors(CellId(i as u32));
            if neighbours.is_empty() {
                continue;
            }
            let mean: f64 = neighbours
                .iter()
                .map(|&nb| snapshot[nb as usize])
                .sum::<f64>()
                / neighbours.len() as f64;
            temp[i] = snapshot[i] + HEAT_TRANSPORT_ALPHA * (mean - snapshot[i]);
        }
    }

    temp
}

/// Ocean proximity for every cell computed by BFS from ocean cells (elevation < 0).
///
/// Returns values in [0, 1]: 1.0 at ocean, decaying linearly to 0 at
/// `OCEAN_PROX_MAX_DIST` hops inland.
fn ocean_proximity_bfs(data: &PlanetData) -> Vec<f32> {
    let n = data.grid.cell_count();
    let mut dist = vec![u32::MAX; n];
    let mut queue = VecDeque::new();

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if data.elevation[i] < 0.0 {
            dist[i] = 0;
            queue.push_back(i);
        }
    }

    while let Some(idx) = queue.pop_front() {
        let d = dist[idx];
        if d >= OCEAN_PROX_MAX_DIST {
            continue;
        }
        for &nb in data.grid.cell_neighbors(CellId(idx as u32)) {
            let nb = nb as usize;
            if dist[nb] == u32::MAX {
                dist[nb] = d + 1;
                queue.push_back(nb);
            }
        }
    }

    dist.iter()
        .map(|&d| {
            if d == u32::MAX {
                0.0_f32
            } else {
                1.0 - d as f32 / OCEAN_PROX_MAX_DIST as f32
            }
        })
        .collect()
}

/// Per-cell mean annual precipitation (mm/year).
///
/// Model: base moisture from ocean proximity, scaled by a Clausius-Clapeyron
/// humidity factor (exponential in temperature) and reduced on high terrain
/// (simplified orographic effect — windward wetting is not modelled).
fn cell_precipitation(data: &PlanetData, temp: &[f64], ocean_proximity: &[f32]) -> Vec<f32> {
    let n = data.grid.cell_count();
    let mut precip = Vec::with_capacity(n);
    for i in 0..n {
        let prox = ocean_proximity[i];
        let t = temp[i];

        // Clausius-Clapeyron: saturation vapour pressure ∝ exp((T−273)/17).
        // Clamp to avoid unrealistically dry polar or unrealistically wet hot extremes.
        let humidity = ((t - 273.0) / 17.0).exp().clamp(0.05, 4.0) as f32;

        // Orographic reduction: high land intercepts and deposits moisture before
        // it reaches further inland, leaving a rain shadow.  Simplified as a linear
        // reduction above 500 m, down to 10% retention at 5 000 m.
        let elev = data.elevation[i] as f32;
        let orographic = if elev > 0.0 {
            (1.0 - ((elev - 500.0).max(0.0) / 4500.0)).max(0.10)
        } else {
            1.0
        };

        let p = (prox * MAX_PRECIP_MM * humidity * orographic).max(MIN_PRECIP_MM);
        precip.push(p);
    }
    precip
}

// ─── Biome assignment ─────────────────────────────────────────────────────────

/// Assign a biome to every cell using temperature, precipitation, elevation, and
/// ocean proximity thresholds (simplified Whittaker climate diagram).
fn assign_biomes(
    data: &PlanetData,
    temp: &[f64],
    precip: &[f32],
    ocean_proximity: &[f32],
) -> Vec<BiomeType> {
    let n = data.grid.cell_count();
    let mut biomes = Vec::with_capacity(n);
    for i in 0..n {
        let elev = data.elevation[i];
        let t = temp[i] as f32;
        let p = precip[i];
        let prox = ocean_proximity[i];

        let biome = if elev < 0.0 {
            if elev < -4000.0 {
                BiomeType::DeepOcean
            } else {
                BiomeType::Ocean
            }
        } else if elev > 4500.0 || t < 263.0 {
            BiomeType::IceCap
        } else if elev > 3000.0 {
            BiomeType::Alpine
        } else if t < 268.0 {
            BiomeType::IceCap
        } else if t < 273.0 {
            BiomeType::Tundra
        } else if t < 283.0 {
            if p < 200.0 {
                BiomeType::ColdDesert
            } else if p < 500.0 {
                BiomeType::ColdSteppe
            } else {
                BiomeType::BorealForest
            }
        } else if t < 293.0 {
            if p < 200.0 {
                BiomeType::ColdSteppe
            } else if p >= 900.0 && prox > 0.55 {
                BiomeType::Wetland
            } else {
                BiomeType::TemperateForest
            }
        } else if t < 297.0 {
            if p < 200.0 {
                BiomeType::HotDesert
            } else if p < 1000.0 {
                BiomeType::TropicalSavanna
            } else {
                BiomeType::TropicalRainforest
            }
        } else {
            // Hot tropical zone.
            if p < 200.0 {
                BiomeType::HotDesert
            } else if p < 1200.0 {
                BiomeType::TropicalSavanna
            } else if prox > 0.70 {
                BiomeType::Mangrove
            } else {
                BiomeType::TropicalRainforest
            }
        };
        biomes.push(biome);
    }
    biomes
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::impacts::run_impacts;
    use crate::planet::tectonics::run_tectonics;
    use crate::planet::{PlanetConfig, PlanetData, TectonicMode};

    fn test_planet() -> PlanetData {
        let config = PlanetConfig {
            seed: 42,
            grid_level: 3,
            tectonic_mode: TectonicMode::Quick,
            tectonic_age_gyr: 1.8,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        run_tectonics(&mut data, |_| {});
        run_impacts(&mut data);
        run_biomes(&mut data);
        data
    }

    #[test]
    fn solar_constant_earth_like() {
        let config = PlanetConfig::default(); // seed=42, Earth radius/mass
        let data = PlanetData::new(config);
        let s0 = solar_constant(&data);
        // Earth's solar constant ≈ 1361 W/m²; star parameters are randomised
        // from seed but must be in habitable-zone range.
        assert!(
            (500.0..3000.0).contains(&s0),
            "Solar constant {s0:.0} W/m² out of plausible range"
        );
    }

    #[test]
    fn insolation_higher_at_equator_than_poles() {
        let s0 = 1361.0; // Earth-like
        let tilt = 0.4091; // 23.44°
        let q_equator = mean_annual_insolation(s0, 0.0, tilt);
        let q_pole = mean_annual_insolation(s0, std::f64::consts::FRAC_PI_2, tilt);
        assert!(
            q_equator > q_pole * 1.5,
            "Equatorial insolation ({q_equator:.1}) should greatly exceed polar ({q_pole:.1})"
        );
        // Equatorial insolation should be ~S0/π ≈ 433 W/m²
        assert!(
            (350.0..500.0).contains(&q_equator),
            "Equatorial insolation {q_equator:.1} W/m² out of expected range"
        );
    }

    #[test]
    fn zero_tilt_gives_no_polar_insolation() {
        let s0 = 1361.0;
        let q_pole = mean_annual_insolation(s0, std::f64::consts::FRAC_PI_2, 0.0);
        // With zero obliquity, the poles receive no direct sunlight.
        assert!(
            q_pole < 1.0,
            "Pole insolation with zero tilt should be near zero, got {q_pole:.1}"
        );
    }

    #[test]
    fn high_tilt_increases_polar_insolation() {
        let s0 = 1361.0;
        let q_low = mean_annual_insolation(s0, std::f64::consts::FRAC_PI_2, 0.1);
        let q_high = mean_annual_insolation(s0, std::f64::consts::FRAC_PI_2, 0.6);
        assert!(
            q_high > q_low * 2.0,
            "Higher obliquity should increase polar insolation: \
             low tilt={q_low:.1}, high tilt={q_high:.1}"
        );
    }

    #[test]
    fn equatorial_cells_warmer_than_polar() {
        let planet = test_planet();
        let n = planet.grid.cell_count();

        // Collect temperature by |sin(lat)| = |pos.y|.
        let mut equatorial_temps: Vec<f32> = Vec::new();
        let mut polar_temps: Vec<f32> = Vec::new();
        for i in 0..n {
            let pos = planet.grid.cell_position(CellId(i as u32));
            let abs_sin_lat = pos.y.abs() as f32;
            if abs_sin_lat < 0.15 {
                equatorial_temps.push(planet.temperature_k[i]);
            } else if abs_sin_lat > 0.85 {
                polar_temps.push(planet.temperature_k[i]);
            }
        }
        if equatorial_temps.is_empty() || polar_temps.is_empty() {
            return; // grid too coarse to have both; skip
        }
        let t_eq = equatorial_temps.iter().sum::<f32>() / equatorial_temps.len() as f32;
        let t_pol = polar_temps.iter().sum::<f32>() / polar_temps.len() as f32;
        assert!(
            t_eq > t_pol,
            "Equatorial mean {t_eq:.1} K should exceed polar mean {t_pol:.1} K"
        );
    }

    #[test]
    fn high_elevation_cells_are_colder() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        // Find cell pairs with same approximate latitude but different elevations.
        let mut low_temps: Vec<f32> = Vec::new();
        let mut high_temps: Vec<f32> = Vec::new();
        for i in 0..n {
            let pos = planet.grid.cell_position(CellId(i as u32));
            let abs_sin_lat = pos.y.abs() as f32;
            if abs_sin_lat < 0.3 {
                let elev = planet.elevation[i] as f32;
                if elev < 100.0 {
                    low_temps.push(planet.temperature_k[i]);
                } else if elev > 2000.0 {
                    high_temps.push(planet.temperature_k[i]);
                }
            }
        }
        if low_temps.is_empty() || high_temps.is_empty() {
            return;
        }
        let mean_low = low_temps.iter().sum::<f32>() / low_temps.len() as f32;
        let mean_high = high_temps.iter().sum::<f32>() / high_temps.len() as f32;
        assert!(
            mean_low > mean_high,
            "Low-elev mean {mean_low:.1} K should exceed high-elev mean {mean_high:.1} K"
        );
    }

    #[test]
    fn ocean_cells_have_proximity_one() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        for i in 0..n {
            if planet.elevation[i] < 0.0 {
                assert_eq!(
                    planet.ocean_proximity[i], 1.0,
                    "Ocean cell {i} should have proximity 1.0"
                );
            }
        }
    }

    #[test]
    fn ocean_proximity_non_negative() {
        let planet = test_planet();
        for &p in &planet.ocean_proximity {
            assert!((0.0..=1.0).contains(&p), "Proximity {p} out of [0,1]");
        }
    }

    #[test]
    fn precipitation_non_negative_everywhere() {
        let planet = test_planet();
        for &p in &planet.precipitation_mm {
            assert!(
                p >= MIN_PRECIP_MM * 0.5,
                "Precipitation {p} mm below minimum"
            );
        }
    }

    #[test]
    fn ocean_cells_classified_as_ocean() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        for i in 0..n {
            if planet.elevation[i] < 0.0 {
                assert!(
                    matches!(planet.biome[i], BiomeType::Ocean | BiomeType::DeepOcean),
                    "Cell {i} at elev {:.0} m got biome {:?}",
                    planet.elevation[i],
                    planet.biome[i]
                );
            }
        }
    }

    #[test]
    fn ice_cap_biome_only_on_cold_cells() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        for i in 0..n {
            if planet.biome[i] == BiomeType::IceCap {
                let t = planet.temperature_k[i];
                let elev = planet.elevation[i] as f32;
                assert!(
                    t < 273.0 || elev > 3000.0 || elev > 4500.0,
                    "IceCap at cell {i} has T={t:.1} K, elev={elev:.0} m — too warm/low"
                );
            }
        }
    }

    #[test]
    fn all_land_cells_have_non_ocean_biome() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        for i in 0..n {
            if planet.elevation[i] >= 0.0 {
                assert!(
                    !matches!(planet.biome[i], BiomeType::Ocean | BiomeType::DeepOcean),
                    "Land cell {i} (elev {:.0} m) has ocean biome",
                    planet.elevation[i]
                );
            }
        }
    }

    #[test]
    fn temperature_in_physical_range() {
        let planet = test_planet();
        for (i, &t) in planet.temperature_k.iter().enumerate() {
            assert!(
                (150.0..400.0).contains(&t),
                "Cell {i} temperature {t:.1} K outside physical range"
            );
        }
    }

    #[test]
    fn biome_counts_are_nonzero_distribution() {
        let planet = test_planet();
        let total = planet.biome.len() as f32;
        let ocean_frac = planet
            .biome
            .iter()
            .filter(|&&b| matches!(b, BiomeType::Ocean | BiomeType::DeepOcean))
            .count() as f32
            / total;
        let land_frac = 1.0 - ocean_frac;
        // Planet has some ocean and some land.
        assert!(ocean_frac > 0.01, "Too little ocean: {ocean_frac:.2}");
        assert!(land_frac > 0.01, "Too little land: {land_frac:.2}");
    }

    #[test]
    fn polar_cells_colder_from_insolation() {
        // Verify the insolation-driven model produces a clear pole-equator
        // temperature gradient — the solar energy distribution naturally
        // cools the poles without any hardcoded gradient constant.
        let planet = test_planet();
        let n = planet.grid.cell_count();

        let mut tropical = Vec::new();
        let mut polar = Vec::new();
        for i in 0..n {
            let pos = planet.grid.cell_position(CellId(i as u32));
            let abs_lat = pos.y.abs();
            if abs_lat < 0.15 {
                tropical.push(planet.temperature_k[i]);
            } else if abs_lat > 0.85 {
                polar.push(planet.temperature_k[i]);
            }
        }
        if tropical.is_empty() || polar.is_empty() {
            return;
        }
        let t_trop = tropical.iter().sum::<f32>() / tropical.len() as f32;
        let t_pol = polar.iter().sum::<f32>() / polar.len() as f32;
        let gradient = t_trop - t_pol;
        assert!(
            gradient > 15.0,
            "Insolation-driven gradient ({gradient:.1} K) should be > 15 K \
             (equator {t_trop:.1} K, pole {t_pol:.1} K)"
        );
    }

    #[test]
    fn ice_albedo_feedback_cools_frozen_cells() {
        // Ice-albedo feedback: cells that freeze should be colder than they
        // would be with land/ocean albedo alone, because ice reflects more
        // sunlight.  Verify by checking that IceCap cells have lower
        // temperature than the ice threshold.
        let planet = test_planet();
        let n = planet.grid.cell_count();
        let ice_temps: Vec<f32> = (0..n)
            .filter(|&i| planet.biome[i] == BiomeType::IceCap)
            .map(|i| planet.temperature_k[i])
            .collect();
        if ice_temps.is_empty() {
            return; // star might be too bright for any ice
        }
        let mean_ice = ice_temps.iter().sum::<f32>() / ice_temps.len() as f32;
        assert!(
            mean_ice < ICE_THRESHOLD_K as f32,
            "Mean IceCap temperature ({mean_ice:.1} K) should be below threshold \
             ({ICE_THRESHOLD_K:.0} K) due to albedo feedback"
        );
    }
}
