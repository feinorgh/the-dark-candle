//! Climate estimation and biome classification for planetary surfaces.
//!
//! A static single-column climate model:
//! stellar insolation + latitude gradient + elevation lapse rate → mean annual
//! temperature; ocean-proximity BFS + Clausius-Clapeyron humidity scaling →
//! mean annual precipitation; simplified Whittaker diagram → biome type.
//!
//! No full atmospheric dynamics are simulated.  The output is a plausible
//! static climate map sufficient for biome and resource distribution.

use super::grid::CellId;
use super::{BiomeType, PlanetData};
use std::collections::VecDeque;

// ─── Climate constants ────────────────────────────────────────────────────────

/// Environmental lapse rate (K/m): temperature decreases with altitude.
const LAPSE_RATE: f64 = 6.5e-3;
/// Assumed planetary albedo (fraction of stellar radiation reflected).
const ALBEDO: f64 = 0.30;
/// Greenhouse warming above blackbody equilibrium (K).  Earth ≈ 33 K.
const GREENHOUSE_K: f64 = 33.0;
/// Annual-mean equator-to-pole temperature gradient (K).  Earth ≈ 60 K.
const POLE_GRADIENT_K: f64 = 60.0;
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
    let t_equator = equatorial_temperature(data);
    let temperature = cell_temperatures(data, t_equator);
    let ocean_proximity = ocean_proximity_bfs(data);
    let precipitation = cell_precipitation(data, &temperature, &ocean_proximity);
    let biomes = assign_biomes(data, &temperature, &precipitation, &ocean_proximity);

    let n = data.grid.cell_count();
    data.temperature_k = (0..n).map(|i| temperature[i] as f32).collect();
    data.precipitation_mm = precipitation;
    data.ocean_proximity = ocean_proximity;
    data.biome = biomes;
}

// ─── Climate model ────────────────────────────────────────────────────────────

/// Mean annual temperature at the equator (K) from stellar parameters.
///
/// Zero-dimensional energy balance: T_eq = T★ × √(R★ / 2d) × (1−α)^0.25 + ΔT_GH
fn equatorial_temperature(data: &PlanetData) -> f64 {
    let star = &data.celestial.star;
    let d = data.celestial.planet_orbit_m;
    star.temperature_k * (star.radius_m / (2.0 * d)).sqrt() * (1.0 - ALBEDO).powf(0.25)
        + GREENHOUSE_K
}

/// Per-cell mean annual temperature (K), adjusting for latitude and elevation.
///
/// Latitude effect: T = T_equator − POLE_GRADIENT × sin²(lat).
/// Elevation effect: subtract LAPSE_RATE × max(0, elevation).
///
/// In this grid's Y-up convention, `pos.y = sin(latitude)` for any cell.
fn cell_temperatures(data: &PlanetData, t_equator: f64) -> Vec<f64> {
    let n = data.grid.cell_count();
    let mut temp = Vec::with_capacity(n);
    for i in 0..n {
        let pos = data.grid.cell_position(CellId(i as u32));
        // pos.y == sin(latitude); sin²(lat) gives polar cooling without trig.
        let t_lat = t_equator - POLE_GRADIENT_K * pos.y * pos.y;
        let t_cell = t_lat - LAPSE_RATE * data.elevation[i].max(0.0);
        temp.push(t_cell);
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
    use crate::planet::{PlanetConfig, PlanetData};

    fn test_planet() -> PlanetData {
        let config = PlanetConfig {
            seed: 42,
            grid_level: 3,
            tectonic_steps: 30,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        run_tectonics(&mut data, |_| {});
        run_impacts(&mut data);
        run_biomes(&mut data);
        data
    }

    #[test]
    fn equatorial_temperature_earth_like() {
        let config = PlanetConfig::default(); // seed=42, Earth radius/mass
        let data = PlanetData::new(config);
        let t = equatorial_temperature(&data);
        // Earth's mean equatorial surface temp ≈ 300 K; allow wide range since
        // the star parameters are randomised from the seed.
        assert!(
            (240.0..360.0).contains(&t),
            "Equatorial temperature {t:.1} K out of plausible range"
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
}
