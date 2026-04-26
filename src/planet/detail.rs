//! Terrain detail: procedural noise, interpolation, and hillshading.
//!
//! Adds sub-cell detail to planetary visualizations. The geodesic grid provides
//! coarse continental-scale data; this module adds fractal terrain texture,
//! smooth interpolation across cell boundaries, and relief shading.

use bevy::math::DVec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

use super::grid::CellId;
use super::{BiomeType, BoundaryType, PlanetData};

// ─── Terrain noise ────────────────────────────────────────────────────────────

/// Procedural noise for sub-cell terrain detail.
///
/// Combines fractal Brownian motion (rolling hills) with ridged multi-fractal
/// (sharp ridgelines, volcanic peaks). The mix is controlled by local terrain
/// roughness so mountains have ridges while ocean floors stay smooth.
pub struct TerrainNoise {
    fbm: Fbm<Perlin>,
    ridged: RidgedMulti<Perlin>,
}

impl TerrainNoise {
    /// Create generators seeded from the planet seed.
    pub fn new(seed: u64) -> Self {
        // Offset seed to avoid correlation with tectonic simulation RNG.
        let s = seed.wrapping_add(7919) as u32;

        // FBM: 6 octaves. Frequency 50 on the unit sphere gives ~130 km
        // features at Earth scale — roughly 2–3 geodesic cells at level 7.
        let fbm = Fbm::<Perlin>::new(s)
            .set_octaves(6)
            .set_frequency(50.0)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        // Ridged: 4 octaves for sharp peaks and fault scarps.
        let ridged = RidgedMulti::<Perlin>::new(s.wrapping_add(137))
            .set_octaves(4)
            .set_frequency(35.0)
            .set_lacunarity(2.2);

        Self { fbm, ridged }
    }

    /// Sample combined noise at a unit-sphere position.
    ///
    /// `roughness` in \[0, 1\] controls amplitude and the FBM/ridged mix.
    /// Returns a displacement in meters.
    pub fn sample(&self, pos: DVec3, roughness: f64) -> f64 {
        let p = [pos.x, pos.y, pos.z];
        let fbm_val = self.fbm.get(p);
        let ridge_val = self.ridged.get(p);

        let mix = roughness * 0.6;
        let combined = fbm_val * (1.0 - mix) + ridge_val * mix;
        combined * roughness * 2000.0
    }
}

// ─── Roughness ────────────────────────────────────────────────────────────────

/// Biome-dependent noise roughness \[0, 1\].
///
/// Higher roughness → larger noise amplitude and more ridged character.
pub fn terrain_roughness(
    biome: BiomeType,
    elevation: f64,
    volcanic: f32,
    boundary: BoundaryType,
) -> f64 {
    let base = match biome {
        BiomeType::Ocean | BiomeType::DeepOcean => 0.08,
        BiomeType::IceCap => 0.12,
        BiomeType::Tundra | BiomeType::ColdDesert => 0.20,
        BiomeType::ColdSteppe | BiomeType::HotDesert => 0.25,
        BiomeType::Wetland | BiomeType::Mangrove => 0.10,
        BiomeType::TemperateForest | BiomeType::BorealForest => 0.30,
        BiomeType::TropicalRainforest | BiomeType::TropicalSavanna => 0.25,
        BiomeType::Alpine => 0.70,
    };

    let elev = (elevation.abs() / 5000.0).clamp(0.0, 1.0) * 0.3;
    let bnd = match boundary {
        BoundaryType::Interior => 0.0,
        BoundaryType::Convergent => 0.40,
        BoundaryType::Divergent => 0.15,
        BoundaryType::Transform => 0.25,
    };
    let vol = volcanic as f64 * 0.3;

    (base + elev + bnd + vol).clamp(0.0, 1.0)
}

// ─── Interpolation ────────────────────────────────────────────────────────────

/// Interpolate elevation at `pos` via IDW from a cell and its neighbors.
///
/// Uses inverse-distance-squared weighting on angular distance. The nearest
/// cell dominates at its center; near cell boundaries, neighbors blend in
/// smoothly.
pub fn interpolate_elevation(data: &PlanetData, pos: DVec3, nearest: CellId) -> f64 {
    let ci = nearest.index();
    let neighbors = data.grid.cell_neighbors(nearest);

    let mut total_w = 0.0;
    let mut weighted_e = 0.0;

    let mut add = |idx: usize| {
        let cpos = data.grid.cell_position(CellId(idx as u32));
        let d = (1.0 - pos.dot(cpos)).max(1e-14);
        let w = 1.0 / (d * d);
        total_w += w;
        weighted_e += w * data.elevation[idx];
    };

    add(ci);
    for &nid in neighbors {
        add(nid as usize);
    }

    weighted_e / total_w
}

/// Compute detailed elevation: interpolation + procedural noise.
///
/// Returns `(elevation_m, nearest_cell_index)`.
///
/// # Ocean surface guarantee
///
/// When the IDW-interpolated elevation is negative (i.e. below sea level), the
/// final result is clamped to at most −2 m.  This prevents high-frequency noise
/// from pushing shallow coastal ocean columns above sea level, which would
/// generate land spikes next to water columns and produce non-collidable terrain
/// walls at the coastline.
pub fn sample_detailed_elevation(
    data: &PlanetData,
    noise: &TerrainNoise,
    pos: DVec3,
    nearest: CellId,
) -> (f64, usize) {
    let ci = nearest.index();
    let interp = interpolate_elevation(data, pos, nearest);
    let roughness = terrain_roughness(
        data.biome[ci],
        interp,
        data.volcanic_activity[ci],
        data.boundary_type[ci],
    );
    let offset = noise.sample(pos, roughness);
    let elevation = if interp < 0.0 {
        // Below-sea-level territory: never let noise lift terrain above sea level.
        (interp + offset).min(-2.0)
    } else {
        interp + offset
    };
    (elevation, ci)
}

// ─── Hillshading ──────────────────────────────────────────────────────────────

/// Sun direction for hillshade computation.
pub struct HillshadeParams {
    sun_x: f64,
    sun_y: f64,
    sun_z: f64,
}

impl Default for HillshadeParams {
    /// Standard NW cartographic illumination (315° azimuth, 45° altitude).
    fn default() -> Self {
        Self::new(315.0_f64.to_radians(), 45.0_f64.to_radians())
    }
}

impl HillshadeParams {
    /// Create from azimuth (from north, clockwise) and altitude (above horizon).
    pub fn new(azimuth: f64, altitude: f64) -> Self {
        Self {
            sun_x: azimuth.sin() * altitude.cos(),
            sun_y: azimuth.cos() * altitude.cos(),
            sun_z: altitude.sin(),
        }
    }
}

/// Compute hillshade illumination for one pixel of an elevation grid.
///
/// `elevations` is row-major (width × height). NaN entries are treated as
/// having the same elevation as the query pixel. `cell_size_m` is the
/// approximate ground distance per pixel. `z_factor` exaggerates slopes.
///
/// Returns illumination in \[0, 1\].
#[allow(clippy::too_many_arguments)]
pub fn hillshade_pixel(
    elevations: &[f64],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    cell_size_m: f64,
    z_factor: f64,
    params: &HillshadeParams,
) -> f32 {
    let center = elevations[y * width + x];
    let get = |xi: usize, yi: usize| {
        let v = elevations[yi * width + xi];
        if v.is_nan() { center } else { v }
    };

    let left = if x > 0 { get(x - 1, y) } else { center };
    let right = if x + 1 < width { get(x + 1, y) } else { center };
    let up = if y > 0 { get(x, y - 1) } else { center };
    let down = if y + 1 < height {
        get(x, y + 1)
    } else {
        center
    };

    let dzdx = (right - left) / (2.0 * cell_size_m) * z_factor;
    let dzdy = (up - down) / (2.0 * cell_size_m) * z_factor;

    // Surface normal in (east, north, up) frame.
    let len = (dzdx * dzdx + dzdy * dzdy + 1.0).sqrt();
    let nx = -dzdx / len;
    let ny = dzdy / len;
    let nz = 1.0 / len;

    let shade = nx * params.sun_x + ny * params.sun_y + nz * params.sun_z;

    // Ambient floor so shadowed areas aren't completely black.
    (shade * 0.55 + 0.45).clamp(0.0, 1.0) as f32
}

// ─── Helper ───────────────────────────────────────────────────────────────────

/// Convert (lat, lon) in radians to a unit-sphere DVec3 (Y-up).
pub fn lat_lon_to_pos(lat: f64, lon: f64) -> DVec3 {
    let cos_lat = lat.cos();
    DVec3::new(cos_lat * lon.sin(), lat.sin(), cos_lat * lon.cos())
}

/// Inverse of [`lat_lon_to_pos`]: convert a unit-length direction (Y-up)
/// back to (lat, lon) in radians.  This is the convention used by the
/// map projection and the voxel terrain (`sample_surface_radius_at`).
pub fn pos_to_lat_lon(dir: DVec3) -> (f64, f64) {
    let d = dir.normalize_or_zero();
    let lat = d.y.clamp(-1.0, 1.0).asin();
    let lon = d.x.atan2(d.z);
    (lat, lon)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::tectonics::run_tectonics;
    use crate::planet::{PlanetConfig, PlanetData};

    #[test]
    fn noise_is_deterministic() {
        let n = TerrainNoise::new(42);
        let pos = DVec3::new(0.5, 0.7, 0.5).normalize();
        let a = n.sample(pos, 0.5);
        let b = n.sample(pos, 0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn noise_varies_with_position() {
        let n = TerrainNoise::new(42);
        let a = n.sample(DVec3::X, 0.5);
        let b = n.sample(DVec3::Y, 0.5);
        assert!((a - b).abs() > 0.1, "noise should vary: {a} vs {b}");
    }

    #[test]
    fn zero_roughness_gives_zero_noise() {
        let n = TerrainNoise::new(99);
        let val = n.sample(DVec3::new(0.3, 0.6, 0.7).normalize(), 0.0);
        assert!(val.abs() < f64::EPSILON);
    }

    #[test]
    fn roughness_ocean_is_low() {
        let r = terrain_roughness(BiomeType::Ocean, -3000.0, 0.0, BoundaryType::Interior);
        assert!(r < 0.3, "ocean roughness should be low, got {r}");
    }

    /// Shallow ocean columns must never produce terrain above sea level (−2 m
    /// margin), regardless of noise.  This is the regression test for the
    /// "phantom walls at coastlines" bug: high-frequency noise was previously
    /// able to push ocean terrain above sea level, generating land spikes in
    /// what appeared to be open water.
    #[test]
    fn shallow_ocean_elevation_stays_below_sea_level() {
        use crate::planet::tectonics::run_tectonics;
        use crate::planet::{PlanetConfig, PlanetData};

        let config = PlanetConfig {
            seed: 77,
            grid_level: 2,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        run_tectonics(&mut data, |_| {});

        let noise = TerrainNoise::new(77);

        // Force all cells to shallow ocean so IDW always returns ~ −50 m.
        let n_cells = data.elevation.len();
        for i in 0..n_cells {
            data.elevation[i] = -50.0;
            data.biome[i] = BiomeType::Ocean;
        }

        // Sample at each cell centre — nearest cell is exactly that cell, so
        // IDW returns exactly −50 m as the interpolated elevation.
        for ci in 0..n_cells {
            let cell = CellId(ci as u32);
            let pos = data.grid.cell_position(cell);
            let (elev, _) = sample_detailed_elevation(&data, &noise, pos, cell);
            assert!(
                elev <= -2.0,
                "shallow ocean elevation {elev:.1} m exceeds −2 m at cell {ci}"
            );
        }
    }

    #[test]
    fn roughness_alpine_convergent_is_high() {
        let r = terrain_roughness(BiomeType::Alpine, 4000.0, 0.5, BoundaryType::Convergent);
        assert!(
            r > 0.8,
            "alpine+convergent roughness should be high, got {r}"
        );
    }

    #[test]
    fn interpolation_near_cell_center() {
        let config = PlanetConfig {
            seed: 42,
            grid_level: 2,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        run_tectonics(&mut data, |_| {});

        let cell = CellId(5);
        let pos = data.grid.cell_position(cell);
        let interp = interpolate_elevation(&data, pos, cell);
        let actual = data.elevation[5];
        assert!(
            (interp - actual).abs() < actual.abs() * 0.15 + 10.0,
            "at cell center: interp={interp}, actual={actual}"
        );
    }

    #[test]
    fn hillshade_flat_returns_consistent() {
        let elevations = vec![100.0; 9];
        let params = HillshadeParams::default();
        let shade = hillshade_pixel(&elevations, 1, 1, 3, 3, 10000.0, 5.0, &params);
        // Flat terrain: normal = (0,0,1), shade = sun_z = sin(45°) ≈ 0.707
        // With ambient: 0.707 * 0.55 + 0.45 = 0.839
        assert!(
            (shade - 0.84).abs() < 0.05,
            "flat shade should be ~0.84, got {shade}"
        );
    }

    #[test]
    fn hillshade_slope_differs_from_flat() {
        let flat = vec![100.0; 9];
        let sloped = vec![
            100.0, 100.0, 100.0, 100.0, 200.0, 400.0, 100.0, 100.0, 100.0,
        ];
        let params = HillshadeParams::default();
        let flat_s = hillshade_pixel(&flat, 1, 1, 3, 3, 1000.0, 5.0, &params);
        let slope_s = hillshade_pixel(&sloped, 1, 1, 3, 3, 1000.0, 5.0, &params);
        assert!(
            (flat_s - slope_s).abs() > 0.01,
            "slope should differ from flat: flat={flat_s}, slope={slope_s}"
        );
    }

    #[test]
    fn hillshade_nan_fallback() {
        let elevations = vec![
            f64::NAN,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ];
        let params = HillshadeParams::default();
        // Should not panic; NaN neighbor falls back to center value.
        let shade = hillshade_pixel(&elevations, 1, 1, 3, 3, 10000.0, 5.0, &params);
        assert!(shade.is_finite());
    }
}
