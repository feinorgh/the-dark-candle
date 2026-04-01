// Biome map: temperature/moisture noise fields for biome-based terrain material
// selection during chunk generation.
//
// The biome map provides lightweight, deterministic lookups of environmental
// parameters at any world (x, z) position without depending on the Bevy ECS
// (no `Res<Assets>`, no `Query`).  This allows terrain generators to select
// surface materials, soil depth, and terrain modifiers based on biome context.

use noise::{NoiseFn, Perlin};

// ── Altitude zone boundaries (world Y) ────────────────────────────────────

/// Below this, coastal sand can appear on flat terrain near sea level.
pub const COASTAL_SAND_ALTITUDE: f64 = 3.0;

/// Above this (relative to sea level), terrain enters the alpine zone
/// where exposed rock replaces soil.
pub const ALPINE_ALTITUDE: f64 = 45.0;

/// Above this (relative to sea level), permanent snow/ice appears.
pub const SNOW_LINE_ALTITUDE: f64 = 60.0;

// ── Slope thresholds (in voxels of height difference per 1 voxel horizontal) ─

/// Slopes steeper than this expose bare stone.
pub const STEEP_SLOPE_THRESHOLD: f64 = 1.5;

/// Slopes steeper than this replace grass with dirt.
pub const MODERATE_SLOPE_THRESHOLD: f64 = 0.8;

/// Compact environmental parameters at a world position.
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentSample {
    /// Temperature in Kelvin.  Higher near "equator" (z=0), lower far away.
    pub temperature_k: f64,
    /// Moisture ratio [0, 1].  0 = arid, 1 = saturated.
    pub moisture: f64,
}

/// Stateless environment sampler.  Generates temperature and moisture at any
/// world (x, z) from noise fields seeded deterministically.
pub struct EnvironmentMap {
    temp_base: Perlin,
    temp_detail: Perlin,
    moisture_base: Perlin,
    moisture_detail: Perlin,
    seed: u32,
}

impl EnvironmentMap {
    pub fn new(seed: u32) -> Self {
        Self {
            temp_base: Perlin::new(seed.wrapping_add(700)),
            temp_detail: Perlin::new(seed.wrapping_add(701)),
            moisture_base: Perlin::new(seed.wrapping_add(710)),
            moisture_detail: Perlin::new(seed.wrapping_add(711)),
            seed,
        }
    }

    /// Sample environment parameters at world position `(x, z)`.
    pub fn sample(&self, x: f64, z: f64) -> EnvironmentSample {
        let temp_k = self.temperature(x, z);
        let moisture = self.moisture(x, z);
        EnvironmentSample {
            temperature_k: temp_k,
            moisture,
        }
    }

    /// Temperature in Kelvin at `(x, z)`.
    ///
    /// Base: 288 K (15 °C) at z=0, decreasing toward high |z| latitudes.
    /// Perturbed by two noise layers for regional variation.
    fn temperature(&self, x: f64, z: f64) -> f64 {
        // Latitude gradient: 288 K at z=0, dropping ~0.003 K/m toward poles
        let latitude_effect = (z.abs() * 0.003).min(50.0);
        let base = 288.0 - latitude_effect;

        // Large-scale climate zones
        let broad = self.temp_base.get([x * 0.0008, z * 0.0008]) * 12.0;
        // Regional variation
        let detail = self.temp_detail.get([x * 0.004, z * 0.004]) * 4.0;

        (base + broad + detail).clamp(200.0, 320.0)
    }

    /// Moisture ratio [0, 1] at `(x, z)`.
    ///
    /// Noise-based with a slight bias: negative z tends drier (rain shadow).
    fn moisture(&self, x: f64, z: f64) -> f64 {
        let broad = self.moisture_base.get([x * 0.001, z * 0.001]);
        let detail = self.moisture_detail.get([x * 0.005, z * 0.005]);
        let raw = 0.5 + broad * 0.35 + detail * 0.15;
        raw.clamp(0.0, 1.0)
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }
}

use super::voxel::MaterialId;

/// Select the surface material based on slope, altitude relative to sea level,
/// and environment parameters.
///
/// `slope`: absolute height gradient (height difference between adjacent
///          voxels).  0 = flat, >1.5 = cliff.
/// `altitude_above_sea`: world_y − sea_level.
/// `env`: temperature/moisture at this position.
/// `has_erosion_override`: if true, an erosion channel already set the material.
pub fn surface_material(
    slope: f64,
    altitude_above_sea: f64,
    env: &EnvironmentSample,
) -> MaterialId {
    // Snow line (cold + high altitude)
    if altitude_above_sea > SNOW_LINE_ALTITUDE && env.temperature_k < 270.0 {
        return MaterialId::ICE;
    }

    // Alpine zone: exposed rock at high altitude
    if altitude_above_sea > ALPINE_ALTITUDE {
        return if slope > MODERATE_SLOPE_THRESHOLD {
            MaterialId::STONE
        } else {
            // Sparse alpine grass/dirt
            MaterialId::DIRT
        };
    }

    // Steep slopes: exposed stone
    if slope > STEEP_SLOPE_THRESHOLD {
        return MaterialId::STONE;
    }

    // Moderate slopes: dirt instead of grass
    if slope > MODERATE_SLOPE_THRESHOLD {
        return MaterialId::DIRT;
    }

    // Coastal sand: near sea level on flat terrain
    if (0.0..COASTAL_SAND_ALTITUDE).contains(&altitude_above_sea) {
        return MaterialId::SAND;
    }

    // Arid regions: sand instead of grass
    if env.moisture < 0.2 && env.temperature_k > 295.0 {
        return MaterialId::SAND;
    }

    // Cold regions: dirt (no grass grows below ~260 K)
    if env.temperature_k < 260.0 {
        return MaterialId::DIRT;
    }

    // Default: grass
    MaterialId::GRASS
}

/// Calculate the approximate slope at a position using finite differences.
///
/// Takes a height-sampling function and computes the gradient magnitude
/// from the four cardinal neighbors.
pub fn compute_slope(sample_height: impl Fn(f64, f64) -> f64, x: f64, z: f64) -> f64 {
    let dx = sample_height(x + 1.0, z) - sample_height(x - 1.0, z);
    let dz = sample_height(x, z + 1.0) - sample_height(x, z - 1.0);
    // Gradient magnitude (rise over 2m run)
    (dx * dx + dz * dz).sqrt() * 0.5
}

/// Adjust soil depth based on slope and local terrain character.
///
/// - Valleys (low slope, below average height) get thicker soil (×1.5)
/// - Ridges (high slope) get thinner soil (×0.3 minimum 1)
/// - Normal terrain keeps the base soil depth
pub fn adjusted_soil_depth(base_depth: f64, slope: f64) -> f64 {
    if slope > STEEP_SLOPE_THRESHOLD {
        // Cliff: almost no soil
        1.0_f64.max(base_depth * 0.3)
    } else if slope > MODERATE_SLOPE_THRESHOLD {
        // Moderate slope: reduced soil
        base_depth * 0.6
    } else if slope < 0.2 {
        // Very flat: accumulated soil (valley-like)
        base_depth * 1.5
    } else {
        base_depth
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_sample_is_deterministic() {
        let map = EnvironmentMap::new(42);
        let s1 = map.sample(100.0, 200.0);
        let s2 = map.sample(100.0, 200.0);
        assert_eq!(s1.temperature_k, s2.temperature_k);
        assert_eq!(s1.moisture, s2.moisture);
    }

    #[test]
    fn temperature_decreases_with_latitude() {
        let map = EnvironmentMap::new(42);
        let t_equator = map.sample(0.0, 0.0).temperature_k;
        let t_polar = map.sample(0.0, 5000.0).temperature_k;
        assert!(
            t_equator > t_polar,
            "Equator ({t_equator:.1} K) should be warmer than polar ({t_polar:.1} K)"
        );
    }

    #[test]
    fn temperature_is_bounded() {
        let map = EnvironmentMap::new(42);
        for i in 0..1000 {
            let x = (i as f64) * 17.3;
            let z = (i as f64) * 23.7 - 500.0;
            let t = map.sample(x, z).temperature_k;
            assert!(
                (200.0..=320.0).contains(&t),
                "Temperature {t} K out of range [200, 320]"
            );
        }
    }

    #[test]
    fn moisture_is_bounded_zero_one() {
        let map = EnvironmentMap::new(42);
        for i in 0..1000 {
            let x = (i as f64) * 11.1;
            let z = (i as f64) * 7.7;
            let m = map.sample(x, z).moisture;
            assert!((0.0..=1.0).contains(&m), "Moisture {m} out of range [0, 1]");
        }
    }

    #[test]
    fn moisture_has_variation() {
        let map = EnvironmentMap::new(42);
        let mut min_m = 1.0_f64;
        let mut max_m = 0.0_f64;
        for i in 0..2000 {
            let x = (i as f64) * 5.3;
            let z = (i as f64) * 3.1;
            let m = map.sample(x, z).moisture;
            min_m = min_m.min(m);
            max_m = max_m.max(m);
        }
        assert!(
            max_m - min_m > 0.3,
            "Moisture should vary: min={min_m:.3}, max={max_m:.3}"
        );
    }

    #[test]
    fn surface_material_steep_slope_is_stone() {
        let env = EnvironmentSample {
            temperature_k: 288.0,
            moisture: 0.5,
        };
        assert_eq!(surface_material(2.0, 20.0, &env), MaterialId::STONE);
    }

    #[test]
    fn surface_material_moderate_slope_is_dirt() {
        let env = EnvironmentSample {
            temperature_k: 288.0,
            moisture: 0.5,
        };
        assert_eq!(surface_material(1.0, 20.0, &env), MaterialId::DIRT);
    }

    #[test]
    fn surface_material_flat_default_is_grass() {
        let env = EnvironmentSample {
            temperature_k: 288.0,
            moisture: 0.5,
        };
        assert_eq!(surface_material(0.1, 20.0, &env), MaterialId::GRASS);
    }

    #[test]
    fn surface_material_snow_at_high_cold() {
        let env = EnvironmentSample {
            temperature_k: 260.0,
            moisture: 0.5,
        };
        assert_eq!(surface_material(0.1, 65.0, &env), MaterialId::ICE);
    }

    #[test]
    fn surface_material_alpine_dirt() {
        let env = EnvironmentSample {
            temperature_k: 280.0,
            moisture: 0.4,
        };
        assert_eq!(surface_material(0.3, 50.0, &env), MaterialId::DIRT);
    }

    #[test]
    fn surface_material_coastal_sand() {
        let env = EnvironmentSample {
            temperature_k: 288.0,
            moisture: 0.5,
        };
        assert_eq!(surface_material(0.1, 1.0, &env), MaterialId::SAND);
    }

    #[test]
    fn surface_material_desert_sand() {
        let env = EnvironmentSample {
            temperature_k: 300.0,
            moisture: 0.1,
        };
        assert_eq!(surface_material(0.1, 20.0, &env), MaterialId::SAND);
    }

    #[test]
    fn surface_material_cold_dirt() {
        let env = EnvironmentSample {
            temperature_k: 250.0,
            moisture: 0.5,
        };
        assert_eq!(surface_material(0.1, 20.0, &env), MaterialId::DIRT);
    }

    #[test]
    fn compute_slope_flat_terrain_is_zero() {
        let slope = compute_slope(|_, _| 64.0, 10.0, 10.0);
        assert!(
            slope.abs() < 1e-10,
            "Flat terrain slope should be ~0, got {slope}"
        );
    }

    #[test]
    fn compute_slope_tilted_terrain() {
        // Height = x → slope of 1.0 in x direction
        let slope = compute_slope(|x, _| x, 10.0, 10.0);
        assert!(
            (slope - 1.0).abs() < 1e-10,
            "Slope should be ~1.0 for height=x, got {slope}"
        );
    }

    #[test]
    fn adjusted_soil_depth_steep_is_thin() {
        let depth = adjusted_soil_depth(5.0, 2.0);
        assert!(depth < 2.0, "Steep slope soil should be thin, got {depth}");
    }

    #[test]
    fn adjusted_soil_depth_flat_is_thick() {
        let depth = adjusted_soil_depth(5.0, 0.1);
        assert!(
            depth > 5.0,
            "Very flat terrain should have thick soil, got {depth}"
        );
    }
}
