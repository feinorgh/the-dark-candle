// Cloud shadow projection and atmospheric fog computation.
//
// Provides pure functions for computing shadow factors from cloud fields
// and fog transmittance from atmospheric humidity/temperature data.
//
// Cloud shadows use Beer-Lambert law: ray-march through cloud LWC field,
// accumulate optical depth, and return shadow factor = exp(-optical_depth).
//
// Atmospheric fog uses exponential height falloff with humidity amplification
// and cold-pooling effects (T < dew_point increases fog density).
//
// All units are SI:
//   - positions/distances: meters
//   - cloud_lwc: kg/m³
//   - humidity: kg/kg (mixing ratio)
//   - temperature/dew_point: Kelvin
//   - pressure: Pascals
//   - extinction coefficients: m⁻¹

use crate::physics::atmosphere;

/// Parameters for cloud shadow computation.
#[derive(Debug, Clone)]
pub struct CloudShadowParams {
    /// Unit vector toward the sun.
    pub sun_direction: [f32; 3],
    /// Shadow blur radius in voxels (larger = softer shadows). Default: 2.0.
    pub shadow_softness: f32,
    /// Minimum light transmission through dense clouds (0.0 = fully blocked). Default: 0.3.
    pub min_shadow_factor: f32,
    /// Cloud extinction coefficient per kg/m³ of liquid water content (m⁻¹). Default: 50.0.
    pub extinction_coeff: f32,
}

impl Default for CloudShadowParams {
    fn default() -> Self {
        Self {
            sun_direction: [0.0, 1.0, 0.0],
            shadow_softness: 2.0,
            min_shadow_factor: 0.3,
            extinction_coeff: 50.0,
        }
    }
}

/// Parameters for height-based exponential fog.
#[derive(Debug, Clone)]
pub struct FogParams {
    /// Base fog density at ground level (extinction coefficient m⁻¹). Default: 0.02.
    pub fog_density_base: f32,
    /// Exponential height falloff rate (m⁻¹). Default: 0.1.
    pub fog_height_falloff: f32,
    /// Humidity amplification factor (dimensionless). Default: 5.0.
    pub humidity_scale: f32,
    /// Cold air amplification factor (K⁻¹). Default: 0.01.
    pub temperature_factor: f32,
    /// Fog color (RGB, sRGB normalized). Default: [0.7, 0.75, 0.8].
    pub fog_color: [f32; 3],
    /// Maximum fog integration distance in meters. Default: 500.0.
    pub max_fog_distance: f32,
}

impl Default for FogParams {
    fn default() -> Self {
        Self {
            fog_density_base: 0.02,
            fog_height_falloff: 0.1,
            humidity_scale: 5.0,
            temperature_factor: 0.01,
            fog_color: [0.7, 0.75, 0.8],
            max_fog_distance: 500.0,
        }
    }
}

/// Compute shadow factor (0.0 = full shadow, 1.0 = full sun) for a surface point.
///
/// Traces a ray from the surface position along the sun direction through the
/// cloud field, accumulating optical depth via Beer-Lambert law. Returns the
/// exponential transmittance clamped to `min_shadow_factor`.
///
/// # Arguments
/// * `surface_pos` - 3D position of the surface point in world coordinates (meters)
/// * `cloud_field` - Flat array of cloud LWC values (kg/m³), indexed [z*size*size + y*size + x]
/// * `size` - Grid dimension (cubic grid: size × size × size)
/// * `params` - Shadow computation parameters
///
/// # Returns
/// Shadow factor in [0, 1], where 1.0 = full sunlight, 0.0 = complete shadow.
pub fn cloud_shadow_factor(
    surface_pos: [f32; 3],
    cloud_field: &[f32],
    size: usize,
    params: &CloudShadowParams,
) -> f32 {
    // Normalize sun direction.
    let sun_len = (params.sun_direction[0].powi(2)
        + params.sun_direction[1].powi(2)
        + params.sun_direction[2].powi(2))
    .sqrt();
    if sun_len < 1e-6 {
        return 1.0; // No sun direction = no shadow.
    }
    let sun_dir = [
        params.sun_direction[0] / sun_len,
        params.sun_direction[1] / sun_len,
        params.sun_direction[2] / sun_len,
    ];

    // Ray-march step size (voxel spacing = 1 meter).
    let step_length = 1.0_f32;
    let max_steps = size * 2; // Traverse up to 2× grid size.

    let mut optical_depth = 0.0_f32;

    for step in 0..max_steps {
        let t = step as f32 * step_length;
        let pos = [
            surface_pos[0] + sun_dir[0] * t,
            surface_pos[1] + sun_dir[1] * t,
            surface_pos[2] + sun_dir[2] * t,
        ];

        // Check if position is within cloud field bounds.
        if pos[0] < 0.0
            || pos[1] < 0.0
            || pos[2] < 0.0
            || pos[0] >= size as f32
            || pos[1] >= size as f32
            || pos[2] >= size as f32
        {
            break; // Exited cloud field.
        }

        // Sample cloud LWC (trilinear interpolation would be better, but use nearest for simplicity).
        let ix = pos[0].floor() as usize;
        let iy = pos[1].floor() as usize;
        let iz = pos[2].floor() as usize;

        if ix < size && iy < size && iz < size {
            let idx = iz * size * size + iy * size + ix;
            let cloud_lwc = cloud_field[idx];

            // Accumulate optical depth: τ += k_ext × LWC × Δs.
            optical_depth += params.extinction_coeff * cloud_lwc * step_length;
        }
    }

    // Beer-Lambert law: transmittance = exp(-τ), clamped to minimum.
    let transmittance = (-optical_depth).exp();
    transmittance.max(params.min_shadow_factor)
}

/// Compute a 2D shadow map for a horizontal surface at `y_level`.
///
/// Returns a size × size grid of shadow factors. Each grid cell (x, z) contains
/// the shadow factor for a surface point at (x, y_level, z).
///
/// # Arguments
/// * `cloud_field` - Flat array of cloud LWC values (kg/m³), size³ elements
/// * `size` - Grid dimension (cubic grid)
/// * `y_level` - Y-coordinate of the horizontal surface
/// * `params` - Shadow computation parameters
///
/// # Returns
/// Flat size × size array of shadow factors, indexed [z * size + x].
pub fn compute_shadow_map(
    cloud_field: &[f32],
    size: usize,
    y_level: usize,
    params: &CloudShadowParams,
) -> Vec<f32> {
    let mut shadow_map = vec![1.0_f32; size * size];

    for z in 0..size {
        for x in 0..size {
            let surface_pos = [x as f32 + 0.5, y_level as f32 + 0.5, z as f32 + 0.5];
            let factor = cloud_shadow_factor(surface_pos, cloud_field, size, params);
            shadow_map[z * size + x] = factor;
        }
    }

    shadow_map
}

/// Compute fog density at a given position based on altitude, humidity, and temperature.
///
/// Fog density uses exponential height falloff, amplified by humidity and cold-pooling:
/// `ρ_fog = ρ₀ × exp(-h × λ) × (1 + w × s_w) × f_cold`
///
/// where:
/// - ρ₀ = base fog density at ground level
/// - h = altitude (meters)
/// - λ = height falloff rate (m⁻¹)
/// - w = humidity mixing ratio (kg/kg)
/// - s_w = humidity amplification scale
/// - f_cold = 1 + max(0, T_dew - T) × k_T (cold air factor)
///
/// # Arguments
/// * `altitude` - Height above ground in meters
/// * `humidity` - Mixing ratio (kg water vapor / kg dry air)
/// * `temperature` - Air temperature in Kelvin
/// * `dew_point` - Dew point temperature in Kelvin
/// * `params` - Fog computation parameters
///
/// # Returns
/// Fog extinction coefficient in m⁻¹.
pub fn fog_density(
    altitude: f32,
    humidity: f32,
    temperature: f32,
    dew_point: f32,
    params: &FogParams,
) -> f32 {
    // Exponential height falloff.
    let height_factor = (-altitude * params.fog_height_falloff).exp();

    // Humidity amplification: humid air = thicker fog.
    let humidity_factor = 1.0 + humidity * params.humidity_scale;

    // Cold pooling: T < dew point increases fog density (morning valley fog).
    let cold_factor = 1.0 + (dew_point - temperature).max(0.0) * params.temperature_factor;

    params.fog_density_base * height_factor * humidity_factor * cold_factor
}

/// Compute fog transmittance along a ray segment.
///
/// Integrates fog density along the ray from `ray_origin` in direction `ray_dir`
/// for a given `distance`. Uses numerical integration with step size = 1 meter.
///
/// # Arguments
/// * `ray_origin` - Starting point of the ray (meters)
/// * `ray_dir` - Unit direction vector
/// * `distance` - Ray length to integrate (meters)
/// * `humidity_field` - Flat size³ array of humidity values (kg/kg)
/// * `temperature_field` - Flat size³ array of temperature values (K)
/// * `pressure_field` - Flat size³ array of pressure values (Pa)
/// * `size` - Grid dimension
/// * `params` - Fog computation parameters
///
/// # Returns
/// (transmittance, fog_color_contribution) where:
/// - transmittance ∈ [0, 1]: fraction of light transmitted (1 = no fog, 0 = fully obscured)
/// - fog_color_contribution: RGB color contribution from fog scattering
#[allow(clippy::too_many_arguments)]
pub fn fog_transmittance(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    distance: f32,
    humidity_field: &[f32],
    temperature_field: &[f32],
    pressure_field: &[f32],
    size: usize,
    params: &FogParams,
) -> (f32, [f32; 3]) {
    // Normalize direction.
    let dir_len = (ray_dir[0].powi(2) + ray_dir[1].powi(2) + ray_dir[2].powi(2)).sqrt();
    if dir_len < 1e-6 {
        return (1.0, [0.0, 0.0, 0.0]); // Degenerate ray.
    }
    let dir = [
        ray_dir[0] / dir_len,
        ray_dir[1] / dir_len,
        ray_dir[2] / dir_len,
    ];

    let integration_distance = distance.min(params.max_fog_distance);
    let step_length = 1.0_f32; // 1 meter steps.
    let num_steps = (integration_distance / step_length).ceil() as usize;

    let mut optical_depth = 0.0_f32;

    for step in 0..num_steps {
        let t = step as f32 * step_length;
        let pos = [
            ray_origin[0] + dir[0] * t,
            ray_origin[1] + dir[1] * t,
            ray_origin[2] + dir[2] * t,
        ];

        // Check bounds.
        if pos[0] < 0.0
            || pos[1] < 0.0
            || pos[2] < 0.0
            || pos[0] >= size as f32
            || pos[1] >= size as f32
            || pos[2] >= size as f32
        {
            break;
        }

        let ix = pos[0].floor() as usize;
        let iy = pos[1].floor() as usize;
        let iz = pos[2].floor() as usize;

        if ix < size && iy < size && iz < size {
            let idx = iz * size * size + iy * size + ix;
            let humidity = humidity_field[idx];
            let temperature = temperature_field[idx];
            let pressure = pressure_field[idx];

            // Compute dew point for cold-pooling effect.
            let dew_pt = atmosphere::dew_point(humidity, pressure);

            // Fog density at this point.
            let density = fog_density(pos[1], humidity, temperature, dew_pt, params);

            // Accumulate optical depth.
            optical_depth += density * step_length;
        }
    }

    // Transmittance from Beer-Lambert law.
    let transmittance = (-optical_depth).exp();

    // Fog color contribution scales with absorbed light.
    let fog_contribution = [
        params.fog_color[0] * (1.0 - transmittance),
        params.fog_color[1] * (1.0 - transmittance),
        params.fog_color[2] * (1.0 - transmittance),
    ];

    (transmittance, fog_contribution)
}

/// Composite fog over an existing scene color.
///
/// Uses the standard fog blending formula:
/// `final_color = scene_color × transmittance + fog_color × (1 - transmittance)`
///
/// # Arguments
/// * `scene_color` - RGB color of the scene before fog (sRGB normalized)
/// * `transmittance` - Fog transmittance in [0, 1]
/// * `fog_color` - RGB fog color (sRGB normalized)
///
/// # Returns
/// Final composited RGB color.
pub fn apply_fog(scene_color: [f32; 3], transmittance: f32, fog_color: [f32; 3]) -> [f32; 3] {
    [
        scene_color[0] * transmittance + fog_color[0] * (1.0 - transmittance),
        scene_color[1] * transmittance + fog_color[1] * (1.0 - transmittance),
        scene_color[2] * transmittance + fog_color[2] * (1.0 - transmittance),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a cloud field with uniform LWC.
    fn uniform_cloud_field(size: usize, lwc: f32) -> Vec<f32> {
        vec![lwc; size * size * size]
    }

    /// Helper: create atmospheric fields with uniform values.
    fn uniform_atmosphere(
        size: usize,
        humidity: f32,
        temperature: f32,
        pressure: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let humidity_field = vec![humidity; size * size * size];
        let temperature_field = vec![temperature; size * size * size];
        let pressure_field = vec![pressure; size * size * size];
        (humidity_field, temperature_field, pressure_field)
    }

    #[test]
    fn shadow_no_clouds_full_sun() {
        let size = 32;
        let cloud_field = uniform_cloud_field(size, 0.0);
        let params = CloudShadowParams::default();

        let surface_pos = [16.0, 10.0, 16.0];
        let factor = cloud_shadow_factor(surface_pos, &cloud_field, size, &params);

        assert!(
            (factor - 1.0).abs() < 1e-5,
            "No clouds should give full sunlight (factor=1.0), got {}",
            factor
        );
    }

    #[test]
    fn shadow_dense_cloud_blocks_light() {
        let size = 32;
        let cloud_field = uniform_cloud_field(size, 1e-3); // Dense cloud: 1 g/m³
        let params = CloudShadowParams {
            extinction_coeff: 50.0,
            min_shadow_factor: 0.3,
            ..Default::default()
        };

        let surface_pos = [16.0, 2.0, 16.0]; // Near bottom.
        let factor = cloud_shadow_factor(surface_pos, &cloud_field, size, &params);

        assert!(
            factor < 0.5,
            "Dense cloud should block most light (factor < 0.5), got {}",
            factor
        );
        assert!(
            factor >= params.min_shadow_factor,
            "Shadow factor should not drop below min_shadow_factor"
        );
    }

    #[test]
    fn shadow_partial_cloud() {
        let size = 32;
        let cloud_field = uniform_cloud_field(size, 1e-4); // Thin cloud: 0.1 g/m³
        let params = CloudShadowParams::default();

        let surface_pos = [16.0, 10.0, 16.0];
        let factor = cloud_shadow_factor(surface_pos, &cloud_field, size, &params);

        assert!(
            factor > 0.5 && factor < 1.0,
            "Thin cloud should give intermediate shadow (0.5 < factor < 1.0), got {}",
            factor
        );
    }

    #[test]
    fn shadow_map_matches_individual() {
        let size = 16;
        let cloud_field = uniform_cloud_field(size, 2e-4);
        let y_level = 5;
        let params = CloudShadowParams::default();

        let shadow_map = compute_shadow_map(&cloud_field, size, y_level, &params);

        // Spot-check a few points.
        for z in [0, 7, 15] {
            for x in [0, 7, 15] {
                let surface_pos = [x as f32 + 0.5, y_level as f32 + 0.5, z as f32 + 0.5];
                let individual = cloud_shadow_factor(surface_pos, &cloud_field, size, &params);
                let from_map = shadow_map[z * size + x];
                assert!(
                    (individual - from_map).abs() < 1e-5,
                    "Shadow map mismatch at ({}, {}): individual={}, map={}",
                    x,
                    z,
                    individual,
                    from_map
                );
            }
        }
    }

    #[test]
    fn shadow_extinction_scales_with_lwc() {
        let size = 32;
        let params = CloudShadowParams::default();
        let surface_pos = [16.0, 5.0, 16.0];

        let thin = uniform_cloud_field(size, 1e-4);
        let thick = uniform_cloud_field(size, 5e-4);

        let factor_thin = cloud_shadow_factor(surface_pos, &thin, size, &params);
        let factor_thick = cloud_shadow_factor(surface_pos, &thick, size, &params);

        assert!(
            factor_thick < factor_thin,
            "Higher LWC should produce darker shadow: thin={}, thick={}",
            factor_thin,
            factor_thick
        );
    }

    #[test]
    fn fog_zero_at_high_altitude() {
        let params = FogParams::default();
        let humidity = 0.01;
        let temperature = 280.0;
        let dew_point = 275.0;

        let high_altitude = 500.0;
        let density = fog_density(high_altitude, humidity, temperature, dew_point, &params);

        assert!(
            density < 1e-3,
            "Fog density should be near zero at high altitude, got {}",
            density
        );
    }

    #[test]
    fn fog_increases_near_ground() {
        let params = FogParams::default();
        let humidity = 0.01;
        let temperature = 280.0;
        let dew_point = 275.0;

        let ground = fog_density(0.0, humidity, temperature, dew_point, &params);
        let high = fog_density(100.0, humidity, temperature, dew_point, &params);

        assert!(
            ground > high,
            "Fog density should be higher at ground level: ground={}, high={}",
            ground,
            high
        );
    }

    #[test]
    fn fog_amplified_by_humidity() {
        let params = FogParams::default();
        let temperature = 280.0;
        let dew_point = 275.0;

        let dry = fog_density(10.0, 0.001, temperature, dew_point, &params);
        let humid = fog_density(10.0, 0.02, temperature, dew_point, &params);

        assert!(
            humid > dry,
            "Higher humidity should increase fog density: dry={}, humid={}",
            dry,
            humid
        );
    }

    #[test]
    fn fog_cold_pool_effect() {
        let params = FogParams::default();
        let humidity = 0.01;

        // Case 1: T > dew_point (no cold pooling).
        let warm = fog_density(10.0, humidity, 285.0, 280.0, &params);

        // Case 2: T < dew_point (cold pooling).
        let cold = fog_density(10.0, humidity, 278.0, 280.0, &params);

        assert!(
            cold > warm,
            "Cold air below dew point should increase fog: warm={}, cold={}",
            warm,
            cold
        );
    }

    #[test]
    fn fog_transmittance_decreases_with_distance() {
        let size = 32;
        let params = FogParams {
            fog_density_base: 0.05,
            ..Default::default()
        };
        let (humidity, temperature, pressure) = uniform_atmosphere(size, 0.01, 280.0, 101325.0);

        let ray_origin = [16.0, 10.0, 16.0];
        let ray_dir = [1.0, 0.0, 0.0];

        let (trans_short, _) = fog_transmittance(
            ray_origin,
            ray_dir,
            10.0,
            &humidity,
            &temperature,
            &pressure,
            size,
            &params,
        );
        let (trans_long, _) = fog_transmittance(
            ray_origin,
            ray_dir,
            50.0,
            &humidity,
            &temperature,
            &pressure,
            size,
            &params,
        );

        assert!(
            trans_long < trans_short,
            "Longer ray should have lower transmittance: short={}, long={}",
            trans_short,
            trans_long
        );
    }

    #[test]
    fn apply_fog_blends_correctly() {
        let scene_color = [1.0, 0.5, 0.2];
        let fog_color = [0.7, 0.75, 0.8];
        let transmittance = 0.6;

        let result = apply_fog(scene_color, transmittance, fog_color);

        // Expected: scene * 0.6 + fog * 0.4
        let expected = [
            1.0 * 0.6 + 0.7 * 0.4,
            0.5 * 0.6 + 0.75 * 0.4,
            0.2 * 0.6 + 0.8 * 0.4,
        ];

        for i in 0..3 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "Fog blend mismatch channel {}: expected {}, got {}",
                i,
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn fog_no_fog_full_transmittance() {
        let size = 32;
        let params = FogParams {
            fog_density_base: 0.0, // No fog.
            ..Default::default()
        };
        let (humidity, temperature, pressure) = uniform_atmosphere(size, 0.0, 288.0, 101325.0);

        let ray_origin = [16.0, 10.0, 16.0];
        let ray_dir = [0.0, 1.0, 0.0];

        let (transmittance, fog_contrib) = fog_transmittance(
            ray_origin,
            ray_dir,
            100.0,
            &humidity,
            &temperature,
            &pressure,
            size,
            &params,
        );

        assert!(
            (transmittance - 1.0).abs() < 1e-5,
            "Zero fog density should give transmittance=1.0, got {}",
            transmittance
        );
        assert!(
            fog_contrib[0].abs() < 1e-5
                && fog_contrib[1].abs() < 1e-5
                && fog_contrib[2].abs() < 1e-5,
            "Zero fog should have no color contribution"
        );
    }
}
