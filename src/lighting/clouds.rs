// CPU-side volumetric cloud renderer using ray-marching with Beer-Lambert extinction
// and light scattering. Reads cloud density data from the LBM gas simulation.

use crate::physics::lbm_gas::plugin::CloudField;
use crate::world::chunk::{CHUNK_SIZE, ChunkCoord};
use std::f32::consts::PI;

/// Parameters controlling volumetric cloud rendering.
#[derive(Debug, Clone, Copy)]
pub struct CloudRenderParams {
    /// Extinction coefficient per kg/m³ of liquid water content (m⁻¹ per kg/m³).
    /// Typical: 1.5 m⁻¹ per kg/m³ (visible light).
    pub extinction_coeff: f32,

    /// Single-scattering albedo (fraction of light scattered vs absorbed).
    /// Typical: 0.99 for water clouds (highly scattering, minimal absorption).
    pub scattering_albedo: f32,

    /// Henyey-Greenstein asymmetry parameter for forward scattering.
    /// Typical: 0.85 for clouds (strong forward bias).
    pub forward_scatter_g: f32,

    /// Fraction of ambient light in cloud base (prevents pure black shadows).
    /// Typical: 0.3 (30% ambient).
    pub ambient_factor: f32,

    /// Maximum ray-march distance in meters.
    pub max_march_distance: f32,

    /// Ray-march step size in meters.
    /// Smaller = better quality, slower. Typical: 0.5–2.0 m.
    pub step_size: f32,

    /// Minimum LWC to consider as cloud (kg/m³).
    /// Below this threshold, density is treated as zero.
    pub density_threshold: f32,
}

impl Default for CloudRenderParams {
    fn default() -> Self {
        Self {
            extinction_coeff: 1.5,
            scattering_albedo: 0.99,
            forward_scatter_g: 0.85,
            ambient_factor: 0.3,
            max_march_distance: 500.0,
            step_size: 1.0,
            density_threshold: 1e-5,
        }
    }
}

/// Result of ray-marching through the cloud volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudSample {
    /// Accumulated cloud color (linear RGB).
    pub color: [f32; 3],

    /// Remaining transmittance (1.0 = fully transparent, 0.0 = fully opaque).
    pub transmittance: f32,
}

impl Default for CloudSample {
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0],
            transmittance: 1.0,
        }
    }
}

/// Sample cloud density at a world position using trilinear interpolation.
///
/// # Arguments
/// * `pos` - World position [x, y, z] in meters.
/// * `cloud_field` - Cloud LWC data from the LBM simulation.
/// * `chunk_size` - Edge length of a chunk in voxels (typically 32).
///
/// # Returns
/// Interpolated liquid water content (kg/m³). Returns 0.0 if outside loaded chunks.
pub fn sample_cloud_density(pos: [f32; 3], cloud_field: &CloudField, chunk_size: usize) -> f32 {
    // Convert world position to voxel coordinates (1 voxel = 1 meter)
    let vx = pos[0];
    let vy = pos[1];
    let vz = pos[2];

    // Chunk coordinate
    let chunk_x = (vx.floor() as i32).div_euclid(chunk_size as i32);
    let chunk_y = (vy.floor() as i32).div_euclid(chunk_size as i32);
    let chunk_z = (vz.floor() as i32).div_euclid(chunk_size as i32);
    let chunk_coord = ChunkCoord::new(chunk_x, chunk_y, chunk_z);

    // Local position within chunk
    let local_x = vx - (chunk_x * chunk_size as i32) as f32;
    let local_y = vy - (chunk_y * chunk_size as i32) as f32;
    let local_z = vz - (chunk_z * chunk_size as i32) as f32;

    // Fractional indices for trilinear interpolation
    let fx = local_x.clamp(0.0, chunk_size as f32 - 1.001);
    let fy = local_y.clamp(0.0, chunk_size as f32 - 1.001);
    let fz = local_z.clamp(0.0, chunk_size as f32 - 1.001);

    let ix = fx.floor() as usize;
    let iy = fy.floor() as usize;
    let iz = fz.floor() as usize;

    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let tz = fz - iz as f32;

    // Sample 8 corners of the voxel cube
    let ix1 = (ix + 1).min(chunk_size - 1);
    let iy1 = (iy + 1).min(chunk_size - 1);
    let iz1 = (iz + 1).min(chunk_size - 1);

    let c000 = cloud_field.get_lwc(&chunk_coord, ix, iy, iz, chunk_size);
    let c100 = cloud_field.get_lwc(&chunk_coord, ix1, iy, iz, chunk_size);
    let c010 = cloud_field.get_lwc(&chunk_coord, ix, iy1, iz, chunk_size);
    let c110 = cloud_field.get_lwc(&chunk_coord, ix1, iy1, iz, chunk_size);
    let c001 = cloud_field.get_lwc(&chunk_coord, ix, iy, iz1, chunk_size);
    let c101 = cloud_field.get_lwc(&chunk_coord, ix1, iy, iz1, chunk_size);
    let c011 = cloud_field.get_lwc(&chunk_coord, ix, iy1, iz1, chunk_size);
    let c111 = cloud_field.get_lwc(&chunk_coord, ix1, iy1, iz1, chunk_size);

    // Trilinear interpolation
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;

    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;

    c0 * (1.0 - tz) + c1 * tz
}

/// Henyey-Greenstein phase function for cloud scattering.
///
/// Describes forward-biased scattering from cloud droplets.
///
/// # Arguments
/// * `cos_theta` - Cosine of angle between incident and scattered ray.
/// * `g` - Asymmetry parameter (0.0 = isotropic, 1.0 = pure forward).
///
/// # Returns
/// Phase function value (dimensionless, normalized to integrate to 1 over sphere).
fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    let norm = (1.0 - g2) / (4.0 * PI);
    norm / denom.powf(1.5)
}

/// Ray-march through the cloud volume, accumulating extinction and in-scattered light.
///
/// Uses Beer-Lambert law for transmittance and Henyey-Greenstein phase function
/// for single-scattering approximation.
///
/// # Arguments
/// * `ray_origin` - Ray start position [x, y, z] in meters.
/// * `ray_dir` - Normalized ray direction [dx, dy, dz].
/// * `sun_dir` - Normalized sun direction [dx, dy, dz].
/// * `sun_color` - Sun color (linear RGB, not intensity-scaled).
/// * `cloud_field` - Cloud LWC data from the LBM simulation.
/// * `params` - Rendering parameters.
///
/// # Returns
/// Accumulated cloud color and transmittance.
pub fn march_cloud_ray(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    sun_dir: [f32; 3],
    sun_color: [f32; 3],
    cloud_field: &CloudField,
    params: &CloudRenderParams,
) -> CloudSample {
    let mut transmittance = 1.0;
    let mut color = [0.0, 0.0, 0.0];
    let mut distance = 0.0;

    // Precompute phase function (scattering angle between ray and sun)
    let cos_theta = ray_dir[0] * sun_dir[0] + ray_dir[1] * sun_dir[1] + ray_dir[2] * sun_dir[2];
    let phase = henyey_greenstein_phase(cos_theta, params.forward_scatter_g);

    let num_steps = (params.max_march_distance / params.step_size) as usize;

    for _ in 0..num_steps {
        // Current sample position
        let pos = [
            ray_origin[0] + ray_dir[0] * distance,
            ray_origin[1] + ray_dir[1] * distance,
            ray_origin[2] + ray_dir[2] * distance,
        ];

        // Sample cloud density
        let density = sample_cloud_density(pos, cloud_field, CHUNK_SIZE);

        if density > params.density_threshold {
            // Optical depth increment for this step
            let extinction = params.extinction_coeff * density;
            let delta_tau = extinction * params.step_size;

            // Beer-Lambert transmittance for this segment
            let segment_transmittance = (-delta_tau).exp();

            // In-scattered light: combines phase function, albedo, and sun illumination
            // Silver lining effect: clouds appear bright when looking toward sun
            // Ambient: prevent pure black in shadow regions
            let scatter_strength = params.scattering_albedo * phase;
            let ambient_light = params.ambient_factor;
            let light_contribution = scatter_strength + ambient_light;

            // Accumulate in-scattered radiance
            // Energy deposited at this step, attenuated by path transmittance
            let in_scatter = [
                sun_color[0] * light_contribution * (1.0 - segment_transmittance) * transmittance,
                sun_color[1] * light_contribution * (1.0 - segment_transmittance) * transmittance,
                sun_color[2] * light_contribution * (1.0 - segment_transmittance) * transmittance,
            ];

            color[0] += in_scatter[0];
            color[1] += in_scatter[1];
            color[2] += in_scatter[2];

            // Update total transmittance
            transmittance *= segment_transmittance;

            // Early exit when fully opaque
            if transmittance < 0.01 {
                transmittance = 0.0;
                break;
            }
        }

        distance += params.step_size;
    }

    CloudSample {
        color,
        transmittance,
    }
}

/// Composite cloud over sky color using standard over operation.
///
/// # Arguments
/// * `cloud` - Cloud sample from ray-marching.
/// * `sky_color` - Background sky color (linear RGB).
///
/// # Returns
/// Final composited color (linear RGB).
pub fn composite_cloud_over_sky(cloud: &CloudSample, sky_color: [f32; 3]) -> [f32; 3] {
    [
        cloud.color[0] + cloud.transmittance * sky_color[0],
        cloud.color[1] + cloud.transmittance * sky_color[1],
        cloud.color[2] + cloud.transmittance * sky_color[2],
    ]
}

/// Estimate cloud coverage in a chunk for LOD / distant rendering.
///
/// Returns the fraction of the chunk that contains cloud density above threshold.
///
/// # Arguments
/// * `cloud_field` - Cloud LWC data.
/// * `chunk_coord` - Chunk to analyze.
/// * `chunk_size` - Edge length of chunk in voxels.
///
/// # Returns
/// Coverage fraction (0.0 = empty, 1.0 = fully cloudy).
pub fn estimate_cloud_coverage(
    cloud_field: &CloudField,
    chunk_coord: &ChunkCoord,
    _chunk_size: usize,
) -> f32 {
    let Some(data) = cloud_field.chunks.get(chunk_coord) else {
        return 0.0;
    };

    let threshold = 1e-5; // Match default density_threshold
    let count = data.iter().filter(|&&lwc| lwc > threshold).count();
    count as f32 / data.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_test_cloud_field(lwc_value: f32) -> CloudField {
        let mut chunks = HashMap::new();
        let size = CHUNK_SIZE;
        let volume = size * size * size;
        let data = vec![lwc_value; volume];
        chunks.insert(ChunkCoord::new(0, 0, 0), data);
        CloudField { chunks }
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    #[test]
    fn empty_cloud_field_full_transmittance() {
        let cloud_field = make_test_cloud_field(0.0);
        let params = CloudRenderParams::default();
        let ray_origin = [16.0, 16.0, 16.0];
        let ray_dir = normalize([0.0, 0.0, 1.0]);
        let sun_dir = normalize([0.0, 1.0, 0.0]);
        let sun_color = [1.0, 1.0, 1.0];

        let sample = march_cloud_ray(
            ray_origin,
            ray_dir,
            sun_dir,
            sun_color,
            &cloud_field,
            &params,
        );

        assert!(
            (sample.transmittance - 1.0).abs() < 0.01,
            "Empty cloud should have full transmittance, got {}",
            sample.transmittance
        );
        assert!(
            sample.color[0].abs() < 0.01,
            "Empty cloud should have zero in-scatter"
        );
    }

    #[test]
    fn dense_cloud_low_transmittance() {
        let cloud_field = make_test_cloud_field(1.0); // 1 kg/m³ LWC
        let params = CloudRenderParams {
            max_march_distance: 32.0,
            step_size: 1.0,
            ..Default::default()
        };

        let ray_origin = [16.0, 16.0, 0.0];
        let ray_dir = normalize([0.0, 0.0, 1.0]);
        let sun_dir = normalize([0.0, 1.0, 0.0]);
        let sun_color = [1.0, 1.0, 1.0];

        let sample = march_cloud_ray(
            ray_origin,
            ray_dir,
            sun_dir,
            sun_color,
            &cloud_field,
            &params,
        );

        // With extinction_coeff = 1.5, 32 meters at 1.0 kg/m³:
        // tau = 1.5 * 1.0 * 32 = 48, T = exp(-48) ≈ 0
        assert!(
            sample.transmittance < 0.01,
            "Dense cloud should have low transmittance, got {}",
            sample.transmittance
        );
    }

    #[test]
    fn beer_lambert_consistency() {
        let cloud_field = make_test_cloud_field(0.5); // 0.5 kg/m³
        let mut params = CloudRenderParams {
            step_size: 0.5,
            ..Default::default()
        };

        let ray_origin = [16.0, 16.0, 0.0];
        let ray_dir = normalize([0.0, 0.0, 1.0]);
        let sun_dir = normalize([0.0, 1.0, 0.0]);
        let sun_color = [1.0, 1.0, 1.0];

        // March 10 meters
        params.max_march_distance = 10.0;
        let sample1 = march_cloud_ray(
            ray_origin,
            ray_dir,
            sun_dir,
            sun_color,
            &cloud_field,
            &params,
        );

        // March 20 meters
        params.max_march_distance = 20.0;
        let sample2 = march_cloud_ray(
            ray_origin,
            ray_dir,
            sun_dir,
            sun_color,
            &cloud_field,
            &params,
        );

        // Optical depth doubles: τ₁ = 0.5 * 1.5 * 10 = 7.5, τ₂ = 15
        // T₂ = exp(-15) ≈ exp(-7.5)² = T₁²
        let expected_t2 = sample1.transmittance * sample1.transmittance;
        assert!(
            (sample2.transmittance - expected_t2).abs() < 0.1,
            "Doubling path length should square transmittance: T1={}, T1²={}, T2={}",
            sample1.transmittance,
            expected_t2,
            sample2.transmittance
        );
    }

    #[test]
    fn forward_scatter_brighter_than_back() {
        let cloud_field = make_test_cloud_field(0.1);
        let params = CloudRenderParams {
            max_march_distance: 10.0,
            step_size: 1.0,
            ..Default::default()
        };

        let ray_origin = [16.0, 16.0, 16.0];
        let sun_dir = normalize([0.0, 0.0, 1.0]);
        let sun_color = [1.0, 1.0, 1.0];

        // Forward scatter: looking toward sun
        let ray_forward = normalize([0.0, 0.0, 1.0]);
        let sample_forward = march_cloud_ray(
            ray_origin,
            ray_forward,
            sun_dir,
            sun_color,
            &cloud_field,
            &params,
        );

        // Back scatter: looking away from sun
        let ray_back = normalize([0.0, 0.0, -1.0]);
        let sample_back = march_cloud_ray(
            ray_origin,
            ray_back,
            sun_dir,
            sun_color,
            &cloud_field,
            &params,
        );

        let brightness_forward =
            sample_forward.color[0] + sample_forward.color[1] + sample_forward.color[2];
        let brightness_back = sample_back.color[0] + sample_back.color[1] + sample_back.color[2];

        assert!(
            brightness_forward > brightness_back,
            "Forward scatter should be brighter than back scatter: fwd={}, back={}",
            brightness_forward,
            brightness_back
        );
    }

    #[test]
    fn cloud_sky_compositing_opaque() {
        let cloud = CloudSample {
            color: [0.8, 0.8, 0.8],
            transmittance: 0.0,
        };
        let sky_color = [0.3, 0.5, 0.9];

        let result = composite_cloud_over_sky(&cloud, sky_color);

        // Opaque cloud: should be pure cloud color
        assert!((result[0] - 0.8).abs() < 0.01);
        assert!((result[1] - 0.8).abs() < 0.01);
        assert!((result[2] - 0.8).abs() < 0.01);
    }

    #[test]
    fn cloud_sky_compositing_transparent() {
        let cloud = CloudSample {
            color: [0.0, 0.0, 0.0],
            transmittance: 1.0,
        };
        let sky_color = [0.3, 0.5, 0.9];

        let result = composite_cloud_over_sky(&cloud, sky_color);

        // Fully transparent cloud: should be pure sky color
        assert!((result[0] - 0.3).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
        assert!((result[2] - 0.9).abs() < 0.01);
    }

    #[test]
    fn density_sampling_correct_lwc() {
        let cloud_field = make_test_cloud_field(0.75);

        let pos = [16.5, 16.5, 16.5]; // Center of chunk
        let density = sample_cloud_density(pos, &cloud_field, CHUNK_SIZE);

        assert!(
            (density - 0.75).abs() < 0.01,
            "Should sample correct LWC, got {}",
            density
        );
    }

    #[test]
    fn cloud_coverage_empty() {
        let cloud_field = make_test_cloud_field(0.0);
        let coverage = estimate_cloud_coverage(&cloud_field, &ChunkCoord::new(0, 0, 0), CHUNK_SIZE);

        assert!(
            coverage.abs() < 0.01,
            "Empty field should have zero coverage, got {}",
            coverage
        );
    }

    #[test]
    fn cloud_coverage_full() {
        let cloud_field = make_test_cloud_field(1.0);
        let coverage = estimate_cloud_coverage(&cloud_field, &ChunkCoord::new(0, 0, 0), CHUNK_SIZE);

        assert!(
            (coverage - 1.0).abs() < 0.01,
            "Full field should have 100% coverage, got {}",
            coverage
        );
    }

    #[test]
    fn early_termination_dense_cloud() {
        let cloud_field = make_test_cloud_field(2.0); // Very dense
        let params = CloudRenderParams {
            max_march_distance: 1000.0,
            step_size: 1.0,
            ..Default::default()
        };

        let ray_origin = [16.0, 16.0, 0.0];
        let ray_dir = normalize([0.0, 0.0, 1.0]);
        let sun_dir = normalize([0.0, 1.0, 0.0]);
        let sun_color = [1.0, 1.0, 1.0];

        let sample = march_cloud_ray(
            ray_origin,
            ray_dir,
            sun_dir,
            sun_color,
            &cloud_field,
            &params,
        );

        // Should terminate early (T < 0.01) before reaching max distance
        assert!(
            sample.transmittance < 0.01,
            "Should terminate early with near-zero transmittance"
        );
    }

    #[test]
    fn henyey_greenstein_forward_bias() {
        let g = 0.85;
        let forward = henyey_greenstein_phase(1.0, g); // cos(0°) = 1
        let backward = henyey_greenstein_phase(-1.0, g); // cos(180°) = -1

        assert!(
            forward > backward,
            "Forward direction should have higher phase value: fwd={}, back={}",
            forward,
            backward
        );
    }

    #[test]
    fn henyey_greenstein_isotropic_when_g_zero() {
        let g = 0.0;
        let forward = henyey_greenstein_phase(1.0, g);
        let backward = henyey_greenstein_phase(-1.0, g);
        let side = henyey_greenstein_phase(0.0, g);

        assert!(
            (forward - backward).abs() < 0.01,
            "Isotropic scattering should be direction-independent"
        );
        assert!(
            (forward - side).abs() < 0.01,
            "Isotropic scattering should be direction-independent"
        );
    }
}
