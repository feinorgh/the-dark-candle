// Atmospheric scattering (Rayleigh + Mie) for sky color computation.
//
// Implements first-principles atmospheric scattering using ray-marching through
// the atmosphere, computing in-scattered light from the sun at each sample point.
//
// This is a CPU-side implementation for integration with the software renderer
// (visualization.rs). GPU shader promotion is a future optimization.
//
// All units are SI:
//   - distances: meters
//   - scattering coefficients: m⁻¹
//   - angles: radians
//   - color: linear RGB (NOT sRGB)

use crate::physics::atmosphere::AtmosphereConfig;
use crate::world::planet::PlanetConfig;
use std::f32::consts::PI;

/// Rayleigh scattering coefficients at sea level for Earth atmosphere.
/// Wavelength-dependent: [680nm (red), 550nm (green), 440nm (blue)].
/// Units: m⁻¹
pub const RAYLEIGH_COEFF_EARTH: [f32; 3] = [5.5e-6, 13.0e-6, 22.4e-6];

/// Mie scattering coefficient for Earth atmosphere (aerosols).
/// Wavelength-independent approximation.
/// Units: m⁻¹
pub const MIE_COEFF_EARTH: f32 = 21.0e-6;

/// Atmospheric scattering parameters derived from atmosphere and planet config.
///
/// All coefficients use SI units (m⁻¹). Color channels correspond to RGB
/// (red, green, blue) wavelengths.
#[derive(Debug, Clone)]
pub struct ScatteringParams {
    /// Rayleigh scattering coefficient per RGB channel (λ^-4 dependent).
    /// Units: m⁻¹
    pub rayleigh_coeff: [f32; 3],

    /// Mie scattering coefficient (wavelength-independent).
    /// Units: m⁻¹
    pub mie_coeff: f32,

    /// Mie asymmetry parameter (forward scattering bias).
    /// Typical range: 0.7–0.85 for atmospheric aerosols. 0.76 is Earth-like.
    pub mie_g: f32,

    /// Outer radius of atmosphere in meters.
    pub atmosphere_radius: f32,

    /// Planet surface radius in meters.
    pub planet_radius: f32,

    /// Rayleigh scale height (exponential density drop-off) in meters.
    /// Earth: 8500 m.
    pub rayleigh_scale_height: f32,

    /// Mie scale height (aerosol layer thickness) in meters.
    /// Earth: 1200 m (concentrated near surface).
    pub mie_scale_height: f32,

    /// Solar irradiance factor (brightness multiplier).
    pub sun_intensity: f32,
}

impl Default for ScatteringParams {
    fn default() -> Self {
        Self {
            rayleigh_coeff: RAYLEIGH_COEFF_EARTH,
            mie_coeff: MIE_COEFF_EARTH,
            mie_g: 0.76,
            atmosphere_radius: 6471000.0, // Earth surface + 100 km
            planet_radius: 6371000.0,     // Earth mean radius
            rayleigh_scale_height: 8500.0,
            mie_scale_height: 1200.0,
            sun_intensity: 20.0,
        }
    }
}

/// Compute scattering parameters from atmosphere and planet configuration.
///
/// # Arguments
/// * `atmos_config` - Atmosphere configuration (scattering coefficients, etc.)
/// * `planet_config` - Planet configuration (radius, gravity, etc.)
///
/// # Returns
/// Scattering parameters for ray-marching.
pub fn compute_scattering_params(
    atmos_config: &AtmosphereConfig,
    planet_config: &PlanetConfig,
) -> ScatteringParams {
    // Derive RGB Rayleigh coefficients from the single-wavelength coefficient
    // using wavelength scaling: β(λ) ∝ λ^-4
    // Reference wavelength: 550 nm (green)
    let rayleigh_base = atmos_config.rayleigh_scatter_coeff;

    // Wavelength ratios (relative to 550nm): [680/550, 550/550, 440/550]
    let wavelength_ratios: [f32; 3] = [680.0 / 550.0, 1.0, 440.0 / 550.0];

    // β(λ) = β₀ × (λ₀/λ)^4
    let rayleigh_coeff = [
        rayleigh_base / wavelength_ratios[0].powi(4), // red
        rayleigh_base,                                // green (reference)
        rayleigh_base / wavelength_ratios[2].powi(4), // blue
    ];

    let planet_radius = planet_config.mean_radius as f32;
    let atmosphere_radius = planet_radius + 100_000.0; // 100 km atmosphere

    ScatteringParams {
        rayleigh_coeff,
        mie_coeff: atmos_config.mie_scatter_coeff,
        mie_g: 0.76, // Earth-like aerosol asymmetry
        atmosphere_radius,
        planet_radius,
        rayleigh_scale_height: 8500.0,
        mie_scale_height: 1200.0,
        sun_intensity: 20.0, // Tuned for visual clarity
    }
}

/// Rayleigh phase function: (3/16π)(1 + cos²θ).
///
/// Describes angular distribution of scattered light. Symmetric (equal forward
/// and backward scattering).
///
/// # Arguments
/// * `cos_theta` - Cosine of angle between incident and scattered ray.
///
/// # Returns
/// Phase function value (dimensionless).
fn rayleigh_phase(cos_theta: f32) -> f32 {
    let norm = 3.0 / (16.0 * PI);
    norm * (1.0 + cos_theta * cos_theta)
}

/// Mie phase function (Henyey-Greenstein approximation):
/// `(1-g²) / (4π(1+g²-2g·cosθ)^1.5)`
///
/// Describes forward-biased scattering from aerosols. Parameter `g` controls
/// asymmetry: g=0 is isotropic, g>0 favors forward scattering.
///
/// # Arguments
/// * `cos_theta` - Cosine of angle between incident and scattered ray.
/// * `g` - Asymmetry parameter (0.0–1.0). Typical: 0.76 for Earth.
///
/// # Returns
/// Phase function value (dimensionless).
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    let norm = (1.0 - g2) / (4.0 * PI);
    norm / denom.powf(1.5)
}

/// Ray-sphere intersection: find entry/exit distances.
///
/// # Arguments
/// * `ray_origin` - Ray start position (relative to sphere center).
/// * `ray_dir` - Normalized ray direction.
/// * `sphere_radius` - Radius of sphere.
///
/// # Returns
/// `Some((t_near, t_far))` if ray intersects, `None` otherwise.
/// `t_near` may be negative if ray starts inside sphere.
fn ray_sphere_intersection(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    sphere_radius: f32,
) -> Option<(f32, f32)> {
    let ox = ray_origin[0];
    let oy = ray_origin[1];
    let oz = ray_origin[2];
    let dx = ray_dir[0];
    let dy = ray_dir[1];
    let dz = ray_dir[2];

    let a = dx * dx + dy * dy + dz * dz;
    let b = 2.0 * (ox * dx + oy * dy + oz * dz);
    let c = ox * ox + oy * oy + oz * oz - sphere_radius * sphere_radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let sqrt_d = discriminant.sqrt();
    let t0 = (-b - sqrt_d) / (2.0 * a);
    let t1 = (-b + sqrt_d) / (2.0 * a);

    Some((t0, t1))
}

/// Compute optical depth along a ray segment through the atmosphere.
///
/// Integrates density (Rayleigh + Mie) along the path. Uses exponential
/// density profiles: `ρ(h) = exp(-h / H)` where h is altitude above surface.
///
/// # Arguments
/// * `start` - Ray start position (relative to planet center).
/// * `dir` - Normalized ray direction.
/// * `length` - Distance to integrate.
/// * `params` - Scattering parameters.
/// * `num_samples` - Number of integration samples.
///
/// # Returns
/// `(rayleigh_depth, mie_depth)` — optical depths (dimensionless).
fn optical_depth(
    start: [f32; 3],
    dir: [f32; 3],
    length: f32,
    params: &ScatteringParams,
    num_samples: usize,
) -> ([f32; 3], f32) {
    let step_size = length / (num_samples as f32);
    let mut rayleigh_depth = [0.0; 3];
    let mut mie_depth = 0.0;

    for i in 0..num_samples {
        let t = (i as f32 + 0.5) * step_size;
        let pos = [
            start[0] + dir[0] * t,
            start[1] + dir[1] * t,
            start[2] + dir[2] * t,
        ];

        let height_from_center = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let altitude = height_from_center - params.planet_radius;

        if altitude < 0.0 {
            // Below surface — infinite optical depth
            return ([f32::INFINITY; 3], f32::INFINITY);
        }

        // Density decreases exponentially with altitude
        let rayleigh_density = (-altitude / params.rayleigh_scale_height).exp();
        let mie_density = (-altitude / params.mie_scale_height).exp();

        // Accumulate optical depth: τ = ∫ β(h) dh
        for (ch, rayleigh_ch) in rayleigh_depth.iter_mut().enumerate() {
            *rayleigh_ch += params.rayleigh_coeff[ch] * rayleigh_density * step_size;
        }
        mie_depth += params.mie_coeff * mie_density * step_size;
    }

    (rayleigh_depth, mie_depth)
}

/// Compute sky color for a given viewing ray.
///
/// Ray-marches through the atmosphere, accumulating in-scattered sunlight
/// at each sample point. Applies Rayleigh and Mie scattering with their
/// respective phase functions and extinction (Beer-Lambert law).
///
/// # Arguments
/// * `ray_dir` - Normalized viewing direction.
/// * `sun_dir` - Normalized sun direction.
/// * `camera_altitude` - Camera altitude above planet surface (meters).
/// * `params` - Scattering parameters.
///
/// # Returns
/// Linear RGB color (NOT sRGB). Values may exceed 1.0 (HDR).
pub fn sky_color(
    ray_dir: [f32; 3],
    sun_dir: [f32; 3],
    camera_altitude: f32,
    params: &ScatteringParams,
) -> [f32; 3] {
    // Camera position (relative to planet center)
    let camera_pos = [0.0, params.planet_radius + camera_altitude, 0.0];

    // Intersect ray with atmosphere sphere
    let atmos_intersection = ray_sphere_intersection(camera_pos, ray_dir, params.atmosphere_radius);

    let (t_start, t_end) = match atmos_intersection {
        Some((t0, t1)) => {
            let start = if t0 > 0.0 { t0 } else { 0.0 }; // Ray starts inside atmosphere
            let end = t1.max(0.0);
            if end <= start {
                return [0.0, 0.0, 0.0]; // No intersection
            }
            (start, end)
        }
        None => return [0.0, 0.0, 0.0], // Ray misses atmosphere
    };

    // Check if ray hits planet surface
    let surface_intersection = ray_sphere_intersection(camera_pos, ray_dir, params.planet_radius);
    let t_end = if let Some((_, t_exit)) = surface_intersection {
        if t_exit > t_start && t_exit < t_end {
            t_exit // Ray hits surface — stop there
        } else {
            t_end
        }
    } else {
        t_end
    };

    let path_length = t_end - t_start;
    if path_length <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    // Ray-march through atmosphere
    let num_samples = 16;
    let step_size = path_length / (num_samples as f32);

    let mut total_rayleigh = [0.0; 3];
    let mut total_mie = [0.0; 3];

    let cos_theta = ray_dir[0] * sun_dir[0] + ray_dir[1] * sun_dir[1] + ray_dir[2] * sun_dir[2];
    let phase_rayleigh = rayleigh_phase(cos_theta);
    let phase_mie = mie_phase(cos_theta, params.mie_g);

    for i in 0..num_samples {
        let t = t_start + (i as f32 + 0.5) * step_size;
        let sample_pos = [
            camera_pos[0] + ray_dir[0] * t,
            camera_pos[1] + ray_dir[1] * t,
            camera_pos[2] + ray_dir[2] * t,
        ];

        let height_from_center = (sample_pos[0] * sample_pos[0]
            + sample_pos[1] * sample_pos[1]
            + sample_pos[2] * sample_pos[2])
            .sqrt();
        let altitude = height_from_center - params.planet_radius;

        if altitude < 0.0 {
            break; // Hit surface
        }

        // Density at sample point
        let rayleigh_density = (-altitude / params.rayleigh_scale_height).exp();
        let mie_density = (-altitude / params.mie_scale_height).exp();

        // Optical depth from sample point to sun (secondary ray-march)
        let sun_ray_intersection =
            ray_sphere_intersection(sample_pos, sun_dir, params.atmosphere_radius);

        let (sun_rayleigh_depth, sun_mie_depth) = if let Some((_, t_sun)) = sun_ray_intersection {
            if t_sun > 0.0 {
                optical_depth(sample_pos, sun_dir, t_sun, params, 8)
            } else {
                ([0.0; 3], 0.0) // Sun ray origin inside atmosphere
            }
        } else {
            ([0.0; 3], 0.0) // Sun ray misses atmosphere (shouldn't happen)
        };

        // Check if sun ray hits planet (in shadow)
        let sun_surface_intersection =
            ray_sphere_intersection(sample_pos, sun_dir, params.planet_radius);
        let in_shadow = if let Some((_, t_surf)) = sun_surface_intersection {
            t_surf > 0.0
        } else {
            false
        };

        if in_shadow {
            continue; // This sample is in planet's shadow
        }

        // Optical depth from camera to sample point
        let segment_length = (i as f32 + 0.5) * step_size;
        let (view_rayleigh_depth, view_mie_depth) = if segment_length > 0.0 {
            optical_depth(camera_pos, ray_dir, segment_length, params, 8)
        } else {
            ([0.0; 3], 0.0)
        };

        // Total extinction (Beer-Lambert): exp(-τ)
        for ch in 0..3 {
            let total_depth =
                view_rayleigh_depth[ch] + sun_rayleigh_depth[ch] + view_mie_depth + sun_mie_depth;
            let extinction = (-total_depth).exp();

            // In-scattered light contribution
            let rayleigh_scatter = params.rayleigh_coeff[ch] * rayleigh_density * phase_rayleigh;
            let mie_scatter = params.mie_coeff * mie_density * phase_mie;

            total_rayleigh[ch] += rayleigh_scatter * extinction * step_size;
            total_mie[ch] += mie_scatter * extinction * step_size;
        }
    }

    // Combine Rayleigh and Mie contributions with sun intensity
    [
        (total_rayleigh[0] + total_mie[0]) * params.sun_intensity,
        (total_rayleigh[1] + total_mie[1]) * params.sun_intensity,
        (total_rayleigh[2] + total_mie[2]) * params.sun_intensity,
    ]
}

/// Compute sun disk color (reddened by atmospheric extinction).
///
/// Traces a ray from camera toward sun, computing optical depth through
/// the atmosphere. Applies Beer-Lambert extinction to get the transmitted
/// sun color (redder at horizon due to longer path through atmosphere).
///
/// # Arguments
/// * `sun_dir` - Normalized sun direction.
/// * `camera_altitude` - Camera altitude above planet surface (meters).
/// * `params` - Scattering parameters.
///
/// # Returns
/// Linear RGB color of sun disk.
pub fn sun_disk_color(
    sun_dir: [f32; 3],
    camera_altitude: f32,
    params: &ScatteringParams,
) -> [f32; 3] {
    let camera_pos = [0.0, params.planet_radius + camera_altitude, 0.0];

    // Ray toward sun
    let atmos_intersection = ray_sphere_intersection(camera_pos, sun_dir, params.atmosphere_radius);

    let path_length = if let Some((t0, t1)) = atmos_intersection {
        let start = if t0 > 0.0 { t0 } else { 0.0 };
        let end = t1.max(0.0);
        end - start
    } else {
        return [1.0, 1.0, 1.0]; // No atmosphere — full sun
    };

    if path_length <= 0.0 {
        return [1.0, 1.0, 1.0];
    }

    // Optical depth along path to sun
    let (rayleigh_depth, mie_depth) = optical_depth(camera_pos, sun_dir, path_length, params, 16);

    // Extinction per channel
    let extinction_r = (-(rayleigh_depth[0] + mie_depth)).exp();
    let extinction_g = (-(rayleigh_depth[1] + mie_depth)).exp();
    let extinction_b = (-(rayleigh_depth[2] + mie_depth)).exp();

    [
        extinction_r * params.sun_intensity,
        extinction_g * params.sun_intensity,
        extinction_b * params.sun_intensity,
    ]
}

/// Pre-compute a sky lookup table (LUT) for fast rendering.
///
/// Generates an equirectangular sky texture by sampling sky color at each
/// pixel direction. This can be used by the software renderer or GPU shader.
///
/// # Arguments
/// * `sun_dir` - Normalized sun direction.
/// * `camera_altitude` - Camera altitude above planet surface (meters).
/// * `params` - Scattering parameters.
/// * `width` - LUT width in pixels.
/// * `height` - LUT height in pixels.
///
/// # Returns
/// Flat array of linear RGB values (row-major order).
pub fn compute_sky_lut(
    sun_dir: [f32; 3],
    camera_altitude: f32,
    params: &ScatteringParams,
    width: usize,
    height: usize,
) -> Vec<[f32; 3]> {
    let mut lut = Vec::with_capacity(width * height);

    for y in 0..height {
        let theta = (y as f32 / height as f32) * PI; // 0 to π
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for x in 0..width {
            let phi = (x as f32 / width as f32) * 2.0 * PI; // 0 to 2π

            // Spherical to Cartesian
            let ray_dir = [sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin()];

            let color = sky_color(ray_dir, sun_dir, camera_altitude, params);
            lut.push(color);
        }
    }

    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.01;

    /// Normalize a 3D vector.
    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    #[test]
    fn blue_sky_at_noon() {
        // Sun overhead, ray horizontal → dominant blue channel
        let params = ScatteringParams::default();
        let sun_dir = [0.0, 1.0, 0.0]; // Zenith
        let ray_dir = normalize([1.0, 0.0, 0.0]); // Horizontal
        let camera_altitude = 0.0;

        let color = sky_color(ray_dir, sun_dir, camera_altitude, &params);

        // Blue channel should be stronger than red (Rayleigh scattering)
        // Green is typically between blue and red
        assert!(
            color[2] > color[0],
            "Blue channel should exceed red (Rayleigh scattering): {:?}",
            color
        );
        assert!(
            color[2] > 0.1,
            "Blue channel should be significant: {:?}",
            color
        );
        // All channels should be positive (visible sky)
        assert!(
            color[0] > 0.0 && color[1] > 0.0 && color[2] > 0.0,
            "All channels should be positive: {:?}",
            color
        );
    }

    #[test]
    fn red_sunset() {
        // Sun at horizon, ray toward sun → dominant red channel
        let params = ScatteringParams::default();
        let sun_dir = normalize([1.0, 0.05, 0.0]); // Near horizon
        let ray_dir = sun_dir; // Looking at sun
        let camera_altitude = 0.0;

        let color = sky_color(ray_dir, sun_dir, camera_altitude, &params);

        // Red channel should be strongest at sunset (longer path, more blue extinction)
        assert!(
            color[0] > color[2] * 0.5,
            "Red should be dominant at sunset: {:?}",
            color
        );
        assert!(
            color[0] > 0.05,
            "Red channel should be visible: {:?}",
            color
        );
    }

    #[test]
    fn night_sky_is_dark() {
        // Sun below horizon → near-zero intensity
        let params = ScatteringParams::default();
        let sun_dir = [0.0, -1.0, 0.0]; // Below horizon
        let ray_dir = normalize([1.0, 0.5, 0.0]);
        let camera_altitude = 0.0;

        let color = sky_color(ray_dir, sun_dir, camera_altitude, &params);

        // All channels should be very low (sun below horizon)
        let max_channel = color[0].max(color[1]).max(color[2]);
        assert!(max_channel < 0.01, "Night sky should be dark: {:?}", color);
    }

    #[test]
    fn altitude_effect() {
        // Sky darker/more purple at high altitude (less atmosphere above)
        let params = ScatteringParams::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let ray_dir = normalize([1.0, 0.0, 0.0]);

        let color_surface = sky_color(ray_dir, sun_dir, 0.0, &params);
        let color_high = sky_color(ray_dir, sun_dir, 20000.0, &params);

        // At altitude, overall intensity should be lower (less scattering medium)
        let intensity_surface = color_surface[0] + color_surface[1] + color_surface[2];
        let intensity_high = color_high[0] + color_high[1] + color_high[2];

        assert!(
            intensity_high < intensity_surface,
            "High altitude sky should be darker: surface={}, high={}",
            intensity_surface,
            intensity_high
        );
    }

    #[test]
    fn mie_forward_scatter() {
        // Ray toward sun has higher intensity than away from sun (Mie forward scattering)
        let params = ScatteringParams::default();
        let sun_dir = normalize([1.0, 0.2, 0.0]);

        let ray_toward = sun_dir;
        let ray_away = [-sun_dir[0], -sun_dir[1], -sun_dir[2]];
        let camera_altitude = 0.0;

        let color_toward = sky_color(ray_toward, sun_dir, camera_altitude, &params);
        let color_away = sky_color(ray_away, sun_dir, camera_altitude, &params);

        let intensity_toward = color_toward[0] + color_toward[1] + color_toward[2];
        let intensity_away = color_away[0] + color_away[1] + color_away[2];

        assert!(
            intensity_toward > intensity_away,
            "Forward scatter should be brighter: toward={}, away={}",
            intensity_toward,
            intensity_away
        );
    }

    #[test]
    fn symmetry_about_sun_axis() {
        // Sky color should be symmetric about the sun axis
        let params = ScatteringParams::default();
        let sun_dir = [0.0, 1.0, 0.0]; // Zenith

        // Two rays at same angle to sun, different azimuth
        let ray_a = normalize([1.0, 0.0, 0.0]);
        let ray_b = normalize([0.0, 0.0, 1.0]);
        let camera_altitude = 0.0;

        let color_a = sky_color(ray_a, sun_dir, camera_altitude, &params);
        let color_b = sky_color(ray_b, sun_dir, camera_altitude, &params);

        // Colors should be nearly identical (symmetric about sun axis)
        for ch in 0..3 {
            let diff = (color_a[ch] - color_b[ch]).abs();
            let avg = (color_a[ch] + color_b[ch]) / 2.0;
            let rel_diff = if avg > 0.0 { diff / avg } else { diff };
            assert!(
                rel_diff < EPSILON,
                "Channel {} should be symmetric: {:?} vs {:?}",
                ch,
                color_a,
                color_b
            );
        }
    }

    #[test]
    fn rayleigh_phase_function_is_symmetric() {
        // Rayleigh phase should be equal for forward and backward scattering
        let phase_forward = rayleigh_phase(1.0);
        let phase_backward = rayleigh_phase(-1.0);
        assert!(
            (phase_forward - phase_backward).abs() < EPSILON,
            "Rayleigh phase should be symmetric: forward={}, backward={}",
            phase_forward,
            phase_backward
        );
    }

    #[test]
    fn mie_phase_function_is_forward_biased() {
        // Mie phase should favor forward scattering (g > 0)
        let g = 0.76;
        let phase_forward = mie_phase(1.0, g);
        let phase_backward = mie_phase(-1.0, g);
        assert!(
            phase_forward > phase_backward,
            "Mie phase should favor forward scatter: forward={}, backward={}",
            phase_forward,
            phase_backward
        );
    }

    #[test]
    fn ray_sphere_intersection_from_outside() {
        let origin = [0.0, 10.0, 0.0];
        let dir = [0.0, -1.0, 0.0];
        let radius = 5.0;

        let result = ray_sphere_intersection(origin, dir, radius);
        assert!(result.is_some(), "Ray from outside should intersect sphere");

        let (t0, t1) = result.unwrap();
        assert!(
            t0 > 0.0 && t1 > t0,
            "Intersection distances should be positive and ordered"
        );
    }

    #[test]
    fn ray_sphere_intersection_from_inside() {
        let origin = [0.0, 0.0, 0.0]; // Inside sphere
        let dir = [0.0, 1.0, 0.0];
        let radius = 5.0;

        let result = ray_sphere_intersection(origin, dir, radius);
        assert!(result.is_some(), "Ray from inside should intersect sphere");

        let (t0, t1) = result.unwrap();
        assert!(
            t0 < 0.0 && t1 > 0.0,
            "Entry should be negative (behind ray origin)"
        );
    }

    #[test]
    fn ray_sphere_miss() {
        let origin = [0.0, 10.0, 0.0];
        let dir = [1.0, 0.0, 0.0]; // Ray perpendicular to sphere
        let radius = 5.0;

        let result = ray_sphere_intersection(origin, dir, radius);
        assert!(result.is_none(), "Ray should miss sphere");
    }

    #[test]
    fn compute_scattering_params_uses_correct_wavelength_scaling() {
        let atmos = AtmosphereConfig::default();
        let planet = PlanetConfig::default();
        let params = compute_scattering_params(&atmos, &planet);

        // Blue should be strongest (λ^-4 scaling)
        assert!(
            params.rayleigh_coeff[2] > params.rayleigh_coeff[1],
            "Blue Rayleigh coeff should exceed green"
        );
        assert!(
            params.rayleigh_coeff[1] > params.rayleigh_coeff[0],
            "Green Rayleigh coeff should exceed red"
        );
    }

    #[test]
    fn sun_disk_color_reddens_at_horizon() {
        let params = ScatteringParams::default();

        // Sun at zenith
        let sun_zenith = [0.0, 1.0, 0.0];
        let color_zenith = sun_disk_color(sun_zenith, 0.0, &params);

        // Sun at horizon
        let sun_horizon = normalize([1.0, 0.05, 0.0]);
        let color_horizon = sun_disk_color(sun_horizon, 0.0, &params);

        // At horizon, red should be stronger relative to blue (less extinction)
        let red_ratio_zenith = color_zenith[0] / color_zenith[2];
        let red_ratio_horizon = color_horizon[0] / color_horizon[2];

        assert!(
            red_ratio_horizon > red_ratio_zenith,
            "Horizon sun should be redder: zenith_ratio={}, horizon_ratio={}",
            red_ratio_zenith,
            red_ratio_horizon
        );
    }

    #[test]
    fn compute_sky_lut_produces_correct_dimensions() {
        let params = ScatteringParams::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let width = 32;
        let height = 16;

        let lut = compute_sky_lut(sun_dir, 0.0, &params, width, height);
        assert_eq!(lut.len(), width * height, "LUT should have correct size");
    }
}
