// Physically-based sky color via Rayleigh scattering.
//
// Models wavelength-dependent scattering (∝ 1/λ⁴) to produce blue skies at
// zenith and orange/red sunsets near the horizon. The optical depth integral
// through the atmosphere determines how much short-wavelength (blue) light is
// scattered versus long-wavelength (red) light that passes through.
//
// Reference: Nishita et al. "Display of The Earth Taking into Account
//            Atmospheric Scattering" (1993).

use std::f32::consts::PI;

/// Rayleigh scattering coefficients at sea level for RGB wavelengths (1/m).
/// λ_R = 680 nm, λ_G = 550 nm, λ_B = 440 nm.
/// β_R(λ) = 8π³(n²−1)² / (3Nλ⁴) where n=1.0003, N=2.545e25 m⁻³
const BETA_R: [f32; 3] = [5.8e-6, 13.5e-6, 33.1e-6];

/// Scale height for Rayleigh scattering (meters).
const H_RAYLEIGH: f32 = 8500.0;

/// Earth radius (meters) — used for atmosphere geometry.
const R_EARTH: f32 = 6_371_000.0;

/// Atmosphere thickness (meters).
const R_ATMOS: f32 = 100_000.0;

/// Number of sample points along the view ray.
const VIEW_SAMPLES: usize = 16;

/// Number of sample points along the light ray (for optical depth to sun).
const LIGHT_SAMPLES: usize = 8;

/// Rayleigh phase function: P(θ) = 3/(16π) × (1 + cos²θ).
fn rayleigh_phase(cos_theta: f32) -> f32 {
    3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta)
}

/// Compute the sky color for a given view direction and sun direction.
///
/// Both `view_dir` and `sun_dir` must be unit vectors. `sun_dir` points
/// **toward** the sun (not the light-travel direction).
///
/// Returns `[R, G, B]` in linear space, values typically in 0.0–2.0+ range
/// (HDR). Caller should tone-map and gamma-correct as needed.
pub fn sky_color(view_dir: [f32; 3], sun_dir: [f32; 3]) -> [f32; 3] {
    // Observer at sea level, looking along view_dir.
    let origin = [0.0_f32, R_EARTH, 0.0];

    // Intersect the view ray with the atmosphere sphere.
    let atmos_dist = ray_sphere_intersect(origin, view_dir, R_EARTH + R_ATMOS);
    let atmos_dist = match atmos_dist {
        Some(d) => d,
        None => return [0.0; 3], // No intersection (shouldn't happen from ground)
    };

    let segment_len = atmos_dist / VIEW_SAMPLES as f32;
    let cos_theta = dot(view_dir, sun_dir);
    let phase = rayleigh_phase(cos_theta);

    let mut total_r = 0.0_f32;
    let mut total_g = 0.0_f32;
    let mut total_b = 0.0_f32;

    // Accumulated optical depth from observer to current sample.
    let mut od_view = [0.0_f32; 3];

    for i in 0..VIEW_SAMPLES {
        let t = (i as f32 + 0.5) * segment_len;
        let sample = [
            origin[0] + view_dir[0] * t,
            origin[1] + view_dir[1] * t,
            origin[2] + view_dir[2] * t,
        ];

        let height = length(sample) - R_EARTH;
        let density = (-height / H_RAYLEIGH).exp();

        // Optical depth for this view segment.
        let od_seg = [
            BETA_R[0] * density * segment_len,
            BETA_R[1] * density * segment_len,
            BETA_R[2] * density * segment_len,
        ];

        od_view[0] += od_seg[0];
        od_view[1] += od_seg[1];
        od_view[2] += od_seg[2];

        // Optical depth from sample point to the sun (through the atmosphere).
        let od_sun = optical_depth_to_sun(sample, sun_dir);

        // Total optical depth = view path + sun path.
        let od_total = [
            od_view[0] + od_sun[0],
            od_view[1] + od_sun[1],
            od_view[2] + od_sun[2],
        ];

        // Transmittance = exp(-optical_depth).
        let attenuation = [
            (-od_total[0]).exp(),
            (-od_total[1]).exp(),
            (-od_total[2]).exp(),
        ];

        // In-scattered light contribution from this segment.
        total_r += density * segment_len * BETA_R[0] * attenuation[0];
        total_g += density * segment_len * BETA_R[1] * attenuation[1];
        total_b += density * segment_len * BETA_R[2] * attenuation[2];
    }

    // Solar intensity factor (arbitrary scale for pleasant brightness).
    let sun_intensity = 20.0;

    [
        total_r * phase * sun_intensity,
        total_g * phase * sun_intensity,
        total_b * phase * sun_intensity,
    ]
}

/// Compute the optical depth from a point in the atmosphere to the sun.
fn optical_depth_to_sun(point: [f32; 3], sun_dir: [f32; 3]) -> [f32; 3] {
    let sun_dist = ray_sphere_intersect(point, sun_dir, R_EARTH + R_ATMOS);
    let sun_dist = match sun_dist {
        Some(d) => d,
        None => return [1e10; 3], // Sun below horizon from this point
    };

    let seg_len = sun_dist / LIGHT_SAMPLES as f32;
    let mut od = [0.0_f32; 3];

    for j in 0..LIGHT_SAMPLES {
        let t = (j as f32 + 0.5) * seg_len;
        let sample = [
            point[0] + sun_dir[0] * t,
            point[1] + sun_dir[1] * t,
            point[2] + sun_dir[2] * t,
        ];

        let height = length(sample) - R_EARTH;
        if height < 0.0 {
            // Ray hit the ground — sun is occluded
            return [1e10; 3];
        }
        let density = (-height / H_RAYLEIGH).exp();

        od[0] += BETA_R[0] * density * seg_len;
        od[1] += BETA_R[1] * density * seg_len;
        od[2] += BETA_R[2] * density * seg_len;
    }

    od
}

/// Intersect a ray with a sphere centered at the origin.
/// Returns the distance to the **far** intersection (entry→exit for rays
/// starting inside, or first hit for rays outside).
fn ray_sphere_intersect(origin: [f32; 3], dir: [f32; 3], radius: f32) -> Option<f32> {
    let b = 2.0 * dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - 4.0 * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t0 = (-b - sqrt_disc) / 2.0;
    let t1 = (-b + sqrt_disc) / 2.0;

    if t1 < 0.0 {
        None
    } else if t0 < 0.0 {
        // Inside the sphere — use far intersection.
        Some(t1)
    } else {
        Some(t0)
    }
}

/// Dot product of two [f32; 3] arrays.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Length of a [f32; 3] vector.
fn length(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

/// Normalize a [f32; 3] vector.
pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(v);
    if len < 1e-10 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Convenience: compute sky color for a direction given sun elevation and
/// azimuth (both in radians). Elevation is 0 at horizon, π/2 at zenith.
pub fn sky_color_from_angles(view_dir: [f32; 3], sun_elevation: f32, sun_azimuth: f32) -> [f32; 3] {
    let sun_dir = [
        sun_azimuth.cos() * sun_elevation.cos(),
        sun_elevation.sin(),
        sun_azimuth.sin() * sun_elevation.cos(),
    ];
    sky_color(view_dir, sun_dir)
}

/// Convert HDR linear RGB to sRGB [u8; 3] with Reinhard tone mapping.
pub fn tonemap_to_srgb(hdr: [f32; 3]) -> [u8; 3] {
    let mapped = [
        hdr[0] / (1.0 + hdr[0]),
        hdr[1] / (1.0 + hdr[1]),
        hdr[2] / (1.0 + hdr[2]),
    ];
    // Apply sRGB gamma (simplified).
    [
        (mapped[0].powf(1.0 / 2.2) * 255.0).clamp(0.0, 255.0) as u8,
        (mapped[1].powf(1.0 / 2.2) * 255.0).clamp(0.0, 255.0) as u8,
        (mapped[2].powf(1.0 / 2.2) * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zenith_is_blue_at_noon() {
        let sun_dir = [0.0, 1.0, 0.0]; // Sun directly overhead
        let view_dir = [0.0, 1.0, 0.0]; // Looking straight up
        let color = sky_color(view_dir, sun_dir);
        // Blue should dominate at zenith due to 1/λ⁴ scattering.
        assert!(
            color[2] > color[0],
            "Blue ({}) should exceed red ({}) at zenith",
            color[2],
            color[0]
        );
    }

    #[test]
    fn horizon_is_reddish_at_sunset() {
        // Sun just above the horizon.
        let sun_dir = normalize([1.0, 0.05, 0.0]);
        // Look toward the sun at the horizon.
        let view_dir = normalize([1.0, 0.0, 0.0]);
        let color = sky_color(view_dir, sun_dir);
        // After the long optical path, red should dominate.
        assert!(
            color[0] > color[2],
            "Red ({}) should exceed blue ({}) toward sun at sunset",
            color[0],
            color[2]
        );
    }

    #[test]
    fn night_sky_is_dark() {
        // Sun well below the horizon.
        let sun_dir = normalize([0.0, -1.0, 1.0]);
        let view_dir = [0.0, 1.0, 0.0];
        let color = sky_color(view_dir, sun_dir);
        // All channels should be very dim.
        let max_channel = color[0].max(color[1]).max(color[2]);
        assert!(
            max_channel < 0.01,
            "Night sky should be very dark, got max channel {max_channel}"
        );
    }

    #[test]
    fn sky_varies_with_sun_elevation() {
        let view_dir = [0.0, 1.0, 0.0];
        let low = sky_color_from_angles(view_dir, 0.1, 0.0);
        let high = sky_color_from_angles(view_dir, 1.0, 0.0);
        let low_brightness = low[0] + low[1] + low[2];
        let high_brightness = high[0] + high[1] + high[2];
        assert!(
            high_brightness > low_brightness,
            "Higher sun should produce brighter sky: low={low_brightness}, high={high_brightness}"
        );
    }

    #[test]
    fn tonemap_clamps_correctly() {
        let black = tonemap_to_srgb([0.0, 0.0, 0.0]);
        assert_eq!(black, [0, 0, 0]);

        let bright = tonemap_to_srgb([10.0, 10.0, 10.0]);
        // Reinhard: 10/(1+10) ≈ 0.909 → gamma → ~244
        for c in bright {
            assert!(c > 200, "Bright HDR should map to high sRGB, got {c}");
        }
    }

    #[test]
    fn rayleigh_phase_symmetric() {
        // Phase function should be symmetric around cos_theta = 0.
        let forward = rayleigh_phase(1.0);
        let backward = rayleigh_phase(-1.0);
        assert!(
            (forward - backward).abs() < 1e-6,
            "Phase should be symmetric: forward={forward}, backward={backward}"
        );
    }

    #[test]
    fn rayleigh_phase_minimum_at_90_degrees() {
        let at_90 = rayleigh_phase(0.0);
        let at_0 = rayleigh_phase(1.0);
        assert!(
            at_0 > at_90,
            "Forward scattering ({at_0}) should exceed 90° ({at_90})"
        );
    }

    #[test]
    fn ray_sphere_from_inside() {
        let origin = [0.0, R_EARTH, 0.0]; // On surface
        let dir = [0.0, 1.0, 0.0]; // Looking up
        let hit = ray_sphere_intersect(origin, dir, R_EARTH + R_ATMOS);
        assert!(hit.is_some(), "Should intersect atmosphere from inside");
        let dist = hit.unwrap();
        // Should be approximately the atmosphere thickness.
        assert!(
            (dist - R_ATMOS).abs() < 1000.0,
            "Distance should be ~{R_ATMOS} m, got {dist}"
        );
    }

    #[test]
    fn beta_r_wavelength_dependence() {
        // β_R should follow 1/λ⁴: blue > green > red.
        const {
            assert!(BETA_R[2] > BETA_R[1]); // Blue scatters more than green
            assert!(BETA_R[1] > BETA_R[0]); // Green scatters more than red
        }
        // Blue/red ratio should be roughly (680/440)^4 ≈ 5.7
        let ratio = BETA_R[2] / BETA_R[0];
        assert!(
            (ratio - 5.7).abs() < 0.5,
            "Blue/red scattering ratio should be ~5.7, got {ratio}"
        );
    }

    #[test]
    fn opposite_horizon_less_red_at_sunset() {
        // Sun near the horizon on +X side.
        let sun_dir = normalize([1.0, 0.05, 0.0]);
        // Look away from the sun.
        let away_dir = normalize([-1.0, 0.1, 0.0]);
        let toward_dir = normalize([1.0, 0.1, 0.0]);

        let away_color = sky_color(away_dir, sun_dir);
        let toward_color = sky_color(toward_dir, sun_dir);

        // Looking toward the sun should have more red.
        assert!(
            toward_color[0] > away_color[0],
            "Toward-sun should be redder: toward_r={}, away_r={}",
            toward_color[0],
            away_color[0]
        );
    }
}
