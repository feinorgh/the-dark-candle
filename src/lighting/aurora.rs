//! Aurora rendering: dedicated outer-shell sphere mesh with a custom additive
//! material that integrates emission over the auroral oval.
//!
//! Architecture overview lives in
//! `docs/superpowers/specs/2026-05-14-aurora-design.md`.
//!
//! The Rust side of this module owns three roles:
//!   1. `AuroraMaterial` — Bevy `Material` with `AlphaMode::Add`, backed by
//!      `assets/shaders/aurora.wgsl`.
//!   2. Systems — `spawn_aurora_shell` (Startup), `update_aurora_material`
//!      (Update), `anchor_aurora_shell_to_planet` (PostUpdate after
//!      `TransformSystems::Propagate`).
//!   3. CPU oracle helpers (`magnetic_north_axis`, `aurora_band_mask`,
//!      `aurora_day_side_factor`) — mirror the WGSL logic and serve as the
//!      test fixtures for the algorithm.

use bevy::math::{DVec3, Vec3};

/// Compute the magnetic-north unit vector in the same frame as
/// `planet_north_axis`, given a `(lat_offset_deg, lon_offset_deg)` tilt.
///
/// The tilt is built in an orthonormal basis derived from `planet_north_axis`:
///   - `up    = planet_north_axis.normalize()`
///   - `east  = up × X̂` (or `up × Ẑ` when `up` is near-X)
///   - `south = east × up`
///
/// `lat_offset_deg` is the tilt magnitude: at `lat = 0` the result equals
/// `up`; at `lat = 90°` it lies in the equatorial plane.
///
/// `lon_offset_deg` selects the tilt direction in the equatorial plane:
///   - `0°`  → `east`
///   - `90°` → `south`
///   - and rotates right-handed around `up` thereafter.
///
/// The convention of `east`/`south` directions is body-frame-dependent (it
/// follows the orthonormal-basis construction above); planets that want a
/// specific geographic alignment of their magnetic offset should choose
/// `lon_offset_deg` accordingly. `planet_north_axis` MUST be a unit vector.
pub fn magnetic_north_axis(
    planet_north_axis: DVec3,
    lat_offset_deg: f64,
    lon_offset_deg: f64,
) -> DVec3 {
    let up = planet_north_axis.normalize();
    let arb = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
    let east = up.cross(arb).normalize();
    let south = east.cross(up).normalize();

    let lat = lat_offset_deg.to_radians();
    let lon = lon_offset_deg.to_radians();
    let tilt_dir = east * lon.cos() + south * lon.sin();
    (up * lat.cos() + tilt_dir * lat.sin()).normalize()
}

/// f32 convenience wrapper for the GPU uniform path.
pub fn magnetic_north_axis_f32(
    planet_north_axis: Vec3,
    lat_offset_deg: f32,
    lon_offset_deg: f32,
) -> Vec3 {
    magnetic_north_axis(
        planet_north_axis.as_dvec3(),
        lat_offset_deg as f64,
        lon_offset_deg as f64,
    )
    .as_vec3()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnetic_north_axis_default_matches_geographic() {
        let geo = DVec3::Y;
        let mag = magnetic_north_axis(geo, 0.0, 0.0);
        assert!((mag - geo).length() < 1e-9, "mag={mag:?}");
    }

    #[test]
    fn magnetic_north_axis_with_lat_tilt_rotates_toward_equator() {
        let geo = DVec3::Y;
        let mag = magnetic_north_axis(geo, 10.0, 0.0);
        let expected_y = 10.0_f64.to_radians().cos();
        assert!(
            (mag.y - expected_y).abs() < 1e-9,
            "mag.y={}, want {}",
            mag.y,
            expected_y
        );
        let horiz = (mag.x * mag.x + mag.z * mag.z).sqrt();
        let expected_horiz = 10.0_f64.to_radians().sin();
        assert!((horiz - expected_horiz).abs() < 1e-9);
    }

    #[test]
    fn magnetic_north_axis_with_lon_rotates_tilt_direction() {
        let geo = DVec3::Y;
        let a = magnetic_north_axis(geo, 5.0, 0.0);
        let b = magnetic_north_axis(geo, 5.0, 90.0);
        let horiz_a = DVec3::new(a.x, 0.0, a.z).normalize();
        let horiz_b = DVec3::new(b.x, 0.0, b.z).normalize();
        let dot = horiz_a.dot(horiz_b);
        assert!(dot.abs() < 1e-9, "dot={}", dot);
    }

    #[test]
    fn magnetic_north_axis_handles_x_aligned_planet_axis() {
        // |up.x| >= 0.9 triggers the arb = Z fallback. Result must still be
        // a unit vector with the requested lat tilt magnitude.
        let geo = DVec3::X;
        let mag = magnetic_north_axis(geo, 15.0, 0.0);
        assert!((mag.length() - 1.0).abs() < 1e-9);
        let expected_x = 15.0_f64.to_radians().cos();
        assert!(
            (mag.x - expected_x).abs() < 1e-9,
            "mag.x={}, want {}",
            mag.x,
            expected_x
        );
        // Tilt magnitude away from up: sqrt(1 - x²) = sin(15°).
        let off_axis = (mag.y * mag.y + mag.z * mag.z).sqrt();
        let expected_off = 15.0_f64.to_radians().sin();
        assert!((off_axis - expected_off).abs() < 1e-9);
    }
}
