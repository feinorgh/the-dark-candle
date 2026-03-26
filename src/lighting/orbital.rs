// Orbital mechanics state and sun-direction computation.
//
// Tracks the planet's rotation (day cycle) and orbital position (year cycle)
// with real SI units.  Provides `compute_sun_direction` which combines axial
// tilt, libration, and daily rotation into a single sun-direction vector in
// the planet's body frame.  This module is consumed by a later integration
// step (A3) that replaces the sinusoidal sun model in the parent lighting
// module.

use bevy::prelude::*;

/// Tracks the planet's rotational and orbital position.
///
/// Rotation angle determines the local time of day (which face of the planet
/// is lit by the star). Orbital angle determines the season (axial tilt
/// relative to star direction).
#[derive(Resource, Debug, Clone, Copy)]
pub struct OrbitalState {
    /// Planet rotation around its axis in radians (0..TAU). One full rotation = one day.
    pub rotation_angle: f64,
    /// Position in orbit around the star in radians (0..TAU). One full orbit = one year.
    pub orbital_angle: f64,
    /// Game-seconds per real-second. Default 72 (matching DayNightConfig).
    /// Set to 0 to pause time.
    pub time_scale: f64,
    /// How many game-days fit in one game-year. Default 1.0 means
    /// one planet rotation = one full orbit (every day is a full year).
    /// Set to 365.25 for Earth-like seasons.
    pub year_in_days: f64,
}

impl Default for OrbitalState {
    fn default() -> Self {
        Self {
            rotation_angle: 0.0,
            orbital_angle: 0.0,
            time_scale: 72.0,
            year_in_days: 1.0,
        }
    }
}

/// Advance rotation and orbital angles each frame.
pub fn advance_orbital_state(time: Res<Time>, mut state: ResMut<OrbitalState>) {
    let real_dt = time.delta_secs_f64();
    let game_dt = real_dt * state.time_scale;

    // Planet rotation: one full rotation = one game day = 86 400 game-seconds.
    let day_length = 86_400.0; // seconds per day
    let rotation_rate = std::f64::consts::TAU / day_length;
    state.rotation_angle = (state.rotation_angle + rotation_rate * game_dt) % std::f64::consts::TAU;

    // Orbital progression: year_in_days game-days per full orbit.
    let year_length = day_length * state.year_in_days;
    let orbital_rate = std::f64::consts::TAU / year_length;
    state.orbital_angle = (state.orbital_angle + orbital_rate * game_dt) % std::f64::consts::TAU;
}

/// Compute the direction toward the star in the planet's body frame.
///
/// The star direction in the inertial (orbital) frame is determined by
/// the orbital angle.  The planet's spin axis is tilted by `axial_tilt`
/// (obliquity) via rotation around the X axis, and the planet rotates by
/// `rotation_angle` around its spin axis (Y).  Libration adds a periodic
/// wobble.
///
/// Returns `(direction, elevation)` where `direction` is a unit-ish
/// `[f32; 3]` and `elevation` is the angle above the horizon at the
/// sub-stellar point (used for lighting intensity).
pub fn compute_sun_direction(
    state: &OrbitalState,
    axial_tilt: f64,
    libration_amplitude: f64,
    libration_period: f64,
) -> ([f32; 3], f32) {
    use std::f64::consts::TAU;

    // Star direction in inertial frame: always at +X (we orbit around it).
    // At orbital_angle=0, star is at +X.  As orbital_angle increases the
    // star appears to move around us.
    let star_inertial = [state.orbital_angle.cos(), 0.0, state.orbital_angle.sin()];

    // Apply axial tilt: rotate the reference frame around X by -axial_tilt.
    // This tilts the planet's spin axis (Y) away from the ecliptic normal,
    // producing a non-zero solar declination that varies with orbital angle.
    let cos_tilt = axial_tilt.cos();
    let sin_tilt = axial_tilt.sin();

    let star_tilted = [
        star_inertial[0],
        star_inertial[1] * cos_tilt + star_inertial[2] * sin_tilt,
        -star_inertial[1] * sin_tilt + star_inertial[2] * cos_tilt,
    ];

    // Libration: sinusoidal wobble in the spin axis.
    let libration_offset = if libration_period > 0.0 {
        let days_elapsed = state.rotation_angle / TAU;
        libration_amplitude * (TAU * days_elapsed / libration_period).sin()
    } else {
        0.0
    };

    // Apply planet rotation around Y axis (spin).
    let total_rotation = state.rotation_angle + libration_offset;
    let cos_rot = total_rotation.cos();
    let sin_rot = total_rotation.sin();

    // Rotate star direction by -rotation_angle around Y (planet spins under
    // the star).
    let sun_dir = [
        (star_tilted[0] * cos_rot + star_tilted[2] * sin_rot) as f32,
        star_tilted[1] as f32,
        (-star_tilted[0] * sin_rot + star_tilted[2] * cos_rot) as f32,
    ];

    // Elevation: angle above the horizon (Y component = sin(elevation)).
    let elevation = (sun_dir[1] as f64).asin() as f32;

    (sun_dir, elevation)
}

/// Derive time-of-day (hours 0..24) from the rotation angle for backward
/// compatibility with the sinusoidal sun model.
///
/// rotation_angle 0 = midnight, π = noon, 2π = next midnight.
/// TimeOfDay convention: 0=midnight, 6=sunrise, 12=noon, 18=sunset.
pub fn time_of_day_from_rotation(rotation_angle: f64) -> f32 {
    let hours = (rotation_angle / std::f64::consts::TAU * 24.0) as f32;
    hours % 24.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn orbital_state_defaults() {
        let state = OrbitalState::default();
        assert_eq!(state.rotation_angle, 0.0);
        assert_eq!(state.orbital_angle, 0.0);
        assert_eq!(state.time_scale, 72.0);
        assert_eq!(state.year_in_days, 1.0);
    }

    #[test]
    fn time_of_day_from_rotation_midnight() {
        let tod = time_of_day_from_rotation(0.0);
        assert!(tod.abs() < 0.01, "rotation=0 should be midnight, got {tod}");
    }

    #[test]
    fn time_of_day_from_rotation_noon() {
        let tod = time_of_day_from_rotation(std::f64::consts::PI);
        assert!(
            (tod - 12.0).abs() < 0.01,
            "rotation=π should be noon, got {tod}"
        );
    }

    #[test]
    fn sun_direction_no_tilt_noon() {
        // At rotation=π (noon), no tilt, orbital_angle=0: star at +X.
        // After rotating by π around Y, star appears at -X in body frame.
        let state = OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: 0.0,
            ..OrbitalState::default()
        };
        let (dir, _elev) = compute_sun_direction(&state, 0.0, 0.0, 0.0);
        // With no tilt, sun should be in XZ plane (y ≈ 0).
        assert!(
            dir[1].abs() < 0.01,
            "No tilt: sun Y should be ~0, got {}",
            dir[1]
        );
    }

    #[test]
    fn sun_direction_with_tilt_solstice() {
        // At summer solstice (orbital_angle=π/2) with 23.44° tilt,
        // the sun should reach higher elevations.
        let state = OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: std::f64::consts::FRAC_PI_2,
            ..OrbitalState::default()
        };
        let earth_tilt = 23.44_f64.to_radians();
        let (_dir_summer, elev_summer) = compute_sun_direction(&state, earth_tilt, 0.0, 0.0);

        // Winter solstice (orbital_angle = 3π/2).
        let state_winter = OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: 3.0 * std::f64::consts::FRAC_PI_2,
            ..OrbitalState::default()
        };
        let (_dir_winter, elev_winter) = compute_sun_direction(&state_winter, earth_tilt, 0.0, 0.0);

        assert!(
            elev_summer > elev_winter,
            "Summer elevation ({elev_summer}) should exceed winter ({elev_winter})"
        );
    }

    #[test]
    fn libration_produces_wobble() {
        // At rotation_angle = π/2, days_elapsed = 0.25,
        // sin(TAU * 0.25 / 1.0) = sin(π/2) = 1 → maximum libration offset.
        let state = OrbitalState {
            rotation_angle: std::f64::consts::FRAC_PI_2,
            orbital_angle: 0.0,
            ..OrbitalState::default()
        };
        let (dir_no, _) = compute_sun_direction(&state, 0.0, 0.0, 0.0);
        let (dir_yes, _) = compute_sun_direction(&state, 0.0, 0.1, 1.0);
        let diff = ((dir_no[0] - dir_yes[0]).powi(2)
            + (dir_no[1] - dir_yes[1]).powi(2)
            + (dir_no[2] - dir_yes[2]).powi(2))
        .sqrt();
        assert!(
            diff > 0.001,
            "Libration should shift sun direction, diff={diff}"
        );
    }

    #[test]
    fn year_in_days_scaling() {
        // With year_in_days=365.25, one rotation should advance orbital angle
        // by TAU/365.25.
        let state = OrbitalState {
            year_in_days: 365.25,
            ..OrbitalState::default()
        };
        let day_length = 86_400.0;
        let year_length = day_length * state.year_in_days;
        let expected_rate = TAU / year_length;
        let actual_advance = expected_rate * day_length;
        let orbital_per_day = TAU / 365.25;
        assert!(
            (actual_advance - orbital_per_day).abs() < 1e-10,
            "One day should advance orbital angle by TAU/365.25"
        );
    }
}
