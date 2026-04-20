// Orbital mechanics state and sun-direction computation.
//
// Tracks the planet's rotation (day cycle) and orbital position (year cycle)
// with real SI units.  Provides `compute_sun_direction` which combines axial
// tilt, libration, and daily rotation into a single sun-direction vector in
// the planet's body frame.  This module is consumed by a later integration
// step (A3) that replaces the sinusoidal sun model in the parent lighting
// module.

use bevy::math::DVec3;
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

/// Compute the direction toward the star as seen by a surface observer.
///
/// Uses the standard astronomical solar position formula:
///   altitude = arcsin(sin(φ)sin(δ) + cos(φ)cos(δ)cos(H))
///   azimuth  = atan2(-sin(H)cos(δ), sin(δ)cos(φ) - cos(δ)sin(φ)cos(H))
///
/// where φ = observer latitude, δ = solar declination (from axial tilt +
/// orbital position), H = hour angle (from rotation + libration).
///
/// Returns `(direction, elevation)` where `direction` is a unit `[f32; 3]`
/// in the observer's local frame (Y = up, X = east, Z = south) and
/// `elevation` is the angle above the horizon in radians.
pub fn compute_sun_direction(
    state: &OrbitalState,
    axial_tilt: f64,
    libration_amplitude: f64,
    libration_period: f64,
) -> ([f32; 3], f32) {
    compute_sun_direction_at_latitude(
        state,
        axial_tilt,
        libration_amplitude,
        libration_period,
        DEFAULT_OBSERVER_LATITUDE,
    )
}

/// Default observer latitude: 45° N — gives a pleasant day/night cycle
/// with noon sun elevation of ~45° (or up to ~68° at summer solstice with
/// Earth-like axial tilt).
pub const DEFAULT_OBSERVER_LATITUDE: f64 = std::f64::consts::FRAC_PI_4;

/// Compute the direction **toward the star** in the planet's body frame.
///
/// Unlike [`compute_sun_direction`], this is independent of any observer's
/// latitude — it returns a single unit `DVec3` that is the same for the
/// entire planet. The planet's rotation axis is assumed to be the Y axis;
/// if your `PlanetConfig.rotation_axis` differs, the caller must rotate
/// the result accordingly.
///
/// The coordinate frame is:
///   +Y = rotation axis (north pole)
///   +X = reference meridian at orbital_angle=0
///   +Z = completes right-handed system
///
/// Declination δ comes from axial tilt and orbital position.
/// Hour angle H comes from rotation angle and libration.
///
/// The star direction in equatorial coordinates:
///   x = cos(δ) cos(H)
///   y = sin(δ)
///   z = −cos(δ) sin(H)
pub fn compute_sun_direction_world(
    state: &OrbitalState,
    axial_tilt: f64,
    libration_amplitude: f64,
    libration_period: f64,
) -> DVec3 {
    use std::f64::consts::TAU;

    let declination = (axial_tilt.sin() * state.orbital_angle.sin()).asin();

    let libration_offset = if libration_period > 0.0 {
        let days_elapsed = state.rotation_angle / TAU;
        libration_amplitude * (TAU * days_elapsed / libration_period).sin()
    } else {
        0.0
    };

    // H=0 at local noon of the reference meridian, positive westward.
    let hour_angle = std::f64::consts::PI - state.rotation_angle - libration_offset;

    let cos_dec = declination.cos();
    let sin_dec = declination.sin();
    let cos_h = hour_angle.cos();
    let sin_h = hour_angle.sin();

    DVec3::new(cos_dec * cos_h, sin_dec, -cos_dec * sin_h).normalize()
}

/// Compute sun direction at a specific observer latitude.
///
/// See [`compute_sun_direction`] for the formula and return values.
pub fn compute_sun_direction_at_latitude(
    state: &OrbitalState,
    axial_tilt: f64,
    libration_amplitude: f64,
    libration_period: f64,
    observer_latitude: f64,
) -> ([f32; 3], f32) {
    use std::f64::consts::TAU;

    // Solar declination: sin(δ) = sin(axial_tilt) × sin(orbital_angle).
    // At orbital_angle = π/2 (summer solstice): δ = axial_tilt.
    // At orbital_angle = 0 or π (equinoxes): δ = 0.
    let declination = (axial_tilt.sin() * state.orbital_angle.sin()).asin();

    // Libration: sinusoidal wobble added to the hour angle.
    let libration_offset = if libration_period > 0.0 {
        let days_elapsed = state.rotation_angle / TAU;
        libration_amplitude * (TAU * days_elapsed / libration_period).sin()
    } else {
        0.0
    };

    // Hour angle: rotation_angle=0 → midnight (H=π), rotation_angle=π → noon (H=0).
    // Convention: H=0 at local noon, positive westward.
    let hour_angle = std::f64::consts::PI - state.rotation_angle - libration_offset;

    let sin_lat = observer_latitude.sin();
    let cos_lat = observer_latitude.cos();
    let sin_dec = declination.sin();
    let cos_dec = declination.cos();
    let cos_h = hour_angle.cos();
    let sin_h = hour_angle.sin();

    // Solar altitude (elevation above horizon).
    let sin_alt = sin_lat * sin_dec + cos_lat * cos_dec * cos_h;
    let altitude = sin_alt.asin();

    // Solar azimuth (measured from south, positive westward).
    let az_y = -sin_h * cos_dec;
    let az_x = sin_dec * cos_lat - cos_dec * sin_lat * cos_h;
    let azimuth = az_y.atan2(az_x);

    // Convert to a direction vector in the observer's local frame:
    //   Y = up (zenith), X = east, Z = south.
    let cos_alt = altitude.cos();
    let dir = [
        (azimuth.sin() * cos_alt) as f32, // east component
        sin_alt as f32,                   // up component
        (azimuth.cos() * cos_alt) as f32, // south component
    ];

    (dir, altitude as f32)
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

/// System: handle time acceleration keybindings.
///
/// - `BracketLeft` (`[`): halve time scale (min 1.0)
/// - `BracketRight` (`]`): double time scale (max 100_000.0)
/// - `Backslash` (`\`): reset to default (72.0)
/// - `KeyP`: toggle pause (set to 0 or restore previous)
pub fn time_acceleration_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<OrbitalState>,
    mut saved_scale: Local<Option<f64>>,
) {
    if keyboard.just_pressed(KeyCode::BracketLeft) {
        state.time_scale = (state.time_scale / 2.0).max(1.0);
        info!("Time scale: {}×", state.time_scale);
    }
    if keyboard.just_pressed(KeyCode::BracketRight) {
        state.time_scale = (state.time_scale * 2.0).min(100_000.0);
        info!("Time scale: {}×", state.time_scale);
    }
    if keyboard.just_pressed(KeyCode::Backslash) {
        state.time_scale = 72.0;
        *saved_scale = None;
        info!("Time scale reset to 72×");
    }
    if keyboard.just_pressed(KeyCode::KeyP) {
        if state.time_scale == 0.0 {
            state.time_scale = saved_scale.unwrap_or(72.0);
            *saved_scale = None;
            info!("Time resumed: {}×", state.time_scale);
        } else {
            *saved_scale = Some(state.time_scale);
            state.time_scale = 0.0;
            info!("Time paused");
        }
    }
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
    fn sun_direction_no_tilt_noon_high_elevation() {
        // At rotation=π (noon), no tilt, orbital_angle=0.
        // At 45° latitude the noon elevation should be ~45°.
        let state = OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: 0.0,
            ..OrbitalState::default()
        };
        let (_dir, elev) = compute_sun_direction(&state, 0.0, 0.0, 0.0);
        let elev_deg = elev.to_degrees();
        assert!(
            (elev_deg - 45.0).abs() < 1.0,
            "Noon at 45°N should be ~45° elevation, got {elev_deg:.1}°"
        );
    }

    #[test]
    fn sun_direction_noon_at_equator() {
        // At the equator (lat=0) with no tilt at noon, sun should be overhead (~90°).
        let state = OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: 0.0,
            ..OrbitalState::default()
        };
        let (_dir, elev) = compute_sun_direction_at_latitude(&state, 0.0, 0.0, 0.0, 0.0);
        let elev_deg = elev.to_degrees();
        assert!(
            (elev_deg - 90.0).abs() < 1.0,
            "Noon at equator should be ~90° elevation, got {elev_deg:.1}°"
        );
    }

    #[test]
    fn sun_direction_midnight_below_horizon() {
        // At rotation=0 (midnight) the sun should be below the horizon.
        let state = OrbitalState {
            rotation_angle: 0.0,
            orbital_angle: 0.0,
            ..OrbitalState::default()
        };
        let (_dir, elev) = compute_sun_direction(&state, 0.0, 0.0, 0.0);
        assert!(
            elev < 0.0,
            "Midnight elevation should be negative, got {:.1}°",
            elev.to_degrees()
        );
    }

    #[test]
    fn sun_direction_with_tilt_solstice() {
        // At summer solstice (orbital_angle=π/2) with 23.44° tilt,
        // noon elevation should be higher than at winter solstice.
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
            "Summer elevation ({:.1}°) should exceed winter ({:.1}°)",
            elev_summer.to_degrees(),
            elev_winter.to_degrees()
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

    #[test]
    fn time_scale_halving_has_minimum() {
        let mut scale = 2.0_f64;
        scale = (scale / 2.0).max(1.0);
        assert_eq!(scale, 1.0);
        scale = (scale / 2.0).max(1.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn time_scale_doubling_has_maximum() {
        let mut scale = 50_000.0_f64;
        scale = (scale * 2.0).min(100_000.0);
        assert_eq!(scale, 100_000.0);
        scale = (scale * 2.0).min(100_000.0);
        assert_eq!(scale, 100_000.0);
    }

    #[test]
    fn pause_sets_zero_and_resume_restores() {
        let mut state = OrbitalState {
            time_scale: 144.0,
            ..Default::default()
        };
        let saved = state.time_scale;
        state.time_scale = 0.0;
        assert_eq!(state.time_scale, 0.0);
        state.time_scale = saved;
        assert_eq!(state.time_scale, 144.0);
    }

    // ── compute_sun_direction_world tests ──

    #[test]
    fn world_sun_noon_points_toward_reference_meridian() {
        // rotation=π → noon on the reference meridian. Sun should be in +X
        // direction (cos(H)=cos(0)=1, sin(H)=0 → x=1, z=0).
        let state = OrbitalState {
            rotation_angle: std::f64::consts::PI,
            orbital_angle: 0.0,
            ..Default::default()
        };
        let dir = compute_sun_direction_world(&state, 0.0, 0.0, 0.0);
        assert!(dir.x > 0.9, "Noon sun should point toward +X, got {dir:?}");
        assert!(dir.y.abs() < 0.01, "No declination → Y≈0, got {dir:?}");
        assert!(dir.z.abs() < 0.01, "Noon → Z≈0, got {dir:?}");
    }

    #[test]
    fn world_sun_midnight_points_away_from_reference_meridian() {
        // rotation=0 → midnight. H=π → cos(π)=-1 → x=-1.
        let state = OrbitalState {
            rotation_angle: 0.0,
            orbital_angle: 0.0,
            ..Default::default()
        };
        let dir = compute_sun_direction_world(&state, 0.0, 0.0, 0.0);
        assert!(dir.x < -0.9, "Midnight sun should point -X, got {dir:?}");
    }

    #[test]
    fn world_sun_is_unit_vector() {
        let state = OrbitalState {
            rotation_angle: 1.234,
            orbital_angle: 0.567,
            ..Default::default()
        };
        let dir = compute_sun_direction_world(&state, 0.4, 0.1, 1.0);
        let len = dir.length();
        assert!(
            (len - 1.0).abs() < 1e-10,
            "Should be unit vector, length={len}"
        );
    }

    #[test]
    fn world_sun_solstice_has_declination() {
        // Summer solstice: orbital_angle=π/2, tilt=23.44°.
        // Declination should equal axial_tilt.
        let tilt = 23.44_f64.to_radians();
        let state = OrbitalState {
            rotation_angle: std::f64::consts::PI, // noon
            orbital_angle: std::f64::consts::FRAC_PI_2,
            ..Default::default()
        };
        let dir = compute_sun_direction_world(&state, tilt, 0.0, 0.0);
        // At noon with full declination, Y = sin(tilt) ≈ 0.3978
        assert!(
            (dir.y - tilt.sin()).abs() < 0.01,
            "Summer solstice noon Y should be sin(tilt)={}, got {}",
            tilt.sin(),
            dir.y
        );
    }

    #[test]
    fn world_sun_independent_of_observer() {
        // The function takes no observer latitude — verify same result
        // regardless of what compute_sun_direction gives for different latitudes.
        let state = OrbitalState {
            rotation_angle: 2.0,
            orbital_angle: 1.0,
            ..Default::default()
        };
        let dir1 = compute_sun_direction_world(&state, 0.2, 0.0, 0.0);
        let dir2 = compute_sun_direction_world(&state, 0.2, 0.0, 0.0);
        assert_eq!(dir1, dir2);
    }

    #[test]
    fn world_sun_equinox_in_equatorial_plane() {
        // At equinox (orbital_angle=0), declination=0 → sun in XZ plane.
        let state = OrbitalState {
            rotation_angle: 1.5,
            orbital_angle: 0.0,
            ..Default::default()
        };
        let dir = compute_sun_direction_world(&state, 0.4, 0.0, 0.0);
        assert!(
            dir.y.abs() < 1e-10,
            "Equinox sun should be in equatorial plane, Y={}",
            dir.y
        );
    }
}
