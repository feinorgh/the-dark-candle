//! Physically-based optics — Snell's law, Fresnel equations, total internal
//! reflection (TIR), and temperature-dependent refractive index of air.
//!
//! # SI Units
//! - Angles: radians
//! - Refractive index: dimensionless
//! - Temperature: Kelvin
//! - Wavelength (for dispersion helpers): metres
//!
//! # Physics summary
//!
//! ## Snell's law
//! At a boundary between media with indices n₁ and n₂:
//! ```text
//! n₁ sin θ₁ = n₂ sin θ₂
//! ```
//! Returns the refracted direction vector, or `None` if total internal
//! reflection occurs (θ₁ > θ_c).
//!
//! ## Fresnel equations (dielectric, unpolarized)
//! The fraction of intensity reflected at an interface:
//! ```text
//! R_s = ((n₁ cos θ₁ − n₂ cos θ₂) / (n₁ cos θ₁ + n₂ cos θ₂))²
//! R_p = ((n₁ cos θ₂ − n₂ cos θ₁) / (n₁ cos θ₂ + n₂ cos θ₁))²
//! R   = (R_s + R_p) / 2
//! T   = 1 − R
//! ```
//!
//! ## Temperature-dependent n of air
//! Air density decreases with temperature; so does its refractive index.
//! Useful for simulating mirages and heat shimmer.
//! Based on the Edlén formula (simplified):
//! ```text
//! n(T) ≈ 1 + (n₀ − 1) × (T₀ / T) × (P / P₀)
//! ```
//! where T₀ = 288.15 K, n₀ = 1.000 293, P₀ = 101 325 Pa.

/// Refractive index of air at standard temperature (288.15 K) and pressure.
pub const N_AIR_STP: f32 = 1.000_293;

/// Standard temperature for air (K).
const T0_AIR: f32 = 288.15;

/// Standard pressure for air (Pa).
const P0_AIR: f32 = 101_325.0;

// ---------------------------------------------------------------------------
// Snell's law
// ---------------------------------------------------------------------------

/// Compute the refracted direction vector using Snell's law.
///
/// # Arguments
/// - `incident` — unit incident ray direction (pointing **toward** the surface).
/// - `normal`   — unit surface normal pointing **away** from the medium the ray
///   is entering (i.e. pointing into the medium the ray is leaving).
/// - `n1`       — refractive index of the medium the ray is leaving.
/// - `n2`       — refractive index of the medium the ray is entering.
///
/// # Returns
/// The refracted unit direction, or `None` if total internal reflection occurs.
///
/// # Example
/// ```
/// # use the_dark_candle::lighting::optics::snell_refract;
/// // Ray going straight down through water surface (n1=1.0, n2=1.33).
/// let refracted = snell_refract([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 1.0, 1.33);
/// assert!(refracted.is_some()); // no TIR going into denser medium
/// ```
pub fn snell_refract(
    incident: [f32; 3],
    normal: [f32; 3],
    n1: f32,
    n2: f32,
) -> Option<[f32; 3]> {
    // cos θ₁ = −(incident · normal)  (incident points toward surface)
    let cos_i = -(dot(incident, normal));
    let ratio = n1 / n2;
    let sin2_t = ratio * ratio * (1.0 - cos_i * cos_i);

    if sin2_t > 1.0 {
        // Total internal reflection — no transmitted ray.
        return None;
    }

    let cos_t = (1.0 - sin2_t).sqrt();
    // Refracted direction: ratio × incident + (ratio × cos_i − cos_t) × normal
    let r = [
        ratio * incident[0] + (ratio * cos_i - cos_t) * normal[0],
        ratio * incident[1] + (ratio * cos_i - cos_t) * normal[1],
        ratio * incident[2] + (ratio * cos_i - cos_t) * normal[2],
    ];
    Some(normalize(r))
}

// ---------------------------------------------------------------------------
// Fresnel equations (unpolarized dielectric)
// ---------------------------------------------------------------------------

/// Compute the Fresnel reflectance for an unpolarized dielectric interface.
///
/// Returns R in [0, 1] — the fraction of intensity reflected.
/// The transmitted fraction is `1 − R` (neglecting absorption).
///
/// Uses the exact Fresnel equations for s- and p-polarization, averaged.
/// Handles total internal reflection by returning 1.0 when sin²θ_t > 1.
///
/// # Arguments
/// - `cos_i` — cosine of the angle of incidence (must be ≥ 0).
/// - `n1`    — refractive index of the incident medium.
/// - `n2`    — refractive index of the transmitted medium.
pub fn fresnel_reflectance(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let cos_i = cos_i.abs().clamp(0.0, 1.0);
    let sin2_i = 1.0 - cos_i * cos_i;
    let sin2_t = (n1 / n2).powi(2) * sin2_i;

    if sin2_t >= 1.0 {
        return 1.0; // Total internal reflection.
    }

    let cos_t = (1.0 - sin2_t).sqrt();

    // s-polarization (perpendicular)
    let rs_num = n1 * cos_i - n2 * cos_t;
    let rs_den = n1 * cos_i + n2 * cos_t;
    let rs = if rs_den.abs() < 1e-10 {
        1.0
    } else {
        (rs_num / rs_den).powi(2)
    };

    // p-polarization (parallel)
    let rp_num = n1 * cos_t - n2 * cos_i;
    let rp_den = n1 * cos_t + n2 * cos_i;
    let rp = if rp_den.abs() < 1e-10 {
        1.0
    } else {
        (rp_num / rp_den).powi(2)
    };

    // Unpolarized: average of the two polarizations.
    0.5 * (rs + rp)
}

/// Transmitted fraction at a dielectric interface: `T = 1 − R`.
#[inline]
pub fn fresnel_transmittance(cos_i: f32, n1: f32, n2: f32) -> f32 {
    1.0 - fresnel_reflectance(cos_i, n1, n2)
}

// ---------------------------------------------------------------------------
// Total internal reflection
// ---------------------------------------------------------------------------

/// Returns `true` if the angle of incidence exceeds the critical angle for
/// TIR (only possible when n1 > n2, i.e. going from denser to rarer medium).
///
/// # Arguments
/// - `cos_i` — cosine of the angle of incidence (≥ 0).
/// - `n1`    — refractive index of the incident medium.
/// - `n2`    — refractive index of the transmitted medium.
#[inline]
pub fn is_total_internal_reflection(cos_i: f32, n1: f32, n2: f32) -> bool {
    if n1 <= n2 {
        return false; // TIR only possible from denser → rarer.
    }
    let sin2_i = 1.0 - cos_i.clamp(0.0, 1.0).powi(2);
    let sin2_t = (n1 / n2).powi(2) * sin2_i;
    sin2_t >= 1.0
}

/// Critical angle θ_c (radians) for total internal reflection.
///
/// Returns `None` if n1 ≤ n2 (TIR is impossible).
/// ```text
/// sin θ_c = n2 / n1
/// ```
pub fn critical_angle(n1: f32, n2: f32) -> Option<f32> {
    if n1 <= n2 {
        return None;
    }
    Some((n2 / n1).clamp(0.0, 1.0).asin())
}

// ---------------------------------------------------------------------------
// Reflection direction
// ---------------------------------------------------------------------------

/// Compute the specular reflection direction.
///
/// # Arguments
/// - `incident` — unit incident ray direction (pointing toward surface).
/// - `normal`   — unit surface normal (pointing away from surface).
///
/// Returns the reflected unit direction.
/// ```text
/// reflected = incident − 2 (incident · normal) normal
/// ```
pub fn reflect_dir(incident: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let d = dot(incident, normal);
    let r = [
        incident[0] - 2.0 * d * normal[0],
        incident[1] - 2.0 * d * normal[1],
        incident[2] - 2.0 * d * normal[2],
    ];
    normalize(r)
}

// ---------------------------------------------------------------------------
// Temperature-dependent refractive index of air
// ---------------------------------------------------------------------------

/// Refractive index of air at the given temperature `t_k` (Kelvin) and
/// pressure `p_pa` (Pascals).
///
/// Based on the simplified Edlén formula for visible light:
/// ```text
/// n(T, P) ≈ 1 + (n₀ − 1) × (T₀ / T) × (P / P₀)
/// ```
/// where T₀ = 288.15 K, n₀ = 1.000 293 (standard air), P₀ = 101 325 Pa.
///
/// Hot air (e.g. above asphalt in summer ≈ 340 K) has a slightly lower n
/// than cold air, bending light upward — the mirage effect.
pub fn temperature_to_n_air(t_k: f32, p_pa: f32) -> f32 {
    let delta_n0 = N_AIR_STP - 1.0;
    1.0 + delta_n0 * (T0_AIR / t_k.max(1.0)) * (p_pa / P0_AIR)
}

/// Gradient of refractive index per Kelvin for air at standard pressure.
///
/// dn/dT ≈ −(n₀ − 1) × T₀ / T²
///
/// Negative: n decreases as temperature rises.
pub fn dn_dt_air(t_k: f32) -> f32 {
    let delta_n0 = N_AIR_STP - 1.0;
    -delta_n0 * T0_AIR / (t_k * t_k).max(1.0)
}

// ---------------------------------------------------------------------------
// Cauchy dispersion (optional per-channel refraction)
// ---------------------------------------------------------------------------

/// Cauchy's equation for wavelength-dependent refractive index.
///
/// ```text
/// n(λ) = A + B / λ²
/// ```
/// where λ is the wavelength in metres.
///
/// `a` and `b` are material-specific Cauchy coefficients.
/// Typical values for borosilicate glass: A = 1.5220, B = 4.61 × 10⁻¹⁵ m².
pub fn cauchy_n(wavelength_m: f32, a: f32, b: f32) -> f32 {
    a + b / (wavelength_m * wavelength_m)
}

/// Per-channel refractive indices (R, G, B) from Cauchy coefficients.
///
/// Uses canonical wavelengths: R = 680 nm, G = 550 nm, B = 440 nm.
pub fn cauchy_n_rgb(a: f32, b: f32) -> [f32; 3] {
    const R_M: f32 = 680e-9;
    const G_M: f32 = 550e-9;
    const B_M: f32 = 440e-9;
    [cauchy_n(R_M, a, b), cauchy_n(G_M, a, b), cauchy_n(B_M, a, b)]
}

// ---------------------------------------------------------------------------
// Vector math helpers (no external deps needed for pure optics math)
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return v;
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn vec_approx_eq(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    // --- Snell's law ---

    #[test]
    fn snell_normal_incidence_no_bending() {
        // At normal incidence (straight down), no bending regardless of n.
        let incident = [0.0_f32, -1.0, 0.0];
        let normal = [0.0_f32, 1.0, 0.0];
        let refracted = snell_refract(incident, normal, 1.0, 1.33).unwrap();
        // Should still point straight down.
        assert!(vec_approx_eq(refracted, [0.0, -1.0, 0.0]));
    }

    #[test]
    fn snell_bends_toward_normal_entering_denser() {
        // 45° incidence, air → glass (1.0 → 1.52).
        // θ₂ = arcsin(sin(45°) / 1.52) ≈ 27.7°
        let s = std::f32::consts::FRAC_1_SQRT_2; // sin(45°) = cos(45°)
        let incident = [s, -s, 0.0]; // 45° from vertical
        let normal = [0.0, 1.0, 0.0];
        let refracted = snell_refract(incident, normal, 1.0, 1.52).unwrap();
        // y-component should be more negative (steeper) than incident.
        assert!(refracted[1] < incident[1], "should bend toward normal");
        // Should still be a unit vector.
        let len = (refracted[0].powi(2) + refracted[1].powi(2) + refracted[2].powi(2)).sqrt();
        assert!(approx_eq(len, 1.0));
    }

    #[test]
    fn snell_tir_from_glass_to_air() {
        // 45° in glass (n=1.52) → air (n=1.0). Critical angle ≈ 41.1°.
        // 45° > 41.1° → TIR.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let incident = [s, -s, 0.0];
        let normal = [0.0, 1.0, 0.0];
        let result = snell_refract(incident, normal, 1.52, 1.0);
        assert!(result.is_none(), "should TIR");
    }

    #[test]
    fn snell_no_tir_going_into_denser() {
        // Going into denser medium can never TIR.
        let incident = [0.707, -0.707, 0.0];
        let normal = [0.0, 1.0, 0.0];
        let result = snell_refract(incident, normal, 1.0, 1.52);
        assert!(result.is_some());
    }

    // --- Fresnel ---

    #[test]
    fn fresnel_normal_incidence_air_to_glass() {
        // R at normal incidence = ((n1-n2)/(n1+n2))²
        // Air→glass: ((1.0-1.52)/(1.0+1.52))² = (0.52/2.52)² ≈ 0.0426
        let r = fresnel_reflectance(1.0, 1.0, 1.52);
        assert!((r - 0.0426).abs() < 0.001, "R = {r}");
    }

    #[test]
    fn fresnel_grazing_angle_approaches_one() {
        // At 90° (cos_i ≈ 0), reflectance → 1 for any n.
        let r = fresnel_reflectance(0.001, 1.0, 1.52);
        assert!(r > 0.99, "R at grazing = {r}");
    }

    #[test]
    fn fresnel_tir_returns_one() {
        // Above critical angle, R = 1.
        let r = fresnel_reflectance(0.1, 1.52, 1.0); // shallow angle in glass
        assert!(approx_eq(r, 1.0), "R during TIR = {r}");
    }

    #[test]
    fn fresnel_transmittance_complement() {
        let cos_i = 0.866; // 30°
        let r = fresnel_reflectance(cos_i, 1.0, 1.33);
        let t = fresnel_transmittance(cos_i, 1.0, 1.33);
        assert!(approx_eq(r + t, 1.0), "R+T = {}", r + t);
    }

    // --- Brewster's angle ---
    #[test]
    fn fresnel_near_brewster_p_pol_minimized() {
        // At Brewster's angle θ_B = arctan(n2/n1), R_p = 0.
        // For air→glass: θ_B = arctan(1.52) ≈ 56.7°
        let theta_b = (1.52_f32).atan(); // radians
        let cos_b = theta_b.cos();
        // Full Fresnel average is NOT zero at Brewster, but is at a local min.
        let r_below = fresnel_reflectance(cos_b + 0.1, 1.0, 1.52);
        let r_at = fresnel_reflectance(cos_b, 1.0, 1.52);
        let r_above = fresnel_reflectance(cos_b - 0.1, 1.0, 1.52);
        // r_at should be smaller than neighbors (local minimum).
        assert!(r_at < r_below + 0.01 || r_at < r_above + 0.01);
    }

    // --- TIR check ---

    #[test]
    fn tir_check_glass_to_air_above_critical() {
        assert!(is_total_internal_reflection(0.5, 1.52, 1.0)); // steep angle
    }

    #[test]
    fn tir_check_air_to_glass_never_tir() {
        // Going into denser: never TIR.
        for cos_i in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert!(!is_total_internal_reflection(cos_i, 1.0, 1.52));
        }
    }

    // --- Critical angle ---

    #[test]
    fn critical_angle_glass_to_air() {
        // sin θ_c = 1.0 / 1.52 → θ_c ≈ 41.1° = 0.717 rad
        let theta_c = critical_angle(1.52, 1.0).unwrap();
        assert!((theta_c - 0.717).abs() < 0.002, "θ_c = {theta_c}");
    }

    #[test]
    fn critical_angle_water_to_air() {
        // sin θ_c = 1.0 / 1.33 → θ_c ≈ 48.75°
        let theta_c = critical_angle(1.33, 1.0).unwrap();
        let expected = (1.0_f32 / 1.33).asin();
        assert!((theta_c - expected).abs() < 1e-5);
    }

    #[test]
    fn critical_angle_none_when_entering_denser() {
        assert!(critical_angle(1.0, 1.52).is_none());
    }

    // --- Reflection direction ---

    #[test]
    fn reflect_normal_incidence() {
        // Straight down hitting upward normal → straight up.
        let r = reflect_dir([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(vec_approx_eq(r, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn reflect_45_degrees() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let r = reflect_dir([s, -s, 0.0], [0.0, 1.0, 0.0]);
        assert!(vec_approx_eq(r, [s, s, 0.0]));
    }

    // --- Temperature-dependent n ---

    #[test]
    fn n_air_stp_close_to_constant() {
        let n = temperature_to_n_air(T0_AIR, P0_AIR);
        assert!((n - N_AIR_STP).abs() < 1e-6, "n at STP = {n}");
    }

    #[test]
    fn n_air_decreases_with_temperature() {
        let n_cold = temperature_to_n_air(250.0, P0_AIR);
        let n_hot = temperature_to_n_air(400.0, P0_AIR);
        assert!(n_cold > n_hot, "n should decrease with T");
    }

    #[test]
    fn n_air_decreases_with_pressure() {
        let n_high_p = temperature_to_n_air(288.15, 200_000.0);
        let n_low_p = temperature_to_n_air(288.15, 50_000.0);
        assert!(n_high_p > n_low_p, "n should increase with P");
    }

    #[test]
    fn dn_dt_air_is_negative() {
        let dn = dn_dt_air(288.15);
        assert!(dn < 0.0, "dn/dT should be negative");
    }

    // --- Cauchy dispersion ---

    #[test]
    fn cauchy_glass_rgb_blue_higher_than_red() {
        // Borosilicate glass: A=1.522, B=4.61e-15
        let [nr, ng, nb] = cauchy_n_rgb(1.522, 4.61e-15);
        assert!(nb > ng, "blue n > green n");
        assert!(ng > nr, "green n > red n");
    }

    #[test]
    fn cauchy_n_zero_b_returns_a() {
        let n = cauchy_n(550e-9, 1.5, 0.0);
        assert!(approx_eq(n, 1.5));
    }
}
