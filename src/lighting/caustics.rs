//! Caustic light concentration from refractive surfaces.
//!
//! Caustics are focused light patterns produced when a refractive or reflective
//! surface bends parallel rays toward convergent paths — the shimmering bright
//! patterns on the bottom of a sunlit pool, the bright ring cast by a glass
//! sphere, or light focused by a water drop.
//!
//! # Approach
//!
//! Two complementary methods are provided:
//!
//! 1. **Analytical Jacobian** — for a flat refractive interface at uniform
//!    angle of incidence, the caustic concentration factor is the inverse
//!    Jacobian of the refraction mapping between solid angles. This gives the
//!    irradiance multiplier at the receiver plane as a closed-form expression
//!    of the incidence angle and refractive indices.
//!
//!    ```text
//!    C = (n₂/n₁)² × (cos θ₁ / cos θ₂)
//!    ```
//!
//!    At normal incidence (θ₁ = 0): C = (n₂/n₁)², so light entering water
//!    from air is concentrated by a factor of ≈ 1.77.
//!
//! 2. **Photon beam tracing** — shoots a diverging bundle of photons from an
//!    emitter through a flat interface plane and records hit positions on a
//!    receiver plane. Kernel density estimation then gives local irradiance.
//!    Handles arbitrary angle of incidence and non-uniform spread.
//!
//! # SI Units
//! - Positions: metres (m)
//! - Irradiance: W/m² (normalized relative to incident; use multiplier × L_sun)
//! - Angles: radians
//!
//! # References
//! - Watt (1990), "Light–water interaction using backward beam tracing"
//! - Musgrave (2002), "A model for Mie scattering and caustic effects"
//! - Glassner (1995), "Principles of Digital Image Synthesis" §6.3

use crate::lighting::optics::{fresnel_transmittance, snell_refract};

// ---------------------------------------------------------------------------
// Photon beam tracing
// ---------------------------------------------------------------------------

/// A photon hit produced by `trace_caustic_beam`.
///
/// Records where a single refracted photon lands on the receiver plane,
/// along with the Fresnel-attenuated RGB power it carries.
#[derive(Debug, Clone, PartialEq)]
pub struct CausticPhoton {
    /// Position on the receiver plane (x, y, z) in metres.
    /// The z coordinate equals the `receiver_z` argument of `trace_caustic_beam`.
    pub pos: [f32; 3],

    /// Relative RGB power at this hit point (all channels in [0, 1]).
    ///
    /// Reduced from 1.0 by the Fresnel transmittance at the interface.
    /// For non-dispersive media all three channels are equal; for dispersive
    /// media with `n1_rgb`/`n2_rgb` variants they may differ.
    pub rgb: [f32; 3],
}

/// Trace a diverging bundle of photons through a flat horizontal refractive
/// interface and record where they land on a receiver plane below.
///
/// Shoots `count` rays from `origin`, spread within a cone of half-angle
/// `spread_rad` around the central direction `dir`. Each ray is refracted at
/// `interface_z` according to Snell's law and then intersected with the
/// receiver plane at `receiver_z`.
///
/// The receiver plane must be on the transmitted side of the interface
/// (i.e., `receiver_z` must be past `interface_z` in the direction of travel).
///
/// # Arguments
/// - `origin`           — photon source position (m)
/// - `dir`              — central ray direction (unit vector toward the interface)
/// - `interface_z`      — z-coordinate of the refractive interface (m)
/// - `receiver_z`       — z-coordinate of the receiver plane (m)
/// - `n1`               — refractive index on the source side (e.g. 1.0 for air)
/// - `n2`               — refractive index on the transmitted side (e.g. 1.33 water)
/// - `interface_normal` — unit surface normal of the interface, pointing from
///   the n2 side toward the n1 side (e.g. `[0,0,1]` upward)
/// - `count`            — number of photons to trace (more → smoother irradiance)
/// - `spread_rad`       — half-angle of the emission cone (radians)
///
/// # Returns
/// List of photon hits on the receiver plane. Photons that undergo TIR are
/// omitted. The returned list may be shorter than `count`.
///
/// # Notes
/// This function uses a z-axis–aligned interface for simplicity. The
/// implementation assumes the receiver plane is below the interface (in −z).
/// For a +z–pointing interface normal the photon must be travelling in −z.
#[allow(clippy::too_many_arguments)]
pub fn trace_caustic_beam(
    origin: [f32; 3],
    dir: [f32; 3],
    interface_z: f32,
    receiver_z: f32,
    n1: f32,
    n2: f32,
    interface_normal: [f32; 3],
    count: u32,
    spread_rad: f32,
) -> Vec<CausticPhoton> {
    let mut photons = Vec::with_capacity(count as usize);
    let dir_n = normalize3(dir);
    let perp = perpendicular_to(dir_n);
    let perp2 = cross3(dir_n, perp);

    for i in 0..count {
        // Distribute photons uniformly in azimuth; sample elevation via
        // stratified cos(elevation) within the spread cone.
        let t = if count > 1 {
            i as f32 / (count - 1) as f32
        } else {
            0.5
        };
        let azimuth = t * std::f32::consts::TAU;
        // Uniform distribution in cos(θ) from cos(spread) to 1.
        let cos_spread = spread_rad.cos();
        let cos_theta = cos_spread + (1.0 - cos_spread) * t;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        // Perturb the central direction by (azimuth, sin_theta) within the cone.
        let ray_dir = normalize3([
            dir_n[0] + sin_theta * (azimuth.cos() * perp[0] + azimuth.sin() * perp2[0]),
            dir_n[1] + sin_theta * (azimuth.cos() * perp[1] + azimuth.sin() * perp2[1]),
            dir_n[2] + sin_theta * (azimuth.cos() * perp[2] + azimuth.sin() * perp2[2]),
        ]);

        // Find where this ray hits the interface plane (z = interface_z).
        if ray_dir[2].abs() < 1e-10 {
            // Ray nearly parallel to interface — degenerate, skip.
            continue;
        }
        let t_iface = (interface_z - origin[2]) / ray_dir[2];
        if t_iface < 0.0 {
            // Interface is behind the origin — skip.
            continue;
        }
        let hit_iface = [
            origin[0] + ray_dir[0] * t_iface,
            origin[1] + ray_dir[1] * t_iface,
            interface_z,
        ];

        // Refract through the interface.
        let cos_i = dot3(ray_dir, interface_normal).abs();
        let Some(refracted) = snell_refract(ray_dir, interface_normal, n1, n2) else {
            // Total internal reflection — photon does not reach receiver.
            continue;
        };
        let transmittance = fresnel_transmittance(cos_i, n1, n2);

        // Find where the refracted ray hits the receiver plane.
        if refracted[2].abs() < 1e-10 {
            continue;
        }
        let t_recv = (receiver_z - hit_iface[2]) / refracted[2];
        if t_recv < 0.0 {
            continue;
        }
        let hit_recv = [
            hit_iface[0] + refracted[0] * t_recv,
            hit_iface[1] + refracted[1] * t_recv,
            receiver_z,
        ];

        photons.push(CausticPhoton {
            pos: hit_recv,
            rgb: [transmittance, transmittance, transmittance],
        });
    }

    photons
}

/// Estimate caustic irradiance at a receiver point using kernel density
/// estimation over a photon set.
///
/// Accumulates the power of all photons within `radius` metres of `point` on
/// the XY plane, then normalizes by the kernel area to give irradiance in
/// units of [incident power / m²].
///
/// Multiply the returned values by the source radiance (W/m²/sr) to get
/// absolute irradiance, or use relative to the direct-light value to get
/// the caustic brightness multiplier.
///
/// # Arguments
/// - `photons` — photon list from `trace_caustic_beam`
/// - `point`   — 2-D position on the receiver plane (x, y) in metres
/// - `radius`  — kernel radius (m); smaller = sharper but noisier estimate
///
/// # Returns
/// Per-channel irradiance `[R, G, B]` in units of [source power / m²].
pub fn caustic_irradiance_at(photons: &[CausticPhoton], point: [f32; 2], radius: f32) -> [f32; 3] {
    let r2 = radius * radius;
    let mut sum = [0.0_f32; 3];
    for photon in photons {
        let dx = photon.pos[0] - point[0];
        let dy = photon.pos[1] - point[1];
        if dx * dx + dy * dy <= r2 {
            sum[0] += photon.rgb[0];
            sum[1] += photon.rgb[1];
            sum[2] += photon.rgb[2];
        }
    }
    // Normalize by the kernel area π r².
    let area = std::f32::consts::PI * r2;
    [sum[0] / area, sum[1] / area, sum[2] / area]
}

// ---------------------------------------------------------------------------
// Analytical caustic concentration factor
// ---------------------------------------------------------------------------

/// Analytical caustic concentration factor for a flat refractive interface.
///
/// At a flat interface between two media, a pencil of rays at angle of
/// incidence θ₁ refracts to angle θ₂ (Snell's law). The solid angle of the
/// refracted bundle differs from the incident bundle by the Jacobian of the
/// refraction transformation:
///
/// ```text
/// dΩ₂/dΩ₁ = (n₁/n₂)² × (cos θ₂ / cos θ₁)
/// ```
///
/// Because irradiance scales as the inverse of solid angle (conservation of
/// étendue), the irradiance at the receiver plane is multiplied by:
///
/// ```text
/// C = dΩ₁/dΩ₂ = (n₂/n₁)² × (cos θ₁ / cos θ₂)
/// ```
///
/// At normal incidence (θ₁ = θ₂ = 0): C = (n₂/n₁)². For air→water this is
/// ≈ 1.77 — light entering water from a calm surface is concentrated by 77%.
///
/// The caustic is brightest at moderate angles and diverges (→ ∞) toward the
/// critical angle (where cos θ₂ → 0 and TIR is imminent). Above the critical
/// angle (TIR), no transmitted light reaches the receiver: C = 0.
///
/// # Arguments
/// - `cos_i` — cosine of the angle of incidence (must be ≥ 0)
/// - `n1`    — refractive index of the incident medium
/// - `n2`    — refractive index of the transmitted medium
///
/// # Returns
/// Irradiance concentration factor (dimensionless). Values > 1 indicate
/// focusing; < 1 indicate spreading. Returns 0 for TIR.
pub fn refraction_caustic_factor(cos_i: f32, n1: f32, n2: f32) -> f32 {
    // Compute sin²θ_t via Snell's law: n1 sinθ₁ = n2 sinθ₂
    let sin2_i = 1.0 - cos_i * cos_i;
    let ratio = n1 / n2;
    let sin2_t = ratio * ratio * sin2_i;

    if sin2_t >= 1.0 {
        // Total internal reflection — no transmitted light.
        return 0.0;
    }

    let cos_t = (1.0 - sin2_t).sqrt();
    if cos_t < 1e-6 {
        // Grazing angle approaching TIR — caustic intensity diverges.
        // In practice this is limited by beam width and wave effects.
        return f32::INFINITY;
    }

    // C = (n₂/n₁)² × (cos θ₁ / cos θ₂)
    let n21 = n2 / n1;
    n21 * n21 * (cos_i / cos_t)
}

/// Absolute underwater irradiance from a surface sunbeam.
///
/// Combines the analytical caustic factor with the Fresnel transmittance to
/// give the fraction of incident solar irradiance that reaches a flat
/// horizontal receiver at any depth below a flat calm water surface.
///
/// ```text
/// E_under = E_sun × T_Fresnel × C_caustic
/// ```
///
/// Note: at shallow water depths the distribution is uniform (the pencil of
/// transmitted rays has not yet reconverged). At greater depth the caustic
/// pattern dissolves into diffuse illumination due to scattering. This
/// function models the **peak** concentration.
///
/// # Arguments
/// - `cos_sun` — cosine of the solar zenith angle (1.0 = overhead sun)
/// - `n_water` — refractive index of water (≈ 1.33)
///
/// # Returns
/// Fraction of incident irradiance transmitted and concentrated below the surface.
pub fn underwater_irradiance_fraction(cos_sun: f32, n_water: f32) -> f32 {
    let t_fresnel = fresnel_transmittance(cos_sun, 1.0, n_water);
    let caustic = refraction_caustic_factor(cos_sun, 1.0, n_water);
    if caustic.is_infinite() {
        // Very close to critical angle — use Fresnel only, caustic is unphysically large.
        return t_fresnel;
    }
    t_fresnel * caustic
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return v;
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Compute a vector perpendicular to `v` (arbitrary but consistent choice).
fn perpendicular_to(v: [f32; 3]) -> [f32; 3] {
    // Choose the least-aligned standard axis as the "up" for the cross product.
    let ax = v[0].abs();
    let ay = v[1].abs();
    let az = v[2].abs();
    let other = if ax <= ay && ax <= az {
        [1.0_f32, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalize3(cross3(v, other))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Analytical caustic factor ---

    #[test]
    fn caustic_factor_normal_incidence_air_to_water() {
        // At θ₁ = 0: C = (n₂/n₁)² = 1.33² ≈ 1.769
        let f = refraction_caustic_factor(1.0, 1.0, 1.33);
        let expected = 1.33_f32.powi(2);
        assert!(
            (f - expected).abs() < 0.01,
            "normal incidence caustic factor = {f}, expected {expected}"
        );
    }

    #[test]
    fn caustic_factor_concentrates_light_entering_dense_medium() {
        // Going into denser medium: n₂ > n₁ → C > 1 at normal incidence.
        let f = refraction_caustic_factor(1.0, 1.0, 1.52);
        assert!(f > 1.0, "light should be concentrated entering glass: {f}");
    }

    #[test]
    fn caustic_factor_zero_for_tir() {
        // TIR: glass to air at steep angle (> critical angle ≈ 41°)
        let f = refraction_caustic_factor(0.1, 1.52, 1.0);
        assert_eq!(f, 0.0, "TIR should give caustic factor 0");
    }

    #[test]
    fn caustic_factor_exits_denser_medium_spreads_light() {
        // Exiting denser medium at moderate angle: n₁ > n₂ → C < 1 (spreading)
        let f = refraction_caustic_factor(0.95, 1.33, 1.0); // water to air, near-normal
        // At near-normal incidence: C = (n₂/n₁)² × (cos θ₁ / cos θ₂) < 1
        assert!(f < 1.0, "exiting dense medium should spread light: {f}");
    }

    #[test]
    fn caustic_factor_increases_toward_critical_angle() {
        // Caustic factor grows as θ₁ → θ_c (cos_i → small value).
        // For glass→air: θ_c ≈ 41.1° → cos_i ≈ 0.753
        // Compare f at 20° vs 35°:
        let f_20 = refraction_caustic_factor(20f32.to_radians().cos(), 1.52, 1.0);
        let f_35 = refraction_caustic_factor(35f32.to_radians().cos(), 1.52, 1.0);
        assert!(
            f_35 > f_20,
            "caustic factor should grow toward critical angle: f_20={f_20}, f_35={f_35}"
        );
    }

    // --- Photon beam tracing ---

    #[test]
    fn trace_caustic_beam_produces_photons() {
        let photons = trace_caustic_beam(
            [0.0, 0.0, 10.0], // source above water
            [0.0, 0.0, -1.0], // pointing straight down
            0.0,              // water surface at z = 0
            -5.0,             // receiver 5 m below surface
            1.0,              // air
            1.33,             // water
            [0.0, 0.0, 1.0],  // interface normal (upward)
            32,
            0.1,
        );
        assert!(!photons.is_empty(), "should produce photons");
    }

    #[test]
    fn trace_caustic_beam_photons_have_valid_transmittance() {
        let photons = trace_caustic_beam(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
            0.0,
            -5.0,
            1.0,
            1.33,
            [0.0, 0.0, 1.0],
            16,
            0.05,
        );
        for p in &photons {
            for &ch in &p.rgb {
                assert!((0.0..=1.0).contains(&ch), "transmittance {ch} out of [0,1]");
            }
        }
    }

    #[test]
    fn caustic_irradiance_positive_near_center() {
        let photons = trace_caustic_beam(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
            0.0,
            -5.0,
            1.0,
            1.33,
            [0.0, 0.0, 1.0],
            64,
            0.05,
        );
        let irr = caustic_irradiance_at(&photons, [0.0, 0.0], 0.5);
        assert!(
            irr[0] > 0.0,
            "irradiance at center should be positive: {:?}",
            irr
        );
    }

    #[test]
    fn caustic_irradiance_zero_far_from_photons() {
        let photons = trace_caustic_beam(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
            0.0,
            -5.0,
            1.0,
            1.33,
            [0.0, 0.0, 1.0],
            16,
            0.01,
        );
        // Far from the beam axis — no photons should hit here.
        let irr = caustic_irradiance_at(&photons, [1000.0, 1000.0], 0.01);
        assert_eq!(irr[0], 0.0, "no photons far from beam");
    }

    #[test]
    fn underwater_irradiance_fraction_overhead_sun() {
        // Overhead sun: cos_sun = 1.0
        // T_Fresnel at normal incidence ≈ 0.98 (very little reflection)
        // C ≈ 1.77 → E_under / E_sun ≈ 1.73
        let frac = underwater_irradiance_fraction(1.0, 1.33);
        assert!(
            frac > 1.0,
            "overhead sun should concentrate light under water: {frac}"
        );
    }

    #[test]
    fn perpendicular_to_is_orthogonal() {
        let v = normalize3([1.0, 2.0, 3.0]);
        let p = perpendicular_to(v);
        let d = dot3(v, p);
        assert!(
            d.abs() < 1e-5,
            "perpendicular vector not orthogonal: dot = {d}"
        );
    }

    #[test]
    fn perpendicular_to_is_unit_length() {
        let v = normalize3([0.0, 1.0, 0.0]);
        let p = perpendicular_to(v);
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "perpendicular vector not unit length: {len}"
        );
    }
}
