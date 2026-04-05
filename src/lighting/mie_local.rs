//! Local Mie scattering for particle-laden voxels.
//!
//! Atmospheric Mie scattering at planetary scale lives in `scattering.rs`.
//! This module handles **voxel-scale** Mie scattering: steam, ash, smoke, and
//! suspended water droplets (fog, cloud interiors). These particles are
//! comparable to or larger than visible wavelengths (0.1–100 µm), producing:
//!
//! - Forward-peaked glow around light sources viewed through fog or steam.
//! - Wavelength-independent scattering → white/grey appearance (unlike blue
//!   Rayleigh sky scattering which is strongly wavelength-dependent).
//! - Reduced visibility inside particle clouds (extinction).
//!
//! # Physics
//!
//! For spherical particles in the geometric optics regime (radius ≫ λ,
//! e.g. water droplets 10–100 µm), the extinction efficiency Q_ext ≈ 2 and
//! the Mie scattering coefficient is:
//!
//! ```text
//! β_Mie = Q_ext × π r² × N = 2 π r² N       [m⁻¹]
//! ```
//!
//! where r = particle radius (m) and N = number density (m⁻³).
//!
//! For simplicity we use representative tabulated values per material type
//! rather than computing from first-principles (which would require knowing
//! exact particle size distributions).
//!
//! Phase function: Henyey–Greenstein, which approximates the full Mie
//! solution with a single asymmetry parameter g:
//!
//! ```text
//! p(θ) = (1 − g²) / [ 4π (1 + g² − 2g cosθ)^1.5 ]
//! ```
//!
//! g = 0 gives isotropic scatter; g = 1 gives complete forward scattering.
//! Typical values: water cloud g ≈ 0.85, ash g ≈ 0.65.
//!
//! # SI Units
//! - Scattering coefficient β: m⁻¹
//! - Path length: m
//! - Transmittance: dimensionless (0–1)
//! - Phase function: dimensionless (normalized to integrate to 1 over 4π sr)

use std::f32::consts::PI;

use crate::world::voxel::MaterialId;

/// Mie scattering parameters for a particle-laden voxel material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalMieParams {
    /// Mie scattering coefficient β_Mie (m⁻¹).
    ///
    /// Controls how rapidly the medium scatters (and therefore extinguishes)
    /// light. Higher values → denser particle cloud → shorter visibility.
    pub coeff: f32,

    /// Henyey–Greenstein asymmetry parameter g (−1 to 1).
    ///
    /// g = 0: isotropic, g > 0: forward-peaked, g < 0: backward-peaked.
    /// Water clouds/steam: ~0.85, ash/smoke: ~0.65.
    pub g: f32,
}

/// Return Mie scattering parameters for a given material, if it is a
/// particle-scattering medium.
///
/// Returns `None` for non-scattering materials (clear solids, air, pure
/// transparent liquids such as clean water).
///
/// # Material values
///
/// | Material | β_Mie (m⁻¹) | g    | Notes                       |
/// |----------|-------------|------|-----------------------------|
/// | Steam    | 50.0        | 0.85 | Dense cloud/fog equivalent  |
/// | Ash      | 20.0        | 0.65 | Mixed particle sizes        |
pub fn mie_params_for_material(id: MaterialId) -> Option<LocalMieParams> {
    match id {
        MaterialId::STEAM => Some(LocalMieParams {
            // Steam / water droplet cloud: dense, strong forward scatter.
            // β_Mie = 50 m⁻¹ corresponds to ~20 cm visibility in thick steam.
            coeff: 50.0,
            g: 0.85,
        }),
        MaterialId::ASH => Some(LocalMieParams {
            // Ash particles (1–10 µm): moderate extinction, less forward-biased
            // than pure water droplets.
            coeff: 20.0,
            g: 0.65,
        }),
        _ => None,
    }
}

/// Mie transmittance through a column of particle-laden medium.
///
/// Wavelength-independent (for particles much larger than λ): all RGB channels
/// attenuate equally. Uses Beer-Lambert extinction:
///
/// ```text
/// T = exp(−β_Mie × d)
/// ```
///
/// # Arguments
/// - `coeff`    — Mie scattering coefficient β_Mie (m⁻¹)
/// - `path_len` — optical path length through the medium (m)
///
/// # Returns
/// Transmittance in [0, 1].
pub fn mie_transmittance(coeff: f32, path_len: f32) -> f32 {
    (-(coeff * path_len)).exp()
}

/// Per-channel Mie transmittance (identical for all channels).
///
/// Because Mie scattering from particles much larger than the wavelength is
/// wavelength-independent, the RGB transmittance is uniform. This contrasts
/// with Rayleigh scattering (λ⁻⁴) and Beer-Lambert absorption (material-
/// specific per-channel α).
///
/// # Arguments
/// - `coeff`    — Mie scattering coefficient β_Mie (m⁻¹)
/// - `path_len` — optical path length (m)
///
/// # Returns
/// `[T, T, T]` — same transmittance on all three channels.
pub fn mie_transmittance_rgb(coeff: f32, path_len: f32) -> [f32; 3] {
    let t = mie_transmittance(coeff, path_len);
    [t, t, t]
}

/// Henyey–Greenstein Mie phase function.
///
/// Gives the probability density of scattering a photon from direction θ
/// relative to the forward direction:
///
/// ```text
/// p(θ) = (1 − g²) / [ 4π (1 + g² − 2g cosθ)^1.5 ]
/// ```
///
/// Normalized: ∫ p(θ) dΩ = 1 over the full sphere.
///
/// # Arguments
/// - `cos_theta` — cosine of angle between incident and scattered direction
/// - `g`         — asymmetry parameter (see `LocalMieParams::g`)
///
/// # Returns
/// Phase function value (sr⁻¹). Multiply by β_Mie × dΩ × dl to get
/// in-scattered radiance per path element.
pub fn mie_phase_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    if denom < 1e-10 {
        // Pathological: avoid division by zero at exact forward peak.
        return 0.0;
    }
    (1.0 - g2) / (4.0 * PI * denom)
}

/// In-scatter factor: how much forward-scattered sunlight enters the view ray
/// through a Mie medium.
///
/// Combines the Mie phase function with the scattering coefficient to give
/// a single factor that, multiplied by the incident sun radiance and the
/// voxel path length dl, gives the in-scattered contribution per voxel:
///
/// ```text
/// ΔL_in = L_sun × mie_in_scatter_factor(cos_θ, params) × dl
/// ```
///
/// This produces the characteristic halo / corona glow visible around a
/// bright light source viewed through steam or fog.
///
/// # Arguments
/// - `cos_theta` — cosine of the angle between the sun and the view direction
/// - `params`    — Mie parameters for the medium
///
/// # Returns
/// In-scatter coefficient (m⁻¹ sr⁻¹).
pub fn mie_in_scatter_factor(cos_theta: f32, params: LocalMieParams) -> f32 {
    params.coeff * mie_phase_hg(cos_theta, params.g)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn steam_has_mie_params() {
        let p = mie_params_for_material(MaterialId::STEAM);
        assert!(p.is_some());
        let p = p.unwrap();
        assert!(p.coeff > 0.0, "steam should scatter light");
        assert!(p.g > 0.0 && p.g < 1.0, "g should be in (0, 1): {}", p.g);
    }

    #[test]
    fn ash_has_mie_params() {
        let p = mie_params_for_material(MaterialId::ASH);
        assert!(p.is_some());
        let p = p.unwrap();
        assert!(p.coeff > 0.0);
    }

    #[test]
    fn air_has_no_local_mie_params() {
        assert!(mie_params_for_material(MaterialId::AIR).is_none());
    }

    #[test]
    fn water_has_no_local_mie_params() {
        // Pure liquid water: no significant Mie particles.
        assert!(mie_params_for_material(MaterialId::WATER).is_none());
    }

    #[test]
    fn mie_transmittance_unity_at_zero_path() {
        let t = mie_transmittance(50.0, 0.0);
        assert!((t - 1.0).abs() < EPS, "T at d=0 = {t}");
    }

    #[test]
    fn mie_transmittance_decreases_with_distance() {
        let t1 = mie_transmittance(50.0, 0.1);
        let t2 = mie_transmittance(50.0, 0.5);
        assert!(
            t2 < t1,
            "T({t2}) should be less than T({t1}) at longer distance"
        );
    }

    #[test]
    fn mie_transmittance_approaches_zero_for_large_path() {
        let t = mie_transmittance(50.0, 10.0); // 10 m through dense steam
        assert!(t < 0.01, "T should approach 0 for large path: {t}");
    }

    #[test]
    fn mie_transmittance_rgb_all_channels_equal() {
        let [r, g, b] = mie_transmittance_rgb(20.0, 0.3);
        assert!((r - g).abs() < EPS, "R and G should be equal: {r} vs {g}");
        assert!((g - b).abs() < EPS, "G and B should be equal: {g} vs {b}");
        assert!(r > 0.0 && r <= 1.0);
    }

    #[test]
    fn mie_phase_hg_forward_biased() {
        let g = 0.85_f32; // steam
        let phase_forward = mie_phase_hg(1.0, g); // toward source
        let phase_backward = mie_phase_hg(-1.0, g); // away from source
        let phase_side = mie_phase_hg(0.0, g); // perpendicular
        assert!(
            phase_forward > phase_side,
            "forward ({phase_forward}) > side ({phase_side})"
        );
        assert!(
            phase_forward > phase_backward,
            "forward ({phase_forward}) > backward ({phase_backward})"
        );
    }

    #[test]
    fn mie_phase_hg_isotropic_at_g_zero() {
        // g = 0 → isotropic → p(θ) = 1/(4π) for all θ
        let p_fwd = mie_phase_hg(1.0, 0.0);
        let p_back = mie_phase_hg(-1.0, 0.0);
        let expected = 1.0 / (4.0 * std::f32::consts::PI);
        assert!((p_fwd - expected).abs() < 1e-5, "p_fwd = {p_fwd}");
        assert!((p_back - expected).abs() < 1e-5, "p_back = {p_back}");
    }

    #[test]
    fn mie_in_scatter_factor_positive() {
        let p = LocalMieParams {
            coeff: 50.0,
            g: 0.85,
        };
        let f = mie_in_scatter_factor(0.9, p); // near-forward viewing
        assert!(f > 0.0, "in-scatter factor should be positive: {f}");
    }

    #[test]
    fn mie_in_scatter_factor_larger_near_sun() {
        // Viewing almost toward the sun (cos_theta → 1) gives more in-scatter.
        let p = LocalMieParams {
            coeff: 50.0,
            g: 0.85,
        };
        let f_toward = mie_in_scatter_factor(0.95, p);
        let f_away = mie_in_scatter_factor(-0.5, p);
        assert!(
            f_toward > f_away,
            "in-scatter toward sun ({f_toward}) should exceed away ({f_away})"
        );
    }

    #[test]
    fn steam_coeff_greater_than_ash() {
        let steam = mie_params_for_material(MaterialId::STEAM).unwrap();
        let ash = mie_params_for_material(MaterialId::ASH).unwrap();
        assert!(
            steam.coeff > ash.coeff,
            "steam β ({}) should exceed ash β ({})",
            steam.coeff,
            ash.coeff
        );
    }
}
