// Photometric helpers for the sky catalogue.
//
// All functions are pure (no allocation, no I/O); they form the conversion
// layer between the physical state stored in `catalogue.rs` (mass,
// temperature, distance, …) and the rendering layer (linear-RGB colour,
// linear flux per pixel) used by the cubemap baker.
//
// References:
// * Mass–luminosity relation: Salaris & Cassisi 2005, *Evolution of Stars
//   and Stellar Populations*; piecewise power-law form.
// * Mass–temperature relation: empirical fits to the Hertzsprung-Russell
//   main sequence (good to ~10 % over 0.1–60 M_⊙).
// * Blackbody → sRGB: Tanner Helland's approximation, widely used in
//   real-time rendering (good visual match for 1 000–40 000 K).
// * Apparent magnitude / flux: standard Pogson definition.

/// Solar bolometric absolute magnitude (IAU 2015 nominal value).
pub const M_BOL_SUN: f32 = 4.74;

// ─── Mass → luminosity / temperature (main sequence) ─────────────────────────

/// Mass–luminosity relation for main-sequence stars (piecewise power-law).
///
/// `mass_solar` is M / M_⊙.  Returns L / L_⊙.
///
/// The four-segment fit covers brown dwarfs and very low-mass stars
/// (M < 0.43), low/middle main sequence (0.43–2 M_⊙), upper main sequence
/// (2–55 M_⊙) and high-mass stars (> 55 M_⊙) where the dependence flattens.
pub fn mass_to_luminosity(mass_solar: f32) -> f32 {
    let m = mass_solar.max(0.01);
    if m < 0.43 {
        0.23 * m.powf(2.3)
    } else if m < 2.0 {
        m.powi(4)
    } else if m < 55.0 {
        1.4 * m.powf(3.5)
    } else {
        32_000.0 * m
    }
}

/// Mass → effective surface temperature (K) for main-sequence stars.
///
/// Uses an empirical relation T ≈ 5778 × M^0.505, calibrated against the
/// Sun (M = 1, T = 5778 K) and adjusted to give the right values at the
/// extremes of the main sequence.  Brown dwarfs (M < 0.08) are approximated
/// by capping at 700 K; we treat them as "Y dwarfs" in the catalogue.
pub fn mass_to_temperature(mass_solar: f32) -> f32 {
    let m = mass_solar.max(0.01);
    if m < 0.08 {
        // Sub-stellar: roughly Y/T-dwarf range.
        (700.0 + 1_500.0 * (m / 0.08)).min(2_400.0)
    } else {
        (5778.0 * m.powf(0.505)).clamp(2_400.0, 50_000.0)
    }
}

// ─── Magnitude / flux ────────────────────────────────────────────────────────

/// Bolometric absolute magnitude from luminosity.
///
/// `M_bol = M_bol_⊙ − 2.5 · log₁₀(L / L_⊙)`.
pub fn absolute_magnitude_from_luminosity(luminosity_solar: f32) -> f32 {
    M_BOL_SUN - 2.5 * luminosity_solar.max(1e-12).log10()
}

/// Apparent magnitude from absolute magnitude and distance (parsecs).
///
/// `m = M + 5 · log₁₀(d / 10 pc)`.
pub fn apparent_magnitude(absolute_mag: f32, distance_pc: f32) -> f32 {
    // Floor protects against log10(0); 1e-9 pc ≈ 30 m, well below any
    // realistic stellar distance but still allows the unit test for the
    // Sun at 1 AU (~4.85e-6 pc) to pass.
    absolute_mag + 5.0 * (distance_pc.max(1e-9) / 10.0).log10()
}

/// Linear flux from apparent magnitude.
///
/// `F = F₀ · 10^(−0.4 · m)`.  The reference flux `F₀` is left at 1.0 — the
/// cubemap baker chooses the absolute scale via a single brightness
/// multiplier, so all that matters here is the *relative* flux between
/// objects.
pub fn flux_from_magnitude(magnitude: f32) -> f32 {
    10.0_f32.powf(-0.4 * magnitude)
}

// ─── Blackbody → linear sRGB ─────────────────────────────────────────────────

/// Blackbody temperature (K) → linear-light sRGB triple, unit-normalised
/// (the brightest channel is 1.0).
///
/// Brightness is *not* encoded here — multiply by `flux_from_magnitude(m)`
/// to get the radiometric flux contributed by an object of magnitude `m`.
///
/// The fit comes from Tanner Helland's well-known piecewise approximation
/// to a Planck spectrum convolved with the sRGB primaries, then converted
/// from gamma-encoded sRGB back to linear light.  It is accurate enough
/// (≲ 5 %) for visual identification of stellar colour and is far cheaper
/// than running CIE 1931 colour matching on every catalogue entry.
pub fn blackbody_to_linear_rgb(temperature_k: f32) -> [f32; 3] {
    // Helland's fit operates on T / 100.
    let t = temperature_k.clamp(1_000.0, 40_000.0) / 100.0;

    let r = if t <= 66.0 {
        255.0
    } else {
        329.698_73 * (t - 60.0).powf(-0.133_204_76)
    };
    let g = if t <= 66.0 {
        99.470_8 * t.ln() - 161.119_57
    } else {
        288.122_17 * (t - 60.0).powf(-0.075_514_85)
    };
    let b = if t >= 66.0 {
        255.0
    } else if t <= 19.0 {
        0.0
    } else {
        138.517_73 * (t - 10.0).ln() - 305.044_8
    };

    let srgb = [
        (r / 255.0).clamp(0.0, 1.0),
        (g / 255.0).clamp(0.0, 1.0),
        (b / 255.0).clamp(0.0, 1.0),
    ];

    // Gamma-decode to linear light (sRGB → linear).
    let linear = [
        srgb_to_linear(srgb[0]),
        srgb_to_linear(srgb[1]),
        srgb_to_linear(srgb[2]),
    ];

    // Unit-normalise so brightness is decoupled from colour.
    let m = linear[0].max(linear[1]).max(linear[2]).max(1e-6);
    [linear[0] / m, linear[1] / m, linear[2] / m]
}

/// Standard sRGB-to-linear gamma decode (IEC 61966-2-1).
#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 1 AU = 1 / 206264.806 pc.  Sun's apparent V-band magnitude is
    /// approximately −26.74; our simplified bolometric calculation gives a
    /// value within ~0.5 mag, which is fine for a procedural catalogue.
    #[test]
    fn solar_apparent_magnitude_sane() {
        let l_sun = 1.0;
        let m_abs = absolute_magnitude_from_luminosity(l_sun);
        assert!(
            (m_abs - M_BOL_SUN).abs() < 1e-4,
            "M_abs(L=1) = {m_abs}, expected {M_BOL_SUN}",
        );
        let one_au_pc = 1.0 / 206_264.81_f32;
        let m_app = apparent_magnitude(m_abs, one_au_pc);
        assert!(
            (m_app + 26.74).abs() < 1.0,
            "Sun at 1 AU should have m ≈ −26.74, got {m_app}",
        );
    }

    #[test]
    fn mass_luminosity_solar() {
        let l = mass_to_luminosity(1.0);
        assert!((l - 1.0).abs() < 1e-3, "L(M=1) = {l}, expected 1.0",);
    }

    #[test]
    fn mass_temperature_solar() {
        let t = mass_to_temperature(1.0);
        assert!((t - 5778.0).abs() < 1.0, "T(M=1) = {t}, expected 5778",);
    }

    #[test]
    fn mass_temperature_monotonic() {
        let t_low = mass_to_temperature(0.2);
        let t_sun = mass_to_temperature(1.0);
        let t_high = mass_to_temperature(20.0);
        assert!(
            t_low < t_sun && t_sun < t_high,
            "T must increase with mass: {t_low} {t_sun} {t_high}",
        );
    }

    #[test]
    fn blackbody_solar_near_white() {
        let rgb = blackbody_to_linear_rgb(5778.0);
        // Sun (5778 K): all three channels should be reasonably high (it
        // looks white to us by definition of the sRGB gamut, but in linear
        // light there is a small red bias relative to D65).
        for c in rgb.iter() {
            assert!(*c > 0.4, "channel {c} too low for solar blackbody {rgb:?}",);
        }
    }

    #[test]
    fn blackbody_cool_star_is_red() {
        let rgb = blackbody_to_linear_rgb(3000.0);
        assert!(rgb[0] > rgb[2], "3000 K should be red-dominant: {rgb:?}");
        assert!(rgb[2] < 0.3, "3000 K should have weak blue: {rgb:?}");
    }

    #[test]
    fn blackbody_hot_star_is_blue() {
        let rgb = blackbody_to_linear_rgb(15_000.0);
        assert!(
            rgb[2] >= rgb[0],
            "15 000 K should be blue-dominant: {rgb:?}",
        );
    }

    #[test]
    fn flux_pogson_step() {
        // A 5-mag difference is exactly a factor of 100 in flux.
        let f0 = flux_from_magnitude(0.0);
        let f5 = flux_from_magnitude(5.0);
        assert!(
            ((f0 / f5) - 100.0).abs() < 1e-3,
            "5-mag step should be 100×, got {}",
            f0 / f5,
        );
    }
}
