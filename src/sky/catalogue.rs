// Procedural celestial catalogue: stars, nebulae, galaxies, Milky Way.
//
// The catalogue lives in a "celestial inertial frame" with axes parallel to
// the planet's body frame at `OrbitalState.rotation_angle = 0`.  It is
// generated once per system from the system seed and is fixed for the entire
// game session.  Each object stores enough real physical state (mass,
// temperature, distance, luminosity, redshift) that future re-baking for
// other EM bands or long-exposure modes only needs the same catalogue plus
// different bake parameters.
//
// All quantities use SI or astronomy-standard units (parsec, solar mass /
// luminosity, Kelvin, magnitude).  See `src/sky/spectrum.rs` for the
// magnitude/flux/blackbody conversions used downstream.

use bevy::math::DVec3;
use serde::{Deserialize, Serialize};

// ─── Spectral & object-kind enums ─────────────────────────────────────────────

/// Morgan-Keenan spectral class plus brown-dwarf classes (L, T, Y).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpectralClass {
    /// ≥ 30 000 K, blue, very rare and luminous.
    O,
    /// 10 000 – 30 000 K, blue-white.
    B,
    /// 7 500 – 10 000 K, white.
    A,
    /// 6 000 – 7 500 K, yellow-white.
    F,
    /// 5 200 – 6 000 K, yellow (Sun is G2V).
    G,
    /// 3 700 – 5 200 K, orange.
    K,
    /// 2 400 – 3 700 K, red, the most common stellar class.
    M,
    /// 1 300 – 2 400 K, brown dwarf, deep red.
    L,
    /// 700 – 1 300 K, brown dwarf, methane-band, magenta-ish.
    T,
    /// < 700 K, brown dwarf, near-infrared only.
    Y,
}

impl SpectralClass {
    /// Map an effective surface temperature (K) to the closest spectral class.
    pub fn from_temperature(t_k: f32) -> Self {
        if t_k >= 30_000.0 {
            Self::O
        } else if t_k >= 10_000.0 {
            Self::B
        } else if t_k >= 7_500.0 {
            Self::A
        } else if t_k >= 6_000.0 {
            Self::F
        } else if t_k >= 5_200.0 {
            Self::G
        } else if t_k >= 3_700.0 {
            Self::K
        } else if t_k >= 2_400.0 {
            Self::M
        } else if t_k >= 1_300.0 {
            Self::L
        } else if t_k >= 700.0 {
            Self::T
        } else {
            Self::Y
        }
    }
}

/// Diffuse-nebula morphological/emission class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NebulaKind {
    /// H II regions, supernova remnants — Hα-dominated red glow.
    Emission,
    /// Dust illuminated by nearby hot star, blue scattered light.
    Reflection,
    /// Optically thick dust silhouetted against bright background.
    Dark,
    /// Compact (~< 0.1°), often [O III]-greenish or doubly-ionised cyan.
    Planetary,
}

/// Galaxy morphological class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GalaxyKind {
    /// Disk + spiral arms, intermediate axial ratio.
    Spiral,
    /// Smooth ellipsoidal, redder.
    Elliptical,
    /// Lens-shaped, between spiral and elliptical.
    Lenticular,
    /// Patchy, disturbed shape; often bluer (active star formation).
    Irregular,
}

// ─── Object structs ───────────────────────────────────────────────────────────

/// A single star in the catalogue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Star {
    /// Unit vector toward the star in the celestial inertial frame.
    pub direction: DVec3,
    /// Distance from the planet (parsecs).
    pub distance_pc: f32,
    /// Stellar mass in solar masses (M / M_sun).
    pub mass_solar: f32,
    /// Bolometric luminosity in solar luminosities (L / L_sun).
    pub luminosity_solar: f32,
    /// Effective surface temperature (K).
    pub temperature_k: f32,
    /// Morgan-Keenan spectral class (O B A F G K M L T Y).
    pub spectral_class: SpectralClass,
    /// Apparent V-band magnitude as seen from the planet.
    pub apparent_magnitude_v: f32,
    /// Linear-light sRGB colour from blackbody integration with CIE 1931 CMFs.
    /// Components are unit-normalised (max channel = 1.0); brightness comes
    /// from `apparent_magnitude_v`, not from this colour.
    pub color_linear: [f32; 3],
}

/// A diffuse nebula (gas/dust cloud).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nebula {
    /// Unit vector toward the nebula centre (celestial frame).
    pub direction: DVec3,
    /// Apparent angular semi-major-axis radius (radians).
    pub angular_radius_rad: f32,
    /// Axial ratio b/a (1.0 = circular, < 1 = elongated).
    pub axial_ratio: f32,
    /// Position-angle of the major axis on the sky (radians, CCW from north).
    pub orientation_rad: f32,
    /// Emission/Reflection/Dark/Planetary.
    pub kind: NebulaKind,
    /// Dominant emission wavelength (nm), used by future EM-band re-bakes.
    pub spectrum_peak_nm: f32,
    /// Surface brightness in magnitudes per square arc-second (lower = brighter).
    pub surface_brightness: f32,
    /// Linear-light sRGB tint, unit-normalised; same role as `Star.color_linear`.
    pub color_linear: [f32; 3],
    /// Seed for the procedural internal-noise pattern during cubemap baking.
    pub texture_seed: u32,
}

/// A remote galaxy (Local-Group or beyond).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Galaxy {
    /// Unit vector toward the galaxy centre (celestial frame).
    pub direction: DVec3,
    /// Apparent angular semi-major-axis radius (radians).
    pub angular_radius_rad: f32,
    /// Axial ratio b/a.
    pub axial_ratio: f32,
    /// Position angle on the sky (radians).
    pub orientation_rad: f32,
    /// Spiral / Elliptical / Lenticular / Irregular.
    pub kind: GalaxyKind,
    /// Cosmological redshift z (used by future Doppler-shifted EM-band re-bakes).
    pub redshift_z: f32,
    /// Apparent V-band magnitude integrated over the whole galaxy.
    pub apparent_magnitude_v: f32,
    /// Linear-light sRGB tint, unit-normalised.
    pub color_linear: [f32; 3],
}

/// Procedural Milky-Way model.
///
/// The diffuse galactic glow is *not* splatted as discrete objects — it is
/// sampled analytically inside the cubemap baker from these parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilkyWay {
    /// Unit vector normal to the galactic plane (celestial frame).
    pub plane_normal: DVec3,
    /// Unit vector toward the galactic centre (line of sight to the bulge).
    pub bulge_direction: DVec3,
    /// Angular radius of the central bulge brightness peak (radians).
    pub bulge_radius_rad: f32,
    /// Effective angular thickness of the disk (radians) — controls the sin(b)
    /// falloff in the diffuse model.
    pub disk_thickness_rad: f32,
    /// Procedural-noise seed for the variable-density disk texture.
    pub seed: u32,
}

/// Complete celestial catalogue for one planetary system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialCatalogue {
    /// All catalogued stars (visible + sub-visible to the magnitude cutoff).
    pub stars: Vec<Star>,
    /// Discrete diffuse nebulae.
    pub nebulae: Vec<Nebula>,
    /// Remote galaxies.
    pub galaxies: Vec<Galaxy>,
    /// Procedural Milky-Way parameters (sampled in the baker).
    pub milky_way: MilkyWay,
    /// Seed used to generate this catalogue (= system_seed XOR sky salt).
    pub generator_seed: u64,
}

/// XOR salt mixed into the system seed to derive the sky generator seed.
/// Keeps the celestial catalogue independent of (and reproducible alongside)
/// other generators that consume the same system seed.
pub const SKY_SEED_SALT: u64 = 0xCE1E_57A2_F1E1_D000;
