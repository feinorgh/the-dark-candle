// Procedural generators for the celestial catalogue.
//
// Reads only `(planet_pos_in_galaxy, system_seed)` and produces a
// fully-deterministic `CelestialCatalogue` containing stars (this module),
// nebulae, galaxies, and host-galaxy parameters (added in later phases).
//
// All distributions are seeded from the system seed XORed with
// `SKY_SEED_SALT`, so the catalogue is reproducible alongside (but
// independent of) every other system-seed-driven generator in the project.

use bevy::math::DVec3;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::TAU;

use super::catalogue::{
    CelestialCatalogue, Galaxy, GalaxyKind, HostGalaxy, Nebula, NebulaKind, SKY_SEED_SALT,
    SpectralClass, Star,
};
use super::spectrum::{
    absolute_magnitude_from_luminosity, apparent_magnitude, blackbody_to_linear_rgb,
    mass_to_luminosity, mass_to_temperature,
};

// ─── Tunables ─────────────────────────────────────────────────────────────────

/// Faintest apparent V-magnitude retained in the catalogue.  Anything fainter
/// would never contribute a measurable pixel even in long-exposure mode.
pub const FAINTEST_MAGNITUDE: f32 = 22.0;

/// Number of candidate stars sampled before magnitude/visibility cuts.  At
/// galactic-disk densities the typical retention rate after the magnitude
/// cut is ~5–20 %, giving ~10⁵ stored stars.
pub const STAR_SAMPLE_COUNT: usize = 800_000;

/// Mass cut-off below which a "star" is treated as a brown dwarf and
/// excluded — they never reach naked-eye magnitude in the local volume.
const MIN_STELLAR_MASS_SOLAR: f32 = 0.08;

/// Maximum stellar mass sampled from the IMF (M_⊙).  Real stars rarely
/// exceed 100 M_⊙; sampling a few of them per system gives the bright
/// O-type beacons that anchor the night sky.
const MAX_STELLAR_MASS_SOLAR: f32 = 60.0;

// ─── Public entry point ──────────────────────────────────────────────────────

/// Generate a complete celestial catalogue for a planetary system.
///
/// `system_seed` is the same 64-bit seed used by `CelestialSystem::generate`;
/// the sky-specific generator XORs it with `SKY_SEED_SALT` so the catalogue
/// is independent of other systems sharing the seed.
///
/// In this phase only stars and the host-galaxy *parameters* are populated;
/// `nebulae` and `galaxies` are empty `Vec`s, and the host-galaxy diffuse
/// glow is sampled by the cubemap baker (later phase) from `HostGalaxy`.
pub fn generate_catalogue(system_seed: u64) -> CelestialCatalogue {
    let generator_seed = system_seed ^ SKY_SEED_SALT;
    let mut rng = SmallRng::seed_from_u64(generator_seed);

    let host_galaxy = generate_host_galaxy(&mut rng);
    let stars = generate_stars(&mut rng, &host_galaxy);
    let nebulae = generate_nebulae(&mut rng, &host_galaxy);
    let galaxies = generate_galaxies(&mut rng, &host_galaxy);

    CelestialCatalogue {
        stars,
        nebulae,
        galaxies,
        host_galaxy,
        generator_seed,
    }
}

// ─── host galaxy ───────────────────────────────────────────────────────────────

/// Pick a random orientation and bulge direction for the host galaxy.
fn generate_host_galaxy(rng: &mut SmallRng) -> HostGalaxy {
    let plane_normal = random_unit_vector(rng);

    // Pick a bulge direction lying *in* the galactic plane (perpendicular to
    // the plane normal) by projecting a random direction.
    let raw = random_unit_vector(rng);
    let bulge_direction = (raw - plane_normal * raw.dot(plane_normal)).normalize();

    HostGalaxy {
        plane_normal,
        bulge_direction,
        // Bulge spans ~25° at our line of sight; scale lightly per system.
        bulge_radius_rad: rng.random_range(0.30_f32..0.55_f32),
        // Effective angular thickness — controls the cosh-style fade away
        // from the galactic plane.
        disk_thickness_rad: rng.random_range(0.06_f32..0.12_f32),
        seed: rng.random::<u32>(),
    }
}

// ─── Stars ───────────────────────────────────────────────────────────────────

/// Generate the star list.
///
/// Process for each of `STAR_SAMPLE_COUNT` candidates:
///   1. Sample mass from the Kroupa IMF.
///   2. Compute luminosity and temperature from main-sequence relations.
///   3. Sample a distance from a thin/thick-disk + halo distribution.
///   4. Pick a celestial direction biased toward the galactic plane.
///   5. Compute apparent magnitude; reject if fainter than `FAINTEST_MAGNITUDE`.
fn generate_stars(rng: &mut SmallRng, mw: &HostGalaxy) -> Vec<Star> {
    let mut out = Vec::with_capacity(STAR_SAMPLE_COUNT / 8);

    for _ in 0..STAR_SAMPLE_COUNT {
        let mass = sample_kroupa_mass(rng);
        if mass < MIN_STELLAR_MASS_SOLAR {
            continue;
        }
        let luminosity = mass_to_luminosity(mass);
        let temperature = mass_to_temperature(mass);
        let distance_pc = sample_stellar_distance(rng);
        let m_abs = absolute_magnitude_from_luminosity(luminosity);
        let m_app = apparent_magnitude(m_abs, distance_pc);
        if m_app > FAINTEST_MAGNITUDE {
            continue;
        }

        let direction = sample_galactic_biased_direction(rng, mw);
        let color_linear = blackbody_to_linear_rgb(temperature);

        out.push(Star {
            direction,
            distance_pc,
            mass_solar: mass,
            luminosity_solar: luminosity,
            temperature_k: temperature,
            spectral_class: SpectralClass::from_temperature(temperature),
            apparent_magnitude_v: m_app,
            color_linear,
        });
    }

    out
}

/// Sample stellar mass (M_⊙) from the **Kroupa (2001)** broken power-law IMF.
///
///   ξ(M) ∝ M^−α
///        α = 1.3   for 0.08–0.5 M_⊙
///        α = 2.3   for 0.5 –50  M_⊙
///
/// Implemented by inverse-CDF sampling within each segment, with the
/// segment chosen by mass-fraction weights pre-computed from the integrals.
fn sample_kroupa_mass(rng: &mut SmallRng) -> f32 {
    // Integral of M^−α from a to b is (b^(1−α) − a^(1−α)) / (1 − α).
    let m1 = MIN_STELLAR_MASS_SOLAR as f64;
    let m2 = 0.5_f64;
    let m3 = MAX_STELLAR_MASS_SOLAR as f64;

    let i_low = power_law_integral(m1, m2, 1.3);
    let i_high = power_law_integral(m2, m3, 2.3) * power_law_continuity_factor(m2, 1.3, 2.3);

    let u: f64 = rng.random_range(0.0..(i_low + i_high));
    let mass = if u < i_low {
        sample_power_law(rng, m1, m2, 1.3)
    } else {
        sample_power_law(rng, m2, m3, 2.3)
    };
    mass as f32
}

/// Antiderivative-difference for ξ(M) ∝ M^−α between `a` and `b`.
fn power_law_integral(a: f64, b: f64, alpha: f64) -> f64 {
    let p = 1.0 - alpha;
    if p.abs() < 1e-9 {
        (b / a).ln()
    } else {
        (b.powf(p) - a.powf(p)) / p
    }
}

/// Continuity factor that re-scales the upper segment so that ξ(M) is
/// continuous at the break-point M = 0.5.  `M^−α₁` at 0.5 must equal
/// `C · M^−α₂` at the same M, hence `C = 0.5^(α₁ − α₂)`.
fn power_law_continuity_factor(m_break: f64, alpha_low: f64, alpha_high: f64) -> f64 {
    m_break.powf(alpha_low - alpha_high)
}

/// Inverse-CDF sample of ξ(M) ∝ M^−α on [a, b].
fn sample_power_law(rng: &mut SmallRng, a: f64, b: f64, alpha: f64) -> f64 {
    let p = 1.0 - alpha;
    let u: f64 = rng.random_range(0.0..1.0);
    if p.abs() < 1e-9 {
        a * (b / a).powf(u)
    } else {
        (a.powf(p) + u * (b.powf(p) - a.powf(p))).powf(1.0 / p)
    }
}

/// Sample a stellar distance (parsecs) from a simplified
/// thin-disk + thick-disk + halo cumulative model centred on the planet.
///
/// Distances are drawn out to ~5 kpc; beyond that point sources blur into
/// the host-galaxy diffuse component handled procedurally by the baker.
fn sample_stellar_distance(rng: &mut SmallRng) -> f32 {
    let u: f32 = rng.random_range(0.0..1.0);
    if u < 0.85 {
        // Thin disk: density falls off exponentially with scale length
        // ~3.5 kpc, but volume V ∝ d³ flattens the radial distribution
        // out to a few kpc.  Sample an exponential-truncated cube root.
        let v: f32 = rng.random_range(0.0..1.0);
        let d3: f32 = v * 5_000.0_f32.powi(3);
        d3.cbrt().max(0.5)
    } else if u < 0.97 {
        // Thick disk / older population: out to ~10 kpc, mostly fainter.
        let v: f32 = rng.random_range(0.0..1.0);
        let d3: f32 = v * 10_000.0_f32.powi(3);
        d3.cbrt().max(50.0)
    } else {
        // Halo: very distant, very faint individual stars.
        let v: f32 = rng.random_range(0.0..1.0);
        let d3: f32 = v * 25_000.0_f32.powi(3);
        d3.cbrt().max(500.0)
    }
}

/// Sample a unit direction biased toward the galactic plane.
///
/// The galactic latitude `b` (angle above/below `mw.plane_normal`) is sampled
/// from a Laplace-like density `exp(−|sin b|/h)` with `h = disk_thickness_rad`.
/// Galactic longitude is uniform around the plane.  This produces the
/// concentration of stars in a band that gives the host galaxy its visible shape.
fn sample_galactic_biased_direction(rng: &mut SmallRng, mw: &HostGalaxy) -> DVec3 {
    let h = mw.disk_thickness_rad as f64;
    // Inverse-CDF sample of exp(−|x|/h) on x = sin(b) ∈ [−1, 1] gives
    // x = −h · sign(u−0.5) · ln(1 − 2|u−0.5| · (1 − e^(−1/h))).
    let u: f64 = rng.random_range(0.0_f64..1.0_f64);
    let s = if u < 0.5 { -1.0 } else { 1.0 };
    let q = (2.0 * (u - 0.5).abs()).clamp(0.0, 1.0);
    let one_minus_e = 1.0 - (-1.0 / h).exp();
    let sin_b = s * (-h * (1.0 - q * one_minus_e).ln()).clamp(-1.0, 1.0);
    let cos_b = (1.0 - sin_b * sin_b).max(0.0).sqrt();

    let l: f64 = rng.random_range(0.0_f64..TAU);

    // Build a galactic-frame basis: pole = mw.plane_normal, bulge = mw.bulge_direction,
    // third axis = pole × bulge.
    let pole = mw.plane_normal;
    let bulge = mw.bulge_direction;
    let third = pole.cross(bulge).normalize();

    // (sin b)·pole + cos(b)·(cos l · bulge + sin l · third)
    sin_b * pole + cos_b * (l.cos() * bulge + l.sin() * third)
}

/// Sample a uniformly-distributed unit vector on the sphere.
fn random_unit_vector(rng: &mut SmallRng) -> DVec3 {
    let z: f64 = rng.random_range(-1.0_f64..1.0);
    let phi: f64 = rng.random_range(0.0_f64..TAU);
    let r = (1.0 - z * z).max(0.0).sqrt();
    DVec3::new(r * phi.cos(), r * phi.sin(), z)
}

// ─── Nebulae ─────────────────────────────────────────────────────────────────

/// Total number of nebulae generated per system, before any visibility cut.
/// Realistic: tens of catalogued bright nebulae per local galactic volume,
/// fading rapidly with distance and extinction.
const NEBULA_COUNT_MIN: usize = 18;
const NEBULA_COUNT_MAX: usize = 36;

/// Generate the nebula list, biased toward the galactic plane (where dust
/// and ionised gas concentrate).
fn generate_nebulae(rng: &mut SmallRng, hg: &HostGalaxy) -> Vec<Nebula> {
    let n = rng.random_range(NEBULA_COUNT_MIN..=NEBULA_COUNT_MAX);
    let mut out = Vec::with_capacity(n);

    for _ in 0..n {
        // 80 % concentrated near the galactic plane (the visible disk),
        // 20 % isotropic (foreground / off-plane gas).
        let direction = if rng.random_range(0.0_f32..1.0) < 0.80 {
            sample_galactic_biased_direction(rng, hg)
        } else {
            random_unit_vector(rng)
        };

        // Lognormal-ish angular radius: most are < 1°, a few span several °.
        // log10(r [rad]) sampled uniformly in [-3.0, -1.5] ⇒ 0.057° to 1.81°.
        let log_r: f32 = rng.random_range(-3.0_f32..-1.5);
        let angular_radius_rad = 10.0_f32.powf(log_r);

        let axial_ratio: f32 = rng.random_range(0.35_f32..1.0);
        let orientation_rad: f32 = rng.random_range(0.0_f32..std::f32::consts::TAU);

        let kind_roll: f32 = rng.random_range(0.0_f32..1.0);
        let kind = if kind_roll < 0.50 {
            NebulaKind::Emission
        } else if kind_roll < 0.75 {
            NebulaKind::Reflection
        } else if kind_roll < 0.90 {
            NebulaKind::Dark
        } else {
            NebulaKind::Planetary
        };

        let (color_linear, spectrum_peak_nm, surface_brightness) = match kind {
            NebulaKind::Emission => (
                // Hα-dominated red glow with a faint blue [O III] component.
                [1.00, 0.32, 0.42],
                656.3, // Hα line
                rng.random_range(20.0_f32..23.0),
            ),
            NebulaKind::Reflection => (
                // Dust scattering the blue light of a nearby hot star.
                [0.55, 0.70, 1.00],
                450.0,
                rng.random_range(21.0_f32..23.5),
            ),
            NebulaKind::Dark => (
                // Sign convention: colour is what is *removed* from the
                // background.  The baker treats `kind == Dark` as a
                // multiplicative attenuator of in-band light; the colour
                // here just biases which channel is most attenuated
                // (typical interstellar extinction is bluer-attenuating).
                [1.00, 0.85, 0.65],
                550.0,
                rng.random_range(24.0_f32..26.0),
            ),
            NebulaKind::Planetary => (
                // Doubly-ionised oxygen [O III] cyan-green.
                [0.45, 0.95, 0.85],
                500.7,
                rng.random_range(18.0_f32..21.0),
            ),
        };

        out.push(Nebula {
            direction,
            angular_radius_rad,
            axial_ratio,
            orientation_rad,
            kind,
            spectrum_peak_nm,
            surface_brightness,
            color_linear,
            texture_seed: rng.random::<u32>(),
        });
    }

    out
}

// ─── Galaxies ────────────────────────────────────────────────────────────────

const GALAXY_COUNT_MIN: usize = 80;
const GALAXY_COUNT_MAX: usize = 160;

/// Generate remote galaxies — distributed isotropically with a mild "zone
/// of avoidance" suppression near the host-galaxy plane (the host's own
/// dust extinguishes background galaxies seen along the disk).
fn generate_galaxies(rng: &mut SmallRng, hg: &HostGalaxy) -> Vec<Galaxy> {
    let plane_n = hg.plane_normal;
    let n_target = rng.random_range(GALAXY_COUNT_MIN..=GALAXY_COUNT_MAX);
    let mut out = Vec::with_capacity(n_target);

    let mut attempts = 0usize;
    while out.len() < n_target && attempts < n_target * 8 {
        attempts += 1;
        let dir = random_unit_vector(rng);

        // Zone-of-avoidance: reject ~80 % of galaxies that fall close to
        // the host plane (|sin b| < 0.10).  Real surveys see a similar
        // ~10° band devoid of catalogued galaxies due to extinction.
        let sin_b = dir.dot(plane_n).abs();
        if sin_b < 0.10 && rng.random_range(0.0_f32..1.0) < 0.80 {
            continue;
        }

        let angular_radius_rad: f32 = {
            // Lognormal: most galaxies are tiny (arcsec to arcmin), a few are big.
            let log_r: f32 = rng.random_range(-4.5_f32..-2.0);
            10.0_f32.powf(log_r)
        };
        let axial_ratio: f32 = rng.random_range(0.20_f32..1.0);
        let orientation_rad: f32 = rng.random_range(0.0_f32..std::f32::consts::TAU);

        let kind_roll: f32 = rng.random_range(0.0_f32..1.0);
        let kind = if kind_roll < 0.55 {
            GalaxyKind::Spiral
        } else if kind_roll < 0.80 {
            GalaxyKind::Elliptical
        } else if kind_roll < 0.92 {
            GalaxyKind::Lenticular
        } else {
            GalaxyKind::Irregular
        };

        let color_linear = match kind {
            GalaxyKind::Spiral => [0.85, 0.90, 1.00],
            GalaxyKind::Elliptical => [1.00, 0.88, 0.70],
            GalaxyKind::Lenticular => [1.00, 0.94, 0.82],
            GalaxyKind::Irregular => [0.80, 0.92, 1.00],
        };

        // Apparent magnitude: brighter for larger objects, but mostly
        // very dim.  Sample around 9..16 with a bias toward fainter.
        let mag_base: f32 = rng.random_range(9.0_f32..16.0);
        // Larger angular size correlates with closer / brighter galaxies.
        let size_brightness_bonus = (-2.5 * (angular_radius_rad / 1.0e-3).log10()).clamp(-3.0, 0.0);
        let apparent_magnitude_v = mag_base + size_brightness_bonus;

        // Cosmological redshift: we don't simulate cosmology, but galaxies
        // at typical extragalactic distances have z in [0.001, 0.05] for
        // anything visually resolvable.
        let redshift_z: f32 = rng.random_range(0.0005_f32..0.05);

        out.push(Galaxy {
            direction: dir,
            angular_radius_rad,
            axial_ratio,
            orientation_rad,
            kind,
            redshift_z,
            apparent_magnitude_v,
            color_linear,
        });
    }

    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_fixed_seed() {
        let a = generate_catalogue(0xDEAD_BEEF);
        let b = generate_catalogue(0xDEAD_BEEF);
        assert_eq!(a.stars.len(), b.stars.len());
        for (sa, sb) in a.stars.iter().zip(b.stars.iter()).take(50) {
            assert_eq!(sa.mass_solar, sb.mass_solar);
            assert_eq!(sa.distance_pc, sb.distance_pc);
            assert_eq!(sa.direction, sb.direction);
        }
    }

    #[test]
    fn naked_eye_visible_count_in_realistic_range() {
        // Naked-eye limit ≈ mag 6.5; expect a few thousand visible stars
        // (Earth's catalogue from a dark site is ~5 000–9 000).
        let cat = generate_catalogue(0x1234_5678);
        let visible = cat
            .stars
            .iter()
            .filter(|s| s.apparent_magnitude_v <= 6.5)
            .count();
        assert!(
            (1_000..=20_000).contains(&visible),
            "expected ~1k–20k naked-eye stars, got {visible}",
        );
    }

    #[test]
    fn star_directions_are_unit_vectors() {
        let cat = generate_catalogue(42);
        for s in cat.stars.iter().take(200) {
            let len = s.direction.length();
            assert!((len - 1.0).abs() < 1e-6, "non-unit direction len={len}",);
        }
    }

    #[test]
    fn imf_skews_toward_low_mass() {
        // M-dwarfs (mass < 0.5) should dominate.
        let cat = generate_catalogue(7);
        let total = cat.stars.len() as f32;
        let m_dwarfs = cat.stars.iter().filter(|s| s.mass_solar < 0.5).count() as f32;
        // The catalogue is *magnitude-limited*, so we can only assert that
        // the local galactic population contains a significant low-mass
        // tail; the apparent-magnitude cut removes most distant M-dwarfs.
        let frac = m_dwarfs / total.max(1.0);
        assert!(
            frac > 0.10,
            "M-dwarfs should be a meaningful fraction even after cut, got {frac}",
        );
    }

    #[test]
    fn galactic_plane_is_overdense() {
        // Stars near the galactic plane (|sin b| < disk_thickness) should be
        // overrepresented compared to a uniform sphere.
        let cat = generate_catalogue(99);
        let h = cat.host_galaxy.disk_thickness_rad as f64;
        let pole = cat.host_galaxy.plane_normal;
        let near_plane = cat
            .stars
            .iter()
            .filter(|s| s.direction.dot(pole).abs() < h)
            .count() as f32;
        let frac = near_plane / cat.stars.len() as f32;
        // Uniform-sphere expectation: 2·sin(h) ≈ 2h for small h.  The
        // galactic-plane bias should multiply this by a factor > 2.
        let uniform_expectation = 2.0 * h as f32;
        assert!(
            frac > 2.0 * uniform_expectation,
            "galactic plane should be overdense: frac={frac}, uniform={uniform_expectation}",
        );
    }
}
