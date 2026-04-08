//! Celestial mechanics: star, moons, rings, Keplerian orbits, tidal forces.
//!
//! All quantities use SI units (kg, m, s, K, W) as per project standard.
//! The planet is placed at the origin; all body positions are planet-centred.

use bevy::math::DVec3;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ─── Physical constants ───────────────────────────────────────────────────────

/// Gravitational constant (N·m²/kg²).
const G: f64 = 6.674e-11;
/// Solar mass (kg).
const M_SUN: f64 = 1.989e30;
/// Solar luminosity (W).
const L_SUN: f64 = 3.828e26;
/// Solar radius (m).
const R_SUN: f64 = 6.960e8;
/// Solar effective surface temperature (K).
const T_SUN: f64 = 5778.0;
/// One astronomical unit (m).
pub const AU: f64 = 1.496e11;
/// 2π (convenience alias for `std::f64::consts::TAU`).
const TWO_PI: f64 = std::f64::consts::TAU;

// ─── Data types ───────────────────────────────────────────────────────────────

/// The host star of the planetary system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Star {
    /// Stellar mass (kg).
    pub mass_kg: f64,
    /// Photospheric radius (m).
    pub radius_m: f64,
    /// Effective surface temperature (K).
    pub temperature_k: f64,
    /// Bolometric luminosity (W).
    pub luminosity_w: f64,
    /// Approximate RGB colour (0–1) derived from blackbody spectrum.
    pub color: [f32; 3],
    /// Inner edge of habitable zone (m from star centre).
    pub habitable_zone_inner_m: f64,
}

/// One natural satellite orbiting the planet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Moon {
    /// Moon mass (kg).
    pub mass_kg: f64,
    /// Mean radius (m).
    pub radius_m: f64,
    /// Semi-major axis of orbit (m).
    pub semi_major_axis_m: f64,
    /// Orbital eccentricity (0 = circular, <1 = elliptic).
    pub eccentricity: f64,
    /// Orbital inclination relative to planet equatorial plane (rad).
    pub inclination_rad: f64,
    /// Longitude of the ascending node (rad).
    pub longitude_ascending_node_rad: f64,
    /// Argument of periapsis (rad).
    pub argument_of_periapsis_rad: f64,
    /// Mean anomaly at epoch t = 0 (rad).
    pub mean_anomaly_at_epoch_rad: f64,
    /// Orbital period (s), derived from Kepler's third law.
    pub orbital_period_s: f64,
    /// Surface albedo (0–1).
    pub albedo: f32,
    /// Approximate surface RGB colour.
    pub surface_color: [f32; 3],
}

/// Planetary ring system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ring {
    /// Inner edge distance from planet centre (m).
    pub inner_radius_m: f64,
    /// Outer edge distance from planet centre (m).
    pub outer_radius_m: f64,
    /// Optical depth / opacity (0–1).
    pub opacity: f32,
    /// Ring RGB colour (icy ≈ grey-white, rocky ≈ sandy-brown).
    pub color: [f32; 3],
}

/// A predicted eclipse event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EclipseEvent {
    /// Approximate time of mid-eclipse (s since epoch 0).
    pub time_s: f64,
    /// `true` = solar eclipse (moon occults star); `false` = lunar eclipse.
    pub is_solar: bool,
    /// Maximum fraction of the stellar disc covered (0–1).
    pub magnitude: f64,
}

/// Complete celestial environment of the generated planet.
///
/// Provides methods for computing body positions and derived phenomena
/// (tidal heights, illumination direction, eclipse timing) at any game time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialSystem {
    /// The host star.
    pub star: Star,
    /// Natural satellites (0–4).
    pub moons: Vec<Moon>,
    /// Planetary ring system, if present.
    pub ring: Option<Ring>,
    /// Planet mass (kg), used for tidal calculations.
    pub planet_mass_kg: f64,
    /// Planet equatorial radius (m).
    pub planet_radius_m: f64,
    /// Planet's orbital semi-major axis around the star (m).
    pub planet_orbit_m: f64,
    /// Planet's orbital period around the star (s).
    pub planet_orbital_period_s: f64,
    /// Axial tilt (obliquity) in radians.  Earth ≈ 0.4091 rad (23.44°).
    /// Determines the equator-to-pole insolation gradient and seasonal
    /// extremes.  Generated randomly in [0.05, 0.70] rad (≈ 3°–40°).
    #[serde(default = "default_axial_tilt")]
    pub axial_tilt_rad: f64,
}

/// Default axial tilt for deserialising legacy data without the field.
fn default_axial_tilt() -> f64 {
    0.4091 // Earth's obliquity
}

// ─── Keplerian mechanics ──────────────────────────────────────────────────────

/// Solve Kepler's equation **M = E − e·sin(E)** for the eccentric anomaly *E*.
///
/// Uses Newton–Raphson iteration; converges in ≤ 10 steps for e < 0.9.
///
/// # Arguments
/// * `m` – Mean anomaly (rad, wrapped to 0–2π by the caller).
/// * `e` – Orbital eccentricity (0 ≤ e < 1).
pub fn solve_kepler(m: f64, e: f64) -> f64 {
    let mut ea = m; // initial guess: eccentric ≈ mean anomaly
    for _ in 0..10 {
        let delta = (ea - e * ea.sin() - m) / (1.0 - e * ea.cos());
        ea -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    ea
}

/// Convert eccentric anomaly to true anomaly (rad).
fn true_anomaly(ea: f64, e: f64) -> f64 {
    2.0 * ((((1.0 + e) / (1.0 - e)).sqrt() * (ea / 2.0).tan()).atan())
}

/// Compute the 3D planet-centred position of a moon at time `t` (s since epoch 0).
///
/// Returns a position vector in the planet's equatorial inertial frame (m):
/// +X = vernal equinox, +Y = north pole, +Z = completing right-hand system.
///
/// The orbital elements stored in `moon` fully determine the result; no
/// additional planet-mass argument is required (the period encodes it already).
pub fn moon_position(moon: &Moon, t: f64) -> DVec3 {
    // Mean anomaly M = M₀ + n·t, where n = 2π/T.
    let m =
        (TWO_PI / moon.orbital_period_s * t + moon.mean_anomaly_at_epoch_rad).rem_euclid(TWO_PI);

    let ea = solve_kepler(m, moon.eccentricity);
    let nu = true_anomaly(ea, moon.eccentricity);
    let r = moon.semi_major_axis_m * (1.0 - moon.eccentricity * ea.cos());

    // Position in perifocal (orbital-plane) frame.
    let x_orb = r * nu.cos();
    let y_orb = r * nu.sin();

    // Rotate into inertial frame via 3-1-3 Euler angles: Ω, i, ω.
    // Standard result for the rotation matrix rows:
    //   x = (cos Ω cos ω − sin Ω sin ω cos i) x_orb + (−cos Ω sin ω − sin Ω cos ω cos i) y_orb
    //   y = (sin Ω cos ω + cos Ω sin ω cos i) x_orb + (−sin Ω sin ω + cos Ω cos ω cos i) y_orb
    //   z = (sin ω sin i)                     x_orb + (cos ω sin i)                      y_orb
    let (cos_o, sin_o) = (
        moon.argument_of_periapsis_rad.cos(),
        moon.argument_of_periapsis_rad.sin(),
    );
    let (cos_i, sin_i) = (moon.inclination_rad.cos(), moon.inclination_rad.sin());
    let (cos_bo, sin_bo) = (
        moon.longitude_ascending_node_rad.cos(),
        moon.longitude_ascending_node_rad.sin(),
    );

    let x = (cos_bo * cos_o - sin_bo * sin_o * cos_i) * x_orb
        + (-cos_bo * sin_o - sin_bo * cos_o * cos_i) * y_orb;
    let y = (sin_bo * cos_o + cos_bo * sin_o * cos_i) * x_orb
        + (-sin_bo * sin_o + cos_bo * cos_o * cos_i) * y_orb;
    let z = sin_o * sin_i * x_orb + cos_o * sin_i * y_orb;

    DVec3::new(x, y, z)
}

// ─── Internal generators ──────────────────────────────────────────────────────

/// Approximate RGB colour for a blackbody source at `temperature_k`.
///
/// Uses the Ballesteros (2012) piecewise formula, accurate to ~1% over 1000–40000 K.
#[allow(clippy::excessive_precision)]
fn blackbody_color(temperature_k: f64) -> [f32; 3] {
    let t = temperature_k;

    let r = if t < 6600.0 {
        1.0_f32
    } else {
        let x = (t / 100.0 - 60.0) as f32;
        (329.698_727_4 * x.powf(-0.133_204_76) / 255.0).clamp(0.0, 1.0)
    };

    let g = if t < 6600.0 {
        let x = (t / 100.0) as f32;
        ((99.470_802_6 * x.ln() - 161.119_568_0) / 255.0).clamp(0.0, 1.0)
    } else {
        let x = (t / 100.0 - 60.0) as f32;
        (288.122_169_5 * x.powf(-0.075_514_84) / 255.0).clamp(0.0, 1.0)
    };

    let b = if t >= 6600.0 {
        1.0_f32
    } else if t <= 1900.0 {
        0.0_f32
    } else {
        let x = (t / 100.0 - 10.0) as f32;
        ((138.517_731_2 * x.ln() - 305.044_792_2) / 255.0).clamp(0.0, 1.0)
    };

    [r, g, b]
}

/// Generate the host star from an RNG.
///
/// Mass ratio M/M☉ is drawn log-uniformly from [0.6, 1.8], biasing toward G/K
/// main-sequence stars. Temperature, luminosity, and radius follow standard
/// main-sequence scaling relations (Zombeck 2007).
fn generate_star(rng: &mut SmallRng) -> Star {
    // Log-uniform mass ratio in [0.6, 1.8].
    let u: f64 = rng.random_range(0.0_f64..1.0_f64);
    let mass_ratio = 0.6_f64 * (1.8_f64 / 0.6_f64).powf(u);
    let mass_kg = mass_ratio * M_SUN;

    // Main-sequence scaling: T ∝ M^0.505, L ∝ M^4, R ∝ M^0.8.
    let temperature_k = T_SUN * mass_ratio.powf(0.505);
    let luminosity_w = L_SUN * mass_ratio.powi(4);
    let radius_m = R_SUN * mass_ratio.powf(0.8);
    let color = blackbody_color(temperature_k);

    // Habitable zone inner edge ≈ sqrt(L/L☉) × 0.95 AU.
    let habitable_zone_inner_m = (luminosity_w / L_SUN).sqrt() * 0.95 * AU;

    Star {
        mass_kg,
        radius_m,
        temperature_k,
        luminosity_w,
        color,
        habitable_zone_inner_m,
    }
}

/// Generate 0–4 moons for the planet.
///
/// Count follows an empirical distribution (0 = 40%, 1 = 30%, 2 = 18%, 3 = 9%,
/// 4 = 3%).  Semi-major axes are log-uniform between 2× Roche limit and 100×
/// planet radius.  Masses are drawn uniformly relative to Earth's moon.
fn generate_moons(rng: &mut SmallRng, planet_mass_kg: f64, planet_radius_m: f64) -> Vec<Moon> {
    // Cumulative distribution over {0,1,2,3,4}.
    const WEIGHTS: [f64; 5] = [0.40, 0.70, 0.88, 0.97, 1.00];
    let roll: f64 = rng.random_range(0.0_f64..1.0_f64);
    let count = WEIGHTS.iter().position(|&w| roll < w).unwrap_or(4);

    // Orbital range: from 2× Roche limit to 100× planet radius.
    let roche = 2.44 * planet_radius_m;
    let a_min = roche * 2.0;
    let a_max = 100.0 * planet_radius_m;

    let mut moons = Vec::with_capacity(count);
    for _ in 0..count {
        // Semi-major axis: log-uniform in [a_min, a_max].
        let u: f64 = rng.random_range(0.0_f64..1.0_f64);
        let a = a_min * (a_max / a_min).powf(u);

        let eccentricity: f64 = rng.random_range(0.0_f64..0.15_f64);
        let inclination_rad: f64 = rng.random_range(0.0_f64..0.30_f64);
        let long_asc: f64 = rng.random_range(0.0_f64..TWO_PI);
        let arg_peri: f64 = rng.random_range(0.0_f64..TWO_PI);
        let mean_anom: f64 = rng.random_range(0.0_f64..TWO_PI);

        // Mass: 0.001–0.15 × Earth's moon (7.342e22 kg).
        let mass_frac: f64 = rng.random_range(0.001_f64..0.15_f64);
        let mass_kg = mass_frac * 7.342e22;

        // Radius from assumed bulk density (1 800–3 200 kg/m³).
        let density: f64 = rng.random_range(1800.0_f64..3200.0_f64);
        let radius_m = ((3.0 * mass_kg) / (4.0 * std::f64::consts::PI * density)).cbrt();

        // Orbital period from Kepler's third law: T² = 4π²a³ / (G·M_planet).
        let period_s = TWO_PI * (a.powi(3) / (G * planet_mass_kg)).sqrt();

        // Appearance: icy moons are grey-white, rocky moons are brownish.
        let albedo: f32 = rng.random_range(0.1_f32..0.45_f32);
        let icy = density < 2400.0;
        let surface_color: [f32; 3] = if icy {
            [0.85, 0.85, 0.90]
        } else {
            let base: f32 = rng.random_range(0.45_f32..0.70_f32);
            [base, base * 0.90, base * 0.78]
        };

        moons.push(Moon {
            mass_kg,
            radius_m,
            semi_major_axis_m: a,
            eccentricity,
            inclination_rad,
            longitude_ascending_node_rad: long_asc,
            argument_of_periapsis_rad: arg_peri,
            mean_anomaly_at_epoch_rad: mean_anom,
            orbital_period_s: period_s,
            albedo,
            surface_color,
        });
    }
    moons
}

/// Optionally generate a ring system (~15% probability).
///
/// Rings are placed between 2.0–3.5 × planet radius (between the Roche limit
/// and a stable outer boundary).
fn generate_ring(rng: &mut SmallRng, planet_radius_m: f64) -> Option<Ring> {
    if rng.random_range(0.0_f32..1.0_f32) >= 0.15 {
        return None;
    }
    let inner = planet_radius_m * rng.random_range(2.0_f64..2.6_f64);
    let outer = planet_radius_m * rng.random_range(2.8_f64..3.5_f64);
    let opacity: f32 = rng.random_range(0.2_f32..0.85_f32);
    let icy = rng.random_range(0.0_f32..1.0_f32) < 0.5;
    let color: [f32; 3] = if icy {
        [0.90, 0.92, 0.95]
    } else {
        [0.60, 0.55, 0.45]
    };
    Some(Ring {
        inner_radius_m: inner,
        outer_radius_m: outer,
        opacity,
        color,
    })
}

// ─── CelestialSystem ─────────────────────────────────────────────────────────

impl CelestialSystem {
    /// Generate a celestial system from the planet's physical parameters and seed.
    ///
    /// The planet is placed in the star's habitable zone (factor 0.8–1.4 × the
    /// inner-edge distance) and the orbital period follows Kepler's third law.
    pub fn generate(planet_mass_kg: f64, planet_radius_m: f64, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed ^ 0xCE1E_5741_DEAD_5555);
        let star = generate_star(&mut rng);

        // Place planet in the star's habitable zone.
        let orbit_factor: f64 = rng.random_range(0.8_f64..1.4_f64);
        let planet_orbit_m = star.habitable_zone_inner_m * orbit_factor;
        let planet_orbital_period_s = TWO_PI * (planet_orbit_m.powi(3) / (G * star.mass_kg)).sqrt();

        let moons = generate_moons(&mut rng, planet_mass_kg, planet_radius_m);
        let ring = generate_ring(&mut rng, planet_radius_m);

        // Axial tilt: most terrestrial planets have modest obliquity, but
        // giant impacts can produce high tilts (Uranus ≈ 98°).  We bias toward
        // Earth-like values: [0.05, 0.70] rad ≈ [3°, 40°].
        let axial_tilt_rad: f64 = rng.random_range(0.05_f64..0.70_f64);

        Self {
            star,
            moons,
            ring,
            planet_mass_kg,
            planet_radius_m,
            planet_orbit_m,
            planet_orbital_period_s,
            axial_tilt_rad,
        }
    }

    /// Direction from the planet to the star at time `t` (s since epoch 0).
    ///
    /// The planet orbits in the XZ plane (ecliptic); the returned unit vector
    /// points toward the star in the planet's equatorial inertial frame.
    pub fn star_direction_at(&self, t: f64) -> DVec3 {
        let angle = TWO_PI * t / self.planet_orbital_period_s;
        DVec3::new(angle.cos(), 0.0, angle.sin())
    }

    /// Planet-centred positions (m) of all moons at time `t` (s since epoch 0).
    pub fn moon_positions_at(&self, t: f64) -> Vec<DVec3> {
        self.moons.iter().map(|m| moon_position(m, t)).collect()
    }

    /// Total equilibrium tidal surface displacement (m) at a planet-surface point.
    ///
    /// `cell_pos` is a unit vector on the planet's surface (same frame as moon
    /// positions).  Uses the classical formula:
    ///
    /// h = Σ_i (3/2) × (M_moon_i / M_planet) × (R_planet / r_i)³ × R_planet × P₂(cos θ_i)
    ///
    /// where P₂ is the degree-2 Legendre polynomial and θ_i the zenith angle of
    /// moon i.  Positive values indicate tidal bulge; negative, tidal trough.
    pub fn tidal_height_at(&self, cell_pos: DVec3, t: f64) -> f64 {
        self.moons
            .iter()
            .zip(self.moon_positions_at(t).iter())
            .map(|(moon, pos): (&Moon, &DVec3)| {
                let r = pos.length();
                if r < 1.0 {
                    return 0.0;
                }
                let cos_theta = cell_pos.dot(*pos / r).clamp(-1.0, 1.0);
                let p2 = 1.5 * cos_theta * cos_theta - 0.5;
                1.5 * (moon.mass_kg / self.planet_mass_kg)
                    * (self.planet_radius_m / r).powi(3)
                    * self.planet_radius_m
                    * p2
            })
            .sum()
    }

    /// Find eclipse events in the time range `[start_s, end_s]` sampled at `dt_s`.
    ///
    /// Returns one entry per distinct eclipse (consecutive hits merged into one
    /// event showing the maximum magnitude).  Both solar eclipses (moon occulting
    /// the star) and lunar eclipses (planet shadow falling on the moon) are
    /// detected by comparing angular separations against the sum of apparent radii.
    pub fn find_eclipses(&self, start_s: f64, end_s: f64, dt_s: f64) -> Vec<EclipseEvent> {
        if self.moons.is_empty() || dt_s <= 0.0 {
            return vec![];
        }
        let star_dist = self.planet_orbit_m;
        let ang_r_star = (self.star.radius_m / star_dist).atan();
        // Shadow cone half-angle for lunar eclipses (small-angle approx).
        let shadow_half_angle =
            ((self.star.radius_m - self.planet_radius_m).abs() / star_dist).atan();

        let mut raw: Vec<EclipseEvent> = Vec::new();
        let mut t = start_s;
        while t <= end_s {
            let star_dir = self.star_direction_at(t);
            let moon_positions = self.moon_positions_at(t);

            for (moon, pos) in self.moons.iter().zip(moon_positions.iter()) {
                let moon_r: f64 = pos.length();
                if moon_r < 1.0 {
                    continue;
                }
                let moon_dir: DVec3 = *pos / moon_r;
                let ang_r_moon: f64 = (moon.radius_m / moon_r).atan();

                // ── Solar eclipse: moon between planet and star ─────────────
                let dot_star: f64 = moon_dir.dot(star_dir);
                if dot_star > 0.0 {
                    let ang_sep: f64 = dot_star.clamp(-1.0_f64, 1.0_f64).acos();
                    if ang_sep < ang_r_star + ang_r_moon {
                        let overlap = (ang_r_star + ang_r_moon - ang_sep).min(2.0 * ang_r_moon);
                        let magnitude = (overlap / (2.0 * ang_r_star)).clamp(0.0_f64, 1.0_f64);
                        raw.push(EclipseEvent {
                            time_s: t,
                            is_solar: true,
                            magnitude,
                        });
                    }
                }

                // ── Lunar eclipse: moon inside planet's umbra ───────────────
                let dot_anti: f64 = moon_dir.dot(-star_dir);
                if dot_anti > 0.0 {
                    let ang_off_axis: f64 = dot_anti.clamp(-1.0_f64, 1.0_f64).acos();
                    let umbra_r = shadow_half_angle + ang_r_moon;
                    if ang_off_axis < umbra_r {
                        let magnitude = (1.0_f64 - ang_off_axis / umbra_r).clamp(0.0_f64, 1.0_f64);
                        raw.push(EclipseEvent {
                            time_s: t,
                            is_solar: false,
                            magnitude,
                        });
                    }
                }
            }
            t += dt_s;
        }

        dedup_eclipse_events(raw, 2.0 * dt_s)
    }
}

/// Merge eclipse hits that are within `merge_window_s` of each other into a
/// single event (keeping the highest magnitude).
fn dedup_eclipse_events(mut events: Vec<EclipseEvent>, merge_window_s: f64) -> Vec<EclipseEvent> {
    if events.len() < 2 {
        return events;
    }
    events.sort_by(|a, b| a.time_s.partial_cmp(&b.time_s).unwrap());
    let mut merged: Vec<EclipseEvent> = Vec::new();
    for ev in events {
        match merged.last_mut() {
            Some(last)
                if last.is_solar == ev.is_solar && ev.time_s - last.time_s <= merge_window_s =>
            {
                last.magnitude = last.magnitude.max(ev.magnitude);
            }
            _ => merged.push(ev),
        }
    }
    merged
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Earth-like planet constants for constructing test systems.
    const M_EARTH: f64 = 5.972e24;
    const R_EARTH: f64 = 6.371e6;

    fn earth_moon() -> Moon {
        // Circular, equatorial orbit aligned with +X at t=0.
        let a = 3.844e8_f64;
        let period_s = TWO_PI * (a.powi(3) / (G * M_EARTH)).sqrt();
        Moon {
            mass_kg: 7.342e22,
            radius_m: 1.737e6,
            semi_major_axis_m: a,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            longitude_ascending_node_rad: 0.0,
            argument_of_periapsis_rad: 0.0,
            mean_anomaly_at_epoch_rad: 0.0, // starts at [a, 0, 0]
            orbital_period_s: period_s,
            albedo: 0.12,
            surface_color: [0.5, 0.5, 0.5],
        }
    }

    fn solar_eclipse_system() -> CelestialSystem {
        // Planet at 1 AU with a moon aligned with the star direction at t=0.
        let star = Star {
            mass_kg: M_SUN,
            radius_m: R_SUN,
            temperature_k: 5778.0,
            luminosity_w: L_SUN,
            color: [1.0, 1.0, 0.9],
            habitable_zone_inner_m: AU * 0.95,
        };
        CelestialSystem {
            star,
            moons: vec![earth_moon()],
            ring: None,
            planet_mass_kg: M_EARTH,
            planet_radius_m: R_EARTH,
            planet_orbit_m: AU,
            planet_orbital_period_s: 3.156e7, // ~1 year
            axial_tilt_rad: 0.4091,           // Earth's obliquity
        }
    }

    // ── Kepler solver ────────────────────────────────────────────────────────

    #[test]
    fn kepler_circular_orbit() {
        // Circular orbit: e=0, so E = M always.
        for &m in &[0.0, 0.5, 1.0, 2.0, std::f64::consts::PI] {
            let ea = solve_kepler(m, 0.0);
            assert!((ea - m).abs() < 1e-10, "E should equal M for e=0");
        }
    }

    #[test]
    fn kepler_satisfies_equation() {
        // Verify E - e·sin(E) = M for a variety of eccentric orbits.
        let cases = [(0.3, 1.0), (0.7, 2.5), (0.5, 0.1), (0.85, 3.0)];
        for (e, m) in cases {
            let ea = solve_kepler(m, e);
            let residual = (ea - e * ea.sin() - m).abs();
            assert!(
                residual < 1e-11,
                "Kepler residual {residual} too large for e={e}, M={m}"
            );
        }
    }

    // ── Moon position ────────────────────────────────────────────────────────

    #[test]
    fn moon_position_circular_equatorial_at_epoch() {
        // Circular, equatorial orbit with Ω=ω=M₀=0 → position at [a, 0, 0].
        let moon = earth_moon();
        let pos = moon_position(&moon, 0.0);
        let a = moon.semi_major_axis_m;
        assert!((pos.x - a).abs() / a < 1e-9, "x should be ~a");
        assert!(pos.y.abs() < 1.0, "y should be ~0");
        assert!(pos.z.abs() < 1.0, "z should be ~0");
    }

    #[test]
    fn moon_position_periodic() {
        // After exactly one orbital period the position must repeat.
        let moon = earth_moon();
        let t0 = 1.5e5; // arbitrary offset
        let t1 = t0 + moon.orbital_period_s;
        let p0 = moon_position(&moon, t0);
        let p1 = moon_position(&moon, t1);
        let dist = (p1 - p0).length();
        assert!(
            dist < 1.0,
            "Position should repeat after one period (drift {dist:.3} m)"
        );
    }

    #[test]
    fn moon_period_kepler_third_law() {
        // Kepler's 3rd: T² = 4π²a³ / (G·M_planet).
        let moon = earth_moon();
        let t_expected = TWO_PI * (moon.semi_major_axis_m.powi(3) / (G * M_EARTH)).sqrt();
        let ratio = moon.orbital_period_s / t_expected;
        assert!(
            (ratio - 1.0).abs() < 1e-9,
            "Period deviates from Kepler's 3rd: ratio={ratio}"
        );
    }

    // ── Star properties ──────────────────────────────────────────────────────

    #[test]
    fn star_luminosity_scales_with_mass() {
        // For main-sequence stars L ∝ M^4.  Compare two seeds that yield
        // different mass ratios and verify the luminosity ratio is consistent.
        let mut rng1 = SmallRng::seed_from_u64(1);
        let mut rng2 = SmallRng::seed_from_u64(999);
        let s1 = generate_star(&mut rng1);
        let s2 = generate_star(&mut rng2);
        let mass_ratio = s1.mass_kg / s2.mass_kg;
        let lum_ratio = s1.luminosity_w / s2.luminosity_w;
        let expected_lum_ratio = mass_ratio.powi(4);
        assert!(
            (lum_ratio / expected_lum_ratio - 1.0).abs() < 1e-9,
            "Luminosity ratio {lum_ratio:.4} should equal (M1/M2)^4 = {expected_lum_ratio:.4}"
        );
    }

    #[test]
    fn blackbody_hotter_star_is_bluer() {
        let cool = blackbody_color(3500.0); // M-dwarf, reddish
        let hot = blackbody_color(10000.0); // A-type, bluish-white
        assert!(hot[2] > cool[2], "Hotter star should have more blue");
        assert!(hot[0] < cool[0], "Hotter star should have less red");
    }

    // ── System generation ────────────────────────────────────────────────────

    #[test]
    fn system_is_deterministic() {
        let s1 = CelestialSystem::generate(M_EARTH, R_EARTH, 42);
        let s2 = CelestialSystem::generate(M_EARTH, R_EARTH, 42);
        assert_eq!(s1.moons.len(), s2.moons.len());
        assert!((s1.star.temperature_k - s2.star.temperature_k).abs() < 1e-9);
        assert!((s1.planet_orbit_m - s2.planet_orbit_m).abs() < 1.0);
    }

    #[test]
    fn moon_count_in_valid_range() {
        for seed in 0..50_u64 {
            let sys = CelestialSystem::generate(M_EARTH, R_EARTH, seed);
            assert!(sys.moons.len() <= 4, "Too many moons: {}", sys.moons.len());
        }
    }

    #[test]
    fn ring_probability_near_fifteen_percent() {
        let n = 200_u64;
        let ring_count = (0..n)
            .filter(|&s| {
                CelestialSystem::generate(M_EARTH, R_EARTH, s)
                    .ring
                    .is_some()
            })
            .count();
        let frac = ring_count as f64 / n as f64;
        // Allow ±10% around the target 15%.
        assert!(
            (0.05..=0.25).contains(&frac),
            "Ring fraction {frac:.2} outside expected range 5–25%"
        );
    }

    // ── Tidal forces ─────────────────────────────────────────────────────────

    #[test]
    fn tidal_max_at_sublunar_min_at_equator() {
        // Moon at [a, 0, 0] (earth_moon at t=0).
        let sys = solar_eclipse_system();
        let sub_lunar = DVec3::new(1.0, 0.0, 0.0); // θ = 0   → P₂ = +1.0
        let equatorial = DVec3::new(0.0, 1.0, 0.0); // θ = 90° → P₂ = -0.5
        let h_sub = sys.tidal_height_at(sub_lunar, 0.0);
        let h_eq = sys.tidal_height_at(equatorial, 0.0);
        assert!(h_sub > 0.0, "Sub-lunar tidal height should be positive");
        assert!(h_eq < 0.0, "Equatorial tidal height should be negative");
        assert!(h_sub > h_eq.abs(), "Sub-lunar bulge > equatorial trough");
    }

    #[test]
    fn tidal_anti_sublunar_equals_sublunar() {
        // P₂(cos θ) = P₂(-cos θ), so opposite point has same tidal height.
        let sys = solar_eclipse_system();
        let sub = DVec3::new(1.0, 0.0, 0.0);
        let anti = DVec3::new(-1.0, 0.0, 0.0);
        let h_sub = sys.tidal_height_at(sub, 0.0);
        let h_anti = sys.tidal_height_at(anti, 0.0);
        assert!(
            (h_sub - h_anti).abs() < 1e-6,
            "Sub-lunar ({h_sub:.4}) and anti-lunar ({h_anti:.4}) heights should be equal"
        );
    }

    // ── Eclipse detection ────────────────────────────────────────────────────

    #[test]
    fn solar_eclipse_detected_when_moon_aligned() {
        // At t=0 the moon is at [a, 0, 0] and the star direction is [1, 0, 0].
        // Perfect angular alignment → solar eclipse must be detected.
        let sys = solar_eclipse_system();
        let dt = 3600.0; // 1-hour steps
        let eclipses = sys.find_eclipses(0.0, dt * 5.0, dt);
        let solar: Vec<_> = eclipses.iter().filter(|e| e.is_solar).collect();
        assert!(
            !solar.is_empty(),
            "Solar eclipse must be detected when moon aligns with star"
        );
        assert!(
            solar[0].magnitude > 0.0,
            "Eclipse magnitude should be positive"
        );
    }

    #[test]
    fn no_eclipses_without_moons() {
        let star = Star {
            mass_kg: M_SUN,
            radius_m: R_SUN,
            temperature_k: 5778.0,
            luminosity_w: L_SUN,
            color: [1.0, 1.0, 0.9],
            habitable_zone_inner_m: AU * 0.95,
        };
        let sys = CelestialSystem {
            star,
            moons: vec![],
            ring: None,
            planet_mass_kg: M_EARTH,
            planet_radius_m: R_EARTH,
            planet_orbit_m: AU,
            planet_orbital_period_s: 3.156e7,
            axial_tilt_rad: 0.4091,
        };
        let eclipses = sys.find_eclipses(0.0, 1e8, 1e4);
        assert!(eclipses.is_empty(), "No moons → no eclipses");
    }
}
