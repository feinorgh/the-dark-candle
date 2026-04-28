// Splat nebulae and remote galaxies into the cubemap accumulator.
//
// Both are rendered as anisotropic Gaussian "blobs" on a tangent plane
// around the object's celestial direction:
//
//   I(x, y) = peak · exp( -0.5 · ( (x/σ_a)² + (y/σ_b)² ) )
//
// where (x, y) are tangent-plane offsets along the major / minor axes
// (oriented by `orientation_rad`), σ_a = `angular_radius_rad`, and
// σ_b = σ_a · `axial_ratio`.  Each kernel sample direction is mapped
// back to (face, px, py) via `direction_to_face_pixel`, so blobs that
// straddle face seams write to multiple faces automatically.
//
// Dark nebulae behave differently: they *attenuate* the in-band signal
// rather than adding to it, which is why they need a dedicated pass
// run after every additive contributor.

use bevy::math::Vec3;

use super::catalogue::{CelestialCatalogue, Galaxy, Nebula, NebulaKind};
use super::cubemap::direction_to_face_pixel;
use super::spectrum::flux_from_magnitude;

// ─── Tunables ────────────────────────────────────────────────────────────────

/// Peak linear flux of an emission nebula at unit angular density.  Calibrated
/// alongside `surface_brightness` so a magnitude-21 nebula peaks near the same
/// per-texel intensity as a magnitude-3 star.
const NEBULA_PEAK_AT_REF_MAG: f32 = 0.7;

/// Reference apparent surface brightness that maps to `*_PEAK_AT_REF_MAG`.
const NEBULA_REF_SURFACE_MAG: f32 = 21.0;

/// Number of samples on each axis of the kernel.  Total samples = (2N+1)².
/// 8 → 17² = 289 samples per object; ample for smooth Gaussian profiles.
const KERNEL_HALF_RES: i32 = 8;

/// How many σ the kernel extends in each axis (3 captures > 99 % of energy).
const KERNEL_SIGMA_EXTENT: f32 = 3.0;

/// Galaxy peak modifier vs nebula peak at the same brightness — galaxies
/// are visually fainter per arc-second of solid angle than emission
/// nebulae of the same total flux because their light is distributed
/// over a stellar profile.
const GALAXY_PEAK_SCALE: f32 = 0.45;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Splat all *additive* nebulae (Emission / Reflection / Planetary) into
/// the f32 RGBA accumulator.  Dark nebulae are skipped here.
pub fn splat_additive_nebulae(
    accum: &mut [Vec<f32>; 6],
    size: u32,
    catalogue: &CelestialCatalogue,
) {
    for n in &catalogue.nebulae {
        if matches!(n.kind, NebulaKind::Dark) {
            continue;
        }
        splat_nebula_additive(accum, size, n);
    }
}

/// Apply all dark nebulae as multiplicative attenuators.  Run *after*
/// every additive contribution (stars, host galaxy, additive nebulae,
/// galaxies) so dark clouds correctly silhouette the background.
pub fn splat_dark_nebulae(accum: &mut [Vec<f32>; 6], size: u32, catalogue: &CelestialCatalogue) {
    for n in &catalogue.nebulae {
        if matches!(n.kind, NebulaKind::Dark) {
            splat_dark_nebula(accum, size, n);
        }
    }
}

/// Splat all remote galaxies as additive Gaussian blobs.
pub fn splat_galaxies(accum: &mut [Vec<f32>; 6], size: u32, catalogue: &CelestialCatalogue) {
    for g in &catalogue.galaxies {
        splat_galaxy(accum, size, g);
    }
}

// ─── Internals ───────────────────────────────────────────────────────────────

#[inline]
fn tangent_basis(forward: Vec3, orientation_rad: f32) -> (Vec3, Vec3) {
    // Pick a stable reference up vector to start from; orient in-plane
    // by `orientation_rad` so position-angle is consistent.
    let ref_up = if forward.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let east = ref_up.cross(forward).normalize();
    let north = forward.cross(east);
    let (s, c) = orientation_rad.sin_cos();
    let major = east * c + north * s;
    let minor = -east * s + north * c;
    (major, minor)
}

fn splat_nebula_additive(accum: &mut [Vec<f32>; 6], size: u32, n: &Nebula) {
    let dir = Vec3::new(
        n.direction.x as f32,
        n.direction.y as f32,
        n.direction.z as f32,
    )
    .normalize();
    let (axis_a, axis_b) = tangent_basis(dir, n.orientation_rad);

    let sigma_a = n.angular_radius_rad.max(1.0e-5);
    let sigma_b = (sigma_a * n.axial_ratio).max(1.0e-5);

    // Effective per-texel peak from the surface-brightness magnitude.
    let peak =
        NEBULA_PEAK_AT_REF_MAG * flux_from_magnitude(n.surface_brightness - NEBULA_REF_SURFACE_MAG);

    splat_anisotropic_gaussian(
        accum,
        size,
        dir,
        axis_a,
        axis_b,
        sigma_a,
        sigma_b,
        peak,
        n.color_linear,
        true,
    );
}

fn splat_dark_nebula(accum: &mut [Vec<f32>; 6], size: u32, n: &Nebula) {
    let dir = Vec3::new(
        n.direction.x as f32,
        n.direction.y as f32,
        n.direction.z as f32,
    )
    .normalize();
    let (axis_a, axis_b) = tangent_basis(dir, n.orientation_rad);

    let sigma_a = n.angular_radius_rad.max(1.0e-5);
    let sigma_b = (sigma_a * n.axial_ratio).max(1.0e-5);

    // Dark-nebula opacity: 0 at the edge, up to ~0.85 at the centre.
    let core_opacity: f32 = 0.85;

    let dx_step = sigma_a * KERNEL_SIGMA_EXTENT / KERNEL_HALF_RES as f32;
    let dy_step = sigma_b * KERNEL_SIGMA_EXTENT / KERNEL_HALF_RES as f32;
    let inv_2sigma2_a = 1.0 / (2.0 * sigma_a * sigma_a);
    let inv_2sigma2_b = 1.0 / (2.0 * sigma_b * sigma_b);

    for ky in -KERNEL_HALF_RES..=KERNEL_HALF_RES {
        for kx in -KERNEL_HALF_RES..=KERNEL_HALF_RES {
            let dx = kx as f32 * dx_step;
            let dy = ky as f32 * dy_step;
            let g = (-(dx * dx * inv_2sigma2_a + dy * dy * inv_2sigma2_b)).exp();
            let opacity = core_opacity * g;
            let sample_dir = (dir + axis_a * dx + axis_b * dy).normalize();
            let (face, px, py) = direction_to_face_pixel(sample_dir, size);
            if px < 0 || py < 0 || px >= size as i32 || py >= size as i32 {
                continue;
            }
            let idx = (py as usize * size as usize + px as usize) * 4;
            let buf = &mut accum[face.0 as usize];
            // Channel-tinted attenuation: brighter colour channels are
            // attenuated less (interstellar reddening preserves red).
            let att_r = 1.0 - opacity * (1.0 - n.color_linear[0]);
            let att_g = 1.0 - opacity * (1.0 - n.color_linear[1]);
            let att_b = 1.0 - opacity * (1.0 - n.color_linear[2]);
            buf[idx] *= att_r.max(0.0);
            buf[idx + 1] *= att_g.max(0.0);
            buf[idx + 2] *= att_b.max(0.0);
            buf[idx + 3] *= (1.0 - opacity).max(0.0);
        }
    }
}

fn splat_galaxy(accum: &mut [Vec<f32>; 6], size: u32, g: &Galaxy) {
    let dir = Vec3::new(
        g.direction.x as f32,
        g.direction.y as f32,
        g.direction.z as f32,
    )
    .normalize();
    let (axis_a, axis_b) = tangent_basis(dir, g.orientation_rad);

    let sigma_a = g.angular_radius_rad.max(1.0e-5);
    let sigma_b = (sigma_a * g.axial_ratio).max(1.0e-5);

    // Apparent magnitude → linear flux density, scaled to galaxy regime.
    let peak = GALAXY_PEAK_SCALE * flux_from_magnitude(g.apparent_magnitude_v);

    splat_anisotropic_gaussian(
        accum,
        size,
        dir,
        axis_a,
        axis_b,
        sigma_a,
        sigma_b,
        peak,
        g.color_linear,
        true,
    );
}

#[allow(clippy::too_many_arguments)]
fn splat_anisotropic_gaussian(
    accum: &mut [Vec<f32>; 6],
    size: u32,
    centre: Vec3,
    axis_a: Vec3,
    axis_b: Vec3,
    sigma_a: f32,
    sigma_b: f32,
    peak: f32,
    color_linear: [f32; 3],
    additive: bool,
) {
    if peak <= 0.0 || !additive {
        return;
    }
    let dx_step = sigma_a * KERNEL_SIGMA_EXTENT / KERNEL_HALF_RES as f32;
    let dy_step = sigma_b * KERNEL_SIGMA_EXTENT / KERNEL_HALF_RES as f32;
    let inv_2sigma2_a = 1.0 / (2.0 * sigma_a * sigma_a);
    let inv_2sigma2_b = 1.0 / (2.0 * sigma_b * sigma_b);

    // Two-pass — collect (face, idx, weight), normalise so the integrated
    // flux is independent of how many texels the kernel resolves to.
    let mut hits: Vec<(usize, usize, f32)> = Vec::with_capacity(64);
    let mut total_w = 0.0f32;
    for ky in -KERNEL_HALF_RES..=KERNEL_HALF_RES {
        for kx in -KERNEL_HALF_RES..=KERNEL_HALF_RES {
            let dx = kx as f32 * dx_step;
            let dy = ky as f32 * dy_step;
            let w = (-(dx * dx * inv_2sigma2_a + dy * dy * inv_2sigma2_b)).exp();
            let sample_dir = (centre + axis_a * dx + axis_b * dy).normalize();
            let (face, px, py) = direction_to_face_pixel(sample_dir, size);
            if px < 0 || py < 0 || px >= size as i32 || py >= size as i32 {
                continue;
            }
            let idx = (py as usize * size as usize + px as usize) * 4;
            hits.push((face.0 as usize, idx, w));
            total_w += w;
        }
    }
    if total_w <= 0.0 {
        return;
    }
    let inv_total = 1.0 / total_w;
    for (face, idx, w) in hits {
        let c = peak * w * inv_total;
        let buf = &mut accum[face];
        buf[idx] += c * color_linear[0];
        buf[idx + 1] += c * color_linear[1];
        buf[idx + 2] += c * color_linear[2];
        buf[idx + 3] += c;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sky::catalogue::{CelestialCatalogue, GalaxyKind, HostGalaxy};
    use bevy::math::DVec3;

    fn empty_cat() -> CelestialCatalogue {
        CelestialCatalogue {
            stars: Vec::new(),
            nebulae: Vec::new(),
            galaxies: Vec::new(),
            host_galaxy: HostGalaxy {
                plane_normal: DVec3::new(0.0, 1.0, 0.0),
                bulge_direction: DVec3::new(0.0, 0.0, 1.0),
                bulge_radius_rad: 0.4,
                disk_thickness_rad: 0.10,
                seed: 0,
            },
            generator_seed: 0,
        }
    }

    fn make_accum(size: u32) -> [Vec<f32>; 6] {
        std::array::from_fn(|_| vec![0.0; (size * size) as usize * 4])
    }

    fn total_alpha(accum: &[Vec<f32>; 6]) -> f32 {
        accum
            .iter()
            .map(|f| f.iter().skip(3).step_by(4).sum::<f32>())
            .sum()
    }

    #[test]
    fn additive_nebula_adds_flux() {
        let mut cat = empty_cat();
        cat.nebulae.push(Nebula {
            direction: DVec3::new(0.0, 0.0, 1.0),
            angular_radius_rad: 0.05,
            axial_ratio: 1.0,
            orientation_rad: 0.0,
            kind: NebulaKind::Emission,
            spectrum_peak_nm: 656.3,
            surface_brightness: 20.0,
            color_linear: [1.0, 0.3, 0.4],
            texture_seed: 0,
        });
        let size = 64u32;
        let mut accum = make_accum(size);
        splat_additive_nebulae(&mut accum, size, &cat);
        assert!(total_alpha(&accum) > 0.0);
    }

    #[test]
    fn dark_nebula_attenuates_existing_signal() {
        let size = 64u32;
        let mut accum = make_accum(size);
        // Pre-fill +Z face with uniform 1.0 in alpha.
        for v in accum[4].iter_mut() {
            *v = 1.0;
        }
        let mut cat = empty_cat();
        cat.nebulae.push(Nebula {
            direction: DVec3::new(0.0, 0.0, 1.0),
            angular_radius_rad: 0.10,
            axial_ratio: 1.0,
            orientation_rad: 0.0,
            kind: NebulaKind::Dark,
            spectrum_peak_nm: 550.0,
            surface_brightness: 25.0,
            color_linear: [1.0, 1.0, 1.0],
            texture_seed: 0,
        });
        splat_dark_nebulae(&mut accum, size, &cat);
        // Pixel at face centre should now be < 1.0.
        let centre_alpha =
            accum[4][((size as usize) * (size as usize) / 2 + (size as usize) / 2) * 4 + 3];
        // Centre pixel at exact midpoint:
        let cx = (size / 2) as usize;
        let cy = (size / 2) as usize;
        let idx = (cy * size as usize + cx) * 4 + 3;
        let centre = accum[4][idx];
        let _ = centre_alpha;
        assert!(centre < 1.0, "dark nebula should attenuate centre pixel");
        assert!(centre >= 0.0, "attenuation must clamp non-negative");
    }

    #[test]
    fn galaxy_adds_flux_at_centre() {
        let mut cat = empty_cat();
        cat.galaxies.push(Galaxy {
            direction: DVec3::new(1.0, 0.0, 0.0),
            angular_radius_rad: 0.01,
            axial_ratio: 0.6,
            orientation_rad: 0.5,
            kind: GalaxyKind::Spiral,
            redshift_z: 0.001,
            apparent_magnitude_v: 8.0,
            color_linear: [0.85, 0.90, 1.00],
        });
        let size = 64u32;
        let mut accum = make_accum(size);
        splat_galaxies(&mut accum, size, &cat);
        // Face 0 = +X, centre pixel should be brightest.
        let cx = (size / 2) as usize;
        let cy = (size / 2) as usize;
        let idx = (cy * size as usize + cx) * 4 + 3;
        let centre = accum[0][idx];
        assert!(
            centre > 0.0,
            "galaxy should deposit flux at face-centre pixel"
        );
    }

    #[test]
    fn produces_finite_pixels() {
        let mut cat = empty_cat();
        cat.nebulae.push(Nebula {
            direction: DVec3::new(0.7, 0.3, 0.6).normalize(),
            angular_radius_rad: 0.03,
            axial_ratio: 0.7,
            orientation_rad: 1.2,
            kind: NebulaKind::Reflection,
            spectrum_peak_nm: 450.0,
            surface_brightness: 22.0,
            color_linear: [0.55, 0.7, 1.0],
            texture_seed: 0,
        });
        cat.galaxies.push(Galaxy {
            direction: DVec3::new(-0.4, 0.6, 0.7).normalize(),
            angular_radius_rad: 0.005,
            axial_ratio: 0.4,
            orientation_rad: 0.3,
            kind: GalaxyKind::Elliptical,
            redshift_z: 0.01,
            apparent_magnitude_v: 12.0,
            color_linear: [1.0, 0.88, 0.7],
        });
        let size = 32u32;
        let mut accum = make_accum(size);
        splat_additive_nebulae(&mut accum, size, &cat);
        splat_galaxies(&mut accum, size, &cat);
        for face in &accum {
            for v in face {
                assert!(v.is_finite() && *v >= 0.0);
            }
        }
    }
}
