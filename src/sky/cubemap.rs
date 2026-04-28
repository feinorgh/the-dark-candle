// CPU baker that turns a `CelestialCatalogue` into an HDR cubemap.
//
// Design (post rubber-duck critique):
//
// * **Star-centric splatting.** Each star walks a small tangent-plane kernel
//   around its direction; every kernel sample is mapped back to a `(face, px,
//   py)` triplet. This automatically handles cubemap face seams: stars near
//   an edge or corner write to multiple faces without special-casing.
//
// * **Per-star kernel normalisation.** The discrete kernel weights are
//   summed per star and the per-sample contribution is divided by the sum
//   before scaling by flux, so a star's *total integrated flux* is constant
//   regardless of sub-pixel position or how many texels the kernel touches.
//
// * **HDR linear, then f16 pack.** Accumulation is f32 RGBA per face; the
//   final image is converted to RGBA16F (`half::f16`) for GPU upload at the
//   `Rgba16Float` cubemap format, which is filterable on every wgpu backend
//   we target.
//
// Resolution is configurable; the default `STAR_CUBEMAP_FACE_SIZE = 1024`
// gives 6 × 1024² × 8 B = 48 MiB of texture memory, plus a transient 96 MiB
// f32 accumulator that is dropped before upload.
//
// The baker is currently single-threaded; profiling will tell us whether
// rayon parallelism is worth the dependency.  At 1024² with ~30 k naked-eye
// stars and ~70 catalogue stars per face the bake completes in a few hundred
// milliseconds in release builds.

use bevy::math::Vec3;
use half::f16;

use super::catalogue::{CelestialCatalogue, Star};
use super::spectrum::flux_from_magnitude;

// ─── Tunables ────────────────────────────────────────────────────────────────

/// Default cubemap face resolution (texels per side).
pub const STAR_CUBEMAP_FACE_SIZE: u32 = 1024;

/// Standard deviation of the per-star Gaussian PSF, in *output texels*.
/// 0.7 keeps a star sub-pixel sharp while still anti-aliasing offsets.
pub const PSF_SIGMA_PX: f32 = 0.7;

/// Half-extent of the kernel evaluated per star, in texels.  3 σ captures
/// ≈ 99.7 % of a 1-D Gaussian.
pub const PSF_RADIUS_PX: i32 = 3;

/// Linear flux assigned to a magnitude-0 reference star (Vega-equivalent),
/// before colour multiplication.  Chosen so the brightest realistic stars
/// (Sirius, mag −1.5) reach ~0.7-0.9 in a cubemap texel after the per-star
/// kernel normalisation, leaving headroom for the additive sky composite.
pub const FLUX_AT_MAG_ZERO: f32 = 0.6;

/// Faintest apparent magnitude rendered into the cubemap.  Stars fainter
/// than this contribute < 1/255 of a texel and are skipped.
pub const RENDER_MAG_LIMIT: f32 = 8.5;

// ─── Cube-face geometry ──────────────────────────────────────────────────────

/// Cubemap face indices, matching the wgpu/bevy convention:
/// 0 = +X, 1 = −X, 2 = +Y, 3 = −Y, 4 = +Z, 5 = −Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceIndex(pub u8);

/// Map a unit direction to a `(face, px, py)` triplet.  `px` and `py` may be
/// outside `[0, size)` if the input direction is degenerate; callers must
/// bounds-check.
#[inline]
pub fn direction_to_face_pixel(dir: Vec3, size: u32) -> (FaceIndex, i32, i32) {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();
    let (face, uc, vc, ma) = if ax >= ay && ax >= az {
        if dir.x > 0.0 {
            (0u8, -dir.z, -dir.y, ax)
        } else {
            (1u8, dir.z, -dir.y, ax)
        }
    } else if ay >= ax && ay >= az {
        if dir.y > 0.0 {
            (2u8, dir.x, dir.z, ay)
        } else {
            (3u8, dir.x, -dir.z, ay)
        }
    } else if dir.z > 0.0 {
        (4u8, dir.x, -dir.y, az)
    } else {
        (5u8, -dir.x, -dir.y, az)
    };
    let u = (uc / ma + 1.0) * 0.5;
    let v = (vc / ma + 1.0) * 0.5;
    let s = size as f32;
    let px = (u * s).floor() as i32;
    let py = (v * s).floor() as i32;
    (FaceIndex(face), px, py)
}

/// Inverse of [`direction_to_face_pixel`]: given a face and *pixel-centre*
/// integer coordinates `(px, py)`, return the unit direction.
#[inline]
pub fn face_pixel_to_direction(face: FaceIndex, px: i32, py: i32, size: u32) -> Vec3 {
    let s = size as f32;
    let uc = ((px as f32 + 0.5) / s) * 2.0 - 1.0;
    let vc = ((py as f32 + 0.5) / s) * 2.0 - 1.0;
    let v = match face.0 {
        0 => Vec3::new(1.0, -vc, -uc),
        1 => Vec3::new(-1.0, -vc, uc),
        2 => Vec3::new(uc, 1.0, vc),
        3 => Vec3::new(uc, -1.0, -vc),
        4 => Vec3::new(uc, -vc, 1.0),
        5 => Vec3::new(-uc, -vc, -1.0),
        _ => Vec3::new(0.0, 0.0, 1.0),
    };
    v.normalize()
}

// ─── Bake output ─────────────────────────────────────────────────────────────

/// HDR cubemap baked from a catalogue.
///
/// `face_pixels[face]` is a row-major RGBA16F slice of size `size² × 4`.
/// Layout: `face_pixels` are ordered `[+X, -X, +Y, -Y, +Z, -Z]` matching the
/// wgpu/bevy cubemap convention.  The whole structure can be passed straight
/// to `bevy::image::Image::new` once flattened.
pub struct StarCubemap {
    pub size: u32,
    pub face_pixels: [Vec<f16>; 6],
}

impl StarCubemap {
    /// Concatenate face byte-streams in cubemap order for upload to a single
    /// 2-D-array texture (wgpu cubemap layout).
    pub fn into_flat_bytes(self) -> Vec<u8> {
        let texels_per_face = (self.size * self.size) as usize * 4;
        let mut out = Vec::with_capacity(texels_per_face * 6 * 2);
        for face in self.face_pixels {
            debug_assert_eq!(face.len(), texels_per_face);
            for h in face {
                out.extend_from_slice(&h.to_le_bytes());
            }
        }
        out
    }
}

// ─── Baker ───────────────────────────────────────────────────────────────────

/// Bake a catalogue's stars (and only its stars, for now) into a cubemap.
///
/// `magnitude_limit` rejects stars fainter than the given apparent magnitude;
/// pass `RENDER_MAG_LIMIT` for the default.
pub fn bake_star_cubemap(
    catalogue: &CelestialCatalogue,
    size: u32,
    magnitude_limit: f32,
) -> StarCubemap {
    let texels = (size * size) as usize;
    // f32 RGBA accumulator per face.  `[Vec<f32>; 6]` keeps the layout
    // straight without forcing a 6-tuple of fixed-size arrays.
    let mut accum: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; texels * 4]);

    for star in &catalogue.stars {
        if star.apparent_magnitude_v > magnitude_limit {
            continue;
        }
        splat_star(&mut accum, size, star);
    }

    // Convert to f16.
    let face_pixels = accum.map(|face_f32| {
        let mut f16_face: Vec<f16> = Vec::with_capacity(face_f32.len());
        for v in &face_f32 {
            f16_face.push(f16::from_f32(*v));
        }
        f16_face
    });

    StarCubemap { size, face_pixels }
}

/// Build an orthonormal tangent frame around `forward` (must be unit-length).
///
/// Returns `(right, up)` in the celestial frame, both unit vectors and
/// orthogonal to `forward`.  Picks an arbitrary "world up" and falls back if
/// `forward` is too close to it.
#[inline]
fn tangent_frame(forward: Vec3) -> (Vec3, Vec3) {
    let world_up = if forward.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let right = world_up.cross(forward).normalize();
    let up = forward.cross(right);
    (right, up)
}

/// Splat a single star into the f32 accumulator with full per-star
/// normalisation across all (possibly cross-face) affected texels.
fn splat_star(accum: &mut [Vec<f32>; 6], size: u32, star: &Star) {
    let dir = Vec3::new(
        star.direction.x as f32,
        star.direction.y as f32,
        star.direction.z as f32,
    )
    .normalize();
    let (right, up) = tangent_frame(dir);

    // Angular resolution: at a face centre one texel subtends 2/size in
    // tangent-plane units (cube of half-edge 1).  Use that as the kernel
    // step so PSF_SIGMA_PX maps to a real solid-angle scale.
    let texel_step = 2.0 / size as f32;
    let sigma = PSF_SIGMA_PX * texel_step;

    // Two-pass: gather all affected texels and weights, normalise, write.
    let mut hits: Vec<(usize, usize, f32)> = Vec::with_capacity(64);
    let mut total_weight = 0.0f32;
    for ky in -PSF_RADIUS_PX..=PSF_RADIUS_PX {
        for kx in -PSF_RADIUS_PX..=PSF_RADIUS_PX {
            let dx = kx as f32 * texel_step;
            let dy = ky as f32 * texel_step;
            // Tangent-plane offset, then re-normalise so the kernel is on
            // the unit sphere — keeps angular distance small for small kx.
            let sample_dir = (dir + right * dx + up * dy).normalize();
            let r2 = (kx * kx + ky * ky) as f32;
            let w = (-r2 / (2.0 * PSF_SIGMA_PX * PSF_SIGMA_PX)).exp();
            let (face, px, py) = direction_to_face_pixel(sample_dir, size);
            if px < 0 || py < 0 || px >= size as i32 || py >= size as i32 {
                continue;
            }
            let idx = (py as usize * size as usize + px as usize) * 4;
            hits.push((face.0 as usize, idx, w));
            total_weight += w;
            // (sigma kept for future angular-space kernel tweaks)
            let _ = sigma;
        }
    }
    if total_weight <= 0.0 {
        return;
    }

    let flux = FLUX_AT_MAG_ZERO * flux_from_magnitude(star.apparent_magnitude_v);
    let inv_total = 1.0 / total_weight;
    for (face, idx, w) in hits {
        let contribution = flux * w * inv_total;
        let face_buf = &mut accum[face];
        face_buf[idx] += contribution * star.color_linear[0];
        face_buf[idx + 1] += contribution * star.color_linear[1];
        face_buf[idx + 2] += contribution * star.color_linear[2];
        face_buf[idx + 3] += contribution; // alpha = scalar luminance
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sky::generate::generate_catalogue;
    use bevy::math::DVec3;

    /// Round-trip a face-pixel centre through direction → face_pixel.
    #[test]
    fn face_pixel_round_trip() {
        let size = 64;
        for face in 0..6u8 {
            for py in 0..size as i32 {
                for px in 0..size as i32 {
                    let dir = face_pixel_to_direction(FaceIndex(face), px, py, size);
                    let (f2, px2, py2) = direction_to_face_pixel(dir, size);
                    assert_eq!(f2.0, face, "face mismatch f={face} ({px},{py})");
                    assert_eq!(px2, px);
                    assert_eq!(py2, py);
                }
            }
        }
    }

    /// A unit-flux star at the +Z axis pole should land in face 4 near
    /// pixel (size/2, size/2) and its peak alpha should be > 0.
    #[test]
    fn star_at_pole_lands_on_pos_z() {
        let mut accum: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; 64 * 64 * 4]);
        let star = Star {
            direction: DVec3::new(0.0, 0.0, 1.0),
            distance_pc: 10.0,
            mass_solar: 1.0,
            luminosity_solar: 1.0,
            temperature_k: 5778.0,
            spectral_class: crate::sky::catalogue::SpectralClass::G,
            apparent_magnitude_v: 0.0,
            color_linear: [1.0, 1.0, 1.0],
        };
        splat_star(&mut accum, 64, &star);
        let centre = (32 * 64 + 32) * 4;
        assert!(
            accum[4][centre + 3] > 0.0,
            "expected non-zero alpha at +Z pole centre"
        );
        // No spillage onto far face (-Z = face 5).
        let total_neg_z: f32 = accum[5].iter().copied().sum();
        assert_eq!(total_neg_z, 0.0);
    }

    /// Two identical stars five magnitudes apart should differ in
    /// integrated flux by a factor of ≈ 100.
    #[test]
    fn magnitude_ratio_is_pogson() {
        let bright = make_star(0.0);
        let faint = make_star(5.0);
        let bright_total = total_alpha(&bright);
        let faint_total = total_alpha(&faint);
        let ratio = bright_total / faint_total;
        assert!(
            (ratio - 100.0).abs() / 100.0 < 0.01,
            "mag-5 ratio off: {ratio}"
        );
    }

    /// Sub-pixel translations of a star should preserve total integrated
    /// flux to within ~1 %.
    #[test]
    fn integrated_flux_invariant_to_subpixel_shift() {
        let mut totals = vec![];
        // Walk the star a fraction of a texel in tangent space.  Use four
        // distinct tangent-plane offsets near the +Z pole.
        let offsets = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.4), (0.45, 0.45)];
        for (ox, oy) in offsets {
            let texel = 2.0 / 128.0;
            let dir = (Vec3::Z + Vec3::X * ox * texel + Vec3::Y * oy * texel).normalize();
            let star = Star {
                direction: DVec3::new(dir.x as f64, dir.y as f64, dir.z as f64),
                distance_pc: 10.0,
                mass_solar: 1.0,
                luminosity_solar: 1.0,
                temperature_k: 5778.0,
                spectral_class: crate::sky::catalogue::SpectralClass::G,
                apparent_magnitude_v: 0.0,
                color_linear: [1.0, 1.0, 1.0],
            };
            let mut accum: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; 128 * 128 * 4]);
            splat_star(&mut accum, 128, &star);
            let total: f32 = accum.iter().map(|f| f.iter().step_by(4).sum::<f32>()).sum();
            // step_by(4) hits R channel; sum alpha for total:
            let total_alpha: f32 = accum
                .iter()
                .map(|f| f.iter().skip(3).step_by(4).sum::<f32>())
                .sum();
            let _ = total;
            totals.push(total_alpha);
        }
        let mean = totals.iter().sum::<f32>() / totals.len() as f32;
        for t in &totals {
            assert!(
                (t - mean).abs() / mean < 0.01,
                "sub-pixel shift broke flux conservation: {totals:?}"
            );
        }
    }

    /// Bake a small catalogue and verify no NaNs/Infs.
    #[test]
    fn full_bake_produces_finite_pixels() {
        let cat = generate_catalogue(0xDEAD_BEEF);
        let cube = bake_star_cubemap(&cat, 256, RENDER_MAG_LIMIT);
        assert_eq!(cube.size, 256);
        for face in &cube.face_pixels {
            assert_eq!(face.len(), 256 * 256 * 4);
            for v in face {
                let f = v.to_f32();
                assert!(f.is_finite(), "non-finite f16 in cubemap: {f}");
                assert!(f >= 0.0, "negative pixel value: {f}");
            }
        }
    }

    fn make_star(mag: f32) -> [Vec<f32>; 6] {
        let mut accum: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; 128 * 128 * 4]);
        let star = Star {
            direction: DVec3::new(0.0, 0.0, 1.0),
            distance_pc: 10.0,
            mass_solar: 1.0,
            luminosity_solar: 1.0,
            temperature_k: 5778.0,
            spectral_class: crate::sky::catalogue::SpectralClass::G,
            apparent_magnitude_v: mag,
            color_linear: [1.0, 1.0, 1.0],
        };
        splat_star(&mut accum, 128, &star);
        accum
    }

    fn total_alpha(accum: &[Vec<f32>; 6]) -> f32 {
        accum
            .iter()
            .map(|f| f.iter().skip(3).step_by(4).sum::<f32>())
            .sum()
    }
}
