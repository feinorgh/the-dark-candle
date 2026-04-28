// Procedural host-galaxy diffuse band.
//
// Splats the analytical galactic disk + bulge + 3-D dust noise directly
// into the f32 cubemap accumulator before f16 packing.  The model is
// fully procedural (driven by `HostGalaxy` parameters that were
// randomised at catalogue generation), with no Milky-Way-specific
// constants.
//
// Why per-pixel direct evaluation (vs splatting discrete blobs):
//   * The diffuse component is *continuous*: a smooth band, not a
//     finite set of objects.
//   * Sampling per pixel from the *normalised 3-D direction* makes the
//     noise field consistent across cubemap face seams — opposing
//     edges of two faces resolve to the same world direction and
//     therefore the same density value.
//
// Cost: 6 × size² pixel evaluations.  At the default 1024² that's
// ~6 M pixels, each ≤ 4 fbm octaves of value noise.  Single-threaded
// release-mode bake takes well under a second on commodity hardware.

use bevy::math::Vec3;

use super::catalogue::HostGalaxy;
use super::cubemap::{FaceIndex, face_pixel_to_direction};

// ─── Tunables ────────────────────────────────────────────────────────────────

/// Peak diffuse linear flux at the centre of the galactic bulge.  Sits
/// well below the per-texel flux of the dimmest naked-eye stars so the
/// band coexists with starlight rather than washing it out.
pub const HOST_GALAXY_BULGE_PEAK: f32 = 0.18;

/// Peak diffuse linear flux on the disk midline, away from the bulge.
pub const HOST_GALAXY_DISK_PEAK: f32 = 0.06;

/// Slightly cool / blue-white tint of the disk light (linear sRGB).
pub const DISK_COLOR_LINEAR: [f32; 3] = [0.78, 0.82, 0.95];

/// Warmer tint of the bulge, dominated by old stellar populations.
pub const BULGE_COLOR_LINEAR: [f32; 3] = [1.00, 0.88, 0.70];

/// Spatial frequency of the base octave of dust noise, in cycles per
/// unit-sphere radius.  Larger ⇒ finer-grained dust structure.
const NOISE_BASE_FREQUENCY: f32 = 4.0;

/// Number of fbm octaves used to sculpt the dust lanes.
const NOISE_OCTAVES: u32 = 4;

/// Modulation depth of the noise on the disk intensity.  0 = uniform
/// band, 1 = full dynamic range; 0.55 leaves ≈ 10 % minimum visibility
/// in dark lanes.
const DISK_NOISE_DEPTH: f32 = 0.45;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Add the procedural host-galaxy diffuse band to an existing f32
/// RGBA-per-face cubemap accumulator.  Idempotent w.r.t. cubemap face
/// boundaries: the noise field is sampled from the normalised 3-D
/// direction, so adjacent faces share values along seams.
pub fn splat_host_galaxy(accum: &mut [Vec<f32>; 6], size: u32, hg: &HostGalaxy) {
    let plane_n = Vec3::new(
        hg.plane_normal.x as f32,
        hg.plane_normal.y as f32,
        hg.plane_normal.z as f32,
    )
    .normalize();
    let bulge_dir = Vec3::new(
        hg.bulge_direction.x as f32,
        hg.bulge_direction.y as f32,
        hg.bulge_direction.z as f32,
    )
    .normalize();

    let disk_thickness = hg.disk_thickness_rad.max(0.01);
    let inv_thickness = 1.0 / disk_thickness;

    let bulge_sigma = hg.bulge_radius_rad.max(0.05);
    let inv_2_sigma2_bulge = 1.0 / (2.0 * bulge_sigma * bulge_sigma);

    let seed = hg.seed;
    let s = size as i32;

    for face in 0..6u8 {
        let buf = &mut accum[face as usize];
        for py in 0..s {
            for px in 0..s {
                let dir = face_pixel_to_direction(FaceIndex(face), px, py, size);

                // Galactic latitude — strong exponential falloff with
                // |sin b|; mathematically equivalent to a hyperbolic-secant
                // disk profile to first order near b ≈ 0.
                let sin_b = dir.dot(plane_n).clamp(-1.0, 1.0);
                let disk_falloff = (-sin_b.abs() * inv_thickness).exp();

                // Bulge — Gaussian in true angle from the bulge direction.
                let cos_to_bulge = dir.dot(bulge_dir).clamp(-1.0, 1.0);
                let angle_bulge = cos_to_bulge.acos();
                let bulge_falloff = (-angle_bulge * angle_bulge * inv_2_sigma2_bulge).exp();

                // Dust lanes: fbm of value noise driven by the unit
                // direction, scaled to NOISE_BASE_FREQUENCY cycles/radius.
                let n = fbm3(dir * NOISE_BASE_FREQUENCY, seed, NOISE_OCTAVES);
                // Map [-1, 1] → [1 - depth, 1] so the noise only *removes*
                // light from the disk (dust extinction), never adds.
                let dust = 1.0 - DISK_NOISE_DEPTH * (0.5 - 0.5 * n);

                let disk_i = HOST_GALAXY_DISK_PEAK * disk_falloff * dust;
                let bulge_i = HOST_GALAXY_BULGE_PEAK * bulge_falloff;

                let r = disk_i * DISK_COLOR_LINEAR[0] + bulge_i * BULGE_COLOR_LINEAR[0];
                let g = disk_i * DISK_COLOR_LINEAR[1] + bulge_i * BULGE_COLOR_LINEAR[1];
                let b = disk_i * DISK_COLOR_LINEAR[2] + bulge_i * BULGE_COLOR_LINEAR[2];
                let a = disk_i + bulge_i;

                let idx = (py as usize * size as usize + px as usize) * 4;
                buf[idx] += r;
                buf[idx + 1] += g;
                buf[idx + 2] += b;
                buf[idx + 3] += a;
            }
        }
    }
}

// ─── 3-D value noise ─────────────────────────────────────────────────────────

#[inline]
fn hash3(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    // Wang-style integer hash → uniform f32 in [0, 1).
    let mut h = seed
        ^ (ix as u32).wrapping_mul(0x8da6_b343)
        ^ (iy as u32).wrapping_mul(0xd816_3841)
        ^ (iz as u32).wrapping_mul(0xcb1a_b31f);
    h = h.wrapping_mul(1_274_126_177);
    h ^= h >> 15;
    h = h.wrapping_mul(2_246_822_519);
    h ^= h >> 13;
    h = h.wrapping_mul(3_266_489_917);
    h ^= h >> 16;
    ((h & 0x00FF_FFFF) as f32) / (0x0100_0000_u32 as f32)
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise_3d(p: Vec3, seed: u32) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let xf = p.x - xi;
    let yf = p.y - yi;
    let zf = p.z - zi;
    let u = smoothstep(xf);
    let v = smoothstep(yf);
    let w = smoothstep(zf);
    let ix = xi as i32;
    let iy = yi as i32;
    let iz = zi as i32;
    let c000 = hash3(ix, iy, iz, seed);
    let c100 = hash3(ix + 1, iy, iz, seed);
    let c010 = hash3(ix, iy + 1, iz, seed);
    let c110 = hash3(ix + 1, iy + 1, iz, seed);
    let c001 = hash3(ix, iy, iz + 1, seed);
    let c101 = hash3(ix + 1, iy, iz + 1, seed);
    let c011 = hash3(ix, iy + 1, iz + 1, seed);
    let c111 = hash3(ix + 1, iy + 1, iz + 1, seed);
    let x00 = c000 + (c100 - c000) * u;
    let x10 = c010 + (c110 - c010) * u;
    let x01 = c001 + (c101 - c001) * u;
    let x11 = c011 + (c111 - c011) * u;
    let y0 = x00 + (x10 - x00) * v;
    let y1 = x01 + (x11 - x01) * v;
    let z0 = y0 + (y1 - y0) * w;
    z0 * 2.0 - 1.0
}

fn fbm3(p: Vec3, seed: u32, octaves: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;
    let mut norm = 0.0f32;
    for o in 0..octaves {
        sum += amp * value_noise_3d(p * freq, seed.wrapping_add(o.wrapping_mul(0x9e37_79b9)));
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm > 0.0 { sum / norm } else { 0.0 }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::DVec3;

    fn dummy_hg() -> HostGalaxy {
        HostGalaxy {
            // Plane = equator (normal = +Y).
            plane_normal: DVec3::new(0.0, 1.0, 0.0),
            // Bulge along +Z.
            bulge_direction: DVec3::new(0.0, 0.0, 1.0),
            bulge_radius_rad: 0.4,
            disk_thickness_rad: 0.10,
            seed: 0xC0FF_EE42,
        }
    }

    fn make_accum(size: u32) -> [Vec<f32>; 6] {
        std::array::from_fn(|_| vec![0.0; (size * size) as usize * 4])
    }

    fn alpha_at(accum: &[Vec<f32>; 6], face: u8, size: u32, px: i32, py: i32) -> f32 {
        let idx = (py as usize * size as usize + px as usize) * 4 + 3;
        accum[face as usize][idx]
    }

    /// Pixels close to the galactic plane should be brighter on average
    /// than pixels far from it (large |b|).
    #[test]
    fn plane_is_brighter_than_poles() {
        let size = 64u32;
        let mut accum = make_accum(size);
        splat_host_galaxy(&mut accum, size, &dummy_hg());

        // Face 2 = +Y: the *galactic pole* in this configuration.
        let pole_total: f32 = accum[2].iter().skip(3).step_by(4).sum();
        // Face 4 = +Z: contains the bulge centre and lies in the plane.
        let plane_total: f32 = accum[4].iter().skip(3).step_by(4).sum();

        assert!(
            plane_total > pole_total * 5.0,
            "plane (face +Z, total {plane_total}) should dominate pole (face +Y, total {pole_total})",
        );
    }

    /// The pixel closest to the bulge direction should be brighter than
    /// any pixel on the plane far from the bulge.
    #[test]
    fn bulge_is_local_maximum() {
        let size = 64u32;
        let mut accum = make_accum(size);
        splat_host_galaxy(&mut accum, size, &dummy_hg());

        // Centre of +Z face → +Z direction → bulge centre.
        let centre = alpha_at(&accum, 4, size, (size / 2) as i32, (size / 2) as i32);
        // Centre of +X face → in the plane, ~90° from bulge.
        let on_plane_far = alpha_at(&accum, 0, size, (size / 2) as i32, (size / 2) as i32);

        assert!(
            centre > on_plane_far * 1.5,
            "bulge centre ({centre}) should clearly exceed off-bulge plane pixel ({on_plane_far})",
        );
    }

    /// Cubemap face seams must agree to floating-point tolerance: the
    /// noise field is sampled from the normalised 3-D direction so
    /// neighbouring faces resolve to the same density at shared edges.
    #[test]
    fn no_seams_at_face_edges() {
        let size = 32u32;
        let mut accum = make_accum(size);
        splat_host_galaxy(&mut accum, size, &dummy_hg());

        // Sample the right edge of +X (face 0) and the left edge of -Z
        // (face 5).  Both correspond to the same world direction up to
        // pixel discretisation; the alpha values should be very close.
        for py in 0..size as i32 {
            let a_pos_x_right = alpha_at(&accum, 0, size, (size as i32) - 1, py);
            let a_neg_z_left = alpha_at(&accum, 5, size, 0, py);
            // Allow a small slack for pixel-grid discretisation.
            let denom = a_pos_x_right.max(a_neg_z_left).max(1e-6);
            let rel = (a_pos_x_right - a_neg_z_left).abs() / denom;
            assert!(
                rel < 0.20,
                "seam mismatch py={py}: pos_x_right={a_pos_x_right}, neg_z_left={a_neg_z_left}",
            );
        }
    }

    /// Pixels are non-negative and finite everywhere.
    #[test]
    fn full_bake_produces_finite_pixels() {
        let size = 32u32;
        let mut accum = make_accum(size);
        splat_host_galaxy(&mut accum, size, &dummy_hg());

        for face in &accum {
            for v in face {
                assert!(v.is_finite(), "non-finite pixel: {v}");
                assert!(*v >= 0.0, "negative pixel: {v}");
            }
        }
    }

    /// fbm output stays in roughly [-1, 1] for reasonable inputs.
    #[test]
    fn fbm_bounded() {
        for i in 0..50 {
            let p = Vec3::new(i as f32 * 0.31, i as f32 * -0.17, i as f32 * 0.93);
            let n = fbm3(p, 0xABCD_1234, 4);
            assert!(n.is_finite());
            assert!(n.abs() < 1.5, "fbm out of range: {n} at {p:?}");
        }
    }
}
