// Procedural caustic tile generator.
//
// Produces a square, seamlessly-tiling, zero-mean intensity field that
// approximates the shimmery web pattern cast onto an underwater seabed.
// The field is sampled as a 2-channel texture by `terrain_caustic.wgsl`
// to add caustic sparkle to underwater fragments.
//
// The pattern is derived from a 2-octave seamless Worley/cellular field
// using F2 - F1, which produces the characteristic bright-line lattice
// where two cell boundaries meet. The output is normalized to zero mean
// (so the shader's additive contribution does not double-count diffuse
// lighting from StandardMaterial) and clamped to a sane peak.
//
// Physical placement (scale_m, world UVs) is handled in the shader; this
// module only produces the unitless intensity tile.

use bevy::math::Vec2;

/// Number of feature points per cell side in the base Worley grid.
/// 1 point per cell is the standard seamless Worley arrangement.
const POINTS_PER_CELL: usize = 1;

/// Cells per side at octave 0. Octave 1 doubles this.
const BASE_CELLS: usize = 8;

/// Generate a `resolution x resolution` seamless caustic tile.
///
/// Returns row-major f32 values, zero-mean-normalized. Typical values
/// fall in roughly [-1.0, 4.0]; the long positive tail represents the
/// caustic bright lines.
///
/// `seed` selects a deterministic feature-point arrangement.
pub fn generate_caustic_tile(resolution: usize, seed: u32) -> Vec<f32> {
    assert!(resolution > 0, "resolution must be positive");

    let points_o0 = generate_feature_points(BASE_CELLS, seed);
    let points_o1 = generate_feature_points(BASE_CELLS * 2, seed.wrapping_add(0x9E37_79B9));

    let mut out = vec![0.0_f32; resolution * resolution];
    for y in 0..resolution {
        for x in 0..resolution {
            let uv = Vec2::new(x as f32 / resolution as f32, y as f32 / resolution as f32);
            let o0 = worley_f2_minus_f1(uv, BASE_CELLS, &points_o0);
            let o1 = worley_f2_minus_f1(uv, BASE_CELLS * 2, &points_o1);
            // Bright lines are where F2-F1 is small. Invert and raise to a
            // sharp power to make filaments visible. Mix octaves so the
            // pattern has both a broad lattice and finer ripples.
            let v0 = (1.0 - o0).clamp(0.0, 1.0).powf(6.0);
            let v1 = (1.0 - o1).clamp(0.0, 1.0).powf(8.0);
            out[y * resolution + x] = v0 + 0.5 * v1;
        }
    }

    // Zero-mean normalize so the shader can simply add this to the
    // diffuse term without skewing the average underwater brightness.
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    for v in &mut out {
        *v -= mean;
    }
    out
}

/// Generate `cells * cells * POINTS_PER_CELL` deterministic feature points
/// in unit-square coordinates, ready for seamless Worley wrap.
fn generate_feature_points(cells: usize, seed: u32) -> Vec<Vec2> {
    let mut points = Vec::with_capacity(cells * cells * POINTS_PER_CELL);
    for cy in 0..cells {
        for cx in 0..cells {
            for i in 0..POINTS_PER_CELL {
                let h = hash3(cx as u32, cy as u32, seed.wrapping_add(i as u32));
                let fx = ((h & 0xFFFF) as f32) / 65535.0;
                let fy = (((h >> 16) & 0xFFFF) as f32) / 65535.0;
                let px = (cx as f32 + fx) / cells as f32;
                let py = (cy as f32 + fy) / cells as f32;
                points.push(Vec2::new(px, py));
            }
        }
    }
    points
}

/// Seamless Worley F2 - F1 at `uv` (in unit square). Wraps via 3x3
/// neighbor scan over the cell `uv` falls into.
fn worley_f2_minus_f1(uv: Vec2, cells: usize, points: &[Vec2]) -> f32 {
    let cell_x = (uv.x * cells as f32).floor() as i32;
    let cell_y = (uv.y * cells as f32).floor() as i32;
    let mut f1 = f32::INFINITY;
    let mut f2 = f32::INFINITY;
    let cells_i = cells as i32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let nx = cell_x + dx;
            let ny = cell_y + dy;
            // Wrap cell coords for seamless tiling, and apply the
            // corresponding offset to the candidate point so distances
            // stay correct across the boundary.
            let wnx = nx.rem_euclid(cells_i);
            let wny = ny.rem_euclid(cells_i);
            let off_x = (nx - wnx) as f32 / cells as f32;
            let off_y = (ny - wny) as f32 / cells as f32;
            for i in 0..POINTS_PER_CELL {
                let p = points[(wny as usize * cells + wnx as usize) * POINTS_PER_CELL + i];
                let candidate = Vec2::new(p.x + off_x, p.y + off_y);
                let d = (uv - candidate).length();
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
    }
    // Normalize so values lie roughly in [0, 1] regardless of cell count.
    (f2 - f1) * cells as f32
}

/// Fast integer hash (xorshift mix) producing well-distributed 32-bit values.
fn hash3(x: u32, y: u32, z: u32) -> u32 {
    let mut h = x.wrapping_mul(0x8508_4A89);
    h ^= y.wrapping_mul(0xA9CE_8DC5);
    h ^= z.wrapping_mul(0x6F4A_5B71);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caustic_tile_is_non_empty_and_correct_size() {
        let tile = generate_caustic_tile(64, 1);
        assert_eq!(tile.len(), 64 * 64);
        assert!(tile.iter().any(|v| v.abs() > 1e-3), "tile is entirely zero");
    }

    #[test]
    fn caustic_tile_has_zero_mean() {
        let tile = generate_caustic_tile(64, 7);
        let mean: f32 = tile.iter().sum::<f32>() / tile.len() as f32;
        assert!(mean.abs() < 1e-5, "mean = {} should be ~0", mean);
    }

    #[test]
    fn caustic_tile_peak_is_bounded() {
        let tile = generate_caustic_tile(64, 11);
        let variance: f32 = tile.iter().map(|v| v * v).sum::<f32>() / tile.len() as f32;
        let stddev = variance.sqrt();
        let peak = tile.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Bright filaments dominate; allow a generous 10x stddev cap to
        // catch only truly pathological outliers from the powf sharpening.
        assert!(
            peak < 10.0 * stddev,
            "peak {} >= 10 * stddev {}",
            peak,
            stddev
        );
    }

    #[test]
    fn caustic_tile_is_seamless() {
        // The Worley sampler is mathematically periodic by construction
        // (wraps cell coordinates via rem_euclid). Verify that property
        // directly: sampling at uv and uv + (1, 0) / uv + (0, 1) yields
        // identical results.
        let cells = BASE_CELLS;
        let points = generate_feature_points(cells, 5);
        for &u in &[0.0_f32, 0.13, 0.5, 0.77, 0.999] {
            for &v in &[0.0_f32, 0.07, 0.42, 0.81] {
                let a = worley_f2_minus_f1(Vec2::new(u, v), cells, &points);
                let b = worley_f2_minus_f1(Vec2::new(u + 1.0, v), cells, &points);
                let c = worley_f2_minus_f1(Vec2::new(u, v + 1.0), cells, &points);
                assert!((a - b).abs() < 1e-4, "x wrap broken at ({u},{v})");
                assert!((a - c).abs() < 1e-4, "y wrap broken at ({u},{v})");
            }
        }
    }

    #[test]
    fn different_seeds_produce_different_tiles() {
        let a = generate_caustic_tile(32, 1);
        let b = generate_caustic_tile(32, 2);
        let mut differences = 0;
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > 1e-3 {
                differences += 1;
            }
        }
        assert!(differences > a.len() / 4);
    }
}
