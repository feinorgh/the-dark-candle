// LOD interpolation and smooth transitions for voxel data.
//
// Provides trilinear interpolation between voxel samples at different
// resolutions and blending helpers for LOD transitions. Used to avoid
// visual popping when chunks switch between LOD levels.

#![allow(dead_code)]

use super::octree::OctreeNode;
use super::voxel::{MaterialId, Voxel};

/// Trilinear interpolation of a scalar field at fractional coordinates.
///
/// `sample(ix, iy, iz)` returns a scalar value at integer grid coordinates.
/// `(fx, fy, fz)` are the fractional coordinates within the grid (0.0 .. size as f32).
///
/// Returns the interpolated value. Clamps to grid bounds.
pub fn trilinear_sample<F>(fx: f32, fy: f32, fz: f32, size: usize, sample: F) -> f32
where
    F: Fn(usize, usize, usize) -> f32,
{
    let max = (size - 1) as f32;
    let x = fx.clamp(0.0, max);
    let y = fy.clamp(0.0, max);
    let z = fz.clamp(0.0, max);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;
    let x1 = (x0 + 1).min(size - 1);
    let y1 = (y0 + 1).min(size - 1);
    let z1 = (z0 + 1).min(size - 1);

    let xd = x - x0 as f32;
    let yd = y - y0 as f32;
    let zd = z - z0 as f32;

    // 8 corner samples
    let c000 = sample(x0, y0, z0);
    let c100 = sample(x1, y0, z0);
    let c010 = sample(x0, y1, z0);
    let c110 = sample(x1, y1, z0);
    let c001 = sample(x0, y0, z1);
    let c101 = sample(x1, y0, z1);
    let c011 = sample(x0, y1, z1);
    let c111 = sample(x1, y1, z1);

    // Interpolate along X
    let c00 = c000 * (1.0 - xd) + c100 * xd;
    let c01 = c001 * (1.0 - xd) + c101 * xd;
    let c10 = c010 * (1.0 - xd) + c110 * xd;
    let c11 = c011 * (1.0 - xd) + c111 * xd;

    // Interpolate along Y
    let c0 = c00 * (1.0 - yd) + c10 * yd;
    let c1 = c01 * (1.0 - yd) + c11 * yd;

    // Interpolate along Z
    c0 * (1.0 - zd) + c1 * zd
}

/// Interpolate temperature from a flat voxel array at fractional coordinates.
pub fn interpolate_temperature(voxels: &[Voxel], size: usize, fx: f32, fy: f32, fz: f32) -> f32 {
    trilinear_sample(fx, fy, fz, size, |x, y, z| {
        voxels[z * size * size + y * size + x].temperature
    })
}

/// Interpolate pressure from a flat voxel array at fractional coordinates.
pub fn interpolate_pressure(voxels: &[Voxel], size: usize, fx: f32, fy: f32, fz: f32) -> f32 {
    trilinear_sample(fx, fy, fz, size, |x, y, z| {
        voxels[z * size * size + y * size + x].pressure
    })
}

/// Interpolate a solidity field (1.0 = solid, 0.0 = air) for smooth surface extraction.
pub fn interpolate_solidity(voxels: &[Voxel], size: usize, fx: f32, fy: f32, fz: f32) -> f32 {
    trilinear_sample(fx, fy, fz, size, |x, y, z| {
        if voxels[z * size * size + y * size + x].is_air() {
            0.0
        } else {
            1.0
        }
    })
}

/// Interpolate temperature from an octree at fractional coordinates.
pub fn interpolate_temperature_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    fx: f32,
    fy: f32,
    fz: f32,
) -> f32 {
    trilinear_sample(fx, fy, fz, size, |x, y, z| {
        tree.get(x, y, z, size).temperature
    })
}

/// Interpolate pressure from an octree at fractional coordinates.
pub fn interpolate_pressure_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    fx: f32,
    fy: f32,
    fz: f32,
) -> f32 {
    trilinear_sample(fx, fy, fz, size, |x, y, z| tree.get(x, y, z, size).pressure)
}

/// LOD transition blend factor based on camera distance.
///
/// Returns a blend factor in [0.0, 1.0]:
/// - 0.0: fully at the near LOD (higher detail)
/// - 1.0: fully at the far LOD (lower detail)
///
/// `distance`: camera distance to chunk center (meters).
/// `near_threshold`: distance where transition begins.
/// `far_threshold`: distance where transition completes.
pub fn lod_blend_factor(distance: f32, near_threshold: f32, far_threshold: f32) -> f32 {
    if distance <= near_threshold {
        return 0.0;
    }
    if distance >= far_threshold {
        return 1.0;
    }
    let range = far_threshold - near_threshold;
    if range <= 0.0 {
        return 1.0;
    }
    // Hermite smoothstep for visually smooth transition
    let t = (distance - near_threshold) / range;
    t * t * (3.0 - 2.0 * t)
}

/// Blend two scalar values using a LOD blend factor.
#[inline]
pub fn blend_scalar(near_value: f32, far_value: f32, factor: f32) -> f32 {
    near_value + (far_value - near_value) * factor
}

/// Determine the dominant material at fractional coordinates by nearest-neighbor.
///
/// For material IDs, trilinear interpolation doesn't make sense (they're categorical),
/// so we use nearest-neighbor lookup.
pub fn nearest_material(voxels: &[Voxel], size: usize, fx: f32, fy: f32, fz: f32) -> MaterialId {
    let x = (fx.round() as usize).min(size - 1);
    let y = (fy.round() as usize).min(size - 1);
    let z = (fz.round() as usize).min(size - 1);
    voxels[z * size * size + y * size + x].material
}

/// Nearest-neighbor material lookup from an octree.
pub fn nearest_material_octree(
    tree: &OctreeNode<Voxel>,
    size: usize,
    fx: f32,
    fy: f32,
    fz: f32,
) -> MaterialId {
    let x = (fx.round() as usize).min(size - 1);
    let y = (fy.round() as usize).min(size - 1);
    let z = (fz.round() as usize).min(size - 1);
    tree.get(x, y, z, size).material
}

/// Upsample a coarse voxel grid to a finer resolution using trilinear interpolation
/// for continuous fields (temperature, pressure) and nearest-neighbor for materials.
///
/// `coarse`: voxels at coarse resolution (coarse_size³)
/// `fine_size`: target resolution (must be a multiple of coarse_size)
///
/// Returns a new flat array of `fine_size³` voxels.
pub fn upsample_voxels(coarse: &[Voxel], coarse_size: usize, fine_size: usize) -> Vec<Voxel> {
    assert!(fine_size >= coarse_size);
    assert!(fine_size.is_multiple_of(coarse_size));

    let scale = coarse_size as f32 / fine_size as f32;
    let len = fine_size * fine_size * fine_size;
    let mut result = Vec::with_capacity(len);

    for fz in 0..fine_size {
        for fy in 0..fine_size {
            for fx in 0..fine_size {
                let cx = fx as f32 * scale;
                let cy = fy as f32 * scale;
                let cz = fz as f32 * scale;

                let mat = nearest_material(coarse, coarse_size, cx, cy, cz);
                let temp = interpolate_temperature(coarse, coarse_size, cx, cy, cz);
                let pres = interpolate_pressure(coarse, coarse_size, cx, cy, cz);

                let mut v = Voxel::new(mat);
                v.temperature = temp;
                v.pressure = pres;
                result.push(v);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn air() -> Voxel {
        Voxel::default()
    }

    fn stone() -> Voxel {
        Voxel::new(MaterialId::STONE)
    }

    fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
        z * size * size + y * size + x
    }

    // --- Trilinear interpolation tests ---

    #[test]
    fn trilinear_at_grid_points_returns_exact() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2³
        let sample = |x: usize, y: usize, z: usize| data[z * 4 + y * 2 + x];

        assert!((trilinear_sample(0.0, 0.0, 0.0, 2, sample) - 1.0).abs() < 1e-6);
        assert!((trilinear_sample(1.0, 0.0, 0.0, 2, sample) - 2.0).abs() < 1e-6);
        assert!((trilinear_sample(0.0, 1.0, 0.0, 2, sample) - 3.0).abs() < 1e-6);
        assert!((trilinear_sample(1.0, 1.0, 1.0, 2, sample) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn trilinear_midpoint_is_average() {
        // Uniform field of value 10.0
        let sample = |_x: usize, _y: usize, _z: usize| 10.0;
        let result = trilinear_sample(0.5, 0.5, 0.5, 2, sample);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn trilinear_interpolates_linearly_along_axis() {
        // Linear gradient along X: f(x,y,z) = x
        let size = 4;
        let sample = |x: usize, _y: usize, _z: usize| x as f32;

        let r0 = trilinear_sample(0.0, 0.0, 0.0, size, sample);
        let r1 = trilinear_sample(1.5, 0.0, 0.0, size, sample);
        let r2 = trilinear_sample(3.0, 0.0, 0.0, size, sample);

        assert!((r0 - 0.0).abs() < 1e-6);
        assert!((r1 - 1.5).abs() < 1e-6);
        assert!((r2 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn trilinear_clamps_out_of_bounds() {
        let sample = |_x: usize, _y: usize, _z: usize| 5.0;
        let result = trilinear_sample(-1.0, -1.0, -1.0, 2, sample);
        assert!((result - 5.0).abs() < 1e-6);

        let result2 = trilinear_sample(100.0, 100.0, 100.0, 2, sample);
        assert!((result2 - 5.0).abs() < 1e-6);
    }

    // --- Voxel interpolation tests ---

    #[test]
    fn temperature_interpolation_at_grid_point() {
        let size = 2;
        let mut voxels = vec![air(); size * size * size];
        voxels[idx(0, 0, 0, size)].temperature = 300.0;
        voxels[idx(1, 0, 0, size)].temperature = 400.0;

        let t = interpolate_temperature(&voxels, size, 0.0, 0.0, 0.0);
        assert!((t - 300.0).abs() < 1e-4);

        let t_mid = interpolate_temperature(&voxels, size, 0.5, 0.0, 0.0);
        assert!(
            (t_mid - 350.0).abs() < 1e-4,
            "Midpoint should be 350, got {t_mid}"
        );
    }

    #[test]
    fn solidity_interpolation() {
        let size = 4;
        let mut voxels = vec![air(); size * size * size];
        // Fill bottom half with stone
        for z in 0..size {
            for x in 0..size {
                for y in 0..size / 2 {
                    voxels[idx(x, y, z, size)] = stone();
                }
            }
        }

        // Deep inside solid: should be ~1.0
        assert!(interpolate_solidity(&voxels, size, 1.0, 0.0, 1.0) > 0.9);
        // Deep inside air: should be ~0.0
        assert!(interpolate_solidity(&voxels, size, 1.0, 3.0, 1.0) < 0.1);
        // At the boundary: should be ~0.5
        let boundary = interpolate_solidity(&voxels, size, 1.5, 1.5, 1.5);
        assert!(
            boundary > 0.2 && boundary < 0.8,
            "Boundary solidity={boundary}"
        );
    }

    // --- LOD blend tests ---

    #[test]
    fn blend_factor_at_extremes() {
        assert_eq!(lod_blend_factor(10.0, 100.0, 200.0), 0.0);
        assert_eq!(lod_blend_factor(300.0, 100.0, 200.0), 1.0);
    }

    #[test]
    fn blend_factor_at_midpoint() {
        let f = lod_blend_factor(150.0, 100.0, 200.0);
        // Smoothstep at t=0.5 → 0.5
        assert!(
            (f - 0.5).abs() < 1e-6,
            "Midpoint blend should be 0.5, got {f}"
        );
    }

    #[test]
    fn blend_factor_is_monotonic() {
        let mut prev = 0.0;
        for d in 0..=100 {
            let distance = 100.0 + d as f32;
            let f = lod_blend_factor(distance, 100.0, 200.0);
            assert!(f >= prev, "Blend should be monotonically increasing");
            prev = f;
        }
    }

    #[test]
    fn blend_scalar_works() {
        assert_eq!(blend_scalar(10.0, 20.0, 0.0), 10.0);
        assert_eq!(blend_scalar(10.0, 20.0, 1.0), 20.0);
        assert_eq!(blend_scalar(10.0, 20.0, 0.5), 15.0);
    }

    // --- Nearest material tests ---

    #[test]
    fn nearest_material_at_exact_coord() {
        let size = 4;
        let mut voxels = vec![air(); size * size * size];
        voxels[idx(2, 2, 2, size)] = stone();

        let mat = nearest_material(&voxels, size, 2.0, 2.0, 2.0);
        assert_eq!(mat, MaterialId::STONE);
    }

    #[test]
    fn nearest_material_rounds() {
        let size = 4;
        let mut voxels = vec![air(); size * size * size];
        voxels[idx(2, 2, 2, size)] = stone();

        // Close to (2,2,2) — should round to stone
        let mat = nearest_material(&voxels, size, 1.6, 2.4, 1.7);
        assert_eq!(mat, MaterialId::STONE);

        // Far from (2,2,2) — should round to air
        let mat_air = nearest_material(&voxels, size, 0.1, 0.1, 0.1);
        assert_eq!(mat_air, MaterialId::AIR);
    }

    // --- Upsample tests ---

    #[test]
    fn upsample_preserves_uniform() {
        let size = 2;
        let mut voxels = vec![stone(); size * size * size];
        for v in &mut voxels {
            v.temperature = 500.0;
            v.pressure = 200_000.0;
        }

        let upsampled = upsample_voxels(&voxels, 2, 4);
        assert_eq!(upsampled.len(), 64);

        for v in &upsampled {
            assert_eq!(v.material, MaterialId::STONE);
            assert!((v.temperature - 500.0).abs() < 1e-3);
            assert!((v.pressure - 200_000.0).abs() < 1.0);
        }
    }

    #[test]
    fn upsample_doubles_resolution() {
        let size = 2;
        let mut voxels = vec![air(); size * size * size];
        voxels[0].temperature = 300.0;
        voxels[1].temperature = 400.0;

        let upsampled = upsample_voxels(&voxels, 2, 4);
        assert_eq!(upsampled.len(), 64);

        // Corner should be close to original value
        assert!((upsampled[0].temperature - 300.0).abs() < 50.0);
    }

    // --- Octree interpolation tests ---

    #[test]
    fn octree_temperature_interpolation() {
        use crate::world::voxel_access::flat_to_octree;

        let size = 4;
        let mut voxels = vec![air(); size * size * size];
        voxels[idx(0, 0, 0, size)].temperature = 300.0;
        voxels[idx(1, 0, 0, size)].temperature = 400.0;

        let tree = flat_to_octree(&voxels, size);

        let t = interpolate_temperature_octree(&tree, size, 0.0, 0.0, 0.0);
        assert!((t - 300.0).abs() < 1e-4);

        let t_mid = interpolate_temperature_octree(&tree, size, 0.5, 0.0, 0.0);
        assert!((t_mid - 350.0).abs() < 1e-4);
    }
}
