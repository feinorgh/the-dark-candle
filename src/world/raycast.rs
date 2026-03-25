// Discrete grid ray marching for voxel grids.
//
// Traces rays through a flat `size³` voxel array along the 26 grid-aligned
// directions (6 cardinal, 12 edge-diagonal, 8 corner-diagonal). Two march
// variants:
//
// - `march_grid_ray` — binary opaque/transparent: non-air stops the ray.
// - `march_grid_ray_attenuated` — Beer-Lambert attenuation through
//    semi-transparent media (water, ice, steam). Returns transmittance.
//
// Used by radiative heat transfer (Phase 9a) and later by optics (Phase 11).

use crate::world::voxel::{MaterialId, Voxel};

/// Result of a successful ray march — the first opaque voxel hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    /// Flat index into the `size³` voxel array.
    pub index: usize,
    /// Euclidean distance from origin to hit in voxel units (= meters).
    pub distance: f32,
}

/// Ray hit with Beer-Lambert transmittance through intervening media.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttenuatedRayHit {
    /// The opaque surface that terminated the ray.
    pub hit: RayHit,
    /// Fraction of radiation reaching the target (0.0–1.0).
    /// `transmittance = exp(−Σ(αᵢ × dᵢ))` via Beer-Lambert law.
    pub transmittance: f32,
}

/// Transmittance below this threshold is treated as fully absorbed (early exit).
const MIN_TRANSMITTANCE: f32 = 0.001;

/// All 26 grid-aligned directions: 6 cardinal + 12 edge + 8 corner.
pub const RAY_DIRECTIONS: [[i32; 3]; 26] = [
    // 6 cardinal
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    // 12 edge-diagonal
    [1, 1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, 1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, 1, -1],
    [0, -1, 1],
    [0, -1, -1],
    // 8 corner-diagonal
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
];

/// Precomputed step distance for each direction in [`RAY_DIRECTIONS`].
///
/// Cardinal = 1.0, edge = √2, corner = √3.
const STEP_DISTANCES: [f32; 26] = {
    let mut dists = [0.0_f32; 26];
    let mut i = 0;
    while i < 26 {
        let d = &RAY_DIRECTIONS[i];
        let sq = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f32;
        // const-compatible sqrt approximation: all values are 1, 2, or 3.
        dists[i] = if sq < 1.5 {
            1.0
        } else if sq < 2.5 {
            std::f32::consts::SQRT_2
        } else {
            // √3 — no stdlib constant, so literal value
            1.732_050_8
        };
        i += 1;
    }
    dists
};

/// March a ray through a flat `size³` voxel grid in a discrete direction.
///
/// Starting from `(sx, sy, sz)`, steps one voxel at a time along `dir`
/// (each component −1, 0, or +1). Returns the first non-air voxel hit,
/// or `None` if the ray exits the grid or exceeds `max_steps`.
///
/// The origin voxel itself is never returned as a hit.
pub fn march_grid_ray(
    voxels: &[Voxel],
    size: usize,
    start: [usize; 3],
    dir_index: usize,
    max_steps: usize,
) -> Option<RayHit> {
    debug_assert!(dir_index < 26);
    let dir = RAY_DIRECTIONS[dir_index];
    let step_dist = STEP_DISTANCES[dir_index];

    let mut x = start[0] as i32;
    let mut y = start[1] as i32;
    let mut z = start[2] as i32;
    let bound = size as i32;

    for step in 1..=max_steps {
        x += dir[0];
        y += dir[1];
        z += dir[2];

        if x < 0 || y < 0 || z < 0 || x >= bound || y >= bound || z >= bound {
            return None;
        }

        let ux = x as usize;
        let uy = y as usize;
        let uz = z as usize;
        let idx = uz * size * size + uy * size + ux;

        if !voxels[idx].material.is_air() {
            return Some(RayHit {
                x: ux,
                y: uy,
                z: uz,
                index: idx,
                distance: step as f32 * step_dist,
            });
        }
    }

    None
}

/// March a ray with Beer-Lambert attenuation through semi-transparent media.
///
/// `get_absorption` classifies each material:
/// - `Some(0.0)` — fully transparent (air), no attenuation
/// - `Some(α)` where α > 0 — semi-transparent, attenuate by exp(−α × step_dist)
/// - `None` — opaque surface, terminates the ray (returned as hit)
///
/// Returns the first opaque hit plus the cumulative transmittance through
/// any intervening semi-transparent voxels. Returns `None` if the ray exits
/// the grid, exceeds `max_steps`, or is fully absorbed (transmittance < 0.001).
pub fn march_grid_ray_attenuated<F>(
    voxels: &[Voxel],
    size: usize,
    start: [usize; 3],
    dir_index: usize,
    max_steps: usize,
    get_absorption: F,
) -> Option<AttenuatedRayHit>
where
    F: Fn(MaterialId) -> Option<f32>,
{
    debug_assert!(dir_index < 26);
    let dir = RAY_DIRECTIONS[dir_index];
    let step_dist = STEP_DISTANCES[dir_index];

    let mut x = start[0] as i32;
    let mut y = start[1] as i32;
    let mut z = start[2] as i32;
    let bound = size as i32;
    let mut optical_depth: f32 = 0.0;

    for step in 1..=max_steps {
        x += dir[0];
        y += dir[1];
        z += dir[2];

        if x < 0 || y < 0 || z < 0 || x >= bound || y >= bound || z >= bound {
            return None;
        }

        let ux = x as usize;
        let uy = y as usize;
        let uz = z as usize;
        let idx = uz * size * size + uy * size + ux;

        match get_absorption(voxels[idx].material) {
            Some(alpha) => {
                optical_depth += alpha * step_dist;
                if (-optical_depth).exp() < MIN_TRANSMITTANCE {
                    return None; // fully absorbed by medium
                }
            }
            None => {
                return Some(AttenuatedRayHit {
                    hit: RayHit {
                        x: ux,
                        y: uy,
                        z: uz,
                        index: idx,
                        distance: step as f32 * step_dist,
                    },
                    transmittance: (-optical_depth).exp(),
                });
            }
        }
    }

    None
}

/// Check whether a non-air voxel at `(x, y, z)` has at least one
/// air-adjacent face (i.e. is a surface voxel exposed to radiation).
pub fn is_surface_voxel(voxels: &[Voxel], size: usize, x: usize, y: usize, z: usize) -> bool {
    let bound = size as i32;
    let xi = x as i32;
    let yi = y as i32;
    let zi = z as i32;

    for &(dx, dy, dz) in &[
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ] {
        let nx = xi + dx;
        let ny = yi + dy;
        let nz = zi + dz;
        if nx < 0 || ny < 0 || nz < 0 || nx >= bound || ny >= bound || nz >= bound {
            return true; // grid boundary = exposed
        }
        let ni = nz as usize * size * size + ny as usize * size + nx as usize;
        if voxels[ni].material.is_air() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::MaterialId;

    fn make_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    fn set_solid(voxels: &mut [Voxel], size: usize, x: usize, y: usize, z: usize) {
        let idx = z * size * size + y * size + x;
        voxels[idx].material = MaterialId::STONE;
    }

    #[test]
    fn ray_through_empty_grid_returns_none() {
        let grid = make_grid(8);
        for dir_idx in 0..26 {
            let hit = march_grid_ray(&grid, 8, [4, 4, 4], dir_idx, 8);
            assert!(hit.is_none(), "dir {dir_idx} should find nothing in air");
        }
    }

    #[test]
    fn ray_hits_adjacent_solid() {
        let mut grid = make_grid(8);
        set_solid(&mut grid, 8, 5, 4, 4);
        let hit = march_grid_ray(&grid, 8, [4, 4, 4], 0, 8); // +X
        let hit = hit.expect("should hit solid at (5,4,4)");
        assert_eq!((hit.x, hit.y, hit.z), (5, 4, 4));
        assert!((hit.distance - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ray_hits_distant_solid() {
        let mut grid = make_grid(16);
        set_solid(&mut grid, 16, 10, 4, 4);
        let hit = march_grid_ray(&grid, 16, [4, 4, 4], 0, 16); // +X
        let hit = hit.expect("should hit solid at (10,4,4)");
        assert_eq!(hit.x, 10);
        assert!((hit.distance - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ray_blocked_by_intermediate_solid() {
        let mut grid = make_grid(16);
        set_solid(&mut grid, 16, 7, 4, 4); // blocker
        set_solid(&mut grid, 16, 10, 4, 4); // target behind blocker
        let hit = march_grid_ray(&grid, 16, [4, 4, 4], 0, 16);
        let hit = hit.expect("should hit blocker, not target");
        assert_eq!(hit.x, 7, "should stop at blocker");
    }

    #[test]
    fn ray_stops_at_max_steps() {
        let mut grid = make_grid(32);
        set_solid(&mut grid, 32, 20, 4, 4);
        let hit = march_grid_ray(&grid, 32, [4, 4, 4], 0, 8); // max 8 steps
        assert!(hit.is_none(), "target at distance 16 exceeds max_steps=8");
    }

    #[test]
    fn diagonal_ray_distance_is_correct() {
        let mut grid = make_grid(16);
        set_solid(&mut grid, 16, 7, 7, 7); // 3 steps along (1,1,1) from (4,4,4)
        let hit = march_grid_ray(&grid, 16, [4, 4, 4], 18, 16); // dir 18 = (1,1,1)
        let hit = hit.expect("should hit diagonal solid");
        assert_eq!((hit.x, hit.y, hit.z), (7, 7, 7));
        let expected = 3.0 * 3.0_f32.sqrt();
        assert!(
            (hit.distance - expected).abs() < 0.01,
            "distance {:.3} should be {expected:.3}",
            hit.distance
        );
    }

    #[test]
    fn surface_voxel_detection() {
        let mut grid = make_grid(4);
        // Solid block surrounded by air
        set_solid(&mut grid, 4, 2, 2, 2);
        assert!(is_surface_voxel(&grid, 4, 2, 2, 2));

        // Fill all neighbors to make it interior
        for &(dx, dy, dz) in &[
            (1i32, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let nx = (2i32 + dx) as usize;
            let ny = (2i32 + dy) as usize;
            let nz = (2i32 + dz) as usize;
            set_solid(&mut grid, 4, nx, ny, nz);
        }
        assert!(
            !is_surface_voxel(&grid, 4, 2, 2, 2),
            "fully enclosed voxel is not a surface"
        );
    }

    #[test]
    fn boundary_voxel_is_surface() {
        let mut grid = make_grid(4);
        set_solid(&mut grid, 4, 0, 0, 0);
        assert!(
            is_surface_voxel(&grid, 4, 0, 0, 0),
            "corner voxel touches grid boundary"
        );
    }

    #[test]
    fn ray_directions_count() {
        assert_eq!(RAY_DIRECTIONS.len(), 26);
        assert_eq!(STEP_DISTANCES.len(), 26);
    }

    #[test]
    fn step_distances_are_correct() {
        for (i, dist) in STEP_DISTANCES.iter().enumerate() {
            let d = &RAY_DIRECTIONS[i];
            let expected = ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f32).sqrt();
            assert!(
                (dist - expected).abs() < 0.001,
                "dir {i}: step dist {dist} != expected {expected}"
            );
        }
    }

    // --- Attenuated ray march tests ---

    fn set_material(
        voxels: &mut [Voxel],
        size: usize,
        x: usize,
        y: usize,
        z: usize,
        mat: MaterialId,
    ) {
        voxels[z * size * size + y * size + x].material = mat;
    }

    /// Classify materials for attenuated tests:
    /// Air → Some(0.0), Water → Some(0.1), Stone → None (opaque).
    fn test_absorption(mat: MaterialId) -> Option<f32> {
        if mat.is_air() {
            Some(0.0)
        } else if mat == MaterialId::WATER {
            Some(0.1) // α = 0.1 m⁻¹ for test convenience
        } else {
            None // opaque
        }
    }

    #[test]
    fn attenuated_through_air_has_full_transmittance() {
        let mut grid = make_grid(8);
        set_solid(&mut grid, 8, 6, 4, 4);
        let result = march_grid_ray_attenuated(&grid, 8, [4, 4, 4], 0, 8, test_absorption);
        let result = result.expect("should hit stone");
        assert_eq!(result.hit.x, 6);
        assert!(
            (result.transmittance - 1.0).abs() < f32::EPSILON,
            "pure air path should have transmittance 1.0, got {}",
            result.transmittance
        );
    }

    #[test]
    fn attenuated_through_water_reduces_transmittance() {
        let mut grid = make_grid(8);
        // Water at x=5, stone target at x=6
        set_material(&mut grid, 8, 5, 4, 4, MaterialId::WATER);
        set_material(&mut grid, 8, 6, 4, 4, MaterialId::STONE);
        let result = march_grid_ray_attenuated(&grid, 8, [4, 4, 4], 0, 8, test_absorption);
        let result = result.expect("should hit stone through water");
        assert_eq!(result.hit.x, 6);
        // 1 voxel of water with α=0.1: transmittance = exp(-0.1) ≈ 0.905
        let expected = (-0.1_f32).exp();
        assert!(
            (result.transmittance - expected).abs() < 1e-6,
            "transmittance {:.6} should be {expected:.6}",
            result.transmittance
        );
    }

    #[test]
    fn attenuated_multiple_water_voxels() {
        let mut grid = make_grid(10);
        // 3 voxels of water (x=5,6,7), stone target at x=8
        for x in 5..=7 {
            set_material(&mut grid, 10, x, 4, 4, MaterialId::WATER);
        }
        set_material(&mut grid, 10, 8, 4, 4, MaterialId::STONE);
        let result = march_grid_ray_attenuated(&grid, 10, [4, 4, 4], 0, 10, test_absorption);
        let result = result.expect("should hit stone through water");
        assert_eq!(result.hit.x, 8);
        // 3 voxels × α=0.1 × d=1: transmittance = exp(-0.3) ≈ 0.741
        let expected = (-0.3_f32).exp();
        assert!(
            (result.transmittance - expected).abs() < 1e-5,
            "transmittance {:.6} should be {expected:.6}",
            result.transmittance
        );
    }

    #[test]
    fn attenuated_fully_absorbed_returns_none() {
        let mut grid = make_grid(16);
        // Use high α to ensure full absorption
        let high_absorption = |mat: MaterialId| -> Option<f32> {
            if mat.is_air() {
                Some(0.0)
            } else if mat == MaterialId::WATER {
                Some(10.0) // α = 10 m⁻¹: exp(-10) ≈ 4.5e-5 < MIN_TRANSMITTANCE
            } else {
                None
            }
        };
        // Fill x=5..12 with water (8 voxels), stone target at x=13
        for x in 5..=12 {
            set_material(&mut grid, 16, x, 4, 4, MaterialId::WATER);
        }
        set_material(&mut grid, 16, 13, 4, 4, MaterialId::STONE);
        let result = march_grid_ray_attenuated(&grid, 16, [4, 4, 4], 0, 16, high_absorption);
        assert!(
            result.is_none(),
            "thick water layer with high α should fully absorb the ray"
        );
    }

    #[test]
    fn attenuated_opaque_stops_before_transparent() {
        let mut grid = make_grid(8);
        // Stone at x=5 blocks, water at x=6 is behind it
        set_material(&mut grid, 8, 5, 4, 4, MaterialId::STONE);
        set_material(&mut grid, 8, 6, 4, 4, MaterialId::WATER);
        let result = march_grid_ray_attenuated(&grid, 8, [4, 4, 4], 0, 8, test_absorption);
        let result = result.expect("should hit stone");
        assert_eq!(result.hit.x, 5);
        assert!(
            (result.transmittance - 1.0).abs() < f32::EPSILON,
            "no semi-transparent material before stone"
        );
    }

    #[test]
    fn attenuated_low_alpha_preserves_most_radiation() {
        let mut grid = make_grid(8);
        set_material(&mut grid, 8, 5, 4, 4, MaterialId::WATER);
        set_material(&mut grid, 8, 6, 4, 4, MaterialId::STONE);
        // Use very low absorption coefficient
        let low_absorption = |mat: MaterialId| -> Option<f32> {
            if mat.is_air() {
                Some(0.0)
            } else if mat == MaterialId::WATER {
                Some(0.01) // nearly transparent
            } else {
                None
            }
        };
        let result = march_grid_ray_attenuated(&grid, 8, [4, 4, 4], 0, 8, low_absorption);
        let result = result.expect("should hit stone");
        let expected = (-0.01_f32).exp(); // ≈ 0.99
        assert!(
            result.transmittance > 0.98,
            "low α should preserve most radiation: {:.4}",
            result.transmittance
        );
        assert!(
            (result.transmittance - expected).abs() < 1e-5,
            "transmittance {:.6} should be {expected:.6}",
            result.transmittance
        );
    }
}
