// Discrete grid ray marching for voxel grids.
//
// Traces rays through a flat `size³` voxel array along the 26 grid-aligned
// directions (6 cardinal, 12 edge-diagonal, 8 corner-diagonal). Opaque
// (non-air) voxels terminate the ray; transparent voxels are traversed.
//
// Used by radiative heat transfer (Phase 9a) and later by optics (Phase 11).

use crate::world::voxel::Voxel;

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
}
