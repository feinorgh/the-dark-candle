// Structural integrity: flood-fill connectivity check from anchored voxels.
//
// Solid voxels that are not connected (via face-adjacent solid neighbors)
// to an anchor are considered unsupported and collapse under gravity.
// An "anchor" is any solid voxel at y=0 (bedrock layer) or any voxel of
// a material explicitly marked as anchored (e.g., stone, dirt).
//
// Algorithm:
//   1. Flood-fill from all anchor voxels, marking reachable solids as "supported."
//   2. Any solid voxel NOT marked is unsupported.
//   3. Unsupported voxels are returned as a list for the caller to handle
//      (e.g., convert to falling entities, or drop them one y-level).

#![allow(dead_code)]

use crate::world::voxel::{MaterialId, Voxel};
use std::collections::VecDeque;

/// Returns true if the material is considered solid (not air, not fluid).
fn is_solid(mat: MaterialId) -> bool {
    !mat.is_air() && mat != MaterialId::WATER && mat != MaterialId::LAVA && mat != MaterialId::STEAM
}

/// Returns true if a voxel at y=0 is a natural anchor (bedrock layer).
fn is_anchor(voxel: &Voxel, y: usize) -> bool {
    y == 0 && is_solid(voxel.material)
}

/// 3D index into a flat voxel array of size³.
fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Face-adjacent neighbor offsets (6-connectivity).
const NEIGHBORS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Find all unsupported solid voxels in a chunk.
///
/// Returns a Vec of (x, y, z) coordinates of solid voxels that have no
/// face-connected path to any anchor voxel. These should collapse.
pub fn find_unsupported(voxels: &[Voxel], size: usize) -> Vec<(usize, usize, usize)> {
    let total = size * size * size;
    assert_eq!(voxels.len(), total);

    let mut supported = vec![false; total];
    let mut queue = VecDeque::new();

    // Seed: all anchor voxels
    for z in 0..size {
        for x in 0..size {
            for y in 0..size {
                let i = idx(x, y, z, size);
                if is_anchor(&voxels[i], y) {
                    supported[i] = true;
                    queue.push_back((x, y, z));
                }
            }
        }
    }

    // BFS flood-fill through face-adjacent solid voxels
    while let Some((x, y, z)) = queue.pop_front() {
        for &(dx, dy, dz) in &NEIGHBORS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;

            if nx < 0 || ny < 0 || nz < 0 {
                continue;
            }
            let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
            if nx >= size || ny >= size || nz >= size {
                continue;
            }

            let ni = idx(nx, ny, nz, size);
            if !supported[ni] && is_solid(voxels[ni].material) {
                supported[ni] = true;
                queue.push_back((nx, ny, nz));
            }
        }
    }

    // Collect unsupported solid voxels
    let mut unsupported = Vec::new();
    for z in 0..size {
        for x in 0..size {
            for y in 0..size {
                let i = idx(x, y, z, size);
                if is_solid(voxels[i].material) && !supported[i] {
                    unsupported.push((x, y, z));
                }
            }
        }
    }

    unsupported
}

/// Drop unsupported voxels one y-level (simple gravity collapse).
///
/// Each unsupported voxel moves down by 1 if the space below is air.
/// Returns the number of voxels that moved.
pub fn collapse_unsupported(voxels: &mut [Voxel], size: usize) -> usize {
    let unsupported = find_unsupported(voxels, size);
    let mut moved = 0;

    // Process top-to-bottom so upper voxels don't land on already-moved ones
    let mut sorted = unsupported;
    sorted.sort_by(|a, b| a.1.cmp(&b.1));

    for (x, y, z) in sorted {
        if y == 0 {
            continue; // Can't fall below the grid
        }
        let from = idx(x, y, z, size);
        let to = idx(x, y - 1, z, size);

        if voxels[to].material.is_air() {
            voxels.swap(from, to);
            moved += 1;
        }
    }

    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    #[test]
    fn empty_grid_has_no_unsupported() {
        let grid = make_grid(4);
        let unsupported = find_unsupported(&grid, 4);
        assert!(unsupported.is_empty());
    }

    #[test]
    fn solid_at_y0_is_supported() {
        let mut grid = make_grid(4);
        grid[idx(1, 0, 1, 4)].material = MaterialId::STONE;
        let unsupported = find_unsupported(&grid, 4);
        assert!(unsupported.is_empty());
    }

    #[test]
    fn column_from_ground_is_supported() {
        let mut grid = make_grid(4);
        // Stack of stone from y=0 to y=3
        for y in 0..4 {
            grid[idx(1, y, 1, 4)].material = MaterialId::STONE;
        }
        let unsupported = find_unsupported(&grid, 4);
        assert!(unsupported.is_empty());
    }

    #[test]
    fn floating_block_is_unsupported() {
        let mut grid = make_grid(4);
        // Stone floating at y=3, nothing below
        grid[idx(1, 3, 1, 4)].material = MaterialId::STONE;
        let unsupported = find_unsupported(&grid, 4);
        assert_eq!(unsupported.len(), 1);
        assert_eq!(unsupported[0], (1, 3, 1));
    }

    #[test]
    fn floating_cluster_is_unsupported() {
        let mut grid = make_grid(8);
        // L-shaped cluster floating at y=5
        grid[idx(3, 5, 3, 8)].material = MaterialId::STONE;
        grid[idx(4, 5, 3, 8)].material = MaterialId::STONE;
        grid[idx(4, 5, 4, 8)].material = MaterialId::DIRT;

        let unsupported = find_unsupported(&grid, 8);
        assert_eq!(unsupported.len(), 3);
    }

    #[test]
    fn bridge_connected_to_ground_is_supported() {
        let mut grid = make_grid(8);
        // Pillar from ground
        for y in 0..4 {
            grid[idx(1, y, 1, 8)].material = MaterialId::STONE;
        }
        // Bridge at y=3 extending to x=4
        for x in 2..5 {
            grid[idx(x, 3, 1, 8)].material = MaterialId::STONE;
        }
        let unsupported = find_unsupported(&grid, 8);
        assert!(
            unsupported.is_empty(),
            "Bridge connected to pillar should be supported"
        );
    }

    #[test]
    fn fluids_are_not_structural() {
        let mut grid = make_grid(4);
        // Water at y=0, stone at y=1 — stone is anchored via ground layer (y=0),
        // but water is not solid so doesn't count
        grid[idx(1, 0, 1, 4)].material = MaterialId::WATER;
        grid[idx(1, 1, 1, 4)].material = MaterialId::STONE;

        let unsupported = find_unsupported(&grid, 4);
        // Stone at y=1 is unsupported because water below isn't solid
        assert_eq!(unsupported.len(), 1);
        assert_eq!(unsupported[0], (1, 1, 1));
    }

    #[test]
    fn collapse_drops_floating_block() {
        let mut grid = make_grid(4);
        grid[idx(1, 3, 1, 4)].material = MaterialId::STONE;

        let moved = collapse_unsupported(&mut grid, 4);
        assert_eq!(moved, 1, "One block should fall");
        assert!(grid[idx(1, 3, 1, 4)].material.is_air());
        assert_eq!(grid[idx(1, 2, 1, 4)].material, MaterialId::STONE);
    }

    #[test]
    fn collapse_repeated_reaches_ground() {
        let mut grid = make_grid(4);
        grid[idx(1, 3, 1, 4)].material = MaterialId::STONE;

        // Collapse repeatedly until stable
        for _ in 0..10 {
            if collapse_unsupported(&mut grid, 4) == 0 {
                break;
            }
        }

        // Block should reach y=0 and be anchored
        assert_eq!(grid[idx(1, 0, 1, 4)].material, MaterialId::STONE);
        let unsupported = find_unsupported(&grid, 4);
        assert!(unsupported.is_empty(), "Block at y=0 should be anchored");
    }

    #[test]
    fn supported_block_does_not_move() {
        let mut grid = make_grid(4);
        grid[idx(1, 0, 1, 4)].material = MaterialId::STONE;
        grid[idx(1, 1, 1, 4)].material = MaterialId::STONE;

        let moved = collapse_unsupported(&mut grid, 4);
        assert_eq!(moved, 0, "Supported blocks should not move");
    }
}
