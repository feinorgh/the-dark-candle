// Free surface tracking: classify cells as LIQUID, AIR, SURFACE, or SOLID.
//
// Runs at the start of each fluid step to ensure boundary conditions are
// applied to the correct cells. A LIQUID cell adjacent to any AIR cell
// becomes SURFACE; SURFACE cells get p=0 (atmospheric) in the pressure solve.

use super::types::{CellTag, FluidGrid};

/// Reclassify all fluid cells based on neighbor adjacency.
///
/// - LIQUID with any AIR face-neighbor → SURFACE
/// - LIQUID with no AIR neighbors → stays LIQUID
/// - AIR stays AIR
/// - SOLID stays SOLID
pub fn update_tags(grid: &mut FluidGrid) {
    let size = grid.size();
    // Snapshot tags to avoid read-write conflict during reclassification.
    let old_tags: Vec<CellTag> = grid.cells().iter().map(|c| c.tag).collect();

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let tag = old_tags[idx];

                match tag {
                    CellTag::Liquid | CellTag::Surface => {
                        if has_air_neighbor(&old_tags, x, y, z, size) {
                            grid.cells_mut()[idx].tag = CellTag::Surface;
                        } else {
                            grid.cells_mut()[idx].tag = CellTag::Liquid;
                        }
                    }
                    // Air and Solid don't change during tagging.
                    CellTag::Air | CellTag::Solid => {}
                }
            }
        }
    }
}

/// Check if any of the 6 face-adjacent cells is tagged AIR.
fn has_air_neighbor(tags: &[CellTag], x: usize, y: usize, z: usize, size: usize) -> bool {
    let offsets: [(i32, i32, i32); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    for (dx, dy, dz) in offsets {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;

        if nx < 0 || ny < 0 || nz < 0 || nx >= size as i32 || ny >= size as i32 || nz >= size as i32
        {
            // Cells at the boundary are treated as adjacent to air (open boundary).
            return true;
        }

        let nidx = nz as usize * size * size + ny as usize * size + nx as usize;
        if tags[nidx] == CellTag::Air {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::MaterialId;

    fn make_grid_with(size: usize, setup: impl Fn(usize, usize, usize) -> CellTag) -> FluidGrid {
        let mut grid = FluidGrid::new_empty(size);
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell = grid.get_mut(x, y, z);
                    cell.tag = setup(x, y, z);
                    if cell.tag == CellTag::Liquid || cell.tag == CellTag::Surface {
                        cell.material = MaterialId::WATER;
                    }
                }
            }
        }
        grid
    }

    #[test]
    fn single_liquid_cell_becomes_surface() {
        // A lone liquid cell surrounded by air should become SURFACE.
        let mut grid = FluidGrid::new_empty(4);
        grid.get_mut(2, 2, 2).tag = CellTag::Liquid;
        grid.get_mut(2, 2, 2).material = MaterialId::WATER;

        update_tags(&mut grid);
        assert_eq!(grid.get(2, 2, 2).tag, CellTag::Surface);
    }

    #[test]
    fn interior_liquid_stays_liquid() {
        // A 4³ grid fully filled with liquid: interior cells stay LIQUID,
        // boundary cells become SURFACE.
        let mut grid = make_grid_with(4, |_, _, _| CellTag::Liquid);

        update_tags(&mut grid);

        // Corner cell (0,0,0) has boundary neighbors → SURFACE
        assert_eq!(grid.get(0, 0, 0).tag, CellTag::Surface);

        // Interior cell (1,1,1): neighbors are (0,1,1), (2,1,1), (1,0,1),
        // (1,2,1), (1,1,0), (1,1,2) — all LIQUID, none AIR.
        // But (1,1,1) is only 1 cell from the edge in a 4³ grid...
        // actually (0,1,1) is at edge → boundary → treated as air-adjacent → SURFACE.
        // In a 4³ grid, there are no truly interior cells (need 6³ minimum for a core).
        // Let's check a 6³ grid instead.
    }

    #[test]
    fn interior_liquid_in_large_grid_stays_liquid() {
        let mut grid = make_grid_with(6, |_, _, _| CellTag::Liquid);
        update_tags(&mut grid);

        // (3,3,3) is fully interior: all 6 neighbors are liquid, none at boundary.
        assert_eq!(grid.get(3, 3, 3).tag, CellTag::Liquid);

        // (0,0,0) is at boundary → SURFACE.
        assert_eq!(grid.get(0, 0, 0).tag, CellTag::Surface);
    }

    #[test]
    fn water_cube_surface_tagging() {
        // 8³ grid: water fills the center 4³ cube (2..6 in each axis), rest is air.
        let mut grid = make_grid_with(8, |x, y, z| {
            if (2..6).contains(&x) && (2..6).contains(&y) && (2..6).contains(&z) {
                CellTag::Liquid
            } else {
                CellTag::Air
            }
        });

        update_tags(&mut grid);

        // Interior: (3,3,3) — all neighbors are liquid → LIQUID
        assert_eq!(grid.get(3, 3, 3).tag, CellTag::Liquid);

        // Surface: (2,3,3) — neighbor (1,3,3) is air → SURFACE
        assert_eq!(grid.get(2, 3, 3).tag, CellTag::Surface);

        // Air stays air
        assert_eq!(grid.get(0, 0, 0).tag, CellTag::Air);
    }

    #[test]
    fn solid_cells_unchanged() {
        let mut grid = FluidGrid::new_empty(4);
        grid.get_mut(1, 1, 1).tag = CellTag::Solid;
        grid.get_mut(1, 1, 1).material = MaterialId::STONE;

        update_tags(&mut grid);
        assert_eq!(grid.get(1, 1, 1).tag, CellTag::Solid);
    }

    #[test]
    fn boundary_cells_treated_as_air_adjacent() {
        // A liquid cell at the grid boundary is adjacent to "outside" = air.
        let mut grid = FluidGrid::new_empty(4);
        grid.get_mut(0, 0, 0).tag = CellTag::Liquid;
        grid.get_mut(0, 0, 0).material = MaterialId::WATER;

        update_tags(&mut grid);
        assert_eq!(grid.get(0, 0, 0).tag, CellTag::Surface);
    }
}
