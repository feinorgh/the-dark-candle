// Population streaming (propagation) for D3Q19 lattice.
//
// The streaming step moves distribution functions along their respective
// lattice velocity directions. This is the "advection" of the LBM — each
// f_i at position x moves to position x + e_i.
//
// Uses double-buffering (returns a new grid) to avoid read-write conflicts.
// Solid and liquid cells use half-way bounce-back: populations streaming
// into a wall are reflected back in the opposite direction.

use super::lattice::{E, OPPOSITE, Q};
use super::types::LbmGrid;

/// Stream all populations to their neighbors. Returns a new grid.
///
/// For gas cells: f_i(x+e_i, t+1) = f_i_post(x, t)
/// For wall cells: half-way bounce-back — f_opp(x, t+1) = f_i_post(x, t)
/// At grid edges: populations streaming out are lost (open boundary).
pub fn stream(grid: &LbmGrid) -> LbmGrid {
    let size = grid.size();
    let mut out = LbmGrid::new_empty(size);

    // Zero all gas cell distributions (new_empty initializes to equilibrium)
    for cell in out.cells_mut() {
        cell.f = [0.0; Q];
    }

    // Copy wall cells unchanged
    for idx in 0..grid.cells().len() {
        let cell = &grid.cells()[idx];
        if cell.is_wall() {
            out.cells_mut()[idx] = *cell;
        }
    }

    // Stream gas cells
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let src = grid.get(x, y, z);
                if !src.is_gas() {
                    continue;
                }

                for i in 0..Q {
                    let nx = x as i32 + E[i][0];
                    let ny = y as i32 + E[i][1];
                    let nz = z as i32 + E[i][2];

                    // Check grid bounds
                    if nx < 0
                        || nx >= size as i32
                        || ny < 0
                        || ny >= size as i32
                        || nz < 0
                        || nz >= size as i32
                    {
                        // Open boundary: population is lost (absorbed)
                        // Compensate by keeping it in place (zero-gradient)
                        let dst_idx = grid.index(x, y, z);
                        out.cells_mut()[dst_idx].f[i] += src.f[i];
                        continue;
                    }

                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nz = nz as usize;

                    let neighbor = grid.get(nx, ny, nz);

                    if neighbor.is_wall() {
                        // Half-way bounce-back: reflect into opposite direction at source
                        let opp = OPPOSITE[i];
                        let src_idx = grid.index(x, y, z);
                        out.cells_mut()[src_idx].f[opp] += src.f[i];
                    } else {
                        // Normal streaming: f_i moves to neighbor
                        let dst_idx = grid.index(nx, ny, nz);
                        out.cells_mut()[dst_idx].f[i] += src.f[i];
                    }
                }
            }
        }
    }

    // Preserve material, tag, and scalar fields for gas cells
    for idx in 0..grid.cells().len() {
        let cell = &grid.cells()[idx];
        if cell.is_gas() {
            out.cells_mut()[idx].material = cell.material;
            out.cells_mut()[idx].tag = cell.tag;
        }
    }

    // Advect moisture and cloud_lwc as passive scalars using upwind scheme.
    // Each gas cell's scalars move in the direction of its macroscopic velocity.
    advect_scalars(grid, &mut out);

    out
}

/// Advect moisture and cloud_lwc using first-order upwind from macroscopic velocity.
///
/// For each gas cell, the scalar value is distributed to downstream neighbors
/// proportionally to the velocity-weighted flux through each face. This is
/// simpler than a full scalar-LBM but sufficient for humidity transport.
fn advect_scalars(src: &LbmGrid, dst: &mut LbmGrid) {
    let size = src.size();

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let cell = src.get(x, y, z);
                if !cell.is_gas() {
                    continue;
                }

                let u = cell.velocity();
                let speed = u[0].abs() + u[1].abs() + u[2].abs();

                if speed < 1e-12 || (cell.moisture < 1e-12 && cell.cloud_lwc < 1e-12) {
                    // No velocity or nothing to transport — keep in place
                    let idx = src.index(x, y, z);
                    dst.cells_mut()[idx].moisture += cell.moisture;
                    dst.cells_mut()[idx].cloud_lwc += cell.cloud_lwc;
                    continue;
                }

                // Fraction transported = min(speed, 1.0) to maintain stability
                let transport_frac = speed.min(1.0);
                let remain_frac = 1.0 - transport_frac;

                // Keep remainder in place
                let idx = src.index(x, y, z);
                dst.cells_mut()[idx].moisture += cell.moisture * remain_frac;
                dst.cells_mut()[idx].cloud_lwc += cell.cloud_lwc * remain_frac;

                // Distribute transported fraction along velocity components
                let dirs: [(i32, i32, i32, f32); 6] = [
                    (1, 0, 0, u[0].max(0.0)),
                    (-1, 0, 0, (-u[0]).max(0.0)),
                    (0, 1, 0, u[1].max(0.0)),
                    (0, -1, 0, (-u[1]).max(0.0)),
                    (0, 0, 1, u[2].max(0.0)),
                    (0, 0, -1, (-u[2]).max(0.0)),
                ];

                for (dx, dy, dz, flux) in dirs {
                    if flux < 1e-12 {
                        continue;
                    }
                    let weight = flux / speed;
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;

                    if nx < 0
                        || nx >= size as i32
                        || ny < 0
                        || ny >= size as i32
                        || nz < 0
                        || nz >= size as i32
                    {
                        // Boundary: keep scalar in place
                        dst.cells_mut()[idx].moisture += cell.moisture * transport_frac * weight;
                        dst.cells_mut()[idx].cloud_lwc += cell.cloud_lwc * transport_frac * weight;
                        continue;
                    }

                    let neighbor = src.get(nx as usize, ny as usize, nz as usize);
                    if neighbor.is_wall() {
                        // Wall: keep scalar in place (no penetration)
                        dst.cells_mut()[idx].moisture += cell.moisture * transport_frac * weight;
                        dst.cells_mut()[idx].cloud_lwc += cell.cloud_lwc * transport_frac * weight;
                    } else {
                        let nidx = src.index(nx as usize, ny as usize, nz as usize);
                        dst.cells_mut()[nidx].moisture += cell.moisture * transport_frac * weight;
                        dst.cells_mut()[nidx].cloud_lwc += cell.cloud_lwc * transport_frac * weight;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::lbm_gas::types::{GasCellTag, LbmCell};
    use crate::world::voxel::MaterialId;

    #[test]
    fn streaming_preserves_mass_in_closed_box() {
        // All-gas grid with walls around edges → mass is conserved
        let size = 6;
        let mut grid = LbmGrid::new_empty(size);

        // Create walls on all faces
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x == size - 1 || y == 0 || y == size - 1 || z == 0 || z == size - 1
                    {
                        *grid.get_mut(x, y, z) = LbmCell::new_solid(MaterialId::STONE);
                    }
                }
            }
        }

        // Add a density perturbation in the center
        let center = grid.get_mut(3, 3, 3);
        *center = LbmCell::new_gas(MaterialId::AIR, 1.5);

        let mass_before = grid.total_mass();
        let streamed = stream(&grid);
        let mass_after = streamed.total_mass();

        assert!(
            (mass_after - mass_before).abs() < 1e-3,
            "Mass changed: {mass_before} → {mass_after}"
        );
    }

    #[test]
    fn streaming_moves_populations() {
        let size = 8;
        let mut grid = LbmGrid::new_empty(size);

        // Set all cells to zero distributions except one
        for cell in grid.cells_mut() {
            cell.f = [0.0; Q];
        }

        // Place a population in direction +x (index 1) at center
        let center = grid.get_mut(4, 4, 4);
        center.f[1] = 1.0; // +x direction

        let streamed = stream(&grid);

        // Population should have moved to (5,4,4) in direction 1
        let f_at_dest = streamed.get(5, 4, 4).f[1];
        assert!(
            f_at_dest > 0.99,
            "Population did not move to expected cell: f={f_at_dest}"
        );
    }

    #[test]
    fn bounce_back_reverses_direction() {
        let size = 4;
        let mut grid = LbmGrid::new_empty(size);

        // Clear all distributions
        for cell in grid.cells_mut() {
            cell.f = [0.0; Q];
        }

        // Place a solid wall at (2,1,1)
        *grid.get_mut(2, 1, 1) = LbmCell::new_solid(MaterialId::STONE);

        // Place a +x population at (1,1,1) heading toward the wall
        grid.get_mut(1, 1, 1).f[1] = 1.0; // direction +x

        let streamed = stream(&grid);

        // Should bounce back: f[2] (-x) at (1,1,1)
        let bounced = streamed.get(1, 1, 1).f[2];
        assert!(
            bounced > 0.99,
            "Bounce-back failed: f[-x] at source = {bounced}"
        );
    }

    #[test]
    fn wall_cells_unchanged_after_streaming() {
        let size = 4;
        let mut grid = LbmGrid::new_empty(size);
        *grid.get_mut(0, 0, 0) = LbmCell::new_solid(MaterialId::STONE);

        let streamed = stream(&grid);
        assert_eq!(streamed.get(0, 0, 0).tag, GasCellTag::Solid);
        assert_eq!(streamed.get(0, 0, 0).material, MaterialId::STONE);
    }

    #[test]
    fn uniform_field_stays_uniform() {
        // All cells at equilibrium with zero velocity → streaming should produce
        // the same field (equilibrium is isotropic).
        let size = 6;
        let grid = LbmGrid::new_empty(size);
        let total_before = grid.total_mass();
        let streamed = stream(&grid);
        let total_after = streamed.total_mass();

        // Mass won't be perfectly conserved due to open boundaries, but
        // interior cells should remain close to equilibrium
        let center = streamed.get(3, 3, 3);
        let rho = center.density();
        assert!((rho - 1.0).abs() < 0.01, "Interior density deviated: {rho}");
        // Total mass may drop slightly at open boundaries
        assert!(
            total_after <= total_before + 0.1,
            "Mass increased: {total_before} → {total_after}"
        );
    }

    #[test]
    fn liquid_cells_act_as_walls() {
        let size = 4;
        let mut grid = LbmGrid::new_empty(size);

        for cell in grid.cells_mut() {
            cell.f = [0.0; Q];
        }

        // Liquid wall at (2,1,1)
        *grid.get_mut(2, 1, 1) = LbmCell::new_liquid(MaterialId::WATER);

        // +x population heading toward liquid
        grid.get_mut(1, 1, 1).f[1] = 1.0;

        let streamed = stream(&grid);

        // Should bounce back just like a solid wall
        let bounced = streamed.get(1, 1, 1).f[2];
        assert!(
            bounced > 0.99,
            "Liquid bounce-back failed: f[-x] = {bounced}"
        );
    }
}
