// Pressure projection on the MAC velocity grid.

use super::types::VelocityGrid;
use crate::world::voxel::{MaterialId, Voxel};

fn cell_index(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Compute divergence of the velocity field (∇·u) at each cell center.
///
/// Uses central differences on the staggered grid. Since dx = dy = dz = 1 (voxel
/// scale), divergence is simply the difference of adjacent face velocities.
/// Solid cells have zero divergence.
pub fn compute_divergence(grid: &VelocityGrid, solid: &[bool], size: usize) -> Vec<f32> {
    let vol = size * size * size;
    let mut div = vec![0.0; vol];

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = cell_index(x, y, z, size);
                if solid[idx] {
                    continue;
                }
                div[idx] = (grid.get_u(x + 1, y, z) - grid.get_u(x, y, z))
                    + (grid.get_v(x, y + 1, z) - grid.get_v(x, y, z))
                    + (grid.get_w(x, y, z + 1) - grid.get_w(x, y, z));
            }
        }
    }

    div
}

/// Jacobi pressure solve: make velocity field divergence-free.
///
/// Iteratively solves ∇²p = divergence using Jacobi relaxation.
/// For each non-solid cell, pressure is updated as:
///   p_new = (sum_of_non_solid_neighbor_pressures − divergence) / neighbor_count
pub fn pressure_solve_jacobi(
    divergence: &[f32],
    solid: &[bool],
    size: usize,
    iterations: usize,
) -> Vec<f32> {
    let vol = size * size * size;
    let mut pressure = vec![0.0; vol];
    let mut scratch = vec![0.0; vol];

    const NEIGHBORS: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    let s = size as isize;

    for _ in 0..iterations {
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let idx = cell_index(x, y, z, size);
                    if solid[idx] {
                        scratch[idx] = 0.0;
                        continue;
                    }

                    let p_center = pressure[idx];
                    let mut sum = 0.0_f32;
                    let mut fluid_count = 0_u32;

                    for &(dx, dy, dz) in &NEIGHBORS {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        let nz = z as isize + dz;

                        if nx >= 0 && nx < s && ny >= 0 && ny < s && nz >= 0 && nz < s {
                            let nidx = cell_index(nx as usize, ny as usize, nz as usize, size);
                            if !solid[nidx] {
                                sum += pressure[nidx];
                            }
                            // Solid in-bounds neighbors contribute 0 (Dirichlet wall)
                        } else {
                            // Out-of-bounds: Neumann BC (dp/dn = 0 → p_ghost = p_center)
                            sum += p_center;
                        }
                        fluid_count += 1;
                    }

                    scratch[idx] = if fluid_count > 0 {
                        (sum - divergence[idx]) / fluid_count as f32
                    } else {
                        0.0
                    };
                }
            }
        }

        std::mem::swap(&mut pressure, &mut scratch);
    }

    pressure
}

/// Subtract pressure gradient from velocity field.
///
/// For each interior face, subtracts the pressure difference across the face.
/// Faces adjacent to solid cells or domain boundaries are left unchanged.
pub fn apply_pressure_gradient(
    grid: &mut VelocityGrid,
    pressure: &[f32],
    solid: &[bool],
    size: usize,
) {
    // U-faces at (i, y, z): between cells (i−1,y,z) and (i,y,z)
    for z in 0..size {
        for y in 0..size {
            for i in 1..size {
                let left = cell_index(i - 1, y, z, size);
                let right = cell_index(i, y, z, size);
                if solid[left] || solid[right] {
                    continue;
                }
                let idx = grid.u_index(i, y, z);
                grid.u[idx] -= pressure[right] - pressure[left];
            }
        }
    }

    // V-faces at (x, j, z): between cells (x,j−1,z) and (x,j,z)
    for z in 0..size {
        for j in 1..size {
            for x in 0..size {
                let below = cell_index(x, j - 1, z, size);
                let above = cell_index(x, j, z, size);
                if solid[below] || solid[above] {
                    continue;
                }
                let idx = grid.v_index(x, j, z);
                grid.v[idx] -= pressure[above] - pressure[below];
            }
        }
    }

    // W-faces at (x, y, k): between cells (x,y,k−1) and (x,y,k)
    for k in 1..size {
        for y in 0..size {
            for x in 0..size {
                let back = cell_index(x, y, k - 1, size);
                let front = cell_index(x, y, k, size);
                if solid[back] || solid[front] {
                    continue;
                }
                let idx = grid.w_index(x, y, k);
                grid.w[idx] -= pressure[front] - pressure[back];
            }
        }
    }
}

/// Build a solid mask from a flat voxel array.
///
/// Returns `true` for solid/liquid voxels that should act as walls.
/// Gas-phase materials (air, steam) and particle materials (ash) are non-solid.
pub fn make_solid_mask_from_voxels(voxels: &[Voxel], size: usize) -> Vec<bool> {
    debug_assert_eq!(voxels.len(), size * size * size);
    voxels
        .iter()
        .map(|v| {
            let m = v.material;
            m != MaterialId::AIR && m != MaterialId::STEAM && m != MaterialId::ASH
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_fluid(size: usize) -> Vec<bool> {
        vec![false; size * size * size]
    }

    #[test]
    fn zero_velocity_zero_divergence() {
        let grid = VelocityGrid::new(4);
        let div = compute_divergence(&grid, &all_fluid(4), 4);
        assert!(div.iter().all(|&d| d.abs() < 1e-10));
    }

    #[test]
    fn uniform_velocity_zero_divergence() {
        let mut grid = VelocityGrid::new(4);
        grid.u.fill(3.0);
        grid.v.fill(-1.0);
        grid.w.fill(2.0);
        let div = compute_divergence(&grid, &all_fluid(4), 4);
        assert!(
            div.iter().all(|&d| d.abs() < 1e-10),
            "max div = {}",
            div.iter().map(|d| d.abs()).fold(0.0_f32, f32::max),
        );
    }

    #[test]
    fn pressure_solve_reduces_source_divergence() {
        let mut grid = VelocityGrid::new(4);
        grid.set_u(3, 1, 1, 1.0);
        let solid = all_fluid(4);
        let div = compute_divergence(&grid, &solid, 4);

        let source = cell_index(2, 1, 1, 4);
        let sink = cell_index(3, 1, 1, 4);
        assert!((div[source] - 1.0).abs() < 1e-6);
        assert!((div[sink] - (-1.0)).abs() < 1e-6);

        let pressure = pressure_solve_jacobi(&div, &solid, 4, 100);
        assert!(
            pressure[source].abs() > 1e-6,
            "pressure at source should be non-zero",
        );
    }

    #[test]
    fn divergence_free_after_projection() {
        let mut grid = VelocityGrid::new(4);
        grid.set_u(3, 1, 1, 1.0);
        let solid = all_fluid(4);
        let div = compute_divergence(&grid, &solid, 4);
        let pressure = pressure_solve_jacobi(&div, &solid, 4, 100);
        apply_pressure_gradient(&mut grid, &pressure, &solid, 4);

        let div2 = compute_divergence(&grid, &solid, 4);
        let max_div = div2.iter().map(|d| d.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_div < 1e-5,
            "max divergence after projection = {max_div}, expected near zero",
        );
    }

    #[test]
    fn solid_mask_from_voxels() {
        let air = Voxel {
            material: MaterialId::AIR,
            temperature: 288.15,
            pressure: 101325.0,
            damage: 1.0,
            latent_heat_buffer: 0.0,
        };
        let stone = Voxel {
            material: MaterialId::STONE,
            ..air
        };
        let steam = Voxel {
            material: MaterialId::STEAM,
            ..air
        };
        let ash = Voxel {
            material: MaterialId::ASH,
            ..air
        };
        let water = Voxel {
            material: MaterialId::WATER,
            ..air
        };

        let voxels = vec![air, stone, steam, ash, water, air, air, air];
        let mask = make_solid_mask_from_voxels(&voxels, 2);
        assert_eq!(
            mask,
            vec![false, true, false, false, true, false, false, false],
        );
    }
}
