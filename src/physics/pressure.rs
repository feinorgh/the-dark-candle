// Gas pressure propagation on the voxel grid.
//
// Pressure diffuses through air and gas voxels (steam), blocked by solids
// and liquids. Used for:
//   - Explosion shockwaves (high-pressure source → radial expansion)
//   - Ventilation (pressure equalization through tunnels)
//   - Wind effects (pressure gradients → force on entities)
//
// Model: discrete diffusion similar to heat transfer. Each tick, pressure
// equalizes between neighboring air/gas voxels with a configurable rate.

#![allow(dead_code)]

use crate::world::voxel::{MaterialId, Voxel};

use super::constants;

/// Standard atmospheric pressure at sea level (Pa).
/// Source: NIST / ISO 2533:1975.
pub const ATMOSPHERIC_PRESSURE: f32 = constants::ATMOSPHERIC_PRESSURE;

/// Default diffusion rate per tick (0.0–1.0). Higher = faster equalization.
const DEFAULT_DIFFUSION_RATE: f32 = 0.25;

/// Returns true if pressure can propagate through this material.
fn is_permeable(mat: MaterialId) -> bool {
    mat.is_air() || mat == MaterialId::STEAM
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

/// 3D index into a flat voxel array of size³.
fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Diffuse pressure across a chunk's voxel grid.
///
/// Reads the current state into a snapshot, then updates each permeable voxel's
/// pressure toward the average of its permeable neighbors. Returns the maximum
/// pressure delta applied (useful for convergence checks).
pub fn diffuse_pressure(voxels: &mut [Voxel], size: usize) -> f32 {
    diffuse_pressure_with_rate(voxels, size, DEFAULT_DIFFUSION_RATE)
}

/// Diffuse pressure with a custom rate (for testing or tuning).
pub fn diffuse_pressure_with_rate(voxels: &mut [Voxel], size: usize, rate: f32) -> f32 {
    let total = size * size * size;
    assert_eq!(voxels.len(), total);

    let snapshot: Vec<Voxel> = voxels.to_vec();
    let mut max_delta: f32 = 0.0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let i = idx(x, y, z, size);
                if !is_permeable(snapshot[i].material) {
                    continue;
                }

                let mut sum = 0.0_f32;
                let mut count = 0u32;

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
                    if is_permeable(snapshot[ni].material) {
                        sum += snapshot[ni].pressure;
                        count += 1;
                    }
                }

                if count > 0 {
                    let avg = sum / count as f32;
                    let delta = (avg - snapshot[i].pressure) * rate;
                    voxels[i].pressure = snapshot[i].pressure + delta;
                    max_delta = max_delta.max(delta.abs());
                }
            }
        }
    }

    max_delta
}

/// Apply an explosion at a point: set high pressure at the source.
/// The caller should then call `diffuse_pressure` repeatedly to propagate.
pub fn create_pressure_source(
    voxels: &mut [Voxel],
    size: usize,
    x: usize,
    y: usize,
    z: usize,
    pressure: f32,
) {
    let i = idx(x, y, z, size);
    if is_permeable(voxels[i].material) {
        voxels[i].pressure = pressure;
    }
}

/// Calculate the pressure gradient force at a position (points from high to low pressure).
/// Returns (fx, fy, fz) force vector.
pub fn pressure_gradient(
    voxels: &[Voxel],
    size: usize,
    x: usize,
    y: usize,
    z: usize,
) -> (f32, f32, f32) {
    let mut gx = 0.0_f32;
    let mut gy = 0.0_f32;
    let mut gz = 0.0_f32;

    let center_p = voxels[idx(x, y, z, size)].pressure;

    // X gradient
    if x > 0 {
        let ni = idx(x - 1, y, z, size);
        if is_permeable(voxels[ni].material) {
            gx += center_p - voxels[ni].pressure;
        }
    }
    if x + 1 < size {
        let ni = idx(x + 1, y, z, size);
        if is_permeable(voxels[ni].material) {
            gx -= center_p - voxels[ni].pressure;
        }
    }

    // Y gradient
    if y > 0 {
        let ni = idx(x, y - 1, z, size);
        if is_permeable(voxels[ni].material) {
            gy += center_p - voxels[ni].pressure;
        }
    }
    if y + 1 < size {
        let ni = idx(x, y + 1, z, size);
        if is_permeable(voxels[ni].material) {
            gy -= center_p - voxels[ni].pressure;
        }
    }

    // Z gradient
    if z > 0 {
        let ni = idx(x, y, z - 1, size);
        if is_permeable(voxels[ni].material) {
            gz += center_p - voxels[ni].pressure;
        }
    }
    if z + 1 < size {
        let ni = idx(x, y, z + 1, size);
        if is_permeable(voxels[ni].material) {
            gz -= center_p - voxels[ni].pressure;
        }
    }

    (gx * 0.5, gy * 0.5, gz * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(size: usize) -> Vec<Voxel> {
        let mut grid = vec![Voxel::default(); size * size * size];
        for v in &mut grid {
            v.pressure = ATMOSPHERIC_PRESSURE;
        }
        grid
    }

    #[test]
    fn atmospheric_pressure_is_si() {
        // Source: NIST / ISO 2533 — sea level pressure is 101325 Pa
        assert_eq!(ATMOSPHERIC_PRESSURE, 101_325.0);
    }

    #[test]
    fn uniform_pressure_is_stable() {
        let mut grid = make_grid(4);
        let delta = diffuse_pressure(&mut grid, 4);
        assert!(delta < 1e-1, "Uniform pressure should not change");
    }

    #[test]
    fn high_pressure_diffuses_to_neighbors() {
        let mut grid = make_grid(4);
        // Explosion-level overpressure: 10 atm at center
        grid[idx(2, 2, 2, 4)].pressure = 1_013_250.0;

        let delta = diffuse_pressure(&mut grid, 4);
        assert!(delta > 0.0, "Pressure should spread");

        // Center should have decreased
        assert!(grid[idx(2, 2, 2, 4)].pressure < 1_013_250.0);
        // Neighbor should have increased
        assert!(grid[idx(1, 2, 2, 4)].pressure > ATMOSPHERIC_PRESSURE);
    }

    #[test]
    fn pressure_blocked_by_solid() {
        let mut grid = make_grid(4);
        // Wall of stone between x=1 and x=2
        for y in 0..4 {
            for z in 0..4 {
                grid[idx(2, y, z, 4)].material = MaterialId::STONE;
                grid[idx(2, y, z, 4)].pressure = 0.0; // Solids don't participate
            }
        }

        // High pressure on one side
        grid[idx(1, 2, 2, 4)].pressure = 1_013_250.0;

        diffuse_pressure(&mut grid, 4);

        // Pressure should not cross the wall
        assert!(
            (grid[idx(3, 2, 2, 4)].pressure - ATMOSPHERIC_PRESSURE).abs() < 1.0,
            "Pressure should not cross solid wall"
        );
    }

    #[test]
    fn explosion_propagates_radially() {
        let mut grid = make_grid(8);
        // 50 atm explosion source
        create_pressure_source(&mut grid, 8, 4, 4, 4, 5_066_250.0);

        // Run several diffusion ticks
        for _ in 0..20 {
            diffuse_pressure(&mut grid, 8);
        }

        // Center should still be above atmospheric
        assert!(grid[idx(4, 4, 4, 8)].pressure > ATMOSPHERIC_PRESSURE);

        // Neighbors closer to center should have higher pressure than distant ones
        let near = grid[idx(3, 4, 4, 8)].pressure;
        let far = grid[idx(1, 4, 4, 8)].pressure;
        assert!(
            near >= far,
            "Closer voxels should have higher pressure: near={near}, far={far}"
        );
    }

    #[test]
    fn pressure_converges_to_equilibrium() {
        let mut grid = make_grid(4);
        // 20 atm at center
        grid[idx(2, 2, 2, 4)].pressure = 2_026_500.0;

        // Run many ticks
        for _ in 0..200 {
            diffuse_pressure(&mut grid, 4);
        }

        // All air voxels should converge to roughly the same pressure
        let pressures: Vec<f32> = grid
            .iter()
            .filter(|v| is_permeable(v.material))
            .map(|v| v.pressure)
            .collect();

        let avg = pressures.iter().sum::<f32>() / pressures.len() as f32;
        for &p in &pressures {
            assert!(
                (p - avg).abs() < 100.0,
                "Pressure should converge: p={p}, avg={avg}"
            );
        }
    }

    #[test]
    fn steam_is_permeable() {
        let mut grid = make_grid(4);
        // Fill a row with steam
        for x in 0..4 {
            grid[idx(x, 2, 2, 4)].material = MaterialId::STEAM;
        }
        grid[idx(0, 2, 2, 4)].pressure = 506_625.0; // 5 atm

        diffuse_pressure(&mut grid, 4);

        // Pressure should spread through steam
        assert!(
            grid[idx(1, 2, 2, 4)].pressure > ATMOSPHERIC_PRESSURE,
            "Pressure should diffuse through steam"
        );
    }

    #[test]
    fn pressure_gradient_points_from_high_to_low() {
        let mut grid = make_grid(4);
        grid[idx(0, 2, 2, 4)].pressure = 506_625.0; // 5 atm
        grid[idx(2, 2, 2, 4)].pressure = ATMOSPHERIC_PRESSURE;

        let (gx, _, _) = pressure_gradient(&grid, 4, 1, 2, 2);
        // Gradient should push away from high pressure
        assert!(
            gx < 0.0,
            "Gradient should push away from high pressure: gx={gx}"
        );
    }

    #[test]
    fn zero_rate_means_no_diffusion() {
        let mut grid = make_grid(4);
        grid[idx(2, 2, 2, 4)].pressure = 5_066_250.0;

        let delta = diffuse_pressure_with_rate(&mut grid, 4, 0.0);
        assert!(delta < 1e-1, "Zero rate should mean no change");
        assert_eq!(grid[idx(2, 2, 2, 4)].pressure, 5_066_250.0);
    }

    #[test]
    fn create_pressure_source_on_solid_does_nothing() {
        let mut grid = make_grid(4);
        grid[idx(2, 2, 2, 4)].material = MaterialId::STONE;
        let original = grid[idx(2, 2, 2, 4)].pressure;

        create_pressure_source(&mut grid, 4, 2, 2, 2, 10_000_000.0);
        assert_eq!(
            grid[idx(2, 2, 2, 4)].pressure,
            original,
            "Solid voxels should not become pressure sources"
        );
    }

    // --- Real-world validation tests ---

    #[test]
    fn hydrostatic_pressure_at_depth() {
        // Wikipedia: Hydrostatic pressure P = ρgh
        // 10 m of water: 1000 × 9.80665 × 10 = 98066.5 Pa
        // Total at 10 m depth: 101325 + 98066.5 = 199391.5 Pa
        let water_density = 1000.0_f32;
        let depth = 10.0_f32;
        let p_total = ATMOSPHERIC_PRESSURE + water_density * constants::GRAVITY * depth;
        assert!(
            (p_total - 199_391.5).abs() < 1.0,
            "Got {p_total} Pa, expected ~199391.5 Pa"
        );
    }

    #[test]
    fn barometric_pressure_at_altitude() {
        // Wikipedia: Barometric formula — pressure at 1000 m ≈ 89875 Pa
        let p = constants::pressure_at_altitude(1000.0, constants::SEA_LEVEL_TEMPERATURE);
        assert!(
            (p - 89_875.0).abs() < 500.0,
            "Got {p} Pa at 1000m, expected ~89875 Pa"
        );
    }
}
