// Deposition + erosion.
//
// Pure functions for depositing slow-moving particles onto voxel surfaces
// and eroding surface voxels exposed to strong flow.

use super::types::{AccumulationGrid, Particle, ParticleTag};
use crate::world::voxel::{MaterialId, Voxel};

/// Compute the voxel mass for a given material (density × VOXEL_VOLUME = density × 1 m³).
fn material_voxel_mass(material: MaterialId) -> f32 {
    match material.0 {
        3 => 1000.0, // Water: 1000 kg/m³
        8 => 917.0,  // Ice: 917 kg/m³
        9 => 0.6,    // Steam: ~0.6 kg/m³
        11 => 600.0, // Ash: ~600 kg/m³
        2 => 1500.0, // Dirt/Sand: 1500 kg/m³
        _ => 1000.0,
    }
}

/// Check if a voxel is solid (not air, not steam).
fn is_solid(voxel: &Voxel) -> bool {
    !matches!(voxel.material.0, 0 | 9)
}

/// Flat voxel index: z * size² + y * size + x.
#[inline]
fn voxel_index(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Check if position has a solid surface below (for deposition).
///
/// Returns `true` when the voxel at (x, y, z) is air and y-1 is solid.
fn has_surface_below(voxels: &[Voxel], x: usize, y: usize, z: usize, size: usize) -> bool {
    if y == 0 {
        // Bottom of chunk counts as an implicit surface.
        return true;
    }
    let below = voxel_index(x, y - 1, z, size);
    is_solid(&voxels[below])
}

/// Check if position has any adjacent solid neighbor (6-connected).
fn has_adjacent_solid(voxels: &[Voxel], x: usize, y: usize, z: usize, size: usize) -> bool {
    let neighbors: [(i32, i32, i32); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    for (dx, dy, dz) in neighbors {
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
        if is_solid(&voxels[voxel_index(nx, ny, nz, size)]) {
            return true;
        }
    }
    false
}

/// Deposit slow-moving particles onto surfaces.
///
/// For each airborne particle with speed < `deposit_velocity`:
/// 1. Check if particle's voxel position has a suitable surface
/// 2. If so, add particle mass to `AccumulationGrid`
/// 3. Mark particle as `Deposited`
/// 4. If accumulated mass ≥ material_voxel_mass, convert air voxel to material
///
/// Snow (ice): only deposits on upward-facing surfaces (y-1 is solid, y is air).
/// Ash: deposits on any adjacent solid surface.
/// Water/other: deposits on any surface below.
pub fn deposit_particles(
    particles: &mut [Particle],
    voxels: &mut [Voxel],
    accum: &mut AccumulationGrid,
    size: usize,
    deposit_velocity: f32,
) -> usize {
    let mut voxels_changed = 0;

    for particle in particles.iter_mut() {
        if particle.tag != ParticleTag::Airborne {
            continue;
        }
        if particle.speed() >= deposit_velocity {
            continue;
        }

        // Compute voxel coordinates from particle position (floor, clamped).
        let x = (particle.position[0] as usize).min(size - 1);
        let y = (particle.position[1] as usize).min(size - 1);
        let z = (particle.position[2] as usize).min(size - 1);

        let idx = voxel_index(x, y, z, size);
        // Can only deposit into air voxels.
        if !voxels[idx].material.is_air() {
            continue;
        }

        let can_deposit = match particle.material.0 {
            // Ice/snow: only on upward-facing surfaces (solid below, air at current).
            8 => has_surface_below(voxels, x, y, z, size),
            // Ash: any adjacent solid surface.
            11 => has_adjacent_solid(voxels, x, y, z, size),
            // Water and others: surface below.
            _ => has_surface_below(voxels, x, y, z, size),
        };

        if !can_deposit {
            continue;
        }

        // Deposit mass and mark particle.
        accum.add_mass(x, y, z, particle.mass, particle.material);
        particle.tag = ParticleTag::Deposited;

        // Check if accumulated mass fills the voxel.
        let threshold = material_voxel_mass(particle.material);
        if accum.get_mass(x, y, z) >= threshold {
            voxels[idx].material = accum.get_material(x, y, z);
            voxels[idx].temperature = particle.temperature;
            voxels[idx].damage = 0.0;
            accum.reset_cell(x, y, z);
            voxels_changed += 1;
        }
    }

    voxels_changed
}

/// Approximate surface normal at a voxel (gradient of solid indicator).
///
/// Uses central differences on the 6-neighborhood solid indicator
/// to estimate the direction pointing away from the solid interior.
pub fn surface_normal(voxels: &[Voxel], x: usize, y: usize, z: usize, size: usize) -> [f32; 3] {
    // Sample solid indicator: 1.0 if solid, 0.0 if not.
    let sample = |sx: i32, sy: i32, sz: i32| -> f32 {
        if sx < 0 || sy < 0 || sz < 0 {
            return 0.0;
        }
        let (ux, uy, uz) = (sx as usize, sy as usize, sz as usize);
        if ux >= size || uy >= size || uz >= size {
            return 0.0;
        }
        if is_solid(&voxels[voxel_index(ux, uy, uz, size)]) {
            1.0
        } else {
            0.0
        }
    };

    let ix = x as i32;
    let iy = y as i32;
    let iz = z as i32;

    // Gradient of solid indicator (central differences).
    // Normal points from solid toward air (negative gradient of solid density).
    let nx = sample(ix - 1, iy, iz) - sample(ix + 1, iy, iz);
    let ny = sample(ix, iy - 1, iz) - sample(ix, iy + 1, iz);
    let nz = sample(ix, iy, iz - 1) - sample(ix, iy, iz + 1);

    // Normalize.
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-8 {
        return [0.0, 0.0, 0.0];
    }
    [nx / len, ny / len, nz / len]
}

/// Wind speed threshold for erosion, by material.
fn erosion_threshold(material: MaterialId) -> f32 {
    match material.0 {
        2 => 5.0,  // Dirt/Sand: 5 m/s
        11 => 6.0, // Ash: 6 m/s
        1 => 25.0, // Stone: 25 m/s
        8 => 10.0, // Ice: 10 m/s
        _ => 15.0, // Default
    }
}

/// Erosion rate coefficient (damage per second per m/s of excess wind).
const EROSION_RATE: f32 = 0.05;

/// Erode surface voxels exposed to strong flow.
///
/// For surface voxels (air above, solid at cell) where wind speed exceeds
/// the material's erosion threshold:
/// - Increment voxel damage proportional to (wind_excess × dt)
/// - When damage ≥ 1.0, convert voxel to air and emit a particle
pub fn erode_surface(
    voxels: &mut [Voxel],
    size: usize,
    wind_velocity: [f32; 3],
    particles: &mut Vec<Particle>,
    dt: f32,
) -> usize {
    let wind_speed = (wind_velocity[0] * wind_velocity[0]
        + wind_velocity[1] * wind_velocity[1]
        + wind_velocity[2] * wind_velocity[2])
        .sqrt();

    let mut emitted = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = voxel_index(x, y, z, size);
                let voxel = &voxels[idx];

                if !is_solid(voxel) {
                    continue;
                }
                // Must be a surface voxel: has air above.
                if y + 1 < size {
                    let above = voxel_index(x, y + 1, z, size);
                    if is_solid(&voxels[above]) {
                        continue;
                    }
                }

                let threshold = erosion_threshold(voxel.material);
                if wind_speed <= threshold {
                    continue;
                }

                let excess = wind_speed - threshold;
                let damage_increment = EROSION_RATE * excess * dt;

                // We need to read material before mutating.
                let mat = voxels[idx].material;
                let temp = voxels[idx].temperature;
                voxels[idx].damage += damage_increment;

                if voxels[idx].damage >= 1.0 {
                    // Emit a particle with the eroded material.
                    let pos = [x as f32 + 0.5, y as f32 + 1.0, z as f32 + 0.5];
                    let vel = [
                        wind_velocity[0] * 0.3,
                        1.0, // slight upward kick
                        wind_velocity[2] * 0.3,
                    ];
                    let mass = material_voxel_mass(mat) * 0.001; // small fragment
                    let p = Particle::new(pos, vel, mass, mat).with_temperature(temp);
                    particles.push(p);
                    emitted += 1;

                    // Convert voxel to air.
                    voxels[idx].material = MaterialId::AIR;
                    voxels[idx].damage = 0.0;
                }
            }
        }
    }

    emitted
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small voxel grid (size³) filled with air.
    fn air_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    /// Helper: set a voxel's material in the flat array.
    fn set_mat(voxels: &mut [Voxel], x: usize, y: usize, z: usize, size: usize, mat: MaterialId) {
        let idx = voxel_index(x, y, z, size);
        voxels[idx].material = mat;
    }

    #[test]
    fn slow_particle_near_surface_gets_deposited() {
        let size = 4;
        let mut voxels = air_grid(size);
        // Place stone at y=0 as a surface.
        set_mat(&mut voxels, 1, 0, 1, size, MaterialId::STONE);

        let mut accum = AccumulationGrid::new(size);
        let mut particles = vec![Particle::new(
            [1.0, 1.0, 1.0], // y=1, above stone at y=0
            [0.1, 0.0, 0.0], // slow: speed = 0.1 m/s
            0.5,
            MaterialId::WATER,
        )];

        deposit_particles(&mut particles, &mut voxels, &mut accum, size, 0.5);

        assert_eq!(particles[0].tag, ParticleTag::Deposited);
    }

    #[test]
    fn fast_particle_is_not_deposited() {
        let size = 4;
        let mut voxels = air_grid(size);
        set_mat(&mut voxels, 1, 0, 1, size, MaterialId::STONE);

        let mut accum = AccumulationGrid::new(size);
        let mut particles = vec![Particle::new(
            [1.0, 1.0, 1.0],
            [5.0, 0.0, 0.0], // fast: speed = 5.0 m/s
            0.5,
            MaterialId::WATER,
        )];

        deposit_particles(&mut particles, &mut voxels, &mut accum, size, 0.5);

        assert_eq!(particles[0].tag, ParticleTag::Airborne);
    }

    #[test]
    fn accumulated_mass_fills_voxel() {
        let size = 4;
        let mut voxels = air_grid(size);
        set_mat(&mut voxels, 2, 0, 2, size, MaterialId::STONE);

        let mut accum = AccumulationGrid::new(size);

        // Pre-load accumulation grid so next deposit pushes it over threshold.
        // Ice threshold = 917 kg. Deposit 910 + 10 = 920 > 917.
        accum.add_mass(2, 1, 2, 910.0, MaterialId::ICE);

        let mut particles = vec![Particle::new(
            [2.0, 1.0, 2.0],
            [0.0, 0.0, 0.0], // stationary
            10.0,
            MaterialId::ICE,
        )];

        let changed = deposit_particles(&mut particles, &mut voxels, &mut accum, size, 0.5);

        assert_eq!(changed, 1);
        let idx = voxel_index(2, 1, 2, size);
        assert_eq!(voxels[idx].material, MaterialId::ICE);
        // Accumulation should be reset after conversion.
        assert!(accum.get_mass(2, 1, 2) < 1e-6);
    }

    #[test]
    fn snow_only_deposits_on_upward_surfaces() {
        let size = 4;
        let mut voxels = air_grid(size);
        // Place a solid wall on the side (x=0, y=1, z=1) — no solid below (1,1,1).
        set_mat(&mut voxels, 0, 1, 1, size, MaterialId::STONE);

        let mut accum = AccumulationGrid::new(size);
        let mut particles = vec![Particle::new(
            [1.0, 1.0, 1.0], // adjacent to wall but no solid below
            [0.0, 0.0, 0.0],
            0.5,
            MaterialId::ICE, // snow/ice
        )];

        deposit_particles(&mut particles, &mut voxels, &mut accum, size, 0.5);

        // Snow should NOT deposit — no solid below at (1,0,1).
        assert_eq!(particles[0].tag, ParticleTag::Airborne);

        // Now place solid below and try again.
        set_mat(&mut voxels, 1, 0, 1, size, MaterialId::STONE);
        particles[0].tag = ParticleTag::Airborne;

        deposit_particles(&mut particles, &mut voxels, &mut accum, size, 0.5);

        // Now snow SHOULD deposit — solid below.
        assert_eq!(particles[0].tag, ParticleTag::Deposited);
    }

    #[test]
    fn erosion_damages_and_converts_voxel() {
        let size = 4;
        let mut voxels = air_grid(size);
        // Place a dirt surface at y=1 with air above.
        set_mat(&mut voxels, 1, 1, 1, size, MaterialId::DIRT);

        let mut particles = Vec::new();
        let wind = [20.0, 0.0, 0.0]; // 20 m/s, well above dirt threshold of 5

        // Apply erosion repeatedly until voxel is destroyed.
        let mut total_emitted = 0;
        for _ in 0..100 {
            total_emitted += erode_surface(&mut voxels, size, wind, &mut particles, 0.1);
            let idx = voxel_index(1, 1, 1, size);
            if voxels[idx].material.is_air() {
                break;
            }
        }

        // Voxel should have been converted to air.
        let idx = voxel_index(1, 1, 1, size);
        assert!(
            voxels[idx].material.is_air(),
            "Voxel should have eroded to air"
        );
        assert!(
            total_emitted > 0,
            "Should have emitted at least one particle"
        );
        assert!(!particles.is_empty());
        assert_eq!(particles[0].material, MaterialId::DIRT);
    }

    #[test]
    fn surface_normal_points_away_from_solid() {
        let size = 4;
        let mut voxels = air_grid(size);
        // Fill bottom half (y=0,1) with stone.
        for z in 0..size {
            for x in 0..size {
                set_mat(&mut voxels, x, 0, z, size, MaterialId::STONE);
                set_mat(&mut voxels, x, 1, z, size, MaterialId::STONE);
            }
        }

        // Normal at the solid surface (2, 1, 2): y-1=0 solid, y+1=2 air.
        // gradient ny = sample(iy-1) - sample(iy+1) = 1.0 - 0.0 = +1.0.
        // Normal points upward: from solid interior toward air.
        let n = surface_normal(&voxels, 2, 1, 2, size);
        assert!(
            n[1] > 0.5,
            "Normal y should be positive (pointing from solid toward air), got {:?}",
            n
        );

        // At the air voxel just above: (2, 2, 2). y-1=1 solid, y+1=3 air.
        let n_air = surface_normal(&voxels, 2, 2, 2, size);
        assert!(
            n_air[1] > 0.5,
            "Normal should point upward at air voxel above surface, got {:?}",
            n_air
        );
    }

    #[test]
    fn ash_deposits_on_wall_surface() {
        let size = 4;
        let mut voxels = air_grid(size);
        // Place a wall: stone at (0, 1, 1) — adjacent to (1, 1, 1).
        set_mat(&mut voxels, 0, 1, 1, size, MaterialId::STONE);

        let mut accum = AccumulationGrid::new(size);
        let mut particles = vec![Particle::new(
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0], // stationary
            0.5,
            MaterialId::ASH,
        )];

        deposit_particles(&mut particles, &mut voxels, &mut accum, size, 0.5);

        // Ash should deposit on any adjacent solid.
        assert_eq!(particles[0].tag, ParticleTag::Deposited);
    }

    #[test]
    fn no_erosion_below_threshold() {
        let size = 4;
        let mut voxels = air_grid(size);
        set_mat(&mut voxels, 1, 0, 1, size, MaterialId::DIRT);

        let mut particles = Vec::new();
        let wind = [3.0, 0.0, 0.0]; // 3 m/s, below dirt threshold of 5

        erode_surface(&mut voxels, size, wind, &mut particles, 1.0);

        let idx = voxel_index(1, 0, 1, size);
        assert_eq!(voxels[idx].damage, 0.0, "No damage below threshold");
        assert!(particles.is_empty());
    }

    #[test]
    fn material_voxel_mass_values() {
        assert!((material_voxel_mass(MaterialId::WATER) - 1000.0).abs() < 1e-6);
        assert!((material_voxel_mass(MaterialId::ICE) - 917.0).abs() < 1e-6);
        assert!((material_voxel_mass(MaterialId::ASH) - 600.0).abs() < 1e-6);
        assert!((material_voxel_mass(MaterialId::DIRT) - 1500.0).abs() < 1e-6);
    }
}
