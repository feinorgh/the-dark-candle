// Precipitation: cloud-to-rain/snow conversion via coalescence.
//
// When cloud liquid water content (LWC) exceeds the coalescence threshold,
// precipitation particles are emitted into the FLIP/PIC system. Cloud LWC
// is removed to conserve mass.
//
// Rain vs snow is determined by temperature:
//   T > 273.15 K → rain (water droplets, terminal velocity ~9 m/s)
//   T ≤ 273.15 K → snow (ice crystals, terminal velocity ~1 m/s)
//
// Sub-cloud evaporation (virga): particles passing through unsaturated air
// lose mass at a rate proportional to the saturation deficit.

use super::types::LbmGrid;
use crate::physics::atmosphere::{self, AtmosphereConfig};
use crate::physics::flip_pic::types::Particle;
use crate::world::voxel::MaterialId;

/// Freezing point of water (K).
const FREEZING_POINT: f32 = 273.15;

/// Terminal velocity of rain in m/s (large droplets).
const RAIN_TERMINAL_VELOCITY: f32 = 9.0;

/// Terminal velocity of snow in m/s.
const SNOW_TERMINAL_VELOCITY: f32 = 1.0;

/// Default droplet mass in kg (1 gram).
const DROPLET_MASS: f32 = 0.001;

/// Snow particle mass in kg (lighter than rain).
const SNOW_MASS: f32 = 0.0005;

/// Scan the LBM grid for cells exceeding the cloud coalescence threshold.
/// Emits FLIP/PIC particles (rain or snow) and reduces cloud_lwc accordingly.
///
/// Returns the number of particles emitted.
pub fn precipitate(
    grid: &mut LbmGrid,
    temperatures: &[f32],
    config: &AtmosphereConfig,
    particles: &mut Vec<Particle>,
    dt: f32,
    tick: u64,
) -> usize {
    let size = grid.size();
    let threshold = config.cloud_coalescence_threshold;
    let mut emitted = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let cell = grid.get(x, y, z);
                if !cell.is_gas() || cell.cloud_lwc <= threshold {
                    continue;
                }

                let idx = grid.index(x, y, z);
                let temp = temperatures[idx];

                // Precipitation rate: excess LWC above threshold converts to particles.
                // Rate-limited by dt to avoid draining the entire cloud in one step.
                let excess = cell.cloud_lwc - threshold;
                let conversion_rate = 0.1; // 10% of excess per second
                let mass_to_convert = (excess * conversion_rate * dt).min(excess * 0.5);

                if mass_to_convert < 1e-8 {
                    continue;
                }

                // Determine particle type from temperature
                let (material, velocity, particle_mass) = if temp > FREEZING_POINT {
                    (MaterialId::WATER, -RAIN_TERMINAL_VELOCITY, DROPLET_MASS)
                } else {
                    (MaterialId::ICE, -SNOW_TERMINAL_VELOCITY, SNOW_MASS)
                };

                // Number of full-size particles, plus one smaller remainder particle
                // to avoid losing mass to integer rounding.
                let n_full = (mass_to_convert / particle_mass).floor() as usize;
                let remainder = mass_to_convert - n_full as f32 * particle_mass;
                let has_remainder = remainder > 1e-8;
                let n_particles = n_full + if has_remainder { 1 } else { 0 };

                if n_particles == 0 {
                    continue;
                }

                let actual_mass = if has_remainder {
                    n_full as f32 * particle_mass + remainder
                } else {
                    n_full as f32 * particle_mass
                };

                // Emit particles from the bottom of this cell
                for i in 0..n_particles {
                    let jitter = deterministic_offset(x, y, z, tick, i);
                    let pos = [
                        x as f32 + 0.5 + jitter[0] * 0.4,
                        y as f32, // bottom of cell
                        z as f32 + 0.5 + jitter[2] * 0.4,
                    ];

                    // Wind-advected: add LBM cell velocity to fall velocity
                    let u = cell.velocity();
                    let vel = [u[0], velocity + u[1], u[2]];

                    // Last particle gets the remainder mass
                    let pmass = if i == n_particles - 1 && has_remainder {
                        remainder
                    } else {
                        particle_mass
                    };

                    let p = Particle::new(pos, vel, pmass, material).with_temperature(temp);
                    particles.push(p);
                    emitted += 1;
                }

                // Remove mass from cloud LWC (conservation)
                grid.cells_mut()[idx].cloud_lwc -= actual_mass;
                grid.cells_mut()[idx].cloud_lwc = grid.cells_mut()[idx].cloud_lwc.max(0.0);
            }
        }
    }

    emitted
}

/// Apply sub-cloud evaporation (virga) to falling precipitation particles.
///
/// Particles passing through unsaturated air lose mass proportional to the
/// saturation deficit. When mass reaches zero, the particle is fully evaporated.
///
/// Returns the number of particles fully evaporated.
pub fn apply_virga(
    particles: &mut Vec<Particle>,
    grid: &LbmGrid,
    temperatures: &[f32],
    pressures: &[f32],
    dt: f32,
) -> usize {
    let size = grid.size();
    let mut evaporated_count = 0;
    let virga_rate: f32 = 5e-5; // kg/s per unit deficit

    for particle in particles.iter_mut() {
        if !particle.is_airborne() {
            continue;
        }

        // Only rain/snow particles undergo virga
        if particle.material != MaterialId::WATER && particle.material != MaterialId::ICE {
            continue;
        }

        // Find the grid cell this particle is in
        let gx = particle.position[0] as usize;
        let gy = particle.position[1] as usize;
        let gz = particle.position[2] as usize;

        if gx >= size || gy >= size || gz >= size {
            continue;
        }

        let cell = grid.get(gx, gy, gz);
        if !cell.is_gas() {
            continue;
        }

        let idx = grid.index(gx, gy, gz);
        let temp = temperatures[idx];
        let pres = pressures[idx];

        let q_sat = atmosphere::saturation_humidity(temp, pres);
        let q = cell.moisture;

        // Only evaporate in subsaturated air
        if q < q_sat {
            let deficit = q_sat - q;
            let mass_loss = virga_rate * deficit * dt;
            particle.mass -= mass_loss;

            if particle.mass <= 0.0 {
                particle.mass = 0.0;
                evaporated_count += 1;
            }
        }
    }

    // Remove fully evaporated particles
    particles.retain(|p| p.mass > 0.0 || !p.is_airborne());

    evaporated_count
}

/// Deterministic position offset for precipitation particle placement.
fn deterministic_offset(x: usize, y: usize, z: usize, tick: u64, index: usize) -> [f32; 3] {
    let mut h = 0xa538_cd13_u64;
    h = h.wrapping_mul(31).wrapping_add(x as u64);
    h = h.wrapping_mul(31).wrapping_add(y as u64);
    h = h.wrapping_mul(31).wrapping_add(z as u64);
    h = h.wrapping_mul(31).wrapping_add(tick);
    h = h.wrapping_mul(31).wrapping_add(index as u64);

    // Mix
    h ^= h >> 17;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 31;

    let x_off = ((h & 0xFFFF) as f32 / 65535.0) - 0.5;
    let y_off = (((h >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5;
    let z_off = (((h >> 32) & 0xFFFF) as f32 / 65535.0) - 0.5;
    [x_off, y_off, z_off]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::lbm_gas::types::LbmCell;

    fn default_config() -> AtmosphereConfig {
        AtmosphereConfig::default()
    }

    fn make_box_grid(size: usize) -> LbmGrid {
        let mut grid = LbmGrid::new_empty(size);
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
        grid
    }

    #[test]
    fn no_precipitation_below_threshold() {
        let size = 8;
        let mut grid = make_box_grid(size);
        let config = default_config();

        // Set cloud_lwc just below threshold at center
        grid.get_mut(4, 4, 4).cloud_lwc = config.cloud_coalescence_threshold * 0.5;

        let n = size * size * size;
        let temps = vec![288.15_f32; n];
        let mut particles = Vec::new();

        let emitted = precipitate(&mut grid, &temps, &config, &mut particles, 1.0, 0);
        assert_eq!(emitted, 0, "Should not precipitate below threshold");
        assert!(particles.is_empty());
    }

    #[test]
    fn rain_above_threshold_warm() {
        let size = 8;
        let mut grid = make_box_grid(size);
        let config = default_config();

        // Set high cloud_lwc at a warm temperature (rain)
        let lwc_initial = config.cloud_coalescence_threshold * 10.0;
        grid.get_mut(4, 4, 4).cloud_lwc = lwc_initial;

        let n = size * size * size;
        let temps = vec![288.15_f32; n]; // warm → rain
        let mut particles = Vec::new();

        let emitted = precipitate(&mut grid, &temps, &config, &mut particles, 1.0, 0);
        assert!(emitted > 0, "Should emit rain particles");

        // All particles should be water (not ice)
        for p in &particles {
            assert_eq!(p.material, MaterialId::WATER);
            assert!(p.velocity[1] < 0.0, "Rain should fall downward");
        }

        // Cloud LWC should have decreased
        let lwc_after = grid.get(4, 4, 4).cloud_lwc;
        assert!(
            lwc_after < lwc_initial,
            "Cloud LWC should decrease: {lwc_initial} → {lwc_after}"
        );
    }

    #[test]
    fn snow_below_freezing() {
        let size = 8;
        let mut grid = make_box_grid(size);
        let config = default_config();

        grid.get_mut(4, 4, 4).cloud_lwc = config.cloud_coalescence_threshold * 10.0;

        let n = size * size * size;
        let temps = vec![260.0_f32; n]; // cold → snow
        let mut particles = Vec::new();

        let emitted = precipitate(&mut grid, &temps, &config, &mut particles, 1.0, 0);
        assert!(emitted > 0, "Should emit snow particles");

        for p in &particles {
            assert_eq!(
                p.material,
                MaterialId::ICE,
                "Cold precipitation should be ice"
            );
            assert!(
                p.velocity[1].abs() < 2.0,
                "Snow terminal velocity should be slow: {}",
                p.velocity[1]
            );
        }
    }

    #[test]
    fn mass_conservation_in_precipitation() {
        let size = 8;
        let mut grid = make_box_grid(size);
        let config = default_config();

        let lwc_initial = config.cloud_coalescence_threshold * 20.0;
        grid.get_mut(4, 4, 4).cloud_lwc = lwc_initial;

        let n = size * size * size;
        let temps = vec![288.15_f32; n];
        let mut particles = Vec::new();

        precipitate(&mut grid, &temps, &config, &mut particles, 1.0, 0);

        let lwc_after = grid.get(4, 4, 4).cloud_lwc;
        let particle_mass: f32 = particles.iter().map(|p| p.mass).sum();
        let lwc_removed = lwc_initial - lwc_after;

        // Mass removed from cloud should equal mass in particles
        assert!(
            (lwc_removed - particle_mass).abs() < 1e-6,
            "Mass not conserved: removed={lwc_removed}, particles={particle_mass}"
        );
    }

    #[test]
    fn virga_evaporates_particles_in_dry_air() {
        let size = 8;
        let grid = make_box_grid(size);
        let n = size * size * size;
        let temps = vec![300.0_f32; n];
        let pressures = vec![101_325.0_f32; n];

        // Interior cells have zero moisture → fully subsaturated
        let mut particles = vec![
            Particle::new(
                [4.0, 4.0, 4.0],
                [0.0, -5.0, 0.0],
                DROPLET_MASS,
                MaterialId::WATER,
            )
            .with_temperature(288.0),
        ];

        let evaporated = apply_virga(&mut particles, &grid, &temps, &pressures, 1.0);

        if evaporated > 0 {
            // Particle fully evaporated
            assert!(particles.is_empty() || particles.iter().all(|p| p.mass <= 0.0));
        } else {
            // Particle lost some mass
            assert!(
                particles[0].mass < DROPLET_MASS,
                "Particle should lose mass in dry air: {}",
                particles[0].mass
            );
        }
    }

    #[test]
    fn virga_no_evaporation_in_saturated_air() {
        let size = 8;
        let mut grid = make_box_grid(size);

        // Set moisture at saturation in the cell
        let temp = 288.15;
        let pres = 101_325.0;
        let q_sat = atmosphere::saturation_humidity(temp, pres);
        grid.get_mut(4, 4, 4).moisture = q_sat;

        let n = size * size * size;
        let temps = vec![temp; n];
        let pressures = vec![pres; n];

        let mut particles = vec![
            Particle::new(
                [4.5, 4.5, 4.5],
                [0.0, -5.0, 0.0],
                DROPLET_MASS,
                MaterialId::WATER,
            )
            .with_temperature(temp),
        ];

        apply_virga(&mut particles, &grid, &temps, &pressures, 1.0);

        // Particle mass should be unchanged in saturated air
        assert!(
            (particles[0].mass - DROPLET_MASS).abs() < 1e-8,
            "No evaporation in saturated air: mass={}",
            particles[0].mass
        );
    }

    #[test]
    fn precipitation_is_deterministic() {
        let size = 8;
        let config = default_config();
        let n = size * size * size;
        let temps = vec![288.15_f32; n];

        // Run twice with same inputs
        let mut grid1 = make_box_grid(size);
        grid1.get_mut(4, 4, 4).cloud_lwc = config.cloud_coalescence_threshold * 10.0;
        let mut particles1 = Vec::new();
        precipitate(&mut grid1, &temps, &config, &mut particles1, 1.0, 42);

        let mut grid2 = make_box_grid(size);
        grid2.get_mut(4, 4, 4).cloud_lwc = config.cloud_coalescence_threshold * 10.0;
        let mut particles2 = Vec::new();
        precipitate(&mut grid2, &temps, &config, &mut particles2, 1.0, 42);

        assert_eq!(particles1.len(), particles2.len());
        for (a, b) in particles1.iter().zip(particles2.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.velocity, b.velocity);
            assert_eq!(a.material, b.material);
        }
    }
}
