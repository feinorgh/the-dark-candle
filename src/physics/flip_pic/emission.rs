// Particle spawning from phase transitions.
//
// Pure functions for emitting particles from evaporation, melting, rain,
// and splash events. All functions are deterministic (no rand crate).

use super::types::Particle;
use crate::world::chunk::ChunkCoord;
use crate::world::voxel::{MaterialId, Voxel};

// Physical constants for water.
const WATER_BOILING_POINT: f32 = 373.15; // K
const WATER_CP: f32 = 4186.0; // J/(kg·K)
const WATER_LV: f32 = 2_260_000.0; // J/kg (latent heat of vaporization)
const WATER_LF: f32 = 334_000.0; // J/kg (latent heat of fusion)
const WATER_TRIPLE_POINT: f32 = 273.16; // K

// Lava: extremely high boiling point, rarely evaporates.
const LAVA_BOILING_POINT: f32 = 2500.0; // K
const LAVA_CP: f32 = 1000.0; // J/(kg·K) (approximate basalt)
const LAVA_LV: f32 = 5_000_000.0; // J/kg (approximate)

/// Default droplet mass (kg) — 1 gram.
const DROPLET_MASS: f32 = 0.001;

/// Deterministic hash-based jitter for reproducible particle positions.
///
/// Uses FNV-1a inspired hash for speed. Returns 3 values in \[-0.5, 0.5\].
fn deterministic_jitter(seed: u64, index: usize) -> [f32; 3] {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    // Mix seed.
    for byte in seed.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    // Mix index.
    for byte in (index as u64).to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }

    let x_bits = (h & 0xFFFF) as f32 / 65535.0 - 0.5;
    let y_bits = ((h >> 16) & 0xFFFF) as f32 / 65535.0 - 0.5;
    let z_bits = ((h >> 32) & 0xFFFF) as f32 / 65535.0 - 0.5;
    [x_bits, y_bits, z_bits]
}

/// Hash a voxel position and tick into a deterministic seed.
fn position_seed(x: usize, y: usize, z: usize, tick: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for byte in (x as u64).to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    for byte in (y as u64).to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    for byte in (z as u64).to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    for byte in tick.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Flat voxel index: z * size² + y * size + x.
#[inline]
fn voxel_index(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Emit steam particles from liquid surfaces above boiling point.
///
/// Scans voxels for water/lava above their boiling point with air above.
/// Emission rate is proportional to `(T - T_boil) × Cp / L_v` (simplified).
/// Each emitted particle carries `DROPLET_MASS` kg of steam.
pub fn emit_evaporation(
    voxels: &[Voxel],
    size: usize,
    particles: &mut Vec<Particle>,
    dt: f32,
    tick: u64,
) -> usize {
    let mut emitted = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = voxel_index(x, y, z, size);
                let voxel = &voxels[idx];

                let (boiling_point, cp, lv) = match voxel.material.0 {
                    3 => (WATER_BOILING_POINT, WATER_CP, WATER_LV),
                    10 => (LAVA_BOILING_POINT, LAVA_CP, LAVA_LV),
                    _ => continue,
                };

                if voxel.temperature <= boiling_point {
                    continue;
                }

                // Must have air above to evaporate.
                if y + 1 < size {
                    let above = voxel_index(x, y + 1, z, size);
                    if !voxels[above].material.is_air() {
                        continue;
                    }
                }

                // Emission rate: particles per second per voxel.
                // rate = (T - T_boil) × Cp / L_v  (dimensionless: energy ratio).
                let excess = voxel.temperature - boiling_point;
                let rate = excess * cp / lv;
                let count = (rate * dt).max(0.0);

                // Emit integer particles; fractional part is lost (acceptable at
                // small dt — accumulates over ticks).
                let n = count as usize;
                for i in 0..n {
                    let seed = position_seed(x, y, z, tick);
                    let jitter = deterministic_jitter(seed, i);

                    let pos = [
                        x as f32 + 0.5 + jitter[0] * 0.4,
                        y as f32 + 1.0, // above the liquid surface
                        z as f32 + 0.5 + jitter[2] * 0.4,
                    ];
                    // Upward velocity with slight horizontal jitter.
                    let vel = [jitter[0] * 0.5, 2.0 + jitter[1].abs(), jitter[2] * 0.5];

                    let steam_temp = voxel.temperature;
                    let p = Particle::new(pos, vel, DROPLET_MASS, MaterialId::STEAM)
                        .with_temperature(steam_temp);
                    particles.push(p);
                    emitted += 1;
                }
            }
        }
    }

    emitted
}

/// Emit water particles from melting ice/snow voxels.
///
/// When ice/snow temperature > 273.16 K (triple point of water),
/// emit water particles. Rate proportional to `(T - T_triple) × Cp_ice / L_f`.
pub fn emit_melting(
    voxels: &[Voxel],
    size: usize,
    particles: &mut Vec<Particle>,
    dt: f32,
    tick: u64,
) -> usize {
    const ICE_CP: f32 = 2090.0; // J/(kg·K)

    let mut emitted = 0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = voxel_index(x, y, z, size);
                let voxel = &voxels[idx];

                if voxel.material.0 != 8 {
                    // Not ice.
                    continue;
                }

                if voxel.temperature <= WATER_TRIPLE_POINT {
                    continue;
                }

                let excess = voxel.temperature - WATER_TRIPLE_POINT;
                let rate = excess * ICE_CP / WATER_LF;
                let n = (rate * dt) as usize;

                for i in 0..n {
                    let seed = position_seed(x, y, z, tick);
                    let jitter = deterministic_jitter(seed, i);

                    let pos = [
                        x as f32 + 0.5 + jitter[0] * 0.3,
                        y as f32 + jitter[1].abs() * 0.2,
                        z as f32 + 0.5 + jitter[2] * 0.3,
                    ];
                    // Melt-water drips downward.
                    let vel = [jitter[0] * 0.1, -0.5, jitter[2] * 0.1];

                    let p = Particle::new(pos, vel, DROPLET_MASS, MaterialId::WATER)
                        .with_temperature(voxel.temperature);
                    particles.push(p);
                    emitted += 1;
                }
            }
        }
    }

    emitted
}

/// Emit rain droplets from the top of a chunk.
///
/// Rate is particles per second across the top face. Positions are
/// deterministic based on chunk coordinate and tick counter.
/// Droplets: mass = 0.001 kg, temperature = 283 K, downward velocity ~5 m/s.
pub fn emit_rain(
    chunk_coord: ChunkCoord,
    size: usize,
    particles: &mut Vec<Particle>,
    rate: f32,
    dt: f32,
    tick: u64,
) -> usize {
    let count = (rate * dt) as usize;
    let mut emitted = 0;

    for i in 0..count {
        // Deterministic position within chunk top face.
        let seed = {
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for byte in (chunk_coord.x as u64).to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            for byte in (chunk_coord.y as u64).to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            for byte in (chunk_coord.z as u64).to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            for byte in tick.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            h
        };

        let jitter = deterministic_jitter(seed, i);

        // Map jitter from [-0.5, 0.5] to [0, size].
        let px = (jitter[0] + 0.5) * size as f32;
        let pz = (jitter[2] + 0.5) * size as f32;

        let pos = [
            px.clamp(0.0, size as f32 - 0.01),
            size as f32 - 0.1, // near top
            pz.clamp(0.0, size as f32 - 0.01),
        ];

        // Downward with slight horizontal variation.
        let vel = [jitter[0] * 0.5, -5.0, jitter[2] * 0.5];

        let p = Particle::new(pos, vel, DROPLET_MASS, MaterialId::WATER).with_temperature(283.0);
        particles.push(p);
        emitted += 1;
    }

    emitted
}

/// Emit a burst of splash particles from an impact point.
///
/// Distributes `count` particles in a hemisphere above the impact point.
/// Velocities are reflected from the impact velocity with deterministic spread.
pub fn emit_splash(
    position: [f32; 3],
    impact_velocity: [f32; 3],
    material: MaterialId,
    count: usize,
    particles: &mut Vec<Particle>,
    tick: u64,
) -> usize {
    let impact_speed = (impact_velocity[0] * impact_velocity[0]
        + impact_velocity[1] * impact_velocity[1]
        + impact_velocity[2] * impact_velocity[2])
        .sqrt();

    // Splash speed is a fraction of impact speed.
    let splash_speed = impact_speed * 0.4;
    let mut emitted = 0;

    for i in 0..count {
        let seed = {
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for byte in position[0].to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            for byte in position[1].to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            for byte in position[2].to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            for byte in tick.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            h
        };

        let jitter = deterministic_jitter(seed, i);

        // Distribute in hemisphere above impact point.
        // Use jitter to create spherical-ish distribution with y >= 0.
        let dx = jitter[0] * 2.0; // [-1, 1]
        let dz = jitter[2] * 2.0; // [-1, 1]
        let dy = jitter[1].abs() + 0.3; // [0.3, 0.8] — always upward

        // Normalize direction.
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let vx = dx / len * splash_speed;
        let vy = dy / len * splash_speed;
        let vz = dz / len * splash_speed;

        let mass = DROPLET_MASS * 0.5; // splash droplets are smaller
        let p = Particle::new(position, [vx, vy, vz], mass, material);
        particles.push(p);
        emitted += 1;
    }

    emitted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn air_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    fn set_voxel(
        voxels: &mut [Voxel],
        x: usize,
        y: usize,
        z: usize,
        size: usize,
        mat: MaterialId,
        temp: f32,
    ) {
        let idx = voxel_index(x, y, z, size);
        voxels[idx].material = mat;
        voxels[idx].temperature = temp;
    }

    #[test]
    fn water_above_boiling_emits_steam() {
        let size = 4;
        let mut voxels = air_grid(size);
        // Water at 1000 K (well above 373.15 K boiling point), air above.
        // rate = (1000 - 373.15) * 4186 / 2_260_000 ≈ 1.16
        set_voxel(&mut voxels, 1, 1, 1, size, MaterialId::WATER, 1000.0);

        let mut particles = Vec::new();
        let emitted = emit_evaporation(&voxels, size, &mut particles, 1.0, 42);

        assert!(emitted > 0, "Should emit steam from boiling water");
        assert!(particles.iter().all(|p| p.material == MaterialId::STEAM));
        // Steam should move upward.
        assert!(
            particles.iter().all(|p| p.velocity[1] > 0.0),
            "Steam should have upward velocity"
        );
    }

    #[test]
    fn water_below_boiling_emits_nothing() {
        let size = 4;
        let mut voxels = air_grid(size);
        set_voxel(&mut voxels, 1, 1, 1, size, MaterialId::WATER, 350.0);

        let mut particles = Vec::new();
        let emitted = emit_evaporation(&voxels, size, &mut particles, 1.0, 42);

        assert_eq!(emitted, 0);
        assert!(particles.is_empty());
    }

    #[test]
    fn ice_above_triple_point_emits_water() {
        let size = 4;
        let mut voxels = air_grid(size);
        // Ice at 350 K (well above 273.16 K triple point).
        // rate = (350 - 273.16) * 2090 / 334_000 ≈ 0.481, dt=10 → ~4 particles.
        set_voxel(&mut voxels, 2, 2, 2, size, MaterialId::ICE, 350.0);

        let mut particles = Vec::new();
        let emitted = emit_melting(&voxels, size, &mut particles, 10.0, 10);

        assert!(emitted > 0, "Should emit water from melting ice");
        assert!(particles.iter().all(|p| p.material == MaterialId::WATER));
    }

    #[test]
    fn ice_below_triple_point_emits_nothing() {
        let size = 4;
        let mut voxels = air_grid(size);
        set_voxel(&mut voxels, 2, 2, 2, size, MaterialId::ICE, 260.0);

        let mut particles = Vec::new();
        let emitted = emit_melting(&voxels, size, &mut particles, 1.0, 10);

        assert_eq!(emitted, 0);
    }

    #[test]
    fn rain_produces_deterministic_positions() {
        let coord = ChunkCoord::new(1, 2, 3);
        let size = 32;

        let mut p1 = Vec::new();
        emit_rain(coord, size, &mut p1, 100.0, 1.0, 42);

        let mut p2 = Vec::new();
        emit_rain(coord, size, &mut p2, 100.0, 1.0, 42);

        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.velocity, b.velocity);
        }
    }

    #[test]
    fn rain_droplets_move_downward() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut particles = Vec::new();
        emit_rain(coord, 32, &mut particles, 50.0, 1.0, 1);

        assert!(!particles.is_empty());
        for p in &particles {
            assert!(p.velocity[1] < 0.0, "Rain should fall downward");
            assert!(
                (p.temperature - 283.0).abs() < 0.01,
                "Rain temperature should be 283 K"
            );
        }
    }

    #[test]
    fn splash_produces_upward_hemisphere() {
        let pos = [5.0, 3.0, 5.0];
        let impact_vel = [0.0, -10.0, 0.0];
        let mut particles = Vec::new();

        let emitted = emit_splash(pos, impact_vel, MaterialId::WATER, 20, &mut particles, 99);

        assert_eq!(emitted, 20);
        assert_eq!(particles.len(), 20);

        // All splash particles should have upward velocity (hemisphere).
        for p in &particles {
            assert!(
                p.velocity[1] > 0.0,
                "Splash should go upward, got vy={}",
                p.velocity[1]
            );
        }
    }

    #[test]
    fn deterministic_jitter_is_repeatable() {
        let a = deterministic_jitter(12345, 7);
        let b = deterministic_jitter(12345, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn deterministic_jitter_varies_with_input() {
        let a = deterministic_jitter(100, 0);
        let b = deterministic_jitter(200, 0);
        let c = deterministic_jitter(100, 1);
        // Different seeds/indices should produce different jitter.
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn deterministic_jitter_in_range() {
        for seed in 0..100 {
            for idx in 0..10 {
                let j = deterministic_jitter(seed, idx);
                for &v in &j {
                    assert!(
                        (-0.5..=0.5).contains(&v),
                        "Jitter {v} out of range for seed={seed}, idx={idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn evaporation_rate_scales_with_temperature() {
        let size = 4;
        let mut voxels_hot = air_grid(size);
        let mut voxels_warm = air_grid(size);

        // Hot: 1500 K, Warm: 500 K.
        set_voxel(&mut voxels_hot, 1, 1, 1, size, MaterialId::WATER, 1500.0);
        set_voxel(&mut voxels_warm, 1, 1, 1, size, MaterialId::WATER, 500.0);

        let mut p_hot = Vec::new();
        let mut p_warm = Vec::new();

        emit_evaporation(&voxels_hot, size, &mut p_hot, 1.0, 1);
        emit_evaporation(&voxels_warm, size, &mut p_warm, 1.0, 1);

        assert!(
            p_hot.len() >= p_warm.len(),
            "Hotter water should emit more: hot={}, warm={}",
            p_hot.len(),
            p_warm.len()
        );
    }
}
