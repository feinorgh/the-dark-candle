// Particle advection + terrain collision.
//
// Moves airborne particles through the voxel grid, applying gravity, sphere drag,
// terrain collision, and bounds checking. All physics use real SI constants —
// terminal velocity emerges from the drag/gravity balance rather than being capped.

use super::types::{Particle, ParticleTag};
use crate::physics::constants::{AIR_DENSITY_SEA_LEVEL, GRAVITY};
use crate::world::voxel::{MaterialId, Voxel};

/// Drag coefficient for a smooth sphere (dimensionless).
const CD_SPHERE: f32 = 0.47;

/// Advect all airborne particles: apply gravity, drag, move, collide with terrain.
///
/// For each airborne particle:
/// 1. Apply gravity: `velocity[1] -= GRAVITY * dt`
/// 2. Apply sphere drag: `F/m = -0.5 * ρ_air * Cd * A * |v| * v / mass`
/// 3. Move: `position += velocity * dt`
/// 4. Terrain collision: if new position is in a solid voxel, mark as `Deposited`
/// 5. Bounds check: if outside `[0, size)`, mark as `Absorbed`
pub fn advect_particles(particles: &mut [Particle], voxels: &[Voxel], size: usize, dt: f32) {
    for p in particles.iter_mut() {
        if p.tag != ParticleTag::Airborne {
            continue;
        }

        // 1. Gravity (y-up)
        p.velocity[1] -= GRAVITY * dt;

        // 2. Sphere drag
        let density = material_density(p.material);
        let r = particle_radius(p.mass, density);
        let drag_acc = drag_acceleration(p.velocity, p.mass, r);
        for (i, &acc) in drag_acc.iter().enumerate() {
            p.velocity[i] += acc * dt;
        }

        // CFL warning (speed * dt > 0.5 voxels)
        let speed = (p.velocity[0] * p.velocity[0]
            + p.velocity[1] * p.velocity[1]
            + p.velocity[2] * p.velocity[2])
            .sqrt();
        if speed * dt > 0.5 {
            #[cfg(debug_assertions)]
            eprintln!(
                "CFL warning: speed={:.2} m/s, dt={:.4} s, displacement={:.2} voxels",
                speed,
                dt,
                speed * dt
            );
        }

        // 3. Move
        for i in 0..3 {
            p.position[i] += p.velocity[i] * dt;
        }

        // 4. Bounds check
        let s = size as f32;
        if p.position[0] < 0.0
            || p.position[0] >= s
            || p.position[1] < 0.0
            || p.position[1] >= s
            || p.position[2] < 0.0
            || p.position[2] >= s
        {
            p.tag = ParticleTag::Absorbed;
            continue;
        }

        // 5. Terrain collision
        let ix = p.position[0] as usize;
        let iy = p.position[1] as usize;
        let iz = p.position[2] as usize;
        let idx = iz * size * size + iy * size + ix;
        if is_solid_material(voxels[idx].material) {
            p.tag = ParticleTag::Deposited;
        }
    }
}

/// Returns `true` if the material is considered solid (not air, steam, or ash).
pub fn is_solid_material(mat: MaterialId) -> bool {
    !matches!(mat.0, 0 | 9 | 11)
}

/// Compute sphere drag acceleration (force per unit mass).
///
/// `F/m = -0.5 * ρ_air * Cd * A * |v| * v / mass`
///
/// where `Cd = 0.47` (smooth sphere), `A = π r²`.
pub fn drag_acceleration(velocity: [f32; 3], mass: f32, radius: f32) -> [f32; 3] {
    let speed_sq =
        velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2];
    if speed_sq < 1e-12 || mass < 1e-20 {
        return [0.0; 3];
    }
    let speed = speed_sq.sqrt();
    let area = std::f32::consts::PI * radius * radius;
    // -0.5 * rho * Cd * A * |v| / m
    let coeff = -0.5 * AIR_DENSITY_SEA_LEVEL * CD_SPHERE * area * speed / mass;
    [
        coeff * velocity[0],
        coeff * velocity[1],
        coeff * velocity[2],
    ]
}

/// Compute particle radius assuming a spherical shape.
///
/// `r = (3m / (4π ρ))^(1/3)`
pub fn particle_radius(mass: f32, density: f32) -> f32 {
    (3.0 * mass / (4.0 * std::f32::consts::PI * density)).cbrt()
}

/// Get default density (kg/m³) for a material.
pub fn material_density(material: MaterialId) -> f32 {
    match material.0 {
        0 => 1.225,   // Air
        1 => 2700.0,  // Stone
        2 => 1500.0,  // Dirt/Sand
        3 => 1000.0,  // Water
        8 => 917.0,   // Ice
        9 => 0.6,     // Steam
        10 => 2500.0, // Lava
        11 => 600.0,  // Ash
        _ => 1000.0,  // Default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::flip_pic::types::Particle;

    fn make_particle(pos: [f32; 3], vel: [f32; 3]) -> Particle {
        Particle::new(pos, vel, 0.001, MaterialId::WATER)
    }

    fn air_voxels(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    #[test]
    fn free_fall_acceleration() {
        let mut particles = vec![make_particle([4.0, 16.0, 4.0], [0.0, 0.0, 0.0])];
        let voxels = air_voxels(32);
        let dt = 0.01;

        advect_particles(&mut particles, &voxels, 32, dt);

        // After one step: vy ≈ -g * dt = -0.0981 m/s (drag negligible at low speed)
        let vy = particles[0].velocity[1];
        assert!(
            (vy - (-GRAVITY * dt)).abs() < 0.01,
            "vy={vy}, expected ~{}",
            -GRAVITY * dt
        );
    }

    #[test]
    fn drag_slows_fast_particle() {
        let fast_vel = [0.0, -50.0, 0.0];
        let mut particles = vec![make_particle([4.0, 16.0, 4.0], fast_vel)];
        let voxels = air_voxels(32);
        let dt = 0.001;

        advect_particles(&mut particles, &voxels, 32, dt);

        // Drag opposes motion → vy should be less negative than pure free-fall
        let vy_pure_gravity = fast_vel[1] - GRAVITY * dt;
        assert!(
            particles[0].velocity[1] > vy_pure_gravity,
            "drag should slow descent: got {}, pure gravity={}",
            particles[0].velocity[1],
            vy_pure_gravity
        );
    }

    #[test]
    fn particle_radius_water_1kg() {
        // r = (3 * 1.0 / (4 * pi * 1000))^(1/3) ≈ 0.0621 m
        let r = particle_radius(1.0, 1000.0);
        assert!(
            (r - 0.0621).abs() < 0.001,
            "water 1kg radius = {r}, expected ~0.0621 m"
        );
    }

    #[test]
    fn particle_hitting_solid_is_deposited() {
        let size = 32;
        let mut voxels = air_voxels(size);
        // Fill y=0..=4 with stone (thick solid layer)
        for y in 0..=4 {
            for z in 0..size {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    voxels[idx] = Voxel::new(MaterialId::STONE);
                }
            }
        }
        // Particle at y=5.0 moving gently downward — small dt lands it in y≈4 (stone)
        let mut particles = vec![make_particle([4.0, 5.0, 4.0], [0.0, -5.0, 0.0])];
        let dt = 0.1;

        advect_particles(&mut particles, &voxels, size, dt);

        assert_eq!(
            particles[0].tag,
            ParticleTag::Deposited,
            "particle should be deposited on solid voxel, pos={:?}",
            particles[0].position
        );
    }

    #[test]
    fn particle_outside_bounds_is_absorbed() {
        // Particle near edge moving outward
        let mut particles = vec![make_particle([31.5, 16.0, 16.0], [10.0, 0.0, 0.0])];
        let voxels = air_voxels(32);
        let dt = 0.1; // displacement ~1.0 voxels → exits [0, 32)

        advect_particles(&mut particles, &voxels, 32, dt);

        assert_eq!(
            particles[0].tag,
            ParticleTag::Absorbed,
            "particle should be absorbed when leaving bounds"
        );
    }

    #[test]
    fn terminal_velocity_emerges() {
        // Drop a 1g water droplet from rest for many steps — speed should stabilize.
        let mut particles = vec![make_particle([16.0, 28.0, 16.0], [0.0, 0.0, 0.0])];
        let voxels = air_voxels(32);
        let dt = 0.001;

        for _ in 0..10_000 {
            advect_particles(&mut particles, &voxels, 32, dt);
            if particles[0].tag != ParticleTag::Airborne {
                break;
            }
            // Reset position to keep in bounds (we only care about velocity)
            particles[0].position = [16.0, 28.0, 16.0];
        }

        let final_speed = particles[0].speed();

        // For a 1g water droplet (r ≈ 0.0062 m):
        // v_t = sqrt(2mg / (ρ Cd A)) ≈ 16.8 m/s
        assert!(
            final_speed > 10.0 && final_speed < 25.0,
            "terminal velocity should be reasonable: got {final_speed} m/s"
        );

        // Check convergence: run more steps and verify speed stabilized
        let prev_speed = final_speed;
        for _ in 0..200 {
            advect_particles(&mut particles, &voxels, 32, dt);
            particles[0].position = [16.0, 28.0, 16.0];
        }
        let after_speed = particles[0].speed();
        let change = (after_speed - prev_speed).abs();
        assert!(
            change < 0.5,
            "velocity should stabilize at terminal: changed by {change} m/s"
        );
    }

    #[test]
    fn non_airborne_not_advected() {
        let mut deposited = make_particle([4.0, 4.0, 4.0], [10.0, 10.0, 10.0]);
        deposited.tag = ParticleTag::Deposited;
        let original_pos = deposited.position;
        let original_vel = deposited.velocity;

        let mut particles = vec![deposited];
        let voxels = air_voxels(32);

        advect_particles(&mut particles, &voxels, 32, 0.01);

        assert_eq!(particles[0].position, original_pos);
        assert_eq!(particles[0].velocity, original_vel);
    }

    #[test]
    fn is_solid_material_classification() {
        assert!(!is_solid_material(MaterialId::AIR));
        assert!(!is_solid_material(MaterialId::STEAM));
        assert!(!is_solid_material(MaterialId::ASH));
        assert!(is_solid_material(MaterialId::STONE));
        assert!(is_solid_material(MaterialId::WATER));
        assert!(is_solid_material(MaterialId::LAVA));
        assert!(is_solid_material(MaterialId::ICE));
        assert!(is_solid_material(MaterialId::DIRT));
    }

    #[test]
    fn drag_acceleration_zero_velocity() {
        let a = drag_acceleration([0.0, 0.0, 0.0], 1.0, 0.01);
        assert_eq!(a, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn drag_acceleration_opposes_motion() {
        let a = drag_acceleration([10.0, 0.0, 0.0], 0.001, 0.006);
        // Drag should be in -x direction
        assert!(a[0] < 0.0, "drag should oppose +x motion: ax={}", a[0]);
        assert!((a[1]).abs() < 1e-10);
        assert!((a[2]).abs() < 1e-10);
    }

    #[test]
    fn material_density_known_values() {
        assert!((material_density(MaterialId::WATER) - 1000.0).abs() < 0.1);
        assert!((material_density(MaterialId::STONE) - 2700.0).abs() < 0.1);
        assert!((material_density(MaterialId::AIR) - 1.225).abs() < 0.01);
    }
}
