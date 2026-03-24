// Grid-to-particle velocity update (FLIP/PIC blend).
//
// After the pressure-projection solver updates the velocity grid, this module
// transfers the solved velocities back to Lagrangian particles using a weighted
// blend of FLIP (velocity delta) and PIC (absolute grid velocity).

use super::types::{Particle, ParticleTag, VelocityGrid};

/// Update particle velocities from the solved grid using FLIP/PIC blending.
///
/// For each airborne particle:
///   PIC_velocity = grid_new.velocity_at(particle.position)
///   FLIP_delta   = grid_new.velocity_at(pos) - grid_old.velocity_at(pos)
///   particle.velocity = flip_ratio * (particle.velocity + FLIP_delta)
///                     + (1 - flip_ratio) * PIC_velocity
///
/// `flip_ratio = 0.97` gives a good balance: FLIP preserves vortices while
/// PIC prevents noise accumulation.
pub fn gather_from_grid(
    particles: &mut [Particle],
    grid_old: &VelocityGrid,
    grid_new: &VelocityGrid,
    flip_ratio: f32,
) {
    let pic_ratio = 1.0 - flip_ratio;

    for p in particles.iter_mut() {
        if p.tag != ParticleTag::Airborne {
            continue;
        }

        let pos = p.position;
        let vel_new = grid_new.velocity_at(pos);
        let vel_old = grid_old.velocity_at(pos);

        for i in 0..3 {
            let flip_delta = vel_new[i] - vel_old[i];
            let flip_vel = p.velocity[i] + flip_delta;
            let pic_vel = vel_new[i];
            p.velocity[i] = flip_ratio * flip_vel + pic_ratio * pic_vel;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::MaterialId;

    fn make_particle(pos: [f32; 3], vel: [f32; 3]) -> Particle {
        Particle::new(pos, vel, 0.001, MaterialId::WATER)
    }

    fn uniform_grid(size: usize, ux: f32, vy: f32, wz: f32) -> VelocityGrid {
        let mut g = VelocityGrid::new(size);
        let s = size;
        for k in 0..s {
            for j in 0..s {
                for i in 0..=s {
                    g.set_u(i, j, k, ux);
                }
            }
        }
        for k in 0..s {
            for j in 0..=s {
                for i in 0..s {
                    g.set_v(i, j, k, vy);
                }
            }
        }
        for k in 0..=s {
            for j in 0..s {
                for i in 0..s {
                    g.set_w(i, j, k, wz);
                }
            }
        }
        g
    }

    #[test]
    fn stationary_particle_acquires_uniform_flow() {
        let mut particles = vec![make_particle([4.0, 4.0, 4.0], [0.0, 0.0, 0.0])];
        let grid_old = VelocityGrid::new(8);
        let grid_new = uniform_grid(8, 5.0, 0.0, 0.0);

        gather_from_grid(&mut particles, &grid_old, &grid_new, 0.97);

        // FLIP component: 0.97 * (0 + 5) = 4.85
        // PIC component:  0.03 * 5       = 0.15
        // Total: 5.0
        assert!(
            (particles[0].velocity[0] - 5.0).abs() < 0.01,
            "got {}",
            particles[0].velocity[0]
        );
    }

    #[test]
    fn pure_pic_gives_grid_velocity() {
        let mut particles = vec![make_particle([4.0, 4.0, 4.0], [10.0, -3.0, 7.0])];
        let grid_old = uniform_grid(8, 1.0, 1.0, 1.0);
        let grid_new = uniform_grid(8, 2.0, 3.0, 4.0);

        gather_from_grid(&mut particles, &grid_old, &grid_new, 0.0);

        assert!((particles[0].velocity[0] - 2.0).abs() < 0.01);
        assert!((particles[0].velocity[1] - 3.0).abs() < 0.01);
        assert!((particles[0].velocity[2] - 4.0).abs() < 0.01);
    }

    #[test]
    fn pure_flip_keeps_old_velocity_plus_delta() {
        let initial_vel = [10.0, -3.0, 7.0];
        let mut particles = vec![make_particle([4.0, 4.0, 4.0], initial_vel)];
        let grid_old = uniform_grid(8, 1.0, 2.0, 3.0);
        let grid_new = uniform_grid(8, 1.5, 2.5, 3.5);

        gather_from_grid(&mut particles, &grid_old, &grid_new, 1.0);

        // FLIP: old_vel + (new_grid - old_grid)
        assert!((particles[0].velocity[0] - (10.0 + 0.5)).abs() < 0.01);
        assert!((particles[0].velocity[1] - (-3.0 + 0.5)).abs() < 0.01);
        assert!((particles[0].velocity[2] - (7.0 + 0.5)).abs() < 0.01);
    }

    #[test]
    fn non_airborne_particles_unchanged() {
        let mut deposited = make_particle([4.0, 4.0, 4.0], [1.0, 2.0, 3.0]);
        deposited.tag = ParticleTag::Deposited;
        let mut absorbed = make_particle([4.0, 4.0, 4.0], [4.0, 5.0, 6.0]);
        absorbed.tag = ParticleTag::Absorbed;

        let mut particles = vec![deposited, absorbed];
        let grid_old = VelocityGrid::new(8);
        let grid_new = uniform_grid(8, 99.0, 99.0, 99.0);

        gather_from_grid(&mut particles, &grid_old, &grid_new, 0.97);

        assert_eq!(particles[0].velocity, [1.0, 2.0, 3.0]);
        assert_eq!(particles[1].velocity, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn default_flip_ratio_blends_correctly() {
        let mut particles = vec![make_particle([4.0, 4.0, 4.0], [0.0, 0.0, 0.0])];
        let grid_old = uniform_grid(8, 0.0, 0.0, 0.0);
        let grid_new = uniform_grid(8, 10.0, 0.0, 0.0);

        gather_from_grid(&mut particles, &grid_old, &grid_new, 0.97);

        // Both FLIP and PIC give 10.0 when particle starts at 0
        assert!((particles[0].velocity[0] - 10.0).abs() < 0.1);
    }
}
