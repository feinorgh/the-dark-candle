// Particle-to-grid velocity transfer using trilinear weights.

use super::types::{Particle, VelocityGrid, WeightGrid};

/// Scatter particle velocities to the MAC velocity grid using trilinear weights.
///
/// For each airborne particle, distribute velocity × mass to the 8 surrounding
/// face nodes. The WeightGrid accumulates total mass per node for normalization.
pub fn scatter_to_grid(particles: &[Particle], grid: &mut VelocityGrid, weights: &mut WeightGrid) {
    let s = grid.size;

    for p in particles {
        if !p.is_airborne() {
            continue;
        }

        let [px, py, pz] = p.position;

        if px < 0.0 || px >= s as f32 || py < 0.0 || py >= s as f32 || pz < 0.0 || pz >= s as f32 {
            continue;
        }

        let mass = p.mass;

        // U-faces at (i, y+0.5, z+0.5): shift by (0, -0.5, -0.5)
        scatter_face(
            px,
            py - 0.5,
            pz - 0.5,
            p.velocity[0],
            mass,
            &mut grid.u,
            &mut weights.u,
            s + 1,
            s,
            s,
            |a, b, c| c * s * (s + 1) + b * (s + 1) + a,
        );

        // V-faces at (x+0.5, j, z+0.5): shift by (-0.5, 0, -0.5)
        scatter_face(
            px - 0.5,
            py,
            pz - 0.5,
            p.velocity[1],
            mass,
            &mut grid.v,
            &mut weights.v,
            s,
            s + 1,
            s,
            |a, b, c| c * (s + 1) * s + b * s + a,
        );

        // W-faces at (x+0.5, y+0.5, k): shift by (-0.5, -0.5, 0)
        scatter_face(
            px - 0.5,
            py - 0.5,
            pz,
            p.velocity[2],
            mass,
            &mut grid.w,
            &mut weights.w,
            s,
            s,
            s + 1,
            |a, b, c| c * s * s + b * s + a,
        );
    }
}

/// Scatter a single velocity component to one face grid using trilinear weights.
#[allow(clippy::too_many_arguments)]
fn scatter_face(
    sx: f32,
    sy: f32,
    sz: f32,
    vel: f32,
    mass: f32,
    data: &mut [f32],
    wt: &mut [f32],
    max_a: usize,
    max_b: usize,
    max_c: usize,
    index_fn: impl Fn(usize, usize, usize) -> usize,
) {
    let ax = sx.floor();
    let ay = sy.floor();
    let az = sz.floor();

    let a0 = ax as isize;
    let b0 = ay as isize;
    let c0 = az as isize;

    let fx = sx - ax;
    let fy = sy - ay;
    let fz = sz - az;

    const OFFSETS: [(isize, isize, isize); 8] = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ];

    let tri = [
        (1.0 - fx) * (1.0 - fy) * (1.0 - fz),
        fx * (1.0 - fy) * (1.0 - fz),
        (1.0 - fx) * fy * (1.0 - fz),
        fx * fy * (1.0 - fz),
        (1.0 - fx) * (1.0 - fy) * fz,
        fx * (1.0 - fy) * fz,
        (1.0 - fx) * fy * fz,
        fx * fy * fz,
    ];

    for (k, &(da, db, dc)) in OFFSETS.iter().enumerate() {
        let a = a0 + da;
        let b = b0 + db;
        let c = c0 + dc;

        if a >= 0
            && (a as usize) < max_a
            && b >= 0
            && (b as usize) < max_b
            && c >= 0
            && (c as usize) < max_c
        {
            let idx = index_fn(a as usize, b as usize, c as usize);
            let w = tri[k];
            data[idx] += vel * mass * w;
            wt[idx] += mass * w;
        }
    }
}

/// Normalize grid velocities by accumulated weights.
/// Where weight > 0, divide velocity by weight. Where weight == 0, leave as 0.
pub fn normalize_grid(grid: &mut VelocityGrid, weights: &WeightGrid) {
    normalize_component(&mut grid.u, &weights.u);
    normalize_component(&mut grid.v, &weights.v);
    normalize_component(&mut grid.w, &weights.w);
}

fn normalize_component(vel: &mut [f32], wt: &[f32]) {
    for (v, &w) in vel.iter_mut().zip(wt.iter()) {
        if w > 0.0 {
            *v /= w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_grid, scatter_to_grid};
    use crate::physics::flip_pic::types::{Particle, ParticleTag, VelocityGrid, WeightGrid};
    use crate::world::voxel::MaterialId;

    #[test]
    fn cell_center_distributes_equally_to_8_u_neighbors() {
        let s = 8;
        let mut grid = VelocityGrid::new(s);
        let mut weights = WeightGrid::new(s);

        // U-shift (0, -0.5, -0.5) → shifted pos (2.5, 2.5, 2.5) → fx=fy=fz=0.5
        let p = Particle::new([2.5, 3.0, 3.0], [10.0, 0.0, 0.0], 1.0, MaterialId::WATER);
        scatter_to_grid(&[p], &mut grid, &mut weights);

        let expected_w = 0.125;
        let expected_v = 10.0 * 0.125;

        for di in 0..2_usize {
            for dy in 0..2_usize {
                for dz in 0..2_usize {
                    let idx = grid.u_index(2 + di, 2 + dy, 2 + dz);
                    assert!(
                        (weights.u[idx] - expected_w).abs() < 1e-6,
                        "weight at ({},{},{}) = {}, expected {expected_w}",
                        2 + di,
                        2 + dy,
                        2 + dz,
                        weights.u[idx],
                    );
                    assert!(
                        (grid.u[idx] - expected_v).abs() < 1e-6,
                        "momentum at ({},{},{}) = {}, expected {expected_v}",
                        2 + di,
                        2 + dy,
                        2 + dz,
                        grid.u[idx],
                    );
                }
            }
        }
    }

    #[test]
    fn exact_face_node_gets_full_weight() {
        let s = 8;
        let mut grid = VelocityGrid::new(s);
        let mut weights = WeightGrid::new(s);

        // U-face (3,2,2) lives at world position (3.0, 2.5, 2.5)
        let p = Particle::new([3.0, 2.5, 2.5], [5.0, 0.0, 0.0], 2.0, MaterialId::WATER);
        scatter_to_grid(&[p], &mut grid, &mut weights);

        let target = grid.u_index(3, 2, 2);
        assert!((weights.u[target] - 2.0).abs() < 1e-6);
        assert!((grid.u[target] - 10.0).abs() < 1e-6);

        // Total u-weight equals particle mass (all goes to one node)
        let total_w: f32 = weights.u.iter().sum();
        assert!((total_w - 2.0).abs() < 1e-5);
    }

    #[test]
    fn momentum_conservation() {
        let s = 8;
        let mut grid = VelocityGrid::new(s);
        let mut weights = WeightGrid::new(s);

        let vel = [3.0, -2.0, 7.0];
        let mass = 0.5;
        // Well inside grid so all 8 nodes are in bounds for every component
        let p = Particle::new([4.3, 4.7, 4.1], vel, mass, MaterialId::LAVA);
        scatter_to_grid(&[p], &mut grid, &mut weights);

        let total_u: f32 = grid.u.iter().sum();
        let total_v: f32 = grid.v.iter().sum();
        let total_w: f32 = grid.w.iter().sum();

        assert!(
            (total_u - vel[0] * mass).abs() < 1e-5,
            "u momentum: expected {}, got {total_u}",
            vel[0] * mass,
        );
        assert!(
            (total_v - vel[1] * mass).abs() < 1e-5,
            "v momentum: expected {}, got {total_v}",
            vel[1] * mass,
        );
        assert!(
            (total_w - vel[2] * mass).abs() < 1e-5,
            "w momentum: expected {}, got {total_w}",
            vel[2] * mass,
        );
    }

    #[test]
    fn multiple_particles_accumulate() {
        let s = 8;
        let mut grid = VelocityGrid::new(s);
        let mut weights = WeightGrid::new(s);

        let p1 = Particle::new([3.0, 2.5, 2.5], [4.0, 0.0, 0.0], 1.0, MaterialId::WATER);
        let p2 = Particle::new([3.0, 2.5, 2.5], [6.0, 0.0, 0.0], 1.0, MaterialId::WATER);
        scatter_to_grid(&[p1, p2], &mut grid, &mut weights);

        let target = grid.u_index(3, 2, 2);
        assert!((grid.u[target] - 10.0).abs() < 1e-6);
        assert!((weights.u[target] - 2.0).abs() < 1e-6);

        // After normalization: (4*1 + 6*1) / (1+1) = 5.0
        normalize_grid(&mut grid, &weights);
        assert!((grid.u[target] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn deposited_particle_is_skipped() {
        let s = 8;
        let mut grid = VelocityGrid::new(s);
        let mut weights = WeightGrid::new(s);

        let mut p = Particle::new([4.0, 4.5, 4.5], [10.0, 0.0, 0.0], 1.0, MaterialId::WATER);
        p.tag = ParticleTag::Deposited;
        scatter_to_grid(&[p], &mut grid, &mut weights);

        assert!(grid.u.iter().all(|&v| v.abs() < 1e-10));
    }
}
