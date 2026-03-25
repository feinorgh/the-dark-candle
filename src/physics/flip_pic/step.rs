// Full FLIP/PIC timestep pipeline.
//
// Orchestrates: emission → P2G → pressure solve → G2P → advection → deposition.
// Supports sub-stepping (multiple FLIP steps per FixedUpdate tick).

use super::accumulation::deposit_particles;
use super::advect::advect_particles;
use super::emission::{emit_evaporation, emit_melting};
use super::g2p::gather_from_grid;
use super::grid_solve::{
    apply_pressure_gradient, compute_divergence, make_solid_mask_from_voxels, pressure_solve_jacobi,
};
use super::p2g::{normalize_grid, scatter_to_grid};
use super::types::{AccumulationGrid, Particle, ParticleTag, VelocityGrid, WeightGrid};
use crate::data::FluidConfig;
use crate::world::voxel::{MaterialId, Voxel};

/// Water boiling point (K) — voxels above this can emit steam.
const WATER_BOILING_POINT: f32 = 373.15;
/// Water triple point (K) — ice above this can melt.
const WATER_TRIPLE_POINT: f32 = 273.16;

/// Check whether any voxel in the array can emit particles (hot water or melting ice).
fn has_emission_sources(voxels: &[Voxel]) -> bool {
    voxels.iter().any(|v| {
        (v.material == MaterialId::WATER && v.temperature > WATER_BOILING_POINT)
            || (v.material == MaterialId::ICE && v.temperature > WATER_TRIPLE_POINT)
    })
}

/// Run a complete FLIP/PIC step on particles within a chunk.
///
/// Pipeline (per sub-step):
/// 1. Emission: evaporation + melting
/// 2. Build VelocityGrid via P2G
/// 3. Save grid_old
/// 4. Grid solve: divergence → pressure → gradient
/// 5. G2P: update particle velocities (FLIP/PIC blend)
/// 6. Advect: move particles, apply gravity + drag, collide
/// 7. Accumulate: deposit slow particles
/// 8. Cleanup: remove dead particles, enforce max count
pub fn flip_step(
    particles: &mut Vec<Particle>,
    voxels: &mut [Voxel],
    accum: &mut AccumulationGrid,
    config: &FluidConfig,
    dt: f32,
    tick: u64,
) -> usize {
    // Early exit when there is nothing to simulate.
    if particles.is_empty() && !has_emission_sources(voxels) {
        return 0;
    }

    // Derive chunk edge length from the flat voxel array (cube root).
    let vol = voxels.len();
    let size = (vol as f32).cbrt().round() as usize;
    debug_assert_eq!(size * size * size, vol);

    // --- 1. Emission ---
    emit_evaporation(voxels, size, particles, dt, tick);
    emit_melting(voxels, size, particles, dt, tick);

    if particles.is_empty() {
        return 0;
    }

    // --- 2. P2G: scatter particle velocities to staggered grid ---
    let mut grid = VelocityGrid::new(size);
    let mut weights = WeightGrid::new(size);
    scatter_to_grid(particles, &mut grid, &mut weights);
    normalize_grid(&mut grid, &weights);

    // --- 3. Save pre-solve grid for FLIP delta ---
    let grid_old = grid.clone();

    // --- 4. Pressure projection ---
    let solid = make_solid_mask_from_voxels(voxels, size);
    let div = compute_divergence(&grid, &solid, size);
    let pressure = pressure_solve_jacobi(&div, &solid, size, config.flip_pressure_iterations);
    apply_pressure_gradient(&mut grid, &pressure, &solid, size);

    // --- 5. G2P: transfer solved velocities back to particles ---
    gather_from_grid(particles, &grid_old, &grid, config.flip_ratio);

    // --- 6. Advect particles (gravity, drag, terrain collision) ---
    advect_particles(particles, voxels, size, dt);

    // --- 7. Accumulation: deposit slow particles onto surfaces ---
    let voxels_changed =
        deposit_particles(particles, voxels, accum, size, config.flip_deposit_velocity);

    // --- 8. Cleanup ---
    // Remove non-airborne (deposited/absorbed) particles.
    particles.retain(|p| p.tag == ParticleTag::Airborne);

    // Age surviving particles.
    for p in particles.iter_mut() {
        p.age += dt;
    }

    // Enforce per-chunk particle budget by removing the oldest particles.
    if particles.len() > config.flip_max_particles_per_chunk {
        particles.sort_unstable_by(|a, b| b.age.partial_cmp(&a.age).unwrap());
        particles.truncate(config.flip_max_particles_per_chunk);
    }

    voxels_changed
}

/// Run N sub-steps of FLIP.
pub fn flip_step_n(
    particles: &mut Vec<Particle>,
    voxels: &mut [Voxel],
    accum: &mut AccumulationGrid,
    config: &FluidConfig,
    dt: f32,
    tick: u64,
    n: usize,
) -> usize {
    let sub_dt = dt / n as f32;
    let mut total_changed = 0;
    for i in 0..n {
        total_changed += flip_step(particles, voxels, accum, config, sub_dt, tick + i as u64);
    }
    total_changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn air_voxels(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    fn test_config() -> FluidConfig {
        FluidConfig {
            flip_enabled: true,
            flip_ratio: 0.97,
            flip_deposit_velocity: 0.5,
            flip_max_particles_per_chunk: 4096,
            flip_substeps: 2,
            flip_pressure_iterations: 20,
            ..Default::default()
        }
    }

    #[test]
    fn empty_particles_no_emitters_returns_zero() {
        let config = test_config();
        let size = 4;
        let mut particles = Vec::new();
        let mut voxels = air_voxels(size);
        let mut accum = AccumulationGrid::new(size);

        let changed = flip_step(&mut particles, &mut voxels, &mut accum, &config, 0.01, 0);
        assert_eq!(changed, 0);
        assert!(particles.is_empty());
    }

    #[test]
    fn particles_survive_full_pipeline() {
        let config = test_config();
        let size = 8;
        let mut voxels = air_voxels(size);
        let mut accum = AccumulationGrid::new(size);
        let mut particles = vec![
            Particle::new([4.0, 6.0, 4.0], [0.0, -1.0, 0.0], 0.001, MaterialId::WATER),
            Particle::new([3.0, 5.0, 3.0], [1.0, 0.0, 0.5], 0.001, MaterialId::WATER),
        ];

        let changed = flip_step(&mut particles, &mut voxels, &mut accum, &config, 0.01, 0);

        // Pipeline ran without panicking; particles may still be airborne.
        let _ = changed; // may be 0 or more
        // Particles should still exist (they are in-bounds and in air).
        assert!(!particles.is_empty());
    }

    #[test]
    fn hot_water_emits_particles() {
        let mut config = test_config();
        // Disable deposition so emitted particles aren't immediately deposited.
        config.flip_deposit_velocity = 0.0;
        let size = 8;
        let mut voxels = air_voxels(size);
        // Very hot water at (2,2,2).
        let idx = 2 * size * size + 2 * size + 2;
        voxels[idx] = Voxel {
            material: MaterialId::WATER,
            temperature: 5000.0,
            ..Default::default()
        };
        let mut accum = AccumulationGrid::new(size);
        let mut particles = Vec::new();

        // dt=0.12: emission rate ≈ 8.57*0.12 ≈ 1 particle.
        flip_step(&mut particles, &mut voxels, &mut accum, &config, 0.12, 42);

        assert!(
            !particles.is_empty(),
            "Hot water should emit steam particles that survive advection"
        );
    }

    #[test]
    fn flip_step_n_runs_multiple_substeps() {
        let config = test_config();
        let size = 4;
        let mut voxels = air_voxels(size);
        let mut accum = AccumulationGrid::new(size);
        let mut particles = vec![Particle::new(
            [2.0, 2.5, 2.0],
            [0.0, 0.0, 0.0],
            0.001,
            MaterialId::WATER,
        )];

        let total = flip_step_n(&mut particles, &mut voxels, &mut accum, &config, 0.02, 0, 3);

        // Should have run 3 substeps without panicking.
        let _ = total; // may be 0 or more
    }

    #[test]
    fn max_particles_enforced() {
        let mut config = test_config();
        config.flip_max_particles_per_chunk = 5;
        let size = 4;
        let mut voxels = air_voxels(size);
        let mut accum = AccumulationGrid::new(size);

        // Insert more particles than the limit.
        let mut particles: Vec<Particle> = (0..10)
            .map(|i| {
                let mut p =
                    Particle::new([2.0, 2.0, 2.0], [0.0, 0.0, 0.0], 0.001, MaterialId::WATER);
                p.age = i as f32;
                p
            })
            .collect();

        flip_step(&mut particles, &mut voxels, &mut accum, &config, 0.001, 0);

        assert!(
            particles.len() <= 5,
            "Particle count should be capped at 5, got {}",
            particles.len()
        );
    }

    #[test]
    fn has_emission_sources_detects_hot_water() {
        let size = 4;
        let mut voxels = air_voxels(size);
        assert!(!has_emission_sources(&voxels));

        // Hot water above boiling.
        voxels[0] = Voxel {
            material: MaterialId::WATER,
            temperature: 400.0,
            ..Default::default()
        };
        assert!(has_emission_sources(&voxels));
    }

    #[test]
    fn has_emission_sources_detects_melting_ice() {
        let size = 4;
        let mut voxels = air_voxels(size);
        // Warm ice above triple point.
        voxels[0] = Voxel {
            material: MaterialId::ICE,
            temperature: 280.0,
            ..Default::default()
        };
        assert!(has_emission_sources(&voxels));
    }
}
