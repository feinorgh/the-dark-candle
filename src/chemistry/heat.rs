// Heat transfer via discrete diffusion across voxel neighbors.
//
// Each tick, heat flows from hotter voxels to cooler neighbors proportional to
// the temperature difference and a conductivity coefficient. This runs on
// FixedUpdate for deterministic simulation.

#![allow(dead_code)]

use crate::world::voxel::{MaterialId, Voxel};

/// Thermal conductivity coefficient per material (W/m·K simplified to a 0–1 scale).
/// In the full game this will be loaded from MaterialData; for now use a lookup.
pub fn conductivity(material: MaterialId) -> f32 {
    match material.0 {
        0 => 0.02,  // air — poor conductor
        1 => 0.8,   // stone
        2 => 0.3,   // dirt
        3 => 0.6,   // water
        4 => 1.0,   // iron — excellent conductor
        5 => 0.15,  // wood
        6 => 0.25,  // sand
        7 => 0.1,   // grass
        8 => 0.5,   // ice
        9 => 0.03,  // steam
        10 => 0.9,  // lava
        11 => 0.15, // ash
        _ => 0.1,
    }
}

/// Compute the new temperature for a voxel given its neighbors' temperatures.
///
/// Uses a simple discrete heat equation:
///   T_new = T + rate * Σ(conductivity * (T_neighbor - T)) / neighbor_count
///
/// `rate` controls simulation speed (typically dt * some constant).
/// `neighbors` is a slice of (temperature, conductivity) pairs for adjacent voxels.
pub fn diffuse_temperature(current_temp: f32, neighbors: &[(f32, f32)], rate: f32) -> f32 {
    if neighbors.is_empty() {
        return current_temp;
    }

    let mut heat_flow = 0.0;
    for &(neighbor_temp, neighbor_conductivity) in neighbors {
        let avg_conductivity = neighbor_conductivity * 0.5;
        heat_flow += avg_conductivity * (neighbor_temp - current_temp);
    }

    current_temp + rate * heat_flow / neighbors.len() as f32
}

/// Apply one heat diffusion step to a flat 3D voxel array of size `size^3`.
/// Modifies temperatures in-place using a double-buffer approach.
/// Returns the new temperature buffer.
pub fn diffuse_chunk(voxels: &[Voxel], size: usize, rate: f32) -> Vec<f32> {
    let len = size * size * size;
    assert_eq!(voxels.len(), len);

    let mut new_temps = Vec::with_capacity(len);

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = z * size * size + y * size + x;
                let voxel = &voxels[idx];
                let cond = conductivity(voxel.material);

                let mut neighbors = Vec::with_capacity(6);
                // ±X
                if x > 0 {
                    let n = &voxels[idx - 1];
                    neighbors.push((n.temperature, conductivity(n.material) + cond));
                }
                if x + 1 < size {
                    let n = &voxels[idx + 1];
                    neighbors.push((n.temperature, conductivity(n.material) + cond));
                }
                // ±Y
                if y > 0 {
                    let n = &voxels[idx - size];
                    neighbors.push((n.temperature, conductivity(n.material) + cond));
                }
                if y + 1 < size {
                    let n = &voxels[idx + size];
                    neighbors.push((n.temperature, conductivity(n.material) + cond));
                }
                // ±Z
                if z > 0 {
                    let n = &voxels[idx - size * size];
                    neighbors.push((n.temperature, conductivity(n.material) + cond));
                }
                if z + 1 < size {
                    let n = &voxels[idx + size * size];
                    neighbors.push((n.temperature, conductivity(n.material) + cond));
                }

                new_temps.push(diffuse_temperature(voxel.temperature, &neighbors, rate));
            }
        }
    }

    new_temps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_temperature_stays_stable() {
        let neighbors = vec![(293.0, 0.5), (293.0, 0.5), (293.0, 0.5)];
        let result = diffuse_temperature(293.0, &neighbors, 0.1);
        assert!((result - 293.0).abs() < f32::EPSILON);
    }

    #[test]
    fn heat_flows_from_hot_to_cold() {
        let neighbors = vec![(200.0, 0.8), (200.0, 0.8)];
        let result = diffuse_temperature(400.0, &neighbors, 0.1);
        assert!(result < 400.0, "Hot voxel should cool down");
    }

    #[test]
    fn cold_voxel_warms_from_hot_neighbors() {
        let neighbors = vec![(500.0, 0.8), (500.0, 0.8)];
        let result = diffuse_temperature(200.0, &neighbors, 0.1);
        assert!(result > 200.0, "Cold voxel should warm up");
    }

    #[test]
    fn no_neighbors_keeps_temperature() {
        let result = diffuse_temperature(350.0, &[], 0.1);
        assert_eq!(result, 350.0);
    }

    #[test]
    fn higher_rate_means_faster_transfer() {
        let neighbors = vec![(500.0, 0.8)];
        let slow = diffuse_temperature(200.0, &neighbors, 0.01);
        let fast = diffuse_temperature(200.0, &neighbors, 0.1);
        assert!(fast > slow, "Higher rate should transfer more heat");
    }

    #[test]
    fn diffuse_chunk_uniform_is_stable() {
        let size = 4;
        let voxels: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId::STONE))
            .collect();
        let new_temps = diffuse_chunk(&voxels, size, 0.1);
        for &t in &new_temps {
            assert!(
                (t - 293.0).abs() < f32::EPSILON,
                "Uniform chunk should stay at 293 K, got {t}"
            );
        }
    }

    #[test]
    fn diffuse_chunk_hot_spot_spreads() {
        let size = 4;
        let mut voxels: Vec<Voxel> = (0..size * size * size)
            .map(|_| Voxel::new(MaterialId::STONE))
            .collect();
        // Place a hot spot in the center-ish
        let center = size * size + size + 1;
        voxels[center].temperature = 1000.0;

        let new_temps = diffuse_chunk(&voxels, size, 0.1);

        // Hot spot should have cooled
        assert!(new_temps[center] < 1000.0, "Hot spot should cool down");

        // Neighbors should have warmed
        let neighbor = center + 1;
        assert!(
            new_temps[neighbor] > 293.0,
            "Neighbor of hot spot should warm up"
        );
    }

    #[test]
    fn conductivity_lookup_returns_positive() {
        for id in 0..=7 {
            let c = conductivity(MaterialId(id));
            assert!(c > 0.0, "Material {id} should have positive conductivity");
        }
    }
}
