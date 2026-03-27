// Creature spawning system driven by biome data.
//
// When a new chunk is loaded, this system determines the biome for that
// chunk's location, then uses the biome's spawn table to decide which
// creatures to spawn and where. Spawned creatures get procedurally
// generated stats via the creature generation pipeline.

#![allow(dead_code)]

use crate::procgen::biomes::{BiomeData, SpawnEntry};
use crate::procgen::creatures::SimpleRng;

/// Determine how many creatures to spawn for a given spawn entry in a chunk.
/// Uses chunk coordinates as part of the seed for deterministic per-chunk spawning.
pub fn creatures_to_spawn(entry: &SpawnEntry, chunk_x: i32, chunk_z: i32, seed: u64) -> u32 {
    let hash = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(chunk_x as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(chunk_z as u64);

    let mut rng = SimpleRng::new(hash);
    let roll = rng.next_f32();

    // Probability based on weight (normalized to 0-1 range, capped)
    let probability = (entry.weight / 10.0).min(1.0);
    if roll > probability {
        return 0;
    }

    // If we spawn, pick a count from 1..=max_per_chunk
    let count_roll = rng.next_f32();
    let count = (count_roll * entry.max_per_chunk as f32).ceil() as u32;
    count.min(entry.max_per_chunk).max(1)
}

/// Generate spawn positions within a chunk (local coordinates).
/// Returns Vec of (x, z) positions within the chunk (0..chunk_size).
pub fn spawn_positions(
    count: u32,
    chunk_size: usize,
    chunk_x: i32,
    chunk_z: i32,
    seed: u64,
) -> Vec<(f32, f32)> {
    let hash = seed
        .wrapping_add(chunk_x as u64 * 1000003)
        .wrapping_add(chunk_z as u64 * 999983);
    let mut rng = SimpleRng::new(hash);

    (0..count)
        .map(|_| {
            let x = rng.next_f32() * chunk_size as f32;
            let z = rng.next_f32() * chunk_size as f32;
            (x, z)
        })
        .collect()
}

/// Plan all creature spawns for a chunk based on its biome.
/// Returns Vec of (species_id, local_x, local_z, seed) for each spawn.
pub fn plan_chunk_spawns(
    biome: &BiomeData,
    chunk_x: i32,
    chunk_z: i32,
    chunk_size: usize,
    world_seed: u64,
) -> Vec<(String, f32, f32, u64)> {
    let mut spawns = Vec::new();

    for (entry_idx, entry) in biome.creature_spawns.iter().enumerate() {
        let entry_seed = world_seed.wrapping_add(entry_idx as u64 * 7919);
        let count = creatures_to_spawn(entry, chunk_x, chunk_z, entry_seed);

        if count > 0 {
            let positions = spawn_positions(count, chunk_size, chunk_x, chunk_z, entry_seed);
            for (i, (x, z)) in positions.into_iter().enumerate() {
                let creature_seed = entry_seed
                    .wrapping_mul(chunk_x as u64)
                    .wrapping_add(chunk_z as u64)
                    .wrapping_add(i as u64);
                spawns.push((entry.id.clone(), x, z, creature_seed));
            }
        }
    }

    spawns
}

/// Plan prop spawns for a chunk based on its biome's prop table.
/// Returns Vec of (prop_type, local_x, local_z, seed) for each spawn.
pub fn plan_chunk_prop_spawns(
    biome: &BiomeData,
    chunk_x: i32,
    chunk_z: i32,
    chunk_size: usize,
    world_seed: u64,
) -> Vec<(String, f32, f32, u64)> {
    let mut spawns = Vec::new();
    // Use a different seed offset from creature spawns to avoid correlation
    let prop_base_seed = world_seed.wrapping_add(0xDEAD_BEEF);

    for (entry_idx, entry) in biome.prop_spawns.iter().enumerate() {
        let entry_seed = prop_base_seed.wrapping_add(entry_idx as u64 * 7919);
        let count = creatures_to_spawn(entry, chunk_x, chunk_z, entry_seed);

        if count > 0 {
            let positions = spawn_positions(count, chunk_size, chunk_x, chunk_z, entry_seed);
            for (i, (x, z)) in positions.into_iter().enumerate() {
                let prop_seed = entry_seed
                    .wrapping_mul(chunk_x as u64)
                    .wrapping_add(chunk_z as u64)
                    .wrapping_add(i as u64);
                spawns.push((entry.id.clone(), x, z, prop_seed));
            }
        }
    }

    spawns
}

/// Plan tree spawns for a chunk based on its biome's tree table.
/// Returns Vec of (tree_name, local_x, local_z, seed) for each spawn.
pub fn plan_chunk_tree_spawns(
    biome: &BiomeData,
    chunk_x: i32,
    chunk_z: i32,
    chunk_size: usize,
    world_seed: u64,
) -> Vec<(String, f32, f32, u64)> {
    let mut spawns = Vec::new();
    let tree_base_seed = world_seed.wrapping_add(0xCAFE_BABE);

    for (entry_idx, entry) in biome.tree_spawns.iter().enumerate() {
        let entry_seed = tree_base_seed.wrapping_add(entry_idx as u64 * 7919);
        let count = creatures_to_spawn(entry, chunk_x, chunk_z, entry_seed);

        if count > 0 {
            let positions = spawn_positions(count, chunk_size, chunk_x, chunk_z, entry_seed);
            for (i, (x, z)) in positions.into_iter().enumerate() {
                let tree_seed = entry_seed
                    .wrapping_mul(chunk_x as u64)
                    .wrapping_add(chunk_z as u64)
                    .wrapping_add(i as u64);
                spawns.push((entry.id.clone(), x, z, tree_seed));
            }
        }
    }

    spawns
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_biome() -> BiomeData {
        BiomeData {
            name: "test".into(),
            display_name: "Test Biome".into(),
            height_range: (0.0, 100.0),
            temperature_range: (200.0, 400.0),
            moisture_range: (0.0, 1.0),
            surface_material: "Grass".into(),
            creature_spawns: vec![
                SpawnEntry {
                    id: "wolf".into(),
                    weight: 5.0,
                    max_per_chunk: 3,
                },
                SpawnEntry {
                    id: "rabbit".into(),
                    weight: 10.0,
                    max_per_chunk: 10,
                },
            ],
            item_spawns: vec![],
            prop_spawns: vec![],
            tree_spawns: vec![],
        }
    }

    #[test]
    fn creatures_to_spawn_is_deterministic() {
        let entry = SpawnEntry {
            id: "wolf".into(),
            weight: 5.0,
            max_per_chunk: 3,
        };
        let a = creatures_to_spawn(&entry, 5, 10, 42);
        let b = creatures_to_spawn(&entry, 5, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn creatures_to_spawn_respects_max() {
        let entry = SpawnEntry {
            id: "wolf".into(),
            weight: 10.0,
            max_per_chunk: 3,
        };
        for seed in 0..100 {
            let count = creatures_to_spawn(&entry, 0, 0, seed);
            assert!(count <= 3, "seed {seed}: count={count} exceeds max 3");
        }
    }

    #[test]
    fn spawn_positions_within_chunk() {
        let positions = spawn_positions(10, 32, 5, 5, 42);
        assert_eq!(positions.len(), 10);
        for (x, z) in &positions {
            assert!(*x >= 0.0 && *x < 32.0, "x={x} out of chunk");
            assert!(*z >= 0.0 && *z < 32.0, "z={z} out of chunk");
        }
    }

    #[test]
    fn plan_chunk_spawns_returns_valid_species() {
        let biome = test_biome();
        let spawns = plan_chunk_spawns(&biome, 0, 0, 32, 42);

        for (species, x, z, _seed) in &spawns {
            assert!(
                species == "wolf" || species == "rabbit",
                "Unknown species: {species}"
            );
            assert!(*x >= 0.0 && *x < 32.0, "x={x} out of range");
            assert!(*z >= 0.0 && *z < 32.0, "z={z} out of range");
        }
    }

    #[test]
    fn different_chunks_get_different_spawns() {
        let biome = test_biome();
        let spawns_a = plan_chunk_spawns(&biome, 0, 0, 32, 42);
        let spawns_b = plan_chunk_spawns(&biome, 10, 10, 32, 42);

        // They may have different counts or positions
        let count_differs = spawns_a.len() != spawns_b.len();
        let pos_differs = if !spawns_a.is_empty() && !spawns_b.is_empty() {
            (spawns_a[0].1 - spawns_b[0].1).abs() > f32::EPSILON
        } else {
            true
        };
        assert!(
            count_differs || pos_differs,
            "Different chunks should produce different spawn plans"
        );
    }

    #[test]
    fn empty_biome_spawns_nothing() {
        let biome = BiomeData {
            name: "empty".into(),
            display_name: "Empty".into(),
            height_range: (0.0, 100.0),
            temperature_range: (200.0, 400.0),
            moisture_range: (0.0, 1.0),
            surface_material: "Air".into(),
            creature_spawns: vec![],
            item_spawns: vec![],
            prop_spawns: vec![],
            tree_spawns: vec![],
        };
        let spawns = plan_chunk_spawns(&biome, 0, 0, 32, 42);
        assert!(spawns.is_empty());
    }

    #[test]
    fn spawn_positions_deterministic() {
        let a = spawn_positions(5, 32, 3, 7, 42);
        let b = spawn_positions(5, 32, 3, 7, 42);
        assert_eq!(a, b);
    }

    fn prop_biome() -> BiomeData {
        BiomeData {
            name: "prop_test".into(),
            display_name: "Prop Test".into(),
            height_range: (0.0, 100.0),
            temperature_range: (200.0, 400.0),
            moisture_range: (0.0, 1.0),
            surface_material: "Grass".into(),
            creature_spawns: vec![],
            item_spawns: vec![],
            prop_spawns: vec![
                SpawnEntry {
                    id: "rock".into(),
                    weight: 5.0,
                    max_per_chunk: 8,
                },
                SpawnEntry {
                    id: "pebble".into(),
                    weight: 8.0,
                    max_per_chunk: 20,
                },
            ],
            tree_spawns: vec![],
        }
    }

    #[test]
    fn plan_chunk_prop_spawns_returns_valid_types() {
        let biome = prop_biome();
        let spawns = plan_chunk_prop_spawns(&biome, 0, 0, 32, 42);
        for (prop_type, x, z, _seed) in &spawns {
            assert!(
                prop_type == "rock" || prop_type == "pebble",
                "Unknown prop type: {prop_type}"
            );
            assert!(*x >= 0.0 && *x < 32.0, "x={x} out of range");
            assert!(*z >= 0.0 && *z < 32.0, "z={z} out of range");
        }
    }

    #[test]
    fn plan_chunk_prop_spawns_is_deterministic() {
        let biome = prop_biome();
        let a = plan_chunk_prop_spawns(&biome, 3, 7, 32, 42);
        let b = plan_chunk_prop_spawns(&biome, 3, 7, 32, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn prop_spawns_differ_between_chunks() {
        let biome = prop_biome();
        let a = plan_chunk_prop_spawns(&biome, 0, 0, 32, 42);
        let b = plan_chunk_prop_spawns(&biome, 10, 10, 32, 42);
        let count_differs = a.len() != b.len();
        let pos_differs = if !a.is_empty() && !b.is_empty() {
            (a[0].1 - b[0].1).abs() > f32::EPSILON
        } else {
            true
        };
        assert!(
            count_differs || pos_differs,
            "Different chunks should produce different prop spawn plans"
        );
    }

    #[test]
    fn empty_prop_spawns_returns_nothing() {
        let biome = test_biome(); // has no prop_spawns
        let spawns = plan_chunk_prop_spawns(&biome, 0, 0, 32, 42);
        assert!(spawns.is_empty());
    }
}
