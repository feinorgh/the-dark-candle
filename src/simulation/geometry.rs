// Geometry region builders for simulation scenarios.
//
// Each `Region` variant describes how to fill part of a voxel grid with a
// specific material. Regions are applied in order — later regions overwrite
// earlier ones, enabling layered compositions (e.g. fill a box, then hollow
// it, then place specific materials inside).

use serde::Deserialize;

use crate::data::MaterialRegistry;
use crate::world::voxel::{MaterialId, Voxel};

/// A region description that can populate part of a voxel grid.
///
/// Variants are designed to cover common spatial patterns. Regions are applied
/// in declaration order; later regions overwrite voxels set by earlier ones.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub enum Region {
    /// Fill a rectangular volume with a material.
    Fill {
        material: String,
        min: (usize, usize, usize),
        max: (usize, usize, usize),
    },

    /// Hollow box: fill the outer shell of a rectangular volume.
    Shell {
        material: String,
        min: (usize, usize, usize),
        max: (usize, usize, usize),
        thickness: usize,
    },

    /// Place a single voxel.
    Single {
        material: String,
        pos: (usize, usize, usize),
    },

    /// Place a material at every Nth position within a region.
    /// Useful for mixing gases (e.g. oxygen every 3rd voxel in hydrogen).
    EveryNth {
        material: String,
        min: (usize, usize, usize),
        max: (usize, usize, usize),
        step: usize,
    },

    /// Alternating checkerboard pattern (material placed where (x+y+z) is even).
    Checkerboard {
        material: String,
        min: (usize, usize, usize),
        max: (usize, usize, usize),
    },

    /// Spherical region (all voxels within `radius` of `center`).
    Sphere {
        material: String,
        center: (usize, usize, usize),
        radius: f32,
    },

    /// Fill an entire horizontal layer at a specific Y level.
    Layer { material: String, y: usize },

    /// Randomized heightmap terrain: fills columns from `y_min` up to a
    /// per-column random height in `[y_min, y_max]`.
    ///
    /// Uses a deterministic hash seeded by `seed` so the same scenario always
    /// produces the same terrain.
    ///
    /// TODO: When native octree physics with LOD-based dynamic resolution is
    /// implemented, this region type should optionally generate octree-native
    /// geometry at multiple resolutions rather than a flat voxel grid.
    RandomHeightmap {
        material: String,
        x_range: (usize, usize),
        z_range: (usize, usize),
        y_min: usize,
        y_max: usize,
        seed: u64,
    },
}

/// 3D index into a flat `size³` array.
fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    z * size * size + y * size + x
}

/// Apply a sequence of regions to a flat `size³` voxel array.
///
/// Regions are applied in order. Each region resolves its material name
/// through the `MaterialRegistry` and sets the corresponding voxels.
///
/// Returns an error if a material name cannot be resolved.
pub fn apply_regions(
    voxels: &mut [Voxel],
    size: usize,
    regions: &[Region],
    registry: &MaterialRegistry,
) -> Result<(), String> {
    for region in regions {
        apply_one(voxels, size, region, registry)?;
    }
    Ok(())
}

fn resolve(name: &str, registry: &MaterialRegistry) -> Result<MaterialId, String> {
    registry
        .resolve_name(name)
        .ok_or_else(|| format!("unknown material: {name:?}"))
}

/// Compute a deterministic per-column height in `[y_min, y_max]` using a
/// simple hash of `(x, z, seed)`. No external crate required.
fn deterministic_height(x: usize, z: usize, y_min: usize, y_max: usize, seed: u64) -> usize {
    // splitmix64-style hash
    let mut h = seed
        .wrapping_add(x as u64)
        .wrapping_mul(6_364_136_223_846_793_005);
    h = h
        .wrapping_add(z as u64)
        .wrapping_mul(6_364_136_223_846_793_005);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    let range = (y_max - y_min + 1) as u64;
    if range == 0 {
        return y_min;
    }
    y_min + (h % range) as usize
}

fn apply_one(
    voxels: &mut [Voxel],
    size: usize,
    region: &Region,
    registry: &MaterialRegistry,
) -> Result<(), String> {
    match region {
        Region::Fill { material, min, max } => {
            let mat = resolve(material, registry)?;
            for z in min.2..=max.2.min(size - 1) {
                for y in min.1..=max.1.min(size - 1) {
                    for x in min.0..=max.0.min(size - 1) {
                        voxels[idx(x, y, z, size)].material = mat;
                    }
                }
            }
        }

        Region::Shell {
            material,
            min,
            max,
            thickness,
        } => {
            let mat = resolve(material, registry)?;
            let t = *thickness;
            for z in min.2..=max.2.min(size - 1) {
                for y in min.1..=max.1.min(size - 1) {
                    for x in min.0..=max.0.min(size - 1) {
                        let on_shell = x < min.0 + t
                            || x + t > max.0
                            || y < min.1 + t
                            || y + t > max.1
                            || z < min.2 + t
                            || z + t > max.2;
                        if on_shell {
                            voxels[idx(x, y, z, size)].material = mat;
                        }
                    }
                }
            }
        }

        Region::Single { material, pos } => {
            let mat = resolve(material, registry)?;
            if pos.0 < size && pos.1 < size && pos.2 < size {
                voxels[idx(pos.0, pos.1, pos.2, size)].material = mat;
            }
        }

        Region::EveryNth {
            material,
            min,
            max,
            step,
        } => {
            let mat = resolve(material, registry)?;
            let s = (*step).max(1);
            let mut counter = 0usize;
            for z in min.2..=max.2.min(size - 1) {
                for y in min.1..=max.1.min(size - 1) {
                    for x in min.0..=max.0.min(size - 1) {
                        if counter.is_multiple_of(s) {
                            voxels[idx(x, y, z, size)].material = mat;
                        }
                        counter += 1;
                    }
                }
            }
        }

        Region::Checkerboard { material, min, max } => {
            let mat = resolve(material, registry)?;
            for z in min.2..=max.2.min(size - 1) {
                for y in min.1..=max.1.min(size - 1) {
                    for x in min.0..=max.0.min(size - 1) {
                        if (x + y + z) % 2 == 0 {
                            voxels[idx(x, y, z, size)].material = mat;
                        }
                    }
                }
            }
        }

        Region::Sphere {
            material,
            center,
            radius,
        } => {
            let mat = resolve(material, registry)?;
            let r2 = radius * radius;
            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        let dx = x as f32 - center.0 as f32;
                        let dy = y as f32 - center.1 as f32;
                        let dz = z as f32 - center.2 as f32;
                        if dx * dx + dy * dy + dz * dz <= r2 {
                            voxels[idx(x, y, z, size)].material = mat;
                        }
                    }
                }
            }
        }

        Region::Layer { material, y } => {
            let mat = resolve(material, registry)?;
            if *y < size {
                for z in 0..size {
                    for x in 0..size {
                        voxels[idx(x, *y, z, size)].material = mat;
                    }
                }
            }
        }

        Region::RandomHeightmap {
            material,
            x_range,
            z_range,
            y_min,
            y_max,
            seed,
        } => {
            let mat = resolve(material, registry)?;
            for z in z_range.0..=z_range.1.min(size - 1) {
                for x in x_range.0..=x_range.1.min(size - 1) {
                    let height = deterministic_height(x, z, *y_min, *y_max, *seed);
                    for y in *y_min..=height.min(size - 1) {
                        voxels[idx(x, y, z, size)].material = mat;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{MaterialData, Phase};

    fn test_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: 0,
            name: "Air".into(),
            default_phase: Phase::Gas,
            density: 1.225,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 1,
            name: "Stone".into(),
            default_phase: Phase::Solid,
            density: 2700.0,
            hardness: 0.9,
            color: [0.5, 0.5, 0.5],
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 3,
            name: "Water".into(),
            default_phase: Phase::Liquid,
            density: 1000.0,
            ..Default::default()
        });
        reg
    }

    #[test]
    fn fill_sets_material_in_region() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();

        let regions = vec![Region::Fill {
            material: "Stone".into(),
            min: (1, 1, 1),
            max: (2, 2, 2),
        }];
        apply_regions(&mut voxels, size, &regions, &reg).unwrap();

        assert_eq!(voxels[idx(1, 1, 1, size)].material, MaterialId::STONE);
        assert_eq!(voxels[idx(2, 2, 2, size)].material, MaterialId::STONE);
        assert_eq!(voxels[idx(0, 0, 0, size)].material, MaterialId::AIR);
    }

    #[test]
    fn shell_creates_hollow_box() {
        let size = 6;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();

        let regions = vec![Region::Shell {
            material: "Stone".into(),
            min: (0, 0, 0),
            max: (5, 5, 5),
            thickness: 1,
        }];
        apply_regions(&mut voxels, size, &regions, &reg).unwrap();

        // Corner should be stone
        assert_eq!(voxels[idx(0, 0, 0, size)].material, MaterialId::STONE);
        // Interior should remain air
        assert_eq!(voxels[idx(2, 2, 2, size)].material, MaterialId::AIR);
        assert_eq!(voxels[idx(3, 3, 3, size)].material, MaterialId::AIR);
    }

    #[test]
    fn single_places_one_voxel() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();

        let regions = vec![Region::Single {
            material: "Water".into(),
            pos: (2, 1, 3),
        }];
        apply_regions(&mut voxels, size, &regions, &reg).unwrap();

        assert_eq!(voxels[idx(2, 1, 3, size)].material, MaterialId::WATER);
    }

    #[test]
    fn every_nth_places_periodically() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();

        let regions = vec![Region::EveryNth {
            material: "Stone".into(),
            min: (0, 0, 0),
            max: (3, 0, 0),
            step: 2,
        }];
        apply_regions(&mut voxels, size, &regions, &reg).unwrap();

        // Positions 0 and 2 should be stone (counter 0 and 2)
        assert_eq!(voxels[idx(0, 0, 0, size)].material, MaterialId::STONE);
        assert_eq!(voxels[idx(1, 0, 0, size)].material, MaterialId::AIR);
        assert_eq!(voxels[idx(2, 0, 0, size)].material, MaterialId::STONE);
        assert_eq!(voxels[idx(3, 0, 0, size)].material, MaterialId::AIR);
    }

    #[test]
    fn unknown_material_returns_error() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();

        let regions = vec![Region::Fill {
            material: "Unobtanium".into(),
            min: (0, 0, 0),
            max: (3, 3, 3),
        }];
        let result = apply_regions(&mut voxels, size, &regions, &reg);
        assert!(result.is_err());
    }

    #[test]
    fn later_regions_overwrite_earlier() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();

        let regions = vec![
            Region::Fill {
                material: "Stone".into(),
                min: (0, 0, 0),
                max: (3, 3, 3),
            },
            Region::Single {
                material: "Water".into(),
                pos: (1, 1, 1),
            },
        ];
        apply_regions(&mut voxels, size, &regions, &reg).unwrap();

        assert_eq!(voxels[idx(0, 0, 0, size)].material, MaterialId::STONE);
        assert_eq!(voxels[idx(1, 1, 1, size)].material, MaterialId::WATER);
    }

    #[test]
    fn sphere_fills_within_radius() {
        let size = 8;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let reg = test_registry();

        let regions = vec![Region::Sphere {
            material: "Water".into(),
            center: (4, 4, 4),
            radius: 2.0,
        }];
        apply_regions(&mut voxels, size, &regions, &reg).unwrap();

        // Center should be water
        assert_eq!(voxels[idx(4, 4, 4, size)].material, MaterialId::WATER);
        // Far corner should remain air
        assert_eq!(voxels[idx(0, 0, 0, size)].material, MaterialId::AIR);
    }

    #[test]
    fn random_heightmap_is_deterministic() {
        let size = 8;
        let reg = test_registry();

        let region = Region::RandomHeightmap {
            material: "Stone".into(),
            x_range: (0, 7),
            z_range: (0, 7),
            y_min: 0,
            y_max: 3,
            seed: 42,
        };

        let mut voxels1 = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let mut voxels2 = vec![Voxel::new(MaterialId::AIR); size * size * size];
        apply_regions(&mut voxels1, size, std::slice::from_ref(&region), &reg).unwrap();
        apply_regions(&mut voxels2, size, std::slice::from_ref(&region), &reg).unwrap();

        // Same seed → identical results
        for i in 0..voxels1.len() {
            assert_eq!(voxels1[i].material, voxels2[i].material);
        }

        // Should have stone somewhere
        let stone_count = voxels1
            .iter()
            .filter(|v| v.material == MaterialId::STONE)
            .count();
        assert!(stone_count > 0, "should have placed some stone");
        // All stone should be at y ≤ 3
        for z in 0..size {
            for x in 0..size {
                for y in 4..size {
                    assert_eq!(
                        voxels1[idx(x, y, z, size)].material,
                        MaterialId::AIR,
                        "no stone above y_max=3 at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn random_heightmap_different_seeds_differ() {
        let size = 8;
        let reg = test_registry();

        let mut v1 = vec![Voxel::new(MaterialId::AIR); size * size * size];
        let mut v2 = vec![Voxel::new(MaterialId::AIR); size * size * size];

        apply_regions(
            &mut v1,
            size,
            &[Region::RandomHeightmap {
                material: "Stone".into(),
                x_range: (0, 7),
                z_range: (0, 7),
                y_min: 0,
                y_max: 5,
                seed: 1,
            }],
            &reg,
        )
        .unwrap();
        apply_regions(
            &mut v2,
            size,
            &[Region::RandomHeightmap {
                material: "Stone".into(),
                x_range: (0, 7),
                z_range: (0, 7),
                y_min: 0,
                y_max: 5,
                seed: 999,
            }],
            &reg,
        )
        .unwrap();

        let differ = v1
            .iter()
            .zip(v2.iter())
            .any(|(a, b)| a.material != b.material);
        assert!(differ, "different seeds should produce different terrain");
    }
}
