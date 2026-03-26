// World prop system: static scenery objects scattered across the terrain.
//
// Props are non-living, non-inventory objects (rocks, boulders, logs, pebbles)
// placed by the biome system during chunk decoration. Each prop has a simple
// mesh, material color, and optional collision shape.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::{PropCategory, PropData, SlopePreference};
use crate::procgen::creatures::SimpleRng;
use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::voxel::MaterialId;

/// ECS component for a spawned prop instance.
#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Prop {
    pub prop_type: String,
    pub display_name: String,
    pub category: PropCategory,
    pub material_id: u16,
    pub scale: Vec3,
}

/// Marker component: chunk needs prop decoration.
#[derive(Component)]
pub struct NeedsDecoration;

/// Tracks prop entities belonging to a chunk for cleanup on unload.
#[derive(Component, Default)]
pub struct ChunkProps {
    pub entities: Vec<Entity>,
}

/// Resource holding loaded PropData templates, indexed by prop_type.
#[derive(Resource, Default)]
pub struct PropRegistry {
    props: std::collections::HashMap<String, PropData>,
}

impl PropRegistry {
    pub fn get(&self, prop_type: &str) -> Option<&PropData> {
        self.props.get(prop_type)
    }

    pub fn insert(&mut self, data: PropData) {
        self.props.insert(data.prop_type.clone(), data);
    }

    pub fn len(&self) -> usize {
        self.props.len()
    }

    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }
}

/// Generate a Prop component from a template with scale variation.
pub fn generate_prop(template: &PropData, material_id: u16, seed: u64) -> Prop {
    let mut rng = SimpleRng::new(seed);
    let v = template.scale_variation;

    let scale_x = template.base_scale.0 * (1.0 + rng.next_signed() * v);
    let scale_y = template.base_scale.1 * (1.0 + rng.next_signed() * v);
    let scale_z = template.base_scale.2 * (1.0 + rng.next_signed() * v);

    Prop {
        prop_type: template.prop_type.clone(),
        display_name: template.display_name.clone(),
        category: template.category,
        material_id,
        scale: Vec3::new(scale_x.max(0.01), scale_y.max(0.01), scale_z.max(0.01)),
    }
}

/// Find the surface Y coordinate at a local (x, z) position in a chunk.
/// Scans from top down to find the first solid voxel.
/// Returns None if the column is entirely air.
pub fn surface_height(chunk: &Chunk, x: usize, z: usize) -> Option<usize> {
    (0..CHUNK_SIZE)
        .rev()
        .find(|&y| !chunk.get(x, y, z).material.is_air())
}

/// Estimate slope at a surface position by checking neighbor heights.
/// Returns approximate slope angle in degrees.
pub fn estimate_slope(chunk: &Chunk, x: usize, z: usize) -> f32 {
    let center = match surface_height(chunk, x, z) {
        Some(h) => h as f32,
        None => return 0.0,
    };

    let mut max_diff: f32 = 0.0;
    for (dx, dz) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
        let nx = x as i32 + dx;
        let nz = z as i32 + dz;
        if nx >= 0
            && nx < CHUNK_SIZE as i32
            && nz >= 0
            && nz < CHUNK_SIZE as i32
            && let Some(nh) = surface_height(chunk, nx as usize, nz as usize)
        {
            let diff = (nh as f32 - center).abs();
            max_diff = max_diff.max(diff);
        }
    }

    // slope angle: atan(rise/run), run = 1 voxel = 1 meter
    max_diff.atan().to_degrees()
}

/// Check if a slope value matches a SlopePreference.
pub fn slope_matches(preference: SlopePreference, slope_degrees: f32) -> bool {
    match preference {
        SlopePreference::Flat => slope_degrees < 15.0,
        SlopePreference::Moderate => slope_degrees < 45.0,
        SlopePreference::Steep => slope_degrees > 30.0,
        SlopePreference::Any => true,
    }
}

/// Check if the surface material at a position is suitable for props.
/// Props shouldn't spawn on water, lava, air, or steam.
pub fn is_valid_surface(chunk: &Chunk, x: usize, y: usize, z: usize) -> bool {
    let mat = chunk.get(x, y, z).material;
    mat != MaterialId::AIR
        && mat != MaterialId::WATER
        && mat != MaterialId::LAVA
        && mat != MaterialId::STEAM
        && mat != MaterialId::ICE
}

/// Build a `PropRegistry` by reading all `.prop.ron` files from disk.
///
/// Uses `find_data_dir()` to locate the `assets/data/` directory, then scans
/// the `props/` subdirectory.
pub fn load_prop_registry() -> Result<PropRegistry, String> {
    let dir = crate::data::find_data_dir()?.join("props");
    if !dir.is_dir() {
        return Ok(PropRegistry::default());
    }
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;

    let mut registry = PropRegistry::default();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".prop.ron") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let data: PropData =
            ron::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        registry.insert(data);
    }

    Ok(registry)
}

/// System: decorates newly generated chunks with prop entities.
pub fn decorate_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    material_registry: Res<crate::data::MaterialRegistry>,
    prop_registry: Res<PropRegistry>,
    mut to_decorate: Query<(Entity, &Chunk, &ChunkCoord, &mut ChunkProps), With<NeedsDecoration>>,
    biome_assets: Res<Assets<crate::procgen::biomes::BiomeData>>,
) {
    use crate::procgen::spawning::plan_chunk_prop_spawns;

    if prop_registry.is_empty() {
        return;
    }

    for (chunk_entity, chunk, coord, mut chunk_props) in &mut to_decorate {
        commands.entity(chunk_entity).remove::<NeedsDecoration>();

        // Collect biomes from assets
        let biomes: Vec<&crate::procgen::biomes::BiomeData> =
            biome_assets.iter().map(|(_, b)| b).collect();
        if biomes.is_empty() {
            continue;
        }

        // Simple biome match: use center-column height as proxy altitude.
        let center_height = surface_height(chunk, CHUNK_SIZE / 2, CHUNK_SIZE / 2).unwrap_or(0);
        let world_y = coord.y as f32 * CHUNK_SIZE as f32 + center_height as f32;
        let biome = biomes
            .iter()
            .find(|b| world_y >= b.height_range.0 && world_y <= b.height_range.1)
            .or(biomes.first());
        let Some(biome) = biome else { continue };

        if biome.prop_spawns.is_empty() {
            continue;
        }

        let spawns = plan_chunk_prop_spawns(biome, coord.x, coord.z, CHUNK_SIZE, 42);
        let origin = coord.world_origin();

        for (prop_type, local_x, local_z, seed) in spawns {
            let Some(template) = prop_registry.get(&prop_type) else {
                continue;
            };

            let ix = (local_x as usize).min(CHUNK_SIZE - 1);
            let iz = (local_z as usize).min(CHUNK_SIZE - 1);

            let Some(sy) = surface_height(chunk, ix, iz) else {
                continue;
            };

            if !is_valid_surface(chunk, ix, sy, iz) {
                continue;
            }

            let slope = estimate_slope(chunk, ix, iz);
            if !slope_matches(template.slope_preference, slope) {
                continue;
            }

            let wy = origin.y as f32 + sy as f32 + 1.0;
            if let Some(min_alt) = template.min_altitude
                && wy < min_alt
            {
                continue;
            }
            if let Some(max_alt) = template.max_altitude
                && wy > max_alt
            {
                continue;
            }

            // Resolve material color
            let material_id = material_registry
                .resolve_name(&template.material)
                .unwrap_or(MaterialId::STONE);
            let mat_data = material_registry.get(material_id);
            let color = mat_data
                .map(|m| Color::srgb(m.color[0], m.color[1], m.color[2]))
                .unwrap_or(Color::srgb(0.5, 0.5, 0.5));

            let prop = generate_prop(template, material_id.0, seed);
            let prop_scale = prop.scale;

            let mesh = match template.category {
                PropCategory::Rock => meshes.add(Sphere::new(0.5)),
                _ => meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
            };
            let material = materials.add(StandardMaterial {
                base_color: color,
                perceptual_roughness: 0.9,
                ..default()
            });

            let world_x = origin.x as f32 + local_x;
            let world_z = origin.z as f32 + local_z;

            let mut rng = SimpleRng::new(seed.wrapping_add(7));
            let rotation = Quat::from_rotation_y(rng.next_f32() * std::f32::consts::TAU);

            let prop_entity = commands
                .spawn((
                    prop,
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    Transform::from_xyz(world_x, wy, world_z)
                        .with_scale(prop_scale)
                        .with_rotation(rotation),
                ))
                .id();

            chunk_props.entities.push(prop_entity);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_template() -> PropData {
        PropData {
            prop_type: "rock".into(),
            display_name: "Rock".into(),
            category: PropCategory::Rock,
            material: "Stone".into(),
            base_scale: (0.5, 0.4, 0.5),
            scale_variation: 0.3,
            collision: "aabb".into(),
            slope_preference: SlopePreference::Any,
            min_altitude: None,
            max_altitude: None,
        }
    }

    #[test]
    fn generate_prop_applies_variation() {
        let template = test_template();
        let a = generate_prop(&template, 1, 42);
        let b = generate_prop(&template, 1, 999);
        assert_ne!(a.scale, b.scale);
    }

    #[test]
    fn generate_prop_deterministic() {
        let template = test_template();
        let a = generate_prop(&template, 1, 42);
        let b = generate_prop(&template, 1, 42);
        assert_eq!(a.scale, b.scale);
    }

    #[test]
    fn generate_prop_minimum_scale() {
        let mut template = test_template();
        template.base_scale = (0.01, 0.01, 0.01);
        template.scale_variation = 0.99;
        for seed in 0..100 {
            let prop = generate_prop(&template, 1, seed);
            assert!(prop.scale.x >= 0.01);
            assert!(prop.scale.y >= 0.01);
            assert!(prop.scale.z >= 0.01);
        }
    }

    #[test]
    fn slope_flat_rejects_steep() {
        assert!(slope_matches(SlopePreference::Flat, 10.0));
        assert!(!slope_matches(SlopePreference::Flat, 20.0));
    }

    #[test]
    fn slope_steep_rejects_flat() {
        assert!(slope_matches(SlopePreference::Steep, 45.0));
        assert!(!slope_matches(SlopePreference::Steep, 10.0));
    }

    #[test]
    fn slope_any_accepts_all() {
        assert!(slope_matches(SlopePreference::Any, 0.0));
        assert!(slope_matches(SlopePreference::Any, 90.0));
    }

    #[test]
    fn surface_height_empty_chunk() {
        let chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        assert_eq!(surface_height(&chunk, 5, 5), None);
    }

    #[test]
    fn surface_height_finds_top() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        for y in 0..10 {
            chunk.set_material(5, y, 5, MaterialId::STONE);
        }
        assert_eq!(surface_height(&chunk, 5, 5), Some(9));
    }

    #[test]
    fn is_valid_surface_rejects_liquids() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.set_material(0, 0, 0, MaterialId::WATER);
        assert!(!is_valid_surface(&chunk, 0, 0, 0));

        chunk.set_material(1, 0, 0, MaterialId::LAVA);
        assert!(!is_valid_surface(&chunk, 1, 0, 0));

        chunk.set_material(2, 0, 0, MaterialId::STONE);
        assert!(is_valid_surface(&chunk, 2, 0, 0));
    }

    #[test]
    fn prop_registry_insert_and_get() {
        let mut reg = PropRegistry::default();
        assert!(reg.is_empty());

        reg.insert(test_template());
        assert_eq!(reg.len(), 1);
        assert!(reg.get("rock").is_some());
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn load_prop_registry_from_disk() {
        let registry = load_prop_registry().expect("should load prop files");
        assert!(
            !registry.is_empty(),
            "expected at least one .prop.ron file in assets/data/props/"
        );
        assert!(registry.get("boulder").is_some());
        assert!(registry.get("rock").is_some());
    }
}
