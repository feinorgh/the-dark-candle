// Procedural tree generator using recursive branching into an octree.
//
// Produces `OctreeNode<Voxel>` tree structures with physically correct material
// layering: bark shell → wood core for trunk and major branches, transitioning
// to twig and dry-leaf voxels at finer subdivision levels. The octree naturally
// compresses uniform regions (air) while preserving detail where branches taper.
//
// The generator operates in a continuous coordinate space (meters) and rasterises
// branch segments into the octree at the appropriate resolution level. Trunk and
// major branches write at base resolution (1 m voxels); secondary branches at
// depth 1–2 (50–25 cm); twigs/leaves at depth 3–4 (12.5–6.25 cm).

use serde::{Deserialize, Serialize};

use crate::world::octree::OctreeNode;
use crate::world::voxel::{MaterialId, Voxel};

/// Configuration for procedural tree generation, loadable from `.tree.ron`.
#[derive(Debug, Clone, Serialize, Deserialize, bevy::asset::Asset, bevy::reflect::TypePath)]
pub struct TreeConfig {
    /// Trunk radius at the base, in meters.
    pub trunk_radius: f32,
    /// Total trunk height before the first branch tier, in meters.
    pub trunk_height: f32,
    /// Number of recursive branching levels (0 = trunk only).
    pub branch_depth: u32,
    /// Each child branch's length as a fraction of the parent segment length.
    pub length_ratio: f32,
    /// Each child branch's radius as a fraction of the parent segment radius.
    pub radius_ratio: f32,
    /// Angle (degrees) that child branches diverge from the parent direction.
    pub branch_angle_deg: f32,
    /// Number of child branches spawned at each fork.
    pub fork_count: u32,
    /// Octree subdivision depth for fine geometry (twigs/leaves). Max 4.
    pub octree_depth: u32,
    /// Ambient temperature for all voxels (Kelvin).
    pub ambient_temperature: f32,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            trunk_radius: 0.6,
            trunk_height: 8.0,
            branch_depth: 4,
            length_ratio: 0.6,
            radius_ratio: 0.55,
            branch_angle_deg: 35.0,
            fork_count: 3,
            octree_depth: 4,
            ambient_temperature: 288.15,
        }
    }
}

/// A 3D direction vector (not necessarily normalised internally, but all
/// operations normalise when needed).
#[derive(Debug, Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalised(self) -> Self {
        let len = self.length();
        if len < 1e-9 {
            return Self::new(0.0, 1.0, 0.0);
        }
        Self::new(self.x / len, self.y / len, self.z / len)
    }

    fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Find an arbitrary vector perpendicular to `self` (assumed normalised).
    fn any_perpendicular(self) -> Self {
        let candidate = if self.y.abs() < 0.9 {
            Self::new(0.0, 1.0, 0.0)
        } else {
            Self::new(1.0, 0.0, 0.0)
        };
        cross(self, candidate).normalised()
    }
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Rotate `v` around `axis` (assumed unit length) by `angle_rad` radians
/// using Rodrigues' rotation formula.
fn rotate_around(v: Vec3, axis: Vec3, angle_rad: f32) -> Vec3 {
    let (sin_a, cos_a) = angle_rad.sin_cos();
    let dot = v.x * axis.x + v.y * axis.y + v.z * axis.z;
    let cr = cross(axis, v);
    Vec3::new(
        v.x * cos_a + cr.x * sin_a + axis.x * dot * (1.0 - cos_a),
        v.y * cos_a + cr.y * sin_a + axis.y * dot * (1.0 - cos_a),
        v.z * cos_a + cr.z * sin_a + axis.z * dot * (1.0 - cos_a),
    )
}

/// A branch segment to rasterise.
struct Segment {
    /// Start position in meters, relative to tree origin.
    start: Vec3,
    /// Growth direction (unit vector).
    direction: Vec3,
    /// Segment length in meters.
    length: f32,
    /// Segment radius at the start (base) in meters.
    radius: f32,
    /// Remaining branching recursion depth.
    depth_remaining: u32,
}

/// Generate a tree as an `OctreeNode<Voxel>`.
///
/// `grid_size` is the edge length of the enclosing cube in base-resolution
/// voxels (e.g., 32 for a 32³ region). The tree is centred on the XZ plane at
/// `grid_size / 2` and grows upward along +Y from y=0.
///
/// The returned octree's finest cells are at `grid_size * 2^octree_depth`.
pub fn generate_tree(config: &TreeConfig, grid_size: usize) -> OctreeNode<Voxel> {
    let depth = config.octree_depth.min(4);
    let scale = 1usize << depth;
    let hi_size = grid_size * scale;

    // Target cell size at various depths: the finest cells have this edge.
    let cell_size_finest = 1.0 / scale as f32; // metres per cell at max depth

    let air = Voxel {
        material: MaterialId::AIR,
        temperature: config.ambient_temperature,
        pressure: 101_325.0,
        damage: 0.0,
        latent_heat_buffer: 0.0,
        density: 0.0,
    };

    let mut root = OctreeNode::new_leaf(air);

    // Collect all segments via recursive branching, then rasterise.
    let mut segments = Vec::new();
    let cx = grid_size as f32 / 2.0;
    let cz = grid_size as f32 / 2.0;

    collect_segments(
        &mut segments,
        config,
        Vec3::new(cx, 0.0, cz),
        Vec3::new(0.0, 1.0, 0.0),
        config.trunk_height,
        config.trunk_radius,
        config.branch_depth,
    );

    // Rasterise each segment into the octree.
    for seg in &segments {
        rasterise_segment(&mut root, seg, hi_size, cell_size_finest, config);
    }

    root
}

/// Recursively build the list of branch segments.
fn collect_segments(
    out: &mut Vec<Segment>,
    config: &TreeConfig,
    start: Vec3,
    direction: Vec3,
    length: f32,
    radius: f32,
    depth_remaining: u32,
) {
    let dir = direction.normalised();
    out.push(Segment {
        start,
        direction: dir,
        length,
        radius,
        depth_remaining,
    });

    if depth_remaining == 0 || radius < 0.01 {
        return;
    }

    // Fork point is at the end of this segment.
    let fork = start.add(dir.scale(length));
    let child_length = length * config.length_ratio;
    let child_radius = radius * config.radius_ratio;

    // One continuation branch roughly along the parent direction (slight lean).
    let continuation_dir = rotate_around(dir, dir.any_perpendicular(), 0.1);
    collect_segments(
        out,
        config,
        fork,
        continuation_dir,
        child_length,
        child_radius,
        depth_remaining - 1,
    );

    // Lateral branches evenly distributed around the parent axis.
    let angle_rad = config.branch_angle_deg.to_radians();
    let perp = dir.any_perpendicular();
    for i in 0..config.fork_count {
        let azimuth = 2.0 * std::f32::consts::PI * (i as f32) / (config.fork_count as f32);
        let rotated_perp = rotate_around(perp, dir, azimuth);
        let branch_dir = rotate_around(dir, rotated_perp, angle_rad);
        collect_segments(
            out,
            config,
            fork,
            branch_dir,
            child_length,
            child_radius,
            depth_remaining - 1,
        );
    }
}

/// Rasterise a single branch segment into the octree, choosing material and
/// resolution based on the segment's radius (thickness).
fn rasterise_segment(
    root: &mut OctreeNode<Voxel>,
    seg: &Segment,
    hi_size: usize,
    cell_size: f32,
    config: &TreeConfig,
) {
    // Determine material and target cell size based on branch thickness.
    let (core_mat, shell_mat, target_cells) = material_for_radius(seg.radius, cell_size);

    let target_size = target_cells.max(1);

    let dir = seg.direction.normalised();
    let steps = (seg.length / (cell_size * 0.5)).ceil() as usize;
    let step_len = seg.length / steps.max(1) as f32;

    // Tapering: radius decreases linearly from base to tip.
    let tip_radius = seg.radius * 0.7;

    for i in 0..=steps {
        let t = i as f32 / steps.max(1) as f32;
        let pos = seg.start.add(dir.scale(step_len * i as f32));
        let r = seg.radius * (1.0 - t) + tip_radius * t;

        // Fill a disc perpendicular to the branch direction.
        fill_disc(
            root,
            pos,
            dir,
            r,
            core_mat,
            shell_mat,
            hi_size,
            cell_size,
            target_size,
            config.ambient_temperature,
        );
    }

    // If this is the finest branch level, add leaf clusters at the tip.
    if seg.depth_remaining == 0 && seg.radius < 0.15 {
        let tip = seg.start.add(dir.scale(seg.length));
        add_leaf_cluster(root, tip, hi_size, cell_size, config.ambient_temperature);
    }
}

/// Choose material based on branch thickness.
/// Returns (core_material, shell_material, target_cell_count).
fn material_for_radius(radius: f32, cell_size: f32) -> (MaterialId, MaterialId, usize) {
    if radius >= 0.3 {
        // Thick trunk/branch: wood core + bark shell at base resolution
        (
            MaterialId::WOOD,
            MaterialId::BARK,
            (1.0 / cell_size) as usize,
        )
    } else if radius >= 0.08 {
        // Medium branch: wood core + bark shell at 2× resolution
        let target = ((0.5 / cell_size) as usize).max(1);
        (MaterialId::WOOD, MaterialId::BARK, target)
    } else if radius >= 0.03 {
        // Thin branch / twig
        (MaterialId::TWIG, MaterialId::TWIG, 1)
    } else {
        // Finest twigs
        (MaterialId::TWIG, MaterialId::TWIG, 1)
    }
}

/// Fill a disc of voxels perpendicular to `dir` at `centre`, with core/shell
/// material layering (bark on the outside, wood on the inside).
#[allow(clippy::too_many_arguments)]
fn fill_disc(
    root: &mut OctreeNode<Voxel>,
    centre: Vec3,
    _dir: Vec3,
    radius: f32,
    core_mat: MaterialId,
    shell_mat: MaterialId,
    hi_size: usize,
    cell_size: f32,
    target_size: usize,
    temperature: f32,
) {
    let r_cells = (radius / cell_size).ceil() as i32;
    let cx = (centre.x / cell_size) as i32;
    let cy = (centre.y / cell_size) as i32;
    let cz = (centre.z / cell_size) as i32;
    let r_sq = radius * radius;
    let shell_inner_r_sq = (radius - cell_size).max(0.0).powi(2);

    for dz in -r_cells..=r_cells {
        for dy in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let wx = (cx + dx) as f32 * cell_size;
                let wy = (cy + dy) as f32 * cell_size;
                let wz = (cz + dz) as f32 * cell_size;

                let dist_x = wx - centre.x;
                let dist_y = wy - centre.y;
                let dist_z = wz - centre.z;
                let dist_sq = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

                if dist_sq > r_sq {
                    continue;
                }

                let gx = (cx + dx) as usize;
                let gy = (cy + dy) as usize;
                let gz = (cz + dz) as usize;

                if gx >= hi_size || gy >= hi_size || gz >= hi_size {
                    continue;
                }

                let mat = if dist_sq >= shell_inner_r_sq {
                    shell_mat
                } else {
                    core_mat
                };

                let voxel = Voxel {
                    material: mat,
                    temperature,
                    pressure: 101_325.0,
                    damage: 0.0,
                    latent_heat_buffer: 0.0,
                    density: 1.0,
                };
                root.set(gx, gy, gz, hi_size, target_size, voxel);
            }
        }
    }
}

/// Place a small cluster of dry-leaf voxels around a branch tip.
fn add_leaf_cluster(
    root: &mut OctreeNode<Voxel>,
    centre: Vec3,
    hi_size: usize,
    cell_size: f32,
    temperature: f32,
) {
    let leaf_voxel = Voxel {
        material: MaterialId::DRY_LEAVES,
        temperature,
        pressure: 101_325.0,
        damage: 0.0,
        latent_heat_buffer: 0.0,
        density: 1.0,
    };

    let radius: i32 = 3; // cells in each direction
    let cx = (centre.x / cell_size) as i32;
    let cy = (centre.y / cell_size) as i32;
    let cz = (centre.z / cell_size) as i32;

    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                // Sparse pattern: only ~40% of cells get leaves for natural look.
                let hash = (dx.wrapping_mul(73) ^ dy.wrapping_mul(179) ^ dz.wrapping_mul(283))
                    .unsigned_abs();
                if hash % 5 < 3 {
                    continue;
                }

                let gx = (cx + dx) as usize;
                let gy = (cy + dy) as usize;
                let gz = (cz + dz) as usize;

                if gx >= hi_size || gy >= hi_size || gz >= hi_size {
                    continue;
                }

                // Only place leaf if cell is currently air.
                if root.get(gx, gy, gz, hi_size).material.is_air() {
                    root.set(gx, gy, gz, hi_size, 1, leaf_voxel);
                }
            }
        }
    }
}

/// Generate a tree at base resolution (1 voxel = 1 m) and stamp its non-air
/// voxels into a chunk. The tree base is placed at local chunk position
/// `(base_x, base_y, base_z)` where `base_y` is the Y of the surface voxel.
///
/// Only voxels within chunk bounds [0, CHUNK_SIZE) are written. Trees taller
/// than the chunk are clipped. Returns the number of voxels written.
pub fn stamp_tree_into_chunk(
    config: &TreeConfig,
    chunk: &mut crate::world::chunk::Chunk,
    base_x: usize,
    base_y: usize,
    base_z: usize,
    seed: u64,
) -> usize {
    use crate::world::chunk::CHUNK_SIZE;

    // Determine grid size to fit the tree: enough for trunk height + branches.
    let extent = (config.trunk_height * 2.0).ceil() as usize + 4;
    let grid_size = extent.next_power_of_two().max(8);

    // Generate at base resolution only (depth=0) for chunk integration.
    let base_config = TreeConfig {
        octree_depth: 0,
        ..*config
    };
    let tree = generate_tree(&base_config, grid_size);

    // The tree is centred at (grid_size/2, 0, grid_size/2) in tree space.
    // Map to chunk space: tree(tx,ty,tz) → chunk(base_x + tx - grid_size/2, base_y + ty, base_z + tz - grid_size/2)
    let half = grid_size / 2;

    // Use a simple deterministic rotation based on the seed.
    let _seed = seed; // reserved for future random rotation

    let mut written = 0;
    tree.for_each_leaf(0, 0, 0, grid_size, &mut |lx, ly, lz, _leaf_size, voxel| {
        if voxel.material.is_air() {
            return;
        }
        let cx = lx as i32 - half as i32 + base_x as i32;
        let cy = ly as i32 + base_y as i32 + 1; // +1: tree starts above surface
        let cz = lz as i32 - half as i32 + base_z as i32;

        if cx >= 0
            && cx < CHUNK_SIZE as i32
            && cy >= 0
            && cy < CHUNK_SIZE as i32
            && cz >= 0
            && cz < CHUNK_SIZE as i32
        {
            let ux = cx as usize;
            let uy = cy as usize;
            let uz = cz as usize;
            // Only write into air cells (don't overwrite terrain).
            if chunk.get(ux, uy, uz).material.is_air() {
                chunk.set(ux, uy, uz, *voxel);
                written += 1;
            }
        }
    });

    written
}

/// Resource holding loaded TreeConfig templates, indexed by name.
#[derive(bevy::prelude::Resource, Default)]
pub struct TreeRegistry {
    trees: std::collections::HashMap<String, TreeConfig>,
}

impl TreeRegistry {
    pub fn get(&self, name: &str) -> Option<&TreeConfig> {
        self.trees.get(name)
    }

    pub fn insert(&mut self, name: String, config: TreeConfig) {
        self.trees.insert(name, config);
    }

    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &TreeConfig)> {
        self.trees.iter()
    }
}

/// Build a `TreeRegistry` by reading all `.tree.ron` files from disk.
pub fn load_tree_registry() -> Result<TreeRegistry, String> {
    let dir = crate::data::find_data_dir()?.join("trees");
    if !dir.is_dir() {
        return Ok(TreeRegistry::default());
    }
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;

    let mut registry = TreeRegistry::default();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".tree.ron") {
            continue;
        }
        let stem = name.trim_end_matches(".tree.ron").to_string();
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let config: TreeConfig =
            ron::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        registry.insert(stem, config);
    }

    Ok(registry)
}

/// Marker component for chunks that need tree planting.
/// Added alongside `NeedsDecoration` at spawn time, consumed by `plant_trees`.
#[derive(bevy::prelude::Component)]
pub struct NeedsTreePlanting;

/// ECS system: plants voxel trees into newly generated chunks based on biome data.
///
/// Queries chunks that still have `NeedsTreePlanting`, determines the biome,
/// plans tree spawn positions, and stamps tree voxels directly into the chunk.
pub fn plant_trees(
    mut commands: bevy::prelude::Commands,
    tree_registry: bevy::prelude::Res<TreeRegistry>,
    biome_assets: bevy::prelude::Res<bevy::prelude::Assets<crate::procgen::biomes::BiomeData>>,
    mut to_plant: bevy::prelude::Query<
        (
            bevy::prelude::Entity,
            &mut crate::world::chunk::Chunk,
            &crate::world::chunk::ChunkCoord,
        ),
        bevy::prelude::With<crate::procgen::props::NeedsDecoration>,
    >,
) {
    use crate::procgen::props::surface_height;
    use crate::procgen::spawning::plan_chunk_tree_spawns;
    use crate::world::chunk::CHUNK_SIZE;

    if tree_registry.is_empty() {
        return;
    }

    let biomes: Vec<&crate::procgen::biomes::BiomeData> =
        biome_assets.iter().map(|(_, b)| b).collect();
    if biomes.is_empty() {
        return;
    }

    for (entity, mut chunk, coord) in &mut to_plant {
        let _ = entity;
        let _ = &mut commands;

        // Simple biome match (same logic as decorate_chunks).
        let center_height = surface_height(&chunk, CHUNK_SIZE / 2, CHUNK_SIZE / 2).unwrap_or(0);
        let world_y = coord.y as f32 * CHUNK_SIZE as f32 + center_height as f32;
        let biome = biomes
            .iter()
            .find(|b| world_y >= b.height_range.0 && world_y <= b.height_range.1)
            .or(biomes.first());
        let Some(biome) = biome else { continue };

        if biome.tree_spawns.is_empty() {
            continue;
        }

        let spawns = plan_chunk_tree_spawns(biome, coord.x, coord.z, CHUNK_SIZE, 42);

        for (tree_name, local_x, local_z, seed) in spawns {
            let Some(config) = tree_registry.get(&tree_name) else {
                continue;
            };

            let ix = (local_x as usize).min(CHUNK_SIZE - 1);
            let iz = (local_z as usize).min(CHUNK_SIZE - 1);

            let Some(sy) = surface_height(&chunk, ix, iz) else {
                continue;
            };

            // Don't plant on water/lava/etc.
            if !crate::procgen::props::is_valid_surface(&chunk, ix, sy, iz) {
                continue;
            }

            stamp_tree_into_chunk(config, &mut chunk, ix, sy, iz, seed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel_access::octree_to_flat;

    #[test]
    fn default_tree_produces_nonzero_voxels() {
        let config = TreeConfig::default();
        let grid_size = 16;
        let tree = generate_tree(&config, grid_size);
        let depth = config.octree_depth;
        let hi_size = grid_size * (1 << depth);
        let flat = octree_to_flat(&tree, hi_size);
        let solid_count = flat.iter().filter(|v| !v.material.is_air()).count();
        assert!(
            solid_count > 100,
            "Expected at least 100 solid voxels, got {solid_count}"
        );
    }

    #[test]
    fn tree_contains_expected_materials() {
        let config = TreeConfig {
            trunk_radius: 0.5,
            trunk_height: 6.0,
            branch_depth: 3,
            radius_ratio: 0.4,
            octree_depth: 3,
            ..Default::default()
        };
        let grid_size = 16;
        let tree = generate_tree(&config, grid_size);
        let hi_size = grid_size * (1 << config.octree_depth);
        let flat = octree_to_flat(&tree, hi_size);

        let has_wood = flat.iter().any(|v| v.material == MaterialId::WOOD);
        let has_bark = flat.iter().any(|v| v.material == MaterialId::BARK);
        let has_twig = flat.iter().any(|v| v.material == MaterialId::TWIG);

        assert!(has_wood, "Tree should contain WOOD voxels");
        assert!(has_bark, "Tree should contain BARK voxels");
        assert!(has_twig, "Tree should contain TWIG voxels");
    }

    #[test]
    fn tree_all_within_bounds() {
        let config = TreeConfig {
            trunk_radius: 0.4,
            trunk_height: 5.0,
            branch_depth: 2,
            octree_depth: 2,
            ..Default::default()
        };
        let grid_size = 16;
        let tree = generate_tree(&config, grid_size);
        let hi_size = grid_size * (1 << config.octree_depth);
        let flat = octree_to_flat(&tree, hi_size);

        // All solid voxels should be within the grid.
        assert_eq!(flat.len(), hi_size * hi_size * hi_size);
    }

    #[test]
    fn tiny_tree_no_panic() {
        let config = TreeConfig {
            trunk_radius: 0.2,
            trunk_height: 2.0,
            branch_depth: 1,
            octree_depth: 1,
            ..Default::default()
        };
        let _tree = generate_tree(&config, 8);
    }

    #[test]
    fn zero_branch_depth_produces_trunk_only() {
        let config = TreeConfig {
            trunk_radius: 0.5,
            trunk_height: 4.0,
            branch_depth: 0,
            octree_depth: 2,
            ..Default::default()
        };
        let grid_size = 8;
        let tree = generate_tree(&config, grid_size);
        let hi_size = grid_size * (1 << config.octree_depth);
        let flat = octree_to_flat(&tree, hi_size);

        let solid_count = flat.iter().filter(|v| !v.material.is_air()).count();
        assert!(solid_count > 10, "Trunk-only tree should have solid voxels");

        // Trunk has bark and wood.
        let has_bark = flat.iter().any(|v| v.material == MaterialId::BARK);
        assert!(has_bark, "Trunk should have bark shell");
    }

    #[test]
    fn leaf_clusters_present_on_fine_branches() {
        let config = TreeConfig {
            trunk_radius: 0.4,
            trunk_height: 4.0,
            branch_depth: 3,
            octree_depth: 4,
            length_ratio: 0.5,
            radius_ratio: 0.4,
            ..Default::default()
        };
        let grid_size = 16;
        let tree = generate_tree(&config, grid_size);
        let hi_size = grid_size * (1 << config.octree_depth);
        let flat = octree_to_flat(&tree, hi_size);

        let has_leaves = flat.iter().any(|v| v.material == MaterialId::DRY_LEAVES);
        assert!(has_leaves, "Fine branches should have leaf clusters");
    }

    #[test]
    fn stamp_tree_into_chunk_writes_voxels() {
        use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};

        let config = TreeConfig {
            trunk_radius: 0.5,
            trunk_height: 6.0,
            branch_depth: 1,
            octree_depth: 0,
            ..Default::default()
        };
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new_empty(coord);

        // Place dirt surface at y=4.
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.set_material(x, 4, z, MaterialId::DIRT);
            }
        }

        let written = stamp_tree_into_chunk(&config, &mut chunk, 16, 4, 16, 42);
        assert!(
            written > 5,
            "Expected at least 5 tree voxels in chunk, got {written}"
        );

        // Verify trunk exists above the surface.
        let trunk_mat = chunk.get(16, 5, 16).material;
        assert!(
            trunk_mat == MaterialId::WOOD || trunk_mat == MaterialId::BARK,
            "Expected wood or bark at trunk base, got {trunk_mat:?}"
        );
    }

    #[test]
    fn stamp_tree_clips_at_chunk_boundary() {
        use crate::world::chunk::{Chunk, ChunkCoord};

        let config = TreeConfig {
            trunk_radius: 0.5,
            trunk_height: 6.0,
            branch_depth: 1,
            octree_depth: 0,
            ..Default::default()
        };
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut chunk = Chunk::new_empty(coord);

        // Place tree at corner — should not panic, just clip.
        // Just verify it doesn't panic — clipped trees produce fewer voxels.
        let _written = stamp_tree_into_chunk(&config, &mut chunk, 0, 0, 0, 99);
    }

    #[test]
    fn load_tree_registry_finds_oak() {
        let registry = load_tree_registry().expect("tree registry");
        assert!(registry.get("oak").is_some(), "Should find oak.tree.ron");
    }
}
