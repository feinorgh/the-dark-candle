// Chunk storage for the voxel world.
//
// A Chunk is a fixed-size cubic volume of voxels (CHUNK_SIZE³). Chunks are the
// fundamental unit of world storage, meshing, simulation, and streaming. Each
// chunk tracks whether its voxel data has been modified (dirty flag) so that
// dependent systems (meshing, physics) only reprocess changed regions.

use bevy::prelude::*;

use super::octree::OctreeNode;
use super::voxel::{MaterialId, Voxel};
use super::voxel_access::{VoxelAccess, flat_to_octree, octree_to_flat};

/// ECS component wrapping a sparse voxel octree for a chunk.
///
/// Built from the flat voxel array after terrain generation. Provides
/// compressed representation for meshing and physics coupling.
#[derive(Component)]
pub struct ChunkOctree(pub OctreeNode<Voxel>);

/// Edge length of a chunk in voxels. 32³ = 32,768 voxels per chunk.
pub const CHUNK_SIZE: usize = 32;

/// Total number of voxels in a chunk.
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Integer coordinate identifying a chunk in world space.
/// Each unit corresponds to CHUNK_SIZE voxels.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Component,
)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Convert a world-space voxel position to the chunk coordinate that contains it.
    pub fn from_voxel_pos(vx: i32, vy: i32, vz: i32) -> Self {
        Self {
            x: vx.div_euclid(CHUNK_SIZE as i32),
            y: vy.div_euclid(CHUNK_SIZE as i32),
            z: vz.div_euclid(CHUNK_SIZE as i32),
        }
    }

    /// World-space origin (minimum corner) of this chunk in voxel coordinates.
    pub fn world_origin(&self) -> IVec3 {
        IVec3::new(
            self.x * CHUNK_SIZE as i32,
            self.y * CHUNK_SIZE as i32,
            self.z * CHUNK_SIZE as i32,
        )
    }

    /// World-space position of this chunk's center.
    pub fn world_center(&self) -> Vec3 {
        let origin = self.world_origin();
        let half = CHUNK_SIZE as f32 / 2.0;
        Vec3::new(
            origin.x as f32 + half,
            origin.y as f32 + half,
            origin.z as f32 + half,
        )
    }
}

/// Storage for a cubic volume of voxels.
///
/// Voxels are stored in a flat array indexed as `z * CHUNK_SIZE² + y * CHUNK_SIZE + x`.
/// The dirty flag is set whenever voxel data is modified, allowing downstream systems
/// (meshing, simulation) to skip unchanged chunks.
#[derive(Component)]
pub struct Chunk {
    /// The chunk's position in chunk-space coordinates.
    pub coord: ChunkCoord,
    /// Flat array of voxels, length CHUNK_VOLUME.
    voxels: Vec<Voxel>,
    /// Set to true when voxel data changes; consumers clear it after processing.
    dirty: bool,
}

impl Chunk {
    /// Create a new chunk filled entirely with the given material.
    pub fn new_filled(coord: ChunkCoord, material: MaterialId) -> Self {
        Self {
            coord,
            voxels: vec![Voxel::new(material); CHUNK_VOLUME],
            dirty: true,
        }
    }

    /// Create a new chunk filled with air.
    pub fn new_empty(coord: ChunkCoord) -> Self {
        Self::new_filled(coord, MaterialId::AIR)
    }

    /// Convert (x, y, z) local coordinates to a flat index.
    /// Panics if any coordinate is out of range.
    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
        z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
    }

    /// Get a voxel at local coordinates.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> &Voxel {
        &self.voxels[Self::index(x, y, z)]
    }

    /// Get a mutable reference to a voxel and mark the chunk dirty.
    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut Voxel {
        self.dirty = true;
        &mut self.voxels[Self::index(x, y, z)]
    }

    /// Set a voxel at local coordinates.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        self.dirty = true;
        self.voxels[Self::index(x, y, z)] = voxel;
    }

    /// Set only the material at a position, preserving other voxel state.
    #[inline]
    pub fn set_material(&mut self, x: usize, y: usize, z: usize, material: MaterialId) {
        self.dirty = true;
        self.voxels[Self::index(x, y, z)].material = material;
    }

    /// Whether the chunk has been modified since the dirty flag was last cleared.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag. Called by downstream systems after processing.
    #[inline]
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Read-only access to the raw voxel slice. Useful for meshing and simulation
    /// passes that need bulk access.
    pub fn voxels(&self) -> &[Voxel] {
        &self.voxels
    }

    /// Mutable access to the raw voxel slice. Marks chunk as dirty.
    pub fn voxels_mut(&mut self) -> &mut [Voxel] {
        self.dirty = true;
        &mut self.voxels
    }

    /// Count non-air voxels.
    pub fn solid_count(&self) -> usize {
        self.voxels.iter().filter(|v| v.is_solid()).count()
    }

    /// Check if the chunk is completely empty (all air).
    pub fn is_empty(&self) -> bool {
        self.voxels.iter().all(|v| v.is_air())
    }

    /// Check if the chunk is completely filled (no air).
    pub fn is_full(&self) -> bool {
        self.voxels.iter().all(|v| v.is_solid())
    }

    /// Fill the entire chunk with a material.
    pub fn fill(&mut self, material: MaterialId) {
        self.dirty = true;
        for v in &mut self.voxels {
            v.material = material;
        }
    }

    /// Fill a column (all Y values at given X, Z) with a material up to a height.
    pub fn fill_column(&mut self, x: usize, z: usize, height: usize, material: MaterialId) {
        let h = height.min(CHUNK_SIZE);
        for y in 0..h {
            self.set_material(x, y, z, material);
        }
    }

    // --- Octree integration ---

    /// Build a sparse voxel octree from the current flat voxel data.
    ///
    /// Uniform regions (all-air, all-stone) collapse to single leaf nodes,
    /// saving memory. The returned octree is a snapshot — modifications to
    /// the chunk afterward are not reflected.
    pub fn to_octree(&self) -> OctreeNode<Voxel> {
        flat_to_octree(&self.voxels, CHUNK_SIZE)
    }

    /// Replace the chunk's voxel data from an octree, expanding it to the
    /// flat array representation. Marks the chunk as dirty.
    pub fn load_from_octree(&mut self, octree: &OctreeNode<Voxel>) {
        self.voxels = octree_to_flat(octree, CHUNK_SIZE);
        self.dirty = true;
    }

    /// Clone the flat voxel array. Useful for passing to physics systems
    /// that expect `Vec<Voxel>` without borrowing the chunk.
    pub fn flat_snapshot(&self) -> Vec<Voxel> {
        self.voxels.clone()
    }
}

impl VoxelAccess for Chunk {
    fn get_voxel(&self, x: usize, y: usize, z: usize) -> &Voxel {
        self.get(x, y, z)
    }

    fn get_voxel_at_depth(&self, x: usize, y: usize, z: usize, depth: u8) -> &Voxel {
        let scale = 1usize << depth;
        self.get(x / scale, y / scale, z / scale)
    }

    fn set_voxel(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        self.set(x, y, z, voxel);
    }

    fn set_voxel_at_depth(&mut self, x: usize, y: usize, z: usize, depth: u8, voxel: Voxel) {
        let scale = 1usize << depth;
        self.set(x / scale, y / scale, z / scale, voxel);
    }

    fn depth_at(&self, _x: usize, _y: usize, _z: usize) -> u8 {
        0 // Flat storage is always base resolution.
    }

    fn size(&self) -> usize {
        CHUNK_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_constants() {
        assert_eq!(CHUNK_SIZE, 32);
        assert_eq!(CHUNK_VOLUME, 32 * 32 * 32);
        assert_eq!(CHUNK_VOLUME, 32_768);
    }

    #[test]
    fn new_empty_chunk_is_all_air() {
        let chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        assert!(chunk.is_empty());
        assert!(!chunk.is_full());
        assert_eq!(chunk.solid_count(), 0);
        assert_eq!(chunk.voxels().len(), CHUNK_VOLUME);
    }

    #[test]
    fn new_filled_chunk_is_all_solid() {
        let chunk = Chunk::new_filled(ChunkCoord::new(0, 0, 0), MaterialId::STONE);
        assert!(!chunk.is_empty());
        assert!(chunk.is_full());
        assert_eq!(chunk.solid_count(), CHUNK_VOLUME);
    }

    #[test]
    fn get_and_set_voxel() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        assert!(chunk.get(5, 10, 15).is_air());

        chunk.set_material(5, 10, 15, MaterialId::STONE);
        assert_eq!(chunk.get(5, 10, 15).material, MaterialId::STONE);
        assert_eq!(chunk.solid_count(), 1);
    }

    #[test]
    fn get_mut_marks_dirty() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.clear_dirty();
        assert!(!chunk.is_dirty());

        let voxel = chunk.get_mut(0, 0, 0);
        voxel.temperature = 500.0;
        assert!(chunk.is_dirty());
        assert_eq!(chunk.get(0, 0, 0).temperature, 500.0);
    }

    #[test]
    fn dirty_flag_lifecycle() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        // New chunks start dirty (need initial meshing)
        assert!(chunk.is_dirty());

        chunk.clear_dirty();
        assert!(!chunk.is_dirty());

        // Any modification sets dirty
        chunk.set_material(0, 0, 0, MaterialId::DIRT);
        assert!(chunk.is_dirty());

        chunk.clear_dirty();
        // set() also dirties
        chunk.set(1, 1, 1, Voxel::new(MaterialId::WATER));
        assert!(chunk.is_dirty());

        chunk.clear_dirty();
        // voxels_mut() also dirties
        let _ = chunk.voxels_mut();
        assert!(chunk.is_dirty());
    }

    #[test]
    fn index_corners() {
        // Verify the indexing formula at all 8 corners of the chunk
        assert_eq!(Chunk::index(0, 0, 0), 0);
        assert_eq!(Chunk::index(CHUNK_SIZE - 1, 0, 0), CHUNK_SIZE - 1);
        assert_eq!(
            Chunk::index(0, CHUNK_SIZE - 1, 0),
            (CHUNK_SIZE - 1) * CHUNK_SIZE
        );
        assert_eq!(
            Chunk::index(0, 0, CHUNK_SIZE - 1),
            (CHUNK_SIZE - 1) * CHUNK_SIZE * CHUNK_SIZE
        );
        assert_eq!(
            Chunk::index(CHUNK_SIZE - 1, CHUNK_SIZE - 1, CHUNK_SIZE - 1),
            CHUNK_VOLUME - 1
        );
    }

    #[test]
    fn fill_replaces_all_voxels() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.clear_dirty();

        chunk.fill(MaterialId::STONE);
        assert!(chunk.is_dirty());
        assert!(chunk.is_full());
        assert_eq!(chunk.solid_count(), CHUNK_VOLUME);

        chunk.fill(MaterialId::AIR);
        assert!(chunk.is_empty());
    }

    #[test]
    fn fill_column_respects_height() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));

        chunk.fill_column(5, 5, 10, MaterialId::DIRT);

        // Voxels below height should be dirt
        for y in 0..10 {
            assert_eq!(chunk.get(5, y, 5).material, MaterialId::DIRT);
        }
        // Voxels at and above height should still be air
        for y in 10..CHUNK_SIZE {
            assert!(chunk.get(5, y, 5).is_air());
        }
    }

    #[test]
    fn fill_column_clamps_to_chunk_size() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        // Height exceeding CHUNK_SIZE should be clamped
        chunk.fill_column(0, 0, CHUNK_SIZE + 100, MaterialId::STONE);

        for y in 0..CHUNK_SIZE {
            assert_eq!(chunk.get(0, y, 0).material, MaterialId::STONE);
        }
    }

    #[test]
    fn chunk_coord_from_voxel_pos() {
        // Positive positions
        assert_eq!(
            ChunkCoord::from_voxel_pos(0, 0, 0),
            ChunkCoord::new(0, 0, 0)
        );
        assert_eq!(
            ChunkCoord::from_voxel_pos(31, 31, 31),
            ChunkCoord::new(0, 0, 0)
        );
        assert_eq!(
            ChunkCoord::from_voxel_pos(32, 0, 0),
            ChunkCoord::new(1, 0, 0)
        );
        assert_eq!(
            ChunkCoord::from_voxel_pos(64, 64, 64),
            ChunkCoord::new(2, 2, 2)
        );

        // Negative positions use Euclidean division (rounds toward -∞)
        assert_eq!(
            ChunkCoord::from_voxel_pos(-1, 0, 0),
            ChunkCoord::new(-1, 0, 0)
        );
        assert_eq!(
            ChunkCoord::from_voxel_pos(-32, 0, 0),
            ChunkCoord::new(-1, 0, 0)
        );
        assert_eq!(
            ChunkCoord::from_voxel_pos(-33, 0, 0),
            ChunkCoord::new(-2, 0, 0)
        );
    }

    #[test]
    fn chunk_coord_world_origin() {
        assert_eq!(ChunkCoord::new(0, 0, 0).world_origin(), IVec3::new(0, 0, 0));
        assert_eq!(
            ChunkCoord::new(1, 0, 0).world_origin(),
            IVec3::new(32, 0, 0)
        );
        assert_eq!(
            ChunkCoord::new(-1, -1, -1).world_origin(),
            IVec3::new(-32, -32, -32)
        );
        assert_eq!(
            ChunkCoord::new(2, 3, 4).world_origin(),
            IVec3::new(64, 96, 128)
        );
    }

    #[test]
    fn chunk_coord_world_center() {
        let center = ChunkCoord::new(0, 0, 0).world_center();
        assert_eq!(center, Vec3::new(16.0, 16.0, 16.0));

        let center = ChunkCoord::new(1, 0, 0).world_center();
        assert_eq!(center, Vec3::new(48.0, 16.0, 16.0));
    }

    #[test]
    fn chunk_memory_is_reasonable() {
        // A chunk should be roughly CHUNK_VOLUME * sizeof(Voxel) + overhead
        let voxel_size = std::mem::size_of::<Voxel>();
        let expected_data = CHUNK_VOLUME * voxel_size;
        // 32768 * 16 bytes = 512 KiB per chunk — verify it's within budget (≤ 1 MiB)
        assert!(
            expected_data <= 1024 * 1024,
            "Chunk voxel data is {} bytes ({} KiB), expected ≤ 1024 KiB",
            expected_data,
            expected_data / 1024
        );
    }

    #[test]
    fn set_full_voxel_preserves_state() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        let v = Voxel {
            material: MaterialId::WATER,
            temperature: 350.0,
            pressure: 2.5,
            damage: 0.3,
            latent_heat_buffer: 0.0,
        };
        chunk.set(7, 7, 7, v);

        let stored = chunk.get(7, 7, 7);
        assert_eq!(stored.material, MaterialId::WATER);
        assert_eq!(stored.temperature, 350.0);
        assert_eq!(stored.pressure, 2.5);
        assert_eq!(stored.damage, 0.3);
    }

    // --- Octree integration ---

    #[test]
    fn to_octree_empty_chunk_is_leaf() {
        let chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        let octree = chunk.to_octree();
        assert!(octree.is_leaf());
        assert!(octree.leaf_value().unwrap().is_air());
    }

    #[test]
    fn to_octree_filled_chunk_is_leaf() {
        let chunk = Chunk::new_filled(ChunkCoord::new(0, 0, 0), MaterialId::STONE);
        let octree = chunk.to_octree();
        assert!(octree.is_leaf());
        assert_eq!(octree.leaf_value().unwrap().material, MaterialId::STONE);
    }

    #[test]
    fn to_octree_mixed_chunk_is_branch() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.set_material(0, 0, 0, MaterialId::STONE);
        let octree = chunk.to_octree();
        assert!(octree.is_branch());
    }

    #[test]
    fn octree_roundtrip_preserves_data() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        // Create a terrain pattern.
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.fill_column(x, z, 16, MaterialId::STONE);
            }
        }
        chunk.set_material(5, 20, 5, MaterialId::WATER);

        let octree = chunk.to_octree();
        let mut restored = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        restored.load_from_octree(&octree);

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert_eq!(
                        chunk.get(x, y, z),
                        restored.get(x, y, z),
                        "Mismatch at ({x}, {y}, {z})"
                    );
                }
            }
        }
    }

    #[test]
    fn load_from_octree_marks_dirty() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.clear_dirty();
        let octree = OctreeNode::new_leaf(Voxel::new(MaterialId::STONE));
        chunk.load_from_octree(&octree);
        assert!(chunk.is_dirty());
    }

    #[test]
    fn flat_snapshot_is_independent_copy() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        chunk.set_material(0, 0, 0, MaterialId::STONE);
        let snap = chunk.flat_snapshot();

        // Modify chunk after snapshot.
        chunk.set_material(0, 0, 0, MaterialId::AIR);

        // Snapshot should still have stone.
        assert_eq!(snap[0].material, MaterialId::STONE);
        assert!(chunk.get(0, 0, 0).is_air());
    }

    #[test]
    fn octree_compresses_uniform_regions() {
        let chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        let octree = chunk.to_octree();
        // Uniform chunk should collapse to 1 node.
        assert_eq!(octree.node_count(), 1);
        assert!(octree.leaf_count() < CHUNK_VOLUME);
    }

    // --- VoxelAccess trait ---

    #[test]
    fn chunk_voxel_access_get_and_set() {
        let mut chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        assert!(chunk.get_voxel(0, 0, 0).is_air());

        chunk.set_voxel(5, 10, 15, Voxel::new(MaterialId::STONE));
        assert!(chunk.get_voxel(5, 10, 15).is_solid());
        assert_eq!(chunk.size(), CHUNK_SIZE);
    }

    #[test]
    fn chunk_voxel_access_depth_always_zero() {
        let chunk = Chunk::new_empty(ChunkCoord::new(0, 0, 0));
        assert_eq!(chunk.depth_at(0, 0, 0), 0);
        assert_eq!(chunk.depth_at(31, 31, 31), 0);
    }

    use super::super::octree::OctreeNode;
}
