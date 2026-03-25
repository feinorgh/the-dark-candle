// Voxel access abstraction layer.
//
// Provides a uniform interface for reading and writing voxels regardless of
// the underlying storage format (flat array or octree). Physics and simulation
// systems use this trait so they work transparently with both representations.

use super::octree::OctreeNode;
use super::voxel::Voxel;

/// Uniform read/write interface over voxel storage.
///
/// Coordinates `(x, y, z)` are local to the container (chunk). Implementations
/// must handle their own bounds checking.
pub trait VoxelAccess {
    /// Get a reference to the voxel at `(x, y, z)`.
    fn get_voxel(&self, x: usize, y: usize, z: usize) -> &Voxel;

    /// Get the voxel at a specific subdivision depth. Depth 0 is the base
    /// resolution (1 cell = 1 base-voxel). Higher depths address sub-voxel
    /// cells within that base voxel.
    ///
    /// `(x, y, z)` are coordinates at the given depth's resolution.
    /// For a chunk of CHUNK_SIZE=32 at depth 0, valid range is 0..32.
    /// At depth 1, valid range is 0..64 (each base voxel splits into 2³).
    fn get_voxel_at_depth(&self, x: usize, y: usize, z: usize, depth: u8) -> &Voxel;

    /// Set the voxel at `(x, y, z)` at base resolution.
    fn set_voxel(&mut self, x: usize, y: usize, z: usize, voxel: Voxel);

    /// Set the voxel at a specific subdivision depth.
    fn set_voxel_at_depth(&mut self, x: usize, y: usize, z: usize, depth: u8, voxel: Voxel);

    /// The current subdivision depth at position `(x, y, z)`.
    /// Returns 0 for uniform base-resolution cells, higher values for
    /// subdivided regions.
    fn depth_at(&self, x: usize, y: usize, z: usize) -> u8;

    /// Edge length of the container at base resolution.
    fn size(&self) -> usize;
}

/// Adapter: flat `Vec<Voxel>` as a `VoxelAccess` (backward compatibility).
///
/// Always operates at depth 0. Sub-voxel depth requests return the base voxel.
pub struct FlatVoxelStorage<'a> {
    voxels: &'a [Voxel],
    size: usize,
}

pub struct FlatVoxelStorageMut<'a> {
    voxels: &'a mut [Voxel],
    size: usize,
}

impl<'a> FlatVoxelStorage<'a> {
    pub fn new(voxels: &'a [Voxel], size: usize) -> Self {
        debug_assert_eq!(voxels.len(), size * size * size);
        Self { voxels, size }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < self.size && y < self.size && z < self.size);
        z * self.size * self.size + y * self.size + x
    }
}

impl<'a> FlatVoxelStorageMut<'a> {
    pub fn new(voxels: &'a mut [Voxel], size: usize) -> Self {
        debug_assert_eq!(voxels.len(), size * size * size);
        Self { voxels, size }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < self.size && y < self.size && z < self.size);
        z * self.size * self.size + y * self.size + x
    }
}

impl VoxelAccess for FlatVoxelStorage<'_> {
    fn get_voxel(&self, x: usize, y: usize, z: usize) -> &Voxel {
        let idx = self.index(x, y, z);
        &self.voxels[idx]
    }

    fn get_voxel_at_depth(&self, x: usize, y: usize, z: usize, depth: u8) -> &Voxel {
        // Flat storage has no subdivision — map sub-voxel coords to base coords.
        let scale = 1usize << depth;
        self.get_voxel(x / scale, y / scale, z / scale)
    }

    fn set_voxel(&mut self, _x: usize, _y: usize, _z: usize, _voxel: Voxel) {
        // Read-only adapter: immutable flat storage cannot be written.
        panic!("FlatVoxelStorage is read-only; use FlatVoxelStorageMut for mutation");
    }

    fn set_voxel_at_depth(&mut self, _x: usize, _y: usize, _z: usize, _depth: u8, _voxel: Voxel) {
        panic!("FlatVoxelStorage is read-only; use FlatVoxelStorageMut for mutation");
    }

    fn depth_at(&self, _x: usize, _y: usize, _z: usize) -> u8 {
        0 // Always base resolution.
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl VoxelAccess for FlatVoxelStorageMut<'_> {
    fn get_voxel(&self, x: usize, y: usize, z: usize) -> &Voxel {
        let idx = self.index(x, y, z);
        &self.voxels[idx]
    }

    fn get_voxel_at_depth(&self, x: usize, y: usize, z: usize, depth: u8) -> &Voxel {
        let scale = 1usize << depth;
        self.get_voxel(x / scale, y / scale, z / scale)
    }

    fn set_voxel(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        let idx = self.index(x, y, z);
        self.voxels[idx] = voxel;
    }

    fn set_voxel_at_depth(&mut self, x: usize, y: usize, z: usize, depth: u8, voxel: Voxel) {
        let scale = 1usize << depth;
        self.set_voxel(x / scale, y / scale, z / scale, voxel);
    }

    fn depth_at(&self, _x: usize, _y: usize, _z: usize) -> u8 {
        0
    }

    fn size(&self) -> usize {
        self.size
    }
}

/// Adapter: `OctreeNode<Voxel>` as a `VoxelAccess`.
///
/// Supports full multi-resolution access. The octree root represents a volume
/// of `size³` base-resolution cells. Subdivision beyond base resolution is
/// addressed via `depth` > 0.
pub struct OctreeVoxelStorage<'a> {
    root: &'a OctreeNode<Voxel>,
    size: usize,
}

pub struct OctreeVoxelStorageMut<'a> {
    root: &'a mut OctreeNode<Voxel>,
    size: usize,
}

impl<'a> OctreeVoxelStorage<'a> {
    pub fn new(root: &'a OctreeNode<Voxel>, size: usize) -> Self {
        Self { root, size }
    }
}

impl<'a> OctreeVoxelStorageMut<'a> {
    pub fn new(root: &'a mut OctreeNode<Voxel>, size: usize) -> Self {
        Self { root, size }
    }
}

impl VoxelAccess for OctreeVoxelStorage<'_> {
    fn get_voxel(&self, x: usize, y: usize, z: usize) -> &Voxel {
        self.root.get(x, y, z, self.size)
    }

    fn get_voxel_at_depth(&self, x: usize, y: usize, z: usize, depth: u8) -> &Voxel {
        let scale = 1usize << depth;
        let full_size = self.size * scale;
        self.root.get(x, y, z, full_size)
    }

    fn set_voxel(&mut self, _x: usize, _y: usize, _z: usize, _voxel: Voxel) {
        panic!("OctreeVoxelStorage is read-only; use OctreeVoxelStorageMut for mutation");
    }

    fn set_voxel_at_depth(&mut self, _x: usize, _y: usize, _z: usize, _depth: u8, _voxel: Voxel) {
        panic!("OctreeVoxelStorage is read-only; use OctreeVoxelStorageMut for mutation");
    }

    fn depth_at(&self, x: usize, y: usize, z: usize) -> u8 {
        self.depth_at_recursive(self.root, x, y, z, self.size, 0)
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl OctreeVoxelStorage<'_> {
    fn depth_at_recursive(
        &self,
        node: &OctreeNode<Voxel>,
        x: usize,
        y: usize,
        z: usize,
        size: usize,
        current_depth: u8,
    ) -> u8 {
        match node {
            OctreeNode::Leaf(_) => current_depth,
            OctreeNode::Branch(children) => {
                let half = size / 2;
                if half == 0 {
                    return current_depth;
                }
                let xi = if x >= half { 1 } else { 0 };
                let yi = if y >= half { 1 } else { 0 };
                let zi = if z >= half { 1 } else { 0 };
                let idx = (zi << 2) | (yi << 1) | xi;
                self.depth_at_recursive(
                    &children[idx],
                    x - xi * half,
                    y - yi * half,
                    z - zi * half,
                    half,
                    current_depth + 1,
                )
            }
        }
    }
}

impl VoxelAccess for OctreeVoxelStorageMut<'_> {
    fn get_voxel(&self, x: usize, y: usize, z: usize) -> &Voxel {
        self.root.get(x, y, z, self.size)
    }

    fn get_voxel_at_depth(&self, x: usize, y: usize, z: usize, depth: u8) -> &Voxel {
        let scale = 1usize << depth;
        let full_size = self.size * scale;
        self.root.get(x, y, z, full_size)
    }

    fn set_voxel(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        self.root.set(x, y, z, self.size, 1, voxel);
    }

    fn set_voxel_at_depth(&mut self, x: usize, y: usize, z: usize, depth: u8, voxel: Voxel) {
        let scale = 1usize << depth;
        let full_size = self.size * scale;
        self.root.set(x, y, z, full_size, 1, voxel);
    }

    fn depth_at(&self, x: usize, y: usize, z: usize) -> u8 {
        depth_at_recursive_ref(self.root, x, y, z, self.size, 0)
    }

    fn size(&self) -> usize {
        self.size
    }
}

fn depth_at_recursive_ref(
    node: &OctreeNode<Voxel>,
    x: usize,
    y: usize,
    z: usize,
    size: usize,
    current_depth: u8,
) -> u8 {
    match node {
        OctreeNode::Leaf(_) => current_depth,
        OctreeNode::Branch(children) => {
            let half = size / 2;
            if half == 0 {
                return current_depth;
            }
            let xi = if x >= half { 1 } else { 0 };
            let yi = if y >= half { 1 } else { 0 };
            let zi = if z >= half { 1 } else { 0 };
            let idx = (zi << 2) | (yi << 1) | xi;
            depth_at_recursive_ref(
                &children[idx],
                x - xi * half,
                y - yi * half,
                z - zi * half,
                half,
                current_depth + 1,
            )
        }
    }
}

/// Convert an octree to a flat voxel array at base resolution.
///
/// Useful for backward compatibility with systems that still expect `&[Voxel]`.
pub fn octree_to_flat(root: &OctreeNode<Voxel>, size: usize) -> Vec<Voxel> {
    let volume = size * size * size;
    let mut flat = vec![Voxel::default(); volume];
    root.for_each_leaf(0, 0, 0, size, &mut |lx, ly, lz, leaf_size, voxel| {
        // Fill all base-resolution cells covered by this leaf.
        for dz in 0..leaf_size {
            for dy in 0..leaf_size {
                for dx in 0..leaf_size {
                    let x = lx + dx;
                    let y = ly + dy;
                    let z = lz + dz;
                    if x < size && y < size && z < size {
                        flat[z * size * size + y * size + x] = *voxel;
                    }
                }
            }
        }
    });
    flat
}

/// Convert a flat voxel array into an octree, collapsing uniform regions.
pub fn flat_to_octree(voxels: &[Voxel], size: usize) -> OctreeNode<Voxel> {
    debug_assert_eq!(voxels.len(), size * size * size);
    build_octree_recursive(voxels, 0, 0, 0, size, size)
}

fn build_octree_recursive(
    voxels: &[Voxel],
    x: usize,
    y: usize,
    z: usize,
    size: usize,
    full_size: usize,
) -> OctreeNode<Voxel> {
    if size == 1 {
        let idx = z * full_size * full_size + y * full_size + x;
        return OctreeNode::Leaf(voxels[idx]);
    }

    // Check if the entire region is uniform.
    let first_idx = z * full_size * full_size + y * full_size + x;
    let first = &voxels[first_idx];
    let mut uniform = true;
    'outer: for dz in 0..size {
        for dy in 0..size {
            for dx in 0..size {
                let idx = (z + dz) * full_size * full_size + (y + dy) * full_size + (x + dx);
                if &voxels[idx] != first {
                    uniform = false;
                    break 'outer;
                }
            }
        }
    }

    if uniform {
        return OctreeNode::Leaf(*first);
    }

    let half = size / 2;
    let children: [OctreeNode<Voxel>; 8] = std::array::from_fn(|i| {
        let cx = x + (i & 1) * half;
        let cy = y + ((i >> 1) & 1) * half;
        let cz = z + ((i >> 2) & 1) * half;
        build_octree_recursive(voxels, cx, cy, cz, half, full_size)
    });

    OctreeNode::Branch(Box::new(children))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::CHUNK_SIZE;
    use crate::world::voxel::MaterialId;

    fn air() -> Voxel {
        Voxel::default()
    }

    fn stone() -> Voxel {
        Voxel::new(MaterialId::STONE)
    }

    // --- FlatVoxelStorage ---

    #[test]
    fn flat_storage_get_and_size() {
        let voxels = vec![air(); 8]; // 2×2×2
        let store = FlatVoxelStorage::new(&voxels, 2);
        assert_eq!(store.size(), 2);
        assert!(store.get_voxel(0, 0, 0).is_air());
        assert!(store.get_voxel(1, 1, 1).is_air());
    }

    #[test]
    fn flat_storage_depth_always_zero() {
        let voxels = vec![air(); 8];
        let store = FlatVoxelStorage::new(&voxels, 2);
        assert_eq!(store.depth_at(0, 0, 0), 0);
        assert_eq!(store.depth_at(1, 1, 1), 0);
    }

    #[test]
    fn flat_storage_get_at_depth_maps_to_base() {
        let mut voxels = vec![air(); 8]; // 2×2×2
        voxels[0] = stone(); // (0,0,0) = stone
        let store = FlatVoxelStorage::new(&voxels, 2);

        // Depth 1: coords 0..4. (0,0,0) and (1,1,1) both map to base (0,0,0).
        assert!(store.get_voxel_at_depth(0, 0, 0, 1).is_solid());
        assert!(store.get_voxel_at_depth(1, 1, 1, 1).is_solid());
        // (2,0,0) maps to base (1,0,0) = air.
        assert!(store.get_voxel_at_depth(2, 0, 0, 1).is_air());
    }

    #[test]
    fn flat_storage_mut_set_and_get() {
        let mut voxels = vec![air(); 8];
        {
            let mut store = FlatVoxelStorageMut::new(&mut voxels, 2);
            store.set_voxel(1, 1, 1, stone());
            assert!(store.get_voxel(1, 1, 1).is_solid());
        }
        // Verify the underlying array was modified.
        let idx = 4 + 2 + 1; // z*4 + y*2 + x
        assert!(voxels[idx].is_solid());
    }

    // --- OctreeVoxelStorage ---

    #[test]
    fn octree_storage_get_uniform() {
        let tree = OctreeNode::new_leaf(air());
        let store = OctreeVoxelStorage::new(&tree, CHUNK_SIZE);
        assert!(store.get_voxel(0, 0, 0).is_air());
        assert!(store.get_voxel(31, 31, 31).is_air());
        assert_eq!(store.size(), CHUNK_SIZE);
    }

    #[test]
    fn octree_storage_depth_leaf_is_zero() {
        let tree = OctreeNode::new_leaf(air());
        let store = OctreeVoxelStorage::new(&tree, 4);
        assert_eq!(store.depth_at(0, 0, 0), 0);
    }

    #[test]
    fn octree_storage_depth_subdivided() {
        let mut tree = OctreeNode::new_leaf(air());
        tree.set(0, 0, 0, 4, 1, stone());
        let store = OctreeVoxelStorage::new(&tree, 4);

        // The cell at (0,0,0) was set at resolution 1 within a size-4 tree,
        // so it required 2 levels of subdivision (4→2→1).
        assert!(store.depth_at(0, 0, 0) >= 2);
        // Uniform regions may have lower depth.
        assert!(store.depth_at(3, 3, 3) <= 2);
    }

    #[test]
    fn octree_storage_mut_set_and_get() {
        let mut tree = OctreeNode::new_leaf(air());
        {
            let mut store = OctreeVoxelStorageMut::new(&mut tree, 4);
            store.set_voxel(2, 3, 1, stone());
        }
        assert!(tree.get(2, 3, 1, 4).is_solid());
        assert!(tree.get(0, 0, 0, 4).is_air());
    }

    #[test]
    fn octree_storage_mut_set_at_depth() {
        let mut tree = OctreeNode::new_leaf(air());
        {
            let mut store = OctreeVoxelStorageMut::new(&mut tree, 4);
            // Depth 1: size becomes 8 (4 * 2^1). Set cell (7,7,7) at depth 1.
            store.set_voxel_at_depth(7, 7, 7, 1, stone());
        }
        // At the full resolution of 8, cell (7,7,7) should be stone.
        assert!(tree.get(7, 7, 7, 8).is_solid());
    }

    // --- Flat ↔ Octree conversion ---

    #[test]
    fn flat_to_octree_uniform() {
        let voxels = vec![air(); 64]; // 4×4×4
        let tree = flat_to_octree(&voxels, 4);
        assert!(tree.is_leaf());
        assert!(tree.leaf_value().unwrap().is_air());
    }

    #[test]
    fn flat_to_octree_single_different() {
        let mut voxels = vec![air(); 64]; // 4×4×4
        voxels[0] = stone(); // (0,0,0) = stone
        let tree = flat_to_octree(&voxels, 4);
        assert!(tree.is_branch());
        assert!(tree.get(0, 0, 0, 4).is_solid());
        assert!(tree.get(1, 0, 0, 4).is_air());
    }

    #[test]
    fn octree_to_flat_roundtrip() {
        let mut voxels = vec![air(); 64]; // 4×4×4
        // Set a few cells.
        voxels[0] = stone();
        voxels[63] = stone();
        voxels[32] = Voxel::new(MaterialId::WATER);

        let tree = flat_to_octree(&voxels, 4);
        let roundtrip = octree_to_flat(&tree, 4);

        assert_eq!(voxels.len(), roundtrip.len());
        for (i, (a, b)) in voxels.iter().zip(roundtrip.iter()).enumerate() {
            assert_eq!(a, b, "Mismatch at index {i}");
        }
    }

    #[test]
    fn octree_to_flat_chunk_size_roundtrip() {
        // Test with CHUNK_SIZE (32) — the real use case.
        let mut voxels = vec![air(); CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
        // Create a terrain-like pattern: solid below y=16, air above.
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    if y < 16 {
                        let idx = z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x;
                        voxels[idx] = stone();
                    }
                }
            }
        }

        let tree = flat_to_octree(&voxels, CHUNK_SIZE);
        let roundtrip = octree_to_flat(&tree, CHUNK_SIZE);

        assert_eq!(voxels.len(), roundtrip.len());
        for (i, (a, b)) in voxels.iter().zip(roundtrip.iter()).enumerate() {
            assert_eq!(a, b, "Mismatch at flat index {i}");
        }

        // The tree should be more compact than the flat array because large
        // uniform regions (all-air above, all-stone below) collapse.
        assert!(
            tree.leaf_count() < CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE,
            "Octree should be sparser than flat: {} leaves vs {} cells",
            tree.leaf_count(),
            CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
        );
    }
}
