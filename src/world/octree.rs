// Sparse Voxel Octree (SVO) data structure.
//
// A generic octree where each node is either a uniform Leaf holding a single
// value for the entire region, or a Branch subdivided into 8 children (2×2×2).
// This enables adaptive resolution: uniform regions collapse to a single node
// while detailed regions subdivide to the configured maximum depth.
//
// Depth 0 is the root (coarsest). Each depth level halves the cell size:
//   depth 0 = full extent, depth 1 = half, depth 2 = quarter, ...
//
// Child indices follow Morton (Z-order) layout:
//   index = (z_bit << 2) | (y_bit << 1) | x_bit
//   where each bit indicates the high (1) or low (0) half along that axis.

use serde::{Deserialize, Serialize};

/// A sparse octree node, generic over the stored value type.
///
/// `T` must be `Clone + PartialEq` so that branches can be collapsed back into
/// leaves when all children become identical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OctreeNode<T: Clone + PartialEq> {
    /// A uniform region: every cell in this subtree holds the same value.
    Leaf(T),
    /// A subdivided region with exactly 8 children.
    Branch(Box<[OctreeNode<T>; 8]>),
}

impl<T: Clone + PartialEq> OctreeNode<T> {
    /// Create a uniform leaf node.
    pub fn new_leaf(value: T) -> Self {
        OctreeNode::Leaf(value)
    }

    /// Create a branch where all 8 children start as identical leaves.
    pub fn new_uniform_branch(value: T) -> Self {
        OctreeNode::Branch(Box::new(std::array::from_fn(|_| {
            OctreeNode::Leaf(value.clone())
        })))
    }

    /// Returns true if this node is a leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, OctreeNode::Leaf(_))
    }

    /// Returns true if this node is a branch.
    pub fn is_branch(&self) -> bool {
        matches!(self, OctreeNode::Branch(_))
    }

    /// Returns the leaf value if this is a leaf node.
    pub fn leaf_value(&self) -> Option<&T> {
        match self {
            OctreeNode::Leaf(v) => Some(v),
            OctreeNode::Branch(_) => None,
        }
    }

    /// Compute the child index for a position within a node of the given `size`.
    ///
    /// `x`, `y`, `z` are local coordinates within the current node's extent.
    /// Returns (child_index, local_x, local_y, local_z) where the local
    /// coordinates are relative to the selected child.
    #[inline]
    fn child_index(x: usize, y: usize, z: usize, half: usize) -> (usize, usize, usize, usize) {
        let xi = if x >= half { 1 } else { 0 };
        let yi = if y >= half { 1 } else { 0 };
        let zi = if z >= half { 1 } else { 0 };
        let idx = (zi << 2) | (yi << 1) | xi;
        (idx, x - xi * half, y - yi * half, z - zi * half)
    }

    /// Get the value at position `(x, y, z)` within a region of `size` cells
    /// per edge. Traverses up to `max_depth` levels.
    ///
    /// If a leaf is reached before `max_depth`, its value is returned (the
    /// region is uniform at that scale). Coordinates must be in `0..size`.
    pub fn get(&self, x: usize, y: usize, z: usize, size: usize) -> &T {
        debug_assert!(
            x < size && y < size && z < size,
            "OctreeNode::get out of bounds"
        );
        match self {
            OctreeNode::Leaf(v) => v,
            OctreeNode::Branch(children) => {
                let half = size / 2;
                if half == 0 {
                    // Reached minimum resolution — should not happen with correct max_depth,
                    // but guard against it.
                    panic!("OctreeNode::get reached zero-size subdivision");
                }
                let (idx, lx, ly, lz) = Self::child_index(x, y, z, half);
                children[idx].get(lx, ly, lz, half)
            }
        }
    }

    /// Set the value at position `(x, y, z)` within a region of `size` cells.
    ///
    /// Subdivides leaf nodes as needed to reach per-cell granularity, then sets
    /// the value. After setting, attempts to collapse the parent if all siblings
    /// became identical. `target_size` is the edge length of the cell at the
    /// desired resolution (1 for per-cell, 2 for 2×2×2 blocks, etc.).
    pub fn set(&mut self, x: usize, y: usize, z: usize, size: usize, target_size: usize, value: T) {
        debug_assert!(
            x < size && y < size && z < size,
            "OctreeNode::set out of bounds"
        );
        debug_assert!(target_size > 0 && target_size <= size);

        if size == target_size {
            *self = OctreeNode::Leaf(value);
            return;
        }

        // Subdivide leaf if necessary.
        if let OctreeNode::Leaf(v) = self {
            *self = OctreeNode::new_uniform_branch(v.clone());
        }

        if let OctreeNode::Branch(children) = self {
            let half = size / 2;
            let (idx, lx, ly, lz) = Self::child_index(x, y, z, half);
            children[idx].set(lx, ly, lz, half, target_size, value);

            // Try to collapse: if all children are identical leaves, merge.
            self.try_collapse();
        }
    }

    /// If this is a branch where all 8 children are identical leaves,
    /// collapse it into a single leaf. Returns true if collapsed.
    pub fn try_collapse(&mut self) -> bool {
        let should_collapse = if let OctreeNode::Branch(children) = self {
            if let Some(first_val) = children[0].leaf_value() {
                children[1..]
                    .iter()
                    .all(|c| c.leaf_value() == Some(first_val))
            } else {
                false
            }
        } else {
            false
        };

        if should_collapse {
            if let OctreeNode::Branch(children) = self {
                if let OctreeNode::Leaf(v) = &children[0] {
                    *self = OctreeNode::Leaf(v.clone());
                    return true;
                }
            }
        }
        false
    }

    /// Subdivide this node if it is a leaf. Each child becomes a leaf with the
    /// same value. No-op if already a branch.
    pub fn subdivide(&mut self) {
        if let OctreeNode::Leaf(v) = self {
            *self = OctreeNode::new_uniform_branch(v.clone());
        }
    }

    /// Recursively collapse all branches where every child is an identical leaf.
    /// Returns the total number of nodes collapsed.
    pub fn collapse_recursive(&mut self) -> usize {
        let mut count = 0;
        if let OctreeNode::Branch(children) = self {
            for child in children.iter_mut() {
                count += child.collapse_recursive();
            }
            if self.try_collapse() {
                count += 1;
            }
        }
        count
    }

    /// Count the total number of nodes in the tree (leaves + branches).
    pub fn node_count(&self) -> usize {
        match self {
            OctreeNode::Leaf(_) => 1,
            OctreeNode::Branch(children) => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
        }
    }

    /// Count only the leaf nodes.
    pub fn leaf_count(&self) -> usize {
        match self {
            OctreeNode::Leaf(_) => 1,
            OctreeNode::Branch(children) => children.iter().map(|c| c.leaf_count()).sum::<usize>(),
        }
    }

    /// Approximate memory usage in bytes (not including allocator overhead).
    pub fn memory_bytes(&self) -> usize {
        let node_size = std::mem::size_of::<Self>();
        match self {
            OctreeNode::Leaf(_) => node_size,
            OctreeNode::Branch(children) => {
                // The Box<[OctreeNode<T>; 8]> allocation
                node_size + children.iter().map(|c| c.memory_bytes()).sum::<usize>()
            }
        }
    }

    /// Maximum depth of the tree (0 = leaf only).
    pub fn max_depth(&self) -> usize {
        match self {
            OctreeNode::Leaf(_) => 0,
            OctreeNode::Branch(children) => {
                1 + children.iter().map(|c| c.max_depth()).max().unwrap_or(0)
            }
        }
    }

    /// Visit every leaf node with its position and size.
    ///
    /// The callback receives `(x, y, z, size, &T)` where `(x, y, z)` is the
    /// minimum corner in the coordinate space of the root node whose edge
    /// length is `root_size`.
    pub fn for_each_leaf<F>(&self, x: usize, y: usize, z: usize, size: usize, f: &mut F)
    where
        F: FnMut(usize, usize, usize, usize, &T),
    {
        match self {
            OctreeNode::Leaf(v) => f(x, y, z, size, v),
            OctreeNode::Branch(children) => {
                let half = size / 2;
                for (i, child) in children.iter().enumerate() {
                    let cx = x + (i & 1) * half;
                    let cy = y + ((i >> 1) & 1) * half;
                    let cz = z + ((i >> 2) & 1) * half;
                    child.for_each_leaf(cx, cy, cz, half, f);
                }
            }
        }
    }

    /// Visit every leaf node mutably.
    pub fn for_each_leaf_mut<F>(&mut self, x: usize, y: usize, z: usize, size: usize, f: &mut F)
    where
        F: FnMut(usize, usize, usize, usize, &mut T),
    {
        match self {
            OctreeNode::Leaf(v) => f(x, y, z, size, v),
            OctreeNode::Branch(children) => {
                let half = size / 2;
                for (i, child) in children.iter_mut().enumerate() {
                    let cx = x + (i & 1) * half;
                    let cy = y + ((i >> 1) & 1) * half;
                    let cz = z + ((i >> 2) & 1) * half;
                    child.for_each_leaf_mut(cx, cy, cz, half, f);
                }
            }
        }
    }
}

impl<T: Clone + PartialEq + Default> OctreeNode<T> {
    /// Create a leaf filled with the default value.
    pub fn new_default() -> Self {
        OctreeNode::Leaf(T::default())
    }
}

impl<T: Clone + PartialEq> PartialEq for OctreeNode<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (OctreeNode::Leaf(a), OctreeNode::Leaf(b)) => a == b,
            (OctreeNode::Branch(a), OctreeNode::Branch(b)) => a == b,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test type
    type Tree = OctreeNode<u8>;

    // --- Construction ---

    #[test]
    fn leaf_creation() {
        let node = Tree::new_leaf(42);
        assert!(node.is_leaf());
        assert!(!node.is_branch());
        assert_eq!(node.leaf_value(), Some(&42));
    }

    #[test]
    fn uniform_branch_creation() {
        let node = Tree::new_uniform_branch(7);
        assert!(node.is_branch());
        assert_eq!(node.leaf_value(), None);
        if let OctreeNode::Branch(children) = &node {
            for child in children.iter() {
                assert_eq!(child.leaf_value(), Some(&7));
            }
        }
    }

    #[test]
    fn default_creation() {
        let node = OctreeNode::<u8>::new_default();
        assert_eq!(node.leaf_value(), Some(&0));
    }

    // --- Child index computation ---

    #[test]
    fn child_index_corners_size_2() {
        // Size 2 → half = 1
        assert_eq!(Tree::child_index(0, 0, 0, 1), (0, 0, 0, 0)); // (0,0,0) → child 0
        assert_eq!(Tree::child_index(1, 0, 0, 1), (1, 0, 0, 0)); // (1,0,0) → child 1
        assert_eq!(Tree::child_index(0, 1, 0, 1), (2, 0, 0, 0)); // (0,1,0) → child 2
        assert_eq!(Tree::child_index(1, 1, 0, 1), (3, 0, 0, 0)); // (1,1,0) → child 3
        assert_eq!(Tree::child_index(0, 0, 1, 1), (4, 0, 0, 0)); // (0,0,1) → child 4
        assert_eq!(Tree::child_index(1, 0, 1, 1), (5, 0, 0, 0)); // (1,0,1) → child 5
        assert_eq!(Tree::child_index(0, 1, 1, 1), (6, 0, 0, 0)); // (0,1,1) → child 6
        assert_eq!(Tree::child_index(1, 1, 1, 1), (7, 0, 0, 0)); // (1,1,1) → child 7
    }

    #[test]
    fn child_index_preserves_local_coords() {
        // Size 8 → half = 4. Point (5, 2, 7) → child (1,0,1) = 5, local (1, 2, 3)
        let (idx, lx, ly, lz) = Tree::child_index(5, 2, 7, 4);
        assert_eq!(idx, 5);
        assert_eq!((lx, ly, lz), (1, 2, 3));
    }

    // --- Get ---

    #[test]
    fn get_from_leaf_returns_value() {
        let node = Tree::new_leaf(99);
        // Any valid coordinate in a leaf returns the same value.
        assert_eq!(*node.get(0, 0, 0, 4), 99);
        assert_eq!(*node.get(3, 3, 3, 4), 99);
    }

    #[test]
    fn get_from_branch_traverses() {
        let mut node = Tree::new_uniform_branch(0);
        // Manually set child 7 (1,1,1) to a different leaf.
        if let OctreeNode::Branch(children) = &mut node {
            children[7] = OctreeNode::Leaf(42);
        }
        // Size 2: position (1,1,1) maps to child 7.
        assert_eq!(*node.get(1, 1, 1, 2), 42);
        assert_eq!(*node.get(0, 0, 0, 2), 0);
    }

    // --- Set ---

    #[test]
    fn set_on_leaf_subdivides_and_stores() {
        let mut node = Tree::new_leaf(0);
        node.set(3, 3, 3, 4, 1, 99);

        // The modified cell should have the new value.
        assert_eq!(*node.get(3, 3, 3, 4), 99);
        // Other cells should retain the original value.
        assert_eq!(*node.get(0, 0, 0, 4), 0);
        assert_eq!(*node.get(2, 2, 2, 4), 0);

        // Tree should have subdivided.
        assert!(node.is_branch());
    }

    #[test]
    fn set_collapses_when_all_same() {
        let mut node = Tree::new_leaf(0);
        // Set one cell to 1, then set it back to 0.
        node.set(0, 0, 0, 2, 1, 1);
        assert!(node.is_branch());

        node.set(0, 0, 0, 2, 1, 0);
        // After setting back, all children are 0 — should collapse to leaf.
        assert!(node.is_leaf());
        assert_eq!(node.leaf_value(), Some(&0));
    }

    #[test]
    fn set_at_target_size_larger_than_1() {
        let mut node = Tree::new_leaf(0);
        // Size 8, target_size 4: sets a 4×4×4 quadrant.
        node.set(0, 0, 0, 8, 4, 5);

        assert!(node.is_branch());
        // The (0,0,0) quadrant should be 5.
        assert_eq!(*node.get(0, 0, 0, 8), 5);
        assert_eq!(*node.get(3, 3, 3, 8), 5);
        // The (4,0,0) quadrant should still be 0.
        assert_eq!(*node.get(4, 0, 0, 8), 0);
    }

    #[test]
    fn set_entire_region_to_same_value_collapses() {
        let mut node = Tree::new_leaf(0);
        // Set the entire 2×2×2 to 5, cell by cell.
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    node.set(x, y, z, 2, 1, 5);
                }
            }
        }
        // Should have collapsed back to a leaf.
        assert!(node.is_leaf());
        assert_eq!(node.leaf_value(), Some(&5));
    }

    // --- Subdivide / Collapse ---

    #[test]
    fn subdivide_leaf_becomes_branch() {
        let mut node = Tree::new_leaf(10);
        node.subdivide();
        assert!(node.is_branch());
        // All children should be leaves with the original value.
        assert_eq!(*node.get(0, 0, 0, 2), 10);
        assert_eq!(*node.get(1, 1, 1, 2), 10);
    }

    #[test]
    fn subdivide_branch_is_noop() {
        let mut node = Tree::new_uniform_branch(10);
        let count_before = node.node_count();
        node.subdivide();
        assert_eq!(node.node_count(), count_before);
    }

    #[test]
    fn try_collapse_uniform_branch() {
        let mut node = Tree::new_uniform_branch(7);
        assert!(node.try_collapse());
        assert!(node.is_leaf());
        assert_eq!(node.leaf_value(), Some(&7));
    }

    #[test]
    fn try_collapse_non_uniform_branch() {
        let mut node = Tree::new_uniform_branch(7);
        if let OctreeNode::Branch(children) = &mut node {
            children[3] = OctreeNode::Leaf(8);
        }
        assert!(!node.try_collapse());
        assert!(node.is_branch());
    }

    #[test]
    fn collapse_recursive_multi_level() {
        // Build a 3-level tree where all leaves are ultimately the same.
        let mut node = Tree::new_uniform_branch(1);
        if let OctreeNode::Branch(children) = &mut node {
            // Subdivide child 0 into another branch of 1s.
            children[0] = Tree::new_uniform_branch(1);
        }
        // Before collapse: node is branch > (branch of leaves, 7 leaves).
        assert_eq!(node.max_depth(), 2);

        let collapsed = node.collapse_recursive();
        assert!(collapsed >= 2); // At least 2 branches collapsed.
        assert!(node.is_leaf());
        assert_eq!(node.leaf_value(), Some(&1));
    }

    // --- Statistics ---

    #[test]
    fn node_count_leaf() {
        let node = Tree::new_leaf(0);
        assert_eq!(node.node_count(), 1);
        assert_eq!(node.leaf_count(), 1);
    }

    #[test]
    fn node_count_one_branch() {
        let node = Tree::new_uniform_branch(0);
        assert_eq!(node.node_count(), 9); // 1 branch + 8 leaves
        assert_eq!(node.leaf_count(), 8);
    }

    #[test]
    fn max_depth_leaf() {
        assert_eq!(Tree::new_leaf(0).max_depth(), 0);
    }

    #[test]
    fn max_depth_one_level() {
        assert_eq!(Tree::new_uniform_branch(0).max_depth(), 1);
    }

    #[test]
    fn max_depth_nested() {
        let mut node = Tree::new_uniform_branch(0);
        if let OctreeNode::Branch(children) = &mut node {
            children[0] = Tree::new_uniform_branch(0);
        }
        assert_eq!(node.max_depth(), 2);
    }

    #[test]
    fn memory_bytes_positive() {
        let node = Tree::new_leaf(0);
        assert!(node.memory_bytes() > 0);
        let branch = Tree::new_uniform_branch(0);
        assert!(branch.memory_bytes() > node.memory_bytes());
    }

    // --- Iteration ---

    #[test]
    fn for_each_leaf_visits_all_cells() {
        let mut node = Tree::new_leaf(0);
        node.set(0, 0, 0, 4, 1, 1);
        node.set(3, 3, 3, 4, 1, 2);

        let mut visited = Vec::new();
        node.for_each_leaf(0, 0, 0, 4, &mut |x, y, z, size, val| {
            visited.push((x, y, z, size, *val));
        });

        // Should have visited multiple leaves.
        assert!(!visited.is_empty());

        // Check that our two special cells are present.
        assert!(visited.iter().any(|&(x, y, z, s, v)| x == 0
            && y == 0
            && z == 0
            && s == 1
            && v == 1));
        assert!(visited.iter().any(|&(x, y, z, s, v)| x == 3
            && y == 3
            && z == 3
            && s == 1
            && v == 2));
    }

    #[test]
    fn for_each_leaf_uniform_single_visit() {
        let node = Tree::new_leaf(42);
        let mut count = 0;
        node.for_each_leaf(0, 0, 0, 8, &mut |_, _, _, size, val| {
            assert_eq!(size, 8);
            assert_eq!(*val, 42);
            count += 1;
        });
        assert_eq!(count, 1);
    }

    #[test]
    fn for_each_leaf_mut_modifies_values() {
        let mut node = Tree::new_uniform_branch(0);
        // Double-check it has 8 leaves.
        assert_eq!(node.leaf_count(), 8);

        // Set all leaves to 99.
        node.for_each_leaf_mut(0, 0, 0, 2, &mut |_, _, _, _, val| {
            *val = 99;
        });

        // Verify all are 99 now.
        node.for_each_leaf(0, 0, 0, 2, &mut |_, _, _, _, val| {
            assert_eq!(*val, 99);
        });
    }

    // --- Equality ---

    #[test]
    fn equality_leaf() {
        assert_eq!(Tree::new_leaf(1), Tree::new_leaf(1));
        assert_ne!(Tree::new_leaf(1), Tree::new_leaf(2));
    }

    #[test]
    fn equality_branch() {
        let a = Tree::new_uniform_branch(5);
        let b = Tree::new_uniform_branch(5);
        assert_eq!(a, b);

        let c = Tree::new_uniform_branch(6);
        assert_ne!(a, c);
    }

    #[test]
    fn equality_leaf_vs_branch() {
        let leaf = Tree::new_leaf(5);
        let branch = Tree::new_uniform_branch(5);
        assert_ne!(leaf, branch);
    }

    // --- Edge cases ---

    #[test]
    fn set_and_get_size_1() {
        let mut node = Tree::new_leaf(0);
        node.set(0, 0, 0, 1, 1, 42);
        assert_eq!(*node.get(0, 0, 0, 1), 42);
        assert!(node.is_leaf());
    }

    #[test]
    fn deep_subdivision_and_retrieval() {
        // Simulate a 16×16×16 tree, set one cell at maximum depth.
        let mut node = Tree::new_leaf(0u8);
        node.set(15, 15, 15, 16, 1, 255);

        assert_eq!(*node.get(15, 15, 15, 16), 255);
        assert_eq!(*node.get(0, 0, 0, 16), 0);
        assert_eq!(*node.get(14, 14, 14, 16), 0);
        assert!(node.max_depth() <= 4); // log2(16) = 4
    }

    #[test]
    fn repeated_set_same_cell() {
        let mut node = Tree::new_leaf(0);
        for i in 0..100 {
            node.set(1, 1, 1, 4, 1, i);
        }
        assert_eq!(*node.get(1, 1, 1, 4), 99);
    }

    #[test]
    fn large_tree_fill_and_collapse() {
        let mut node = Tree::new_leaf(0u8);
        // Fill a 32×32×32 tree with value 1, cell by cell.
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    node.set(x, y, z, 32, 1, 1);
                }
            }
        }
        // All cells are 1, but tree may be deeply subdivided.
        // Recursive collapse should reduce it back to a single leaf.
        node.collapse_recursive();
        assert!(node.is_leaf());
        assert_eq!(node.leaf_value(), Some(&1));
    }

    // --- Voxel-type compatibility ---

    #[test]
    fn works_with_struct_values() {
        #[derive(Debug, Clone, PartialEq)]
        struct TestVoxel {
            material: u16,
            temperature: f32,
        }

        let default = TestVoxel {
            material: 0,
            temperature: 288.15,
        };
        let stone = TestVoxel {
            material: 1,
            temperature: 288.15,
        };

        let mut tree = OctreeNode::new_leaf(default.clone());
        tree.set(0, 0, 0, 4, 1, stone.clone());

        assert_eq!(*tree.get(0, 0, 0, 4), stone);
        assert_eq!(*tree.get(3, 3, 3, 4), default);
    }
}
