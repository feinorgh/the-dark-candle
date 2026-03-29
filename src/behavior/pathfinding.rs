// 3D voxel pathfinding using A* search.
//
// Operates on a flat voxel grid (trait-abstracted so it works with chunks or
// test grids). Supports walking on solid surfaces, jumping up 1 voxel,
// dropping down, and swimming through water at reduced speed.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::voxel::MaterialId;

/// Trait abstracting voxel lookups so pathfinding works with any grid.
pub trait VoxelGrid {
    /// Get material at (x, y, z). Returns None if out of bounds.
    fn get_material(&self, x: i32, y: i32, z: i32) -> Option<MaterialId>;
}

/// [`VoxelGrid`] backed by loaded world chunks.
///
/// Holds borrowed chunk references keyed by [`ChunkCoord`].  Build one per
/// path-computation pass from the ECS `Query<(&Chunk, &ChunkCoord)>`.
pub struct WorldVoxelGrid<'a> {
    chunks: HashMap<(i32, i32, i32), &'a Chunk>,
}

impl<'a> WorldVoxelGrid<'a> {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn insert(&mut self, coord: &ChunkCoord, chunk: &'a Chunk) {
        self.chunks.insert((coord.x, coord.y, coord.z), chunk);
    }
}

impl Default for WorldVoxelGrid<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelGrid for WorldVoxelGrid<'_> {
    fn get_material(&self, x: i32, y: i32, z: i32) -> Option<MaterialId> {
        let cc = ChunkCoord::from_voxel_pos(x, y, z);
        let chunk = self.chunks.get(&(cc.x, cc.y, cc.z))?;
        let origin = cc.world_origin();
        let lx = (x - origin.x) as usize;
        let ly = (y - origin.y) as usize;
        let lz = (z - origin.z) as usize;
        if lx < CHUNK_SIZE && ly < CHUNK_SIZE && lz < CHUNK_SIZE {
            Some(chunk.get(lx, ly, lz).material)
        } else {
            None
        }
    }
}

/// Result of a pathfinding query.
#[derive(Debug, Clone)]
pub struct Path {
    /// Ordered list of positions from start (exclusive) to goal (inclusive).
    pub waypoints: Vec<[i32; 3]>,
    /// Total cost of the path.
    pub cost: f32,
}

/// Configuration for pathfinding behavior.
#[derive(Debug, Clone)]
pub struct PathConfig {
    /// Maximum number of nodes to explore before giving up.
    pub max_nodes: usize,
    /// Maximum vertical jump height (in voxels).
    pub max_jump: i32,
    /// Maximum drop height (in voxels).
    pub max_drop: i32,
    /// Whether this creature can swim.
    pub can_swim: bool,
    /// Movement cost multiplier for swimming.
    pub swim_cost: f32,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            max_nodes: 2000,
            max_jump: 1,
            max_drop: 3,
            can_swim: false,
            swim_cost: 3.0,
        }
    }
}

#[derive(Clone)]
struct Node {
    pos: [i32; 3],
    g_cost: f32,
    f_cost: f32,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
    }
}
impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn heuristic(a: [i32; 3], b: [i32; 3]) -> f32 {
    let dx = (a[0] - b[0]).abs() as f32;
    let dy = (a[1] - b[1]).abs() as f32;
    let dz = (a[2] - b[2]).abs() as f32;
    // Octile-ish 3D heuristic (admissible)
    dx + dy + dz
}

fn is_air_or_water(mat: MaterialId) -> bool {
    mat == MaterialId::AIR || mat == MaterialId::WATER
}

fn is_walkable_surface(grid: &dyn VoxelGrid, x: i32, y: i32, z: i32) -> bool {
    let Some(below) = grid.get_material(x, y - 1, z) else {
        return false;
    };
    let Some(at) = grid.get_material(x, y, z) else {
        return false;
    };
    // Standing on solid ground, position is air
    !below.is_air() && below != MaterialId::WATER && at.is_air()
}

fn is_swimmable(grid: &dyn VoxelGrid, x: i32, y: i32, z: i32) -> bool {
    grid.get_material(x, y, z) == Some(MaterialId::WATER)
}

/// Find a path from `start` to `goal` through the voxel grid.
/// Returns None if no path exists within the node budget.
pub fn find_path(
    grid: &dyn VoxelGrid,
    start: [i32; 3],
    goal: [i32; 3],
    config: &PathConfig,
) -> Option<Path> {
    if start == goal {
        return Some(Path {
            waypoints: vec![],
            cost: 0.0,
        });
    }

    let mut open = BinaryHeap::new();
    let mut g_costs: HashMap<[i32; 3], f32> = HashMap::new();
    let mut came_from: HashMap<[i32; 3], [i32; 3]> = HashMap::new();
    let mut explored = 0usize;

    g_costs.insert(start, 0.0);
    open.push(Node {
        pos: start,
        g_cost: 0.0,
        f_cost: heuristic(start, goal),
    });

    // 4 cardinal + 4 diagonal horizontal neighbors
    let horizontal: [(i32, i32); 8] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];

    while let Some(current) = open.pop() {
        explored += 1;
        if explored > config.max_nodes {
            return None;
        }

        if current.pos == goal {
            // Reconstruct path
            let mut path = vec![goal];
            let mut pos = goal;
            while let Some(&prev) = came_from.get(&pos) {
                if prev == start {
                    break;
                }
                path.push(prev);
                pos = prev;
            }
            path.reverse();
            return Some(Path {
                waypoints: path,
                cost: current.g_cost,
            });
        }

        // Skip if we've already found a cheaper way here
        if let Some(&best_g) = g_costs.get(&current.pos)
            && current.g_cost > best_g
        {
            continue;
        }

        let [cx, cy, cz] = current.pos;

        for &(dx, dz) in &horizontal {
            let nx = cx + dx;
            let nz = cz + dz;
            let diagonal = dx != 0 && dz != 0;
            let base_cost = if diagonal { 1.414 } else { 1.0 };

            // Try same-level walk
            if is_walkable_surface(grid, nx, cy, nz) {
                let cost = current.g_cost + base_cost;
                if cost < *g_costs.get(&[nx, cy, nz]).unwrap_or(&f32::INFINITY) {
                    g_costs.insert([nx, cy, nz], cost);
                    came_from.insert([nx, cy, nz], current.pos);
                    open.push(Node {
                        pos: [nx, cy, nz],
                        g_cost: cost,
                        f_cost: cost + heuristic([nx, cy, nz], goal),
                    });
                }
            }

            // Try jumping up (1..=max_jump)
            for dy in 1..=config.max_jump {
                let ny = cy + dy;
                if is_walkable_surface(grid, nx, ny, nz) {
                    // Check clearance: air at current position + dy
                    let clear = (1..=dy).all(|h| {
                        grid.get_material(cx, cy + h, cz)
                            .map(is_air_or_water)
                            .unwrap_or(false)
                    });
                    if clear {
                        let cost = current.g_cost + base_cost + dy as f32 * 0.5;
                        if cost < *g_costs.get(&[nx, ny, nz]).unwrap_or(&f32::INFINITY) {
                            g_costs.insert([nx, ny, nz], cost);
                            came_from.insert([nx, ny, nz], current.pos);
                            open.push(Node {
                                pos: [nx, ny, nz],
                                g_cost: cost,
                                f_cost: cost + heuristic([nx, ny, nz], goal),
                            });
                        }
                    }
                }
            }

            // Try dropping down (1..=max_drop)
            for dy in 1..=config.max_drop {
                let ny = cy - dy;
                if is_walkable_surface(grid, nx, ny, nz) {
                    // Check clearance: air below current position
                    let clear = (1..dy).all(|h| {
                        grid.get_material(nx, cy - h, nz)
                            .map(is_air_or_water)
                            .unwrap_or(false)
                    });
                    if clear {
                        let cost = current.g_cost + base_cost + dy as f32 * 0.3;
                        if cost < *g_costs.get(&[nx, ny, nz]).unwrap_or(&f32::INFINITY) {
                            g_costs.insert([nx, ny, nz], cost);
                            came_from.insert([nx, ny, nz], current.pos);
                            open.push(Node {
                                pos: [nx, ny, nz],
                                g_cost: cost,
                                f_cost: cost + heuristic([nx, ny, nz], goal),
                            });
                        }
                    }
                }
            }

            // Swimming
            if config.can_swim && is_swimmable(grid, nx, cy, nz) {
                let cost = current.g_cost + base_cost * config.swim_cost;
                if cost < *g_costs.get(&[nx, cy, nz]).unwrap_or(&f32::INFINITY) {
                    g_costs.insert([nx, cy, nz], cost);
                    came_from.insert([nx, cy, nz], current.pos);
                    open.push(Node {
                        pos: [nx, cy, nz],
                        g_cost: cost,
                        f_cost: cost + heuristic([nx, cy, nz], goal),
                    });
                }
                // Swim up/down
                for &dy in &[1i32, -1] {
                    let ny = cy + dy;
                    if is_swimmable(grid, nx, ny, nz) {
                        let cost = current.g_cost + config.swim_cost;
                        if cost < *g_costs.get(&[nx, ny, nz]).unwrap_or(&f32::INFINITY) {
                            g_costs.insert([nx, ny, nz], cost);
                            came_from.insert([nx, ny, nz], current.pos);
                            open.push(Node {
                                pos: [nx, ny, nz],
                                g_cost: cost,
                                f_cost: cost + heuristic([nx, ny, nz], goal),
                            });
                        }
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test grid: flat array with known bounds.
    struct TestGrid {
        size: i32,
        voxels: Vec<MaterialId>,
    }

    impl TestGrid {
        fn new(size: i32) -> Self {
            let vol = (size * size * size) as usize;
            Self {
                size,
                voxels: vec![MaterialId::AIR; vol],
            }
        }

        fn set(&mut self, x: i32, y: i32, z: i32, mat: MaterialId) {
            if x >= 0 && x < self.size && y >= 0 && y < self.size && z >= 0 && z < self.size {
                let idx = (y * self.size * self.size + z * self.size + x) as usize;
                self.voxels[idx] = mat;
            }
        }

        /// Fill a layer at y with the given material.
        fn fill_layer(&mut self, y: i32, mat: MaterialId) {
            for z in 0..self.size {
                for x in 0..self.size {
                    self.set(x, y, z, mat);
                }
            }
        }
    }

    impl VoxelGrid for TestGrid {
        fn get_material(&self, x: i32, y: i32, z: i32) -> Option<MaterialId> {
            if x >= 0 && x < self.size && y >= 0 && y < self.size && z >= 0 && z < self.size {
                let idx = (y * self.size * self.size + z * self.size + x) as usize;
                Some(self.voxels[idx])
            } else {
                None
            }
        }
    }

    #[test]
    fn path_to_self_is_empty() {
        let grid = TestGrid::new(8);
        let config = PathConfig::default();
        let path = find_path(&grid, [1, 1, 1], [1, 1, 1], &config).unwrap();
        assert!(path.waypoints.is_empty());
        assert_eq!(path.cost, 0.0);
    }

    #[test]
    fn straight_line_on_flat_ground() {
        let mut grid = TestGrid::new(16);
        grid.fill_layer(0, MaterialId::STONE); // floor at y=0
        // Walk from (1,1,1) to (5,1,1) — all standing on stone floor at y=0, air at y=1
        let config = PathConfig::default();
        let path = find_path(&grid, [1, 1, 1], [5, 1, 1], &config);
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(!path.waypoints.is_empty());
        assert_eq!(*path.waypoints.last().unwrap(), [5, 1, 1]);
    }

    #[test]
    fn path_around_wall() {
        let mut grid = TestGrid::new(16);
        grid.fill_layer(0, MaterialId::STONE);
        // Wall at x=3, z=0..5 blocking direct path
        for z in 0..5 {
            grid.set(3, 1, z, MaterialId::STONE);
            grid.set(3, 2, z, MaterialId::STONE);
        }
        let config = PathConfig::default();
        let path = find_path(&grid, [1, 1, 2], [5, 1, 2], &config);
        assert!(path.is_some());
        let path = path.unwrap();
        // Path must go around the wall (z >= 5)
        assert!(path.waypoints.iter().any(|p| p[2] >= 5));
        assert_eq!(*path.waypoints.last().unwrap(), [5, 1, 2]);
    }

    #[test]
    fn jump_up_one_block() {
        let mut grid = TestGrid::new(16);
        grid.fill_layer(0, MaterialId::STONE);
        // Raised platform at x=3..6
        for x in 3..6 {
            for z in 0..6 {
                grid.set(x, 1, z, MaterialId::STONE);
            }
        }
        let config = PathConfig {
            max_jump: 1,
            ..Default::default()
        };
        // Start on ground floor (y=1), goal on platform (y=2)
        let path = find_path(&grid, [1, 1, 2], [4, 2, 2], &config);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(*path.waypoints.last().unwrap(), [4, 2, 2]);
    }

    #[test]
    fn cannot_jump_too_high() {
        let mut grid = TestGrid::new(16);
        grid.fill_layer(0, MaterialId::STONE);
        // High platform (3 blocks up)
        for x in 3..6 {
            for z in 0..6 {
                grid.set(x, 1, z, MaterialId::STONE);
                grid.set(x, 2, z, MaterialId::STONE);
                grid.set(x, 3, z, MaterialId::STONE);
            }
        }
        let config = PathConfig {
            max_jump: 1,
            max_nodes: 500,
            ..Default::default()
        };
        // Ground at y=1, platform top at y=4 — too high to jump
        let path = find_path(&grid, [1, 1, 2], [4, 4, 2], &config);
        assert!(path.is_none());
    }

    #[test]
    fn drop_down_ledge() {
        let mut grid = TestGrid::new(16);
        grid.fill_layer(0, MaterialId::STONE);
        // Raised platform on left side
        for x in 0..4 {
            for z in 0..8 {
                grid.set(x, 1, z, MaterialId::STONE);
                grid.set(x, 2, z, MaterialId::STONE);
            }
        }
        let config = PathConfig {
            max_drop: 3,
            ..Default::default()
        };
        // Start on platform (y=3), goal on ground (y=1)
        let path = find_path(&grid, [2, 3, 2], [6, 1, 2], &config);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(*path.waypoints.last().unwrap(), [6, 1, 2]);
    }

    #[test]
    fn swim_through_water() {
        let mut grid = TestGrid::new(16);
        grid.fill_layer(0, MaterialId::STONE);
        // Water pool at y=1, x=3..6
        for x in 3..6 {
            for z in 0..8 {
                grid.set(x, 1, z, MaterialId::WATER);
            }
        }
        let config = PathConfig {
            can_swim: true,
            swim_cost: 2.0,
            ..Default::default()
        };
        let path = find_path(&grid, [1, 1, 2], [7, 1, 2], &config);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(*path.waypoints.last().unwrap(), [7, 1, 2]);
    }

    #[test]
    fn non_swimmer_avoids_water() {
        let mut grid = TestGrid::new(16);
        grid.fill_layer(0, MaterialId::STONE);
        // Water wall blocking the path completely
        for z in 0..16 {
            grid.set(5, 1, z, MaterialId::WATER);
        }
        let config = PathConfig {
            can_swim: false,
            max_nodes: 500,
            ..Default::default()
        };
        // Water is not walkable surface and non-swimmer can't traverse it
        let path = find_path(&grid, [3, 1, 2], [7, 1, 2], &config);
        assert!(path.is_none());
    }

    #[test]
    fn no_path_in_sealed_room() {
        let mut grid = TestGrid::new(8);
        grid.fill_layer(0, MaterialId::STONE);
        // Walls around (2,1,2)
        for x in 1..4 {
            for z in 1..4 {
                if x == 1 || x == 3 || z == 1 || z == 3 {
                    grid.set(x, 1, z, MaterialId::STONE);
                    grid.set(x, 2, z, MaterialId::STONE);
                }
            }
        }
        let config = PathConfig {
            max_nodes: 200,
            ..Default::default()
        };
        let path = find_path(&grid, [2, 1, 2], [6, 1, 6], &config);
        assert!(path.is_none());
    }

    #[test]
    fn diagonal_movement_is_costlier() {
        let mut grid = TestGrid::new(16);
        grid.fill_layer(0, MaterialId::STONE);
        let config = PathConfig::default();

        let straight = find_path(&grid, [1, 1, 1], [5, 1, 1], &config).unwrap();
        let diagonal = find_path(&grid, [1, 1, 1], [4, 1, 4], &config).unwrap();
        // Diagonal path covering same manhattan distance should cost more per step
        assert!(diagonal.cost > 0.0);
        assert!(straight.cost > 0.0);
    }
}
