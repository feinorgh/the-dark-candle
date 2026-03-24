// Core data types for the AMR Navier-Stokes fluid simulation.
//
// FluidGrid is a flat array of FluidCells with the same z*size²+y*size+x
// indexing as Chunk. Each cell tracks velocity, pressure correction, and a
// tag classifying it as solid, liquid, air, or free surface.

use crate::data::MaterialRegistry;
use crate::world::chunk::Chunk;
use crate::world::voxel::{MaterialId, Voxel};

/// Classification of a fluid cell for boundary condition handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CellTag {
    /// Empty space — participates in pressure BCs near liquid.
    #[default]
    Air,
    /// Fully submerged liquid cell.
    Liquid,
    /// Liquid cell adjacent to at least one air cell (free surface).
    Surface,
    /// Impenetrable solid — enforces no-slip or free-slip velocity BC.
    Solid,
}

/// Per-cell fluid state. Stored separately from Voxel to avoid bloating
/// the 16-byte voxel struct used by terrain, chemistry, and biology.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidCell {
    /// Velocity in m/s (x, y, z).
    pub velocity: [f32; 3],
    /// Pressure correction from the Poisson solve, in Pascals.
    pub pressure: f32,
    /// Cell classification for boundary conditions.
    pub tag: CellTag,
    /// Material ID of the fluid occupying this cell (or air/solid).
    pub material: MaterialId,
}

impl Default for FluidCell {
    fn default() -> Self {
        Self {
            velocity: [0.0; 3],
            pressure: 0.0,
            tag: CellTag::Air,
            material: MaterialId::AIR,
        }
    }
}

impl FluidCell {
    pub fn new_liquid(material: MaterialId) -> Self {
        Self {
            velocity: [0.0; 3],
            pressure: 0.0,
            tag: CellTag::Liquid,
            material,
        }
    }

    pub fn new_solid(material: MaterialId) -> Self {
        Self {
            velocity: [0.0; 3],
            pressure: 0.0,
            tag: CellTag::Solid,
            material,
        }
    }

    pub fn is_fluid(&self) -> bool {
        matches!(self.tag, CellTag::Liquid | CellTag::Surface)
    }

    pub fn speed(&self) -> f32 {
        let [vx, vy, vz] = self.velocity;
        (vx * vx + vy * vy + vz * vz).sqrt()
    }
}

/// Flat-array grid of fluid cells for a single chunk.
///
/// Uses the same `z * size * size + y * size + x` indexing as `Chunk`.
/// Exists only for chunks that contain fluid voxels; chunks with no
/// fluids don't need a FluidGrid.
#[derive(Debug, Clone)]
pub struct FluidGrid {
    cells: Vec<FluidCell>,
    size: usize,
}

impl FluidGrid {
    /// Create a grid filled with air cells.
    pub fn new_empty(size: usize) -> Self {
        Self {
            cells: vec![FluidCell::default(); size * size * size],
            size,
        }
    }

    /// Build a FluidGrid from chunk voxel data.
    ///
    /// Scans each voxel's material to classify cells:
    /// - Air material → `CellTag::Air`
    /// - Fluid material (liquid phase) → `CellTag::Liquid`
    /// - Solid material → `CellTag::Solid`
    ///
    /// Surface tagging is deferred to `update_tags()` which examines neighbors.
    /// If `registry` is `None`, uses a hardcoded fallback for known fluid IDs.
    pub fn from_chunk(chunk: &Chunk, registry: Option<&MaterialRegistry>) -> Self {
        let size = cube_root_exact(chunk.voxels().len());
        let mut grid = Self::new_empty(size);

        for (i, voxel) in chunk.voxels().iter().enumerate() {
            grid.cells[i] = classify_voxel(voxel, registry);
        }

        grid
    }

    /// Build a FluidGrid from a raw voxel slice (for testing / pure-function use).
    pub fn from_voxels(voxels: &[Voxel], size: usize, registry: Option<&MaterialRegistry>) -> Self {
        assert_eq!(
            voxels.len(),
            size * size * size,
            "voxel count must equal size³"
        );
        let mut grid = Self::new_empty(size);
        for (i, voxel) in voxels.iter().enumerate() {
            grid.cells[i] = classify_voxel(voxel, registry);
        }
        grid
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < self.size && y < self.size && z < self.size);
        z * self.size * self.size + y * self.size + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> &FluidCell {
        &self.cells[self.index(x, y, z)]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut FluidCell {
        let idx = self.index(x, y, z);
        &mut self.cells[idx]
    }

    /// Raw slice access for bulk operations.
    #[inline]
    pub fn cells(&self) -> &[FluidCell] {
        &self.cells
    }

    /// Mutable raw slice access.
    #[inline]
    pub fn cells_mut(&mut self) -> &mut [FluidCell] {
        &mut self.cells
    }

    /// Convert (x, y, z) from a flat index.
    #[inline]
    pub fn coords_from_index(&self, idx: usize) -> (usize, usize, usize) {
        let x = idx % self.size;
        let y = (idx / self.size) % self.size;
        let z = idx / (self.size * self.size);
        (x, y, z)
    }

    /// Count cells with a given tag.
    pub fn count_tag(&self, tag: CellTag) -> usize {
        self.cells.iter().filter(|c| c.tag == tag).count()
    }

    /// True if any cell is liquid or surface.
    pub fn has_fluid(&self) -> bool {
        self.cells.iter().any(|c| c.is_fluid())
    }

    /// Maximum speed across all cells (for CFL checks).
    pub fn max_speed(&self) -> f32 {
        self.cells.iter().map(|c| c.speed()).fold(0.0_f32, f32::max)
    }

    /// Total fluid volume in cubic meters (count of fluid cells × voxel_size³).
    /// At base resolution (1 voxel = 1 m), this equals the cell count.
    pub fn fluid_volume(&self) -> f32 {
        self.cells.iter().filter(|c| c.is_fluid()).count() as f32
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Classify a single voxel into a FluidCell.
fn classify_voxel(voxel: &Voxel, registry: Option<&MaterialRegistry>) -> FluidCell {
    if voxel.material.is_air() {
        return FluidCell::default();
    }

    let is_liquid = if let Some(reg) = registry {
        reg.get(voxel.material)
            .map(|m| m.default_phase == crate::data::Phase::Liquid)
            .unwrap_or(false)
    } else {
        is_known_fluid(voxel.material)
    };

    if is_liquid {
        FluidCell::new_liquid(voxel.material)
    } else {
        FluidCell::new_solid(voxel.material)
    }
}

/// Fallback fluid detection for known material IDs when no registry is available.
fn is_known_fluid(mat: MaterialId) -> bool {
    matches!(mat.0, 3 | 10) // water=3, lava=10
}

/// Integer cube root for perfect cubes. Panics if `n` is not a perfect cube.
fn cube_root_exact(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let approx = (n as f64).cbrt().round() as usize;
    // Check neighbors to handle floating-point imprecision
    for candidate in [approx.saturating_sub(1), approx, approx + 1] {
        if candidate * candidate * candidate == n {
            return candidate;
        }
    }
    panic!("voxel count {} is not a perfect cube", n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_root_exact_works() {
        assert_eq!(cube_root_exact(0), 0);
        assert_eq!(cube_root_exact(1), 1);
        assert_eq!(cube_root_exact(8), 2);
        assert_eq!(cube_root_exact(27), 3);
        assert_eq!(cube_root_exact(64), 4);
        assert_eq!(cube_root_exact(32768), 32);
    }

    #[test]
    #[should_panic(expected = "not a perfect cube")]
    fn cube_root_exact_panics_on_non_cube() {
        cube_root_exact(10);
    }

    #[test]
    fn cell_tag_default_is_air() {
        assert_eq!(CellTag::default(), CellTag::Air);
    }

    #[test]
    fn fluid_cell_default_is_air_at_rest() {
        let cell = FluidCell::default();
        assert_eq!(cell.tag, CellTag::Air);
        assert_eq!(cell.velocity, [0.0, 0.0, 0.0]);
        assert_eq!(cell.pressure, 0.0);
        assert_eq!(cell.material, MaterialId::AIR);
        assert!(!cell.is_fluid());
    }

    #[test]
    fn fluid_cell_new_liquid() {
        let cell = FluidCell::new_liquid(MaterialId::WATER);
        assert_eq!(cell.tag, CellTag::Liquid);
        assert_eq!(cell.material, MaterialId::WATER);
        assert!(cell.is_fluid());
        assert_eq!(cell.speed(), 0.0);
    }

    #[test]
    fn fluid_cell_new_solid() {
        let cell = FluidCell::new_solid(MaterialId::STONE);
        assert_eq!(cell.tag, CellTag::Solid);
        assert!(!cell.is_fluid());
    }

    #[test]
    fn fluid_cell_speed() {
        let cell = FluidCell {
            velocity: [3.0, 4.0, 0.0],
            ..Default::default()
        };
        assert!((cell.speed() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn grid_new_empty() {
        let grid = FluidGrid::new_empty(4);
        assert_eq!(grid.size(), 4);
        assert_eq!(grid.cells().len(), 64);
        assert_eq!(grid.count_tag(CellTag::Air), 64);
        assert!(!grid.has_fluid());
    }

    #[test]
    fn grid_indexing_matches_chunk_convention() {
        let grid = FluidGrid::new_empty(4);
        // z * size² + y * size + x
        assert_eq!(grid.index(0, 0, 0), 0);
        assert_eq!(grid.index(1, 0, 0), 1);
        assert_eq!(grid.index(0, 1, 0), 4);
        assert_eq!(grid.index(0, 0, 1), 16);
        assert_eq!(grid.index(3, 3, 3), 63);
    }

    #[test]
    fn grid_coords_from_index_roundtrip() {
        let grid = FluidGrid::new_empty(8);
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let idx = grid.index(x, y, z);
                    let (rx, ry, rz) = grid.coords_from_index(idx);
                    assert_eq!(
                        (x, y, z),
                        (rx, ry, rz),
                        "roundtrip failed for ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn grid_get_set() {
        let mut grid = FluidGrid::new_empty(4);
        grid.get_mut(1, 2, 3).tag = CellTag::Liquid;
        grid.get_mut(1, 2, 3).material = MaterialId::WATER;
        assert_eq!(grid.get(1, 2, 3).tag, CellTag::Liquid);
        assert!(grid.has_fluid());
        assert_eq!(grid.count_tag(CellTag::Liquid), 1);
    }

    #[test]
    fn grid_from_voxels_classifies_correctly() {
        let size = 4;
        let mut voxels = vec![Voxel::default(); size * size * size];
        // Place some water and stone
        voxels[0] = Voxel::new(MaterialId::WATER);
        voxels[1] = Voxel::new(MaterialId::LAVA);
        voxels[2] = Voxel::new(MaterialId::STONE);
        // rest are air

        let grid = FluidGrid::from_voxels(&voxels, size, None);
        assert_eq!(grid.get(0, 0, 0).tag, CellTag::Liquid);
        assert_eq!(grid.get(0, 0, 0).material, MaterialId::WATER);
        assert_eq!(grid.get(1, 0, 0).tag, CellTag::Liquid);
        assert_eq!(grid.get(1, 0, 0).material, MaterialId::LAVA);
        assert_eq!(grid.get(2, 0, 0).tag, CellTag::Solid);
        assert_eq!(grid.get(3, 0, 0).tag, CellTag::Air);
    }

    #[test]
    fn grid_fluid_volume() {
        let size = 4;
        let mut voxels = vec![Voxel::default(); size * size * size];
        voxels[0] = Voxel::new(MaterialId::WATER);
        voxels[1] = Voxel::new(MaterialId::WATER);
        voxels[2] = Voxel::new(MaterialId::WATER);
        let grid = FluidGrid::from_voxels(&voxels, size, None);
        assert!((grid.fluid_volume() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn grid_max_speed() {
        let mut grid = FluidGrid::new_empty(4);
        grid.get_mut(0, 0, 0).velocity = [1.0, 0.0, 0.0];
        grid.get_mut(1, 1, 1).velocity = [3.0, 4.0, 0.0]; // speed=5
        grid.get_mut(2, 2, 2).velocity = [0.0, 2.0, 0.0];
        assert!((grid.max_speed() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn classify_voxel_without_registry() {
        let water = Voxel::new(MaterialId::WATER);
        let stone = Voxel::new(MaterialId::STONE);
        let air = Voxel::default();

        let wc = classify_voxel(&water, None);
        assert_eq!(wc.tag, CellTag::Liquid);

        let sc = classify_voxel(&stone, None);
        assert_eq!(sc.tag, CellTag::Solid);

        let ac = classify_voxel(&air, None);
        assert_eq!(ac.tag, CellTag::Air);
    }

    #[test]
    fn is_known_fluid_identifies_water_and_lava() {
        assert!(is_known_fluid(MaterialId::WATER));
        assert!(is_known_fluid(MaterialId::LAVA));
        assert!(!is_known_fluid(MaterialId::STONE));
        assert!(!is_known_fluid(MaterialId::AIR));
        assert!(!is_known_fluid(MaterialId::ICE));
        assert!(!is_known_fluid(MaterialId::STEAM));
    }
}
