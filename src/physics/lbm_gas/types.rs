// Core data types for the LBM gas simulation.
//
// LbmGrid is a flat array of LbmCells with the same z*size²+y*size+x
// indexing as Chunk. Each cell stores 19 distribution functions for the
// D3Q19 lattice plus the material occupying that cell.

use crate::data::MaterialRegistry;
use crate::world::chunk::Chunk;
use crate::world::voxel::{MaterialId, Voxel};
use crate::world::voxel_access::VoxelAccess;

use super::lattice::{self, Q};

/// Classification of an LBM cell for boundary handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GasCellTag {
    /// Gas cell — participates in LBM streaming and collision.
    #[default]
    Gas,
    /// Solid wall — bounce-back boundary condition.
    Solid,
    /// Liquid surface — treated as a wall from the gas side.
    Liquid,
}

/// Per-cell LBM state. Stores the 19 distribution functions plus metadata.
///
/// Size: 19 × 4 + 2 + 4 + 4 + padding ≈ 88 bytes per cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LbmCell {
    /// Distribution functions for D3Q19 lattice.
    pub f: [f32; Q],
    /// Material ID of the gas occupying this cell (air, steam, etc.).
    pub material: MaterialId,
    /// Cell classification for boundary conditions.
    pub tag: GasCellTag,
    /// Water vapor mixing ratio (kg vapor / kg dry air). Advected as passive scalar.
    pub moisture: f32,
    /// Liquid water content from condensation (kg/m³). Cloud density field.
    pub cloud_lwc: f32,
}

impl Default for LbmCell {
    fn default() -> Self {
        Self {
            f: lattice::equilibrium(1.0, [0.0; 3]),
            material: MaterialId::AIR,
            tag: GasCellTag::Gas,
            moisture: 0.0,
            cloud_lwc: 0.0,
        }
    }
}

impl LbmCell {
    /// Create a gas cell initialized to equilibrium at given density and zero velocity.
    pub fn new_gas(material: MaterialId, rho: f32) -> Self {
        Self {
            f: lattice::equilibrium(rho, [0.0; 3]),
            material,
            tag: GasCellTag::Gas,
            moisture: 0.0,
            cloud_lwc: 0.0,
        }
    }

    /// Create a gas cell with initial moisture content.
    pub fn new_gas_moist(material: MaterialId, rho: f32, moisture: f32) -> Self {
        Self {
            f: lattice::equilibrium(rho, [0.0; 3]),
            material,
            tag: GasCellTag::Gas,
            moisture,
            cloud_lwc: 0.0,
        }
    }

    /// Create a solid wall cell. Distributions are zeroed (unused).
    pub fn new_solid(material: MaterialId) -> Self {
        Self {
            f: [0.0; Q],
            material,
            tag: GasCellTag::Solid,
            moisture: 0.0,
            cloud_lwc: 0.0,
        }
    }

    /// Create a liquid boundary cell. Treated as wall from gas side.
    pub fn new_liquid(material: MaterialId) -> Self {
        Self {
            f: [0.0; Q],
            material,
            tag: GasCellTag::Liquid,
            moisture: 0.0,
            cloud_lwc: 0.0,
        }
    }

    /// Whether this cell participates in LBM collision and streaming.
    pub fn is_gas(&self) -> bool {
        self.tag == GasCellTag::Gas
    }

    /// Whether this cell acts as a bounce-back wall.
    pub fn is_wall(&self) -> bool {
        matches!(self.tag, GasCellTag::Solid | GasCellTag::Liquid)
    }

    /// Recover macroscopic density: ρ = Σf_i.
    pub fn density(&self) -> f32 {
        self.f.iter().sum()
    }

    /// Recover macroscopic velocity: u = Σ(f_i × e_i) / ρ.
    pub fn velocity(&self) -> [f32; 3] {
        let rho = self.density();
        if rho.abs() < 1e-12 {
            return [0.0; 3];
        }
        let mut u = [0.0f32; 3];
        for i in 0..Q {
            let ei = lattice::E[i];
            u[0] += self.f[i] * ei[0] as f32;
            u[1] += self.f[i] * ei[1] as f32;
            u[2] += self.f[i] * ei[2] as f32;
        }
        u[0] /= rho;
        u[1] /= rho;
        u[2] /= rho;
        u
    }

    /// Mach number: |u| / cs (lattice units). Should stay below ~0.3.
    pub fn mach_number(&self) -> f32 {
        let u = self.velocity();
        let speed = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        speed / lattice::CS2.sqrt()
    }
}

/// Flat-array LBM grid with same indexing as Chunk.
#[derive(Debug, Clone)]
pub struct LbmGrid {
    cells: Vec<LbmCell>,
    size: usize,
}

impl LbmGrid {
    /// Create an empty grid of the given edge length, all cells initialized
    /// to gas equilibrium at ρ=1.0, u=0.
    pub fn new_empty(size: usize) -> Self {
        Self {
            cells: vec![LbmCell::default(); size * size * size],
            size,
        }
    }

    /// Create a grid from a flat voxel array. Gas voxels become gas cells
    /// initialized to equilibrium; solids/liquids become wall cells.
    pub fn from_voxels(voxels: &[Voxel], size: usize, registry: Option<&MaterialRegistry>) -> Self {
        let n = size * size * size;
        assert_eq!(voxels.len(), n, "Voxel count must match size³");

        let cells: Vec<LbmCell> = voxels.iter().map(|v| classify_voxel(v, registry)).collect();

        Self { cells, size }
    }

    /// Create a grid from a Chunk.
    pub fn from_chunk(chunk: &Chunk, registry: Option<&MaterialRegistry>) -> Self {
        Self::from_voxels(chunk.voxels(), chunk.size(), registry)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Flat index from 3D coordinates: z*size²+y*size+x.
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    /// 3D coordinates from flat index.
    pub fn coords_from_index(&self, idx: usize) -> (usize, usize, usize) {
        let z = idx / (self.size * self.size);
        let rem = idx % (self.size * self.size);
        let y = rem / self.size;
        let x = rem % self.size;
        (x, y, z)
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> &LbmCell {
        &self.cells[self.index(x, y, z)]
    }

    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut LbmCell {
        let idx = self.index(x, y, z);
        &mut self.cells[idx]
    }

    pub fn cells(&self) -> &[LbmCell] {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut [LbmCell] {
        &mut self.cells
    }

    /// Count cells with a given tag.
    pub fn count_tag(&self, tag: GasCellTag) -> usize {
        self.cells.iter().filter(|c| c.tag == tag).count()
    }

    /// Whether any gas cell has non-equilibrium distributions (i.e., flow).
    pub fn has_gas(&self) -> bool {
        self.cells.iter().any(|c| c.is_gas())
    }

    /// Maximum Mach number across all gas cells.
    pub fn max_mach(&self) -> f32 {
        self.cells
            .iter()
            .filter(|c| c.is_gas())
            .map(|c| c.mach_number())
            .fold(0.0f32, f32::max)
    }

    /// Total gas mass in the grid (sum of densities for gas cells).
    pub fn total_mass(&self) -> f32 {
        self.cells
            .iter()
            .filter(|c| c.is_gas())
            .map(|c| c.density())
            .sum()
    }
}

/// Classify a single voxel into an LBM cell.
fn classify_voxel(voxel: &Voxel, registry: Option<&MaterialRegistry>) -> LbmCell {
    let mat = voxel.material;

    if is_gas_material(mat, registry) {
        // Density from voxel pressure via ideal gas approximation:
        // For simplicity, use normalized lattice density ≈ 1.0 for ambient.
        // Pressure deviations will drive flow.
        let rho = pressure_to_lattice_density(voxel.pressure);
        LbmCell::new_gas(mat, rho)
    } else if is_liquid_material(mat, registry) {
        LbmCell::new_liquid(mat)
    } else {
        LbmCell::new_solid(mat)
    }
}

/// Check if a material is a gas (participates in LBM).
fn is_gas_material(mat: MaterialId, registry: Option<&MaterialRegistry>) -> bool {
    if let Some(reg) = registry
        && let Some(data) = reg.get(mat)
    {
        return data.default_phase == crate::data::Phase::Gas;
    }
    // Hardcoded fallback: air (0) and steam (9)
    mat == MaterialId::AIR || mat == MaterialId::STEAM
}

/// Check if a material is a liquid (wall from gas perspective).
fn is_liquid_material(mat: MaterialId, registry: Option<&MaterialRegistry>) -> bool {
    if let Some(reg) = registry
        && let Some(data) = reg.get(mat)
    {
        return data.default_phase == crate::data::Phase::Liquid;
    }
    // Hardcoded fallback: water (3) and lava (10)
    mat == MaterialId::WATER || mat == MaterialId::LAVA
}

/// Convert physical pressure (Pa) to lattice density.
/// Uses linear mapping: ρ_lattice = P_physical / P_atmospheric.
/// At sea level (101325 Pa), ρ_lattice = 1.0.
fn pressure_to_lattice_density(pressure_pa: f32) -> f32 {
    const ATMOSPHERIC: f32 = 101_325.0;
    (pressure_pa / ATMOSPHERIC).max(0.01)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lbm_cell_is_gas_at_equilibrium() {
        let cell = LbmCell::default();
        assert!(cell.is_gas());
        assert!(!cell.is_wall());
        assert_eq!(cell.material, MaterialId::AIR);
        assert!((cell.density() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn new_gas_cell_has_correct_density() {
        let cell = LbmCell::new_gas(MaterialId::STEAM, 0.6);
        assert!(cell.is_gas());
        assert_eq!(cell.material, MaterialId::STEAM);
        assert!(
            (cell.density() - 0.6).abs() < 1e-5,
            "density = {}, expected 0.6",
            cell.density()
        );
    }

    #[test]
    fn new_solid_cell_is_wall() {
        let cell = LbmCell::new_solid(MaterialId::STONE);
        assert!(cell.is_wall());
        assert!(!cell.is_gas());
        assert_eq!(cell.tag, GasCellTag::Solid);
    }

    #[test]
    fn new_liquid_cell_is_wall() {
        let cell = LbmCell::new_liquid(MaterialId::WATER);
        assert!(cell.is_wall());
        assert_eq!(cell.tag, GasCellTag::Liquid);
    }

    #[test]
    fn gas_cell_velocity_at_rest_is_zero() {
        let cell = LbmCell::new_gas(MaterialId::AIR, 1.0);
        let u = cell.velocity();
        for (d, &val) in u.iter().enumerate() {
            assert!(val.abs() < 1e-6, "velocity[{d}] = {val}, expected 0");
        }
    }

    #[test]
    fn gas_cell_velocity_with_flow() {
        let u_in = [0.05, -0.03, 0.02];
        let cell = LbmCell {
            f: lattice::equilibrium(1.0, u_in),
            material: MaterialId::AIR,
            tag: GasCellTag::Gas,
            moisture: 0.0,
            cloud_lwc: 0.0,
        };
        let u_out = cell.velocity();
        for d in 0..3 {
            assert!(
                (u_out[d] - u_in[d]).abs() < 1e-5,
                "velocity[{d}] = {}, expected {}",
                u_out[d],
                u_in[d]
            );
        }
    }

    #[test]
    fn mach_number_at_rest_is_zero() {
        let cell = LbmCell::new_gas(MaterialId::AIR, 1.0);
        assert!(cell.mach_number() < 1e-6);
    }

    #[test]
    fn lbm_grid_new_empty() {
        let grid = LbmGrid::new_empty(4);
        assert_eq!(grid.size(), 4);
        assert_eq!(grid.cells().len(), 64);
        assert!(grid.has_gas());
        assert_eq!(grid.count_tag(GasCellTag::Gas), 64);
    }

    #[test]
    fn lbm_grid_indexing_roundtrip() {
        let grid = LbmGrid::new_empty(8);
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let idx = grid.index(x, y, z);
                    let (rx, ry, rz) = grid.coords_from_index(idx);
                    assert_eq!((rx, ry, rz), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn lbm_grid_from_voxels_classifies_materials() {
        let size = 4;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        // Place a stone block and a water block
        voxels[0] = Voxel::new(MaterialId::STONE);
        voxels[1] = Voxel::new(MaterialId::WATER);
        voxels[2] = Voxel::new(MaterialId::STEAM);

        let grid = LbmGrid::from_voxels(&voxels, size, None);
        assert_eq!(grid.get(0, 0, 0).tag, GasCellTag::Solid); // stone
        assert_eq!(grid.get(1, 0, 0).tag, GasCellTag::Liquid); // water
        assert_eq!(grid.get(2, 0, 0).tag, GasCellTag::Gas); // steam
        assert_eq!(grid.get(3, 0, 0).tag, GasCellTag::Gas); // air
    }

    #[test]
    fn total_mass_counts_only_gas_cells() {
        let size = 2;
        let mut voxels = vec![Voxel::new(MaterialId::AIR); size * size * size];
        voxels[0] = Voxel::new(MaterialId::STONE); // solid — excluded from mass

        let grid = LbmGrid::from_voxels(&voxels, size, None);
        let total = grid.total_mass();
        // 7 gas cells × ~1.0 density each
        assert!(
            (total - 7.0).abs() < 0.1,
            "total mass = {total}, expected ~7.0"
        );
    }

    #[test]
    fn pressure_to_lattice_density_atmospheric() {
        let rho = pressure_to_lattice_density(101_325.0);
        assert!(
            (rho - 1.0).abs() < 1e-5,
            "Atmospheric pressure → ρ={rho}, expected 1.0"
        );
    }

    #[test]
    fn pressure_to_lattice_density_high_pressure() {
        let rho = pressure_to_lattice_density(202_650.0);
        assert!(
            (rho - 2.0).abs() < 1e-4,
            "Double atmospheric → ρ={rho}, expected 2.0"
        );
    }

    #[test]
    fn lbm_cell_size_is_reasonable() {
        let sz = std::mem::size_of::<LbmCell>();
        // 19 f32 (76) + 2 f32 (moisture, cloud_lwc) (8) + MaterialId (2) + GasCellTag (1) + padding ≈ 88
        assert!(sz <= 92, "LbmCell is {sz} bytes, expected ≤92");
    }
}
