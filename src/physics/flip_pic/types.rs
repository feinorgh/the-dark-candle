// Core types for FLIP/PIC particle simulation.
//
// Particle: Lagrangian element carrying position, velocity, mass, material, temperature.
// ParticleBuffer: per-chunk particle storage.
// VelocityGrid: staggered MAC grid for pressure projection.
// AccumulationGrid: sub-voxel mass tracking for particle deposition.

use crate::world::chunk::ChunkCoord;
use crate::world::voxel::MaterialId;

/// Lifecycle state of a particle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleTag {
    /// In flight — participates in P2G/G2P/advection.
    Airborne,
    /// Has deposited into the voxel grid — pending removal.
    Deposited,
    /// Absorbed back into a liquid body — pending removal.
    Absorbed,
}

/// A single Lagrangian particle (~44 bytes).
///
/// All units are SI: position in meters, velocity in m/s, mass in kg,
/// temperature in Kelvin, age in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub mass: f32,
    pub material: MaterialId,
    pub temperature: f32,
    pub age: f32,
    pub tag: ParticleTag,
}

impl Particle {
    pub fn new(position: [f32; 3], velocity: [f32; 3], mass: f32, material: MaterialId) -> Self {
        Self {
            position,
            velocity,
            mass,
            material,
            temperature: 288.15, // ambient
            age: 0.0,
            tag: ParticleTag::Airborne,
        }
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    pub fn speed(&self) -> f32 {
        let [vx, vy, vz] = self.velocity;
        (vx * vx + vy * vy + vz * vz).sqrt()
    }

    pub fn is_airborne(&self) -> bool {
        self.tag == ParticleTag::Airborne
    }
}

/// Per-chunk particle storage.
#[derive(Debug, Clone)]
pub struct ParticleBuffer {
    pub particles: Vec<Particle>,
    pub chunk_coord: ChunkCoord,
}

impl ParticleBuffer {
    pub fn new(coord: ChunkCoord) -> Self {
        Self {
            particles: Vec::new(),
            chunk_coord: coord,
        }
    }

    pub fn airborne_count(&self) -> usize {
        self.particles.iter().filter(|p| p.is_airborne()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Remove all non-airborne particles.
    pub fn remove_dead(&mut self) {
        self.particles.retain(|p| p.is_airborne());
    }
}

/// Staggered MAC (Marker-And-Cell) velocity grid.
///
/// Velocities live on cell faces, not centers:
/// - `u` (x-component): (size+1) × size × size values, on x-faces
/// - `v` (y-component): size × (size+1) × size values, on y-faces
/// - `w` (z-component): size × size × (size+1) values, on z-faces
///
/// This prevents odd-even pressure decoupling and is standard for
/// incompressible flow solvers.
#[derive(Debug, Clone)]
pub struct VelocityGrid {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    pub size: usize,
}

impl VelocityGrid {
    pub fn new(size: usize) -> Self {
        let s = size;
        Self {
            u: vec![0.0; (s + 1) * s * s],
            v: vec![0.0; s * (s + 1) * s],
            w: vec![0.0; s * s * (s + 1)],
            size: s,
        }
    }

    // --- U (x-face) indexing: (size+1) × size × size ---
    // Index: z * size * (size+1) + y * (size+1) + i
    // where i ∈ [0, size] is the face index along x

    pub fn u_index(&self, i: usize, y: usize, z: usize) -> usize {
        let s = self.size;
        z * s * (s + 1) + y * (s + 1) + i
    }

    pub fn get_u(&self, i: usize, y: usize, z: usize) -> f32 {
        self.u[self.u_index(i, y, z)]
    }

    pub fn set_u(&mut self, i: usize, y: usize, z: usize, val: f32) {
        let idx = self.u_index(i, y, z);
        self.u[idx] = val;
    }

    // --- V (y-face) indexing: size × (size+1) × size ---
    // Index: z * (size+1) * size + j * size + x
    // where j ∈ [0, size] is the face index along y

    pub fn v_index(&self, x: usize, j: usize, z: usize) -> usize {
        let s = self.size;
        z * (s + 1) * s + j * s + x
    }

    pub fn get_v(&self, x: usize, j: usize, z: usize) -> f32 {
        self.v[self.v_index(x, j, z)]
    }

    pub fn set_v(&mut self, x: usize, j: usize, z: usize, val: f32) {
        let idx = self.v_index(x, j, z);
        self.v[idx] = val;
    }

    // --- W (z-face) indexing: size × size × (size+1) ---
    // Index: k * size * size + y * size + x
    // where k ∈ [0, size] is the face index along z

    pub fn w_index(&self, x: usize, y: usize, k: usize) -> usize {
        let s = self.size;
        k * s * s + y * s + x
    }

    pub fn get_w(&self, x: usize, y: usize, k: usize) -> f32 {
        self.w[self.w_index(x, y, k)]
    }

    pub fn set_w(&mut self, x: usize, y: usize, k: usize, val: f32) {
        let idx = self.w_index(x, y, k);
        self.w[idx] = val;
    }

    /// Trilinear interpolation of velocity at an arbitrary position.
    ///
    /// Position is in local chunk coordinates: [0, size) for each axis.
    /// Each component is interpolated from its respective face grid.
    pub fn velocity_at(&self, pos: [f32; 3]) -> [f32; 3] {
        let s = self.size;
        [
            self.interp_u(pos, s),
            self.interp_v(pos, s),
            self.interp_w(pos, s),
        ]
    }

    /// Interpolate u-component. U lives on x-faces at (i, y+0.5, z+0.5).
    fn interp_u(&self, pos: [f32; 3], s: usize) -> f32 {
        let px = pos[0].clamp(0.0, s as f32);
        let py = (pos[1] - 0.5).clamp(0.0, (s - 1) as f32);
        let pz = (pos[2] - 0.5).clamp(0.0, (s - 1) as f32);

        let i0 = (px as usize).min(s - 1);
        let i1 = (i0 + 1).min(s);
        let y0 = (py as usize).min(s - 1);
        let y1 = (y0 + 1).min(s - 1);
        let z0 = (pz as usize).min(s - 1);
        let z1 = (z0 + 1).min(s - 1);

        let fx = px - i0 as f32;
        let fy = py - y0 as f32;
        let fz = pz - z0 as f32;

        trilinear(
            self.get_u(i0, y0, z0),
            self.get_u(i1, y0, z0),
            self.get_u(i0, y1, z0),
            self.get_u(i1, y1, z0),
            self.get_u(i0, y0, z1),
            self.get_u(i1, y0, z1),
            self.get_u(i0, y1, z1),
            self.get_u(i1, y1, z1),
            fx,
            fy,
            fz,
        )
    }

    /// Interpolate v-component. V lives on y-faces at (x+0.5, j, z+0.5).
    fn interp_v(&self, pos: [f32; 3], s: usize) -> f32 {
        let px = (pos[0] - 0.5).clamp(0.0, (s - 1) as f32);
        let py = pos[1].clamp(0.0, s as f32);
        let pz = (pos[2] - 0.5).clamp(0.0, (s - 1) as f32);

        let x0 = (px as usize).min(s - 1);
        let x1 = (x0 + 1).min(s - 1);
        let j0 = (py as usize).min(s - 1);
        let j1 = (j0 + 1).min(s);
        let z0 = (pz as usize).min(s - 1);
        let z1 = (z0 + 1).min(s - 1);

        let fx = px - x0 as f32;
        let fy = py - j0 as f32;
        let fz = pz - z0 as f32;

        trilinear(
            self.get_v(x0, j0, z0),
            self.get_v(x1, j0, z0),
            self.get_v(x0, j1, z0),
            self.get_v(x1, j1, z0),
            self.get_v(x0, j0, z1),
            self.get_v(x1, j0, z1),
            self.get_v(x0, j1, z1),
            self.get_v(x1, j1, z1),
            fx,
            fy,
            fz,
        )
    }

    /// Interpolate w-component. W lives on z-faces at (x+0.5, y+0.5, k).
    fn interp_w(&self, pos: [f32; 3], s: usize) -> f32 {
        let px = (pos[0] - 0.5).clamp(0.0, (s - 1) as f32);
        let py = (pos[1] - 0.5).clamp(0.0, (s - 1) as f32);
        let pz = pos[2].clamp(0.0, s as f32);

        let x0 = (px as usize).min(s - 1);
        let x1 = (x0 + 1).min(s - 1);
        let y0 = (py as usize).min(s - 1);
        let y1 = (y0 + 1).min(s - 1);
        let k0 = (pz as usize).min(s - 1);
        let k1 = (k0 + 1).min(s);

        let fx = px - x0 as f32;
        let fy = py - y0 as f32;
        let fz = pz - k0 as f32;

        trilinear(
            self.get_w(x0, y0, k0),
            self.get_w(x1, y0, k0),
            self.get_w(x0, y1, k0),
            self.get_w(x1, y1, k0),
            self.get_w(x0, y0, k1),
            self.get_w(x1, y0, k1),
            self.get_w(x0, y1, k1),
            self.get_w(x1, y1, k1),
            fx,
            fy,
            fz,
        )
    }
}

/// Weight grid for P2G normalization — same layout as VelocityGrid.
#[derive(Debug, Clone)]
pub struct WeightGrid {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    pub size: usize,
}

impl WeightGrid {
    pub fn new(size: usize) -> Self {
        let s = size;
        Self {
            u: vec![0.0; (s + 1) * s * s],
            v: vec![0.0; s * (s + 1) * s],
            w: vec![0.0; s * s * (s + 1)],
            size: s,
        }
    }
}

/// Sub-voxel mass accumulation tracking for particle deposition.
///
/// Each cell tracks how much mass has been deposited by particles.
/// When accumulated mass exceeds material density × voxel volume (1 m³),
/// the voxel transitions from air to the deposited material.
#[derive(Debug, Clone)]
pub struct AccumulationGrid {
    pub mass: Vec<f32>,
    pub material: Vec<MaterialId>,
    pub size: usize,
}

impl AccumulationGrid {
    pub fn new(size: usize) -> Self {
        let vol = size * size * size;
        Self {
            mass: vec![0.0; vol],
            material: vec![MaterialId::AIR; vol],
            size,
        }
    }

    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.size * self.size + y * self.size + x
    }

    pub fn get_mass(&self, x: usize, y: usize, z: usize) -> f32 {
        self.mass[self.index(x, y, z)]
    }

    pub fn get_material(&self, x: usize, y: usize, z: usize) -> MaterialId {
        self.material[self.index(x, y, z)]
    }

    pub fn add_mass(&mut self, x: usize, y: usize, z: usize, amount: f32, mat: MaterialId) {
        let idx = self.index(x, y, z);
        self.mass[idx] += amount;
        self.material[idx] = mat;
    }

    pub fn reset_cell(&mut self, x: usize, y: usize, z: usize) {
        let idx = self.index(x, y, z);
        self.mass[idx] = 0.0;
        self.material[idx] = MaterialId::AIR;
    }
}

/// Trilinear interpolation of 8 corner values.
///
/// fx, fy, fz ∈ [0, 1] are fractional offsets within the cell.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn trilinear(
    c000: f32,
    c100: f32,
    c010: f32,
    c110: f32,
    c001: f32,
    c101: f32,
    c011: f32,
    c111: f32,
    fx: f32,
    fy: f32,
    fz: f32,
) -> f32 {
    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;
    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;
    c0 * (1.0 - fz) + c1 * fz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_creation() {
        let p = Particle::new([1.0, 2.0, 3.0], [0.5, 0.0, 0.0], 0.001, MaterialId::WATER);
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
        assert!((p.speed() - 0.5).abs() < 1e-6);
        assert!(p.is_airborne());
        assert!((p.temperature - 288.15).abs() < 0.01);
    }

    #[test]
    fn particle_with_temperature() {
        let p =
            Particle::new([0.0; 3], [0.0; 3], 0.001, MaterialId::STEAM).with_temperature(373.15);
        assert!((p.temperature - 373.15).abs() < 0.01);
    }

    #[test]
    fn particle_speed() {
        let p = Particle::new([0.0; 3], [3.0, 4.0, 0.0], 1.0, MaterialId::WATER);
        assert!((p.speed() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn particle_buffer_basics() {
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        let mut buf = ParticleBuffer::new(coord);
        assert!(buf.is_empty());
        assert_eq!(buf.airborne_count(), 0);

        buf.particles
            .push(Particle::new([0.0; 3], [0.0; 3], 0.001, MaterialId::WATER));
        assert_eq!(buf.airborne_count(), 1);

        buf.particles[0].tag = ParticleTag::Deposited;
        assert_eq!(buf.airborne_count(), 0);

        buf.remove_dead();
        assert!(buf.is_empty());
    }

    #[test]
    fn velocity_grid_dimensions() {
        let g = VelocityGrid::new(4);
        assert_eq!(g.u.len(), 5 * 4 * 4); // (4+1)*4*4 = 80
        assert_eq!(g.v.len(), 4 * 5 * 4); // 4*(4+1)*4 = 80
        assert_eq!(g.w.len(), 4 * 4 * 5); // 4*4*(4+1) = 80
    }

    #[test]
    fn velocity_grid_indexing() {
        let mut g = VelocityGrid::new(4);
        g.set_u(2, 1, 3, 5.0);
        assert!((g.get_u(2, 1, 3) - 5.0).abs() < 1e-6);

        g.set_v(1, 3, 2, -2.0);
        assert!((g.get_v(1, 3, 2) - (-2.0)).abs() < 1e-6);

        g.set_w(3, 0, 4, 7.5);
        assert!((g.get_w(3, 0, 4) - 7.5).abs() < 1e-6);
    }

    #[test]
    fn velocity_grid_uniform_flow() {
        let mut g = VelocityGrid::new(4);
        let s = g.size;
        // Set all u-faces to 3.0
        for k in 0..s {
            for j in 0..s {
                for i in 0..=s {
                    g.set_u(i, j, k, 3.0);
                }
            }
        }
        let v = g.velocity_at([2.0, 2.0, 2.0]);
        assert!((v[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn velocity_grid_interpolation_center() {
        let mut g = VelocityGrid::new(4);
        // Put v=1.0 on all y-faces around cell (2,2,2)
        for j in 2..=3 {
            for x in 1..=3 {
                for z in 1..=3 {
                    g.set_v(x, j, z, 1.0);
                }
            }
        }
        let v = g.velocity_at([2.5, 2.5, 2.5]);
        assert!((v[1] - 1.0).abs() < 0.2);
    }

    #[test]
    fn trilinear_corner_values() {
        // At corner (0,0,0) → c000
        assert!(
            (trilinear(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0) - 1.0).abs() < 1e-6
        );
        // At corner (1,1,1) → c111
        assert!(
            (trilinear(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 1.0, 1.0) - 8.0).abs() < 1e-6
        );
        // At center (0.5, 0.5, 0.5) → average of all 8
        let avg = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 8.0;
        assert!(
            (trilinear(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 0.5, 0.5) - avg).abs() < 1e-6
        );
    }

    #[test]
    fn accumulation_grid_basics() {
        let mut ag = AccumulationGrid::new(4);
        assert!((ag.get_mass(1, 2, 3)).abs() < 1e-6);

        ag.add_mass(1, 2, 3, 50.0, MaterialId::ICE);
        assert!((ag.get_mass(1, 2, 3) - 50.0).abs() < 1e-6);
        assert_eq!(ag.get_material(1, 2, 3), MaterialId::ICE);

        ag.add_mass(1, 2, 3, 30.0, MaterialId::ICE);
        assert!((ag.get_mass(1, 2, 3) - 80.0).abs() < 1e-6);

        ag.reset_cell(1, 2, 3);
        assert!((ag.get_mass(1, 2, 3)).abs() < 1e-6);
        assert_eq!(ag.get_material(1, 2, 3), MaterialId::AIR);
    }

    #[test]
    fn weight_grid_dimensions() {
        let w = WeightGrid::new(4);
        assert_eq!(w.u.len(), 5 * 4 * 4);
        assert_eq!(w.v.len(), 4 * 5 * 4);
        assert_eq!(w.w.len(), 4 * 4 * 5);
    }
}
