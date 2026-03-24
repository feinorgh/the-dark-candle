// D3Q19 lattice definition for the Lattice Boltzmann Method.
//
// The D3Q19 lattice has 19 discrete velocity directions in 3D: one rest
// velocity, six face-adjacent, and twelve edge-adjacent. This gives a good
// balance between accuracy and memory (19 × 4 bytes = 76 bytes per cell).
//
// All constants here are in lattice units (dx=1, dt=1). Physical unit
// conversion is handled by `LbmScaling`.

/// Number of discrete velocities in the D3Q19 lattice.
pub const Q: usize = 19;

/// Speed of sound squared in lattice units: cs² = 1/3.
pub const CS2: f32 = 1.0 / 3.0;

/// cs⁴ = 1/9, used in non-equilibrium stress tensor computation.
pub const CS4: f32 = 1.0 / 9.0;

/// Discrete velocity vectors for D3Q19.
///
/// Index 0 is the rest velocity (0,0,0).
/// Indices 1–6 are face-adjacent (±x, ±y, ±z).
/// Indices 7–18 are edge-adjacent (±x±y, ±x±z, ±y±z).
pub const E: [[i32; 3]; Q] = [
    // Rest
    [0, 0, 0],
    // Face-adjacent (6)
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    // Edge-adjacent (12)
    [1, 1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
];

/// Lattice weights for D3Q19.
///
/// - w_0 = 1/3 (rest)
/// - w_{1..6} = 1/18 (face-adjacent, |e|² = 1)
/// - w_{7..18} = 1/36 (edge-adjacent, |e|² = 2)
pub const W: [f32; Q] = [
    1.0 / 3.0,  // rest
    1.0 / 18.0, // +x
    1.0 / 18.0, // -x
    1.0 / 18.0, // +y
    1.0 / 18.0, // -y
    1.0 / 18.0, // +z
    1.0 / 18.0, // -z
    1.0 / 36.0, // +x+y
    1.0 / 36.0, // -x+y
    1.0 / 36.0, // +x-y
    1.0 / 36.0, // -x-y
    1.0 / 36.0, // +x+z
    1.0 / 36.0, // -x+z
    1.0 / 36.0, // +x-z
    1.0 / 36.0, // -x-z
    1.0 / 36.0, // +y+z
    1.0 / 36.0, // -y+z
    1.0 / 36.0, // +y-z
    1.0 / 36.0, // -y-z
];

/// Opposite direction index for each velocity. Used for bounce-back BCs.
/// OPPOSITE[i] is the index j such that E[j] = -E[i].
pub const OPPOSITE: [usize; Q] = [
    0,  // 0: rest → rest
    2,  // 1: +x → -x
    1,  // 2: -x → +x
    4,  // 3: +y → -y
    3,  // 4: -y → +y
    6,  // 5: +z → -z
    5,  // 6: -z → +z
    10, // 7: +x+y → -x-y
    9,  // 8: -x+y → +x-y
    8,  // 9: +x-y → -x+y
    7,  // 10: -x-y → +x+y
    14, // 11: +x+z → -x-z
    13, // 12: -x+z → +x-z
    12, // 13: +x-z → -x+z
    11, // 14: -x-z → +x+z
    18, // 15: +y+z → -y-z
    17, // 16: -y+z → +y-z
    16, // 17: +y-z → -y+z
    15, // 18: -y-z → +y+z
];

/// Maxwell-Boltzmann equilibrium distribution for the D3Q19 lattice.
///
/// f_eq_i = w_i × ρ × (1 + (e_i · u)/cs² + (e_i · u)²/(2·cs⁴) - u·u/(2·cs²))
///
/// This is the second-order truncation of the continuous Maxwell-Boltzmann
/// distribution, which is standard for incompressible/weakly compressible LBM.
pub fn equilibrium(rho: f32, u: [f32; 3]) -> [f32; Q] {
    let u_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    let mut f_eq = [0.0f32; Q];

    for i in 0..Q {
        let eu = E[i][0] as f32 * u[0] + E[i][1] as f32 * u[1] + E[i][2] as f32 * u[2];
        f_eq[i] = W[i] * rho * (1.0 + eu / CS2 + (eu * eu) / (2.0 * CS4) - u_sq / (2.0 * CS2));
    }

    f_eq
}

/// Lattice-to-physical unit conversion parameters.
///
/// In LBM, the lattice uses dimensionless units (dx_L=1, dt_L=1).
/// This struct holds the scaling factors to convert between lattice
/// and physical (SI) units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LbmScaling {
    /// Physical length per lattice cell (m). Equal to VOXEL_SIZE.
    pub dx: f32,
    /// Physical time per lattice timestep (s). Derived from τ and ν_eff.
    pub dt: f32,
}

impl LbmScaling {
    /// Create scaling from physical cell size and effective kinematic viscosity.
    ///
    /// Given τ (relaxation time) and ν_eff (physical kinematic viscosity in m²/s),
    /// the lattice timestep maps to:
    ///   dt_phys = ν_lattice × dx² / ν_eff
    /// where ν_lattice = cs² × (τ - 0.5).
    pub fn from_tau_and_viscosity(dx: f32, tau: f32, nu_eff: f32) -> Self {
        let nu_lattice = CS2 * (tau - 0.5);
        let dt = if nu_eff > 0.0 {
            nu_lattice * dx * dx / nu_eff
        } else {
            1.0 // fallback: 1 second per step
        };
        Self { dx, dt }
    }

    /// Convert lattice velocity to physical velocity (m/s).
    pub fn velocity_to_physical(&self, u_lattice: [f32; 3]) -> [f32; 3] {
        let c = self.dx / self.dt;
        [u_lattice[0] * c, u_lattice[1] * c, u_lattice[2] * c]
    }

    /// Convert physical velocity to lattice velocity.
    pub fn velocity_to_lattice(&self, u_physical: [f32; 3]) -> [f32; 3] {
        let c_inv = self.dt / self.dx;
        [
            u_physical[0] * c_inv,
            u_physical[1] * c_inv,
            u_physical[2] * c_inv,
        ]
    }

    /// Convert lattice density to physical pressure (Pa).
    /// P = ρ_lattice × cs² × (dx/dt)²
    pub fn pressure_to_physical(&self, rho_lattice: f32) -> f32 {
        let c_sq = (self.dx / self.dt) * (self.dx / self.dt);
        rho_lattice * CS2 * c_sq
    }

    /// Convert physical force density (N/m³) to lattice force density.
    /// F_lattice = F_physical × dt² / dx
    pub fn force_to_lattice(&self, f_physical: [f32; 3]) -> [f32; 3] {
        let scale = self.dt * self.dt / self.dx;
        [
            f_physical[0] * scale,
            f_physical[1] * scale,
            f_physical[2] * scale,
        ]
    }
}

impl Default for LbmScaling {
    fn default() -> Self {
        Self { dx: 1.0, dt: 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one() {
        let sum: f32 = W.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "D3Q19 weights sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn opposite_directions_are_inverse() {
        for i in 0..Q {
            let j = OPPOSITE[i];
            assert!(j < Q, "OPPOSITE[{i}] = {j} out of range");
            // E[j] should be -E[i]
            for (d, (&ej_d, &ei_d)) in E[j].iter().zip(E[i].iter()).enumerate() {
                assert_eq!(
                    ej_d, -ei_d,
                    "E[OPPOSITE[{i}]][{d}] = {ej_d}, expected {}",
                    -ei_d
                );
            }
        }
    }

    #[test]
    fn opposite_is_involution() {
        for i in 0..Q {
            assert_eq!(
                OPPOSITE[OPPOSITE[i]], i,
                "OPPOSITE[OPPOSITE[{i}]] = {}, expected {i}",
                OPPOSITE[OPPOSITE[i]]
            );
        }
    }

    #[test]
    fn lattice_symmetry_first_moment_vanishes() {
        // Σ w_i × e_i = 0 (isotropy condition)
        let mut sum = [0.0f32; 3];
        for i in 0..Q {
            for d in 0..3 {
                sum[d] += W[i] * E[i][d] as f32;
            }
        }
        for (d, &val) in sum.iter().enumerate() {
            assert!(
                val.abs() < 1e-7,
                "First moment sum[{d}] = {val}, expected 0",
            );
        }
    }

    #[test]
    fn lattice_symmetry_second_moment_is_cs2() {
        // Σ w_i × e_iα × e_iβ = cs² × δ_αβ
        let mut moments = [[0.0f32; 3]; 3];
        for (&w_i, e_i) in W.iter().zip(E.iter()) {
            for (m_row, &ea) in moments.iter_mut().zip(e_i.iter()) {
                for (m_val, &eb) in m_row.iter_mut().zip(e_i.iter()) {
                    *m_val += w_i * ea as f32 * eb as f32;
                }
            }
        }
        for (a, row) in moments.iter().enumerate() {
            for (b, &sum) in row.iter().enumerate() {
                let expected = if a == b { CS2 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-6,
                    "Second moment [{a}][{b}] = {sum}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn equilibrium_at_rest_recovers_density() {
        let rho = 1.225;
        let f_eq = equilibrium(rho, [0.0, 0.0, 0.0]);
        let rho_recovered: f32 = f_eq.iter().sum();
        assert!(
            (rho_recovered - rho).abs() < 1e-5,
            "Recovered ρ = {rho_recovered}, expected {rho}"
        );
    }

    #[test]
    fn equilibrium_recovers_momentum() {
        let rho = 1.0;
        let u = [0.05, -0.03, 0.02];
        let f_eq = equilibrium(rho, u);

        let rho_rec: f32 = f_eq.iter().sum();
        let mut u_rec = [0.0f32; 3];
        for i in 0..Q {
            for d in 0..3 {
                u_rec[d] += f_eq[i] * E[i][d] as f32;
            }
        }
        // Divide by density to get velocity
        for val in &mut u_rec {
            *val /= rho_rec;
        }

        assert!(
            (rho_rec - rho).abs() < 1e-5,
            "Recovered ρ = {rho_rec}, expected {rho}"
        );
        for d in 0..3 {
            assert!(
                (u_rec[d] - u[d]).abs() < 1e-5,
                "Recovered u[{d}] = {}, expected {}",
                u_rec[d],
                u[d]
            );
        }
    }

    #[test]
    fn equilibrium_is_non_negative_for_small_velocity() {
        let rho = 1.0;
        let u = [0.01, 0.01, 0.01];
        let f_eq = equilibrium(rho, u);
        for (i, &fi) in f_eq.iter().enumerate() {
            assert!(fi >= 0.0, "f_eq[{i}] = {fi} is negative for small velocity");
        }
    }

    #[test]
    fn scaling_from_tau_and_viscosity() {
        // Air: ν_eff ≈ 0.005 m²/s (with Smagorinsky), τ = 0.55, dx = 1.0 m
        let s = LbmScaling::from_tau_and_viscosity(1.0, 0.55, 0.005);
        // ν_lattice = (1/3)(0.55-0.5) = 1/60 ≈ 0.01667
        // dt = 0.01667 × 1.0 / 0.005 ≈ 3.333 s
        assert!(
            (s.dt - 10.0 / 3.0).abs() < 0.01,
            "dt = {}, expected ~3.33",
            s.dt
        );
    }

    #[test]
    fn velocity_conversion_roundtrip() {
        let s = LbmScaling { dx: 1.0, dt: 2.0 };
        let u_phys = [10.0, -5.0, 3.0];
        let u_lat = s.velocity_to_lattice(u_phys);
        let u_back = s.velocity_to_physical(u_lat);
        for d in 0..3 {
            assert!(
                (u_back[d] - u_phys[d]).abs() < 1e-5,
                "Roundtrip failed for component {d}"
            );
        }
    }

    #[test]
    fn pressure_conversion() {
        // At rest (ρ_lattice = 1.0), P_lattice = cs² = 1/3
        // P_physical = (1/3) × (dx/dt)²
        let s = LbmScaling { dx: 1.0, dt: 1.0 };
        let p = s.pressure_to_physical(1.0);
        assert!((p - CS2).abs() < 1e-6, "Pressure = {p}, expected {CS2}");
    }

    #[test]
    fn equilibrium_mass_conservation_across_densities() {
        for &rho in &[0.5, 1.0, 1.225, 2.0, 5.0] {
            let f_eq = equilibrium(rho, [0.01, -0.01, 0.005]);
            let sum: f32 = f_eq.iter().sum();
            assert!(
                (sum - rho).abs() < 1e-4,
                "Mass not conserved at ρ={rho}: sum={sum}"
            );
        }
    }

    #[test]
    fn d3q19_has_correct_count() {
        assert_eq!(Q, 19);
        assert_eq!(E.len(), 19);
        assert_eq!(W.len(), 19);
        assert_eq!(OPPOSITE.len(), 19);
    }
}
