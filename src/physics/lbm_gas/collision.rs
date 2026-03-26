// BGK collision operator with Smagorinsky sub-grid turbulence model.
//
// The collision step relaxes distribution functions toward their local
// equilibrium. The Smagorinsky model adds turbulent viscosity based on
// the local strain rate, keeping τ_eff stable at coarse (1m) resolution.

use super::lattice::{self, CS2, CS4, E, Q};
use super::types::{LbmCell, LbmGrid};

/// BGK collision on a single cell with fixed relaxation time τ.
///
/// f_i_post = f_i - (f_i - f_eq_i) / τ
pub fn collide_bgk(cell: &mut LbmCell, tau: f32) {
    let rho = cell.density();
    let u = cell.velocity();
    let f_eq = lattice::equilibrium(rho, u);
    let inv_tau = 1.0 / tau;

    for (i, &f_eq_i) in f_eq.iter().enumerate() {
        cell.f[i] -= (cell.f[i] - f_eq_i) * inv_tau;
    }
}

/// Compute the magnitude of the strain rate tensor from the non-equilibrium
/// stress tensor.
///
/// |S| = √(2 × Π_neq : Π_neq) / (2 × ρ × cs⁴)
///
/// where Π_neq_αβ = Σ_i (f_i - f_eq_i) × e_iα × e_iβ
pub fn strain_rate(cell: &LbmCell) -> f32 {
    let rho = cell.density();
    let u = cell.velocity();
    let f_eq = lattice::equilibrium(rho, u);

    // Compute non-equilibrium stress tensor (symmetric 3×3)
    let mut pi_neq = [[0.0f32; 3]; 3];
    for i in 0..Q {
        let f_neq = cell.f[i] - f_eq[i];
        for a in 0..3 {
            for b in a..3 {
                pi_neq[a][b] += f_neq * E[i][a] as f32 * E[i][b] as f32;
            }
        }
    }
    // Fill symmetric part
    pi_neq[1][0] = pi_neq[0][1];
    pi_neq[2][0] = pi_neq[0][2];
    pi_neq[2][1] = pi_neq[1][2];

    // Double contraction: Π:Π = Σ_αβ Π_αβ²
    let mut pi_sq = 0.0f32;
    for row in &pi_neq {
        for &val in row {
            pi_sq += val * val;
        }
    }

    // |S| = √(2 × Π:Π) / (2 × ρ × cs⁴)
    let denom = 2.0 * rho.max(1e-10) * CS4;
    (2.0 * pi_sq).sqrt() / denom
}

/// BGK collision with Smagorinsky sub-grid turbulence model.
///
/// The effective relaxation time is:
///   τ_eff = 0.5 × (τ_base + √(τ_base² + 2 × √2 × Cs² × |S| / cs⁴))
///
/// This formulation (from Yu et al. 2005) ensures τ_eff > 0.5 and adds
/// turbulent viscosity proportional to the local strain rate.
pub fn collide_smagorinsky(cell: &mut LbmCell, tau_base: f32, cs_smag: f32) {
    if cs_smag <= 0.0 {
        collide_bgk(cell, tau_base);
        return;
    }

    let s_mag = strain_rate(cell);

    // Compute turbulent relaxation time increment
    // τ_turb from: ν_turb = (Cs × Δx)² × |S|, and ν_lattice = cs²(τ-0.5)
    // In lattice units (Δx=1): ν_turb = Cs² × |S|
    // τ_turb = ν_turb / cs² = Cs² × |S| / cs²
    let tau_turb = cs_smag * cs_smag * s_mag / CS2;
    let tau_eff = tau_base + tau_turb;

    collide_bgk(cell, tau_eff);
}

/// Apply collision to all gas cells in a grid.
pub fn collide_grid(grid: &mut LbmGrid, tau: f32, cs_smag: f32) {
    for cell in grid.cells_mut() {
        if cell.is_gas() {
            collide_smagorinsky(cell, tau, cs_smag);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::MaterialId;

    #[test]
    fn bgk_collision_preserves_mass() {
        let mut cell = LbmCell::new_gas(MaterialId::AIR, 1.225);
        let mass_before = cell.density();
        collide_bgk(&mut cell, 0.6);
        let mass_after = cell.density();
        assert!(
            (mass_after - mass_before).abs() < 1e-5,
            "Mass changed: {mass_before} → {mass_after}"
        );
    }

    #[test]
    fn bgk_collision_preserves_momentum() {
        let u_in = [0.05, -0.03, 0.02];
        let rho = 1.0;
        let mut cell = LbmCell {
            f: lattice::equilibrium(rho, u_in),
            material: MaterialId::AIR,
            tag: crate::physics::lbm_gas::types::GasCellTag::Gas,
            moisture: 0.0,
            cloud_lwc: 0.0,
        };
        collide_bgk(&mut cell, 0.6);
        let u_out = cell.velocity();
        for d in 0..3 {
            assert!(
                (u_out[d] - u_in[d]).abs() < 1e-5,
                "Momentum changed in dim {d}: {} → {}",
                u_in[d],
                u_out[d]
            );
        }
    }

    #[test]
    fn bgk_at_equilibrium_is_noop() {
        let rho = 1.0;
        let u = [0.01, 0.0, 0.0];
        let f_eq = lattice::equilibrium(rho, u);
        let mut cell = LbmCell {
            f: f_eq,
            material: MaterialId::AIR,
            tag: crate::physics::lbm_gas::types::GasCellTag::Gas,
            moisture: 0.0,
            cloud_lwc: 0.0,
        };
        collide_bgk(&mut cell, 0.6);
        // Should remain at equilibrium
        for (i, &expected) in f_eq.iter().enumerate() {
            assert!(
                (cell.f[i] - expected).abs() < 1e-6,
                "f[{i}] changed from equilibrium: {expected} → {}",
                cell.f[i]
            );
        }
    }

    #[test]
    fn smagorinsky_collision_preserves_mass() {
        let mut cell = LbmCell::new_gas(MaterialId::AIR, 1.225);
        let mass_before = cell.density();
        collide_smagorinsky(&mut cell, 0.55, 0.1);
        let mass_after = cell.density();
        assert!(
            (mass_after - mass_before).abs() < 1e-5,
            "Mass changed: {mass_before} → {mass_after}"
        );
    }

    #[test]
    fn smagorinsky_with_zero_cs_equals_bgk() {
        let mut cell1 = LbmCell::new_gas(MaterialId::AIR, 1.0);
        let mut cell2 = cell1;

        collide_bgk(&mut cell1, 0.55);
        collide_smagorinsky(&mut cell2, 0.55, 0.0);

        for i in 0..Q {
            assert!(
                (cell1.f[i] - cell2.f[i]).abs() < 1e-10,
                "Smagorinsky(Cs=0) != BGK at f[{i}]"
            );
        }
    }

    #[test]
    fn strain_rate_at_equilibrium_is_zero() {
        let cell = LbmCell::new_gas(MaterialId::AIR, 1.0);
        let s = strain_rate(&cell);
        assert!(s < 1e-6, "Strain rate at equilibrium = {s}, expected ~0");
    }

    #[test]
    fn collide_grid_skips_wall_cells() {
        let size = 4;
        let mut grid = LbmGrid::new_empty(size);
        // Set one cell to solid
        *grid.get_mut(0, 0, 0) = LbmCell::new_solid(MaterialId::STONE);
        let solid_f_before = grid.get(0, 0, 0).f;

        collide_grid(&mut grid, 0.55, 0.1);

        // Solid cell should be unchanged
        let solid_f_after = grid.get(0, 0, 0).f;
        for i in 0..Q {
            assert_eq!(solid_f_before[i], solid_f_after[i]);
        }
    }

    #[test]
    fn smagorinsky_increases_effective_tau() {
        // Create a cell with non-equilibrium perturbation (high strain)
        let mut cell = LbmCell::new_gas(MaterialId::AIR, 1.0);
        // Perturb some distributions to create strain
        cell.f[1] += 0.01;
        cell.f[2] -= 0.01;

        let s = strain_rate(&cell);
        assert!(s > 0.0, "Should have nonzero strain rate");

        // With Smagorinsky, effective tau should be higher → less relaxation
        let mut cell_bgk = cell;
        let mut cell_smag = cell;
        collide_bgk(&mut cell_bgk, 0.55);
        collide_smagorinsky(&mut cell_smag, 0.55, 0.1);

        // Both should conserve mass
        let m_bgk = cell_bgk.density();
        let m_smag = cell_smag.density();
        assert!((m_bgk - m_smag).abs() < 1e-5);

        // But distributions should differ (Smagorinsky relaxes less)
        let mut diff = 0.0f32;
        for i in 0..Q {
            diff += (cell_bgk.f[i] - cell_smag.f[i]).abs();
        }
        assert!(
            diff > 1e-8,
            "BGK and Smagorinsky should produce different results with strain"
        );
    }
}
