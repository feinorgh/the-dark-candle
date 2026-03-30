//! Geological layering, surface rock classification, and ore deposit placement.
//!
//! Surface rock type and geological age are derived deterministically from the
//! tectonic context (plate boundary type, crust type, volcanic activity, elevation).
//! Ore deposits are placed with per-cell deterministic RNG seeded from the planet
//! seed, with probabilities scaled by geological plausibility.
//!
//! Call `run_geology` **after** `run_biomes`; coal placement uses biome data.

use super::{BiomeType, BoundaryType, CrustType, PlanetData, RockType};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ─── Ore deposit bitmask constants ───────────────────────────────────────────

/// Iron ore (magnetite, haematite) — continental sedimentary basins.
pub const ORE_IRON: u16 = 1 << 0;
/// Copper ore (chalcopyrite) — volcanic arcs, porphyry deposits.
pub const ORE_COPPER: u16 = 1 << 1;
/// Gold — hydrothermal veins at faults and felsic intrusions.
pub const ORE_GOLD: u16 = 1 << 2;
/// Coal — ancient organic-rich sedimentary strata below former forests.
pub const ORE_COAL: u16 = 1 << 3;
/// Native sulfur — volcanic fumaroles and crater lakes.
pub const ORE_SULFUR: u16 = 1 << 4;
/// Gemstones (ruby, sapphire, garnet) — high-grade metamorphic zones.
pub const ORE_GEMS: u16 = 1 << 5;
/// Petroleum / natural gas — deep sedimentary basins.
pub const ORE_OIL: u16 = 1 << 6;

// ─── Public entry point ───────────────────────────────────────────────────────

/// Assign surface rock types, geological ages, and ore deposits to all cells.
///
/// Populates `data.surface_rock`, `data.geological_age`, and `data.ore_deposits`.
/// Requires `run_tectonics` (for plate data) and `run_biomes` (for biome data
/// used in coal placement) to have been called first.
pub fn run_geology(data: &mut PlanetData) {
    let surface_rock = assign_surface_rocks(data);
    let geological_age = compute_geological_age(data);
    let ore_deposits = place_ore_deposits(data, &surface_rock);
    data.surface_rock = surface_rock;
    data.geological_age = geological_age;
    data.ore_deposits = ore_deposits;
}

// ─── Surface rock assignment ──────────────────────────────────────────────────

/// Assign the dominant surface rock type to each cell from tectonic context.
fn assign_surface_rocks(data: &PlanetData) -> Vec<RockType> {
    let n = data.grid.cell_count();
    let mut rocks = Vec::with_capacity(n);
    for i in 0..n {
        let bt = data.boundary_type[i];
        let ct = data.crust_type[i];
        let va = data.volcanic_activity[i];
        let elev = data.elevation[i];

        let rock = if elev < -3000.0 {
            // Deep ocean: mantle-derived or fresh basaltic crust.
            if bt == BoundaryType::Divergent {
                RockType::Peridotite
            } else {
                RockType::Basalt
            }
        } else if va > 0.8 {
            // Highly active volcano: silica-rich glass or fresh basalt.
            RockType::Obsidian
        } else if va > 0.4 {
            RockType::Basalt
        } else if bt == BoundaryType::Divergent {
            if elev < 0.0 {
                RockType::Basalt
            } else {
                RockType::Peridotite // exposed upper mantle at rift shoulders
            }
        } else if bt == BoundaryType::Convergent {
            // Mountain belts: metamorphic dominates.
            if ct == CrustType::Continental {
                if elev > 3000.0 {
                    RockType::Quartzite // high-grade exposed cores
                } else {
                    RockType::Gneiss
                }
            } else {
                // Oceanic subduction → accretionary wedge: marble, shale.
                RockType::Marble
            }
        } else if bt == BoundaryType::Transform {
            // Shear zones expose basement rocks.
            RockType::Quartzite
        } else {
            // Plate interior.
            if ct == CrustType::Oceanic {
                RockType::Basalt
            } else if elev < 0.0 {
                RockType::Limestone // shallow-sea carbonate platform
            } else if elev < 200.0 {
                RockType::Shale // low-lying alluvial plains
            } else if elev < 1500.0 {
                RockType::Sandstone // upland sedimentary cover
            } else {
                RockType::Granite // elevated exposed basement
            }
        };
        rocks.push(rock);
    }
    rocks
}

// ─── Geological age ───────────────────────────────────────────────────────────

/// Compute geological age per cell (0.0 = ancient craton, 1.0 = freshly formed).
///
/// Divergent boundaries produce the youngest crust; continental interiors are the
/// oldest.  Volcanic activity adds a youth bias (fresh lava).  A small
/// deterministic per-cell noise term breaks up uniform regions.
fn compute_geological_age(data: &PlanetData) -> Vec<f32> {
    let n = data.grid.cell_count();
    let mut age = Vec::with_capacity(n);
    for i in 0..n {
        let base: f64 = match data.boundary_type[i] {
            BoundaryType::Divergent => 0.85,
            BoundaryType::Convergent => 0.50,
            BoundaryType::Transform => 0.55,
            BoundaryType::Interior => {
                if data.crust_type[i] == CrustType::Continental {
                    0.15
                } else {
                    0.65
                }
            }
        };
        // Volcanic activity skews toward recent formation.
        let volcanic_boost = data.volcanic_activity[i] as f64 * 0.25;
        // Small deterministic noise to break uniform zones.
        let noise = cell_noise(data.config.seed, i) * 0.10 - 0.05;
        let a = (base + volcanic_boost + noise).clamp(0.0, 1.0) as f32;
        age.push(a);
    }
    age
}

// ─── Ore placement ────────────────────────────────────────────────────────────

/// Place ore deposits with plausible geological probabilities.
///
/// Each cell gets an independent `SmallRng` seeded from the planet seed and
/// cell index, ensuring fully deterministic placement.
fn place_ore_deposits(data: &PlanetData, rocks: &[RockType]) -> Vec<u16> {
    let n = data.grid.cell_count();
    let mut deposits = vec![0_u16; n];

    for i in 0..n {
        let mut rng = SmallRng::seed_from_u64(cell_seed(data.config.seed, i));
        let bt = data.boundary_type[i];
        let ct = data.crust_type[i];
        let va = data.volcanic_activity[i];
        let elev = data.elevation[i];
        let rock = rocks[i];
        let biome = if data.biome.is_empty() {
            BiomeType::default()
        } else {
            data.biome[i]
        };

        let is_land = elev >= 0.0;

        // ── Iron: continental sedimentary basins ──────────────────────────────
        if is_land
            && ct == CrustType::Continental
            && bt == BoundaryType::Interior
            && matches!(
                rock,
                RockType::Sandstone | RockType::Shale | RockType::Limestone
            )
            && rng.random_range(0.0_f32..1.0_f32) < 0.10
        {
            deposits[i] |= ORE_IRON;
        }

        // ── Copper: porphyry deposits at volcanic arcs ────────────────────────
        if is_land
            && bt == BoundaryType::Convergent
            && va > 0.20
            && rng.random_range(0.0_f32..1.0_f32) < 0.12
        {
            deposits[i] |= ORE_COPPER;
        }

        // ── Gold: hydrothermal veins at faults and felsic intrusions ──────────
        if is_land
            && (bt == BoundaryType::Transform
                || matches!(rock, RockType::Granite | RockType::Gneiss))
            && rng.random_range(0.0_f32..1.0_f32) < 0.04
        {
            deposits[i] |= ORE_GOLD;
        }

        // ── Coal: former-forest sedimentary basins ────────────────────────────
        if is_land
            && ct == CrustType::Continental
            && matches!(
                biome,
                BiomeType::BorealForest
                    | BiomeType::TemperateForest
                    | BiomeType::TropicalRainforest
                    | BiomeType::TropicalSavanna
            )
            && matches!(
                rock,
                RockType::Shale | RockType::Sandstone | RockType::Limestone
            )
            && rng.random_range(0.0_f32..1.0_f32) < 0.07
        {
            deposits[i] |= ORE_COAL;
        }

        // ── Sulfur: volcanic fumaroles ────────────────────────────────────────
        if va > 0.40 && rng.random_range(0.0_f32..1.0_f32) < 0.18 {
            deposits[i] |= ORE_SULFUR;
        }

        // ── Gemstones: high-grade metamorphic zones ───────────────────────────
        if is_land
            && bt == BoundaryType::Convergent
            && ct == CrustType::Continental
            && matches!(
                rock,
                RockType::Marble | RockType::Quartzite | RockType::Gneiss
            )
            && rng.random_range(0.0_f32..1.0_f32) < 0.05
        {
            deposits[i] |= ORE_GEMS;
        }

        // ── Oil / gas: deep sedimentary basins ────────────────────────────────
        if is_land
            && ct == CrustType::Continental
            && bt == BoundaryType::Interior
            && elev < 300.0
            && data.fault_stress[i] < 0.30
            && matches!(rock, RockType::Shale | RockType::Limestone)
            && rng.random_range(0.0_f32..1.0_f32) < 0.05
        {
            deposits[i] |= ORE_OIL;
        }
    }

    deposits
}

// ─── RNG helpers ─────────────────────────────────────────────────────────────

/// Per-cell deterministic noise in [0, 1).
fn cell_noise(seed: u64, i: usize) -> f64 {
    let h = seed
        .wrapping_add(i as u64)
        .wrapping_mul(0x6c62_272e_07bb_0142)
        .rotate_left(17)
        ^ 0xdead_beef_cafe_babe;
    h as f64 / u64::MAX as f64
}

/// Unique 64-bit RNG seed for cell `i` under planet `seed`.
fn cell_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_add(i as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ 0x5555_5555_5555_5555
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::biomes::run_biomes;
    use crate::planet::impacts::run_impacts;
    use crate::planet::tectonics::run_tectonics;
    use crate::planet::{PlanetConfig, PlanetData};

    fn test_planet() -> PlanetData {
        let config = PlanetConfig {
            seed: 42,
            grid_level: 3,
            tectonic_steps: 30,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        run_tectonics(&mut data, |_| {});
        run_impacts(&mut data);
        run_biomes(&mut data);
        run_geology(&mut data);
        data
    }

    #[test]
    fn volcanic_cells_have_basalt_or_obsidian() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        for i in 0..n {
            if planet.volcanic_activity[i] > 0.80 && planet.elevation[i] >= 0.0 {
                assert!(
                    matches!(
                        planet.surface_rock[i],
                        RockType::Obsidian | RockType::Basalt
                    ),
                    "Highly volcanic cell {i} has rock {:?}",
                    planet.surface_rock[i]
                );
            }
        }
    }

    #[test]
    fn divergent_cells_are_geologically_young() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        let div_ages: Vec<f32> = (0..n)
            .filter(|&i| planet.boundary_type[i] == BoundaryType::Divergent)
            .map(|i| planet.geological_age[i])
            .collect();
        let int_ages: Vec<f32> = (0..n)
            .filter(|&i| {
                planet.boundary_type[i] == BoundaryType::Interior
                    && planet.crust_type[i] == CrustType::Continental
            })
            .map(|i| planet.geological_age[i])
            .collect();
        if div_ages.is_empty() || int_ages.is_empty() {
            return;
        }
        let mean_div = div_ages.iter().sum::<f32>() / div_ages.len() as f32;
        let mean_int = int_ages.iter().sum::<f32>() / int_ages.len() as f32;
        assert!(
            mean_div > mean_int,
            "Divergent mean age {mean_div:.2} should exceed interior mean age {mean_int:.2}"
        );
    }

    #[test]
    fn geological_age_in_range() {
        let planet = test_planet();
        for (i, &a) in planet.geological_age.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&a),
                "Cell {i} geological age {a} out of [0, 1]"
            );
        }
    }

    #[test]
    fn sulfur_only_at_volcanic_cells() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        for i in 0..n {
            if planet.ore_deposits[i] & ORE_SULFUR != 0 {
                assert!(
                    planet.volcanic_activity[i] > 0.40,
                    "Sulfur at cell {i} with low volcanic activity {:.2}",
                    planet.volcanic_activity[i]
                );
            }
        }
    }

    #[test]
    fn copper_only_at_convergent_volcanic_cells() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        for i in 0..n {
            if planet.ore_deposits[i] & ORE_COPPER != 0 {
                assert_eq!(
                    planet.boundary_type[i],
                    BoundaryType::Convergent,
                    "Copper at non-convergent cell {i} ({:?})",
                    planet.boundary_type[i]
                );
                assert!(planet.volcanic_activity[i] > 0.20);
            }
        }
    }

    #[test]
    fn iron_only_on_continental_interior_land() {
        let planet = test_planet();
        let n = planet.grid.cell_count();
        for i in 0..n {
            if planet.ore_deposits[i] & ORE_IRON != 0 {
                assert!(planet.elevation[i] >= 0.0, "Iron in ocean at cell {i}");
                assert_eq!(planet.crust_type[i], CrustType::Continental);
                assert_eq!(planet.boundary_type[i], BoundaryType::Interior);
            }
        }
    }

    #[test]
    fn geology_is_deterministic() {
        let p1 = test_planet();
        let p2 = test_planet();
        assert_eq!(p1.surface_rock, p2.surface_rock);
        assert_eq!(p1.geological_age, p2.geological_age);
        assert_eq!(p1.ore_deposits, p2.ore_deposits);
    }

    #[test]
    fn ore_deposits_present_on_planet() {
        let planet = test_planet();
        let any_deposits = planet.ore_deposits.iter().any(|&d| d != 0);
        assert!(
            any_deposits,
            "No ore deposits found on planet (seed=42, level=3)"
        );
    }
}
