// Universal physical constants in SI units.
//
// Single source of truth for all real-world constants used across the
// simulation. Every module that needs a physical constant imports from here
// rather than defining its own.
//
// Unit system: strict SI (meters, kilograms, seconds, Kelvin, Pascals, etc.)
// Spatial mapping: 1 voxel = 1 meter.
//
// Sources:
//   - NIST CODATA 2018 (fundamental constants)
//   - Wikipedia (derived/environmental constants)

/// Standard gravitational acceleration at Earth's surface (m/s²).
/// Source: NIST CODATA — exact by definition since 1901.
pub const GRAVITY: f32 = 9.806_65;

/// Standard atmospheric pressure at sea level (Pa).
/// Source: NIST / ISO 2533:1975 — exact by definition.
pub const ATMOSPHERIC_PRESSURE: f32 = 101_325.0;

/// Stefan–Boltzmann constant (W/(m²·K⁴)).
/// Source: NIST CODATA 2018.
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Universal (molar) gas constant (J/(mol·K)).
/// Source: NIST CODATA 2018 — exact by definition.
pub const GAS_CONSTANT: f32 = 8.314_462;

/// Mean molar mass of dry air (kg/mol).
/// Source: ISO 2533:1975 standard atmosphere.
pub const MOLAR_MASS_AIR: f32 = 0.028_964_7;

/// Absolute zero (K). Lower bound for all temperatures.
pub const ABSOLUTE_ZERO: f32 = 0.0;

/// Triple point of water (K).
/// Source: NIST — exact by definition (ITS-90 scale).
pub const WATER_TRIPLE_POINT: f32 = 273.16;

/// Standard sea-level temperature for barometric formula (K).
/// Source: ISO 2533:1975 standard atmosphere (15 °C).
pub const SEA_LEVEL_TEMPERATURE: f32 = 288.15;

/// Voxel edge length (m). Defines the spatial scale of the simulation.
pub const VOXEL_SIZE: f32 = 1.0;

/// Voxel face area (m²). Used in heat flux calculations.
pub const VOXEL_FACE_AREA: f32 = VOXEL_SIZE * VOXEL_SIZE;

/// Voxel volume (m³). Used in mass, energy, and pressure calculations.
pub const VOXEL_VOLUME: f32 = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;

/// Sea-level Y coordinate in the voxel world (voxels / meters above y=0).
pub const SEA_LEVEL_Y: f32 = 64.0;

/// Default ambient temperature (K). ~15 °C, matching standard atmosphere.
pub const AMBIENT_TEMPERATURE: f32 = SEA_LEVEL_TEMPERATURE;

/// Density of dry air at sea level and 15 °C (kg/m³).
/// Source: ISO 2533:1975 — derived from ideal gas law at standard conditions.
pub const AIR_DENSITY_SEA_LEVEL: f32 = 1.225;

// ---------------------------------------------------------------------------
// Derived helpers
// ---------------------------------------------------------------------------

/// Barometric formula: pressure at a given altitude above sea level.
///
/// `P(h) = P₀ × exp(−M × g × h / (R × T))`
///
/// where `h` is height above sea level (m), `T` is the local temperature (K).
///
/// Source: Wikipedia — Barometric formula (isothermal approximation).
pub fn pressure_at_altitude(altitude_above_sea_level: f32, temperature_k: f32) -> f32 {
    let exponent =
        -(MOLAR_MASS_AIR * GRAVITY * altitude_above_sea_level) / (GAS_CONSTANT * temperature_k);
    ATMOSPHERIC_PRESSURE * exponent.exp()
}

/// Air density at a given pressure and temperature via the ideal gas law.
///
/// `ρ = P × M / (R × T)`
///
/// Source: Wikipedia — Ideal gas law.
pub fn air_density(pressure_pa: f32, temperature_k: f32) -> f32 {
    (pressure_pa * MOLAR_MASS_AIR) / (GAS_CONSTANT * temperature_k)
}

/// Mass of a voxel given its material density (kg).
///
/// `m = ρ × V` where V = 1 m³.
pub fn voxel_mass(density_kg_m3: f32) -> f32 {
    density_kg_m3 * VOXEL_VOLUME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_matches_nist() {
        // NIST standard gravity: exactly 9.80665 m/s²
        assert!((GRAVITY - 9.806_65).abs() < 1e-5);
    }

    #[test]
    fn atmospheric_pressure_matches_standard() {
        // ISO 2533: exactly 101325 Pa
        assert_eq!(ATMOSPHERIC_PRESSURE, 101_325.0);
    }

    #[test]
    fn barometric_formula_at_sea_level_returns_standard() {
        let p = pressure_at_altitude(0.0, SEA_LEVEL_TEMPERATURE);
        assert!((p - ATMOSPHERIC_PRESSURE).abs() < 1.0);
    }

    #[test]
    fn barometric_formula_at_1000m() {
        // Wikipedia: ~89,875 Pa at 1000 m (isothermal approx at 288.15 K)
        let p = pressure_at_altitude(1000.0, SEA_LEVEL_TEMPERATURE);
        assert!(
            (p - 89_875.0).abs() < 500.0,
            "Got {p} Pa, expected ~89875 Pa"
        );
    }

    #[test]
    fn air_density_at_sea_level() {
        // ISO 2533: 1.225 kg/m³ at 101325 Pa and 288.15 K
        let rho = air_density(ATMOSPHERIC_PRESSURE, SEA_LEVEL_TEMPERATURE);
        assert!(
            (rho - 1.225).abs() < 0.01,
            "Got {rho} kg/m³, expected ~1.225"
        );
    }

    #[test]
    fn voxel_mass_for_water() {
        // 1 m³ of water = 1000 kg
        let m = voxel_mass(1000.0);
        assert_eq!(m, 1000.0);
    }

    #[test]
    fn voxel_mass_for_iron() {
        // 1 m³ of iron = 7874 kg
        let m = voxel_mass(7874.0);
        assert_eq!(m, 7874.0);
    }

    #[test]
    fn pressure_decreases_with_altitude() {
        let p_low = pressure_at_altitude(100.0, SEA_LEVEL_TEMPERATURE);
        let p_high = pressure_at_altitude(1000.0, SEA_LEVEL_TEMPERATURE);
        assert!(p_low > p_high, "Pressure should decrease with altitude");
    }

    #[test]
    fn air_density_decreases_with_altitude() {
        let p_low = pressure_at_altitude(0.0, SEA_LEVEL_TEMPERATURE);
        let p_high = pressure_at_altitude(5000.0, SEA_LEVEL_TEMPERATURE);
        let rho_low = air_density(p_low, SEA_LEVEL_TEMPERATURE);
        let rho_high = air_density(p_high, SEA_LEVEL_TEMPERATURE);
        assert!(
            rho_low > rho_high,
            "Air density should decrease with altitude"
        );
    }

    #[test]
    fn hydrostatic_pressure_at_10m_depth() {
        // Wikipedia: Hydrostatic pressure P = ρgh
        // 10 m of water: 1000 × 9.80665 × 10 = 98066.5 Pa
        let water_density = 1000.0_f32;
        let depth = 10.0_f32;
        let p_hydro = water_density * GRAVITY * depth;
        assert!(
            (p_hydro - 98_066.5).abs() < 1.0,
            "Got {p_hydro} Pa, expected ~98066.5"
        );
    }
}
