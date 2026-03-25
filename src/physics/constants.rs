// Physics constants — split into universal and world-specific.
//
// Universal constants (Stefan-Boltzmann, gas constant, etc.) are loaded
// from `assets/data/universal_constants.ron` and never change between worlds.
//
// World constants (gravity, atmosphere, etc.) are loaded from
// `assets/data/world_constants.ron` and can be overridden per world.
//
// `Default` impls provide NIST/ISO reference values for tests and non-ECS code.
// Backward-compatible module-level `pub const` items and free functions are
// retained so existing callers (`constants::GRAVITY`, etc.) keep working.
//
// Unit system: strict SI (meters, kilograms, seconds, Kelvin, Pascals, etc.)
// Spatial mapping: 1 voxel = 1 meter (configurable via `voxel_size`).
//
// Sources:
//   - NIST CODATA 2018 (fundamental constants)
//   - ISO 2533:1975 (standard atmosphere / Earth defaults)

use bevy::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Universal constants — same in any world
// ---------------------------------------------------------------------------

/// Universal physical constants loaded from `universal_constants.ron`.
///
/// These are fundamental constants of nature that do not vary between worlds.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, Resource)]
pub struct UniversalConstants {
    /// Stefan–Boltzmann constant (W/(m²·K⁴)). NIST CODATA 2018.
    pub stefan_boltzmann: f64,
    /// Universal (molar) gas constant (J/(mol·K)). NIST CODATA 2018.
    pub gas_constant: f32,
    /// Triple point of water (K). NIST ITS-90.
    pub water_triple_point: f32,
    /// Speed of light in vacuum (m/s). NIST CODATA 2018 (exact).
    #[serde(default = "default_speed_of_light")]
    pub speed_of_light: f64,
}

fn default_speed_of_light() -> f64 {
    299_792_458.0
}

impl Default for UniversalConstants {
    fn default() -> Self {
        Self {
            stefan_boltzmann: 5.670_374_419e-8,
            gas_constant: 8.314_462,
            water_triple_point: 273.16,
            speed_of_light: 299_792_458.0,
        }
    }
}

impl UniversalConstants {
    /// Absolute zero (K). Lower bound for all temperatures.
    pub const ABSOLUTE_ZERO: f32 = 0.0;
}

// ---------------------------------------------------------------------------
// World constants — vary per planet / world
// ---------------------------------------------------------------------------

/// World-specific constants loaded from `world_constants.ron`.
///
/// These define the physical environment of a particular world.
/// Defaults are Earth values from ISO 2533:1975.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, Resource)]
pub struct WorldConstants {
    /// Surface gravitational acceleration (m/s²). Earth: 9.80665.
    pub gravity: f32,
    /// Atmospheric pressure at sea level (Pa). Earth: 101325.
    pub atmospheric_pressure: f32,
    /// Mean molar mass of atmosphere (kg/mol). Earth air: 0.0289647.
    pub molar_mass_air: f32,
    /// Sea-level temperature (K). Earth: 288.15 (15 °C).
    pub sea_level_temperature: f32,
    /// Air density at sea level (kg/m³). Earth: 1.225.
    pub air_density_sea_level: f32,
    /// Sea-level Y coordinate in voxel world (voxels above y=0).
    pub sea_level_y: f32,
    /// Voxel edge length (m). Defines spatial scale.
    pub voxel_size: f32,
}

impl Default for WorldConstants {
    fn default() -> Self {
        Self {
            gravity: 9.806_65,
            atmospheric_pressure: 101_325.0,
            molar_mass_air: 0.028_964_7,
            sea_level_temperature: 288.15,
            air_density_sea_level: 1.225,
            sea_level_y: 64.0,
            voxel_size: 1.0,
        }
    }
}

impl WorldConstants {
    /// Voxel face area (m²).
    pub fn voxel_face_area(&self) -> f32 {
        self.voxel_size * self.voxel_size
    }

    /// Voxel volume (m³).
    pub fn voxel_volume(&self) -> f32 {
        self.voxel_size * self.voxel_size * self.voxel_size
    }

    /// Default ambient temperature (K). Same as `sea_level_temperature`.
    pub fn ambient_temperature(&self) -> f32 {
        self.sea_level_temperature
    }

    /// Mass of a voxel given material density (kg).
    pub fn voxel_mass(&self, density_kg_m3: f32) -> f32 {
        density_kg_m3 * self.voxel_volume()
    }
}

// ---------------------------------------------------------------------------
// Combined helpers that need both universal + world constants
// ---------------------------------------------------------------------------

/// Barometric formula: pressure at a given altitude above sea level.
///
/// `P(h) = P₀ × exp(−M × g × h / (R × T))`
pub fn pressure_at_altitude_full(
    world: &WorldConstants,
    universal: &UniversalConstants,
    altitude: f32,
    temperature_k: f32,
) -> f32 {
    let exponent = -(world.molar_mass_air * world.gravity * altitude)
        / (universal.gas_constant * temperature_k);
    world.atmospheric_pressure * exponent.exp()
}

/// Air density via ideal gas law: ρ = PM / (RT).
pub fn air_density_full(
    world: &WorldConstants,
    universal: &UniversalConstants,
    pressure_pa: f32,
    temperature_k: f32,
) -> f32 {
    (pressure_pa * world.molar_mass_air) / (universal.gas_constant * temperature_k)
}

// ---------------------------------------------------------------------------
// Backward-compatible module-level constants (Earth defaults)
// ---------------------------------------------------------------------------

/// Standard gravitational acceleration at Earth's surface (m/s²).
pub const GRAVITY: f32 = 9.806_65;

/// Standard atmospheric pressure at sea level (Pa).
pub const ATMOSPHERIC_PRESSURE: f32 = 101_325.0;

/// Stefan–Boltzmann constant (W/(m²·K⁴)).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Speed of light in vacuum (m/s). NIST CODATA 2018 (exact).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Universal (molar) gas constant (J/(mol·K)).
pub const GAS_CONSTANT: f32 = 8.314_462;

/// Mean molar mass of dry air (kg/mol).
pub const MOLAR_MASS_AIR: f32 = 0.028_964_7;

/// Absolute zero (K).
pub const ABSOLUTE_ZERO: f32 = 0.0;

/// Triple point of water (K).
pub const WATER_TRIPLE_POINT: f32 = 273.16;

/// Standard sea-level temperature (K).
pub const SEA_LEVEL_TEMPERATURE: f32 = 288.15;

/// Voxel edge length (m).
pub const VOXEL_SIZE: f32 = 1.0;

/// Voxel face area (m²).
pub const VOXEL_FACE_AREA: f32 = VOXEL_SIZE * VOXEL_SIZE;

/// Voxel volume (m³).
pub const VOXEL_VOLUME: f32 = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;

/// Sea-level Y coordinate in the voxel world.
pub const SEA_LEVEL_Y: f32 = 64.0;

/// Default ambient temperature (K).
pub const AMBIENT_TEMPERATURE: f32 = SEA_LEVEL_TEMPERATURE;

/// Density of dry air at sea level and 15 °C (kg/m³).
pub const AIR_DENSITY_SEA_LEVEL: f32 = 1.225;

// ---------------------------------------------------------------------------
// Backward-compatible free functions (delegate to defaults)
// ---------------------------------------------------------------------------

/// Barometric formula: pressure at a given altitude above sea level.
pub fn pressure_at_altitude(altitude_above_sea_level: f32, temperature_k: f32) -> f32 {
    pressure_at_altitude_full(
        &WorldConstants::default(),
        &UniversalConstants::default(),
        altitude_above_sea_level,
        temperature_k,
    )
}

/// Air density at a given pressure and temperature via the ideal gas law.
pub fn air_density(pressure_pa: f32, temperature_k: f32) -> f32 {
    air_density_full(
        &WorldConstants::default(),
        &UniversalConstants::default(),
        pressure_pa,
        temperature_k,
    )
}

/// Mass of a voxel given its material density (kg).
pub fn voxel_mass(density_kg_m3: f32) -> f32 {
    WorldConstants::default().voxel_mass(density_kg_m3)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_matches_nist() {
        assert!((GRAVITY - 9.806_65).abs() < 1e-5);
    }

    #[test]
    fn atmospheric_pressure_matches_standard() {
        assert_eq!(ATMOSPHERIC_PRESSURE, 101_325.0);
    }

    #[test]
    fn barometric_formula_at_sea_level_returns_standard() {
        let p = pressure_at_altitude(0.0, SEA_LEVEL_TEMPERATURE);
        assert!((p - ATMOSPHERIC_PRESSURE).abs() < 1.0);
    }

    #[test]
    fn barometric_formula_at_1000m() {
        let p = pressure_at_altitude(1000.0, SEA_LEVEL_TEMPERATURE);
        assert!(
            (p - 89_875.0).abs() < 500.0,
            "Got {p} Pa, expected ~89875 Pa"
        );
    }

    #[test]
    fn air_density_at_sea_level() {
        let rho = air_density(ATMOSPHERIC_PRESSURE, SEA_LEVEL_TEMPERATURE);
        assert!(
            (rho - 1.225).abs() < 0.01,
            "Got {rho} kg/m³, expected ~1.225"
        );
    }

    #[test]
    fn voxel_mass_for_water() {
        let m = voxel_mass(1000.0);
        assert_eq!(m, 1000.0);
    }

    #[test]
    fn voxel_mass_for_iron() {
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
        let water_density = 1000.0_f32;
        let depth = 10.0_f32;
        let p_hydro = water_density * GRAVITY * depth;
        assert!(
            (p_hydro - 98_066.5).abs() < 1.0,
            "Got {p_hydro} Pa, expected ~98066.5"
        );
    }

    // --- Struct tests ---

    #[test]
    fn universal_default_matches_module_constants() {
        let u = UniversalConstants::default();
        assert_eq!(u.stefan_boltzmann, STEFAN_BOLTZMANN);
        assert_eq!(u.gas_constant, GAS_CONSTANT);
        assert_eq!(u.water_triple_point, WATER_TRIPLE_POINT);
        assert_eq!(u.speed_of_light, SPEED_OF_LIGHT);
    }

    #[test]
    fn world_default_matches_module_constants() {
        let w = WorldConstants::default();
        assert_eq!(w.gravity, GRAVITY);
        assert_eq!(w.atmospheric_pressure, ATMOSPHERIC_PRESSURE);
        assert_eq!(w.molar_mass_air, MOLAR_MASS_AIR);
        assert_eq!(w.sea_level_temperature, SEA_LEVEL_TEMPERATURE);
        assert_eq!(w.air_density_sea_level, AIR_DENSITY_SEA_LEVEL);
        assert_eq!(w.sea_level_y, SEA_LEVEL_Y);
        assert_eq!(w.voxel_size, VOXEL_SIZE);
    }

    #[test]
    fn world_derived_values_match_constants() {
        let w = WorldConstants::default();
        assert_eq!(w.voxel_face_area(), VOXEL_FACE_AREA);
        assert_eq!(w.voxel_volume(), VOXEL_VOLUME);
        assert_eq!(w.ambient_temperature(), AMBIENT_TEMPERATURE);
    }

    #[test]
    fn full_functions_match_free_functions() {
        let w = WorldConstants::default();
        let u = UniversalConstants::default();
        assert_eq!(
            pressure_at_altitude_full(&w, &u, 1000.0, 288.15),
            pressure_at_altitude(1000.0, 288.15)
        );
        assert_eq!(
            air_density_full(&w, &u, 101_325.0, 288.15),
            air_density(101_325.0, 288.15)
        );
        assert_eq!(w.voxel_mass(1000.0), voxel_mass(1000.0));
    }

    #[test]
    fn universal_ron_file_parses_to_defaults() {
        let ron_str = std::fs::read_to_string("assets/data/universal_constants.ron")
            .expect("universal_constants.ron should exist");
        let parsed: UniversalConstants =
            ron::from_str(&ron_str).expect("RON should deserialize into UniversalConstants");
        let defaults = UniversalConstants::default();
        assert_eq!(parsed.stefan_boltzmann, defaults.stefan_boltzmann);
        assert_eq!(parsed.gas_constant, defaults.gas_constant);
        assert_eq!(parsed.water_triple_point, defaults.water_triple_point);
        assert_eq!(parsed.speed_of_light, defaults.speed_of_light);
    }

    #[test]
    fn world_ron_file_parses_to_defaults() {
        let ron_str = std::fs::read_to_string("assets/data/world_constants.ron")
            .expect("world_constants.ron should exist");
        let parsed: WorldConstants =
            ron::from_str(&ron_str).expect("RON should deserialize into WorldConstants");
        let defaults = WorldConstants::default();
        assert_eq!(parsed.gravity, defaults.gravity);
        assert_eq!(parsed.atmospheric_pressure, defaults.atmospheric_pressure);
        assert_eq!(parsed.molar_mass_air, defaults.molar_mass_air);
        assert_eq!(parsed.sea_level_temperature, defaults.sea_level_temperature);
        assert_eq!(parsed.air_density_sea_level, defaults.air_density_sea_level);
        assert_eq!(parsed.sea_level_y, defaults.sea_level_y);
        assert_eq!(parsed.voxel_size, defaults.voxel_size);
    }
}
