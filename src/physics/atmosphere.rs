// Atmosphere configuration and thermodynamic calculations.
//
// Defines atmospheric properties (temperature, humidity, scattering coefficients)
// and provides helper functions for saturation vapor pressure, dew point, and
// lifting condensation level calculations.
//
// All values use SI units:
//   - temperature: Kelvin
//   - pressure: Pascals
//   - humidity: kg water vapor / kg dry air (mixing ratio)
//   - altitude: meters
//   - scattering coefficients: m⁻¹
//   - solar constant: W/m²
//
// Data-driven: loaded from `assets/data/atmosphere_config.ron` via `RonAssetPlugin`.

use bevy::prelude::*;
use serde::Deserialize;

/// Latent heat of vaporization of water in J/kg.
pub const LATENT_HEAT_VAPORIZATION: f32 = 2.501e6;

/// Gas constant for water vapor in J/(kg·K).
pub const R_WATER_VAPOR: f32 = 461.5;

/// Gas constant for dry air in J/(kg·K).
pub const R_DRY_AIR: f32 = 287.05;

/// Reference saturation vapor pressure at 273.15 K (0°C) in Pascals.
pub const SATURATION_PRESSURE_REF: f32 = 611.2;

/// Reference temperature for saturation pressure in Kelvin.
pub const SATURATION_TEMP_REF: f32 = 273.15;

/// Configuration for atmospheric properties, loaded from RON.
///
/// The atmosphere model defines temperature lapse rate, humidity baseline,
/// cloud formation thresholds, and optical scattering parameters for sky
/// rendering and weather simulation.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, Resource)]
pub struct AtmosphereConfig {
    /// Surface temperature in Kelvin. Default: 288.15 K (15°C).
    pub surface_temperature: f32,

    /// Temperature lapse rate in K/km (temperature decrease with altitude).
    /// Default: 6.5 K/km (Earth standard atmosphere).
    pub lapse_rate: f32,

    /// Tropopause altitude in meters. Above this, temperature is constant.
    /// Default: 11,000 m (11 km, Earth standard).
    pub tropopause_altitude: f32,

    /// Baseline humidity (mixing ratio) in kg water vapor / kg dry air.
    /// Default: 0.01 (1% humidity by mass).
    pub humidity_baseline: f32,

    /// Enable Coriolis effect in wind calculations.
    /// Default: true.
    pub coriolis_enabled: bool,

    /// Liquid water content threshold for precipitation in kg/m³.
    /// Default: 0.3e-3 (0.3 g/m³).
    pub cloud_coalescence_threshold: f32,

    /// Rayleigh scattering coefficient at 550nm wavelength in m⁻¹.
    /// Default: 5.8e-6 (Earth atmosphere at sea level).
    pub rayleigh_scatter_coeff: f32,

    /// Mie scattering coefficient (aerosols, dust) in m⁻¹.
    /// Default: 21.0e-6 (Earth atmosphere, moderate aerosol loading).
    pub mie_scatter_coeff: f32,

    /// Solar constant (total solar irradiance) in W/m².
    /// Default: 1361.0 (Earth at 1 AU from Sun).
    pub solar_constant: f32,
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            surface_temperature: 288.15,
            lapse_rate: 6.5,
            tropopause_altitude: 11000.0,
            humidity_baseline: 0.01,
            coriolis_enabled: true,
            cloud_coalescence_threshold: 0.3e-3,
            rayleigh_scatter_coeff: 5.8e-6,
            mie_scatter_coeff: 21.0e-6,
            solar_constant: 1361.0,
        }
    }
}

/// Saturation vapor pressure of water at a given temperature.
///
/// Uses the Clausius-Clapeyron equation:
/// `e_s(T) = e₀ × exp(L_v / R_v × (1/T₀ - 1/T))`
///
/// where:
/// - e₀ = 611.2 Pa (saturation pressure at T₀)
/// - L_v = 2.501e6 J/kg (latent heat of vaporization)
/// - R_v = 461.5 J/(kg·K) (gas constant for water vapor)
/// - T₀ = 273.15 K (reference temperature)
///
/// # Arguments
/// * `temperature_k` - Temperature in Kelvin
///
/// # Returns
/// Saturation vapor pressure in Pascals.
pub fn saturation_vapor_pressure(temperature_k: f32) -> f32 {
    let lv_over_rv = LATENT_HEAT_VAPORIZATION / R_WATER_VAPOR;
    let exponent = lv_over_rv * (1.0 / SATURATION_TEMP_REF - 1.0 / temperature_k);
    SATURATION_PRESSURE_REF * exponent.exp()
}

/// Saturation humidity (mixing ratio) at a given temperature and pressure.
///
/// Uses the definition of mixing ratio at saturation:
/// `w_s = (R_d / R_v) × e_s(T) / (P - e_s(T))`
///
/// where:
/// - R_d = 287.05 J/(kg·K) (dry air gas constant)
/// - R_v = 461.5 J/(kg·K) (water vapor gas constant)
/// - e_s(T) = saturation vapor pressure at temperature T
/// - P = total atmospheric pressure
///
/// # Arguments
/// * `temperature_k` - Temperature in Kelvin
/// * `pressure_pa` - Atmospheric pressure in Pascals
///
/// # Returns
/// Saturation mixing ratio in kg water vapor / kg dry air.
pub fn saturation_humidity(temperature_k: f32, pressure_pa: f32) -> f32 {
    let e_s = saturation_vapor_pressure(temperature_k);
    let ratio = R_DRY_AIR / R_WATER_VAPOR;
    ratio * e_s / (pressure_pa - e_s)
}

/// Dew point temperature given humidity and pressure.
///
/// Inverts the Clausius-Clapeyron equation:
/// Given actual vapor pressure `e = humidity × P / (R_d/R_v + humidity)`,
/// solve for temperature:
/// `T_d = 1 / (1/T₀ - R_v/L_v × ln(e/e₀))`
///
/// # Arguments
/// * `humidity` - Mixing ratio in kg water vapor / kg dry air
/// * `pressure_pa` - Atmospheric pressure in Pascals
///
/// # Returns
/// Dew point temperature in Kelvin.
pub fn dew_point(humidity: f32, pressure_pa: f32) -> f32 {
    // Actual vapor pressure from mixing ratio: e = w × P / (R_d/R_v + w)
    let ratio = R_DRY_AIR / R_WATER_VAPOR;
    let e = humidity * pressure_pa / (ratio + humidity);

    // Invert Clausius-Clapeyron: T_d = 1 / (1/T₀ - R_v/L_v × ln(e/e₀))
    let rv_over_lv = R_WATER_VAPOR / LATENT_HEAT_VAPORIZATION;
    let ln_ratio = (e / SATURATION_PRESSURE_REF).ln();
    1.0 / (1.0 / SATURATION_TEMP_REF - rv_over_lv * ln_ratio)
}

/// Lifting condensation level (LCL) for a rising air parcel.
///
/// Altitude at which an air parcel lifted adiabatically from the surface
/// becomes saturated and forms clouds.
///
/// Approximation: `LCL ≈ (T_surface - T_dew) / lapse_rate × 1000.0`
///
/// # Arguments
/// * `surface_temp_k` - Surface temperature in Kelvin
/// * `surface_humidity` - Surface mixing ratio (kg vapor / kg dry air)
/// * `surface_pressure_pa` - Surface pressure in Pascals
/// * `lapse_rate_k_per_km` - Temperature lapse rate in K/km
///
/// # Returns
/// Lifting condensation level altitude in meters above surface.
pub fn lifting_condensation_level(
    surface_temp_k: f32,
    surface_humidity: f32,
    surface_pressure_pa: f32,
    lapse_rate_k_per_km: f32,
) -> f32 {
    let t_dew = dew_point(surface_humidity, surface_pressure_pa);
    let temp_diff = surface_temp_k - t_dew;
    (temp_diff / lapse_rate_k_per_km) * 1000.0
}

/// Temperature at a given altitude using atmospheric lapse rate.
///
/// Below the tropopause: `T = surface_temperature - lapse_rate × altitude / 1000.0`
/// Above the tropopause: constant at tropopause temperature.
///
/// # Arguments
/// * `config` - Atmosphere configuration
/// * `altitude_m` - Altitude above surface in meters
///
/// # Returns
/// Temperature in Kelvin.
pub fn temperature_at_altitude(config: &AtmosphereConfig, altitude_m: f32) -> f32 {
    if altitude_m < config.tropopause_altitude {
        config.surface_temperature - config.lapse_rate * altitude_m / 1000.0
    } else {
        // Above tropopause: constant temperature
        config.surface_temperature - config.lapse_rate * config.tropopause_altitude / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON_PERCENT_5: f32 = 0.05;
    const EPSILON_PERCENT_10: f32 = 0.10;
    const EPSILON_PERCENT_1: f32 = 0.01;
    const EPSILON_ABSOLUTE_0_5: f32 = 0.5;

    #[test]
    fn saturation_vapor_pressure_at_100c() {
        // At 100°C (373.15 K), saturation pressure should be ~101325 Pa (1 atm)
        // Note: The simple Clausius-Clapeyron equation becomes less accurate
        // at high temperatures (>50°C from reference). Tolerance is relaxed to 25%.
        let temp = 373.15;
        let e_s = saturation_vapor_pressure(temp);
        let expected = 101325.0;
        let rel_error = (e_s - expected).abs() / expected;
        assert!(
            rel_error < 0.25,
            "Saturation pressure at 100°C: {} Pa, expected {} Pa (error: {:.1}%)",
            e_s,
            expected,
            rel_error * 100.0
        );
    }

    #[test]
    fn saturation_vapor_pressure_at_20c() {
        // At 20°C (293.15 K), saturation pressure should be ~2338 Pa
        let temp = 293.15;
        let e_s = saturation_vapor_pressure(temp);
        let expected = 2338.0;
        let rel_error = (e_s - expected).abs() / expected;
        assert!(
            rel_error < EPSILON_PERCENT_5,
            "Saturation pressure at 20°C: {} Pa, expected {} Pa (error: {:.1}%)",
            e_s,
            expected,
            rel_error * 100.0
        );
    }

    #[test]
    fn saturation_vapor_pressure_at_0c() {
        // At 0°C (273.15 K), saturation pressure should be ~611.2 Pa (reference value)
        let temp = 273.15;
        let e_s = saturation_vapor_pressure(temp);
        let expected = 611.2;
        let rel_error = (e_s - expected).abs() / expected;
        assert!(
            rel_error < EPSILON_PERCENT_1,
            "Saturation pressure at 0°C: {} Pa, expected {} Pa (error: {:.1}%)",
            e_s,
            expected,
            rel_error * 100.0
        );
    }

    #[test]
    fn saturation_humidity_at_20c_1atm() {
        // At 20°C, 101325 Pa, saturation humidity should be ~0.0147
        let temp = 293.15;
        let pressure = 101325.0;
        let w_s = saturation_humidity(temp, pressure);
        let expected = 0.0147;
        let rel_error = (w_s - expected).abs() / expected;
        assert!(
            rel_error < EPSILON_PERCENT_10,
            "Saturation humidity at 20°C, 1 atm: {}, expected {} (error: {:.1}%)",
            w_s,
            expected,
            rel_error * 100.0
        );
    }

    #[test]
    fn dew_point_roundtrip() {
        // Compute saturation humidity at a temperature, then verify dew point
        // returns the same temperature
        let temp = 293.15; // 20°C
        let pressure = 101325.0;
        let w_s = saturation_humidity(temp, pressure);
        let t_dew = dew_point(w_s, pressure);
        let abs_error = (t_dew - temp).abs();
        assert!(
            abs_error < EPSILON_ABSOLUTE_0_5,
            "Dew point roundtrip: T={} K, w_s={}, T_dew={} K (error: {} K)",
            temp,
            w_s,
            t_dew,
            abs_error
        );
    }

    #[test]
    fn lifting_condensation_level_sanity() {
        // Surface 25°C (298.15 K), dew point 15°C (288.15 K), lapse 6.5 K/km
        // LCL ≈ (298.15 - 288.15) / 6.5 × 1000 = 1538 m
        let surface_temp = 298.15;
        let surface_pressure = 101325.0;
        // Find humidity that gives dew point of 288.15 K
        let target_dew = 288.15;
        // Approximate: we know dew point formula, so solve backwards
        // For simplicity, use saturation humidity at 288.15 K as the actual humidity
        let surface_humidity = saturation_humidity(target_dew, surface_pressure);
        let lapse = 6.5;

        let lcl =
            lifting_condensation_level(surface_temp, surface_humidity, surface_pressure, lapse);
        let expected = 1538.0;
        let rel_error = (lcl - expected).abs() / expected;
        assert!(
            rel_error < EPSILON_PERCENT_10,
            "LCL: {} m, expected {} m (error: {:.1}%)",
            lcl,
            expected,
            rel_error * 100.0
        );
    }

    #[test]
    fn temperature_at_altitude_at_surface() {
        let config = AtmosphereConfig::default();
        let temp = temperature_at_altitude(&config, 0.0);
        assert!(
            (temp - config.surface_temperature).abs() < 1e-6,
            "Temperature at surface should be {}, got {}",
            config.surface_temperature,
            temp
        );
    }

    #[test]
    fn temperature_at_altitude_mid_troposphere() {
        let config = AtmosphereConfig::default();
        // At 5000 m, T = 288.15 - 6.5 × 5 = 255.65 K
        let temp = temperature_at_altitude(&config, 5000.0);
        let expected = 288.15 - 6.5 * 5.0;
        assert!(
            (temp - expected).abs() < 1e-3,
            "Temperature at 5000m should be {} K, got {} K",
            expected,
            temp
        );
    }

    #[test]
    fn temperature_at_altitude_above_tropopause() {
        let config = AtmosphereConfig::default();
        // Above 11000 m, temperature should be constant
        let temp_at_tropopause = temperature_at_altitude(&config, config.tropopause_altitude);
        let temp_above = temperature_at_altitude(&config, config.tropopause_altitude + 5000.0);
        assert!(
            (temp_at_tropopause - temp_above).abs() < 1e-6,
            "Temperature should be constant above tropopause: {} K vs {} K",
            temp_at_tropopause,
            temp_above
        );
    }

    #[test]
    fn default_atmosphere_config_is_physically_reasonable() {
        let config = AtmosphereConfig::default();
        // Surface temperature reasonable (273-323 K range)
        assert!(config.surface_temperature > 273.0 && config.surface_temperature < 323.0);
        // Lapse rate reasonable (1-10 K/km range)
        assert!(config.lapse_rate > 1.0 && config.lapse_rate < 10.0);
        // Tropopause altitude reasonable (8-15 km range)
        assert!(config.tropopause_altitude > 8000.0 && config.tropopause_altitude < 15000.0);
        // Humidity baseline positive
        assert!(config.humidity_baseline > 0.0);
        // Solar constant reasonable (1300-1400 W/m²)
        assert!(config.solar_constant > 1300.0 && config.solar_constant < 1400.0);
        // Scattering coefficients positive
        assert!(config.rayleigh_scatter_coeff > 0.0);
        assert!(config.mie_scatter_coeff > 0.0);
    }
}
