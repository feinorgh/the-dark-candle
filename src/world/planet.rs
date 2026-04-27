// Spherical planet configuration and coordinate helpers.
//
// Defines the planet as a sphere centered at world origin with configurable
// radius, geological layers, rotation, and gravity.  All spatial units are SI
// (meters, radians, seconds).  1 voxel = 1 meter.
//
// The planet model provides:
//   - Radial altitude from planet center
//   - Latitude / longitude from Cartesian position
//   - Local gravity vector (gravitational + centrifugal)
//   - Surface-normal direction (local "up")
//   - Shell membership test for chunk loading
//
// Data-driven: loaded from `assets/data/planet_config.ron` via `RonAssetPlugin`.

use bevy::math::DVec3;
use bevy::prelude::*;
use serde::Deserialize;

use super::erosion::{ErosionConfig, HydraulicErosionConfig};
use super::noise::NoiseConfig;

/// Gravitational constant G in m³/(kg·s²). NIST CODATA 2018.
pub const GRAVITATIONAL_CONSTANT: f64 = 6.674_30e-11;

/// Geological layer defined by radial depth bands.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct GeologicalLayer {
    /// Name for identification (e.g. "inner_core", "mantle", "crust").
    pub name: String,
    /// Inner radius of this layer in meters from planet center.
    pub inner_radius: f64,
    /// Outer radius of this layer in meters from planet center.
    pub outer_radius: f64,
    /// Primary material name for this layer (matched against MaterialRegistry).
    pub material: String,
}

/// Terrain generation mode.
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub enum TerrainMode {
    /// Planet-data-driven spherical terrain (PlanetaryTerrainSampler).
    #[default]
    Planetary,
    /// Flat world with constant -Y gravity. Used for sandbox tests and
    /// headless physics simulations that don't need a spherical planet.
    Flat,
}

/// Configuration for a spherical planet, loaded from RON.
///
/// The planet is a sphere centered at world origin `(0, 0, 0)`.
/// Surface features come from noise sampled in spherical coordinates.
/// Geological layers (core → mantle → crust) are defined by radial bands.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, Resource)]
pub struct PlanetConfig {
    /// Terrain generation mode.
    #[serde(default)]
    pub mode: TerrainMode,

    /// Mean radius of the planet surface in meters.
    /// Default: 32,000 m (32 km — comparable to a large island).
    pub mean_radius: f64,

    /// Sea-level radius in meters from planet center.
    /// Water fills any surface below this radius.
    pub sea_level_radius: f64,

    /// Surface gravitational acceleration in m/s². Earth: 9.80665.
    /// Used to derive planet mass via `M = g·r²/G`.
    pub surface_gravity: f64,

    /// Rotation rate in rad/s. Earth: ~7.292e-5.
    /// Set to 0.0 for a non-rotating planet.
    #[serde(default)]
    pub rotation_rate: f64,

    /// Rotation axis as a unit vector. Default: Y-axis (0, 1, 0).
    #[serde(default = "default_rotation_axis")]
    pub rotation_axis: [f64; 3],

    /// Axial tilt (obliquity) in radians. Earth: 23.44° ≈ 0.4091 rad.
    /// Tilts the spin axis relative to the orbital plane, producing seasons.
    #[serde(default)]
    pub axial_tilt: f64,

    /// Libration amplitude in radians. Produces a periodic wobble in the
    /// apparent sun position. Moon: ~7° ≈ 0.122 rad. Default: 0.0 (none).
    #[serde(default)]
    pub libration_amplitude: f64,

    /// Libration period in game-days. Default: 0.0 (disabled).
    /// Set to 1.0 for diurnal libration (wobble once per day).
    #[serde(default)]
    pub libration_period: f64,

    /// Amplitude of surface displacement noise in meters.
    /// Terrain height varies from `mean_radius - height_scale` to
    /// `mean_radius + height_scale`.
    pub height_scale: f64,

    /// Geological layers from core to surface. Ordered inner-to-outer.
    #[serde(default)]
    pub layers: Vec<GeologicalLayer>,

    /// Noise seed for terrain generation.
    #[serde(default = "default_seed")]
    pub seed: u32,

    /// Continental noise frequency (controls landmass size).
    #[serde(default = "default_continent_freq")]
    pub continent_freq: f64,

    /// Detail noise frequency (controls hills/mountains).
    #[serde(default = "default_detail_freq")]
    pub detail_freq: f64,

    /// Cave noise frequency.
    #[serde(default = "default_cave_freq")]
    pub cave_freq: f64,

    /// Cave carving threshold (lower = fewer caves).
    #[serde(default = "default_cave_threshold")]
    pub cave_threshold: f64,

    /// Soil/regolith depth in meters above bedrock.
    #[serde(default = "default_soil_depth")]
    pub soil_depth: f64,

    /// Composable noise stack configuration.  When present, the terrain
    /// generators use `NoiseStack` instead of the legacy 2-layer Perlin blend.
    /// Fields like `continent_freq` and `detail_freq` are ignored when this
    /// is `Some`.
    #[serde(default)]
    pub noise: Option<NoiseConfig>,

    /// Erosion and valley carving configuration. When present, overrides the
    /// default `ErosionConfig` used by the flat terrain generator.
    #[serde(default)]
    pub erosion: Option<ErosionConfig>,

    /// Hydraulic erosion configuration.  Applied to the heightmap after
    /// noise generation but before voxel fill.
    #[serde(default)]
    pub hydraulic_erosion: Option<HydraulicErosionConfig>,
}

fn default_rotation_axis() -> [f64; 3] {
    [0.0, 1.0, 0.0]
}
fn default_seed() -> u32 {
    42
}
fn default_continent_freq() -> f64 {
    0.005
}
fn default_detail_freq() -> f64 {
    0.02
}
fn default_cave_freq() -> f64 {
    0.03
}
fn default_cave_threshold() -> f64 {
    -0.3
}
fn default_soil_depth() -> f64 {
    4.0
}

impl Default for PlanetConfig {
    fn default() -> Self {
        Self {
            mode: TerrainMode::Planetary,
            mean_radius: 32_000.0,
            sea_level_radius: 32_000.0,
            surface_gravity: 9.806_65,
            rotation_rate: 7.292e-5, // Earth sidereal
            rotation_axis: default_rotation_axis(),
            axial_tilt: 0.0,
            libration_amplitude: 0.0,
            libration_period: 0.0,
            height_scale: 32.0,
            layers: default_layers(),
            seed: 42,
            continent_freq: 0.005,
            detail_freq: 0.02,
            cave_freq: 0.03,
            cave_threshold: -0.3,
            soil_depth: 4.0,
            noise: None,
            erosion: None,
            hydraulic_erosion: None,
        }
    }
}

/// Default geological layers for a 32 km radius planet.
///
/// Proportions roughly follow Earth's internal structure scaled down.
fn default_layers() -> Vec<GeologicalLayer> {
    vec![
        GeologicalLayer {
            name: "inner_core".into(),
            inner_radius: 0.0,
            outer_radius: 5_000.0,
            material: "Iron".into(),
        },
        GeologicalLayer {
            name: "outer_core".into(),
            inner_radius: 5_000.0,
            outer_radius: 11_000.0,
            material: "Iron".into(),
        },
        GeologicalLayer {
            name: "mantle".into(),
            inner_radius: 11_000.0,
            outer_radius: 28_000.0,
            material: "Stone".into(),
        },
        GeologicalLayer {
            name: "crust".into(),
            inner_radius: 28_000.0,
            outer_radius: 32_000.0,
            material: "Stone".into(),
        },
    ]
}

impl PlanetConfig {
    /// Planet mass derived from surface gravity and mean radius.
    ///
    /// From Newton's law: `g = G·M/r²`  →  `M = g·r²/G`.
    pub fn planet_mass(&self) -> f64 {
        self.surface_gravity * self.mean_radius * self.mean_radius / GRAVITATIONAL_CONSTANT
    }

    /// Radial distance from planet center to a world position.
    #[inline]
    pub fn distance_from_center(&self, pos: DVec3) -> f64 {
        pos.length()
    }

    /// Altitude above mean radius (positive = above surface, negative = below).
    #[inline]
    pub fn altitude(&self, pos: DVec3) -> f64 {
        self.distance_from_center(pos) - self.mean_radius
    }

    /// Altitude above sea level radius.
    #[inline]
    pub fn altitude_above_sea_level(&self, pos: DVec3) -> f64 {
        self.distance_from_center(pos) - self.sea_level_radius
    }

    /// Unit vector pointing outward from planet center (local "up").
    ///
    /// Returns `DVec3::Y` if the position is at the exact center (degenerate).
    #[inline]
    pub fn surface_normal(&self, pos: DVec3) -> DVec3 {
        let len = pos.length();
        if len < 1e-10 { DVec3::Y } else { pos / len }
    }

    /// Convert a world position to `(latitude, longitude)` in radians.
    ///
    /// - Latitude: −π/2 (south pole) to +π/2 (north pole).  Measured from
    ///   the equatorial plane, where the equator is perpendicular to
    ///   `rotation_axis`.
    /// - Longitude: −π to +π.  Measured in the equatorial plane from the +X
    ///   axis toward +Z.
    pub fn lat_lon(&self, pos: DVec3) -> (f64, f64) {
        let len = pos.length();
        if len < 1e-10 {
            return (0.0, 0.0);
        }
        let dir = pos / len;

        let axis = DVec3::new(
            self.rotation_axis[0],
            self.rotation_axis[1],
            self.rotation_axis[2],
        )
        .normalize_or(DVec3::Y);

        // Latitude: angle from equatorial plane = asin(dir · axis).
        let sin_lat = dir.dot(axis).clamp(-1.0, 1.0);
        let lat = sin_lat.asin();

        // Project direction onto equatorial plane.
        let equatorial = dir - axis * sin_lat;
        let eq_len = equatorial.length();
        let lon = if eq_len < 1e-12 {
            0.0 // At a pole; longitude is undefined.
        } else {
            let eq_norm = equatorial / eq_len;
            // Build equatorial basis: east = axis × X (or fallback).
            let ref_x = if axis.x.abs() < 0.9 {
                DVec3::X
            } else {
                DVec3::Z
            };
            let east = axis.cross(ref_x).normalize();
            let north_eq = east.cross(axis).normalize();
            eq_norm.dot(north_eq).atan2(eq_norm.dot(east))
        };

        (lat, lon)
    }

    /// Gravitational acceleration vector at a world position.
    ///
    /// Combines Newtonian gravity toward the center and centrifugal pseudo-force
    /// from planetary rotation:
    ///
    /// `g⃗_apparent = g⃗_grav + a⃗_centrifugal`
    ///
    /// where:
    /// - `g⃗_grav = -(G·M / r²) · r̂` (toward center)
    /// - `a⃗_centrifugal = ω² · d⊥` (away from rotation axis)
    pub fn gravity_at(&self, pos: DVec3) -> DVec3 {
        let r = pos.length();
        if r < 1e-10 {
            return DVec3::ZERO;
        }
        let r_hat = pos / r;

        // Newtonian gravitational acceleration: g = -G·M/r² · r̂
        let gm = self.surface_gravity * self.mean_radius * self.mean_radius;
        let g_grav = -(gm / (r * r)) * r_hat;

        // Centrifugal pseudo-force: a_c = ω² · d⊥
        if self.rotation_rate.abs() < 1e-20 {
            return g_grav;
        }

        let omega = self.rotation_rate;
        let axis = DVec3::new(
            self.rotation_axis[0],
            self.rotation_axis[1],
            self.rotation_axis[2],
        )
        .normalize_or(DVec3::Y);

        // d⊥ = component of pos perpendicular to rotation axis
        let along_axis = pos.dot(axis) * axis;
        let d_perp = pos - along_axis;
        let a_centrifugal = omega * omega * d_perp;

        g_grav + a_centrifugal
    }

    /// Local "down" direction at a world position (unit vector).
    ///
    /// This is the normalized apparent gravity vector, defining the local
    /// vertical.  All physics (slope forces, ground detection, buoyancy
    /// direction) derive from this single vector.
    pub fn local_down(&self, pos: DVec3) -> DVec3 {
        let g = self.gravity_at(pos);
        let len = g.length();
        if len < 1e-20 {
            -self.surface_normal(pos)
        } else {
            g / len
        }
    }

    /// Test whether a world position is within the surface shell
    /// (the band of chunks worth loading around the terrain surface).
    ///
    /// `depth_below` and `height_above` are in meters relative to the mean
    /// surface radius.
    pub fn is_in_shell(&self, pos: DVec3, depth_below: f64, height_above: f64) -> bool {
        let r = pos.length();
        let min_r = self.mean_radius - depth_below;
        let max_r = self.mean_radius + height_above;
        r >= min_r && r <= max_r
    }

    /// Surface radius at a given `(lat, lon)` from noise displacement.
    ///
    /// Returns `mean_radius + noise_displacement`.  This is the actual terrain
    /// surface height at that angular position.
    ///
    /// Uses the same 2D noise approach as the flat generator, but with lat/lon
    /// as coordinates instead of world X/Z.
    pub fn surface_radius_at(&self, _lat: f64, _lon: f64, noise_val: f64) -> f64 {
        self.mean_radius + noise_val * self.height_scale
    }

    /// Find the geological layer at a given radial distance from center.
    ///
    /// Returns the layer name and material, or `None` if no layer covers
    /// that radius (above crust → surface/air).
    pub fn layer_at_radius(&self, radius: f64) -> Option<&GeologicalLayer> {
        self.layers
            .iter()
            .find(|l| radius >= l.inner_radius && radius < l.outer_radius)
    }

    /// Whether the planet uses spherical mode.
    /// Returns `false` for `Flat` mode, `true` for `Planetary` mode.
    pub fn is_spherical(&self) -> bool {
        self.mode != TerrainMode::Flat
    }
}

/// Placeholder for future world-generation pipeline ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorldGenPhase {
    /// Geological layer assignment (core → mantle → crust).
    Geology,
    /// Surface heightmap from noise.
    Heightmap,
    /// Cave and tunnel carving.
    Caves,
    /// Biome assignment from climate parameters.
    Biomes,
    /// Ore and mineral deposit placement.
    Ores,
    /// Water body filling (oceans, lakes, rivers).
    Hydrology,
}

/// Placeholder for tectonic plate configuration (future phase).
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct TectonicPlateConfig {
    /// Number of major tectonic plates.
    pub plate_count: u32,
    /// Seed for plate boundary generation.
    pub seed: u64,
    /// Mean drift speed in m/s (geological timescale).
    pub mean_drift_speed: f64,
}

impl Default for TectonicPlateConfig {
    fn default() -> Self {
        Self {
            plate_count: 12,
            seed: 42,
            mean_drift_speed: 1.6e-9, // ~5 cm/year
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn default_config() -> PlanetConfig {
        PlanetConfig::default()
    }

    // --- Basic properties ---

    #[test]
    fn default_config_is_planetary() {
        let cfg = default_config();
        assert!(cfg.is_spherical());
        assert_eq!(cfg.mean_radius, 32_000.0);
        assert!((cfg.surface_gravity - 9.806_65).abs() < 1e-4);
    }

    #[test]
    fn planet_mass_is_physically_consistent() {
        let cfg = default_config();
        let mass = cfg.planet_mass();
        // M = g·r²/G = 9.80665 * 32000² / 6.67430e-11
        let expected = 9.806_65 * 32_000.0 * 32_000.0 / GRAVITATIONAL_CONSTANT;
        assert!(
            (mass - expected).abs() / expected < 1e-6,
            "Mass {mass} != expected {expected}"
        );
        // Sanity: mass should be very large
        assert!(mass > 1e20, "Planet mass too small: {mass}");
    }

    // --- Distance and altitude ---

    #[test]
    fn altitude_at_surface_is_zero() {
        let cfg = default_config();
        let pos = DVec3::new(cfg.mean_radius, 0.0, 0.0);
        assert!(cfg.altitude(pos).abs() < 1e-6);
    }

    #[test]
    fn altitude_above_surface_is_positive() {
        let cfg = default_config();
        let pos = DVec3::new(cfg.mean_radius + 100.0, 0.0, 0.0);
        assert!((cfg.altitude(pos) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn altitude_below_surface_is_negative() {
        let cfg = default_config();
        let pos = DVec3::new(cfg.mean_radius - 500.0, 0.0, 0.0);
        assert!((cfg.altitude(pos) - (-500.0)).abs() < 1e-6);
    }

    #[test]
    fn altitude_above_sea_level() {
        let cfg = default_config();
        let above = DVec3::new(cfg.sea_level_radius + 10.0, 0.0, 0.0);
        assert!((cfg.altitude_above_sea_level(above) - 10.0).abs() < 1e-6);
    }

    // --- Surface normal ---

    #[test]
    fn surface_normal_points_outward() {
        let cfg = default_config();
        let pos = DVec3::new(100.0, 200.0, 300.0);
        let normal = cfg.surface_normal(pos);
        let expected = pos.normalize();
        assert!((normal - expected).length() < 1e-10);
    }

    #[test]
    fn surface_normal_at_origin_is_y() {
        let cfg = default_config();
        let normal = cfg.surface_normal(DVec3::ZERO);
        assert_eq!(normal, DVec3::Y);
    }

    // --- Latitude / longitude ---

    #[test]
    fn lat_lon_at_north_pole() {
        let cfg = default_config();
        // North pole: along +Y (rotation axis)
        let pos = DVec3::new(0.0, cfg.mean_radius, 0.0);
        let (lat, _lon) = cfg.lat_lon(pos);
        assert!(
            (lat - PI / 2.0).abs() < 1e-6,
            "North pole lat should be π/2, got {lat}"
        );
    }

    #[test]
    fn lat_lon_at_south_pole() {
        let cfg = default_config();
        let pos = DVec3::new(0.0, -cfg.mean_radius, 0.0);
        let (lat, _lon) = cfg.lat_lon(pos);
        assert!(
            (lat - (-PI / 2.0)).abs() < 1e-6,
            "South pole lat should be -π/2, got {lat}"
        );
    }

    #[test]
    fn lat_lon_at_equator() {
        let cfg = default_config();
        let pos = DVec3::new(cfg.mean_radius, 0.0, 0.0);
        let (lat, _lon) = cfg.lat_lon(pos);
        assert!(lat.abs() < 1e-6, "Equator lat should be ~0, got {lat}");
    }

    #[test]
    fn lat_lon_varies_around_equator() {
        let cfg = default_config();
        let r = cfg.mean_radius;
        let (_, lon1) = cfg.lat_lon(DVec3::new(r, 0.0, 0.0));
        let (_, lon2) = cfg.lat_lon(DVec3::new(0.0, 0.0, r));
        assert!(
            (lon1 - lon2).abs() > 0.1,
            "Longitude should vary: lon1={lon1}, lon2={lon2}"
        );
    }

    // --- Gravity ---

    #[test]
    fn gravity_at_surface_matches_configured() {
        let cfg = PlanetConfig {
            rotation_rate: 0.0, // No rotation for clean test
            ..default_config()
        };
        let pos = DVec3::new(cfg.mean_radius, 0.0, 0.0);
        let g = cfg.gravity_at(pos);
        // Should point inward (toward center = -X)
        assert!(g.x < 0.0, "Gravity should point toward center");
        // Magnitude should match surface gravity
        let mag = g.length();
        assert!(
            (mag - cfg.surface_gravity).abs() < 1e-4,
            "Surface gravity magnitude {mag} != {}",
            cfg.surface_gravity
        );
    }

    #[test]
    fn gravity_decreases_with_altitude() {
        let cfg = PlanetConfig {
            rotation_rate: 0.0,
            ..default_config()
        };
        let g_surface = cfg
            .gravity_at(DVec3::new(cfg.mean_radius, 0.0, 0.0))
            .length();
        let g_high = cfg
            .gravity_at(DVec3::new(cfg.mean_radius + 1000.0, 0.0, 0.0))
            .length();
        assert!(
            g_surface > g_high,
            "Gravity should decrease with altitude: {g_surface} vs {g_high}"
        );
    }

    #[test]
    fn gravity_inverse_square_law() {
        let cfg = PlanetConfig {
            rotation_rate: 0.0,
            ..default_config()
        };
        let r1 = cfg.mean_radius;
        let r2 = cfg.mean_radius * 2.0;
        let g1 = cfg.gravity_at(DVec3::new(r1, 0.0, 0.0)).length();
        let g2 = cfg.gravity_at(DVec3::new(r2, 0.0, 0.0)).length();
        // g ∝ 1/r² → g1/g2 = (r2/r1)² = 4
        let ratio = g1 / g2;
        assert!(
            (ratio - 4.0).abs() < 0.01,
            "Inverse square ratio: {ratio}, expected 4.0"
        );
    }

    #[test]
    fn gravity_is_radially_symmetric() {
        let cfg = PlanetConfig {
            rotation_rate: 0.0,
            ..default_config()
        };
        let r = cfg.mean_radius;
        let g_x = cfg.gravity_at(DVec3::new(r, 0.0, 0.0)).length();
        let g_y = cfg.gravity_at(DVec3::new(0.0, r, 0.0)).length();
        let g_z = cfg.gravity_at(DVec3::new(0.0, 0.0, r)).length();
        assert!(
            (g_x - g_y).abs() < 1e-6 && (g_y - g_z).abs() < 1e-6,
            "Gravity should be radially symmetric: x={g_x}, y={g_y}, z={g_z}"
        );
    }

    #[test]
    fn centrifugal_reduces_apparent_gravity_at_equator() {
        let cfg = PlanetConfig {
            rotation_rate: 7.292e-5, // Earth rotation
            ..default_config()
        };
        let cfg_no_rot = PlanetConfig {
            rotation_rate: 0.0,
            ..default_config()
        };
        // Equator position (perpendicular to rotation axis Y)
        let pos = DVec3::new(cfg.mean_radius, 0.0, 0.0);
        let g_rot = cfg.gravity_at(pos).length();
        let g_no_rot = cfg_no_rot.gravity_at(pos).length();
        assert!(
            g_no_rot > g_rot,
            "Centrifugal should reduce apparent gravity: {g_no_rot} > {g_rot}"
        );
    }

    #[test]
    fn centrifugal_has_no_effect_at_poles() {
        let cfg = PlanetConfig {
            rotation_rate: 7.292e-5,
            ..default_config()
        };
        let cfg_no_rot = PlanetConfig {
            rotation_rate: 0.0,
            ..default_config()
        };
        // Pole position (along rotation axis Y)
        let pos = DVec3::new(0.0, cfg.mean_radius, 0.0);
        let g_rot = cfg.gravity_at(pos).length();
        let g_no_rot = cfg_no_rot.gravity_at(pos).length();
        assert!(
            (g_rot - g_no_rot).abs() < 1e-10,
            "Centrifugal should be zero at poles: {g_rot} vs {g_no_rot}"
        );
    }

    #[test]
    fn gravity_at_center_is_zero() {
        let cfg = default_config();
        let g = cfg.gravity_at(DVec3::ZERO);
        assert!(g.length() < 1e-20);
    }

    // --- Local down ---

    #[test]
    fn local_down_points_toward_center() {
        let cfg = PlanetConfig {
            rotation_rate: 0.0,
            ..default_config()
        };
        let pos = DVec3::new(cfg.mean_radius, 0.0, 0.0);
        let down = cfg.local_down(pos);
        // Should point inward = -X direction
        assert!(down.x < -0.99, "Local down should point toward center");
    }

    // --- Shell membership ---

    #[test]
    fn is_in_shell_at_surface() {
        let cfg = default_config();
        let pos = DVec3::new(cfg.mean_radius, 0.0, 0.0);
        assert!(cfg.is_in_shell(pos, 100.0, 100.0));
    }

    #[test]
    fn is_in_shell_deep_interior_excluded() {
        let cfg = default_config();
        let pos = DVec3::new(cfg.mean_radius - 500.0, 0.0, 0.0);
        assert!(!cfg.is_in_shell(pos, 100.0, 100.0));
    }

    #[test]
    fn is_in_shell_outer_space_excluded() {
        let cfg = default_config();
        let pos = DVec3::new(cfg.mean_radius + 500.0, 0.0, 0.0);
        assert!(!cfg.is_in_shell(pos, 100.0, 100.0));
    }

    // --- Geological layers ---

    #[test]
    fn layer_at_center_is_inner_core() {
        let cfg = default_config();
        let layer = cfg.layer_at_radius(100.0);
        assert!(layer.is_some());
        assert_eq!(layer.unwrap().name, "inner_core");
    }

    #[test]
    fn layer_at_surface_is_crust() {
        let cfg = default_config();
        let layer = cfg.layer_at_radius(31_000.0);
        assert!(layer.is_some());
        assert_eq!(layer.unwrap().name, "crust");
    }

    #[test]
    fn layer_above_crust_is_none() {
        let cfg = default_config();
        let layer = cfg.layer_at_radius(33_000.0);
        assert!(layer.is_none());
    }

    // --- Surface radius ---

    #[test]
    fn surface_radius_with_zero_noise() {
        let cfg = default_config();
        let r = cfg.surface_radius_at(0.0, 0.0, 0.0);
        assert_eq!(r, cfg.mean_radius);
    }

    #[test]
    fn surface_radius_with_positive_noise() {
        let cfg = default_config();
        let r = cfg.surface_radius_at(0.0, 0.0, 1.0);
        assert_eq!(r, cfg.mean_radius + cfg.height_scale);
    }

    // --- RON deserialization ---

    #[test]
    fn planet_config_ron_roundtrip() {
        let ron_str = std::fs::read_to_string("assets/data/planet_config.ron")
            .expect("planet_config.ron should exist");
        let parsed: PlanetConfig =
            ron::from_str(&ron_str).expect("RON should deserialize into PlanetConfig");
        assert!(parsed.mean_radius > 0.0);
        assert!(parsed.surface_gravity > 0.0);
    }

    // --- Terrain mode ---
}
