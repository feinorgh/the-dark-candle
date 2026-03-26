//! Scene presets — named configurations that override PlanetConfig.
//!
//! Each preset defines a PlanetConfig tuned for a specific demonstration
//! scenario. Presets are selected via `--scene <name>` on the command line.

use super::erosion::ErosionConfig;
use super::planet::{PlanetConfig, TerrainMode};

/// Available scene presets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScenePreset {
    /// A flat-terrain valley with river channels carved by D8 flow
    /// accumulation. Good for demonstrating erosion, fluid simulation,
    /// and atmospheric rendering together.
    ValleyRiver,
}

impl ScenePreset {
    /// Parse a preset name (case-insensitive, underscores or hyphens).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().replace('-', "_").as_str() {
            "valley_river" | "valley" | "river" => Some(Self::ValleyRiver),
            _ => None,
        }
    }

    /// List all available preset names for help text.
    pub fn available_names() -> &'static [&'static str] {
        &["valley_river"]
    }

    /// Build the PlanetConfig for this preset.
    pub fn planet_config(&self) -> PlanetConfig {
        match self {
            Self::ValleyRiver => valley_river_preset(),
        }
    }
}

/// Valley river preset: flat terrain with erosion carving visible channels.
///
/// Tuned parameters:
/// - Flat mode with sea_level at 40 (low, so valleys reach water)
/// - Higher continent_freq for more varied terrain in a small area
/// - Moderate height_scale to keep valleys within visible range
/// - Erosion enabled with a low flow_threshold for denser drainage
fn valley_river_preset() -> PlanetConfig {
    PlanetConfig {
        mode: TerrainMode::Flat,
        mean_radius: 32_000.0,
        sea_level_radius: 40.0,
        surface_gravity: 9.806_65,
        rotation_rate: 0.0,
        rotation_axis: [0.0, 1.0, 0.0],
        height_scale: 24.0,
        layers: Vec::new(),
        seed: 2025,
        continent_freq: 0.008,
        detail_freq: 0.03,
        cave_freq: 0.03,
        cave_threshold: -0.4,
        soil_depth: 3.0,
        erosion: Some(valley_river_erosion_config()),
    }
}

/// Return the ErosionConfig used by the valley river preset.
pub fn valley_river_erosion_config() -> ErosionConfig {
    ErosionConfig {
        enabled: true,
        flow_threshold: 30.0,
        depth_scale: 3.5,
        max_channel_depth: 14.0,
        width_scale: 2.5,
        valley_shape: 0.35,
        region_size: 4096.0,
        cell_size: 8.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_preset_names() {
        assert_eq!(
            ScenePreset::from_name("valley_river"),
            Some(ScenePreset::ValleyRiver)
        );
        assert_eq!(
            ScenePreset::from_name("Valley-River"),
            Some(ScenePreset::ValleyRiver)
        );
        assert_eq!(
            ScenePreset::from_name("valley"),
            Some(ScenePreset::ValleyRiver)
        );
        assert_eq!(ScenePreset::from_name("unknown"), None);
    }

    #[test]
    fn valley_river_config_is_flat() {
        let config = ScenePreset::ValleyRiver.planet_config();
        assert_eq!(config.mode, TerrainMode::Flat);
        assert!(
            config.sea_level_radius < 100.0,
            "Sea level should be low for flat mode"
        );
    }
}
