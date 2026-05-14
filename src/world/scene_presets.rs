//! Scene presets — named configurations that override PlanetConfig.
//!
//! Each preset defines a PlanetConfig tuned for a specific demonstration
//! scenario. Presets are selected via `--scene <name>` on the command line.

use super::erosion::ErosionConfig;
use super::noise::NoiseConfig;
use super::planet::{PlanetConfig, TerrainMode};

/// Available scene presets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScenePreset {
    /// Eroded valleys with river channels carved by D8 flow accumulation.
    ValleyRiver,
    /// Spherical planet driven by the tectonic/biome simulation pipeline.
    SphericalPlanet,
    /// Towering ridged peaks, deep valleys, and a snowline.
    Alpine,
    /// Island chains scattered across open ocean.
    Archipelago,
    /// Flat mesa tops with deep slot canyons.
    DesertCanyon,
    /// Gentle hills and wide grasslands.
    RollingPlains,
    /// Calderas, lava flows, and rugged volcanic peaks.
    Volcanic,
    /// Glacial U-shaped valleys and flat plateaus.
    TundraFjords,
}

impl ScenePreset {
    /// Parse a preset name (case-insensitive, underscores or hyphens).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().replace('-', "_").as_str() {
            "valley_river" | "valley" | "river" => Some(Self::ValleyRiver),
            "spherical_planet" | "spherical" | "planet" => Some(Self::SphericalPlanet),
            "alpine" | "mountains" => Some(Self::Alpine),
            "archipelago" | "islands" => Some(Self::Archipelago),
            "desert_canyon" | "desert" | "canyon" => Some(Self::DesertCanyon),
            "rolling_plains" | "plains" => Some(Self::RollingPlains),
            "volcanic" | "volcano" => Some(Self::Volcanic),
            "tundra_fjords" | "tundra" | "fjords" => Some(Self::TundraFjords),
            _ => None,
        }
    }

    /// List all available preset names for help text.
    pub fn available_names() -> &'static [&'static str] {
        &[
            "valley_river",
            "spherical_planet",
            "alpine",
            "archipelago",
            "desert_canyon",
            "rolling_plains",
            "volcanic",
            "tundra_fjords",
        ]
    }

    /// Build the PlanetConfig for this preset.
    pub fn planet_config(&self) -> PlanetConfig {
        match self {
            Self::ValleyRiver => valley_river_preset(),
            Self::SphericalPlanet => spherical_planet_preset(),
            Self::Alpine => alpine_preset(),
            Self::Archipelago => archipelago_preset(),
            Self::DesertCanyon => desert_canyon_preset(),
            Self::RollingPlains => rolling_plains_preset(),
            Self::Volcanic => volcanic_preset(),
            Self::TundraFjords => tundra_fjords_preset(),
        }
    }
}

// ── Helper: base flat config ───────────────────────────────────────────────

/// Common planetary-mode defaults shared by all presets.
fn base_flat_config() -> PlanetConfig {
    PlanetConfig {
        mode: TerrainMode::Planetary,
        mean_radius: 32_000.0,
        sea_level_radius: 64.0,
        surface_gravity: 9.806_65,
        rotation_rate: 0.0,
        rotation_axis: [0.0, 1.0, 0.0],
        axial_tilt: 0.0,
        libration_amplitude: 0.0,
        libration_period: 0.0,
        magnetic_pole_offset_deg: [0.0, 0.0],
        aurora_strength: 1.0,
        aurora_band_center_deg: 67.0,
        aurora_band_half_width_deg: 5.0,
        height_scale: 32.0,
        layers: Vec::new(),
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

// ── Valley River ───────────────────────────────────────────────────────────

/// Valley river preset: flat terrain with NoiseStack + erosion carving visible
/// channels.
fn valley_river_preset() -> PlanetConfig {
    PlanetConfig {
        sea_level_radius: 40.0,
        height_scale: 24.0,
        seed: 2025,
        cave_freq: 0.03,
        cave_threshold: -0.4,
        soil_depth: 3.0,
        noise: Some(NoiseConfig {
            fbm_octaves: 6,
            fbm_base_freq: 0.008,
            ridged_base_freq: 0.012,
            selector_thresholds: (-0.1, 0.4),
            warp_strength: 30.0,
            micro_amplitude: 1.0,
            ..Default::default()
        }),
        erosion: Some(valley_river_erosion_config()),
        ..base_flat_config()
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

// ── Spherical Planet ───────────────────────────────────────────────────────

/// Spherical planet: configures the voxel world for spherical mode.
/// Uses Earth-scale dimensions (6,371 km radius) with real-world geological
/// structure. The floating-origin system ensures f32 precision is maintained
/// at any position on the surface.
fn spherical_planet_preset() -> PlanetConfig {
    use super::planet::GeologicalLayer;
    PlanetConfig {
        mode: TerrainMode::Planetary,
        mean_radius: 6_371_000.0,
        sea_level_radius: 6_371_000.0,
        surface_gravity: 9.806_65,
        rotation_rate: 7.292e-5,
        rotation_axis: [0.0, 1.0, 0.0],
        axial_tilt: 0.0,
        libration_amplitude: 0.0,
        libration_period: 0.0,
        magnetic_pole_offset_deg: [0.0, 0.0],
        aurora_strength: 1.0,
        aurora_band_center_deg: 67.0,
        aurora_band_half_width_deg: 5.0,
        height_scale: 8_848.0, // Everest-scale relief
        layers: vec![
            GeologicalLayer {
                name: "inner_core".into(),
                inner_radius: 0.0,
                outer_radius: 1_220_000.0,
                material: "Iron".into(),
            },
            GeologicalLayer {
                name: "outer_core".into(),
                inner_radius: 1_220_000.0,
                outer_radius: 3_486_000.0,
                material: "Iron".into(),
            },
            GeologicalLayer {
                name: "mantle".into(),
                inner_radius: 3_486_000.0,
                outer_radius: 6_336_000.0,
                material: "Stone".into(),
            },
            GeologicalLayer {
                name: "crust".into(),
                inner_radius: 6_336_000.0,
                outer_radius: 6_371_000.0,
                material: "Stone".into(),
            },
        ],
        seed: 42,
        continent_freq: 0.005,
        detail_freq: 0.02,
        cave_freq: 0.03,
        cave_threshold: -0.35,
        soil_depth: 4.0,
        noise: Some(NoiseConfig {
            fbm_octaves: 6,
            fbm_base_freq: 0.005,
            ridged_octaves: 5,
            ridged_base_freq: 0.008,
            selector_freq: 0.002,
            selector_thresholds: (-0.3, 0.2),
            warp_strength: 60.0,
            micro_freq: 0.15,
            micro_amplitude: 3.0,
            ..Default::default()
        }),
        erosion: None,
        hydraulic_erosion: None,
    }
}

// ── Alpine ─────────────────────────────────────────────────────────────────

/// Alpine preset: towering ridged peaks, deep valleys, and snowline.
///
/// The terrain selector is biased toward ridged multi-fractal noise, producing
/// sharp mountain ridges.  High `height_scale` creates dramatic vertical relief.
fn alpine_preset() -> PlanetConfig {
    PlanetConfig {
        height_scale: 80.0,
        seed: 7777,
        cave_threshold: -0.35,
        soil_depth: 2.0,
        noise: Some(NoiseConfig {
            fbm_octaves: 7,
            fbm_base_freq: 0.006,
            ridged_octaves: 6,
            ridged_base_freq: 0.008,
            selector_freq: 0.002,
            selector_thresholds: (-0.5, 0.0),
            warp_strength: 50.0,
            micro_freq: 0.2,
            micro_amplitude: 2.0,
            ..Default::default()
        }),
        erosion: Some(ErosionConfig {
            enabled: true,
            flow_threshold: 40.0,
            depth_scale: 4.0,
            max_channel_depth: 16.0,
            width_scale: 2.0,
            valley_shape: 0.25,
            region_size: 4096.0,
            cell_size: 8.0,
        }),
        ..base_flat_config()
    }
}

// ── Archipelago ────────────────────────────────────────────────────────────

/// Archipelago preset: island chains in open ocean.
///
/// Low sea level relative to terrain makes most of the surface underwater,
/// with scattered island peaks poking through.  The continent mask frequency
/// creates many small landmasses.
fn archipelago_preset() -> PlanetConfig {
    PlanetConfig {
        sea_level_radius: 72.0,
        height_scale: 40.0,
        seed: 3141,
        cave_threshold: -0.35,
        soil_depth: 3.0,
        noise: Some(NoiseConfig {
            fbm_octaves: 6,
            fbm_base_freq: 0.007,
            ridged_octaves: 4,
            ridged_base_freq: 0.01,
            selector_freq: 0.004,
            selector_thresholds: (-0.1, 0.3),
            warp_strength: 45.0,
            micro_amplitude: 1.2,
            // Continent mask: many small islands
            continent_enabled: true,
            continent_freq: 0.006,    // higher freq = smaller landmasses
            continent_threshold: 0.2, // only peaks above 0.2 become land
            shelf_blend_width: 0.15,
            ocean_floor_depth: 1.0,
            ocean_floor_amplitude: 0.15,
            ..Default::default()
        }),
        erosion: Some(ErosionConfig {
            enabled: true,
            flow_threshold: 50.0,
            depth_scale: 2.5,
            max_channel_depth: 10.0,
            width_scale: 2.0,
            valley_shape: 0.3,
            region_size: 4096.0,
            cell_size: 8.0,
        }),
        ..base_flat_config()
    }
}

// ── Desert Canyon ──────────────────────────────────────────────────────────

/// Desert canyon preset: flat mesa tops with deep slot canyons.
///
/// Terrain is mostly flat (selector forced toward FBM), but with aggressive
/// cave carving near the surface to create canyon-like features.  Erosion is
/// strong to deepen gullies.
fn desert_canyon_preset() -> PlanetConfig {
    PlanetConfig {
        sea_level_radius: 30.0,
        height_scale: 28.0,
        seed: 5555,
        cave_freq: 0.025,
        cave_threshold: -0.2,
        soil_depth: 1.5,
        noise: Some(NoiseConfig {
            fbm_octaves: 5,
            fbm_base_freq: 0.004,
            fbm_persistence: 0.45,
            ridged_octaves: 4,
            ridged_base_freq: 0.006,
            selector_freq: 0.003,
            selector_thresholds: (0.1, 0.5),
            warp_strength: 35.0,
            micro_freq: 0.12,
            micro_amplitude: 1.0,
            ..Default::default()
        }),
        erosion: Some(ErosionConfig {
            enabled: true,
            flow_threshold: 25.0,
            depth_scale: 5.0,
            max_channel_depth: 18.0,
            width_scale: 1.5,
            valley_shape: 0.15,
            region_size: 4096.0,
            cell_size: 8.0,
        }),
        ..base_flat_config()
    }
}

// ── Rolling Plains ─────────────────────────────────────────────────────────

/// Rolling plains preset: gentle hills and wide grasslands.
///
/// Only FBM noise is used (selector forced below threshold); ridged terrain is
/// absent.  Low `height_scale` for subtle terrain variation.
fn rolling_plains_preset() -> PlanetConfig {
    PlanetConfig {
        height_scale: 16.0,
        seed: 1234,
        cave_threshold: -0.45,
        soil_depth: 5.0,
        noise: Some(NoiseConfig {
            fbm_octaves: 5,
            fbm_base_freq: 0.004,
            fbm_persistence: 0.55,
            ridged_octaves: 3,
            ridged_base_freq: 0.008,
            selector_freq: 0.002,
            // Forces pure FBM everywhere (Perlin max is ~1.0, never reaches 1.1)
            selector_thresholds: (1.1, 1.2),
            warp_strength: 25.0,
            micro_freq: 0.1,
            micro_amplitude: 0.8,
            ..Default::default()
        }),
        erosion: Some(ErosionConfig {
            enabled: true,
            flow_threshold: 60.0,
            depth_scale: 2.0,
            max_channel_depth: 8.0,
            width_scale: 3.0,
            valley_shape: 0.5,
            region_size: 4096.0,
            cell_size: 8.0,
        }),
        ..base_flat_config()
    }
}

// ── Volcanic ───────────────────────────────────────────────────────────────

/// Volcanic preset: calderas, lava flows, and rugged peaks.
///
/// Ridged noise dominates for jagged volcanic terrain.  Dense cave carving
/// creates lava tube candidates at depth.  High height_scale for dramatic
/// volcanic cones.
fn volcanic_preset() -> PlanetConfig {
    PlanetConfig {
        height_scale: 60.0,
        seed: 6660,
        cave_freq: 0.035,
        cave_threshold: -0.25,
        soil_depth: 2.0,
        noise: Some(NoiseConfig {
            fbm_octaves: 6,
            fbm_base_freq: 0.006,
            ridged_octaves: 6,
            ridged_gain: 2.5,
            ridged_base_freq: 0.007,
            selector_freq: 0.003,
            // Selector biased toward ridged (mountains dominant)
            selector_thresholds: (-0.4, -0.1),
            warp_strength: 55.0,
            warp_freq: 0.003,
            micro_freq: 0.18,
            micro_amplitude: 2.0,
            ..Default::default()
        }),
        erosion: Some(ErosionConfig {
            enabled: true,
            flow_threshold: 35.0,
            depth_scale: 3.5,
            max_channel_depth: 12.0,
            width_scale: 2.0,
            valley_shape: 0.2,
            region_size: 4096.0,
            cell_size: 8.0,
        }),
        ..base_flat_config()
    }
}

// ── Tundra Fjords ──────────────────────────────────────────────────────────

/// Tundra fjords preset: glacial U-shaped valleys and flat plateaus.
///
/// Moderate terrain with U-shaped erosion (high valley_shape).  The terrain
/// selector produces wide flat areas interspersed with carved fjord valleys.
fn tundra_fjords_preset() -> PlanetConfig {
    PlanetConfig {
        height_scale: 36.0,
        seed: 9090,
        cave_threshold: -0.4,
        soil_depth: 2.5,
        noise: Some(NoiseConfig {
            fbm_octaves: 6,
            fbm_base_freq: 0.005,
            ridged_octaves: 4,
            ridged_base_freq: 0.007,
            selector_freq: 0.003,
            selector_thresholds: (-0.15, 0.35),
            warp_strength: 35.0,
            micro_freq: 0.12,
            micro_amplitude: 1.0,
            ..Default::default()
        }),
        erosion: Some(ErosionConfig {
            enabled: true,
            flow_threshold: 35.0,
            depth_scale: 4.0,
            max_channel_depth: 15.0,
            width_scale: 3.0,
            valley_shape: 0.8,
            region_size: 4096.0,
            cell_size: 8.0,
        }),
        ..base_flat_config()
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
        assert_eq!(
            ScenePreset::from_name("spherical_planet"),
            Some(ScenePreset::SphericalPlanet)
        );
        assert_eq!(
            ScenePreset::from_name("planet"),
            Some(ScenePreset::SphericalPlanet)
        );
        assert_eq!(
            ScenePreset::from_name("spherical"),
            Some(ScenePreset::SphericalPlanet)
        );
        assert_eq!(ScenePreset::from_name("alpine"), Some(ScenePreset::Alpine));
        assert_eq!(
            ScenePreset::from_name("mountains"),
            Some(ScenePreset::Alpine)
        );
        assert_eq!(
            ScenePreset::from_name("archipelago"),
            Some(ScenePreset::Archipelago)
        );
        assert_eq!(
            ScenePreset::from_name("islands"),
            Some(ScenePreset::Archipelago)
        );
        assert_eq!(
            ScenePreset::from_name("desert_canyon"),
            Some(ScenePreset::DesertCanyon)
        );
        assert_eq!(
            ScenePreset::from_name("desert"),
            Some(ScenePreset::DesertCanyon)
        );
        assert_eq!(
            ScenePreset::from_name("rolling_plains"),
            Some(ScenePreset::RollingPlains)
        );
        assert_eq!(
            ScenePreset::from_name("plains"),
            Some(ScenePreset::RollingPlains)
        );
        assert_eq!(
            ScenePreset::from_name("volcanic"),
            Some(ScenePreset::Volcanic)
        );
        assert_eq!(
            ScenePreset::from_name("volcano"),
            Some(ScenePreset::Volcanic)
        );
        assert_eq!(
            ScenePreset::from_name("tundra_fjords"),
            Some(ScenePreset::TundraFjords)
        );
        assert_eq!(
            ScenePreset::from_name("tundra"),
            Some(ScenePreset::TundraFjords)
        );
        assert_eq!(
            ScenePreset::from_name("fjords"),
            Some(ScenePreset::TundraFjords)
        );
        assert_eq!(ScenePreset::from_name("unknown"), None);
    }

    #[test]
    fn valley_river_config_is_planetary() {
        let config = ScenePreset::ValleyRiver.planet_config();
        assert_eq!(config.mode, TerrainMode::Planetary);
        assert!(
            config.sea_level_radius < 100.0,
            "Sea level should be low for this preset"
        );
    }

    #[test]
    fn spherical_planet_config_is_spherical() {
        let config = ScenePreset::SphericalPlanet.planet_config();
        assert_eq!(config.mode, TerrainMode::Planetary);
        assert!(
            config.mean_radius > 1_000.0,
            "Spherical planet radius should be at least 1 km"
        );
        assert!(
            !config.layers.is_empty(),
            "Spherical planet should have geological layers"
        );
    }

    #[test]
    fn all_presets_produce_valid_configs() {
        for name in ScenePreset::available_names() {
            let preset = ScenePreset::from_name(name).unwrap();
            let config = preset.planet_config();
            assert!(
                config.height_scale > 0.0,
                "{name}: height_scale must be positive"
            );
            assert!(config.seed > 0, "{name}: seed must be non-zero");
            assert!(
                config.cave_threshold < 0.0,
                "{name}: cave_threshold should be negative"
            );
            assert!(
                config.soil_depth > 0.0,
                "{name}: soil_depth must be positive"
            );
        }
    }

    #[test]
    fn new_presets_use_noise_stack() {
        for name in &[
            "alpine",
            "archipelago",
            "desert_canyon",
            "rolling_plains",
            "volcanic",
            "tundra_fjords",
        ] {
            let preset = ScenePreset::from_name(name).unwrap();
            let config = preset.planet_config();
            assert!(config.noise.is_some(), "{name} should use NoiseStack");
        }
    }

    #[test]
    fn valley_river_migrated_to_noise_stack() {
        let config = ScenePreset::ValleyRiver.planet_config();
        assert!(
            config.noise.is_some(),
            "valley_river should now use NoiseStack"
        );
    }

    #[test]
    fn alpine_has_high_relief() {
        let config = ScenePreset::Alpine.planet_config();
        assert!(
            config.height_scale >= 60.0,
            "Alpine should have high relief"
        );
        let noise = config.noise.as_ref().unwrap();
        assert!(
            noise.selector_thresholds.0 < -0.3,
            "Alpine selector should be biased toward ridged"
        );
    }

    #[test]
    fn rolling_plains_forces_pure_fbm() {
        let config = ScenePreset::RollingPlains.planet_config();
        assert!(config.height_scale <= 20.0, "Plains should have low relief");
        let noise = config.noise.as_ref().unwrap();
        assert!(
            noise.selector_thresholds.0 > 1.0,
            "Plains selector should force pure FBM"
        );
    }

    #[test]
    fn available_names_matches_from_name() {
        for name in ScenePreset::available_names() {
            assert!(
                ScenePreset::from_name(name).is_some(),
                "available_names() lists '{name}' but from_name() doesn't recognize it"
            );
        }
    }
}
