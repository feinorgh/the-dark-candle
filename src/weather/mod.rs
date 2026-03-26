//! Weather systems: particle simulation, rendering, and atmospheric effects.

pub mod accumulation;
pub mod emitter;
pub mod particle_render;
pub mod wind_upload;

pub use accumulation::{AccumulationConfig, SurfaceAccumulation};
pub use emitter::PrecipitationEmitter;
pub use particle_render::{
    ParticleMeshMarker, ParticleReadback, ParticleRenderConfig, ParticleRenderPlugin,
};
pub use wind_upload::{GpuWeatherState, WindFieldUploader};

use bevy::prelude::*;

use wind_upload::{extract_wind_field, upload_wind_to_gpu};

/// Top-level weather plugin — registers all weather sub-plugins.
pub struct WeatherPlugin;

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ParticleRenderPlugin)
            .init_resource::<WindFieldUploader>()
            .init_resource::<GpuWeatherState>()
            .init_resource::<AccumulationConfig>()
            .add_systems(FixedUpdate, extract_wind_field)
            .add_systems(Update, upload_wind_to_gpu)
            .add_systems(
                Update,
                (
                    accumulation::init_accumulation,
                    accumulation::track_ground_impacts,
                    accumulation::apply_accumulation,
                )
                    .chain(),
            );
        emitter::register(app);
    }
}
