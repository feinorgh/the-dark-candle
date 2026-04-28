// SkyPlugin — generate the celestial catalogue and bake it into a cubemap.
//
// The plugin runs a single update system that waits for `PlanetaryData` to be
// inserted, derives the catalogue and HDR cubemap from `PlanetData.seed`, and
// publishes a `StarCubemapHandle` resource consumed by `update_sky_material`
// in `crate::lighting::sky_dome`.
//
// The bake itself happens on the main thread (a few hundred ms in release).
// Once we hit that as a frame-stall we'll move it into an `AsyncComputeTask`.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, ImageSampler};
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};

use crate::sky::catalogue::CelestialCatalogue;
use crate::sky::cubemap::{RENDER_MAG_LIMIT, STAR_CUBEMAP_FACE_SIZE, bake_star_cubemap};
use crate::sky::generate::generate_catalogue;
use crate::world::PlanetaryData;

/// Bevy plugin: register the catalogue + cubemap bake startup pipeline.
pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, generate_and_bake_once);
    }
}

/// Resource: the baked star cubemap, populated once after `PlanetaryData` is
/// available.  `update_sky_material` copies this handle into `SkyMaterial`.
#[derive(Resource, Clone)]
pub struct StarCubemapHandle(pub Handle<Image>);

/// Resource: the procedural celestial catalogue, kept resident so future
/// systems (telescope camera, multi-band re-bake, long-exposure mode) can
/// re-render without re-generating.
#[derive(Resource, Clone)]
pub struct SkyCatalogue(pub Arc<CelestialCatalogue>);

/// One-shot system: generate catalogue + bake cubemap on the first frame
/// `PlanetaryData` is available.
fn generate_and_bake_once(
    mut done: Local<bool>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    planetary: Option<Res<PlanetaryData>>,
) {
    if *done {
        return;
    }
    let Some(data) = planetary else {
        return;
    };
    let seed = data.0.config.seed;
    info!("Generating celestial catalogue for system seed {seed:#x}…");
    let catalogue = generate_catalogue(seed);
    info!(
        "Catalogue: {} stars, MW plane normal = {:?}",
        catalogue.stars.len(),
        catalogue.milky_way.plane_normal
    );

    let cube = bake_star_cubemap(&catalogue, STAR_CUBEMAP_FACE_SIZE, RENDER_MAG_LIMIT);
    let size = cube.size;
    let bytes = cube.into_flat_bytes();

    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        label: Some("sky_star_cubemap_view"),
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image.sampler = ImageSampler::linear();

    let handle = images.add(image);
    info!("Star cubemap baked at {size}x{size}x6 (Rgba16Float)");

    commands.insert_resource(StarCubemapHandle(handle));
    commands.insert_resource(SkyCatalogue(Arc::new(catalogue)));
    *done = true;
}
