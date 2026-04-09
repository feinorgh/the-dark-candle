// Global planet map: equirectangular projection of the planet with player marker.
//
// Uses the existing CPU render_projection() on first open, then caches the
// result as a Bevy Image handle.

use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::camera::FpsCamera;
use crate::hud::Player;
use crate::planet::projections::{Projection, render_projection};
use crate::planet::render::ColourMode;
use crate::world::PlanetaryData;
use crate::world::planet::PlanetConfig;

use super::ui::{MapCoordText, MapImageNode, MapView, MapViewState, PlayerMarker};

/// Width of the cached global projection image.
const PROJECTION_WIDTH: u32 = 1024;

/// Cached global map projection.
#[derive(Default)]
pub struct GlobalMapCache {
    handle: Option<Handle<Image>>,
    width: u32,
    height: u32,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn update_global_map(
    state: Res<MapViewState>,
    planetary: Option<Res<PlanetaryData>>,
    planet_config: Option<Res<PlanetConfig>>,
    cam_q: Query<&Transform, (With<FpsCamera>, With<Player>)>,
    mut images: ResMut<Assets<Image>>,
    mut map_node_q: Query<(&mut ImageNode, &mut Node), (With<MapImageNode>, Without<PlayerMarker>)>,
    mut coord_text_q: Query<&mut Text, With<MapCoordText>>,
    mut marker_q: Query<&mut Node, (With<PlayerMarker>, Without<MapImageNode>)>,
    mut cache: Local<GlobalMapCache>,
) {
    if state.view != MapView::Global {
        return;
    }

    let Some(planetary) = planetary else {
        // Not a planetary world — show placeholder text.
        if let Ok(mut text) = coord_text_q.single_mut() {
            **text = "Global map not available (flat world)".to_string();
        }
        return;
    };

    // Render the projection on first access.
    if cache.handle.is_none() {
        let planet_data = &*planetary.0;
        let rgb_img = render_projection(
            planet_data,
            &Projection::Equirectangular,
            &ColourMode::Elevation,
            PROJECTION_WIDTH,
        );

        let (w, h) = (rgb_img.width(), rgb_img.height());
        cache.width = w;
        cache.height = h;

        // Convert RGB → RGBA for Bevy.
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for pixel in rgb_img.pixels() {
            rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
        }

        let image = Image::new(
            Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            rgba,
            TextureFormat::Rgba8UnormSrgb,
            default(),
        );

        cache.handle = Some(images.add(image));
    }

    // Update the UI image and apply zoom/pan transform.
    if let (Ok((mut img_node, mut node)), Some(h)) = (map_node_q.single_mut(), cache.handle.clone())
    {
        img_node.image = h;
        let zoom = state.global_zoom;
        let pan = state.global_pan;
        node.width = Val::Percent(zoom * 100.0);
        node.height = Val::Percent(zoom * 100.0);
        // Position so that the viewport center shows image UV (0.5 - pan.x, 0.5 - pan.y).
        node.left = Val::Percent((1.0 - zoom) * 50.0 + zoom * pan.x * 100.0);
        node.top = Val::Percent((1.0 - zoom) * 50.0 + zoom * pan.y * 100.0);
    }

    // Compute player position on the map.
    let Ok(cam_tf) = cam_q.single() else {
        return;
    };

    let player_pos = cam_tf.translation;
    let (lat, lon) = if let Some(ref config) = planet_config {
        config.lat_lon(player_pos.as_dvec3())
    } else {
        (0.0, 0.0)
    };

    // Equirectangular: x maps to longitude, y maps to latitude.
    // lon ∈ [-π, π] → u ∈ [0, 1]
    // lat ∈ [-π/2, π/2] → v ∈ [1, 0] (top = +90°, bottom = -90°)
    let u = (lon / std::f64::consts::TAU + 0.5) as f32;
    let v = (0.5 - lat / std::f64::consts::PI) as f32;

    // Apply zoom and pan.
    let zoom = state.global_zoom;
    let pan = state.global_pan;
    let u_zoomed = (u - 0.5) * zoom + 0.5 + pan.x * zoom;
    let v_zoomed = (v - 0.5) * zoom + 0.5 + pan.y * zoom;

    let marker_pct_x = u_zoomed * 100.0;
    let marker_pct_y = v_zoomed * 100.0;

    if let Ok(mut node) = marker_q.single_mut() {
        node.left = Val::Percent(marker_pct_x.clamp(0.0, 100.0));
        node.top = Val::Percent(marker_pct_y.clamp(0.0, 100.0));
    }

    // Update coordinate text.
    let lat_deg = lat.to_degrees();
    let lon_deg = lon.to_degrees();
    let ns = if lat_deg >= 0.0 { "N" } else { "S" };
    let ew = if lon_deg >= 0.0 { "E" } else { "W" };

    if let Ok(mut text) = coord_text_q.single_mut() {
        **text = format!(
            "{:.1}°{ns} {:.1}°{ew}  Zoom: {:.1}×",
            lat_deg.abs(),
            lon_deg.abs(),
            zoom,
        );
    }
}
