// Local discovery map: top-down view of visited chunk columns.
//
// Each pixel represents one chunk column (32 m), coloured by biome.
// Undiscovered regions are transparent (the dark overlay shows through).

use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::camera::FpsCamera;
use crate::hud::Player;
use crate::planet::BiomeType;
use crate::world::chunk::CHUNK_SIZE;

use super::discovery::DiscoveredColumns;
use super::ui::{MapCoordText, MapImageNode, MapView, MapViewState, PlayerMarker};

/// Map image side length in pixels at zoom=1 (1 pixel per chunk).
const BASE_MAP_SIZE: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn update_local_map(
    state: Res<MapViewState>,
    discovered: Res<DiscoveredColumns>,
    cam_q: Query<&crate::floating_origin::WorldPosition, (With<FpsCamera>, With<Player>)>,
    mut images: ResMut<Assets<Image>>,
    mut map_node_q: Query<&mut ImageNode, With<MapImageNode>>,
    mut coord_text_q: Query<&mut Text, With<MapCoordText>>,
    mut marker_q: Query<&mut Node, With<PlayerMarker>>,
    mut cached_handle: Local<Option<Handle<Image>>>,
) {
    if state.view != MapView::Local {
        return;
    }

    let Ok(cam_wp) = cam_q.single() else {
        return;
    };

    let player_pos = cam_wp.0.as_vec3();
    let player_chunk_x = (player_pos.x as i32).div_euclid(CHUNK_SIZE as i32);
    let player_chunk_z = (player_pos.z as i32).div_euclid(CHUNK_SIZE as i32);

    let zoom = state.local_zoom;
    let map_px = BASE_MAP_SIZE * zoom;
    let half_chunks = (BASE_MAP_SIZE / 2) as i32;

    // Build the map image.
    let format = TextureFormat::Rgba8UnormSrgb;
    let pixel_size = 4usize; // RGBA = 4 bytes
    let mut data = vec![0u8; (map_px * map_px) as usize * pixel_size];

    for cz_offset in -half_chunks..half_chunks {
        for cx_offset in -half_chunks..half_chunks {
            let cx = player_chunk_x + cx_offset;
            let cz = player_chunk_z + cz_offset;
            let key = [cx, cz];

            let color = if let Some(col) = discovered.columns.get(&key) {
                biome_rgba(col.biome)
            } else {
                [0, 0, 0, 0] // Transparent = undiscovered
            };

            // Each chunk occupies `zoom × zoom` pixels.
            let base_px_x = ((cx_offset + half_chunks) as u32) * zoom;
            let base_px_y = ((cz_offset + half_chunks) as u32) * zoom;

            for dy in 0..zoom {
                for dx in 0..zoom {
                    let px = base_px_x + dx;
                    let py = base_px_y + dy;
                    if px < map_px && py < map_px {
                        let idx = ((py * map_px + px) as usize) * pixel_size;
                        data[idx..idx + 4].copy_from_slice(&color);
                    }
                }
            }
        }
    }

    let image = Image::new(
        Extent3d {
            width: map_px,
            height: map_px,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        format,
        default(),
    );

    // Reuse or create handle.
    let handle = match cached_handle.as_ref() {
        Some(h) => {
            let _ = images.insert(h.id(), image);
            h.clone()
        }
        None => {
            let h = images.add(image);
            *cached_handle = Some(h.clone());
            h
        }
    };

    // Update the UI node's image.
    if let Ok(mut img_node) = map_node_q.single_mut() {
        img_node.image = handle;
    }

    // Player marker stays centred.
    if let Ok(mut node) = marker_q.single_mut() {
        node.left = Val::Percent(50.0);
        node.top = Val::Percent(50.0);
    }

    // Update coordinate text.
    if let Ok(mut text) = coord_text_q.single_mut() {
        **text = format!(
            "Position: ({:.0}, {:.0}, {:.0})  Chunk: ({}, {})  Zoom: {}×",
            player_pos.x, player_pos.y, player_pos.z, player_chunk_x, player_chunk_z, zoom,
        );
    }
}

/// Map `BiomeType` to an RGBA u8 colour for the local map.
fn biome_rgba(biome: BiomeType) -> [u8; 4] {
    let [r, g, b] = match biome {
        BiomeType::Ocean => [38, 77, 153],
        BiomeType::DeepOcean => [13, 26, 89],
        BiomeType::IceCap => [230, 242, 255],
        BiomeType::Tundra => [178, 191, 178],
        BiomeType::BorealForest => [51, 102, 64],
        BiomeType::ColdSteppe => [153, 153, 102],
        BiomeType::TemperateForest => [38, 140, 51],
        BiomeType::Alpine => [140, 128, 115],
        BiomeType::TropicalSavanna => [178, 166, 77],
        BiomeType::TropicalRainforest => [13, 102, 26],
        BiomeType::HotDesert => [217, 191, 128],
        BiomeType::ColdDesert => [153, 140, 128],
        BiomeType::Wetland => [77, 128, 89],
        BiomeType::Mangrove => [64, 115, 77],
    };
    [r, g, b, 255]
}
