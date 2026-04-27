//! Debug visualisation for the V2 cubed-sphere chunk pipeline.
//!
//! **F6** — toggle chunk wireframe overlay.
//!
//! When enabled:
//! - Every loaded V2 chunk is drawn as an oriented wireframe cuboid, coloured
//!   by LOD level (LOD 0 = green … LOD 4 = red).
//! - A translucent circle marks sea level at the player's current position.
//! - A line segment and sphere show the expected terrain surface height at the
//!   player's (lat, lon).
//! - Diagnostic text is printed to stdout (one line per frame so it can be
//!   tailed; also shown in-engine via F3 HUD via [`V2PipelineStats`]).

use bevy::math::{DVec3, Isometry3d};
use bevy::prelude::*;

use crate::floating_origin::{RenderOrigin, WorldPosition};
use crate::world::chunk::CHUNK_SIZE;
use crate::world::planet::PlanetConfig;
use crate::world::v2::chunk_manager::{V2ChunkCoord, V2ChunkMarker, V2TerrainGen};

// ── Colour palette by LOD ────────────────────────────────────────────────────

const LOD_COLOURS: [Color; 5] = [
    Color::srgb(0.0, 1.0, 0.2), // LOD 0 — green  (closest)
    Color::srgb(0.0, 0.9, 0.9), // LOD 1 — cyan
    Color::srgb(1.0, 1.0, 0.0), // LOD 2 — yellow
    Color::srgb(1.0, 0.5, 0.0), // LOD 3 — orange
    Color::srgb(1.0, 0.1, 0.1), // LOD 4 — red    (coarsest)
];

fn lod_colour(lod: u8) -> Color {
    let idx = (lod as usize).min(LOD_COLOURS.len() - 1);
    LOD_COLOURS[idx]
}

// ── Resource ─────────────────────────────────────────────────────────────────

/// Toggle state for the F6 chunk debug overlay.
#[derive(Resource, Default)]
pub struct ChunkDebugViz {
    pub enabled: bool,
}

// ── Systems ──────────────────────────────────────────────────────────────────

/// Handle F6 to toggle the chunk wireframe overlay.
pub fn toggle_chunk_debug_viz(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<ChunkDebugViz>,
) {
    if keyboard.just_pressed(KeyCode::F6) {
        state.enabled = !state.enabled;
        info!(
            "[ChunkDebug] Chunk wireframe overlay {} (F6)",
            if state.enabled { "ENABLED" } else { "DISABLED" }
        );
    }
}

/// Draw chunk wireframes + height reference markers when F6 overlay is active.
pub fn draw_chunk_debug_viz(
    state: Res<ChunkDebugViz>,
    mut gizmos: Gizmos,
    planet: Res<PlanetConfig>,
    origin: Res<RenderOrigin>,
    terrain_gen: Option<Res<V2TerrainGen>>,
    camera_q: Query<&WorldPosition, With<crate::camera::FpsCamera>>,
    chunks_q: Query<(&Transform, &V2ChunkCoord), With<V2ChunkMarker>>,
) {
    if !state.enabled {
        return;
    }

    let cs_f = CHUNK_SIZE as f32;
    let cs_half_f = cs_f / 2.0;

    // ── Chunk wireframes ─────────────────────────────────────────────────────
    for (transform, coord) in &chunks_q {
        let colour = lod_colour(coord.0.lod);
        // Reconstruct the chunk centre in render space.
        // transform.translation = adjusted = centre - rotation * (scale * CS/2)
        // therefore centre = translation + rotation * (scale * CS/2)
        let half_local = transform.scale * cs_half_f;
        let centre = transform.translation + transform.rotation * half_local;

        // The side lengths of the bounding box in world space.
        let side = transform.scale * cs_f;

        // Draw 12 edges of the oriented bounding box.
        let iso = Isometry3d::new(centre, transform.rotation);
        gizmos.rounded_cuboid(iso, side, colour);
    }

    // ── Height reference markers ─────────────────────────────────────────────
    let Ok(cam_world) = camera_q.single() else {
        return;
    };
    let cam_wpos: DVec3 = cam_world.0;
    let cam_r = cam_wpos.length();

    // Camera position in render space.
    let cam_render = {
        let d = cam_wpos - origin.0;
        Vec3::new(d.x as f32, d.y as f32, d.z as f32)
    };

    // Radially-outward unit direction at the camera (approx. "up").
    let up = if cam_render.length_squared() > 1e-6 {
        cam_render.normalize()
    } else {
        Vec3::Y
    };

    // Sea level marker — draw a circle in the tangent plane at sea level.
    let sea_r = planet.sea_level_radius;
    let sea_offset = (sea_r - cam_r) as f32;
    let sea_pos = cam_render + up * sea_offset;
    let sea_iso = Isometry3d::new(sea_pos, Quat::from_rotation_arc(Vec3::Z, up));
    gizmos.circle(sea_iso, 256.0, Color::srgba(0.2, 0.5, 1.0, 0.7));

    // Terrain surface marker — sample the terrain height at the player's lat/lon.
    if let Some(tg) = terrain_gen {
        let dir = cam_wpos.normalize();
        let (lat, lon) = crate::planet::detail::pos_to_lat_lon(dir);
        let surface_r = tg.0.sample_surface_radius_at(lat, lon);
        let surface_offset = (surface_r - cam_r) as f32;
        let surface_pos = cam_render + up * surface_offset;

        // Vertical line from sea level to surface height.
        let line_lo = cam_render + up * sea_offset.min(surface_offset);
        let line_hi = cam_render + up * sea_offset.max(surface_offset);
        gizmos.line(line_lo, line_hi, Color::srgb(1.0, 0.8, 0.0));

        // Small sphere at the expected surface height.
        gizmos.sphere(
            Isometry3d::from_translation(surface_pos),
            32.0,
            Color::srgb(1.0, 0.6, 0.0),
        );

        // Diagnostic stdout line (not every frame — only when enabled so it
        // doesn't spam when the overlay is off).
        let alt_above_sea = cam_r - sea_r;
        let alt_above_surface = cam_r - surface_r;
        trace!(
            "[ChunkDebug] lat={:.2}° lon={:.2}° | alt(sea)={:.1}m alt(surface)={:.1}m | surface_r={:.1}m sea_r={:.1}m",
            lat.to_degrees(),
            lon.to_degrees(),
            alt_above_sea,
            alt_above_surface,
            surface_r,
            sea_r,
        );
    }
}
