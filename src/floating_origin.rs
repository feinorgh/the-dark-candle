// Floating-origin system for large-world precision.
//
// All simulation-relevant entities store their true position as `WorldPosition(DVec3)`
// in planet-centered f64 coordinates. Bevy `Transform` is render-space only:
// `Transform.translation = (WorldPosition - RenderOrigin).as_vec3()`.
//
// The `RenderOrigin` is rebased (shifted) when the camera moves more than
// `REBASE_THRESHOLD` meters from the current origin. On rebase, all entity
// Transforms are updated by subtracting the shift delta.

use bevy::math::DVec3;
use bevy::prelude::*;

/// Rebase when camera offset from RenderOrigin exceeds this distance (meters).
/// At 512 m, f32 precision is ~0.00003 m — invisible for 1 m voxels.
const REBASE_THRESHOLD: f64 = 512.0;

/// True world position of an entity in planet-centered f64 coordinates.
///
/// This is the source of truth for position. The Bevy `Transform` is derived
/// from this by subtracting the current `RenderOrigin`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct WorldPosition(pub DVec3);

impl WorldPosition {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self(DVec3::new(x, y, z))
    }

    pub fn from_dvec3(v: DVec3) -> Self {
        Self(v)
    }

    pub fn from_vec3(v: Vec3) -> Self {
        Self(DVec3::new(v.x as f64, v.y as f64, v.z as f64))
    }

    /// Convert to render-space Vec3 relative to the given origin.
    pub fn render_offset(&self, origin: &RenderOrigin) -> Vec3 {
        let d = self.0 - origin.0;
        Vec3::new(d.x as f32, d.y as f32, d.z as f32)
    }
}

/// The current rendering origin in planet-centered f64 coordinates.
///
/// All Bevy Transforms are relative to this point. Resets periodically
/// to keep f32 values near zero.
#[derive(Resource, Debug, Clone, Copy)]
pub struct RenderOrigin(pub DVec3);

impl Default for RenderOrigin {
    fn default() -> Self {
        Self(DVec3::ZERO)
    }
}

/// Indicates that a rebase just occurred this frame. Contains the shift delta.
///
/// Systems that cache absolute positions can check for this resource to update.
/// Inserted and removed each frame a rebase occurs.
#[derive(Resource, Debug, Clone, Copy)]
pub struct RenderOriginShift(pub DVec3);

/// Plugin that manages the floating-origin rebase cycle.
pub struct FloatingOriginPlugin;

impl Plugin for FloatingOriginPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderOrigin>()
            .add_systems(
                PostUpdate,
                rebase_origin
                    .before(bevy::transform::TransformSystems::Propagate),
            );
    }
}

/// Check if the camera has moved far enough to trigger a rebase.
/// If so, shift RenderOrigin and update all entity Transforms.
fn rebase_origin(
    mut origin: ResMut<RenderOrigin>,
    mut commands: Commands,
    camera_q: Query<&WorldPosition, With<Camera>>,
    mut transform_q: Query<&mut Transform, Without<Camera>>,
    mut camera_transform_q: Query<(&WorldPosition, &mut Transform), With<Camera>>,
    existing_shift: Option<Res<RenderOriginShift>>,
) {
    // Remove previous frame's shift marker
    if existing_shift.is_some() {
        commands.remove_resource::<RenderOriginShift>();
    }

    let Ok(cam_world) = camera_q.single() else {
        return;
    };

    let offset = cam_world.0 - origin.0;
    if offset.length() < REBASE_THRESHOLD {
        return;
    }

    // Rebase: new origin = camera's current world position
    let shift = cam_world.0 - origin.0;
    origin.0 = cam_world.0;
    let shift_f32 = Vec3::new(shift.x as f32, shift.y as f32, shift.z as f32);

    commands.insert_resource(RenderOriginShift(shift));

    // Update all non-camera entity Transforms
    for mut transform in &mut transform_q {
        transform.translation -= shift_f32;
    }

    // Update camera Transform (set to zero offset since origin = camera pos)
    for (wp, mut transform) in &mut camera_transform_q {
        let d = wp.0 - origin.0;
        transform.translation = Vec3::new(d.x as f32, d.y as f32, d.z as f32);
    }
}

/// Sync a WorldPosition entity's Transform.translation to render-space.
///
/// This system runs for entities that have both WorldPosition and Transform,
/// keeping the Transform in sync with the floating origin. Camera entities
/// handle their own sync in the camera systems.
pub fn sync_world_position_to_transform(
    origin: Res<RenderOrigin>,
    mut query: Query<(&WorldPosition, &mut Transform), (Changed<WorldPosition>, Without<Camera>)>,
) {
    for (wp, mut transform) in &mut query {
        transform.translation = wp.render_offset(&origin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_position_render_offset() {
        let wp = WorldPosition(DVec3::new(6_371_100.0, 0.0, 0.0));
        let origin = RenderOrigin(DVec3::new(6_371_000.0, 0.0, 0.0));
        let offset = wp.render_offset(&origin);
        assert!((offset.x - 100.0).abs() < 0.01);
        assert!(offset.y.abs() < 0.01);
        assert!(offset.z.abs() < 0.01);
    }

    #[test]
    fn world_position_from_vec3() {
        let v = Vec3::new(32000.0, 100.0, -500.0);
        let wp = WorldPosition::from_vec3(v);
        assert!((wp.0.x - 32000.0).abs() < 0.1);
        assert!((wp.0.y - 100.0).abs() < 0.1);
        assert!((wp.0.z - (-500.0)).abs() < 0.1);
    }

    #[test]
    fn render_offset_near_zero_when_at_origin() {
        let wp = WorldPosition(DVec3::new(6_371_000.0, 0.0, 0.0));
        let origin = RenderOrigin(DVec3::new(6_371_000.0, 0.0, 0.0));
        let offset = wp.render_offset(&origin);
        assert!(offset.length() < 0.01);
    }

    #[test]
    fn render_offset_preserves_precision_at_large_distances() {
        // At Earth-scale, f64→f32 offset should be precise when origin is nearby
        let wp = WorldPosition(DVec3::new(6_371_000.5, 100.25, -200.75));
        let origin = RenderOrigin(DVec3::new(6_371_000.0, 100.0, -200.0));
        let offset = wp.render_offset(&origin);
        assert!((offset.x - 0.5).abs() < 0.001);
        assert!((offset.y - 0.25).abs() < 0.001);
        assert!((offset.z - (-0.75)).abs() < 0.001);
    }
}
