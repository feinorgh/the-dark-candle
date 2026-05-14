//! Terrain caustic material — an [`ExtendedMaterial`] over [`StandardMaterial`]
//! that adds an underwater caustic-light contribution on top of the standard
//! PBR fragment.
//!
//! This is the Phase B0 scaffolding: the shader is wired up and compiles, but
//! does not yet modulate the output.  Phase C will add the underwater mask
//! and procedural caustic projection.
//!
//! # Why an `ExtendedMaterial`?
//!
//! The voxel terrain currently uses `StandardMaterial`; we want to keep all of
//! Bevy's PBR work (vertex colors, shadow sampling, distance fog, GI) and only
//! ADD a per-fragment underwater term.  Wrapping `StandardMaterial` in an
//! extension keeps the upstream pipeline unchanged while giving us a single
//! point to inject the caustic uniform and shader logic.

use bevy::asset::Asset;
use bevy::math::Vec4;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::floating_origin::RenderOrigin;
use crate::lighting::SunWorldDirection;
use crate::world::planet::PlanetConfig;

/// Asset path of the terrain caustic fragment shader.
pub const TERRAIN_CAUSTIC_SHADER: &str = "shaders/terrain_caustic.wgsl";

/// The combined terrain material type used by all chunk, stitch, and corner
/// meshes.
pub type TerrainCausticMaterial = ExtendedMaterial<StandardMaterial, TerrainCausticExt>;

/// Caustic extension uniform.  Field layout must match `CausticUniform` in
/// `assets/shaders/terrain_caustic.wgsl`.
///
/// All vectors are 16-byte aligned per WGSL `uniform` storage rules; auxiliary
/// scalars live in the trailing `.w` lane of the [`Self::params`] vector to
/// keep the total struct size a multiple of 16 bytes.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct TerrainCausticExt {
    /// `xyz` = sun direction in render-space (unit vector FROM surface TO sun).
    /// `w`   = unused.
    #[uniform(100)]
    pub sun_dir: Vec4,

    /// `xyz` = planet center in render-space (`== -RenderOrigin`).
    /// `w`   = unused.
    #[uniform(100)]
    pub planet_center: Vec4,

    /// Packed scalars:
    /// - `x` = sea level radius (m)
    /// - `y` = caustic tile size (m)
    /// - `z` = depth falloff (1/m, Beer–Lambert)
    /// - `w` = caustic strength multiplier (0 disables the effect)
    #[uniform(100)]
    pub params: Vec4,
}

impl Default for TerrainCausticExt {
    fn default() -> Self {
        Self {
            sun_dir: Vec4::new(0.0, 1.0, 0.0, 0.0),
            planet_center: Vec4::new(0.0, -6_371_000.0, 0.0, 0.0),
            // Phase B0: strength = 0 means the caustic shader is bound but
            // contributes nothing to the output.  Phase C will set this to
            // a nominal 1.0 once the projection math is in place.
            params: Vec4::new(6_371_000.0, 8.0, 0.05, 0.0),
        }
    }
}

impl MaterialExtension for TerrainCausticExt {
    fn fragment_shader() -> ShaderRef {
        TERRAIN_CAUSTIC_SHADER.into()
    }

    // Deferred path uses the same shader (it short-circuits via the
    // `PREPASS_PIPELINE` shader_def — see the WGSL).
    fn deferred_fragment_shader() -> ShaderRef {
        TERRAIN_CAUSTIC_SHADER.into()
    }
}

/// Construct a [`TerrainCausticMaterial`] for chunk faces (single-sided).
pub fn make_chunk_material() -> TerrainCausticMaterial {
    ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            // Force forward rendering: in deferred mode `ExtendedMaterial` can
            // only modify `PbrInput`, not the post-lighting output, so the
            // caustic contribution (added after `apply_pbr_lighting` in
            // Phase C) would be silently dropped.
            opaque_render_method: bevy::pbr::OpaqueRendererMethod::Forward,
            ..default()
        },
        extension: TerrainCausticExt::default(),
    }
}

/// Construct a [`TerrainCausticMaterial`] for stitch/corner-cap meshes which
/// need to be drawn from both sides (the boundary ribbons are 1-D in their
/// minor axis).
pub fn make_stitch_material() -> TerrainCausticMaterial {
    ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            double_sided: true,
            cull_mode: None,
            opaque_render_method: bevy::pbr::OpaqueRendererMethod::Forward,
            ..default()
        },
        extension: TerrainCausticExt::default(),
    }
}

/// Shared handles to the three terrain materials used by the chunk manager,
/// pairwise stitches, and corner caps.  Stored as a resource so all three
/// systems pick up the same material instance and so the
/// [`update_terrain_caustic_uniform`] system has a single mutation point.
#[derive(Resource, Default)]
pub struct TerrainCausticHandles {
    pub chunk: Option<Handle<TerrainCausticMaterial>>,
    pub stitch: Option<Handle<TerrainCausticMaterial>>,
    pub corner: Option<Handle<TerrainCausticMaterial>>,
}

/// Nominal physical parameters for the underwater caustic effect.  Values
/// are in SI units (metres / dimensionless cosine) and are documented in
/// the WGSL `CausticUniform` block.
pub const CAUSTIC_TILE_SIZE_M: f32 = 8.0;
pub const CAUSTIC_DEPTH_FALLOFF_INV_M: f32 = 0.06;
pub const CAUSTIC_STRENGTH: f32 = 1.0;

/// Phase D: per-frame update of the caustic uniform on all three terrain
/// materials.  Reads:
///   - [`SunWorldDirection`] for the live solar direction,
///   - [`RenderOrigin`] for the floating-origin offset (so we can express
///     the planet centre in render-space, where shader fragments live),
///   - [`PlanetConfig::sea_level_radius`] for the underwater mask boundary.
///
/// Must run in `PostUpdate` AFTER the floating-origin rebase, otherwise on
/// rebase frames the planet centre lags by one frame and the underwater
/// mask shimmers at the seam.
pub fn update_terrain_caustic_uniform(
    handles: Res<TerrainCausticHandles>,
    sun: Res<SunWorldDirection>,
    origin: Res<RenderOrigin>,
    planet: Res<PlanetConfig>,
    mut materials: ResMut<Assets<TerrainCausticMaterial>>,
) {
    // Sun direction in render-space is the same as world-space (rotation
    // only; floating origin is a translation).  Convert f64 to f32 here.
    let s = sun.0;
    let sun_dir = Vec4::new(s.x as f32, s.y as f32, s.z as f32, 0.0);

    // Planet centre is at the world origin (0,0,0); in render-space that
    // becomes `-RenderOrigin`.
    let pc = -origin.0;
    let planet_center = Vec4::new(pc.x as f32, pc.y as f32, pc.z as f32, 0.0);

    let params = Vec4::new(
        planet.sea_level_radius as f32,
        CAUSTIC_TILE_SIZE_M,
        CAUSTIC_DEPTH_FALLOFF_INV_M,
        CAUSTIC_STRENGTH,
    );

    for h in [
        handles.chunk.as_ref(),
        handles.stitch.as_ref(),
        handles.corner.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(mat) = materials.get_mut(h) {
            mat.extension.sun_dir = sun_dir;
            mat.extension.planet_center = planet_center;
            mat.extension.params = params;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uniform_is_disabled() {
        // Phase B0 default must not modulate the output (strength = 0).
        let ext = TerrainCausticExt::default();
        assert_eq!(ext.params.w, 0.0, "caustic strength must default to 0");
    }

    #[test]
    fn chunk_material_is_forward() {
        let mat = make_chunk_material();
        assert!(matches!(
            mat.base.opaque_render_method,
            bevy::pbr::OpaqueRendererMethod::Forward
        ));
    }

    #[test]
    fn stitch_material_is_double_sided_and_forward() {
        let mat = make_stitch_material();
        assert!(mat.base.double_sided);
        assert!(mat.base.cull_mode.is_none());
        assert!(matches!(
            mat.base.opaque_render_method,
            bevy::pbr::OpaqueRendererMethod::Forward
        ));
    }
}
