// Terrain + ExtendedMaterial caustic shader.
//
// Phase B0 (current): a near-no-op that uses the StandardMaterial PBR pipeline
// unchanged.  This proves the bind-group layout and shader compile against the
// terrain meshes.  Phase C will add the underwater mask + procedural caustic
// projection on top.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

// Caustic extension uniform.
//
// Field layout MUST match `CausticUniform` on the Rust side
// (`src/lighting/terrain_caustic_material.rs`).  All vectors are 16-byte
// aligned per WGSL std140-ish rules; the helper scalars live in the `.w`
// component of the trailing Vec4 to keep the struct a multiple of 16 bytes.
struct CausticUniform {
    // xyz = sun direction in render-space (unit vector pointing FROM surface TO sun)
    // w   = unused
    sun_dir: vec4<f32>,

    // xyz = planet center in render-space (== -RenderOrigin)
    // w   = unused
    planet_center: vec4<f32>,

    // x = sea level radius (m)
    // y = tile size in meters (caustic texture tile period)
    // z = depth falloff coefficient (1/m, Beer-Lambert)
    // w = caustic strength multiplier (0 disables effect; 1 is nominal)
    params: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> caustic: CausticUniform;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Generate the standard PBR input from the StandardMaterial bindings.
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    // Phase B0: caustic extension is bound but does not modify output.
    // Reference the uniform once so the binding isn't optimized out.
    let _unused = caustic.params.w * 0.0;
    out.color.r += _unused;
#endif

    return out;
}
