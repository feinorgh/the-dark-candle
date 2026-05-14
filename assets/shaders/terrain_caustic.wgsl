// Terrain + ExtendedMaterial caustic shader.
//
// Phase C: adds an underwater caustic contribution to the StandardMaterial
// PBR output.  The contribution is:
//   - masked smoothly by depth below sea level (avoids waterline stipple
//     from f32 precision at planet-radius scale),
//   - modulated by the sun-up angle (no caustic at night / at the horizon),
//   - attenuated by Beer–Lambert extinction over depth,
//   - patterned by a 2-octave procedural Worley F2 - F1 field in
//     world-XZ space (matches the CPU oracle in
//     `src/lighting/caustic_tile.rs`),
//   - zero-mean (shifted by the empirical mean of the Worley field) so the
//     average underwater brightness is unchanged — the caustic only adds
//     sparkle, it does NOT double-count diffuse light from PBR.
//
// World-XZ coordinates in the floating-origin frame stay small (camera-
// relative), which keeps Worley UVs within f32-friendly precision even at
// a 6.37 Mm planet radius.

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

// --- procedural seamless Worley ----------------------------------------

fn _hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3)),
    );
    return fract(sin(q) * 43758.547);
}

fn _worley_f2_minus_f1(p: vec2<f32>) -> f32 {
    let ip = floor(p);
    let fp = p - ip;
    var f1: f32 = 1.0e9;
    var f2: f32 = 1.0e9;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offs = vec2<f32>(f32(x), f32(y));
            let h = _hash22(ip + offs);
            let pt = offs + h;
            let d = length(fp - pt);
            if (d < f1) { f2 = f1; f1 = d; }
            else if (d < f2) { f2 = d; }
        }
    }
    return f2 - f1;
}

// Sharpened "bright filament" Worley sample: high where F2-F1 is small.
fn _caustic_sample(uv: vec2<f32>) -> f32 {
    let w0 = _worley_f2_minus_f1(uv);
    let w1 = _worley_f2_minus_f1(uv * 2.0);
    let v0 = pow(clamp(1.0 - w0, 0.0, 1.0), 6.0);
    let v1 = pow(clamp(1.0 - w1, 0.0, 1.0), 8.0);
    // Empirical mean of `v0 + 0.5 * v1` for hash-uniform feature points;
    // matches the zero-mean normalization in caustic_tile.rs to within a
    // few percent.
    return (v0 + 0.5 * v1) - 0.18;
}

// -----------------------------------------------------------------------

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

    // ---- caustic contribution (forward path only) --------------------
    let strength = caustic.params.w;
    if (strength > 0.0) {
        let world_pos = in.world_position.xyz;
        let radial = world_pos - caustic.planet_center.xyz;
        let radius = length(radial);
        let depth = caustic.params.x - radius;  // sea level - radius
        // Smooth waterline mask over ~2 m so f32 precision noise at
        // planet scale doesn't stipple the surface boundary.
        let mask = smoothstep(0.0, 2.0, depth);
        if (mask > 0.0) {
            let tile_size = max(caustic.params.y, 0.001);
            let uv = world_pos.xz / tile_size;
            let pattern = _caustic_sample(uv);
            // Sun-zenith factor: caustics are strongest under a high sun.
            let up = radial / max(radius, 1.0);
            let cos_sun = max(0.0, dot(up, caustic.sun_dir.xyz));
            let attenuation = exp(-caustic.params.z * depth);
            let contribution = pattern * mask * cos_sun * attenuation * strength;
            // Slight cool-white tint so the sparkle reads as light-through-water.
            out.color = out.color + vec4<f32>(
                contribution * 0.85,
                contribution * 0.95,
                contribution * 1.00,
                0.0,
            );
        }
    }
#endif

    return out;
}

