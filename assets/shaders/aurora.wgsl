// Aurora — volumetric shell rendered as additive blend over the framebuffer.
// Placeholder: outputs vec4<f32>(0.0, 0.0, 0.0, 1.0). Real ray-march lands in Task 7.

#import bevy_pbr::mesh_functions::get_world_from_local
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::forward_io::Vertex

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> planet_center_render: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> magnetic_north_axis: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> sun_world_direction: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> aurora_params: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<uniform> aurora_band: vec4<f32>;

struct AuroraOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

@vertex
fn vertex(in: Vertex) -> AuroraOut {
    let world_from_local = get_world_from_local(in.instance_index);
    let world_pos = (world_from_local * vec4<f32>(in.position, 1.0)).xyz;
    let clip = view.clip_from_world * vec4<f32>(world_pos, 1.0);
    return AuroraOut(clip, world_pos);
}

@fragment
fn fragment(in: AuroraOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
