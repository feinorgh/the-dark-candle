// Particle physics step — compute shader for weather particle simulation.
//
// Dispatched as 1D workgroups (256 threads each) over the particle array.
// Each thread processes one particle: applies gravity, aerodynamic drag,
// and wind forces, then integrates position and velocity via forward Euler.

struct ParticleParams {
    dt: f32,
    particle_count: u32,
    gravity: f32,
    grid_size: u32,
    wind_global: vec3<f32>,
    cell_size: f32,
}

struct GpuParticle {
    position: vec3<f32>,
    life: f32,
    velocity: vec3<f32>,
    kind: u32,
    mass: f32,
    _pad: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: ParticleParams;
@group(1) @binding(0) var<storage, read> wind_field: array<vec4<f32>>;
@group(2) @binding(0) var<storage, read_write> particles: array<GpuParticle>;

// Standard air density at sea level (kg/m³).
const RHO_AIR: f32 = 1.225;

// Drag coefficient per particle kind.
fn drag_coefficient(kind: u32) -> f32 {
    switch kind {
        case 1u: { return 1.2; }       // Snow: high drag (flat crystal)
        case 3u: { return 0.45; }      // Hail: smooth sphere
        default: { return 0.47; }      // Rain (0) and Sand (2): sphere
    }
}

// Cross-sectional area per particle kind (m²).
fn cross_section_area(kind: u32) -> f32 {
    switch kind {
        case 0u: { return 7.07e-6; }   // Rain: π × (1.5e-3)²
        case 1u: { return 2.5e-5; }    // Snow: larger effective area
        case 2u: { return 3.14e-6; }   // Sand: π × (1e-3)²
        case 3u: { return 1.26e-5; }   // Hail: π × (2e-3)²
        default: { return 7.07e-6; }
    }
}

// Sample wind velocity from the 3D grid via trilinear interpolation.
fn sample_wind(pos: vec3<f32>) -> vec3<f32> {
    let gs = i32(params.grid_size);
    if gs <= 0 {
        return params.wind_global;
    }

    let gc = pos / params.cell_size;
    let g0 = vec3<i32>(floor(gc));

    // Clamp so both g0 and g0+1 are within [0, grid_size-1].
    let max_idx = gs - 1;
    let c0 = clamp(g0, vec3<i32>(0), vec3<i32>(max_idx - 1));
    let c1 = c0 + vec3<i32>(1);

    let frac = gc - vec3<f32>(c0);

    // Flat index: z * size * size + y * size + x
    let s = gs;
    let i000 = c0.z * s * s + c0.y * s + c0.x;
    let i100 = c0.z * s * s + c0.y * s + c1.x;
    let i010 = c0.z * s * s + c1.y * s + c0.x;
    let i110 = c0.z * s * s + c1.y * s + c1.x;
    let i001 = c1.z * s * s + c0.y * s + c0.x;
    let i101 = c1.z * s * s + c0.y * s + c1.x;
    let i011 = c1.z * s * s + c1.y * s + c0.x;
    let i111 = c1.z * s * s + c1.y * s + c1.x;

    let w000 = wind_field[i000].xyz;
    let w100 = wind_field[i100].xyz;
    let w010 = wind_field[i010].xyz;
    let w110 = wind_field[i110].xyz;
    let w001 = wind_field[i001].xyz;
    let w101 = wind_field[i101].xyz;
    let w011 = wind_field[i011].xyz;
    let w111 = wind_field[i111].xyz;

    // Trilinear interpolation.
    let fx = frac.x;
    let fy = frac.y;
    let fz = frac.z;

    let c00 = mix(w000, w100, fx);
    let c10 = mix(w010, w110, fx);
    let c01 = mix(w001, w101, fx);
    let c11 = mix(w011, w111, fx);
    let c0v = mix(c00, c10, fy);
    let c1v = mix(c01, c11, fy);
    return mix(c0v, c1v, fz);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.particle_count {
        return;
    }

    var p = particles[idx];

    // Dead particles are inert.
    if p.life <= 0.0 {
        return;
    }

    let wind = sample_wind(p.position);
    let v_rel = p.velocity - wind;
    let speed_rel = length(v_rel);

    // Gravity: F_g = m × (0, −g, 0)
    let f_gravity = vec3<f32>(0.0, -params.gravity * p.mass, 0.0);

    // Aerodynamic drag: F_d = −½ ρ_air Cd A |v_rel| v_rel
    let cd = drag_coefficient(p.kind);
    let area = cross_section_area(p.kind);
    let f_drag = v_rel * (-0.5 * RHO_AIR * cd * area * speed_rel);

    let f_total = f_gravity + f_drag;
    let accel = f_total / p.mass;

    // Forward Euler integration.
    p.velocity = p.velocity + accel * params.dt;
    p.position = p.position + p.velocity * params.dt;

    // Decrement lifetime.
    p.life = p.life - params.dt;

    // Ground boundary: kill particle on contact.
    if p.position.y < 0.0 {
        p.life = 0.0;
    }

    particles[idx] = p;
}
