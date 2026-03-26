//! Billboard-quad rendering for weather particles.
//!
//! Rebuilds a single dynamic [`Mesh`] each frame with camera-facing quads for
//! every alive particle in [`ParticleReadback`].  Rain particles are elongated
//! velocity-aligned streaks; snow particles are small square billboards.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

use crate::gpu::particles::{GpuParticle, kind};

// ─── Resources & Components ────────────────────────────────────────────────

/// Latest particle positions read back from the GPU (or CPU sim) for rendering.
#[derive(Resource, Default)]
pub struct ParticleReadback {
    /// Current frame's particle data.
    pub particles: Vec<GpuParticle>,
}

/// Marker component for the entity that carries the combined particle mesh.
#[derive(Component)]
pub struct ParticleMeshMarker;

/// Tunable knobs for particle billboard rendering.
#[derive(Resource, Debug, Clone)]
pub struct ParticleRenderConfig {
    /// Performance cap: maximum quads emitted per frame.
    pub max_render_particles: usize,
    /// Rain streak length in meters.
    pub rain_streak_length: f32,
    /// Rain streak width in meters.
    pub rain_streak_width: f32,
    /// Snow quad half-extent in meters (full side = 2 × this is wrong — this
    /// is the full side length; halved internally).
    pub snow_quad_size: f32,
    /// Master toggle for the particle renderer.
    pub enabled: bool,
}

impl Default for ParticleRenderConfig {
    fn default() -> Self {
        Self {
            max_render_particles: 20_000,
            rain_streak_length: 0.3,
            rain_streak_width: 0.02,
            snow_quad_size: 0.05,
            enabled: true,
        }
    }
}

// ─── Plugin ────────────────────────────────────────────────────────────────

/// Registers the particle billboard renderer.
pub struct ParticleRenderPlugin;

impl Plugin for ParticleRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleReadback>()
            .init_resource::<ParticleRenderConfig>()
            .add_systems(Startup, spawn_particle_mesh)
            .add_systems(Update, update_particle_mesh);
    }
}

// ─── Systems ───────────────────────────────────────────────────────────────

/// Spawn the single entity that will hold the combined particle mesh.
fn spawn_particle_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    commands.spawn((
        ParticleMeshMarker,
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            double_sided: true,
            cull_mode: None,
            depth_bias: 1.0,
            ..default()
        })),
        Transform::default(),
        Visibility::default(),
    ));
}

/// Rebuild the particle mesh every frame from the current readback data.
fn update_particle_mesh(
    readback: Res<ParticleReadback>,
    config: Res<ParticleRenderConfig>,
    camera_q: Query<&GlobalTransform, With<Camera3d>>,
    mesh_q: Query<&Mesh3d, With<ParticleMeshMarker>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !config.enabled {
        return;
    }

    let Ok(cam_gt) = camera_q.single() else {
        return;
    };
    let cam_forward = cam_gt.forward().as_vec3();

    let Ok(mesh_handle) = mesh_q.single() else {
        return;
    };

    let Some(mesh) = meshes.get_mut(&mesh_handle.0) else {
        return;
    };

    let new_mesh = build_particle_mesh(&readback.particles, cam_forward, &config);
    *mesh = new_mesh;
}

/// Four corner positions plus an RGBA colour shared by all four vertices.
type QuadVerts = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 4]);

// ─── Mesh builder ──────────────────────────────────────────────────────────

/// Build a single [`Mesh`] containing billboard quads for all alive particles.
fn build_particle_mesh(
    particles: &[GpuParticle],
    camera_forward: Vec3,
    config: &ParticleRenderConfig,
) -> Mesh {
    let cap = config.max_render_particles;

    // Pre-compute billboard basis vectors.
    let right = camera_forward.cross(Vec3::Y).normalize_or(Vec3::X);
    let up = right.cross(camera_forward).normalize();

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let normal = (-camera_forward).to_array();
    let mut count: usize = 0;

    for p in particles {
        if p.life <= 0.0 || count >= cap {
            if count >= cap {
                break;
            }
            continue;
        }
        count += 1;

        let base = positions.len() as u32;
        let pos = Vec3::from(p.position);

        let (v0, v1, v2, v3, color) = match p.kind {
            kind::RAIN => rain_quad(pos, p.velocity, camera_forward, config),
            _ => snow_quad(pos, right, up, config),
        };

        positions.extend_from_slice(&[v0, v1, v2, v3]);
        normals.extend_from_slice(&[normal; 4]);
        colors.extend_from_slice(&[color; 4]);
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Generate a velocity-aligned rain streak quad.
fn rain_quad(
    pos: Vec3,
    velocity: [f32; 3],
    camera_forward: Vec3,
    config: &ParticleRenderConfig,
) -> QuadVerts {
    let vel = Vec3::from(velocity);
    let vel_dir = vel.normalize_or(Vec3::NEG_Y);
    let half_len = config.rain_streak_length / 2.0;
    let side =
        vel_dir.cross(camera_forward).normalize_or(Vec3::X) * (config.rain_streak_width / 2.0);

    let tail = pos - vel_dir * half_len;
    let head = pos + vel_dir * half_len;

    let color = [0.7, 0.8, 1.0, 0.4];
    (
        (tail - side).to_array(),
        (tail + side).to_array(),
        (head + side).to_array(),
        (head - side).to_array(),
        color,
    )
}

/// Generate a camera-facing square snow quad.
fn snow_quad(pos: Vec3, right: Vec3, up: Vec3, config: &ParticleRenderConfig) -> QuadVerts {
    let half = config.snow_quad_size / 2.0;
    let color = [1.0, 1.0, 1.0, 0.8];
    (
        (pos + (-right - up) * half).to_array(),
        (pos + (right - up) * half).to_array(),
        (pos + (right + up) * half).to_array(),
        (pos + (-right + up) * half).to_array(),
        color,
    )
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_particle(particle_kind: u32, life: f32) -> GpuParticle {
        GpuParticle {
            position: [10.0, 50.0, 10.0],
            life,
            velocity: [0.0, -5.0, 0.0],
            kind: particle_kind,
            mass: 1.0e-4,
            _pad: [0.0; 7],
        }
    }

    fn default_config() -> ParticleRenderConfig {
        ParticleRenderConfig::default()
    }

    fn forward() -> Vec3 {
        Vec3::NEG_Z
    }

    #[test]
    fn empty_particles_produce_empty_mesh() {
        let mesh = build_particle_mesh(&[], forward(), &default_config());
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .expect("positions attribute should exist");
        assert_eq!(positions.len(), 0);
    }

    #[test]
    fn rain_generates_elongated_quads() {
        let config = default_config();
        let p = make_particle(kind::RAIN, 5.0);
        let mesh = build_particle_mesh(&[p], forward(), &config);

        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions");
        assert_eq!(positions.len(), 4, "one quad = 4 vertices");

        // Extract raw vertex positions.
        let verts: Vec<Vec3> = match positions {
            bevy::mesh::VertexAttributeValues::Float32x3(v) => {
                v.iter().map(|a| Vec3::from(*a)).collect()
            }
            _ => panic!("unexpected attribute format"),
        };

        // The quad should be taller than it is wide.
        let height = (verts[2] - verts[0]).length();
        let width = (verts[1] - verts[0]).length();
        assert!(
            height > width,
            "rain quad should be elongated: height={height}, width={width}"
        );
    }

    #[test]
    fn snow_generates_square_quads() {
        let config = default_config();
        let p = make_particle(kind::SNOW, 5.0);
        let mesh = build_particle_mesh(&[p], forward(), &config);

        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions");
        assert_eq!(positions.len(), 4);

        let verts: Vec<Vec3> = match positions {
            bevy::mesh::VertexAttributeValues::Float32x3(v) => {
                v.iter().map(|a| Vec3::from(*a)).collect()
            }
            _ => panic!("unexpected attribute format"),
        };

        let side_a = (verts[1] - verts[0]).length();
        let side_b = (verts[2] - verts[1]).length();
        let ratio = side_a / side_b;
        assert!(
            (0.95..=1.05).contains(&ratio),
            "snow quad should be roughly square: side_a={side_a}, side_b={side_b}, ratio={ratio}"
        );
    }

    #[test]
    fn max_render_cap_respected() {
        let mut config = default_config();
        config.max_render_particles = 10;

        let particles: Vec<GpuParticle> =
            (0..100).map(|_| make_particle(kind::RAIN, 5.0)).collect();

        let mesh = build_particle_mesh(&particles, forward(), &config);
        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions");
        assert_eq!(
            positions.len(),
            40,
            "10 quads × 4 verts = 40, got {}",
            positions.len()
        );
    }

    #[test]
    fn dead_particles_skipped() {
        let config = default_config();
        let particles = vec![
            make_particle(kind::RAIN, 0.0),
            make_particle(kind::SNOW, -1.0),
            make_particle(kind::RAIN, 5.0),
        ];

        let mesh = build_particle_mesh(&particles, forward(), &config);
        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions");
        assert_eq!(
            positions.len(),
            4,
            "only the one alive particle should produce vertices"
        );
    }
}
