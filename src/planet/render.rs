//! Interactive 3D globe renderer for planetary visualization.
//!
//! Builds a triangle mesh from the geodesic grid and displays it in a Bevy
//! window with an orbital camera. Supports multiple colour modes, elevation
//! exaggeration, celestial overlays, and screenshot export.

use bevy::asset::RenderAssetUsages;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

use super::grid::CellId;
use super::{BiomeType, PlanetData, RockType};

// ─── Colour modes ─────────────────────────────────────────────────────────────

/// Visualization colour mode for the globe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColourMode {
    /// Blue (deep ocean) → green (lowland) → brown (highland) → white (peak).
    #[default]
    Elevation,
    /// Distinct colour per biome classification.
    Biome,
    /// Random colour per tectonic plate.
    Plates,
    /// Red (young) → blue (ancient).
    GeologicalAge,
    /// Yellow (thin) → purple (thick).
    CrustDepth,
    /// Grey (none) → cyan (peak) at epoch 0.
    TidalAmplitude,
    /// Colour per surface rock type.
    Rock,
    /// Blue (cold) → green → red (hot).
    Temperature,
}

impl ColourMode {
    fn label(self) -> &'static str {
        match self {
            Self::Elevation => "Elevation",
            Self::Biome => "Biome",
            Self::Plates => "Plates",
            Self::GeologicalAge => "Geological Age",
            Self::CrustDepth => "Crust Depth",
            Self::TidalAmplitude => "Tidal Amplitude",
            Self::Rock => "Surface Rock",
            Self::Temperature => "Temperature",
        }
    }

    /// Parse a colour mode from a CLI name string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "elevation" => Some(Self::Elevation),
            "biome" => Some(Self::Biome),
            "plates" => Some(Self::Plates),
            "age" | "geological_age" => Some(Self::GeologicalAge),
            "crust" | "crust_depth" => Some(Self::CrustDepth),
            "tidal" | "tidal_amplitude" => Some(Self::TidalAmplitude),
            "rock" => Some(Self::Rock),
            "temperature" | "temp" => Some(Self::Temperature),
            _ => None,
        }
    }
}

// ─── Colour mapping ───────────────────────────────────────────────────────────

pub(crate) fn cell_color(data: &PlanetData, cell: usize, mode: &ColourMode) -> [f32; 4] {
    match mode {
        ColourMode::Elevation => elevation_color(data.elevation[cell]),
        ColourMode::Biome => biome_color(data.biome[cell]),
        ColourMode::Plates => plate_color(data.plate_id[cell]),
        ColourMode::GeologicalAge => age_color(data.geological_age[cell]),
        ColourMode::CrustDepth => crust_depth_color(data.crust_depth[cell]),
        ColourMode::TidalAmplitude => {
            let pos = data.grid.cell_position(CellId(cell as u32));
            tidal_color(data.celestial.tidal_height_at(pos, 0.0))
        }
        ColourMode::Rock => rock_color(data.surface_rock[cell]),
        ColourMode::Temperature => temperature_color(data.temperature_k[cell]),
    }
}

fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn elevation_color(elev: f64) -> [f32; 4] {
    let e = elev as f32;
    let rgb = if e < -4000.0 {
        [0.05, 0.05, 0.3]
    } else if e < 0.0 {
        lerp_rgb([0.05, 0.05, 0.3], [0.2, 0.4, 0.7], (e + 4000.0) / 4000.0)
    } else if e < 500.0 {
        lerp_rgb([0.15, 0.5, 0.15], [0.3, 0.6, 0.2], e / 500.0)
    } else if e < 2000.0 {
        lerp_rgb([0.3, 0.6, 0.2], [0.6, 0.4, 0.2], (e - 500.0) / 1500.0)
    } else if e < 5000.0 {
        lerp_rgb([0.6, 0.4, 0.2], [0.7, 0.7, 0.7], (e - 2000.0) / 3000.0)
    } else {
        lerp_rgb(
            [0.7, 0.7, 0.7],
            [1.0, 1.0, 1.0],
            ((e - 5000.0) / 4000.0).min(1.0),
        )
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn biome_color(biome: BiomeType) -> [f32; 4] {
    let rgb = match biome {
        BiomeType::Ocean => [0.15, 0.30, 0.60],
        BiomeType::DeepOcean => [0.05, 0.10, 0.35],
        BiomeType::IceCap => [0.90, 0.95, 1.00],
        BiomeType::Tundra => [0.70, 0.75, 0.70],
        BiomeType::BorealForest => [0.20, 0.40, 0.25],
        BiomeType::ColdSteppe => [0.60, 0.60, 0.40],
        BiomeType::TemperateForest => [0.15, 0.55, 0.20],
        BiomeType::Alpine => [0.55, 0.50, 0.45],
        BiomeType::TropicalSavanna => [0.70, 0.65, 0.30],
        BiomeType::TropicalRainforest => [0.05, 0.40, 0.10],
        BiomeType::HotDesert => [0.85, 0.75, 0.50],
        BiomeType::ColdDesert => [0.60, 0.55, 0.50],
        BiomeType::Wetland => [0.30, 0.50, 0.35],
        BiomeType::Mangrove => [0.25, 0.45, 0.30],
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn plate_color(plate_id: u8) -> [f32; 4] {
    const PALETTE: [[f32; 3]; 16] = [
        [0.90, 0.20, 0.20],
        [0.20, 0.70, 0.30],
        [0.20, 0.30, 0.90],
        [0.90, 0.80, 0.20],
        [0.80, 0.30, 0.80],
        [0.30, 0.80, 0.80],
        [0.90, 0.50, 0.20],
        [0.50, 0.20, 0.70],
        [0.60, 0.80, 0.30],
        [0.30, 0.50, 0.70],
        [0.80, 0.60, 0.50],
        [0.40, 0.70, 0.60],
        [0.70, 0.30, 0.50],
        [0.50, 0.60, 0.30],
        [0.30, 0.40, 0.60],
        [0.70, 0.50, 0.70],
    ];
    let rgb = PALETTE[plate_id as usize % PALETTE.len()];
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn age_color(age: f32) -> [f32; 4] {
    let rgb = lerp_rgb([0.20, 0.20, 0.80], [0.80, 0.20, 0.20], age);
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn crust_depth_color(depth: f32) -> [f32; 4] {
    let t = (depth / 50_000.0).clamp(0.0, 1.0);
    let rgb = lerp_rgb([0.90, 0.90, 0.20], [0.50, 0.10, 0.70], t);
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn tidal_color(height: f64) -> [f32; 4] {
    let t = ((height as f32 + 1.0) / 2.0).clamp(0.0, 1.0);
    let rgb = lerp_rgb([0.30, 0.30, 0.30], [0.20, 0.80, 0.90], t);
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn rock_color(rock: RockType) -> [f32; 4] {
    let rgb = match rock {
        RockType::Basalt => [0.25, 0.25, 0.30],
        RockType::Granite => [0.70, 0.65, 0.60],
        RockType::Sandstone => [0.80, 0.70, 0.50],
        RockType::Limestone => [0.85, 0.85, 0.80],
        RockType::Shale => [0.40, 0.40, 0.35],
        RockType::Marble => [0.90, 0.90, 0.85],
        RockType::Quartzite => [0.75, 0.70, 0.65],
        RockType::Obsidian => [0.15, 0.15, 0.20],
        RockType::Peridotite => [0.35, 0.45, 0.25],
        RockType::Gneiss => [0.55, 0.50, 0.50],
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn temperature_color(temp_k: f32) -> [f32; 4] {
    let t = ((temp_k - 200.0) / 120.0).clamp(0.0, 1.0);
    let rgb = if t < 0.5 {
        lerp_rgb([0.10, 0.20, 0.80], [0.20, 0.80, 0.30], t * 2.0)
    } else {
        lerp_rgb([0.20, 0.80, 0.30], [0.90, 0.20, 0.10], (t - 0.5) * 2.0)
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
}

// ─── Mesh building ────────────────────────────────────────────────────────────

/// Build a Bevy [`Mesh`] from planet data using the given colour mode.
///
/// Each geodesic cell becomes a polygon fan of 5 (pentagon) or 6 (hexagon)
/// triangles. Ring vertices sit at the spherical centroids of the Delaunay
/// triangles meeting at each cell. Positions are displaced radially by
/// `elevation × exaggeration / radius`.
pub fn build_globe_mesh(data: &PlanetData, mode: &ColourMode, exaggeration: f64) -> Mesh {
    let n = data.grid.cell_count();
    let radius = data.config.radius_m;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(7 * n);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(7 * n);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(7 * n);
    let mut indices: Vec<u32> = Vec::with_capacity(18 * n);

    for ci in 0..n {
        let cell_pos = data.grid.cell_position(CellId(ci as u32));
        let cell_elev = data.elevation[ci];
        let cell_r = 1.0 + exaggeration * cell_elev / radius;
        let center = cell_pos * cell_r;
        let normal = [cell_pos.x as f32, cell_pos.y as f32, cell_pos.z as f32];
        let color = cell_color(data, ci, mode);

        let base_idx = positions.len() as u32;

        // Center vertex.
        positions.push([center.x as f32, center.y as f32, center.z as f32]);
        normals.push(normal);
        colors.push(color);

        // Ring vertices at triangle centroids.
        let neighbors = data.grid.cell_neighbors(CellId(ci as u32));
        let nn = neighbors.len();
        let nn_u32 = nn as u32;

        for j in 0..nn {
            let pos_j = data.grid.cell_position(CellId(neighbors[j]));
            let pos_next = data.grid.cell_position(CellId(neighbors[(j + 1) % nn]));
            let centroid = (cell_pos + pos_j + pos_next).normalize();
            let avg_elev = (cell_elev
                + data.elevation[neighbors[j] as usize]
                + data.elevation[neighbors[(j + 1) % nn] as usize])
                / 3.0;
            let v_r = 1.0 + exaggeration * avg_elev / radius;
            let v = centroid * v_r;

            positions.push([v.x as f32, v.y as f32, v.z as f32]);
            normals.push([centroid.x as f32, centroid.y as f32, centroid.z as f32]);
            colors.push(color);
        }

        // Triangle fan: center → ring[j] → ring[j+1].
        for j in 0..nn_u32 {
            indices.push(base_idx);
            indices.push(base_idx + 1 + j);
            indices.push(base_idx + 1 + (j + 1) % nn_u32);
        }
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

/// Build a flat annular mesh in the XZ plane (for planetary rings).
fn build_ring_mesh(inner_r: f64, outer_r: f64, segments: u32) -> Mesh {
    let mut positions = Vec::with_capacity(((segments + 1) * 2) as usize);
    let mut normals = Vec::with_capacity(((segments + 1) * 2) as usize);
    let mut indices = Vec::with_capacity((segments * 6) as usize);

    for i in 0..=segments {
        let angle = std::f64::consts::TAU * i as f64 / segments as f64;
        let (sin, cos) = angle.sin_cos();
        let c = cos as f32;
        let s = sin as f32;

        positions.push([c * inner_r as f32, 0.0, s * inner_r as f32]);
        normals.push([0.0, 1.0, 0.0]);
        positions.push([c * outer_r as f32, 0.0, s * outer_r as f32]);
        normals.push([0.0, 1.0, 0.0]);
    }

    for i in 0..segments {
        let base = i * 2;
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
        indices.push(base + 1);
        indices.push(base + 3);
        indices.push(base + 2);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ─── Bevy app components & resources ──────────────────────────────────────────

#[derive(Resource)]
struct GlobeState {
    data: PlanetData,
    mode: ColourMode,
    exaggeration: f64,
    needs_rebuild: bool,
}

#[derive(Component)]
struct GlobeMesh;

#[derive(Component)]
struct OrbitalCamera {
    distance: f32,
    latitude: f32,
    longitude: f32,
}

#[derive(Component)]
struct MoonMarker;

#[derive(Component)]
struct RingMarker;

#[derive(Component)]
struct StarLight;

// ─── Systems ──────────────────────────────────────────────────────────────────

fn setup_globe(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<GlobeState>,
) {
    // Globe mesh.
    let globe = build_globe_mesh(&state.data, &state.mode, state.exaggeration);
    let mesh_handle = meshes.add(globe);
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });
    commands.spawn((
        GlobeMesh,
        Mesh3d(mesh_handle),
        MeshMaterial3d(material),
        Transform::IDENTITY,
    ));

    // Star directional light.
    let star_dir = state.data.celestial.star_direction_at(0.0);
    let star_col = state.data.celestial.star.color;
    let light_dir = Vec3::new(-star_dir.x as f32, -star_dir.y as f32, -star_dir.z as f32);
    commands.spawn((
        StarLight,
        DirectionalLight {
            illuminance: 50_000.0,
            color: Color::srgb(star_col[0], star_col[1], star_col[2]),
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_arc(Vec3::NEG_Z, light_dir)),
    ));

    // Ambient light so the dark hemisphere is visible.
    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        ..default()
    });

    // Moon markers.
    let moon_positions = state.data.celestial.moon_positions_at(0.0);
    let moon_mesh = meshes.add(Sphere::new(0.05));
    for (i, pos) in moon_positions.iter().enumerate() {
        let moon = &state.data.celestial.moons[i];
        let dist = (moon.semi_major_axis_m / state.data.config.radius_m).clamp(1.5, 5.0);
        let dir = pos.normalize();
        let p = dir * dist;
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(
                moon.surface_color[0],
                moon.surface_color[1],
                moon.surface_color[2],
            ),
            ..default()
        });
        commands.spawn((
            MoonMarker,
            Mesh3d(moon_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(p.x as f32, p.y as f32, p.z as f32),
        ));
    }

    // Planetary ring (if present).
    if let Some(ring) = &state.data.celestial.ring {
        let inner = ring.inner_radius_m / state.data.config.radius_m;
        let outer = ring.outer_radius_m / state.data.config.radius_m;
        let ring_mesh_handle = meshes.add(build_ring_mesh(inner, outer, 64));
        let ring_mat = materials.add(StandardMaterial {
            base_color: Color::srgba(ring.color[0], ring.color[1], ring.color[2], ring.opacity),
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            unlit: true,
            ..default()
        });
        commands.spawn((
            RingMarker,
            Mesh3d(ring_mesh_handle),
            MeshMaterial3d(ring_mat),
            Transform::IDENTITY,
        ));
    }

    // Orbital camera.
    commands.spawn((
        Camera3d::default(),
        OrbitalCamera {
            distance: 3.0,
            latitude: 0.3,
            longitude: 0.0,
        },
        Transform::from_xyz(0.0, 0.9, 2.85).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn orbital_camera(
    mouse_button: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut query: Query<(&mut OrbitalCamera, &mut Transform)>,
) {
    let Ok((mut cam, mut tf)) = query.single_mut() else {
        return;
    };

    if mouse_button.pressed(MouseButton::Left) {
        let delta = motion.delta;
        cam.longitude -= delta.x * 0.005;
        cam.latitude = (cam.latitude + delta.y * 0.005).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    if scroll.delta.y.abs() > 0.0 {
        cam.distance = (cam.distance - scroll.delta.y * 0.3).clamp(1.2, 20.0);
    }

    let x = cam.distance * cam.latitude.cos() * cam.longitude.sin();
    let y = cam.distance * cam.latitude.sin();
    let z = cam.distance * cam.latitude.cos() * cam.longitude.cos();
    *tf = Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y);
}

fn switch_colour_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<GlobeState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut globe_query: Query<&mut Mesh3d, With<GlobeMesh>>,
) {
    let new_mode = [
        (KeyCode::Digit1, ColourMode::Elevation),
        (KeyCode::Digit2, ColourMode::Biome),
        (KeyCode::Digit3, ColourMode::Plates),
        (KeyCode::Digit4, ColourMode::GeologicalAge),
        (KeyCode::Digit5, ColourMode::CrustDepth),
        (KeyCode::Digit6, ColourMode::TidalAmplitude),
        (KeyCode::Digit7, ColourMode::Rock),
        (KeyCode::Digit8, ColourMode::Temperature),
    ]
    .into_iter()
    .find(|&(key, _)| keyboard.just_pressed(key))
    .map(|(_, mode)| mode);

    if let Some(mode) = new_mode.filter(|&m| m != state.mode) {
        state.mode = mode;
        state.needs_rebuild = true;
        println!("Colour mode: {}", mode.label());
    }

    if keyboard.just_pressed(KeyCode::Equal) || keyboard.just_pressed(KeyCode::NumpadAdd) {
        state.exaggeration *= 1.5;
        state.needs_rebuild = true;
        println!("Elevation exaggeration: {:.1}×", state.exaggeration);
    }
    if keyboard.just_pressed(KeyCode::Minus) || keyboard.just_pressed(KeyCode::NumpadSubtract) {
        state.exaggeration /= 1.5;
        state.needs_rebuild = true;
        println!("Elevation exaggeration: {:.1}×", state.exaggeration);
    }

    if state.needs_rebuild {
        state.needs_rebuild = false;
        let new_mesh = build_globe_mesh(&state.data, &state.mode, state.exaggeration);
        let handle = meshes.add(new_mesh);
        if let Ok(mut mesh3d) = globe_query.single_mut() {
            mesh3d.0 = handle;
        }
    }
}

fn screenshot_system(keyboard: Res<ButtonInput<KeyCode>>, mut commands: Commands) {
    if !keyboard.just_pressed(KeyCode::F12) {
        return;
    }
    if let Err(e) = std::fs::create_dir_all("screenshots") {
        eprintln!("Failed to create screenshots/ directory: {e}");
        return;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = format!("screenshots/globe_{ts}.png");
    commands
        .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
        .observe(bevy::render::view::screenshot::save_to_disk(path.clone()));
    println!("Screenshot → {path}");
}

// ─── Entry point ──────────────────────────────────────────────────────────────

/// Launch the interactive 3D globe viewer.
///
/// Controls:
/// - **Left-drag**: rotate the globe
/// - **Scroll**: zoom in / out
/// - **1–8**: switch colour mode (elevation, biome, plates, geological age,
///   crust depth, tidal amplitude, rock, temperature)
/// - **+/−**: adjust elevation exaggeration
/// - **F12**: save screenshot to `screenshots/`
pub fn run_globe_viewer(data: PlanetData) {
    println!("Globe viewer controls:");
    println!(
        "  Left-drag: rotate | Scroll: zoom | 1-8: colour mode | +/-: exaggeration | F12: screenshot"
    );

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Worldgen Globe Viewer".into(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(GlobeState {
            data,
            mode: ColourMode::Elevation,
            exaggeration: 50.0,
            needs_rebuild: false,
        })
        .add_systems(Startup, setup_globe)
        .add_systems(
            Update,
            (orbital_camera, switch_colour_mode, screenshot_system),
        )
        .run();
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::biomes::run_biomes;
    use crate::planet::geology::run_geology;
    use crate::planet::impacts::run_impacts;
    use crate::planet::tectonics::run_tectonics;
    use crate::planet::{PlanetConfig, PlanetData};
    use bevy::mesh::VertexAttributeValues;

    fn test_planet() -> PlanetData {
        let config = PlanetConfig {
            seed: 42,
            grid_level: 2,
            ..Default::default()
        };
        let mut data = PlanetData::new(config);
        run_tectonics(&mut data, |_| {});
        run_impacts(&mut data);
        run_biomes(&mut data);
        run_geology(&mut data);
        data
    }

    #[test]
    fn globe_mesh_has_valid_geometry() {
        let data = test_planet();
        let mesh = build_globe_mesh(&data, &ColourMode::Elevation, 50.0);

        let pos = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        let norms = mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap();
        let cols = mesh.attribute(Mesh::ATTRIBUTE_COLOR).unwrap();
        let idx = mesh.indices().unwrap();

        let vert_count = match pos {
            VertexAttributeValues::Float32x3(v) => v.len(),
            _ => panic!("unexpected position format"),
        };
        assert!(vert_count > 0);

        let norm_count = match norms {
            VertexAttributeValues::Float32x3(v) => v.len(),
            _ => panic!("unexpected normal format"),
        };
        assert_eq!(vert_count, norm_count);

        let color_count = match cols {
            VertexAttributeValues::Float32x4(v) => v.len(),
            _ => panic!("unexpected color format"),
        };
        assert_eq!(vert_count, color_count);

        match idx {
            Indices::U32(v) => {
                for &i in v {
                    assert!((i as usize) < vert_count, "index {i} >= {vert_count}");
                }
                assert_eq!(v.len() % 3, 0, "index count not divisible by 3");
            }
            _ => panic!("expected U32 indices"),
        }
    }

    #[test]
    fn globe_mesh_vertex_count_matches_cells() {
        let data = test_planet();
        let mesh = build_globe_mesh(&data, &ColourMode::Biome, 50.0);
        let n = data.grid.cell_count();

        let vert_count = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(v) => v.len(),
            _ => panic!(),
        };

        let pentagons = (0..n)
            .filter(|&i| data.grid.is_pentagon(CellId(i as u32)))
            .count();
        let hexagons = n - pentagons;
        // Each cell: 1 center + N ring vertices (5 or 6).
        let expected = pentagons * 6 + hexagons * 7;
        assert_eq!(vert_count, expected);
    }

    #[test]
    fn all_colour_modes_produce_valid_rgba() {
        let data = test_planet();
        let modes = [
            ColourMode::Elevation,
            ColourMode::Biome,
            ColourMode::Plates,
            ColourMode::GeologicalAge,
            ColourMode::CrustDepth,
            ColourMode::TidalAmplitude,
            ColourMode::Rock,
            ColourMode::Temperature,
        ];

        for mode in &modes {
            for ci in 0..data.grid.cell_count() {
                let [r, g, b, a] = cell_color(&data, ci, mode);
                assert!(
                    (0.0..=1.0).contains(&r)
                        && (0.0..=1.0).contains(&g)
                        && (0.0..=1.0).contains(&b),
                    "colour out of range for {mode:?} cell {ci}: [{r}, {g}, {b}]"
                );
                assert!(
                    (a - 1.0).abs() < f32::EPSILON,
                    "alpha should be 1.0, got {a}"
                );
            }
        }
    }

    #[test]
    fn ring_mesh_geometry() {
        let mesh = build_ring_mesh(1.5, 2.5, 32);
        let vert_count = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(v) => v.len(),
            _ => panic!(),
        };
        // (32 + 1) × 2 = 66 vertices.
        assert_eq!(vert_count, 66);

        match mesh.indices().unwrap() {
            // 32 quads × 2 triangles × 3 indices = 192.
            Indices::U32(v) => assert_eq!(v.len(), 192),
            _ => panic!(),
        }
    }

    #[test]
    fn elevation_color_gradients() {
        let [_, _, b_deep, _] = elevation_color(-8000.0);
        let [_, g_low, _, _] = elevation_color(100.0);
        let [_, _, _, a] = elevation_color(3000.0);

        assert!(b_deep > 0.2, "deep ocean should have blue > 0.2");
        assert!(g_low > 0.4, "lowland should have green > 0.4");
        assert!((a - 1.0).abs() < f32::EPSILON, "alpha should be 1.0");
    }

    #[test]
    fn globe_mesh_normals_point_outward() {
        let data = test_planet();
        // Zero exaggeration: all vertices on the unit sphere.
        let mesh = build_globe_mesh(&data, &ColourMode::Elevation, 0.0);

        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(v) => v,
            _ => panic!(),
        };
        let normals_attr = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap() {
            VertexAttributeValues::Float32x3(v) => v,
            _ => panic!(),
        };

        for (pos, norm) in positions.iter().zip(normals_attr.iter()) {
            let dot = pos[0] * norm[0] + pos[1] * norm[1] + pos[2] * norm[2];
            assert!(dot > 0.9, "normal should point outward (dot={dot:.3})");
        }
    }

    #[test]
    fn globe_mesh_positions_near_unit_sphere() {
        let data = test_planet();
        let mesh = build_globe_mesh(&data, &ColourMode::Elevation, 0.0);

        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(v) => v,
            _ => panic!(),
        };

        for p in positions {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(
                (r - 1.0).abs() < 0.01,
                "vertex should be near unit sphere, got r={r}"
            );
        }
    }
}
