//! Ground accumulation systems for snow, rain moisture, and sand deposits.
//!
//! Tracks particle ground impacts from the GPU weather simulation and converts
//! accumulated depth / moisture into voxel material changes once thresholds are
//! reached.

use bevy::prelude::*;

use crate::data::MaterialRegistry;
use crate::gpu::particles::{self, GpuParticle};
use crate::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use crate::world::voxel::Voxel;

use super::particle_render::ParticleReadback;

/// Number of columns per chunk (one per (x, z) pair).
const COLUMN_COUNT: usize = CHUNK_SIZE * CHUNK_SIZE;

/// Maximum particle Y position considered a ground impact (meters).
const GROUND_IMPACT_CEILING: f32 = 2.0;

// ─── Components ────────────────────────────────────────────────────────────

/// Per-chunk surface accumulation tracking for precipitation and wind deposits.
///
/// Each vector is indexed by `z * CHUNK_SIZE + x` (one entry per column).
#[derive(Component)]
pub struct SurfaceAccumulation {
    /// Snow depth per surface column (meters).
    pub snow_depth: Vec<f32>,
    /// Surface moisture per column (kg/m²).
    pub surface_moisture: Vec<f32>,
    /// Sand deposit depth per column (meters).
    pub sand_depth: Vec<f32>,
}

impl Default for SurfaceAccumulation {
    fn default() -> Self {
        Self {
            snow_depth: vec![0.0; COLUMN_COUNT],
            surface_moisture: vec![0.0; COLUMN_COUNT],
            sand_depth: vec![0.0; COLUMN_COUNT],
        }
    }
}

// ─── Resources ─────────────────────────────────────────────────────────────

/// Configuration for precipitation accumulation rates and conversion thresholds.
#[derive(Resource)]
pub struct AccumulationConfig {
    /// Snow depth added per snowflake impact (meters).
    pub snow_per_particle: f32,
    /// Moisture added per raindrop impact (kg/m²).
    pub moisture_per_particle: f32,
    /// Snow depth at which a snow voxel is placed (meters).
    pub snow_conversion_threshold: f32,
    /// Moisture level at which a puddle voxel forms (kg/m²).
    pub puddle_threshold: f32,
    /// Sand depth at which a sand voxel is placed (meters).
    pub sand_conversion_threshold: f32,
    /// Master toggle for accumulation processing.
    pub enabled: bool,
}

impl Default for AccumulationConfig {
    fn default() -> Self {
        Self {
            snow_per_particle: 0.001,
            moisture_per_particle: 0.0335,
            snow_conversion_threshold: 0.5,
            puddle_threshold: 5.0,
            sand_conversion_threshold: 0.3,
            enabled: true,
        }
    }
}

// ─── Systems ───────────────────────────────────────────────────────────────

/// Attaches [`SurfaceAccumulation`] to any chunk that lacks one.
pub fn init_accumulation(
    mut commands: Commands,
    new_chunks: Query<Entity, (With<Chunk>, Without<SurfaceAccumulation>)>,
) {
    for entity in &new_chunks {
        commands
            .entity(entity)
            .insert(SurfaceAccumulation::default());
    }
}

/// Reads dead near-ground particles from [`ParticleReadback`] and increments
/// per-column accumulation values on the corresponding chunk.
pub fn track_ground_impacts(
    config: Res<AccumulationConfig>,
    readback: Res<ParticleReadback>,
    mut chunk_q: Query<(&ChunkCoord, &mut SurfaceAccumulation)>,
) {
    if !config.enabled {
        return;
    }

    for particle in &readback.particles {
        if !is_ground_impact(particle) {
            continue;
        }

        let wx = particle.position[0].floor() as i32;
        let wz = particle.position[2].floor() as i32;

        let chunk_coord = ChunkCoord::from_voxel_pos(wx, 0, wz);
        let origin = chunk_coord.world_origin();
        let lx = (wx - origin.x).clamp(0, CHUNK_SIZE as i32 - 1) as usize;
        let lz = (wz - origin.z).clamp(0, CHUNK_SIZE as i32 - 1) as usize;
        let col_idx = lz * CHUNK_SIZE + lx;

        for (coord, mut accum) in &mut chunk_q {
            if *coord == chunk_coord {
                match particle.kind {
                    particles::kind::RAIN => {
                        accum.surface_moisture[col_idx] += config.moisture_per_particle;
                    }
                    particles::kind::SNOW => {
                        accum.snow_depth[col_idx] += config.snow_per_particle;
                    }
                    particles::kind::SAND => {
                        accum.sand_depth[col_idx] += config.snow_per_particle;
                    }
                    _ => {}
                }
                break;
            }
        }
    }
}

/// Converts accumulated depths/moisture into voxel material changes when the
/// configured thresholds are reached.
pub fn apply_accumulation(
    config: Res<AccumulationConfig>,
    registry: Res<MaterialRegistry>,
    mut chunk_q: Query<(&mut Chunk, &mut SurfaceAccumulation)>,
) {
    if !config.enabled {
        return;
    }

    let snow_id = registry.resolve_name("Snow");
    let water_id = registry.resolve_name("Water");
    let sand_id = registry.resolve_name("Sand");

    for (mut chunk, mut accum) in &mut chunk_q {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let col = z * CHUNK_SIZE + x;

                if let Some(id) = snow_id
                    && accum.snow_depth[col] >= config.snow_conversion_threshold
                    && let Some(y) = first_air_above_surface(&chunk, x, z)
                {
                    chunk.set(x, y, z, Voxel::new(id));
                    accum.snow_depth[col] = 0.0;
                }

                if let Some(id) = water_id
                    && accum.surface_moisture[col] >= config.puddle_threshold
                    && let Some(y) = first_air_above_surface(&chunk, x, z)
                {
                    chunk.set(x, y, z, Voxel::new(id));
                    accum.surface_moisture[col] = 0.0;
                }

                if let Some(id) = sand_id
                    && accum.sand_depth[col] >= config.sand_conversion_threshold
                    && let Some(y) = first_air_above_surface(&chunk, x, z)
                {
                    chunk.set(x, y, z, Voxel::new(id));
                    accum.sand_depth[col] = 0.0;
                }
            }
        }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Returns `true` when a particle represents a ground impact: lifetime expired
/// and position is near ground level.
fn is_ground_impact(p: &GpuParticle) -> bool {
    p.life <= 0.0 && p.position[1] < GROUND_IMPACT_CEILING
}

/// Finds the Y of the first air voxel directly above the topmost solid voxel
/// in the given column.  Returns `None` when the column is entirely air or
/// entirely solid.
fn first_air_above_surface(chunk: &Chunk, x: usize, z: usize) -> Option<usize> {
    for y in (0..CHUNK_SIZE).rev() {
        if !chunk.get(x, y, z).material.is_air() {
            let place_y = y + 1;
            return if place_y < CHUNK_SIZE {
                Some(place_y)
            } else {
                None
            };
        }
    }
    None
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::MaterialId;

    /// Helper: build a dead particle of a given kind at the specified position.
    fn dead_particle(kind: u32, x: f32, y: f32, z: f32) -> GpuParticle {
        GpuParticle {
            position: [x, y, z],
            life: 0.0,
            velocity: [0.0; 3],
            kind,
            mass: 0.001,
            _pad: [0.0; 7],
        }
    }

    /// Helper: build a minimal Bevy [`App`] wired for accumulation systems.
    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.init_resource::<AccumulationConfig>();
        app.init_resource::<ParticleReadback>();
        app.init_resource::<MaterialRegistry>();
        app.add_systems(
            Update,
            (init_accumulation, track_ground_impacts, apply_accumulation).chain(),
        );
        app
    }

    #[test]
    fn default_accumulation_is_zero() {
        let accum = SurfaceAccumulation::default();
        assert_eq!(accum.snow_depth.len(), COLUMN_COUNT);
        assert_eq!(accum.surface_moisture.len(), COLUMN_COUNT);
        assert_eq!(accum.sand_depth.len(), COLUMN_COUNT);
        assert!(accum.snow_depth.iter().all(|&v| v == 0.0));
        assert!(accum.surface_moisture.iter().all(|&v| v == 0.0));
        assert!(accum.sand_depth.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn snow_accumulates_from_impact() {
        let mut app = test_app();
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        app.world_mut().spawn((
            Chunk::new_empty(coord),
            coord,
            SurfaceAccumulation::default(),
        ));

        let mut readback = ParticleReadback::default();
        readback
            .particles
            .push(dead_particle(particles::kind::SNOW, 5.0, 0.5, 3.0));
        app.insert_resource(readback);

        app.update();

        let col_idx = 3 * CHUNK_SIZE + 5;
        let accum = app
            .world_mut()
            .query::<&SurfaceAccumulation>()
            .single(app.world())
            .unwrap();
        assert!(
            accum.snow_depth[col_idx] > 0.0,
            "snow depth should increase after snow particle impact"
        );
    }

    #[test]
    fn rain_accumulates_moisture() {
        let mut app = test_app();
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        app.world_mut().spawn((
            Chunk::new_empty(coord),
            coord,
            SurfaceAccumulation::default(),
        ));

        let mut readback = ParticleReadback::default();
        readback
            .particles
            .push(dead_particle(particles::kind::RAIN, 10.0, 1.0, 7.0));
        app.insert_resource(readback);

        app.update();

        let col_idx = 7 * CHUNK_SIZE + 10;
        let accum = app
            .world_mut()
            .query::<&SurfaceAccumulation>()
            .single(app.world())
            .unwrap();
        assert!(
            accum.surface_moisture[col_idx] > 0.0,
            "moisture should increase after rain particle impact"
        );
    }

    #[test]
    fn snow_conversion_at_threshold() {
        let mut app = test_app();
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };

        // Register "Snow" material so the conversion path activates.
        let snow_id = MaterialId(200);
        let mut registry = MaterialRegistry::new();
        registry.insert(crate::data::MaterialData {
            id: snow_id.0,
            name: "Snow".into(),
            ..Default::default()
        });
        app.insert_resource(registry);

        // Build a chunk with a solid floor (stone at y=0).
        let mut chunk = Chunk::new_empty(coord);
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                chunk.set(x, 0, z, Voxel::new(MaterialId::STONE));
            }
        }

        // Pre-load accumulation just below threshold.
        let mut accum = SurfaceAccumulation::default();
        let col_idx = 4 * CHUNK_SIZE + 4;
        accum.snow_depth[col_idx] = 0.499;
        app.world_mut().spawn((chunk, coord, accum));

        // One more particle should push over the 0.5 m threshold.
        let mut readback = ParticleReadback::default();
        readback
            .particles
            .push(dead_particle(particles::kind::SNOW, 4.0, 0.5, 4.0));
        app.insert_resource(readback);

        app.update();

        // Voxel at (4, 1, 4) should now be snow.
        let chunk_ref = app
            .world_mut()
            .query::<&Chunk>()
            .single(app.world())
            .unwrap();
        assert_eq!(
            chunk_ref.get(4, 1, 4).material,
            snow_id,
            "voxel above stone floor should be converted to snow"
        );
    }

    #[test]
    fn disabled_config_skips_accumulation() {
        let mut app = test_app();
        app.insert_resource(AccumulationConfig {
            enabled: false,
            ..Default::default()
        });

        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        app.world_mut().spawn((
            Chunk::new_empty(coord),
            coord,
            SurfaceAccumulation::default(),
        ));

        let mut readback = ParticleReadback::default();
        readback
            .particles
            .push(dead_particle(particles::kind::SNOW, 5.0, 0.5, 3.0));
        app.insert_resource(readback);

        app.update();

        let accum = app
            .world_mut()
            .query::<&SurfaceAccumulation>()
            .single(app.world())
            .unwrap();
        assert!(
            accum.snow_depth.iter().all(|&v| v == 0.0),
            "nothing should accumulate when disabled"
        );
    }

    #[test]
    fn out_of_bounds_particle_ignored() {
        let mut app = test_app();
        let coord = ChunkCoord { x: 0, y: 0, z: 0 };
        app.world_mut().spawn((
            Chunk::new_empty(coord),
            coord,
            SurfaceAccumulation::default(),
        ));

        let mut readback = ParticleReadback::default();
        // Particle far above ground — should not count as impact.
        readback
            .particles
            .push(dead_particle(particles::kind::SNOW, 5.0, 100.0, 3.0));
        app.insert_resource(readback);

        app.update();

        let accum = app
            .world_mut()
            .query::<&SurfaceAccumulation>()
            .single(app.world())
            .unwrap();
        assert!(
            accum.snow_depth.iter().all(|&v| v == 0.0),
            "high-altitude dead particles should not trigger accumulation"
        );
    }
}
