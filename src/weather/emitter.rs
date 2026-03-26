use bevy::prelude::*;

use crate::gpu::particles::{self, GpuParticle};
use crate::physics::lbm_gas::plugin::CloudField;
use crate::world::chunk::CHUNK_SIZE;

// ---------------------------------------------------------------------------
// Physical constants (ISA atmosphere)
// ---------------------------------------------------------------------------

/// ISA standard sea-level temperature (K).
const ISA_SEA_LEVEL_TEMP: f32 = 288.15;

/// ISA tropospheric lapse rate (K/m).
const ISA_LAPSE_RATE: f32 = 0.0065;

// ---------------------------------------------------------------------------
// Emitter tuning defaults
// ---------------------------------------------------------------------------

/// Default max GPU particles emitted per fixed-update tick.
const DEFAULT_MAX_EMIT: usize = 500;

/// Default cloud liquid-water-content threshold for precipitation (kg/m³).
/// Real-world autoconversion starts around 0.3 g/m³.
const DEFAULT_CLOUD_THRESHOLD: f32 = 0.3e-3;

/// Freezing point of water (K) — rain/snow boundary.
const FREEZING_POINT: f32 = 273.15;

// ---------------------------------------------------------------------------
// Per-kind particle parameters (SI)
// ---------------------------------------------------------------------------

/// Rain drop lifetime (s).
const RAIN_LIFE: f32 = 30.0;
/// Snowflake lifetime (s) — longer because it falls slower.
const SNOW_LIFE: f32 = 60.0;

/// Rain initial downward velocity (m/s).
const RAIN_INITIAL_VELOCITY_Y: f32 = -2.0;
/// Snow initial downward velocity (m/s).
const SNOW_INITIAL_VELOCITY_Y: f32 = -0.5;

/// Rain drop mass (kg) — typical drizzle drop ≈ 33.5 µg.
const RAIN_DROP_MASS: f32 = 3.35e-5;
/// Snowflake mass (kg) — typical ≈ 3 µg.
const SNOWFLAKE_MASS: f32 = 3.0e-6;

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

/// Bridges LBM cloud fields to the GPU particle system by buffering pending
/// precipitation particles each tick.
#[derive(Resource)]
pub struct PrecipitationEmitter {
    /// Buffer of particles waiting to be uploaded to the GPU.
    pending: Vec<GpuParticle>,
    /// Maximum particles emitted per `FixedUpdate` tick.
    pub max_emit_per_tick: usize,
    /// Cloud LWC threshold for precipitation (kg/m³).
    pub cloud_threshold: f32,
    /// Temperature below which precipitation is snow instead of rain (K).
    pub snow_temperature: f32,
}

impl Default for PrecipitationEmitter {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            max_emit_per_tick: DEFAULT_MAX_EMIT,
            cloud_threshold: DEFAULT_CLOUD_THRESHOLD,
            snow_temperature: FREEZING_POINT,
        }
    }
}

impl PrecipitationEmitter {
    /// Take all pending particles, leaving the buffer empty.
    pub fn drain_pending(&mut self) -> Vec<GpuParticle> {
        std::mem::take(&mut self.pending)
    }

    /// Number of particles currently waiting in the buffer.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

// ---------------------------------------------------------------------------
// Deterministic spatial hash for sub-voxel jitter
// ---------------------------------------------------------------------------

fn simple_hash(x: u32, y: u32, z: u32, seed: u32) -> f32 {
    let h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(z.wrapping_mul(1_274_126_177))
        .wrapping_add(seed);
    let h = (h ^ (h >> 13)).wrapping_mul(1_103_515_245);
    (h & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32
}

// ---------------------------------------------------------------------------
// Core logic (pure function, easy to test)
// ---------------------------------------------------------------------------

fn emit_from_cloud_field(cloud_field: &CloudField, emitter: &mut PrecipitationEmitter) {
    let size = CHUNK_SIZE;

    for (coord, lwc_data) in &cloud_field.chunks {
        let origin = coord.world_origin();

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    if emitter.pending.len() >= emitter.max_emit_per_tick {
                        return;
                    }

                    let idx = z * size * size + y * size + x;
                    if idx >= lwc_data.len() {
                        continue;
                    }

                    let lwc = lwc_data[idx];
                    if lwc <= emitter.cloud_threshold {
                        continue;
                    }

                    // ISA lapse-rate temperature estimate at this altitude.
                    let altitude_m = origin.y as f32 + y as f32;
                    let temp_k = ISA_SEA_LEVEL_TEMP - ISA_LAPSE_RATE * altitude_m;
                    let is_snow = temp_k < emitter.snow_temperature;

                    // Sub-voxel jitter (0..1 range per axis).
                    let jx = simple_hash(x as u32, y as u32, z as u32, 0);
                    let jy = simple_hash(x as u32, y as u32, z as u32, 1);
                    let jz = simple_hash(x as u32, y as u32, z as u32, 2);

                    let wx = origin.x as f32 + x as f32 + jx;
                    let wy = origin.y as f32 + y as f32 + jy;
                    let wz = origin.z as f32 + z as f32 + jz;

                    let (life, vy, kind, mass) = if is_snow {
                        (
                            SNOW_LIFE,
                            SNOW_INITIAL_VELOCITY_Y,
                            particles::kind::SNOW,
                            SNOWFLAKE_MASS,
                        )
                    } else {
                        (
                            RAIN_LIFE,
                            RAIN_INITIAL_VELOCITY_Y,
                            particles::kind::RAIN,
                            RAIN_DROP_MASS,
                        )
                    };

                    emitter.pending.push(GpuParticle {
                        position: [wx, wy, wz],
                        life,
                        velocity: [0.0, vy, 0.0],
                        kind,
                        mass,
                        _pad: [0.0; 7],
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bevy system
// ---------------------------------------------------------------------------

/// Scans `CloudField` for cells exceeding the LWC threshold and queues GPU
/// precipitation particles (rain or snow) into `PrecipitationEmitter`.
pub fn scan_clouds_for_precipitation(
    cloud_field: Res<CloudField>,
    mut emitter: ResMut<PrecipitationEmitter>,
) {
    emit_from_cloud_field(&cloud_field, &mut emitter);
}

// ---------------------------------------------------------------------------
// Plugin registration helper
// ---------------------------------------------------------------------------

pub fn register(app: &mut App) {
    app.init_resource::<PrecipitationEmitter>()
        .add_systems(FixedUpdate, scan_clouds_for_precipitation);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::ChunkCoord;
    use std::collections::HashMap;

    /// Helper: build a `CloudField` with one chunk of LWC data.
    fn cloud_field_one_chunk(coord: ChunkCoord, lwc_data: Vec<f32>) -> CloudField {
        let mut chunks = HashMap::new();
        chunks.insert(coord, lwc_data);
        CloudField { chunks }
    }

    // -- 1. default_thresholds -------------------------------------------------

    #[test]
    fn default_thresholds() {
        let e = PrecipitationEmitter::default();
        assert_eq!(e.max_emit_per_tick, DEFAULT_MAX_EMIT);
        assert!((e.cloud_threshold - DEFAULT_CLOUD_THRESHOLD).abs() < f32::EPSILON);
        assert!((e.snow_temperature - FREEZING_POINT).abs() < f32::EPSILON);
        assert!(e.pending.is_empty());
    }

    // -- 2. rain_emitted_above_threshold ---------------------------------------

    #[test]
    fn rain_emitted_above_threshold() {
        // Chunk at y=0 → altitude ≈ 0 m → temp ≈ 288 K (warm → rain).
        let coord = ChunkCoord::new(0, 0, 0);
        let size = CHUNK_SIZE;
        let mut lwc = vec![0.0; size * size * size];
        lwc[0] = 1.0e-3; // well above threshold

        let cf = cloud_field_one_chunk(coord, lwc);
        let mut emitter = PrecipitationEmitter::default();
        emit_from_cloud_field(&cf, &mut emitter);

        assert_eq!(emitter.pending.len(), 1);
        assert_eq!(emitter.pending[0].kind, particles::kind::RAIN);
        assert!((emitter.pending[0].life - RAIN_LIFE).abs() < f32::EPSILON);
        assert!((emitter.pending[0].mass - RAIN_DROP_MASS).abs() < f32::EPSILON);
    }

    // -- 3. snow_emitted_below_freezing ----------------------------------------

    #[test]
    fn snow_emitted_below_freezing() {
        // Place chunk high enough that ISA lapse rate drops below 273.15 K.
        // altitude needed: (288.15 - 273.15) / 0.0065 ≈ 2308 m → chunk y ≈ 73
        // Use y = 100 (altitude 3200 m → temp ≈ 267.35 K → snow).
        let coord = ChunkCoord::new(0, 100, 0);
        let size = CHUNK_SIZE;
        let mut lwc = vec![0.0; size * size * size];
        lwc[0] = 1.0e-3;

        let cf = cloud_field_one_chunk(coord, lwc);
        let mut emitter = PrecipitationEmitter::default();
        emit_from_cloud_field(&cf, &mut emitter);

        assert_eq!(emitter.pending.len(), 1);
        assert_eq!(emitter.pending[0].kind, particles::kind::SNOW);
        assert!((emitter.pending[0].life - SNOW_LIFE).abs() < f32::EPSILON);
        assert!((emitter.pending[0].mass - SNOWFLAKE_MASS).abs() < f32::EPSILON);
    }

    // -- 4. emission_capped_at_max ---------------------------------------------

    #[test]
    fn emission_capped_at_max() {
        let coord = ChunkCoord::new(0, 0, 0);
        let size = CHUNK_SIZE;
        // Fill every cell above threshold — 32³ = 32768 cells, far more than
        // max_emit_per_tick (500).
        let lwc = vec![1.0e-3; size * size * size];

        let cf = cloud_field_one_chunk(coord, lwc);
        let mut emitter = PrecipitationEmitter::default();
        emit_from_cloud_field(&cf, &mut emitter);

        assert_eq!(emitter.pending.len(), emitter.max_emit_per_tick);
    }

    // -- 5. drain_clears_pending -----------------------------------------------

    #[test]
    fn drain_clears_pending() {
        let coord = ChunkCoord::new(0, 0, 0);
        let size = CHUNK_SIZE;
        let mut lwc = vec![0.0; size * size * size];
        lwc[0] = 1.0e-3;

        let cf = cloud_field_one_chunk(coord, lwc);
        let mut emitter = PrecipitationEmitter::default();
        emit_from_cloud_field(&cf, &mut emitter);
        assert!(!emitter.pending.is_empty());

        let drained = emitter.drain_pending();
        assert!(!drained.is_empty());
        assert!(emitter.pending.is_empty());
        assert_eq!(emitter.pending_count(), 0);
    }
}
