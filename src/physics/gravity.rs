// Entity gravity, drag, buoyancy, and ground detection.
//
// Any entity with a `PhysicsBody` component automatically receives gravity,
// ground collision, and velocity integration against the voxel terrain.
// Entities that also carry `Mass` + `DragProfile` get the full force model
// where terminal velocity, buoyancy, and drag emerge naturally from constituent
// forces.  Entities without those components fall back to simple gravity +
// safety-cap behavior.
//
// Supports two modes:
//   **Flat**: fixed -Y gravity (legacy).
//   **Spherical**: radial gravity toward planet center with centrifugal
//     pseudo-force from planetary rotation.
//
// Runs on `FixedUpdate` for deterministic simulation.

#![allow(dead_code)]

use bevy::prelude::*;

use crate::world::chunk::Chunk;
use crate::world::chunk_manager::ChunkMap;
use crate::world::collision::{ground_height_at, ground_height_radial};
use crate::world::planet::PlanetConfig;

use super::constants;
use super::shapes::PhysicsMaterial;

/// Gravity acceleration in m/s² — re-exported from `constants` for convenience.
/// Source: NIST CODATA — exactly 9.80665 m/s².
pub const GRAVITY: f32 = constants::GRAVITY;

/// Absolute velocity safety cap (m/s). Acts as a backstop even with the
/// force model to prevent numerical blowup in degenerate cases (e.g.
/// zero drag area). 200 m/s is well above any realistic terminal velocity
/// in air (~53 m/s for a human).
pub const VELOCITY_SAFETY_CAP: f32 = 200.0;

/// Mass in kilograms. Required (along with `DragProfile`) for force-based physics.
/// Entities without `Mass` fall back to simple gravity + safety-cap behavior.
#[derive(Component, Debug)]
pub struct Mass(pub f32);

/// Drag profile for aerodynamic / hydrodynamic drag calculation.
#[derive(Component, Debug)]
pub struct DragProfile {
    /// Drag coefficient (dimensionless). ~1.0 for cube, ~0.47 for sphere, ~1.2 for upright human.
    pub coefficient: f32,
    /// Reference cross-sectional area in m². Projected area normal to velocity.
    pub area: f32,
    /// Displaced volume in m³ (used for buoyancy). ~0.07 m³ for a human (~70 liters).
    pub volume: f32,
}

impl Default for DragProfile {
    fn default() -> Self {
        Self {
            coefficient: 1.2, // upright human
            area: 0.7,        // ~0.7 m² for a human
            volume: 0.07,     // ~70 liters
        }
    }
}

// ---------------------------------------------------------------------------
// Pure force helper functions
// ---------------------------------------------------------------------------

/// Gravitational force: F_g = m × g (directed downward).
pub fn gravitational_force(mass: f32) -> f32 {
    mass * GRAVITY
}

/// Buoyancy force: F_b = ρ_medium × V_displaced × g (directed upward).
/// `medium_density` in kg/m³, `displaced_volume` in m³.
pub fn buoyancy_force(medium_density: f32, displaced_volume: f32) -> f32 {
    medium_density * displaced_volume * GRAVITY
}

/// Drag force magnitude: F_d = ½ × ρ × v² × C_d × A.
/// Always opposes the direction of motion.
pub fn drag_force(medium_density: f32, speed: f32, drag_coefficient: f32, area: f32) -> f32 {
    0.5 * medium_density * speed * speed * drag_coefficient * area
}

/// Friction force: F_f = μ × F_normal.
/// Only meaningful when the entity is grounded.
pub fn friction_force(friction_coefficient: f32, normal_force: f32) -> f32 {
    friction_coefficient * normal_force
}

/// Theoretical terminal velocity: v_t = sqrt(2 × m × g / (ρ × C_d × A)).
/// Returns `VELOCITY_SAFETY_CAP` if the denominator is zero or negative.
pub fn terminal_velocity(mass: f32, medium_density: f32, drag_coefficient: f32, area: f32) -> f32 {
    let denom = medium_density * drag_coefficient * area;
    if denom <= 0.0 {
        return VELOCITY_SAFETY_CAP;
    }
    (2.0 * mass * GRAVITY / denom).sqrt()
}

/// Marker + state for any entity affected by gravity.
#[derive(Component, Debug)]
pub struct PhysicsBody {
    /// Current velocity (full 3D; gravity affects Y component).
    pub velocity: Vec3,
    /// Whether the entity is resting on a solid surface.
    pub grounded: bool,
    /// Multiplier for gravity. 1.0 = normal, 0.0 = weightless.
    pub gravity_scale: f32,
    /// Distance from the entity's Transform origin to its feet.
    /// Ground collision places the origin at `ground_y + foot_offset`.
    pub foot_offset: f32,
}

impl Default for PhysicsBody {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            grounded: false,
            gravity_scale: 1.0,
            foot_offset: 0.0,
        }
    }
}

impl PhysicsBody {
    /// Create a physics body with a specific foot offset (e.g. half-height for centered origins).
    pub fn with_foot_offset(mut self, offset: f32) -> Self {
        self.foot_offset = offset;
        self
    }

    /// Create a weightless physics body (no gravity, still has velocity).
    pub fn weightless() -> Self {
        Self {
            gravity_scale: 0.0,
            ..Default::default()
        }
    }
}

/// Default ground friction coefficient (dimensionless).
/// Fallback for entities without a `PhysicsMaterial` component.
/// Approximate for rubber-on-concrete; per-surface values are preferred.
const DEFAULT_GROUND_FRICTION: f32 = 0.6;

/// Apply forces to all entities with a `PhysicsBody`.
///
/// In **flat** mode, gravity is fixed -Y.  In **spherical** mode, gravity
/// points toward the planet center with inverse-square falloff and centrifugal
/// correction from planetary rotation.
///
/// Entities that also have `Mass` + `DragProfile` receive the full force model
/// (gravity, buoyancy, drag, ground friction) and terminal velocity emerges
/// naturally.  Entities with only `PhysicsBody` fall back to simple gravity
/// acceleration with a hard safety cap.
///
/// Ground friction uses the entity's `PhysicsMaterial` if present, otherwise
/// falls back to `DEFAULT_GROUND_FRICTION`.
///
/// Also resolves ground collision against the voxel terrain.
#[allow(clippy::type_complexity)]
pub fn apply_forces(
    time: Res<Time>,
    chunk_map: Option<Res<ChunkMap>>,
    chunks: Query<&Chunk>,
    planet: Res<PlanetConfig>,
    mut bodies: Query<(
        &mut PhysicsBody,
        &mut Transform,
        Option<&Mass>,
        Option<&DragProfile>,
        Option<&PhysicsMaterial>,
    )>,
) {
    let dt = time.delta_secs();
    if dt == 0.0 {
        return;
    }

    // TODO: Sample the voxel at the entity position to determine medium density
    // (air vs. water vs. lava). For now, assume air at sea level everywhere.
    let medium_density = constants::AIR_DENSITY_SEA_LEVEL;

    let is_spherical = planet.is_spherical();

    for (mut body, mut transform, mass, drag_profile, phys_mat) in &mut bodies {
        if body.gravity_scale == 0.0 {
            // Apply velocity but no gravity (e.g. floating items)
            transform.translation += body.velocity * dt;
            continue;
        }

        let ground_friction = phys_mat
            .map(|m| m.friction)
            .unwrap_or(DEFAULT_GROUND_FRICTION);

        // Compute local "down" direction and gravity magnitude.
        let (local_down, gravity_mag) = if is_spherical {
            let pos = transform.translation.as_dvec3();
            let g_vec = planet.gravity_at(pos);
            let mag = g_vec.length() as f32;
            let down = if mag > 1e-10 {
                g_vec.as_vec3() / mag
            } else {
                Vec3::NEG_Y
            };
            (down, mag)
        } else {
            (Vec3::NEG_Y, GRAVITY)
        };

        let local_up = -local_down;

        match (mass, drag_profile) {
            // Full force model
            (Some(m), Some(dp)) => {
                let mass_kg = m.0;

                // Decompose velocity into "along gravity" and "lateral" components.
                let v_down_mag = body.velocity.dot(local_down);
                let v_lateral = body.velocity - local_down * v_down_mag;

                // --- Forces along gravity axis ---
                let f_gravity = mass_kg * gravity_mag * body.gravity_scale;
                let f_buoyancy = buoyancy_force(medium_density, dp.volume);

                let speed_down = v_down_mag.abs();
                let drag_down = drag_force(medium_density, speed_down, dp.coefficient, dp.area);
                // Drag opposes velocity direction along gravity axis
                let f_drag_down = if v_down_mag > 0.0 {
                    -drag_down // Moving downward → drag pushes upward (negative in down-axis)
                } else {
                    drag_down // Moving upward → drag pushes downward
                };

                // Net force along gravity: gravity pulls down, buoyancy pushes up, drag opposes
                // Convention: positive = in local_down direction
                let net_f_down = f_gravity - f_buoyancy + f_drag_down;
                let accel_down = net_f_down / mass_kg;
                // Apply acceleration: positive accel_down → velocity in local_down direction
                body.velocity += local_down * accel_down * dt;

                // --- Lateral forces (drag + optional ground friction) ---
                let speed_lat = v_lateral.length();
                if speed_lat > 1e-6 {
                    let dir_lat = v_lateral / speed_lat;
                    let mut decel_lat =
                        drag_force(medium_density, speed_lat, dp.coefficient, dp.area);

                    if body.grounded {
                        let normal_force = f_gravity - f_buoyancy;
                        if normal_force > 0.0 {
                            decel_lat += friction_force(ground_friction, normal_force);
                        }
                    }

                    let decel_mag = (decel_lat / mass_kg) * dt;
                    if decel_mag >= speed_lat {
                        // Remove lateral velocity entirely
                        body.velocity -= v_lateral;
                    } else {
                        body.velocity -= dir_lat * decel_mag;
                    }
                }
            }
            // Fallback: simple gravity + safety cap (no Mass/DragProfile)
            _ => {
                let gs = body.gravity_scale;
                body.velocity += local_down * gravity_mag * gs * dt;
            }
        }

        // Absolute backstop — prevents numerical blowup in degenerate cases.
        let speed = body.velocity.length();
        if speed > VELOCITY_SAFETY_CAP {
            body.velocity *= VELOCITY_SAFETY_CAP / speed;
        }

        // Integrate velocity
        transform.translation += body.velocity * dt;

        // Ground collision (requires V1 chunk data)
        if let Some(ref chunk_map) = chunk_map {
            if is_spherical {
                if let Some(ground_r) =
                    ground_height_radial(transform.translation, chunk_map, &chunks)
                {
                    let entity_r = transform.translation.length();
                    let feet_r = entity_r - body.foot_offset;
                    if feet_r <= ground_r {
                        // Push entity outward to stand on surface
                        let surface_normal = local_up;
                        let correction = ground_r + body.foot_offset - entity_r;
                        transform.translation += surface_normal * correction;

                        // Zero the velocity component toward the surface
                        let v_toward_ground = body.velocity.dot(local_down);
                        if v_toward_ground > 0.0 {
                            body.velocity -= local_down * v_toward_ground;
                        }
                        body.grounded = true;
                    } else {
                        body.grounded = false;
                    }
                }
            } else if let Some(ground_y) = ground_height_at(
                transform.translation.x,
                transform.translation.z,
                chunk_map,
                &chunks,
            ) {
                let feet_y = transform.translation.y - body.foot_offset;
                if feet_y <= ground_y {
                    transform.translation.y = ground_y + body.foot_offset;
                    if body.velocity.y < 0.0 {
                        body.velocity.y = 0.0;
                    }
                    body.grounded = true;
                } else {
                    body.grounded = false;
                }
            }
        }
    }
}

/// System set for physics ordering.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GravitySet;

pub struct GravityPlugin;

impl Plugin for GravityPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(FixedUpdate, apply_forces.in_set(GravitySet));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::constants;

    // --- Component defaults ---

    #[test]
    fn default_physics_body() {
        let body = PhysicsBody::default();
        assert_eq!(body.velocity, Vec3::ZERO);
        assert!(!body.grounded);
        assert_eq!(body.gravity_scale, 1.0);
        assert_eq!(body.foot_offset, 0.0);
    }

    #[test]
    fn with_foot_offset_builder() {
        let body = PhysicsBody::default().with_foot_offset(0.9);
        assert_eq!(body.foot_offset, 0.9);
        assert_eq!(body.gravity_scale, 1.0);
    }

    #[test]
    fn weightless_body_has_zero_gravity() {
        let body = PhysicsBody::weightless();
        assert_eq!(body.gravity_scale, 0.0);
        assert_eq!(body.velocity, Vec3::ZERO);
    }

    #[test]
    fn default_drag_profile() {
        let dp = DragProfile::default();
        assert_eq!(dp.coefficient, 1.2);
        assert_eq!(dp.area, 0.7);
        assert_eq!(dp.volume, 0.07);
    }

    // --- Constant sanity checks ---

    #[test]
    fn gravity_matches_real_world() {
        assert!(
            (GRAVITY - 9.806_65).abs() < 1e-4,
            "GRAVITY should match NIST standard: got {GRAVITY}"
        );
    }

    // --- Pure force functions ---

    #[test]
    fn gravitational_force_80kg() {
        // F = m*g = 80 * 9.80665 ≈ 784.532 N
        let f = gravitational_force(80.0);
        assert!((f - 784.532).abs() < 0.1, "Got {f} N");
    }

    #[test]
    fn buoyancy_force_water() {
        // 1 m³ in water: F_b = 1000 * 1.0 * 9.80665 ≈ 9806.65 N
        let f = buoyancy_force(1000.0, 1.0);
        assert!((f - 9806.65).abs() < 0.1, "Got {f} N");
    }

    #[test]
    fn drag_force_basic() {
        // F_d = 0.5 * 1.225 * 10² * 1.2 * 0.7 = 0.5 * 1.225 * 100 * 0.84 = 51.45 N
        let f = drag_force(1.225, 10.0, 1.2, 0.7);
        assert!((f - 51.45).abs() < 0.1, "Got {f} N");
    }

    #[test]
    fn drag_force_zero_speed_is_zero() {
        let f = drag_force(1.225, 0.0, 1.2, 0.7);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn friction_force_basic() {
        // F_f = μ * F_n = 0.6 * 100 = 60 N
        let f = friction_force(0.6, 100.0);
        assert!((f - 60.0).abs() < 1e-4, "Got {f} N, expected 60.0 N");
    }

    // --- Terminal velocity ---

    #[test]
    fn terminal_velocity_human() {
        // v_t = sqrt(2 * 80 * 9.80665 / (1.225 * 1.2 * 0.7)) ≈ 39.05 m/s
        // Note: the classic "53 m/s" skydiver figure uses lower C_d (~0.7).
        // With our upright-human C_d=1.2 and A=0.7 m², the math gives ~39 m/s.
        let vt = terminal_velocity(80.0, 1.225, 1.2, 0.7);
        let expected = (2.0 * 80.0 * GRAVITY / (1.225 * 1.2 * 0.7_f32)).sqrt();
        assert!(
            (vt - expected).abs() < 0.1,
            "Human terminal velocity: got {vt} m/s, expected ~{expected} m/s"
        );
        // Sanity: should be in a reasonable range for a human
        assert!(
            vt > 30.0 && vt < 60.0,
            "Terminal velocity {vt} m/s out of plausible range"
        );
    }

    #[test]
    fn terminal_velocity_zero_area_returns_safety_cap() {
        let vt = terminal_velocity(80.0, 1.225, 1.2, 0.0);
        assert_eq!(vt, VELOCITY_SAFETY_CAP);
    }

    #[test]
    fn terminal_velocity_zero_density_returns_safety_cap() {
        let vt = terminal_velocity(80.0, 0.0, 1.2, 0.7);
        assert_eq!(vt, VELOCITY_SAFETY_CAP);
    }

    // --- Buoyancy: floats vs. sinks ---

    #[test]
    fn wood_floats_in_water() {
        // Wood ~600 kg/m³, 1 m³ displaced in water (1000 kg/m³)
        let f_buoyancy = buoyancy_force(1000.0, 1.0);
        let f_gravity = gravitational_force(600.0);
        assert!(
            f_buoyancy > f_gravity,
            "Wood should float: buoyancy {f_buoyancy} N > gravity {f_gravity} N"
        );
    }

    #[test]
    fn iron_sinks_in_water() {
        // Iron ~7874 kg/m³, 1 m³ displaced in water (1000 kg/m³)
        let f_buoyancy = buoyancy_force(1000.0, 1.0);
        let f_gravity = gravitational_force(7874.0);
        assert!(
            f_buoyancy < f_gravity,
            "Iron should sink: buoyancy {f_buoyancy} N < gravity {f_gravity} N"
        );
    }

    // --- Drag opposes velocity ---

    #[test]
    fn drag_always_decelerates() {
        let mass = 80.0_f32;
        let cd = 1.2_f32;
        let area = 0.7_f32;
        let rho = constants::AIR_DENSITY_SEA_LEVEL;

        // Falling at 30 m/s
        let speed = 30.0_f32;
        let f_drag = drag_force(rho, speed, cd, area);
        let decel = f_drag / mass; // m/s²
        assert!(decel > 0.0, "Drag should produce positive deceleration");

        // Faster speed → more drag
        let f_drag_fast = drag_force(rho, 60.0, cd, area);
        assert!(f_drag_fast > f_drag, "Drag should increase with speed");
    }

    // --- Integration tests (simple kinematics, no ECS) ---

    #[test]
    fn velocity_integration() {
        let mut body = PhysicsBody::default();
        let dt = 1.0 / 60.0;

        body.velocity.y -= GRAVITY * body.gravity_scale * dt;
        assert!(body.velocity.y < 0.0, "Should be falling");
    }

    #[test]
    fn freefall_velocity_after_2_seconds() {
        let g = constants::GRAVITY;
        let t = 2.0_f32;
        let mut vy = 0.0_f32;
        let dt = 1.0 / 60.0;
        let steps = (t / dt) as u32;

        for _ in 0..steps {
            vy -= g * dt;
        }

        let expected = -g * t;
        let tolerance = 0.5;
        assert!(
            (vy - expected).abs() < tolerance,
            "After {t}s freefall: got {vy} m/s, expected ~{expected} m/s"
        );
    }

    #[test]
    fn freefall_distance_after_3_seconds() {
        let g = constants::GRAVITY;
        let t = 3.0_f32;
        let dt = 1.0 / 60.0;
        let steps = (t / dt) as u32;
        let mut vy = 0.0_f32;
        let mut pos_y = 100.0_f32;

        for _ in 0..steps {
            vy -= g * dt;
            pos_y += vy * dt;
        }

        let distance_fallen = 100.0 - pos_y;
        let expected = 0.5 * g * t * t;
        let tolerance = 1.0;
        assert!(
            (distance_fallen - expected).abs() < tolerance,
            "After {t}s freefall: fell {distance_fallen} m, expected ~{expected} m"
        );
    }

    #[test]
    fn jump_velocity_for_desired_height() {
        let g = constants::GRAVITY;
        let desired_height = 1.25_f32;
        let v0 = (2.0 * g * desired_height).sqrt();

        assert!(
            (v0 - 4.952).abs() < 0.01,
            "Jump velocity for {desired_height}m: got {v0}, expected ~4.952 m/s"
        );

        let dt = 1.0 / 240.0;
        let mut vy = v0;
        let mut max_y = 0.0_f32;
        let mut y = 0.0_f32;

        for _ in 0..1000 {
            vy -= g * dt;
            y += vy * dt;
            max_y = max_y.max(y);
            if vy < 0.0 && y < 0.0 {
                break;
            }
        }

        assert!(
            (max_y - desired_height).abs() < 0.05,
            "Simulated peak: {max_y} m, expected ~{desired_height} m"
        );
    }

    #[test]
    fn grounded_stops_downward_velocity() {
        let mut body = PhysicsBody::default();
        body.velocity.y = -10.0;
        body.grounded = false;

        if body.velocity.y < 0.0 {
            body.velocity.y = 0.0;
        }
        body.grounded = true;

        assert_eq!(body.velocity.y, 0.0);
        assert!(body.grounded);
    }

    #[test]
    fn horizontal_velocity_preserved_on_ground() {
        let mut body = PhysicsBody {
            velocity: Vec3::new(5.0, -10.0, 3.0),
            ..Default::default()
        };

        if body.velocity.y < 0.0 {
            body.velocity.y = 0.0;
        }
        body.grounded = true;

        assert_eq!(body.velocity.x, 5.0);
        assert_eq!(body.velocity.z, 3.0);
        assert_eq!(body.velocity.y, 0.0);
    }

    // --- Force model integration (pure math, no ECS) ---

    #[test]
    fn force_model_reaches_terminal_velocity() {
        // Simulate 80 kg human falling in air with full force model.
        // Should converge near theoretical terminal velocity (~53 m/s).
        let mass = 80.0_f32;
        let cd = 1.2_f32;
        let area = 0.7_f32;
        let volume = 0.07_f32;
        let rho = constants::AIR_DENSITY_SEA_LEVEL;
        let dt = 1.0 / 60.0;

        let vt_theory = terminal_velocity(mass, rho, cd, area);

        let mut vy = 0.0_f32;
        // Simulate 30 seconds of freefall (plenty to converge)
        let steps = (30.0 / dt) as u32;
        for _ in 0..steps {
            let f_grav = gravitational_force(mass);
            let f_buoy = buoyancy_force(rho, volume);
            let speed = vy.abs();
            let f_drag = drag_force(rho, speed, cd, area);
            let drag_sign = if vy > 0.0 { -f_drag } else { f_drag };
            let net = -f_grav + f_buoy + drag_sign;
            vy += (net / mass) * dt;
        }

        let final_speed = vy.abs();
        assert!(
            (final_speed - vt_theory).abs() < 1.0,
            "Should converge to terminal velocity: got {final_speed} m/s, expected ~{vt_theory} m/s"
        );
    }

    #[test]
    fn force_model_buoyancy_slows_sinking() {
        // An entity in a dense medium should sink slower than in vacuum.
        let mass = 80.0_f32;
        let rho = constants::AIR_DENSITY_SEA_LEVEL;
        let volume = 0.07_f32;

        // One step with buoyancy
        let f_grav = gravitational_force(mass);
        let f_buoy = buoyancy_force(rho, volume);
        let net_with_buoyancy = -f_grav + f_buoy;
        let accel_with = net_with_buoyancy / mass;

        // One step without buoyancy (vacuum)
        let accel_without = -constants::GRAVITY;

        // With buoyancy, downward acceleration magnitude should be smaller
        assert!(
            accel_with.abs() < accel_without.abs(),
            "Buoyancy should reduce downward acceleration: {accel_with} vs {accel_without}"
        );
    }

    // --- Real-world validation tests ---

    #[test]
    fn neutral_buoyancy_near_zero_net_force() {
        // Entity with density ≈ water density should have near-zero net vertical force
        // Wikipedia: Archimedes' principle - neutral buoyancy when ρ_object ≈ ρ_fluid
        let mass = 1000.0; // 1 m³ of water-density material
        let volume = 1.0;
        let fluid_density = 1000.0; // water
        let f_grav = gravitational_force(mass);
        let f_buoy = buoyancy_force(fluid_density, volume);
        let net = f_grav - f_buoy;
        assert!(
            net.abs() < 0.01,
            "Neutral buoyancy: net force should be ~0, got {net}"
        );
    }

    #[test]
    fn dense_object_higher_terminal_velocity() {
        // Wikipedia: Terminal velocity increases with mass
        // Iron sphere vs wood sphere, same size (1m³), same drag
        let cd = 0.47; // sphere
        let area = std::f32::consts::PI * 0.5 * 0.5; // r=0.5m sphere cross-section
        let rho_air = 1.225;

        let iron_mass = 7874.0; // kg/m³ × 1m³
        let wood_mass = 600.0;

        let vt_iron = terminal_velocity(iron_mass, rho_air, cd, area);
        let vt_wood = terminal_velocity(wood_mass, rho_air, cd, area);

        assert!(
            vt_iron > vt_wood,
            "Iron ({vt_iron:.1}) should have higher terminal velocity than wood ({vt_wood:.1})"
        );
        // Iron ~137 m/s, Wood ~38 m/s (both in air)
    }

    #[test]
    fn terminal_velocity_lower_in_denser_medium() {
        // Wikipedia: drag increases with medium density
        let mass = 80.0;
        let cd = 1.2;
        let area = 0.7;

        let vt_air = terminal_velocity(mass, 1.225, cd, area);
        let vt_water = terminal_velocity(mass, 1000.0, cd, area);

        assert!(
            vt_air > vt_water * 5.0,
            "Terminal velocity in air ({vt_air:.1}) should be much higher than in water ({vt_water:.1})"
        );
    }

    #[test]
    fn ice_much_less_friction_than_stone() {
        // Wikipedia: Coefficient of friction - ice μ≈0.05, stone μ≈0.7
        let mass = 80.0;
        let normal = mass * GRAVITY; // standing on surface

        let f_ice = friction_force(0.05, normal);
        let f_stone = friction_force(0.7, normal);

        // Stone friction should be 14× ice friction (0.7/0.05 = 14)
        let ratio = f_stone / f_ice;
        assert!(
            (ratio - 14.0).abs() < 0.1,
            "Stone/ice friction ratio should be 14, got {ratio:.1}"
        );
    }

    #[test]
    fn friction_force_80kg_on_stone() {
        // F_f = μ × m × g = 0.7 × 80 × 9.80665 ≈ 549.2 N
        // Wikipedia: friction force equation
        let mass = 80.0;
        let mu = 0.7; // stone
        let normal = mass * GRAVITY;
        let f = friction_force(mu, normal);
        let expected = 0.7 * 80.0 * 9.80665;
        assert!(
            (f - expected).abs() < 0.1,
            "Friction on stone: expected {expected:.1} N, got {f:.1} N"
        );
    }

    #[test]
    fn buoyancy_force_1m3_in_water_is_9810n() {
        // F_b = ρ × V × g = 1000 × 1.0 × 9.80665 = 9806.65 N
        // Wikipedia: Archimedes' principle
        let f = buoyancy_force(1000.0, 1.0);
        assert!(
            (f - 9806.65).abs() < 1.0,
            "1m³ buoyancy in water: expected ~9807 N, got {f:.1}"
        );
    }
}
