// Narrow-phase collision detection between entity collision shapes.
//
// Consumes `BroadPhasePairs` and tests each candidate pair using the exact
// geometry of their `CollisionShape` components. Produces a `Contacts`
// resource containing `ContactManifold` entries for overlapping pairs.
//
// Supported shape pair tests:
//   - Sphere vs Sphere
//   - Sphere vs AABB
//   - AABB vs AABB
//   - Capsule vs Sphere
//   - Capsule vs AABB
//   - Capsule vs Capsule

use bevy::prelude::*;

use super::broad_phase::BroadPhasePairs;
use super::shapes::{CollisionShape, CombinedMaterial, PhysicsMaterial};

// ---------------------------------------------------------------------------
// Contact data
// ---------------------------------------------------------------------------

/// A single contact point between two entities.
#[derive(Debug, Clone, Copy)]
pub struct ContactPoint {
    /// World-space contact position.
    pub point: Vec3,
    /// Contact normal pointing from entity B toward entity A.
    pub normal: Vec3,
    /// Penetration depth (positive when overlapping).
    pub depth: f32,
}

/// All contacts between a pair of entities for one tick.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    pub entity_a: Entity,
    pub entity_b: Entity,
    pub contacts: Vec<ContactPoint>,
    pub material: CombinedMaterial,
}

/// Resource holding all contact manifolds for the current tick.
#[derive(Resource, Default, Debug)]
pub struct Contacts {
    pub manifolds: Vec<ContactManifold>,
}

// ---------------------------------------------------------------------------
// Overlap tests (pure geometry — no ECS)
// ---------------------------------------------------------------------------

/// Closest point on a line segment (a, b) to point p.
fn closest_point_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// Closest point on an AABB (defined by half-extents centered at origin) to a point.
fn closest_point_on_aabb(half: Vec3, point: Vec3) -> Vec3 {
    Vec3::new(
        point.x.clamp(-half.x, half.x),
        point.y.clamp(-half.y, half.y),
        point.z.clamp(-half.z, half.z),
    )
}

/// Sphere vs Sphere overlap test.
pub fn sphere_vs_sphere(
    pos_a: Vec3,
    radius_a: f32,
    pos_b: Vec3,
    radius_b: f32,
) -> Option<ContactPoint> {
    let diff = pos_a - pos_b;
    let dist_sq = diff.length_squared();
    let sum_r = radius_a + radius_b;

    if dist_sq >= sum_r * sum_r {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 { diff / dist } else { Vec3::Y };
    let depth = sum_r - dist;
    let point = pos_b + normal * (radius_b - depth * 0.5);

    Some(ContactPoint {
        point,
        normal,
        depth,
    })
}

/// Sphere vs AABB overlap test.
///
/// `aabb_pos` is the AABB center, `half` is its half-extents.
pub fn sphere_vs_aabb(
    sphere_pos: Vec3,
    radius: f32,
    aabb_pos: Vec3,
    half: Vec3,
) -> Option<ContactPoint> {
    // Work in AABB-local space
    let local = sphere_pos - aabb_pos;
    let closest = closest_point_on_aabb(half, local);
    let diff = local - closest;
    let dist_sq = diff.length_squared();

    if dist_sq >= radius * radius {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 {
        diff / dist
    } else {
        // Sphere center inside AABB — push out along smallest overlap axis
        let overlap = half - local.abs();
        if overlap.x <= overlap.y && overlap.x <= overlap.z {
            Vec3::new(local.x.signum(), 0.0, 0.0)
        } else if overlap.y <= overlap.z {
            Vec3::new(0.0, local.y.signum(), 0.0)
        } else {
            Vec3::new(0.0, 0.0, local.z.signum())
        }
    };

    let depth = radius - dist;
    let point = aabb_pos + closest;

    Some(ContactPoint {
        point,
        normal,
        depth,
    })
}

/// AABB vs AABB overlap test (Separating Axis Theorem).
pub fn aabb_vs_aabb(pos_a: Vec3, half_a: Vec3, pos_b: Vec3, half_b: Vec3) -> Option<ContactPoint> {
    let diff = pos_a - pos_b;
    let overlap = (half_a + half_b) - diff.abs();

    if overlap.x <= 0.0 || overlap.y <= 0.0 || overlap.z <= 0.0 {
        return None;
    }

    // Push out along axis of minimum penetration
    let (normal, depth) = if overlap.x <= overlap.y && overlap.x <= overlap.z {
        (Vec3::new(diff.x.signum(), 0.0, 0.0), overlap.x)
    } else if overlap.y <= overlap.z {
        (Vec3::new(0.0, diff.y.signum(), 0.0), overlap.y)
    } else {
        (Vec3::new(0.0, 0.0, diff.z.signum()), overlap.z)
    };

    let point = (pos_a + pos_b) * 0.5;

    Some(ContactPoint {
        point,
        normal,
        depth,
    })
}

/// Capsule vs Sphere overlap test.
///
/// The capsule axis runs along Y, centered at `cap_pos`.
pub fn capsule_vs_sphere(
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    sphere_pos: Vec3,
    sphere_radius: f32,
) -> Option<ContactPoint> {
    // Capsule segment endpoints
    let seg_a = cap_pos + Vec3::new(0.0, cap_half_height, 0.0);
    let seg_b = cap_pos - Vec3::new(0.0, cap_half_height, 0.0);
    let closest = closest_point_on_segment(seg_a, seg_b, sphere_pos);

    sphere_vs_sphere(closest, cap_radius, sphere_pos, sphere_radius)
}

/// Capsule vs AABB overlap test.
///
/// Approximates by finding the closest point on the capsule axis to the AABB,
/// then performing a sphere-vs-AABB test at that point.
pub fn capsule_vs_aabb(
    cap_pos: Vec3,
    cap_radius: f32,
    cap_half_height: f32,
    aabb_pos: Vec3,
    half: Vec3,
) -> Option<ContactPoint> {
    let seg_a = cap_pos + Vec3::new(0.0, cap_half_height, 0.0);
    let seg_b = cap_pos - Vec3::new(0.0, cap_half_height, 0.0);

    // Find the point on the capsule axis closest to the AABB center
    let closest_on_axis = closest_point_on_segment(seg_a, seg_b, aabb_pos);

    sphere_vs_aabb(closest_on_axis, cap_radius, aabb_pos, half)
}

/// Capsule vs Capsule overlap test.
///
/// Reduces to closest-points-on-two-segments, then sphere-vs-sphere.
pub fn capsule_vs_capsule(
    pos_a: Vec3,
    radius_a: f32,
    half_height_a: f32,
    pos_b: Vec3,
    radius_b: f32,
    half_height_b: f32,
) -> Option<ContactPoint> {
    let a_top = pos_a + Vec3::new(0.0, half_height_a, 0.0);
    let a_bot = pos_a - Vec3::new(0.0, half_height_a, 0.0);
    let b_top = pos_b + Vec3::new(0.0, half_height_b, 0.0);
    let b_bot = pos_b - Vec3::new(0.0, half_height_b, 0.0);

    let (closest_a, closest_b) = closest_points_on_segments(a_top, a_bot, b_top, b_bot);

    sphere_vs_sphere(closest_a, radius_a, closest_b, radius_b)
}

/// Closest points between two line segments.
fn closest_points_on_segments(a1: Vec3, a2: Vec3, b1: Vec3, b2: Vec3) -> (Vec3, Vec3) {
    let d1 = a2 - a1;
    let d2 = b2 - b1;
    let r = a1 - b1;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a < 1e-12 && e < 1e-12 {
        return (a1, b1);
    }

    let (s, t);
    if a < 1e-12 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e < 1e-12 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b_val = d1.dot(d2);
            let denom = a * e - b_val * b_val;

            s = if denom.abs() > 1e-12 {
                ((b_val * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let t_nom = b_val * s + f;
            t = if e.abs() > 1e-12 {
                (t_nom / e).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    }

    (a1 + d1 * s, b1 + d2 * t)
}

// ---------------------------------------------------------------------------
// Dispatch: test any pair of shapes
// ---------------------------------------------------------------------------

/// Test two positioned shapes for overlap, returning a contact if found.
pub fn test_shapes(
    shape_a: &CollisionShape,
    pos_a: Vec3,
    shape_b: &CollisionShape,
    pos_b: Vec3,
) -> Option<ContactPoint> {
    use CollisionShape::*;

    match (shape_a, shape_b) {
        (Sphere { radius: ra }, Sphere { radius: rb }) => sphere_vs_sphere(pos_a, *ra, pos_b, *rb),
        (Sphere { radius }, Aabb { half_extents }) => {
            sphere_vs_aabb(pos_a, *radius, pos_b, *half_extents)
        }
        (Aabb { half_extents }, Sphere { radius }) => {
            // Flip normal direction
            sphere_vs_aabb(pos_b, *radius, pos_a, *half_extents).map(|mut c| {
                c.normal = -c.normal;
                c
            })
        }
        (Aabb { half_extents: ha }, Aabb { half_extents: hb }) => {
            aabb_vs_aabb(pos_a, *ha, pos_b, *hb)
        }
        (
            Capsule {
                radius: ra,
                half_height: hha,
            },
            Sphere { radius: rb },
        ) => capsule_vs_sphere(pos_a, *ra, *hha, pos_b, *rb),
        (
            Sphere { radius: ra },
            Capsule {
                radius: rb,
                half_height: hhb,
            },
        ) => capsule_vs_sphere(pos_b, *rb, *hhb, pos_a, *ra).map(|mut c| {
            c.normal = -c.normal;
            c
        }),
        (
            Capsule {
                radius: ra,
                half_height: hha,
            },
            Aabb { half_extents: hb },
        ) => capsule_vs_aabb(pos_a, *ra, *hha, pos_b, *hb),
        (
            Aabb { half_extents: ha },
            Capsule {
                radius: rb,
                half_height: hhb,
            },
        ) => capsule_vs_aabb(pos_b, *rb, *hhb, pos_a, *ha).map(|mut c| {
            c.normal = -c.normal;
            c
        }),
        (
            Capsule {
                radius: ra,
                half_height: hha,
            },
            Capsule {
                radius: rb,
                half_height: hhb,
            },
        ) => capsule_vs_capsule(pos_a, *ra, *hha, pos_b, *rb, *hhb),
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// Run narrow-phase collision detection on all broad-phase pairs.
pub fn narrow_phase_detect(
    pairs: Res<BroadPhasePairs>,
    query: Query<(&Transform, &CollisionShape, Option<&PhysicsMaterial>)>,
    mut contacts: ResMut<Contacts>,
) {
    contacts.manifolds.clear();

    let default_mat = PhysicsMaterial::default();

    for &(entity_a, entity_b) in &pairs.pairs {
        let Ok((tf_a, shape_a, mat_a)) = query.get(entity_a) else {
            continue;
        };
        let Ok((tf_b, shape_b, mat_b)) = query.get(entity_b) else {
            continue;
        };

        if let Some(contact) = test_shapes(shape_a, tf_a.translation, shape_b, tf_b.translation) {
            let mat_a = mat_a.unwrap_or(&default_mat);
            let mat_b = mat_b.unwrap_or(&default_mat);
            let combined = PhysicsMaterial::combined(mat_a, mat_b);

            contacts.manifolds.push(ContactManifold {
                entity_a,
                entity_b,
                contacts: vec![contact],
                material: combined,
            });
        }
    }
}

/// System set for narrow-phase ordering.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct NarrowPhaseSet;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Sphere vs Sphere ---

    #[test]
    fn spheres_overlapping() {
        let c = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(c.is_some());
        let c = c.unwrap();
        assert!((c.depth - 0.5).abs() < 1e-4, "depth = {}", c.depth);
        // Normal points from B toward A (i.e. in -X direction since A is at origin)
        assert!(
            c.normal.x < 0.0,
            "normal should point from B toward A: {:?}",
            c.normal
        );
    }

    #[test]
    fn spheres_just_touching() {
        let c = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::new(2.0, 0.0, 0.0), 1.0);
        assert!(c.is_none(), "Touching but not overlapping");
    }

    #[test]
    fn spheres_separated() {
        let c = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(c.is_none());
    }

    #[test]
    fn spheres_concentric() {
        let c = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::ZERO, 0.5);
        assert!(c.is_some());
        let c = c.unwrap();
        assert!((c.depth - 1.5).abs() < 1e-4);
    }

    // --- Sphere vs AABB ---

    #[test]
    fn sphere_inside_aabb() {
        let c = sphere_vs_aabb(Vec3::ZERO, 0.5, Vec3::ZERO, Vec3::splat(2.0));
        assert!(c.is_some());
    }

    #[test]
    fn sphere_touching_aabb_face() {
        let c = sphere_vs_aabb(Vec3::new(2.5, 0.0, 0.0), 1.0, Vec3::ZERO, Vec3::splat(2.0));
        assert!(c.is_some());
        let c = c.unwrap();
        assert!(c.normal.x > 0.0, "Should push sphere in +X");
    }

    #[test]
    fn sphere_far_from_aabb() {
        let c = sphere_vs_aabb(Vec3::new(10.0, 0.0, 0.0), 1.0, Vec3::ZERO, Vec3::splat(2.0));
        assert!(c.is_none());
    }

    // --- AABB vs AABB ---

    #[test]
    fn aabbs_overlapping_x() {
        let c = aabb_vs_aabb(
            Vec3::ZERO,
            Vec3::splat(1.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::splat(1.0),
        );
        assert!(c.is_some());
        let c = c.unwrap();
        assert!((c.depth - 0.5).abs() < 1e-4);
        assert!(c.normal.x.abs() > 0.5, "Should resolve along X");
    }

    #[test]
    fn aabbs_separated() {
        let c = aabb_vs_aabb(
            Vec3::ZERO,
            Vec3::splat(1.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::splat(1.0),
        );
        assert!(c.is_none());
    }

    #[test]
    fn aabbs_touching_edge() {
        let c = aabb_vs_aabb(
            Vec3::ZERO,
            Vec3::splat(1.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::splat(1.0),
        );
        assert!(
            c.is_none(),
            "Exactly touching = zero overlap = no collision"
        );
    }

    // --- Capsule vs Sphere ---

    #[test]
    fn capsule_sphere_side_collision() {
        let c = capsule_vs_sphere(Vec3::ZERO, 0.5, 1.0, Vec3::new(0.8, 0.0, 0.0), 0.5);
        assert!(c.is_some());
    }

    #[test]
    fn capsule_sphere_tip_collision() {
        let c = capsule_vs_sphere(Vec3::ZERO, 0.5, 1.0, Vec3::new(0.0, 1.8, 0.0), 0.5);
        assert!(c.is_some());
    }

    #[test]
    fn capsule_sphere_no_collision() {
        let c = capsule_vs_sphere(Vec3::ZERO, 0.5, 1.0, Vec3::new(5.0, 0.0, 0.0), 0.5);
        assert!(c.is_none());
    }

    // --- Capsule vs AABB ---

    #[test]
    fn capsule_aabb_overlapping() {
        let c = capsule_vs_aabb(
            Vec3::ZERO,
            0.5,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::splat(1.0),
        );
        assert!(c.is_some());
    }

    #[test]
    fn capsule_aabb_separated() {
        let c = capsule_vs_aabb(
            Vec3::ZERO,
            0.5,
            1.0,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::splat(1.0),
        );
        assert!(c.is_none());
    }

    // --- Capsule vs Capsule ---

    #[test]
    fn capsules_parallel_overlapping() {
        let c = capsule_vs_capsule(Vec3::ZERO, 0.5, 1.0, Vec3::new(0.8, 0.0, 0.0), 0.5, 1.0);
        assert!(c.is_some());
    }

    #[test]
    fn capsules_separated() {
        let c = capsule_vs_capsule(Vec3::ZERO, 0.5, 1.0, Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0);
        assert!(c.is_none());
    }

    // --- Shape dispatch ---

    #[test]
    fn test_shapes_dispatch_sphere_sphere() {
        let a = CollisionShape::sphere(1.0);
        let b = CollisionShape::sphere(1.0);
        let c = test_shapes(&a, Vec3::ZERO, &b, Vec3::new(1.5, 0.0, 0.0));
        assert!(c.is_some());
    }

    #[test]
    fn test_shapes_dispatch_aabb_sphere() {
        let a = CollisionShape::aabb(2.0, 2.0, 2.0);
        let b = CollisionShape::sphere(0.5);
        let c = test_shapes(&a, Vec3::ZERO, &b, Vec3::new(1.2, 0.0, 0.0));
        assert!(c.is_some());
    }

    #[test]
    fn test_shapes_dispatch_capsule_capsule() {
        let a = CollisionShape::capsule(0.3, 1.8);
        let b = CollisionShape::capsule(0.3, 1.8);
        let c = test_shapes(&a, Vec3::ZERO, &b, Vec3::new(0.5, 0.0, 0.0));
        assert!(c.is_some());
    }

    // --- Segment utility ---

    #[test]
    fn closest_point_on_segment_midpoint() {
        let p = closest_point_on_segment(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        assert!((p.y).abs() < 1e-6);
    }

    #[test]
    fn closest_point_on_segment_endpoint() {
        let p = closest_point_on_segment(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 5.0, 0.0),
        );
        assert!((p.y - 1.0).abs() < 1e-6);
    }
}
