// Cubed-sphere coordinate system for the V2 rendering pipeline.
//
// Maps a sphere onto 6 cube faces with uniform chunk grids. Each chunk is
// addressed by (face, u, v, layer) where (u, v) are tangent-plane positions
// and layer is the radial depth/height relative to the planet surface.
//
// The cubed-sphere projection normalizes cube-face points to the unit sphere,
// giving near-uniform cell sizes (max distortion ratio ~1.22 at face corners).

use bevy::math::{DVec3, Mat3, Quat, Vec3};

use crate::world::chunk::CHUNK_SIZE;

/// The six faces of a cube, used as the primary addressing axis for chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CubeFace {
    /// +X face (east)
    PosX = 0,
    /// −X face (west)
    NegX = 1,
    /// +Y face (top / north pole)
    PosY = 2,
    /// −Y face (bottom / south pole)
    NegY = 3,
    /// +Z face (south)
    PosZ = 4,
    /// −Z face (north)
    NegZ = 5,
}

impl CubeFace {
    pub const ALL: [CubeFace; 6] = [
        CubeFace::PosX,
        CubeFace::NegX,
        CubeFace::PosY,
        CubeFace::NegY,
        CubeFace::PosZ,
        CubeFace::NegZ,
    ];

    /// The outward-facing normal of this cube face (unit vector).
    pub fn normal(self) -> DVec3 {
        match self {
            CubeFace::PosX => DVec3::X,
            CubeFace::NegX => DVec3::NEG_X,
            CubeFace::PosY => DVec3::Y,
            CubeFace::NegY => DVec3::NEG_Y,
            CubeFace::PosZ => DVec3::Z,
            CubeFace::NegZ => DVec3::NEG_Z,
        }
    }

    /// The two tangent basis vectors for this face: (right, up).
    ///
    /// These define the local (u, v) coordinate axes on the face.
    /// `right` corresponds to increasing u, `up` to increasing v.
    pub fn tangent_axes(self) -> (DVec3, DVec3) {
        match self {
            //             right     up
            CubeFace::PosX => (DVec3::NEG_Z, DVec3::Y),
            CubeFace::NegX => (DVec3::Z, DVec3::Y),
            CubeFace::PosY => (DVec3::X, DVec3::NEG_Z),
            CubeFace::NegY => (DVec3::X, DVec3::Z),
            CubeFace::PosZ => (DVec3::X, DVec3::Y),
            CubeFace::NegZ => (DVec3::NEG_X, DVec3::Y),
        }
    }

    /// Determine which cube face a unit-sphere direction belongs to.
    ///
    /// The face is chosen by the component with the largest absolute value.
    pub fn from_unit_dir(dir: DVec3) -> Self {
        let ax = dir.x.abs();
        let ay = dir.y.abs();
        let az = dir.z.abs();
        if ax >= ay && ax >= az {
            if dir.x > 0.0 {
                CubeFace::PosX
            } else {
                CubeFace::NegX
            }
        } else if ay >= ax && ay >= az {
            if dir.y > 0.0 {
                CubeFace::PosY
            } else {
                CubeFace::NegY
            }
        } else if dir.z > 0.0 {
            CubeFace::PosZ
        } else {
            CubeFace::NegZ
        }
    }
}

/// Chunk coordinate on the cubed sphere.
///
/// - `face`: which of the 6 cube faces this chunk belongs to
/// - `u, v`: integer position on the face grid, in chunk units (each = CHUNK_SIZE meters)
/// - `layer`: radial layer relative to the mean surface (0 = surface, +1 = above, −1 = below)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeSphereCoord {
    pub face: CubeFace,
    pub u: i32,
    pub v: i32,
    pub layer: i32,
}

impl CubeSphereCoord {
    pub const fn new(face: CubeFace, u: i32, v: i32, layer: i32) -> Self {
        Self { face, u, v, layer }
    }

    /// How many chunks span one edge of a cube face for a given planet radius.
    ///
    /// The face edge length on the unit cube is 2. The arc length on the sphere
    /// for that edge is `π/2 * radius`. We divide by `CHUNK_SIZE` to get chunk count.
    pub fn face_chunks_per_edge(mean_radius: f64) -> f64 {
        let cs = CHUNK_SIZE as f64;
        (std::f64::consts::FRAC_PI_2 * mean_radius / cs).ceil()
    }

    /// Compute the unit-sphere direction for the center of this chunk's (u, v) cell.
    ///
    /// Maps the chunk's center on the face to a point on the unit cube,
    /// then normalizes to the unit sphere.
    pub fn unit_sphere_dir(&self, face_chunks_per_edge: f64) -> DVec3 {
        // Map (u, v) from chunk-grid coordinates to [-1, 1] on the face.
        // u=0, v=0 is at the center-left-bottom of the face grid.
        let half = face_chunks_per_edge / 2.0;
        let nu = (self.u as f64 + 0.5 - half) / half; // [-1, 1]
        let nv = (self.v as f64 + 0.5 - half) / half; // [-1, 1]

        let normal = self.face.normal();
        let (right, up) = self.face.tangent_axes();

        // Point on the unit cube face
        let cube_point = normal + right * nu + up * nv;

        // Project to unit sphere
        cube_point.normalize()
    }

    /// World-space center position and orientation for this chunk.
    ///
    /// Returns `(translation, rotation)` where:
    /// - `translation` is the world position of the chunk's local origin
    /// - `rotation` aligns local Y with the radial "up" direction
    ///
    /// `mean_radius` is the planet's mean surface radius in meters.
    /// `face_chunks_per_edge` is how many chunks span one face edge.
    pub fn world_transform(&self, mean_radius: f64, face_chunks_per_edge: f64) -> (Vec3, Quat) {
        let (translation, rotation, _scale) =
            self.world_transform_scaled(mean_radius, face_chunks_per_edge);
        (translation, rotation)
    }

    /// World-space transform with tangent-plane scale factors.
    ///
    /// Returns `(translation, rotation, scale)` where:
    /// - `translation` is the world position of the chunk's local origin
    /// - `rotation` is face-aligned: local X → face U direction (projected
    ///   onto tangent plane), local Y → radial up, local Z → right-hand cross
    /// - `scale.x` and `scale.z` compensate for gnomonic projection distortion
    ///   so that each chunk covers its correct angular footprint on the sphere.
    ///   `scale.y` is always 1.0 (radial layers are uniform).
    ///
    /// At face center the scale is ~1.27 (chunk must cover more arc than CS),
    /// at face edges ~0.64 (chunk must cover less arc). Without this scaling,
    /// there are visible gaps at face centers and overlaps at edges.
    pub fn world_transform_scaled(
        &self,
        mean_radius: f64,
        face_chunks_per_edge: f64,
    ) -> (Vec3, Quat, Vec3) {
        let dir = self.unit_sphere_dir(face_chunks_per_edge);
        let cs = CHUNK_SIZE as f64;

        let center_r = mean_radius + self.layer as f64 * cs;

        let world_pos = dir * center_r;
        let translation = Vec3::new(world_pos.x as f32, world_pos.y as f32, world_pos.z as f32);

        // ── Deterministic face-aligned rotation ──
        // Local Y = radial up.
        // Local X = face U tangent axis projected onto the tangent plane.
        // Local Z = right × up (right-handed cross product).
        let local_up = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32);
        let (face_right_d, _) = self.face.tangent_axes();
        let face_right =
            Vec3::new(face_right_d.x as f32, face_right_d.y as f32, face_right_d.z as f32);

        let right = (face_right - local_up * face_right.dot(local_up)).normalize();
        let forward = right.cross(local_up);

        let rotation = Quat::from_mat3(&Mat3::from_cols(right, local_up, forward));

        // ── Tangent-plane scale from gnomonic angular footprint ──
        let half = face_chunks_per_edge / 2.0;
        let nu = (self.u as f64 + 0.5 - half) / half;
        let nv = (self.v as f64 + 0.5 - half) / half;
        let dnu = 0.5 / half; // half-width of one chunk in cube-face coords

        let normal_d = self.face.normal();
        let (right_d, up_d) = self.face.tangent_axes();

        // Directions at the chunk's U and V edges on the unit cube, projected
        // to the unit sphere. The angle between them gives the true arc extent.
        let dir_u_lo = (normal_d + right_d * (nu - dnu) + up_d * nv).normalize();
        let dir_u_hi = (normal_d + right_d * (nu + dnu) + up_d * nv).normalize();
        let dir_v_lo = (normal_d + right_d * nu + up_d * (nv - dnu)).normalize();
        let dir_v_hi = (normal_d + right_d * nu + up_d * (nv + dnu)).normalize();

        let angle_u = dir_u_lo.dot(dir_u_hi).clamp(-1.0, 1.0).acos();
        let angle_v = dir_v_lo.dot(dir_v_hi).clamp(-1.0, 1.0).acos();

        // Scale = (actual arc extent at this radius) / (nominal chunk size CS).
        let scale_x = (center_r * angle_u / cs) as f32;
        let scale_z = (center_r * angle_v / cs) as f32;

        (translation, rotation, Vec3::new(scale_x, 1.0, scale_z))
    }

    /// The 6 face-neighbors of this chunk (±u, ±v, ±layer).
    ///
    /// For ±u and ±v, handles cross-face wrapping when the neighbor falls
    /// off the edge of this face.
    pub fn neighbors(&self, face_chunks_per_edge: i32) -> [CubeSphereCoord; 6] {
        let max = face_chunks_per_edge; // valid range: [0, max)
        [
            self.neighbor_u(1, max),
            self.neighbor_u(-1, max),
            self.neighbor_v(1, max),
            self.neighbor_v(-1, max),
            // Layer neighbors are always on the same face
            CubeSphereCoord::new(self.face, self.u, self.v, self.layer + 1),
            CubeSphereCoord::new(self.face, self.u, self.v, self.layer - 1),
        ]
    }

    fn neighbor_u(&self, delta: i32, max: i32) -> CubeSphereCoord {
        let nu = self.u + delta;
        if nu >= 0 && nu < max {
            return CubeSphereCoord::new(self.face, nu, self.v, self.layer);
        }
        // Cross-face: step off the u-edge
        let (adj_face, adj_u, adj_v) = cross_face_u(self.face, self.v, nu, max);
        CubeSphereCoord::new(adj_face, adj_u, adj_v, self.layer)
    }

    fn neighbor_v(&self, delta: i32, max: i32) -> CubeSphereCoord {
        let nv = self.v + delta;
        if nv >= 0 && nv < max {
            return CubeSphereCoord::new(self.face, self.u, nv, self.layer);
        }
        // Cross-face: step off the v-edge
        let (adj_face, adj_u, adj_v) = cross_face_v(self.face, self.u, nv, max);
        CubeSphereCoord::new(adj_face, adj_u, adj_v, self.layer)
    }
}

/// Convert a world-space position to the nearest `CubeSphereCoord`.
pub fn world_pos_to_coord(
    pos: DVec3,
    mean_radius: f64,
    face_chunks_per_edge: f64,
) -> CubeSphereCoord {
    let r = pos.length();
    let dir = if r > 1e-10 { pos / r } else { DVec3::Y };
    let cs = CHUNK_SIZE as f64;

    let face = CubeFace::from_unit_dir(dir);
    let (right, up) = face.tangent_axes();
    let normal = face.normal();

    // Project direction onto the cube face: find the scalar t such that
    // (dir * t) · normal = 1.0 (intersection with the unit cube face).
    let dn = dir.dot(normal);
    let t = if dn.abs() > 1e-12 { 1.0 / dn } else { 1.0 };
    let cube_point = dir * t;

    // Extract (u, v) in [-1, 1] from the cube-face point.
    let nu = cube_point.dot(right); // [-1, 1]
    let nv = cube_point.dot(up); // [-1, 1]

    // Convert to chunk grid coordinates.
    let half = face_chunks_per_edge / 2.0;
    let u = (nu * half + half).floor() as i32;
    let v = (nv * half + half).floor() as i32;

    // Radial layer from the surface.
    let layer = ((r - mean_radius) / cs).floor() as i32;

    CubeSphereCoord::new(face, u, v, layer)
}

/// Cross-face neighbor lookup when stepping off the u-edge of a face.
///
/// Given a face, the current v coordinate, the out-of-bounds u, and the
/// face grid size, returns (adjacent_face, new_u, new_v).
fn cross_face_u(face: CubeFace, v: i32, u_oob: i32, max: i32) -> (CubeFace, i32, i32) {
    let last = max - 1;
    if u_oob >= max {
        // Stepped off the +u edge
        match face {
            CubeFace::PosX => (CubeFace::NegZ, 0, v),
            CubeFace::NegX => (CubeFace::PosZ, 0, v),
            CubeFace::PosY => (CubeFace::PosX, v, last),
            CubeFace::NegY => (CubeFace::PosX, v, 0),
            CubeFace::PosZ => (CubeFace::PosX, 0, v),
            CubeFace::NegZ => (CubeFace::NegX, 0, v),
        }
    } else {
        // Stepped off the -u edge (u_oob < 0)
        match face {
            CubeFace::PosX => (CubeFace::PosZ, last, v),
            CubeFace::NegX => (CubeFace::NegZ, last, v),
            CubeFace::PosY => (CubeFace::NegX, last - v, last),
            CubeFace::NegY => (CubeFace::NegX, last - v, 0),
            CubeFace::PosZ => (CubeFace::NegX, last, v),
            CubeFace::NegZ => (CubeFace::PosX, last, v),
        }
    }
}

/// Cross-face neighbor lookup when stepping off the v-edge of a face.
fn cross_face_v(face: CubeFace, u: i32, v_oob: i32, max: i32) -> (CubeFace, i32, i32) {
    let last = max - 1;
    if v_oob >= max {
        // Stepped off the +v edge
        match face {
            CubeFace::PosX => (CubeFace::PosY, last, last - u),
            CubeFace::NegX => (CubeFace::PosY, 0, u),
            CubeFace::PosY => (CubeFace::NegZ, u, last),
            CubeFace::NegY => (CubeFace::PosZ, u, 0),
            CubeFace::PosZ => (CubeFace::PosY, u, 0),
            CubeFace::NegZ => (CubeFace::PosY, last - u, last),
        }
    } else {
        // Stepped off the -v edge (v_oob < 0)
        match face {
            CubeFace::PosX => (CubeFace::NegY, last, u),
            CubeFace::NegX => (CubeFace::NegY, 0, last - u),
            CubeFace::PosY => (CubeFace::PosZ, u, last),
            CubeFace::NegY => (CubeFace::NegZ, u, 0),
            CubeFace::PosZ => (CubeFace::NegY, u, last),
            CubeFace::NegZ => (CubeFace::NegY, last - u, 0),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_RADIUS: f64 = 32_000.0;
    const TEST_FACE_CHUNKS: f64 = 1000.0;
    const TEST_FACE_CHUNKS_I: i32 = 1000;

    #[test]
    fn face_from_unit_dir_identifies_dominant_axis() {
        assert_eq!(CubeFace::from_unit_dir(DVec3::X), CubeFace::PosX);
        assert_eq!(CubeFace::from_unit_dir(DVec3::NEG_X), CubeFace::NegX);
        assert_eq!(CubeFace::from_unit_dir(DVec3::Y), CubeFace::PosY);
        assert_eq!(CubeFace::from_unit_dir(DVec3::NEG_Y), CubeFace::NegY);
        assert_eq!(CubeFace::from_unit_dir(DVec3::Z), CubeFace::PosZ);
        assert_eq!(CubeFace::from_unit_dir(DVec3::NEG_Z), CubeFace::NegZ);
    }

    #[test]
    fn face_from_unit_dir_diagonal() {
        // (1, 0.5, 0.3) → dominant X → PosX
        let dir = DVec3::new(1.0, 0.5, 0.3).normalize();
        assert_eq!(CubeFace::from_unit_dir(dir), CubeFace::PosX);
    }

    #[test]
    fn unit_sphere_dir_is_normalized() {
        let coord = CubeSphereCoord::new(CubeFace::PosX, 500, 500, 0);
        let dir = coord.unit_sphere_dir(TEST_FACE_CHUNKS);
        let len = dir.length();
        assert!(
            (len - 1.0).abs() < 1e-10,
            "unit_sphere_dir should be unit length, got {len}"
        );
    }

    #[test]
    fn unit_sphere_dir_face_center_matches_normal() {
        // Center chunk of each face should point along the face normal.
        let half = (TEST_FACE_CHUNKS as i32) / 2;
        for face in CubeFace::ALL {
            let coord = CubeSphereCoord::new(face, half, half, 0);
            let dir = coord.unit_sphere_dir(TEST_FACE_CHUNKS);
            let expected = face.normal();
            let dot = dir.dot(expected);
            assert!(
                dot > 0.99,
                "Face {face:?} center dir should be ~face normal, dot={dot}"
            );
        }
    }

    #[test]
    fn world_transform_y_axis_is_radial() {
        let coord = CubeSphereCoord::new(CubeFace::PosZ, 500, 500, 0);
        let (translation, rotation) = coord.world_transform(TEST_RADIUS, TEST_FACE_CHUNKS);
        let local_y = rotation * Vec3::Y;
        let expected_up = translation.normalize();
        let dot = local_y.dot(expected_up);
        assert!(
            dot > 0.999,
            "Local Y should align with radial direction, dot={dot}"
        );
    }

    #[test]
    fn world_transform_frame_is_orthonormal() {
        let coord = CubeSphereCoord::new(CubeFace::PosX, 300, 700, -2);
        let (_translation, rotation) = coord.world_transform(TEST_RADIUS, TEST_FACE_CHUNKS);
        let x = rotation * Vec3::X;
        let y = rotation * Vec3::Y;
        let z = rotation * Vec3::Z;

        // Orthogonality
        assert!(x.dot(y).abs() < 1e-5, "X·Y should be ~0");
        assert!(x.dot(z).abs() < 1e-5, "X·Z should be ~0");
        assert!(y.dot(z).abs() < 1e-5, "Y·Z should be ~0");

        // Unit length
        assert!((x.length() - 1.0).abs() < 1e-5);
        assert!((y.length() - 1.0).abs() < 1e-5);
        assert!((z.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn world_transform_layer_offsets_radially() {
        let c0 = CubeSphereCoord::new(CubeFace::PosZ, 500, 500, 0);
        let c1 = CubeSphereCoord::new(CubeFace::PosZ, 500, 500, 1);
        let (t0, _) = c0.world_transform(TEST_RADIUS, TEST_FACE_CHUNKS);
        let (t1, _) = c1.world_transform(TEST_RADIUS, TEST_FACE_CHUNKS);
        let r0 = t0.length();
        let r1 = t1.length();
        let expected_diff = CHUNK_SIZE as f32;
        let actual_diff = r1 - r0;
        assert!(
            (actual_diff - expected_diff).abs() < 0.1,
            "Layer +1 should be {expected_diff}m further, got {actual_diff}m"
        );
    }

    #[test]
    fn roundtrip_world_pos_to_coord() {
        let original = CubeSphereCoord::new(CubeFace::PosX, 400, 600, 0);
        let (translation, _) = original.world_transform(TEST_RADIUS, TEST_FACE_CHUNKS);
        let pos = DVec3::new(
            translation.x as f64,
            translation.y as f64,
            translation.z as f64,
        );
        let recovered = world_pos_to_coord(pos, TEST_RADIUS, TEST_FACE_CHUNKS);
        assert_eq!(
            recovered.face, original.face,
            "Face mismatch: {recovered:?} vs {original:?}"
        );
        assert!(
            (recovered.u - original.u).abs() <= 1,
            "u mismatch: {recovered:?} vs {original:?}"
        );
        assert!(
            (recovered.v - original.v).abs() <= 1,
            "v mismatch: {recovered:?} vs {original:?}"
        );
        assert!(
            (recovered.layer - original.layer).abs() <= 1,
            "layer mismatch: {recovered:?} vs {original:?}"
        );
    }

    #[test]
    fn roundtrip_multiple_faces() {
        let coords = [
            CubeSphereCoord::new(CubeFace::PosX, 500, 500, 0),
            CubeSphereCoord::new(CubeFace::NegX, 200, 800, -1),
            CubeSphereCoord::new(CubeFace::PosY, 500, 500, 2),
            CubeSphereCoord::new(CubeFace::NegY, 500, 500, 0),
            CubeSphereCoord::new(CubeFace::PosZ, 300, 700, 0),
            CubeSphereCoord::new(CubeFace::NegZ, 600, 400, 1),
        ];
        for original in coords {
            let (translation, _) = original.world_transform(TEST_RADIUS, TEST_FACE_CHUNKS);
            let pos = DVec3::new(
                translation.x as f64,
                translation.y as f64,
                translation.z as f64,
            );
            let recovered = world_pos_to_coord(pos, TEST_RADIUS, TEST_FACE_CHUNKS);
            assert_eq!(
                recovered.face, original.face,
                "Face mismatch for {original:?}: got {recovered:?}"
            );
        }
    }

    #[test]
    fn same_face_neighbors_stay_on_face() {
        let coord = CubeSphereCoord::new(CubeFace::PosZ, 500, 500, 0);
        let nbrs = coord.neighbors(TEST_FACE_CHUNKS_I);
        // ±u, ±v should remain on PosZ
        for n in &nbrs[0..4] {
            assert_eq!(n.face, CubeFace::PosZ, "Expected same face for {n:?}");
        }
        // ±layer should remain on PosZ
        assert_eq!(nbrs[4].layer, 1);
        assert_eq!(nbrs[5].layer, -1);
    }

    #[test]
    fn cross_face_neighbor_u_positive_edge() {
        // At u = max-1 on PosZ, stepping +u should go to PosX
        let max = TEST_FACE_CHUNKS_I;
        let coord = CubeSphereCoord::new(CubeFace::PosZ, max - 1, 500, 0);
        let nbrs = coord.neighbors(max);
        let plus_u = nbrs[0]; // +u neighbor
        assert_eq!(plus_u.face, CubeFace::PosX, "Should cross to PosX");
        assert_eq!(plus_u.u, 0, "Should enter at u=0 on PosX");
    }

    #[test]
    fn cross_face_neighbor_u_negative_edge() {
        // At u = 0 on PosZ, stepping -u should go to NegX
        let max = TEST_FACE_CHUNKS_I;
        let coord = CubeSphereCoord::new(CubeFace::PosZ, 0, 500, 0);
        let nbrs = coord.neighbors(max);
        let minus_u = nbrs[1]; // -u neighbor
        assert_eq!(minus_u.face, CubeFace::NegX, "Should cross to NegX");
        assert_eq!(minus_u.u, max - 1, "Should enter at u=max-1");
    }

    #[test]
    fn cross_face_neighbor_v_positive_edge() {
        // At v = max-1 on PosZ, stepping +v should go to PosY
        let max = TEST_FACE_CHUNKS_I;
        let coord = CubeSphereCoord::new(CubeFace::PosZ, 500, max - 1, 0);
        let nbrs = coord.neighbors(max);
        let plus_v = nbrs[2]; // +v neighbor
        assert_eq!(plus_v.face, CubeFace::PosY, "Should cross to PosY");
    }

    #[test]
    fn neighbor_layer_preserves_face_and_uv() {
        let coord = CubeSphereCoord::new(CubeFace::NegX, 300, 400, 2);
        let nbrs = coord.neighbors(TEST_FACE_CHUNKS_I);
        let above = nbrs[4]; // +layer
        let below = nbrs[5]; // -layer
        assert_eq!(above, CubeSphereCoord::new(CubeFace::NegX, 300, 400, 3));
        assert_eq!(below, CubeSphereCoord::new(CubeFace::NegX, 300, 400, 1));
    }

    #[test]
    fn tangent_scale_larger_at_face_center_than_edge() {
        // Gnomonic distortion: face center needs larger scale than face edge.
        let half = TEST_FACE_CHUNKS_I / 2;
        let center = CubeSphereCoord::new(CubeFace::PosZ, half, half, 0);
        let edge = CubeSphereCoord::new(CubeFace::PosZ, TEST_FACE_CHUNKS_I - 1, half, 0);

        let (_, _, scale_center) =
            center.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);
        let (_, _, scale_edge) =
            edge.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);

        assert!(
            scale_center.x > scale_edge.x,
            "Face center scale_x ({}) should be > edge scale_x ({})",
            scale_center.x,
            scale_edge.x,
        );
    }

    #[test]
    fn tangent_scale_is_symmetric() {
        // Chunks equidistant from face center should have the same scale.
        let half = TEST_FACE_CHUNKS_I / 2;
        let left = CubeSphereCoord::new(CubeFace::PosZ, half - 100, half, 0);
        let right = CubeSphereCoord::new(CubeFace::PosZ, half + 100, half, 0);

        let (_, _, scale_l) = left.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);
        let (_, _, scale_r) = right.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);

        assert!(
            (scale_l.x - scale_r.x).abs() < 0.01,
            "Symmetric chunks should have similar scale: {} vs {}",
            scale_l.x,
            scale_r.x,
        );
    }

    #[test]
    fn adjacent_chunks_edges_meet() {
        // After scaling, adjacent chunks' world-space edges should approximately meet.
        let half = TEST_FACE_CHUNKS_I / 2;
        let cs = CHUNK_SIZE as f32;

        let a = CubeSphereCoord::new(CubeFace::PosZ, half, half, 0);
        let b = CubeSphereCoord::new(CubeFace::PosZ, half + 1, half, 0);

        let (center_a, rot_a, scale_a) =
            a.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);
        let (center_b, rot_b, scale_b) =
            b.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);

        // Right edge of chunk A: center_a + rot_a * (cs/2 * scale_a.x, 0, 0)
        let edge_a = center_a + rot_a * Vec3::new(cs / 2.0 * scale_a.x, 0.0, 0.0);
        // Left edge of chunk B: center_b + rot_b * (-cs/2 * scale_b.x, 0, 0)
        let edge_b = center_b + rot_b * Vec3::new(-cs / 2.0 * scale_b.x, 0.0, 0.0);

        let gap = (edge_a - edge_b).length();
        assert!(
            gap < 1.0,
            "Adjacent chunk edges should meet within 1m, got {gap:.3}m gap"
        );
    }
}
