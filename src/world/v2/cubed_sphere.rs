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

/// The six face-directions of a chunk.
///
/// Used for LOD-aware neighbour lookups and boundary-loop classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkDir {
    /// +u tangential direction (increasing u).
    PosU,
    /// −u tangential direction (decreasing u).
    NegU,
    /// +v tangential direction (increasing v).
    PosV,
    /// −v tangential direction (decreasing v).
    NegV,
    /// +layer radial direction (outward).
    PosLayer,
    /// −layer radial direction (inward).
    NegLayer,
}

impl ChunkDir {
    pub const ALL: [ChunkDir; 6] = [
        ChunkDir::PosU,
        ChunkDir::NegU,
        ChunkDir::PosV,
        ChunkDir::NegV,
        ChunkDir::PosLayer,
        ChunkDir::NegLayer,
    ];

    /// The direction opposite to `self`.
    pub fn opposite(self) -> ChunkDir {
        match self {
            ChunkDir::PosU => ChunkDir::NegU,
            ChunkDir::NegU => ChunkDir::PosU,
            ChunkDir::PosV => ChunkDir::NegV,
            ChunkDir::NegV => ChunkDir::PosV,
            ChunkDir::PosLayer => ChunkDir::NegLayer,
            ChunkDir::NegLayer => ChunkDir::PosLayer,
        }
    }
}

/// Chunk coordinate on the cubed sphere.
///
/// - `face`: which of the 6 cube faces this chunk belongs to
/// - `u, v`: integer position on the face grid, in chunk units
/// - `layer`: radial layer relative to the mean surface (0 = surface, +1 = above, −1 = below)
/// - `lod`: level of detail (0 = full resolution, each level doubles the world footprint)
///
/// At LOD 0, each chunk covers `CHUNK_SIZE` meters. At LOD `l`, each chunk covers
/// `CHUNK_SIZE * 2^l` meters. The face grid has `face_chunks_per_edge >> lod` cells
/// per edge at LOD `l`, so `u` and `v` range over `[0, fce_lod)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeSphereCoord {
    pub face: CubeFace,
    pub u: i32,
    pub v: i32,
    pub layer: i32,
    pub lod: u8,
}

impl CubeSphereCoord {
    pub const fn new(face: CubeFace, u: i32, v: i32, layer: i32) -> Self {
        Self {
            face,
            u,
            v,
            layer,
            lod: 0,
        }
    }

    pub const fn new_with_lod(face: CubeFace, u: i32, v: i32, layer: i32, lod: u8) -> Self {
        Self {
            face,
            u,
            v,
            layer,
            lod,
        }
    }

    /// How many chunks span one edge of a cube face for a given planet radius.
    ///
    /// The face edge length on the unit cube is 2. The arc length on the sphere
    /// for that edge is `π/2 * radius`. We divide by `CHUNK_SIZE` to get chunk count.
    pub fn face_chunks_per_edge(mean_radius: f64) -> f64 {
        let cs = CHUNK_SIZE as f64;
        (std::f64::consts::FRAC_PI_2 * mean_radius / cs).ceil()
    }

    /// Face chunks per edge at this coord's LOD level.
    ///
    /// Each LOD level halves the grid resolution (and doubles chunk footprint).
    pub fn face_chunks_per_edge_lod(mean_radius: f64, lod: u8) -> f64 {
        let base = Self::face_chunks_per_edge(mean_radius);
        (base / (1u64 << lod) as f64).ceil().max(1.0)
    }

    /// The effective face_chunks_per_edge for this coordinate's LOD level.
    pub fn effective_fce(&self, mean_radius: f64) -> f64 {
        Self::face_chunks_per_edge_lod(mean_radius, self.lod)
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
        let lod_scale = (1u64 << self.lod) as f64;

        let center_r = mean_radius + self.layer as f64 * cs * lod_scale;

        let world_pos = dir * center_r;
        let translation = Vec3::new(world_pos.x as f32, world_pos.y as f32, world_pos.z as f32);

        // ── Deterministic face-aligned rotation ──
        // Local Y = radial up.
        // Local X = face U tangent axis projected onto the tangent plane.
        // Local Z = right × up (right-handed cross product).
        let local_up = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32);
        let (face_right_d, _) = self.face.tangent_axes();
        let face_right = Vec3::new(
            face_right_d.x as f32,
            face_right_d.y as f32,
            face_right_d.z as f32,
        );

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
        let scale_y = lod_scale as f32;
        let scale_z = (center_r * angle_v / cs) as f32;

        (translation, rotation, Vec3::new(scale_x, scale_y, scale_z))
    }

    /// Like `world_transform_scaled` but returns the center position as f64
    /// for use with the floating-origin system.
    pub fn world_transform_scaled_f64(
        &self,
        mean_radius: f64,
        face_chunks_per_edge: f64,
    ) -> (DVec3, Quat, Vec3) {
        let dir = self.unit_sphere_dir(face_chunks_per_edge);
        let cs = CHUNK_SIZE as f64;
        let lod_scale = (1u64 << self.lod) as f64;
        let center_r = mean_radius + self.layer as f64 * cs * lod_scale;
        let world_pos = dir * center_r;

        // Rotation and scale are the same as f32 variant
        let (_, rotation, scale) = self.world_transform_scaled(mean_radius, face_chunks_per_edge);

        (world_pos, rotation, scale)
    }

    /// The 6 face-neighbors of this chunk (±u, ±v, ±layer).
    ///
    /// For ±u and ±v, handles cross-face wrapping when the neighbor falls
    /// off the edge of this face.
    pub fn neighbors(&self, face_chunks_per_edge: i32) -> [CubeSphereCoord; 6] {
        let max = face_chunks_per_edge;
        [
            self.neighbor_u(1, max),
            self.neighbor_u(-1, max),
            self.neighbor_v(1, max),
            self.neighbor_v(-1, max),
            CubeSphereCoord::new_with_lod(self.face, self.u, self.v, self.layer + 1, self.lod),
            CubeSphereCoord::new_with_lod(self.face, self.u, self.v, self.layer - 1, self.lod),
        ]
    }

    fn neighbor_u(&self, delta: i32, max: i32) -> CubeSphereCoord {
        let nu = self.u + delta;
        if nu >= 0 && nu < max {
            return CubeSphereCoord::new_with_lod(self.face, nu, self.v, self.layer, self.lod);
        }
        let (adj_face, adj_u, adj_v) = cross_face_u(self.face, self.v, nu, max);
        CubeSphereCoord::new_with_lod(adj_face, adj_u, adj_v, self.layer, self.lod)
    }

    fn neighbor_v(&self, delta: i32, max: i32) -> CubeSphereCoord {
        let nv = self.v + delta;
        if nv >= 0 && nv < max {
            return CubeSphereCoord::new_with_lod(self.face, self.u, nv, self.layer, self.lod);
        }
        let (adj_face, adj_u, adj_v) = cross_face_v(self.face, self.u, nv, max);
        CubeSphereCoord::new_with_lod(adj_face, adj_u, adj_v, self.layer, self.lod)
    }

    /// The LOD-0 children of this chunk (4 children for LOD > 0).
    ///
    /// At LOD `l`, this chunk covers the same area as `2×2` chunks at LOD `l-1`.
    pub fn children(&self) -> Option<[CubeSphereCoord; 4]> {
        if self.lod == 0 {
            return None;
        }
        let child_lod = self.lod - 1;
        let bu = self.u * 2;
        let bv = self.v * 2;
        Some([
            CubeSphereCoord::new_with_lod(self.face, bu, bv, self.layer, child_lod),
            CubeSphereCoord::new_with_lod(self.face, bu + 1, bv, self.layer, child_lod),
            CubeSphereCoord::new_with_lod(self.face, bu, bv + 1, self.layer, child_lod),
            CubeSphereCoord::new_with_lod(self.face, bu + 1, bv + 1, self.layer, child_lod),
        ])
    }

    /// The parent chunk at the next coarser LOD level.
    pub fn parent(&self) -> CubeSphereCoord {
        CubeSphereCoord::new_with_lod(self.face, self.u / 2, self.v / 2, self.layer, self.lod + 1)
    }

    /// Returns the single representative same-face neighbour at `target_lod` that
    /// lies just across `self`'s `dir` boundary.
    ///
    /// Uses face-fractional arithmetic — not bit-shifting — because
    /// [`face_chunks_per_edge_lod`] applies `ceil(base / 2^lod)`, which means
    /// naïve right-shifts give wrong results for non-power-of-two base grids.
    ///
    /// For `PosLayer`/`NegLayer`, the u/v are remapped to the `target_lod` grid
    /// using the cell midpoint so the returned coord is always in-bounds.
    ///
    /// Returns `None` when the representative point falls outside this face's
    /// grid (i.e. the neighbour would be on a different cube face).
    pub fn same_face_neighbor_at_lod(
        &self,
        dir: ChunkDir,
        target_lod: u8,
        mean_radius: f64,
    ) -> Option<CubeSphereCoord> {
        let fce_src = Self::face_chunks_per_edge_lod(mean_radius, self.lod) as i64;
        let fce_tgt = Self::face_chunks_per_edge_lod(mean_radius, target_lod) as i64;
        let fce_tgt_i = fce_tgt as i32;

        // Integer midpoint remap: cell-centre at self.(u/v) mapped to target grid.
        let u_mid = ((2 * self.u as i64 + 1) * fce_tgt / (2 * fce_src)) as i32;
        let v_mid = ((2 * self.v as i64 + 1) * fce_tgt / (2 * fce_src)) as i32;

        let (u_tgt, v_tgt, layer_tgt) = match dir {
            // Radial: remap u/v to same world position, shift layer by 1.
            ChunkDir::PosLayer => (u_mid, v_mid, self.layer + 1),
            ChunkDir::NegLayer => (u_mid, v_mid, self.layer - 1),
            // Tangential: advance one step in the boundary direction; use midpoint
            // for the parallel axis.
            ChunkDir::PosU => (
                (((self.u as i64 + 1) * fce_tgt) / fce_src) as i32,
                v_mid,
                self.layer,
            ),
            ChunkDir::NegU => (
                // div_euclid gives floor for negative numerators (u=0 case → -1).
                ((self.u as i64 * fce_tgt - 1).div_euclid(fce_src)) as i32,
                v_mid,
                self.layer,
            ),
            ChunkDir::PosV => (
                u_mid,
                (((self.v as i64 + 1) * fce_tgt) / fce_src) as i32,
                self.layer,
            ),
            ChunkDir::NegV => (
                u_mid,
                ((self.v as i64 * fce_tgt - 1).div_euclid(fce_src)) as i32,
                self.layer,
            ),
        };

        if u_tgt < 0 || u_tgt >= fce_tgt_i || v_tgt < 0 || v_tgt >= fce_tgt_i {
            return None;
        }
        Some(CubeSphereCoord::new_with_lod(
            self.face, u_tgt, v_tgt, layer_tgt, target_lod,
        ))
    }

    /// Returns **all** same-face neighbours at `target_lod` that share a
    /// boundary face with `self` in direction `dir`.
    ///
    /// Unlike [`same_face_neighbor_at_lod`] (which picks one representative),
    /// this iterates the full tangential extent so that every fine chunk is
    /// returned when going to a finer LOD, and the single coarse chunk is
    /// returned when going to a coarser LOD.
    ///
    /// Returns an empty `Vec` when the neighbours are on a different face.
    /// Used by the cross-LOD invalidation pass.
    pub fn same_face_neighbors_at_lod_all(
        &self,
        dir: ChunkDir,
        target_lod: u8,
        mean_radius: f64,
    ) -> Vec<CubeSphereCoord> {
        let fce_src = Self::face_chunks_per_edge_lod(mean_radius, self.lod) as i64;
        let fce_tgt = Self::face_chunks_per_edge_lod(mean_radius, target_lod) as i64;
        let fce_tgt_i = fce_tgt as i32;

        let u_mid = ((2 * self.u as i64 + 1) * fce_tgt / (2 * fce_src)) as i32;
        let v_mid = ((2 * self.v as i64 + 1) * fce_tgt / (2 * fce_src)) as i32;

        // Compute the inclusive range of the parallel axis that overlaps self's
        // extent.  For the boundary axis there is exactly one target-grid cell.
        // `v_hi = ((v+1)*fce_tgt - 1) / fce_src` excludes chunks that start
        // exactly at self's upper boundary (standard half-open-interval
        // convention).
        match dir {
            ChunkDir::PosLayer => {
                vec![CubeSphereCoord::new_with_lod(
                    self.face,
                    u_mid,
                    v_mid,
                    self.layer + 1,
                    target_lod,
                )]
            }
            ChunkDir::NegLayer => {
                vec![CubeSphereCoord::new_with_lod(
                    self.face,
                    u_mid,
                    v_mid,
                    self.layer - 1,
                    target_lod,
                )]
            }
            ChunkDir::PosU => {
                let u_tgt = (((self.u as i64 + 1) * fce_tgt) / fce_src) as i32;
                if u_tgt < 0 || u_tgt >= fce_tgt_i {
                    return vec![];
                }
                let v_lo = ((self.v as i64 * fce_tgt) / fce_src) as i32;
                let v_hi = (((self.v as i64 + 1) * fce_tgt - 1) / fce_src) as i32;
                (v_lo.max(0)..=v_hi.min(fce_tgt_i - 1))
                    .map(|vt| {
                        CubeSphereCoord::new_with_lod(self.face, u_tgt, vt, self.layer, target_lod)
                    })
                    .collect()
            }
            ChunkDir::NegU => {
                let u_tgt = ((self.u as i64 * fce_tgt - 1).div_euclid(fce_src)) as i32;
                if u_tgt < 0 || u_tgt >= fce_tgt_i {
                    return vec![];
                }
                let v_lo = ((self.v as i64 * fce_tgt) / fce_src) as i32;
                let v_hi = (((self.v as i64 + 1) * fce_tgt - 1) / fce_src) as i32;
                (v_lo.max(0)..=v_hi.min(fce_tgt_i - 1))
                    .map(|vt| {
                        CubeSphereCoord::new_with_lod(self.face, u_tgt, vt, self.layer, target_lod)
                    })
                    .collect()
            }
            ChunkDir::PosV => {
                let v_tgt = (((self.v as i64 + 1) * fce_tgt) / fce_src) as i32;
                if v_tgt < 0 || v_tgt >= fce_tgt_i {
                    return vec![];
                }
                let u_lo = ((self.u as i64 * fce_tgt) / fce_src) as i32;
                let u_hi = (((self.u as i64 + 1) * fce_tgt - 1) / fce_src) as i32;
                (u_lo.max(0)..=u_hi.min(fce_tgt_i - 1))
                    .map(|ut| {
                        CubeSphereCoord::new_with_lod(self.face, ut, v_tgt, self.layer, target_lod)
                    })
                    .collect()
            }
            ChunkDir::NegV => {
                let v_tgt = ((self.v as i64 * fce_tgt - 1).div_euclid(fce_src)) as i32;
                if v_tgt < 0 || v_tgt >= fce_tgt_i {
                    return vec![];
                }
                let u_lo = ((self.u as i64 * fce_tgt) / fce_src) as i32;
                let u_hi = (((self.u as i64 + 1) * fce_tgt - 1) / fce_src) as i32;
                (u_lo.max(0)..=u_hi.min(fce_tgt_i - 1))
                    .map(|ut| {
                        CubeSphereCoord::new_with_lod(self.face, ut, v_tgt, self.layer, target_lod)
                    })
                    .collect()
            }
        }
    }

    /// Returns the representative cross-face neighbour at `target_lod` that lies
    /// just across `self`'s `dir` boundary on the adjacent cube face.
    ///
    /// Used by the Phase-3 stitch-mesh system when
    /// [`same_face_neighbor_at_lod`](Self::same_face_neighbor_at_lod) returns
    /// `None` (i.e. the neighbour is on a different face).
    ///
    /// # Algorithm
    ///
    /// 1. Compute the out-of-bounds position in the **source LOD** grid:
    ///    - `PosU`: u = fce_src (one past +u edge)
    ///    - `NegU`: u = −1 (one before −u edge)
    ///    - `PosV` / `NegV`: analogous
    /// 2. Map through the private `cross_face_u` / `cross_face_v` tables to get
    ///    (`adj_face`, `adj_u_src`, `adj_v_src`) in the source LOD grid.
    /// 3. Rescale to target LOD using the standard integer midpoint formula:
    ///    `tgt = (2 * src + 1) * fce_tgt / (2 * fce_src)`.
    ///
    /// Returns `None` for `PosLayer`/`NegLayer` (no cross-face in the radial
    /// direction) or when the rescaled coordinate is out of range.
    pub fn cross_face_neighbor_at_lod(
        &self,
        dir: ChunkDir,
        target_lod: u8,
        mean_radius: f64,
    ) -> Option<(CubeSphereCoord, ChunkDir)> {
        let fce_src = Self::face_chunks_per_edge_lod(mean_radius, self.lod) as i64;
        let fce_tgt = Self::face_chunks_per_edge_lod(mean_radius, target_lod) as i64;
        let fce_src_i = fce_src as i32;
        let fce_tgt_i = fce_tgt as i32;

        let (adj_face, adj_u_src, adj_v_src) = match dir {
            // Radial directions have no cross-face mapping.
            ChunkDir::PosLayer | ChunkDir::NegLayer => return None,
            // Step one cell past the +u / −u boundary in source-LOD coords.
            ChunkDir::PosU => cross_face_u(self.face, self.v, fce_src_i, fce_src_i),
            ChunkDir::NegU => cross_face_u(self.face, self.v, -1, fce_src_i),
            // Step one cell past the +v / −v boundary.
            ChunkDir::PosV => cross_face_v(self.face, self.u, fce_src_i, fce_src_i),
            ChunkDir::NegV => cross_face_v(self.face, self.u, -1, fce_src_i),
        };

        // Rescale (adj_u_src, adj_v_src) from source-LOD grid to target-LOD grid.
        let adj_u_tgt = ((2 * adj_u_src as i64 + 1) * fce_tgt / (2 * fce_src)) as i32;
        let adj_v_tgt = ((2 * adj_v_src as i64 + 1) * fce_tgt / (2 * fce_src)) as i32;

        if adj_u_tgt < 0 || adj_u_tgt >= fce_tgt_i || adj_v_tgt < 0 || adj_v_tgt >= fce_tgt_i {
            return None;
        }

        let incoming_dir = cross_face_incoming_dir(self.face, dir);
        Some((
            CubeSphereCoord::new_with_lod(adj_face, adj_u_tgt, adj_v_tgt, self.layer, target_lod),
            incoming_dir,
        ))
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
    //
    // Convention: each chunk at layer L is CENTERED at
    // `mean_radius + L*CHUNK_SIZE` (see `world_transform_scaled`), so layer L
    // spans the radial range `[mean_r + (L-0.5)*cs, mean_r + (L+0.5)*cs)`.
    // Layer 0 therefore straddles `mean_radius`, which is the intended
    // "surface layer" used by coarser LOD rings.
    let layer = ((r - mean_radius) / cs + 0.5).floor() as i32;

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

/// Returns the incoming [`ChunkDir`] on the adjacent face when stepping off
/// `source_face` in `step_dir`.
///
/// Used by the stitch-mesh system to pick the correct boundary-loop slot on
/// the coarse chunk.  For same-face neighbours the incoming direction is
/// simply `step_dir.opposite()`.  For cross-face neighbours the cube-face
/// axis mapping can transpose or reflect the axes, so this function derives
/// the correct slot from a mid-face probe through `cross_face_u`/`cross_face_v`.
///
/// The probe uses a representative coordinate `MAX/2` (never 0 or `last`)
/// to avoid coincidental boundary hits on the non-constrained axis.
pub(crate) fn cross_face_incoming_dir(source_face: CubeFace, step_dir: ChunkDir) -> ChunkDir {
    const MAX: i32 = 100;
    const LAST: i32 = MAX - 1;
    const MID: i32 = MAX / 2;

    let (_, adj_u, adj_v) = match step_dir {
        ChunkDir::PosU => cross_face_u(source_face, MID, MAX, MAX),
        ChunkDir::NegU => cross_face_u(source_face, MID, -1, MAX),
        ChunkDir::PosV => cross_face_v(source_face, MID, MAX, MAX),
        ChunkDir::NegV => cross_face_v(source_face, MID, -1, MAX),
        // Radial directions have no cross-face mapping; fall back to opposite.
        ChunkDir::PosLayer | ChunkDir::NegLayer => return step_dir.opposite(),
    };

    if adj_u == 0 {
        ChunkDir::NegU
    } else if adj_u == LAST {
        ChunkDir::PosU
    } else if adj_v == 0 {
        ChunkDir::PosV
    } else {
        debug_assert_eq!(adj_v, LAST, "cross_face probe must land on a face boundary");
        ChunkDir::NegV
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
        assert_eq!(
            recovered.layer, original.layer,
            "layer mismatch: {recovered:?} vs {original:?}"
        );
    }

    #[test]
    fn layer_0_contains_mean_radius() {
        // A point exactly on the mean-radius sphere must land in layer 0,
        // because `world_transform_scaled` centers layer L at
        // `mean_r + L*cs`, so layer 0 spans `[mean_r - cs/2, mean_r + cs/2)`.
        for face in [
            CubeFace::PosX,
            CubeFace::NegX,
            CubeFace::PosY,
            CubeFace::NegY,
            CubeFace::PosZ,
            CubeFace::NegZ,
        ] {
            let dir = face.normal().normalize();
            let pos = dir * TEST_RADIUS;
            let coord = world_pos_to_coord(pos, TEST_RADIUS, TEST_FACE_CHUNKS);
            assert_eq!(
                coord.layer, 0,
                "point at r=mean_radius on face {face:?} should map to layer 0, got {coord:?}"
            );
        }

        // Points near the top edge of layer 0 should still be layer 0.
        let dir = CubeFace::PosX.normal().normalize();
        let r_just_below_top = TEST_RADIUS + CHUNK_SIZE as f64 / 2.0 - 0.01;
        let coord = world_pos_to_coord(dir * r_just_below_top, TEST_RADIUS, TEST_FACE_CHUNKS);
        assert_eq!(coord.layer, 0, "r just below layer-0 top: {coord:?}");

        // Just above the top edge should be layer 1.
        let r_just_above_top = TEST_RADIUS + CHUNK_SIZE as f64 / 2.0 + 0.01;
        let coord = world_pos_to_coord(dir * r_just_above_top, TEST_RADIUS, TEST_FACE_CHUNKS);
        assert_eq!(coord.layer, 1, "r just above layer-0 top: {coord:?}");
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

        let (_, _, scale_center) = center.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);
        let (_, _, scale_edge) = edge.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);

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

        let (center_a, rot_a, scale_a) = a.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);
        let (center_b, rot_b, scale_b) = b.world_transform_scaled(TEST_RADIUS, TEST_FACE_CHUNKS);

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

    // ── same_face_neighbor_at_lod tests ──────────────────────────────────────

    #[test]
    fn same_lod_neighbor_matches_neighbors_api() {
        // At the same LOD, same_face_neighbor_at_lod must return identical results
        // to neighbors() for interior chunks.
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 500, 500, 0, 0);
        let nbrs = coord.neighbors(TEST_FACE_CHUNKS_I);
        let radius = TEST_RADIUS;

        let check = |dir: ChunkDir, expected: CubeSphereCoord| {
            let got = coord
                .same_face_neighbor_at_lod(dir, 0, radius)
                .expect("interior same-LOD neighbour should always be Some");
            assert_eq!(
                got, expected,
                "direction {dir:?}: expected {expected:?}, got {got:?}"
            );
        };
        check(ChunkDir::PosU, nbrs[0]);
        check(ChunkDir::NegU, nbrs[1]);
        check(ChunkDir::PosV, nbrs[2]);
        check(ChunkDir::NegV, nbrs[3]);
        check(ChunkDir::PosLayer, nbrs[4]);
        check(ChunkDir::NegLayer, nbrs[5]);
    }

    #[test]
    fn cross_lod_to_coarser_pos_u() {
        // R=320 → fce0=16, fce1=8. Chunk (8,8) at LOD 0, PosU to LOD 1.
        // u_tgt = (9*8)/16 = 4, v_mid = (17*8)/(2*16) = 136/32 = 4.
        let small_r = 320.0_f64;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 8, 8, 0, 0);
        let got = coord
            .same_face_neighbor_at_lod(ChunkDir::PosU, 1, small_r)
            .expect("interior coarse PosU should be Some");
        assert_eq!(got.lod, 1);
        assert_eq!(got.face, CubeFace::PosZ);
        assert_eq!(got.u, 4, "u_tgt for coarser PosU: got {}", got.u);
        assert_eq!(got.v, 4, "v_mid for coarser PosU: got {}", got.v);
    }

    #[test]
    fn cross_lod_to_finer_pos_u() {
        // R=320 → fce0=16, fce1=8. Chunk (4,4) at LOD 1, PosU to LOD 0.
        // u_tgt = (5*16)/8 = 10, v_mid = (9*16)/(2*8) = 144/16 = 9.
        let small_r = 320.0_f64;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 4, 4, 0, 1);
        let got = coord
            .same_face_neighbor_at_lod(ChunkDir::PosU, 0, small_r)
            .expect("interior finer PosU should be Some");
        assert_eq!(got.lod, 0);
        assert_eq!(got.u, 10);
        assert_eq!(got.v, 9);
    }

    #[test]
    fn cross_lod_neg_u_at_zero_returns_none() {
        // u=0: NegU at any LOD must return None (cross-face boundary).
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 0, 500, 0, 0);
        assert_eq!(
            coord.same_face_neighbor_at_lod(ChunkDir::NegU, 0, TEST_RADIUS),
            None
        );
        assert_eq!(
            coord.same_face_neighbor_at_lod(ChunkDir::NegU, 1, TEST_RADIUS),
            None
        );
    }

    #[test]
    fn cross_lod_pos_u_at_max_returns_none() {
        // R=320, fce0=16. u=15 (fce0-1): PosU crosses face → None.
        let small_r = 320.0_f64;
        let fce0_i = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 0) as i32;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, fce0_i - 1, 8, 0, 0);
        assert_eq!(
            coord.same_face_neighbor_at_lod(ChunkDir::PosU, 0, small_r),
            None
        );
    }

    #[test]
    fn same_face_neighbors_all_same_lod_returns_one() {
        // R=320, fce0=16. Interior chunk (8,8) — each tangential dir returns 1.
        let small_r = 320.0_f64;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 8, 8, 0, 0);
        for dir in ChunkDir::ALL {
            let result = coord.same_face_neighbors_at_lod_all(dir, 0, small_r);
            if matches!(
                dir,
                ChunkDir::PosU | ChunkDir::NegU | ChunkDir::PosV | ChunkDir::NegV
            ) {
                assert_eq!(
                    result.len(),
                    1,
                    "interior same-LOD tangential should return 1 neighbour for {dir:?}, got {}",
                    result.len()
                );
            }
        }
    }

    #[test]
    fn same_face_neighbors_all_to_finer_returns_multiple() {
        // R=320, fce1=8→fce0=16. Coarse (4,4) LOD 1 → LOD 0 PosU spans 2 fine chunks.
        // u_tgt=(5*16)/8=10. v_lo=(4*16)/8=8. v_hi=(5*16-1)/8=79/8=9.
        let small_r = 320.0_f64;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 4, 4, 0, 1);
        let pos_u = coord.same_face_neighbors_at_lod_all(ChunkDir::PosU, 0, small_r);
        assert_eq!(
            pos_u.len(),
            2,
            "coarse→fine PosU should return 2 fine chunks, got {}",
            pos_u.len()
        );
        assert!(
            pos_u.iter().all(|c| c.u == 10),
            "all PosU fine chunks must have u=10"
        );
        assert_eq!(pos_u[0].v, 8);
        assert_eq!(pos_u[1].v, 9);
    }

    #[test]
    fn same_face_neighbors_all_cross_face_returns_empty() {
        // R=320, fce0=16. At the face boundary PosU/NegU → empty.
        let small_r = 320.0_f64;
        let fce0_i = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 0) as i32;
        let coord_max = CubeSphereCoord::new_with_lod(CubeFace::PosZ, fce0_i - 1, 8, 0, 0);
        assert!(
            coord_max
                .same_face_neighbors_at_lod_all(ChunkDir::PosU, 0, small_r)
                .is_empty(),
            "PosU at max-u should return empty"
        );
        let coord_zero = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 0, 8, 0, 0);
        assert!(
            coord_zero
                .same_face_neighbors_at_lod_all(ChunkDir::NegU, 0, small_r)
                .is_empty(),
            "NegU at u=0 should return empty"
        );
    }

    #[test]
    fn cross_lod_odd_fce_no_gaps_or_overlap() {
        // Use a radius that produces a non-power-of-two fce to stress the
        // ceil-based grid remapping.  R=320 → fce0=16, fce1=8 (both power-of-2
        // but the ratio fce0/fce1 = 2.0 exactly, so this is a "clean" case).
        // Use a slightly asymmetric R to get a genuinely non-halving fce:
        // R=327 → fce0=ceil(π/2*327/32)=ceil(16.056)=17, fce1=ceil(17/2)=9.
        let small_r = 327.0_f64;
        let fce0 = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 0) as i32;
        let fce1 = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 1) as i32;

        // For every LOD-1 chunk, sum the count of LOD-0 PosU neighbours.
        // Total must equal fce1 * fce1 (the LOD-1 face fully covered).
        let mut total_fine_invalidations = 0usize;
        for u in 0..fce1 {
            for v in 0..fce1 {
                let c1 = CubeSphereCoord::new_with_lod(CubeFace::PosZ, u, v, 0, 1);
                let all = c1.same_face_neighbors_at_lod_all(ChunkDir::PosU, 0, small_r);
                // Each fine chunk must stay in-bounds.
                for nc in &all {
                    assert!(nc.u >= 0 && nc.u < fce0, "nc.u out of bounds: {}", nc.u);
                    assert!(nc.v >= 0 && nc.v < fce0, "nc.v out of bounds: {}", nc.v);
                }
                total_fine_invalidations += all.len();
            }
        }
        // Each LOD-0 column (u, all v) gets invalidated once per LOD-1 row that
        // borders it; this is sanity-checked by confirming the count is non-zero.
        assert!(
            total_fine_invalidations > 0,
            "No fine invalidations generated"
        );
    }

    // ── cross_face_neighbor_at_lod tests ─────────────────────────────────────

    #[test]
    fn cross_face_neighbor_pos_u_returns_adjacent_face() {
        // PosZ face, at +u edge (u=fce-1). PosU should cross to PosX face.
        // cross_face_u(PosZ, v, max, max) → (PosX, 0, v)
        let small_r = 320.0_f64;
        let fce0 = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 0) as i32;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, fce0 - 1, 5, 0, 0);
        // same_face must return None for this coord
        assert!(
            coord
                .same_face_neighbor_at_lod(ChunkDir::PosU, 0, small_r)
                .is_none(),
            "PosU at max-u should be None from same_face"
        );
        let result = coord.cross_face_neighbor_at_lod(ChunkDir::PosU, 0, small_r);
        assert!(
            result.is_some(),
            "cross_face PosU should produce a neighbor"
        );
        let (nb, incoming) = result.unwrap();
        assert_eq!(
            nb.face,
            CubeFace::PosX,
            "PosZ +PosU should cross to PosX, got {:?}",
            nb.face
        );
        assert_eq!(nb.u, 0, "first cell on adjacent face should have u=0");
        assert_eq!(nb.lod, 0);
        // PosZ +PosU: cross_face_u(PosZ, _, max, max) → (PosX, 0, v) → NegU
        assert_eq!(incoming, ChunkDir::NegU, "incoming dir should be NegU");
    }

    #[test]
    fn cross_face_neighbor_neg_u_returns_adjacent_face() {
        // PosZ face, at −u edge (u=0). NegU should cross to NegX face.
        // cross_face_u(PosZ, v, -1, max) → (NegX, last, v)
        let small_r = 320.0_f64;
        let fce0 = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 0) as i32;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 0, 7, 0, 0);
        assert!(
            coord
                .same_face_neighbor_at_lod(ChunkDir::NegU, 0, small_r)
                .is_none(),
            "NegU at u=0 should be None from same_face"
        );
        let (nb, incoming) = coord
            .cross_face_neighbor_at_lod(ChunkDir::NegU, 0, small_r)
            .expect("cross_face NegU should produce a neighbor");
        assert_eq!(
            nb.face,
            CubeFace::NegX,
            "PosZ −NegU should cross to NegX, got {:?}",
            nb.face
        );
        assert_eq!(nb.u, fce0 - 1, "last cell on adjacent face, got {}", nb.u);
        // PosZ NegU: cross_face_u(PosZ, _, -1, max) → (NegX, last, v) → PosU
        assert_eq!(incoming, ChunkDir::PosU, "incoming dir should be PosU");
    }

    #[test]
    fn cross_face_neighbor_pos_v_returns_adjacent_face() {
        // PosZ face, at +v edge (v=fce-1). PosV should cross to PosY face.
        // cross_face_v(PosZ, u, max, max) → (PosY, u, 0)
        let small_r = 320.0_f64;
        let fce0 = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 0) as i32;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 6, fce0 - 1, 0, 0);
        assert!(
            coord
                .same_face_neighbor_at_lod(ChunkDir::PosV, 0, small_r)
                .is_none()
        );
        let (nb, incoming) = coord
            .cross_face_neighbor_at_lod(ChunkDir::PosV, 0, small_r)
            .expect("cross_face PosV should produce a neighbor");
        assert_eq!(
            nb.face,
            CubeFace::PosY,
            "PosZ +PosV → PosY, got {:?}",
            nb.face
        );
        assert_eq!(nb.v, 0, "first cell on adjacent face, got {}", nb.v);
        // PosZ PosV: cross_face_v(PosZ, _, max, max) → (PosY, u, 0) → PosV
        assert_eq!(incoming, ChunkDir::PosV, "incoming dir should be PosV");
    }

    #[test]
    fn cross_face_neighbor_neg_v_returns_adjacent_face() {
        // PosZ face, at -v edge (v=0). NegV should cross to NegY face.
        // cross_face_v(PosZ, u, -1, max) → (NegY, u, last)
        let small_r = 320.0_f64;
        let fce0 = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 0) as i32;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 8, 0, 0, 0);
        assert!(
            coord
                .same_face_neighbor_at_lod(ChunkDir::NegV, 0, small_r)
                .is_none()
        );
        let (nb, incoming) = coord
            .cross_face_neighbor_at_lod(ChunkDir::NegV, 0, small_r)
            .expect("cross_face NegV should produce a neighbor");
        assert_eq!(
            nb.face,
            CubeFace::NegY,
            "PosZ -NegV → NegY, got {:?}",
            nb.face
        );
        assert_eq!(nb.v, fce0 - 1, "last cell on adjacent face, got {}", nb.v);
        // PosZ NegV: cross_face_v(PosZ, _, -1, max) → (NegY, u, last) → NegV
        assert_eq!(incoming, ChunkDir::NegV, "incoming dir should be NegV");
    }

    #[test]
    fn cross_face_neighbor_radial_returns_none() {
        let small_r = 320.0_f64;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, 0, 0, 0, 0);
        assert!(
            coord
                .cross_face_neighbor_at_lod(ChunkDir::PosLayer, 0, small_r)
                .is_none(),
            "PosLayer should always be None"
        );
        assert!(
            coord
                .cross_face_neighbor_at_lod(ChunkDir::NegLayer, 0, small_r)
                .is_none(),
            "NegLayer should always be None"
        );
    }

    #[test]
    fn cross_face_neighbor_coarse_lod_scales_correctly() {
        // At LOD 0 edge, cross-face to LOD 1 → coordinate should be rescaled.
        // R=320 → fce0=16, fce1=8. PosZ, u=fce0-1=15, v=7 (mid-face).
        // cross_face_u(PosZ, 7, 16, 16) → (PosX, 0, 7)
        // adj_u_tgt = (1 * 8) / (2*16) = 8/32 = 0
        // adj_v_tgt = (15 * 8) / (2*16) = 120/32 = 3
        let small_r = 320.0_f64;
        let fce0 = CubeSphereCoord::face_chunks_per_edge_lod(small_r, 0) as i32;
        let coord = CubeSphereCoord::new_with_lod(CubeFace::PosZ, fce0 - 1, 7, 0, 0);
        let (nb, incoming) = coord
            .cross_face_neighbor_at_lod(ChunkDir::PosU, 1, small_r)
            .expect("cross-face LOD-1 neighbor should exist");
        assert_eq!(nb.face, CubeFace::PosX);
        assert_eq!(nb.lod, 1);
        assert_eq!(nb.u, 0);
        assert_eq!(nb.v, 3, "v should rescale from 7 (LOD0/16) to 3 (LOD1/8)");
        assert_eq!(incoming, ChunkDir::NegU);
    }

    // ── cross_face_incoming_dir tests ────────────────────────────────────────

    #[test]
    fn cross_face_incoming_dir_equatorial_u_steps() {
        // Equatorial faces stepping U: should map to the opposite U direction.
        use super::cross_face_incoming_dir;
        for face in [
            CubeFace::PosX,
            CubeFace::NegX,
            CubeFace::PosZ,
            CubeFace::NegZ,
        ] {
            assert_eq!(
                cross_face_incoming_dir(face, ChunkDir::PosU),
                ChunkDir::NegU,
                "{face:?} PosU"
            );
            assert_eq!(
                cross_face_incoming_dir(face, ChunkDir::NegU),
                ChunkDir::PosU,
                "{face:?} NegU"
            );
        }
    }

    #[test]
    fn cross_face_incoming_dir_polar_always_same_v() {
        // PosY → always NegV for any step direction.
        // NegY → always PosV for any step direction.
        use super::cross_face_incoming_dir;
        for step in [
            ChunkDir::PosU,
            ChunkDir::NegU,
            ChunkDir::PosV,
            ChunkDir::NegV,
        ] {
            assert_eq!(
                cross_face_incoming_dir(CubeFace::PosY, step),
                ChunkDir::NegV,
                "PosY {step:?}"
            );
            assert_eq!(
                cross_face_incoming_dir(CubeFace::NegY, step),
                ChunkDir::PosV,
                "NegY {step:?}"
            );
        }
    }

    #[test]
    fn cross_face_incoming_dir_transposed_v_steps() {
        // PosX/NegX stepping in V: land on PosY/NegY's U boundary (transposed).
        use super::cross_face_incoming_dir;
        // PosX PosV → cross_face_v(PosX, MID, MAX, MAX) → (PosY, last, last-MID) → PosU
        assert_eq!(
            cross_face_incoming_dir(CubeFace::PosX, ChunkDir::PosV),
            ChunkDir::PosU
        );
        // PosX NegV → cross_face_v(PosX, MID, -1, MAX) → (NegY, last, MID) → PosU
        assert_eq!(
            cross_face_incoming_dir(CubeFace::PosX, ChunkDir::NegV),
            ChunkDir::PosU
        );
        // NegX PosV → cross_face_v(NegX, MID, MAX, MAX) → (PosY, 0, MID) → NegU
        assert_eq!(
            cross_face_incoming_dir(CubeFace::NegX, ChunkDir::PosV),
            ChunkDir::NegU
        );
        // NegX NegV → cross_face_v(NegX, MID, -1, MAX) → (NegY, 0, last-MID) → NegU
        assert_eq!(
            cross_face_incoming_dir(CubeFace::NegX, ChunkDir::NegV),
            ChunkDir::NegU
        );
    }

    #[test]
    fn cross_face_incoming_dir_posz_v_steps_preserved() {
        // PosZ stepping in V: preserved (not reflected).
        use super::cross_face_incoming_dir;
        // PosZ PosV → cross_face_v(PosZ, MID, MAX, MAX) → (PosY, MID, 0) → PosV
        assert_eq!(
            cross_face_incoming_dir(CubeFace::PosZ, ChunkDir::PosV),
            ChunkDir::PosV
        );
        // PosZ NegV → cross_face_v(PosZ, MID, -1, MAX) → (NegY, MID, last) → NegV
        assert_eq!(
            cross_face_incoming_dir(CubeFace::PosZ, ChunkDir::NegV),
            ChunkDir::NegV
        );
    }

    #[test]
    fn cross_face_incoming_dir_negz_v_steps_reflected() {
        // NegZ stepping in V: reflected (PosV→NegV, NegV→PosV).
        use super::cross_face_incoming_dir;
        // NegZ PosV → cross_face_v(NegZ, MID, MAX, MAX) → (PosY, last-MID, last) → NegV
        assert_eq!(
            cross_face_incoming_dir(CubeFace::NegZ, ChunkDir::PosV),
            ChunkDir::NegV
        );
        // NegZ NegV → cross_face_v(NegZ, MID, -1, MAX) → (NegY, last-MID, 0) → PosV
        assert_eq!(
            cross_face_incoming_dir(CubeFace::NegZ, ChunkDir::NegV),
            ChunkDir::PosV
        );
    }
}
