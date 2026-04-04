// Discrete grid ray marching for voxel grids.
//
// Traces rays through a flat `size³` voxel array along the 26 grid-aligned
// directions (6 cardinal, 12 edge-diagonal, 8 corner-diagonal). Two march
// variants:
//
// - `march_grid_ray` — binary opaque/transparent: non-air stops the ray.
// - `march_grid_ray_attenuated` — Beer-Lambert attenuation through
//    semi-transparent media (water, ice, steam). Returns transmittance.
//
// Used by radiative heat transfer (Phase 9a) and later by optics (Phase 11).

use crate::world::voxel::{MaterialId, Voxel};

/// Result of a successful ray march — the first opaque voxel hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    /// Flat index into the `size³` voxel array.
    pub index: usize,
    /// Euclidean distance from origin to hit in voxel units (= meters).
    pub distance: f32,
}

/// Ray hit with Beer-Lambert transmittance through intervening media.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttenuatedRayHit {
    /// The opaque surface that terminated the ray.
    pub hit: RayHit,
    /// Fraction of radiation reaching the target (0.0–1.0).
    /// `transmittance = exp(−Σ(αᵢ × dᵢ))` via Beer-Lambert law.
    pub transmittance: f32,
}

/// Transmittance below this threshold is treated as fully absorbed (early exit).
const MIN_TRANSMITTANCE: f32 = 0.001;

/// All 26 grid-aligned directions: 6 cardinal + 12 edge + 8 corner.
pub const RAY_DIRECTIONS: [[i32; 3]; 26] = [
    // 6 cardinal
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    // 12 edge-diagonal
    [1, 1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, 1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, 1, -1],
    [0, -1, 1],
    [0, -1, -1],
    // 8 corner-diagonal
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
];

/// Precomputed step distance for each direction in [`RAY_DIRECTIONS`].
///
/// Cardinal = 1.0, edge = √2, corner = √3.
const STEP_DISTANCES: [f32; 26] = {
    let mut dists = [0.0_f32; 26];
    let mut i = 0;
    while i < 26 {
        let d = &RAY_DIRECTIONS[i];
        let sq = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f32;
        // const-compatible sqrt approximation: all values are 1, 2, or 3.
        dists[i] = if sq < 1.5 {
            1.0
        } else if sq < 2.5 {
            std::f32::consts::SQRT_2
        } else {
            // √3 — no stdlib constant, so literal value
            1.732_050_8
        };
        i += 1;
    }
    dists
};

/// March a ray through a flat `size³` voxel grid in a discrete direction.
///
/// Starting from `(sx, sy, sz)`, steps one voxel at a time along `dir`
/// (each component −1, 0, or +1). Returns the first non-air voxel hit,
/// or `None` if the ray exits the grid or exceeds `max_steps`.
///
/// The origin voxel itself is never returned as a hit.
pub fn march_grid_ray(
    voxels: &[Voxel],
    size: usize,
    start: [usize; 3],
    dir_index: usize,
    max_steps: usize,
) -> Option<RayHit> {
    debug_assert!(dir_index < 26);
    let dir = RAY_DIRECTIONS[dir_index];
    let step_dist = STEP_DISTANCES[dir_index];

    let mut x = start[0] as i32;
    let mut y = start[1] as i32;
    let mut z = start[2] as i32;
    let bound = size as i32;

    for step in 1..=max_steps {
        x += dir[0];
        y += dir[1];
        z += dir[2];

        if x < 0 || y < 0 || z < 0 || x >= bound || y >= bound || z >= bound {
            return None;
        }

        let ux = x as usize;
        let uy = y as usize;
        let uz = z as usize;
        let idx = uz * size * size + uy * size + ux;

        if !voxels[idx].material.is_air() {
            return Some(RayHit {
                x: ux,
                y: uy,
                z: uz,
                index: idx,
                distance: step as f32 * step_dist,
            });
        }
    }

    None
}

/// March a ray with Beer-Lambert attenuation through semi-transparent media.
///
/// `get_absorption` classifies each material:
/// - `Some(0.0)` — fully transparent (air), no attenuation
/// - `Some(α)` where α > 0 — semi-transparent, attenuate by exp(−α × step_dist)
/// - `None` — opaque surface, terminates the ray (returned as hit)
///
/// Returns the first opaque hit plus the cumulative transmittance through
/// any intervening semi-transparent voxels. Returns `None` if the ray exits
/// the grid, exceeds `max_steps`, or is fully absorbed (transmittance < 0.001).
pub fn march_grid_ray_attenuated<F>(
    voxels: &[Voxel],
    size: usize,
    start: [usize; 3],
    dir_index: usize,
    max_steps: usize,
    get_absorption: F,
) -> Option<AttenuatedRayHit>
where
    F: Fn(MaterialId) -> Option<f32>,
{
    debug_assert!(dir_index < 26);
    let dir = RAY_DIRECTIONS[dir_index];
    let step_dist = STEP_DISTANCES[dir_index];

    let mut x = start[0] as i32;
    let mut y = start[1] as i32;
    let mut z = start[2] as i32;
    let bound = size as i32;
    let mut optical_depth: f32 = 0.0;

    for step in 1..=max_steps {
        x += dir[0];
        y += dir[1];
        z += dir[2];

        if x < 0 || y < 0 || z < 0 || x >= bound || y >= bound || z >= bound {
            return None;
        }

        let ux = x as usize;
        let uy = y as usize;
        let uz = z as usize;
        let idx = uz * size * size + uy * size + ux;

        match get_absorption(voxels[idx].material) {
            Some(alpha) => {
                optical_depth += alpha * step_dist;
                if (-optical_depth).exp() < MIN_TRANSMITTANCE {
                    return None; // fully absorbed by medium
                }
            }
            None => {
                return Some(AttenuatedRayHit {
                    hit: RayHit {
                        x: ux,
                        y: uy,
                        z: uz,
                        index: idx,
                        distance: step as f32 * step_dist,
                    },
                    transmittance: (-optical_depth).exp(),
                });
            }
        }
    }

    None
}

/// Check whether a non-air voxel at `(x, y, z)` has at least one
/// air-adjacent face (i.e. is a surface voxel exposed to radiation).
pub fn is_surface_voxel(voxels: &[Voxel], size: usize, x: usize, y: usize, z: usize) -> bool {
    let bound = size as i32;
    let xi = x as i32;
    let yi = y as i32;
    let zi = z as i32;

    for &(dx, dy, dz) in &[
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ] {
        let nx = xi + dx;
        let ny = yi + dy;
        let nz = zi + dz;
        if nx < 0 || ny < 0 || nz < 0 || nx >= bound || ny >= bound || nz >= bound {
            return true; // grid boundary = exposed
        }
        let ni = nz as usize * size * size + ny as usize * size + nx as usize;
        if voxels[ni].material.is_air() {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Arbitrary-direction DDA raymarcher (Amanatides & Woo)
// ---------------------------------------------------------------------------

/// Result of a DDA ray march through a voxel grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdaHit {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    /// Which axis the ray crossed to enter this voxel (0=X, 1=Y, 2=Z).
    pub face_axis: usize,
    /// Sign of the face normal along `face_axis` (+1.0 or −1.0).
    pub face_sign: f32,
    /// Distance from ray origin to the hit point.
    pub t: f32,
}

impl DdaHit {
    /// Face normal as `[f32; 3]`.
    pub fn face_normal(&self) -> [f32; 3] {
        let mut n = [0.0_f32; 3];
        n[self.face_axis] = self.face_sign;
        n
    }
}

/// DDA ray march result with per-channel RGB transmittance (Beer-Lambert).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdaAttenuatedHit {
    pub hit: DdaHit,
    /// Per-channel transmittance `[R, G, B]` through intervening media.
    pub transmittance: [f32; 3],
}

/// March an arbitrary ray through a flat `size³` voxel grid using the
/// Amanatides & Woo DDA algorithm.
///
/// `origin` and `dir` are in world-space coordinates where one voxel = one
/// unit. The grid occupies `[0, size)³`. Handles rays originating outside the
/// grid by advancing to the AABB entry point.
///
/// Returns the first non-air voxel hit, or `None` if the ray misses or
/// exceeds `max_dist`.
pub fn dda_march_ray(
    voxels: &[Voxel],
    size: usize,
    origin: [f32; 3],
    dir: [f32; 3],
    max_dist: f32,
) -> Option<DdaHit> {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if len < 1e-10 {
        return None;
    }
    let dir = [dir[0] / len, dir[1] / len, dir[2] / len];
    let fs = size as f32;

    let (t_enter, t_exit) = ray_aabb(origin, dir, fs);
    if t_enter >= t_exit || t_exit < 0.0 {
        return None;
    }

    let t_start = t_enter.max(0.0) + 0.001;
    let pos = [
        origin[0] + dir[0] * t_start,
        origin[1] + dir[1] * t_start,
        origin[2] + dir[2] * t_start,
    ];

    let bound = size as i32;
    let mut ix = (pos[0].floor() as i32).clamp(0, bound - 1);
    let mut iy = (pos[1].floor() as i32).clamp(0, bound - 1);
    let mut iz = (pos[2].floor() as i32).clamp(0, bound - 1);

    let step_x: i32 = if dir[0] >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir[1] >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir[2] >= 0.0 { 1 } else { -1 };

    let dt_x = if dir[0].abs() > 1e-10 {
        (1.0 / dir[0]).abs()
    } else {
        f32::MAX
    };
    let dt_y = if dir[1].abs() > 1e-10 {
        (1.0 / dir[1]).abs()
    } else {
        f32::MAX
    };
    let dt_z = if dir[2].abs() > 1e-10 {
        (1.0 / dir[2]).abs()
    } else {
        f32::MAX
    };

    let mut t_max_x = if dir[0] >= 0.0 {
        ((ix + 1) as f32 - pos[0]) * dt_x
    } else {
        (pos[0] - ix as f32) * dt_x
    };
    let mut t_max_y = if dir[1] >= 0.0 {
        ((iy + 1) as f32 - pos[1]) * dt_y
    } else {
        (pos[1] - iy as f32) * dt_y
    };
    let mut t_max_z = if dir[2] >= 0.0 {
        ((iz + 1) as f32 - pos[2]) * dt_z
    } else {
        (pos[2] - iz as f32) * dt_z
    };

    let mut t_total = t_start;
    let mut last_axis = 1_usize;
    let max_steps = size * 3;

    for _ in 0..max_steps {
        if ix < 0 || iy < 0 || iz < 0 {
            return None;
        }
        let (ux, uy, uz) = (ix as usize, iy as usize, iz as usize);
        if ux >= size || uy >= size || uz >= size {
            return None;
        }

        let idx = uz * size * size + uy * size + ux;
        if !voxels[idx].material.is_air() {
            return Some(DdaHit {
                x: ux,
                y: uy,
                z: uz,
                face_axis: last_axis,
                face_sign: match last_axis {
                    0 => -step_x as f32,
                    1 => -step_y as f32,
                    _ => -step_z as f32,
                },
                t: t_total,
            });
        }

        // Advance (Amanatides & Woo step).
        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                t_total = t_start + t_max_x;
                t_max_x += dt_x;
                ix += step_x;
                last_axis = 0;
            } else {
                t_total = t_start + t_max_z;
                t_max_z += dt_z;
                iz += step_z;
                last_axis = 2;
            }
        } else if t_max_y < t_max_z {
            t_total = t_start + t_max_y;
            t_max_y += dt_y;
            iy += step_y;
            last_axis = 1;
        } else {
            t_total = t_start + t_max_z;
            t_max_z += dt_z;
            iz += step_z;
            last_axis = 2;
        }

        if t_total - t_start > max_dist {
            return None;
        }
    }

    None
}

/// March a ray with per-channel RGB Beer-Lambert attenuation.
///
/// `get_absorption_rgb` returns per-channel absorption coefficients for a
/// material: `Some([α_r, α_g, α_b])` for transparent/semi-transparent,
/// `None` for opaque (terminates the ray).
///
/// Returns the first opaque hit plus per-channel transmittance through any
/// intervening media.
pub fn dda_march_ray_attenuated<F>(
    voxels: &[Voxel],
    size: usize,
    origin: [f32; 3],
    dir: [f32; 3],
    max_dist: f32,
    get_absorption_rgb: F,
) -> Option<DdaAttenuatedHit>
where
    F: Fn(MaterialId) -> Option<[f32; 3]>,
{
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if len < 1e-10 {
        return None;
    }
    let dir = [dir[0] / len, dir[1] / len, dir[2] / len];
    let fs = size as f32;

    let (t_enter, t_exit) = ray_aabb(origin, dir, fs);
    if t_enter >= t_exit || t_exit < 0.0 {
        return None;
    }

    let t_start = t_enter.max(0.0) + 0.001;
    let pos = [
        origin[0] + dir[0] * t_start,
        origin[1] + dir[1] * t_start,
        origin[2] + dir[2] * t_start,
    ];

    let bound = size as i32;
    let mut ix = (pos[0].floor() as i32).clamp(0, bound - 1);
    let mut iy = (pos[1].floor() as i32).clamp(0, bound - 1);
    let mut iz = (pos[2].floor() as i32).clamp(0, bound - 1);

    let step_x: i32 = if dir[0] >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir[1] >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir[2] >= 0.0 { 1 } else { -1 };

    let dt_x = if dir[0].abs() > 1e-10 {
        (1.0 / dir[0]).abs()
    } else {
        f32::MAX
    };
    let dt_y = if dir[1].abs() > 1e-10 {
        (1.0 / dir[1]).abs()
    } else {
        f32::MAX
    };
    let dt_z = if dir[2].abs() > 1e-10 {
        (1.0 / dir[2]).abs()
    } else {
        f32::MAX
    };

    let mut t_max_x = if dir[0] >= 0.0 {
        ((ix + 1) as f32 - pos[0]) * dt_x
    } else {
        (pos[0] - ix as f32) * dt_x
    };
    let mut t_max_y = if dir[1] >= 0.0 {
        ((iy + 1) as f32 - pos[1]) * dt_y
    } else {
        (pos[1] - iy as f32) * dt_y
    };
    let mut t_max_z = if dir[2] >= 0.0 {
        ((iz + 1) as f32 - pos[2]) * dt_z
    } else {
        (pos[2] - iz as f32) * dt_z
    };

    let mut t_total = t_start;
    let mut last_axis = 1_usize;
    let max_steps = size * 3;
    let mut optical_depth = [0.0_f32; 3];
    let mut prev_t = t_start;

    for _ in 0..max_steps {
        if ix < 0 || iy < 0 || iz < 0 {
            return None;
        }
        let (ux, uy, uz) = (ix as usize, iy as usize, iz as usize);
        if ux >= size || uy >= size || uz >= size {
            return None;
        }

        let idx = uz * size * size + uy * size + ux;
        let mat = voxels[idx].material;

        match get_absorption_rgb(mat) {
            Some(alpha_rgb) => {
                // Distance through this voxel (approximate — DDA step length).
                let step_len = t_total - prev_t;
                let d = step_len.max(0.5); // minimum half-voxel
                optical_depth[0] += alpha_rgb[0] * d;
                optical_depth[1] += alpha_rgb[1] * d;
                optical_depth[2] += alpha_rgb[2] * d;

                // Early exit if fully absorbed on all channels.
                let max_od = optical_depth[0].max(optical_depth[1]).max(optical_depth[2]);
                if (-max_od).exp() < MIN_TRANSMITTANCE {
                    return None;
                }
            }
            None => {
                // Opaque hit.
                return Some(DdaAttenuatedHit {
                    hit: DdaHit {
                        x: ux,
                        y: uy,
                        z: uz,
                        face_axis: last_axis,
                        face_sign: match last_axis {
                            0 => -step_x as f32,
                            1 => -step_y as f32,
                            _ => -step_z as f32,
                        },
                        t: t_total,
                    },
                    transmittance: [
                        (-optical_depth[0]).exp(),
                        (-optical_depth[1]).exp(),
                        (-optical_depth[2]).exp(),
                    ],
                });
            }
        }

        prev_t = t_total;

        // Advance (Amanatides & Woo step).
        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                t_total = t_start + t_max_x;
                t_max_x += dt_x;
                ix += step_x;
                last_axis = 0;
            } else {
                t_total = t_start + t_max_z;
                t_max_z += dt_z;
                iz += step_z;
                last_axis = 2;
            }
        } else if t_max_y < t_max_z {
            t_total = t_start + t_max_y;
            t_max_y += dt_y;
            iy += step_y;
            last_axis = 1;
        } else {
            t_total = t_start + t_max_z;
            t_max_z += dt_z;
            iz += step_z;
            last_axis = 2;
        }

        if t_total - t_start > max_dist {
            return None;
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Refractive DDA ray march (Snell's law + Fresnel at material boundaries)
// ---------------------------------------------------------------------------

/// A single refracted/reflected path segment from a refractive ray march.
#[derive(Debug, Clone, PartialEq)]
pub struct RefractiveSegment {
    /// Grid-space origin of this segment.
    pub origin: [f32; 3],
    /// Unit direction of this segment.
    pub dir: [f32; 3],
    /// Refractive index of the medium for this segment.
    pub n: f32,
    /// Distance travelled in this segment (metres, since 1 voxel = 1 m).
    pub length: f32,
    /// Fresnel transmittance at the boundary that started this segment (0–1).
    /// 1.0 for the first segment (no prior boundary).
    pub transmittance: f32,
}

/// Result of a refractive DDA ray march.
#[derive(Debug, Clone, PartialEq)]
pub struct RefractiveMarchResult {
    /// Ordered list of path segments from origin to the terminal hit or
    /// the maximum distance.
    pub segments: Vec<RefractiveSegment>,
    /// The terminal opaque hit, if any.
    pub hit: Option<DdaHit>,
    /// Total accumulated reflectance loss (product of Fresnel transmittances).
    pub total_transmittance: f32,
}

/// March a ray through a voxel grid applying Snell's law refraction and
/// Fresnel reflection at material boundaries.
///
/// At each voxel crossing where the refractive index changes:
/// - Computes the surface normal from the DDA crossing axis.
/// - Applies Snell's law to compute the refracted direction.
/// - Computes the Fresnel transmittance for that interface.
/// - If total internal reflection occurs, the ray is reflected instead.
///
/// The march terminates when:
/// - An opaque voxel (no refractive index) is hit.
/// - `max_dist` total path length is exceeded.
/// - `max_bounces` TIR reflections have occurred.
///
/// # Arguments
/// - `get_n` — returns `Some(n)` if the material is refractive, `None` if opaque.
/// - `max_dist` — maximum total path length in metres.
/// - `max_bounces` — maximum number of TIR reflections before terminating.
pub fn dda_march_ray_refractive<F>(
    voxels: &[Voxel],
    size: usize,
    origin: [f32; 3],
    dir: [f32; 3],
    max_dist: f32,
    max_bounces: u32,
    get_n: F,
) -> RefractiveMarchResult
where
    F: Fn(MaterialId) -> Option<f32>,
{
    use crate::lighting::optics::{fresnel_reflectance, reflect_dir, snell_refract};

    let mut segments: Vec<RefractiveSegment> = Vec::new();
    let mut total_transmittance = 1.0_f32;
    let mut bounces = 0u32;
    let mut dist_remaining = max_dist;

    // Current ray state.
    let mut cur_origin = origin;
    let mut cur_dir = normalize3(dir);

    // Determine starting refractive index (the medium the ray begins in).
    let ox = cur_origin[0].floor() as i32;
    let oy = cur_origin[1].floor() as i32;
    let oz = cur_origin[2].floor() as i32;
    let start_n = if ox >= 0
        && oy >= 0
        && oz >= 0
        && (ox as usize) < size
        && (oy as usize) < size
        && (oz as usize) < size
    {
        let idx = (oz as usize) * size * size + (oy as usize) * size + (ox as usize);
        let mat = voxels[idx].material;
        get_n(mat).unwrap_or(1.0)
    } else {
        1.0 // Outside grid → air.
    };
    let mut cur_n = start_n;
    let mut seg_start = cur_origin;
    let mut seg_transmittance = 1.0_f32;

    loop {
        if dist_remaining <= 0.0 {
            // Flush the current in-progress segment.
            segments.push(RefractiveSegment {
                origin: seg_start,
                dir: cur_dir,
                n: cur_n,
                length: max_dist - dist_remaining,
                transmittance: seg_transmittance,
            });
            break;
        }

        // March one step with the DDA.
        let Some(hit) = dda_march_ray(voxels, size, cur_origin, cur_dir, dist_remaining) else {
            // No hit — ray escapes the grid.
            let seg_len = (max_dist - dist_remaining).min(dist_remaining);
            segments.push(RefractiveSegment {
                origin: seg_start,
                dir: cur_dir,
                n: cur_n,
                length: seg_len,
                transmittance: seg_transmittance,
            });
            break;
        };

        let new_mat = voxels[hit.z * size * size + hit.y * size + hit.x].material;
        let new_n_opt = get_n(new_mat);

        // Opaque surface — terminate.
        let Some(new_n) = new_n_opt else {
            let seg_len = hit.t;
            segments.push(RefractiveSegment {
                origin: seg_start,
                dir: cur_dir,
                n: cur_n,
                length: seg_len,
                transmittance: seg_transmittance,
            });
            return RefractiveMarchResult {
                segments,
                hit: Some(DdaHit {
                    x: hit.x,
                    y: hit.y,
                    z: hit.z,
                    face_axis: hit.face_axis,
                    face_sign: hit.face_sign,
                    t: hit.t,
                }),
                total_transmittance,
            };
        };

        // Refractive boundary — compute normal pointing out of the new medium.
        let face_normal = hit.face_normal();
        // The normal from DDA points in the direction the ray came from.
        // We need it pointing from new medium → old medium (away from the surface
        // the ray is entering), so flip it to point into the old medium.
        let normal_toward_incident = [
            -face_normal[0] * hit.face_sign,
            -face_normal[1] * hit.face_sign,
            -face_normal[2] * hit.face_sign,
        ];

        let cos_i = -dot3(cur_dir, normal_toward_incident).abs();
        let r = fresnel_reflectance(cos_i.abs(), cur_n, new_n);
        let t = 1.0 - r;

        // Finish the current segment up to the boundary.
        let boundary_pos = [
            cur_origin[0] + cur_dir[0] * hit.t,
            cur_origin[1] + cur_dir[1] * hit.t,
            cur_origin[2] + cur_dir[2] * hit.t,
        ];
        segments.push(RefractiveSegment {
            origin: seg_start,
            dir: cur_dir,
            n: cur_n,
            length: hit.t,
            transmittance: seg_transmittance,
        });
        dist_remaining -= hit.t;

        // Try to refract.
        if let Some(refracted_dir) =
            snell_refract(cur_dir, normal_toward_incident, cur_n, new_n)
        {
            // Successful refraction — continue with the refracted ray.
            total_transmittance *= t;
            cur_n = new_n;
            cur_dir = refracted_dir;
        } else {
            // Total internal reflection.
            if bounces >= max_bounces {
                break;
            }
            bounces += 1;
            total_transmittance *= r;
            cur_dir = reflect_dir(cur_dir, normal_toward_incident);
            // n stays the same after TIR.
        }

        cur_origin = [
            boundary_pos[0] + cur_dir[0] * 1e-4,
            boundary_pos[1] + cur_dir[1] * 1e-4,
            boundary_pos[2] + cur_dir[2] * 1e-4,
        ];
        seg_start = cur_origin;
        seg_transmittance = total_transmittance;
    }

    RefractiveMarchResult {
        segments,
        hit: None,
        total_transmittance,
    }
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return v;
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Ray-AABB intersection for a grid bounding box `[0, size]³`.
/// Returns `(t_near, t_far)`.
fn ray_aabb(origin: [f32; 3], dir: [f32; 3], size: f32) -> (f32, f32) {
    let inv = [
        if dir[0].abs() > 1e-10 {
            1.0 / dir[0]
        } else {
            f32::MAX.copysign(dir[0])
        },
        if dir[1].abs() > 1e-10 {
            1.0 / dir[1]
        } else {
            f32::MAX.copysign(dir[1])
        },
        if dir[2].abs() > 1e-10 {
            1.0 / dir[2]
        } else {
            f32::MAX.copysign(dir[2])
        },
    ];

    let t0x = -origin[0] * inv[0];
    let t1x = (size - origin[0]) * inv[0];
    let t0y = -origin[1] * inv[1];
    let t1y = (size - origin[1]) * inv[1];
    let t0z = -origin[2] * inv[2];
    let t1z = (size - origin[2]) * inv[2];

    let t_near = t0x.min(t1x).max(t0y.min(t1y)).max(t0z.min(t1z));
    let t_far = t0x.max(t1x).min(t0y.max(t1y)).min(t0z.max(t1z));
    (t_near, t_far)
}

/// Estimate surface normal from the voxel grid via central differences
/// of the "solidity" field (0 = air, 1 = solid).
pub fn estimate_surface_normal(
    voxels: &[Voxel],
    size: usize,
    x: usize,
    y: usize,
    z: usize,
) -> [f32; 3] {
    let s = |px: i32, py: i32, pz: i32| -> f32 {
        if px < 0 || py < 0 || pz < 0 {
            return 0.0;
        }
        let (ux, uy, uz) = (px as usize, py as usize, pz as usize);
        if ux >= size || uy >= size || uz >= size {
            return 0.0;
        }
        let idx = uz * size * size + uy * size + ux;
        if voxels[idx].material.is_air() {
            0.0
        } else {
            1.0
        }
    };

    let (ix, iy, iz) = (x as i32, y as i32, z as i32);
    let nx = s(ix - 1, iy, iz) - s(ix + 1, iy, iz);
    let ny = s(ix, iy - 1, iz) - s(ix, iy + 1, iz);
    let nz = s(ix, iy, iz - 1) - s(ix, iy, iz + 1);

    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-10 {
        [0.0, 1.0, 0.0]
    } else {
        [nx / len, ny / len, nz / len]
    }
}

/// Check if a point is in shadow (any opaque voxel between it and the light).
pub fn is_shadowed(voxels: &[Voxel], size: usize, pos: [f32; 3], to_light: [f32; 3]) -> bool {
    // Offset slightly to avoid self-intersection.
    let len =
        (to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2]).sqrt();
    if len < 1e-10 {
        return false;
    }
    let dir = [to_light[0] / len, to_light[1] / len, to_light[2] / len];
    let origin = [
        pos[0] + dir[0] * 0.5,
        pos[1] + dir[1] * 0.5,
        pos[2] + dir[2] * 0.5,
    ];
    dda_march_ray(voxels, size, origin, dir, size as f32 * 2.0).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::voxel::MaterialId;

    fn make_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    fn set_solid(voxels: &mut [Voxel], size: usize, x: usize, y: usize, z: usize) {
        let idx = z * size * size + y * size + x;
        voxels[idx].material = MaterialId::STONE;
    }

    #[test]
    fn ray_through_empty_grid_returns_none() {
        let grid = make_grid(8);
        for dir_idx in 0..26 {
            let hit = march_grid_ray(&grid, 8, [4, 4, 4], dir_idx, 8);
            assert!(hit.is_none(), "dir {dir_idx} should find nothing in air");
        }
    }

    #[test]
    fn ray_hits_adjacent_solid() {
        let mut grid = make_grid(8);
        set_solid(&mut grid, 8, 5, 4, 4);
        let hit = march_grid_ray(&grid, 8, [4, 4, 4], 0, 8); // +X
        let hit = hit.expect("should hit solid at (5,4,4)");
        assert_eq!((hit.x, hit.y, hit.z), (5, 4, 4));
        assert!((hit.distance - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ray_hits_distant_solid() {
        let mut grid = make_grid(16);
        set_solid(&mut grid, 16, 10, 4, 4);
        let hit = march_grid_ray(&grid, 16, [4, 4, 4], 0, 16); // +X
        let hit = hit.expect("should hit solid at (10,4,4)");
        assert_eq!(hit.x, 10);
        assert!((hit.distance - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ray_blocked_by_intermediate_solid() {
        let mut grid = make_grid(16);
        set_solid(&mut grid, 16, 7, 4, 4); // blocker
        set_solid(&mut grid, 16, 10, 4, 4); // target behind blocker
        let hit = march_grid_ray(&grid, 16, [4, 4, 4], 0, 16);
        let hit = hit.expect("should hit blocker, not target");
        assert_eq!(hit.x, 7, "should stop at blocker");
    }

    #[test]
    fn ray_stops_at_max_steps() {
        let mut grid = make_grid(32);
        set_solid(&mut grid, 32, 20, 4, 4);
        let hit = march_grid_ray(&grid, 32, [4, 4, 4], 0, 8); // max 8 steps
        assert!(hit.is_none(), "target at distance 16 exceeds max_steps=8");
    }

    #[test]
    fn diagonal_ray_distance_is_correct() {
        let mut grid = make_grid(16);
        set_solid(&mut grid, 16, 7, 7, 7); // 3 steps along (1,1,1) from (4,4,4)
        let hit = march_grid_ray(&grid, 16, [4, 4, 4], 18, 16); // dir 18 = (1,1,1)
        let hit = hit.expect("should hit diagonal solid");
        assert_eq!((hit.x, hit.y, hit.z), (7, 7, 7));
        let expected = 3.0 * 3.0_f32.sqrt();
        assert!(
            (hit.distance - expected).abs() < 0.01,
            "distance {:.3} should be {expected:.3}",
            hit.distance
        );
    }

    #[test]
    fn surface_voxel_detection() {
        let mut grid = make_grid(4);
        // Solid block surrounded by air
        set_solid(&mut grid, 4, 2, 2, 2);
        assert!(is_surface_voxel(&grid, 4, 2, 2, 2));

        // Fill all neighbors to make it interior
        for &(dx, dy, dz) in &[
            (1i32, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let nx = (2i32 + dx) as usize;
            let ny = (2i32 + dy) as usize;
            let nz = (2i32 + dz) as usize;
            set_solid(&mut grid, 4, nx, ny, nz);
        }
        assert!(
            !is_surface_voxel(&grid, 4, 2, 2, 2),
            "fully enclosed voxel is not a surface"
        );
    }

    #[test]
    fn boundary_voxel_is_surface() {
        let mut grid = make_grid(4);
        set_solid(&mut grid, 4, 0, 0, 0);
        assert!(
            is_surface_voxel(&grid, 4, 0, 0, 0),
            "corner voxel touches grid boundary"
        );
    }

    #[test]
    fn ray_directions_count() {
        assert_eq!(RAY_DIRECTIONS.len(), 26);
        assert_eq!(STEP_DISTANCES.len(), 26);
    }

    #[test]
    fn step_distances_are_correct() {
        for (i, dist) in STEP_DISTANCES.iter().enumerate() {
            let d = &RAY_DIRECTIONS[i];
            let expected = ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f32).sqrt();
            assert!(
                (dist - expected).abs() < 0.001,
                "dir {i}: step dist {dist} != expected {expected}"
            );
        }
    }

    // --- Attenuated ray march tests ---

    fn set_material(
        voxels: &mut [Voxel],
        size: usize,
        x: usize,
        y: usize,
        z: usize,
        mat: MaterialId,
    ) {
        voxels[z * size * size + y * size + x].material = mat;
    }

    /// Classify materials for attenuated tests:
    /// Air → Some(0.0), Water → Some(0.1), Stone → None (opaque).
    fn test_absorption(mat: MaterialId) -> Option<f32> {
        if mat.is_air() {
            Some(0.0)
        } else if mat == MaterialId::WATER {
            Some(0.1) // α = 0.1 m⁻¹ for test convenience
        } else {
            None // opaque
        }
    }

    #[test]
    fn attenuated_through_air_has_full_transmittance() {
        let mut grid = make_grid(8);
        set_solid(&mut grid, 8, 6, 4, 4);
        let result = march_grid_ray_attenuated(&grid, 8, [4, 4, 4], 0, 8, test_absorption);
        let result = result.expect("should hit stone");
        assert_eq!(result.hit.x, 6);
        assert!(
            (result.transmittance - 1.0).abs() < f32::EPSILON,
            "pure air path should have transmittance 1.0, got {}",
            result.transmittance
        );
    }

    #[test]
    fn attenuated_through_water_reduces_transmittance() {
        let mut grid = make_grid(8);
        // Water at x=5, stone target at x=6
        set_material(&mut grid, 8, 5, 4, 4, MaterialId::WATER);
        set_material(&mut grid, 8, 6, 4, 4, MaterialId::STONE);
        let result = march_grid_ray_attenuated(&grid, 8, [4, 4, 4], 0, 8, test_absorption);
        let result = result.expect("should hit stone through water");
        assert_eq!(result.hit.x, 6);
        // 1 voxel of water with α=0.1: transmittance = exp(-0.1) ≈ 0.905
        let expected = (-0.1_f32).exp();
        assert!(
            (result.transmittance - expected).abs() < 1e-6,
            "transmittance {:.6} should be {expected:.6}",
            result.transmittance
        );
    }

    #[test]
    fn attenuated_multiple_water_voxels() {
        let mut grid = make_grid(10);
        // 3 voxels of water (x=5,6,7), stone target at x=8
        for x in 5..=7 {
            set_material(&mut grid, 10, x, 4, 4, MaterialId::WATER);
        }
        set_material(&mut grid, 10, 8, 4, 4, MaterialId::STONE);
        let result = march_grid_ray_attenuated(&grid, 10, [4, 4, 4], 0, 10, test_absorption);
        let result = result.expect("should hit stone through water");
        assert_eq!(result.hit.x, 8);
        // 3 voxels × α=0.1 × d=1: transmittance = exp(-0.3) ≈ 0.741
        let expected = (-0.3_f32).exp();
        assert!(
            (result.transmittance - expected).abs() < 1e-5,
            "transmittance {:.6} should be {expected:.6}",
            result.transmittance
        );
    }

    #[test]
    fn attenuated_fully_absorbed_returns_none() {
        let mut grid = make_grid(16);
        // Use high α to ensure full absorption
        let high_absorption = |mat: MaterialId| -> Option<f32> {
            if mat.is_air() {
                Some(0.0)
            } else if mat == MaterialId::WATER {
                Some(10.0) // α = 10 m⁻¹: exp(-10) ≈ 4.5e-5 < MIN_TRANSMITTANCE
            } else {
                None
            }
        };
        // Fill x=5..12 with water (8 voxels), stone target at x=13
        for x in 5..=12 {
            set_material(&mut grid, 16, x, 4, 4, MaterialId::WATER);
        }
        set_material(&mut grid, 16, 13, 4, 4, MaterialId::STONE);
        let result = march_grid_ray_attenuated(&grid, 16, [4, 4, 4], 0, 16, high_absorption);
        assert!(
            result.is_none(),
            "thick water layer with high α should fully absorb the ray"
        );
    }

    #[test]
    fn attenuated_opaque_stops_before_transparent() {
        let mut grid = make_grid(8);
        // Stone at x=5 blocks, water at x=6 is behind it
        set_material(&mut grid, 8, 5, 4, 4, MaterialId::STONE);
        set_material(&mut grid, 8, 6, 4, 4, MaterialId::WATER);
        let result = march_grid_ray_attenuated(&grid, 8, [4, 4, 4], 0, 8, test_absorption);
        let result = result.expect("should hit stone");
        assert_eq!(result.hit.x, 5);
        assert!(
            (result.transmittance - 1.0).abs() < f32::EPSILON,
            "no semi-transparent material before stone"
        );
    }

    #[test]
    fn attenuated_low_alpha_preserves_most_radiation() {
        let mut grid = make_grid(8);
        set_material(&mut grid, 8, 5, 4, 4, MaterialId::WATER);
        set_material(&mut grid, 8, 6, 4, 4, MaterialId::STONE);
        // Use very low absorption coefficient
        let low_absorption = |mat: MaterialId| -> Option<f32> {
            if mat.is_air() {
                Some(0.0)
            } else if mat == MaterialId::WATER {
                Some(0.01) // nearly transparent
            } else {
                None
            }
        };
        let result = march_grid_ray_attenuated(&grid, 8, [4, 4, 4], 0, 8, low_absorption);
        let result = result.expect("should hit stone");
        let expected = (-0.01_f32).exp(); // ≈ 0.99
        assert!(
            result.transmittance > 0.98,
            "low α should preserve most radiation: {:.4}",
            result.transmittance
        );
        assert!(
            (result.transmittance - expected).abs() < 1e-5,
            "transmittance {:.6} should be {expected:.6}",
            result.transmittance
        );
    }

    // --- Arbitrary DDA raymarcher tests ---

    #[test]
    fn dda_ray_hits_adjacent_solid() {
        let mut grid = make_grid(8);
        set_solid(&mut grid, 8, 5, 4, 4);
        let hit = dda_march_ray(&grid, 8, [4.5, 4.5, 4.5], [1.0, 0.0, 0.0], 10.0);
        let hit = hit.expect("should hit solid at (5,4,4)");
        assert_eq!((hit.x, hit.y, hit.z), (5, 4, 4));
        assert_eq!(hit.face_axis, 0, "face should be X axis");
        assert!(hit.face_sign < 0.0, "face should point toward -X");
    }

    #[test]
    fn dda_ray_diagonal_hit() {
        let mut grid = make_grid(16);
        set_solid(&mut grid, 16, 8, 8, 4);
        let hit = dda_march_ray(&grid, 16, [4.5, 4.5, 4.5], [1.0, 1.0, 0.0], 20.0);
        let hit = hit.expect("should hit solid diagonally");
        assert_eq!((hit.x, hit.y, hit.z), (8, 8, 4));
    }

    #[test]
    fn dda_ray_from_outside_grid() {
        let mut grid = make_grid(8);
        set_solid(&mut grid, 8, 0, 4, 4);
        let hit = dda_march_ray(&grid, 8, [-5.0, 4.5, 4.5], [1.0, 0.0, 0.0], 20.0);
        let hit = hit.expect("should hit solid entering from outside");
        assert_eq!((hit.x, hit.y, hit.z), (0, 4, 4));
    }

    #[test]
    fn dda_ray_misses_grid() {
        let grid = make_grid(8);
        let hit = dda_march_ray(&grid, 8, [4.5, 4.5, 4.5], [0.0, 1.0, 0.0], 20.0);
        assert!(hit.is_none(), "should find nothing in empty grid");
    }

    #[test]
    fn dda_ray_max_dist_limit() {
        let mut grid = make_grid(32);
        set_solid(&mut grid, 32, 20, 4, 4);
        let hit = dda_march_ray(&grid, 32, [4.5, 4.5, 4.5], [1.0, 0.0, 0.0], 5.0);
        assert!(hit.is_none(), "target beyond max_dist should not be found");
    }

    #[test]
    fn dda_face_normal_correctness() {
        let hit = DdaHit {
            x: 5,
            y: 4,
            z: 4,
            face_axis: 0,
            face_sign: -1.0,
            t: 1.0,
        };
        assert_eq!(hit.face_normal(), [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn dda_attenuated_through_air_full_transmittance() {
        let mut grid = make_grid(8);
        set_solid(&mut grid, 8, 6, 4, 4);
        let absorption = |mat: MaterialId| -> Option<[f32; 3]> {
            if mat.is_air() { Some([0.0; 3]) } else { None }
        };
        let result =
            dda_march_ray_attenuated(&grid, 8, [4.5, 4.5, 4.5], [1.0, 0.0, 0.0], 10.0, absorption);
        let result = result.expect("should hit stone");
        assert_eq!(result.hit.x, 6);
        for ch in 0..3 {
            assert!(
                (result.transmittance[ch] - 1.0).abs() < 0.01,
                "air should have full transmittance on channel {ch}"
            );
        }
    }

    #[test]
    fn dda_attenuated_colored_absorption() {
        let mut grid = make_grid(10);
        // 2 voxels of water (x=5,6), stone target at x=7
        for x in 5..=6 {
            set_material(&mut grid, 10, x, 4, 4, MaterialId::WATER);
        }
        set_material(&mut grid, 10, 7, 4, 4, MaterialId::STONE);

        // Water absorbs red more than blue (like real water).
        let absorption = |mat: MaterialId| -> Option<[f32; 3]> {
            if mat.is_air() {
                Some([0.0; 3])
            } else if mat == MaterialId::WATER {
                Some([0.5, 0.1, 0.05]) // Red absorbed most
            } else {
                None
            }
        };

        let result = dda_march_ray_attenuated(
            &grid,
            10,
            [4.5, 4.5, 4.5],
            [1.0, 0.0, 0.0],
            15.0,
            absorption,
        );
        let result = result.expect("should hit stone through water");
        assert_eq!(result.hit.x, 7);
        // Blue should be most transmitted, red least.
        assert!(
            result.transmittance[2] > result.transmittance[0],
            "Blue transmittance ({}) should exceed red ({})",
            result.transmittance[2],
            result.transmittance[0]
        );
    }

    #[test]
    fn is_shadowed_detects_blocker() {
        let mut grid = make_grid(16);
        set_solid(&mut grid, 16, 8, 10, 8); // blocker above
        assert!(
            is_shadowed(&grid, 16, [8.5, 5.5, 8.5], [0.0, 1.0, 0.0]),
            "voxel above should cast shadow"
        );
    }

    #[test]
    fn is_shadowed_open_sky() {
        let grid = make_grid(16);
        assert!(
            !is_shadowed(&grid, 16, [8.5, 5.5, 8.5], [0.0, 1.0, 0.0]),
            "empty grid should not shadow"
        );
    }

    #[test]
    fn estimate_surface_normal_top_face() {
        let mut grid = make_grid(8);
        // Solid block at (4,4,4) with air above → normal should point up.
        set_solid(&mut grid, 8, 4, 4, 4);
        let n = estimate_surface_normal(&grid, 8, 4, 4, 4);
        // Y component should be dominant (pointing up from isolated cube).
        // All faces exposed, so normal averages to something; check it's nonzero.
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 0.01,
            "Normal should be unit length, got {len}"
        );
    }

    // -----------------------------------------------------------------------
    // Refractive DDA tests
    // -----------------------------------------------------------------------

    /// Helper: make a grid where a centre slab [0..size, y_lo..y_hi, 0..size]
    /// is filled with a "glass" material (ID 12, n=1.52) and everything else
    /// is air (ID 0, n=1.0).
    fn make_refractive_slab(size: usize, y_lo: usize, y_hi: usize) -> Vec<Voxel> {
        let mut grid = vec![Voxel::default(); size * size * size];
        for z in 0..size {
            for y in y_lo..y_hi {
                for x in 0..size {
                    let idx = z * size * size + y * size + x;
                    grid[idx].material = MaterialId::GLASS;
                    grid[idx].density = 1.0;
                }
            }
        }
        grid
    }

    fn refractive_n(mat: MaterialId) -> Option<f32> {
        match mat {
            MaterialId::AIR => Some(1.0),
            MaterialId::GLASS => Some(1.52),
            MaterialId::WATER => Some(1.33),
            _ => None, // Opaque
        }
    }

    #[test]
    fn refractive_march_straight_through_homogeneous() {
        // Uniform air grid — no refraction, ray travels in a straight line.
        let size = 16;
        let grid = make_grid(size);
        let origin = [8.0, 8.0, 0.5];
        let dir = [0.0, 0.0, 1.0];
        let result =
            dda_march_ray_refractive(&grid, size, origin, dir, 10.0, 4, |m| refractive_n(m));
        // Should produce segments with the same direction throughout.
        assert!(!result.segments.is_empty());
        for seg in &result.segments {
            assert!(
                (seg.dir[2] - 1.0).abs() < 1e-4,
                "direction should not change in air"
            );
        }
    }

    #[test]
    fn refractive_march_normal_incidence_no_deflection() {
        // Normal incidence (ray perpendicular to slab surface) — Snell's law
        // gives no angular deflection, only a change of n.
        let size = 16;
        let grid = make_refractive_slab(size, 6, 10);
        let origin = [8.0, 0.5, 8.0];
        let dir = [0.0, 1.0, 0.0]; // Straight up into slab.
        let result =
            dda_march_ray_refractive(&grid, size, origin, dir, 15.0, 4, |m| refractive_n(m));
        // All segments should point in the +Y direction.
        for seg in &result.segments {
            assert!(
                seg.dir[1] > 0.99,
                "normal incidence — no lateral deflection expected, got {:?}",
                seg.dir
            );
        }
        // Transmittance should be close to 1 (glass Fresnel R ≈ 4% per surface).
        assert!(
            result.total_transmittance > 0.9,
            "expected high transmittance through glass slab, got {}",
            result.total_transmittance
        );
    }

    #[test]
    fn refractive_march_oblique_changes_direction() {
        // Oblique incidence — refracted ray must change direction inside slab.
        let size = 16;
        let grid = make_refractive_slab(size, 6, 10);
        // 45° from above.
        let d = 1.0_f32 / 2.0_f32.sqrt();
        let dir = [0.0, d, d];
        let origin = [8.0, 0.5, 8.0];
        let result =
            dda_march_ray_refractive(&grid, size, origin, dir, 20.0, 4, |m| refractive_n(m));
        // Find a segment inside the glass (n ≈ 1.52).
        let glass_segs: Vec<_> = result.segments.iter().filter(|s| s.n > 1.4).collect();
        // Must have at least one glass segment.
        assert!(
            !glass_segs.is_empty(),
            "expected at least one segment inside glass"
        );
        // The z-component inside glass should be smaller than the incident z-component
        // (ray bends toward the normal, so the angle with the normal decreases).
        let glass_z = glass_segs[0].dir[2];
        assert!(
            glass_z < d + 0.01,
            "refracted ray should bend toward normal in denser medium; z={glass_z} vs incident z={d}"
        );
    }

    #[test]
    fn refractive_march_tir_stays_inside() {
        // Supercritical angle from inside glass → TIR, ray stays inside.
        // Critical angle for glass→air: arcsin(1/1.52) ≈ 41.1°
        // Use an angle of 60° from the normal (well above 41°).
        let size = 32;
        let grid = make_refractive_slab(size, 0, 32); // Entire grid is glass.
        let d = (60.0_f32.to_radians()).sin();
        let dir = [d, (60.0_f32.to_radians()).cos(), 0.0]; // 60° from Y-normal.
        let origin = [16.0, 16.0, 16.0];
        let result =
            dda_march_ray_refractive(&grid, size, origin, dir, 10.0, 4, |m| refractive_n(m));
        // With TIR all segments should remain at n ≈ 1.52.
        for seg in &result.segments {
            assert!(
                seg.n > 1.4,
                "TIR — ray must stay in glass, but seg.n={}",
                seg.n
            );
        }
    }
}
