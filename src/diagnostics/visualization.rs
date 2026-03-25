// Frame renderer: converts a voxel grid slice into an RGB image buffer.
//
// Designed for headless use in the simulation test harness. Each frame
// represents one simulation tick and can be piped to ffmpeg for video
// encoding or saved as individual PNGs.
//
// View modes:
// - Slice: 2D cross-section at a fixed depth along an axis
// - TopDown: orthographic Y-axis raycast (first opaque voxel wins)
// - Perspective: 3D perspective camera with DDA raymarching, Lambertian
//   shading, and shadow casting for realistic terrain rendering

use image::{Rgb, RgbImage};
use serde::Deserialize;

use crate::data::MaterialRegistry;
use crate::world::voxel::Voxel;

/// Which axis to slice along when using `ViewMode::Slice`.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceAxis {
    /// Slice at a fixed X, showing the YZ plane.
    X,
    /// Slice at a fixed Y, showing the XZ plane (top-down).
    #[default]
    Y,
    /// Slice at a fixed Z, showing the XY plane.
    Z,
}

/// How to project the 3D grid into a 2D frame.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub enum ViewMode {
    /// A single cross-section through the grid.
    Slice {
        axis: SliceAxis,
        /// Depth index along the slice axis. Clamped to `[0, grid_size)`.
        depth: usize,
    },
    /// Raycast down the Y axis; the first opaque voxel determines the color.
    TopDown,
    /// 3D perspective camera with lighting and shadows.
    Perspective {
        /// Camera position in voxel-space coordinates.
        eye: (f32, f32, f32),
        /// Look-at target in voxel-space coordinates.
        target: (f32, f32, f32),
        /// Vertical field of view in degrees.
        fov_degrees: f32,
        /// Output image width in pixels.
        width: u32,
        /// Output image height in pixels.
        height: u32,
    },
}

/// Directional light for perspective rendering.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct SceneLight {
    /// Normalized direction FROM the light source (points away from the light).
    /// For a sun at upper-left: e.g. `(-0.5, -0.8, -0.3)` (auto-normalized).
    pub direction: (f32, f32, f32),
    /// Light color (RGB, 0.0–1.0).
    pub color: (f32, f32, f32),
    /// Intensity multiplier (1.0 = full strength).
    pub intensity: f32,
    /// Ambient light floor (0.0–1.0). Prevents pure-black shadows.
    pub ambient: f32,
}

impl Default for SceneLight {
    fn default() -> Self {
        Self {
            direction: (-0.4, -0.8, -0.3),
            color: (1.0, 1.0, 0.95),
            intensity: 1.0,
            ambient: 0.15,
        }
    }
}

impl Default for ViewMode {
    fn default() -> Self {
        Self::Slice {
            axis: SliceAxis::default(),
            depth: 0,
        }
    }
}

/// What physical quantity determines voxel color.
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub enum ColorMode {
    /// Use `MaterialData.color` directly.
    #[default]
    Material,
    /// Blue (cold) → Red (hot) heatmap.
    Temperature {
        /// Lower bound in Kelvin (maps to blue).
        min_k: f32,
        /// Upper bound in Kelvin (maps to red).
        max_k: f32,
    },
    /// Green (low) → Yellow (high) pressure map.
    Pressure {
        /// Lower bound in Pascals.
        min_pa: f32,
        /// Upper bound in Pascals.
        max_pa: f32,
    },
    /// Material color blended with incandescent glow above 800 K.
    /// Mirrors the in-game thermal glow ramp: dark red → cherry →
    /// orange → yellow-white. HDR emissive values are tone-mapped
    /// to [0, 255] for the output image.
    Incandescence,
}

/// Map a normalized value in `[0, 1]` to a blue→cyan→green→yellow→red gradient.
fn heatmap_rgb(t: f32) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    // Five-stop gradient: blue → cyan → green → yellow → red
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

/// Resolve a single voxel to an RGB pixel based on the color mode.
fn voxel_color(voxel: &Voxel, registry: &MaterialRegistry, color_mode: &ColorMode) -> Rgb<u8> {
    if voxel.material.is_air() {
        return Rgb([0, 0, 0]);
    }

    match color_mode {
        ColorMode::Material => {
            if let Some(mat) = registry.get(voxel.material) {
                Rgb([
                    (mat.color[0] * 255.0) as u8,
                    (mat.color[1] * 255.0) as u8,
                    (mat.color[2] * 255.0) as u8,
                ])
            } else {
                Rgb([255, 0, 255]) // magenta = unknown material
            }
        }
        ColorMode::Temperature { min_k, max_k } => {
            let range = max_k - min_k;
            let t = if range > 0.0 {
                (voxel.temperature - min_k) / range
            } else {
                0.5
            };
            heatmap_rgb(t)
        }
        ColorMode::Pressure { min_pa, max_pa } => {
            let range = max_pa - min_pa;
            let t = if range > 0.0 {
                (voxel.pressure - min_pa) / range
            } else {
                0.5
            };
            heatmap_rgb(t)
        }
        ColorMode::Incandescence => {
            // Base color from material data
            let base = if let Some(mat) = registry.get(voxel.material) {
                [mat.color[0], mat.color[1], mat.color[2], 1.0]
            } else {
                [0.8, 0.0, 0.8, 1.0]
            };
            let hdr = crate::world::meshing::incandescence_color(base, voxel.temperature);
            // Tone-map HDR → LDR via Reinhard
            fn tonemap(v: f32) -> u8 {
                ((v / (1.0 + v)) * 255.0).min(255.0) as u8
            }
            Rgb([tonemap(hdr[0]), tonemap(hdr[1]), tonemap(hdr[2])])
        }
    }
}

/// Index into a flat `size³` voxel array.
fn idx(x: usize, y: usize, z: usize, size: usize) -> usize {
    x + z * size + y * size * size
}

// ---------------------------------------------------------------------------
// DDA Voxel Raymarcher (arbitrary direction)
// ---------------------------------------------------------------------------

/// A simple 3D vector for the software renderer (avoids bevy dep in this module).
#[derive(Debug, Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::new(0.0, 1.0, 0.0);
        }
        Self::new(self.x / len, self.y / len, self.z / len)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

/// Result of a DDA ray-voxel intersection.
struct DdaHit {
    /// Voxel coordinates of the hit.
    x: usize,
    y: usize,
    z: usize,
    /// Which axis the ray entered through (0=X, 1=Y, 2=Z).
    face_axis: usize,
    /// Sign of the entry direction along face_axis (+1 or -1).
    face_sign: f32,
    /// Distance from ray origin to the hit.
    t: f32,
}

/// DDA (Digital Differential Analyzer) raymarcher through a voxel grid.
/// Returns the first non-air voxel hit, or `None` if the ray exits the grid.
fn dda_march(
    origin: Vec3,
    dir: Vec3,
    voxels: &[Voxel],
    size: usize,
    max_dist: f32,
) -> Option<DdaHit> {
    let dir = dir.normalized();
    let fs = size as f32;

    // Advance ray to grid AABB entry if starting outside.
    let (t_enter, t_exit) = ray_aabb(origin, dir, fs);
    if t_enter >= t_exit || t_exit < 0.0 {
        return None;
    }

    let t_start = t_enter.max(0.0) + 0.001;
    let pos = origin.add(dir.scale(t_start));

    // Current voxel coordinates (clamped into grid).
    let mut ix = (pos.x.floor() as i32).clamp(0, size as i32 - 1);
    let mut iy = (pos.y.floor() as i32).clamp(0, size as i32 - 1);
    let mut iz = (pos.z.floor() as i32).clamp(0, size as i32 - 1);

    // Step direction (+1 or -1) along each axis.
    let step_x: i32 = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir.z >= 0.0 { 1 } else { -1 };

    // Distance along ray to cross one full voxel on each axis.
    let dt_x = if dir.x.abs() > 1e-10 {
        (1.0 / dir.x).abs()
    } else {
        f32::MAX
    };
    let dt_y = if dir.y.abs() > 1e-10 {
        (1.0 / dir.y).abs()
    } else {
        f32::MAX
    };
    let dt_z = if dir.z.abs() > 1e-10 {
        (1.0 / dir.z).abs()
    } else {
        f32::MAX
    };

    // Distance to the next voxel boundary on each axis.
    let mut t_max_x = if dir.x >= 0.0 {
        ((ix + 1) as f32 - pos.x) * dt_x
    } else {
        (pos.x - ix as f32) * dt_x
    };
    let mut t_max_y = if dir.y >= 0.0 {
        ((iy + 1) as f32 - pos.y) * dt_y
    } else {
        (pos.y - iy as f32) * dt_y
    };
    let mut t_max_z = if dir.z >= 0.0 {
        ((iz + 1) as f32 - pos.z) * dt_z
    } else {
        (pos.z - iz as f32) * dt_z
    };

    let mut t_total = t_start;
    let mut last_axis = 1_usize; // Y by default

    let max_steps = (size * 3) as u32;
    for _ in 0..max_steps {
        // Check bounds
        if ix < 0 || iy < 0 || iz < 0 {
            return None;
        }
        let ux = ix as usize;
        let uy = iy as usize;
        let uz = iz as usize;
        if ux >= size || uy >= size || uz >= size {
            return None;
        }

        // Hit test
        let voxel = &voxels[idx(ux, uy, uz, size)];
        if !voxel.material.is_air() {
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

        // Advance to next voxel boundary (Amanatides & Woo).
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

/// Ray-AABB intersection for the grid bounding box [0, size]³.
fn ray_aabb(origin: Vec3, dir: Vec3, size: f32) -> (f32, f32) {
    let inv = Vec3::new(
        if dir.x.abs() > 1e-10 {
            1.0 / dir.x
        } else {
            f32::MAX.copysign(dir.x)
        },
        if dir.y.abs() > 1e-10 {
            1.0 / dir.y
        } else {
            f32::MAX.copysign(dir.y)
        },
        if dir.z.abs() > 1e-10 {
            1.0 / dir.z
        } else {
            f32::MAX.copysign(dir.z)
        },
    );

    let t0x = (0.0 - origin.x) * inv.x;
    let t1x = (size - origin.x) * inv.x;
    let t0y = (0.0 - origin.y) * inv.y;
    let t1y = (size - origin.y) * inv.y;
    let t0z = (0.0 - origin.z) * inv.z;
    let t1z = (size - origin.z) * inv.z;

    let t_near = t0x.min(t1x).max(t0y.min(t1y)).max(t0z.min(t1z));
    let t_far = t0x.max(t1x).min(t0y.max(t1y)).min(t0z.max(t1z));
    (t_near, t_far)
}

/// Estimate surface normal from the voxel grid via central differences.
/// Returns the gradient of the "solidity" field (0=air, 1=solid).
fn estimate_normal(x: usize, y: usize, z: usize, voxels: &[Voxel], size: usize) -> Vec3 {
    let s = |px: i32, py: i32, pz: i32| -> f32 {
        if px < 0 || py < 0 || pz < 0 {
            return 0.0;
        }
        let (ux, uy, uz) = (px as usize, py as usize, pz as usize);
        if ux >= size || uy >= size || uz >= size {
            return 0.0;
        }
        if voxels[idx(ux, uy, uz, size)].material.is_air() {
            0.0
        } else {
            1.0
        }
    };

    let (ix, iy, iz) = (x as i32, y as i32, z as i32);
    let nx = s(ix - 1, iy, iz) - s(ix + 1, iy, iz);
    let ny = s(ix, iy - 1, iz) - s(ix, iy + 1, iz);
    let nz = s(ix, iy, iz - 1) - s(ix, iy, iz + 1);

    Vec3::new(nx, ny, nz).normalized()
}

/// Check if a point is in shadow by marching toward the light.
fn is_shadowed(pos: Vec3, light_dir: Vec3, voxels: &[Voxel], size: usize) -> bool {
    // March from hit point toward the light (opposite of light direction).
    let to_light = light_dir.scale(-1.0);
    // Offset origin slightly to avoid self-intersection.
    let origin = pos.add(to_light.scale(0.5));
    dda_march(origin, to_light, voxels, size, size as f32 * 2.0).is_some()
}

/// Compute the sky/background color via Rayleigh scattering.
fn sky_color(ray_dir: Vec3, light_dir: Vec3) -> Rgb<u8> {
    use crate::lighting::sky;

    let view = [ray_dir.x, ray_dir.y, ray_dir.z];
    // light_dir points toward the surface; sun_dir points toward the sun.
    let sun_dir = sky::normalize([-light_dir.x, -light_dir.y, -light_dir.z]);

    let hdr = sky::sky_color(view, sun_dir);

    // Add sun disk and glow on top of the atmospheric color.
    let sun_dot = ray_dir.dot(light_dir.scale(-1.0).normalized()).max(0.0);
    let (hdr_r, hdr_g, hdr_b) = if sun_dot > 0.995 {
        // Sun disk — bright white-yellow
        (hdr[0] + 5.0, hdr[1] + 4.5, hdr[2] + 3.0)
    } else if sun_dot > 0.98 {
        // Sun glow corona
        let glow = (sun_dot - 0.98) / 0.015;
        (
            hdr[0] + 2.0 * glow,
            hdr[1] + 1.8 * glow,
            hdr[2] + 1.2 * glow,
        )
    } else {
        (hdr[0], hdr[1], hdr[2])
    };

    let srgb = sky::tonemap_to_srgb([hdr_r, hdr_g, hdr_b]);
    Rgb(srgb)
}

// ---------------------------------------------------------------------------
// Perspective rendering
// ---------------------------------------------------------------------------

/// Camera parameters for perspective rendering.
struct PerspectiveCamera {
    eye: Vec3,
    target: Vec3,
    fov_degrees: f32,
    width: u32,
    height: u32,
}

/// Render a perspective view with Lambertian shading and shadows.
fn render_perspective(
    voxels: &[Voxel],
    size: usize,
    registry: &MaterialRegistry,
    color_mode: &ColorMode,
    cam: &PerspectiveCamera,
    light: &SceneLight,
) -> RgbImage {
    let mut img = RgbImage::new(cam.width, cam.height);

    // Camera basis vectors
    let forward = cam.target.sub(cam.eye).normalized();
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = forward.cross(world_up).normalized();
    let up = right.cross(forward).normalized();

    let fov_rad = cam.fov_degrees.to_radians();
    let half_h = (fov_rad / 2.0).tan();
    let aspect = cam.width as f32 / cam.height as f32;
    let half_w = half_h * aspect;

    // Normalized light direction (pointing FROM the light toward the scene).
    let light_dir = Vec3::new(light.direction.0, light.direction.1, light.direction.2).normalized();
    let light_col = (light.color.0, light.color.1, light.color.2);
    let max_march = (size as f32) * 3.0;

    for py in 0..cam.height {
        for px in 0..cam.width {
            // Normalized device coordinates [-1, 1]
            let u = (2.0 * px as f32 / cam.width as f32 - 1.0) * half_w;
            let v = (1.0 - 2.0 * py as f32 / cam.height as f32) * half_h;

            // Ray direction
            let dir = forward.add(right.scale(u)).add(up.scale(v)).normalized();

            // March the ray
            let pixel = if let Some(hit) = dda_march(cam.eye, dir, voxels, size, max_march) {
                let voxel = &voxels[idx(hit.x, hit.y, hit.z, size)];
                let base = voxel_color(voxel, registry, color_mode);

                // Surface normal from face axis or gradient estimate.
                let normal = if hit.face_axis == 0 {
                    Vec3::new(hit.face_sign, 0.0, 0.0)
                } else if hit.face_axis == 1 {
                    Vec3::new(0.0, hit.face_sign, 0.0)
                } else {
                    Vec3::new(0.0, 0.0, hit.face_sign)
                };

                // Smooth the normal using gradient for non-flat surfaces.
                let grad = estimate_normal(hit.x, hit.y, hit.z, voxels, size);
                let blended = normal.scale(0.5).add(grad.scale(0.5)).normalized();

                // Lambertian diffuse: max(0, -light_dir · normal)
                let n_dot_l = blended.dot(light_dir.scale(-1.0)).max(0.0);

                // Shadow test
                let hit_pos = Vec3::new(hit.x as f32 + 0.5, hit.y as f32 + 0.5, hit.z as f32 + 0.5);
                let in_shadow = is_shadowed(hit_pos, light_dir, voxels, size);

                let shadow_factor = if in_shadow { 0.0 } else { 1.0 };

                // Final lighting: ambient uses neutral white, directional uses
                // the light color so dawn/dusk warmth only tints direct light.
                let diffuse = n_dot_l * light.intensity * shadow_factor;

                // Depth fog (gentle fade toward background at far distances)
                let fog_start = size as f32 * 0.5;
                let fog_end = size as f32 * 2.5;
                let fog = ((hit.t - fog_start) / (fog_end - fog_start)).clamp(0.0, 1.0);

                let bg = sky_color(dir, light_dir);

                let r = (base.0[0] as f32 * (light.ambient + diffuse * light_col.0)).min(255.0);
                let g = (base.0[1] as f32 * (light.ambient + diffuse * light_col.1)).min(255.0);
                let b = (base.0[2] as f32 * (light.ambient + diffuse * light_col.2)).min(255.0);

                Rgb([
                    (r * (1.0 - fog) + bg.0[0] as f32 * fog) as u8,
                    (g * (1.0 - fog) + bg.0[1] as f32 * fog) as u8,
                    (b * (1.0 - fog) + bg.0[2] as f32 * fog) as u8,
                ])
            } else {
                sky_color(dir, light_dir)
            };

            img.put_pixel(px, py, pixel);
        }
    }

    img
}

/// Render a single frame from the voxel grid.
///
/// For `Slice` and `TopDown` modes the image is `(grid_size * scale)²`.
/// For `Perspective` mode the image is `(grid_size * scale, grid_size * scale)`
/// by default but uses proper 3D perspective projection with lighting.
pub fn render_frame(
    voxels: &[Voxel],
    size: usize,
    registry: &MaterialRegistry,
    view: &ViewMode,
    color_mode: &ColorMode,
    scale: u32,
) -> RgbImage {
    render_frame_lit(
        voxels,
        size,
        registry,
        view,
        color_mode,
        scale,
        &SceneLight::default(),
    )
}

/// Render a single frame with explicit lighting parameters.
///
/// The `light` parameter only affects `ViewMode::Perspective`; 2D modes ignore it.
pub fn render_frame_lit(
    voxels: &[Voxel],
    size: usize,
    registry: &MaterialRegistry,
    view: &ViewMode,
    color_mode: &ColorMode,
    scale: u32,
    light: &SceneLight,
) -> RgbImage {
    let dim = (size as u32) * scale;

    match view {
        ViewMode::Slice { axis, depth } => {
            let mut img = RgbImage::new(dim, dim);
            let d = (*depth).min(size.saturating_sub(1));
            for row in 0..size {
                for col in 0..size {
                    let (x, y, z) = match axis {
                        SliceAxis::X => (d, row, col),
                        SliceAxis::Y => (col, d, row),
                        SliceAxis::Z => (col, row, d),
                    };
                    let voxel = &voxels[idx(x, y, z, size)];
                    let color = voxel_color(voxel, registry, color_mode);

                    for sy in 0..scale {
                        for sx in 0..scale {
                            img.put_pixel(col as u32 * scale + sx, row as u32 * scale + sy, color);
                        }
                    }
                }
            }
            img
        }
        ViewMode::TopDown => {
            let mut img = RgbImage::new(dim, dim);
            for z in 0..size {
                for x in 0..size {
                    let mut color = Rgb([0, 0, 0]);
                    for y in (0..size).rev() {
                        let voxel = &voxels[idx(x, y, z, size)];
                        if !voxel.material.is_air() {
                            color = voxel_color(voxel, registry, color_mode);
                            break;
                        }
                    }
                    for sy in 0..scale {
                        for sx in 0..scale {
                            img.put_pixel(x as u32 * scale + sx, z as u32 * scale + sy, color);
                        }
                    }
                }
            }
            img
        }
        ViewMode::Perspective {
            eye,
            target,
            fov_degrees,
            width,
            height,
        } => render_perspective(
            voxels,
            size,
            registry,
            color_mode,
            &PerspectiveCamera {
                eye: Vec3::new(eye.0, eye.1, eye.2),
                target: Vec3::new(target.0, target.1, target.2),
                fov_degrees: *fov_degrees,
                width: *width,
                height: *height,
            },
            light,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MaterialData;
    use crate::world::voxel::MaterialId;

    fn test_registry() -> MaterialRegistry {
        let mut reg = MaterialRegistry::new();
        reg.insert(MaterialData {
            id: 0,
            name: "Air".into(),
            color: [0.0, 0.0, 0.0],
            transparent: true,
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 1,
            name: "Stone".into(),
            color: [0.5, 0.5, 0.5],
            ..Default::default()
        });
        reg.insert(MaterialData {
            id: 3,
            name: "Water".into(),
            color: [0.0, 0.3, 0.8],
            transparent: true,
            ..Default::default()
        });
        reg
    }

    fn make_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    #[test]
    fn render_slice_correct_dimensions() {
        let size = 4;
        let scale = 2;
        let grid = make_grid(size);
        let reg = test_registry();
        let view = ViewMode::Slice {
            axis: SliceAxis::Y,
            depth: 0,
        };
        let img = render_frame(&grid, size, &reg, &view, &ColorMode::Material, scale);
        assert_eq!(img.width(), (size as u32) * scale);
        assert_eq!(img.height(), (size as u32) * scale);
    }

    #[test]
    fn render_topdown_correct_dimensions() {
        let size = 8;
        let scale = 1;
        let grid = make_grid(size);
        let reg = test_registry();
        let img = render_frame(
            &grid,
            size,
            &reg,
            &ViewMode::TopDown,
            &ColorMode::Material,
            scale,
        );
        assert_eq!(img.width(), size as u32);
        assert_eq!(img.height(), size as u32);
    }

    #[test]
    fn air_renders_black() {
        let size = 2;
        let grid = make_grid(size);
        let reg = test_registry();
        let img = render_frame(
            &grid,
            size,
            &reg,
            &ViewMode::TopDown,
            &ColorMode::Material,
            1,
        );
        assert_eq!(*img.get_pixel(0, 0), Rgb([0, 0, 0]));
    }

    #[test]
    fn stone_renders_correct_color() {
        let size = 2;
        let mut grid = make_grid(size);
        grid[idx(0, 0, 0, size)].material = MaterialId::STONE;
        let reg = test_registry();
        let view = ViewMode::Slice {
            axis: SliceAxis::Y,
            depth: 0,
        };
        let img = render_frame(&grid, size, &reg, &view, &ColorMode::Material, 1);
        // Stone color = [0.5, 0.5, 0.5] → [127, 127, 127]
        assert_eq!(*img.get_pixel(0, 0), Rgb([127, 127, 127]));
    }

    #[test]
    fn temperature_heatmap_extremes() {
        let cold = heatmap_rgb(0.0);
        let hot = heatmap_rgb(1.0);
        // Cold = blue, hot = red
        assert_eq!(cold, Rgb([0, 0, 255]));
        assert_eq!(hot, Rgb([255, 0, 0]));
    }

    #[test]
    fn depth_clamped_to_grid_bounds() {
        let size = 4;
        let grid = make_grid(size);
        let reg = test_registry();
        let view = ViewMode::Slice {
            axis: SliceAxis::Y,
            depth: 999, // out of bounds
        };
        // Should not panic
        let img = render_frame(&grid, size, &reg, &view, &ColorMode::Material, 1);
        assert_eq!(img.width(), size as u32);
    }

    #[test]
    fn topdown_finds_first_opaque() {
        let size = 4;
        let mut grid = make_grid(size);
        // Place stone at y=2, water at y=1 in column (0, _, 0)
        grid[idx(0, 2, 0, size)].material = MaterialId::STONE;
        grid[idx(0, 1, 0, size)].material = MaterialId::WATER;
        let reg = test_registry();
        let img = render_frame(
            &grid,
            size,
            &reg,
            &ViewMode::TopDown,
            &ColorMode::Material,
            1,
        );
        // Top-down should find stone at y=2 (higher) first
        assert_eq!(*img.get_pixel(0, 0), Rgb([127, 127, 127]));
    }

    // ----- DDA raymarcher tests -----

    #[test]
    fn dda_hits_solid_voxel_straight_down() {
        let size = 8;
        let mut grid = make_grid(size);
        grid[idx(4, 2, 4, size)].material = MaterialId::STONE;

        let origin = Vec3::new(4.5, 7.5, 4.5);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let hit = dda_march(origin, dir, &grid, size, 100.0);

        assert!(hit.is_some(), "should hit stone voxel");
        let h = hit.unwrap();
        assert_eq!((h.x, h.y, h.z), (4, 2, 4));
        assert_eq!(h.face_axis, 1); // entered from Y axis
    }

    #[test]
    fn dda_misses_empty_grid() {
        let size = 8;
        let grid = make_grid(size); // all air
        let origin = Vec3::new(4.5, 7.5, 4.5);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        assert!(dda_march(origin, dir, &grid, size, 100.0).is_none());
    }

    #[test]
    fn dda_hits_from_outside_grid() {
        let size = 8;
        let mut grid = make_grid(size);
        // Fill y=0 plane with stone
        for x in 0..size {
            for z in 0..size {
                grid[idx(x, 0, z, size)].material = MaterialId::STONE;
            }
        }

        // Ray from above and outside the grid, angled in
        let origin = Vec3::new(4.0, 12.0, 4.0);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let hit = dda_march(origin, dir, &grid, size, 100.0);
        assert!(hit.is_some(), "should enter grid and hit ground");
    }

    #[test]
    fn dda_diagonal_hit() {
        let size = 8;
        let mut grid = make_grid(size);
        grid[idx(6, 0, 6, size)].material = MaterialId::STONE;

        let origin = Vec3::new(0.5, 4.0, 0.5);
        let dir = Vec3::new(1.0, -0.6, 1.0);
        let hit = dda_march(origin, dir, &grid, size, 100.0);
        assert!(hit.is_some(), "diagonal ray should find stone");
        let h = hit.unwrap();
        assert_eq!((h.x, h.y, h.z), (6, 0, 6));
    }

    #[test]
    fn shadow_ray_detects_occlusion() {
        let size = 8;
        let mut grid = make_grid(size);
        // Overhanging block at (4, 5, 4)
        grid[idx(4, 5, 4, size)].material = MaterialId::STONE;

        let below = Vec3::new(4.5, 3.5, 4.5);
        let light_from_above = Vec3::new(0.0, -1.0, 0.0);
        assert!(
            is_shadowed(below, light_from_above, &grid, size),
            "point below overhang should be in shadow"
        );
    }

    #[test]
    fn shadow_ray_clear_when_no_blocker() {
        let size = 8;
        let grid = make_grid(size); // all air
        let pos = Vec3::new(4.5, 3.5, 4.5);
        let light_from_above = Vec3::new(0.0, -1.0, 0.0);
        assert!(
            !is_shadowed(pos, light_from_above, &grid, size),
            "no blocker = no shadow"
        );
    }

    #[test]
    fn perspective_renders_correct_dimensions() {
        let size = 4;
        let grid = make_grid(size);
        let reg = test_registry();
        let view = ViewMode::Perspective {
            eye: (8.0, 8.0, 8.0),
            target: (2.0, 2.0, 2.0),
            fov_degrees: 60.0,
            width: 64,
            height: 48,
        };
        let img = render_frame(&grid, size, &reg, &view, &ColorMode::Material, 1);
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 48);
    }

    #[test]
    fn perspective_renders_terrain_with_shading() {
        let size = 8;
        let mut grid = make_grid(size);
        // Fill y=0 with stone to form a ground plane
        for x in 0..size {
            for z in 0..size {
                grid[idx(x, 0, z, size)].material = MaterialId::STONE;
            }
        }
        let reg = test_registry();
        let light = SceneLight {
            direction: (-0.4, -0.8, -0.3),
            intensity: 1.0,
            ambient: 0.1,
            ..Default::default()
        };
        let view = ViewMode::Perspective {
            eye: (4.0, 6.0, -2.0),
            target: (4.0, 0.0, 4.0),
            fov_degrees: 60.0,
            width: 32,
            height: 32,
        };
        let img = render_frame_lit(&grid, size, &reg, &view, &ColorMode::Material, 1, &light);

        // Center-bottom pixel should show lit stone (not black sky or pure ambient)
        let center_px = *img.get_pixel(16, 28);
        // Stone base = 127, with ambient=0.1 minimum → at least ~12
        assert!(
            center_px.0[0] > 10 || center_px.0[1] > 10 || center_px.0[2] > 10,
            "ground should be visible, got {:?}",
            center_px
        );
    }
}
