// Frame renderer: converts a voxel grid slice into an RGB image buffer.
//
// Designed for headless use in the simulation test harness. Each frame
// represents one simulation tick and can be piped to ffmpeg for video
// encoding or saved as individual PNGs.

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

/// Render a single frame from the voxel grid.
///
/// Returns an `RgbImage` of dimensions `(grid_size * scale, grid_size * scale)`.
pub fn render_frame(
    voxels: &[Voxel],
    size: usize,
    registry: &MaterialRegistry,
    view: &ViewMode,
    color_mode: &ColorMode,
    scale: u32,
) -> RgbImage {
    let dim = (size as u32) * scale;
    let mut img = RgbImage::new(dim, dim);

    match view {
        ViewMode::Slice { axis, depth } => {
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

                    // Fill the scaled pixel block
                    for sy in 0..scale {
                        for sx in 0..scale {
                            img.put_pixel(col as u32 * scale + sx, row as u32 * scale + sy, color);
                        }
                    }
                }
            }
        }
        ViewMode::TopDown => {
            for z in 0..size {
                for x in 0..size {
                    // March down from the top (highest Y) until we hit an
                    // opaque (non-air) voxel.
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
        }
    }

    img
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
}
