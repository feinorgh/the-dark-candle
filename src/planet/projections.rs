//! 2D map projection rendering for planetary data.
//!
//! Renders planetary data as equirectangular, Mollweide, or orthographic
//! projection images. Uses a spatial index for fast nearest-cell lookups.
//! When detail is enabled (default), applies IDW interpolation, procedural
//! noise, and hillshading for realistic terrain relief.

use std::f64::consts::{FRAC_PI_2, PI, SQRT_2, TAU};

use bevy::math::DVec3;
use image::{Rgb, RgbImage};

use super::PlanetData;
use super::detail::{
    HillshadeParams, TerrainNoise, hillshade_pixel, lat_lon_to_pos, sample_detailed_elevation,
};
use super::grid::{CellId, IcosahedralGrid};
use super::render::{ColourMode, cell_color, elevation_color};

// ─── Projection types ─────────────────────────────────────────────────────────

/// Map projection type.
#[derive(Debug, Clone)]
pub enum Projection {
    /// Simple lat/lon grid. Distorted at poles. 2:1 aspect ratio.
    Equirectangular,
    /// Equal-area elliptical projection. 2:1 aspect ratio.
    Mollweide,
    /// Hemisphere view from one direction. 1:1 aspect ratio.
    Orthographic {
        /// Centre longitude in degrees.
        center_lon_deg: f64,
    },
}

impl Projection {
    /// Natural image height for the given width.
    pub fn natural_height(&self, width: u32) -> u32 {
        match self {
            Self::Equirectangular | Self::Mollweide => width / 2,
            Self::Orthographic { .. } => width,
        }
    }

    /// Inverse projection: pixel (x, y) → (latitude, longitude) in radians.
    ///
    /// Returns `None` if the pixel falls outside the projection boundary
    /// (e.g. outside the Mollweide ellipse or orthographic circle).
    fn inverse(&self, x: u32, y: u32, w: u32, h: u32) -> Option<(f64, f64)> {
        match self {
            Self::Equirectangular => {
                let lon = -PI + TAU * x as f64 / w as f64;
                let lat = FRAC_PI_2 - PI * y as f64 / h as f64;
                Some((lat, lon))
            }
            Self::Mollweide => mollweide_inverse(x, y, w, h),
            Self::Orthographic { center_lon_deg } => {
                orthographic_inverse(x, y, w, h, center_lon_deg.to_radians())
            }
        }
    }

    /// Parse a projection name from CLI input.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "equirectangular" | "equirect" => Some(Self::Equirectangular),
            "mollweide" => Some(Self::Mollweide),
            "orthographic" | "ortho" => Some(Self::Orthographic {
                center_lon_deg: 0.0,
            }),
            _ => None,
        }
    }
}

// ─── Inverse projection math ─────────────────────────────────────────────────

fn mollweide_inverse(x: u32, y: u32, w: u32, h: u32) -> Option<(f64, f64)> {
    let px = (2.0 * x as f64 / w as f64 - 1.0) * 2.0 * SQRT_2;
    let py = (1.0 - 2.0 * y as f64 / h as f64) * SQRT_2;

    // Inside the ellipse?
    if (px / (2.0 * SQRT_2)).powi(2) + (py / SQRT_2).powi(2) > 1.0 {
        return None;
    }

    let sin_theta = (py / SQRT_2).clamp(-1.0, 1.0);
    let theta = sin_theta.asin();
    let lat_arg = (2.0 * theta + (2.0 * theta).sin()) / PI;
    let lat = lat_arg.clamp(-1.0, 1.0).asin();

    let cos_theta = theta.cos();
    let lon = if cos_theta.abs() < 1e-10 {
        0.0
    } else {
        PI * px / (2.0 * SQRT_2 * cos_theta)
    };

    if lon.abs() > PI + 0.001 {
        return None;
    }

    Some((lat, lon.clamp(-PI, PI)))
}

fn orthographic_inverse(x: u32, y: u32, w: u32, h: u32, center_lon: f64) -> Option<(f64, f64)> {
    let px = 2.0 * x as f64 / w as f64 - 1.0;
    let py = 1.0 - 2.0 * y as f64 / h as f64;
    let rr = px * px + py * py;

    if rr > 1.0 {
        return None;
    }

    // Equatorial orthographic (center_lat = 0).
    let lat = py.clamp(-1.0, 1.0).asin();
    let z = (1.0 - rr).max(0.0).sqrt();
    let lon = center_lon + px.atan2(z);

    // Wrap to [−π, π].
    let lon = ((lon + PI) % TAU + TAU) % TAU - PI;

    Some((lat, lon))
}

/// Convert (lat, lon) in radians to a unit-sphere DVec3 (Y-up).
fn lat_lon_to_dvec3(lat: f64, lon: f64) -> DVec3 {
    let cos_lat = lat.cos();
    DVec3::new(cos_lat * lon.sin(), lat.sin(), cos_lat * lon.cos())
}

// ─── Spatial index ────────────────────────────────────────────────────────────

/// Grid-based spatial index for fast nearest-cell lookup by (lat, lon).
///
/// Partitions the sphere into 1° × 1° bins. For a query point, only cells
/// in nearby bins are checked (with wider lon search near the poles).
struct SpatialIndex {
    lat_bins: usize,
    lon_bins: usize,
    bins: Vec<Vec<u32>>,
}

impl SpatialIndex {
    fn new(grid: &IcosahedralGrid) -> Self {
        let lat_bins = 180;
        let lon_bins = 360;
        let mut bins = vec![Vec::new(); lat_bins * lon_bins];

        for id in grid.cell_ids() {
            let (lat, lon) = grid.cell_lat_lon(id);
            let lb = ((lat + FRAC_PI_2) / PI * lat_bins as f64).clamp(0.0, (lat_bins - 1) as f64)
                as usize;
            let lob =
                ((lon + PI) / TAU * lon_bins as f64).clamp(0.0, (lon_bins - 1) as f64) as usize;
            bins[lb * lon_bins + lob].push(id.0);
        }

        Self {
            lat_bins,
            lon_bins,
            bins,
        }
    }

    fn nearest_cell(&self, grid: &IcosahedralGrid, lat: f64, lon: f64) -> CellId {
        let lb = ((lat + FRAC_PI_2) / PI * self.lat_bins as f64)
            .clamp(0.0, (self.lat_bins - 1) as f64) as usize;
        let lob = ((lon + PI) / TAU * self.lon_bins as f64).clamp(0.0, (self.lon_bins - 1) as f64)
            as usize;

        // Widen longitude search near the poles where 1° of longitude is tiny.
        let cos_lat = lat.cos().abs().max(0.01);
        let extra_lon = ((2.0 / cos_lat) as i32)
            .max(2)
            .min(self.lon_bins as i32 / 2);

        let query = lat_lon_to_dvec3(lat, lon);
        let mut best_id = CellId(0);
        let mut best_dot = f64::NEG_INFINITY;

        for dlat in -2..=2_i32 {
            let lat_idx = (lb as i32 + dlat).clamp(0, self.lat_bins as i32 - 1) as usize;
            for dlon in -extra_lon..=extra_lon {
                let lon_idx = ((lob as i32 + dlon) % self.lon_bins as i32 + self.lon_bins as i32)
                    as usize
                    % self.lon_bins;

                for &cell_id in &self.bins[lat_idx * self.lon_bins + lon_idx] {
                    let pos = grid.cell_position(CellId(cell_id));
                    let dot = query.dot(pos);
                    if dot > best_dot {
                        best_dot = dot;
                        best_id = CellId(cell_id);
                    }
                }
            }
        }

        best_id
    }
}

// ─── Rendering ────────────────────────────────────────────────────────────────

const BACKGROUND: Rgb<u8> = Rgb([10, 10, 20]);

fn color_f32_to_rgb(c: [f32; 4]) -> Rgb<u8> {
    Rgb([
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    ])
}

/// Render a map projection of planet data as an RGB image.
///
/// Uses two-pass rendering: first computes detailed elevation at every pixel
/// (interpolated + noise), then applies hillshading and colour mapping.
pub fn render_projection(
    data: &PlanetData,
    projection: &Projection,
    mode: &ColourMode,
    width: u32,
) -> RgbImage {
    let height = projection.natural_height(width);
    let index = SpatialIndex::new(&data.grid);
    let noise = TerrainNoise::new(data.config.seed);
    let w = width as usize;
    let h = height as usize;

    // Pass 1: compute detailed elevation and nearest-cell for every pixel.
    let mut elevations = vec![f64::NAN; w * h];
    let mut cells = vec![0usize; w * h];

    for y in 0..height {
        for x in 0..width {
            if let Some((lat, lon)) = projection.inverse(x, y, width, height) {
                let cell = index.nearest_cell(&data.grid, lat, lon);
                let pos = lat_lon_to_pos(lat, lon);
                let (elev, ci) = sample_detailed_elevation(data, &noise, pos, cell);
                let idx = y as usize * w + x as usize;
                elevations[idx] = elev;
                cells[idx] = ci;
            }
        }
    }

    // Approximate ground distance per pixel (using equatorial circumference).
    let cell_size_m = PI * data.config.radius_m / h as f64;

    // Pass 2: hillshade and colour.
    let hs = HillshadeParams::default();
    let z_factor = 15.0;
    let mut img = RgbImage::from_pixel(width, height, BACKGROUND);

    for y in 0..height {
        for x in 0..width {
            let idx = y as usize * w + x as usize;
            if elevations[idx].is_nan() {
                continue;
            }

            let shade = hillshade_pixel(
                &elevations,
                x as usize,
                y as usize,
                w,
                h,
                cell_size_m,
                z_factor,
                &hs,
            );

            // Use detailed elevation for Elevation mode; nearest cell for others.
            let base = match mode {
                ColourMode::Elevation => elevation_color(elevations[idx]),
                _ => cell_color(data, cells[idx], mode),
            };

            let shaded = [base[0] * shade, base[1] * shade, base[2] * shade, base[3]];
            img.put_pixel(x, y, color_f32_to_rgb(shaded));
        }
    }

    img
}

/// Render a rotating orthographic animation.
///
/// Produces `frames` images at evenly spaced longitudes and encodes them
/// using [`FrameEncoder`](crate::diagnostics::video::FrameEncoder).
pub fn render_animation(
    data: &PlanetData,
    mode: &ColourMode,
    width: u32,
    frames: u32,
    output_path: &str,
) -> Result<(), String> {
    use crate::diagnostics::video::FrameEncoder;

    let height = width; // orthographic is 1:1
    let w = width as usize;
    let h = height as usize;
    let index = SpatialIndex::new(&data.grid);
    let noise = TerrainNoise::new(data.config.seed);
    let hs = HillshadeParams::default();
    let cell_size_m = PI * data.config.radius_m / h as f64;
    let z_factor = 15.0;
    let mut encoder = FrameEncoder::new(output_path, width, height, 30)?;

    let mut elevations = vec![f64::NAN; w * h];
    let mut cells = vec![0usize; w * h];

    for f in 0..frames {
        let center_lon = TAU * f as f64 / frames as f64;

        // Reset buffers.
        elevations.fill(f64::NAN);

        // Pass 1: elevation.
        for y in 0..height {
            for x in 0..width {
                if let Some((lat, lon)) = orthographic_inverse(x, y, width, height, center_lon) {
                    let cell = index.nearest_cell(&data.grid, lat, lon);
                    let pos = lat_lon_to_pos(lat, lon);
                    let (elev, ci) = sample_detailed_elevation(data, &noise, pos, cell);
                    let idx = y as usize * w + x as usize;
                    elevations[idx] = elev;
                    cells[idx] = ci;
                }
            }
        }

        // Pass 2: hillshade + colour.
        let mut img = RgbImage::from_pixel(width, height, BACKGROUND);
        for y in 0..height {
            for x in 0..width {
                let idx = y as usize * w + x as usize;
                if elevations[idx].is_nan() {
                    continue;
                }

                let shade = hillshade_pixel(
                    &elevations,
                    x as usize,
                    y as usize,
                    w,
                    h,
                    cell_size_m,
                    z_factor,
                    &hs,
                );
                let base = match mode {
                    ColourMode::Elevation => elevation_color(elevations[idx]),
                    _ => cell_color(data, cells[idx], mode),
                };
                let shaded = [base[0] * shade, base[1] * shade, base[2] * shade, base[3]];
                img.put_pixel(x, y, color_f32_to_rgb(shaded));
            }
        }

        encoder.push_frame(&img)?;
        if f % 30 == 0 {
            println!("  Animation frame {f}/{frames}");
        }
    }

    encoder.finish()
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
    fn equirectangular_center_pixel() {
        let (lat, lon) = Projection::Equirectangular
            .inverse(500, 250, 1000, 500)
            .unwrap();
        assert!(lat.abs() < 0.01, "center lat should be ~0, got {lat}");
        assert!(lon.abs() < 0.01, "center lon should be ~0, got {lon}");
    }

    #[test]
    fn mollweide_center_pixel() {
        let (lat, lon) = Projection::Mollweide.inverse(500, 250, 1000, 500).unwrap();
        assert!(lat.abs() < 0.01, "center lat should be ~0, got {lat}");
        assert!(lon.abs() < 0.01, "center lon should be ~0, got {lon}");
    }

    #[test]
    fn mollweide_outside_ellipse() {
        assert!(Projection::Mollweide.inverse(0, 0, 1000, 500).is_none());
        assert!(Projection::Mollweide.inverse(999, 0, 1000, 500).is_none());
    }

    #[test]
    fn orthographic_center_and_edge() {
        let proj = Projection::Orthographic {
            center_lon_deg: 45.0,
        };
        let (lat, lon) = proj.inverse(500, 500, 1000, 1000).unwrap();
        assert!(lat.abs() < 0.01, "center lat should be ~0, got {lat}");
        assert!(
            (lon - 45.0_f64.to_radians()).abs() < 0.01,
            "center lon should be ~45°, got {lon}"
        );
        // Corner pixel is outside the circle.
        assert!(proj.inverse(0, 0, 1000, 1000).is_none());
    }

    #[test]
    fn spatial_index_finds_correct_cell() {
        let data = test_planet();
        let index = SpatialIndex::new(&data.grid);

        // For each cell, the index should return itself or a very close neighbor.
        for id in data.grid.cell_ids() {
            let (lat, lon) = data.grid.cell_lat_lon(id);
            let found = index.nearest_cell(&data.grid, lat, lon);
            if found != id {
                let pos = data.grid.cell_position(id);
                let found_pos = data.grid.cell_position(found);
                let dot = pos.dot(found_pos);
                assert!(
                    dot > 0.999,
                    "spatial index returned distant cell for {id}: dot={dot}"
                );
            }
        }
    }

    #[test]
    fn equirectangular_image_dimensions() {
        let data = test_planet();
        let img = render_projection(
            &data,
            &Projection::Equirectangular,
            &ColourMode::Elevation,
            200,
        );
        assert_eq!(img.width(), 200);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn mollweide_has_background_corners() {
        let data = test_planet();
        let img = render_projection(&data, &Projection::Mollweide, &ColourMode::Biome, 200);
        assert_eq!(*img.get_pixel(0, 0), BACKGROUND);
    }

    #[test]
    fn orthographic_is_square() {
        let data = test_planet();
        let proj = Projection::Orthographic {
            center_lon_deg: 0.0,
        };
        let img = render_projection(&data, &proj, &ColourMode::Temperature, 200);
        assert_eq!(img.width(), 200);
        assert_eq!(img.height(), 200);
    }

    #[test]
    fn equirectangular_no_background_pixels() {
        let data = test_planet();
        let img = render_projection(
            &data,
            &Projection::Equirectangular,
            &ColourMode::Elevation,
            100,
        );
        // Every pixel should be coloured (no background).
        for y in 0..img.height() {
            for x in 0..img.width() {
                assert_ne!(
                    *img.get_pixel(x, y),
                    BACKGROUND,
                    "pixel ({x},{y}) should not be background"
                );
            }
        }
    }

    #[test]
    fn projection_from_name() {
        assert!(Projection::from_name("equirect").is_some());
        assert!(Projection::from_name("mollweide").is_some());
        assert!(Projection::from_name("ortho").is_some());
        assert!(Projection::from_name("mercator").is_none());
    }
}
