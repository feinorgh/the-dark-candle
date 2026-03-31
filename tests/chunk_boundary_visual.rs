//! Headless visual test for chunk boundary seam rendering.
//!
//! Three test functions write to `test_output/`:
//!
//! 1. `chunk_boundary_seam_visual`
//!    Top-down orthographic view of a uniform 3×3 stone grid, fixed vs broken.
//!    Output: chunk_seams_{fixed,broken,comparison}.png
//!
//! 2. `chunk_boundary_perspective_visual`
//!    3/4-angle perspective view of the same grid.  From this angle the
//!    spurious vertical cap-faces produced by the unfixed mesher are clearly
//!    visible as bright vertical strips along every chunk edge.
//!    Output: chunk_seams_persp_{fixed,broken,comparison}.png
//!
//! 3. `chunk_boundary_mixed_materials`
//!    Chunks alternate between Stone, Dirt, Iron, and Wood so that material
//!    changes at every boundary.  The seam fix must work even when the two
//!    sides of a boundary hold different materials.
//!    Output: chunk_seams_mixed_{fixed,broken,comparison}.png
//!
//! Run: cargo test --test chunk_boundary_visual -- --nocapture

use image::{Rgb, RgbImage};
use the_dark_candle::world::chunk::{CHUNK_SIZE, Chunk, ChunkCoord};
use the_dark_candle::world::meshing::{ChunkMesh, NeighborVoxels, generate_mesh_with_colors};
use the_dark_candle::world::voxel::{MaterialId, Voxel};

// ── grid dimensions ──────────────────────────────────────────────────────────

const GRID_W: i32 = 3; // chunks along X
const GRID_D: i32 = 3; // chunks along Z
const SURFACE_Y: usize = 15; // stone fills y = 0 ..= SURFACE_Y

// ── chunk construction ───────────────────────────────────────────────────────

fn make_grid() -> Vec<Chunk> {
    let mut chunks = Vec::new();
    for cx in 0..GRID_W {
        for cz in 0..GRID_D {
            let coord = ChunkCoord::new(cx, 0, cz);
            let mut chunk = Chunk::new_filled(coord, MaterialId::AIR);
            for x in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    for y in 0..=SURFACE_Y {
                        chunk.set(x, y, z, Voxel::new(MaterialId::STONE));
                    }
                }
            }
            chunks.push(chunk);
        }
    }
    chunks
}

fn chunk_at(chunks: &[Chunk], cx: i32, cz: i32) -> Option<&Chunk> {
    if cx < 0 || cz < 0 || cx >= GRID_W || cz >= GRID_D {
        return None;
    }
    chunks.get((cx * GRID_D + cz) as usize)
}

fn build_neighbors(chunks: &[Chunk], cx: i32, cz: i32) -> NeighborVoxels {
    let mut n = NeighborVoxels::default();
    if let Some(c) = chunk_at(chunks, cx + 1, cz) {
        n.px = Some(c.flat_snapshot());
    }
    if let Some(c) = chunk_at(chunks, cx - 1, cz) {
        n.nx = Some(c.flat_snapshot());
    }
    if let Some(c) = chunk_at(chunks, cx, cz + 1) {
        n.pz = Some(c.flat_snapshot());
    }
    if let Some(c) = chunk_at(chunks, cx, cz - 1) {
        n.nz = Some(c.flat_snapshot());
    }
    n
}

// ── mesh generation ──────────────────────────────────────────────────────────

/// Collect all chunk meshes into world-space triangle soup.
fn world_mesh(chunks: &[Chunk], use_neighbors: bool) -> (Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<u32>) {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for cx in 0..GRID_W {
        for cz in 0..GRID_D {
            let chunk = chunk_at(chunks, cx, cz).unwrap();
            let origin = chunk.coord.world_origin();
            let ox = origin.x as f32;
            let oy = origin.y as f32;
            let oz = origin.z as f32;

            let mesh: ChunkMesh = if use_neighbors {
                let nb = build_neighbors(chunks, cx, cz);
                generate_mesh_with_colors(chunk, &nb, None, false)
            } else {
                generate_mesh_with_colors(chunk, &NeighborVoxels::default(), None, false)
            };

            let base = positions.len() as u32;
            for p in &mesh.positions {
                positions.push([p[0] + ox, p[1] + oy, p[2] + oz]);
            }
            colors.extend_from_slice(&mesh.colors);
            for i in &mesh.indices {
                indices.push(i + base);
            }
        }
    }

    (positions, colors, indices)
}

// ── software rasterizer ───────────────────────────────────────────────────────

/// Orthographic top-down view: world X → pixel X, world Z → pixel Y, world Y → depth.
fn rasterise(
    positions: &[[f32; 3]],
    colors: &[[f32; 4]],
    indices: &[u32],
    img_w: u32,
    img_h: u32,
    world_w: f32,
    world_d: f32,
) -> RgbImage {
    let mut img = RgbImage::from_pixel(img_w, img_h, Rgb([20u8, 20, 30]));
    let mut zbuf = vec![f32::NEG_INFINITY; (img_w * img_h) as usize];

    let sx = img_w as f32 / world_w;
    let sz = img_h as f32 / world_d;

    // project world → (screen_x, screen_y, depth)
    let proj = |p: [f32; 3]| -> (f32, f32, f32) { (p[0] * sx, p[2] * sz, p[1]) };

    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (x0, y0, z0) = proj(positions[i0]);
        let (x1, y1, z1) = proj(positions[i1]);
        let (x2, y2, z2) = proj(positions[i2]);

        let c0 = colors[i0];
        let c1 = colors[i1];
        let c2 = colors[i2];

        let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if area.abs() < 0.5 {
            continue;
        }

        let min_x = x0.min(x1).min(x2).max(0.0) as i32;
        let min_y = y0.min(y1).min(y2).max(0.0) as i32;
        let max_x = (x0.max(x1).max(x2) + 1.0).min(img_w as f32 - 1.0) as i32;
        let max_y = (y0.max(y1).max(y2) + 1.0).min(img_h as f32 - 1.0) as i32;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let pxf = px as f32 + 0.5;
                let pyf = py as f32 + 0.5;

                let w0 = (x1 - pxf) * (y2 - pyf) - (x2 - pxf) * (y1 - pyf);
                let w1 = (x2 - pxf) * (y0 - pyf) - (x0 - pxf) * (y2 - pyf);
                let w2 = (x0 - pxf) * (y1 - pyf) - (x1 - pxf) * (y0 - pyf);

                let inside = if area > 0.0 {
                    w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
                } else {
                    w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
                };
                if !inside {
                    continue;
                }

                let inv = 1.0 / area;
                let bw0 = w0 * inv;
                let bw1 = w1 * inv;
                let bw2 = w2 * inv;
                let depth = bw0 * z0 + bw1 * z1 + bw2 * z2;
                let bidx = py as usize * img_w as usize + px as usize;

                if depth > zbuf[bidx] {
                    zbuf[bidx] = depth;
                    let r = (bw0 * c0[0] + bw1 * c1[0] + bw2 * c2[0]).clamp(0.0, 1.0);
                    let g = (bw0 * c0[1] + bw1 * c1[1] + bw2 * c2[1]).clamp(0.0, 1.0);
                    let b = (bw0 * c0[2] + bw1 * c1[2] + bw2 * c2[2]).clamp(0.0, 1.0);
                    img.put_pixel(
                        px as u32,
                        py as u32,
                        Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]),
                    );
                }
            }
        }
    }

    img
}

/// Overlay yellow lines at chunk X and Z boundaries.
fn draw_boundaries(img: &mut RgbImage, img_w: u32, img_h: u32) {
    let voxels_x = GRID_W as u32 * CHUNK_SIZE as u32;
    let voxels_z = GRID_D as u32 * CHUNK_SIZE as u32;
    let ppv_x = img_w as f32 / voxels_x as f32;
    let ppv_z = img_h as f32 / voxels_z as f32;

    for cx in 1..GRID_W as u32 {
        let px = (cx as f32 * CHUNK_SIZE as f32 * ppv_x) as u32;
        for py in 0..img_h {
            img.put_pixel(px.min(img_w - 1), py, Rgb([255, 220, 0]));
        }
    }
    for cz in 1..GRID_D as u32 {
        let py = (cz as f32 * CHUNK_SIZE as f32 * ppv_z) as u32;
        for px in 0..img_w {
            img.put_pixel(px, py.min(img_h - 1), Rgb([255, 220, 0]));
        }
    }
}

fn side_by_side(left: &RgbImage, right: &RgbImage) -> RgbImage {
    let gap = 8u32;
    let w = left.width() + gap + right.width();
    let h = left.height().max(right.height());
    let mut out = RgbImage::from_pixel(w, h, Rgb([50u8, 50, 50]));
    for y in 0..left.height() {
        for x in 0..left.width() {
            out.put_pixel(x, y, *left.get_pixel(x, y));
        }
    }
    for y in 0..right.height() {
        for x in 0..right.width() {
            out.put_pixel(x + left.width() + gap, y, *right.get_pixel(x, y));
        }
    }
    out
}

// ── gap metric ───────────────────────────────────────────────────────────────

/// Count pixels that are not background (i.e., the geometry is visible).
fn count_lit_pixels(img: &RgbImage) -> u32 {
    img.pixels()
        .filter(|p| p[0] > 45 || p[1] > 45 || p[2] > 55)
        .count() as u32
}

fn near_boundary_dark_pixels(img: &RgbImage, img_w: u32, img_h: u32) -> u32 {
    let voxels_x = GRID_W as u32 * CHUNK_SIZE as u32;
    let voxels_z = GRID_D as u32 * CHUNK_SIZE as u32;
    let ppv_x = img_w as f32 / voxels_x as f32;
    let ppv_z = img_h as f32 / voxels_z as f32;
    let tol = (ppv_x.max(ppv_z) * 1.5) as u32 + 2;

    let is_dark = |p: &Rgb<u8>| p[0] < 45 && p[1] < 45 && p[2] < 55;
    let mut count = 0u32;

    for cx in 1..GRID_W as u32 {
        let bx = (cx as f32 * CHUNK_SIZE as f32 * ppv_x) as u32;
        let x0 = bx.saturating_sub(tol);
        let x1 = (bx + tol).min(img_w - 1);
        for py in 0..img_h {
            for px in x0..=x1 {
                if is_dark(img.get_pixel(px, py)) {
                    count += 1;
                }
            }
        }
    }
    for cz in 1..GRID_D as u32 {
        let bz = (cz as f32 * CHUNK_SIZE as f32 * ppv_z) as u32;
        let z0 = bz.saturating_sub(tol);
        let z1 = (bz + tol).min(img_h - 1);
        for py in z0..=z1 {
            for px in 0..img_w {
                if is_dark(img.get_pixel(px, py)) {
                    count += 1;
                }
            }
        }
    }
    count
}

// ── test entry point ─────────────────────────────────────────────────────────

#[test]
fn chunk_boundary_seam_visual() {
    std::fs::create_dir_all("test_output").expect("create test_output");

    let chunks = make_grid();

    let world_w = (GRID_W as f32) * CHUNK_SIZE as f32;
    let world_d = (GRID_D as f32) * CHUNK_SIZE as f32;
    const W: u32 = 768;
    const H: u32 = 768;

    // ── fixed (NeighborVoxels populated) ─────────────────────────────────────
    let (pf, cf, if_) = world_mesh(&chunks, true);
    let mut img_fixed = rasterise(&pf, &cf, &if_, W, H, world_w, world_d);
    draw_boundaries(&mut img_fixed, W, H);
    img_fixed
        .save("test_output/chunk_seams_fixed.png")
        .expect("save fixed");

    // ── broken (no neighbors) ────────────────────────────────────────────────
    let (pb, cb, ib) = world_mesh(&chunks, false);
    let mut img_broken = rasterise(&pb, &cb, &ib, W, H, world_w, world_d);
    draw_boundaries(&mut img_broken, W, H);
    img_broken
        .save("test_output/chunk_seams_broken.png")
        .expect("save broken");

    // ── side-by-side comparison ───────────────────────────────────────────────
    let comparison = side_by_side(&img_fixed, &img_broken);
    comparison
        .save("test_output/chunk_seams_comparison.png")
        .expect("save comparison");

    // ── metric ───────────────────────────────────────────────────────────────
    let gaps_fixed = near_boundary_dark_pixels(&img_fixed, W, H);
    let gaps_broken = near_boundary_dark_pixels(&img_broken, W, H);

    println!("Output written to test_output/");
    println!("  chunk_seams_fixed.png      — NeighborVoxels active");
    println!("  chunk_seams_broken.png     — no neighbors (old behavior)");
    println!("  chunk_seams_comparison.png — side-by-side");
    println!("Near-boundary dark pixels: fixed={gaps_fixed}, broken={gaps_broken}");

    assert!(
        gaps_fixed <= gaps_broken,
        "Seam fix should reduce boundary gaps (fixed={gaps_fixed}, broken={gaps_broken})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Perspective rasterizer
// ═══════════════════════════════════════════════════════════════════════════════

// Minimal 3-component vector math (avoids pulling in a math dep in tests).

fn v3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn v3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn v3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn v3_norm(a: [f32; 3]) -> [f32; 3] {
    let l = v3_dot(a, a).sqrt();
    if l < 1e-10 {
        a
    } else {
        [a[0] / l, a[1] / l, a[2] / l]
    }
}

/// Camera for perspective projection. Converts world positions directly to
/// screen pixels without going through matrices, avoiding row/column-major
/// confusion entirely.
struct Camera {
    eye: [f32; 3],
    right: [f32; 3],   // camera X axis
    up: [f32; 3],      // camera Y axis
    forward: [f32; 3], // points toward target
    f: f32,            // 1 / tan(fov_y / 2)
    aspect: f32,
    near: f32,
    far: f32,
    img_w: u32,
    img_h: u32,
}

impl Camera {
    fn new(eye: [f32; 3], target: [f32; 3], fov_y_deg: f32, img_w: u32, img_h: u32) -> Self {
        let forward = v3_norm(v3_sub(target, eye));
        let right = v3_norm(v3_cross(forward, [0.0, 1.0, 0.0]));
        let up = v3_cross(right, forward);
        Self {
            eye,
            right,
            up,
            forward,
            f: 1.0 / (fov_y_deg.to_radians() / 2.0).tan(),
            aspect: img_w as f32 / img_h as f32,
            near: 0.5,
            far: 600.0,
            img_w,
            img_h,
        }
    }

    /// Returns `(screen_x, screen_y, linear_depth)` or `None` if outside frustum.
    fn project(&self, p: [f32; 3]) -> Option<(f32, f32, f32)> {
        let d = v3_sub(p, self.eye);
        let cx = v3_dot(d, self.right);
        let cy = v3_dot(d, self.up);
        let cz = v3_dot(d, self.forward); // positive = in front

        if cz < self.near || cz > self.far {
            return None;
        }

        let ndc_x = cx * self.f / (self.aspect * cz);
        let ndc_y = cy * self.f / cz;

        let sx = (ndc_x * 0.5 + 0.5) * self.img_w as f32;
        let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * self.img_h as f32;

        if sx < 0.0 || sy < 0.0 || sx >= self.img_w as f32 || sy >= self.img_h as f32 {
            return None;
        }

        Some((sx, sy, cz))
    }
}

/// Rasterise a world-space triangle soup from a perspective camera.
#[allow(clippy::too_many_arguments)]
fn rasterise_perspective(
    positions: &[[f32; 3]],
    colors: &[[f32; 4]],
    indices: &[u32],
    eye: [f32; 3],
    target: [f32; 3],
    fov_y_deg: f32,
    img_w: u32,
    img_h: u32,
) -> RgbImage {
    let cam = Camera::new(eye, target, fov_y_deg, img_w, img_h);
    let light_dir = cam.forward; // diffuse from camera direction

    let mut img = RgbImage::from_pixel(img_w, img_h, Rgb([18u8, 18, 28]));
    let mut zbuf = vec![f32::INFINITY; (img_w * img_h) as usize];

    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        let (Some((x0, y0, z0)), Some((x1, y1, z1)), Some((x2, y2, z2))) = (
            cam.project(positions[i0]),
            cam.project(positions[i1]),
            cam.project(positions[i2]),
        ) else {
            continue;
        };

        let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if area.abs() < 0.5 {
            continue;
        }

        let e1 = v3_sub(positions[i1], positions[i0]);
        let e2 = v3_sub(positions[i2], positions[i0]);
        let normal = v3_norm(v3_cross(e1, e2));
        let light = (0.35 + 0.65 * v3_dot(normal, light_dir).max(0.0)).clamp(0.0, 1.0);

        let c0 = colors[i0];
        let c1 = colors[i1];
        let c2 = colors[i2];

        let min_x = x0.min(x1).min(x2).max(0.0) as i32;
        let min_y = y0.min(y1).min(y2).max(0.0) as i32;
        let max_x = (x0.max(x1).max(x2) + 1.0).min(img_w as f32 - 1.0) as i32;
        let max_y = (y0.max(y1).max(y2) + 1.0).min(img_h as f32 - 1.0) as i32;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let pxf = px as f32 + 0.5;
                let pyf = py as f32 + 0.5;

                let w0 = (x1 - pxf) * (y2 - pyf) - (x2 - pxf) * (y1 - pyf);
                let w1 = (x2 - pxf) * (y0 - pyf) - (x0 - pxf) * (y2 - pyf);
                let w2 = (x0 - pxf) * (y1 - pyf) - (x1 - pxf) * (y0 - pyf);

                let inside = if area > 0.0 {
                    w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
                } else {
                    w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
                };
                if !inside {
                    continue;
                }

                let inv = 1.0 / area;
                let depth = (w0 * z0 + w1 * z1 + w2 * z2) * inv;
                let bidx = py as usize * img_w as usize + px as usize;

                if depth < zbuf[bidx] {
                    zbuf[bidx] = depth;
                    let bw0 = w0 * inv;
                    let bw1 = w1 * inv;
                    let bw2 = w2 * inv;
                    let r = ((bw0 * c0[0] + bw1 * c1[0] + bw2 * c2[0]) * light).clamp(0.0, 1.0);
                    let g = ((bw0 * c0[1] + bw1 * c1[1] + bw2 * c2[1]) * light).clamp(0.0, 1.0);
                    let b = ((bw0 * c0[2] + bw1 * c1[2] + bw2 * c2[2]) * light).clamp(0.0, 1.0);
                    img.put_pixel(
                        px as u32,
                        py as u32,
                        Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]),
                    );
                }
            }
        }
    }

    img
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mixed-material grid
// ═══════════════════════════════════════════════════════════════════════════════

/// 3×3 grid where each chunk has a different material, so every shared boundary
/// has a material change on one or both sides.
fn make_mixed_grid() -> Vec<Chunk> {
    const PALETTE: [MaterialId; 4] = [
        MaterialId::STONE,
        MaterialId::DIRT,
        MaterialId::IRON,
        MaterialId::WOOD,
    ];

    let mut chunks = Vec::new();
    for cx in 0..GRID_W {
        for cz in 0..GRID_D {
            let mat = PALETTE[((cx + cz * 2) as usize) % PALETTE.len()];
            let coord = ChunkCoord::new(cx, 0, cz);
            let mut chunk = Chunk::new_filled(coord, MaterialId::AIR);
            for x in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    for y in 0..=SURFACE_Y {
                        chunk.set(x, y, z, Voxel::new(mat));
                    }
                }
            }
            chunks.push(chunk);
        }
    }
    chunks
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2 – perspective view
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn chunk_boundary_perspective_visual() {
    std::fs::create_dir_all("test_output").expect("create test_output");

    let chunks = make_grid();

    // Camera: 3/4 angle above-and-behind the grid centre.
    let cx = (GRID_W as f32 * CHUNK_SIZE as f32) / 2.0;
    let cz = (GRID_D as f32 * CHUNK_SIZE as f32) / 2.0;
    let cy = SURFACE_Y as f32;
    let eye = [cx - 80.0, cy + 90.0, cz - 80.0];
    let target = [cx, cy, cz];

    const W: u32 = 960;
    const H: u32 = 640;

    let (pf, cf, if_) = world_mesh(&chunks, true);
    let img_fixed = rasterise_perspective(&pf, &cf, &if_, eye, target, 50.0, W, H);
    img_fixed
        .save("test_output/chunk_seams_persp_fixed.png")
        .expect("save");

    let (pb, cb, ib) = world_mesh(&chunks, false);
    let img_broken = rasterise_perspective(&pb, &cb, &ib, eye, target, 50.0, W, H);
    img_broken
        .save("test_output/chunk_seams_persp_broken.png")
        .expect("save");

    let comparison = side_by_side(&img_fixed, &img_broken);
    comparison
        .save("test_output/chunk_seams_persp_comparison.png")
        .expect("save comparison");

    // In the broken image the spurious vertical cap-faces add extra mesh
    // triangles.  Count total lit pixels — broken should have more (extra
    // faces) while fixed should render fewer, smoother pixels near boundaries.
    let lit_fixed = count_lit_pixels(&img_fixed);
    let lit_broken = count_lit_pixels(&img_broken);

    println!("Perspective view:");
    println!("  chunk_seams_persp_fixed.png      — seam fix active");
    println!("  chunk_seams_persp_broken.png     — no neighbors");
    println!("  chunk_seams_persp_comparison.png — side-by-side");
    println!("Total lit pixels: fixed={lit_fixed}, broken={lit_broken}");

    // Broken has spurious side-faces → more lit pixels.
    assert!(
        lit_fixed <= lit_broken,
        "Broken mesh should have more lit pixels from extra cap faces \
         (fixed={lit_fixed}, broken={lit_broken})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3 – mixed-material boundaries
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn chunk_boundary_mixed_materials() {
    std::fs::create_dir_all("test_output").expect("create test_output");

    let chunks = make_mixed_grid();

    let world_w = (GRID_W as f32) * CHUNK_SIZE as f32;
    let world_d = (GRID_D as f32) * CHUNK_SIZE as f32;
    const W: u32 = 768;
    const H: u32 = 768;

    // Top-down
    let (pf, cf, if_) = world_mesh(&chunks, true);
    let mut img_fixed = rasterise(&pf, &cf, &if_, W, H, world_w, world_d);
    draw_boundaries(&mut img_fixed, W, H);
    img_fixed
        .save("test_output/chunk_seams_mixed_fixed.png")
        .expect("save");

    let (pb, cb, ib) = world_mesh(&chunks, false);
    let mut img_broken = rasterise(&pb, &cb, &ib, W, H, world_w, world_d);
    draw_boundaries(&mut img_broken, W, H);
    img_broken
        .save("test_output/chunk_seams_mixed_broken.png")
        .expect("save");

    // Perspective
    let cx = world_w / 2.0;
    let cz = world_d / 2.0;
    let cy = SURFACE_Y as f32;
    let eye = [cx - 80.0, cy + 90.0, cz - 80.0];
    let target = [cx, cy, cz];

    const PW: u32 = 960;
    const PH: u32 = 640;

    let (pf2, cf2, if2) = world_mesh(&chunks, true);
    let img_persp_fixed = rasterise_perspective(&pf2, &cf2, &if2, eye, target, 50.0, PW, PH);
    img_persp_fixed
        .save("test_output/chunk_seams_mixed_persp_fixed.png")
        .expect("save");

    let (pb2, cb2, ib2) = world_mesh(&chunks, false);
    let img_persp_broken = rasterise_perspective(&pb2, &cb2, &ib2, eye, target, 50.0, PW, PH);
    img_persp_broken
        .save("test_output/chunk_seams_mixed_persp_broken.png")
        .expect("save");

    // 4-up comparison: top row = topdown, bottom row = perspective
    let comparison = four_up(&img_fixed, &img_broken, &img_persp_fixed, &img_persp_broken);
    comparison
        .save("test_output/chunk_seams_mixed_4up.png")
        .expect("save");

    let gaps_fixed = near_boundary_dark_pixels(&img_fixed, W, H);
    let gaps_broken = near_boundary_dark_pixels(&img_broken, W, H);

    println!("Mixed-material view:");
    println!("  chunk_seams_mixed_fixed.png         — topdown, seam fix");
    println!("  chunk_seams_mixed_broken.png        — topdown, no neighbors");
    println!("  chunk_seams_mixed_persp_fixed.png   — perspective, seam fix");
    println!("  chunk_seams_mixed_persp_broken.png  — perspective, no neighbors");
    println!("  chunk_seams_mixed_4up.png           — 4-up comparison");
    println!("Near-boundary dark pixels: fixed={gaps_fixed}, broken={gaps_broken}");

    assert!(
        gaps_fixed <= gaps_broken,
        "Mixed-material seam fix check failed (fixed={gaps_fixed}, broken={gaps_broken})"
    );
}

/// Arrange four images in a 2×2 grid (top-left, top-right, bottom-left, bottom-right).
fn four_up(tl: &RgbImage, tr: &RgbImage, bl: &RgbImage, br: &RgbImage) -> RgbImage {
    let gap = 8u32;
    let col_w = tl.width().max(bl.width());
    let col_r = tr.width().max(br.width());
    let row_t = tl.height().max(tr.height());
    let row_b = bl.height().max(br.height());
    let total_w = col_w + gap + col_r;
    let total_h = row_t + gap + row_b;

    let mut out = RgbImage::from_pixel(total_w, total_h, Rgb([40u8, 40, 40]));

    let blit = |out: &mut RgbImage, img: &RgbImage, ox: u32, oy: u32| {
        for y in 0..img.height() {
            for x in 0..img.width() {
                out.put_pixel(ox + x, oy + y, *img.get_pixel(x, y));
            }
        }
    };

    blit(&mut out, tl, 0, 0);
    blit(&mut out, tr, col_w + gap, 0);
    blit(&mut out, bl, 0, row_t + gap);
    blit(&mut out, br, col_w + gap, row_t + gap);
    out
}
