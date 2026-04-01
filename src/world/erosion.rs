#![allow(dead_code)]

//! D8 Flow Accumulation and Valley Carving
//!
//! This module implements a hydraulic erosion preprocessing pipeline for
//! terrain generation. It computes water flow patterns across a heightmap
//! and carves realistic river valleys and drainage channels.
//!
//! ## Algorithm Overview
//!
//! 1. **Sink Filling** — Uses a priority-flood algorithm (Planchon &
//!    Darboux, 2002) to eliminate closed depressions in the heightmap.
//!    A min-heap processes cells from the boundary inward, raising any
//!    cell that sits below its pour point by a small epsilon to guarantee
//!    drainage.
//!
//! 2. **D8 Flow Direction** — Each cell is assigned one of eight cardinal
//!    or diagonal flow directions based on the steepest downhill gradient
//!    to a neighbour. Diagonal neighbours use a √2 distance factor for
//!    correct slope computation.
//!
//! 3. **Flow Accumulation** — Cells are sorted by descending height and
//!    traversed in order. Each cell contributes `1 + its own accumulation`
//!    to its downstream neighbour, producing a map of upstream catchment
//!    area.
//!
//! 4. **Valley Carving** — Accumulated flow drives channel geometry:
//!    depth scales with `ln(flow)`, width with `√flow`. A blendable V/U
//!    cross-section profile shapes the valley walls. Material overrides
//!    place sand on channel beds and expose stone on steep walls.

use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::voxel::MaterialId;

// -------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------

/// D8 direction offsets in grid coordinates `(dx, dy)`.
///
/// Index 0 = North, proceeding clockwise:
/// N, NE, E, SE, S, SW, W, NW.
const D8_OFFSETS: [(i32, i32); 8] = [
    (0, -1),  // 0: N
    (1, -1),  // 1: NE
    (1, 0),   // 2: E
    (1, 1),   // 3: SE
    (0, 1),   // 4: S
    (-1, 1),  // 5: SW
    (-1, 0),  // 6: W
    (-1, -1), // 7: NW
];

/// Unit-length direction vectors for each D8 index.
const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

const D8_UNIT_VECTORS: [(f64, f64); 8] = [
    (0.0, -1.0),              // N
    (INV_SQRT2, -INV_SQRT2),  // NE
    (1.0, 0.0),               // E
    (INV_SQRT2, INV_SQRT2),   // SE
    (0.0, 1.0),               // S
    (-INV_SQRT2, INV_SQRT2),  // SW
    (-1.0, 0.0),              // W
    (-INV_SQRT2, -INV_SQRT2), // NW
];

/// Small height increment added during sink filling to ensure every
/// filled cell has a strictly downhill path to the boundary.
const SINK_FILL_EPSILON: f64 = 1e-5;

/// Maximum search radius (in grid cells) when looking for the nearest
/// channel in [`FlowMap::nearest_channel_info`].
const CHANNEL_SEARCH_RADIUS: i32 = 10;

// -------------------------------------------------------------------
// ErosionConfig
// -------------------------------------------------------------------

/// Configuration parameters for the erosion and valley-carving system.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ErosionConfig {
    /// Whether erosion processing is enabled.
    pub enabled: bool,
    /// Minimum flow accumulation required to begin carving a channel.
    pub flow_threshold: f32,
    /// Multiplier applied to `ln(flow_accum)` when computing depth.
    pub depth_scale: f32,
    /// Absolute maximum channel depth in metres.
    pub max_channel_depth: f32,
    /// Multiplier applied to `sqrt(flow_accum)` for channel width.
    pub width_scale: f32,
    /// Cross-section blend factor.
    /// `0.0` = V-shaped (linear), `1.0` = U-shaped (parabolic).
    pub valley_shape: f32,
    /// Side length of the flow-map region in metres.
    pub region_size: f32,
    /// Grid cell resolution in metres.
    pub cell_size: f32,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            flow_threshold: 50.0,
            depth_scale: 3.0,
            max_channel_depth: 12.0,
            width_scale: 2.0,
            valley_shape: 0.3,
            region_size: 4096.0,
            cell_size: 8.0,
        }
    }
}

// -------------------------------------------------------------------
// ChannelInfo
// -------------------------------------------------------------------

/// Information about the nearest river channel relative to a query
/// point.
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    /// Upstream catchment area expressed as accumulated cell count.
    pub flow_accumulation: f64,
    /// Perpendicular distance from the channel centre-line in metres.
    pub distance: f64,
    /// Unit vector indicating the channel flow direction `(x, z)`.
    pub direction: (f64, f64),
}

// -------------------------------------------------------------------
// HeapCell (private — used by the priority-flood sink filler)
// -------------------------------------------------------------------

#[derive(PartialEq)]
struct HeapCell {
    height: f64,
    index: usize,
}

impl Eq for HeapCell {}

impl PartialOrd for HeapCell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.height
            .partial_cmp(&other.height)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.index.cmp(&other.index))
    }
}

// -------------------------------------------------------------------
// FlowMap
// -------------------------------------------------------------------

/// Pre-computed drainage map covering a rectangular region of terrain.
///
/// Stores the sink-filled heightmap, D8 flow directions, and
/// accumulated upstream catchment area for every cell in the grid.
pub struct FlowMap {
    width: usize,
    height: usize,
    cell_size: f64,
    origin_x: f64,
    origin_z: f64,
    /// Sink-filled heightmap.
    heights: Vec<f64>,
    /// D8 flow direction per cell (0–7, or 255 = no downhill).
    flow_direction: Vec<u8>,
    /// Number of upstream cells that drain through each cell.
    flow_accumulation: Vec<f64>,
}

impl FlowMap {
    // -- construction -----------------------------------------------

    /// Build a flow map by sampling `height_fn(world_x, world_z)` over
    /// a square grid centred at (`center_x`, `center_z`).
    ///
    /// Steps:
    /// 1. Sample terrain heights onto the grid.
    /// 2. Fill sinks (Planchon & Darboux priority-flood).
    /// 3. Compute D8 flow directions (steepest descent).
    /// 4. Accumulate upstream flow (descending-height traversal).
    pub fn compute(
        height_fn: impl Fn(f64, f64) -> f64,
        region_size: f64,
        cell_size: f64,
        center_x: f64,
        center_z: f64,
    ) -> Self {
        let cell_size = cell_size.max(0.001);
        let w = ((region_size / cell_size).ceil() as usize).max(2);
        let h = w;
        let origin_x = center_x - (w as f64 * cell_size) / 2.0;
        let origin_z = center_z - (h as f64 * cell_size) / 2.0;

        // 1. Sample heights.
        let mut raw = vec![0.0_f64; w * h];
        for gy in 0..h {
            for gx in 0..w {
                let wx = origin_x + (gx as f64 + 0.5) * cell_size;
                let wz = origin_z + (gy as f64 + 0.5) * cell_size;
                raw[gy * w + gx] = height_fn(wx, wz);
            }
        }

        // 2. Fill sinks.
        let filled = fill_sinks(&raw, w, h);

        // 3. D8 flow directions.
        let directions = compute_d8_directions(&filled, w, h, cell_size);

        // 4. Flow accumulation.
        let accumulation = compute_flow_accumulation(&filled, &directions, w, h);

        Self {
            width: w,
            height: h,
            cell_size,
            origin_x,
            origin_z,
            heights: filled,
            flow_direction: directions,
            flow_accumulation: accumulation,
        }
    }

    // -- grid accessors ---------------------------------------------

    /// Number of columns in the grid.
    pub fn grid_width(&self) -> usize {
        self.width
    }

    /// Number of rows in the grid.
    pub fn grid_height(&self) -> usize {
        self.height
    }

    /// Filled height at grid cell (`gx`, `gy`), or `None` if out of
    /// bounds.
    pub fn get_height(&self, gx: usize, gy: usize) -> Option<f64> {
        if gx < self.width && gy < self.height {
            Some(self.heights[gy * self.width + gx])
        } else {
            None
        }
    }

    /// D8 direction code at grid cell, or `None` if out of bounds.
    pub fn get_flow_direction(&self, gx: usize, gy: usize) -> Option<u8> {
        if gx < self.width && gy < self.height {
            Some(self.flow_direction[gy * self.width + gx])
        } else {
            None
        }
    }

    /// Flow accumulation at grid cell, or `None` if out of bounds.
    pub fn get_flow_accumulation(&self, gx: usize, gy: usize) -> Option<f64> {
        if gx < self.width && gy < self.height {
            Some(self.flow_accumulation[gy * self.width + gx])
        } else {
            None
        }
    }

    // -- world-space queries ----------------------------------------

    /// Bilinear-interpolated flow accumulation at a world position.
    ///
    /// Coordinates outside the region are clamped to the nearest edge
    /// cell.
    pub fn flow_at(&self, world_x: f64, world_z: f64) -> f64 {
        let gx = (world_x - self.origin_x) / self.cell_size - 0.5;
        let gz = (world_z - self.origin_z) / self.cell_size - 0.5;

        let x0 = gx.floor() as i64;
        let z0 = gz.floor() as i64;
        let fx = gx - x0 as f64;
        let fz = gz - z0 as f64;

        let sample = |ix: i64, iz: i64| -> f64 {
            let cx = ix.clamp(0, self.width as i64 - 1) as usize;
            let cz = iz.clamp(0, self.height as i64 - 1) as usize;
            self.flow_accumulation[cz * self.width + cx]
        };

        let v00 = sample(x0, z0);
        let v10 = sample(x0 + 1, z0);
        let v01 = sample(x0, z0 + 1);
        let v11 = sample(x0 + 1, z0 + 1);

        let top = v00 * (1.0 - fx) + v10 * fx;
        let bot = v01 * (1.0 - fx) + v11 * fx;
        top * (1.0 - fz) + bot * fz
    }

    /// Unit direction vector of water flow at a world position.
    ///
    /// Returns `None` when the position lies outside the grid or the
    /// underlying cell has no downhill neighbour.
    pub fn flow_direction_at(&self, world_x: f64, world_z: f64) -> Option<(f64, f64)> {
        let (gx, gy) = self.world_to_grid(world_x, world_z)?;
        let dir = self.flow_direction[gy * self.width + gx];
        if dir >= 8 {
            return None;
        }
        Some(D8_UNIT_VECTORS[dir as usize])
    }

    /// Find the strongest nearby channel above `flow_threshold`.
    ///
    /// Searches a local neighbourhood of grid cells (within
    /// [`CHANNEL_SEARCH_RADIUS`]) and returns [`ChannelInfo`] for the
    /// cell with the highest flow accumulation, or `None` if nothing
    /// exceeds the threshold.
    pub fn nearest_channel_info(
        &self,
        world_x: f64,
        world_z: f64,
        flow_threshold: f32,
    ) -> Option<ChannelInfo> {
        let threshold = flow_threshold as f64;
        let (cx, cy) = self.world_to_grid_nearest(world_x, world_z);

        let mut best_accum = threshold;
        let mut best_idx: Option<usize> = None;

        let r = CHANNEL_SEARCH_RADIUS;
        for dy in -r..=r {
            let gy = cy as i32 + dy;
            if gy < 0 || gy >= self.height as i32 {
                continue;
            }
            for dx in -r..=r {
                let gx = cx as i32 + dx;
                if gx < 0 || gx >= self.width as i32 {
                    continue;
                }
                let idx = gy as usize * self.width + gx as usize;
                let acc = self.flow_accumulation[idx];
                let dir = self.flow_direction[idx];
                // Only consider cells with a valid flow direction.
                if acc > best_accum && dir < 8 {
                    best_accum = acc;
                    best_idx = Some(idx);
                }
            }
        }

        let idx = best_idx?;
        let dir = self.flow_direction[idx];
        let unit = D8_UNIT_VECTORS[dir as usize];

        // World position of the channel cell centre.
        let cell_gx = idx % self.width;
        let cell_gy = idx / self.width;
        let cell_wx = self.origin_x + (cell_gx as f64 + 0.5) * self.cell_size;
        let cell_wz = self.origin_z + (cell_gy as f64 + 0.5) * self.cell_size;

        // Perpendicular distance from the channel centre-line.
        let vx = world_x - cell_wx;
        let vz = world_z - cell_wz;
        let along = vx * unit.0 + vz * unit.1;
        let perp_sq = (vx * vx + vz * vz) - along * along;
        let distance = perp_sq.max(0.0).sqrt();

        Some(ChannelInfo {
            flow_accumulation: self.flow_accumulation[idx],
            distance,
            direction: unit,
        })
    }

    // -- private helpers --------------------------------------------

    /// Convert a world position to the containing grid cell, or `None`
    /// if the position is outside the grid.
    fn world_to_grid(&self, world_x: f64, world_z: f64) -> Option<(usize, usize)> {
        let gx = ((world_x - self.origin_x) / self.cell_size) as i64;
        let gy = ((world_z - self.origin_z) / self.cell_size) as i64;
        if gx < 0 || gy < 0 || gx >= self.width as i64 || gy >= self.height as i64 {
            return None;
        }
        Some((gx as usize, gy as usize))
    }

    /// Convert a world position to the nearest valid grid cell,
    /// clamping to the grid boundary.
    fn world_to_grid_nearest(&self, world_x: f64, world_z: f64) -> (usize, usize) {
        let gx = ((world_x - self.origin_x) / self.cell_size)
            .round()
            .clamp(0.0, (self.width - 1) as f64) as usize;
        let gy = ((world_z - self.origin_z) / self.cell_size)
            .round()
            .clamp(0.0, (self.height - 1) as f64) as usize;
        (gx, gy)
    }
}

// -------------------------------------------------------------------
// Private algorithm helpers
// -------------------------------------------------------------------

/// Priority-flood sink filling (Planchon & Darboux, 2002).
///
/// Processes boundary cells first via a min-heap, propagating inward.
/// Any interior cell whose original height sits below its pour-point
/// neighbour is raised to `neighbour_height + ε`, guaranteeing every
/// cell a strictly downhill path to the boundary.
fn fill_sinks(raw: &[f64], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;
    let mut filled = raw.to_vec();
    let mut visited = vec![false; n];
    let mut heap: BinaryHeap<Reverse<HeapCell>> = BinaryHeap::with_capacity(n);

    // Seed the heap with all boundary cells.
    for gy in 0..h {
        for gx in 0..w {
            if gx == 0 || gx == w - 1 || gy == 0 || gy == h - 1 {
                let idx = gy * w + gx;
                visited[idx] = true;
                heap.push(Reverse(HeapCell {
                    height: filled[idx],
                    index: idx,
                }));
            }
        }
    }

    while let Some(Reverse(cell)) = heap.pop() {
        let cx = (cell.index % w) as i32;
        let cy = (cell.index / w) as i32;

        for &(dx, dy) in &D8_OFFSETS {
            let nx = cx + dx;
            let ny = cy + dy;
            if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if visited[ni] {
                continue;
            }
            visited[ni] = true;

            // Raise the neighbour if it sits below the current
            // pour point, otherwise keep its natural height.
            if filled[ni] < filled[cell.index] {
                filled[ni] = filled[cell.index] + SINK_FILL_EPSILON;
            }

            heap.push(Reverse(HeapCell {
                height: filled[ni],
                index: ni,
            }));
        }
    }

    filled
}

/// Assign a D8 flow direction to every cell based on steepest descent.
///
/// Slope is computed as `(h_self - h_neighbour) / distance`, where
/// diagonal distance is `cell_size × √2` and cardinal distance is
/// `cell_size`. The direction with the largest positive slope wins.
fn compute_d8_directions(heights: &[f64], w: usize, h: usize, cell_size: f64) -> Vec<u8> {
    let n = w * h;
    let mut dirs = vec![255_u8; n];
    let diag = cell_size * std::f64::consts::SQRT_2;

    for gy in 0..h {
        for gx in 0..w {
            let idx = gy * w + gx;
            let h0 = heights[idx];
            let mut best_slope = 0.0_f64;
            let mut best_dir = 255_u8;

            for (d, &(dx, dy)) in D8_OFFSETS.iter().enumerate() {
                let nx = gx as i32 + dx;
                let ny = gy as i32 + dy;
                if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                let dist = if dx != 0 && dy != 0 { diag } else { cell_size };
                let slope = (h0 - heights[ni]) / dist;
                if slope > best_slope {
                    best_slope = slope;
                    best_dir = d as u8;
                }
            }

            dirs[idx] = best_dir;
        }
    }

    dirs
}

/// Compute upstream flow accumulation by traversing cells from highest
/// to lowest. Each cell contributes `1 + own_accumulation` to its
/// downstream neighbour.
fn compute_flow_accumulation(heights: &[f64], dirs: &[u8], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;
    let mut accum = vec![0.0_f64; n];

    // Sort cell indices by descending filled height.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        heights[b]
            .partial_cmp(&heights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &idx in &order {
        let dir = dirs[idx];
        if dir >= 8 {
            continue;
        }
        let (dx, dy) = D8_OFFSETS[dir as usize];
        let nx = (idx % w) as i32 + dx;
        let ny = (idx / w) as i32 + dy;
        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
            let ni = ny as usize * w + nx as usize;
            accum[ni] += 1.0 + accum[idx];
        }
    }

    accum
}

// -------------------------------------------------------------------
// Valley carving (public free functions)
// -------------------------------------------------------------------

/// Maximum channel depth in metres for a given flow accumulation.
///
/// Depth scales logarithmically:
/// `min(max_channel_depth, depth_scale × ln(flow_accum))`.
/// Returns `0.0` for accumulation values ≤ 1.
pub fn channel_depth(flow_accum: f64, config: &ErosionConfig) -> f64 {
    if flow_accum <= 1.0 {
        return 0.0;
    }
    let depth = config.depth_scale as f64 * flow_accum.ln();
    depth.clamp(0.0, config.max_channel_depth as f64)
}

/// Channel half-width in metres for a given flow accumulation.
///
/// Width scales with the square root:
/// `width_scale × √flow_accum`.
pub fn channel_width(flow_accum: f64, config: &ErosionConfig) -> f64 {
    if flow_accum <= 0.0 {
        return 0.0;
    }
    config.width_scale as f64 * flow_accum.sqrt()
}

/// Carve a valley cross-section into the terrain.
///
/// Returns the adjusted height and an optional material override:
///
/// - Sand (`MaterialId::SAND`) on the channel bed (inner 20%).
/// - Stone (`MaterialId::STONE`) on steep exposed walls (20–50%
///   when the channel is deeper than 3 m).
/// - `None` outside the carved zone or for shallow channels.
pub fn carve_valley(
    base_height: f64,
    channel_info: &ChannelInfo,
    config: &ErosionConfig,
) -> (f64, Option<MaterialId>) {
    let max_depth = channel_depth(channel_info.flow_accumulation, config);
    let width = channel_width(channel_info.flow_accumulation, config);

    if width <= 0.0 || channel_info.distance > width {
        return (base_height, None);
    }

    // Normalised distance: 0 at centre, 1 at edge.
    let t = channel_info.distance / width;

    // Blend between V-shaped (linear) and U-shaped (parabolic).
    let v_profile = t;
    let u_profile = t * t;
    let vs = config.valley_shape as f64;
    let blended = vs * u_profile + (1.0 - vs) * v_profile;

    let carved = base_height - max_depth * (1.0 - blended);

    let material = if t < 0.2 {
        Some(MaterialId::SAND)
    } else if t < 0.5 && max_depth > 3.0 {
        Some(MaterialId::STONE) // exposed rock on steep walls
    } else {
        None
    };

    (carved, material)
}

// -------------------------------------------------------------------
// Hydraulic Erosion
// -------------------------------------------------------------------

/// Simulation mode for hydraulic erosion.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Default)]
pub enum HydraulicMode {
    /// Droplet simulation only (fine gullies).
    Droplet,
    /// Grid-based simulation only (broad shaping).
    Grid,
    /// Grid pass → repose slippage → droplet pass.
    #[default]
    Combined,
}

/// Configuration for hydraulic erosion simulation.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct HydraulicErosionConfig {
    /// Whether hydraulic erosion is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Simulation mode.
    #[serde(default)]
    pub mode: HydraulicMode,
    /// Number of droplet simulation iterations (per grid cell, approximately).
    #[serde(default = "default_droplet_iterations")]
    pub droplet_iterations: u32,
    /// Maximum lifetime of each droplet (steps before evaporation).
    #[serde(default = "default_droplet_lifetime")]
    pub droplet_lifetime: u32,
    /// Sediment capacity factor (higher = droplet can carry more).
    #[serde(default = "default_sediment_capacity")]
    pub sediment_capacity: f64,
    /// Erosion speed: fraction of deficit eroded per step.
    #[serde(default = "default_erosion_speed")]
    pub erosion_speed: f64,
    /// Deposition speed: fraction of excess deposited per step.
    #[serde(default = "default_deposition_speed")]
    pub deposition_speed: f64,
    /// Evaporation rate per step (fraction of water lost).
    #[serde(default = "default_evaporation_rate")]
    pub evaporation_rate: f64,
    /// Minimum slope for sediment capacity (prevents zero capacity on flat).
    #[serde(default = "default_min_slope")]
    pub min_slope: f64,
    /// Inertia: how much the droplet retains its previous direction (0–1).
    #[serde(default = "default_inertia")]
    pub inertia: f64,
    /// Gravity for droplet velocity (m/s², typically 9.8).
    #[serde(default = "default_droplet_gravity")]
    pub gravity: f64,
    /// Number of grid-based erosion iterations.
    #[serde(default = "default_grid_iterations")]
    pub grid_iterations: u32,
    /// Angle-of-repose tangent: slopes above this cause material slippage.
    #[serde(default = "default_repose_tangent")]
    pub repose_tangent: f64,
}

fn default_droplet_iterations() -> u32 {
    50_000
}
fn default_droplet_lifetime() -> u32 {
    64
}
fn default_sediment_capacity() -> f64 {
    8.0
}
fn default_erosion_speed() -> f64 {
    0.3
}
fn default_deposition_speed() -> f64 {
    0.3
}
fn default_evaporation_rate() -> f64 {
    0.01
}
fn default_min_slope() -> f64 {
    0.01
}
fn default_inertia() -> f64 {
    0.3
}
fn default_droplet_gravity() -> f64 {
    9.80665
}
fn default_grid_iterations() -> u32 {
    20
}
fn default_repose_tangent() -> f64 {
    0.8 // ~38° angle of repose (typical for loose sediment)
}

impl Default for HydraulicErosionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: HydraulicMode::default(),
            droplet_iterations: default_droplet_iterations(),
            droplet_lifetime: default_droplet_lifetime(),
            sediment_capacity: default_sediment_capacity(),
            erosion_speed: default_erosion_speed(),
            deposition_speed: default_deposition_speed(),
            evaporation_rate: default_evaporation_rate(),
            min_slope: default_min_slope(),
            inertia: default_inertia(),
            gravity: default_droplet_gravity(),
            grid_iterations: default_grid_iterations(),
            repose_tangent: default_repose_tangent(),
        }
    }
}

/// Run hydraulic erosion on a heightmap grid in-place.
///
/// `heights` is a row-major `w × h` grid.  `cell_size` is the spatial
/// resolution in meters.  `config` controls the simulation parameters.
/// `seed` provides determinism for droplet spawn locations.
pub fn hydraulic_erode(
    heights: &mut [f64],
    w: usize,
    h: usize,
    cell_size: f64,
    config: &HydraulicErosionConfig,
    seed: u32,
) {
    if !config.enabled || w < 3 || h < 3 {
        return;
    }

    match config.mode {
        HydraulicMode::Droplet => {
            droplet_erode(heights, w, h, cell_size, config, seed);
        }
        HydraulicMode::Grid => {
            grid_erode(heights, w, h, cell_size, config);
            repose_slippage(heights, w, h, cell_size, config);
        }
        HydraulicMode::Combined => {
            grid_erode(heights, w, h, cell_size, config);
            repose_slippage(heights, w, h, cell_size, config);
            droplet_erode(heights, w, h, cell_size, config, seed);
        }
    }
}

/// Droplet-based erosion (simplified Beyer variant).
///
/// Spawns droplets at pseudo-random positions.  Each droplet flows downhill,
/// eroding steep slopes and depositing on flat areas until it evaporates.
fn droplet_erode(
    heights: &mut [f64],
    w: usize,
    h: usize,
    cell_size: f64,
    config: &HydraulicErosionConfig,
    seed: u32,
) {
    let n = config.droplet_iterations;

    for i in 0..n {
        // Deterministic pseudo-random spawn position
        let hash = simple_hash(seed, i);
        let mut px = (hash % (w as u32 - 2)) as f64 + 1.0;
        let mut pz = ((hash / (w as u32)) % (h as u32 - 2)) as f64 + 1.0;
        let mut dx = 0.0_f64;
        let mut dz = 0.0_f64;
        let mut speed = 0.0_f64;
        let mut water = 1.0_f64;
        let mut sediment = 0.0_f64;

        for _ in 0..config.droplet_lifetime {
            let ix = px as usize;
            let iz = pz as usize;

            if ix < 1 || ix >= w - 1 || iz < 1 || iz >= h - 1 {
                break;
            }

            // Compute gradient via bilinear interpolation of neighbors
            let (gx, gz) = heightmap_gradient(heights, w, ix, iz, cell_size);

            // Apply inertia
            dx = dx * config.inertia - gx * (1.0 - config.inertia);
            dz = dz * config.inertia - gz * (1.0 - config.inertia);

            // Normalize direction
            let len = (dx * dx + dz * dz).sqrt();
            if len < 1e-10 {
                break;
            }
            dx /= len;
            dz /= len;

            // Move droplet
            let new_px = px + dx;
            let new_pz = pz + dz;

            let new_ix = new_px as usize;
            let new_iz = new_pz as usize;
            if new_ix < 1 || new_ix >= w - 1 || new_iz < 1 || new_iz >= h - 1 {
                break;
            }

            // Height difference
            let old_h = heights[iz * w + ix];
            let new_h = heights[new_iz * w + new_ix];
            let delta_h = new_h - old_h;

            // Sediment capacity based on slope
            let slope = (-delta_h / cell_size).max(config.min_slope);
            let capacity = slope * speed * water * config.sediment_capacity;

            if sediment > capacity || delta_h > 0.0 {
                // Deposit sediment
                let deposit = if delta_h > 0.0 {
                    // Going uphill: deposit min of sediment or height difference
                    sediment.min(delta_h)
                } else {
                    (sediment - capacity) * config.deposition_speed
                };
                heights[iz * w + ix] += deposit;
                sediment -= deposit;
            } else {
                // Erode
                let erode = ((capacity - sediment) * config.erosion_speed).min(-delta_h);
                heights[iz * w + ix] -= erode;
                sediment += erode;
            }

            // Update velocity and evaporate
            speed = (speed * speed + delta_h.abs() * config.gravity).sqrt();
            water *= 1.0 - config.evaporation_rate;

            px = new_px;
            pz = new_pz;

            if water < 0.001 {
                break;
            }
        }
    }
}

/// Grid-based erosion: water/sediment/velocity field simulation.
///
/// Distributes water uniformly and iteratively moves it downhill,
/// eroding steep cells and depositing on flat cells.
fn grid_erode(
    heights: &mut [f64],
    w: usize,
    h: usize,
    cell_size: f64,
    config: &HydraulicErosionConfig,
) {
    let n = w * h;
    let mut water = vec![0.01_f64; n];
    let mut sediment = vec![0.0_f64; n];
    let mut new_water = vec![0.0_f64; n];
    let mut new_sediment = vec![0.0_f64; n];

    for _ in 0..config.grid_iterations {
        // Rain: add water everywhere
        for w_cell in water.iter_mut() {
            *w_cell += 0.01;
        }

        // Flow: move water downhill
        new_water.fill(0.0);
        new_sediment.fill(0.0);

        for z in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = z * w + x;
                let h_cur = heights[idx] + water[idx];

                // Find lowest neighbor
                let mut best_idx = idx;
                let mut best_h = h_cur;

                for &(ddx, ddz) in &D8_OFFSETS {
                    let nx = x as i32 + ddx;
                    let nz = z as i32 + ddz;
                    if nx >= 0 && (nx as usize) < w && nz >= 0 && (nz as usize) < h {
                        let ni = nz as usize * w + nx as usize;
                        let nh = heights[ni] + water[ni];
                        if nh < best_h {
                            best_h = nh;
                            best_idx = ni;
                        }
                    }
                }

                if best_idx != idx {
                    let delta = (h_cur - best_h).min(water[idx]);
                    let transfer_frac = delta / h_cur.max(1e-10);
                    let water_transfer = water[idx] * transfer_frac * 0.5;
                    let sed_transfer = sediment[idx] * transfer_frac * 0.5;

                    new_water[best_idx] += water_transfer;
                    new_water[idx] -= water_transfer;
                    new_sediment[best_idx] += sed_transfer;
                    new_sediment[idx] -= sed_transfer;
                }
            }
        }

        // Apply transfers
        for i in 0..n {
            water[i] = (water[i] + new_water[i]).max(0.0);
            sediment[i] = (sediment[i] + new_sediment[i]).max(0.0);
        }

        // Erosion/deposition
        for z in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = z * w + x;
                let slope = local_slope(heights, w, h, x, z, cell_size);
                let capacity = water[idx] * slope * config.sediment_capacity * 0.1;

                if sediment[idx] > capacity {
                    let deposit = (sediment[idx] - capacity) * config.deposition_speed;
                    heights[idx] += deposit;
                    sediment[idx] -= deposit;
                } else {
                    let erode = (capacity - sediment[idx]) * config.erosion_speed * 0.1;
                    heights[idx] -= erode;
                    sediment[idx] += erode;
                }
            }
        }

        // Evaporation
        for w_cell in water.iter_mut() {
            *w_cell *= 1.0 - config.evaporation_rate * 5.0;
        }
    }
}

/// Angle-of-repose slippage: flatten slopes steeper than the repose angle.
fn repose_slippage(
    heights: &mut [f64],
    w: usize,
    h: usize,
    cell_size: f64,
    config: &HydraulicErosionConfig,
) {
    for _ in 0..3 {
        // Multiple passes for convergence
        for z in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = z * w + x;

                for &(ddx, ddz) in &D8_OFFSETS {
                    let nx = (x as i32 + ddx) as usize;
                    let nz = (z as i32 + ddz) as usize;
                    if nx < w && nz < h {
                        let ni = nz * w + nx;
                        let diff = heights[idx] - heights[ni];
                        let dist = if ddx.abs() + ddz.abs() == 2 {
                            cell_size * std::f64::consts::SQRT_2
                        } else {
                            cell_size
                        };
                        let max_d = config.repose_tangent * dist;

                        if diff > max_d {
                            let excess = (diff - max_d) * 0.5;
                            heights[idx] -= excess;
                            heights[ni] += excess;
                        }
                    }
                }
            }
        }
    }
}

// ── Hydraulic erosion helpers ──────────────────────────────────────────

/// Compute heightmap gradient at `(ix, iz)` using central differences.
fn heightmap_gradient(
    heights: &[f64],
    w: usize,
    ix: usize,
    iz: usize,
    cell_size: f64,
) -> (f64, f64) {
    let left = heights[iz * w + (ix - 1)];
    let right = heights[iz * w + (ix + 1)];
    let up = heights[(iz - 1) * w + ix];
    let down = heights[(iz + 1) * w + ix];
    let gx = (right - left) / (2.0 * cell_size);
    let gz = (down - up) / (2.0 * cell_size);
    (gx, gz)
}

/// Local slope magnitude at `(x, z)`.
fn local_slope(
    heights: &[f64],
    w: usize,
    h: usize,
    x: usize,
    z: usize,
    cell_size: f64,
) -> f64 {
    if x < 1 || x >= w - 1 || z < 1 || z >= h - 1 {
        return 0.0;
    }
    let (gx, gz) = heightmap_gradient(heights, w, x, z, cell_size);
    (gx * gx + gz * gz).sqrt()
}

/// Simple deterministic hash for pseudo-random droplet placement.
fn simple_hash(seed: u32, index: u32) -> u32 {
    let mut x = seed.wrapping_mul(1103515245).wrapping_add(index);
    x ^= x >> 16;
    x = x.wrapping_mul(0x45d9f3b);
    x ^= x >> 16;
    x = x.wrapping_mul(0x45d9f3b);
    x ^= x >> 16;
    x
}

// -------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    /// Helper: build a FlowMap from a height function on a small grid
    /// centred at the origin.
    fn small_flow_map(height_fn: impl Fn(f64, f64) -> f64, region: f64, cell: f64) -> FlowMap {
        FlowMap::compute(height_fn, region, cell, 0.0, 0.0)
    }

    // 1 -----------------------------------------------------------
    #[test]
    fn test_erosion_config_defaults() {
        let c = ErosionConfig::default();
        assert!(c.enabled);
        assert!((c.flow_threshold - 50.0).abs() < f32::EPSILON);
        assert!((c.depth_scale - 3.0).abs() < f32::EPSILON);
        assert!((c.max_channel_depth - 12.0).abs() < f32::EPSILON);
        assert!((c.width_scale - 2.0).abs() < f32::EPSILON);
        assert!((c.valley_shape - 0.3).abs() < f32::EPSILON);
        assert!((c.region_size - 4096.0).abs() < f32::EPSILON);
        assert!((c.cell_size - 8.0).abs() < f32::EPSILON);
    }

    // 2 -----------------------------------------------------------
    #[test]
    fn test_flow_map_tilted_plane() {
        // height = -z  →  terrain slopes downward to the south.
        let map = small_flow_map(|_x, z| -z, 128.0, 8.0);
        let w = map.grid_width();
        let h = map.grid_height();
        assert_eq!(w, 16);
        assert_eq!(h, 16);

        // Every cell except the bottom row should flow south (dir 4).
        for gy in 0..h - 1 {
            for gx in 0..w {
                let dir = map.get_flow_direction(gx, gy).expect("in bounds");
                assert_eq!(dir, 4, "cell ({gx},{gy}) should flow south, got {dir}");
            }
        }

        // Accumulation should increase from north to south.
        for gx in 0..w {
            let mut prev = -1.0_f64;
            for gy in 0..h {
                let acc = map.get_flow_accumulation(gx, gy).expect("in bounds");
                assert!(acc >= prev, "accum at ({gx},{gy})={acc} < prev={prev}");
                prev = acc;
            }
        }

        // Bottom row should collect all upstream cells in column.
        for gx in 0..w {
            let acc = map.get_flow_accumulation(gx, h - 1).expect("in bounds");
            assert!(
                acc >= (h - 1) as f64 - TOL,
                "bottom cell ({gx},{}) accum={acc}, expected >= {}",
                h - 1,
                h - 1
            );
        }
    }

    // 3 -----------------------------------------------------------
    #[test]
    fn test_sink_filling() {
        // Flat plane at 10.0 with a single-cell pit.
        // Grid 16×16, cell_size=8, origin=(-64,-64).
        // Cell (8,8) centre is at world (4, 4).
        let map = FlowMap::compute(
            |x, z| {
                if (x - 4.0).abs() < 1.0 && (z - 4.0).abs() < 1.0 {
                    0.0 // pit
                } else {
                    10.0
                }
            },
            128.0,
            8.0,
            0.0,
            0.0,
        );

        let cx = map.grid_width() / 2; // 8
        let cy = map.grid_height() / 2; // 8
        let h = map.get_height(cx, cy).expect("in bounds");
        assert!(
            (h - 10.0).abs() < 0.1,
            "pit should be filled to ~10.0, got {h}"
        );
    }

    // 4 -----------------------------------------------------------
    #[test]
    fn test_flow_accumulation_convergence() {
        // Bowl with slight southward tilt so there is a clear outlet.
        let map = FlowMap::compute(|x, z| x * x + z * z + 0.5 * z, 256.0, 8.0, 0.0, 0.0);

        let w = map.grid_width();
        let h = map.grid_height();

        // Centre should exceed all four corners.
        let centre = map.get_flow_accumulation(w / 2, h / 2).expect("in bounds");
        let corners = [
            map.get_flow_accumulation(0, 0).expect("in bounds"),
            map.get_flow_accumulation(w - 1, 0).expect("in bounds"),
            map.get_flow_accumulation(0, h - 1).expect("in bounds"),
            map.get_flow_accumulation(w - 1, h - 1).expect("in bounds"),
        ];
        for (i, &c) in corners.iter().enumerate() {
            assert!(
                centre > c,
                "centre acc {centre} should exceed corner {i} \
                 ({c})"
            );
        }

        // Maximum accumulation should be substantial (>10% of cells).
        let max_acc = (0..w * h)
            .map(|i| map.flow_accumulation[i])
            .fold(0.0_f64, f64::max);
        assert!(
            max_acc > (w * h) as f64 * 0.1,
            "max accumulation {max_acc} is too low"
        );
    }

    // 5 -----------------------------------------------------------
    #[test]
    fn test_channel_depth_scales_with_log() {
        let cfg = ErosionConfig::default();

        // At e, ln(e) = 1  →  depth = 3.0 × 1 = 3.0.
        let d_e = channel_depth(std::f64::consts::E, &cfg);
        assert!(
            (d_e - 3.0).abs() < TOL,
            "depth at e should be 3.0, got {d_e}"
        );

        // At 100, depth = min(12, 3 × ln(100)).
        let d100 = channel_depth(100.0, &cfg);
        let expected = (3.0_f64 * 100.0_f64.ln()).min(cfg.max_channel_depth as f64);
        assert!(
            (d100 - expected).abs() < TOL,
            "depth at 100: expected {expected}, got {d100}"
        );

        // At or below 1.0, depth is 0.
        assert!(channel_depth(1.0, &cfg).abs() < TOL);
        assert!(channel_depth(0.5, &cfg).abs() < TOL);
    }

    // 6 -----------------------------------------------------------
    #[test]
    fn test_channel_width_scales_with_sqrt() {
        let cfg = ErosionConfig::default();

        // width_scale(2) × sqrt(100) = 20.
        let w100 = channel_width(100.0, &cfg);
        let expected = 2.0 * 10.0;
        assert!(
            (w100 - expected).abs() < TOL,
            "width at 100: expected {expected}, got {w100}"
        );

        let w0 = channel_width(0.0, &cfg);
        assert!(w0.abs() < TOL, "width at 0 should be 0");
    }

    // 7 -----------------------------------------------------------
    #[test]
    fn test_carve_valley_at_center() {
        let cfg = ErosionConfig::default();
        let info = ChannelInfo {
            flow_accumulation: 100.0,
            distance: 0.0,
            direction: (0.0, 1.0),
        };
        let base = 50.0;
        let (carved, _) = carve_valley(base, &info, &cfg);
        let depth = channel_depth(100.0, &cfg);
        assert!(
            (carved - (base - depth)).abs() < TOL,
            "at centre the full depth should be carved"
        );
    }

    // 8 -----------------------------------------------------------
    #[test]
    fn test_carve_valley_beyond_width() {
        let cfg = ErosionConfig::default();
        let width = channel_width(100.0, &cfg);
        let info = ChannelInfo {
            flow_accumulation: 100.0,
            distance: width + 1.0,
            direction: (0.0, 1.0),
        };
        let base = 50.0;
        let (carved, mat) = carve_valley(base, &info, &cfg);
        assert!(
            (carved - base).abs() < TOL,
            "beyond width: height should be unchanged"
        );
        assert!(mat.is_none(), "beyond width: no material override");
    }

    // 9 -----------------------------------------------------------
    #[test]
    fn test_carve_valley_material_override() {
        let cfg = ErosionConfig::default();
        let accum = 100.0;
        let width = channel_width(accum, &cfg);
        let depth = channel_depth(accum, &cfg);
        assert!(depth > 3.0, "need depth > 3 for STONE test");

        // t < 0.2 → SAND
        let bed = ChannelInfo {
            flow_accumulation: accum,
            distance: 0.1 * width,
            direction: (0.0, 1.0),
        };
        let (_, mat) = carve_valley(50.0, &bed, &cfg);
        assert_eq!(mat, Some(MaterialId::SAND), "bed should be SAND");

        // t ∈ [0.2, 0.5) with depth > 3.0 → STONE
        let wall = ChannelInfo {
            flow_accumulation: accum,
            distance: 0.35 * width,
            direction: (0.0, 1.0),
        };
        let (_, mat) = carve_valley(50.0, &wall, &cfg);
        assert_eq!(mat, Some(MaterialId::STONE), "steep wall should be STONE");

        // t ≥ 0.5 → None
        let outer = ChannelInfo {
            flow_accumulation: accum,
            distance: 0.6 * width,
            direction: (0.0, 1.0),
        };
        let (_, mat) = carve_valley(50.0, &outer, &cfg);
        assert!(mat.is_none(), "outer zone: no material override");
    }

    // 10 ----------------------------------------------------------
    #[test]
    fn test_flow_map_determinism() {
        let height_fn = |x: f64, z: f64| x.sin() + z.cos();
        let a = small_flow_map(height_fn, 64.0, 4.0);
        let b = small_flow_map(height_fn, 64.0, 4.0);

        assert_eq!(a.grid_width(), b.grid_width());
        assert_eq!(a.grid_height(), b.grid_height());

        for gy in 0..a.grid_height() {
            for gx in 0..a.grid_width() {
                assert_eq!(
                    a.get_height(gx, gy),
                    b.get_height(gx, gy),
                    "height mismatch at ({gx},{gy})"
                );
                assert_eq!(
                    a.get_flow_direction(gx, gy),
                    b.get_flow_direction(gx, gy),
                    "direction mismatch at ({gx},{gy})"
                );
                let aa = a.get_flow_accumulation(gx, gy).expect("in bounds");
                let ba = b.get_flow_accumulation(gx, gy).expect("in bounds");
                assert!(
                    (aa - ba).abs() < TOL,
                    "accum mismatch at ({gx},{gy}): \
                     {aa} vs {ba}"
                );
            }
        }
    }

    // 11 ----------------------------------------------------------
    #[test]
    fn test_flow_at_interpolation() {
        // Tilted plane: accumulation at column x, row y ≈ y.
        let map = small_flow_map(|_x, z| -z, 128.0, 8.0);

        let cs = map.cell_size;
        let ox = map.origin_x;
        let oz = map.origin_z;

        // At exact cell centre the value should match the cell.
        let gx = 5_usize;
        let gy = 5_usize;
        let wx = ox + (gx as f64 + 0.5) * cs;
        let wz = oz + (gy as f64 + 0.5) * cs;
        let expected = map.get_flow_accumulation(gx, gy).expect("in bounds");
        let got = map.flow_at(wx, wz);
        assert!(
            (got - expected).abs() < 0.01,
            "cell centre: expected {expected}, got {got}"
        );

        // Halfway between rows 5 and 6 ≈ their average.
        let wz_mid = oz + (gy as f64 + 1.0) * cs;
        let a5 = map.get_flow_accumulation(gx, gy).expect("in bounds");
        let a6 = map.get_flow_accumulation(gx, gy + 1).expect("in bounds");
        let avg = (a5 + a6) / 2.0;
        let got_mid = map.flow_at(wx, wz_mid);
        assert!(
            (got_mid - avg).abs() < 0.5,
            "midpoint: expected ~{avg}, got {got_mid}"
        );
    }

    // 12 ----------------------------------------------------------
    #[test]
    fn test_v_shape_vs_u_shape() {
        let accum = 200.0;
        let make_info = |d: f64| ChannelInfo {
            flow_accumulation: accum,
            distance: d,
            direction: (0.0, 1.0),
        };

        let v_cfg = ErosionConfig {
            valley_shape: 0.0,
            ..ErosionConfig::default()
        };
        let u_cfg = ErosionConfig {
            valley_shape: 1.0,
            ..ErosionConfig::default()
        };

        let width = channel_width(accum, &v_cfg);
        let depth = channel_depth(accum, &v_cfg);
        let info = make_info(0.5 * width); // t = 0.5

        let (v_carved, _) = carve_valley(100.0, &info, &v_cfg);
        let (u_carved, _) = carve_valley(100.0, &info, &u_cfg);

        // At t = 0.5:
        //   V profile = 0.5  → carved = 100 - depth × 0.5
        //   U profile = 0.25 → carved = 100 - depth × 0.75
        // U-shape carves deeper (lower height) at the midpoint.
        assert!(
            u_carved < v_carved,
            "U ({u_carved}) should be lower than V ({v_carved})"
        );

        let expected_v = 100.0 - depth * (1.0 - 0.5);
        let expected_u = 100.0 - depth * (1.0 - 0.25);
        assert!(
            (v_carved - expected_v).abs() < TOL,
            "V mismatch: {v_carved} vs {expected_v}"
        );
        assert!(
            (u_carved - expected_u).abs() < TOL,
            "U mismatch: {u_carved} vs {expected_u}"
        );
    }

    // ── Hydraulic erosion tests ─────────────────────────────────────────

    fn default_hydraulic_config() -> HydraulicErosionConfig {
        HydraulicErosionConfig {
            enabled: true,
            ..Default::default()
        }
    }

    fn tilted_heightmap(w: usize, h: usize, cell_size: f64) -> Vec<f64> {
        let mut heights = vec![0.0; w * h];
        for z in 0..h {
            for x in 0..w {
                heights[z * w + x] = (x as f64) * cell_size * 0.5;
            }
        }
        heights
    }

    #[test]
    fn hydraulic_config_defaults_are_sensible() {
        let cfg = HydraulicErosionConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.droplet_iterations > 0);
        assert!(cfg.droplet_lifetime > 0);
        assert!(cfg.sediment_capacity > 0.0);
        assert!(cfg.erosion_speed > 0.0);
        assert!(cfg.deposition_speed > 0.0);
        assert!(cfg.evaporation_rate > 0.0);
        assert!(cfg.gravity > 0.0);
        assert!(cfg.repose_tangent > 0.0);
    }

    #[test]
    fn droplet_erosion_modifies_terrain() {
        let w = 64;
        let h = 64;
        let cell_size = 1.0;
        let mut heights = tilted_heightmap(w, h, cell_size);
        let original: Vec<f64> = heights.clone();
        let config = HydraulicErosionConfig {
            enabled: true,
            mode: HydraulicMode::Droplet,
            droplet_iterations: 5000,
            ..Default::default()
        };

        droplet_erode(&mut heights, w, h, cell_size, &config, 42);

        let mut changed = 0;
        for (a, b) in original.iter().zip(heights.iter()) {
            if (a - b).abs() > 1e-10 {
                changed += 1;
            }
        }
        assert!(
            changed > 10,
            "Droplet erosion should modify multiple cells, got {changed}"
        );
    }

    #[test]
    fn grid_erosion_modifies_terrain() {
        let w = 32;
        let h = 32;
        let cell_size = 2.0;
        let mut heights = tilted_heightmap(w, h, cell_size);
        let original: Vec<f64> = heights.clone();
        let config = default_hydraulic_config();

        grid_erode(&mut heights, w, h, cell_size, &config);

        let mut changed = 0;
        for (a, b) in original.iter().zip(heights.iter()) {
            if (a - b).abs() > 1e-10 {
                changed += 1;
            }
        }
        assert!(
            changed > 10,
            "Grid erosion should modify multiple cells, got {changed}"
        );
    }

    #[test]
    fn repose_slippage_flattens_steep_slopes() {
        let w = 16;
        let h = 16;
        let cell_size = 1.0;
        let mut heights = vec![0.0; w * h];
        for z in 0..h {
            for x in w / 2..w {
                heights[z * w + x] = 100.0;
            }
        }

        let config = default_hydraulic_config();
        repose_slippage(&mut heights, w, h, cell_size, &config);

        let edge_x = w / 2;
        let z_mid = h / 2;
        let left = heights[z_mid * w + (edge_x - 1)];
        let right = heights[z_mid * w + edge_x];
        let diff = right - left;
        assert!(
            diff < 100.0,
            "Cliff should be less steep after slippage, diff={diff}"
        );
    }

    #[test]
    fn combined_erosion_modifies_terrain() {
        let w = 32;
        let h = 32;
        let cell_size = 2.0;
        let mut heights = tilted_heightmap(w, h, cell_size);
        let original: Vec<f64> = heights.clone();
        let config = HydraulicErosionConfig {
            enabled: true,
            mode: HydraulicMode::Combined,
            droplet_iterations: 1000,
            grid_iterations: 5,
            ..Default::default()
        };

        hydraulic_erode(&mut heights, w, h, cell_size, &config, 42);

        let total_change: f64 = original
            .iter()
            .zip(heights.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            total_change > 1.0,
            "Combined erosion should produce meaningful changes, total_change={total_change}"
        );
    }

    #[test]
    fn disabled_hydraulic_erosion_does_nothing() {
        let w = 16;
        let h = 16;
        let cell_size = 1.0;
        let mut heights = tilted_heightmap(w, h, cell_size);
        let original: Vec<f64> = heights.clone();

        let config = HydraulicErosionConfig::default();
        hydraulic_erode(&mut heights, w, h, cell_size, &config, 42);

        assert_eq!(heights, original, "Disabled erosion should not modify terrain");
    }

    #[test]
    fn droplet_erosion_is_deterministic() {
        let w = 32;
        let h = 32;
        let cell_size = 1.0;
        let config = HydraulicErosionConfig {
            enabled: true,
            mode: HydraulicMode::Droplet,
            droplet_iterations: 2000,
            ..Default::default()
        };

        let mut h1 = tilted_heightmap(w, h, cell_size);
        let mut h2 = tilted_heightmap(w, h, cell_size);
        droplet_erode(&mut h1, w, h, cell_size, &config, 42);
        droplet_erode(&mut h2, w, h, cell_size, &config, 42);

        assert_eq!(h1, h2, "Same seed should produce identical erosion");
    }

    #[test]
    fn simple_hash_is_deterministic() {
        assert_eq!(simple_hash(42, 0), simple_hash(42, 0));
        assert_ne!(simple_hash(42, 0), simple_hash(42, 1));
        assert_ne!(simple_hash(42, 0), simple_hash(43, 0));
    }
}
