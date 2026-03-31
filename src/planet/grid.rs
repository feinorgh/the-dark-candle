//! Icosahedral geodesic grid on the unit sphere.
//!
//! Constructed by recursive subdivision of a regular icosahedron. Each vertex
//! of the subdivided mesh becomes a cell center, producing a grid of hexagonal
//! cells (and 12 pentagonal cells at the original icosahedron vertices).
//!
//! ## Algorithm
//!
//! 1. Start with the 12 vertices and 20 triangular faces of a regular
//!    icosahedron inscribed in the unit sphere.
//! 2. Recursively subdivide each triangle into 4 sub-triangles by inserting
//!    midpoints on each edge and projecting them onto the unit sphere.
//! 3. The resulting vertex set defines the cell centers. Neighbors are the
//!    vertices connected by mesh edges, ordered counter-clockwise when viewed
//!    from outside the sphere.
//! 4. Cell areas are computed using the barycentric dual: each face's spherical
//!    area is split equally among its 3 vertices.

use bevy::math::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Unique identifier for a cell in the geodesic grid.
///
/// Internally a `u32` index into the grid's arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CellId(pub u32);

impl CellId {
    /// Convert to array index.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for CellId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cell({})", self.0)
    }
}

/// An icosahedral geodesic grid on the unit sphere.
///
/// Each cell corresponds to a vertex of the subdivided icosahedron. The 12
/// original icosahedron vertices produce pentagonal cells (5 neighbors); all
/// others are hexagonal (6 neighbors).
///
/// Cell count at subdivision level *L* = 10 × 4^L + 2:
/// - Level 0: 12
/// - Level 4: 2,562
/// - Level 6: 40,962
/// - Level 7: 163,842
/// - Level 9: 2,621,442
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcosahedralGrid {
    /// Subdivision level used to construct this grid.
    level: u32,
    /// Unit-sphere position of each cell center, stored as `[x, y, z]`.
    positions: Vec<[f64; 3]>,
    /// Neighbor cell indices for each cell, ordered counter-clockwise
    /// when viewed from outside the sphere.
    neighbors: Vec<Vec<u32>>,
    /// Spherical area of each cell (barycentric dual area on unit sphere).
    areas: Vec<f64>,
}

impl IcosahedralGrid {
    /// Build a new geodesic grid at the given subdivision level.
    ///
    /// Higher levels produce finer grids with exponentially more cells.
    /// Level 7 (~164K cells) is recommended for interactive use; level 9
    /// (~2.6M cells) for production quality.
    ///
    /// # Panics
    ///
    /// Panics if `level > 10` (would exceed `u32` addressing capacity).
    pub fn new(level: u32) -> Self {
        assert!(level <= 10, "Subdivision level {level} too high (max 10)");
        let (vertices, faces) = build_subdivided_icosahedron(level);
        let neighbors = build_ordered_adjacency(&vertices, &faces);
        let areas = compute_barycentric_areas(&vertices, &faces);
        Self {
            level,
            positions: vertices,
            neighbors,
            areas,
        }
    }

    /// Subdivision level used to build this grid.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Total number of cells.
    pub fn cell_count(&self) -> usize {
        self.positions.len()
    }

    /// Expected cell count for a given subdivision level.
    pub fn expected_cell_count(level: u32) -> usize {
        10 * 4_usize.pow(level) + 2
    }

    /// Unit-sphere position of a cell center as a `DVec3`.
    pub fn cell_position(&self, id: CellId) -> DVec3 {
        DVec3::from_array(self.positions[id.index()])
    }

    /// Neighbor cell IDs, ordered counter-clockwise from outside the sphere.
    pub fn cell_neighbors(&self, id: CellId) -> &[u32] {
        &self.neighbors[id.index()]
    }

    /// Number of neighbors (5 for pentagons, 6 for hexagons).
    pub fn cell_neighbor_count(&self, id: CellId) -> usize {
        self.neighbors[id.index()].len()
    }

    /// Whether this cell is one of the 12 pentagonal cells.
    pub fn is_pentagon(&self, id: CellId) -> bool {
        self.neighbors[id.index()].len() == 5
    }

    /// Spherical area of the cell's Voronoi region on the unit sphere.
    pub fn cell_area(&self, id: CellId) -> f64 {
        self.areas[id.index()]
    }

    /// Convert a cell position to (latitude, longitude) in radians.
    ///
    /// - Latitude: −π/2 (south pole) to π/2 (north pole).
    /// - Longitude: −π to π.
    /// - Uses Y-up convention (matching Bevy's coordinate system).
    pub fn cell_lat_lon(&self, id: CellId) -> (f64, f64) {
        let p = self.positions[id.index()];
        // Y is the vertical axis in Bevy's coordinate system.
        let lat = p[1].clamp(-1.0, 1.0).asin();
        let lon = p[0].atan2(p[2]);
        (lat, lon)
    }

    /// Find the nearest cell to a given (latitude, longitude) in radians.
    ///
    /// Uses brute-force O(n) search via dot-product comparison.
    pub fn nearest_cell(&self, lat: f64, lon: f64) -> CellId {
        let target = lat_lon_to_unit_vec(lat, lon);
        let mut best_id = 0_u32;
        let mut best_dot = f64::NEG_INFINITY;
        for (i, pos) in self.positions.iter().enumerate() {
            let d = target[0] * pos[0] + target[1] * pos[1] + target[2] * pos[2];
            if d > best_dot {
                best_dot = d;
                best_id = i as u32;
            }
        }
        CellId(best_id)
    }

    /// Find the nearest cell to a unit-sphere position vector.
    ///
    /// The input is normalized internally; it need not be exactly unit length.
    pub fn nearest_cell_from_pos(&self, pos: DVec3) -> CellId {
        let target = pos.normalize();
        let mut best_id = 0_u32;
        let mut best_dot = f64::NEG_INFINITY;
        for (i, p) in self.positions.iter().enumerate() {
            let d = target.x * p[0] + target.y * p[1] + target.z * p[2];
            if d > best_dot {
                best_dot = d;
                best_id = i as u32;
            }
        }
        CellId(best_id)
    }

    /// Iterator over all cell IDs.
    pub fn cell_ids(&self) -> impl Iterator<Item = CellId> {
        (0..self.positions.len() as u32).map(CellId)
    }

    /// Access the raw position data as `[x, y, z]` slices.
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }

    /// Access the raw neighbor data for all cells.
    pub fn all_neighbors(&self) -> &[Vec<u32>] {
        &self.neighbors
    }

    /// Access the raw area data for all cells.
    pub fn all_areas(&self) -> &[f64] {
        &self.areas
    }
}

// ---------------------------------------------------------------------------
// CellIndex — fast nearest-cell spatial lookup
// ---------------------------------------------------------------------------

/// Grid-based spatial index for fast nearest-cell lookups.
///
/// Partitions the sphere into 1° × 1° latitude/longitude bins. For each
/// query only cells in nearby bins are checked (with a wider longitude window
/// near the poles), giving sub-millisecond performance at all subdivision
/// levels including level 9 (~2.6 M cells).
///
/// Build once with [`CellIndex::build`] and then call [`CellIndex::nearest_cell`]
/// per voxel (or per column) during chunk generation.
pub struct CellIndex {
    lat_bins: usize,
    lon_bins: usize,
    bins: Vec<Vec<u32>>,
    /// Latitude bin search radius, derived from grid cell spacing.
    lat_search: i32,
}

impl CellIndex {
    /// Build a spatial index from the given grid.  O(n) construction.
    pub fn build(grid: &IcosahedralGrid) -> Self {
        const LAT_BINS: usize = 180;
        const LON_BINS: usize = 360;
        let mut bins = vec![Vec::new(); LAT_BINS * LON_BINS];

        for id in grid.cell_ids() {
            let (lat, lon) = grid.cell_lat_lon(id);
            let lb = ((lat + FRAC_PI_2) / PI * LAT_BINS as f64).clamp(0.0, (LAT_BINS - 1) as f64)
                as usize;
            let lob =
                ((lon + PI) / TAU * LON_BINS as f64).clamp(0.0, (LON_BINS - 1) as f64) as usize;
            bins[lb * LON_BINS + lob].push(id.0);
        }

        let n = grid.cell_count();
        // Approximate inter-cell angular spacing in degrees; add margin for
        // safety so the nearest cell is never outside the searched window.
        let spacing_deg = (4.0 * PI / n as f64).sqrt().to_degrees();
        let lat_search = (spacing_deg.ceil() as i32 + 2).max(2);

        Self {
            lat_bins: LAT_BINS,
            lon_bins: LON_BINS,
            bins,
            lat_search,
        }
    }

    /// Find the nearest grid cell to a unit-sphere position vector.
    ///
    /// The vector need not be exactly unit length — it is normalised
    /// internally.  Returns the nearest `CellId` using a dot-product
    /// comparison over the relevant bins.
    pub fn nearest_cell(&self, grid: &IcosahedralGrid, pos: DVec3) -> CellId {
        let target = pos.normalize_or(DVec3::Y);

        // Derive (lat, lon) from the target direction (Y-up convention).
        let lat = target.y.asin().clamp(-FRAC_PI_2, FRAC_PI_2);
        let lon = target.x.atan2(target.z); // atan2(X, Z) — Y-up grid

        let lb = ((lat + FRAC_PI_2) / PI * self.lat_bins as f64)
            .clamp(0.0, (self.lat_bins - 1) as f64) as usize;
        let lob = ((lon + PI) / TAU * self.lon_bins as f64).clamp(0.0, (self.lon_bins - 1) as f64)
            as usize;

        // Widen longitude search near the poles where 1° of lon subtends
        // a very small arc, so cells may fall in far-away lon bins.
        let cos_lat = lat.cos().abs().max(0.01);
        let extra_lon = ((2.0 / cos_lat) as i32)
            .max(2)
            .min(self.lon_bins as i32 / 2);

        let mut best_id = CellId(0);
        let mut best_dot = f64::NEG_INFINITY;

        for dlat in -self.lat_search..=self.lat_search {
            let lat_idx = (lb as i32 + dlat).clamp(0, self.lat_bins as i32 - 1) as usize;
            for dlon in -extra_lon..=extra_lon {
                let lon_idx = (lob as i32 + dlon).rem_euclid(self.lon_bins as i32) as usize;
                for &cell_id in &self.bins[lat_idx * self.lon_bins + lon_idx] {
                    let cpos = grid.cell_position(CellId(cell_id));
                    let dot = target.dot(cpos);
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

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

/// Convert (latitude, longitude) in radians to a unit-sphere `[x, y, z]` vector.
/// Y-up convention: Y = sin(lat), X = cos(lat)·sin(lon), Z = cos(lat)·cos(lon).
fn lat_lon_to_unit_vec(lat: f64, lon: f64) -> [f64; 3] {
    let cos_lat = lat.cos();
    [
        cos_lat * lon.sin(), // X
        lat.sin(),           // Y (up)
        cos_lat * lon.cos(), // Z
    ]
}

/// Build the base icosahedron and recursively subdivide `level` times.
fn build_subdivided_icosahedron(level: u32) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let (mut vertices, mut faces) = base_icosahedron();
    for _ in 0..level {
        (vertices, faces) = subdivide(&vertices, &faces);
    }
    (vertices, faces)
}

/// The 12 vertices and 20 faces of a regular icosahedron on the unit sphere.
///
/// Vertices use the golden-ratio construction: cyclic permutations of
/// (0, ±1, ±φ) normalized to unit length. Faces have outward-facing
/// counter-clockwise winding.
fn base_icosahedron() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = 1.0 / norm;
    let b = phi / norm;

    #[rustfmt::skip]
    let vertices = vec![
        [-a,  b, 0.0], //  0
        [ a,  b, 0.0], //  1
        [-a, -b, 0.0], //  2
        [ a, -b, 0.0], //  3
        [0.0, -a,  b], //  4
        [0.0,  a,  b], //  5
        [0.0, -a, -b], //  6
        [0.0,  a, -b], //  7
        [ b, 0.0, -a], //  8
        [ b, 0.0,  a], //  9
        [-b, 0.0, -a], // 10
        [-b, 0.0,  a], // 11
    ];

    // 20 faces grouped as: 5 around vertex 0, 5 adjacent (upper equatorial),
    // 5 around vertex 3, 5 adjacent (lower equatorial).
    #[rustfmt::skip]
    let faces = vec![
        [0, 11,  5], [0,  5,  1], [0, 1,  7], [0,  7, 10], [0, 10, 11],
        [1,  5,  9], [5, 11,  4], [11, 10, 2], [10, 7,  6], [7,  1,  8],
        [3,  9,  4], [3,  4,  2], [3,  2,  6], [3,  6,  8], [3,  8,  9],
        [4,  9,  5], [2,  4, 11], [6,  2, 10], [8,  6,  7], [9,  8,  1],
    ];

    (vertices, faces)
}

/// Subdivide each triangle into 4 by inserting edge midpoints projected
/// onto the unit sphere.
///
/// Returns the expanded vertex list and the new face list (4× the input).
fn subdivide(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut new_vertices = vertices.to_vec();
    let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_faces = Vec::with_capacity(faces.len() * 4);

    for &[a, b, c] in faces {
        let ab = edge_midpoint(a, b, &mut new_vertices, &mut midpoint_cache);
        let bc = edge_midpoint(b, c, &mut new_vertices, &mut midpoint_cache);
        let ca = edge_midpoint(c, a, &mut new_vertices, &mut midpoint_cache);

        // Split into 4 sub-triangles preserving outward CCW winding.
        new_faces.push([a, ab, ca]);
        new_faces.push([ab, b, bc]);
        new_faces.push([ca, bc, c]);
        new_faces.push([ab, bc, ca]);
    }

    (new_vertices, new_faces)
}

/// Get or create the midpoint vertex for an edge, projected onto the unit sphere.
fn edge_midpoint(
    a: u32,
    b: u32,
    vertices: &mut Vec<[f64; 3]>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = (a.min(b), a.max(b));
    if let Some(&mid) = cache.get(&key) {
        return mid;
    }
    let va = DVec3::from_array(vertices[a as usize]);
    let vb = DVec3::from_array(vertices[b as usize]);
    let mid = ((va + vb) * 0.5).normalize();
    let idx = vertices.len() as u32;
    vertices.push(mid.to_array());
    cache.insert(key, idx);
    idx
}

/// Build adjacency lists with neighbors ordered counter-clockwise
/// when viewed from outside the sphere.
fn build_ordered_adjacency(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<Vec<u32>> {
    let n = vertices.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];

    for &[a, b, c] in faces {
        push_unique(&mut adj[a as usize], b);
        push_unique(&mut adj[a as usize], c);
        push_unique(&mut adj[b as usize], a);
        push_unique(&mut adj[b as usize], c);
        push_unique(&mut adj[c as usize], a);
        push_unique(&mut adj[c as usize], b);
    }

    // Order each neighbor list counter-clockwise around the vertex.
    for (i, neighbors) in adj.iter_mut().enumerate() {
        let center = DVec3::from_array(vertices[i]);
        order_neighbors_ccw(center, neighbors, vertices);
    }

    adj
}

/// Append `val` to `vec` only if not already present.
fn push_unique(vec: &mut Vec<u32>, val: u32) {
    if !vec.contains(&val) {
        vec.push(val);
    }
}

/// Sort neighbor indices counter-clockwise around `center` on the unit sphere.
///
/// Constructs a local tangent frame at `center` and sorts neighbors by their
/// angle in that frame. The tangent frame uses Y-up as the reference direction,
/// falling back to X-up near the poles.
fn order_neighbors_ccw(center: DVec3, neighbors: &mut [u32], vertices: &[[f64; 3]]) {
    if neighbors.len() < 2 {
        return;
    }

    // Build an orthonormal tangent frame {e1, e2} at `center`.
    // Normal = center (outward on unit sphere).
    let up = if center.y.abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let e1 = center.cross(up).normalize();
    let e2 = center.cross(e1);

    // Compute angle of each neighbor in the tangent plane.
    let mut angles: Vec<(u32, f64)> = neighbors
        .iter()
        .map(|&n| {
            let d = DVec3::from_array(vertices[n as usize]) - center;
            let x = d.dot(e1);
            let y = d.dot(e2);
            (n, y.atan2(x))
        })
        .collect();

    angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (i, (id, _)) in angles.into_iter().enumerate() {
        neighbors[i] = id;
    }
}

/// Compute barycentric dual area for each vertex.
///
/// Each face's spherical area is split equally among its 3 vertices. This is
/// a standard approximation of the Voronoi cell area and is exact in the limit
/// of uniform meshes (which the geodesic grid closely approximates).
fn compute_barycentric_areas(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<f64> {
    let n = vertices.len();
    let mut areas = vec![0.0_f64; n];

    for &[a, b, c] in faces {
        let va = DVec3::from_array(vertices[a as usize]);
        let vb = DVec3::from_array(vertices[b as usize]);
        let vc = DVec3::from_array(vertices[c as usize]);
        let face_area = spherical_triangle_area(va, vb, vc);
        let third = face_area / 3.0;
        areas[a as usize] += third;
        areas[b as usize] += third;
        areas[c as usize] += third;
    }

    areas
}

/// Area of a spherical triangle on the unit sphere.
///
/// Uses the Van Oosterom–Strackee formula for the solid angle subtended by
/// three unit vectors:
///
/// Ω = 2 × atan2(|a · (b × c)|, 1 + a·b + b·c + c·a)
fn spherical_triangle_area(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let numerator = a.dot(b.cross(c)).abs();
    let denominator = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * numerator.atan2(denominator)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn cell_count_level_0() {
        let grid = IcosahedralGrid::new(0);
        assert_eq!(grid.cell_count(), 12);
        assert_eq!(grid.cell_count(), IcosahedralGrid::expected_cell_count(0));
    }

    #[test]
    fn cell_count_level_1() {
        let grid = IcosahedralGrid::new(1);
        assert_eq!(grid.cell_count(), 42);
        assert_eq!(grid.cell_count(), IcosahedralGrid::expected_cell_count(1));
    }

    #[test]
    fn cell_count_level_2() {
        let grid = IcosahedralGrid::new(2);
        assert_eq!(grid.cell_count(), 162);
        assert_eq!(grid.cell_count(), IcosahedralGrid::expected_cell_count(2));
    }

    #[test]
    fn cell_count_formula_matches() {
        for level in 0..=5 {
            let grid = IcosahedralGrid::new(level);
            assert_eq!(
                grid.cell_count(),
                IcosahedralGrid::expected_cell_count(level),
                "Cell count mismatch at level {level}"
            );
        }
    }

    #[test]
    fn all_positions_on_unit_sphere() {
        let grid = IcosahedralGrid::new(3);
        for id in grid.cell_ids() {
            let pos = grid.cell_position(id);
            let len = pos.length();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "Cell {id}: length = {len}, expected 1.0"
            );
        }
    }

    #[test]
    fn exactly_12_pentagons() {
        for level in 0..=4 {
            let grid = IcosahedralGrid::new(level);
            let pentagons = grid.cell_ids().filter(|&id| grid.is_pentagon(id)).count();
            assert_eq!(
                pentagons, 12,
                "Level {level}: expected 12 pentagons, got {pentagons}"
            );
        }
    }

    #[test]
    fn neighbor_symmetry() {
        let grid = IcosahedralGrid::new(2);
        for id in grid.cell_ids() {
            for &n in grid.cell_neighbors(id) {
                assert!(
                    grid.cell_neighbors(CellId(n)).contains(&id.0),
                    "{id} lists neighbor {n}, but Cell({n}) does not reciprocate"
                );
            }
        }
    }

    #[test]
    fn neighbor_counts_are_5_or_6() {
        let grid = IcosahedralGrid::new(2);
        for id in grid.cell_ids() {
            let count = grid.cell_neighbor_count(id);
            assert!(
                count == 5 || count == 6,
                "{id}: expected 5 or 6 neighbors, got {count}"
            );
        }
    }

    #[test]
    fn total_area_equals_sphere() {
        let grid = IcosahedralGrid::new(3);
        let total: f64 = grid.all_areas().iter().sum();
        let expected = 4.0 * PI;
        assert!(
            (total - expected).abs() < 1e-6,
            "Total area = {total}, expected {expected} (4π)"
        );
    }

    #[test]
    fn areas_all_positive() {
        let grid = IcosahedralGrid::new(2);
        for id in grid.cell_ids() {
            let area = grid.cell_area(id);
            assert!(area > 0.0, "{id}: area = {area}, expected > 0");
        }
    }

    #[test]
    fn areas_roughly_uniform() {
        let grid = IcosahedralGrid::new(3);
        let areas = grid.all_areas();
        let mean = areas.iter().sum::<f64>() / areas.len() as f64;
        for (i, &area) in areas.iter().enumerate() {
            let ratio = area / mean;
            assert!(
                (0.75..=1.25).contains(&ratio),
                "Cell {i}: area ratio = {ratio:.4}, expected near 1.0"
            );
        }
    }

    #[test]
    fn nearest_cell_round_trips() {
        let grid = IcosahedralGrid::new(2);
        for id in grid.cell_ids() {
            let (lat, lon) = grid.cell_lat_lon(id);
            let found = grid.nearest_cell(lat, lon);
            assert_eq!(
                found, id,
                "Round-trip failed: {id} → ({lat:.6}, {lon:.6}) → {found}"
            );
        }
    }

    #[test]
    fn lat_lon_in_valid_range() {
        let grid = IcosahedralGrid::new(2);
        for id in grid.cell_ids() {
            let (lat, lon) = grid.cell_lat_lon(id);
            assert!(
                (-PI / 2.0..=PI / 2.0).contains(&lat),
                "{id}: lat = {lat}, out of range"
            );
            assert!((-PI..=PI).contains(&lon), "{id}: lon = {lon}, out of range");
        }
    }

    #[test]
    fn base_icosahedron_uniform_edge_lengths() {
        let (vertices, faces) = base_icosahedron();
        let mut edge_lengths = Vec::new();
        for &[a, b, c] in &faces {
            for &(i, j) in &[(a, b), (b, c), (c, a)] {
                let va = DVec3::from_array(vertices[i as usize]);
                let vb = DVec3::from_array(vertices[j as usize]);
                edge_lengths.push((va - vb).length());
            }
        }
        let first = edge_lengths[0];
        for (i, &len) in edge_lengths.iter().enumerate() {
            assert!(
                (len - first).abs() < 1e-10,
                "Edge {i}: length = {len}, expected {first}"
            );
        }
    }

    #[test]
    fn base_icosahedron_outward_normals() {
        let (vertices, faces) = base_icosahedron();
        for (i, &[a, b, c]) in faces.iter().enumerate() {
            let va = DVec3::from_array(vertices[a as usize]);
            let vb = DVec3::from_array(vertices[b as usize]);
            let vc = DVec3::from_array(vertices[c as usize]);
            let centroid = (va + vb + vc) / 3.0;
            let normal = (vb - va).cross(vc - va);
            assert!(
                normal.dot(centroid) > 0.0,
                "Face {i} [{a}, {b}, {c}]: normal points inward"
            );
        }
    }

    #[test]
    fn nearest_cell_from_pos_matches_lat_lon() {
        let grid = IcosahedralGrid::new(2);
        // Test a few known directions.
        let north = DVec3::new(0.0, 1.0, 0.0);
        let south = DVec3::new(0.0, -1.0, 0.0);
        let n_cell = grid.nearest_cell_from_pos(north);
        let s_cell = grid.nearest_cell_from_pos(south);
        // They should differ.
        assert_ne!(
            n_cell, s_cell,
            "North and south poles should map to different cells"
        );
        // North pole cell should have high latitude.
        let (lat, _) = grid.cell_lat_lon(n_cell);
        assert!(
            lat > 0.5,
            "North pole cell latitude = {lat}, expected > 0.5"
        );
    }

    #[test]
    fn cell_index_nearest_matches_brute_force() {
        let grid = IcosahedralGrid::new(3);
        let index = CellIndex::build(&grid);
        // Test a handful of positions around the sphere.
        let test_positions = [
            DVec3::Y,
            DVec3::NEG_Y,
            DVec3::X,
            DVec3::NEG_X,
            DVec3::Z,
            DVec3::new(0.6, 0.6, 0.5).normalize(),
        ];
        for pos in test_positions {
            let fast = index.nearest_cell(&grid, pos);
            let brute = grid.nearest_cell_from_pos(pos);
            assert_eq!(
                fast, brute,
                "CellIndex mismatch at {pos}: fast={fast}, brute={brute}"
            );
        }
    }

    #[test]
    fn cell_index_round_trips_all_cells() {
        let grid = IcosahedralGrid::new(2);
        let index = CellIndex::build(&grid);
        for id in grid.cell_ids() {
            let pos = grid.cell_position(id);
            let found = index.nearest_cell(&grid, pos);
            assert_eq!(found, id, "CellIndex round-trip failed: {id} → {found}");
        }
    }
}
