// Death and decomposition: entity removal → corpse voxels → decay → nutrients.
//
// When a creature dies, it is converted into organic material voxels at its
// position. These voxels then decay over time, releasing nutrients into the
// soil (modifying neighbor voxel properties). This closes the biology→chemistry
// feedback loop.

#![allow(dead_code)]

use crate::data::BodySize;
use crate::world::voxel::{MaterialId, Voxel};

/// Material ID for organic matter (decaying corpse material).
/// We reuse a convention: organic matter is a new material.
pub const ORGANIC_MATTER: MaterialId = MaterialId(12);

/// How many voxels a corpse occupies based on body size.
pub fn corpse_voxel_count(size: BodySize) -> usize {
    match size {
        BodySize::Tiny => 1,
        BodySize::Small => 2,
        BodySize::Medium => 4,
        BodySize::Large => 8,
        BodySize::Huge => 16,
    }
}

/// State of a decomposing corpse in the voxel world.
#[derive(Debug, Clone)]
pub struct CorpseState {
    /// Voxel positions (x, y, z) that this corpse occupies.
    pub positions: Vec<(usize, usize, usize)>,
    /// Remaining decay ticks before the corpse dissolves.
    pub decay_remaining: u32,
    /// Total nutrients to release when fully decayed.
    pub nutrient_value: f32,
}

impl CorpseState {
    pub fn new(positions: Vec<(usize, usize, usize)>, size: BodySize) -> Self {
        let (decay_time, nutrients) = match size {
            BodySize::Tiny => (200, 10.0),
            BodySize::Small => (400, 25.0),
            BodySize::Medium => (800, 50.0),
            BodySize::Large => (1200, 100.0),
            BodySize::Huge => (2000, 200.0),
        };
        Self {
            positions,
            decay_remaining: decay_time,
            nutrient_value: nutrients,
        }
    }

    /// Whether the corpse has fully decayed.
    pub fn is_decayed(&self) -> bool {
        self.decay_remaining == 0
    }
}

/// Place corpse voxels at the given position in a voxel grid.
/// Returns the positions that were placed.
pub fn place_corpse(
    voxels: &mut [Voxel],
    size: usize,
    center_x: usize,
    center_y: usize,
    center_z: usize,
    body_size: BodySize,
) -> Vec<(usize, usize, usize)> {
    let count = corpse_voxel_count(body_size);
    let mut placed = Vec::new();

    // Place corpse voxels in a small cluster around the center
    let offsets: [(i32, i32, i32); 16] = [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 0, 1),
        (-1, 0, -1),
        (0, 1, 0),
        (1, 0, -1),
        (-1, 0, 1),
        (0, -1, 0),
        (1, 1, 0),
        (-1, 1, 0),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
    ];

    for &(dx, dy, dz) in offsets.iter().take(count) {
        let nx = center_x as i32 + dx;
        let ny = center_y as i32 + dy;
        let nz = center_z as i32 + dz;

        if nx < 0 || ny < 0 || nz < 0 {
            continue;
        }
        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
        if nx >= size || ny >= size || nz >= size {
            continue;
        }

        let idx = nz * size * size + ny * size + nx;
        if voxels[idx].material.is_air() {
            voxels[idx].material = ORGANIC_MATTER;
            placed.push((nx, ny, nz));
        }
    }

    placed
}

/// Tick decay for a corpse. Returns nutrients released this tick.
/// When fully decayed, the corpse voxels should be removed (converted to air).
pub fn tick_decay(corpse: &mut CorpseState) -> f32 {
    if corpse.is_decayed() {
        return 0.0;
    }

    corpse.decay_remaining = corpse.decay_remaining.saturating_sub(1);

    if corpse.is_decayed() {
        // Release all remaining nutrients on final tick
        corpse.nutrient_value
    } else {
        // Release nutrients gradually
        let per_tick = corpse.nutrient_value / (corpse.decay_remaining as f32 + 1.0);
        corpse.nutrient_value -= per_tick;
        per_tick
    }
}

/// Remove decayed corpse voxels from the grid (convert back to air).
pub fn remove_corpse(voxels: &mut [Voxel], size: usize, positions: &[(usize, usize, usize)]) {
    for &(x, y, z) in positions {
        let idx = z * size * size + y * size + x;
        if idx < voxels.len() && voxels[idx].material == ORGANIC_MATTER {
            voxels[idx] = Voxel::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(size: usize) -> Vec<Voxel> {
        vec![Voxel::default(); size * size * size]
    }

    #[test]
    fn corpse_voxel_count_scales_with_size() {
        assert!(corpse_voxel_count(BodySize::Huge) > corpse_voxel_count(BodySize::Tiny));
    }

    #[test]
    fn place_corpse_creates_organic_voxels() {
        let mut grid = make_grid(8);
        let positions = place_corpse(&mut grid, 8, 4, 4, 4, BodySize::Medium);
        assert!(!positions.is_empty());
        for &(x, y, z) in &positions {
            let idx = z * 64 + y * 8 + x;
            assert_eq!(grid[idx].material, ORGANIC_MATTER);
        }
    }

    #[test]
    fn place_corpse_respects_bounds() {
        let mut grid = make_grid(4);
        // Corner placement — should not panic
        let positions = place_corpse(&mut grid, 4, 0, 0, 0, BodySize::Large);
        assert!(!positions.is_empty());
        for &(x, y, z) in &positions {
            assert!(x < 4 && y < 4 && z < 4);
        }
    }

    #[test]
    fn place_corpse_only_in_air() {
        let mut grid = make_grid(8);
        // Fill center with stone
        let idx = 4 * 64 + 4 * 8 + 4;
        grid[idx].material = MaterialId::STONE;

        let positions = place_corpse(&mut grid, 8, 4, 4, 4, BodySize::Medium);
        // Should not overwrite stone
        assert_eq!(grid[idx].material, MaterialId::STONE);
        assert!(!positions.contains(&(4, 4, 4)));
    }

    #[test]
    fn decay_reduces_remaining() {
        let mut corpse = CorpseState::new(vec![(4, 4, 4)], BodySize::Tiny);
        let initial = corpse.decay_remaining;
        tick_decay(&mut corpse);
        assert_eq!(corpse.decay_remaining, initial - 1);
    }

    #[test]
    fn decay_releases_nutrients() {
        let mut corpse = CorpseState::new(vec![(4, 4, 4)], BodySize::Small);
        let nutrients = tick_decay(&mut corpse);
        assert!(nutrients > 0.0);
    }

    #[test]
    fn fully_decayed_releases_remaining_nutrients() {
        let mut corpse = CorpseState::new(vec![(4, 4, 4)], BodySize::Tiny);
        corpse.decay_remaining = 1;
        let remaining = corpse.nutrient_value;
        let nutrients = tick_decay(&mut corpse);
        assert!(corpse.is_decayed());
        // Should release whatever was left
        assert!(nutrients > 0.0);
        assert!(nutrients <= remaining + 0.01);
    }

    #[test]
    fn remove_corpse_clears_voxels() {
        let mut grid = make_grid(8);
        let positions = place_corpse(&mut grid, 8, 4, 4, 4, BodySize::Medium);
        remove_corpse(&mut grid, 8, &positions);
        for &(x, y, z) in &positions {
            let idx = z * 64 + y * 8 + x;
            assert!(grid[idx].material.is_air());
        }
    }

    #[test]
    fn remove_corpse_only_removes_organic() {
        let mut grid = make_grid(8);
        let idx = 4 * 64 + 4 * 8 + 4;
        grid[idx].material = MaterialId::STONE;
        // Try to remove at stone position
        remove_corpse(&mut grid, 8, &[(4, 4, 4)]);
        assert_eq!(grid[idx].material, MaterialId::STONE);
    }
}
