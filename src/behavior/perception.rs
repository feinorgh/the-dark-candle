// Perception system: sight, hearing, smell as range-limited queries.
//
// Each sense has a different range and penetration model:
// - Sight: long range, blocked by solid voxels (line-of-sight)
// - Hearing: medium range, passes through walls (attenuated)
// - Smell: short range, only through air/gas (follows airflow)
//
// Perception generates sensory events that drive the needs and utility systems.

use bevy::prelude::*;

/// Sensory capabilities of a creature.
#[derive(Component, Debug, Clone)]
pub struct Senses {
    /// Maximum sight range in voxels.
    pub sight_range: f32,
    /// Maximum hearing range in voxels.
    pub hearing_range: f32,
    /// Maximum smell range in voxels.
    pub smell_range: f32,
}

impl Default for Senses {
    fn default() -> Self {
        Self {
            sight_range: 32.0,
            hearing_range: 20.0,
            smell_range: 12.0,
        }
    }
}

/// A perceived entity in the world.
#[derive(Debug, Clone)]
pub struct Percept {
    /// Position of the perceived entity.
    pub position: [f32; 3],
    /// How the entity was detected.
    pub sense: SenseType,
    /// What kind of thing was perceived.
    pub kind: PerceptKind,
    /// Distance from perceiver to target.
    pub distance: f32,
    /// Signal strength (0.0–1.0, decreases with distance and occlusion).
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenseType {
    Sight,
    Hearing,
    Smell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptKind {
    Creature,
    Food,
    Threat,
    Item,
}

/// Trait abstracting voxel lookups for line-of-sight checks.
pub trait PerceptionGrid {
    fn is_opaque(&self, x: i32, y: i32, z: i32) -> bool;
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool;
}

/// Check line-of-sight between two positions using 3D Bresenham ray marching.
/// Returns true if the line is unobstructed (no opaque voxels in the way).
pub fn has_line_of_sight(grid: &dyn PerceptionGrid, from: [f32; 3], to: [f32; 3]) -> bool {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let dz = to[2] - from[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist < 0.5 {
        return true;
    }

    let steps = dist.ceil() as i32;
    let step_x = dx / steps as f32;
    let step_y = dy / steps as f32;
    let step_z = dz / steps as f32;

    // Check intermediate positions (skip start, skip end)
    for i in 1..steps {
        let x = (from[0] + step_x * i as f32).floor() as i32;
        let y = (from[1] + step_y * i as f32).floor() as i32;
        let z = (from[2] + step_z * i as f32).floor() as i32;
        if grid.is_opaque(x, y, z) {
            return false;
        }
    }

    true
}

/// Calculate sight perception strength (decreases with distance, 0 if blocked).
pub fn sight_strength(
    grid: &dyn PerceptionGrid,
    from: [f32; 3],
    to: [f32; 3],
    sight_range: f32,
) -> f32 {
    let dist = point_distance(from, to);
    if dist > sight_range {
        return 0.0;
    }
    if !has_line_of_sight(grid, from, to) {
        return 0.0;
    }
    // Linear falloff with distance
    1.0 - (dist / sight_range)
}

/// Calculate hearing perception strength (passes through walls, attenuated).
/// Each solid voxel between attenuates by `wall_attenuation`.
pub fn hearing_strength(
    grid: &dyn PerceptionGrid,
    from: [f32; 3],
    to: [f32; 3],
    hearing_range: f32,
    wall_attenuation: f32,
) -> f32 {
    let dist = point_distance(from, to);
    if dist > hearing_range {
        return 0.0;
    }

    let base_strength = 1.0 - (dist / hearing_range);

    // Count solid voxels between
    let steps = dist.ceil() as i32;
    if steps == 0 {
        return base_strength;
    }

    let dx = (to[0] - from[0]) / steps as f32;
    let dy = (to[1] - from[1]) / steps as f32;
    let dz = (to[2] - from[2]) / steps as f32;

    let mut walls = 0;
    for i in 1..steps {
        let x = (from[0] + dx * i as f32).floor() as i32;
        let y = (from[1] + dy * i as f32).floor() as i32;
        let z = (from[2] + dz * i as f32).floor() as i32;
        if grid.is_solid(x, y, z) {
            walls += 1;
        }
    }

    (base_strength * wall_attenuation.powi(walls)).max(0.0)
}

/// Calculate smell perception strength (only through air/non-solid voxels).
pub fn smell_strength(
    grid: &dyn PerceptionGrid,
    from: [f32; 3],
    to: [f32; 3],
    smell_range: f32,
) -> f32 {
    let dist = point_distance(from, to);
    if dist > smell_range {
        return 0.0;
    }

    // Smell is blocked by any solid voxel
    let steps = dist.ceil() as i32;
    if steps == 0 {
        return 1.0;
    }

    let dx = (to[0] - from[0]) / steps as f32;
    let dy = (to[1] - from[1]) / steps as f32;
    let dz = (to[2] - from[2]) / steps as f32;

    for i in 1..steps {
        let x = (from[0] + dx * i as f32).floor() as i32;
        let y = (from[1] + dy * i as f32).floor() as i32;
        let z = (from[2] + dz * i as f32).floor() as i32;
        if grid.is_solid(x, y, z) {
            return 0.0;
        }
    }

    1.0 - (dist / smell_range)
}

/// Calculate threat level from a list of perceived threats.
/// Returns 0.0 (no threats) to 1.0 (extreme danger).
pub fn compute_threat_level(threats: &[Percept]) -> f32 {
    if threats.is_empty() {
        return 0.0;
    }

    // Strongest threat signal dominates, with slight additive from others
    let max_strength = threats.iter().map(|t| t.strength).fold(0.0f32, f32::max);

    let count_bonus = (threats.len() as f32 - 1.0) * 0.1;

    (max_strength + count_bonus).min(1.0)
}

fn point_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPerceptionGrid {
        size: i32,
        voxels: Vec<bool>, // true = solid/opaque
    }

    impl TestPerceptionGrid {
        fn new(size: i32) -> Self {
            Self {
                size,
                voxels: vec![false; (size * size * size) as usize],
            }
        }

        fn set_solid(&mut self, x: i32, y: i32, z: i32) {
            if x >= 0 && x < self.size && y >= 0 && y < self.size && z >= 0 && z < self.size {
                let idx = (y * self.size * self.size + z * self.size + x) as usize;
                self.voxels[idx] = true;
            }
        }
    }

    impl PerceptionGrid for TestPerceptionGrid {
        fn is_opaque(&self, x: i32, y: i32, z: i32) -> bool {
            if x >= 0 && x < self.size && y >= 0 && y < self.size && z >= 0 && z < self.size {
                self.voxels[(y * self.size * self.size + z * self.size + x) as usize]
            } else {
                false
            }
        }

        fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
            self.is_opaque(x, y, z)
        }
    }

    #[test]
    fn clear_line_of_sight() {
        let grid = TestPerceptionGrid::new(16);
        assert!(has_line_of_sight(&grid, [1.0, 1.0, 1.0], [10.0, 1.0, 1.0]));
    }

    #[test]
    fn blocked_line_of_sight() {
        let mut grid = TestPerceptionGrid::new(16);
        grid.set_solid(5, 1, 1);
        assert!(!has_line_of_sight(&grid, [1.0, 1.0, 1.0], [10.0, 1.0, 1.0]));
    }

    #[test]
    fn sight_zero_when_blocked() {
        let mut grid = TestPerceptionGrid::new(16);
        grid.set_solid(5, 1, 1);
        let s = sight_strength(&grid, [1.0, 1.0, 1.0], [10.0, 1.0, 1.0], 32.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn sight_decreases_with_distance() {
        let grid = TestPerceptionGrid::new(32);
        let near = sight_strength(&grid, [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], 32.0);
        let far = sight_strength(&grid, [0.0, 0.0, 0.0], [20.0, 0.0, 0.0], 32.0);
        assert!(near > far);
        assert!(near > 0.0);
        assert!(far > 0.0);
    }

    #[test]
    fn sight_zero_beyond_range() {
        let grid = TestPerceptionGrid::new(16);
        let s = sight_strength(&grid, [0.0, 0.0, 0.0], [50.0, 0.0, 0.0], 32.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn hearing_passes_through_walls() {
        let mut grid = TestPerceptionGrid::new(16);
        grid.set_solid(5, 1, 1);
        let s = hearing_strength(&grid, [1.0, 1.0, 1.0], [10.0, 1.0, 1.0], 20.0, 0.5);
        assert!(s > 0.0); // still heard, but attenuated
    }

    #[test]
    fn hearing_attenuated_by_walls() {
        let grid_clear = TestPerceptionGrid::new(16);
        let mut grid_wall = TestPerceptionGrid::new(16);
        grid_wall.set_solid(5, 1, 1);

        let s_clear = hearing_strength(&grid_clear, [1.0, 1.0, 1.0], [10.0, 1.0, 1.0], 20.0, 0.5);
        let s_wall = hearing_strength(&grid_wall, [1.0, 1.0, 1.0], [10.0, 1.0, 1.0], 20.0, 0.5);
        assert!(s_clear > s_wall);
    }

    #[test]
    fn smell_blocked_by_solid() {
        let mut grid = TestPerceptionGrid::new(16);
        grid.set_solid(5, 1, 1);
        let s = smell_strength(&grid, [1.0, 1.0, 1.0], [10.0, 1.0, 1.0], 12.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn smell_works_in_clear_air() {
        let grid = TestPerceptionGrid::new(16);
        let s = smell_strength(&grid, [1.0, 1.0, 1.0], [5.0, 1.0, 1.0], 12.0);
        assert!(s > 0.0);
    }

    #[test]
    fn threat_level_from_no_threats() {
        assert_eq!(compute_threat_level(&[]), 0.0);
    }

    #[test]
    fn threat_level_from_single_threat() {
        let threats = vec![Percept {
            position: [5.0, 1.0, 5.0],
            sense: SenseType::Sight,
            kind: PerceptKind::Threat,
            distance: 5.0,
            strength: 0.7,
        }];
        let level = compute_threat_level(&threats);
        assert!((level - 0.7).abs() < 0.001);
    }

    #[test]
    fn threat_level_increases_with_multiple_threats() {
        let threats = vec![
            Percept {
                position: [5.0, 1.0, 5.0],
                sense: SenseType::Sight,
                kind: PerceptKind::Threat,
                distance: 5.0,
                strength: 0.5,
            },
            Percept {
                position: [3.0, 1.0, 3.0],
                sense: SenseType::Hearing,
                kind: PerceptKind::Threat,
                distance: 3.0,
                strength: 0.6,
            },
        ];
        let level = compute_threat_level(&threats);
        assert!(level > 0.6); // more than just the max
        assert!(level <= 1.0);
    }

    #[test]
    fn threat_level_caps_at_one() {
        let threats: Vec<Percept> = (0..10)
            .map(|i| Percept {
                position: [i as f32, 1.0, 0.0],
                sense: SenseType::Sight,
                kind: PerceptKind::Threat,
                distance: 2.0,
                strength: 0.9,
            })
            .collect();
        let level = compute_threat_level(&threats);
        assert!(level <= 1.0);
    }
}
