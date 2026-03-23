// Data-driven scenario tests.
//
// Each sub-directory under `tests/cases/` holds `.scenario.ron` files for one
// category of test (physics, chemistry, terrain). This file discovers and runs
// all of them automatically — to add a new test, just drop a new RON file into
// the appropriate directory.

use serde::Deserialize;

use the_dark_candle::chemistry::reactions::{check_reaction, ReactionData};
use the_dark_candle::data::MaterialRegistry;
use the_dark_candle::physics::gravity::{GRAVITY, TERMINAL_VELOCITY};
use the_dark_candle::world::chunk::{Chunk, ChunkCoord, CHUNK_VOLUME};
use the_dark_candle::world::terrain::{TerrainConfig, TerrainGenerator};
use the_dark_candle::world::voxel::MaterialId;

// ── Scenario type definitions ─────────────────────────────────────────────────

/// Pure-math gravity scenario: simulates the integration loop without ECS.
#[derive(Deserialize, Debug)]
struct PhysicsScenario {
    description: String,
    initial_velocity_y: f32,
    gravity_scale: f32,
    /// Number of fixed-step iterations to simulate.
    steps: u32,
    /// Simulated seconds per step.
    delta_secs: f32,
    /// Expected velocity_y at the end (compared with `tolerance`).
    expect_velocity_y: f32,
    tolerance: f32,
}

/// Inline chemistry reaction scenario: embeds the rule directly in the file.
#[derive(Deserialize, Debug)]
struct ChemistryScenario {
    description: String,
    /// Name-to-ID mapping for building a MaterialRegistry in the test.
    materials: std::collections::HashMap<String, u16>,
    rule: ReactionData,
    material_a: u16,
    material_b: u16,
    temperature: f32,
    expect_reacts: bool,
    /// Expected `new_material_a` ID if reaction occurs.
    expect_output_a: Option<u16>,
}

/// Terrain generation scenario: runs the generator and checks voxel statistics.
#[derive(Deserialize, Debug)]
struct TerrainScenario {
    description: String,
    config: TerrainConfig,
    chunk_coord: (i32, i32, i32),
    expect_has_air: Option<bool>,
    expect_has_solid: Option<bool>,
    expect_solid_fraction_gt: Option<f32>,
    expect_solid_fraction_lt: Option<f32>,
}

// ── Generic discovery runner ──────────────────────────────────────────────────

fn run_all<T, F>(glob_pattern: &str, runner: F)
where
    T: serde::de::DeserializeOwned + std::fmt::Debug,
    F: Fn(&T) -> Result<(), String>,
{
    let entries: Vec<_> = glob::glob(glob_pattern)
        .expect("invalid glob pattern")
        .collect();

    assert!(
        !entries.is_empty(),
        "No scenario files found matching {glob_pattern}"
    );

    let mut failures = Vec::new();
    for entry in entries {
        let path = entry.expect("glob entry error");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let scenario: T =
            ron::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()));
        if let Err(msg) = runner(&scenario) {
            failures.push(format!("{}: {msg}", path.display()));
        }
    }

    if !failures.is_empty() {
        panic!("Scenario failures:\n{}", failures.join("\n"));
    }
}

// ── Category-specific runners ─────────────────────────────────────────────────

fn run_physics(s: &PhysicsScenario) -> Result<(), String> {
    let mut vel = s.initial_velocity_y;
    for _ in 0..s.steps {
        vel -= GRAVITY * s.gravity_scale * s.delta_secs;
        vel = vel.max(-TERMINAL_VELOCITY);
    }
    if !approx::abs_diff_eq!(vel, s.expect_velocity_y, epsilon = s.tolerance) {
        return Err(format!(
            "{}: expected velocity_y ≈ {} (±{}), got {vel}",
            s.description, s.expect_velocity_y, s.tolerance
        ));
    }
    Ok(())
}

fn run_chemistry(s: &ChemistryScenario) -> Result<(), String> {
    use the_dark_candle::data::{MaterialData, Phase};

    let mut registry = MaterialRegistry::new();
    for (name, &id) in &s.materials {
        registry.insert(MaterialData {
            id,
            name: name.clone(),
            default_phase: Phase::Solid,
            density: 1000.0,
            melting_point: None,
            boiling_point: None,
            ignition_point: None,
            hardness: 0.5,
            color: [0.5, 0.5, 0.5],
            transparent: false,
            melted_into: None,
            boiled_into: None,
            frozen_into: None,
            condensed_into: None,
        });
    }

    let result = check_reaction(
        &s.rule,
        MaterialId(s.material_a),
        MaterialId(s.material_b),
        s.temperature,
        &registry,
    );

    let reacted = result.is_some();
    if reacted != s.expect_reacts {
        return Err(format!(
            "{}: expected reacts={}, got {}",
            s.description, s.expect_reacts, reacted
        ));
    }

    if let Some(expected_id) = s.expect_output_a {
        let actual_id = result
            .as_ref()
            .map(|r| r.new_material_a.0)
            .unwrap_or(s.material_a);
        if actual_id != expected_id {
            return Err(format!(
                "{}: expected output_a={expected_id}, got {actual_id}",
                s.description
            ));
        }
    }

    Ok(())
}

fn run_terrain(s: &TerrainScenario) -> Result<(), String> {
    let (cx, cy, cz) = s.chunk_coord;
    let coord = ChunkCoord::new(cx, cy, cz);
    let generator = TerrainGenerator::new(s.config.clone());
    let mut chunk = Chunk::new_empty(coord);
    generator.generate_chunk(&mut chunk);

    let solid_count = chunk
        .voxels()
        .iter()
        .filter(|v| !v.material.is_air())
        .count();
    let air_count = CHUNK_VOLUME - solid_count;
    let solid_fraction = solid_count as f32 / CHUNK_VOLUME as f32;

    if let Some(expected) = s.expect_has_air {
        let has_air = air_count > 0;
        if has_air != expected {
            return Err(format!(
                "{}: expect_has_air={expected}, got {has_air} (air voxels: {air_count})",
                s.description
            ));
        }
    }

    if let Some(expected) = s.expect_has_solid {
        let has_solid = solid_count > 0;
        if has_solid != expected {
            return Err(format!(
                "{}: expect_has_solid={expected}, got {has_solid} (solid voxels: {solid_count})",
                s.description
            ));
        }
    }

    if let Some(min_frac) = s.expect_solid_fraction_gt {
        if solid_fraction <= min_frac {
            return Err(format!(
                "{}: expected solid fraction > {min_frac}, got {solid_fraction:.4}",
                s.description
            ));
        }
    }

    if let Some(max_frac) = s.expect_solid_fraction_lt {
        if solid_fraction >= max_frac {
            return Err(format!(
                "{}: expected solid fraction < {max_frac}, got {solid_fraction:.4}",
                s.description
            ));
        }
    }

    Ok(())
}

// ── Test entrypoints ──────────────────────────────────────────────────────────

#[test]
fn all_physics_scenarios() {
    let pattern = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/cases/physics/*.scenario.ron"
    );
    run_all(pattern, run_physics);
}

#[test]
fn all_chemistry_scenarios() {
    let pattern = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/cases/chemistry/*.scenario.ron"
    );
    run_all(pattern, run_chemistry);
}

#[test]
fn all_terrain_scenarios() {
    let pattern = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/cases/terrain/*.scenario.ron"
    );
    run_all(pattern, run_terrain);
}
