// Integration tests verifying all RON asset files in assets/data/ are valid.
// These tests parse every RON file directly (without Bevy's asset pipeline)
// to catch data errors early, before they become runtime panics.

use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct EnemyData {
    name: String,
    health: f32,
    speed: f32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
enum Phase {
    Solid,
    Liquid,
    Gas,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct MaterialData {
    id: u16,
    name: String,
    default_phase: Phase,
    density: f32,
    melting_point: Option<f32>,
    boiling_point: Option<f32>,
    ignition_point: Option<f32>,
    hardness: f32,
    color: [f32; 3],
    transparent: bool,
}

/// Validate that every .enemy.ron file in assets/data/ deserializes correctly.
#[test]
fn all_enemy_ron_files_are_valid() {
    let pattern = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/data/**/*.enemy.ron");
    let files: Vec<_> = glob::glob(pattern)
        .expect("Failed to read glob pattern")
        .collect();

    assert!(
        !files.is_empty(),
        "No .enemy.ron files found in assets/data/"
    );

    for entry in files {
        let path = entry.expect("Failed to read glob entry");
        let contents = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        let _data: EnemyData = ron::from_str(&contents)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    }
}

/// Validate that every .material.ron file in assets/data/materials/ deserializes correctly.
#[test]
fn all_material_ron_files_are_valid() {
    let pattern = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/data/materials/*.material.ron"
    );
    let files: Vec<_> = glob::glob(pattern)
        .expect("Failed to read glob pattern")
        .collect();

    assert!(
        !files.is_empty(),
        "No .material.ron files found in assets/data/materials/"
    );

    for entry in files {
        let path = entry.expect("Failed to read glob entry");
        let contents = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        let _data: MaterialData = ron::from_str(&contents)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    }
}
