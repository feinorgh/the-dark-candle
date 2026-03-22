// Integration tests verifying all RON asset files in assets/data/ are valid.
// These tests parse every RON file directly (without Bevy's asset pipeline)
// to catch data errors early, before they become runtime panics.

/// Validate that every .enemy.ron file in assets/data/ deserializes correctly.
#[test]
fn all_enemy_ron_files_are_valid() {
    use serde::Deserialize;

    #[derive(Deserialize, Debug)]
    #[allow(dead_code)]
    struct EnemyData {
        name: String,
        health: f32,
        speed: f32,
    }

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
