// Data-driven simulation scenario tests.
//
// Each `.simulation.ron` file in `tests/cases/simulation/` defines a complete
// multi-tick simulation test: grid setup, materials, reactions, ignition,
// simulation parameters, and assertions.
//
// To add a new test, create a new `.simulation.ron` file in that directory.
// It will be auto-discovered and executed.

use the_dark_candle::simulation::scenario::{SimulationScenario, run_scenario};

/// Discover and run all `.simulation.ron` files.
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
        panic!("Simulation scenario failures:\n{}", failures.join("\n"));
    }
}

#[test]
fn all_simulation_scenarios() {
    let pattern = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/cases/simulation/*.simulation.ron"
    );
    run_all::<SimulationScenario, _>(pattern, run_scenario);
}
