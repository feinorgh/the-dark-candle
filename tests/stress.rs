//! Gameplay stress / fuzzy regression tests.
//!
//! Discovers `*.stress.ron` files under `tests/cases/stress/` and runs each as
//! a `StressScenario` against the headless `StressApp` harness. Also hosts
//! proptest-driven property generators (added in later tasks).

use serde::Deserialize;

use the_dark_candle::test_support::stress::{
    InvariantFailure, InvariantSet, PlanetPreset, StressApp,
};

#[derive(Deserialize, Debug)]
struct StressScenario {
    description: String,
    seed: u64,
    planet: PresetSpec,
    #[serde(default = "default_warmup_ticks")]
    warmup_ticks: u32,
    teleports: Vec<Teleport>,
    invariants: Vec<InvariantSpec>,
    #[serde(default)]
    chunk_load_rate_min: Option<f32>,
}

#[derive(Deserialize, Debug, Clone, Copy)]
enum PresetSpec {
    Earth,
    SmallPlanet,
}

fn default_warmup_ticks() -> u32 {
    30
}

#[derive(Deserialize, Debug)]
struct Teleport {
    lat: f64,
    lon: f64,
    altitude_m: f64,
    /// Number of ticks to advance after the teleport before the next one.
    then_tick: u32,
}

#[derive(Deserialize, Debug, Clone, Copy)]
enum InvariantSpec {
    Panics,
    Finite,
    NoOverflow,
    ChunkCache,
    LoadRate,
}

fn preset_to_harness(p: PresetSpec) -> PlanetPreset {
    match p {
        PresetSpec::Earth => PlanetPreset::Earth,
        PresetSpec::SmallPlanet => PlanetPreset::SmallPlanet,
    }
}

fn invariants_to_set(specs: &[InvariantSpec]) -> InvariantSet {
    let mut s = InvariantSet::default();
    for sp in specs {
        s = s | match sp {
            InvariantSpec::Panics => InvariantSet::PANICS,
            InvariantSpec::Finite => InvariantSet::FINITE,
            InvariantSpec::NoOverflow => InvariantSet::NO_OVERFLOW,
            InvariantSpec::ChunkCache => InvariantSet::CHUNK_CACHE,
            InvariantSpec::LoadRate => InvariantSet::LOAD_RATE,
        };
    }
    s
}

fn run_scenario(scenario: &StressScenario, path: &std::path::Path) -> Result<(), String> {
    let mut app = StressApp::new(scenario.seed, preset_to_harness(scenario.planet));
    app.tick_n(scenario.warmup_ticks);

    for tp in &scenario.teleports {
        app.teleport(tp.lat, tp.lon, tp.altitude_m);
        app.tick_n(tp.then_tick);
    }

    let which = invariants_to_set(&scenario.invariants);
    let failures = app.assert_invariants_with_min_rate(which, scenario.chunk_load_rate_min);

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Scenario failed: {}\n  description: {}\n  failures:\n{}",
            path.display(),
            scenario.description,
            failures
                .iter()
                .map(|f| format!("    - {f:?}"))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
}

#[test]
fn run_all_stress_scenarios() {
    let pattern = "tests/cases/stress/*.stress.ron";
    let entries: Vec<_> = glob::glob(pattern).expect("invalid glob pattern").collect();

    assert!(
        !entries.is_empty(),
        "No stress scenarios found matching {pattern}"
    );

    let mut errors = Vec::new();
    for entry in entries {
        let path = entry.expect("glob entry");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
        let scenario: StressScenario =
            ron::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e));

        if let Err(msg) = run_scenario(&scenario, &path) {
            errors.push(msg);
        }
    }

    assert!(
        errors.is_empty(),
        "{} stress scenario(s) failed:\n\n{}",
        errors.len(),
        errors.join("\n\n")
    );
}

// Suppress dead-code warnings for InvariantFailure variants pulled into scope
// (they appear inside formatted failure strings via Debug).
#[allow(dead_code)]
fn _import_used(_f: &InvariantFailure) {}
