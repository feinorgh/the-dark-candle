# Working With AI Agents

Guidelines for collaborating with AI coding agents on The Dark Candle, based on lessons learned building the simulation stack (Phases 0–7, 338+ tests).

## What Works Well

### AGENTS.md Is Critical

The `AGENTS.md` file at the repository root is the single highest-ROI file for AI collaboration. It contains Bevy 0.18 API rules, architecture constraints, and project conventions that prevent hallucinations.

**Keep expanding it** with every API gotcha you discover. During development, AGENTS.md caught dozens of issues: deprecated bundles, old camera APIs, incorrect `Entity` constructors, wrong `AmbientLight` usage, and more.

### Pure Functions + Unit Tests = Safe AI Territory

Every simulation module follows the same pattern:

1. **Pure logic** in standalone functions (no Bevy dependency).
2. **Bevy plugin** just wires systems to the ECS schedule.
3. **Trait-based abstractions** (e.g., `VoxelGrid`, `PerceptionGrid`) so tests use lightweight test grids instead of full chunks.

This is ideal for AI agents because they can write, compile, and validate code without needing a running game instance. Preserve this separation as the codebase grows.

### Phased, Scoped Tasks

Well-defined tasks like "Do Phase 6" work well because each task has clear boundaries, inputs, and outputs. Vague requests like "make the game better" or "improve performance" will produce low-quality results.

## Recommendations

### Keep Sessions Focused on One Concern

As the codebase grows past what fits in a single context window (~50+ files), start fresh sessions for unrelated work. Use session checkpoints and `plan.md` to carry state between sessions. A session that tries to touch rendering, physics, and AI simultaneously will produce worse results than three focused sessions.

### Use the Explore Agent Liberally

Before making changes to existing code, have the agent answer "how does X currently work?" rather than relying on its memory of files read many turns ago. Stale context is the primary source of incorrect edits. Reading a file costs far less than debugging a bad edit.

### Reference Code, Don't Assume It

When asking for changes to existing files, name the specific file path (e.g., "modify `src/physics/fluids.rs`") rather than describing the file from memory. The agent should always `view` a file before editing it.

### Run Tests After Every Change

The pre-commit hook enforces `cargo fmt`, `cargo clippy`, and `cargo test` on every commit. This fast feedback loop catches hallucinations before they compound. Never skip it, and never batch multiple unrelated changes into one commit.

### Keep AGENTS.md as a Living Document

Add entries for:

- **New constants**: MaterialId values (currently 0–12), chunk sizes, physics constants.
- **Module boundaries**: what imports what, which modules are pure vs. Bevy-dependent.
- **Patterns that worked**: snapshot-based simulation, `TestGrid` trait pattern for pathfinding/perception, `SimpleRng` for deterministic generation.
- **API gotchas**: any new Bevy 0.18 surprises discovered during development.

### Write Tests First for New Features

TDD catches hallucinations earlier than writing tests after the fact. The pattern:

1. Define the data structures and function signatures.
2. Write tests that express the expected behavior.
3. Implement until tests pass.
4. Run clippy and the full test suite.

### Avoid Sweeping Refactors in One Session

"Rename X everywhere" or "restructure the module hierarchy" risks silent breakage across many files. Prefer small, verifiable changes. If a large refactor is needed, break it into phases with test verification at each step.

### For Larger Features (Rendering, Audio, Networking)

As the project moves beyond pure simulation into Bevy-integrated systems:

- **One session per system** — don't mix rendering work with physics work.
- **Start each session** by reading `AGENTS.md` + the specific module's `mod.rs`.
- **Maintain the pure-logic / Bevy-integration split** — keep game logic testable without a running app, even when adding rendering or input handling.
- **Expect more Bevy API friction** — rendering, assets, and windowing change between Bevy versions more than ECS fundamentals do. Document every workaround in AGENTS.md.

## Current Codebase Structure

For reference, the simulation stack as of Phase 7 completion:

```
src/
├── main.rs              # App entry, registers all plugins
├── camera/              # FPS camera, movement, gravity
├── world/               # Voxel types, chunks, terrain, meshing, collision
├── physics/             # Gravity, AABB collision, fluids, integrity, pressure
├── chemistry/           # Materials, heat transfer, reactions, state transitions
├── biology/             # Metabolism, health, growth, decay, plants
├── behavior/            # Needs, utility AI, pathfinding, behaviors, perception
├── social/              # Relationships, factions, reputation, group behaviors
├── entities/            # Enemy components
├── procgen/             # Creature/item generation, biomes, spawning
└── data/                # RON data structs, asset loading
```

All simulation modules use pure functions with comprehensive test suites. Bevy plugins are thin wrappers that schedule these functions as ECS systems.
