---
name: render-diagnostic
description: >-
    Headlessly run The Dark Candle, capture screenshots or video of the rendering
    pipeline, and analyse the output. Use this skill whenever the user reports a
    visual or rendering problem (terrain holes, floating chunks, dark sky, missing
    LOD, sky/atmosphere/star issues, lighting glitches, "still looks bad", etc.),
    or attaches a screenshot of an in-game artefact, or asks for a fresh capture
    after a render-pipeline change.
---

# Render Diagnostic Skill

This skill compresses the visual debug loop. Instead of asking the user to take
screenshots and paste them, the agent drives the game itself via the existing
`AgentCapturePlugin` harness, then inspects the output.

## When to invoke this skill (proactively)

Trigger automatically — without waiting to be asked — when ANY of these apply:

- The user describes a visual/rendering symptom: holes, floating chunks, missing
  geometry, wrong colours, black faces, dark/empty sky, missing stars, broken
  LOD seams, atmosphere artefacts, lighting/shadow problems, fog issues.
- The user attaches a screenshot of an in-game artefact and asks what it is or
  why it looks wrong.
- The user has just merged or asked to merge a change touching `src/world/`,
  `src/lighting/`, `src/gpu/`, `src/world/v2/`, atmosphere, sky, or meshing,
  and a verification capture would close the loop.
- The user says things like "verify your work", "take a fresh shot", or
  "does it still look bad?".

Do NOT invoke for non-visual issues (logic bugs, panics with no rendering
component, unit-test failures, data-loading bugs).

## Procedure

### 1. Cross-reference open issues first

Before capturing, read `issues.json` and look for open bugs whose `category` is
`render`, `sky`, `lighting`, `atmosphere`, `lod`, `meshing`, or whose
`affected_files` overlap the modules touched in the current task. If the
symptom matches an existing issue, mention the issue ID and use the same
`spawn` coordinates if it records any. This stops duplicate issues being
filed and lets the user track recurrence.

### 2. Choose capture parameters

Default: a single screenshot at `coastline` (interesting terrain features).
Adjust based on the reported symptom:

| Symptom | Suggested flags |
|---|---|
| Generic "looks wrong" / first-pass diagnostic | `--spawn coastline --settle 240 --capture screenshot` |
| LOD seams / chunk popping while moving | `--capture video --capture-frames 150 --capture-fps 30 --settle 180` |
| Sky / sun / atmosphere / stars | `--spawn 0.0,0.0 --settle 240` (and a second capture rotated to look up if the harness supports it) |
| Subsurface / collision / "falling through" | `--spawn random-land --settle 300` |
| Specific reported coordinate | `--spawn LAT,LON` matching the user's report |
| Regression compared to known-good | Same `--spawn` and `--settle` as the previous capture in `agent_captures/` |

Always use `--planet --planet-level 7` unless the user specifies otherwise,
and `--settle 240` or higher (level 7 needs time to mesh).

### 3. Run the harness

Use a unique output directory per run so captures aren't overwritten:

```bash
OUT="agent_captures/$(date -u +%Y%m%d-%H%M%S)-<short-symptom-tag>"
cargo run --release --features bevy/dynamic_linking -- \
  --planet --planet-level 7 \
  --spawn coastline \
  --agent --settle 240 --capture screenshot \
  --capture-out "$OUT"
```

Notes:
- Prefer `--release` for capture runs — debug builds frequently fail to
  finish meshing within the settle window on level 7.
- If `cargo run` fails to compile, fix the build first; do not paper over
  with a stale capture.
- If the harness exits before producing `meta.json`, treat it as a
  diagnostic failure and report the stderr to the user instead of guessing.

### 4. Inspect the captures

After the run completes:

1. Read `<OUT>/meta.json` to confirm the capture parameters and file list.
2. View each PNG in `meta.files` with the `view` tool — the agent can
   actually look at the image. Describe what is visible: terrain
   coverage, sky colour, presence/absence of artefacts, HUD readings
   (chunk count, FPS).
3. Compare against the user's reported symptom. If the symptom is
   reproduced, proceed to diagnose. If it is NOT reproduced, say so
   explicitly and ask whether the user is seeing it on a different
   spawn or build.

### 5. Multi-hypothesis captures (when comparing fixes)

When the user previously asked to compare "Test A vs Test B" style
alternatives, run BOTH captures in parallel using two background bash
sessions or `task` subagents, each writing to its own `--capture-out`
directory. Present the two screenshots side by side in the response.

### 6. Update `issues.json`

- If the capture reproduces a NEW visual bug, append an entry with a
  fresh ID (e.g. `RENDER-NNN`, `SKY-NNN`), severity, suspected cause,
  and the path to the capture under `evidence`.
- If the capture confirms an existing open bug is FIXED, update its
  `status` to `resolved` with the resolution details and the commit
  SHA that fixed it.
- If the capture shows the bug is still present after a fix attempt,
  add a comment to the issue (or update `last_seen`) — do not silently
  re-run.

## Reference

Full documentation of the capture harness, including all CLI flags,
spawn modes, and the `meta.json` schema, lives in
[`docs/agent-capture.md`](../../../docs/agent-capture.md). The plugin
implementation is in `src/diagnostics/` (search for `AgentCapturePlugin`).

## Anti-patterns

- ❌ Asking the user to take a screenshot when this skill is available.
- ❌ Running a capture without first checking `issues.json`.
- ❌ Reusing `agent_captures/` as the output directory (it overwrites
  previous baselines like `render010_baseline/` and `lod_fix_run/`).
- ❌ Reporting "looks fine" without actually viewing the PNG with the
  `view` tool.
- ❌ Iterating fix → capture → "still bad" more than twice without
  pausing to consult the rubber-duck agent on the hypothesis.
