# Agent Capture — AI Rendering Analysis Guide

Agent capture mode lets an AI agent launch the game headlessly (or with a window), wait
for terrain chunks to load, capture a screenshot or video, and exit automatically. The
output files and a `meta.json` descriptor are written to a configurable directory for
downstream analysis.

## Quick start

```bash
# Single screenshot after 180 frames of settling
cargo run --features bevy/dynamic_linking -- \
  --planet --planet-level 7 \
  --spawn 45.0,0.0 \
  --agent --settle 180 --capture screenshot \
  --capture-out agent_captures/

# 5-second video (150 frames @ 30 fps)
cargo run --features bevy/dynamic_linking -- \
  --planet --planet-level 7 \
  --spawn coastline \
  --agent --settle 120 --capture video \
  --capture-frames 150 --capture-fps 30 \
  --capture-out agent_captures/video/
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--agent` | false | Enable agent mode. Suppresses cursor grab and auto-exits after capture. |
| `--settle <n>` | 120 | Frames to wait before capturing. Increase for slower terrain generation. |
| `--capture <mode>` | screenshot | `screenshot` or `video`. |
| `--capture-frames <n>` | 120 | Frames to record (video mode only). |
| `--capture-out <dir>` | `agent_captures` | Output directory. Created if absent. |
| `--capture-fps <n>` | 30 | Target FPS for ffmpeg encoding (video mode only). |
| `--initial-yaw-deg <deg>` | 0.0 | Rotate the camera's initial look direction by this many degrees around the surface normal. Use `180` to face the opposite direction (often puts the sun in front of the camera). |
| `--initial-pitch-deg <deg>` | 0.0 | Tilt the camera up (positive) or down (negative). `-45` gives a useful overhead terrain view. |

### Spawn modes (used with `--spawn`)

| Mode | Description |
|---|---|
| `LAT,LON` | Explicit coordinates in degrees, e.g. `45.0,-12.5`. |
| `random-land` | Random point above sea level. |
| `coastline` | Random land point near a water boundary. |
| *(omitted)* | Default: 45°N 0°E. |

## Output files

After a run, the output directory contains:

```
agent_captures/
  frame_00000.png      # screenshot, or first video frame
  frame_00001.png      # (video mode only)
  …
  capture.mp4          # (video mode only — requires ffmpeg)
  meta.json            # metadata descriptor
```

### `meta.json` format

```json
{
  "mode": "screenshot",
  "spawn_lat_deg": 45.0,
  "spawn_lon_deg": 0.0,
  "settle_frames": 180,
  "captured_at_unix": 1745712000,
  "files": ["agent_captures/frame_00000.png"]
}
```

## Capture phases

The `AgentCapturePlugin` drives the session through four phases:

1. **Settling** — waits `settle_frames` game frames so chunks generate and mesh.
2. **Capturing** — spawns `Screenshot::primary_window()` observers (one per frame for
   video). Bevy's GPU readback is async, so there is an additional 5-frame grace period.
3. **Encoding** (video only) — invokes `ffmpeg` to encode the PNG sequence into `capture.mp4`.
   If ffmpeg is not found, the PNG frames are preserved and a warning is logged.
4. **Done** — writes `meta.json` and calls `AppExit::Success`.

## Video encoding requirements

- `ffmpeg` must be available on `PATH` for MP4 encoding.
- If absent, PNG frames are kept and the process exits cleanly (no crash).
- The encode command uses `libx264` with `yuv420p` pixel format and pads to even dimensions.

## Example: automated quality analysis

```python
import subprocess, json, pathlib, sys

out = pathlib.Path("agent_captures")
result = subprocess.run([
    "cargo", "run", "--features", "bevy/dynamic_linking", "--",
    "--planet", "--planet-level", "7",
    "--spawn", "coastline",
    "--agent", "--settle", "240", "--capture", "screenshot",
    "--capture-out", str(out),
], capture_output=True, text=True)

meta = json.loads((out / "meta.json").read_text())
print("Captured:", meta["files"])
# Pass meta["files"][0] to an image analysis model
```

## Notes for agents

- Use at least `--settle 120` on a level-5 planet; use `240+` for level 7.
- Combine with `--spawn coastline` to land near interesting terrain features.
- For rendering regression checks, compare multiple screenshots at the same coordinates.
- The HUD diagnostics (chunk count, FPS) remain visible in captures — they are useful
  for verifying terrain is fully loaded before the capture fires.
