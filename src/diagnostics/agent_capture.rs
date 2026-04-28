// Agent capture instrumentation — allows AI agents to spawn in-game,
// wait for terrain to load, and capture screenshots or video frames for
// qualitative rendering analysis.
//
// Activated via the `--agent` CLI flag. All behaviour is driven by the
// `AgentCaptureConfig` resource inserted in `main.rs` before `app.run()`.
//
// ## Capture phases
// 1. **Settling** — wait `settle_frames` frames so chunks generate and mesh.
// 2. **Capturing** — trigger one screenshot (screenshot mode) or N screenshots
//    one per frame (video mode), saved as numbered PNGs.
//    We trigger the screenshot and then wait an extra `SAVE_GRACE_FRAMES` frames
//    to ensure Bevy's async GPU readback has completed.
// 3. **Encoding** — (video only) invoke ffmpeg to encode the PNG sequence.
// 4. **Done** — write `meta.json` and exit via `AppExit`.

use std::path::PathBuf;

use bevy::app::AppExit;
use bevy::ecs::message::MessageWriter;
use bevy::prelude::*;

use crate::camera::SpawnLocation;

/// Extra frames to wait after triggering the last screenshot before moving on.
/// Bevy's screenshot readback is asynchronous (GPU → CPU copy), so we need a
/// small grace period to ensure the PNG is written before we proceed.
const SAVE_GRACE_FRAMES: u32 = 5;

/// Whether to capture a single screenshot or a multi-frame video.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CaptureMode {
    #[default]
    Screenshot,
    Video,
}

/// Configuration for agent capture mode. Insert before `app.run()`.
#[derive(Resource, Debug, Clone)]
pub struct AgentCaptureConfig {
    /// Frames to wait before capturing (lets chunks load and mesh).
    pub settle_frames: u32,
    /// Screenshot or video.
    pub mode: CaptureMode,
    /// For video: number of screenshots to record.
    pub capture_frames: u32,
    /// For interval video: capture one screenshot every N game frames.
    /// 1 = every frame (dense video), N>1 = time-lapse (N frames apart).
    pub capture_interval: u32,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Video FPS for ffmpeg encoding.
    pub fps: u32,
    /// Initial camera yaw offset in degrees, applied at spawn by rotating the
    /// default look direction around the surface normal.  0° = default (faces
    /// along the tangent toward what happens to be "east" on the spawn face).
    /// 180° flips the camera to face the opposite direction (useful for getting
    /// the sun in front of the camera in agent captures).
    pub initial_yaw_deg: f32,
    /// Initial camera pitch offset in degrees.  Negative = look down, positive =
    /// look up.  Applied after yaw.  −45° is a useful overhead-ish terrain view.
    pub initial_pitch_deg: f32,
    /// If true, advance the orbital rotation angle to solar noon at the spawn
    /// longitude so the terrain is fully lit for captures.  Without this the
    /// default rotation places solar noon at lon=0°, which leaves most spawn
    /// locations in darkness.
    pub force_daylight: bool,
}

impl Default for AgentCaptureConfig {
    fn default() -> Self {
        Self {
            settle_frames: 120,
            mode: CaptureMode::Screenshot,
            capture_frames: 120,
            capture_interval: 1,
            output_dir: PathBuf::from("agent_captures"),
            fps: 30,
            initial_yaw_deg: 0.0,
            initial_pitch_deg: 0.0,
            force_daylight: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum CapturePhase {
    Settling,
    Capturing,
    /// Waiting for GPU readback to finish writing PNGs.
    Waiting {
        frames_left: u32,
    },
    Encoding,
    Done,
}

/// Internal state for agent capture. Managed by `AgentCapturePlugin`.
#[derive(Resource)]
struct AgentCaptureState {
    frame: u32,
    phase: CapturePhase,
    saved_paths: Vec<PathBuf>,
    spawn_lat_deg: f64,
    spawn_lon_deg: f64,
}

/// Bevy plugin that drives agent capture when `AgentCaptureConfig` is present.
pub struct AgentCapturePlugin;

impl Plugin for AgentCapturePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AgentCaptureState {
            frame: 0,
            phase: CapturePhase::Settling,
            saved_paths: Vec::new(),
            spawn_lat_deg: 0.0,
            spawn_lon_deg: 0.0,
        })
        .add_systems(
            Update,
            agent_capture_tick.run_if(resource_exists::<AgentCaptureConfig>),
        );
    }
}

/// Master tick: advances phase, triggers captures, encodes video, and exits.
fn agent_capture_tick(
    config: Res<AgentCaptureConfig>,
    mut state: ResMut<AgentCaptureState>,
    mut commands: Commands,
    mut app_exit: MessageWriter<AppExit>,
    spawn_loc: Option<Res<SpawnLocation>>,
) {
    state.frame += 1;

    match state.phase.clone() {
        // ── Phase 1: Settling ──────────────────────────────────────────────
        CapturePhase::Settling => {
            if state.frame == 1 {
                if let Some(loc) = spawn_loc {
                    state.spawn_lat_deg = loc.lat;
                    state.spawn_lon_deg = loc.lon;
                }
                info!(
                    "[AgentCapture] Settling for {} frames before capture…",
                    config.settle_frames
                );
            }

            if state.frame >= config.settle_frames {
                info!("[AgentCapture] Settle complete. Starting capture…");
                if let Err(e) = std::fs::create_dir_all(&config.output_dir) {
                    error!("[AgentCapture] Failed to create output dir: {e}");
                }
                state.phase = CapturePhase::Capturing;
                state.frame = 0;
            }
        }

        // ── Phase 2: Capturing ─────────────────────────────────────────────
        CapturePhase::Capturing => {
            let frame_idx = state.saved_paths.len();
            let all_captured = match config.mode {
                CaptureMode::Screenshot => frame_idx >= 1,
                CaptureMode::Video => frame_idx >= config.capture_frames as usize,
            };

            if all_captured {
                // All screenshots triggered — wait for GPU readback.
                state.phase = CapturePhase::Waiting {
                    frames_left: SAVE_GRACE_FRAMES,
                };
            } else {
                let interval = config.capture_interval.max(1);
                let on_interval = state.frame.is_multiple_of(interval);
                let should_capture = on_interval
                    && match config.mode {
                        CaptureMode::Screenshot => frame_idx == 0,
                        CaptureMode::Video => true,
                    };

                if should_capture {
                    let path = config.output_dir.join(format!("frame_{frame_idx:05}.png"));

                    commands
                        .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
                        .observe(bevy::render::view::screenshot::save_to_disk(path.clone()));

                    info!("[AgentCapture] Triggered screenshot → {}", path.display());
                    state.saved_paths.push(path);
                }
            }
        }

        // ── Phase 2b: Waiting for GPU readback ────────────────────────────
        CapturePhase::Waiting { frames_left } => {
            if frames_left == 0 {
                match config.mode {
                    CaptureMode::Screenshot => state.phase = CapturePhase::Done,
                    CaptureMode::Video => {
                        state.phase = CapturePhase::Encoding;
                        state.frame = 0;
                    }
                }
            } else {
                state.phase = CapturePhase::Waiting {
                    frames_left: frames_left - 1,
                };
            }
        }

        // ── Phase 3: Encoding (video only) ────────────────────────────────
        CapturePhase::Encoding => {
            // Only run once on first frame after transition.
            if state.frame != 1 {
                return;
            }

            let output_video = config.output_dir.join("capture.mp4");
            let frame_pattern = config
                .output_dir
                .join("frame_%05d.png")
                .to_string_lossy()
                .to_string();

            info!(
                "[AgentCapture] Encoding {} frames to {}…",
                state.saved_paths.len(),
                output_video.display()
            );

            let status = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-framerate",
                    &config.fps.to_string(),
                    "-i",
                    &frame_pattern,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    // Pad to even dimensions (libx264 requirement).
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    output_video.to_str().unwrap_or("capture.mp4"),
                ])
                .status();

            match status {
                Ok(s) if s.success() => {
                    info!("[AgentCapture] Video encoded → {}", output_video.display());
                    state.saved_paths.push(output_video);
                }
                Ok(s) => {
                    warn!("[AgentCapture] ffmpeg exited with {s} — PNG frames preserved");
                }
                Err(e) => {
                    warn!("[AgentCapture] ffmpeg not available ({e}) — PNG frames preserved");
                }
            }

            state.phase = CapturePhase::Done;
        }

        // ── Phase 4: Done ─────────────────────────────────────────────────
        CapturePhase::Done => {
            if state.frame == 1 {
                write_metadata(&config, &state);
                info!("[AgentCapture] Capture complete. Exiting.");
                app_exit.write(AppExit::Success);
            } else if state.frame > 60 {
                // Bevy AppExit hasn't been processed by the runner (common
                // under headless/Xvfb setups). Abort immediately without
                // running destructors to avoid segfaults from wgpu teardown
                // on partially-shut-down threads.
                warn!("[AgentCapture] AppExit stalled — aborting process.");
                std::process::abort();
            }
        }
    }
}

/// Write `meta.json` with capture details for downstream analysis.
fn write_metadata(config: &AgentCaptureConfig, state: &AgentCaptureState) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mode_str = match config.mode {
        CaptureMode::Screenshot => "screenshot",
        CaptureMode::Video => "video",
    };

    let files_json: Vec<String> = state
        .saved_paths
        .iter()
        .map(|p| format!("\"{}\"", p.display()))
        .collect();

    let json = format!(
        r#"{{
  "mode": "{mode_str}",
  "spawn_lat_deg": {lat},
  "spawn_lon_deg": {lon},
  "settle_frames": {settle},
  "initial_yaw_deg": {yaw},
  "initial_pitch_deg": {pitch},
  "force_daylight": {force_daylight},
  "captured_at_unix": {now},
  "files": [{files}]
}}
"#,
        lat = state.spawn_lat_deg,
        lon = state.spawn_lon_deg,
        settle = config.settle_frames,
        yaw = config.initial_yaw_deg,
        pitch = config.initial_pitch_deg,
        force_daylight = config.force_daylight,
        files = files_json.join(", "),
    );

    let meta_path = config.output_dir.join("meta.json");
    if let Err(e) = std::fs::write(&meta_path, json) {
        warn!("[AgentCapture] Failed to write meta.json: {e}");
    } else {
        info!("[AgentCapture] Metadata written → {}", meta_path.display());
    }
}
