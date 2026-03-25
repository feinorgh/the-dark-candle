// Video encoder: pipes raw RGB frames to ffmpeg or falls back to PNG output.
//
// The encoder is designed for headless simulation visualization. It spawns an
// ffmpeg subprocess and streams raw pixel data via stdin, producing an MP4 (or
// other format) without writing intermediate frame files. When ffmpeg is not
// available, it saves individual PNG frames to a directory instead.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use image::RgbImage;
use serde::Deserialize;

use super::visualization::{ColorMode, ViewMode};

/// Full video output configuration, embeddable in `SimulationScenario`.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct VideoConfig {
    /// Output path (e.g. `"output.mp4"`). For PNG fallback, a directory is
    /// created at `<path>_frames/`.
    pub path: String,
    /// How to project the 3D grid into 2D.
    #[serde(default)]
    pub view: ViewMode,
    /// What physical quantity to visualize.
    #[serde(default)]
    pub color_mode: ColorMode,
    /// Frames per second in the output video (default: 30).
    #[serde(default = "default_fps")]
    pub fps: u32,
    /// Pixel scale factor per voxel (default: 4).
    #[serde(default = "default_scale")]
    pub scale: u32,
}

fn default_fps() -> u32 {
    30
}

fn default_scale() -> u32 {
    4
}

/// Active encoder state. Created by [`FrameEncoder::new`] and consumed by
/// [`FrameEncoder::push_frame`] / [`FrameEncoder::finish`].
pub enum FrameEncoder {
    /// Piping raw frames to ffmpeg.
    Ffmpeg {
        child: Child,
        width: u32,
        height: u32,
    },
    /// Saving individual PNGs (fallback).
    PngFallback { dir: PathBuf, frame_count: u32 },
}

impl FrameEncoder {
    /// Create a new encoder.
    ///
    /// Tries ffmpeg first; falls back to PNG directory output if ffmpeg is not
    /// found. The `width` and `height` are the pixel dimensions of each frame
    /// (i.e. `grid_size * scale`).
    pub fn new(output_path: &str, width: u32, height: u32, fps: u32) -> Result<Self, String> {
        // Try to launch ffmpeg
        let ffmpeg_result = Command::new("ffmpeg")
            .args([
                "-y", // overwrite output
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                &format!("{width}x{height}"),
                "-r",
                &fps.to_string(),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                // Ensure dimensions are even (x264 requirement)
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-movflags",
                "+faststart",
                output_path,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        match ffmpeg_result {
            Ok(child) => {
                eprintln!("  Video encoder: piping to ffmpeg → {output_path}");
                Ok(Self::Ffmpeg {
                    child,
                    width,
                    height,
                })
            }
            Err(_) => {
                // Fallback: create a directory for PNG frames
                let dir = PathBuf::from(format!(
                    "{}_frames",
                    Path::new(output_path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("video")
                ));
                std::fs::create_dir_all(&dir).map_err(|e| {
                    format!("Failed to create frame directory {}: {e}", dir.display())
                })?;
                eprintln!(
                    "  Video encoder: ffmpeg not found, saving PNGs to {}/",
                    dir.display()
                );
                Ok(Self::PngFallback {
                    dir,
                    frame_count: 0,
                })
            }
        }
    }

    /// Push one frame to the encoder.
    pub fn push_frame(&mut self, img: &RgbImage) -> Result<(), String> {
        match self {
            Self::Ffmpeg {
                child,
                width,
                height,
            } => {
                debug_assert_eq!(img.width(), *width);
                debug_assert_eq!(img.height(), *height);

                let stdin = child
                    .stdin
                    .as_mut()
                    .ok_or("ffmpeg stdin closed unexpectedly")?;
                stdin
                    .write_all(img.as_raw())
                    .map_err(|e| format!("Failed to write frame to ffmpeg: {e}"))?;
                Ok(())
            }
            Self::PngFallback { dir, frame_count } => {
                let path = dir.join(format!("frame_{frame_count:05}.png"));
                img.save(&path)
                    .map_err(|e| format!("Failed to save {}: {e}", path.display()))?;
                *frame_count += 1;
                Ok(())
            }
        }
    }

    /// Finalize the video. For ffmpeg, closes stdin and waits for the process
    /// to finish. For PNG fallback, prints a helpful ffmpeg command.
    pub fn finish(self) -> Result<(), String> {
        match self {
            Self::Ffmpeg { mut child, .. } => {
                // Close stdin to signal EOF
                drop(child.stdin.take());
                let status = child
                    .wait()
                    .map_err(|e| format!("Failed to wait for ffmpeg: {e}"))?;
                if status.success() {
                    eprintln!("  Video encoding complete.");
                    Ok(())
                } else {
                    Err(format!("ffmpeg exited with status: {status}"))
                }
            }
            Self::PngFallback { dir, frame_count } => {
                eprintln!("  Saved {frame_count} PNG frames to {}/", dir.display());
                eprintln!(
                    "  To encode: ffmpeg -framerate 30 -i {}/frame_%%05d.png \
                     -c:v libx264 -pix_fmt yuv420p output.mp4",
                    dir.display()
                );
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn png_fallback_creates_frames() {
        let dir = std::env::temp_dir().join("tdc_video_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut encoder = FrameEncoder::PngFallback {
            dir: dir.clone(),
            frame_count: 0,
        };

        std::fs::create_dir_all(&dir).unwrap();

        let img = RgbImage::from_pixel(4, 4, Rgb([128, 64, 32]));
        encoder.push_frame(&img).unwrap();
        encoder.push_frame(&img).unwrap();

        assert!(dir.join("frame_00000.png").exists());
        assert!(dir.join("frame_00001.png").exists());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn video_config_deserializes_defaults() {
        let ron_str = r#"(
            path: "test.mp4",
        )"#;
        let config: VideoConfig = ron::from_str(ron_str).unwrap();
        assert_eq!(config.path, "test.mp4");
        assert_eq!(config.fps, 30);
        assert_eq!(config.scale, 4);
        assert_eq!(config.color_mode, ColorMode::Material);
    }
}
