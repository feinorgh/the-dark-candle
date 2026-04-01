// Composable multi-octave noise for terrain generation.
//
// `NoiseStack` replaces the hardcoded 2-layer Perlin blend with a flexible
// noise pipeline:
//
//   1. **FBM** (Fractal Brownian Motion) — standard multi-octave sum.
//   2. **Ridged multi-fractal** — sharp ridges from `(1 - |noise|)²`.
//   3. **Terrain type selector** — low-frequency noise blends FBM ↔ ridged.
//   4. **Domain warping** — offsets sample coordinates for organic shapes.
//   5. **Micro-detail** — high-frequency, low-amplitude surface roughness.
//
// All functions are deterministic given a seed and pure (no mutation),
// enabling parallel chunk generation.

use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

/// Configuration for the composable noise stack.
///
/// Each field has a sensible default.  Scene presets override specific values
/// to produce different terrain characters.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NoiseConfig {
    // ── FBM ────────────────────────────────────────────────────────────
    /// Number of FBM octaves.  More octaves = finer detail.
    #[serde(default = "default_fbm_octaves")]
    pub fbm_octaves: u32,
    /// Amplitude decay per octave.  0.5 is standard.
    #[serde(default = "default_fbm_persistence")]
    pub fbm_persistence: f64,
    /// Frequency multiplier per octave.
    #[serde(default = "default_fbm_lacunarity")]
    pub fbm_lacunarity: f64,
    /// Base frequency for FBM evaluation.
    #[serde(default = "default_fbm_base_freq")]
    pub fbm_base_freq: f64,

    // ── Ridged multi-fractal ───────────────────────────────────────────
    /// Number of ridged-noise octaves.
    #[serde(default = "default_ridged_octaves")]
    pub ridged_octaves: u32,
    /// Feedback gain for ridged noise weighting.
    #[serde(default = "default_ridged_gain")]
    pub ridged_gain: f64,
    /// Base frequency for ridged evaluation.
    #[serde(default = "default_ridged_base_freq")]
    pub ridged_base_freq: f64,

    // ── Terrain type selector ──────────────────────────────────────────
    /// Frequency of the selector noise.
    #[serde(default = "default_selector_freq")]
    pub selector_freq: f64,
    /// `(low, high)` thresholds.  Below `low` → pure FBM; above `high` →
    /// pure ridged; between → smoothstep blend.
    #[serde(default = "default_selector_thresholds")]
    pub selector_thresholds: (f64, f64),

    // ── Domain warping ─────────────────────────────────────────────────
    /// Warp displacement amplitude in meters (voxels).
    #[serde(default = "default_warp_strength")]
    pub warp_strength: f64,
    /// Frequency of the warp noise fields.
    #[serde(default = "default_warp_freq")]
    pub warp_freq: f64,

    // ── Micro-detail ───────────────────────────────────────────────────
    /// Frequency of the surface micro-detail layer.
    #[serde(default = "default_micro_freq")]
    pub micro_freq: f64,
    /// Amplitude of the micro-detail layer (±voxels).
    #[serde(default = "default_micro_amplitude")]
    pub micro_amplitude: f64,

    // ── Continent mask ──────────────────────────────────────────────────
    /// Whether to apply continent/ocean masking.
    #[serde(default)]
    pub continent_enabled: bool,
    /// Frequency of the continent noise field.
    #[serde(default = "default_continent_freq")]
    pub continent_freq: f64,
    /// Threshold for land vs ocean.  Continent noise above this → land.
    /// Below `threshold - shelf_blend_width` → deep ocean.
    #[serde(default = "default_continent_threshold")]
    pub continent_threshold: f64,
    /// Width of the continental shelf transition zone (in noise-space units).
    #[serde(default = "default_shelf_blend_width")]
    pub shelf_blend_width: f64,
    /// Ocean floor base depth in normalized noise units (relative to height_scale).
    /// A value of 1.25 with height_scale=32 gives an ocean floor ~40 voxels below sea level.
    #[serde(default = "default_ocean_floor_depth")]
    pub ocean_floor_depth: f64,
    /// Amplitude of ocean floor terrain variation in normalized noise units.
    #[serde(default = "default_ocean_floor_amplitude")]
    pub ocean_floor_amplitude: f64,
}

// ── Default value fns (for serde) ──────────────────────────────────────────

fn default_fbm_octaves() -> u32 {
    6
}
fn default_fbm_persistence() -> f64 {
    0.5
}
fn default_fbm_lacunarity() -> f64 {
    2.0
}
fn default_fbm_base_freq() -> f64 {
    0.005
}
fn default_ridged_octaves() -> u32 {
    5
}
fn default_ridged_gain() -> f64 {
    2.0
}
fn default_ridged_base_freq() -> f64 {
    0.008
}
fn default_selector_freq() -> f64 {
    0.003
}
fn default_selector_thresholds() -> (f64, f64) {
    (-0.2, 0.3)
}
fn default_warp_strength() -> f64 {
    40.0
}
fn default_warp_freq() -> f64 {
    0.004
}
fn default_micro_freq() -> f64 {
    0.15
}
fn default_micro_amplitude() -> f64 {
    1.5
}
fn default_continent_freq() -> f64 {
    0.002
}
fn default_continent_threshold() -> f64 {
    0.05
}
fn default_shelf_blend_width() -> f64 {
    0.15
}
fn default_ocean_floor_depth() -> f64 {
    1.25 // With height_scale=32, this is ~40 voxels below sea level
}
fn default_ocean_floor_amplitude() -> f64 {
    0.25 // With height_scale=32, ±8 voxels of ocean floor variation
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            fbm_octaves: default_fbm_octaves(),
            fbm_persistence: default_fbm_persistence(),
            fbm_lacunarity: default_fbm_lacunarity(),
            fbm_base_freq: default_fbm_base_freq(),
            ridged_octaves: default_ridged_octaves(),
            ridged_gain: default_ridged_gain(),
            ridged_base_freq: default_ridged_base_freq(),
            selector_freq: default_selector_freq(),
            selector_thresholds: default_selector_thresholds(),
            warp_strength: default_warp_strength(),
            warp_freq: default_warp_freq(),
            micro_freq: default_micro_freq(),
            micro_amplitude: default_micro_amplitude(),
            continent_enabled: false,
            continent_freq: default_continent_freq(),
            continent_threshold: default_continent_threshold(),
            shelf_blend_width: default_shelf_blend_width(),
            ocean_floor_depth: default_ocean_floor_depth(),
            ocean_floor_amplitude: default_ocean_floor_amplitude(),
        }
    }
}

/// Composable noise stack for terrain height generation.
///
/// Built from a `NoiseConfig` and a seed.  Evaluates a pipeline of noise
/// functions to produce terrain height at any 2D position.
///
/// The evaluation pipeline:
/// 1. Domain warp: offset `(x, z)` by two independent noise fields.
/// 2. Selector: low-frequency noise to choose FBM vs ridged.
/// 3. FBM / ridged multi-fractal: multi-octave noise evaluation.
/// 4. Blend: smoothstep interpolation between FBM and ridged.
///
/// Micro-detail is evaluated separately via `sample_with_detail()` and adds
/// high-frequency, low-amplitude variation at the surface.
pub struct NoiseStack {
    config: NoiseConfig,
    /// Seed offsets for deterministic sub-noise generation.
    seed: u32,
}

impl NoiseStack {
    pub fn new(seed: u32, config: NoiseConfig) -> Self {
        Self { config, seed }
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &NoiseConfig {
        &self.config
    }

    // ── Core noise functions ───────────────────────────────────────────

    /// Fractal Brownian Motion: multi-octave Perlin sum.
    ///
    /// Returns a value roughly in `[-1, 1]` (normalized by the geometric
    /// series sum of amplitudes).
    pub fn fbm(&self, x: f64, z: f64) -> f64 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = self.config.fbm_base_freq;
        let mut normalization = 0.0;

        for i in 0..self.config.fbm_octaves {
            let perlin = Perlin::new(self.seed.wrapping_add(i));
            value += amplitude * perlin.get([x * frequency, z * frequency]);
            normalization += amplitude;
            amplitude *= self.config.fbm_persistence;
            frequency *= self.config.fbm_lacunarity;
        }

        if normalization > 0.0 {
            value / normalization
        } else {
            0.0
        }
    }

    /// Ridged multi-fractal noise: produces sharp mountain ridges.
    ///
    /// Uses `1.0 - |noise|` inversion to create ridges at noise zero-crossings,
    /// then squares the signal to sharpen.  Each octave is weighted by the
    /// previous octave's output for detail concentration on ridges.
    ///
    /// Returns a value in `[0, ~1]` (not normalized to exact bounds).
    pub fn ridged(&self, x: f64, z: f64) -> f64 {
        let mut value = 0.0;
        let mut weight = 1.0;
        let mut frequency = self.config.ridged_base_freq;
        let mut amplitude = 1.0;
        let persistence = 0.5;
        let mut normalization = 0.0;

        for i in 0..self.config.ridged_octaves {
            let perlin = Perlin::new(self.seed.wrapping_add(50 + i));
            let signal_raw = perlin.get([x * frequency, z * frequency]);
            let mut signal = 1.0 - signal_raw.abs();
            signal *= signal; // sharpen ridges
            signal *= weight;
            weight = (signal * self.config.ridged_gain).clamp(0.0, 1.0);

            value += signal * amplitude;
            normalization += amplitude;
            amplitude *= persistence;
            frequency *= self.config.fbm_lacunarity; // reuse lacunarity
        }

        if normalization > 0.0 {
            value / normalization
        } else {
            0.0
        }
    }

    /// Terrain type selector value at `(x, z)`.
    ///
    /// Low-frequency noise in `[-1, 1]` used to blend between FBM and ridged
    /// terrain.
    fn selector_value(&self, x: f64, z: f64) -> f64 {
        let perlin = Perlin::new(self.seed.wrapping_add(100));
        perlin.get([x * self.config.selector_freq, z * self.config.selector_freq])
    }

    /// Domain warp offsets.  Returns `(warp_x, warp_z)` to add to the sample
    /// coordinates before evaluating terrain noise.
    fn warp_offsets(&self, x: f64, z: f64) -> (f64, f64) {
        let warp_x_noise = Perlin::new(self.seed.wrapping_add(200));
        let warp_z_noise = Perlin::new(self.seed.wrapping_add(201));
        let wx = warp_x_noise.get([x * self.config.warp_freq, z * self.config.warp_freq])
            * self.config.warp_strength;
        let wz = warp_z_noise.get([x * self.config.warp_freq, z * self.config.warp_freq])
            * self.config.warp_strength;
        (wx, wz)
    }

    /// Micro-detail noise.  High-frequency, low-amplitude layer for visual
    /// surface roughness.  Intended to be added only at the surface.
    pub fn micro_detail(&self, x: f64, z: f64) -> f64 {
        let perlin = Perlin::new(self.seed.wrapping_add(300));
        perlin.get([x * self.config.micro_freq, z * self.config.micro_freq])
            * self.config.micro_amplitude
    }

    // ── Continent mask ─────────────────────────────────────────────────

    /// Raw continent noise value at `(x, z)`.
    ///
    /// Values > `continent_threshold` → land,
    /// values < `continent_threshold - shelf_blend_width` → deep ocean,
    /// in between → continental shelf (gradual transition).
    pub fn continent_value(&self, x: f64, z: f64) -> f64 {
        let perlin = Perlin::new(self.seed.wrapping_add(350));
        perlin.get([
            x * self.config.continent_freq,
            z * self.config.continent_freq,
        ])
    }

    /// Low-amplitude ocean floor terrain at `(x, z)`.
    ///
    /// Returns normalized noise for underwater terrain variation.
    fn ocean_floor_noise(&self, x: f64, z: f64) -> f64 {
        let perlin = Perlin::new(self.seed.wrapping_add(360));
        perlin.get([x * 0.01, z * 0.01]) * self.config.ocean_floor_amplitude
    }

    /// Classify a position as land / shelf / ocean and return a blend factor.
    ///
    /// Returns:
    /// - `1.0` for full land (standard terrain applies)
    /// - `0.0` for deep ocean (ocean floor applies)
    /// - `(0, 1)` for continental shelf (blended transition)
    pub fn continent_blend(&self, x: f64, z: f64) -> f64 {
        let cv = self.continent_value(x, z);
        let threshold = self.config.continent_threshold;
        let shelf = self.config.shelf_blend_width;
        let ocean_edge = threshold - shelf;

        if cv >= threshold {
            1.0
        } else if cv <= ocean_edge {
            0.0
        } else {
            smoothstep(ocean_edge, threshold, cv)
        }
    }

    // ── Composite sampling ─────────────────────────────────────────────

    /// Sample raw terrain height at `(x, z)`, excluding micro-detail.
    ///
    /// Returns a value roughly in `[-1, 1]` representing the land terrain shape.
    /// When continent masking is disabled, this is the final terrain noise.
    fn sample_land(&self, x: f64, z: f64) -> f64 {
        // 1. Domain warp
        let (wx, wz) = self.warp_offsets(x, z);
        let sx = x + wx;
        let sz = z + wz;

        // 2. Selector
        let sel = self.selector_value(sx, sz);
        let (lo, hi) = self.config.selector_thresholds;

        // 3. FBM and/or ridged
        if sel < lo {
            // Pure FBM (flat/rolling terrain)
            self.fbm(sx, sz) * 0.6
        } else if sel > hi {
            // Pure ridged (mountain terrain)
            self.ridged(sx, sz) * 1.5 - 0.5
        } else {
            // Blended transition
            let t = smoothstep(lo, hi, sel);
            let fbm_val = self.fbm(sx, sz) * 0.6;
            let ridged_val = self.ridged(sx, sz) * 1.5 - 0.5;
            lerp(fbm_val, ridged_val, t)
        }
    }

    /// Sample terrain height at `(x, z)`, excluding micro-detail.
    ///
    /// When continent masking is enabled, blends between land terrain
    /// and ocean floor based on the continent noise field.  Returns a
    /// normalized value (roughly `[-1.5, 1.5]`) to be scaled by
    /// `height_scale` and offset by `sea_level`.
    pub fn sample(&self, x: f64, z: f64) -> f64 {
        let land = self.sample_land(x, z);

        if !self.config.continent_enabled {
            return land;
        }

        let blend = self.continent_blend(x, z);

        if blend >= 1.0 {
            land
        } else {
            // Ocean floor in normalized units: negative → below sea level
            let ocean = -self.config.ocean_floor_depth + self.ocean_floor_noise(x, z);
            if blend <= 0.0 {
                ocean
            } else {
                lerp(ocean, land, blend)
            }
        }
    }

    /// Sample terrain height at `(x, z)`, including micro-detail.
    ///
    /// Use this for the final voxel-fill pass.  The micro-detail adds surface
    /// roughness (±`micro_amplitude` voxels) without changing the large-scale
    /// terrain shape.
    pub fn sample_with_detail(&self, x: f64, z: f64) -> f64 {
        self.sample(x, z) + self.micro_detail(x, z)
    }
}

// ── Math helpers ───────────────────────────────────────────────────────────

/// Hermite smoothstep: smooth interpolation from 0 to 1 as `x` moves from
/// `edge0` to `edge1`.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation between `a` and `b` by factor `t`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_stack() -> NoiseStack {
        NoiseStack::new(42, NoiseConfig::default())
    }

    // ── FBM ────────────────────────────────────────────────────────────

    #[test]
    fn fbm_is_deterministic() {
        let stack = default_stack();
        let a = stack.fbm(100.0, 200.0);
        let b = stack.fbm(100.0, 200.0);
        assert_eq!(a, b);
    }

    #[test]
    fn fbm_is_bounded() {
        let stack = default_stack();
        for x in (-500..500).step_by(17) {
            for z in (-500..500).step_by(17) {
                let v = stack.fbm(x as f64, z as f64);
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "FBM out of [-1, 1]: {v} at ({x}, {z})"
                );
            }
        }
    }

    #[test]
    fn fbm_varies_across_space() {
        let stack = default_stack();
        let mut values = Vec::new();
        for x in (0..1000).step_by(100) {
            values.push(stack.fbm(x as f64, 0.0));
        }
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min > 0.01,
            "FBM shows no variation: min={min}, max={max}"
        );
    }

    #[test]
    fn fbm_different_seeds_differ() {
        let a = NoiseStack::new(1, NoiseConfig::default());
        let b = NoiseStack::new(9999, NoiseConfig::default());
        let va = a.fbm(50.0, 50.0);
        let vb = b.fbm(50.0, 50.0);
        assert!(
            (va - vb).abs() > 1e-6,
            "Different seeds should produce different FBM values"
        );
    }

    // ── Ridged ─────────────────────────────────────────────────────────

    #[test]
    fn ridged_is_non_negative() {
        let stack = default_stack();
        for x in (-500..500).step_by(17) {
            for z in (-500..500).step_by(17) {
                let v = stack.ridged(x as f64, z as f64);
                assert!(v >= 0.0, "Ridged is negative: {v} at ({x}, {z})");
            }
        }
    }

    #[test]
    fn ridged_is_deterministic() {
        let stack = default_stack();
        let a = stack.ridged(100.0, 200.0);
        let b = stack.ridged(100.0, 200.0);
        assert_eq!(a, b);
    }

    #[test]
    fn ridged_varies_across_space() {
        let stack = default_stack();
        let mut values = Vec::new();
        for x in (0..1000).step_by(100) {
            values.push(stack.ridged(x as f64, 0.0));
        }
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min > 0.01,
            "Ridged shows no variation: min={min}, max={max}"
        );
    }

    // ── Selector ───────────────────────────────────────────────────────

    #[test]
    fn selector_is_bounded() {
        let stack = default_stack();
        for x in (-500..500).step_by(23) {
            for z in (-500..500).step_by(23) {
                let v = stack.selector_value(x as f64, z as f64);
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "Selector out of [-1, 1]: {v} at ({x}, {z})"
                );
            }
        }
    }

    // ── Domain warp ────────────────────────────────────────────────────

    #[test]
    fn warp_offsets_are_bounded() {
        let stack = default_stack();
        for x in (-500..500).step_by(31) {
            for z in (-500..500).step_by(31) {
                let (wx, wz) = stack.warp_offsets(x as f64, z as f64);
                let max_w = stack.config.warp_strength;
                assert!(
                    wx.abs() <= max_w && wz.abs() <= max_w,
                    "Warp offset ({wx}, {wz}) exceeds strength {max_w}"
                );
            }
        }
    }

    // ── Micro-detail ───────────────────────────────────────────────────

    #[test]
    fn micro_detail_is_bounded() {
        let stack = default_stack();
        let max_amp = stack.config.micro_amplitude;
        for x in (-500..500).step_by(13) {
            for z in (-500..500).step_by(13) {
                let v = stack.micro_detail(x as f64, z as f64);
                assert!(
                    v.abs() <= max_amp,
                    "Micro-detail {v} exceeds amplitude {max_amp}"
                );
            }
        }
    }

    // ── Composite sample ───────────────────────────────────────────────

    #[test]
    fn sample_is_deterministic() {
        let stack = default_stack();
        let a = stack.sample(100.0, 200.0);
        let b = stack.sample(100.0, 200.0);
        assert_eq!(a, b);
    }

    #[test]
    fn sample_with_detail_is_deterministic() {
        let stack = default_stack();
        let a = stack.sample_with_detail(100.0, 200.0);
        let b = stack.sample_with_detail(100.0, 200.0);
        assert_eq!(a, b);
    }

    #[test]
    fn sample_varies_across_space() {
        let stack = default_stack();
        let mut values = Vec::new();
        for x in (0..2000).step_by(200) {
            values.push(stack.sample(x as f64, 0.0));
        }
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min > 0.01,
            "Composite sample shows no variation: min={min}, max={max}"
        );
    }

    #[test]
    fn sample_with_detail_differs_from_sample() {
        let stack = default_stack();
        let without = stack.sample(77.0, 33.0);
        let with = stack.sample_with_detail(77.0, 33.0);
        // Micro-detail should add a small offset (not guaranteed non-zero,
        // but highly unlikely to be exactly zero at a random point).
        assert!(
            (with - without).abs() < stack.config.micro_amplitude + 0.01,
            "Detail offset is too large"
        );
    }

    // ── Smoothstep / lerp helpers ──────────────────────────────────────

    #[test]
    fn smoothstep_boundaries() {
        assert!((smoothstep(0.0, 1.0, 0.0)).abs() < 1e-10);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 1e-10);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-10);
        // Clamped outside range
        assert!((smoothstep(0.0, 1.0, -1.0)).abs() < 1e-10);
        assert!((smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lerp_interpolates() {
        assert!((lerp(0.0, 10.0, 0.0)).abs() < 1e-10);
        assert!((lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
    }

    // ── Config variations ──────────────────────────────────────────────

    #[test]
    fn zero_octaves_returns_zero() {
        let stack = NoiseStack::new(
            42,
            NoiseConfig {
                fbm_octaves: 0,
                ridged_octaves: 0,
                ..Default::default()
            },
        );
        assert_eq!(stack.fbm(50.0, 50.0), 0.0);
        assert_eq!(stack.ridged(50.0, 50.0), 0.0);
    }

    #[test]
    fn zero_warp_strength_produces_no_offset() {
        let stack = NoiseStack::new(
            42,
            NoiseConfig {
                warp_strength: 0.0,
                ..Default::default()
            },
        );
        let (wx, wz) = stack.warp_offsets(100.0, 200.0);
        assert_eq!(wx, 0.0);
        assert_eq!(wz, 0.0);
    }

    #[test]
    fn selector_thresholds_control_blend() {
        // Force pure FBM: set selector thresholds so that all values are above lo.
        // Since Perlin is in [-1, 1], setting lo=1.1 ensures we always take the
        // "below lo" branch (pure FBM).
        let stack_fbm_only = NoiseStack::new(
            42,
            NoiseConfig {
                selector_thresholds: (1.1, 1.2),
                warp_strength: 0.0,
                ..Default::default()
            },
        );

        // Force pure ridged: set thresholds so selector is always above hi.
        let stack_ridged_only = NoiseStack::new(
            42,
            NoiseConfig {
                selector_thresholds: (-1.2, -1.1),
                warp_strength: 0.0,
                ..Default::default()
            },
        );

        let v_fbm = stack_fbm_only.sample(100.0, 100.0);
        let v_ridged = stack_ridged_only.sample(100.0, 100.0);
        // They should differ because they use different noise algorithms.
        assert!(
            (v_fbm - v_ridged).abs() > 1e-6,
            "Pure FBM and pure ridged should differ: fbm={v_fbm}, ridged={v_ridged}"
        );
    }

    // ── Continent mask tests ────────────────────────────────────────────

    fn continent_stack() -> NoiseStack {
        NoiseStack::new(
            42,
            NoiseConfig {
                continent_enabled: true,
                continent_freq: 0.002,
                continent_threshold: 0.05,
                shelf_blend_width: 0.15,
                ocean_floor_depth: 1.25,
                ocean_floor_amplitude: 0.25,
                ..Default::default()
            },
        )
    }

    #[test]
    fn continent_value_is_deterministic() {
        let stack = continent_stack();
        let v1 = stack.continent_value(100.0, 200.0);
        let v2 = stack.continent_value(100.0, 200.0);
        assert_eq!(v1, v2, "Continent value should be deterministic");
    }

    #[test]
    fn continent_blend_extremes() {
        let stack = continent_stack();
        // Scan many positions — we should find both full-land and deep-ocean
        let mut found_land = false;
        let mut found_ocean = false;
        let mut found_shelf = false;

        for i in 0..5000 {
            let x = (i as f64) * 7.3;
            let z = (i as f64) * 11.1;
            let blend = stack.continent_blend(x, z);
            assert!((0.0..=1.0).contains(&blend), "Blend {blend} out of range");

            if blend >= 1.0 {
                found_land = true;
            }
            if blend <= 0.0 {
                found_ocean = true;
            }
            if blend > 0.01 && blend < 0.99 {
                found_shelf = true;
            }
        }

        assert!(found_land, "Should find some land positions");
        assert!(found_ocean, "Should find some ocean positions");
        assert!(found_shelf, "Should find some shelf/transition positions");
    }

    #[test]
    fn continent_ocean_is_below_land() {
        let stack = continent_stack();
        // At ocean positions (blend=0), sample should be negative (below sea level)
        // At land positions (blend=1), sample should be the normal land terrain

        let mut ocean_samples = Vec::new();
        let mut land_samples = Vec::new();

        for i in 0..5000 {
            let x = (i as f64) * 3.7;
            let z = (i as f64) * 5.1;
            let blend = stack.continent_blend(x, z);
            let height = stack.sample(x, z);

            if blend <= 0.0 {
                ocean_samples.push(height);
            } else if blend >= 1.0 {
                land_samples.push(height);
            }
        }

        assert!(!ocean_samples.is_empty(), "No ocean samples found");
        assert!(!land_samples.is_empty(), "No land samples found");

        let ocean_avg: f64 = ocean_samples.iter().sum::<f64>() / ocean_samples.len() as f64;
        let land_avg: f64 = land_samples.iter().sum::<f64>() / land_samples.len() as f64;

        assert!(
            ocean_avg < land_avg,
            "Ocean avg ({ocean_avg:.3}) should be below land avg ({land_avg:.3})"
        );
        // Ocean should be well below zero (below sea level)
        assert!(
            ocean_avg < -0.5,
            "Ocean avg ({ocean_avg:.3}) should be well below sea level (< -0.5)"
        );
    }

    #[test]
    fn continent_disabled_has_no_effect() {
        let stack_off = NoiseStack::new(42, NoiseConfig::default());
        let stack_on = continent_stack();

        // With continent disabled, sample should equal sample_land
        let v_off = stack_off.sample(100.0, 200.0);
        let v_land = stack_off.sample_land(100.0, 200.0);
        assert_eq!(
            v_off, v_land,
            "Disabled continent should pass through land noise"
        );

        // With continent enabled at a land position, it should also equal land
        // (find a position with blend=1.0)
        for i in 0..5000 {
            let x = (i as f64) * 7.3;
            let z = (i as f64) * 11.1;
            if stack_on.continent_blend(x, z) >= 1.0 {
                let enabled = stack_on.sample(x, z);
                let land = stack_on.sample_land(x, z);
                assert!(
                    (enabled - land).abs() < 1e-10,
                    "At land position, continent sample ({enabled}) should equal land ({land})"
                );
                break;
            }
        }
    }

    #[test]
    fn continent_shelf_smoothly_transitions() {
        let stack = continent_stack();
        // Find a shelf region and verify blend is smooth
        let mut shelf_positions = Vec::new();

        for i in 0..10000 {
            let x = (i as f64) * 1.7;
            let z = (i as f64) * 2.9;
            let blend = stack.continent_blend(x, z);
            if blend > 0.05 && blend < 0.95 {
                shelf_positions.push((x, z, blend));
            }
            if shelf_positions.len() >= 20 {
                break;
            }
        }

        assert!(
            shelf_positions.len() >= 5,
            "Should find shelf transition positions, got {}",
            shelf_positions.len()
        );

        // At shelf positions, height should be between ocean and land extremes
        for &(x, z, blend) in &shelf_positions {
            let height = stack.sample(x, z);
            let land = stack.sample_land(x, z);
            let ocean = -stack.config.ocean_floor_depth + stack.ocean_floor_noise(x, z);
            let expected = lerp(ocean, land, blend);
            assert!(
                (height - expected).abs() < 1e-10,
                "Shelf height {height} should equal lerp({ocean}, {land}, {blend}) = {expected}"
            );
        }
    }
}
