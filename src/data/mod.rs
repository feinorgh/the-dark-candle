use std::collections::HashMap;
use std::path::Path;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::{Deserialize, Serialize};

use crate::world::voxel::MaterialId;

pub struct DataPlugin;

impl Plugin for DataPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<EnemyData>::new(&["enemy.ron"]))
            .add_plugins(RonAssetPlugin::<MaterialData>::new(&["material.ron"]))
            .add_plugins(RonAssetPlugin::<CreatureData>::new(&["creature.ron"]))
            .add_plugins(RonAssetPlugin::<ItemData>::new(&["item.ron"]))
            .add_plugins(RonAssetPlugin::<FluidConfig>::new(&["fluid_config.ron"]))
            .add_plugins(RonAssetPlugin::<PropData>::new(&["prop.ron"]))
            .add_plugins(RonAssetPlugin::<crate::procgen::tree::TreeConfig>::new(&[
                "tree.ron",
            ]));

        // Build MaterialRegistry synchronously from .material.ron files on disk.
        match load_material_registry() {
            Ok(registry) => {
                info!("Loaded MaterialRegistry with {} materials", registry.len());
                app.insert_resource(registry);
            }
            Err(e) => {
                warn!("Failed to load MaterialRegistry: {e}");
                app.init_resource::<MaterialRegistry>();
            }
        }
    }
}

/// Raw data loaded from `.enemy.ron` files.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct EnemyData {
    pub name: String,
    pub health: f32,
    pub speed: f32,
}

/// Material phase (solid, liquid, gas) at standard conditions.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phase {
    #[default]
    Solid,
    Liquid,
    Gas,
}

fn default_albedo() -> f32 {
    0.3
}

/// Spectral power distribution across physical wavelength bands.
///
/// Each field is the fraction of total `emission_power` emitted in that band.
/// Values should sum to 1.0.  Phase 1 uses `visible` (rendering) and
/// `thermal_ir` + `near_ir` (heat transfer).  `uv` and `near_ir` are stored
/// for future systems (fluorescence, night-vision goggles, IR scanners).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct EmissionSpectrum {
    /// UV band (280–380 nm). Fluorescence excitation, germicidal lamps.
    #[serde(default)]
    pub uv: f32,
    /// Visible band (380–700 nm). Determines rendered glow via `emission_color`.
    #[serde(default = "EmissionSpectrum::default_visible")]
    pub visible: f32,
    /// Near-IR band (700–1 400 nm). Night-vision devices, IR remote LEDs.
    #[serde(default = "EmissionSpectrum::default_near_ir")]
    pub near_ir: f32,
    /// Thermal IR band (1.4–15 µm). Radiative heat transfer, thermal cameras.
    #[serde(default = "EmissionSpectrum::default_thermal_ir")]
    pub thermal_ir: f32,
}

impl EmissionSpectrum {
    fn default_visible() -> f32 {
        0.85
    }
    fn default_near_ir() -> f32 {
        0.10
    }
    fn default_thermal_ir() -> f32 {
        0.05
    }

    /// Fraction of power that contributes to heating (near-IR + thermal IR).
    pub fn heating_fraction(&self) -> f32 {
        self.near_ir + self.thermal_ir
    }
}

impl Default for EmissionSpectrum {
    fn default() -> Self {
        Self {
            uv: 0.0,
            visible: 0.85,
            near_ir: 0.10,
            thermal_ir: 0.05,
        }
    }
}

/// Physical and chemical properties of a material, loaded from `.material.ron`.
/// The `id` field maps to `MaterialId` in the voxel system.
///
/// All values use SI units:
///   - density: kg/m³
///   - temperatures (melting, boiling, ignition): Kelvin
///   - thermal_conductivity: W/(m·K)
///   - specific_heat_capacity: J/(kg·K)
///   - hardness: Mohs scale (0–10)
///   - viscosity: Pa·s
///   - latent heats: J/kg
///   - heat_of_combustion: J/kg
///   - molar_mass: kg/mol
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq, Default)]
pub struct MaterialData {
    /// Numeric ID matching `MaterialId` in the voxel system (0 = air).
    pub id: u16,
    pub name: String,
    /// Phase at standard temperature/pressure.
    pub default_phase: Phase,
    /// Density in kg/m³.
    pub density: f32,
    /// Melting point in Kelvin (None for materials that don't melt, e.g. air).
    pub melting_point: Option<f32>,
    /// Boiling point in Kelvin.
    pub boiling_point: Option<f32>,
    /// Ignition temperature in Kelvin (None if non-flammable).
    pub ignition_point: Option<f32>,
    /// Scratch resistance on the Mohs scale (0 = air/fluids, 10 = diamond).
    /// Source: Wikipedia — Mohs scale of mineral hardness.
    pub hardness: f32,
    /// Base color for rendering (RGB, 0.0–1.0).
    pub color: [f32; 3],
    /// Whether light passes through this material.
    pub transparent: bool,

    // --- Thermal properties (SI) ---
    /// Thermal conductivity in W/(m·K). How fast heat flows through this material.
    /// Source: Wikipedia — List of thermal conductivities.
    #[serde(default)]
    pub thermal_conductivity: f32,
    /// Specific heat capacity in J/(kg·K). Energy to raise 1 kg by 1 K.
    /// Source: Wikipedia — Table of specific heat capacities.
    #[serde(default)]
    pub specific_heat_capacity: f32,
    /// Energy absorbed during solid→liquid transition (J/kg). None if no melting.
    /// Source: Wikipedia — Enthalpy of fusion.
    #[serde(default)]
    pub latent_heat_fusion: Option<f32>,
    /// Energy absorbed during liquid→gas transition (J/kg). None if no boiling.
    /// Source: Wikipedia — Enthalpy of vaporization.
    #[serde(default)]
    pub latent_heat_vaporization: Option<f32>,
    /// Emissivity for radiative heat transfer (0.0–1.0, dimensionless).
    #[serde(default)]
    pub emissivity: f32,
    /// Absorption coefficient for semi-transparent materials (m⁻¹).
    /// Beer-Lambert attenuation: transmittance = exp(−α × d).
    /// `None` = fully opaque (blocks radiation), `Some(α)` = attenuates per meter.
    #[serde(default)]
    pub absorption_coefficient: Option<f32>,

    // --- Mechanical properties (SI) ---
    /// Dynamic viscosity in Pa·s. Only meaningful for fluids.
    /// Source: Wikipedia — Viscosity.
    #[serde(default)]
    pub viscosity: Option<f32>,
    /// Kinetic friction coefficient (dimensionless, 0.0–1.0).
    /// Source: Wikipedia — Friction § Coefficient of friction.
    #[serde(default)]
    pub friction_coefficient: f32,
    /// Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly elastic).
    #[serde(default)]
    pub restitution: f32,
    /// Young's modulus in Pascals. Stiffness of the material.
    /// Source: Wikipedia — Young's modulus.
    #[serde(default)]
    pub youngs_modulus: Option<f32>,
    /// Maximum tensile stress before fracture (Pa). None for fluids/gases.
    /// Source: Wikipedia — Tensile strength.
    #[serde(default)]
    pub tensile_strength: Option<f32>,
    /// Maximum compressive stress before crushing (Pa). None for fluids/gases.
    /// Source: Wikipedia — Compressive strength.
    #[serde(default)]
    pub compressive_strength: Option<f32>,
    /// Maximum shear stress before failure (Pa). None for fluids/gases.
    /// Source: Wikipedia — Shear strength.
    #[serde(default)]
    pub shear_strength: Option<f32>,
    /// Maximum bending stress before snapping (Pa). None for fluids/gases.
    /// Source: Wikipedia — Flexural strength.
    #[serde(default)]
    pub flexural_strength: Option<f32>,
    /// Resistance to crack propagation (Pa·√m). None for fluids/gases.
    /// Source: Wikipedia — Fracture toughness.
    #[serde(default)]
    pub fracture_toughness: Option<f32>,

    // --- Chemical / combustion properties (SI) ---
    /// Energy released per kg when burned (J/kg). None if non-flammable.
    /// Source: Wikipedia — Heat of combustion.
    #[serde(default)]
    pub heat_of_combustion: Option<f32>,
    /// Molar mass in kg/mol. Used for ideal gas law calculations (gases only).
    #[serde(default)]
    pub molar_mass: Option<f32>,

    // --- Optical properties (SI) ---
    /// Index of refraction (dimensionless). Ratio of speed of light in vacuum
    /// to speed in this material. Determines bending at interfaces (Snell's law).
    /// Air ≈ 1.0003, water = 1.33, ice = 1.31, glass = 1.52, diamond = 2.42.
    /// `None` = opaque, no refraction. Source: Wikipedia — List of refractive indices.
    #[serde(default)]
    pub refractive_index: Option<f32>,
    /// Specular reflectivity (0.0–1.0, dimensionless). Fraction of incoming light
    /// reflected at normal incidence. Metals have high reflectivity (iron ≈ 0.65,
    /// polished steel ≈ 0.70). Most non-metals use Fresnel equations instead.
    /// `None` = derive from Fresnel (default for non-metals).
    #[serde(default)]
    pub reflectivity: Option<f32>,
    /// Per-channel RGB absorption coefficients (m⁻¹) for colored light
    /// attenuation through transparent materials (Beer-Lambert law).
    /// `[α_R, α_G, α_B]` — higher values mean more absorption on that channel.
    /// Water absorbs red strongly, giving blue tint: `[0.45, 0.07, 0.02]`.
    /// `None` = derive from scalar absorption_coefficient (equal on all channels).
    #[serde(default)]
    pub absorption_rgb: Option<[f32; 3]>,
    /// Cauchy B dispersion coefficient (m²). When `Some`, the material splits
    /// white light into a spectrum (dispersion). Combined with `refractive_index`
    /// (treated as n at the green reference wavelength 550 nm), gives per-channel
    /// refractive indices via Cauchy's equation: n(λ) = A + B/λ².
    ///
    /// Higher B = more dispersion (wider colour splitting):
    /// - Borosilicate glass: 4.61e-15 m²
    /// - Fused quartz / SiO₂: 3.4e-15 m²
    /// - Diamond: 1.1e-14 m² (very high dispersion)
    ///
    /// `None` = non-dispersive (single refractive index for all wavelengths).
    #[serde(default)]
    pub cauchy_b: Option<f32>,
    /// Surface albedo (fraction of incident solar radiation reflected).
    /// Range 0.0–1.0. Default: 0.3.
    /// Low values (dark surfaces) absorb more heat.
    /// Water ≈ 0.06, snow/ice ≈ 0.8, soil ≈ 0.2, stone ≈ 0.3.
    #[serde(default = "default_albedo")]
    pub albedo: f32,

    // --- Active emission properties (SI) ---
    /// Total radiant exitance in W/m² (power emitted per unit surface area).
    /// `None` = not an active emitter (default).  Independent of `emissivity`
    /// which governs passive Stefan-Boltzmann radiation.
    /// Warm-white LED panel ≈ 5 000, 60 W incandescent bulb surface ≈ 7 000.
    #[serde(default)]
    pub emission_power: Option<f32>,
    /// Visible emission colour (RGB, 0.0–1.0).  Determines the rendered glow
    /// colour.  `None` = warm white default `[1.0, 0.93, 0.84]`.
    #[serde(default)]
    pub emission_color: Option<[f32; 3]>,
    /// Spectral power distribution across UV / visible / near-IR / thermal-IR
    /// bands.  Each field is a fraction of `emission_power` (should sum to 1.0).
    /// `None` = LED-like default (85 % visible, 10 % near-IR, 5 % thermal-IR).
    #[serde(default)]
    pub emission_spectrum: Option<EmissionSpectrum>,

    // --- Phase transition targets ---
    /// Material name this becomes when heated above melting_point (solid → liquid).
    #[serde(default)]
    pub melted_into: Option<String>,
    /// Material name this becomes when heated above boiling_point (liquid → gas).
    #[serde(default)]
    pub boiled_into: Option<String>,
    /// Material name this becomes when cooled below melting_point (liquid → solid).
    #[serde(default)]
    pub frozen_into: Option<String>,
    /// Material name this becomes when cooled below boiling_point (gas → liquid).
    #[serde(default)]
    pub condensed_into: Option<String>,
}

impl MaterialData {
    /// Default warm-white emission colour (~3000 K correlated colour temperature).
    const DEFAULT_EMISSION_COLOR: [f32; 3] = [1.0, 0.93, 0.84];

    /// Whether this material is an active light emitter.
    pub fn is_emissive(&self) -> bool {
        matches!(self.emission_power, Some(p) if p > 0.0)
    }

    /// Resolved emission colour: explicit `emission_color` or warm-white default.
    pub fn resolved_emission_color(&self) -> [f32; 3] {
        self.emission_color.unwrap_or(Self::DEFAULT_EMISSION_COLOR)
    }

    /// Resolved emission spectrum: explicit or LED-like default.
    pub fn resolved_emission_spectrum(&self) -> EmissionSpectrum {
        self.emission_spectrum.unwrap_or_default()
    }

    /// Get per-channel RGB absorption coefficients for light transport.
    ///
    /// Returns `Some([α_R, α_G, α_B])` for transparent/semi-transparent
    /// materials, `None` for fully opaque. Falls back from `absorption_rgb`
    /// to scalar `absorption_coefficient` (broadcast to all channels).
    pub fn light_absorption_rgb(&self) -> Option<[f32; 3]> {
        if let Some(rgb) = self.absorption_rgb {
            return Some(rgb);
        }
        if self.transparent {
            if let Some(alpha) = self.absorption_coefficient {
                return Some([alpha, alpha, alpha]);
            }
            // Transparent but no absorption specified → fully transparent.
            return Some([0.0, 0.0, 0.0]);
        }
        None // Opaque
    }

    /// Per-channel refractive indices (R, G, B) accounting for dispersion.
    ///
    /// If `cauchy_b` is set, applies Cauchy dispersion so blue bends more than
    /// red (`n_B > n_G > n_R`). If only `refractive_index` is set, all three
    /// channels share the same index. Returns `None` for opaque materials.
    ///
    /// # Example
    /// ```
    /// # use the_dark_candle::data::MaterialData;
    /// let mut glass = MaterialData::default();
    /// glass.refractive_index = Some(1.52);
    /// glass.cauchy_b = Some(4.61e-15);
    /// let [nr, ng, nb] = glass.dispersion_n_rgb().unwrap();
    /// assert!(nb > ng && ng > nr);
    /// ```
    pub fn dispersion_n_rgb(&self) -> Option<[f32; 3]> {
        let base_n = self.refractive_index?;
        if let Some(b) = self.cauchy_b {
            Some(crate::lighting::optics::dispersive_n_rgb(base_n, b))
        } else {
            Some([base_n, base_n, base_n])
        }
    }
}

/// Global resource holding handles to loaded data assets.
#[derive(Resource, Default)]
pub struct GameAssets {
    pub goblin_data: Handle<EnemyData>,
}

// --- Creature Data ---

/// Dietary classification for creatures.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diet {
    Herbivore,
    Carnivore,
    Omnivore,
    Scavenger,
}

/// Size category affecting collision, visibility, and resource needs.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodySize {
    Tiny,
    Small,
    Medium,
    Large,
    Huge,
}

/// Base statistics for a creature species, loaded from `.creature.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct CreatureData {
    /// Unique species identifier string (e.g., "wolf", "deer").
    pub species: String,
    /// Display name shown to the player.
    pub display_name: String,
    /// Base health points.
    pub base_health: f32,
    /// Base movement speed (voxels per second).
    pub base_speed: f32,
    /// Base attack damage (0 for passive creatures).
    #[serde(default)]
    pub base_attack: f32,
    /// Body size category.
    pub body_size: BodySize,
    /// Dietary classification.
    pub diet: Diet,
    /// Collision half-extents (x, y, z) for AABB.
    pub hitbox: (f32, f32, f32),
    /// Base color for rendering (RGB, 0.0–1.0).
    pub color: [f32; 3],
    /// How much variation is allowed in stats (0.0–1.0, fraction of base).
    #[serde(default = "default_variation")]
    pub stat_variation: f32,
    /// Preferred biome names (empty = spawns anywhere).
    #[serde(default)]
    pub preferred_biomes: Vec<String>,
    /// Whether the creature is hostile to the player by default.
    #[serde(default)]
    pub hostile: bool,
    /// Lifespan in simulation ticks (None = immortal).
    #[serde(default)]
    pub lifespan: Option<u32>,
    /// Optional explicit body plan override for procedural animation.
    /// When `None`, the plan is derived from `body_size` via
    /// `BodyPlan::default_for_size`. Useful for species whose silhouette
    /// shouldn't match the body-size default (e.g., a tiny serpent, or a
    /// huge biped).
    #[serde(default)]
    pub body_plan: Option<crate::bodies::procedural_body::BodyPlan>,
}

fn default_variation() -> f32 {
    0.1
}

// --- Item Data ---

/// Category of item affecting how it can be used.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemCategory {
    Tool,
    Weapon,
    Armor,
    Food,
    Material,
    Container,
    Misc,
}

/// Template for procedural item generation, loaded from `.item.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct ItemData {
    /// Unique item type identifier (e.g., "sword", "pickaxe").
    pub item_type: String,
    /// Display name template (may include "{material}" placeholder).
    pub display_name: String,
    /// Item category.
    pub category: ItemCategory,
    /// Material name this item is primarily made of (influences properties).
    pub primary_material: String,
    /// Base weight in kg (modified by material density).
    pub base_weight: f32,
    /// Base durability (modified by material hardness).
    pub base_durability: f32,
    /// Base damage for weapons/tools (0 for non-weapons).
    #[serde(default)]
    pub base_damage: f32,
    /// Base armor value (0 for non-armor).
    #[serde(default)]
    pub base_armor: f32,
    /// Nutritional value if food (calories, 0 for non-food).
    #[serde(default)]
    pub nutrition: f32,
    /// Whether this item can be stacked in inventory.
    #[serde(default)]
    pub stackable: bool,
    /// Maximum stack size (only relevant if stackable).
    #[serde(default = "default_max_stack")]
    pub max_stack: u32,
}

fn default_max_stack() -> u32 {
    64
}

/// Category of world prop for spawn filtering.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropCategory {
    Rock,
    Vegetation,
    Debris,
    Misc,
}

/// Slope preference for prop placement.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlopePreference {
    /// Only flat ground (slope < 15°)
    Flat,
    /// Moderate slopes OK (< 45°)
    Moderate,
    /// Prefers steep slopes (> 30°)
    Steep,
    /// Any slope
    #[default]
    Any,
}

/// Template for world scenery props (rocks, logs, etc.), loaded from `.prop.ron`.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct PropData {
    /// Unique identifier (e.g., "boulder", "pebble").
    pub prop_type: String,
    /// Display name.
    pub display_name: String,
    /// Category for filtering.
    pub category: PropCategory,
    /// Material name (looked up in MaterialRegistry for color/physics).
    pub material: String,
    /// Base scale in meters (x, y, z).
    pub base_scale: (f32, f32, f32),
    /// Random scale variation factor (0.0 = no variation, 0.3 = ±30%).
    #[serde(default)]
    pub scale_variation: f32,
    /// Collision shape: "sphere", "aabb", or "none".
    #[serde(default = "default_collision")]
    pub collision: String,
    /// Slope preference for placement.
    #[serde(default)]
    pub slope_preference: SlopePreference,
    /// Minimum altitude (world units) for spawning, if restricted.
    #[serde(default)]
    pub min_altitude: Option<f32>,
    /// Maximum altitude (world units) for spawning, if restricted.
    #[serde(default)]
    pub max_altitude: Option<f32>,
}

fn default_collision() -> String {
    "aabb".into()
}

/// Lookup table from MaterialId to material properties, with name-based resolution.
/// Built at startup from loaded MaterialData assets.
#[derive(Resource, Debug, Default, Clone)]
pub struct MaterialRegistry {
    materials: HashMap<u16, MaterialData>,
    name_to_id: HashMap<String, u16>,
}

impl MaterialRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a material's properties, indexing by both ID and name.
    pub fn insert(&mut self, data: MaterialData) {
        self.name_to_id.insert(data.name.clone(), data.id);
        self.materials.insert(data.id, data);
    }

    /// Look up material data by numeric ID.
    pub fn get(&self, id: MaterialId) -> Option<&MaterialData> {
        self.materials.get(&id.0)
    }

    /// Look up material data by name.
    pub fn get_by_name(&self, name: &str) -> Option<&MaterialData> {
        let id = self.name_to_id.get(name)?;
        self.materials.get(id)
    }

    /// Resolve a material name to its numeric MaterialId.
    pub fn resolve_name(&self, name: &str) -> Option<MaterialId> {
        self.name_to_id.get(name).map(|&id| MaterialId(id))
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }
}

/// Build a `MaterialRegistry` by reading all `.material.ron` files from disk.
///
/// Searches `assets/data/materials/` relative to `CARGO_MANIFEST_DIR` (tests)
/// or the current working directory (runtime).
pub fn load_material_registry() -> Result<MaterialRegistry, String> {
    let dir = find_data_dir()?.join("materials");
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;

    let mut reg = MaterialRegistry::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".material.ron") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let data: MaterialData =
            ron::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        reg.insert(data);
    }

    if reg.is_empty() {
        return Err(format!("no .material.ron files found in {}", dir.display()));
    }
    Ok(reg)
}

/// Locate the `assets/data/` directory.
///
/// Checks `CARGO_MANIFEST_DIR` first (for tests), then falls back to paths
/// relative to the current working directory.
pub fn find_data_dir() -> Result<std::path::PathBuf, String> {
    if let Ok(dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let path = Path::new(&dir).join("assets").join("data");
        if path.is_dir() {
            return Ok(path);
        }
    }
    for candidate in ["assets/data", "data"] {
        let path = Path::new(candidate);
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }
    }
    Err("cannot find assets/data/ directory".into())
}

/// Configuration for fluid simulation systems, loaded from
/// `assets/data/fluid_config.ron`.
///
/// Contains parameters for both the AMR Navier-Stokes liquid solver and the
/// LBM gas solver. All values use SI units where applicable.
#[derive(Deserialize, Asset, TypePath, Debug, Clone, PartialEq)]
pub struct FluidConfig {
    // --- AMR Navier-Stokes (liquids) ---
    /// Maximum Jacobi iterations for the pressure Poisson solver.
    #[serde(default = "default_pressure_iterations")]
    pub pressure_solver_iterations: usize,

    /// Maximum allowed CFL number before warning. Accuracy degrades above 1.0.
    #[serde(default = "default_cfl_max")]
    pub cfl_max: f32,

    /// Default fluid density in kg/m³ (used when material lookup fails).
    #[serde(default = "default_density")]
    pub density_default: f32,

    /// Jacobi iterations for the viscosity diffusion solve.
    #[serde(default = "default_diffusion_iterations")]
    pub diffusion_iterations: usize,

    // --- LBM (gases) ---
    /// BGK relaxation time τ. Must be > 0.5 for stability.
    /// Higher values increase effective viscosity (more diffusion).
    /// Default 0.55 — stable with Smagorinsky sub-grid model.
    #[serde(default = "default_lbm_tau")]
    pub lbm_tau: f32,

    /// Smagorinsky constant Cs for sub-grid turbulence model.
    /// Standard value 0.1 (Smagorinsky 1963, Lilly 1966).
    /// Set to 0.0 to disable sub-grid model (pure BGK).
    #[serde(default = "default_lbm_smagorinsky_cs")]
    pub lbm_smagorinsky_cs: f32,

    /// Number of LBM sub-steps per FixedUpdate tick.
    /// Each sub-step advances gas dynamics by dt_lattice seconds.
    #[serde(default = "default_lbm_steps_per_tick")]
    pub lbm_steps_per_tick: usize,

    /// Whether the LBM gas simulation is active.
    #[serde(default = "default_lbm_enabled")]
    pub lbm_enabled: bool,

    // --- FLIP/PIC (particles) ---
    /// Whether the FLIP/PIC particle simulation is active.
    #[serde(default = "default_flip_enabled")]
    pub flip_enabled: bool,

    /// FLIP/PIC blend ratio. 0.0 = pure PIC (stable, diffusive),
    /// 1.0 = pure FLIP (detail-preserving, noisy). Default 0.97.
    #[serde(default = "default_flip_ratio")]
    pub flip_ratio: f32,

    /// Velocity threshold (m/s) below which particles deposit onto surfaces.
    #[serde(default = "default_flip_deposit_velocity")]
    pub flip_deposit_velocity: f32,

    /// Maximum particles per chunk (memory budget).
    #[serde(default = "default_flip_max_particles")]
    pub flip_max_particles_per_chunk: usize,

    /// Number of FLIP sub-steps per FixedUpdate tick for CFL stability.
    #[serde(default = "default_flip_substeps")]
    pub flip_substeps: usize,

    /// Jacobi iterations for the FLIP pressure projection.
    #[serde(default = "default_flip_pressure_iterations")]
    pub flip_pressure_iterations: usize,
}

fn default_pressure_iterations() -> usize {
    50
}
fn default_cfl_max() -> f32 {
    1.0
}
fn default_density() -> f32 {
    1000.0
}
fn default_diffusion_iterations() -> usize {
    20
}
fn default_lbm_tau() -> f32 {
    0.55
}
fn default_lbm_smagorinsky_cs() -> f32 {
    0.1
}
fn default_lbm_steps_per_tick() -> usize {
    1
}
fn default_lbm_enabled() -> bool {
    true
}
fn default_flip_enabled() -> bool {
    true
}
fn default_flip_ratio() -> f32 {
    0.97
}
fn default_flip_deposit_velocity() -> f32 {
    0.5
}
fn default_flip_max_particles() -> usize {
    4096
}
fn default_flip_substeps() -> usize {
    2
}
fn default_flip_pressure_iterations() -> usize {
    20
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            pressure_solver_iterations: default_pressure_iterations(),
            cfl_max: default_cfl_max(),
            density_default: default_density(),
            diffusion_iterations: default_diffusion_iterations(),
            lbm_tau: default_lbm_tau(),
            lbm_smagorinsky_cs: default_lbm_smagorinsky_cs(),
            lbm_steps_per_tick: default_lbm_steps_per_tick(),
            lbm_enabled: default_lbm_enabled(),
            flip_enabled: default_flip_enabled(),
            flip_ratio: default_flip_ratio(),
            flip_deposit_velocity: default_flip_deposit_velocity(),
            flip_max_particles_per_chunk: default_flip_max_particles(),
            flip_substeps: default_flip_substeps(),
            flip_pressure_iterations: default_flip_pressure_iterations(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enemy_data_deserializes_from_ron() {
        let ron_str = r#"EnemyData(name: "Goblin", health: 50.0, speed: 2.5)"#;
        let data: EnemyData = ron::from_str(ron_str).expect("Failed to deserialize EnemyData");
        assert_eq!(data.name, "Goblin");
        assert_eq!(data.health, 50.0);
        assert_eq!(data.speed, 2.5);
    }

    #[test]
    fn enemy_data_deserializes_multiline_ron() {
        let ron_str = r#"
            EnemyData(
                name: "Skeleton",
                health: 100.0,
                speed: 1.5,
            )
        "#;
        let data: EnemyData = ron::from_str(ron_str).expect("Failed to deserialize EnemyData");
        assert_eq!(data.name, "Skeleton");
        assert_eq!(data.health, 100.0);
        assert_eq!(data.speed, 1.5);
    }

    #[test]
    fn enemy_data_rejects_missing_fields() {
        let ron_str = r#"EnemyData(name: "Orc", health: 80.0)"#;
        assert!(ron::from_str::<EnemyData>(ron_str).is_err());
    }

    #[test]
    fn enemy_data_clone_is_equal() {
        let data = EnemyData {
            name: "Troll".into(),
            health: 200.0,
            speed: 0.8,
        };
        assert_eq!(data, data.clone());
    }

    #[test]
    fn goblin_ron_file_is_valid() {
        let contents = include_str!("../../assets/data/goblin.enemy.ron");
        let data: EnemyData =
            ron::from_str(contents).expect("goblin.enemy.ron failed to deserialize");
        assert!(!data.name.is_empty());
        assert!(data.health > 0.0);
        assert!(data.speed > 0.0);
    }

    #[test]
    fn material_data_deserializes_from_ron() {
        let ron_str = r#"
            MaterialData(
                id: 1,
                name: "Stone",
                default_phase: Solid,
                density: 2700.0,
                melting_point: Some(1473.0),
                boiling_point: Some(2773.0),
                ignition_point: None,
                hardness: 6.5,
                color: (0.5, 0.5, 0.5),
                transparent: false,
                thermal_conductivity: 2.5,
                specific_heat_capacity: 790.0,
                latent_heat_fusion: Some(400000.0),
                friction_coefficient: 0.7,
                restitution: 0.2,
                emissivity: 0.93,
                melted_into: Some("Lava"),
            )
        "#;
        let data: MaterialData =
            ron::from_str(ron_str).expect("Failed to deserialize MaterialData");
        assert_eq!(data.name, "Stone");
        assert_eq!(data.id, 1);
        assert_eq!(data.default_phase, Phase::Solid);
        assert_eq!(data.melting_point, Some(1473.0));
        assert_eq!(data.ignition_point, None);
        assert!(!data.transparent);
        assert_eq!(data.melted_into, Some("Lava".into()));
        // Verify new SI fields
        assert_eq!(data.thermal_conductivity, 2.5);
        assert_eq!(data.specific_heat_capacity, 790.0);
        assert_eq!(data.latent_heat_fusion, Some(400000.0));
        assert_eq!(data.friction_coefficient, 0.7);
    }

    #[test]
    fn material_data_air_has_no_melting_point() {
        let ron_str = r#"
            MaterialData(
                id: 0,
                name: "Air",
                default_phase: Gas,
                density: 1.225,
                melting_point: None,
                boiling_point: None,
                ignition_point: None,
                hardness: 0.0,
                color: (0.8, 0.9, 1.0),
                transparent: true,
                thermal_conductivity: 0.026,
                specific_heat_capacity: 1005.0,
                molar_mass: Some(0.0289647),
            )
        "#;
        let data: MaterialData = ron::from_str(ron_str).expect("Failed to deserialize air");
        assert!(data.transparent);
        assert_eq!(data.hardness, 0.0);
        assert_eq!(data.melting_point, None);
        assert_eq!(data.molar_mass, Some(0.0289647));
    }

    #[test]
    fn material_data_flammable_has_ignition_point() {
        let ron_str = r#"
            MaterialData(
                id: 5,
                name: "Wood",
                default_phase: Solid,
                density: 600.0,
                melting_point: None,
                boiling_point: None,
                ignition_point: Some(573.0),
                hardness: 2.0,
                color: (0.6, 0.4, 0.2),
                transparent: false,
                thermal_conductivity: 0.15,
                specific_heat_capacity: 1700.0,
                heat_of_combustion: Some(15000000.0),
            )
        "#;
        let data: MaterialData = ron::from_str(ron_str).expect("Failed to deserialize wood");
        assert_eq!(data.ignition_point, Some(573.0));
        assert_eq!(data.heat_of_combustion, Some(15_000_000.0));
    }

    #[test]
    fn material_data_new_fields_default_to_zero() {
        // Minimal RON with only the original fields — new fields should default
        let ron_str = r#"
            MaterialData(
                id: 99,
                name: "Test",
                default_phase: Solid,
                density: 1000.0,
                melting_point: None,
                boiling_point: None,
                ignition_point: None,
                hardness: 1.0,
                color: (1.0, 1.0, 1.0),
                transparent: false,
            )
        "#;
        let data: MaterialData = ron::from_str(ron_str).expect("Failed to deserialize");
        assert_eq!(data.thermal_conductivity, 0.0);
        assert_eq!(data.specific_heat_capacity, 0.0);
        assert_eq!(data.viscosity, None);
        assert_eq!(data.friction_coefficient, 0.0);
        assert_eq!(data.restitution, 0.0);
        assert_eq!(data.emissivity, 0.0);
        assert_eq!(data.latent_heat_fusion, None);
        assert_eq!(data.latent_heat_vaporization, None);
        assert_eq!(data.heat_of_combustion, None);
        assert_eq!(data.molar_mass, None);
        assert_eq!(data.youngs_modulus, None);
    }

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
            let data: MaterialData = ron::from_str(&contents)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
            assert!(!data.name.is_empty(), "{}: name is empty", path.display());
            assert!(data.density >= 0.0, "{}: negative density", path.display());
            assert!(
                data.hardness >= 0.0 && data.hardness <= 10.0,
                "{}: hardness out of Mohs range (0-10)",
                path.display()
            );
        }
    }

    // --- CreatureData tests ---

    #[test]
    fn creature_data_deserializes_from_ron() {
        let ron_str = r#"
            CreatureData(
                species: "wolf",
                display_name: "Grey Wolf",
                base_health: 80.0,
                base_speed: 6.0,
                base_attack: 15.0,
                body_size: Medium,
                diet: Carnivore,
                hitbox: (0.4, 0.5, 0.8),
                color: (0.5, 0.5, 0.5),
                hostile: true,
            )
        "#;
        let data: CreatureData =
            ron::from_str(ron_str).expect("Failed to deserialize CreatureData");
        assert_eq!(data.species, "wolf");
        assert_eq!(data.base_health, 80.0);
        assert_eq!(data.diet, Diet::Carnivore);
        assert_eq!(data.body_size, BodySize::Medium);
        assert!(data.hostile);
        assert_eq!(data.stat_variation, 0.1); // default
        assert!(data.preferred_biomes.is_empty()); // default
        assert!(data.lifespan.is_none()); // default
    }

    #[test]
    fn creature_data_with_all_fields() {
        let ron_str = r#"
            CreatureData(
                species: "deer",
                display_name: "Forest Deer",
                base_health: 50.0,
                base_speed: 8.0,
                base_attack: 0.0,
                body_size: Large,
                diet: Herbivore,
                hitbox: (0.5, 0.7, 1.0),
                color: (0.6, 0.4, 0.2),
                stat_variation: 0.2,
                preferred_biomes: ["forest", "meadow"],
                hostile: false,
                lifespan: Some(50000),
            )
        "#;
        let data: CreatureData = ron::from_str(ron_str).expect("Failed to deserialize deer");
        assert_eq!(data.stat_variation, 0.2);
        assert_eq!(data.preferred_biomes, vec!["forest", "meadow"]);
        assert_eq!(data.lifespan, Some(50000));
        assert!(!data.hostile);
    }

    #[test]
    fn creature_data_rejects_missing_species() {
        let ron_str = r#"CreatureData(display_name: "?", base_health: 1.0, base_speed: 1.0, body_size: Tiny, diet: Omnivore, hitbox: (0.1, 0.1, 0.1), color: (1.0, 1.0, 1.0))"#;
        assert!(ron::from_str::<CreatureData>(ron_str).is_err());
    }

    #[test]
    fn all_creature_ron_files_are_valid() {
        let pattern = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/data/creatures/*.creature.ron"
        );
        let files: Vec<_> = glob::glob(pattern)
            .expect("Failed to read glob pattern")
            .collect();

        assert!(
            !files.is_empty(),
            "No .creature.ron files found in assets/data/creatures/"
        );

        for entry in files {
            let path = entry.expect("Failed to read glob entry");
            let contents = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
            let data: CreatureData = ron::from_str(&contents)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
            assert!(
                !data.species.is_empty(),
                "{}: species is empty",
                path.display()
            );
            assert!(
                data.base_health > 0.0,
                "{}: health must be positive",
                path.display()
            );
            assert!(
                data.base_speed > 0.0,
                "{}: speed must be positive",
                path.display()
            );
        }
    }

    // --- ItemData tests ---

    #[test]
    fn item_data_deserializes_from_ron() {
        let ron_str = r#"
            ItemData(
                item_type: "sword",
                display_name: "Iron Sword",
                category: Weapon,
                primary_material: "Iron",
                base_weight: 2.5,
                base_durability: 100.0,
                base_damage: 20.0,
            )
        "#;
        let data: ItemData = ron::from_str(ron_str).expect("Failed to deserialize ItemData");
        assert_eq!(data.item_type, "sword");
        assert_eq!(data.category, ItemCategory::Weapon);
        assert_eq!(data.primary_material, "Iron");
        assert_eq!(data.base_damage, 20.0);
        assert!(!data.stackable); // default
        assert_eq!(data.max_stack, 64); // default
    }

    #[test]
    fn item_data_food_with_nutrition() {
        let ron_str = r#"
            ItemData(
                item_type: "apple",
                display_name: "Apple",
                category: Food,
                primary_material: "Air",
                base_weight: 0.2,
                base_durability: 10.0,
                nutrition: 150.0,
                stackable: true,
                max_stack: 16,
            )
        "#;
        let data: ItemData = ron::from_str(ron_str).expect("Failed to deserialize food");
        assert_eq!(data.category, ItemCategory::Food);
        assert_eq!(data.nutrition, 150.0);
        assert!(data.stackable);
        assert_eq!(data.max_stack, 16);
    }

    #[test]
    fn item_data_rejects_missing_fields() {
        let ron_str = r#"ItemData(item_type: "rock", display_name: "Rock")"#;
        assert!(ron::from_str::<ItemData>(ron_str).is_err());
    }

    #[test]
    fn all_item_ron_files_are_valid() {
        let pattern = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/data/items/*.item.ron");
        let files: Vec<_> = glob::glob(pattern)
            .expect("Failed to read glob pattern")
            .collect();

        assert!(
            !files.is_empty(),
            "No .item.ron files found in assets/data/items/"
        );

        for entry in files {
            let path = entry.expect("Failed to read glob entry");
            let contents = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
            let data: ItemData = ron::from_str(&contents)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
            assert!(
                !data.item_type.is_empty(),
                "{}: item_type is empty",
                path.display()
            );
            assert!(
                data.base_weight >= 0.0,
                "{}: negative weight",
                path.display()
            );
            assert!(
                data.base_durability >= 0.0,
                "{}: negative durability",
                path.display()
            );
        }
    }
}
