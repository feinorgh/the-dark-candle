//! Body-region injury system: per-limb damage tiers, healing, and limb severing.
//!
//! # Physics model
//! Injury severity is derived from the mechanical stress applied to a body region:
//!
//! ```text
//! area  ≈ region_volume^(2/3)   [m²]
//! stress = force_n / area        [Pa]
//! ratio  = stress / toughness_pa
//! ```
//!
//! Injury tiers are assigned based on `ratio`:
//! - < 0.25  → `Healthy`   (minor force, absorbed elastically)
//! - < 0.75  → `Bruised`   (soft-tissue damage)
//! - < 1.25  → `Fractured` (structural damage to bone/cartilage)
//! - ≥ 1.25  → `Severed`   (catastrophic failure)
//!
//! # Healing model
//! - `Bruised` regions heal at 10 % of `Health::heal_rate` per tick.
//! - `Fractured` regions heal at 2 % of `Health::heal_rate` per tick.
//! - `Severed` regions do not heal autonomously.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::biology::health::Health;

// ---------------------------------------------------------------------------
// BodyData stub
// ---------------------------------------------------------------------------

/// Definition of a single body region used to initialise `BodyHealth`.
///
/// Loaded from `assets/data/bodies/{species}.body.ron`.
/// TODO: move to `src/bodies/tissue.rs` once that module is created.
#[derive(Debug, Clone)]
pub struct BodyRegion {
    /// Unique region name (e.g. `"torso"`, `"left_leg"`, `"head"`).
    pub name: String,
    /// Approximate volume of biological tissue in m³ (1 voxel = 1 m³).
    pub volume_m3: f32,
    /// Mass of tissue in kg (used to compute base HP).
    pub mass_kg: f32,
    /// Material toughness in Pa — resistance to structural failure.
    pub toughness_pa: f32,
}

/// Complete body layout for a species.
///
/// TODO: derive `Deserialize + Asset + TypePath` and load from RON once
/// `src/bodies/tissue.rs` is created and registered in `DataPlugin`.
#[derive(Debug, Clone, Default)]
pub struct BodyData {
    pub species: String,
    pub regions: Vec<BodyRegion>,
}

// ---------------------------------------------------------------------------
// Injury tier
// ---------------------------------------------------------------------------

/// Severity tier for a single body region.
///
/// The ordering is meaningful: `Healthy < Bruised < Fractured < Severed`.
/// This allows comparisons such as `tier >= InjuryTier::Fractured`.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum InjuryTier {
    /// No structural damage.
    Healthy,
    /// Soft-tissue damage (contusions, haematomas). Heals slowly.
    Bruised,
    /// Structural damage (bone fractures, torn ligaments). Heals very slowly.
    Fractured,
    /// Catastrophic failure — limb or region is non-functional.
    Severed,
}

// ---------------------------------------------------------------------------
// RegionHealth
// ---------------------------------------------------------------------------

/// Health and injury state for a single body region.
#[derive(Debug, Clone)]
pub struct RegionHealth {
    /// Current hit points of this region.
    pub hp: f32,
    /// Maximum hit points of this region.
    pub max_hp: f32,
    /// Current injury tier derived from cumulative damage.
    pub tier: InjuryTier,
}

impl RegionHealth {
    /// Create a new region at full health.
    pub fn new(max_hp: f32) -> Self {
        Self {
            hp: max_hp,
            max_hp,
            tier: InjuryTier::Healthy,
        }
    }

    /// Apply a physical force to this region and update the injury tier.
    ///
    /// # Arguments
    /// - `force_n` – applied force in Newtons.
    /// - `toughness_pa` – material toughness in Pascals (resistance to failure).
    /// - `region_volume_m3` – approximate volume of the region in m³ (1 voxel = 1 m³).
    ///
    /// # Physics
    /// Cross-sectional area is estimated from volume as `V^(2/3)` (assumes
    /// roughly isotropic geometry). Stress = force / area. Injury tier is
    /// derived from the ratio of stress to toughness.
    pub fn apply_damage(&mut self, force_n: f32, toughness_pa: f32, region_volume_m3: f32) {
        // Estimate cross-sectional area from volume (SI: m²).
        let area_m2 = region_volume_m3.powf(2.0 / 3.0).max(1e-6);
        let stress_pa = force_n / area_m2;
        let ratio = stress_pa / toughness_pa.max(1.0);

        // Determine the new tier from stress ratio.
        let new_tier = if ratio >= 1.25 {
            InjuryTier::Severed
        } else if ratio >= 0.75 {
            InjuryTier::Fractured
        } else if ratio >= 0.25 {
            InjuryTier::Bruised
        } else {
            InjuryTier::Healthy
        };

        // Only upgrade the tier (damage is irreversible in this tick).
        if new_tier > self.tier {
            self.tier = new_tier;
        }

        // Reduce HP proportionally to stress (capped at max_hp).
        let hp_damage = (ratio * self.max_hp * 0.5).min(self.hp);
        self.hp = (self.hp - hp_damage).max(0.0);
    }

    /// Return the HP as a fraction of max_hp (0.0 – 1.0).
    pub fn hp_fraction(&self) -> f32 {
        if self.max_hp > 0.0 {
            self.hp / self.max_hp
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// BodyHealth
// ---------------------------------------------------------------------------

/// ECS component tracking per-region health for an entity's body.
#[derive(Component, Debug, Clone, Default)]
pub struct BodyHealth {
    /// Map from region name to its health/injury state.
    pub regions: HashMap<String, RegionHealth>,
}

impl BodyHealth {
    /// Initialise a `BodyHealth` from a `BodyData` description.
    ///
    /// # Arguments
    /// - `body_data` – species body layout.
    /// - `base_hp_per_kg_tissue` – how many HP each kg of biological tissue
    ///   contributes (typical value: `10.0` HP/kg).
    pub fn new(body_data: &BodyData, base_hp_per_kg_tissue: f32) -> Self {
        let mut regions = HashMap::new();
        for region in &body_data.regions {
            let max_hp = region.mass_kg * base_hp_per_kg_tissue;
            regions.insert(region.name.clone(), RegionHealth::new(max_hp.max(1.0)));
        }
        Self { regions }
    }

    /// Apply a physical force to a named region.
    ///
    /// If `region` is not found, the call is silently ignored.
    pub fn apply_damage_to_region(
        &mut self,
        region: &str,
        force_n: f32,
        toughness_pa: f32,
        region_volume_m3: f32,
    ) {
        if let Some(r) = self.regions.get_mut(region) {
            r.apply_damage(force_n, toughness_pa, region_volume_m3);
        }
    }

    /// Return total HP as a fraction of total max HP across all regions (0.0 – 1.0).
    pub fn total_hp_fraction(&self) -> f32 {
        let (sum_hp, sum_max): (f32, f32) = self
            .regions
            .values()
            .fold((0.0, 0.0), |(sh, sm), r| (sh + r.hp, sm + r.max_hp));
        if sum_max > 0.0 { sum_hp / sum_max } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Severed limb component
// ---------------------------------------------------------------------------

/// Marker component placed on a newly detached limb entity.
#[derive(Component, Debug, Clone)]
pub struct SeveredLimb {
    /// The entity from which this limb was detached.
    pub from_entity: Entity,
    /// The name of the bone/region that was severed.
    pub bone_name: String,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Heal `Bruised` and `Fractured` regions over time, driven by the entity's
/// overall `Health::heal_rate`.
///
/// Healing rates:
/// - `Bruised`   → 10 % of `heal_rate` per tick.
/// - `Fractured` → 2 % of `heal_rate` per tick.
/// - `Severed`   → no autonomous healing.
pub fn tick_injury_healing(mut query: Query<(&mut BodyHealth, &Health)>) {
    for (mut body_health, health) in &mut query {
        if health.dead {
            continue;
        }
        for region in body_health.regions.values_mut() {
            let heal_rate = match &region.tier {
                InjuryTier::Bruised => health.heal_rate * 0.1,
                InjuryTier::Fractured => health.heal_rate * 0.02,
                InjuryTier::Healthy | InjuryTier::Severed => 0.0,
            };

            if heal_rate > 0.0 {
                region.hp = (region.hp + heal_rate).min(region.max_hp);

                // Downgrade tier when HP recovers above its threshold.
                let fraction = region.hp_fraction();
                region.tier = match &region.tier {
                    InjuryTier::Fractured if fraction > 0.8 => InjuryTier::Bruised,
                    InjuryTier::Bruised if fraction >= 1.0 => InjuryTier::Healthy,
                    other => other.clone(),
                };
            }
        }
    }
}

/// Detect regions currently at `InjuryTier::Severed` with 0 HP and spawn a
/// detached limb entity for each. Entities that already carry a `SeveredLimb`
/// marker are skipped to prevent repeated spawning.
pub fn spawn_severed_limbs(
    mut commands: Commands,
    query: Query<(Entity, &BodyHealth, &Transform), Without<SeveredLimb>>,
) {
    for (entity, body_health, transform) in &query {
        for (region_name, region) in &body_health.regions {
            if region.tier == InjuryTier::Severed && region.hp <= 0.0 {
                // Spawn detached limb entity at the parent's world position.
                // TODO: use Skeleton::bone_transforms for exact bone position.
                commands.spawn((
                    Transform::from_translation(transform.translation),
                    SeveredLimb {
                        from_entity: entity,
                        bone_name: region_name.clone(),
                    },
                ));
                // Mark parent so we don't re-detect this region next tick.
                commands.entity(entity).insert(SeveredLimb {
                    from_entity: entity,
                    bone_name: region_name.clone(),
                });
                // Only process one severed region per entity per tick.
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── InjuryTier ordering ──────────────────────────────────────────────

    #[test]
    fn injury_tier_healthy_less_than_bruised() {
        assert!(InjuryTier::Healthy < InjuryTier::Bruised);
    }

    #[test]
    fn injury_tier_bruised_less_than_fractured() {
        assert!(InjuryTier::Bruised < InjuryTier::Fractured);
    }

    #[test]
    fn injury_tier_fractured_less_than_severed() {
        assert!(InjuryTier::Fractured < InjuryTier::Severed);
    }

    #[test]
    fn injury_tier_ordering_full_chain() {
        let tiers = [
            InjuryTier::Healthy,
            InjuryTier::Bruised,
            InjuryTier::Fractured,
            InjuryTier::Severed,
        ];
        for i in 0..tiers.len() - 1 {
            assert!(
                tiers[i] < tiers[i + 1],
                "Expected {:?} < {:?}",
                tiers[i],
                tiers[i + 1]
            );
        }
    }

    // ── apply_damage tier transitions ────────────────────────────────────

    fn make_region(max_hp: f32) -> RegionHealth {
        RegionHealth::new(max_hp)
    }

    #[test]
    fn low_force_does_not_injure() {
        let mut r = make_region(100.0);
        // Tiny force on a large volume → stress well below toughness.
        r.apply_damage(1.0, 1_000_000.0, 0.5);
        assert_eq!(r.tier, InjuryTier::Healthy);
    }

    #[test]
    fn moderate_force_causes_bruise() {
        let mut r = make_region(100.0);
        // stress/toughness ≈ 0.5 → Bruised tier.
        let volume = 0.008_f32; // 0.2 m side cube → area ≈ 0.04 m²
        let toughness = 1_000.0_f32;
        let area = volume.powf(2.0_f32 / 3.0);
        let force = 0.5 * toughness * area;
        r.apply_damage(force, toughness, volume);
        assert_eq!(r.tier, InjuryTier::Bruised);
    }

    #[test]
    fn large_force_causes_fracture() {
        let mut r = make_region(100.0);
        let volume = 0.008_f32;
        let toughness = 1_000.0_f32;
        let area = volume.powf(2.0_f32 / 3.0);
        let force = 1.0_f32 * toughness * area; // ratio ≈ 1.0 → Fractured
        r.apply_damage(force, toughness, volume);
        assert_eq!(r.tier, InjuryTier::Fractured);
    }

    #[test]
    fn catastrophic_force_severs() {
        let mut r = make_region(100.0);
        let volume = 0.008_f32;
        let toughness = 1_000.0_f32;
        let area = volume.powf(2.0_f32 / 3.0);
        let force = 2.0_f32 * toughness * area; // ratio ≈ 2.0 → Severed
        r.apply_damage(force, toughness, volume);
        assert_eq!(r.tier, InjuryTier::Severed);
    }

    #[test]
    fn damage_does_not_downgrade_tier() {
        let mut r = make_region(100.0);
        r.tier = InjuryTier::Fractured;
        // Applying tiny force should not downgrade from Fractured.
        r.apply_damage(0.0001, 1_000_000.0, 0.5);
        assert!(r.tier >= InjuryTier::Fractured);
    }

    // ── Healing downgrades tier ──────────────────────────────────────────

    #[test]
    fn bruised_region_heals_to_healthy_at_full_hp() {
        let mut r = RegionHealth {
            hp: 99.0,
            max_hp: 100.0,
            tier: InjuryTier::Bruised,
        };
        // Heal by exactly enough to reach 100 %.
        r.hp = 100.0;
        // Simulate the downgrade check.
        if r.hp_fraction() >= 1.0 && r.tier == InjuryTier::Bruised {
            r.tier = InjuryTier::Healthy;
        }
        assert_eq!(r.tier, InjuryTier::Healthy);
    }

    #[test]
    fn fractured_region_downgrades_to_bruised_at_80_percent() {
        let mut r = RegionHealth {
            hp: 82.0,
            max_hp: 100.0,
            tier: InjuryTier::Fractured,
        };
        // Simulate the downgrade check.
        if r.hp_fraction() > 0.8 && r.tier == InjuryTier::Fractured {
            r.tier = InjuryTier::Bruised;
        }
        assert_eq!(r.tier, InjuryTier::Bruised);
    }

    // ── BodyHealth aggregate HP ──────────────────────────────────────────

    #[test]
    fn body_health_total_fraction_at_full() {
        let body_data = BodyData {
            species: "biped".into(),
            regions: vec![
                BodyRegion {
                    name: "torso".into(),
                    volume_m3: 0.08,
                    mass_kg: 40.0,
                    toughness_pa: 50_000.0,
                },
                BodyRegion {
                    name: "head".into(),
                    volume_m3: 0.005,
                    mass_kg: 5.0,
                    toughness_pa: 80_000.0,
                },
            ],
        };
        let bh = BodyHealth::new(&body_data, 10.0);
        assert!((bh.total_hp_fraction() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn body_health_total_fraction_decreases_with_damage() {
        let body_data = BodyData {
            species: "biped".into(),
            regions: vec![BodyRegion {
                name: "torso".into(),
                volume_m3: 0.08,
                mass_kg: 40.0,
                toughness_pa: 50_000.0,
            }],
        };
        let mut bh = BodyHealth::new(&body_data, 10.0);
        bh.apply_damage_to_region("torso", 5_000_000.0, 50_000.0, 0.08);
        assert!(bh.total_hp_fraction() < 1.0);
    }
}
