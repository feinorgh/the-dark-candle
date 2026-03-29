pub mod decay;
pub mod growth;
pub mod health;
pub mod metabolism;
pub mod plants;

use bevy::prelude::*;

use crate::procgen::creatures::Creature;

/// System set for biology systems running on `FixedUpdate`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BiologySet;

/// Reason a creature died.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeathCause {
    OldAge,
    Health,
}

/// Event fired when a creature dies. Carries both a component flag
/// (`Health.dead`) for queries and this one-shot message for reactions.
#[derive(Message, Debug, Clone)]
pub struct CreatureDied {
    pub entity: Entity,
    pub cause: DeathCause,
}

pub struct BiologyPlugin;

impl Plugin for BiologyPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<CreatureDied>().add_systems(
            FixedUpdate,
            (
                tick_metabolism_system,
                tick_health_system.after(tick_metabolism_system),
                tick_growth_system,
                handle_creature_death
                    .after(tick_health_system)
                    .after(tick_growth_system),
            )
                .in_set(BiologySet)
                .after(crate::chemistry::runtime::ChemistrySet),
        );
    }
}

/// Depletes energy each tick; applies starvation damage when energy runs out.
fn tick_metabolism_system(
    mut query: Query<(&mut metabolism::Metabolism, &mut health::Health), With<Creature>>,
    time: Res<Time<Fixed>>,
) {
    let dt = time.delta_secs();
    for (mut meta, mut hp) in &mut query {
        let starvation = metabolism::tick_metabolism(&mut meta, dt);
        if starvation > 0.0 {
            hp.take_damage(starvation, health::DamageType::Starvation);
        }
    }
}

/// Processes status effects and natural healing.
fn tick_health_system(
    mut query: Query<(Entity, &mut health::Health, &metabolism::Metabolism), With<Creature>>,
    mut death_events: MessageWriter<CreatureDied>,
) {
    for (entity, mut hp, meta) in &mut query {
        if hp.dead {
            continue;
        }
        health::tick_health(&mut hp, meta.energy_fraction());
        if hp.dead {
            death_events.write(CreatureDied {
                entity,
                cause: DeathCause::Health,
            });
        }
    }
}

/// Ages creatures and detects death from old age.
fn tick_growth_system(
    mut query: Query<(Entity, &mut growth::Growth, &mut health::Health), With<Creature>>,
    mut death_events: MessageWriter<CreatureDied>,
) {
    for (entity, mut g, mut hp) in &mut query {
        if hp.dead {
            continue;
        }
        if growth::tick_growth(&mut g) {
            hp.dead = true;
            hp.current = 0.0;
            death_events.write(CreatureDied {
                entity,
                cause: DeathCause::OldAge,
            });
        }
    }
}

/// Despawns dead creatures.
// TODO: place corpse voxels via decay::place_corpse() before despawning.
fn handle_creature_death(mut commands: Commands, mut events: MessageReader<CreatureDied>) {
    for event in events.read() {
        commands.entity(event.entity).despawn();
    }
}
