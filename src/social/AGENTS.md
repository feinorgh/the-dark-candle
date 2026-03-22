# Social Module

Entity relationships, faction groups, reputation from observed actions, and group coordination behaviors.

## Files

| File | Purpose |
|------|---------|
| `relationships.rs` | `CreatureId`, per-entity trust/familiarity/hostility |
| `factions.rs` | `FactionId`, groups, territory claims, inter-faction standings |
| `reputation.rs` | Witness-based reputation effects, faction propagation |
| `group_behaviors.rs` | Cooperative hunting, territory defense, food sharing, rally |

## Key Types

- `CreatureId(u64)` — maps to `Entity::to_bits()` for ECS integration
- `Relationship { trust: f32, familiarity: f32, hostility: f32 }`
- `FactionId(u32)` — arbitrary group identifier
- `FactionRelation { standing: f32 }` — inter-faction disposition

## Disposition Thresholds

- **Friendly:** `trust - hostility > 0.1`
- **Hostile:** `trust - hostility < -0.1`
- **Same faction default:** disposition = 0.8
- **Alliance:** inter-faction standing > 0.5
- **War:** inter-faction standing < -0.5

## Reputation Effect Strengths

| Action | Trust Δ (friend target) | Hostility Δ (friend target) |
|--------|------------------------|----------------------------|
| Attack | -0.3 | +0.4 |
| Kill | -0.6 | +0.7 |
| Help/Defend | +0.3 | -0.2 |
| ShareFood | +0.2 | -0.1 |
| Steal | -0.4 | +0.3 |
| Flee | -0.1 | 0.0 |

Effects are inverted when the observer dislikes the target. Faction propagation applies at 50% strength.

## Dependencies

- **Imports from:** (self-contained — no external crate:: imports)
- **Imported by:** (standalone; behavior module will query relationships for decision-making)

## Patterns

- `Relationships` is a `HashMap<CreatureId, Relationship>` wrapped in a Component.
- `FactionRegistry` is a central resource holding all factions, standings, and creature-faction mappings.
- `decay_relationships()` drifts values toward neutral over time; `prune_forgotten()` removes zero-familiarity entries.
- `propagate_to_faction()` excludes both the action's actor and the propagating witness from receiving secondhand info.

## Gotchas

- `Relationships::get()` takes `CreatureId` by value (it's `Copy`), not by reference. Don't write `.get(&id)`.
- `FactionRegistry` uses canonical pair ordering `(min_id, max_id)` for relation keys — `get_standing(A, B)` and `get_standing(B, A)` return the same value.
- Territory claims are `Vec<[i32; 2]>` (chunk XZ coords) with deduplication on insert. Large factions with many claims may want a `HashSet` in the future.
