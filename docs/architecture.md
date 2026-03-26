# Architecture & Data Flow

## 1. The Entity Component System (ECS)
This game strictly adheres to Bevy's ECS model.
- **Components:** Pure data structs. No logic.
- **Systems:** Pure logic functions. They operate on Components via Queries.
- **Resources:** Global singletons (e.g., Score, Time, Asset handles).

## 2. The Data-Driven Pipeline (YAML / RON)
To ensure maximum flexibility, all game data is loaded at runtime via `bevy_common_assets`.

### How to create a new Data Type:
If the user requests a new game entity (e.g., "Add a Goblin enemy"):
1. **Create the Rust Struct:** Create a data container in `src/data/` deriving the necessary traits.
   ```rust
   #[derive(serde::Deserialize, bevy::asset::Asset, bevy::reflect::TypePath)]
   pub struct EnemyData {
       pub name: String,
       pub health: f32,
       pub speed: f32,
   }
