use std::path::Path;

use crate::planet::PlanetData;

const PLANET_DIR: &str = "assets/data/planets";

pub fn planet_path(seed: u64) -> std::path::PathBuf {
    Path::new(PLANET_DIR).join(format!("{seed}.bin"))
}

pub fn save_planet(data: &PlanetData) -> Result<(), Box<dyn std::error::Error>> {
    let path = planet_path(data.config.seed);
    std::fs::create_dir_all(PLANET_DIR)?;
    let bytes = bincode::serde::encode_to_vec(data, bincode::config::standard())?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load_planet(seed: u64) -> Option<PlanetData> {
    let path = planet_path(seed);
    let bytes = std::fs::read(path).ok()?;
    let (data, _) =
        bincode::serde::decode_from_slice::<PlanetData, _>(&bytes, bincode::config::standard())
            .ok()?;
    Some(data)
}
