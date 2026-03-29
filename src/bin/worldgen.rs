//! Standalone planetary world generator and visualizer.
//!
//! Usage: `cargo run --bin worldgen -- --seed 42 --level 6 --stats`

use clap::Parser;
use the_dark_candle::planet::{PlanetConfig, PlanetData};

/// Planetary world generator and visualizer.
#[derive(Parser, Debug)]
#[command(name = "worldgen", about = "Generate and visualize planetary worlds")]
struct Args {
    /// World generation seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Geodesic grid subdivision level (0–10).
    #[arg(long, default_value_t = 7)]
    level: u32,

    /// Planet radius in kilometers.
    #[arg(long, default_value_t = 6371.0)]
    radius_km: f64,

    /// Print generation statistics.
    #[arg(long)]
    stats: bool,
}

fn main() {
    let args = Args::parse();

    let config = PlanetConfig {
        seed: args.seed,
        grid_level: args.level,
        radius_m: args.radius_km * 1000.0,
        ..Default::default()
    };

    println!(
        "Generating planet (seed={}, level={})...",
        config.seed, config.grid_level
    );

    let start = std::time::Instant::now();
    let planet = PlanetData::new(config);
    let elapsed = start.elapsed();

    println!("Generated in {elapsed:.2?}");
    println!("  Cells: {}", planet.grid.cell_count());
    println!(
        "  Pentagons: {}",
        planet
            .grid
            .cell_ids()
            .filter(|&id| planet.grid.is_pentagon(id))
            .count()
    );

    if args.stats {
        let areas = planet.grid.all_areas();
        let total: f64 = areas.iter().sum();
        let mean = total / areas.len() as f64;
        let min = areas.iter().copied().fold(f64::INFINITY, f64::min);
        let max = areas.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        println!(
            "  Total area: {total:.6} (expected {:.6})",
            4.0 * std::f64::consts::PI
        );
        println!("  Cell area — min: {min:.8}, mean: {mean:.8}, max: {max:.8}");
        println!("  Area ratio (max/min): {:.4}", max / min);

        // Convert to real-world scale.
        let r = planet.config.radius_m;
        let scale = r * r; // area on sphere of radius r = unit_area × r²
        let avg_km2 = mean * scale / 1e6;
        let side_km = avg_km2.sqrt();
        println!(
            "  Avg cell area at radius {:.0} km: {avg_km2:.2} km² (~{side_km:.1} km side)",
            r / 1000.0
        );
    }
}
