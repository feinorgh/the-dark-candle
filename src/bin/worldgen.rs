//! Standalone planetary world generator and visualizer.
//!
//! Usage: `cargo run --bin worldgen -- --seed 42 --level 6 --stats`

use clap::Parser;
use the_dark_candle::planet::tectonics::run_tectonics;
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
    let mut planet = PlanetData::new(config);
    let grid_elapsed = start.elapsed();
    println!("  Grid built in {grid_elapsed:.2?}");

    let tec_start = std::time::Instant::now();
    run_tectonics(&mut planet, |_| {});
    let tec_elapsed = tec_start.elapsed();
    println!("  Tectonics done in {tec_elapsed:.2?}");

    let elapsed = start.elapsed();
    println!("Total: {elapsed:.2?}");
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
        let scale = r * r; // area on sphere of radius r = unit_area * r^2
        let avg_km2 = mean * scale / 1e6;
        let side_km = avg_km2.sqrt();
        println!(
            "  Avg cell area at radius {:.0} km: {avg_km2:.2} km^2 (~{side_km:.1} km side)",
            r / 1000.0
        );

        // Tectonic stats.
        let elevs = &planet.elevation;
        let land_cells = elevs.iter().filter(|&&e| e >= 0.0).count();
        let ocean_cells = elevs.iter().filter(|&&e| e < 0.0).count();
        let elev_min = elevs.iter().copied().fold(f64::INFINITY, f64::min);
        let elev_max = elevs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let n = elevs.len() as f64;
        let elev_mean = elevs.iter().sum::<f64>() / n;
        println!("\n  Tectonic results:");
        println!("    Elevation range: {elev_min:.0} m to {elev_max:.0} m (mean {elev_mean:.0} m)");
        println!(
            "    Land cells: {land_cells} ({:.1}%)",
            100.0 * land_cells as f64 / n
        );
        println!(
            "    Ocean cells: {ocean_cells} ({:.1}%)",
            100.0 * ocean_cells as f64 / n
        );

        use the_dark_candle::planet::BoundaryType;
        let conv = planet
            .boundary_type
            .iter()
            .filter(|&&b| b == BoundaryType::Convergent)
            .count();
        let divg = planet
            .boundary_type
            .iter()
            .filter(|&&b| b == BoundaryType::Divergent)
            .count();
        let xfrm = planet
            .boundary_type
            .iter()
            .filter(|&&b| b == BoundaryType::Transform)
            .count();
        println!("    Boundary cells — convergent: {conv}, divergent: {divg}, transform: {xfrm}");

        let volcanic: usize = planet
            .volcanic_activity
            .iter()
            .filter(|&&v| v > 0.01)
            .count();
        println!("    Active volcanic cells: {volcanic}");
    }
}
