//! Standalone planetary world generator and visualizer.
//!
//! Usage: `cargo run --bin worldgen -- --seed 42 --level 6 --stats`

use clap::Parser;
use the_dark_candle::planet::biomes::run_biomes;
use the_dark_candle::planet::geology::{
    ORE_COAL, ORE_COPPER, ORE_GEMS, ORE_GOLD, ORE_IRON, ORE_OIL, ORE_SULFUR, run_geology,
};
use the_dark_candle::planet::impacts::run_impacts;
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

    /// Open interactive 3D globe viewer.
    #[arg(long)]
    globe: bool,

    /// Export map projection (equirectangular, mollweide, orthographic).
    #[arg(long)]
    projection: Option<String>,

    /// Output file path for projection export.
    #[arg(long, default_value = "world.png")]
    output: String,

    /// Image width for projection / animation export (pixels).
    #[arg(long, default_value_t = 2048)]
    width: u32,

    /// Colour mode for projection / globe (elevation, biome, plates, age,
    /// crust, tidal, rock, temp).
    #[arg(long, default_value = "elevation")]
    colourmode: String,

    /// Export a rotating orthographic animation to the given path.
    #[arg(long)]
    animate: Option<String>,

    /// Use GPU compute shaders for projection rendering.
    #[arg(long)]
    gpu: bool,

    /// Enable tectonic time-lapse mode in the globe viewer.
    /// Captures simulation snapshots for step-by-step playback.
    #[arg(long)]
    timelapse: bool,
}

fn main() {
    let args = Args::parse();

    let config = PlanetConfig {
        seed: args.seed,
        grid_level: args.level,
        radius_m: args.radius_km * 1000.0,
        mass_kg: 5.972e24, // Earth-like mass
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
    let history = if args.timelapse {
        use the_dark_candle::planet::tectonics::run_tectonics_with_history;
        let h = run_tectonics_with_history(&mut planet, None, |_| {});
        println!(
            "  Tectonics done in {:.2?} ({} frames captured)",
            tec_start.elapsed(),
            h.frame_count()
        );
        Some(h)
    } else {
        run_tectonics(&mut planet, |_| {});
        println!("  Tectonics done in {:.2?}", tec_start.elapsed());
        None
    };

    let imp_start = std::time::Instant::now();
    run_impacts(&mut planet);
    let imp_elapsed = imp_start.elapsed();
    println!("  Impacts done in {imp_elapsed:.2?}");

    let bio_start = std::time::Instant::now();
    run_biomes(&mut planet);
    let bio_elapsed = bio_start.elapsed();
    println!("  Biomes done in {bio_elapsed:.2?}");

    let geo_start = std::time::Instant::now();
    run_geology(&mut planet);
    let geo_elapsed = geo_start.elapsed();
    println!("  Geology done in {geo_elapsed:.2?}");

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

        // Impact stats.
        use the_dark_candle::planet::impacts::generate_impact_events;
        let events =
            generate_impact_events(planet.config.bombardment_intensity, planet.config.seed);
        println!("\n  Impact results:");
        println!("    Events generated: {}", events.len());
        if !events.is_empty() {
            let max_depth = events
                .iter()
                .map(|e| e.depth_m)
                .fold(f64::NEG_INFINITY, f64::max);
            let max_radius = events
                .iter()
                .map(|e| e.radius_cells)
                .fold(f64::NEG_INFINITY, f64::max);
            println!("    Largest crater: depth {max_depth:.0} m, radius {max_radius:.1} cells");
        }

        // Celestial stats.
        let cel = &planet.celestial;
        let star = &cel.star;
        let star_class = match star.temperature_k as u32 {
            0..=3700 => "M",
            3701..=5200 => "K",
            5201..=6000 => "G",
            6001..=7500 => "F",
            7501..=10000 => "A",
            _ => "B/O",
        };
        println!("\n  Celestial system:");
        println!(
            "    Star: {star_class}-type, T={:.0} K, L={:.3e} W, R={:.3e} m",
            star.temperature_k, star.luminosity_w, star.radius_m
        );
        println!(
            "    Planet orbit: {:.3e} m ({:.2} AU), period {:.2} Earth-years",
            cel.planet_orbit_m,
            cel.planet_orbit_m / the_dark_candle::planet::celestial::AU,
            cel.planet_orbital_period_s / 3.156e7
        );
        println!("    Moons: {}", cel.moons.len());
        for (i, moon) in cel.moons.iter().enumerate() {
            println!(
                "      Moon {}: R={:.0} km, a={:.3e} m, T={:.2} days",
                i + 1,
                moon.radius_m / 1000.0,
                moon.semi_major_axis_m,
                moon.orbital_period_s / 86400.0
            );
        }
        if let Some(ring) = &cel.ring {
            println!(
                "    Ring: {:.3e}–{:.3e} m, opacity={:.2}",
                ring.inner_radius_m, ring.outer_radius_m, ring.opacity
            );
        } else {
            println!("    Ring: none");
        }

        // Tidal range at epoch 0 over a sample of cells.
        if !cel.moons.is_empty() {
            use the_dark_candle::planet::grid::CellId;
            let t = 0.0;
            let tidal: Vec<f64> = (0..planet.grid.cell_count())
                .map(|i| {
                    let pos = planet.grid.cell_position(CellId(i as u32));
                    cel.tidal_height_at(pos, t)
                })
                .collect();
            let t_min = tidal.iter().copied().fold(f64::INFINITY, f64::min);
            let t_max = tidal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!("    Tidal range at epoch: {t_min:.2} m to {t_max:.2} m");
        }

        // Biome stats.
        use the_dark_candle::planet::BiomeType;
        let n = planet.grid.cell_count() as f32;
        let ocean_cells = planet
            .biome
            .iter()
            .filter(|&&b| matches!(b, BiomeType::Ocean | BiomeType::DeepOcean))
            .count();
        let land_cells = n as usize - ocean_cells;
        let temp_min = planet
            .temperature_k
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let temp_max = planet
            .temperature_k
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let temp_mean = planet.temperature_k.iter().sum::<f32>() / n;
        let precip_mean = planet.precipitation_mm.iter().sum::<f32>() / n;

        println!("\n  Biome results:");
        println!("    Temperature: {temp_min:.0} K to {temp_max:.0} K (mean {temp_mean:.0} K)");
        println!("    Mean precipitation: {precip_mean:.0} mm/year");
        println!("    Land cells with biome data: {land_cells}");

        // Count top biomes.
        let mut biome_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for &b in &planet.biome {
            *biome_counts.entry(format!("{b:?}")).or_default() += 1;
        }
        let mut biome_list: Vec<_> = biome_counts.iter().collect();
        biome_list.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for (name, count) in biome_list.iter().take(5) {
            println!("      {name}: {count} ({:.1}%)", 100.0 * **count as f32 / n);
        }

        println!("\n  Geology results:");
        let mut rock_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for &r in &planet.surface_rock {
            *rock_counts.entry(format!("{r:?}")).or_default() += 1;
        }
        let mut rock_list: Vec<_> = rock_counts.iter().collect();
        rock_list.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for (name, count) in rock_list.iter().take(5) {
            println!("      {name}: {count} ({:.1}%)", 100.0 * **count as f32 / n);
        }

        let ore_cells = |mask: u16| {
            planet
                .ore_deposits
                .iter()
                .filter(|&&d| d & mask != 0)
                .count()
        };
        println!("    Ore deposits (land cells with each type):");
        println!(
            "      Iron: {}, Copper: {}, Gold: {}, Coal: {}",
            ore_cells(ORE_IRON),
            ore_cells(ORE_COPPER),
            ore_cells(ORE_GOLD),
            ore_cells(ORE_COAL)
        );
        println!(
            "      Sulfur: {}, Gems: {}, Oil: {}",
            ore_cells(ORE_SULFUR),
            ore_cells(ORE_GEMS),
            ore_cells(ORE_OIL)
        );
    }

    // ── Projection export ─────────────────────────────────────────────────
    use the_dark_candle::planet::projections::{Projection, render_animation, render_projection};
    use the_dark_candle::planet::render::ColourMode;

    let mode = ColourMode::from_name(&args.colourmode).unwrap_or_default();

    if let Some(proj_name) = &args.projection {
        let Some(projection) = Projection::from_name(proj_name) else {
            eprintln!(
                "Unknown projection '{proj_name}'. Use: equirectangular, mollweide, orthographic"
            );
            return;
        };
        let start = std::time::Instant::now();
        let img = if args.gpu {
            use the_dark_candle::gpu::render_projection_gpu;
            println!("Using GPU compute path...");
            render_projection_gpu(&planet, &projection, &mode, args.width)
        } else {
            render_projection(&planet, &projection, &mode, args.width)
        };
        println!(
            "Projection rendered in {:.2?} ({}×{})",
            start.elapsed(),
            img.width(),
            img.height()
        );
        img.save(&args.output)
            .expect("Failed to save projection image");
        println!("Saved to {}", args.output);
    }

    if let Some(anim_path) = &args.animate {
        let start = std::time::Instant::now();
        println!("Rendering animation ({} frames)...", 360);
        let result = if args.gpu {
            use the_dark_candle::gpu::render_animation_gpu;
            println!("Using GPU compute path...");
            render_animation_gpu(&planet, &mode, args.width, 360, anim_path)
        } else {
            render_animation(&planet, &mode, args.width, 360, anim_path)
        };
        if let Err(e) = result {
            eprintln!("Animation error: {e}");
        } else {
            println!("Animation done in {:.2?} → {anim_path}", start.elapsed());
        }
    }

    if args.globe {
        use the_dark_candle::planet::render::run_globe_viewer;
        run_globe_viewer(planet, history);
    }
}
