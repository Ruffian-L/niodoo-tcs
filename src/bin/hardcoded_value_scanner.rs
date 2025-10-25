//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// Hardcoded Value Scanner CLI
///
/// Binary replacement for scripts/hardcoded_value_buster.py
use clap::Parser;
use niodoo_feeling::bullshit_buster::{HardcodedValueScanner, ScanConfig};
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber;

#[derive(Parser)]
#[command(
    name = "hardcoded-value-scanner",
    about = "Detect and analyze hardcoded values in codebase",
    version
)]
struct Args {
    /// Directory to scan
    #[arg(value_name = "DIRECTORY")]
    directory: PathBuf,

    /// Output report file
    #[arg(short, long, default_value = "hardcoded_analysis_report.json")]
    output: PathBuf,

    /// Generate replacement suggestions
    #[arg(short, long)]
    generate_suggestions: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Setup logging
    let log_level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    info!("Scanning directory: {:?}", args.directory);

    // Create scanner configuration
    let mut config = ScanConfig::default();
    config.output_file = args.output.clone();
    config.generate_suggestions = args.generate_suggestions;

    // Create scanner
    let scanner = HardcodedValueScanner::new(config)?;

    // Scan directory
    let mut result = scanner.scan_directory(&args.directory)?;

    // Generate suggestions if requested
    if args.generate_suggestions {
        info!("Generating replacement suggestions...");
        scanner.generate_suggestions(&mut result);
    }

    // Print summary
    tracing::info!("\n📊 HARDCODED VALUE SCAN REPORT 📊");
    tracing::info!("Files scanned: {}", result.summary.total_files_scanned);
    tracing::info!(
        "Files with hardcoded values: {}",
        result.summary.files_with_hardcoded_values
    );
    tracing::info!(
        "Total hardcoded instances: {}",
        result.summary.total_hardcoded_instances
    );

    // Show top offenders
    if !result.top_offenders.is_empty() {
        tracing::info!("\n🔥 TOP OFFENDERS:");
        for (i, offender) in result.top_offenders.iter().take(10).enumerate() {
            tracing::info!(
                "  {}. {:?}: {} instances",
                i + 1,
                offender.file,
                offender.instance_count
            );
        }
    }

    // Save report
    scanner.save_report(&result)?;
    tracing::info!("\n✅ Report saved to: {:?}", args.output);

    // Exit with error code if hardcoded values found
    if result.summary.total_hardcoded_instances > 0 {
        tracing::error!(
            "Found {} hardcoded values across {} files",
            result.summary.total_hardcoded_instances, result.summary.files_with_hardcoded_values
        );
        std::process::exit(1);
    }

    Ok(())
}
