/// Bullshit Detector CLI
///
/// Binary replacement for scripts/fresh_bullshit_scan.py
use clap::Parser;
use niodoo_feeling::bullshit_buster::BullshitDetector;
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber;

#[derive(Parser)]
#[command(
    name = "bullshit-detector",
    about = "Detect fake/placeholder/stub code in codebase",
    version
)]
struct Args {
    /// Directory to scan
    #[arg(value_name = "DIRECTORY")]
    directory: PathBuf,

    /// Output report file
    #[arg(short, long, default_value = "bullshit_detection_report.json")]
    output: PathBuf,

    /// Maximum number of offenders to display
    #[arg(short = 'n', long, default_value = "10")]
    top_n: usize,

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

    info!(
        "Scanning directory for fake/stub code: {:?}",
        args.directory
    );

    // Create detector
    let detector = BullshitDetector::new()?;

    // Scan directory
    let result = detector.scan_directory(&args.directory)?;

    // Print summary
    tracing::info!("\n📊 BULLSHIT DETECTION REPORT 📊");
    tracing::info!("Total files analyzed: {}", result.summary.total_files);
    tracing::info!("Files with fake code: {}", result.summary.files_with_fake);
    tracing::info!("Total fake instances: {}", result.summary.fake_instances);
    tracing::info!(
        "Fake code percentage: {:.1}%",
        result.summary.fake_percentage
    );

    // Show category breakdown
    if !result.category_counts.is_empty() {
        tracing::info!("\n📈 FAKE CODE CATEGORIES:");
        let mut categories: Vec<_> = result.category_counts.iter().collect();
        categories.sort_by(|a, b| b.1.cmp(a.1));

        for (category, count) in categories {
            let percentage = if result.summary.fake_instances > 0 {
                (*count as f64 / result.summary.fake_instances as f64) * 100.0
            } else {
                0.0
            };
            tracing::info!("  {}: {} instances ({:.1}%)", category, count, percentage);
        }
    }

    // Show worst offenders
    if !result.worst_offenders.is_empty() {
        tracing::info!("\n🚨 WORST OFFENDERS:");
        for (i, offender) in result.worst_offenders.iter().take(args.top_n).enumerate() {
            tracing::info!(
                "  {}. {:?}: {} fake instances",
                i + 1,
                offender.file,
                offender.fake_count
            );
            tracing::info!("     Categories: {}", offender.categories.join(", "));
        }
    }

    // Save detailed report
    detector.save_report(&result, &args.output)?;
    tracing::info!("\n✅ Detailed results saved to: {:?}", args.output);

    // Exit with error code if fake code found
    if result.summary.fake_instances > 0 {
        tracing::error!(
            "Found {} fake code instances across {} files ({:.1}% of files)",
            result.summary.fake_instances,
            result.summary.files_with_fake,
            result.summary.fake_percentage
        );
        std::process::exit(1);
    }

    tracing::info!("\n✨ No fake code detected! Clean codebase.");
    Ok(())
}
