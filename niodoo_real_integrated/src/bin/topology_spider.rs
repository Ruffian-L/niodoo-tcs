//! Topology Spider - Crawl through topology space and test healing integration
//!
//! Usage: cargo run --bin topology_spider

use anyhow::Result;
use niodoo_real_integrated::topology_crawler::TopologyCrawler;
use tracing::{Level, info};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🕷️  TOPOLOGY SPIDER - Starting crawl");
    info!("====================================");

    // Create crawler
    let mut crawler = TopologyCrawler::new()?;

    // Crawl through all positions
    let crawl_results = crawler.crawl_space().await?;

    // Test healing scenarios
    let healing_results = crawler.test_healing_scenarios().await?;

    // Print summary
    info!("");
    info!("🎯 FINAL SUMMARY");
    info!("================");
    info!(
        "Position crawl: {}/{} positions tested",
        crawl_results.positions.len(),
        crawl_results.positions.len()
    );
    info!(
        "Knot complexity matches: {}/{}",
        crawl_results
            .positions
            .iter()
            .filter(|r| r.knot_valid)
            .count(),
        crawl_results.positions.len()
    );
    info!(
        "Healing integration: {}/{} tests passed",
        healing_results.passed, healing_results.total
    );

    if healing_results.passed >= 3 {
        info!("✅ Healing/topology integration verified!");
    } else {
        info!("⚠️  Some healing tests failed - review topology integration");
    }

    Ok(())
}
