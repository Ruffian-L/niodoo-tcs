//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// Configure MCP CLI
///
/// Binary replacement for configure_mcp.py
use clap::Parser;
use niodoo_feeling::config::configure_claude_mcp;
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber;

#[derive(Parser)]
#[command(
    name = "configure-mcp",
    about = "Configure Claude Code MCP servers for Niodoo-Feeling project",
    version
)]
struct Args {
    /// Project path (defaults to current directory)
    #[arg(short, long)]
    project_path: Option<PathBuf>,

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

    // Determine project path
    let project_path = args
        .project_path
        .unwrap_or_else(|| std::env::current_dir().expect("Failed to get current directory"));

    info!("Configuring MCP servers for project: {:?}", project_path);

    // Configure MCP
    if let Err(e) = configure_claude_mcp(&project_path) {
        tracing::error!("Failed to configure MCP: {}", e);
        std::process::exit(1);
    }

    tracing::info!("\n✅ MCP configuration complete!");
    tracing::info!("\nNext steps:");
    tracing::info!("1. Restart Claude Code (or reload the window)");
    tracing::info!("2. Check for MCP tools with: /mcp");
    tracing::info!("3. Tools should appear as: mcp__query_embeddings, mcp__embed_repo, etc.");

    Ok(())
}
