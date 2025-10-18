/// MCP STDIO Bridge Binary
///
/// Binary replacement for mcp_stdio_wrapper.py
use clap::Parser;
use niodoo_feeling::mcp::StdioBridge;
use tracing::{error, info};
use tracing_subscriber;

#[derive(Parser)]
#[command(
    name = "mcp-stdio-bridge",
    about = "MCP STDIO bridge for El Chapo v3.2",
    version
)]
struct Args {
    /// MCP server URL
    #[arg(
        short,
        long,
        env = "MCP_SERVER_URL",
        default_value = "http://localhost:8002"
    )]
    server_url: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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

    info!("Starting MCP STDIO Bridge");
    info!("Server URL: {}", args.server_url);

    // Create bridge
    let bridge = StdioBridge::new(Some(args.server_url));

    // Run bridge
    if let Err(e) = bridge.run().await {
        tracing::error!("Bridge error: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
