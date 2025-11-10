//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Mind's Eye Web Visualizer
//!
//! Web-based 3D visualization of NIODOO cognitive state using Three.js.
//! Perfect for SSH/RunPod scenarios - just open in browser!
//! Serves HTML page that connects to NIODOO telemetry stream via WebSocket.

mod web_server;

use clap::Parser;

#[derive(Parser)]
#[command(name = "niodoo-visualizer")]
#[command(about = "Web-based real-time visualization of NIODOO cognitive state")]
struct Args {
    /// Port to serve web interface on
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    println!("🧠 NIODOO Mind's Eye Web Visualizer");
    println!("====================================");
    println!("Starting web server on port {}...", args.port);
    println!();
    println!("📡 To use:");
    println!("   1. Start NIODOO with: NIODOO_TELEMETRY_ENABLED=true cargo run");
    println!("   2. Open browser to: http://localhost:{}", args.port);
    println!("   3. If SSH'd in, use port forwarding:");
    println!("      ssh -L {}:localhost:{} user@runpod-host", args.port, args.port);
    println!();
    
    web_server::start_web_server(args.port).await
}

