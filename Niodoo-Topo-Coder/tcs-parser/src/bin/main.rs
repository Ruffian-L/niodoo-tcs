//! tcs-parser CLI - Code-to-Topology Analysis Tool
//!
//! Analyzes source code files and outputs topological features + complexity metrics
//! in JSON format for NIODOO-CODE training data generation.
//!
//! NO PRINTLN except for final JSON output. Use log crate for diagnostics.

use clap::{Parser, Subcommand};
use anyhow::{Context, Result};
use log::{info, error};
use serde_json;
use std::path::PathBuf;
use std::fs;

use tcs_parser::{
    get_ast,
    graph::ast_to_graph,
    matrix::graph_to_matrix,
    complexity::compute_complexity,
};

#[derive(Parser)]
#[command(name = "tcs-parser")]
#[command(about = "Code-to-Topology Analysis for NIODOO-CODE", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze a source code file
    Analyze {
        /// Path to the source code file
        #[arg(short, long)]
        file: PathBuf,

        /// Programming language (rust, python)
        #[arg(short, long)]
        language: String,

        /// Output file path (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();

    match cli.command {
        Commands::Analyze { file, language, output } => {
            analyze_file(&file, &language, output.as_deref())?;
        }
    }

    Ok(())
}

fn analyze_file(file_path: &PathBuf, language: &str, output_path: Option<&std::path::Path>) -> Result<()> {
    info!("Analyzing file: {:?}", file_path);

    // Read source code
    let source_code = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file: {:?}", file_path))?;

    // Parse to AST
    info!("Parsing AST...");
    let tree = get_ast(&source_code, language)
        .with_context(|| format!("Failed to parse {} code", language))?;

    // Build control flow graph
    info!("Building control flow graph...");
    let graph = ast_to_graph(&tree, &source_code)
        .context("Failed to build control flow graph")?;

    // Convert to adjacency matrix
    info!("Converting to adjacency matrix...");
    let matrix = graph_to_matrix(&graph)
        .context("Failed to convert graph to matrix")?;

    // Compute complexity metrics
    info!("Computing complexity metrics...");
    let complexity = compute_complexity(&tree, &source_code);

    // Build JSON output
    let output = serde_json::json!({
        "file": file_path.to_string_lossy(),
        "language": language,
        "graph": {
            "node_count": graph.node_count(),
            "edge_count": graph.edge_count(),
        },
        "matrix": {
            "shape": matrix.shape(),
            "data": matrix.iter().copied().collect::<Vec<f64>>(),
        },
        "complexity": complexity,
    });

    // Write output
    let json_str = serde_json::to_string_pretty(&output)
        .context("Failed to serialize JSON")?;

    if let Some(out_path) = output_path {
        info!("Writing output to: {:?}", out_path);
        fs::write(out_path, json_str)
            .with_context(|| format!("Failed to write output to: {:?}", out_path))?;
    } else {
        // Output to stdout
        println!("{}", json_str);
    }

    info!("Analysis complete!");
    Ok(())
}

