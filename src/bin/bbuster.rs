//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::path::PathBuf;
use clap::{Parser, Subcommand};
use walkdir::WalkDir;
use anyhow::Result;
use tracing::info;

mod parser; // Import the parser module (to be implemented)
mod topology_mapper; // Import topology mapper (to be implemented)
mod detectors; // Import detectors (to be implemented)
mod emotional; // Import emotional analyzer (to be implemented)
mod report; // Import report generator (to be implemented)

use parser::CodeParser;
use topology_mapper::TopologyMapper;
use detectors::DetectorRegistry;
use emotional::EmotionalAnalyzer;
use report::ReportGenerator;

// Existing imports from core
use niodoo_consciousness::dual_mobius_gaussian::DualMobiusGaussian;
use niodoo_consciousness::gaussian_process::GaussianProcess;
use niodoo_consciousness::feeling_model::FeelingModel;

#[derive(Parser)]
#[command(name = "bbuster")]
#[command(about = "Bullshit Buster - Code review with Gaussian Möbius Topology")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Scan {
        /// Target file or directory
        target: PathBuf,
        
        /// Enable topology-based analysis
        #[arg(long)]
        topo_flip: bool,
        
        /// Enable emotional overlay
        #[arg(long)]
        emotional: bool,
        
        /// Output format (terminal, json)
        #[arg(long, default_value = "terminal")]
        format: String,
        
        /// Number of k-flips for topology analysis
        #[arg(long, default_value = "3")]
        k_flips: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Scan { target, topo_flip, emotional, format, k_flips } => {
            let parser = CodeParser::new();
            let asts = parser.parse_directory(&target)?;

            let mobius = DualMobiusGaussian::new(); // Use existing
            let mapper = TopologyMapper::new(&mobius);
            let topologies = asts.iter().map(|ast| mapper.map_to_topology(ast)).collect::<Result<Vec<_>, _>>()?;

            let registry = DetectorRegistry::new();
            let issues: Vec<_> = topologies
                .iter()
                .flat_map(|topo| registry.run_all(topo))
                .collect();

            let feeling_model = FeelingModel::new(); // Use existing
            let emotional_analyzer = EmotionalAnalyzer::new(&feeling_model);
            let emotional_issues: Vec<_> = issues
                .iter()
                .map(|issue| emotional_analyzer.analyze_emotion(issue))
                .collect();

            let report_gen = ReportGenerator::new(format);
            let report = report_gen.generate(&issues, &emotional_issues, &topologies)?;
            info!("{}", report);
        }
    }

    Ok(())
}
