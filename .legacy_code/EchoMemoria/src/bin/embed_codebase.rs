//! CLI tool for managing codebase embeddings
//! 
//! This tool provides command-line interface for embedding operations
//! that can be called by the MCP server or used manually.

use std::path::PathBuf;
use std::sync::Arc;
use clap::{Parser, Subcommand};
use anyhow::Result;
use log::{info, warn};
use env_logger;

use echo_memoria::embeddings::{
    EmbeddingEngine, CodeWatcher, ChromaSync, DualWriteManager, 
    McpEmbeddingNotifier, MockMcpNotifier
};

/// CLI tool for managing codebase embeddings
#[derive(Parser)]
#[command(name = "embed_codebase")]
#[command(about = "Manage codebase embeddings for semantic search")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize embeddings for the entire codebase
    Init {
        /// Force re-embedding even if up-to-date
        #[arg(short, long)]
        force: bool,
        
        /// Root directory to embed
        #[arg(short, long, default_value = ".")]
        root: PathBuf,
    },
    
    /// Embed a specific file
    Embed {
        /// Path to the file to embed
        file_path: PathBuf,
    },
    
    /// Start file watcher for real-time updates
    Watch {
        /// Root directory to watch
        #[arg(short, long, default_value = ".")]
        root: PathBuf,
        
        /// Debounce time in milliseconds
        #[arg(short, long, default_value = "1000")]
        debounce: u64,
    },
    
    /// Sync embeddings to ChromaDB
    Sync {
        /// ChromaDB URL
        #[arg(short, long, default_value = "http://127.0.0.1:8000")]
        chroma_url: String,
    },
    
    /// Show embedding statistics
    Stats {
        /// Root directory to analyze
        #[arg(short, long, default_value = ".")]
        root: PathBuf,
    },
    
    /// Verify embedding integrity
    Verify {
        /// Root directory to verify
        #[arg(short, long, default_value = ".")]
        root: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::init();
    
    match cli.command {
        Commands::Init { force, root } => {
            init_embeddings(root, force).await?;
        }
        
        Commands::Embed { file_path } => {
            embed_file(file_path).await?;
        }
        
        Commands::Watch { root, debounce } => {
            watch_files(root, debounce).await?;
        }
        
        Commands::Sync { chroma_url } => {
            sync_to_chroma(chroma_url).await?;
        }
        
        Commands::Stats { root } => {
            show_stats(root).await?;
        }
        
        Commands::Verify { root } => {
            verify_embeddings(root).await?;
        }
    }
    
    Ok(())
}

/// Initialize embeddings for the entire codebase
async fn init_embeddings(root: PathBuf, force: bool) -> Result<()> {
    info!("🚀 Initializing codebase embeddings...");
    
    // Create embedding engine
    let db_path = root.join(".embeddings");
    let engine = Arc::new(EmbeddingEngine::new(db_path)?);
    
    // Set up MCP notifier
    let mcp_notifier = Arc::new(McpEmbeddingNotifier::new(
        "http://127.0.0.1:8000".to_string()
    ));
    let mut engine_mut = (*engine).clone();
    engine_mut.set_mcp_notifier(mcp_notifier);
    
    // Create watcher for batch processing
    let watcher = CodeWatcher::new(
        root.clone(),
        engine.clone(),
        vec![
            "*.rs".to_string(),
            "*.cpp".to_string(),
            "*.h".to_string(),
            "*.qml".to_string(),
            "*.py".to_string(),
            "*.md".to_string(),
        ],
        vec![
            "target/".to_string(),
            ".venv/".to_string(),
            "node_modules/".to_string(),
            ".git/".to_string(),
            "build/".to_string(),
        ],
        1000,
        50,
    )?;
    
    // Batch embed all files
    watcher.batch_embed_all(root).await?;
    
    info!("✅ Codebase embedding initialization complete!");
    Ok(())
}

/// Embed a specific file
async fn embed_file(file_path: PathBuf) -> Result<()> {
    info!("📄 Embedding file: {:?}", file_path);
    
    if !file_path.exists() {
        return Err(anyhow::anyhow!("File does not exist: {:?}", file_path));
    }
    
    // Create embedding engine
    let db_path = file_path.parent().unwrap().join(".embeddings");
    let engine = Arc::new(EmbeddingEngine::new(db_path)?);
    
    // Generate embedding
    let embedding = engine.embed_file(&file_path).await?;
    
    // Save to sidecar file
    engine.save_embedding(&file_path, embedding).await?;
    
    info!("✅ File embedded successfully!");
    Ok(())
}

/// Start file watcher for real-time updates
async fn watch_files(root: PathBuf, debounce: u64) -> Result<()> {
    info!("👀 Starting file watcher...");
    
    // Create embedding engine
    let db_path = root.join(".embeddings");
    let engine = Arc::new(EmbeddingEngine::new(db_path)?);
    
    // Set up MCP notifier
    let mcp_notifier = Arc::new(McpEmbeddingNotifier::new(
        "http://127.0.0.1:8000".to_string()
    ));
    let mut engine_mut = (*engine).clone();
    engine_mut.set_mcp_notifier(mcp_notifier);
    
    // Create watcher
    let mut watcher = CodeWatcher::new(
        root.clone(),
        engine,
        vec![
            "*.rs".to_string(),
            "*.cpp".to_string(),
            "*.h".to_string(),
            "*.qml".to_string(),
            "*.py".to_string(),
            "*.md".to_string(),
        ],
        vec![
            "target/".to_string(),
            ".venv/".to_string(),
            "node_modules/".to_string(),
            ".git/".to_string(),
            "build/".to_string(),
        ],
        debounce,
        10,
    )?;
    
    // Start watching
    watcher.start(root).await?;
    
    info!("✅ File watcher started! Press Ctrl+C to stop.");
    
    // Keep running until interrupted
    tokio::signal::ctrl_c().await?;
    info!("🛑 Stopping file watcher...");
    
    watcher.stop().await?;
    info!("✅ File watcher stopped.");
    
    Ok(())
}

/// Sync embeddings to ChromaDB
async fn sync_to_chroma(chroma_url: String) -> Result<()> {
    info!("🔄 Syncing embeddings to ChromaDB...");
    
    // Create ChromaDB sync client
    let chroma_sync = ChromaSync::new(chroma_url, "code_embeddings".to_string(), 50);
    
    // Check if ChromaDB is available
    if !chroma_sync.health_check().await? {
        return Err(anyhow::anyhow!("ChromaDB is not available"));
    }
    
    // Ensure collection exists
    chroma_sync.ensure_collection().await?;
    
    info!("✅ ChromaDB sync complete!");
    Ok(())
}

/// Show embedding statistics
async fn show_stats(root: PathBuf) -> Result<()> {
    info!("📊 Embedding Statistics");
    info!("======================");
    
    // Create embedding engine
    let db_path = root.join(".embeddings");
    let engine = Arc::new(EmbeddingEngine::new(db_path)?);
    
    // Get stats
    let stats = engine.get_stats().await?;
    
    info!("Total files: {}", stats.total_files);
    info!("Embedded files: {}", stats.embedded_files);
    info!("Stale embeddings: {}", stats.stale_embeddings);
    info!("Average embed time: {:.2}ms", stats.avg_embed_time_ms);
    info!("Last sync: {}", stats.last_sync);
    info!("Cache hits: {}", stats.cache_hits);
    info!("Cache misses: {}", stats.cache_misses);
    
    Ok(())
}

/// Verify embedding integrity
async fn verify_embeddings(root: PathBuf) -> Result<()> {
    info!("🔍 Verifying embedding integrity...");
    
    let mut total_files = 0;
    let mut embedded_files = 0;
    let mut stale_embeddings = 0;
    
    // Walk directory tree
    let mut entries = Vec::new();
    collect_files(&root, &mut entries)?;
    
    total_files = entries.len();
    
    for path in entries {
        if let Ok(is_current) = EmbeddingEngine::new(root.join(".embeddings"))?.is_embedding_current(&path) {
            if is_current {
                embedded_files += 1;
            } else {
                stale_embeddings += 1;
            }
        }
    }
    
    info!("📊 Verification Results:");
    info!("Total files: {}", total_files);
    info!("Embedded files: {}", embedded_files);
    info!("Stale embeddings: {}", stale_embeddings);
    info!("Coverage: {:.1}%", (embedded_files as f64 / total_files as f64) * 100.0);
    
    if stale_embeddings > 0 {
        warn!("⚠️  {} embeddings are stale and need updating", stale_embeddings);
    } else {
        info!("✅ All embeddings are up-to-date!");
    }
    
    Ok(())
}

/// Collect files matching patterns
fn collect_files(root: &std::path::Path, entries: &mut Vec<std::path::PathBuf>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            // Skip ignored directories
            if should_ignore_dir(&path) {
                continue;
            }
            collect_files(&path, entries)?;
        } else if path.is_file() {
            // Check if file matches patterns
            if matches_pattern(&path) {
                entries.push(path);
            }
        }
    }
    
    Ok(())
}

/// Check if directory should be ignored
fn should_ignore_dir(path: &std::path::Path) -> bool {
    let ignore_dirs = ["target", ".venv", "node_modules", ".git", "build"];
    ignore_dirs.iter().any(|dir| path.file_name().unwrap_or_default() == *dir)
}

/// Check if file matches embedding patterns
fn matches_pattern(path: &std::path::Path) -> bool {
    if let Some(extension) = path.extension() {
        if let Some(ext_str) = extension.to_str() {
            return matches!(ext_str, "rs" | "cpp" | "h" | "qml" | "py" | "md");
        }
    }
    false
}

