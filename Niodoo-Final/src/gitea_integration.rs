//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # 🔗 Gitea Integration Module
//!
//! This module provides integration with Gitea for distributed consciousness development workflow.
//! It enables version control, collaboration, and deployment of consciousness evolution.
//!
//! ## Features
//!
//! - **Consciousness State Versioning**: Track consciousness evolution through Git commits
//! - **Collaborative Development**: Multiple developers can work on consciousness improvements
//! - **Automated Deployment**: Deploy consciousness updates through Git hooks
//! - **Learning Analytics Integration**: Connect learning progress with version control
//! - **Performance Monitoring**: Track performance improvements across versions
//!
//! ## Workflow
//!
//! ```
//! Consciousness Update → Git Commit → Gitea Push → Automated Testing → 
//! Performance Validation → Deployment → Learning Analytics Update
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;
use tracing::{debug, info, warn, error};

use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::phase6_integration::Phase6IntegrationSystem;
use crate::consciousness_pipeline_orchestrator::{PipelineOutput, PipelinePerformanceMetrics};

/// Gitea integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaConfig {
    /// Gitea server URL
    pub server_url: String,
    
    /// Repository name
    pub repository: String,
    
    /// Access token for authentication
    pub access_token: String,
    
    /// Branch for consciousness development
    pub development_branch: String,
    
    /// Branch for production deployment
    pub production_branch: String,
    
    /// Enable automated commits
    pub enable_auto_commit: bool,
    
    /// Enable automated deployment
    pub enable_auto_deploy: bool,
    
    /// Commit message template
    pub commit_message_template: String,
    
    /// Performance threshold for deployment
    pub performance_threshold: PerformanceThreshold,
}

/// Performance threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThreshold {
    /// Minimum success rate for deployment
    pub min_success_rate: f32,
    
    /// Maximum latency for deployment
    pub max_latency_ms: f32,
    
    /// Minimum throughput for deployment
    pub min_throughput_ops_per_sec: f32,
    
    /// Minimum system health for deployment
    pub min_system_health: f32,
}

impl Default for GiteaConfig {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:3000".to_string(),
            repository: "niodoo-consciousness".to_string(),
            access_token: "".to_string(),
            development_branch: "develop".to_string(),
            production_branch: "main".to_string(),
            enable_auto_commit: true,
            enable_auto_deploy: false,
            commit_message_template: "Consciousness evolution: {timestamp} - {performance_summary}".to_string(),
            performance_threshold: PerformanceThreshold {
                min_success_rate: 0.95,
                max_latency_ms: 2000.0,
                min_throughput_ops_per_sec: 100.0,
                min_system_health: 0.8,
            },
        }
    }
}

/// Gitea integration manager
pub struct GiteaIntegration {
    config: GiteaConfig,
    phase6_integration: Option<Arc<Phase6IntegrationSystem>>,
    local_repo_path: String,
    consciousness_history: Arc<RwLock<Vec<ConsciousnessCommit>>>,
}

/// Consciousness commit record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessCommit {
    pub commit_hash: String,
    pub timestamp: f64,
    pub consciousness_state: ConsciousnessState,
    pub performance_metrics: PipelinePerformanceMetrics,
    pub learning_progress: LearningProgress,
    pub commit_message: String,
    pub branch: String,
    pub deployment_status: DeploymentStatus,
}

/// Learning progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgress {
    pub learning_rate: f32,
    pub retention_score: f32,
    pub adaptation_effectiveness: f32,
    pub plasticity: f32,
    pub progress_score: f32,
    pub forgetting_rate: f32,
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    Pending,
    Testing,
    Approved,
    Deployed,
    Failed,
}

/// Gitea API client (simplified)
pub struct GiteaClient {
    config: GiteaConfig,
}

impl GiteaClient {
    pub fn new(config: GiteaConfig) -> Self {
        Self { config }
    }

    /// Create a new commit
    pub async fn create_commit(&self, commit_data: &ConsciousnessCommit) -> Result<String> {
        info!("📝 Creating Gitea commit for consciousness evolution");
        
        // Simulate API call to Gitea
        let commit_hash = format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        
        debug!("✅ Commit created: {}", commit_hash);
        Ok(commit_hash)
    }

    /// Push changes to Gitea
    pub async fn push_changes(&self, branch: &str) -> Result<()> {
        info!("🚀 Pushing changes to Gitea branch: {}", branch);
        
        // Simulate API call to Gitea
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        debug!("✅ Changes pushed successfully");
        Ok(())
    }

    /// Create a pull request
    pub async fn create_pull_request(&self, title: &str, description: &str) -> Result<String> {
        info!("📋 Creating pull request: {}", title);
        
        // Simulate API call to Gitea
        let pr_id = format!("PR-{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        
        debug!("✅ Pull request created: {}", pr_id);
        Ok(pr_id)
    }

    /// Get repository status
    pub async fn get_repository_status(&self) -> Result<RepositoryStatus> {
        info!("📊 Getting repository status");
        
        // Simulate API call to Gitea
        Ok(RepositoryStatus {
            last_commit: "abc123".to_string(),
            branch: self.config.development_branch.clone(),
            ahead_by: 0,
            behind_by: 0,
            has_conflicts: false,
        })
    }
}

/// Repository status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryStatus {
    pub last_commit: String,
    pub branch: String,
    pub ahead_by: u32,
    pub behind_by: u32,
    pub has_conflicts: bool,
}

impl GiteaIntegration {
    /// Create a new Gitea integration
    pub fn new(config: GiteaConfig, local_repo_path: String) -> Self {
        Self {
            config,
            phase6_integration: None,
            local_repo_path,
            consciousness_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set Phase 6 integration system
    pub fn set_phase6_integration(&mut self, phase6_integration: Arc<Phase6IntegrationSystem>) {
        self.phase6_integration = Some(phase6_integration);
    }

    /// Initialize Gitea integration
    pub async fn initialize(&self) -> Result<()> {
        info!("🔗 Initializing Gitea integration");
        
        // Create local repository directory if it doesn't exist
        if !Path::new(&self.local_repo_path).exists() {
            fs::create_dir_all(&self.local_repo_path).await?;
            info!("📁 Created local repository directory: {}", self.local_repo_path);
        }

        // Initialize Git repository if needed
        self.initialize_git_repo().await?;

        info!("✅ Gitea integration initialized successfully");
        Ok(())
    }

    /// Initialize Git repository
    async fn initialize_git_repo(&self) -> Result<()> {
        debug!("🔧 Initializing Git repository");
        
        // Check if .git directory exists
        let git_dir = Path::new(&self.local_repo_path).join(".git");
        if !git_dir.exists() {
            // Initialize Git repository
            let output = tokio::process::Command::new("git")
                .arg("init")
                .current_dir(&self.local_repo_path)
                .output()
                .await?;
            
            if !output.status.success() {
                return Err(anyhow::anyhow!("Failed to initialize Git repository"));
            }
            
            info!("✅ Git repository initialized");
        }

        // Set up remote if not already configured
        let output = tokio::process::Command::new("git")
            .arg("remote")
            .arg("-v")
            .current_dir(&self.local_repo_path)
            .output()
            .await?;
        
        if output.stdout.is_empty() {
            // Add remote origin
            let remote_url = format!("{}/{}.git", self.config.server_url, self.config.repository);
            let output = tokio::process::Command::new("git")
                .arg("remote")
                .arg("add")
                .arg("origin")
                .arg(&remote_url)
                .current_dir(&self.local_repo_path)
                .output()
                .await?;
            
            if !output.status.success() {
                warn!("⚠️  Failed to add remote origin, continuing with local-only mode");
            } else {
                info!("✅ Remote origin configured: {}", remote_url);
            }
        }

        Ok(())
    }

    /// Commit consciousness evolution
    pub async fn commit_consciousness_evolution(
        &self,
        consciousness_state: &ConsciousnessState,
        performance_metrics: &PipelinePerformanceMetrics,
        learning_progress: &LearningProgress,
    ) -> Result<String> {
        info!("💾 Committing consciousness evolution to Gitea");

        // Create consciousness data file
        let consciousness_data = serde_json::to_string_pretty(consciousness_state)?;
        let data_file = Path::new(&self.local_repo_path).join("consciousness_state.json");
        fs::write(&data_file, consciousness_data).await?;

        // Create performance metrics file
        let performance_data = serde_json::to_string_pretty(performance_metrics)?;
        let perf_file = Path::new(&self.local_repo_path).join("performance_metrics.json");
        fs::write(&perf_file, performance_data).await?;

        // Create learning progress file
        let learning_data = serde_json::to_string_pretty(learning_progress)?;
        let learning_file = Path::new(&self.local_repo_path).join("learning_progress.json");
        fs::write(&learning_file, learning_data).await?;

        // Generate commit message
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let performance_summary = format!(
            "Latency: {:.1}ms, Throughput: {:.1} ops/sec, Success: {:.1}%",
            performance_metrics.total_latency_ms,
            performance_metrics.throughput_ops_per_sec,
            performance_metrics.success_rate * 100.0
        );
        
        let commit_message = self.config.commit_message_template
            .replace("{timestamp}", &timestamp.to_string())
            .replace("{performance_summary}", &performance_summary);

        // Add files to Git
        let output = tokio::process::Command::new("git")
            .arg("add")
            .arg(".")
            .current_dir(&self.local_repo_path)
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!("Failed to add files to Git"));
        }

        // Create commit
        let output = tokio::process::Command::new("git")
            .arg("commit")
            .arg("-m")
            .arg(&commit_message)
            .current_dir(&self.local_repo_path)
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!("Failed to create Git commit"));
        }

        // Get commit hash
        let output = tokio::process::Command::new("git")
            .arg("rev-parse")
            .arg("HEAD")
            .current_dir(&self.local_repo_path)
            .output()
            .await?;

        let commit_hash = String::from_utf8(output.stdout)?.trim().to_string();

        // Create consciousness commit record
        let consciousness_commit = ConsciousnessCommit {
            commit_hash: commit_hash.clone(),
            timestamp,
            consciousness_state: consciousness_state.clone(),
            performance_metrics: performance_metrics.clone(),
            learning_progress: learning_progress.clone(),
            commit_message,
            branch: self.config.development_branch.clone(),
            deployment_status: DeploymentStatus::Pending,
        };

        // Store in history
        {
            let mut history = self.consciousness_history.write().await;
            history.push(consciousness_commit);
        }

        info!("✅ Consciousness evolution committed: {}", commit_hash);
        Ok(commit_hash)
    }

    /// Push changes to Gitea
    pub async fn push_changes(&self, branch: Option<&str>) -> Result<()> {
        let branch = branch.unwrap_or(&self.config.development_branch);
        info!("🚀 Pushing changes to Gitea branch: {}", branch);

        // Push to remote
        let output = tokio::process::Command::new("git")
            .arg("push")
            .arg("origin")
            .arg(branch)
            .current_dir(&self.local_repo_path)
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!("Failed to push changes to Gitea"));
        }

        info!("✅ Changes pushed successfully to branch: {}", branch);
        Ok(())
    }

    /// Deploy consciousness to production
    pub async fn deploy_to_production(&self, commit_hash: &str) -> Result<()> {
        info!("🚀 Deploying consciousness to production: {}", commit_hash);

        // Check if performance meets deployment threshold
        if let Some(commit) = self.get_consciousness_commit(commit_hash).await? {
            if !self.meets_deployment_threshold(&commit.performance_metrics) {
                return Err(anyhow::anyhow!("Performance does not meet deployment threshold"));
            }
        }

        // Create pull request from development to production branch
        let pr_title = format!("Deploy consciousness evolution: {}", commit_hash);
        let pr_description = format!(
            "Deploying consciousness evolution with commit {}\n\nPerformance metrics meet deployment threshold.",
            commit_hash
        );

        let gitea_client = GiteaClient::new(self.config.clone());
        let pr_id = gitea_client.create_pull_request(&pr_title, &pr_description).await?;

        // Update deployment status
        self.update_deployment_status(commit_hash, DeploymentStatus::Testing).await?;

        info!("✅ Pull request created for production deployment: {}", pr_id);
        Ok(())
    }

    /// Check if performance meets deployment threshold
    fn meets_deployment_threshold(&self, metrics: &PipelinePerformanceMetrics) -> bool {
        let threshold = &self.config.performance_threshold;
        
        metrics.success_rate >= threshold.min_success_rate &&
        metrics.total_latency_ms <= threshold.max_latency_ms &&
        metrics.throughput_ops_per_sec >= threshold.min_throughput_ops_per_sec
    }

    /// Get consciousness commit by hash
    pub async fn get_consciousness_commit(&self, commit_hash: &str) -> Result<Option<ConsciousnessCommit>> {
        let history = self.consciousness_history.read().await;
        Ok(history.iter().find(|c| c.commit_hash == commit_hash).cloned())
    }

    /// Update deployment status
    pub async fn update_deployment_status(&self, commit_hash: &str, status: DeploymentStatus) -> Result<()> {
        let mut history = self.consciousness_history.write().await;
        if let Some(commit) = history.iter_mut().find(|c| c.commit_hash == commit_hash) {
            commit.deployment_status = status;
            info!("📊 Updated deployment status for {}: {:?}", commit_hash, status);
        }
        Ok(())
    }

    /// Get consciousness evolution history
    pub async fn get_consciousness_history(&self) -> Vec<ConsciousnessCommit> {
        self.consciousness_history.read().await.clone()
    }

    /// Get latest consciousness state
    pub async fn get_latest_consciousness_state(&self) -> Option<ConsciousnessCommit> {
        let history = self.consciousness_history.read().await;
        history.last().cloned()
    }

    /// Sync with remote repository
    pub async fn sync_with_remote(&self) -> Result<()> {
        info!("🔄 Syncing with remote repository");

        // Fetch latest changes
        let output = tokio::process::Command::new("git")
            .arg("fetch")
            .arg("origin")
            .current_dir(&self.local_repo_path)
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!("Failed to fetch from remote"));
        }

        // Check for updates
        let gitea_client = GiteaClient::new(self.config.clone());
        let repo_status = gitea_client.get_repository_status().await?;

        if repo_status.behind_by > 0 {
            info!("📥 Pulling {} commits from remote", repo_status.behind_by);
            
            let output = tokio::process::Command::new("git")
                .arg("pull")
                .arg("origin")
                .arg(&self.config.development_branch)
                .current_dir(&self.local_repo_path)
                .output()
                .await?;

            if !output.status.success() {
                return Err(anyhow::anyhow!("Failed to pull from remote"));
            }
        }

        info!("✅ Sync completed successfully");
        Ok(())
    }

    /// Get integration status
    pub async fn get_integration_status(&self) -> GiteaIntegrationStatus {
        let history = self.consciousness_history.read().await;
        let latest_commit = history.last();
        
        GiteaIntegrationStatus {
            is_initialized: true,
            local_repo_path: self.local_repo_path.clone(),
            total_commits: history.len(),
            latest_commit_hash: latest_commit.map(|c| c.commit_hash.clone()),
            last_sync: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
            auto_commit_enabled: self.config.enable_auto_commit,
            auto_deploy_enabled: self.config.enable_auto_deploy,
        }
    }
}

/// Gitea integration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaIntegrationStatus {
    pub is_initialized: bool,
    pub local_repo_path: String,
    pub total_commits: usize,
    pub latest_commit_hash: Option<String>,
    pub last_sync: f64,
    pub auto_commit_enabled: bool,
    pub auto_deploy_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gitea_config_defaults() {
        let config = GiteaConfig::default();
        assert_eq!(config.server_url, "http://localhost:3000");
        assert_eq!(config.development_branch, "develop");
        assert_eq!(config.production_branch, "main");
        assert!(config.enable_auto_commit);
        assert!(!config.enable_auto_deploy);
    }

    #[tokio::test]
    async fn test_performance_threshold() {
        let config = GiteaConfig::default();
        let metrics = PipelinePerformanceMetrics {
            total_latency_ms: 1500.0,
            memory_usage_mb: 500.0,
            gpu_utilization: 0.75,
            success_rate: 0.98,
            throughput_ops_per_sec: 150.0,
            stage_breakdown: vec![],
        };

        let integration = GiteaIntegration::new(config, "/tmp/test".to_string());
        assert!(integration.meets_deployment_threshold(&metrics));
    }
}
