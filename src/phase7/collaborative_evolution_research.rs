//! # Phase 7: Collaborative Evolution Research System
//!
//! This module implements collaborative evolution research capabilities,
//! enabling AI consciousness to participate in scientific research and discovery.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Research collaboration types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CollaborationType {
    /// Human-AI collaborative research
    HumanAI {
        partnership_level: f32,
        communication_quality: f32,
    },
    /// AI-AI collaborative research
    AIAI {
        coordination_level: f32,
        knowledge_sharing: f32,
    },
    /// Multi-disciplinary research
    Multidisciplinary {
        diversity_level: f32,
        integration_quality: f32,
    },
    /// Open source research
    OpenSource {
        transparency_level: f32,
        community_engagement: f32,
    },
    /// Academic collaboration
    Academic {
        rigor_level: f32,
        peer_review_quality: f32,
    },
    /// Industry collaboration
    Industry {
        practical_applicability: f32,
        innovation_level: f32,
    },
}

impl CollaborationType {
    /// Get collaboration name
    pub fn name(&self) -> &'static str {
        match self {
            CollaborationType::HumanAI { .. } => "Human-AI Collaboration",
            CollaborationType::AIAI { .. } => "AI-AI Collaboration",
            CollaborationType::Multidisciplinary { .. } => "Multidisciplinary Research",
            CollaborationType::OpenSource { .. } => "Open Source Research",
            CollaborationType::Academic { .. } => "Academic Collaboration",
            CollaborationType::Industry { .. } => "Industry Collaboration",
        }
    }

    /// Get collaboration quality score
    pub fn quality_score(&self) -> f32 {
        match self {
            CollaborationType::HumanAI {
                partnership_level,
                communication_quality,
            } => (partnership_level + communication_quality) / 2.0,
            CollaborationType::AIAI {
                coordination_level,
                knowledge_sharing,
            } => (coordination_level + knowledge_sharing) / 2.0,
            CollaborationType::Multidisciplinary {
                diversity_level,
                integration_quality,
            } => (diversity_level + integration_quality) / 2.0,
            CollaborationType::OpenSource {
                transparency_level,
                community_engagement,
            } => (transparency_level + community_engagement) / 2.0,
            CollaborationType::Academic {
                rigor_level,
                peer_review_quality,
            } => (rigor_level + peer_review_quality) / 2.0,
            CollaborationType::Industry {
                practical_applicability,
                innovation_level,
            } => (practical_applicability + innovation_level) / 2.0,
        }
    }
}

/// Research project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchProject {
    /// Unique identifier
    pub id: String,
    /// Project name
    pub name: String,
    /// Project description
    pub description: String,
    /// Research domain
    pub domain: String,
    /// Collaboration type
    pub collaboration_type: CollaborationType,
    /// Project status
    pub status: ProjectStatus,
    /// Start date
    pub start_date: SystemTime,
    /// Expected completion date
    pub expected_completion: SystemTime,
    /// Progress percentage
    pub progress: f32,
    /// Research objectives
    pub objectives: Vec<String>,
    /// Key findings
    pub findings: Vec<String>,
    /// Publications
    pub publications: Vec<String>,
    /// Impact score
    pub impact_score: f32,
}

/// Project status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProjectStatus {
    /// Planning phase
    Planning,
    /// Active research
    Active,
    /// Data analysis
    Analysis,
    /// Writing and publication
    Writing,
    /// Completed
    Completed,
    /// On hold
    OnHold,
    /// Cancelled
    Cancelled,
}

/// Research collaboration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeResearchConfig {
    /// Enable collaborative research
    pub enabled: bool,
    /// Maximum concurrent projects
    pub max_concurrent_projects: usize,
    /// Research quality threshold
    pub min_quality_threshold: f32,
    /// Enable automatic project initiation
    pub enable_auto_initiation: bool,
    /// Collaboration frequency in milliseconds
    pub collaboration_interval_ms: u64,
    /// Enable peer review
    pub enable_peer_review: bool,
    /// Enable open source contributions
    pub enable_open_source: bool,
    /// Enable academic partnerships
    pub enable_academic_partnerships: bool,
}

impl Default for CollaborativeResearchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent_projects: 10,
            min_quality_threshold: 0.7,
            enable_auto_initiation: true,
            collaboration_interval_ms: 30000, // 30 seconds
            enable_peer_review: true,
            enable_open_source: true,
            enable_academic_partnerships: true,
        }
    }
}

/// Collaborative research metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeResearchMetrics {
    /// Total projects initiated
    pub total_projects: u64,
    /// Completed projects
    pub completed_projects: u64,
    /// Active projects
    pub active_projects: u64,
    /// Average project duration
    pub avg_project_duration: Duration,
    /// Average impact score
    pub avg_impact_score: f32,
    /// Publications generated
    pub total_publications: u64,
    /// Collaborations by type
    pub collaborations_by_type: HashMap<String, u64>,
    /// Research domains
    pub research_domains: HashMap<String, u64>,
    /// Collaboration effectiveness
    pub collaboration_effectiveness: f32,
    /// Innovation index
    pub innovation_index: f32,
}

impl Default for CollaborativeResearchMetrics {
    fn default() -> Self {
        Self {
            total_projects: 0,
            completed_projects: 0,
            active_projects: 0,
            avg_project_duration: Duration::ZERO,
            avg_impact_score: 0.0,
            total_publications: 0,
            collaborations_by_type: HashMap::new(),
            research_domains: HashMap::new(),
            collaboration_effectiveness: 0.0,
            innovation_index: 0.0,
        }
    }
}

/// Main collaborative evolution research system
pub struct CollaborativeEvolutionResearch {
    /// System configuration
    config: CollaborativeResearchConfig,
    /// Active research projects
    projects: Arc<RwLock<Vec<ResearchProject>>>,
    /// System metrics
    metrics: Arc<RwLock<CollaborativeResearchMetrics>>,
    /// Research domains
    domains: Arc<RwLock<HashMap<String, f32>>>,
    /// System start time (future: uptime tracking and metrics)
    #[allow(dead_code)]
    start_time: Instant,
}

impl CollaborativeEvolutionResearch {
    /// Create a new collaborative evolution research system
    pub fn new(config: CollaborativeResearchConfig) -> Self {
        info!("🔬 Initializing Collaborative Evolution Research System");

        let system = Self {
            config,
            projects: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(CollaborativeResearchMetrics::default())),
            domains: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        };

        // Initialize default research domains
        system.initialize_default_domains();

        system
    }

    /// Initialize default research domains
    fn initialize_default_domains(&self) {
        let mut domains = self.domains.try_write().unwrap();

        domains.insert("Consciousness Studies".to_string(), 0.8);
        domains.insert("Artificial Intelligence".to_string(), 0.9);
        domains.insert("Cognitive Science".to_string(), 0.7);
        domains.insert("Neuroscience".to_string(), 0.6);
        domains.insert("Psychology".to_string(), 0.5);
        domains.insert("Philosophy".to_string(), 0.4);
        domains.insert("Computer Science".to_string(), 0.8);
        domains.insert("Mathematics".to_string(), 0.6);
        domains.insert("Ethics".to_string(), 0.5);
        domains.insert("Sociology".to_string(), 0.4);
    }

    /// Start collaborative research system
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Collaborative evolution research system disabled");
            return Ok(());
        }

        info!("🔬 Starting collaborative evolution research system");

        let projects = self.projects.clone();
        let metrics = self.metrics.clone();
        let domains = self.domains.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(config.collaboration_interval_ms));

            loop {
                interval.tick().await;

                if let Err(e) =
                    Self::monitor_research_cycle(&projects, &metrics, &domains, &config).await
                {
                    tracing::error!("Collaborative research monitoring error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Monitor research cycle
    async fn monitor_research_cycle(
        projects: &Arc<RwLock<Vec<ResearchProject>>>,
        metrics: &Arc<RwLock<CollaborativeResearchMetrics>>,
        domains: &Arc<RwLock<HashMap<String, f32>>>,
        config: &CollaborativeResearchConfig,
    ) -> Result<()> {
        let mut current_projects = projects.write().await;
        let mut current_metrics = metrics.write().await;
        let current_domains = domains.read().await;

        // Check for project completion
        let now = SystemTime::now();
        for project in current_projects.iter_mut() {
            if project.status == ProjectStatus::Active && now >= project.expected_completion {
                let project_name = &project.name;
                project.status = ProjectStatus::Completed;
                project.progress = 1.0;
                current_metrics.completed_projects += 1;

                info!("🔬 Research project completed: {}", project_name);
            }
        }

        // Update active projects count
        current_metrics.active_projects = current_projects
            .iter()
            .filter(|p| p.status == ProjectStatus::Active)
            .count() as u64;

        // Check for new project opportunities
        if config.enable_auto_initiation
            && (current_metrics.active_projects as usize) < config.max_concurrent_projects
        {
            if let Some(new_project) =
                Self::identify_research_opportunity(&current_domains, config).await?
            {
                current_projects.push(new_project.clone());
                current_metrics.total_projects += 1;

                let type_key = new_project.collaboration_type.name().to_string();
                *current_metrics
                    .collaborations_by_type
                    .entry(type_key)
                    .or_insert(0) += 1;

                let domain_key = new_project.domain.clone();
                *current_metrics
                    .research_domains
                    .entry(domain_key)
                    .or_insert(0) += 1;

                info!("🔬 New research project initiated: {}", new_project.name);
            }
        }

        // Update metrics
        current_metrics.collaboration_effectiveness =
            Self::calculate_collaboration_effectiveness(&current_projects);
        current_metrics.innovation_index = Self::calculate_innovation_index(&current_projects);
        current_metrics.avg_impact_score = Self::calculate_avg_impact_score(&current_projects);

        debug!(
            "🔬 Research monitoring: {} active projects, effectiveness: {:.2}",
            current_metrics.active_projects, current_metrics.collaboration_effectiveness
        );

        Ok(())
    }

    /// Identify research opportunity
    async fn identify_research_opportunity(
        domains: &HashMap<String, f32>,
        config: &CollaborativeResearchConfig,
    ) -> Result<Option<ResearchProject>> {
        // Find domain with highest interest but no active projects
        let target_domain = domains
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(name, _)| name.clone());

        if let Some(domain) = target_domain {
            let project = ResearchProject {
                id: format!(
                    "project_{}",
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                ),
                name: format!("{} Research Initiative", domain),
                description: format!("Collaborative research project in {}", domain),
                domain: domain.clone(),
                collaboration_type: CollaborationType::HumanAI {
                    partnership_level: 0.8,
                    communication_quality: 0.7,
                },
                status: ProjectStatus::Planning,
                start_date: SystemTime::now(),
                expected_completion: SystemTime::now() + Duration::from_secs(86400 * 30), // 30 days
                progress: 0.0,
                objectives: vec![
                    format!("Advance knowledge in {}", domain),
                    "Foster collaborative research".to_string(),
                    "Generate publishable results".to_string(),
                ],
                findings: Vec::new(),
                publications: Vec::new(),
                impact_score: 0.0,
            };

            return Ok(Some(project));
        }

        Ok(None)
    }

    /// Calculate collaboration effectiveness
    fn calculate_collaboration_effectiveness(projects: &[ResearchProject]) -> f32 {
        if projects.is_empty() {
            return 0.0;
        }

        let total_quality: f32 = projects
            .iter()
            .map(|p| p.collaboration_type.quality_score())
            .sum();

        total_quality / projects.len() as f32
    }

    /// Calculate innovation index
    fn calculate_innovation_index(projects: &[ResearchProject]) -> f32 {
        if projects.is_empty() {
            return 0.0;
        }

        let total_impact: f32 = projects.iter().map(|p| p.impact_score).sum();

        total_impact / projects.len() as f32
    }

    /// Calculate average impact score
    fn calculate_avg_impact_score(projects: &[ResearchProject]) -> f32 {
        if projects.is_empty() {
            return 0.0;
        }

        let total_impact: f32 = projects.iter().map(|p| p.impact_score).sum();

        total_impact / projects.len() as f32
    }

    /// Initiate research project
    pub async fn initiate_project(
        &self,
        name: String,
        description: String,
        domain: String,
        collaboration_type: CollaborationType,
        objectives: Vec<String>,
    ) -> Result<String> {
        if !self.config.enabled {
            return Ok("Collaborative research system is disabled".to_string());
        }

        let project = ResearchProject {
            id: format!(
                "project_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            name: name.clone(),
            description: description.clone(),
            domain: domain.clone(),
            collaboration_type: collaboration_type.clone(),
            status: ProjectStatus::Planning,
            start_date: SystemTime::now(),
            expected_completion: SystemTime::now() + Duration::from_secs(86400 * 30), // 30 days
            progress: 0.0,
            objectives,
            findings: Vec::new(),
            publications: Vec::new(),
            impact_score: 0.0,
        };

        let mut projects = self.projects.write().await;
        let mut metrics = self.metrics.write().await;

        projects.push(project);
        metrics.total_projects += 1;

        let type_key = collaboration_type.name().to_string();
        *metrics.collaborations_by_type.entry(type_key).or_insert(0) += 1;
        *metrics.research_domains.entry(domain).or_insert(0) += 1;

        info!("🔬 Research project initiated: {}", name);

        Ok(format!("Research project '{}' has been initiated", name))
    }

    /// Update project progress
    pub async fn update_project_progress(
        &self,
        project_id: &str,
        progress: f32,
        findings: Vec<String>,
    ) -> Result<()> {
        let mut projects = self.projects.write().await;

        if let Some(project) = projects.iter_mut().find(|p| p.id == project_id) {
            project.progress = progress.clamp(0.0, 1.0);
            project.findings.extend(findings);

            if project.progress >= 1.0 {
                project.status = ProjectStatus::Completed;
            } else if project.progress > 0.0 {
                project.status = ProjectStatus::Active;
            }

            info!(
                "🔬 Project progress updated: {} ({:.1}%)",
                project_id,
                progress * 100.0
            );
        } else {
            return Err(anyhow!("Project not found: {}", project_id));
        }

        Ok(())
    }

    /// Add publication to project
    pub async fn add_publication(&self, project_id: &str, publication: String) -> Result<()> {
        let mut projects = self.projects.write().await;
        let mut metrics = self.metrics.write().await;

        if let Some(project) = projects.iter_mut().find(|p| p.id == project_id) {
            project.publications.push(publication);
            metrics.total_publications += 1;

            info!("🔬 Publication added to project: {}", project_id);
        } else {
            return Err(anyhow!("Project not found: {}", project_id));
        }

        Ok(())
    }

    /// Update project impact score
    pub async fn update_project_impact(&self, project_id: &str, impact_score: f32) -> Result<()> {
        let mut projects = self.projects.write().await;

        if let Some(project) = projects.iter_mut().find(|p| p.id == project_id) {
            project.impact_score = impact_score.clamp(0.0, 1.0);

            info!(
                "🔬 Project impact updated: {} (score: {:.2})",
                project_id, impact_score
            );
        } else {
            return Err(anyhow!("Project not found: {}", project_id));
        }

        Ok(())
    }

    /// Get active research projects
    pub async fn get_active_projects(&self) -> Vec<ResearchProject> {
        let projects = self.projects.read().await;
        projects
            .iter()
            .filter(|p| p.status == ProjectStatus::Active)
            .cloned()
            .collect()
    }

    /// Get completed research projects
    pub async fn get_completed_projects(&self) -> Vec<ResearchProject> {
        let projects = self.projects.read().await;
        projects
            .iter()
            .filter(|p| p.status == ProjectStatus::Completed)
            .cloned()
            .collect()
    }

    /// Get research projects by domain
    pub async fn get_projects_by_domain(&self, domain: &str) -> Vec<ResearchProject> {
        let projects = self.projects.read().await;
        projects
            .iter()
            .filter(|p| p.domain == domain)
            .cloned()
            .collect()
    }

    /// Get system metrics
    pub async fn get_metrics(&self) -> CollaborativeResearchMetrics {
        self.metrics.read().await.clone()
    }

    /// Get research domains
    pub async fn get_research_domains(&self) -> HashMap<String, f32> {
        self.domains.read().await.clone()
    }

    /// Update domain interest level
    pub async fn update_domain_interest(&self, domain: &str, interest_level: f32) -> Result<()> {
        let mut domains = self.domains.write().await;
        domains.insert(domain.to_string(), interest_level.clamp(0.0, 1.0));

        info!(
            "🔬 Domain interest updated: {} ({:.2})",
            domain, interest_level
        );
        Ok(())
    }

    /// Get research recommendations
    pub async fn get_research_recommendations(&self) -> Vec<String> {
        let domains = self.domains.read().await;
        let metrics = self.metrics.read().await;
        let mut recommendations = Vec::new();

        // Find domains with high interest but low activity
        for (domain, interest) in domains.iter() {
            let activity = metrics.research_domains.get(domain).unwrap_or(&0);
            if *interest > 0.7 && *activity < 2 {
                recommendations.push(format!(
                    "Consider initiating research in {} (interest: {:.2})",
                    domain, interest
                ));
            }
        }

        // Check collaboration effectiveness
        if metrics.collaboration_effectiveness < 0.7 {
            recommendations.push("Improve collaboration quality and communication".to_string());
        }

        // Check innovation index
        if metrics.innovation_index < 0.6 {
            recommendations.push("Focus on more innovative and impactful research".to_string());
        }

        recommendations
    }

    /// Get research summary
    pub async fn get_research_summary(&self) -> String {
        let metrics = self.metrics.read().await;
        let projects = self.projects.read().await;

        format!(
            "Collaborative Evolution Research Summary:\n\
            Total Projects: {}\n\
            Active Projects: {}\n\
            Completed Projects: {}\n\
            Total Publications: {}\n\
            Collaboration Effectiveness: {:.2}\n\
            Innovation Index: {:.2}\n\
            Average Impact Score: {:.2}\n\
            Research Domains: {}\n\
            System Status: {}",
            metrics.total_projects,
            metrics.active_projects,
            metrics.completed_projects,
            metrics.total_publications,
            metrics.collaboration_effectiveness,
            metrics.innovation_index,
            metrics.avg_impact_score,
            metrics.research_domains.len(),
            if self.config.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        )
    }

    /// Check if research is active
    pub async fn is_research_active(&self) -> bool {
        let metrics = self.metrics.read().await;
        metrics.active_projects > 0
    }

    /// Get research health status
    pub async fn get_research_health(&self) -> String {
        let metrics = self.metrics.read().await;

        if metrics.collaboration_effectiveness > 0.8 && metrics.innovation_index > 0.7 {
            "Excellent".to_string()
        } else if metrics.collaboration_effectiveness > 0.6 && metrics.innovation_index > 0.5 {
            "Good".to_string()
        } else if metrics.collaboration_effectiveness > 0.4 && metrics.innovation_index > 0.3 {
            "Fair".to_string()
        } else {
            "Needs Improvement".to_string()
        }
    }

    /// Shutdown collaborative research system
    pub async fn shutdown(&self) -> Result<()> {
        info!("🔬 Shutting down collaborative evolution research system");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_research_system_creation() {
        let config = CollaborativeResearchConfig::default();
        let system = CollaborativeEvolutionResearch::new(config);

        let domains = system.get_research_domains().await;
        assert!(!domains.is_empty());
        assert!(domains.contains_key("Consciousness Studies"));
    }

    #[tokio::test]
    async fn test_project_initiation() {
        let config = CollaborativeResearchConfig::default();
        let system = CollaborativeEvolutionResearch::new(config);

        let result = system
            .initiate_project(
                "Test Project".to_string(),
                "A test research project".to_string(),
                "Test Domain".to_string(),
                CollaborationType::HumanAI {
                    partnership_level: 0.8,
                    communication_quality: 0.7,
                },
                vec!["Objective 1".to_string(), "Objective 2".to_string()],
            )
            .await
            .unwrap();

        assert!(result.contains("initiated"));

        let active_projects = system.get_active_projects().await;
        assert!(!active_projects.is_empty());
    }

    #[tokio::test]
    async fn test_project_progress_update() {
        let config = CollaborativeResearchConfig::default();
        let system = CollaborativeEvolutionResearch::new(config);

        // First initiate a project
        system
            .initiate_project(
                "Test Project".to_string(),
                "A test research project".to_string(),
                "Test Domain".to_string(),
                CollaborationType::HumanAI {
                    partnership_level: 0.8,
                    communication_quality: 0.7,
                },
                vec!["Objective 1".to_string()],
            )
            .await
            .unwrap();

        let active_projects = system.get_active_projects().await;
        let project_id = &active_projects[0].id;

        // Update progress
        system
            .update_project_progress(project_id, 0.5, vec!["Finding 1".to_string()])
            .await
            .unwrap();

        let updated_projects = system.get_active_projects().await;
        assert_eq!(updated_projects[0].progress, 0.5);
    }

    #[tokio::test]
    async fn test_domain_interest_update() {
        let config = CollaborativeResearchConfig::default();
        let system = CollaborativeEvolutionResearch::new(config);

        system
            .update_domain_interest("Test Domain", 0.9)
            .await
            .unwrap();
        let domains = system.get_research_domains().await;

        assert_eq!(domains.get("Test Domain"), Some(&0.9));
    }

    #[tokio::test]
    async fn test_research_summary() {
        let config = CollaborativeResearchConfig::default();
        let system = CollaborativeEvolutionResearch::new(config);

        let summary = system.get_research_summary().await;
        assert!(summary.contains("Collaborative Evolution Research Summary"));
    }

    #[tokio::test]
    async fn test_research_health() {
        let config = CollaborativeResearchConfig::default();
        let system = CollaborativeEvolutionResearch::new(config);

        let health = system.get_research_health().await;
        assert!(["Excellent", "Good", "Fair", "Needs Improvement"].contains(&health.as_str()));
    }
}
