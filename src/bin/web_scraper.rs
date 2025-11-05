//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Web Scraper for Consciousness Learning Training Data
//!
//! Scrapes emotional, psychological, and consciousness-related content
//! from websites to generate training data for the QLoRA fine-tuning pipeline.

use niodoo_core::qwen_curator::LearningEvent;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};

/// Learning event scraped from web content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapedLearningEvent {
    /// Timestamp of scraping
    pub timestamp: String,
    /// Source URL
    pub url: String,
    /// Title of the content
    pub title: String,
    /// Main content text
    pub content: String,
    /// Author if available
    pub author: Option<String>,
    /// Tags/categories for content classification
    pub tags: Vec<String>,
}

/// Web scraper configuration
#[derive(Debug, Clone)]
pub struct ScraperConfig {
    /// Target websites to scrape
    pub target_sites: Vec<String>,
    /// Content selectors for different sites
    pub content_selectors: Vec<String>,
    /// Title selectors
    pub title_selectors: Vec<String>,
    /// Author selectors (optional)
    pub author_selectors: Vec<String>,
    /// Keywords to filter relevant content
    pub keywords: Vec<String>,
    /// Maximum pages per site
    pub max_pages_per_site: usize,
    /// Delay between requests (seconds)
    pub request_delay: u64,
    /// Output file path
    pub output_path: String,
}

impl Default for ScraperConfig {
    fn default() -> Self {
        Self {
            target_sites: vec![
                "https://www.reddit.com/r/psychology/".to_string(),
                "https://www.reddit.com/r/Therapy/".to_string(),
                "https://www.reddit.com/r/consciousness/".to_string(),
                "https://www.reddit.com/r/Meditation/".to_string(),
                "https://www.reddit.com/r/philosophy/".to_string(),
                "https://www.psychologytoday.com/".to_string(),
                "https://www.psychcentral.com/".to_string(),
            ],
            content_selectors: vec![
                ".post-content".to_string(),
                ".comment-content".to_string(),
                ".article-content".to_string(),
                ".entry-content".to_string(),
                ".content".to_string(),
                "article".to_string(),
                ".post".to_string(),
            ],
            title_selectors: vec![
                "h1".to_string(),
                ".post-title".to_string(),
                ".title".to_string(),
                "title".to_string(),
            ],
            author_selectors: vec![
                ".author".to_string(),
                ".byline".to_string(),
                "[data-author]".to_string(),
            ],
            keywords: vec![
                "emotion".to_string(),
                "feeling".to_string(),
                "consciousness".to_string(),
                "mindfulness".to_string(),
                "therapy".to_string(),
                "anxiety".to_string(),
                "depression".to_string(),
                "healing".to_string(),
                "self-awareness".to_string(),
                "meditation".to_string(),
                "psychology".to_string(),
                "mental health".to_string(),
            ],
            max_pages_per_site: 50,
            request_delay: 2,
            output_path: "scraped_training_data.json".to_string(),
        }
    }
}

/// Web scraper for consciousness training data
pub struct ConsciousnessScraper {
    client: Client,
    config: ScraperConfig,
    visited_urls: HashSet<String>,
}

impl ConsciousnessScraper {
    pub fn new(config: ScraperConfig) -> Self {
        let client = Client::builder()
            .user_agent("Consciousness-Learning-Scraper/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            config,
            visited_urls: HashSet::new(),
        }
    }

    /// Start scraping process
    pub async fn scrape(
        &mut self,
    ) -> Result<Vec<ScrapedLearningEvent>, Box<dyn std::error::Error>> {
        println!("🕷️ Starting consciousness training data scraping...");
        println!("📊 Target sites: {}", self.config.target_sites.len());
        println!("🔍 Keywords: {:?}", self.config.keywords);

        let mut all_events = Vec::new();

        // Collect site URLs to avoid borrowing conflict
        let site_urls: Vec<String> = self.config.target_sites.clone();

        for site_url in site_urls {
            println!("🌐 Scraping site: {}", site_url);
            match self.scrape_site(&site_url).await {
                Ok(events) => {
                    println!("✅ Collected {} events from {}", events.len(), site_url);
                    all_events.extend(events);
                }
                Err(e) => {
                    println!("❌ Failed to scrape {}: {}", site_url, e);
                }
            }

            // Delay between sites
            sleep(Duration::from_secs(self.config.request_delay * 2)).await;
        }

        println!("🎯 Total scraped events: {}", all_events.len());
        Ok(all_events)
    }

    /// Scrape a single site
    async fn scrape_site(
        &mut self,
        base_url: &str,
    ) -> Result<Vec<ScrapedLearningEvent>, Box<dyn std::error::Error>> {
        let mut events = Vec::new();
        let mut urls_to_visit = vec![base_url.to_string()];
        let mut pages_scraped = 0;

        while pages_scraped < self.config.max_pages_per_site && !urls_to_visit.is_empty() {
            let current_url = urls_to_visit.remove(0);

            if self.visited_urls.contains(&current_url) {
                continue;
            }

            self.visited_urls.insert(current_url.clone());

            match self.scrape_page(&current_url).await {
                Ok(page_events) => {
                    events.extend(page_events);
                    pages_scraped += 1;

                    // Extract more URLs from the page (simplified - just add some variations)
                    if pages_scraped < self.config.max_pages_per_site {
                        // Add pagination or related URLs here if needed
                    }
                }
                Err(e) => {
                    println!("⚠️ Failed to scrape page {}: {}", current_url, e);
                }
            }

            // Rate limiting
            sleep(Duration::from_secs(self.config.request_delay)).await;
        }

        Ok(events)
    }

    /// Scrape a single page
    async fn scrape_page(
        &self,
        url: &str,
    ) -> Result<Vec<ScrapedLearningEvent>, Box<dyn std::error::Error>> {
        println!("📄 Scraping: {}", url);

        let response = self.client.get(url).send().await?;
        let html = response.text().await?;
        let document = Html::parse_document(&html);

        let mut events = Vec::new();

        // Try different content selectors
        for selector_str in &self.config.content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    let content = element
                        .text()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .trim()
                        .to_string();

                    if content.len() < 100 {
                        continue; // Skip too short content
                    }

                    // Check if content contains relevant keywords
                    let content_lower = content.to_lowercase();
                    let has_keywords = self
                        .config
                        .keywords
                        .iter()
                        .any(|keyword| content_lower.contains(keyword));

                    if !has_keywords {
                        continue;
                    }

                    // Extract title
                    let title = self
                        .extract_title(&document)
                        .unwrap_or_else(|| "Untitled".to_string());

                    // Extract author
                    let author = self.extract_author(&document);

                    // Create timestamp
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs()
                        .to_string();

                    // Generate tags based on keywords found
                    let tags = self
                        .config
                        .keywords
                        .iter()
                        .filter(|keyword| content_lower.contains(*keyword))
                        .cloned()
                        .collect();

                    let event = ScrapedLearningEvent {
                        timestamp,
                        url: url.to_string(),
                        title,
                        content,
                        author,
                        tags,
                    };

                    events.push(event);
                }
            }
        }

        Ok(events)
    }

    /// Extract title from page
    fn extract_title(&self, document: &Html) -> Option<String> {
        for selector_str in &self.config.title_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let title = element
                        .text()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .trim()
                        .to_string();
                    if !title.is_empty() {
                        return Some(title);
                    }
                }
            }
        }
        None
    }

    /// Extract author from page
    fn extract_author(&self, document: &Html) -> Option<String> {
        for selector_str in &self.config.author_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let author = element
                        .text()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .trim()
                        .to_string();
                    if !author.is_empty() {
                        return Some(author);
                    }
                }
            }
        }
        None
    }

    /// Save scraped data to file
    pub fn save_to_file(
        &self,
        events: &[ScrapedLearningEvent],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(events)?;
        let mut file = File::create(&self.config.output_path)?;
        file.write_all(json.as_bytes())?;
        println!(
            "💾 Saved {} events to {}",
            events.len(),
            self.config.output_path
        );
        Ok(())
    }

    /// Convert scraped events to LearningEvent format for training
    pub fn convert_to_learning_events(
        &self,
        scraped_events: &[ScrapedLearningEvent],
    ) -> Vec<LearningEvent> {
        scraped_events
            .iter()
            .map(|scraped| {
                LearningEvent {
                    timestamp: scraped.timestamp.clone(),
                    input: format!(
                        "{}: {}",
                        scraped.title,
                        scraped.content.chars().take(200).collect::<String>()
                    ),
                    response: scraped.content.clone(),
                    emotional_state: None, // Could be inferred from content
                    coherence: None,       // Could be computed
                    memory_activations: None,
                    topology_metrics: None,
                }
            })
            .collect()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Consciousness Training Data Web Scraper");
    println!("==========================================");

    let config = ScraperConfig::default();
    let mut scraper = ConsciousnessScraper::new(config);

    // Scrape data
    let scraped_events = scraper.scrape().await?;

    // Save raw scraped data
    scraper.save_to_file(&scraped_events)?;

    // Convert to training format
    let learning_events = scraper.convert_to_learning_events(&scraped_events);

    // Save in training format
    let training_json = serde_json::to_string_pretty(&learning_events)?;
    let training_path = "scraped_learning_events.json";
    let mut training_file = File::create(training_path)?;
    training_file.write_all(training_json.as_bytes())?;

    println!("✅ Scraping complete!");
    println!("📊 Total events collected: {}", scraped_events.len());
    println!("💾 Raw data saved to: scraped_training_data.json");
    println!("🎯 Training data saved to: {}", training_path);

    Ok(())
}
