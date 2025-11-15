//! LoCoMo Benchmark Integration
//!
//! Long-Context Conversational Memory benchmark for validating:
//! - Context ingestion of full conversational histories
//! - Single-hop QA
//! - Multi-hop QA
//! - Temporal reasoning QA
//! - Adversarial QA

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoCoMoTest {
    pub conversation_id: String,
    pub conversation_history: Vec<ConversationTurn>,
    pub questions: Vec<Question>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub turn_id: usize,
    pub speaker: String, // "user" or "assistant"
    pub content: String,
    pub timestamp: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub question_id: String,
    pub category: QuestionCategory,
    pub question: String,
    pub expected_answer: Option<String>,
    pub expected_keywords: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum QuestionCategory {
    SingleHop,
    MultiHop,
    Temporal,
    Adversarial,
}

impl std::fmt::Display for QuestionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuestionCategory::SingleHop => write!(f, "SingleHop"),
            QuestionCategory::MultiHop => write!(f, "MultiHop"),
            QuestionCategory::Temporal => write!(f, "Temporal"),
            QuestionCategory::Adversarial => write!(f, "Adversarial"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoCoMoResult {
    pub question_id: String,
    pub category: QuestionCategory,
    pub response: String,
    pub f1_score: f64,
    pub exact_match: bool,
    pub keywords_matched: usize,
    pub keywords_total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoCoMoBenchmarkResults {
    pub test_id: String,
    pub timestamp: String,
    pub results: Vec<LoCoMoResult>,
    pub category_scores: HashMap<String, CategoryScores>,
    pub overall_f1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryScores {
    pub f1_score: f64,
    pub exact_match_rate: f64,
    pub keyword_match_rate: f64,
    pub count: usize,
}

pub struct LoCoMoRunner {
    // Will be populated with test cases
}

impl LoCoMoRunner {
    pub fn new() -> Self {
        Self {}
    }

    /// Load LoCoMo test cases from JSON file
    pub fn load_tests<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<LoCoMoTest>, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let tests: Vec<LoCoMoTest> = serde_json::from_str(&content)?;
        Ok(tests)
    }

    /// Run a single LoCoMo test case
    pub async fn run_test<F>(
        &self,
        test: &LoCoMoTest,
        process_fn: F,
    ) -> Result<LoCoMoBenchmarkResults, anyhow::Error>
    where
        F: Fn(
            &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
        >,
    {
        let mut results = Vec::new();
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        let mut category_f1_sum: HashMap<String, f64> = HashMap::new();
        let mut category_exact_match_sum: HashMap<String, usize> = HashMap::new();
        let mut category_keyword_match_sum: HashMap<String, usize> = HashMap::new();
        let mut category_keyword_total_sum: HashMap<String, usize> = HashMap::new();

        // Build full conversation context
        let conversation_context = self.build_conversation_context(test);

        for question in &test.questions {
            // Inject conversation history into prompt
            let full_prompt = format!(
                "{}\n\nQuestion: {}\nAnswer:",
                conversation_context, question.question
            );

            // Process through pipeline
            let response = process_fn(&full_prompt).await?;

            // Evaluate response
            let result = self.evaluate_response(question, &response);

            // Update category statistics
            let cat_key = format!("{:?}", result.category);
            *category_counts.entry(cat_key.clone()).or_insert(0) += 1;
            *category_f1_sum.entry(cat_key.clone()).or_insert(0.0) += result.f1_score;
            if result.exact_match {
                *category_exact_match_sum.entry(cat_key.clone()).or_insert(0) += 1;
            }
            *category_keyword_match_sum
                .entry(cat_key.clone())
                .or_insert(0) += result.keywords_matched;
            *category_keyword_total_sum
                .entry(cat_key.clone())
                .or_insert(0) += result.keywords_total;

            results.push(result);
        }

        // Compute category scores
        let mut category_scores = HashMap::new();
        for (cat_key, count) in category_counts {
            let f1_avg = category_f1_sum.get(&cat_key).copied().unwrap_or(0.0) / count as f64;
            let exact_match_rate =
                category_exact_match_sum.get(&cat_key).copied().unwrap_or(0) as f64 / count as f64;
            let keyword_match_rate = if let (Some(matched), Some(total)) = (
                category_keyword_match_sum.get(&cat_key),
                category_keyword_total_sum.get(&cat_key),
            ) {
                if *total > 0 {
                    *matched as f64 / *total as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };

            category_scores.insert(
                cat_key.clone(),
                CategoryScores {
                    f1_score: f1_avg,
                    exact_match_rate,
                    keyword_match_rate,
                    count,
                },
            );
        }

        // Compute overall F1
        let overall_f1 = results.iter().map(|r| r.f1_score).sum::<f64>() / results.len() as f64;

        Ok(LoCoMoBenchmarkResults {
            test_id: test.conversation_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            results,
            category_scores,
            overall_f1,
        })
    }

    fn build_conversation_context(&self, test: &LoCoMoTest) -> String {
        let mut context = String::new();
        context.push_str("Conversation History:\n");

        for turn in &test.conversation_history {
            context.push_str(&format!("{}: {}\n", turn.speaker, turn.content));
        }

        context
    }

    fn evaluate_response(&self, question: &Question, response: &str) -> LoCoMoResult {
        let mut f1_score = 0.0;
        let mut exact_match = false;
        let mut keywords_matched = 0;

        // Check exact match
        if let Some(ref expected) = question.expected_answer {
            let response_lower = response.to_lowercase();
            let expected_lower = expected.to_lowercase();

            if response_lower.contains(&expected_lower) || expected_lower.contains(&response_lower)
            {
                exact_match = true;
                f1_score = 1.0;
            }
        }

        // Check keyword matching
        let response_lower = response.to_lowercase();
        for keyword in &question.expected_keywords {
            if response_lower.contains(&keyword.to_lowercase()) {
                keywords_matched += 1;
            }
        }

        // Compute F1 if not exact match
        if !exact_match && !question.expected_keywords.is_empty() {
            let precision = keywords_matched as f64 / question.expected_keywords.len() as f64;
            // Recall approximation: assume all keywords should be present
            let recall = if keywords_matched == question.expected_keywords.len() {
                1.0
            } else {
                keywords_matched as f64 / question.expected_keywords.len() as f64
            };

            if precision + recall > 0.0 {
                f1_score = 2.0 * precision * recall / (precision + recall);
            }
        }

        LoCoMoResult {
            question_id: question.question_id.clone(),
            category: question.category,
            response: response.to_string(),
            f1_score,
            exact_match,
            keywords_matched,
            keywords_total: question.expected_keywords.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_response() {
        let runner = LoCoMoRunner::new();
        let question = Question {
            question_id: "test-1".to_string(),
            category: QuestionCategory::SingleHop,
            question: "What is the capital?".to_string(),
            expected_answer: Some("Paris".to_string()),
            expected_keywords: vec!["Paris".to_string(), "France".to_string()],
        };

        let result = runner.evaluate_response(&question, "The capital is Paris, France.");
        assert!(result.exact_match || result.f1_score > 0.0);
        assert!(result.keywords_matched >= 1);
    }
}
