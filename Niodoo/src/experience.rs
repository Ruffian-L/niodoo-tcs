use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct ExperiencePayload<'a> {
    pub id: &'a str,
    pub prompt: &'a str,
    pub response: &'a str,
    pub context: &'a str,
    pub task_type: &'a str,
    pub quality_score: Option<i32>,
    pub success_score: f32,
    pub rouge_l: f32,
    pub feedback: &'a str,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<&'a Value>,
}

#[derive(Debug)]
pub struct Experience {
    pub id: String,
    pub prompt: String,
    pub response: String,
    pub context: String,
    pub task_type: String,
    pub quality_score: Option<i32>,
    pub success_score: f32,
    pub rouge_l: f32,
    pub feedback: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<Value>,
}

impl Experience {
    pub fn new(
        prompt: String,
        response: String,
        context: String,
        task_type: String,
        quality_score: Option<i32>,
        rouge_l: f32,
        feedback: String,
        success_score: f32,
        metadata: Option<Value>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            prompt,
            response,
            context,
            task_type,
            quality_score,
            success_score,
            rouge_l,
            feedback,
            timestamp: Utc::now(),
            metadata,
        }
    }

    pub fn as_payload(&self) -> ExperiencePayload<'_> {
        ExperiencePayload {
            id: &self.id,
            prompt: &self.prompt,
            response: &self.response,
            context: &self.context,
            task_type: &self.task_type,
            quality_score: self.quality_score,
            success_score: self.success_score,
            rouge_l: self.rouge_l,
            feedback: &self.feedback,
            timestamp: self.timestamp,
            metadata: self.metadata.as_ref(),
        }
    }
}
