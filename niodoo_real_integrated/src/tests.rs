#![cfg(test)]

use anyhow::Result;
use mockall::mock;
use mockall::predicate;

use crate::erag::CollapseResult;
use crate::test_support::mock_pipeline;

trait EmbedService {
    fn embed(&self, prompt: &str) -> Result<Vec<f32>>;
}

trait CollapseService {
    fn collapse(&self, embedding: &[f32]) -> Result<CollapseResult>;
}

mock! {
    pub EmbedServiceMock {}
    impl EmbedService for EmbedServiceMock {
        fn embed(&self, prompt: &str) -> Result<Vec<f32>>;
    }
}

mock! {
    pub CollapseServiceMock {}
    impl CollapseService for CollapseServiceMock {
        fn collapse(&self, embedding: &[f32]) -> Result<CollapseResult>;
    }
}

struct MockPipelineFacade<E, M>
where
    E: EmbedService,
    M: CollapseService,
{
    embedder: E,
    erag: M,
}

impl<E, M> MockPipelineFacade<E, M>
where
    E: EmbedService,
    M: CollapseService,
{
    fn new(embedder: E, erag: M) -> Self {
        Self { embedder, erag }
    }

    async fn process_prompt(&self, prompt: &str) -> Result<CollapseResult> {
        let embedding = self.embedder.embed(prompt)?;
        self.erag.collapse(&embedding)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn mock_pipeline_embed_stage() -> Result<()> {
    let harness = mock_pipeline("embed").await?;
    assert!(harness.pipeline().dataset_stats.sample_count > 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_process_prompt_with_mock_clients() -> Result<()> {
    let prompt = "Möbius convergence smoke";
    let embedding = vec![0.42f32, 0.84f32, 0.21f32, 0.63f32];
    let expected_context = "Synthetic Möbius trace".to_string();

    let mut embedder = MockEmbedServiceMock::new();
    let capture_prompt = prompt.to_string();
    let embedding_clone = embedding.clone();
    embedder
        .expect_embed()
        .with(predicate::eq(capture_prompt))
        .times(1)
        .returning(move |_| Ok(embedding_clone.clone()));

    let mut erag = MockCollapseServiceMock::new();
    let embedding_verification = embedding.clone();
    let context_clone = expected_context.clone();
    erag.expect_collapse()
        .withf(move |received| received == embedding_verification.as_slice())
        .times(1)
        .returning(move |_| {
            Ok(CollapseResult {
                top_hits: Vec::new(),
                aggregated_context: context_clone.clone(),
                average_similarity: 0.82,
                curator_quality: Some(0.8),
                failure_type: None,
                failure_details: None,
            })
        });

    let facade = MockPipelineFacade::new(embedder, erag);
    let collapse = facade.process_prompt(prompt).await?;

    assert_eq!(collapse.aggregated_context, expected_context);
    assert!(collapse.average_similarity >= 0.8);
    assert_eq!(collapse.failure_type, None);
    Ok(())
}
