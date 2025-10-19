//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use lru::LruCache;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::sync::Arc;
use tokenizers::Tokenizer;

// Use a trait for future flexibility
pub trait SentenceEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

pub struct SemanticTransformer {
    model: BertModel,
    tokenizer: Tokenizer,
    cache: Arc<Mutex<LruCache<String, Vec<f32>>>>,
}

impl SemanticTransformer {
    pub fn new(model_path: &str) -> Result<Self> {
        // Dynamic model loading with error handling
        let device = Device::Cpu;
        let config = Config::default(); // Use default config for now
        let vb = VarBuilder::from_pth(model_path, DType::F32, &device)?;
        let model = BertModel::load(vb, &config)?;

        let tokenizer = Tokenizer::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            cache: Arc::new(Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(1024).unwrap(),
            ))),
        })
    }

    fn preprocess(&self, text: &str) -> Result<(Tensor, Tensor)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        let input_ids_tensor = Tensor::new(input_ids.as_slice(), &Device::Cpu)?;
        let attention_mask_tensor = Tensor::new(attention_mask.as_slice(), &Device::Cpu)?;
        Ok((input_ids_tensor, attention_mask_tensor))
    }

    fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let (input_ids, attention_mask) = self.preprocess(text)?;

        // Compute embeddings
        let outputs = self.model.forward(&input_ids, &attention_mask, None)?;
        let embeddings = &outputs;

        // Convert to f32 vec
        let embedding_vec: Vec<f32> = embeddings.to_vec1()?.into_iter().map(|x: f32| x).collect();

        Ok(embedding_vec)
    }
}

impl SentenceEmbedder for SemanticTransformer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        {
            let mut cache = self.cache.lock();
            if let Some(cached_embedding) = cache.get(text) {
                return Ok(cached_embedding.clone());
            }
        }

        // Compute and cache embedding
        let embedding = self.compute_embedding(text)?;

        {
            let mut cache = self.cache.lock();
            cache.put(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }
}

// ==================================================================================
// DEPENDENCY INJECTION PATTERN (RECOMMENDED)
// ==================================================================================

/// Embedding service with dependency injection pattern
///
/// This is the recommended way to use embeddings in your application.
/// It provides better testability, clearer ownership, and no hidden initialization requirements.
///
/// # Example
/// ```
/// use niodoo_feeling::embeddings::EmbeddingService;
///
/// let service = EmbeddingService::new("/path/to/model")?;
/// let embedding = service.embed("test text")?;
/// ```
pub struct EmbeddingService {
    embedder: Arc<SemanticTransformer>,
}

impl EmbeddingService {
    /// Create a new embedding service with the specified model
    pub fn new(model_path: &str) -> Result<Self> {
        let embedder = Arc::new(SemanticTransformer::new(model_path)?);
        Ok(Self { embedder })
    }

    /// Embed a single text string
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder.embed(text)
    }

    /// Embed multiple text strings in batch
    pub fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|text| self.embedder.embed(text))
            .collect::<Result<Vec<Vec<f32>>>>()
    }

    /// Get a cloned Arc reference to the embedder for shared ownership
    pub fn get_embedder(&self) -> Arc<SemanticTransformer> {
        Arc::clone(&self.embedder)
    }
}

// ==================================================================================
// LEGACY GLOBAL API (DEPRECATED - use EmbeddingService instead)
// ==================================================================================

/// Global embedder instance (deprecated)
///
/// # Deprecation Notice
/// This global state pattern is deprecated. Use `EmbeddingService` instead for better
/// testability and clearer ownership semantics.
///
/// # Thread Safety
/// This is thread-safe using `Lazy` + `Arc<Mutex<_>>` pattern, but requires explicit
/// initialization which can lead to runtime errors if forgotten.
static GLOBAL_EMBEDDER: Lazy<Arc<Mutex<Option<SemanticTransformer>>>> =
    Lazy::new(|| Arc::new(Mutex::new(None)));

/// Initialize global embedder with model path (deprecated)
///
/// # Deprecation Notice
/// Use `EmbeddingService::new()` instead. This function will be removed in a future version.
///
/// # Thread Safety
/// This function is thread-safe and can be called from multiple threads.
/// Subsequent calls will replace the previous embedder instance.
///
/// # Example (deprecated pattern)
/// ```ignore
/// use niodoo_feeling::embeddings::init_global_embedder;
///
/// init_global_embedder("/path/to/model")?;
/// ```
#[deprecated(since = "0.1.0", note = "Use EmbeddingService::new() instead")]
pub fn init_global_embedder(model_path: &str) -> Result<()> {
    let embedder = SemanticTransformer::new(model_path)?;
    let mut global_embedder = GLOBAL_EMBEDDER.lock();
    *global_embedder = Some(embedder);
    tracing::warn!("init_global_embedder is deprecated. Use EmbeddingService::new() instead.");
    Ok(())
}

/// Helper function for global embeddings (deprecated)
///
/// # Deprecation Notice
/// Use `EmbeddingService::embed()` instead. This function will be removed in a future version.
///
/// # Errors
/// Returns error if global embedder not initialized via `init_global_embedder()`.
///
/// # Example (deprecated pattern)
/// ```ignore
/// use niodoo_feeling::embeddings::{init_global_embedder, global_embed};
///
/// init_global_embedder("/path/to/model")?;
/// let embedding = global_embed("test text")?;
/// ```
#[deprecated(since = "0.1.0", note = "Use EmbeddingService::embed() instead")]
pub fn global_embed(text: &str) -> Result<Vec<f32>> {
    let global_embedder = GLOBAL_EMBEDDER.lock();
    match &*global_embedder {
        Some(embedder) => embedder.embed(text),
        None => anyhow::bail!("Global embedder not initialized. Call init_global_embedder first."),
    }
}

// Cosine similarity for semantic comparison
pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude1: f32 = vec1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let magnitude2: f32 = vec2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1 * magnitude2)
    }
}

/// Batch embedding for performance (deprecated)
///
/// # Deprecation Notice
/// Use `EmbeddingService::batch_embed()` instead. This function will be removed in a future version.
///
/// # Errors
/// Returns error if global embedder not initialized via `init_global_embedder()`.
///
/// # Example (deprecated pattern)
/// ```ignore
/// use niodoo_feeling::embeddings::{init_global_embedder, batch_embed};
///
/// init_global_embedder("/path/to/model")?;
/// let texts = vec!["text1".to_string(), "text2".to_string()];
/// let embeddings = batch_embed(&texts)?;
/// ```
#[deprecated(since = "0.1.0", note = "Use EmbeddingService::batch_embed() instead")]
pub fn batch_embed(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let global_embedder = GLOBAL_EMBEDDER.lock();
    match &*global_embedder {
        Some(embedder) => texts
            .iter()
            .map(|text| embedder.embed(text))
            .collect::<Result<Vec<Vec<f32>>>>(),
        None => anyhow::bail!("Global embedder not initialized. Call init_global_embedder first."),
    }
}
