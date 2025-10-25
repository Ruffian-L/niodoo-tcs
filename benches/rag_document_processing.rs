use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tracing::{info, error, warn};
use niodoo_consciousness::rag::ingestion::IngestionEngine;
use std::time::Duration;

/// Generate test document of specified size
fn generate_test_document(size_kb: usize) -> String {
    let words = vec![
        "consciousness", "memory", "Möbius", "Gaussian", "process",
        "empathy", "neurodivergent", "hallucination", "resonance",
        "IIT", "Phi", "integration", "information", "theory"
    ];

    let target_size = size_kb * 1024;
    let mut doc = String::with_capacity(target_size);
    let mut current_size = 0;

    while current_size < target_size {
        for word in &words {
            doc.push_str(word);
            doc.push(' ');
            current_size += word.len() + 1;

            if current_size >= target_size {
                break;
            }
        }
    }

    doc
}

/// Benchmark document chunking with different sizes
fn bench_chunking_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_chunking");
    group.measurement_time(Duration::from_secs(10));

    // Test different document sizes (10KB to 1MB)
    for size_kb in [10, 50, 100, 500, 1000].iter() {
        let doc = generate_test_document(*size_kb);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size_kb)),
            &doc,
            |b, doc| {
                b.iter(|| {
                    let mut engine = IngestionEngine::new(512); // 512 words per chunk
                    let chunks = engine.chunk_text(
                        black_box(doc),
                        "test_doc.md"
                    ).unwrap();
                    black_box(chunks)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different chunk sizes on same document
fn bench_chunk_size_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_size_optimization");
    group.measurement_time(Duration::from_secs(10));

    let doc = generate_test_document(100); // 100KB test doc

    // Test different chunk sizes
    for chunk_size in [128, 256, 512, 1024, 2048].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_words", chunk_size)),
            chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let mut engine = IngestionEngine::new(chunk_size);
                    let chunks = engine.chunk_text(
                        black_box(&doc),
                        "test_doc.md"
                    ).unwrap();
                    black_box(chunks)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark entity extraction overhead
fn bench_entity_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_extraction");

    let text_with_entities = "The Möbius transformation in Gaussian process memory with IIT and Phi consciousness integration using neurodivergent empathy hallucination detection.";
    let text_without_entities = "The transformation in process memory with consciousness integration using detection.";

    group.bench_function("with_entities", |b| {
        b.iter(|| {
            let engine = IngestionEngine::new(512);
            let entities = engine.extract_entities(black_box(text_with_entities));
            black_box(entities)
        });
    });

    group.bench_function("without_entities", |b| {
        b.iter(|| {
            let engine = IngestionEngine::new(512);
            let entities = engine.extract_entities(black_box(text_without_entities));
            black_box(entities)
        });
    });

    group.finish();
}

/// Benchmark markdown stripping
fn bench_markdown_stripping(c: &mut Criterion) {
    let markdown_doc = r#"
# Heading 1

This is a test document with **bold** and *italic* text.

## Heading 2

```rust
fn test() {
    tracing::info!("code block");
}
```

- List item 1
- List item 2

`inline code` and more text.
"#;

    c.bench_function("markdown_stripping", |b| {
        b.iter(|| {
            let engine = IngestionEngine::new(512);
            let stripped = engine.strip_markdown(black_box(markdown_doc));
            black_box(stripped)
        });
    });
}

/// Benchmark memory allocation patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    // Small chunks (low memory)
    group.bench_function("small_chunks_256", |b| {
        let doc = generate_test_document(100);
        b.iter(|| {
            let mut engine = IngestionEngine::new(256);
            let chunks = engine.chunk_text(black_box(&doc), "test.md").unwrap();
            black_box(chunks)
        });
    });

    // Large chunks (high memory)
    group.bench_function("large_chunks_2048", |b| {
        let doc = generate_test_document(100);
        b.iter(|| {
            let mut engine = IngestionEngine::new(2048);
            let chunks = engine.chunk_text(black_box(&doc), "test.md").unwrap();
            black_box(chunks)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_chunking_sizes,
    bench_chunk_size_variations,
    bench_entity_extraction,
    bench_markdown_stripping,
    bench_memory_patterns
);

criterion_main!(benches);
