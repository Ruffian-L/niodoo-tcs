#![cfg(feature = "legacy_tests")]

//! Phase 0 Training Validation Tests
//!
//! Tests to validate that training actually updates weights and that the training bug fix works correctly.

use anyhow::Result;
use niodoo_real_integrated::lora_trainer::{LoRAConfig, LoRATrainer};

/// Create a test batch with random data
fn create_test_batch(
    size: usize,
    input_dim: usize,
    output_dim: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..size)
        .map(|_| {
            let input: Vec<f32> = (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let target: Vec<f32> = (0..output_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            (input, target)
        })
        .collect()
}

#[test]
fn test_weights_actually_update() -> Result<()> {
    let config = LoRAConfig {
        input_dim: 128,
        output_dim: 128,
        rank: 8,
        alpha: 16.0,
        use_fp16: false,
    };

    let mut trainer = LoRATrainer::with_config(config)?;
    let batch = create_test_batch(4, 128, 128);

    // Capture initial weights
    let initial_weight_a = trainer.adapter().lora_a().to_vec2::<f32>()?;
    let initial_weight_b = trainer.adapter().lora_b().to_vec2::<f32>()?;

    // Run ONE training step
    let loss = trainer.train_batch(&batch, 0.001)?;

    // Capture final weights
    let final_weight_a = trainer.adapter().lora_a().to_vec2::<f32>()?;
    let final_weight_b = trainer.adapter().lora_b().to_vec2::<f32>()?;

    // Compute weight differences
    let weight_diff_a: f64 = initial_weight_a
        .iter()
        .zip(final_weight_a.iter())
        .map(|(init, fin)| ((init - fin) as f64).abs())
        .sum();
    let weight_diff_b: f64 = initial_weight_b
        .iter()
        .zip(final_weight_b.iter())
        .map(|(init, fin)| ((init - fin) as f64).abs())
        .sum();

    let total_diff = weight_diff_a + weight_diff_b;

    println!(
        "✅ Weight update test: diff = {:.9}, loss = {:.6}",
        total_diff, loss
    );

    assert!(
        total_diff > 1e-6,
        "TRAINING BUG: Weights did not update! diff = {:.9}",
        total_diff
    );

    Ok(())
}

#[test]
fn test_loss_decreases() -> Result<()> {
    let config = LoRAConfig {
        input_dim: 128,
        output_dim: 128,
        rank: 8,
        alpha: 16.0,
        use_fp16: false,
    };

    let mut trainer = LoRATrainer::with_config(config)?;
    let batch = create_test_batch(4, 128, 128);

    // Train on the same batch multiple times - loss should decrease
    let initial_loss = trainer.train_batch(&batch, 0.001)?;

    let mut previous_loss = initial_loss;
    for step in 0..100 {
        let current_loss = trainer.train_batch(&batch, 0.001)?;

        if step % 20 == 0 {
            println!("Step {}: loss = {:.6}", step, current_loss);
        }

        previous_loss = current_loss;
    }

    let final_loss = previous_loss;
    let reduction = (initial_loss - final_loss) / initial_loss;

    println!(
        "✅ Loss decrease test: initial={:.6}, final={:.6}, reduction={:.2}%",
        initial_loss,
        final_loss,
        reduction * 100.0
    );

    assert!(
        reduction > 0.5,
        "Loss did not decrease sufficiently: initial={:.6}, final={:.6}, reduction={:.2}%",
        initial_loss,
        final_loss,
        reduction * 100.0
    );

    Ok(())
}

#[test]
fn test_gradients_exist() -> Result<()> {
    let config = LoRAConfig {
        input_dim: 128,
        output_dim: 128,
        rank: 8,
        alpha: 16.0,
        use_fp16: false,
    };

    let mut trainer = LoRATrainer::with_config(config)?;
    let batch = create_test_batch(4, 128, 128);

    // Train a batch and verify gradients were computed
    let loss = trainer.train_batch(&batch, 0.001)?;

    // If loss is significant, gradients should have been computed
    // We can't directly access gradients, but we can verify weights changed
    let initial_weight_a = trainer.adapter().lora_a().to_vec2::<f32>()?;
    let initial_weight_b = trainer.adapter().lora_b().to_vec2::<f32>()?;

    // Train again
    let _ = trainer.train_batch(&batch, 0.001)?;

    let final_weight_a = trainer.adapter().lora_a().to_vec2::<f32>()?;
    let final_weight_b = trainer.adapter().lora_b().to_vec2::<f32>()?;

    let weight_diff_a: f64 = initial_weight_a
        .iter()
        .zip(final_weight_a.iter())
        .map(|(init, fin)| ((init - fin) as f64).abs())
        .sum();
    let weight_diff_b: f64 = initial_weight_b
        .iter()
        .zip(final_weight_b.iter())
        .map(|(init, fin)| ((init - fin) as f64).abs())
        .sum();

    let total_diff = weight_diff_a + weight_diff_b;

    println!(
        "✅ Gradient test: loss={:.6}, weight_diff={:.9}",
        loss, total_diff
    );

    if loss > 0.001 {
        assert!(
            total_diff > 1e-6,
            "Gradients should have been computed but weights didn't update! loss={:.6}, diff={:.9}",
            loss,
            total_diff
        );
    }

    Ok(())
}

#[test]
fn test_epoch_0_updates_weights() -> Result<()> {
    // This test specifically validates the Phase 0 bug fix:
    // Weights should update even on epoch 0
    let config = LoRAConfig {
        input_dim: 128,
        output_dim: 128,
        rank: 8,
        alpha: 16.0,
        use_fp16: false,
    };

    let mut trainer = LoRATrainer::with_config(config)?;
    let data = create_test_batch(8, 128, 128);

    // Capture initial weights
    let initial_weight_a = trainer.adapter().lora_a().to_vec2::<f32>()?;
    let initial_weight_b = trainer.adapter().lora_b().to_vec2::<f32>()?;

    // Train for 1 epoch (epoch 0)
    let _loss = trainer.train_epoch(&data, 0.001, 0)?;

    // Capture final weights
    let final_weight_a = trainer.adapter().lora_a().to_vec2::<f32>()?;
    let final_weight_b = trainer.adapter().lora_b().to_vec2::<f32>()?;

    // Compute weight differences
    let weight_diff_a: f64 = initial_weight_a
        .iter()
        .zip(final_weight_a.iter())
        .map(|(init, fin)| ((init - fin) as f64).abs())
        .sum();
    let weight_diff_b: f64 = initial_weight_b
        .iter()
        .zip(final_weight_b.iter())
        .map(|(init, fin)| ((init - fin) as f64).abs())
        .sum();

    let total_diff = weight_diff_a + weight_diff_b;

    println!("✅ Epoch 0 weight update test: diff = {:.9}", total_diff);

    assert!(
        total_diff > 1e-6,
        "PHASE 0 BUG FIX FAILED: Weights did not update on epoch 0! diff = {:.9}",
        total_diff
    );

    Ok(())
}
