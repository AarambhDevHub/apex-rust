use apex_rust::config::{get_tiny_lora_config, get_tiny_vision_config};
use apex_rust::model::ApexModel;
use apex_rust::train::{
    clip_gradients, expand_labels_for_visual_tokens, get_lr, global_grad_norm,
    load_adapter_safetensors, load_model_safetensors, save_adapter_safetensors,
    save_checkpoint_metadata, save_model_safetensors, AdamWConfig, AdamWOptimizer, AdamWState,
    CheckpointMetadata, TrainingState,
};
use candle_core::Device;

fn small_lora_config() -> apex_rust::config::ApexConfig {
    let mut cfg = get_tiny_lora_config();
    cfg.model.n_layers = 2;
    cfg.attention.global_layer_freq = 2;
    cfg.moe.enabled = false;
    cfg.skip_gate.enabled = false;
    cfg
}

#[test]
fn cosine_warmup_scheduler_hits_expected_regions() {
    let lr0 = get_lr(0, 10, 100, 1.0, 0.1);
    let lr5 = get_lr(5, 10, 100, 1.0, 0.1);
    let lr_end = get_lr(100, 10, 100, 1.0, 0.1);
    assert_eq!(lr0, 0.0);
    assert!((lr5 - 0.5).abs() < 1e-12);
    assert!((lr_end - 0.1).abs() < 1e-12);
}

#[test]
fn adamw_update_and_gradient_clipping_work() -> apex_rust::Result<()> {
    let cfg = AdamWConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        grad_clip: 1.0,
    };
    let mut param = vec![1.0_f32, -1.0];
    let grad = vec![0.5_f32, -0.5];
    let mut state = AdamWState::new(param.len());
    apex_rust::train::adamw_update(&mut param, &grad, &mut state, &cfg)?;
    assert_eq!(state.step, 1);
    assert!(param[0] < 1.0);
    assert!(param[1] > -1.0);

    let mut grads = vec![vec![3.0_f32, 4.0]];
    let norm = clip_gradients(&mut grads, 1.0)?;
    assert!((norm - 5.0).abs() < 1e-6);
    assert!((global_grad_norm(&grads) - 1.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn adamw_optimizer_and_training_state_track_steps() -> apex_rust::Result<()> {
    let config = AdamWConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.01,
        grad_clip: 0.5,
    };
    let mut optimizer = AdamWOptimizer::new(&[2], config)?;
    let mut params = vec![vec![1.0_f32, 2.0]];
    let grads = vec![vec![1.0_f32, 1.0]];
    let stats = optimizer.step(&mut params, &grads)?;
    assert_eq!(stats.step, 1);
    assert!(stats.clipped);
    assert!(params[0][0] < 1.0);

    let mut state = TrainingState::new(2);
    assert!(!state.advance_micro_step());
    assert!(state.advance_micro_step());
    assert!(state.update_best_val_loss(2.0));
    assert!(!state.update_best_val_loss(3.0));
    Ok(())
}

#[test]
fn checkpoint_metadata_is_json_readable() -> apex_rust::Result<()> {
    let cfg = small_lora_config();
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("metadata.json");
    save_checkpoint_metadata(&path, 7, 2, 1.25, Some(&cfg))?;
    let text = std::fs::read_to_string(path)?;
    let metadata: CheckpointMetadata = serde_json::from_str(&text)?;
    assert_eq!(metadata.step, 7);
    assert_eq!(metadata.epoch, 2);
    assert_eq!(metadata.loss, 1.25);
    assert!(metadata.config.is_some());
    Ok(())
}

#[test]
fn adapter_safetensors_contains_lora_tensors() -> apex_rust::Result<()> {
    let cfg = small_lora_config();
    let model = ApexModel::new(cfg, Device::Cpu)?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("adapter.safetensors");
    save_adapter_safetensors(&path, &model)?;
    let bytes = std::fs::read(path)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| apex_rust::ApexError::Checkpoint(e.to_string()))?;
    assert!(tensors.names().iter().any(|name| name.contains("lora_A")));
    assert!(tensors.names().iter().any(|name| name.contains("lora_B")));
    Ok(())
}

#[test]
fn model_safetensors_load_roundtrips_weights() -> apex_rust::Result<()> {
    let cfg = small_lora_config();
    let model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let expected = model.embedding.flatten_all()?.to_vec1::<f32>()?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("model.safetensors");
    save_model_safetensors(&path, &model)?;

    let mut loaded = ApexModel::new(cfg, Device::Cpu)?;
    let report = load_model_safetensors(&path, &mut loaded, true)?;
    assert!(report.loaded > 0);
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    let actual = loaded.embedding.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn adapter_safetensors_load_roundtrips_adapter_weights() -> apex_rust::Result<()> {
    let cfg = small_lora_config();
    let model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let expected = adapter_tensor_values(&model, "lora_A.weight")?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("adapter.safetensors");
    save_adapter_safetensors(&path, &model)?;

    let mut loaded = ApexModel::new(cfg, Device::Cpu)?;
    let report = load_adapter_safetensors(&path, &mut loaded, true)?;
    assert!(report.loaded > 0);
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    let actual = adapter_tensor_values(&loaded, "lora_A.weight")?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn train_label_expansion_masks_visual_tokens() -> apex_rust::Result<()> {
    let cfg = get_tiny_vision_config();
    let ids = vec![vec![1, cfg.vision.image_token_id, 2, 3]];
    let labels = vec![vec![10, 11, 12, 13]];
    let expanded = expand_labels_for_visual_tokens(
        &ids,
        &labels,
        cfg.vision.image_token_id,
        cfg.vision.n_visual_tokens,
        -100,
    )?;
    assert_eq!(
        expanded[0].len(),
        labels[0].len() - 1 + cfg.vision.n_visual_tokens
    );
    assert!(expanded[0][1..1 + cfg.vision.n_visual_tokens]
        .iter()
        .all(|label| *label == -100));
    Ok(())
}

fn adapter_tensor_values(model: &ApexModel, suffix: &str) -> apex_rust::Result<Vec<f32>> {
    let (_, tensor) = model
        .adapter_tensors()
        .into_iter()
        .find(|(name, _)| name.ends_with(suffix))
        .expect("expected adapter tensor");
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}
