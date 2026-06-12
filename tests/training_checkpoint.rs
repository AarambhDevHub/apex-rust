use apex_rust::config::{get_tiny_lora_config, get_tiny_vision_config};
use apex_rust::model::ApexModel;
use apex_rust::train::{
    expand_labels_for_visual_tokens, get_lr, save_adapter_safetensors, save_checkpoint_metadata,
    CheckpointMetadata,
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
