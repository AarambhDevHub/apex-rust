use apex_rust::alignment::{dpo_loss, grpo_advantages};
use apex_rust::config::{
    get_tiny_config, get_tiny_dora_config, get_tiny_qdora_config, get_tiny_qlora_config,
    get_tiny_vision_config,
};
use apex_rust::data::{
    read_preference_jsonl, read_pretrain_jsonl, read_sft_jsonl, read_vision_jsonl,
};
use apex_rust::generation::{apply_top_k, apply_top_p, sample_next_token, GenerationConfig};
use apex_rust::model::{
    build_apex_attention_mask, dequantize_4bit_weight, pack_4bit_indices, precompute_rope_cache,
    quantize_4bit_weight, unpack_4bit_indices, ApexModel, LoadBalancer, RmsNorm,
};
use apex_rust::tokenizer::{ApexTokenizer, ChatMessage};
use apex_rust::{eval, train};
use candle_core::{Device, Tensor};

fn fast_config() -> apex_rust::config::ApexConfig {
    let mut cfg = get_tiny_config();
    cfg.model.n_layers = 2;
    cfg.attention.global_layer_freq = 2;
    cfg.moe.enabled = false;
    cfg.skip_gate.enabled = false;
    cfg.multi_token_head.enabled = true;
    cfg.model.max_seq_len = 64;
    cfg.training.seq_len = 16;
    cfg
}

#[test]
fn config_roundtrip_and_validation() -> apex_rust::Result<()> {
    let cfg = fast_config();
    cfg.validate()?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("config.yaml");
    cfg.to_yaml(&path)?;
    let loaded = apex_rust::config::ApexConfig::from_yaml(&path)?;
    assert_eq!(loaded.model.d_model, cfg.model.d_model);
    assert_eq!(loaded.attention.global_layer_freq, 2);
    Ok(())
}

#[test]
fn tokenizer_formats_chat_and_token_types() -> apex_rust::Result<()> {
    let tok = ApexTokenizer::new(None::<&str>)?;
    let ids = tok.encode_chat(
        &[
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hi".to_string(),
            },
        ],
        false,
        false,
    )?;
    assert!(ids.contains(&tok.user_token_id()));
    assert!(ids.contains(&tok.assistant_token_id()));
    let types = tok.get_token_types(&ids);
    assert_eq!(types.len(), ids.len());
    assert!(types.contains(&2));
    Ok(())
}

#[test]
fn data_loaders_accept_original_samples() -> apex_rust::Result<()> {
    let tok = ApexTokenizer::new(None::<&str>)?;
    let pretrain = read_pretrain_jsonl("APEX-1/data/samples/tiny_text.jsonl", &tok, 16)?;
    let sft = read_sft_jsonl("APEX-1/data/samples/tiny_sft.jsonl", &tok, 32)?;
    let pref = read_preference_jsonl("APEX-1/data/samples/tiny_preference.jsonl", &tok, 32, 16)?;
    let vision = read_vision_jsonl("APEX-1/data/samples/tiny_vision.jsonl", None::<&str>)?;
    assert!(!pretrain.is_empty());
    assert!(!sft.is_empty());
    assert!(!pref.is_empty());
    assert!(!vision.is_empty());
    Ok(())
}

#[test]
fn rope_mask_norm_and_load_balancer_work() -> apex_rust::Result<()> {
    let device = Device::Cpu;
    let (cos, sin, factor) = precompute_rope_cache(8, 16, 10_000.0, 2.0, &device)?;
    assert_eq!(cos.dims(), &[16, 8]);
    assert_eq!(sin.dims(), &[16, 8]);
    assert!(factor > 1.0);
    let mask = build_apex_attention_mask(2, 5, 2, false);
    assert!(mask[0][1]);
    assert!(!mask[4][1]);
    let norm = RmsNorm::new(4, &device)?;
    let x = Tensor::from_vec(vec![1.0_f32; 8], (1, 2, 4), &device)?;
    let y = norm.forward(&x)?;
    assert_eq!(y.dims(), &[1, 2, 4]);
    let mut balancer = LoadBalancer::new(4, 0.1);
    let stats = balancer.update(&[vec![0, 1], vec![1, 2]]);
    assert!(stats.max_load >= stats.min_load);
    Ok(())
}

#[test]
fn model_forward_and_losses_work() -> apex_rust::Result<()> {
    let cfg = fast_config();
    let mut model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let tokens = vec![(9..25).collect::<Vec<u32>>()];
    let out = model.forward(&tokens, None, 0, None, true)?;
    assert_eq!(out.logits.dims(), &[1, 16, cfg.model.vocab_size]);
    assert!(out.hidden_states.is_some());
    assert!(out.spec_logits.is_some());
    let loss = train::compute_pretrain_loss(
        &out.logits,
        out.spec_logits.as_deref(),
        &tokens,
        cfg.multi_token_head.lambda_spec,
    )?;
    assert!(loss.loss_total.is_finite());
    let token_types = vec![vec![2u8; 16]];
    let sft = train::compute_sft_loss(&out.logits, &tokens, &token_types)?;
    assert!(sft.valid_tokens > 0);
    Ok(())
}

#[test]
fn model_safetensors_export_is_readable() -> apex_rust::Result<()> {
    let cfg = fast_config();
    let model = ApexModel::new(cfg, Device::Cpu)?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("model.safetensors");
    train::save_model_safetensors(&path, &model)?;
    let bytes = std::fs::read(&path)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| apex_rust::ApexError::Checkpoint(e.to_string()))?;
    assert!(tensors.names().contains(&"embedding.weight"));
    assert!(tensors.names().contains(&"final_norm.weight"));
    Ok(())
}

#[test]
fn peft_variants_insert_and_merge() -> apex_rust::Result<()> {
    for cfg in [
        apex_rust::config::get_tiny_lora_config(),
        get_tiny_qlora_config(),
        get_tiny_dora_config(),
        get_tiny_qdora_config(),
    ] {
        let mut cfg = cfg;
        cfg.model.n_layers = 2;
        cfg.attention.global_layer_freq = 2;
        cfg.moe.enabled = false;
        cfg.skip_gate.enabled = false;
        let mut model = ApexModel::new(cfg, Device::Cpu)?;
        assert!(model.trainable_parameters() < model.total_parameters());
        assert!(model.count_lora_modules() > 0);
        model.merge_and_unload_adapters()?;
        assert_eq!(model.count_lora_modules(), 0);
    }
    Ok(())
}

#[test]
fn four_bit_quantization_roundtrips_shape() -> apex_rust::Result<()> {
    let device = Device::Cpu;
    let indices = vec![0u8, 1, 2, 15, 8];
    let packed = pack_4bit_indices(&indices);
    assert_eq!(unpack_4bit_indices(&packed, indices.len()), indices);
    let weight = Tensor::from_vec(vec![-1.0_f32, -0.5, 0.5, 1.0], (2, 2), &device)?;
    let q = quantize_4bit_weight(&weight, "nf4", true)?;
    let restored = dequantize_4bit_weight(&q, &device)?;
    assert_eq!(restored.dims(), &[2, 2]);
    Ok(())
}

#[test]
fn generation_sampling_filters_logits() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    apply_top_k(&mut logits, 2);
    assert!(logits[0].is_infinite());
    apply_top_p(&mut logits, 0.9);
    let token = sample_next_token(
        &[0.0, 1.0, 3.0],
        &GenerationConfig {
            temperature: 0.0,
            ..GenerationConfig::default()
        },
        0.0,
        &[],
    );
    assert_eq!(token, 2);
}

#[test]
fn vision_inserts_visual_embeddings() -> apex_rust::Result<()> {
    let mut cfg = get_tiny_vision_config();
    cfg.model.n_layers = 2;
    cfg.attention.global_layer_freq = 2;
    cfg.moe.enabled = false;
    cfg.skip_gate.enabled = false;
    cfg.multi_token_head.enabled = false;
    cfg.training.seq_len = 16;
    let mut vision = apex_rust::vision::ApexVisionModel::new(cfg.clone(), Device::Cpu)?;
    let image = Tensor::from_vec(
        vec![0.5_f32; cfg.vision.in_channels * cfg.vision.image_size * cfg.vision.image_size],
        (
            1,
            cfg.vision.in_channels,
            cfg.vision.image_size,
            cfg.vision.image_size,
        ),
        &Device::Cpu,
    )?;
    let tokens = vec![vec![1, cfg.vision.image_token_id, 2, 3]];
    let out = vision.forward(&tokens, Some(&image), 0, false)?;
    assert_eq!(out.logits.dim(1)?, 3 + cfg.vision.n_visual_tokens);
    Ok(())
}

#[test]
fn alignment_and_eval_helpers_work() -> apex_rust::Result<()> {
    let metrics = dpo_loss(-1.0, -2.0, -1.5, -1.8, 0.1, 0.0);
    assert!(metrics.loss.is_finite());
    let adv = grpo_advantages(&[1.0, 2.0, 3.0]);
    assert_eq!(adv.len(), 3);
    let report =
        eval::evaluate_generated_texts(&["hello world".to_string(), "hello rust".to_string()]);
    assert_eq!(report.count, 2);
    Ok(())
}
