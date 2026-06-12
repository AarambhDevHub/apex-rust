use apex_rust::alignment::{
    dpo_loss, extract_thinking_text, grpo_advantages, reward_model_loss, score_combined_reward,
    ConstitutionalAI, ProcessRewardModel, RewardModel, RewardWeights,
};
use apex_rust::config::{
    get_tiny_config, get_tiny_dora_config, get_tiny_qdora_config, get_tiny_qlora_config,
    get_tiny_vision_config,
};
use apex_rust::data::{
    read_preference_jsonl, read_pretrain_jsonl, read_sft_jsonl, read_vision_jsonl,
};
use apex_rust::generation::{
    apply_top_k, apply_top_p, sample_next_token, ApexGenerator, GenerationConfig,
};
use apex_rust::model::{
    build_apex_attention_mask, dequantize_4bit_weight, pack_4bit_indices, precompute_rope_cache,
    quantize_4bit_weight, unpack_4bit_indices, ApexModel, LoadBalancer, RmsNorm,
};
use apex_rust::tokenizer::{ApexTokenizer, ChatMessage};
use apex_rust::utils::{
    build_architecture_diagram, build_layer_table, estimate_detailed_flops, flops_summary_text,
    inspection_markdown, parameter_summary_text, verify_shapes, ModelInspection,
};
use apex_rust::vision::{ImagePreprocessor, ImageTensor};
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
fn utility_reports_and_shape_checker_work() -> apex_rust::Result<()> {
    let cfg = fast_config();
    let mut model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let inspection = ModelInspection::from_model(&model);
    assert_eq!(inspection.layers.len(), cfg.model.n_layers);
    assert!(inspection_markdown(&inspection, true).contains("Layer Map"));
    assert!(parameter_summary_text(&model).contains("Total parameters"));

    let diagram = build_architecture_diagram(&cfg, "APEX Rust");
    assert!(diagram.contains("Layer 00"));
    let table = build_layer_table(&cfg);
    assert!(table.contains("Global MLA"));

    let flops = estimate_detailed_flops(&cfg, 8);
    assert!(flops.total > 0.0);
    assert!(flops_summary_text(&cfg, 8).contains("Total"));

    let shapes = verify_shapes(&mut model, 2, 8)?;
    assert!(shapes.all_passed(), "{:?}", shapes.checks);
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
fn generation_speculative_decoding_smoke() -> apex_rust::Result<()> {
    let cfg = fast_config();
    let mut model = ApexModel::new(cfg, Device::Cpu)?;
    let mut generator = ApexGenerator::new(
        &mut model,
        GenerationConfig {
            max_new_tokens: 4,
            temperature: 0.0,
            top_p: 1.0,
            use_speculative: true,
            ..GenerationConfig::default()
        },
    );
    let out = generator.generate(vec![1, 2, 3], 0)?;
    assert!(!out.token_ids.is_empty());
    assert_eq!(out.total_tokens, out.token_ids.len());
    assert!(out.total_tokens <= 4);
    Ok(())
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
fn vision_preprocessor_converts_resizes_and_batches() -> apex_rust::Result<()> {
    let mut cfg = get_tiny_vision_config();
    cfg.vision.image_size = 4;
    cfg.vision.in_channels = 3;
    let pre = ImagePreprocessor::new(&cfg);

    let hwc = ImageTensor::from_hwc(
        vec![
            255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 255.0,
        ],
        2,
        2,
        3,
    )?;
    let processed = pre.preprocess(&hwc)?;
    assert_eq!(processed.channels, 3);
    assert_eq!(processed.height, 4);
    assert_eq!(processed.width, 4);
    assert_eq!(processed.pixels.len(), 3 * 4 * 4);

    let gray = ImageTensor::from_chw(vec![0.25; 4], 1, 2, 2)?;
    let rgb = gray.to_channels(3)?;
    assert_eq!(rgb.channels, 3);
    assert_eq!(&rgb.pixels[0..4], &[0.25; 4]);
    assert_eq!(&rgb.pixels[4..8], &[0.25; 4]);
    assert_eq!(&rgb.pixels[8..12], &[0.25; 4]);

    let batch = pre.batch_to_tensor(&[hwc, gray], &Device::Cpu)?;
    assert_eq!(batch.dims(), &[2, 3, 4, 4]);
    Ok(())
}

#[cfg(feature = "image")]
#[test]
fn vision_preprocessor_loads_image_file() -> apex_rust::Result<()> {
    let mut cfg = get_tiny_vision_config();
    cfg.vision.image_size = 4;
    cfg.vision.in_channels = 3;
    let pre = ImagePreprocessor::new(&cfg);

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("sample.png");
    let mut img = image::RgbImage::new(2, 2);
    img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
    img.put_pixel(1, 0, image::Rgb([0, 255, 0]));
    img.put_pixel(0, 1, image::Rgb([0, 0, 255]));
    img.put_pixel(1, 1, image::Rgb([255, 255, 255]));
    img.save(&path)
        .map_err(|e| apex_rust::ApexError::Data(e.to_string()))?;

    let processed = pre.load_path(&path)?;
    assert_eq!(processed.channels, 3);
    assert_eq!(processed.height, 4);
    assert_eq!(processed.width, 4);
    let tensor = pre.load_path_to_tensor(&path, &Device::Cpu)?;
    assert_eq!(tensor.dims(), &[1, 3, 4, 4]);
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

#[test]
fn reward_and_process_reward_models_work() -> apex_rust::Result<()> {
    let cfg = fast_config();
    let model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let mut reward = RewardModel::new(model.clone(), cfg.model.d_model, true)?;
    let mask = vec![vec![1u8, 1, 1, 1]];
    let rewards = reward.forward(&[vec![1, 2, 3, 4]], Some(&mask))?;
    assert_eq!(rewards.len(), 1);
    let loss = reward_model_loss(&[rewards[0] + 1.0], &[rewards[0]])?;
    assert!(loss.is_finite());

    let mut prm = ProcessRewardModel::new(model, cfg.model.d_model, true)?;
    let scores = prm.score_steps(&[1, 2], &[vec![3], vec![4, 5]])?;
    assert_eq!(scores.len(), 2);
    assert!(scores.iter().all(|score| (0.0..=1.0).contains(score)));

    let steps = vec!["check the intermediate result".to_string()];
    assert!(prm.score_steps_from_text("prompt", &steps, None).is_err());
    let tok = ApexTokenizer::new(None::<&str>)?;
    let text_scores = prm.score_steps_from_text("prompt", &steps, Some(&tok))?;
    assert_eq!(text_scores.len(), 1);
    Ok(())
}

#[test]
fn constitutional_and_combined_rewards_work() -> apex_rust::Result<()> {
    let generator =
        |prompt: &str, _max_tokens: usize, _temperature: f64| -> apex_rust::Result<String> {
            if prompt.starts_with("Original response:") {
                Ok("A safer revised response.".to_string())
            } else if prompt.contains("dangerous materials") {
                Ok("YES - violates the dangerous materials principle.".to_string())
            } else {
                Ok("NO - no violation.".to_string())
            }
        };
    let mut constitutional = ConstitutionalAI::new(
        generator,
        Some(vec![
            "Never assist with creating weapons or dangerous materials.".to_string(),
            "Be helpful, harmless, and honest.".to_string(),
        ]),
        0.1,
        0.3,
    );
    let critiques = constitutional.critique_response("unsafe response", Some("prompt"));
    assert_eq!(critiques.len(), 2);
    assert!(critiques.iter().any(|critique| critique.violated));
    let revised = constitutional.revise_response("unsafe response", Some("prompt"));
    assert_eq!(revised.violation_count, 1);
    assert_eq!(revised.revised_response, "A safer revised response.");

    let steps = extract_thinking_text("x<|thinking|>\nfirst\nsecond\n<|/thinking|>y");
    assert_eq!(steps, vec!["first".to_string(), "second".to_string()]);
    let breakdown = score_combined_reward(
        Some(1.0),
        Some(&[0.5, 1.0]),
        Some(0.0),
        RewardWeights::default(),
    )?;
    assert!((breakdown.total - 0.65).abs() < 1e-9);
    Ok(())
}
