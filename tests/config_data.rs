use apex_rust::config::{
    get_tiny_adapter_dpo_config, get_tiny_config, get_tiny_dora_config,
    get_tiny_dora_inference_config, get_tiny_lora_config, get_tiny_lora_inference_config,
    get_tiny_qdora_config, get_tiny_qdora_inference_config, get_tiny_qlora_config,
    get_tiny_qlora_inference_config, get_tiny_vision_config, ApexConfig, PeftMethod,
};
use apex_rust::data::{
    batch_preference_samples, batch_pretrain_samples, batch_sft_samples, collate_vision_text_batch,
    format_preference_example, pack_tokens, read_preference_jsonl, read_vision_jsonl, BatchOptions,
    SftSample, StreamingPretrainDataset, IGNORE_INDEX,
};
use apex_rust::tokenizer::ApexTokenizer;

#[test]
fn presets_validate() -> apex_rust::Result<()> {
    for cfg in [
        get_tiny_config(),
        get_tiny_lora_config(),
        get_tiny_qlora_config(),
        get_tiny_dora_config(),
        get_tiny_qdora_config(),
        get_tiny_adapter_dpo_config(PeftMethod::Lora),
        get_tiny_lora_inference_config(),
        get_tiny_qlora_inference_config(),
        get_tiny_dora_inference_config(),
        get_tiny_qdora_inference_config(),
        get_tiny_vision_config(),
    ] {
        cfg.validate()?;
    }
    Ok(())
}

#[test]
fn yaml_config_presets_validate() -> apex_rust::Result<()> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("configs");
    for name in [
        "tiny.yaml",
        "small.yaml",
        "medium.yaml",
        "large.yaml",
        "tiny_vision.yaml",
        "small_vision.yaml",
        "medium_vision.yaml",
        "large_vision.yaml",
        "tiny_lora.yaml",
        "tiny_qlora.yaml",
        "tiny_dora.yaml",
        "tiny_qdora.yaml",
        "tiny_lora_dpo.yaml",
        "tiny_qlora_dpo.yaml",
        "tiny_dora_dpo.yaml",
        "tiny_qdora_dpo.yaml",
        "tiny_lora_inference.yaml",
        "tiny_qlora_inference.yaml",
        "tiny_dora_inference.yaml",
        "tiny_qdora_inference.yaml",
    ] {
        let cfg = ApexConfig::from_yaml(root.join(name))?;
        cfg.validate()?;
    }
    Ok(())
}

#[test]
fn inference_preset_loads_generation_defaults() -> apex_rust::Result<()> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("configs");
    let cfg = ApexConfig::from_yaml(root.join("tiny_lora_inference.yaml"))?;
    assert!(cfg.peft.enabled);
    assert_eq!(cfg.peft.method, PeftMethod::Lora);
    assert_eq!(cfg.generation.max_new_tokens, 64);
    assert!(cfg.generation.use_speculative);
    Ok(())
}

#[test]
fn validation_rejects_bad_head_geometry() {
    let mut cfg = get_tiny_config();
    cfg.model.n_heads_kv = 3;
    let err = cfg
        .validate()
        .expect_err("invalid head geometry should fail");
    assert!(err.to_string().contains("n_heads_q"));
}

#[test]
fn pack_tokens_drops_tiny_tail_after_full_sample() {
    let samples = pack_tokens(&[1, 2, 3, 4, 5], 4, 0);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].input_ids, vec![1, 2, 3, 4]);
    assert_eq!(samples[0].attention_mask, vec![1, 1, 1, 1]);
}

#[test]
fn preference_aliases_are_supported() -> apex_rust::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("pref.jsonl");
    std::fs::write(
        &path,
        r#"{"instruction":"Pick one","accepted":"yes","dispreferred":"no"}"#,
    )?;
    let tok = ApexTokenizer::new(None::<&str>)?;
    let samples = read_preference_jsonl(&path, &tok, 32, 16)?;
    assert_eq!(samples.len(), 1);
    assert!(samples[0].chosen_ids.len() > samples[0].prompt_len);
    assert!(samples[0].rejected_ids.len() > samples[0].prompt_len);
    Ok(())
}

#[test]
fn preference_formatter_can_skip_chat_template() -> apex_rust::Result<()> {
    let tok = ApexTokenizer::new(None::<&str>)?;
    let sample = format_preference_example(&tok, "prompt", "chosen", "rejected", 16, 8, false)?;
    assert!(!sample.prompt_ids.is_empty());
    assert_eq!(sample.prompt_len, sample.prompt_ids.len());
    Ok(())
}

#[test]
fn vision_prompt_response_schema_is_supported() -> apex_rust::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("vision.jsonl");
    std::fs::write(
        &path,
        r#"{"image":"image.png","prompt":"<|img|> describe","response":"a chart"}"#,
    )?;
    let samples = read_vision_jsonl(&path, None::<&str>)?;
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].prompt, "<|img|> describe");
    assert_eq!(samples[0].response, "a chart");
    assert!(samples[0].image.ends_with("image.png"));
    Ok(())
}

#[test]
fn batch_builders_collate_core_sample_types() -> apex_rust::Result<()> {
    let tok = ApexTokenizer::new(None::<&str>)?;
    let pretrain = pack_tokens(&[1, 2, 3, 4, 5, 6, 7, 8], 4, tok.pad_token_id());
    let batches = batch_pretrain_samples(&pretrain, BatchOptions::new(2, false, false, 0)?);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].input_ids.len(), 2);

    let sft = vec![
        SftSample {
            input_ids: vec![1, 2, 0],
            token_types: vec![0, 2, 0],
        },
        SftSample {
            input_ids: vec![1, 3, 0],
            token_types: vec![0, 1, 0],
        },
    ];
    let sft_batches = batch_sft_samples(&sft, BatchOptions::new(2, false, true, 0)?);
    assert_eq!(sft_batches[0].token_types[0], vec![0, 2, 0]);

    let pref = vec![
        format_preference_example(&tok, "prompt", "chosen", "rejected", 16, 8, false)?,
        format_preference_example(&tok, "prompt 2", "yes", "no", 16, 8, false)?,
    ];
    let pref_batches = batch_preference_samples(
        &pref,
        BatchOptions::new(2, false, true, 0)?,
        tok.pad_token_id(),
        24,
    );
    assert_eq!(pref_batches.len(), 1);
    assert_eq!(pref_batches[0].chosen_ids.len(), 2);
    assert_eq!(
        pref_batches[0].chosen_ids[0].len(),
        pref_batches[0].chosen_attention_mask[0].len()
    );
    Ok(())
}

#[test]
fn streaming_pretrain_dataset_emits_final_attention_mask() -> apex_rust::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("stream.txt");
    std::fs::write(&path, "abcd\n")?;
    let tok = ApexTokenizer::new(None::<&str>)?;
    let stream = StreamingPretrainDataset::new(vec![path], &tok, 8, false, 7)?;
    let samples = stream.collect_samples(None)?;
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].input_ids.len(), 8);
    assert_eq!(samples[0].attention_mask, vec![1, 1, 1, 1, 0, 0, 0, 0]);
    Ok(())
}

#[test]
fn vision_text_collator_masks_prompt_labels() -> apex_rust::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("vision.jsonl");
    std::fs::write(
        &path,
        r#"{"image":"image.png","prompt":"describe","response":"a chart"}"#,
    )?;
    let samples = read_vision_jsonl(&path, None::<&str>)?;
    let tok = ApexTokenizer::new(None::<&str>)?;
    let batch = collate_vision_text_batch(&samples, &tok, 64, tok.pad_token_id(), IGNORE_INDEX)?;
    assert_eq!(batch.token_ids.len(), 1);
    assert_eq!(batch.labels.len(), 1);
    assert_eq!(batch.token_ids[0].len(), batch.labels[0].len());
    assert!(batch.labels[0].contains(&IGNORE_INDEX));
    assert_eq!(batch.prompts[0], "describe");
    Ok(())
}
