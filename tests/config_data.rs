use apex_rust::config::{
    get_tiny_adapter_dpo_config, get_tiny_config, get_tiny_dora_config, get_tiny_lora_config,
    get_tiny_qdora_config, get_tiny_qlora_config, get_tiny_vision_config, ApexConfig, PeftMethod,
};
use apex_rust::data::{
    format_preference_example, pack_tokens, read_preference_jsonl, read_vision_jsonl,
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
    ] {
        let cfg = ApexConfig::from_yaml(root.join(name))?;
        cfg.validate()?;
    }
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
