use apex_rust::config::{
    get_tiny_adapter_dpo_config, get_tiny_config, get_tiny_vision_config, PeftMethod,
};
use apex_rust::data::format_preference_example;
use apex_rust::model::ApexModel;
use apex_rust::tokenizer::ApexTokenizer;
use apex_rust::train::{
    train_adapter_dpo_steps, train_grpo_steps, train_pretrain_steps, TrainableVariables,
};
use apex_rust::utils::resolve_runtime;
use apex_rust::vision::{ApexVisionModel, NativeVisionEncoder};
use candle_core::{Device, IndexOp, Tensor};

fn fast_text_config() -> apex_rust::config::ApexConfig {
    let mut cfg = get_tiny_config();
    cfg.model.n_layers = 2;
    cfg.attention.global_layer_freq = 2;
    cfg.moe.enabled = false;
    cfg.skip_gate.enabled = false;
    cfg.multi_token_head.enabled = false;
    cfg.training.seq_len = 8;
    cfg.training.batch_size = 1;
    cfg.training.warmup_steps = 1;
    cfg.training.max_steps = 4;
    cfg
}

fn fast_preference_config() -> apex_rust::config::ApexConfig {
    let mut cfg = get_tiny_adapter_dpo_config(PeftMethod::Lora);
    cfg.model.n_layers = 2;
    cfg.attention.global_layer_freq = 2;
    cfg.moe.enabled = false;
    cfg.skip_gate.enabled = false;
    cfg.multi_token_head.enabled = false;
    cfg.adapter_dpo.max_prompt_len = 16;
    cfg.adapter_dpo.max_response_len = 8;
    cfg.training.warmup_steps = 1;
    cfg.training.max_steps = 4;
    cfg
}

#[test]
fn pretrain_runner_applies_optimizer_step() -> apex_rust::Result<()> {
    let cfg = fast_text_config();
    let mut model = ApexModel::new(cfg, Device::Cpu)?;
    let before = model.embedding.i((9, 0))?.to_scalar::<f32>()?;
    let samples = vec![vec![9, 10, 11, 12, 13, 14, 15, 16]];
    let report = train_pretrain_steps(&mut model, &samples, 1, false)?;
    let after = model.embedding.i((9, 0))?.to_scalar::<f32>()?;
    assert_eq!(report.optimizer_steps, 1);
    assert!(report.trainable_parameters > 0);
    assert!(report.final_loss.is_finite());
    assert_ne!(before, after);
    Ok(())
}

#[test]
fn adapter_dpo_and_grpo_runners_apply_steps() -> apex_rust::Result<()> {
    let cfg = fast_preference_config();
    let tok = ApexTokenizer::new(None::<&str>)?;
    let sample = format_preference_example(
        &tok,
        "Say hi.",
        "Hi.",
        "No.",
        cfg.adapter_dpo.max_prompt_len,
        cfg.adapter_dpo.max_response_len,
        true,
    )?;
    let mut policy = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let mut reference = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let dpo = train_adapter_dpo_steps(
        &mut policy,
        Some(&mut reference),
        std::slice::from_ref(&sample),
        1,
        false,
    )?;
    assert_eq!(dpo.optimizer_steps, 1);
    assert!(dpo.trainable_parameters > 0);

    let mut grpo_policy = ApexModel::new(cfg, Device::Cpu)?;
    let grpo = train_grpo_steps(&mut grpo_policy, &[sample], 1, false)?;
    assert_eq!(grpo.optimizer_steps, 1);
    assert!(grpo.final_loss.is_finite());
    Ok(())
}

#[test]
fn vision_encoder_depth_changes_parameter_count_and_registers_variables() -> apex_rust::Result<()> {
    let mut shallow = get_tiny_vision_config();
    shallow.vision.n_layers = 0;
    let mut deep = shallow.clone();
    deep.vision.n_layers = 2;
    let shallow_encoder = NativeVisionEncoder::new(&shallow, &Device::Cpu)?;
    let deep_encoder = NativeVisionEncoder::new(&deep, &Device::Cpu)?;
    assert!(deep_encoder.parameters() > shallow_encoder.parameters());

    let mut model = ApexVisionModel::new(deep.clone(), Device::Cpu)?;
    let variables = TrainableVariables::attach_vision_model(&mut model)?;
    assert!(variables.parameter_count() > deep_encoder.parameters());
    let image = Tensor::from_vec(
        vec![0.5_f32; deep.vision.in_channels * deep.vision.image_size * deep.vision.image_size],
        (
            1,
            deep.vision.in_channels,
            deep.vision.image_size,
            deep.vision.image_size,
        ),
        &Device::Cpu,
    )?;
    let tokens = model.encoder.forward(&image)?;
    assert_eq!(tokens.dim(2)?, deep.vision.d_vision);
    Ok(())
}

#[test]
fn runtime_resolution_handles_cpu_fallback_and_cuda_error() -> apex_rust::Result<()> {
    let runtime = resolve_runtime("auto", Some("fp16"))?;
    assert_eq!(runtime.device_label, "cpu");
    assert!(runtime.fallback_reason.is_some());
    assert!(resolve_runtime("cuda:0", Some("fp16")).is_err());
    Ok(())
}
