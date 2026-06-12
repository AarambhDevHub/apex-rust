use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use serde::{Deserialize, Serialize};

use crate::config::ApexConfig;
use crate::data::{PreferenceSample, SftSample};
use crate::error::{ApexError, Result};
use crate::model::ApexModel;
use crate::tensor;

use super::autograd::{
    adapter_dpo_loss_tensor, clip_grad_store, grpo_clipped_loss_tensor, pretrain_loss_tensor,
    sequence_logprob_tensor, sft_loss_tensor, AutogradStepStats,
};
use super::losses::{compute_pretrain_loss, LossMetrics};
use super::scheduler::get_lr;
use super::variables::{scope_for_model, TrainableVariables};

/// Runs one forward/loss step without updating model parameters.
pub fn dry_run_pretrain_step(model: &mut ApexModel, tokens: &[Vec<u32>]) -> Result<LossMetrics> {
    let out = model.forward(tokens, None, 0, None, false)?;
    compute_pretrain_loss(
        &out.logits,
        out.spec_logits.as_deref(),
        tokens,
        model.config.multi_token_head.lambda_spec,
    )
}

/// Summary returned by a real training loop.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingReport {
    /// Number of requested outer steps.
    pub steps: usize,
    /// Number of optimizer updates actually applied.
    pub optimizer_steps: usize,
    /// Last detached loss value.
    pub final_loss: f64,
    /// Last pre-clip gradient norm.
    pub final_grad_norm: f64,
    /// Number of trainable tensors registered with Candle.
    pub trainable_tensors: usize,
    /// Number of trainable scalar parameters registered with Candle.
    pub trainable_parameters: usize,
}

/// Trains the model with next-token pretraining loss.
pub fn train_pretrain_steps(
    model: &mut ApexModel,
    samples: &[Vec<u32>],
    steps: usize,
    dry_run: bool,
) -> Result<TrainingReport> {
    ensure_non_empty(samples, "pretrain")?;
    let variables = TrainableVariables::attach_model(model, scope_for_model(model))?;
    let mut optimizer = build_adamw(&variables, &model.config)?;
    let mut report = report_for(&variables, steps.max(1));
    let accumulation = accumulation_steps(&model.config, dry_run);
    for step in 0..steps.max(1) {
        let mut combined = None;
        let mut logged_loss = 0.0;
        let mut logged = 0usize;
        for micro in 0..accumulation {
            let idx = (step * accumulation + micro) % samples.len();
            let batch = &[samples[idx].clone()];
            let output = model.forward(batch, None, 0, None, false)?;
            let loss = pretrain_loss_tensor(
                &output.logits,
                output.spec_logits.as_deref(),
                batch,
                model.config.multi_token_head.lambda_spec,
            )?;
            logged_loss += loss.metrics.loss_total;
            logged += 1;
            append_scaled_loss(&mut combined, &loss.loss, accumulation)?;
        }
        report.final_loss = logged_loss / logged.max(1) as f64;
        if dry_run {
            break;
        }
        let stats = optimizer_step(
            combined.ok_or_else(|| ApexError::Model("empty pretrain loss".to_string()))?,
            &variables,
            &mut optimizer,
            &model.config,
            report.optimizer_steps + 1,
        )?;
        report.optimizer_steps += 1;
        report.final_grad_norm = stats.grad_norm;
    }
    Ok(report)
}

/// Trains the model with assistant-token SFT loss.
pub fn train_sft_steps(
    model: &mut ApexModel,
    samples: &[SftSample],
    steps: usize,
    dry_run: bool,
) -> Result<TrainingReport> {
    ensure_non_empty(samples, "sft")?;
    let variables = TrainableVariables::attach_model(model, scope_for_model(model))?;
    let mut optimizer = build_adamw(&variables, &model.config)?;
    let mut report = report_for(&variables, steps.max(1));
    let accumulation = accumulation_steps(&model.config, dry_run);
    for step in 0..steps.max(1) {
        let mut combined = None;
        let mut logged_loss = 0.0;
        let mut logged = 0usize;
        for micro in 0..accumulation {
            let sample = &samples[(step * accumulation + micro) % samples.len()];
            let ids = std::slice::from_ref(&sample.input_ids);
            let types = std::slice::from_ref(&sample.token_types);
            let output = model.forward(ids, None, 0, None, false)?;
            let loss = sft_loss_tensor(&output.logits, ids, types)?;
            logged_loss += loss.metrics.loss_total;
            logged += 1;
            append_scaled_loss(&mut combined, &loss.loss, accumulation)?;
        }
        report.final_loss = logged_loss / logged.max(1) as f64;
        if dry_run {
            break;
        }
        let stats = optimizer_step(
            combined.ok_or_else(|| ApexError::Model("empty sft loss".to_string()))?,
            &variables,
            &mut optimizer,
            &model.config,
            report.optimizer_steps + 1,
        )?;
        report.optimizer_steps += 1;
        report.final_grad_norm = stats.grad_norm;
    }
    Ok(report)
}

/// Trains adapter policy weights with DPO preference loss.
pub fn train_adapter_dpo_steps(
    policy: &mut ApexModel,
    mut reference: Option<&mut ApexModel>,
    samples: &[PreferenceSample],
    steps: usize,
    dry_run: bool,
) -> Result<TrainingReport> {
    ensure_non_empty(samples, "adapter-dpo")?;
    let variables = TrainableVariables::attach_model(policy, scope_for_model(policy))?;
    let mut optimizer = build_adamw(&variables, &policy.config)?;
    let mut report = report_for(&variables, steps.max(1));
    let accumulation = accumulation_steps(&policy.config, dry_run);
    for step in 0..steps.max(1) {
        let mut combined = None;
        let mut logged_loss = 0.0;
        let mut logged = 0usize;
        for micro in 0..accumulation {
            let sample = &samples[(step * accumulation + micro) % samples.len()];
            let loss = adapter_dpo_loss_tensor(
                policy,
                reference.as_deref_mut(),
                sample,
                policy.config.adapter_dpo.beta,
                policy.config.adapter_dpo.label_smoothing,
                policy.config.adapter_dpo.reference_free,
                policy.config.adapter_dpo.length_normalize,
            )?;
            logged_loss += loss.metrics.loss_total;
            logged += 1;
            append_scaled_loss(&mut combined, &loss.loss, accumulation)?;
        }
        report.final_loss = logged_loss / logged.max(1) as f64;
        if dry_run {
            break;
        }
        let stats = optimizer_step(
            combined.ok_or_else(|| ApexError::Model("empty adapter-dpo loss".to_string()))?,
            &variables,
            &mut optimizer,
            &policy.config,
            report.optimizer_steps + 1,
        )?;
        report.optimizer_steps += 1;
        report.final_grad_norm = stats.grad_norm;
    }
    Ok(report)
}

/// Trains policy weights with a compact preference-row GRPO objective.
pub fn train_grpo_steps(
    policy: &mut ApexModel,
    samples: &[PreferenceSample],
    steps: usize,
    dry_run: bool,
) -> Result<TrainingReport> {
    ensure_non_empty(samples, "grpo")?;
    let variables = TrainableVariables::attach_model(policy, scope_for_model(policy))?;
    let mut optimizer = build_adamw(&variables, &policy.config)?;
    let mut report = report_for(&variables, steps.max(1));
    let accumulation = accumulation_steps(&policy.config, dry_run);
    for step in 0..steps.max(1) {
        let mut combined = None;
        let mut logged_loss = 0.0;
        let mut logged = 0usize;
        for micro in 0..accumulation {
            let sample = &samples[(step * accumulation + micro) % samples.len()];
            let old_chosen = crate::alignment::sequence_logprob(
                policy,
                &sample.chosen_ids,
                sample.prompt_len,
                policy.config.adapter_dpo.length_normalize,
            )?;
            let old_rejected = crate::alignment::sequence_logprob(
                policy,
                &sample.rejected_ids,
                sample.prompt_len,
                policy.config.adapter_dpo.length_normalize,
            )?;
            let new_chosen = sequence_logprob_tensor(
                policy,
                &sample.chosen_ids,
                sample.prompt_len,
                policy.config.adapter_dpo.length_normalize,
            )?;
            let new_rejected = sequence_logprob_tensor(
                policy,
                &sample.rejected_ids,
                sample.prompt_len,
                policy.config.adapter_dpo.length_normalize,
            )?;
            let advantages = crate::alignment::grpo_advantages(&[1.0, 0.0]);
            let loss = grpo_clipped_loss_tensor(
                &[old_chosen, old_rejected],
                &[new_chosen, new_rejected],
                &advantages,
                policy.config.grpo.clip_eps,
            )?;
            logged_loss += loss.to_scalar::<f32>()? as f64;
            logged += 1;
            append_scaled_loss(&mut combined, &loss, accumulation)?;
        }
        report.final_loss = logged_loss / logged.max(1) as f64;
        if dry_run {
            break;
        }
        let stats = optimizer_step(
            combined.ok_or_else(|| ApexError::Model("empty grpo loss".to_string()))?,
            &variables,
            &mut optimizer,
            &policy.config,
            report.optimizer_steps + 1,
        )?;
        report.optimizer_steps += 1;
        report.final_grad_norm = stats.grad_norm;
    }
    Ok(report)
}

fn ensure_non_empty<T>(samples: &[T], name: &str) -> Result<()> {
    if samples.is_empty() {
        return Err(ApexError::Data(format!("{name} training requires samples")));
    }
    Ok(())
}

fn report_for(variables: &TrainableVariables, steps: usize) -> TrainingReport {
    TrainingReport {
        steps,
        optimizer_steps: 0,
        final_loss: 0.0,
        final_grad_norm: 0.0,
        trainable_tensors: variables.len(),
        trainable_parameters: variables.parameter_count(),
    }
}

fn accumulation_steps(config: &ApexConfig, dry_run: bool) -> usize {
    if dry_run {
        1
    } else {
        config.training.gradient_accumulation_steps.max(1)
    }
}

fn build_adamw(variables: &TrainableVariables, config: &ApexConfig) -> Result<AdamW> {
    if variables.is_empty() {
        return Err(ApexError::Config(
            "no trainable tensors were registered".to_string(),
        ));
    }
    Ok(AdamW::new(
        variables.vars(),
        ParamsAdamW {
            lr: config.training.peak_lr,
            beta1: config.training.beta1,
            beta2: config.training.beta2,
            eps: config.training.eps,
            weight_decay: config.training.weight_decay,
        },
    )?)
}

fn optimizer_step(
    loss: Tensor,
    variables: &TrainableVariables,
    optimizer: &mut AdamW,
    config: &ApexConfig,
    step: usize,
) -> Result<AutogradStepStats> {
    optimizer.set_learning_rate(get_lr(
        step,
        config.training.warmup_steps,
        config.training.max_steps,
        config.training.peak_lr,
        config.training.min_lr_ratio,
    ));
    let mut grads = loss.backward()?;
    let stats = clip_grad_store(&mut grads, variables, config.training.grad_clip)?;
    optimizer.step(&grads)?;
    Ok(stats)
}

fn append_scaled_loss(
    total: &mut Option<Tensor>,
    loss: &Tensor,
    accumulation: usize,
) -> Result<()> {
    let scaled = loss.broadcast_div(&tensor::scalar(accumulation.max(1) as f64, loss.device())?)?;
    *total = Some(match total.take() {
        Some(prev) => prev.broadcast_add(&scaled)?,
        None => scaled,
    });
    Ok(())
}
