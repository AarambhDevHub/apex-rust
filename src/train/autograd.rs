//! Differentiable training losses and gradient utilities built on Candle autograd.

use candle_core::{backprop::GradStore, IndexOp, Tensor};
use serde::{Deserialize, Serialize};

use crate::data::PreferenceSample;
use crate::error::{ApexError, Result};
use crate::model::ApexModel;
use crate::tensor;

use super::losses::LossMetrics;
use super::variables::TrainableVariables;

/// A differentiable scalar loss plus detached metrics for logging.
#[derive(Clone)]
pub struct AutogradLoss {
    /// Scalar loss tensor that should be passed to `backward`.
    pub loss: Tensor,
    /// Detached numeric metrics for logs and checkpoint metadata.
    pub metrics: LossMetrics,
}

/// Scalar stats returned after gradient clipping.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AutogradStepStats {
    /// Global gradient norm before clipping.
    pub grad_norm: f64,
    /// Whether the gradients were scaled down.
    pub clipped: bool,
}

/// Computes graph-preserving next-token pretraining loss.
pub fn pretrain_loss_tensor(
    logits: &Tensor,
    spec_logits: Option<&[Tensor]>,
    token_ids: &[Vec<u32>],
    lambda_spec: f64,
) -> Result<AutogradLoss> {
    let main = shifted_cross_entropy_tensor(logits, token_ids, 1, None)?;
    let mut total = main.loss.clone();
    let mut spec_sum = 0.0;
    let mut spec_count = 0usize;
    if let Some(specs) = spec_logits {
        for (offset, spec) in specs.iter().enumerate() {
            let k = offset + 1;
            if token_ids
                .first()
                .map(Vec::len)
                .unwrap_or(0)
                .saturating_sub(k)
                < 1
            {
                break;
            }
            let spec_loss = shifted_cross_entropy_tensor(spec, token_ids, k, None)?;
            if spec_loss.metrics.valid_tokens > 0 {
                let weighted = spec_loss
                    .loss
                    .broadcast_mul(&tensor::scalar(lambda_spec, spec.device())?)?;
                total = total.broadcast_add(&weighted)?;
                spec_sum += spec_loss.metrics.loss_main;
                spec_count += 1;
            }
        }
    }
    let loss_total = scalar_value(&total)?;
    let loss_spec_avg = if spec_count > 0 {
        spec_sum / spec_count as f64
    } else {
        0.0
    };
    Ok(AutogradLoss {
        loss: total,
        metrics: LossMetrics {
            loss_total,
            loss_main: main.metrics.loss_main,
            loss_spec_avg,
            valid_tokens: main.metrics.valid_tokens,
        },
    })
}

/// Computes graph-preserving assistant-token SFT loss.
pub fn sft_loss_tensor(
    logits: &Tensor,
    token_ids: &[Vec<u32>],
    token_types: &[Vec<u8>],
) -> Result<AutogradLoss> {
    shifted_cross_entropy_tensor(logits, token_ids, 1, Some(token_types))
}

/// Computes graph-preserving multimodal SFT loss from expanded label rows.
pub fn vision_sft_loss_tensor(
    logits: &Tensor,
    labels: &[Vec<i64>],
    ignore_index: i64,
) -> Result<AutogradLoss> {
    let dims = logits.dims();
    if dims.len() != 3 {
        return Err(ApexError::Shape(format!(
            "vision_sft_loss_tensor expects [B,S,V], got {dims:?}"
        )));
    }
    let (b, s, _) = (dims[0], dims[1], dims[2]);
    if labels.len() != b || labels.iter().any(|row| row.len() != s) {
        return Err(ApexError::Shape(
            "vision labels must match logits batch and sequence length".to_string(),
        ));
    }
    let mut total = None;
    let mut count = 0usize;
    for (batch, row) in labels.iter().enumerate().take(b) {
        for pos in 0..s.saturating_sub(1) {
            let label = row[pos + 1];
            if label == ignore_index {
                continue;
            }
            let term = cross_entropy_row(&logits.i((batch, pos, ..))?, label)?;
            append_loss_term(&mut total, term)?;
            count += 1;
        }
    }
    finalize_cross_entropy(total, count, logits.device())
}

/// Computes a differentiable response log-probability for preference training.
pub fn sequence_logprob_tensor(
    model: &mut ApexModel,
    token_ids: &[u32],
    response_start: usize,
    length_normalize: bool,
) -> Result<Tensor> {
    if token_ids.len() < 2 {
        return Err(ApexError::Data(
            "sequence_logprob_tensor requires at least two tokens".to_string(),
        ));
    }
    let output = model.forward(&[token_ids.to_vec()], None, response_start, None, false)?;
    let logits = output.logits.i((0, .., ..))?;
    let mut total = None;
    let mut count = 0usize;
    for (pos, token_id) in token_ids
        .iter()
        .copied()
        .enumerate()
        .skip(response_start.max(1))
    {
        if pos == 0 || pos > logits.dim(0)? {
            continue;
        }
        let label = token_id as usize;
        let row = logits.i((pos - 1, ..))?;
        let log_probs = tensor::log_softmax_last(&row)?;
        if label >= log_probs.dim(0)? {
            return Err(ApexError::Data(format!(
                "label {label} outside vocab {}",
                log_probs.dim(0)?
            )));
        }
        append_loss_term(&mut total, log_probs.i(label)?)?;
        count += 1;
    }
    let sum = total.unwrap_or(tensor::scalar(0.0, logits.device())?);
    if length_normalize && count > 0 {
        Ok(sum.broadcast_div(&tensor::scalar(count as f64, logits.device())?)?)
    } else {
        Ok(sum)
    }
}

/// Builds a graph-preserving DPO loss from policy tensors and reference scalars.
pub fn dpo_loss_tensor(
    policy_chosen: &Tensor,
    policy_rejected: &Tensor,
    reference_chosen: f64,
    reference_rejected: f64,
    beta: f64,
    label_smoothing: f64,
) -> Result<Tensor> {
    let policy_delta = policy_chosen.broadcast_sub(policy_rejected)?;
    let reference_delta = tensor::scalar(
        reference_chosen - reference_rejected,
        policy_chosen.device(),
    )?;
    let margin = policy_delta
        .broadcast_sub(&reference_delta)?
        .broadcast_mul(&tensor::scalar(beta, policy_chosen.device())?)?;
    let positive_loss = softplus(&margin.neg()?)?;
    let negative_loss = softplus(&margin)?;
    positive_loss
        .broadcast_mul(&tensor::scalar(
            1.0 - label_smoothing,
            policy_chosen.device(),
        )?)?
        .broadcast_add(
            &negative_loss
                .broadcast_mul(&tensor::scalar(label_smoothing, policy_chosen.device())?)?,
        )
        .map_err(Into::into)
}

/// Computes adapter-DPO tensor loss and detached metrics for one sample.
pub fn adapter_dpo_loss_tensor(
    policy: &mut ApexModel,
    reference: Option<&mut ApexModel>,
    sample: &PreferenceSample,
    beta: f64,
    label_smoothing: f64,
    reference_free: bool,
    length_normalize: bool,
) -> Result<AutogradLoss> {
    let policy_chosen = sequence_logprob_tensor(
        policy,
        &sample.chosen_ids,
        sample.prompt_len,
        length_normalize,
    )?;
    let policy_rejected = sequence_logprob_tensor(
        policy,
        &sample.rejected_ids,
        sample.prompt_len,
        length_normalize,
    )?;
    let (reference_chosen, reference_rejected) = if reference_free {
        (0.0, 0.0)
    } else {
        let reference = reference.ok_or_else(|| {
            ApexError::Data(
                "adapter_dpo_loss_tensor requires a reference model unless reference_free=true"
                    .to_string(),
            )
        })?;
        (
            crate::alignment::sequence_logprob(
                reference,
                &sample.chosen_ids,
                sample.prompt_len,
                length_normalize,
            )?,
            crate::alignment::sequence_logprob(
                reference,
                &sample.rejected_ids,
                sample.prompt_len,
                length_normalize,
            )?,
        )
    };
    let loss = dpo_loss_tensor(
        &policy_chosen,
        &policy_rejected,
        reference_chosen,
        reference_rejected,
        beta,
        label_smoothing,
    )?;
    let metrics = crate::alignment::dpo_loss(
        scalar_value(&policy_chosen)?,
        scalar_value(&policy_rejected)?,
        reference_chosen,
        reference_rejected,
        beta,
        label_smoothing,
    );
    Ok(AutogradLoss {
        loss,
        metrics: LossMetrics {
            loss_total: metrics.loss,
            loss_main: metrics.loss,
            loss_spec_avg: 0.0,
            valid_tokens: sample
                .chosen_ids
                .len()
                .saturating_sub(sample.prompt_len.max(1)),
        },
    })
}

/// Computes graph-preserving PPO-style clipped GRPO policy loss.
pub fn grpo_clipped_loss_tensor(
    old_logprobs: &[f64],
    new_logprobs: &[Tensor],
    advantages: &[f64],
    clip_eps: f64,
) -> Result<Tensor> {
    if old_logprobs.len() != new_logprobs.len() || old_logprobs.len() != advantages.len() {
        return Err(ApexError::Shape(
            "old_logprobs, new_logprobs, and advantages lengths must match".to_string(),
        ));
    }
    let Some(first) = new_logprobs.first() else {
        return tensor::scalar(0.0, &candle_core::Device::Cpu);
    };
    let mut total = None;
    for ((old, new), advantage) in old_logprobs.iter().zip(new_logprobs).zip(advantages) {
        let delta = new.broadcast_sub(&tensor::scalar(*old, new.device())?)?;
        let ratio = delta.exp()?;
        let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)?;
        let unclipped_obj = ratio.broadcast_mul(&tensor::scalar(*advantage, new.device())?)?;
        let clipped_obj = clipped.broadcast_mul(&tensor::scalar(*advantage, new.device())?)?;
        let term = unclipped_obj.broadcast_minimum(&clipped_obj)?.neg()?;
        append_loss_term(&mut total, term)?;
    }
    Ok(total
        .unwrap_or(tensor::scalar(0.0, first.device())?)
        .broadcast_div(&tensor::scalar(new_logprobs.len() as f64, first.device())?)?)
}

/// Clips gradients in a Candle `GradStore` and returns the pre-clip norm.
pub fn clip_grad_store(
    grads: &mut GradStore,
    variables: &TrainableVariables,
    max_norm: f64,
) -> Result<AutogradStepStats> {
    if max_norm < 0.0 {
        return Err(ApexError::Config(
            "max_norm must be non-negative".to_string(),
        ));
    }
    let mut sum_sq = 0.0;
    for item in &variables.tensors {
        if let Some(grad) = grads.get(item.var.as_tensor()) {
            let values = grad.flatten_all()?.to_vec1::<f32>()?;
            sum_sq += values
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>();
        }
    }
    let grad_norm = sum_sq.sqrt();
    let clipped = max_norm > 0.0 && grad_norm > max_norm;
    if clipped {
        let scale = max_norm / (grad_norm + 1e-12);
        for item in &variables.tensors {
            if let Some(grad) = grads.get(item.var.as_tensor()) {
                let scaled = grad.broadcast_mul(&tensor::scalar(scale, grad.device())?)?;
                grads.insert(item.var.as_tensor(), scaled);
            }
        }
    }
    Ok(AutogradStepStats { grad_norm, clipped })
}

fn shifted_cross_entropy_tensor(
    logits: &Tensor,
    token_ids: &[Vec<u32>],
    offset: usize,
    token_types: Option<&[Vec<u8>]>,
) -> Result<AutogradLoss> {
    let dims = logits.dims();
    if dims.len() != 3 {
        return Err(ApexError::Shape(format!(
            "shifted_cross_entropy_tensor expects [B,S,V], got {dims:?}"
        )));
    }
    let (b, s, _) = (dims[0], dims[1], dims[2]);
    if offset >= s {
        return Err(ApexError::Shape("offset leaves no labels".to_string()));
    }
    validate_token_rows(token_ids, b, s)?;
    if let Some(types) = token_types {
        if types.len() != b || types.iter().any(|row| row.len() != s) {
            return Err(ApexError::Shape(
                "token_types must match logits batch and sequence length".to_string(),
            ));
        }
    }
    let mut total = None;
    let mut count = 0usize;
    for batch in 0..b {
        for pos in 0..s - offset {
            let label_pos = pos + offset;
            if token_types
                .map(|types| types[batch][label_pos] != 2)
                .unwrap_or(false)
            {
                continue;
            }
            let label = i64::from(token_ids[batch][label_pos]);
            let term = cross_entropy_row(&logits.i((batch, pos, ..))?, label)?;
            append_loss_term(&mut total, term)?;
            count += 1;
        }
    }
    finalize_cross_entropy(total, count, logits.device())
}

fn validate_token_rows(token_ids: &[Vec<u32>], batch: usize, seq_len: usize) -> Result<()> {
    if token_ids.len() != batch || token_ids.iter().any(|row| row.len() != seq_len) {
        return Err(ApexError::Shape(
            "token_ids must match logits batch and sequence length".to_string(),
        ));
    }
    Ok(())
}

fn cross_entropy_row(logits: &Tensor, label: i64) -> Result<Tensor> {
    let idx =
        usize::try_from(label).map_err(|_| ApexError::Data(format!("negative label {label}")))?;
    let vocab = logits.dim(0)?;
    if idx >= vocab {
        return Err(ApexError::Data(format!(
            "label {idx} outside vocab {vocab}"
        )));
    }
    Ok(tensor::log_softmax_last(logits)?.i(idx)?.neg()?)
}

fn append_loss_term(total: &mut Option<Tensor>, term: Tensor) -> Result<()> {
    *total = Some(match total.take() {
        Some(prev) => prev.broadcast_add(&term)?,
        None => term,
    });
    Ok(())
}

fn finalize_cross_entropy(
    total: Option<Tensor>,
    count: usize,
    device: &candle_core::Device,
) -> Result<AutogradLoss> {
    let sum = total.unwrap_or(tensor::scalar(0.0, device)?);
    let loss = if count > 0 {
        sum.broadcast_div(&tensor::scalar(count as f64, device)?)?
    } else {
        sum
    };
    let loss_value = scalar_value(&loss)?;
    Ok(AutogradLoss {
        loss,
        metrics: LossMetrics {
            loss_total: loss_value,
            loss_main: loss_value,
            loss_spec_avg: 0.0,
            valid_tokens: count,
        },
    })
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    x.exp()?
        .broadcast_add(&tensor::scalar(1.0, x.device())?)?
        .log()
        .map_err(Into::into)
}

fn scalar_value(tensor: &Tensor) -> Result<f64> {
    Ok(f64::from(tensor.to_scalar::<f32>()?))
}
