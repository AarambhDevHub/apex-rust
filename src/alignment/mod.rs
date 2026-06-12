use serde::{Deserialize, Serialize};

use crate::data::PreferenceSample;
use crate::error::{ApexError, Result};
use crate::model::ApexModel;
use crate::tensor;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DpoMetrics {
    pub loss: f64,
    pub chosen_logprob: f64,
    pub rejected_logprob: f64,
    pub reward_margin: f64,
    pub accuracy: f64,
}

pub fn sequence_logprob(
    model: &mut ApexModel,
    token_ids: &[u32],
    response_start: usize,
    length_normalize: bool,
) -> Result<f64> {
    if token_ids.len() < 2 {
        return Err(ApexError::Data(
            "sequence_logprob requires at least two tokens".to_string(),
        ));
    }
    let output = model.forward(&[token_ids.to_vec()], None, response_start, None, false)?;
    let logits = output.logits.i((0, .., ..))?;
    let log_probs = tensor::log_softmax_last(&logits)?.to_vec2::<f32>()?;
    let mut sum = 0.0;
    let mut count = 0usize;
    let start = response_start.max(1);
    for (pos, token_id) in token_ids.iter().copied().enumerate().skip(start) {
        let label = token_id as usize;
        if let Some(row) = log_probs.get(pos - 1) {
            if label < row.len() {
                sum += f64::from(row[label]);
                count += 1;
            }
        }
    }
    if length_normalize && count > 0 {
        Ok(sum / count as f64)
    } else {
        Ok(sum)
    }
}

pub fn dpo_loss(
    policy_chosen: f64,
    policy_rejected: f64,
    reference_chosen: f64,
    reference_rejected: f64,
    beta: f64,
    label_smoothing: f64,
) -> DpoMetrics {
    let chosen_reward = policy_chosen - reference_chosen;
    let rejected_reward = policy_rejected - reference_rejected;
    let margin = beta * (chosen_reward - rejected_reward);
    let positive_loss = -log_sigmoid(margin);
    let negative_loss = -log_sigmoid(-margin);
    let loss = (1.0 - label_smoothing) * positive_loss + label_smoothing * negative_loss;
    DpoMetrics {
        loss,
        chosen_logprob: policy_chosen,
        rejected_logprob: policy_rejected,
        reward_margin: chosen_reward - rejected_reward,
        accuracy: if chosen_reward > rejected_reward {
            1.0
        } else {
            0.0
        },
    }
}

pub fn adapter_dpo_step(
    policy: &mut ApexModel,
    reference: Option<&mut ApexModel>,
    sample: &PreferenceSample,
    beta: f64,
    label_smoothing: f64,
    reference_free: bool,
    length_normalize: bool,
) -> Result<DpoMetrics> {
    let policy_chosen = sequence_logprob(
        policy,
        &sample.chosen_ids,
        sample.prompt_len,
        length_normalize,
    )?;
    let policy_rejected = sequence_logprob(
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
                "adapter_dpo_step requires a reference model unless reference_free=true"
                    .to_string(),
            )
        })?;
        (
            sequence_logprob(
                reference,
                &sample.chosen_ids,
                sample.prompt_len,
                length_normalize,
            )?,
            sequence_logprob(
                reference,
                &sample.rejected_ids,
                sample.prompt_len,
                length_normalize,
            )?,
        )
    };
    Ok(dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta,
        label_smoothing,
    ))
}

pub fn grpo_advantages(rewards: &[f64]) -> Vec<f64> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
    let std = var.sqrt().max(1e-8);
    rewards.iter().map(|r| (r - mean) / std).collect()
}

pub fn clipped_policy_loss(
    old_logprobs: &[f64],
    new_logprobs: &[f64],
    advantages: &[f64],
    clip_eps: f64,
) -> Result<f64> {
    if old_logprobs.len() != new_logprobs.len() || old_logprobs.len() != advantages.len() {
        return Err(ApexError::Shape(
            "old_logprobs, new_logprobs, and advantages lengths must match".to_string(),
        ));
    }
    if old_logprobs.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0;
    for ((old, new), adv) in old_logprobs.iter().zip(new_logprobs).zip(advantages) {
        let ratio = (new - old).exp();
        let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
        total += -f64::min(ratio * adv, clipped * adv);
    }
    Ok(total / old_logprobs.len() as f64)
}

fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

use candle_core::IndexOp;
