//! Group Relative Policy Optimization reward normalization and policy loss.

use crate::error::{ApexError, Result};

/// Normalizes reward values into zero-mean, unit-variance GRPO advantages.
pub fn grpo_advantages(rewards: &[f64]) -> Vec<f64> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
    let std = var.sqrt().max(1e-8);
    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// Computes PPO-style clipped policy loss from old/new log-probs and advantages.
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
