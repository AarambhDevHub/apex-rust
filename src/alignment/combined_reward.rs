//! Weighted reward composition for GRPO-style alignment.

use serde::{Deserialize, Serialize};

use crate::error::{ApexError, Result};

/// Weights applied to outcome, process, and constitutional signals.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RewardWeights {
    /// Weight for final-answer correctness.
    pub outcome: f64,
    /// Weight for reasoning-process quality.
    pub process: f64,
    /// Weight for constitutional safety and policy compliance.
    pub constitutional: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            outcome: 0.5,
            process: 0.2,
            constitutional: 0.3,
        }
    }
}

/// Individual reward signals plus the weighted total.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RewardBreakdown {
    /// Outcome reward used for the final answer.
    pub outcome_reward: f64,
    /// Mean process reward used for reasoning steps.
    pub process_reward: f64,
    /// Constitutional score used for policy compliance.
    pub constitutional_score: f64,
    /// Weighted total reward.
    pub total: f64,
}

/// Computes the weighted combined reward from three scalar signals.
pub fn combined_reward(
    outcome_reward: f64,
    process_reward: f64,
    constitutional_score: f64,
    weights: RewardWeights,
) -> f64 {
    weights.outcome * outcome_reward
        + weights.process * process_reward
        + weights.constitutional * constitutional_score
}

/// Builds a reward breakdown from optional signal values and process step scores.
pub fn score_combined_reward(
    outcome_reward: Option<f64>,
    process_step_scores: Option<&[f64]>,
    constitutional_score: Option<f64>,
    weights: RewardWeights,
) -> Result<RewardBreakdown> {
    validate_weights(weights)?;
    let outcome_reward = outcome_reward.unwrap_or(0.5);
    let process_reward = process_step_scores
        .filter(|scores| !scores.is_empty())
        .map(|scores| scores.iter().sum::<f64>() / scores.len() as f64)
        .unwrap_or(0.5);
    let constitutional_score = constitutional_score.unwrap_or(1.0);
    Ok(RewardBreakdown {
        outcome_reward,
        process_reward,
        constitutional_score,
        total: combined_reward(
            outcome_reward,
            process_reward,
            constitutional_score,
            weights,
        ),
    })
}

/// Extracts line-based thinking steps from `<|thinking|>...<|/thinking|>` text.
pub fn extract_thinking_text(response: &str) -> Vec<String> {
    let start_tag = "<|thinking|>";
    let end_tag = "<|/thinking|>";
    let Some(start_idx) = response.find(start_tag) else {
        return Vec::new();
    };
    let Some(end_idx) = response.find(end_tag) else {
        return Vec::new();
    };
    if end_idx <= start_idx {
        return Vec::new();
    }
    response[start_idx + start_tag.len()..end_idx]
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Validates that reward weights are finite and non-negative.
pub fn validate_weights(weights: RewardWeights) -> Result<()> {
    for (name, value) in [
        ("outcome", weights.outcome),
        ("process", weights.process),
        ("constitutional", weights.constitutional),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(ApexError::Config(format!(
                "reward weight {name} must be finite and non-negative, got {value}"
            )));
        }
    }
    Ok(())
}
