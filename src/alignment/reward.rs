//! Scalar reward model and Bradley-Terry preference loss.

use candle_core::Tensor;

use crate::error::{ApexError, Result};
use crate::model::ApexModel;
use crate::tensor;

/// A transformer backbone plus a scalar reward head.
#[derive(Clone)]
pub struct RewardModel {
    /// Backbone used as a hidden-state feature extractor.
    pub backbone: ApexModel,
    /// Linear reward-head weight with shape `[d_model]`.
    pub reward_weight: Tensor,
    /// Scalar reward-head bias.
    pub reward_bias: f32,
    /// Whether training code should treat the backbone as frozen.
    pub freeze_backbone: bool,
}

impl RewardModel {
    /// Creates a reward model with a randomly initialized scalar head.
    pub fn new(backbone: ApexModel, d_model: usize, freeze_backbone: bool) -> Result<Self> {
        if d_model != backbone.config.model.d_model {
            return Err(ApexError::Shape(format!(
                "reward head d_model {d_model} does not match backbone d_model {}",
                backbone.config.model.d_model
            )));
        }
        let device = backbone.device.clone();
        let reward_weight = tensor::randn(&[d_model], 0.0, 0.02, &device)?;
        Ok(Self {
            backbone,
            reward_weight,
            reward_bias: 0.0,
            freeze_backbone,
        })
    }

    /// Creates a reward model from an explicit head tensor.
    pub fn with_head(
        backbone: ApexModel,
        reward_weight: Tensor,
        reward_bias: f32,
        freeze_backbone: bool,
    ) -> Result<Self> {
        let dims = reward_weight.dims();
        if dims != [backbone.config.model.d_model] {
            return Err(ApexError::Shape(format!(
                "reward_weight must be [{}], got {dims:?}",
                backbone.config.model.d_model
            )));
        }
        Ok(Self {
            backbone,
            reward_weight,
            reward_bias,
            freeze_backbone,
        })
    }

    /// Scores tokenized sequences, using each sequence's last non-padded token.
    pub fn forward(
        &mut self,
        token_ids: &[Vec<u32>],
        attention_mask: Option<&[Vec<u8>]>,
    ) -> Result<Vec<f64>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let seq_len = token_ids[0].len();
        let last_indices = last_non_padded_indices(attention_mask, token_ids.len(), seq_len)?;
        let output = self.backbone.forward(token_ids, None, 0, None, true)?;
        let hidden = output
            .hidden_states
            .ok_or_else(|| ApexError::Model("reward model did not return hidden states".into()))?
            .to_vec3::<f32>()?;
        let weight = self.reward_weight.to_vec1::<f32>()?;
        let mut rewards = Vec::with_capacity(token_ids.len());
        for (batch_idx, position) in last_indices.into_iter().enumerate() {
            let last_hidden = hidden
                .get(batch_idx)
                .and_then(|row| row.get(position))
                .ok_or_else(|| {
                    ApexError::Shape(format!(
                        "hidden state missing batch {batch_idx} position {position}"
                    ))
                })?;
            rewards.push(linear_score(last_hidden, &weight, self.reward_bias));
        }
        Ok(rewards)
    }

    /// Scores a single tokenized sequence.
    pub fn score_sequence(
        &mut self,
        token_ids: &[u32],
        attention_mask: Option<&[u8]>,
    ) -> Result<f64> {
        let masks = attention_mask.map(|mask| vec![mask.to_vec()]);
        let scores = self.forward(&[token_ids.to_vec()], masks.as_deref())?;
        scores
            .into_iter()
            .next()
            .ok_or_else(|| ApexError::Data("empty reward score output".into()))
    }
}

/// Computes Bradley-Terry preference loss for chosen and rejected rewards.
pub fn reward_model_loss(reward_chosen: &[f64], reward_rejected: &[f64]) -> Result<f64> {
    if reward_chosen.len() != reward_rejected.len() {
        return Err(ApexError::Shape(
            "reward_chosen and reward_rejected lengths must match".to_string(),
        ));
    }
    if reward_chosen.is_empty() {
        return Ok(0.0);
    }
    let loss = reward_chosen
        .iter()
        .zip(reward_rejected)
        .map(|(chosen, rejected)| -log_sigmoid(chosen - rejected))
        .sum::<f64>()
        / reward_chosen.len() as f64;
    Ok(loss)
}

/// Returns last non-zero mask positions, or the final sequence position without a mask.
pub fn last_non_padded_indices(
    attention_mask: Option<&[Vec<u8>]>,
    batch_size: usize,
    seq_len: usize,
) -> Result<Vec<usize>> {
    if seq_len == 0 {
        return Err(ApexError::Data(
            "cannot score reward for empty sequence".to_string(),
        ));
    }
    let Some(mask) = attention_mask else {
        return Ok(vec![seq_len - 1; batch_size]);
    };
    if mask.len() != batch_size {
        return Err(ApexError::Shape(format!(
            "attention_mask batch {} does not match token batch {batch_size}",
            mask.len()
        )));
    }
    mask.iter()
        .map(|row| {
            if row.len() != seq_len {
                return Err(ApexError::Shape(format!(
                    "attention_mask row length {} does not match sequence length {seq_len}",
                    row.len()
                )));
            }
            Ok(row.iter().rposition(|&v| v != 0).unwrap_or(0))
        })
        .collect()
}

fn linear_score(hidden: &[f32], weight: &[f32], bias: f32) -> f64 {
    hidden
        .iter()
        .zip(weight)
        .map(|(h, w)| f64::from(*h) * f64::from(*w))
        .sum::<f64>()
        + f64::from(bias)
}

fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}
