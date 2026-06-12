//! Process Reward Model utilities for step-level reasoning scores.

use candle_core::Tensor;

use crate::error::{ApexError, Result};
use crate::model::ApexModel;
use crate::tensor;
use crate::tokenizer::ApexTokenizer;

/// A backbone plus sigmoid scalar head for scoring reasoning steps.
#[derive(Clone)]
pub struct ProcessRewardModel {
    /// Backbone used as a hidden-state feature extractor.
    pub backbone: ApexModel,
    /// Step-head weight with shape `[d_model]`.
    pub step_weight: Tensor,
    /// Step-head scalar bias.
    pub step_bias: f32,
    /// Whether training code should treat the backbone as frozen.
    pub freeze_backbone: bool,
}

impl ProcessRewardModel {
    /// Creates a PRM with a randomly initialized step head.
    pub fn new(backbone: ApexModel, d_model: usize, freeze_backbone: bool) -> Result<Self> {
        if d_model != backbone.config.model.d_model {
            return Err(ApexError::Shape(format!(
                "PRM d_model {d_model} does not match backbone d_model {}",
                backbone.config.model.d_model
            )));
        }
        let device = backbone.device.clone();
        let step_weight = tensor::randn(&[d_model], 0.0, 0.02, &device)?;
        Ok(Self {
            backbone,
            step_weight,
            step_bias: 0.0,
            freeze_backbone,
        })
    }

    /// Creates a PRM from an explicit step-head tensor.
    pub fn with_head(
        backbone: ApexModel,
        step_weight: Tensor,
        step_bias: f32,
        freeze_backbone: bool,
    ) -> Result<Self> {
        let dims = step_weight.dims();
        if dims != [backbone.config.model.d_model] {
            return Err(ApexError::Shape(format!(
                "step_weight must be [{}], got {dims:?}",
                backbone.config.model.d_model
            )));
        }
        Ok(Self {
            backbone,
            step_weight,
            step_bias,
            freeze_backbone,
        })
    }

    /// Scores the last token of a full prompt-plus-step context.
    pub fn forward(&mut self, input_ids: &[u32]) -> Result<f64> {
        if input_ids.is_empty() {
            return Err(ApexError::Data(
                "ProcessRewardModel::forward requires at least one token".to_string(),
            ));
        }
        let output = self
            .backbone
            .forward(&[input_ids.to_vec()], None, 0, None, true)?;
        let hidden = output
            .hidden_states
            .ok_or_else(|| ApexError::Model("PRM did not return hidden states".into()))?
            .to_vec3::<f32>()?;
        let last_hidden = hidden
            .first()
            .and_then(|row| row.last())
            .ok_or_else(|| ApexError::Shape("PRM hidden state is empty".to_string()))?;
        let weight = self.step_weight.to_vec1::<f32>()?;
        Ok(sigmoid(linear_score(last_hidden, &weight, self.step_bias)))
    }

    /// Scores each reasoning step after appending it to the cumulative context.
    pub fn score_steps(
        &mut self,
        prompt_ids: &[u32],
        step_ids_list: &[Vec<u32>],
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::with_capacity(step_ids_list.len());
        let mut context = prompt_ids.to_vec();
        for step_ids in step_ids_list {
            context.extend(step_ids.iter().copied());
            scores.push(self.forward(&context)?);
        }
        Ok(scores)
    }

    /// Tokenizes a prompt and reasoning steps, then scores each cumulative step.
    pub fn score_steps_from_text(
        &mut self,
        prompt: &str,
        steps: &[String],
        tokenizer: Option<&ApexTokenizer>,
    ) -> Result<Vec<f64>> {
        let tokenizer = tokenizer.ok_or_else(|| {
            ApexError::Tokenizer(
                "score_steps_from_text requires a tokenizer; use score_steps for pretokenized IDs"
                    .to_string(),
            )
        })?;
        let prompt_ids = tokenizer.encode(prompt, false)?;
        self.score_steps_from_text_pretokenized(&prompt_ids, steps, tokenizer)
    }

    /// Scores text steps when the prompt has already been tokenized.
    pub fn score_steps_from_text_pretokenized(
        &mut self,
        prompt_ids: &[u32],
        step_texts: &[String],
        tokenizer: &ApexTokenizer,
    ) -> Result<Vec<f64>> {
        let mut step_ids = Vec::with_capacity(step_texts.len());
        for step in step_texts {
            step_ids.push(tokenizer.encode(&format!("\n{step}"), false)?);
        }
        self.score_steps(prompt_ids, &step_ids)
    }
}

fn linear_score(hidden: &[f32], weight: &[f32], bias: f32) -> f64 {
    hidden
        .iter()
        .zip(weight)
        .map(|(h, w)| f64::from(*h) * f64::from(*w))
        .sum::<f64>()
        + f64::from(bias)
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}
