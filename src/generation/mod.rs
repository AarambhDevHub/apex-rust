//! Autoregressive generation, sampling filters, and speculative-token helpers.

use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use candle_core::IndexOp;

use crate::error::Result;
use crate::model::{cache_len, ApexModel, KvCache};
use crate::tensor;

/// Sampling and generation options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to produce.
    pub max_new_tokens: usize,
    /// Default sampling temperature.
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    pub top_p: f64,
    /// Top-k sampling cutoff; zero disables top-k.
    pub top_k: usize,
    /// Penalty applied to already generated token logits.
    pub repetition_penalty: f64,
    /// Enables thinking-token temperature switching.
    pub enable_thinking: bool,
    /// Maximum thinking-token count before an end-thinking token is forced.
    pub max_thinking_tokens: usize,
    /// Temperature used inside the thinking span.
    pub thinking_temperature: f64,
    /// Temperature used after the thinking span.
    pub output_temperature: f64,
    /// Uses the speculative-generation entry point.
    pub use_speculative: bool,
    /// Token ID that stops generation.
    pub eos_token_id: u32,
    /// Padding token ID carried for tokenizer compatibility.
    pub pad_token_id: u32,
    /// Thinking-start token ID.
    pub thinking_start_id: u32,
    /// Thinking-end token ID.
    pub thinking_end_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            enable_thinking: false,
            max_thinking_tokens: 1024,
            thinking_temperature: 0.6,
            output_temperature: 0.3,
            use_speculative: false,
            eos_token_id: 2,
            pad_token_id: 0,
            thinking_start_id: 6,
            thinking_end_id: 7,
        }
    }
}

/// Generated token IDs and simple stopping statistics.
#[derive(Debug, Clone, Default)]
pub struct GenerationOutput {
    /// Newly generated token IDs.
    pub token_ids: Vec<u32>,
    /// Number of tokens produced while in thinking mode.
    pub thinking_tokens: usize,
    /// Total number of generated tokens.
    pub total_tokens: usize,
    /// Whether generation stopped on EOS.
    pub finished: bool,
}

/// Applies repetition penalty in place to token logits.
pub fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f64) {
    if (penalty - 1.0).abs() < f64::EPSILON {
        return;
    }
    for id in generated {
        let idx = *id as usize;
        if let Some(v) = logits.get_mut(idx) {
            if *v > 0.0 {
                *v /= penalty as f32;
            } else {
                *v *= penalty as f32;
            }
        }
    }
}

/// Masks logits outside the largest `top_k` values.
pub fn apply_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let keep = tensor::top_k_indices(logits, top_k);
    let mut allowed = vec![false; logits.len()];
    for idx in keep {
        allowed[idx] = true;
    }
    for (idx, v) in logits.iter_mut().enumerate() {
        if !allowed[idx] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Masks logits outside the smallest set whose probabilities exceed `top_p`.
pub fn apply_top_p(logits: &mut [f32], top_p: f64) {
    if top_p >= 1.0 {
        return;
    }
    let probs = tensor::safe_probs_from_logits(logits);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let mut cumulative = 0.0_f32;
    let mut keep = vec![false; logits.len()];
    for (rank, idx) in order.into_iter().enumerate() {
        cumulative += probs[idx];
        keep[idx] = true;
        if cumulative > top_p as f32 && rank > 0 {
            break;
        }
    }
    for (idx, v) in logits.iter_mut().enumerate() {
        if !keep[idx] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Samples one token from logits after repetition, top-k, top-p, and temperature.
pub fn sample_next_token(
    raw_logits: &[f32],
    cfg: &GenerationConfig,
    temperature: f64,
    generated: &[u32],
) -> u32 {
    let mut logits = raw_logits.to_vec();
    apply_repetition_penalty(&mut logits, generated, cfg.repetition_penalty);
    if temperature <= 0.0 {
        return tensor::top_k_indices(&logits, 1)
            .into_iter()
            .next()
            .unwrap_or(0) as u32;
    }
    for v in &mut logits {
        *v /= temperature as f32;
    }
    apply_top_k(&mut logits, cfg.top_k);
    apply_top_p(&mut logits, cfg.top_p);
    let probs = tensor::safe_probs_from_logits(&logits);
    let mut rng = rand::rng();
    match WeightedIndex::new(&probs) {
        Ok(dist) => dist.sample(&mut rng) as u32,
        Err(_) => tensor::top_k_indices(&logits, 1)
            .into_iter()
            .next()
            .unwrap_or(0) as u32,
    }
}

/// Converts logits into filtered probabilities using the same sampling controls.
pub fn filtered_probabilities(
    raw_logits: &[f32],
    cfg: &GenerationConfig,
    temperature: f64,
    generated: &[u32],
) -> Vec<f32> {
    if raw_logits.is_empty() {
        return Vec::new();
    }
    let mut logits = raw_logits.to_vec();
    apply_repetition_penalty(&mut logits, generated, cfg.repetition_penalty);
    if temperature <= 0.0 {
        let mut probs = vec![0.0; logits.len()];
        if let Some(idx) = tensor::top_k_indices(&logits, 1).into_iter().next() {
            probs[idx] = 1.0;
        }
        return probs;
    }
    for v in &mut logits {
        *v /= temperature as f32;
    }
    apply_top_k(&mut logits, cfg.top_k);
    apply_top_p(&mut logits, cfg.top_p);
    tensor::safe_probs_from_logits(&logits)
}

/// Stateful generator borrowing a mutable model during decoding.
pub struct ApexGenerator<'a> {
    /// Model used for forward passes and KV-cache updates.
    pub model: &'a mut ApexModel,
    /// Sampling and token-control settings.
    pub config: GenerationConfig,
}

impl<'a> ApexGenerator<'a> {
    /// Creates a generator for the provided model and config.
    pub fn new(model: &'a mut ApexModel, config: GenerationConfig) -> Self {
        Self { model, config }
    }

    /// Generates tokens from an input prompt with incremental KV-cache reuse.
    pub fn generate(&mut self, input_ids: Vec<u32>, prefix_len: usize) -> Result<GenerationOutput> {
        if self.config.use_speculative {
            return self.generate_with_speculative(input_ids, prefix_len);
        }
        self.generate_standard(input_ids, prefix_len)
    }

    /// Generates tokens with the standard one-token decode loop.
    fn generate_standard(
        &mut self,
        input_ids: Vec<u32>,
        prefix_len: usize,
    ) -> Result<GenerationOutput> {
        let cfg = self.config.clone();
        let mut generated = Vec::new();
        let mut thinking_tokens = 0usize;
        let mut in_thinking = false;
        let mut current_temp = cfg.temperature;
        let mut output = self
            .model
            .forward(&[input_ids], None, prefix_len, None, false)?;
        let mut next_logits = last_logits(&output.logits)?;
        let mut kv_caches = Some(output.kv_caches);
        for _step in 0..cfg.max_new_tokens {
            let mut token = sample_next_token(&next_logits, &cfg, current_temp, &generated);
            generated.push(token);
            if token == cfg.eos_token_id {
                break;
            }
            if cfg.enable_thinking {
                if token == cfg.thinking_start_id {
                    in_thinking = true;
                    current_temp = cfg.thinking_temperature;
                } else if in_thinking {
                    thinking_tokens += 1;
                    if thinking_tokens >= cfg.max_thinking_tokens {
                        token = cfg.thinking_end_id;
                        generated.push(token);
                        in_thinking = false;
                        current_temp = cfg.output_temperature;
                    }
                }
                if token == cfg.thinking_end_id && in_thinking {
                    in_thinking = false;
                    current_temp = cfg.output_temperature;
                }
            }
            output = self
                .model
                .forward(&[vec![token]], None, 0, kv_caches.as_deref(), false)?;
            next_logits = last_logits(&output.logits)?;
            kv_caches = Some(output.kv_caches);
        }
        Ok(GenerationOutput {
            finished: generated.last().copied() == Some(cfg.eos_token_id),
            total_tokens: generated.len(),
            token_ids: generated,
            thinking_tokens,
        })
    }

    /// Runs the speculative-generation entry point.
    pub fn generate_with_speculative(
        &mut self,
        input_ids: Vec<u32>,
        prefix_len: usize,
    ) -> Result<GenerationOutput> {
        let cfg = self.config.clone();
        let draft_head_count = self
            .model
            .multi_token_head
            .as_ref()
            .map(|head| head.heads.len())
            .unwrap_or(0);
        if cfg.enable_thinking || draft_head_count == 0 {
            return self.generate_standard(input_ids, prefix_len);
        }

        let mut context_ids = input_ids.clone();
        let mut generated = Vec::new();
        let mut output = self
            .model
            .forward(&[input_ids], None, prefix_len, None, true)?;
        let mut next_logits = last_logits(&output.logits)?;
        let mut hidden_states = output.hidden_states.clone();
        let mut kv_caches = Some(output.kv_caches);

        while generated.len() < cfg.max_new_tokens {
            let main_token = sample_next_token(&next_logits, &cfg, cfg.temperature, &generated);
            push_generated(&mut generated, &mut context_ids, main_token);
            if main_token == cfg.eos_token_id || generated.len() >= cfg.max_new_tokens {
                break;
            }

            let draft_ids = match (&self.model.multi_token_head, hidden_states.as_ref()) {
                (Some(head), Some(hidden)) => head.draft_tokens(hidden)?,
                _ => Vec::new(),
            };
            if draft_ids.is_empty() {
                output =
                    self.model
                        .forward(&[vec![main_token]], None, 0, kv_caches.as_deref(), true)?;
                next_logits = last_logits(&output.logits)?;
                hidden_states = output.hidden_states.clone();
                kv_caches = Some(output.kv_caches);
                continue;
            }

            let remaining = cfg.max_new_tokens.saturating_sub(generated.len());
            let draft_ids = draft_ids.into_iter().take(remaining).collect::<Vec<_>>();
            let mut verify_input = Vec::with_capacity(1 + draft_ids.len());
            verify_input.push(main_token);
            verify_input.extend(draft_ids.iter().copied());

            output = self
                .model
                .forward(&[verify_input], None, 0, kv_caches.as_deref(), true)?;
            let verify_logits = output.logits.to_vec3::<f32>()?;
            kv_caches = Some(output.kv_caches);
            hidden_states = output.hidden_states.clone();

            let Some(token_logits) = verify_logits.first() else {
                output =
                    self.model
                        .forward(&[context_ids.clone()], None, prefix_len, None, true)?;
                next_logits = last_logits(&output.logits)?;
                hidden_states = output.hidden_states.clone();
                kv_caches = Some(output.kv_caches);
                continue;
            };
            let vocab_len = token_logits.first().map(Vec::len).unwrap_or(0).max(1);
            let mut accepted = 0usize;
            let mut rejected = false;

            for (idx, draft_id) in draft_ids.iter().copied().enumerate() {
                if generated.len() >= cfg.max_new_tokens {
                    break;
                }
                let Some(logits) = token_logits.get(idx) else {
                    break;
                };
                let target_prob = token_probability(logits, draft_id, &cfg, &generated);
                let draft_prob = 1.0 / vocab_len as f64;
                let accept_prob = (target_prob / draft_prob.max(1e-10)).clamp(0.0, 1.0);
                let accept = if cfg.temperature <= 0.0 {
                    sample_next_token(logits, &cfg, 0.0, &generated) == draft_id
                } else {
                    rand::rng().random::<f64>() < accept_prob
                };

                if accept {
                    push_generated(&mut generated, &mut context_ids, draft_id);
                    accepted += 1;
                    if draft_id == cfg.eos_token_id {
                        break;
                    }
                } else {
                    let resampled = sample_next_token(logits, &cfg, cfg.temperature, &generated);
                    push_generated(&mut generated, &mut context_ids, resampled);
                    rejected = true;
                    output =
                        self.model
                            .forward(&[context_ids.clone()], None, prefix_len, None, true)?;
                    next_logits = last_logits(&output.logits)?;
                    hidden_states = output.hidden_states.clone();
                    kv_caches = Some(output.kv_caches);
                    break;
                }
            }

            if generated.last().copied() == Some(cfg.eos_token_id)
                || generated.len() >= cfg.max_new_tokens
            {
                break;
            }
            if rejected {
                continue;
            }
            let next_idx = accepted.min(token_logits.len().saturating_sub(1));
            if let Some(logits) = token_logits.get(next_idx) {
                next_logits = logits.clone();
            } else {
                output =
                    self.model
                        .forward(&[context_ids.clone()], None, prefix_len, None, true)?;
                next_logits = last_logits(&output.logits)?;
                hidden_states = output.hidden_states.clone();
                kv_caches = Some(output.kv_caches);
            }
        }

        Ok(GenerationOutput {
            finished: generated.last().copied() == Some(cfg.eos_token_id),
            total_tokens: generated.len(),
            token_ids: generated,
            thinking_tokens: 0,
        })
    }

    /// Returns the previous sequence length represented by a KV-cache set.
    pub fn prev_len(kv_caches: &[KvCache]) -> usize {
        kv_caches.first().map(cache_len).unwrap_or(0)
    }
}

fn last_logits(logits: &candle_core::Tensor) -> Result<Vec<f32>> {
    let seq = logits.dim(1)?;
    Ok(logits.i((0, seq - 1, ..))?.to_vec1::<f32>()?)
}

fn token_probability(
    logits: &[f32],
    token_id: u32,
    cfg: &GenerationConfig,
    generated: &[u32],
) -> f64 {
    filtered_probabilities(logits, cfg, cfg.temperature, generated)
        .get(token_id as usize)
        .copied()
        .unwrap_or(0.0) as f64
}

fn push_generated(generated: &mut Vec<u32>, context_ids: &mut Vec<u32>, token_id: u32) {
    generated.push(token_id);
    context_ids.push(token_id);
}
