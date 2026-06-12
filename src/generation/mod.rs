use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use candle_core::IndexOp;

use crate::error::Result;
use crate::model::{cache_len, ApexModel, KvCache};
use crate::tensor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f64,
    pub enable_thinking: bool,
    pub max_thinking_tokens: usize,
    pub thinking_temperature: f64,
    pub output_temperature: f64,
    pub use_speculative: bool,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub thinking_start_id: u32,
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

#[derive(Debug, Clone, Default)]
pub struct GenerationOutput {
    pub token_ids: Vec<u32>,
    pub thinking_tokens: usize,
    pub total_tokens: usize,
    pub finished: bool,
}

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

pub struct ApexGenerator<'a> {
    pub model: &'a mut ApexModel,
    pub config: GenerationConfig,
}

impl<'a> ApexGenerator<'a> {
    pub fn new(model: &'a mut ApexModel, config: GenerationConfig) -> Self {
        Self { model, config }
    }

    pub fn generate(&mut self, input_ids: Vec<u32>, prefix_len: usize) -> Result<GenerationOutput> {
        if self.config.use_speculative {
            return self.generate_with_speculative(input_ids, prefix_len);
        }
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

    pub fn generate_with_speculative(
        &mut self,
        input_ids: Vec<u32>,
        prefix_len: usize,
    ) -> Result<GenerationOutput> {
        let mut out = self.generate(input_ids, prefix_len)?;
        out.token_ids.truncate(self.config.max_new_tokens);
        out.total_tokens = out.token_ids.len();
        Ok(out)
    }

    pub fn prev_len(kv_caches: &[KvCache]) -> usize {
        kv_caches.first().map(cache_len).unwrap_or(0)
    }
}

fn last_logits(logits: &candle_core::Tensor) -> Result<Vec<f32>> {
    let seq = logits.dim(1)?;
    Ok(logits.i((0, seq - 1, ..))?.to_vec1::<f32>()?)
}
