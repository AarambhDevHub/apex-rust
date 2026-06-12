use candle_core::{Device, Tensor};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};
use crate::tensor;

use super::attention::{AttentionKind, KvCache};
use super::block::TransformerBlock;
use super::ffn::{DenseFfn, FfnKind};
use super::mask::build_apex_attention_mask;
use super::multi_token_head::MultiTokenHead;
use super::norm::RmsNorm;
use super::rope::precompute_rope_cache;
use super::skip_gate::SkipGate;

#[derive(Clone)]
pub struct ModelOutput {
    pub logits: Tensor,
    pub spec_logits: Option<Vec<Tensor>>,
    pub kv_caches: Vec<KvCache>,
    pub hidden_states: Option<Tensor>,
}

#[derive(Clone)]
pub struct ApexModel {
    pub config: ApexConfig,
    pub device: Device,
    pub embedding: Tensor,
    pub embed_scale: f64,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: RmsNorm,
    pub multi_token_head: Option<MultiTokenHead>,
    pub cos_cache: Tensor,
    pub sin_cache: Tensor,
    pub cos_cache_rope: Tensor,
    pub sin_cache_rope: Tensor,
    pub attn_factor: f64,
}

impl ApexModel {
    pub fn new(config: ApexConfig, device: Device) -> Result<Self> {
        config.validate()?;
        let m = &config.model;
        let embedding = tensor::randn(&[m.vocab_size, m.d_model], 0.0, 0.02, &device)?;
        let blocks = (0..m.n_layers)
            .map(|i| TransformerBlock::new(i, &config, &device))
            .collect::<Result<Vec<_>>>()?;
        let (cos_cache, sin_cache, attn_factor) = precompute_rope_cache(
            m.d_head,
            m.max_seq_len,
            m.rope_base,
            m.rope_scaling,
            &device,
        )?;
        let (cos_cache_rope, sin_cache_rope, _) = precompute_rope_cache(
            m.d_head_rope,
            m.max_seq_len,
            m.rope_base,
            m.rope_scaling,
            &device,
        )?;
        Ok(Self {
            embed_scale: (m.d_model as f64).sqrt(),
            final_norm: RmsNorm::new(m.d_model, &device)?,
            multi_token_head: if config.multi_token_head.enabled {
                Some(MultiTokenHead::new(
                    m.d_model,
                    m.vocab_size,
                    config.multi_token_head.n_predict,
                    &device,
                )?)
            } else {
                None
            },
            config,
            device,
            embedding,
            blocks,
            cos_cache,
            sin_cache,
            cos_cache_rope,
            sin_cache_rope,
            attn_factor,
        })
    }

    pub fn forward(
        &mut self,
        token_ids: &[Vec<u32>],
        positions: Option<Vec<usize>>,
        prefix_len: usize,
        kv_caches: Option<&[KvCache]>,
        return_hidden: bool,
    ) -> Result<ModelOutput> {
        if token_ids.is_empty() || token_ids[0].is_empty() {
            return Err(ApexError::Data(
                "token_ids must be non-empty [B,S]".to_string(),
            ));
        }
        let s = token_ids[0].len();
        if !token_ids.iter().all(|row| row.len() == s) {
            return Err(ApexError::Data(
                "all token rows must have equal length".to_string(),
            ));
        }
        let x = embed_tokens(&self.embedding, token_ids)?
            .broadcast_mul(&tensor::scalar(self.embed_scale, &self.device)?)?;
        self.forward_embeddings(&x, positions, prefix_len, kv_caches, return_hidden)
    }

    pub fn forward_embeddings(
        &mut self,
        scaled_embeddings: &Tensor,
        positions: Option<Vec<usize>>,
        prefix_len: usize,
        kv_caches: Option<&[KvCache]>,
        return_hidden: bool,
    ) -> Result<ModelOutput> {
        let dims = scaled_embeddings.dims();
        if dims.len() != 3 {
            return Err(ApexError::Shape(format!(
                "forward_embeddings expects [B,S,D], got {dims:?}"
            )));
        }
        let s = dims[1];
        if dims[2] != self.config.model.d_model {
            return Err(ApexError::Shape(format!(
                "embedding dim {} does not match d_model {}",
                dims[2], self.config.model.d_model
            )));
        }
        let mut x = scaled_embeddings.clone();
        let positions = positions.unwrap_or_else(|| {
            let prev = kv_caches
                .and_then(|c| c.first())
                .map(cache_len)
                .unwrap_or(0);
            (prev..prev + s).collect()
        });
        if positions.len() != s {
            return Err(ApexError::Shape(format!(
                "positions length {} does not match sequence length {s}",
                positions.len()
            )));
        }
        let mut new_caches = Vec::with_capacity(self.blocks.len());
        for (idx, block) in self.blocks.iter_mut().enumerate() {
            let global = block.is_global;
            let mask = build_apex_attention_mask(
                if kv_caches.is_none() { prefix_len } else { 0 },
                s,
                self.config.attention.local_window,
                global,
            );
            let layer_cache = kv_caches.and_then(|c| c.get(idx));
            let (cos, sin) = if global {
                (&self.cos_cache_rope, &self.sin_cache_rope)
            } else {
                (&self.cos_cache, &self.sin_cache)
            };
            let (next_x, cache) =
                block.forward(&x, cos, sin, &positions, Some(&mask), layer_cache)?;
            x = next_x;
            new_caches.push(cache);
        }
        let hidden = self.final_norm.forward(&x)?;
        let logits = tensor::linear(&hidden, &self.embedding, None)?;
        let spec_logits = self
            .multi_token_head
            .as_ref()
            .map(|h| h.forward(&hidden))
            .transpose()?;
        Ok(ModelOutput {
            logits,
            spec_logits,
            kv_caches: new_caches,
            hidden_states: if return_hidden { Some(hidden) } else { None },
        })
    }

    pub fn total_parameters(&self) -> usize {
        self.embedding.elem_count()
            + self.final_norm.parameters()
            + self
                .blocks
                .iter()
                .map(TransformerBlock::parameters)
                .sum::<usize>()
            + self
                .multi_token_head
                .as_ref()
                .map(MultiTokenHead::parameters)
                .unwrap_or(0)
    }

    pub fn trainable_parameters(&self) -> usize {
        if self.config.peft.enabled {
            self.blocks
                .iter()
                .map(TransformerBlock::trainable_parameters)
                .sum::<usize>()
        } else {
            self.total_parameters()
        }
    }

    pub fn active_parameters(&self) -> usize {
        let mut total = self.embedding.elem_count() + self.final_norm.parameters();
        for block in &self.blocks {
            total += block.norm1.parameters() + block.norm2.parameters() + block.attn.parameters();
            match &block.ffn {
                FfnKind::Dense(f) => total += f.parameters(),
                FfnKind::Moe(m) => {
                    total += m
                        .shared_experts
                        .iter()
                        .map(DenseFfn::parameters)
                        .sum::<usize>();
                    total += m
                        .routed_experts
                        .first()
                        .map(DenseFfn::parameters)
                        .unwrap_or(0)
                        * m.n_active;
                    total += m.router.parameters();
                }
            }
            total += block
                .skip_gate
                .as_ref()
                .map(SkipGate::parameters)
                .unwrap_or(0);
        }
        total
    }

    pub fn count_lora_modules(&self) -> usize {
        let mut count = 0;
        for block in &self.blocks {
            count += count_lora_attention(&block.attn);
            count += count_lora_ffn(&block.ffn);
        }
        count
    }

    pub fn count_qlora_modules(&self) -> usize {
        let mut count = 0;
        for block in &self.blocks {
            count += count_quant_attention(&block.attn);
            count += count_quant_ffn(&block.ffn);
        }
        count
    }

    pub fn count_dora_modules(&self) -> usize {
        let mut count = 0;
        for block in &self.blocks {
            count += count_dora_attention(&block.attn);
            count += count_dora_ffn(&block.ffn);
        }
        count
    }

    pub fn merge_and_unload_adapters(&mut self) -> Result<()> {
        for block in &mut self.blocks {
            block.merge_and_unload()?;
        }
        self.config.peft.enabled = false;
        Ok(())
    }

    pub fn named_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        out.push(("embedding.weight".to_string(), self.embedding.clone()));
        out.push((
            "final_norm.weight".to_string(),
            self.final_norm.weight.clone(),
        ));
        for (idx, block) in self.blocks.iter().enumerate() {
            block.named_tensors(&format!("blocks.{idx}"), &mut out)?;
        }
        if let Some(head) = &self.multi_token_head {
            head.named_tensors("multi_token_head", &mut out);
        }
        Ok(out)
    }

    pub fn adapter_tensors(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (idx, block) in self.blocks.iter().enumerate() {
            block.adapter_tensors(&format!("blocks.{idx}"), &mut out);
        }
        out
    }
}

pub fn cache_len(cache: &KvCache) -> usize {
    match cache {
        KvCache::Mla { c_kv, .. } => c_kv.dim(1).unwrap_or(0),
        KvCache::Gqa { k, .. } => k.dim(2).unwrap_or(0),
    }
}

fn embed_tokens(embedding: &Tensor, token_ids: &[Vec<u32>]) -> Result<Tensor> {
    let rows = embedding.to_vec2::<f32>()?;
    let b = token_ids.len();
    let s = token_ids[0].len();
    let d = embedding.dim(1)?;
    let mut values = Vec::with_capacity(b * s * d);
    for row in token_ids {
        for &id in row {
            let idx = id as usize;
            if idx >= rows.len() {
                return Err(ApexError::Data(format!(
                    "token id {id} outside embedding vocab"
                )));
            }
            values.extend_from_slice(&rows[idx]);
        }
    }
    Ok(Tensor::from_vec(values, (b, s, d), embedding.device())?)
}

fn count_lora_attention(attn: &AttentionKind) -> usize {
    match attn {
        AttentionKind::Mla(a) => [
            &a.w_dkv, &a.w_uk, &a.w_uv, &a.w_dq, &a.w_uq, &a.w_kr, &a.w_qr, &a.w_o,
        ]
        .iter()
        .filter(|l| l.is_lora())
        .count(),
        AttentionKind::Gqa(a) => [&a.w_q, &a.w_k, &a.w_v, &a.w_o]
            .iter()
            .filter(|l| l.is_lora())
            .count(),
    }
}

fn count_quant_attention(attn: &AttentionKind) -> usize {
    match attn {
        AttentionKind::Mla(a) => [
            &a.w_dkv, &a.w_uk, &a.w_uv, &a.w_dq, &a.w_uq, &a.w_kr, &a.w_qr, &a.w_o,
        ]
        .iter()
        .filter(|l| l.is_quantized_adapter())
        .count(),
        AttentionKind::Gqa(a) => [&a.w_q, &a.w_k, &a.w_v, &a.w_o]
            .iter()
            .filter(|l| l.is_quantized_adapter())
            .count(),
    }
}

fn count_dora_attention(attn: &AttentionKind) -> usize {
    match attn {
        AttentionKind::Mla(a) => [
            &a.w_dkv, &a.w_uk, &a.w_uv, &a.w_dq, &a.w_uq, &a.w_kr, &a.w_qr, &a.w_o,
        ]
        .iter()
        .filter(|l| l.is_dora())
        .count(),
        AttentionKind::Gqa(a) => [&a.w_q, &a.w_k, &a.w_v, &a.w_o]
            .iter()
            .filter(|l| l.is_dora())
            .count(),
    }
}

fn count_lora_dense(ffn: &DenseFfn) -> usize {
    [&ffn.w_gate, &ffn.w_up, &ffn.w_down]
        .iter()
        .filter(|l| l.is_lora())
        .count()
}

fn count_quant_dense(ffn: &DenseFfn) -> usize {
    [&ffn.w_gate, &ffn.w_up, &ffn.w_down]
        .iter()
        .filter(|l| l.is_quantized_adapter())
        .count()
}

fn count_dora_dense(ffn: &DenseFfn) -> usize {
    [&ffn.w_gate, &ffn.w_up, &ffn.w_down]
        .iter()
        .filter(|l| l.is_dora())
        .count()
}

fn count_lora_ffn(ffn: &FfnKind) -> usize {
    match ffn {
        FfnKind::Dense(f) => count_lora_dense(f),
        FfnKind::Moe(m) => {
            m.shared_experts.iter().map(count_lora_dense).sum::<usize>()
                + m.routed_experts.iter().map(count_lora_dense).sum::<usize>()
                + usize::from(m.router.is_lora())
        }
    }
}

fn count_quant_ffn(ffn: &FfnKind) -> usize {
    match ffn {
        FfnKind::Dense(f) => count_quant_dense(f),
        FfnKind::Moe(m) => {
            m.shared_experts
                .iter()
                .map(count_quant_dense)
                .sum::<usize>()
                + m.routed_experts
                    .iter()
                    .map(count_quant_dense)
                    .sum::<usize>()
                + usize::from(m.router.is_quantized_adapter())
        }
    }
}

fn count_dora_ffn(ffn: &FfnKind) -> usize {
    match ffn {
        FfnKind::Dense(f) => count_dora_dense(f),
        FfnKind::Moe(m) => {
            m.shared_experts.iter().map(count_dora_dense).sum::<usize>()
                + m.routed_experts.iter().map(count_dora_dense).sum::<usize>()
                + usize::from(m.router.is_dora())
        }
    }
}
