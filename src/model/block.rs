use candle_core::{Device, Tensor};

use crate::config::{is_global_layer, is_moe_layer, ApexConfig};
use crate::error::Result;

use super::attention::{AttentionKind, GqaAttention, KvCache, MlaAttention};
use super::ffn::{DenseFfn, FfnKind, MoeFfn};
use super::norm::RmsNorm;
use super::skip_gate::SkipGate;

/// One transformer block with attention, FFN/MoE, RMSNorm, and optional skip gate.
#[derive(Clone)]
pub struct TransformerBlock {
    /// Zero-based layer index.
    pub layer_idx: usize,
    /// Whether this block uses global MLA attention.
    pub is_global: bool,
    /// Whether this block uses a MoE feed-forward layer.
    pub is_moe: bool,
    /// Pre-attention normalization.
    pub norm1: RmsNorm,
    /// Pre-FFN normalization.
    pub norm2: RmsNorm,
    /// Attention implementation selected for this layer.
    pub attn: AttentionKind,
    /// Feed-forward implementation selected for this layer.
    pub ffn: FfnKind,
    /// Optional token-level gate for skipping FFN contribution.
    pub skip_gate: Option<SkipGate>,
}

impl TransformerBlock {
    /// Creates a transformer block according to layer-placement rules.
    pub fn new(layer_idx: usize, cfg: &ApexConfig, device: &Device) -> Result<Self> {
        let is_global = is_global_layer(layer_idx, cfg.attention.global_layer_freq);
        let is_moe = is_moe_layer(layer_idx, &cfg.moe);
        Ok(Self {
            layer_idx,
            is_global,
            is_moe,
            norm1: RmsNorm::new(cfg.model.d_model, device)?,
            norm2: RmsNorm::new(cfg.model.d_model, device)?,
            attn: if is_global {
                AttentionKind::Mla(MlaAttention::new(
                    cfg,
                    &format!("blocks.{layer_idx}.attn"),
                    device,
                )?)
            } else {
                AttentionKind::Gqa(GqaAttention::new(
                    cfg,
                    &format!("blocks.{layer_idx}.attn"),
                    device,
                )?)
            },
            ffn: if is_moe {
                FfnKind::Moe(MoeFfn::new(
                    cfg,
                    &format!("blocks.{layer_idx}.ffn"),
                    device,
                )?)
            } else {
                FfnKind::Dense(DenseFfn::new(
                    cfg,
                    &format!("blocks.{layer_idx}.ffn"),
                    device,
                )?)
            },
            skip_gate: if cfg.skip_gate.enabled {
                Some(SkipGate::new(
                    cfg.model.d_model,
                    cfg.skip_gate.hidden_dim,
                    cfg.skip_gate.threshold,
                    device,
                )?)
            } else {
                None
            },
        })
    }

    /// Runs attention, residual additions, FFN/MoE, and optional skip gating.
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &[usize],
        mask: Option<&[Vec<bool>]>,
        cache: Option<&KvCache>,
    ) -> Result<(Tensor, KvCache)> {
        let (h, new_cache) =
            self.attn
                .forward(&self.norm1.forward(x)?, cos, sin, positions, mask, cache)?;
        let mut y = x.broadcast_add(&h)?;
        let ffn_out = self.ffn.forward(&self.norm2.forward(&y)?)?;
        if let Some(gate) = &self.skip_gate {
            let run_mask = gate.run_mask(&y)?;
            y = y.broadcast_add(&ffn_out.broadcast_mul(&run_mask)?)?;
        } else {
            y = y.broadcast_add(&ffn_out)?;
        }
        Ok((y, new_cache))
    }

    /// Returns total represented parameters.
    pub fn parameters(&self) -> usize {
        self.norm1.parameters()
            + self.norm2.parameters()
            + self.attn.parameters()
            + self.ffn.parameters()
            + self
                .skip_gate
                .as_ref()
                .map(SkipGate::parameters)
                .unwrap_or(0)
    }

    /// Returns trainable parameters under the current adapter policy.
    pub fn trainable_parameters(&self) -> usize {
        self.norm1.parameters()
            + self.norm2.parameters()
            + self.attn.trainable_parameters()
            + self.ffn.trainable_parameters()
            + self
                .skip_gate
                .as_ref()
                .map(SkipGate::parameters)
                .unwrap_or(0)
    }

    /// Merges and unloads adapter layers in attention and FFN modules.
    pub fn merge_and_unload(&mut self) -> Result<()> {
        self.attn.merge_and_unload()?;
        self.ffn.merge_and_unload()
    }

    /// Appends full block tensors to a named checkpoint list.
    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) -> Result<()> {
        out.push((format!("{prefix}.norm1.weight"), self.norm1.weight.clone()));
        out.push((format!("{prefix}.norm2.weight"), self.norm2.weight.clone()));
        self.attn.named_tensors(&format!("{prefix}.attn"), out)?;
        self.ffn.named_tensors(&format!("{prefix}.ffn"), out)?;
        if let Some(gate) = &self.skip_gate {
            gate.fc1
                .named_tensors(&format!("{prefix}.skip_gate.fc1"), out);
            gate.fc2
                .named_tensors(&format!("{prefix}.skip_gate.fc2"), out);
        }
        Ok(())
    }

    /// Appends only adapter tensors in this block.
    pub fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        self.attn.adapter_tensors(&format!("{prefix}.attn"), out);
        self.ffn.adapter_tensors(&format!("{prefix}.ffn"), out);
    }
}
