use candle_core::{Device, Tensor};

use crate::config::{is_global_layer, is_moe_layer, ApexConfig};
use crate::error::Result;

use super::attention::{AttentionKind, GqaAttention, KvCache, MlaAttention};
use super::ffn::{DenseFfn, FfnKind, MoeFfn};
use super::norm::RmsNorm;
use super::skip_gate::SkipGate;

#[derive(Clone)]
pub struct TransformerBlock {
    pub layer_idx: usize,
    pub is_global: bool,
    pub is_moe: bool,
    pub norm1: RmsNorm,
    pub norm2: RmsNorm,
    pub attn: AttentionKind,
    pub ffn: FfnKind,
    pub skip_gate: Option<SkipGate>,
}

impl TransformerBlock {
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

    pub fn merge_and_unload(&mut self) -> Result<()> {
        self.attn.merge_and_unload()?;
        self.ffn.merge_and_unload()
    }

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

    pub fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        self.attn.adapter_tensors(&format!("{prefix}.attn"), out);
        self.ffn.adapter_tensors(&format!("{prefix}.ffn"), out);
    }
}
