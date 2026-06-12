use candle_core::{Device, Tensor};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};
use crate::tensor;

use super::linear::LinearLayer;

#[derive(Clone)]
pub struct DenseFfn {
    pub w_gate: LinearLayer,
    pub w_up: LinearLayer,
    pub w_down: LinearLayer,
}

impl DenseFfn {
    pub fn new(cfg: &ApexConfig, prefix: &str, device: &Device) -> Result<Self> {
        let m = &cfg.model;
        Ok(Self {
            w_gate: LinearLayer::new(
                &format!("{prefix}.W_gate"),
                m.d_model,
                m.d_ffn,
                false,
                cfg,
                device,
            )?,
            w_up: LinearLayer::new(
                &format!("{prefix}.W_up"),
                m.d_model,
                m.d_ffn,
                false,
                cfg,
                device,
            )?,
            w_down: LinearLayer::new(
                &format!("{prefix}.W_down"),
                m.d_ffn,
                m.d_model,
                false,
                cfg,
                device,
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = tensor::silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        self.w_down.forward(&gate.broadcast_mul(&up)?)
    }

    pub fn parameters(&self) -> usize {
        self.w_gate.parameters() + self.w_up.parameters() + self.w_down.parameters()
    }

    pub fn trainable_parameters(&self) -> usize {
        self.w_gate.trainable_parameters()
            + self.w_up.trainable_parameters()
            + self.w_down.trainable_parameters()
    }

    pub fn merge_and_unload(&mut self) -> Result<()> {
        self.w_gate.merge_and_unload()?;
        self.w_up.merge_and_unload()?;
        self.w_down.merge_and_unload()
    }

    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) -> Result<()> {
        self.w_gate
            .named_tensors(&format!("{prefix}.W_gate"), out)?;
        self.w_up.named_tensors(&format!("{prefix}.W_up"), out)?;
        self.w_down
            .named_tensors(&format!("{prefix}.W_down"), out)?;
        Ok(())
    }

    pub fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        self.w_gate
            .adapter_tensors(&format!("{prefix}.W_gate"), out);
        self.w_up.adapter_tensors(&format!("{prefix}.W_up"), out);
        self.w_down
            .adapter_tensors(&format!("{prefix}.W_down"), out);
    }
}

#[derive(Clone)]
pub struct MoeFfn {
    pub n_experts: usize,
    pub n_active: usize,
    pub shared_experts: Vec<DenseFfn>,
    pub routed_experts: Vec<DenseFfn>,
    pub router: LinearLayer,
    pub expert_bias: Vec<f32>,
    pub last_top_k_idx: Vec<Vec<usize>>,
}

impl MoeFfn {
    pub fn new(cfg: &ApexConfig, prefix: &str, device: &Device) -> Result<Self> {
        let shared_experts = (0..cfg.moe.n_shared)
            .map(|i| DenseFfn::new(cfg, &format!("{prefix}.shared.{i}"), device))
            .collect::<Result<Vec<_>>>()?;
        let routed_experts = (0..cfg.moe.n_experts)
            .map(|i| DenseFfn::new(cfg, &format!("{prefix}.expert.{i}"), device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            n_experts: cfg.moe.n_experts,
            n_active: cfg.moe.n_active,
            shared_experts,
            routed_experts,
            router: LinearLayer::new(
                &format!("{prefix}.router"),
                cfg.model.d_model,
                cfg.moe.n_experts,
                false,
                cfg,
                device,
            )?,
            expert_bias: vec![0.0; cfg.moe.n_experts],
            last_top_k_idx: Vec::new(),
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(ApexError::Shape(format!(
                "MoE expects [B,S,D], got {dims:?}"
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let mut out = tensor::zeros(dims, x.device())?;
        for expert in &self.shared_experts {
            out = out.broadcast_add(&expert.forward(x)?)?;
        }
        let flat = x.reshape((b * s, d))?;
        let mut router_logits = self.router.forward(&flat)?.to_vec2::<f32>()?;
        self.last_top_k_idx.clear();
        let mut weights_by_expert = vec![vec![0.0_f32; b * s]; self.n_experts];
        for (token_idx, row) in router_logits.iter_mut().enumerate() {
            for (logit, bias) in row.iter_mut().zip(&self.expert_bias) {
                *logit += *bias;
            }
            let top = tensor::top_k_indices(row, self.n_active);
            let max = top
                .iter()
                .map(|&idx| row[idx])
                .fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = top.iter().map(|&idx| (row[idx] - max).exp()).sum();
            for &idx in &top {
                weights_by_expert[idx][token_idx] = (row[idx] - max).exp() / denom.max(1e-8);
            }
            self.last_top_k_idx.push(top);
        }
        for (expert_idx, expert) in self.routed_experts.iter().enumerate() {
            if weights_by_expert[expert_idx].iter().all(|w| *w == 0.0) {
                continue;
            }
            let expert_out = expert.forward(x)?;
            let mask =
                Tensor::from_vec(weights_by_expert[expert_idx].clone(), (b, s, 1), x.device())?;
            out = out.broadcast_add(&expert_out.broadcast_mul(&mask)?)?;
        }
        Ok(out)
    }

    pub fn set_expert_bias(&mut self, bias: &[f32]) {
        self.expert_bias.clear();
        self.expert_bias.extend_from_slice(bias);
    }

    pub fn parameters(&self) -> usize {
        self.shared_experts
            .iter()
            .map(DenseFfn::parameters)
            .sum::<usize>()
            + self
                .routed_experts
                .iter()
                .map(DenseFfn::parameters)
                .sum::<usize>()
            + self.router.parameters()
    }

    pub fn trainable_parameters(&self) -> usize {
        self.shared_experts
            .iter()
            .map(DenseFfn::trainable_parameters)
            .sum::<usize>()
            + self
                .routed_experts
                .iter()
                .map(DenseFfn::trainable_parameters)
                .sum::<usize>()
            + self.router.trainable_parameters()
    }

    pub fn merge_and_unload(&mut self) -> Result<()> {
        for e in &mut self.shared_experts {
            e.merge_and_unload()?;
        }
        for e in &mut self.routed_experts {
            e.merge_and_unload()?;
        }
        self.router.merge_and_unload()
    }

    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) -> Result<()> {
        for (idx, expert) in self.shared_experts.iter().enumerate() {
            expert.named_tensors(&format!("{prefix}.shared.{idx}"), out)?;
        }
        for (idx, expert) in self.routed_experts.iter().enumerate() {
            expert.named_tensors(&format!("{prefix}.expert.{idx}"), out)?;
        }
        self.router
            .named_tensors(&format!("{prefix}.router"), out)?;
        Ok(())
    }

    pub fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        for (idx, expert) in self.shared_experts.iter().enumerate() {
            expert.adapter_tensors(&format!("{prefix}.shared.{idx}"), out);
        }
        for (idx, expert) in self.routed_experts.iter().enumerate() {
            expert.adapter_tensors(&format!("{prefix}.expert.{idx}"), out);
        }
        self.router
            .adapter_tensors(&format!("{prefix}.router"), out);
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum FfnKind {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

impl FfnKind {
    pub(crate) fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(f) => f.forward(x),
            Self::Moe(f) => f.forward(x),
        }
    }

    pub(crate) fn parameters(&self) -> usize {
        match self {
            Self::Dense(f) => f.parameters(),
            Self::Moe(f) => f.parameters(),
        }
    }

    pub(crate) fn trainable_parameters(&self) -> usize {
        match self {
            Self::Dense(f) => f.trainable_parameters(),
            Self::Moe(f) => f.trainable_parameters(),
        }
    }

    pub(crate) fn merge_and_unload(&mut self) -> Result<()> {
        match self {
            Self::Dense(f) => f.merge_and_unload(),
            Self::Moe(f) => f.merge_and_unload(),
        }
    }

    pub(crate) fn named_tensors(
        &self,
        prefix: &str,
        out: &mut Vec<(String, Tensor)>,
    ) -> Result<()> {
        match self {
            Self::Dense(f) => f.named_tensors(prefix, out),
            Self::Moe(f) => f.named_tensors(prefix, out),
        }
    }

    pub(crate) fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        match self {
            Self::Dense(f) => f.adapter_tensors(prefix, out),
            Self::Moe(f) => f.adapter_tensors(prefix, out),
        }
    }
}
