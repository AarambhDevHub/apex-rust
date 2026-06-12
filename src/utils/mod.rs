use serde::{Deserialize, Serialize};

use crate::config::ApexConfig;
use crate::model::{ApexModel, AttentionKind, FfnKind};

/// Parameter counts and adapter module counts for one model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterReport {
    /// Total represented parameters.
    pub total_parameters: usize,
    /// Estimated parameters active for one token path.
    pub active_parameters: usize,
    /// Parameters considered trainable by the current configuration.
    pub trainable_parameters: usize,
    /// Number of LoRA-family adapter modules.
    pub lora_modules: usize,
    /// Number of quantized adapter modules.
    pub qlora_modules: usize,
    /// Number of DoRA-family adapter modules.
    pub dora_modules: usize,
}

impl ParameterReport {
    /// Builds a parameter report from a model instance.
    pub fn from_model(model: &ApexModel) -> Self {
        Self {
            total_parameters: model.total_parameters(),
            active_parameters: model.active_parameters(),
            trainable_parameters: model.trainable_parameters(),
            lora_modules: model.count_lora_modules(),
            qlora_modules: model.count_qlora_modules(),
            dora_modules: model.count_dora_modules(),
        }
    }
}

/// Per-layer architecture and parameter summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerReport {
    /// Zero-based layer index.
    pub layer_idx: usize,
    /// Attention implementation name.
    pub attention: String,
    /// Feed-forward implementation name.
    pub ffn: String,
    /// Whether a skip gate is present.
    pub skip_gate: bool,
    /// Layer parameter count.
    pub parameters: usize,
}

/// Full inspection report for model structure and configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInspection {
    /// Human-readable model family label.
    pub model: String,
    /// Configuration used to instantiate the model.
    pub config: ApexConfig,
    /// Parameter summary.
    pub parameters: ParameterReport,
    /// Per-layer summaries.
    pub layers: Vec<LayerReport>,
}

impl ModelInspection {
    /// Builds an inspection report from a model instance.
    pub fn from_model(model: &ApexModel) -> Self {
        let layers = model
            .blocks
            .iter()
            .map(|block| LayerReport {
                layer_idx: block.layer_idx,
                attention: match &block.attn {
                    AttentionKind::Mla(_) => "mla".to_string(),
                    AttentionKind::Gqa(_) => "gqa_sliding_window".to_string(),
                },
                ffn: match &block.ffn {
                    FfnKind::Dense(_) => "dense_swiglu".to_string(),
                    FfnKind::Moe(m) => format!(
                        "moe_{}x{}_shared{}",
                        m.n_active,
                        m.n_experts,
                        m.shared_experts.len()
                    ),
                },
                skip_gate: block.skip_gate.is_some(),
                parameters: block.parameters(),
            })
            .collect();
        Self {
            model: "APEX-1 Rust Candle".to_string(),
            config: model.config.clone(),
            parameters: ParameterReport::from_model(model),
            layers,
        }
    }
}

/// Returns a human-readable architecture summary.
pub fn architecture_text(model: &ApexModel) -> String {
    let mut out = String::new();
    out.push_str("APEX-1 Rust Candle\n");
    out.push_str(&format!(
        "d_model={} layers={} vocab={} max_seq={}\n",
        model.config.model.d_model,
        model.config.model.n_layers,
        model.config.model.vocab_size,
        model.config.model.max_seq_len
    ));
    out.push_str(&format!(
        "attention: MLA every {} layers, GQA local_window={}\n",
        model.config.attention.global_layer_freq, model.config.attention.local_window
    ));
    out.push_str(&format!(
        "moe: enabled={} experts={} active={} shared={}\n",
        model.config.moe.enabled,
        model.config.moe.n_experts,
        model.config.moe.n_active,
        model.config.moe.n_shared
    ));
    for block in &model.blocks {
        let attn = match &block.attn {
            AttentionKind::Mla(_) => "MLA",
            AttentionKind::Gqa(_) => "GQA",
        };
        let ffn = match &block.ffn {
            FfnKind::Dense(_) => "Dense",
            FfnKind::Moe(_) => "MoE",
        };
        out.push_str(&format!(
            "layer {:02}: {:>3} + {:>5} + skip_gate={}\n",
            block.layer_idx,
            attn,
            ffn,
            block.skip_gate.is_some()
        ));
    }
    out
}

/// Rough floating-point operation estimate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct FlopsEstimate {
    /// Sequence length used by the estimate.
    pub sequence_length: usize,
    /// Batch size used by the estimate.
    pub batch_size: usize,
    /// Estimated forward-pass FLOPs.
    pub forward_flops: f64,
    /// Estimated training FLOPs, approximated as three forward passes.
    pub train_flops: f64,
}

/// Estimates forward and training FLOPs from model dimensions.
pub fn estimate_flops(
    cfg: &ApexConfig,
    batch_size: usize,
    sequence_length: usize,
) -> FlopsEstimate {
    let m = &cfg.model;
    let tokens = batch_size as f64 * sequence_length as f64;
    let attn = 4.0 * tokens * m.d_model as f64 * m.d_model as f64;
    let ffn = 6.0 * tokens * m.d_model as f64 * m.d_ffn as f64;
    let moe_multiplier = if cfg.moe.enabled {
        (cfg.moe.n_shared + cfg.moe.n_active) as f64
    } else {
        1.0
    };
    let layer = attn + ffn * moe_multiplier;
    let forward = layer * m.n_layers as f64;
    FlopsEstimate {
        sequence_length,
        batch_size,
        forward_flops: forward,
        train_flops: forward * 3.0,
    }
}
