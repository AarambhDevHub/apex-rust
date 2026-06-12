//! Model inspection reports and architecture text.

use serde::{Deserialize, Serialize};

use crate::config::ApexConfig;
use crate::model::{ApexModel, AttentionKind, FfnKind};

use super::params::{format_parameter_count, ParameterReport};

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
    /// Number of global MLA layers.
    pub global_layers: usize,
    /// Number of local GQA sliding-window layers.
    pub local_layers: usize,
    /// Number of MoE FFN layers.
    pub moe_layers: usize,
    /// Number of dense FFN layers.
    pub dense_layers: usize,
    /// Number of layers containing skip gates.
    pub skip_gate_layers: usize,
    /// Whether the config enables vision.
    pub vision_enabled: bool,
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
            .collect::<Vec<_>>();
        let global_layers = layers
            .iter()
            .filter(|layer| layer.attention == "mla")
            .count();
        let moe_layers = layers
            .iter()
            .filter(|layer| layer.ffn.starts_with("moe"))
            .count();
        let skip_gate_layers = layers.iter().filter(|layer| layer.skip_gate).count();
        Self {
            model: "APEX-1 Rust Candle".to_string(),
            config: model.config.clone(),
            parameters: ParameterReport::from_model(model),
            global_layers,
            local_layers: layers.len().saturating_sub(global_layers),
            moe_layers,
            dense_layers: layers.len().saturating_sub(moe_layers),
            skip_gate_layers,
            vision_enabled: model.config.vision.enabled,
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

/// Formats a model inspection as Markdown.
pub fn inspection_markdown(report: &ModelInspection, show_layers: bool) -> String {
    let mut lines = vec![
        format!("# {} Inspection", report.model),
        String::new(),
        format!(
            "Total parameters:      {}",
            format_parameter_count(report.parameters.total_parameters)
        ),
        format!(
            "Trainable parameters:  {}",
            format_parameter_count(report.parameters.trainable_parameters)
        ),
        format!(
            "Active parameters:     {}",
            format_parameter_count(report.parameters.active_parameters)
        ),
        format!("Vocabulary size:       {}", report.config.model.vocab_size),
        format!("Hidden size d_model:   {}", report.config.model.d_model),
        format!("Layers:                {}", report.layers.len()),
        format!("Global MLA layers:     {}", report.global_layers),
        format!("Local GQA+SW layers:   {}", report.local_layers),
        format!("MoE layers:            {}", report.moe_layers),
        format!("Dense FFN layers:      {}", report.dense_layers),
        format!("Skip-gate layers:      {}", report.skip_gate_layers),
        format!("Vision enabled:        {}", report.vision_enabled),
    ];
    if show_layers {
        lines.extend([
            String::new(),
            "## Layer Map".to_string(),
            String::new(),
            "| Layer | Attention | FFN | Skip Gate | Params |".to_string(),
            "|---:|---|---|---|---:|".to_string(),
        ]);
        for layer in &report.layers {
            lines.push(format!(
                "| {} | {} | {} | {} | {} |",
                layer.layer_idx,
                layer.attention,
                layer.ffn,
                layer.skip_gate,
                format_parameter_count(layer.parameters)
            ));
        }
    }
    lines.join("\n")
}
