//! Parameter counting and formatting utilities.

use serde::{Deserialize, Serialize};

use crate::model::ApexModel;

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

/// Coarse component-level parameter breakdown.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterBreakdown {
    /// Token embedding and tied LM-head matrix parameters.
    pub embedding: usize,
    /// Transformer block parameter count.
    pub transformer_blocks: usize,
    /// Final normalization parameters.
    pub final_norm: usize,
    /// Multi-token prediction head parameters.
    pub multi_token_head: usize,
    /// Total represented parameters.
    pub total: usize,
}

impl ParameterBreakdown {
    /// Builds a component-level breakdown from a model instance.
    pub fn from_model(model: &ApexModel) -> Self {
        let embedding = model.embedding.elem_count();
        let transformer_blocks = model.blocks.iter().map(|block| block.parameters()).sum();
        let final_norm = model.final_norm.parameters();
        let multi_token_head = model
            .multi_token_head
            .as_ref()
            .map(|head| head.parameters())
            .unwrap_or(0);
        Self {
            embedding,
            transformer_blocks,
            final_norm,
            multi_token_head,
            total: embedding + transformer_blocks + final_norm + multi_token_head,
        }
    }
}

/// Formats a parameter count with K/M/B suffixes.
pub fn format_parameter_count(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Returns a human-readable parameter summary.
pub fn parameter_summary_text(model: &ApexModel) -> String {
    let report = ParameterReport::from_model(model);
    let breakdown = ParameterBreakdown::from_model(model);
    let active_ratio =
        100.0 * report.active_parameters as f64 / report.total_parameters.max(1) as f64;
    [
        "APEX-1 Parameter Summary".to_string(),
        "========================".to_string(),
        format!(
            "Total parameters:     {} ({})",
            format_parameter_count(report.total_parameters),
            report.total_parameters
        ),
        format!(
            "Trainable parameters: {} ({})",
            format_parameter_count(report.trainable_parameters),
            report.trainable_parameters
        ),
        format!(
            "Active parameters:    {} ({:.1}%)",
            format_parameter_count(report.active_parameters),
            active_ratio
        ),
        String::new(),
        "Component breakdown:".to_string(),
        format!(
            "  embedding           {}",
            format_parameter_count(breakdown.embedding)
        ),
        format!(
            "  transformer_blocks  {}",
            format_parameter_count(breakdown.transformer_blocks)
        ),
        format!(
            "  final_norm          {}",
            format_parameter_count(breakdown.final_norm)
        ),
        format!(
            "  multi_token_head    {}",
            format_parameter_count(breakdown.multi_token_head)
        ),
    ]
    .join("\n")
}
