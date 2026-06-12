//! Text architecture diagrams and layer tables.

use crate::config::{is_global_layer, is_moe_layer, ApexConfig};

/// Builds a readable ASCII architecture diagram from a config.
pub fn build_architecture_diagram(config: &ApexConfig, title: &str) -> String {
    let m = &config.model;
    let a = &config.attention;
    let mut lines = Vec::new();
    lines.push(title.to_string());
    lines.push("=".repeat(title.len()));
    lines.push(String::new());
    lines.push(format!(
        "d_model={}, layers={}, vocab={}, max_seq_len={}",
        m.d_model, m.n_layers, m.vocab_size, m.max_seq_len
    ));
    lines.push(format!(
        "global_layer_freq={}, local_window={}",
        a.global_layer_freq, a.local_window
    ));
    lines.push(String::new());

    if config.vision.enabled {
        lines.push("Image Input".to_string());
        lines.push(format!(
            "  `- Vision Encoder: {}, image={}px, patch={}px",
            config.vision.encoder_type, config.vision.image_size, config.vision.patch_size
        ));
        lines.push(format!(
            "      `- Vision Projector: {}, visual_tokens={}",
            config.vision.projector_type, config.vision.n_visual_tokens
        ));
        lines.push("          `- Insert at <|img|> inside token embedding stream".to_string());
        lines.push(String::new());
    }

    lines.push("Text Input".to_string());
    lines.push("  `- Token Embedding x sqrt(d_model)".to_string());
    lines.push("      `- Transformer Blocks".to_string());

    for layer_idx in 0..m.n_layers {
        let attn = if is_global_layer(layer_idx, a.global_layer_freq) {
            "Global MLA"
        } else {
            "Local GQA+SW"
        };
        let ffn = if is_moe_layer(layer_idx, &config.moe) {
            "MoE FFN"
        } else {
            "Dense FFN"
        };
        let skip = if config.skip_gate.enabled {
            "SkipGate"
        } else {
            "NoSkipGate"
        };
        let connector = if layer_idx < m.n_layers - 1 {
            "          |-"
        } else {
            "          `-"
        };
        lines.push(format!(
            "{connector} Layer {layer_idx:02}: {attn} + {ffn} + {skip}"
        ));
    }
    lines.push("              `- Final RMSNorm".to_string());
    lines.push("                  `- Tied LM Head -> logits".to_string());
    if config.multi_token_head.enabled {
        lines.push("                  `- Multi-token speculative heads".to_string());
    }
    lines.join("\n")
}

/// Builds a Markdown table describing each transformer layer.
pub fn build_layer_table(config: &ApexConfig) -> String {
    let mut lines = vec![
        "| Layer | Attention | FFN | Skip Gate |".to_string(),
        "|---:|---|---|---|".to_string(),
    ];
    for layer_idx in 0..config.model.n_layers {
        let attn = if is_global_layer(layer_idx, config.attention.global_layer_freq) {
            "Global MLA"
        } else {
            "Local GQA+SW"
        };
        let ffn = if is_moe_layer(layer_idx, &config.moe) {
            "MoE"
        } else {
            "Dense"
        };
        let skip = if config.skip_gate.enabled {
            "Enabled"
        } else {
            "Disabled"
        };
        lines.push(format!("| {layer_idx} | {attn} | {ffn} | {skip} |"));
    }
    lines.join("\n")
}
