//! FLOPs estimation helpers.

use serde::{Deserialize, Serialize};

use crate::config::{is_global_layer, is_moe_layer, ApexConfig};

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

/// Component-level FLOPs estimate for one forward pass.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DetailedFlopsEstimate {
    /// Sequence length used by the estimate.
    pub sequence_length: usize,
    /// Embedding lookup FLOPs, kept as zero because it is memory traffic.
    pub embedding: f64,
    /// Attention FLOPs across all layers.
    pub attention_total: f64,
    /// FFN FLOPs across all layers, including SwiGLU elementwise multiply.
    pub ffn_total: f64,
    /// RMSNorm FLOPs.
    pub rmsnorm: f64,
    /// Tied LM-head projection FLOPs.
    pub lm_head: f64,
    /// Number of global MLA layers.
    pub global_layers: usize,
    /// Number of local GQA layers.
    pub local_layers: usize,
    /// Total forward FLOPs.
    pub total: f64,
}

/// Estimates forward and training FLOPs from model dimensions.
pub fn estimate_flops(
    cfg: &ApexConfig,
    batch_size: usize,
    sequence_length: usize,
) -> FlopsEstimate {
    let detailed = estimate_detailed_flops(cfg, sequence_length);
    let forward = detailed.total * batch_size as f64;
    FlopsEstimate {
        sequence_length,
        batch_size,
        forward_flops: forward,
        train_flops: forward * 3.0,
    }
}

/// Estimates component-level FLOPs for a single sequence.
pub fn estimate_detailed_flops(cfg: &ApexConfig, sequence_length: usize) -> DetailedFlopsEstimate {
    let m = &cfg.model;
    let s = sequence_length as f64;
    let mut attention_total = 0.0;
    let mut ffn_total = 0.0;
    let mut global_layers = 0usize;
    let mut local_layers = 0usize;

    for layer_idx in 0..m.n_layers {
        if is_global_layer(layer_idx, cfg.attention.global_layer_freq) {
            global_layers += 1;
            let q_comp = 2.0 * s * m.d_model as f64 * m.d_q_compressed as f64;
            let q_decomp = 2.0 * s * m.d_q_compressed as f64 * (m.n_heads_q * m.d_head) as f64;
            let kv_comp = 2.0 * s * m.d_model as f64 * m.d_kv_compressed as f64;
            let kv_decomp =
                2.0 * s * m.d_kv_compressed as f64 * (m.n_heads_kv * m.d_head * 2) as f64;
            let rope = 2.0
                * s
                * m.d_model as f64
                * (m.n_heads_q + m.n_heads_kv) as f64
                * m.d_head_rope as f64;
            let d_total = (m.d_head + m.d_head_rope) as f64;
            let attn_score = 2.0 * m.n_heads_q as f64 * s * s * d_total;
            let attn_value = 2.0 * m.n_heads_q as f64 * s * s * m.d_head as f64;
            let out_proj = 2.0 * s * (m.n_heads_q * m.d_head) as f64 * m.d_model as f64;
            attention_total +=
                q_comp + q_decomp + kv_comp + kv_decomp + rope + attn_score + attn_value + out_proj;
        } else {
            local_layers += 1;
            let window = sequence_length.min(cfg.attention.local_window) as f64;
            let qkv = 2.0
                * s
                * m.d_model as f64
                * (m.n_heads_q + 2 * m.n_heads_kv) as f64
                * m.d_head as f64;
            let attn_score = 2.0 * m.n_heads_q as f64 * s * window * m.d_head as f64;
            let attn_value = 2.0 * m.n_heads_q as f64 * s * window * m.d_head as f64;
            let out_proj = 2.0 * s * (m.n_heads_q * m.d_head) as f64 * m.d_model as f64;
            attention_total += qkv + attn_score + attn_value + out_proj;
        }

        if is_moe_layer(layer_idx, &cfg.moe) {
            let expert = 2.0 * s * m.d_model as f64 * m.d_ffn as f64 * 3.0 + s * m.d_ffn as f64;
            let shared = cfg.moe.n_shared as f64 * expert;
            let routed = cfg.moe.n_active as f64 * expert;
            let router = 2.0 * s * m.d_model as f64 * cfg.moe.n_experts as f64;
            ffn_total += shared + routed + router;
        } else {
            ffn_total += 2.0 * s * m.d_model as f64 * m.d_ffn as f64 * 3.0 + s * m.d_ffn as f64;
        }
    }

    let rmsnorm = 2.0 * m.n_layers as f64 * 2.0 * s * m.d_model as f64;
    let lm_head = 2.0 * s * m.d_model as f64 * m.vocab_size as f64;
    DetailedFlopsEstimate {
        sequence_length,
        embedding: 0.0,
        attention_total,
        ffn_total,
        rmsnorm,
        lm_head,
        global_layers,
        local_layers,
        total: attention_total + ffn_total + rmsnorm + lm_head,
    }
}

/// Formats FLOPs with human-readable units.
pub fn format_flops(flops: f64) -> String {
    if flops >= 1e15 {
        format!("{:.1} PFLOPs", flops / 1e15)
    } else if flops >= 1e12 {
        format!("{:.1} TFLOPs", flops / 1e12)
    } else if flops >= 1e9 {
        format!("{:.1} GFLOPs", flops / 1e9)
    } else if flops >= 1e6 {
        format!("{:.1} MFLOPs", flops / 1e6)
    } else {
        format!("{flops:.0} FLOPs")
    }
}

/// Returns a human-readable FLOPs summary.
pub fn flops_summary_text(cfg: &ApexConfig, sequence_length: usize) -> String {
    let flops = estimate_detailed_flops(cfg, sequence_length);
    [
        "APEX-1 FLOPs Estimate".to_string(),
        "======================".to_string(),
        format!("Sequence length:  {}", sequence_length),
        format!("Global layers:    {}", flops.global_layers),
        format!("Local layers:     {}", flops.local_layers),
        String::new(),
        format!("Attention:        {}", format_flops(flops.attention_total)),
        format!("FFN:              {}", format_flops(flops.ffn_total)),
        format!("RMSNorm:          {}", format_flops(flops.rmsnorm)),
        format!("LM Head:          {}", format_flops(flops.lm_head)),
        "----------------------".to_string(),
        format!("Total:            {}", format_flops(flops.total)),
    ]
    .join("\n")
}
