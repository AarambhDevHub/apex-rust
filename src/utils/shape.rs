//! Forward-pass shape verification helpers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::config::is_global_layer;
use crate::error::Result;
use crate::model::{ApexModel, KvCache};

/// Result of shape verification checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShapeVerificationReport {
    /// Per-check pass/fail flags.
    pub checks: BTreeMap<String, bool>,
    /// Number of checks that passed.
    pub passed: usize,
    /// Total number of checks.
    pub total: usize,
}

impl ShapeVerificationReport {
    /// Returns true when every shape check passed.
    pub fn all_passed(&self) -> bool {
        self.passed == self.total
    }
}

/// Runs a deterministic forward pass and verifies model tensor shapes.
pub fn verify_shapes(
    model: &mut ApexModel,
    batch_size: usize,
    sequence_length: usize,
) -> Result<ShapeVerificationReport> {
    let cfg = model.config.clone();
    let tokens = synthetic_tokens(batch_size, sequence_length, cfg.model.vocab_size);
    let output = model.forward(&tokens, None, sequence_length / 2, None, true)?;
    let mut checks = BTreeMap::new();

    checks.insert(
        "logits_shape".to_string(),
        output.logits.dims() == [batch_size, sequence_length, cfg.model.vocab_size],
    );
    if let Some(hidden) = output.hidden_states.as_ref() {
        checks.insert(
            "hidden_shape".to_string(),
            hidden.dims() == [batch_size, sequence_length, cfg.model.d_model],
        );
    } else {
        checks.insert("hidden_shape".to_string(), false);
    }
    if let Some(spec_logits) = output.spec_logits.as_ref() {
        for (idx, spec) in spec_logits.iter().enumerate() {
            checks.insert(
                format!("spec_logits_{idx}_shape"),
                spec.dims() == [batch_size, sequence_length, cfg.model.vocab_size],
            );
        }
    }
    checks.insert(
        "n_kv_caches".to_string(),
        output.kv_caches.len() == cfg.model.n_layers,
    );

    for (layer_idx, cache) in output.kv_caches.iter().enumerate() {
        let global = is_global_layer(layer_idx, cfg.attention.global_layer_freq);
        match (global, cache) {
            (true, KvCache::Mla { c_kv, k_rope }) => {
                checks.insert(
                    format!("layer_{layer_idx}_mla_c_kv"),
                    c_kv.dims() == [batch_size, sequence_length, cfg.model.d_kv_compressed],
                );
                checks.insert(
                    format!("layer_{layer_idx}_mla_k_rope"),
                    k_rope.dims()
                        == [
                            batch_size,
                            cfg.model.n_heads_kv,
                            sequence_length,
                            cfg.model.d_head_rope,
                        ],
                );
            }
            (false, KvCache::Gqa { k, v }) => {
                let expected_cache_len = sequence_length.min(cfg.attention.local_window);
                checks.insert(
                    format!("layer_{layer_idx}_gqa_k"),
                    k.dims()
                        == [
                            batch_size,
                            cfg.model.n_heads_kv,
                            expected_cache_len,
                            cfg.model.d_head,
                        ],
                );
                checks.insert(
                    format!("layer_{layer_idx}_gqa_v"),
                    v.dims()
                        == [
                            batch_size,
                            cfg.model.n_heads_kv,
                            expected_cache_len,
                            cfg.model.d_head,
                        ],
                );
            }
            (true, _) => {
                checks.insert(format!("layer_{layer_idx}_mla_cache_kind"), false);
            }
            (false, _) => {
                checks.insert(format!("layer_{layer_idx}_gqa_cache_kind"), false);
            }
        }
    }
    checks.insert(
        "weight_tying_shape".to_string(),
        model.embedding.dims() == [cfg.model.vocab_size, cfg.model.d_model],
    );

    let passed = checks.values().filter(|&&ok| ok).count();
    let total = checks.len();
    Ok(ShapeVerificationReport {
        checks,
        passed,
        total,
    })
}

fn synthetic_tokens(batch_size: usize, sequence_length: usize, vocab_size: usize) -> Vec<Vec<u32>> {
    (0..batch_size)
        .map(|batch| {
            (0..sequence_length)
                .map(|pos| ((batch * sequence_length + pos) % vocab_size.max(1)) as u32)
                .collect()
        })
        .collect()
}
