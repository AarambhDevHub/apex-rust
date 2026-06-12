use candle_core::{Device, Tensor};

use crate::error::Result;

/// Builds an APEX attention mask with bidirectional prefix and causal suffix.
pub fn build_apex_attention_mask(
    prefix_len: usize,
    total_len: usize,
    local_window: usize,
    global: bool,
) -> Vec<Vec<bool>> {
    let mut mask = vec![vec![false; total_len]; total_len];
    for row_mask in mask.iter_mut().take(prefix_len.min(total_len)) {
        for item in row_mask.iter_mut().take(prefix_len.min(total_len)) {
            *item = true;
        }
    }
    for (row, row_mask) in mask.iter_mut().enumerate().skip(prefix_len) {
        if global {
            for item in row_mask.iter_mut().take(row + 1) {
                *item = true;
            }
        } else {
            let start = row.saturating_sub(local_window.saturating_sub(1));
            for item in row_mask.iter_mut().take(row + 1).skip(start) {
                *item = true;
            }
        }
    }
    mask
}

/// Converts a boolean attention mask into an additive `[1,1,Q,K]` tensor.
pub fn additive_mask(
    mask: &[Vec<bool>],
    q_len: usize,
    kv_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(q_len * kv_len);
    for r in 0..q_len {
        for c in 0..kv_len {
            let ok = mask
                .get(r)
                .and_then(|row| row.get(c))
                .copied()
                .unwrap_or(c <= r || q_len == 1);
            values.push(if ok { 0.0_f32 } else { -1.0e9_f32 });
        }
    }
    Ok(Tensor::from_vec(values, (1, 1, q_len, kv_len), device)?)
}
