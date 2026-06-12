use serde::{Deserialize, Serialize};

use crate::error::{ApexError, Result};
use crate::tensor;

/// Scalar loss values returned by training loss helpers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossMetrics {
    /// Total objective including optional auxiliary terms.
    pub loss_total: f64,
    /// Main next-token cross-entropy loss.
    pub loss_main: f64,
    /// Average speculative-head loss.
    pub loss_spec_avg: f64,
    /// Number of non-ignored target tokens.
    pub valid_tokens: usize,
}

/// Computes next-token pretraining loss plus optional speculative-head loss.
pub fn compute_pretrain_loss(
    logits: &candle_core::Tensor,
    spec_logits: Option<&[candle_core::Tensor]>,
    token_ids: &[Vec<u32>],
    lambda_spec: f64,
) -> Result<LossMetrics> {
    let (flat_logits, labels) = shifted_logits_and_labels(logits, token_ids, 1, None)?;
    let (loss_main_sum, count) = tensor::cross_entropy_sum(&flat_logits, &labels, -100)?;
    let loss_main = loss_main_sum / count.max(1) as f64;
    let mut spec_total = 0.0;
    let mut spec_count = 0usize;
    if let Some(specs) = spec_logits {
        for (offset, spec) in specs.iter().enumerate() {
            let k = offset + 1;
            if token_ids[0].len().saturating_sub(k) < 1 {
                break;
            }
            let (flat, labels) = shifted_logits_and_labels(spec, token_ids, k, None)?;
            let (sum, n) = tensor::cross_entropy_sum(&flat, &labels, -100)?;
            if n > 0 {
                spec_total += sum / n as f64;
                spec_count += 1;
            }
        }
    }
    let loss_spec_avg = if spec_count > 0 {
        spec_total / spec_count as f64
    } else {
        0.0
    };
    Ok(LossMetrics {
        loss_total: loss_main + lambda_spec * loss_spec_avg,
        loss_main,
        loss_spec_avg,
        valid_tokens: count,
    })
}

/// Computes supervised fine-tuning loss over assistant tokens only.
pub fn compute_sft_loss(
    logits: &candle_core::Tensor,
    token_ids: &[Vec<u32>],
    token_types: &[Vec<u8>],
) -> Result<LossMetrics> {
    let (flat, labels) = shifted_logits_and_labels(logits, token_ids, 1, Some(token_types))?;
    let (sum, count) = tensor::cross_entropy_sum(&flat, &labels, -100)?;
    let loss = sum / count.max(1) as f64;
    Ok(LossMetrics {
        loss_total: loss,
        loss_main: loss,
        loss_spec_avg: 0.0,
        valid_tokens: count,
    })
}

/// Computes multimodal SFT loss from already-expanded label rows.
pub fn compute_vision_sft_loss(
    logits: &candle_core::Tensor,
    labels: &[Vec<i64>],
    ignore_index: i64,
) -> Result<LossMetrics> {
    let dims = logits.dims();
    let (b, s, v) = (dims[0], dims[1], dims[2]);
    let mut flat_rows = Vec::with_capacity(b * (s - 1) * v);
    let logits_vec = logits.to_vec3::<f32>()?;
    let mut shifted_labels = Vec::with_capacity(b * (s - 1));
    for batch in 0..b {
        for pos in 0..s - 1 {
            flat_rows.extend_from_slice(&logits_vec[batch][pos]);
            shifted_labels.push(labels[batch][pos + 1]);
        }
    }
    let flat = candle_core::Tensor::from_vec(flat_rows, (b * (s - 1), v), logits.device())?;
    let (sum, count) = tensor::cross_entropy_sum(&flat, &shifted_labels, ignore_index)?;
    let loss = sum / count.max(1) as f64;
    Ok(LossMetrics {
        loss_total: loss,
        loss_main: loss,
        loss_spec_avg: 0.0,
        valid_tokens: count,
    })
}

/// Expands labels to match inserted visual tokens, marking visual labels ignored.
pub fn expand_labels_for_visual_tokens(
    token_ids: &[Vec<u32>],
    labels: &[Vec<i64>],
    image_token_id: u32,
    n_visual_tokens: usize,
    ignore_index: i64,
) -> Result<Vec<Vec<i64>>> {
    if token_ids.len() != labels.len() {
        return Err(ApexError::Data(
            "token_ids and labels batch mismatch".to_string(),
        ));
    }
    let mut rows = Vec::with_capacity(token_ids.len());
    for (ids, labs) in token_ids.iter().zip(labels) {
        let idx = ids
            .iter()
            .position(|id| *id == image_token_id)
            .unwrap_or(if ids.is_empty() { 0 } else { 1 });
        let mut row = Vec::new();
        if ids.contains(&image_token_id) {
            row.extend_from_slice(&labs[..idx]);
            row.extend(std::iter::repeat_n(ignore_index, n_visual_tokens));
            row.extend_from_slice(&labs[idx + 1..]);
        } else {
            row.extend_from_slice(&labs[..idx]);
            row.extend(std::iter::repeat_n(ignore_index, n_visual_tokens));
            row.extend_from_slice(&labs[idx..]);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn shifted_logits_and_labels(
    logits: &candle_core::Tensor,
    token_ids: &[Vec<u32>],
    offset: usize,
    token_types: Option<&[Vec<u8>]>,
) -> Result<(candle_core::Tensor, Vec<i64>)> {
    let dims = logits.dims();
    let (b, s, v) = (dims[0], dims[1], dims[2]);
    if offset >= s {
        return Err(ApexError::Shape("offset leaves no labels".to_string()));
    }
    let logits_vec = logits.to_vec3::<f32>()?;
    let rows = b * (s - offset);
    let mut flat = Vec::with_capacity(rows * v);
    let mut labels = Vec::with_capacity(rows);
    for batch in 0..b {
        for (pos, row_logits) in logits_vec[batch].iter().enumerate().take(s - offset) {
            flat.extend_from_slice(row_logits);
            let label_pos = pos + offset;
            let mut label = i64::from(token_ids[batch][label_pos]);
            if let Some(types) = token_types {
                if types[batch][label_pos] != 2 {
                    label = -100;
                }
            }
            labels.push(label);
        }
    }
    Ok((
        candle_core::Tensor::from_vec(flat, (rows, v), logits.device())?,
        labels,
    ))
}
