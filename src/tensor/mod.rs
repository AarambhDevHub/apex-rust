//! Small Candle tensor helpers used by model, loss, and generation code.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use rand_distr::{Distribution, Normal};

use crate::error::{ApexError, Result};

/// Returns the CPU device used by the default runtime.
pub fn device_cpu() -> Device {
    Device::Cpu
}

/// Creates a normally initialized float tensor.
pub fn randn(shape: &[usize], mean: f64, std: f64, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let normal = Normal::new(mean, std)
        .map_err(|e| ApexError::Model(format!("invalid normal initializer: {e}")))?;
    let mut rng = rand::rng();
    let values: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng) as f32).collect();
    Ok(Tensor::from_vec(values, shape, device)?)
}

/// Creates a float tensor filled with zeros.
pub fn zeros(shape: &[usize], device: &Device) -> Result<Tensor> {
    Ok(Tensor::zeros(shape, DType::F32, device)?)
}

/// Creates a float tensor filled with ones.
pub fn ones(shape: &[usize], device: &Device) -> Result<Tensor> {
    Ok(Tensor::ones(shape, DType::F32, device)?)
}

/// Creates a scalar float tensor.
pub fn scalar(v: f64, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(vec![v as f32], (), device)?)
}

/// Applies a linear projection to the last dimension of an arbitrary-rank tensor.
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let x_dims = x.dims();
    let w_dims = weight.dims();
    if w_dims.len() != 2 || x_dims.is_empty() {
        return Err(ApexError::Shape(format!(
            "linear expects x rank >=1 and weight rank 2, got x={x_dims:?} weight={w_dims:?}"
        )));
    }
    let in_features = *x_dims.last().unwrap_or(&0);
    if in_features != w_dims[1] {
        return Err(ApexError::Shape(format!(
            "linear input dim {in_features} does not match weight dim {}",
            w_dims[1]
        )));
    }
    let out_features = w_dims[0];
    let outer: usize = x_dims[..x_dims.len() - 1].iter().product();
    let flat = x.reshape((outer, in_features))?;
    let y = flat.matmul(&weight.t()?)?;
    match bias {
        Some(b) => {
            let y = y.broadcast_add(b)?;
            restore_linear_shape(y, x_dims, out_features)
        }
        None => restore_linear_shape(y, x_dims, out_features),
    }
}

/// Restores the original batch dimensions after a flattened linear projection.
fn restore_linear_shape(y: Tensor, input_shape: &[usize], out_features: usize) -> Result<Tensor> {
    if input_shape.len() == 2 {
        return Ok(y);
    }
    let mut shape = input_shape.to_vec();
    if let Some(last) = shape.last_mut() {
        *last = out_features;
    }
    Ok(y.reshape(shape.as_slice())?)
}

/// Applies the SiLU activation.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// Applies approximate GELU activation.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

/// Applies the sigmoid activation.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::sigmoid(x)?)
}

/// Applies softmax along the final dimension.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::softmax(x, D::Minus1)?)
}

/// Applies log-softmax along the final dimension.
pub fn log_softmax_last(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::log_softmax(x, D::Minus1)?)
}

/// Applies RMSNorm with a learned scale vector.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let sq = x.sqr()?;
    let mean = sq.mean_keepdim(D::Minus1)?;
    let denom = mean.broadcast_add(&scalar(eps, x.device())?)?.sqrt()?;
    Ok(x.broadcast_div(&denom)?.broadcast_mul(weight)?)
}

/// Normalizes each row of a weight matrix by its L2 norm.
pub fn l2_normalize_rows(weight: &Tensor, eps: f64) -> Result<Tensor> {
    let denom = weight.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let denom = denom.broadcast_add(&scalar(eps, weight.device())?)?;
    Ok(weight.broadcast_div(&denom)?)
}

/// Repeats key/value heads to match the query-head count in grouped attention.
pub fn repeat_kv_heads(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let dims = x.dims();
    if dims.len() != 4 {
        return Err(ApexError::Shape(format!(
            "repeat_kv_heads expects rank 4, got {:?}",
            dims
        )));
    }
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let mut heads = Vec::with_capacity(h * groups);
    for head in 0..h {
        let slice = x.i((.., head, .., ..))?.unsqueeze(1)?;
        for _ in 0..groups {
            heads.push(slice.clone());
        }
    }
    Ok(Tensor::cat(&heads, 1)?.reshape((b, h * groups, s, d))?)
}

/// Concatenates two tensors along the final dimension.
pub fn concat_last(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(Tensor::cat(&[a, b], D::Minus1)?)
}

/// Computes summed cross-entropy loss over valid labels.
pub fn cross_entropy_sum(
    logits: &Tensor,
    labels: &[i64],
    ignore_index: i64,
) -> Result<(f64, usize)> {
    let dims = logits.dims();
    if dims.len() != 2 {
        return Err(ApexError::Shape(format!(
            "cross_entropy_sum expects [N,V], got {:?}",
            dims
        )));
    }
    if dims[0] != labels.len() {
        return Err(ApexError::Shape(format!(
            "labels length {} does not match logits rows {}",
            labels.len(),
            dims[0]
        )));
    }
    let probs = log_softmax_last(logits)?;
    let rows = probs.to_vec2::<f32>()?;
    let mut loss = 0.0;
    let mut count = 0usize;
    for (row, &label) in rows.iter().zip(labels) {
        if label == ignore_index {
            continue;
        }
        let idx = usize::try_from(label)
            .map_err(|_| ApexError::Data(format!("negative label {label}")))?;
        if idx >= row.len() {
            return Err(ApexError::Data(format!(
                "label {idx} outside vocab {}",
                row.len()
            )));
        }
        loss -= f64::from(row[idx]);
        count += 1;
    }
    Ok((loss, count))
}

/// Flattens a rank-1 or rank-2 integer tensor into a vector.
pub fn flatten2_i64(t: &Tensor) -> Result<Vec<i64>> {
    match t.dims().len() {
        1 => Ok(t.to_vec1::<i64>()?),
        2 => Ok(t.to_vec2::<i64>()?.into_iter().flatten().collect()),
        other => Err(ApexError::Shape(format!(
            "expected rank 1 or 2 integer tensor, got rank {other}"
        ))),
    }
}

/// Returns indices of the largest `k` values in descending order.
pub fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    idx.truncate(k.min(values.len()));
    idx
}

/// Converts logits to a normalized probability vector with finite fallbacks.
pub fn safe_probs_from_logits(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![1.0 / logits.len().max(1) as f32; logits.len()];
    }
    let mut exps: Vec<f32> = logits
        .iter()
        .map(|v| if v.is_finite() { (v - max).exp() } else { 0.0 })
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let p = 1.0 / exps.len().max(1) as f32;
        exps.fill(p);
        return exps;
    }
    for v in &mut exps {
        *v /= sum;
    }
    exps
}
