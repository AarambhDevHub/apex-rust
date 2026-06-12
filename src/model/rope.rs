use candle_core::{Device, Tensor};

use crate::error::{ApexError, Result};

pub fn precompute_rope_cache(
    d_head: usize,
    max_seq_len: usize,
    rope_base: f64,
    scale_factor: f64,
    device: &Device,
) -> Result<(Tensor, Tensor, f64)> {
    let mut theta = Vec::with_capacity(d_head / 2);
    for i in (0..d_head).step_by(2) {
        theta.push((1.0 / rope_base.powf(i as f64 / d_head as f64)) as f32);
    }
    let (theta, attn_factor) = apply_yarn_scaling_vec(&theta, scale_factor, 32.0, 1.0);
    let mut cos = Vec::with_capacity(max_seq_len * d_head);
    let mut sin = Vec::with_capacity(max_seq_len * d_head);
    for pos in 0..max_seq_len {
        for &freq in &theta {
            let angle = pos as f32 * freq;
            let c = angle.cos();
            let s = angle.sin();
            cos.push(c);
            cos.push(c);
            sin.push(s);
            sin.push(s);
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_seq_len, d_head), device)?,
        Tensor::from_vec(sin, (max_seq_len, d_head), device)?,
        attn_factor,
    ))
}

pub fn apply_yarn_scaling_vec(
    theta: &[f32],
    scale_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> (Vec<f32>, f64) {
    if scale_factor <= 1.0 {
        return (theta.to_vec(), 1.0);
    }
    let mut out = Vec::with_capacity(theta.len());
    for &t in theta {
        let wavelength = 2.0 * std::f64::consts::PI / (f64::from(t).max(1e-30));
        let scaled = if wavelength < beta_fast {
            f64::from(t)
        } else if wavelength > beta_slow * scale_factor {
            f64::from(t) / scale_factor
        } else {
            let interp = (wavelength / beta_slow - 1.0) / (scale_factor - 1.0);
            f64::from(t) / (interp * scale_factor + (1.0 - interp))
        };
        out.push(scaled as f32);
    }
    (out, 0.1 * scale_factor.ln() + 1.0)
}

fn rotate_half_vec(v: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len());
    for pair in v.chunks(2) {
        if pair.len() == 2 {
            out.push(-pair[1]);
            out.push(pair[0]);
        }
    }
    out
}

pub(crate) fn apply_rope_single(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &[usize],
) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() != 4 {
        return Err(ApexError::Shape(format!(
            "apply_rope_single expects rank 4, got {dims:?}"
        )));
    }
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let input = x.flatten_all()?.to_vec1::<f32>()?;
    let cos_rows = cos.to_vec2::<f32>()?;
    let sin_rows = sin.to_vec2::<f32>()?;
    let mut out = Vec::with_capacity(b * h * s * d);
    for batch in 0..b {
        for head in 0..h {
            for (tok, pos) in positions.iter().copied().enumerate().take(s) {
                let base = ((batch * h + head) * s + tok) * d;
                let vals = &input[base..base + d];
                let rotated = rotate_half_vec(vals);
                for i in 0..d {
                    out.push(vals[i] * cos_rows[pos][i] + rotated[i] * sin_rows[pos][i]);
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, (b, h, s, d), x.device())?)
}

pub(crate) fn apply_rope_pair(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &[usize],
) -> Result<(Tensor, Tensor)> {
    Ok((
        apply_rope_single(q, cos, sin, positions)?,
        apply_rope_single(k, cos, sin, positions)?,
    ))
}
