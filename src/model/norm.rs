use candle_core::{Device, Tensor};

use crate::error::Result;
use crate::tensor;

/// Root-mean-square layer normalization with a learned scale vector.
#[derive(Clone)]
pub struct RmsNorm {
    /// Learned per-channel scale.
    pub weight: Tensor,
    /// Numerical epsilon added to the RMS denominator.
    pub eps: f64,
}

impl RmsNorm {
    /// Creates an RMSNorm initialized with unit weights.
    pub fn new(d_model: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: tensor::ones(&[d_model], device)?,
            eps: 1e-6,
        })
    }

    /// Normalizes the last dimension of `x`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        tensor::rms_norm(x, &self.weight, self.eps)
    }

    /// Returns the number of learned parameters.
    pub fn parameters(&self) -> usize {
        self.weight.elem_count()
    }
}
