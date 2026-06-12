use candle_core::{Device, Tensor};

use crate::error::Result;
use crate::tensor;

#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    pub fn new(d_model: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: tensor::ones(&[d_model], device)?,
            eps: 1e-6,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        tensor::rms_norm(x, &self.weight, self.eps)
    }

    pub fn parameters(&self) -> usize {
        self.weight.elem_count()
    }
}
