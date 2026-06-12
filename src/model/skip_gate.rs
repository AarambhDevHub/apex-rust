use candle_core::{Device, Tensor};

use crate::error::Result;
use crate::tensor;

use super::linear::PlainLinear;

/// Two-layer gate that decides whether each token should run the FFN path.
#[derive(Clone)]
pub struct SkipGate {
    /// First gate projection from model hidden size to gate hidden size.
    pub fc1: PlainLinear,
    /// Second gate projection to one probability per token.
    pub fc2: PlainLinear,
    /// Threshold used to convert probabilities into a hard run mask.
    pub threshold: f64,
}

impl SkipGate {
    /// Creates a skip gate MLP.
    pub fn new(d_model: usize, hidden: usize, threshold: f64, device: &Device) -> Result<Self> {
        Ok(Self {
            fc1: PlainLinear::new("skip_gate.0", d_model, hidden, true, device)?,
            fc2: PlainLinear::new("skip_gate.2", hidden, 1, true, device)?,
            threshold,
        })
    }

    /// Returns sigmoid gate probabilities with shape `[B,S,1]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        tensor::sigmoid(&self.fc2.forward(&tensor::silu(&self.fc1.forward(x)?)?)?)
    }

    /// Returns a hard mask where one means the FFN output is used.
    pub fn run_mask(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.forward(x)?.to_vec3::<f32>()?;
        let dims = x.dims();
        let mut values = Vec::with_capacity(dims[0] * dims[1]);
        for batch in gate {
            for item in batch {
                values.push(if f64::from(item[0]) < self.threshold {
                    0.0_f32
                } else {
                    1.0_f32
                });
            }
        }
        Ok(Tensor::from_vec(values, (dims[0], dims[1], 1), x.device())?)
    }

    /// Returns the number of gate parameters.
    pub fn parameters(&self) -> usize {
        self.fc1.parameters() + self.fc2.parameters()
    }
}
