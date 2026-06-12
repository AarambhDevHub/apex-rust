use candle_core::{Device, IndexOp, Tensor};

use crate::error::Result;
use crate::tensor;

use super::linear::PlainLinear;

#[derive(Clone)]
pub struct MultiTokenHead {
    pub heads: Vec<PlainLinear>,
}

impl MultiTokenHead {
    pub fn new(
        d_model: usize,
        vocab_size: usize,
        n_predict: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            heads: (0..n_predict)
                .map(|i| {
                    PlainLinear::new(
                        format!("multi_token_head.{i}"),
                        d_model,
                        vocab_size,
                        false,
                        device,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Vec<Tensor>> {
        self.heads.iter().map(|head| head.forward(hidden)).collect()
    }

    pub fn draft_tokens(&self, hidden: &Tensor) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(self.heads.len());
        for logits in self.forward(hidden)? {
            let last = logits.i((0, logits.dim(1)? - 1, ..))?.to_vec1::<f32>()?;
            let idx = tensor::top_k_indices(&last, 1)
                .into_iter()
                .next()
                .unwrap_or(0);
            ids.push(idx as u32);
        }
        Ok(ids)
    }

    pub fn parameters(&self) -> usize {
        self.heads.iter().map(PlainLinear::parameters).sum()
    }

    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        for (idx, head) in self.heads.iter().enumerate() {
            head.named_tensors(&format!("{prefix}.{idx}"), out);
        }
    }
}
