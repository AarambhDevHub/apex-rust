use candle_core::{Device, IndexOp, Tensor};

use crate::error::Result;
use crate::tensor;

use super::linear::PlainLinear;

/// Set of auxiliary heads that predict multiple future tokens from hidden states.
#[derive(Clone)]
pub struct MultiTokenHead {
    /// One linear vocabulary projection per future-token offset.
    pub heads: Vec<PlainLinear>,
}

impl MultiTokenHead {
    /// Creates `n_predict` vocabulary heads.
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

    /// Returns logits from every auxiliary head.
    pub fn forward(&self, hidden: &Tensor) -> Result<Vec<Tensor>> {
        self.heads.iter().map(|head| head.forward(hidden)).collect()
    }

    /// Greedily drafts one token from each auxiliary head at the last position.
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

    /// Returns the number of parameters across all heads.
    pub fn parameters(&self) -> usize {
        self.heads.iter().map(PlainLinear::parameters).sum()
    }

    /// Appends named head tensors to a checkpoint list.
    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        for (idx, head) in self.heads.iter().enumerate() {
            head.named_tensors(&format!("{prefix}.{idx}"), out);
        }
    }
}
