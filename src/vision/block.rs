//! Native ViT transformer block used by the vision encoder.

use candle_core::{Device, Tensor};

use crate::error::{ApexError, Result};
use crate::model::{PlainLinear, RmsNorm};
use crate::tensor;

/// One pre-norm transformer block for image patch tokens.
#[derive(Clone)]
pub struct VisionTransformerBlock {
    /// Number of vision attention heads.
    pub n_heads: usize,
    /// Per-head hidden size.
    pub d_head: usize,
    /// Pre-attention normalization.
    pub norm1: RmsNorm,
    /// Pre-MLP normalization.
    pub norm2: RmsNorm,
    /// Query projection.
    pub w_q: PlainLinear,
    /// Key projection.
    pub w_k: PlainLinear,
    /// Value projection.
    pub w_v: PlainLinear,
    /// Output projection.
    pub w_o: PlainLinear,
    /// First MLP projection.
    pub fc1: PlainLinear,
    /// Second MLP projection back to `d_vision`.
    pub fc2: PlainLinear,
}

impl VisionTransformerBlock {
    /// Creates a vision transformer block.
    pub fn new(
        prefix: &str,
        d_vision: usize,
        n_heads: usize,
        mlp_ratio: f64,
        device: &Device,
    ) -> Result<Self> {
        if n_heads == 0 || !d_vision.is_multiple_of(n_heads) {
            return Err(ApexError::Config(
                "vision block requires d_vision divisible by n_heads".to_string(),
            ));
        }
        let d_head = d_vision / n_heads;
        let mlp_hidden = ((d_vision as f64 * mlp_ratio).round() as usize).max(d_vision);
        Ok(Self {
            n_heads,
            d_head,
            norm1: RmsNorm::new(d_vision, device)?,
            norm2: RmsNorm::new(d_vision, device)?,
            w_q: PlainLinear::new(format!("{prefix}.W_Q"), d_vision, d_vision, true, device)?,
            w_k: PlainLinear::new(format!("{prefix}.W_K"), d_vision, d_vision, true, device)?,
            w_v: PlainLinear::new(format!("{prefix}.W_V"), d_vision, d_vision, true, device)?,
            w_o: PlainLinear::new(format!("{prefix}.W_O"), d_vision, d_vision, true, device)?,
            fc1: PlainLinear::new(
                format!("{prefix}.mlp.fc1"),
                d_vision,
                mlp_hidden,
                true,
                device,
            )?,
            fc2: PlainLinear::new(
                format!("{prefix}.mlp.fc2"),
                mlp_hidden,
                d_vision,
                true,
                device,
            )?,
        })
    }

    /// Applies self-attention and MLP residual paths to vision tokens.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(ApexError::Shape(format!(
                "vision transformer block expects [B,S,D], got {dims:?}"
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let h = self.norm1.forward(x)?;
        let q = self
            .w_q
            .forward(&h)?
            .reshape((b, s, self.n_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w_k
            .forward(&h)?
            .reshape((b, s, self.n_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w_v
            .forward(&h)?
            .reshape((b, s, self.n_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .broadcast_div(&tensor::scalar((self.d_head as f64).sqrt(), x.device())?)?;
        let attn = tensor::softmax_last(&scores)?
            .contiguous()?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, s, d))?;
        let y = x.broadcast_add(&self.w_o.forward(&attn)?)?;
        let mlp = self
            .fc2
            .forward(&tensor::gelu(&self.fc1.forward(&self.norm2.forward(&y)?)?)?)?;
        Ok(y.broadcast_add(&mlp)?)
    }

    /// Returns the number of stored block parameters.
    pub fn parameters(&self) -> usize {
        self.norm1.parameters()
            + self.norm2.parameters()
            + self.w_q.parameters()
            + self.w_k.parameters()
            + self.w_v.parameters()
            + self.w_o.parameters()
            + self.fc1.parameters()
            + self.fc2.parameters()
    }
}
