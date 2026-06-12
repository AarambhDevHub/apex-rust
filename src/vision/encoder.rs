use candle_core::Device;
use candle_core::Tensor;

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};
use crate::model::{PlainLinear, RmsNorm};
use crate::tensor;

use super::block::VisionTransformerBlock;

/// Native ViT encoder that converts image patches into contextual vision tokens.
#[derive(Clone)]
pub struct NativeVisionEncoder {
    /// Expected square image size.
    pub image_size: usize,
    /// Square patch size.
    pub patch_size: usize,
    /// Number of input channels.
    pub channels: usize,
    /// Vision token hidden size.
    pub d_vision: usize,
    /// Linear projection applied to flattened patches.
    pub patch_proj: PlainLinear,
    /// Learned CLS token prepended to patch tokens.
    pub cls_token: Tensor,
    /// Learned positional embedding for CLS plus patch tokens.
    pub pos_embed: Tensor,
    /// Transformer blocks that contextualize CLS and patch tokens.
    pub blocks: Vec<VisionTransformerBlock>,
    /// Final token normalization.
    pub norm: RmsNorm,
}

impl NativeVisionEncoder {
    /// Creates the patch encoder and learned vision embeddings.
    pub fn new(cfg: &ApexConfig, device: &Device) -> Result<Self> {
        let v = &cfg.vision;
        let patches_per_side = v.image_size / v.patch_size;
        let n_patches = patches_per_side * patches_per_side;
        Ok(Self {
            image_size: v.image_size,
            patch_size: v.patch_size,
            channels: v.in_channels,
            d_vision: v.d_vision,
            patch_proj: PlainLinear::new(
                "vision.patch_proj",
                v.in_channels * v.patch_size * v.patch_size,
                v.d_vision,
                true,
                device,
            )?,
            cls_token: tensor::zeros(&[1, 1, v.d_vision], device)?,
            pos_embed: tensor::randn(&[1, n_patches + 1, v.d_vision], 0.0, 0.02, device)?,
            blocks: (0..v.n_layers)
                .map(|idx| {
                    VisionTransformerBlock::new(
                        &format!("vision.encoder.blocks.{idx}"),
                        v.d_vision,
                        v.n_heads,
                        v.mlp_ratio,
                        device,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            norm: RmsNorm::new(v.d_vision, device)?,
        })
    }

    /// Encodes an image tensor `[B,C,H,W]` into `[B,N+1,Dv]` tokens.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let dims = image.dims();
        if dims.len() != 4 {
            return Err(ApexError::Shape(format!(
                "vision encoder expects [B,C,H,W], got {dims:?}"
            )));
        }
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        if c != self.channels || h != self.image_size || w != self.image_size {
            return Err(ApexError::Shape(format!(
                "vision encoder expected [B,{},{},{}], got {dims:?}",
                self.channels, self.image_size, self.image_size
            )));
        }
        let patch = self.patch_size;
        let patches_side = h / patch;
        let images = image.flatten_all()?.to_vec1::<f32>()?;
        let mut flattened = Vec::with_capacity(b * patches_side * patches_side * c * patch * patch);
        for batch in 0..b {
            for py in 0..patches_side {
                for px in 0..patches_side {
                    for channel in 0..c {
                        for iy in 0..patch {
                            let row = py * patch + iy;
                            for ix in 0..patch {
                                let col = px * patch + ix;
                                let idx = ((batch * c + channel) * h + row) * w + col;
                                flattened.push(images[idx]);
                            }
                        }
                    }
                }
            }
        }
        let n_patches = patches_side * patches_side;
        let patch_dim = c * patch * patch;
        let patches = Tensor::from_vec(flattened, (b, n_patches, patch_dim), image.device())?;
        let patch_tokens = self.patch_proj.forward(&patches)?;
        let cls = self.cls_token.broadcast_as((b, 1, self.d_vision))?;
        let mut tokens = Tensor::cat(&[&cls, &patch_tokens], 1)?.broadcast_add(&self.pos_embed)?;
        for block in &self.blocks {
            tokens = block.forward(&tokens)?;
        }
        self.norm.forward(&tokens)
    }

    /// Returns the number of vision encoder parameters.
    pub fn parameters(&self) -> usize {
        self.patch_proj.parameters()
            + self.cls_token.elem_count()
            + self.pos_embed.elem_count()
            + self
                .blocks
                .iter()
                .map(VisionTransformerBlock::parameters)
                .sum::<usize>()
            + self.norm.parameters()
    }
}
