use candle_core::Device;
use candle_core::{IndexOp, Tensor};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};
use crate::model::PlainLinear;
use crate::tensor;

/// Projects vision tokens into the language model hidden space.
#[derive(Clone)]
pub struct VisionToTextProjector {
    /// Projector mode, currently `perceiver` or `mlp`.
    pub projector_type: String,
    /// Number of visual tokens emitted for insertion into text.
    pub n_visual_tokens: usize,
    /// Input projection from vision hidden size to projector hidden size.
    pub input_proj: PlainLinear,
    /// Optional hidden MLP layers.
    pub hidden_layers: Vec<PlainLinear>,
    /// Output projection from projector hidden size to text hidden size.
    pub output_proj: PlainLinear,
    /// Learned latent tokens used by the perceiver-style selector.
    pub latents: Option<Tensor>,
}

impl VisionToTextProjector {
    /// Creates a projector from vision and model config.
    pub fn new(cfg: &ApexConfig, device: &Device) -> Result<Self> {
        let v = &cfg.vision;
        let hidden = v.projector_hidden_dim;
        let mut hidden_layers = Vec::new();
        for idx in 0..v.projector_layers.saturating_sub(1) {
            hidden_layers.push(PlainLinear::new(
                format!("vision.projector.hidden.{idx}"),
                hidden,
                hidden,
                true,
                device,
            )?);
        }
        let latents = if v.projector_type == "perceiver" {
            Some(tensor::randn(
                &[1, v.n_visual_tokens, v.d_vision],
                0.0,
                0.02,
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            projector_type: v.projector_type.clone(),
            n_visual_tokens: v.n_visual_tokens,
            input_proj: PlainLinear::new(
                "vision.projector.input",
                v.d_vision,
                hidden,
                true,
                device,
            )?,
            hidden_layers,
            output_proj: PlainLinear::new(
                "vision.projector.output",
                hidden,
                cfg.model.d_model,
                true,
                device,
            )?,
            latents,
        })
    }

    /// Selects visual tokens and projects them into text embedding space.
    pub fn forward(&self, vision_tokens: &Tensor) -> Result<Tensor> {
        let selected = if self.projector_type == "perceiver" {
            self.perceiver_select(vision_tokens)?
        } else {
            self.mlp_select(vision_tokens)?
        };
        let mut h = tensor::gelu(&self.input_proj.forward(&selected)?)?;
        for layer in &self.hidden_layers {
            h = tensor::gelu(&layer.forward(&h)?)?;
        }
        self.output_proj.forward(&h)
    }

    /// Selects tokens using learned latents plus pooled/source token features.
    fn perceiver_select(&self, vision_tokens: &Tensor) -> Result<Tensor> {
        let dims = vision_tokens.dims();
        if dims.len() != 3 {
            return Err(ApexError::Shape(format!(
                "projector expects [B,S,D], got {dims:?}"
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let pooled = mean_tokens(vision_tokens)?;
        let latents = self
            .latents
            .as_ref()
            .ok_or_else(|| ApexError::Model("perceiver projector missing latents".to_string()))?
            .broadcast_as((b, self.n_visual_tokens, d))?;
        let mut values = latents.to_vec3::<f32>()?;
        let pooled_values = pooled.to_vec2::<f32>()?;
        for batch in 0..b {
            for (token, token_values) in values[batch]
                .iter_mut()
                .enumerate()
                .take(self.n_visual_tokens)
            {
                let source_idx = token * s / self.n_visual_tokens.max(1);
                let source = vision_tokens
                    .i((batch, source_idx.min(s - 1), ..))?
                    .to_vec1::<f32>()?;
                for feature in 0..d {
                    token_values[feature] += 0.5 * pooled_values[batch][feature] + source[feature];
                }
            }
        }
        Ok(Tensor::from_vec(
            values.into_iter().flatten().flatten().collect::<Vec<_>>(),
            (b, self.n_visual_tokens, d),
            vision_tokens.device(),
        )?)
    }

    /// Selects evenly spaced vision tokens for the MLP projector path.
    fn mlp_select(&self, vision_tokens: &Tensor) -> Result<Tensor> {
        let dims = vision_tokens.dims();
        if dims.len() != 3 {
            return Err(ApexError::Shape(format!(
                "projector expects [B,S,D], got {dims:?}"
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let mut values = Vec::with_capacity(b * self.n_visual_tokens * d);
        let tokens = vision_tokens.to_vec3::<f32>()?;
        for batch in tokens.iter().take(b) {
            for token in 0..self.n_visual_tokens {
                let source_idx = token * s / self.n_visual_tokens.max(1);
                values.extend_from_slice(&batch[source_idx.min(s - 1)]);
            }
        }
        Ok(Tensor::from_vec(
            values,
            (b, self.n_visual_tokens, d),
            vision_tokens.device(),
        )?)
    }

    /// Returns the number of projector parameters.
    pub fn parameters(&self) -> usize {
        self.input_proj.parameters()
            + self.output_proj.parameters()
            + self
                .hidden_layers
                .iter()
                .map(PlainLinear::parameters)
                .sum::<usize>()
            + self.latents.as_ref().map(Tensor::elem_count).unwrap_or(0)
    }
}

/// Mean-pools `[B,S,D]` tokens across the sequence dimension.
fn mean_tokens(tokens: &Tensor) -> Result<Tensor> {
    let dims = tokens.dims();
    let values = tokens.to_vec3::<f32>()?;
    let (b, s, d) = (dims[0], dims[1], dims[2]);
    let mut out = Vec::with_capacity(b * d);
    for batch in values.iter().take(b) {
        for feature in 0..d {
            let sum: f32 = batch.iter().take(s).map(|row| row[feature]).sum();
            out.push(sum / s.max(1) as f32);
        }
    }
    Ok(Tensor::from_vec(out, (b, d), tokens.device())?)
}
