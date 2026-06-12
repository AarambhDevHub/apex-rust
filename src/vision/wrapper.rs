use candle_core::{Device, Tensor};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};
use crate::model::{ApexModel, ModelOutput};
use crate::tensor;

use super::encoder::NativeVisionEncoder;
use super::projector::VisionToTextProjector;

/// Multimodal wrapper that inserts projected vision tokens into the language model.
#[derive(Clone)]
pub struct ApexVisionModel {
    /// Underlying text language model.
    pub language_model: ApexModel,
    /// Native image patch encoder.
    pub encoder: NativeVisionEncoder,
    /// Vision-to-text projector.
    pub projector: VisionToTextProjector,
    /// Token ID that is replaced by visual token embeddings.
    pub image_token_id: u32,
}

impl ApexVisionModel {
    /// Creates a vision-enabled model wrapper.
    pub fn new(config: ApexConfig, device: Device) -> Result<Self> {
        if !config.vision.enabled {
            return Err(ApexError::Config(
                "ApexVisionModel requires vision.enabled=true".to_string(),
            ));
        }
        Ok(Self {
            encoder: NativeVisionEncoder::new(&config, &device)?,
            projector: VisionToTextProjector::new(&config, &device)?,
            image_token_id: config.vision.image_token_id,
            language_model: ApexModel::new(config, device)?,
        })
    }

    /// Encodes an image into projected text-space visual embeddings.
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        self.projector.forward(&self.encoder.forward(image)?)
    }

    /// Runs text-only or image-conditioned forward pass.
    pub fn forward(
        &mut self,
        token_ids: &[Vec<u32>],
        image: Option<&Tensor>,
        prefix_len: usize,
        return_hidden: bool,
    ) -> Result<ModelOutput> {
        match image {
            Some(img) => {
                let visual_embeds = self.encode_image(img)?;
                self.forward_with_visual_embeddings(
                    token_ids,
                    &visual_embeds,
                    prefix_len,
                    return_hidden,
                )
            }
            None => self
                .language_model
                .forward(token_ids, None, prefix_len, None, return_hidden),
        }
    }

    /// Runs a forward pass after inserting already-computed visual embeddings.
    pub fn forward_with_visual_embeddings(
        &mut self,
        token_ids: &[Vec<u32>],
        visual_embeds: &Tensor,
        prefix_len: usize,
        return_hidden: bool,
    ) -> Result<ModelOutput> {
        let embeddings = self.splice_visual_embeddings(token_ids, visual_embeds)?;
        self.language_model
            .forward_embeddings(&embeddings, None, prefix_len, None, return_hidden)
    }

    /// Replaces each image token with the projected visual embedding sequence.
    pub fn splice_visual_embeddings(
        &self,
        token_ids: &[Vec<u32>],
        visual_embeds: &Tensor,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(ApexError::Data("token_ids must be non-empty".to_string()));
        }
        let dims = visual_embeds.dims();
        if dims.len() != 3 {
            return Err(ApexError::Shape(format!(
                "visual_embeds must be [B,V,D], got {dims:?}"
            )));
        }
        let b = token_ids.len();
        let visual_b = dims[0];
        let v_tokens = dims[1];
        let d = dims[2];
        if visual_b != b {
            return Err(ApexError::Shape(format!(
                "visual batch {visual_b} does not match token batch {b}"
            )));
        }
        let mut row_lengths = Vec::with_capacity(b);
        for row in token_ids {
            let replacements = row.iter().filter(|&&id| id == self.image_token_id).count();
            row_lengths.push(row.len() + replacements * v_tokens.saturating_sub(1));
        }
        let out_len = row_lengths[0];
        if row_lengths.iter().any(|&len| len != out_len) {
            return Err(ApexError::Shape(
                "all rows must have equal length after visual-token insertion".to_string(),
            ));
        }
        let text_embeds = embed_tokens_from_model(&self.language_model.embedding, token_ids)?;
        let text_values = text_embeds.to_vec3::<f32>()?;
        let visual_values = visual_embeds.to_vec3::<f32>()?;
        let mut values = Vec::with_capacity(b * out_len * d);
        for batch in 0..b {
            for (pos, &id) in token_ids[batch].iter().enumerate() {
                if id == self.image_token_id {
                    for visual_token in visual_values[batch].iter().take(v_tokens) {
                        values.extend_from_slice(visual_token);
                    }
                } else {
                    values.extend_from_slice(&text_values[batch][pos]);
                }
            }
        }
        let scaled = Tensor::from_vec(values, (b, out_len, d), visual_embeds.device())?;
        scaled
            .broadcast_mul(&tensor::scalar(
                self.language_model.embed_scale,
                visual_embeds.device(),
            )?)
            .map_err(Into::into)
    }

    /// Returns total parameters across text model, encoder, and projector.
    pub fn total_parameters(&self) -> usize {
        self.language_model.total_parameters()
            + self.encoder.parameters()
            + self.projector.parameters()
    }
}

/// Expands one label row to ignore labels for inserted visual tokens.
pub fn expand_labels_for_visual_tokens(
    labels: &[i64],
    input_ids: &[u32],
    image_token_id: u32,
    n_visual_tokens: usize,
    ignore_index: i64,
) -> Vec<i64> {
    let mut out = Vec::new();
    for (&label, &token) in labels.iter().zip(input_ids) {
        if token == image_token_id {
            out.extend(std::iter::repeat_n(ignore_index, n_visual_tokens));
        } else {
            out.push(label);
        }
    }
    out
}

/// Embeds token IDs using the text model's embedding table.
fn embed_tokens_from_model(embedding: &Tensor, token_ids: &[Vec<u32>]) -> Result<Tensor> {
    let rows = embedding.to_vec2::<f32>()?;
    let b = token_ids.len();
    let s = token_ids[0].len();
    let d = embedding.dim(1)?;
    let mut values = Vec::with_capacity(b * s * d);
    for row in token_ids {
        for &id in row {
            let idx = id as usize;
            if idx >= rows.len() {
                return Err(ApexError::Data(format!(
                    "token id {id} outside embedding vocab"
                )));
            }
            values.extend_from_slice(&rows[idx]);
        }
    }
    Ok(Tensor::from_vec(values, (b, s, d), embedding.device())?)
}
