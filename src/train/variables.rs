//! Trainable variable registration for Candle optimizer-backed loops.

use candle_core::{Tensor, Var};

use crate::error::Result;
use crate::model::{
    ApexModel, AttentionKind, BaseLinear, DenseFfn, FfnKind, LinearLayer, PlainLinear,
};
use crate::vision::{
    ApexVisionModel, NativeVisionEncoder, VisionToTextProjector, VisionTransformerBlock,
};

/// Which parts of a model should be registered as optimizer variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainableScope {
    /// Register every floating-point parameter that participates in text forward passes.
    Full,
    /// Register only adapter tensors plus any explicitly trainable PEFT base tensors.
    AdaptersOnly,
}

/// A named optimizer variable backed by a model tensor.
#[derive(Clone)]
pub struct TrainableTensor {
    /// Stable checkpoint-style tensor name.
    pub name: String,
    /// Candle variable handle used by optimizers and gradient stores.
    pub var: Var,
}

/// Collection of trainable variables attached to a model instance.
#[derive(Clone, Default)]
pub struct TrainableVariables {
    /// Named variable handles.
    pub tensors: Vec<TrainableTensor>,
}

impl TrainableVariables {
    /// Replaces eligible tensors in `model` with variable-backed tensors.
    pub fn attach_model(model: &mut ApexModel, scope: TrainableScope) -> Result<Self> {
        let mut vars = Self::default();
        if matches!(scope, TrainableScope::Full) {
            vars.attach_tensor("embedding.weight", &mut model.embedding)?;
            vars.attach_tensor("final_norm.weight", &mut model.final_norm.weight)?;
        }
        for (idx, block) in model.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{idx}");
            if matches!(scope, TrainableScope::Full) {
                vars.attach_tensor(format!("{prefix}.norm1.weight"), &mut block.norm1.weight)?;
                vars.attach_tensor(format!("{prefix}.norm2.weight"), &mut block.norm2.weight)?;
            }
            vars.attach_attention(&format!("{prefix}.attn"), &mut block.attn, scope)?;
            vars.attach_ffn(&format!("{prefix}.ffn"), &mut block.ffn, scope)?;
            if matches!(scope, TrainableScope::Full) {
                if let Some(gate) = &mut block.skip_gate {
                    vars.attach_plain(&format!("{prefix}.skip_gate.fc1"), &mut gate.fc1, true)?;
                    vars.attach_plain(&format!("{prefix}.skip_gate.fc2"), &mut gate.fc2, true)?;
                }
            }
        }
        if matches!(scope, TrainableScope::Full) {
            if let Some(head) = &mut model.multi_token_head {
                for (idx, layer) in head.heads.iter_mut().enumerate() {
                    vars.attach_plain(&format!("multi_token_head.{idx}"), layer, true)?;
                }
            }
        }
        Ok(vars)
    }

    /// Replaces eligible tensors in a vision-language wrapper with variable-backed tensors.
    pub fn attach_vision_model(model: &mut ApexVisionModel) -> Result<Self> {
        let language_scope = scope_for_model(&model.language_model);
        let mut vars = if model.language_model.config.vision.freeze_language_model {
            Self::default()
        } else {
            Self::attach_model(&mut model.language_model, language_scope)?
        };
        if !model.language_model.config.vision.freeze_vision_encoder {
            vars.attach_vision_encoder("vision.encoder", &mut model.encoder)?;
        }
        vars.attach_projector("vision.projector", &mut model.projector)?;
        Ok(vars)
    }

    /// Returns cloned Candle variable handles for optimizer construction.
    pub fn vars(&self) -> Vec<Var> {
        self.tensors.iter().map(|item| item.var.clone()).collect()
    }

    /// Returns the number of registered tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true when no tensors are registered.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns the total scalar parameter count represented by registered tensors.
    pub fn parameter_count(&self) -> usize {
        self.tensors
            .iter()
            .map(|item| item.var.as_tensor().elem_count())
            .sum()
    }

    /// Returns the registered tensor names in optimizer order.
    pub fn names(&self) -> Vec<String> {
        self.tensors.iter().map(|item| item.name.clone()).collect()
    }

    fn attach_tensor(&mut self, name: impl Into<String>, tensor: &mut Tensor) -> Result<()> {
        let var = Var::from_tensor(tensor)?;
        *tensor = var.as_tensor().clone();
        self.tensors.push(TrainableTensor {
            name: name.into(),
            var,
        });
        Ok(())
    }

    fn attach_plain(
        &mut self,
        prefix: &str,
        layer: &mut PlainLinear,
        include_even_if_frozen: bool,
    ) -> Result<()> {
        if include_even_if_frozen || layer.trainable {
            self.attach_tensor(format!("{prefix}.weight"), &mut layer.weight)?;
            if let Some(bias) = &mut layer.bias {
                self.attach_tensor(format!("{prefix}.bias"), bias)?;
            }
        }
        Ok(())
    }

    fn attach_base(
        &mut self,
        prefix: &str,
        base: &mut BaseLinear,
        scope: TrainableScope,
    ) -> Result<()> {
        match base {
            BaseLinear::Plain(layer) => {
                let include = matches!(scope, TrainableScope::Full) || layer.trainable;
                self.attach_plain(prefix, layer, include)
            }
            BaseLinear::Quantized(layer) => {
                if matches!(scope, TrainableScope::Full) {
                    if let Some(bias) = &mut layer.bias {
                        self.attach_tensor(format!("{prefix}.bias"), bias)?;
                    }
                }
                Ok(())
            }
        }
    }

    fn attach_linear(
        &mut self,
        prefix: &str,
        layer: &mut LinearLayer,
        scope: TrainableScope,
    ) -> Result<()> {
        match layer {
            LinearLayer::Plain(layer) => {
                let include = matches!(scope, TrainableScope::Full) || layer.trainable;
                self.attach_plain(prefix, layer, include)
            }
            LinearLayer::Lora {
                base,
                lora_a,
                lora_b,
                dora_magnitude,
                ..
            } => {
                self.attach_base(&format!("{prefix}.base"), base, scope)?;
                self.attach_plain(&format!("{prefix}.lora_A"), lora_a, true)?;
                self.attach_plain(&format!("{prefix}.lora_B"), lora_b, true)?;
                if let Some(mag) = dora_magnitude {
                    self.attach_tensor(format!("{prefix}.dora_magnitude"), mag)?;
                }
                Ok(())
            }
        }
    }

    fn attach_attention(
        &mut self,
        prefix: &str,
        attn: &mut AttentionKind,
        scope: TrainableScope,
    ) -> Result<()> {
        match attn {
            AttentionKind::Mla(attn) => {
                self.attach_linear(&format!("{prefix}.W_DKV"), &mut attn.w_dkv, scope)?;
                self.attach_linear(&format!("{prefix}.W_UK"), &mut attn.w_uk, scope)?;
                self.attach_linear(&format!("{prefix}.W_UV"), &mut attn.w_uv, scope)?;
                self.attach_linear(&format!("{prefix}.W_DQ"), &mut attn.w_dq, scope)?;
                self.attach_linear(&format!("{prefix}.W_UQ"), &mut attn.w_uq, scope)?;
                self.attach_linear(&format!("{prefix}.W_KR"), &mut attn.w_kr, scope)?;
                self.attach_linear(&format!("{prefix}.W_QR"), &mut attn.w_qr, scope)?;
                self.attach_linear(&format!("{prefix}.W_O"), &mut attn.w_o, scope)
            }
            AttentionKind::Gqa(attn) => {
                self.attach_linear(&format!("{prefix}.W_Q"), &mut attn.w_q, scope)?;
                self.attach_linear(&format!("{prefix}.W_K"), &mut attn.w_k, scope)?;
                self.attach_linear(&format!("{prefix}.W_V"), &mut attn.w_v, scope)?;
                self.attach_linear(&format!("{prefix}.W_O"), &mut attn.w_o, scope)
            }
        }
    }

    fn attach_dense(
        &mut self,
        prefix: &str,
        ffn: &mut DenseFfn,
        scope: TrainableScope,
    ) -> Result<()> {
        self.attach_linear(&format!("{prefix}.W_gate"), &mut ffn.w_gate, scope)?;
        self.attach_linear(&format!("{prefix}.W_up"), &mut ffn.w_up, scope)?;
        self.attach_linear(&format!("{prefix}.W_down"), &mut ffn.w_down, scope)
    }

    fn attach_ffn(&mut self, prefix: &str, ffn: &mut FfnKind, scope: TrainableScope) -> Result<()> {
        match ffn {
            FfnKind::Dense(ffn) => self.attach_dense(prefix, ffn, scope),
            FfnKind::Moe(ffn) => {
                for (idx, expert) in ffn.shared_experts.iter_mut().enumerate() {
                    self.attach_dense(&format!("{prefix}.shared.{idx}"), expert, scope)?;
                }
                for (idx, expert) in ffn.routed_experts.iter_mut().enumerate() {
                    self.attach_dense(&format!("{prefix}.expert.{idx}"), expert, scope)?;
                }
                self.attach_linear(&format!("{prefix}.router"), &mut ffn.router, scope)
            }
        }
    }

    fn attach_vision_encoder(
        &mut self,
        prefix: &str,
        encoder: &mut NativeVisionEncoder,
    ) -> Result<()> {
        self.attach_plain(
            &format!("{prefix}.patch_proj"),
            &mut encoder.patch_proj,
            true,
        )?;
        self.attach_tensor(format!("{prefix}.cls_token"), &mut encoder.cls_token)?;
        self.attach_tensor(format!("{prefix}.pos_embed"), &mut encoder.pos_embed)?;
        for (idx, block) in encoder.blocks.iter_mut().enumerate() {
            self.attach_vision_block(&format!("{prefix}.blocks.{idx}"), block)?;
        }
        self.attach_tensor(format!("{prefix}.norm.weight"), &mut encoder.norm.weight)
    }

    fn attach_vision_block(
        &mut self,
        prefix: &str,
        block: &mut VisionTransformerBlock,
    ) -> Result<()> {
        self.attach_tensor(format!("{prefix}.norm1.weight"), &mut block.norm1.weight)?;
        self.attach_tensor(format!("{prefix}.norm2.weight"), &mut block.norm2.weight)?;
        self.attach_plain(&format!("{prefix}.W_Q"), &mut block.w_q, true)?;
        self.attach_plain(&format!("{prefix}.W_K"), &mut block.w_k, true)?;
        self.attach_plain(&format!("{prefix}.W_V"), &mut block.w_v, true)?;
        self.attach_plain(&format!("{prefix}.W_O"), &mut block.w_o, true)?;
        self.attach_plain(&format!("{prefix}.mlp.fc1"), &mut block.fc1, true)?;
        self.attach_plain(&format!("{prefix}.mlp.fc2"), &mut block.fc2, true)
    }

    fn attach_projector(
        &mut self,
        prefix: &str,
        projector: &mut VisionToTextProjector,
    ) -> Result<()> {
        self.attach_plain(&format!("{prefix}.input"), &mut projector.input_proj, true)?;
        for (idx, layer) in projector.hidden_layers.iter_mut().enumerate() {
            self.attach_plain(&format!("{prefix}.hidden.{idx}"), layer, true)?;
        }
        self.attach_plain(
            &format!("{prefix}.output"),
            &mut projector.output_proj,
            true,
        )?;
        if let Some(latents) = &mut projector.latents {
            self.attach_tensor(format!("{prefix}.latents"), latents)?;
        }
        Ok(())
    }
}

/// Chooses full-model or adapter-only optimization from the config state.
pub fn scope_for_model(model: &ApexModel) -> TrainableScope {
    if model.config.peft.enabled {
        TrainableScope::AdaptersOnly
    } else {
        TrainableScope::Full
    }
}
