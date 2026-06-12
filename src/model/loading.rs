//! Loading named tensors back into model and adapter structures.

use std::collections::{HashMap, HashSet};

use candle_core::Tensor;

use crate::error::{ApexError, Result};

use super::attention::{AttentionKind, GqaAttention, MlaAttention};
use super::ffn::{DenseFfn, FfnKind, MoeFfn};
use super::linear::{quantize_4bit_weight, BaseLinear, LinearLayer, PlainLinear};
use super::ApexModel;

/// Summary returned after applying tensors to a model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorLoadReport {
    /// Number of tensors copied into model fields.
    pub loaded: usize,
    /// Expected tensors missing from the input file.
    pub missing: Vec<String>,
    /// Input tensors that did not match any loadable model field.
    pub unexpected: Vec<String>,
}

/// Loads a full model tensor map into an existing model instance.
pub fn load_full_tensors(
    model: &mut ApexModel,
    tensors: &HashMap<String, Tensor>,
    strict: bool,
) -> Result<TensorLoadReport> {
    load_model_tensors(model, tensors, LoadMode::Full, strict)
}

/// Loads adapter-only tensors into an existing adapter-enabled model instance.
pub fn load_adapter_tensors(
    model: &mut ApexModel,
    tensors: &HashMap<String, Tensor>,
    strict: bool,
) -> Result<TensorLoadReport> {
    load_model_tensors(model, tensors, LoadMode::AdapterOnly, strict)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadMode {
    Full,
    AdapterOnly,
}

fn load_model_tensors(
    model: &mut ApexModel,
    tensors: &HashMap<String, Tensor>,
    mode: LoadMode,
    strict: bool,
) -> Result<TensorLoadReport> {
    let mut state = LoadState::new(tensors, strict);
    if mode == LoadMode::Full {
        state.load_tensor("embedding.weight", &mut model.embedding)?;
        state.load_tensor("final_norm.weight", &mut model.final_norm.weight)?;
    }
    for (idx, block) in model.blocks.iter_mut().enumerate() {
        load_block(&format!("blocks.{idx}"), block, &mut state, mode)?;
    }
    if mode == LoadMode::Full {
        if let Some(head) = &mut model.multi_token_head {
            for (idx, layer) in head.heads.iter_mut().enumerate() {
                load_plain_linear(&format!("multi_token_head.{idx}"), layer, &mut state)?;
            }
        }
    }
    Ok(state.finish())
}

fn load_block(
    prefix: &str,
    block: &mut super::TransformerBlock,
    state: &mut LoadState<'_>,
    mode: LoadMode,
) -> Result<()> {
    if mode == LoadMode::Full {
        state.load_tensor(&format!("{prefix}.norm1.weight"), &mut block.norm1.weight)?;
        state.load_tensor(&format!("{prefix}.norm2.weight"), &mut block.norm2.weight)?;
    }
    load_attention(&format!("{prefix}.attn"), &mut block.attn, state, mode)?;
    load_ffn(&format!("{prefix}.ffn"), &mut block.ffn, state, mode)?;
    if mode == LoadMode::Full {
        if let Some(gate) = &mut block.skip_gate {
            load_plain_linear(&format!("{prefix}.skip_gate.fc1"), &mut gate.fc1, state)?;
            load_plain_linear(&format!("{prefix}.skip_gate.fc2"), &mut gate.fc2, state)?;
        }
    }
    Ok(())
}

fn load_attention(
    prefix: &str,
    attention: &mut AttentionKind,
    state: &mut LoadState<'_>,
    mode: LoadMode,
) -> Result<()> {
    match attention {
        AttentionKind::Mla(attn) => load_mla(prefix, attn, state, mode),
        AttentionKind::Gqa(attn) => load_gqa(prefix, attn, state, mode),
    }
}

fn load_mla(
    prefix: &str,
    attn: &mut MlaAttention,
    state: &mut LoadState<'_>,
    mode: LoadMode,
) -> Result<()> {
    for (name, layer) in [
        ("W_DKV", &mut attn.w_dkv),
        ("W_UK", &mut attn.w_uk),
        ("W_UV", &mut attn.w_uv),
        ("W_DQ", &mut attn.w_dq),
        ("W_UQ", &mut attn.w_uq),
        ("W_KR", &mut attn.w_kr),
        ("W_QR", &mut attn.w_qr),
        ("W_O", &mut attn.w_o),
    ] {
        load_linear_layer(&format!("{prefix}.{name}"), layer, state, mode)?;
    }
    Ok(())
}

fn load_gqa(
    prefix: &str,
    attn: &mut GqaAttention,
    state: &mut LoadState<'_>,
    mode: LoadMode,
) -> Result<()> {
    for (name, layer) in [
        ("W_Q", &mut attn.w_q),
        ("W_K", &mut attn.w_k),
        ("W_V", &mut attn.w_v),
        ("W_O", &mut attn.w_o),
    ] {
        load_linear_layer(&format!("{prefix}.{name}"), layer, state, mode)?;
    }
    Ok(())
}

fn load_ffn(
    prefix: &str,
    ffn: &mut FfnKind,
    state: &mut LoadState<'_>,
    mode: LoadMode,
) -> Result<()> {
    match ffn {
        FfnKind::Dense(ffn) => load_dense_ffn(prefix, ffn, state, mode),
        FfnKind::Moe(ffn) => load_moe_ffn(prefix, ffn, state, mode),
    }
}

fn load_dense_ffn(
    prefix: &str,
    ffn: &mut DenseFfn,
    state: &mut LoadState<'_>,
    mode: LoadMode,
) -> Result<()> {
    for (name, layer) in [
        ("W_gate", &mut ffn.w_gate),
        ("W_up", &mut ffn.w_up),
        ("W_down", &mut ffn.w_down),
    ] {
        load_linear_layer(&format!("{prefix}.{name}"), layer, state, mode)?;
    }
    Ok(())
}

fn load_moe_ffn(
    prefix: &str,
    ffn: &mut MoeFfn,
    state: &mut LoadState<'_>,
    mode: LoadMode,
) -> Result<()> {
    for (idx, expert) in ffn.shared_experts.iter_mut().enumerate() {
        load_dense_ffn(&format!("{prefix}.shared.{idx}"), expert, state, mode)?;
    }
    for (idx, expert) in ffn.routed_experts.iter_mut().enumerate() {
        load_dense_ffn(&format!("{prefix}.expert.{idx}"), expert, state, mode)?;
    }
    load_linear_layer(&format!("{prefix}.router"), &mut ffn.router, state, mode)
}

fn load_linear_layer(
    prefix: &str,
    layer: &mut LinearLayer,
    state: &mut LoadState<'_>,
    mode: LoadMode,
) -> Result<()> {
    match layer {
        LinearLayer::Plain(plain) => {
            if mode == LoadMode::Full {
                load_plain_linear(prefix, plain, state)?;
            }
        }
        LinearLayer::Lora {
            base,
            lora_a,
            lora_b,
            dora_magnitude,
            ..
        } => {
            if mode == LoadMode::Full {
                load_base_linear(&format!("{prefix}.base"), base, state)?;
            }
            load_plain_linear(&format!("{prefix}.lora_A"), lora_a, state)?;
            load_plain_linear(&format!("{prefix}.lora_B"), lora_b, state)?;
            if let Some(mag) = dora_magnitude {
                state.load_tensor(&format!("{prefix}.dora_magnitude"), mag)?;
            }
        }
    }
    Ok(())
}

fn load_base_linear(prefix: &str, base: &mut BaseLinear, state: &mut LoadState<'_>) -> Result<()> {
    match base {
        BaseLinear::Plain(plain) => load_plain_linear(prefix, plain, state),
        BaseLinear::Quantized(quantized) => {
            let name = format!("{prefix}.weight");
            if let Some(tensor) = state.get_tensor(&name)? {
                let dims = tensor.dims();
                if dims != [quantized.weight.shape.0, quantized.weight.shape.1] {
                    return Err(ApexError::Shape(format!(
                        "{name} shape {dims:?} does not match quantized base shape {:?}",
                        quantized.weight.shape
                    )));
                }
                quantized.weight = quantize_4bit_weight(
                    &tensor,
                    &quantized.weight.quant_type,
                    quantized.weight.double_quant,
                )?;
                state.mark_loaded(name);
            } else if state.strict {
                state.missing.push(name);
            }
            if let Some(bias) = &mut quantized.bias {
                state.load_tensor(&format!("{prefix}.bias"), bias)?;
            }
            Ok(())
        }
    }
}

fn load_plain_linear(
    prefix: &str,
    layer: &mut PlainLinear,
    state: &mut LoadState<'_>,
) -> Result<()> {
    state.load_tensor(&format!("{prefix}.weight"), &mut layer.weight)?;
    if let Some(bias) = &mut layer.bias {
        state.load_tensor(&format!("{prefix}.bias"), bias)?;
    }
    Ok(())
}

struct LoadState<'a> {
    tensors: &'a HashMap<String, Tensor>,
    loaded: HashSet<String>,
    missing: Vec<String>,
    strict: bool,
}

impl<'a> LoadState<'a> {
    fn new(tensors: &'a HashMap<String, Tensor>, strict: bool) -> Self {
        Self {
            tensors,
            loaded: HashSet::new(),
            missing: Vec::new(),
            strict,
        }
    }

    fn get_tensor(&self, name: &str) -> Result<Option<Tensor>> {
        self.tensors
            .get(name)
            .map(|tensor| Ok(tensor.clone()))
            .transpose()
    }

    fn mark_loaded(&mut self, name: String) {
        self.loaded.insert(name);
    }

    fn load_tensor(&mut self, name: &str, target: &mut Tensor) -> Result<()> {
        if let Some(tensor) = self.get_tensor(name)? {
            if tensor.dims() != target.dims() {
                return Err(ApexError::Shape(format!(
                    "{name} shape {:?} does not match model shape {:?}",
                    tensor.dims(),
                    target.dims()
                )));
            }
            *target = tensor;
            self.loaded.insert(name.to_string());
        } else if self.strict {
            self.missing.push(name.to_string());
        }
        Ok(())
    }

    fn finish(self) -> TensorLoadReport {
        let unexpected = self
            .tensors
            .keys()
            .filter(|name| !self.loaded.contains(*name))
            .cloned()
            .collect();
        TensorLoadReport {
            loaded: self.loaded.len(),
            missing: self.missing,
            unexpected,
        }
    }
}
