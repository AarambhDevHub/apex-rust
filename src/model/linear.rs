use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::config::{ApexConfig, PeftConfig, PeftMethod};
use crate::error::{ApexError, Result};
use crate::tensor;

/// Standard trainable linear projection.
#[derive(Clone)]
pub struct PlainLinear {
    /// Stable module name used for targeting and checkpoint keys.
    pub name: String,
    /// Weight matrix with shape `[out_features, in_features]`.
    pub weight: Tensor,
    /// Optional output bias.
    pub bias: Option<Tensor>,
    /// Whether the base weight counts as trainable under PEFT.
    pub trainable: bool,
}

impl PlainLinear {
    /// Creates a normally initialized linear layer.
    pub fn new(
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            weight: tensor::randn(&[out_features, in_features], 0.0, 0.02, device)?,
            bias: if bias {
                Some(tensor::zeros(&[out_features], device)?)
            } else {
                None
            },
            trainable: true,
        })
    }

    /// Applies the linear projection to the last dimension of `x`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        tensor::linear(x, &self.weight, self.bias.as_ref())
    }

    /// Returns the number of stored weight and bias parameters.
    pub fn parameters(&self) -> usize {
        self.weight.elem_count() + self.bias.as_ref().map(Tensor::elem_count).unwrap_or(0)
    }

    /// Appends this layer's tensors to a named checkpoint list.
    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        out.push((format!("{prefix}.weight"), self.weight.clone()));
        if let Some(bias) = &self.bias {
            out.push((format!("{prefix}.bias"), bias.clone()));
        }
    }
}

/// Row-wise 4-bit quantized weight payload.
#[derive(Clone, Serialize, Deserialize)]
pub struct QuantizedWeight4Bit {
    /// Packed two-per-byte 4-bit codebook indices.
    pub qweight: Vec<u8>,
    /// Original matrix shape `(out_features, in_features)`.
    pub shape: (usize, usize),
    /// Codebook type, usually `nf4` or `fp4`.
    pub quant_type: String,
    /// Whether row scales are themselves quantized.
    pub double_quant: bool,
    /// Per-row scales when double quantization is disabled.
    pub scales: Vec<f32>,
    /// Quantized per-row scales when double quantization is enabled.
    pub scale_q: Vec<u8>,
    /// Shared scale used to reconstruct `scale_q`.
    pub scale_scale: f32,
}

const NF4_CODEBOOK: [f32; 16] = [
    -1.0000000, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0000000,
    0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.562_617, 0.7229568, 1.0000000,
];

/// Returns the 16-value codebook for the requested 4-bit quantization type.
pub fn codebook(quant_type: &str) -> Result<[f32; 16]> {
    match quant_type {
        "nf4" => Ok(NF4_CODEBOOK),
        "fp4" => {
            let mut out = [0.0; 16];
            for (i, v) in out.iter_mut().enumerate() {
                *v = -1.0 + 2.0 * i as f32 / 15.0;
            }
            Ok(out)
        }
        other => Err(ApexError::Model(format!("unknown quant_type {other}"))),
    }
}

/// Packs 4-bit indices into bytes using low/high nibbles.
pub fn pack_4bit_indices(indices: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len().div_ceil(2));
    for pair in indices.chunks(2) {
        let low = pair[0] & 0x0f;
        let high = pair.get(1).copied().unwrap_or(0) & 0x0f;
        out.push(low | (high << 4));
    }
    out
}

/// Unpacks byte-packed 4-bit indices into one index per value.
pub fn unpack_4bit_indices(packed: &[u8], num_values: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        out.push(byte & 0x0f);
        out.push((byte >> 4) & 0x0f);
    }
    out.truncate(num_values);
    out
}

/// Quantizes a rank-2 weight tensor row-wise into 4-bit codebook indices.
pub fn quantize_4bit_weight(
    weight: &Tensor,
    quant_type: &str,
    double_quant: bool,
) -> Result<QuantizedWeight4Bit> {
    let dims = weight.dims();
    if dims.len() != 2 {
        return Err(ApexError::Shape(format!(
            "quantize_4bit_weight expects rank 2, got {dims:?}"
        )));
    }
    let (out_features, in_features) = (dims[0], dims[1]);
    let rows = weight.to_vec2::<f32>()?;
    let cb = codebook(quant_type)?;
    let mut indices = Vec::with_capacity(out_features * in_features);
    let mut scales = Vec::with_capacity(out_features);
    for row in rows {
        let scale = row.iter().fold(1e-8_f32, |a, b| a.max(b.abs()));
        scales.push(scale);
        for v in row {
            let normalized = (v / scale).clamp(-1.0, 1.0);
            let mut best = 0u8;
            let mut best_dist = f32::INFINITY;
            for (idx, &q) in cb.iter().enumerate() {
                let dist = (normalized - q).abs();
                if dist < best_dist {
                    best = idx as u8;
                    best_dist = dist;
                }
            }
            indices.push(best);
        }
    }
    let qweight = pack_4bit_indices(&indices);
    let scale_scale = scales.iter().copied().fold(1e-8_f32, f32::max);
    let scale_q = if double_quant {
        scales
            .iter()
            .map(|s| ((*s / scale_scale * 255.0).round().clamp(0.0, 255.0)) as u8)
            .collect()
    } else {
        Vec::new()
    };
    Ok(QuantizedWeight4Bit {
        qweight,
        shape: (out_features, in_features),
        quant_type: quant_type.to_string(),
        double_quant,
        scales: if double_quant { Vec::new() } else { scales },
        scale_q,
        scale_scale,
    })
}

/// Reconstructs a floating-point weight tensor from a 4-bit payload.
pub fn dequantize_4bit_weight(q: &QuantizedWeight4Bit, device: &Device) -> Result<Tensor> {
    let cb = codebook(&q.quant_type)?;
    let (out_features, in_features) = q.shape;
    let n = out_features * in_features;
    let indices = unpack_4bit_indices(&q.qweight, n);
    let mut values = Vec::with_capacity(n);
    for row in 0..out_features {
        let scale = if q.double_quant {
            f32::from(q.scale_q[row]) / 255.0 * q.scale_scale
        } else {
            q.scales[row]
        };
        for col in 0..in_features {
            values.push(cb[indices[row * in_features + col] as usize] * scale);
        }
    }
    Ok(Tensor::from_vec(
        values,
        (out_features, in_features),
        device,
    )?)
}

/// Linear layer backed by a 4-bit quantized base weight.
#[derive(Clone)]
pub struct QuantizedLinear4Bit {
    /// Stable module name used for checkpoint keys.
    pub name: String,
    /// Quantized base weight.
    pub weight: QuantizedWeight4Bit,
    /// Optional bias stored in floating point.
    pub bias: Option<Tensor>,
}

impl QuantizedLinear4Bit {
    /// Quantizes a plain linear layer into a 4-bit layer.
    pub fn from_plain(base: &PlainLinear, quant_type: &str, double_quant: bool) -> Result<Self> {
        Ok(Self {
            name: base.name.clone(),
            weight: quantize_4bit_weight(&base.weight, quant_type, double_quant)?,
            bias: base.bias.clone(),
        })
    }

    /// Dequantizes the base weight on the requested device.
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        dequantize_4bit_weight(&self.weight, device)
    }

    /// Applies the dequantized linear projection.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.dequantize(x.device())?;
        tensor::linear(x, &weight, self.bias.as_ref())
    }

    /// Returns original bytes, quantized bytes, and compression ratio.
    pub fn storage_summary(&self) -> (usize, usize, f64) {
        let float_bytes = self.weight.shape.0 * self.weight.shape.1 * 4;
        let mut quant_bytes = self.weight.qweight.len();
        if self.weight.double_quant {
            quant_bytes += self.weight.scale_q.len() + 4;
        } else {
            quant_bytes += self.weight.scales.len() * 4;
        }
        let ratio = float_bytes as f64 / quant_bytes.max(1) as f64;
        (float_bytes, quant_bytes, ratio)
    }
}

/// Base layer used below LoRA/DoRA adapters.
#[derive(Clone)]
pub enum BaseLinear {
    /// Floating-point base projection.
    Plain(PlainLinear),
    /// Four-bit quantized base projection.
    Quantized(QuantizedLinear4Bit),
}

impl BaseLinear {
    /// Applies the base projection.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(p) => p.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }

    /// Returns the floating-point weight, dequantizing if necessary.
    pub fn weight(&self, device: &Device) -> Result<Tensor> {
        match self {
            Self::Plain(p) => Ok(p.weight.clone()),
            Self::Quantized(q) => q.dequantize(device),
        }
    }

    /// Returns the optional base bias.
    pub fn bias(&self) -> Option<&Tensor> {
        match self {
            Self::Plain(p) => p.bias.as_ref(),
            Self::Quantized(q) => q.bias.as_ref(),
        }
    }

    /// Converts the base into a plain floating-point layer.
    pub fn into_plain(self, device: &Device) -> Result<PlainLinear> {
        match self {
            Self::Plain(p) => Ok(p),
            Self::Quantized(q) => {
                let name = q.name.clone();
                let weight = q.dequantize(device)?;
                Ok(PlainLinear {
                    name,
                    weight,
                    bias: q.bias,
                    trainable: false,
                })
            }
        }
    }
}

/// Linear layer that may include LoRA, QLoRA, DoRA, or QDoRA adapters.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum LinearLayer {
    /// Plain floating-point linear projection.
    Plain(PlainLinear),
    /// Adapter-wrapped base projection.
    Lora {
        /// Frozen or trainable base layer.
        base: BaseLinear,
        /// Low-rank input projection.
        lora_a: PlainLinear,
        /// Low-rank output projection.
        lora_b: PlainLinear,
        /// Adapter scale `alpha / rank`.
        scaling: f64,
        /// Whether adapter weights have been merged into the base.
        merged: bool,
        /// PEFT method used by this layer.
        method: PeftMethod,
        /// DoRA row magnitudes, present for DoRA/QDoRA.
        dora_magnitude: Option<Tensor>,
    },
}

impl LinearLayer {
    /// Creates a linear layer and wraps it with PEFT adapters when targeted.
    pub fn new(
        name: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
        cfg: &ApexConfig,
        device: &Device,
    ) -> Result<Self> {
        let base = PlainLinear::new(name, in_features, out_features, bias, device)?;
        if cfg.peft.enabled && matches_target(name, &cfg.peft.target_modules) {
            Self::wrap(base, &cfg.peft, device)
        } else {
            Ok(Self::Plain(base))
        }
    }

    /// Builds the requested adapter wrapper around a plain base layer.
    fn wrap(base: PlainLinear, peft: &PeftConfig, device: &Device) -> Result<Self> {
        let in_features = base.weight.dim(1)?;
        let out_features = base.weight.dim(0)?;
        let quantized = matches!(peft.method, PeftMethod::Qlora | PeftMethod::Qdora);
        let base = if quantized {
            BaseLinear::Quantized(QuantizedLinear4Bit::from_plain(
                &base,
                &peft.quant_type,
                peft.double_quant,
            )?)
        } else {
            BaseLinear::Plain(PlainLinear {
                trainable: !peft.freeze_base_model,
                ..base
            })
        };
        let lora_a = PlainLinear::new("lora_A", in_features, peft.r, false, device)?;
        let mut lora_b = PlainLinear::new("lora_B", peft.r, out_features, false, device)?;
        lora_b.weight = tensor::zeros(&[out_features, peft.r], device)?;
        let dora_magnitude = if matches!(peft.method, PeftMethod::Dora | PeftMethod::Qdora) {
            let weight = base.weight(device)?;
            let rows = weight.to_vec2::<f32>()?;
            let mags: Vec<f32> = rows
                .iter()
                .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
                .collect();
            Some(Tensor::from_vec(mags, (out_features, 1), device)?)
        } else {
            None
        };
        Ok(Self::Lora {
            base,
            lora_a,
            lora_b,
            scaling: peft.alpha as f64 / peft.r as f64,
            merged: false,
            method: peft.method.clone(),
            dora_magnitude,
        })
    }

    /// Applies the base projection plus active adapter update.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(p) => p.forward(x),
            Self::Lora {
                base,
                lora_a,
                lora_b,
                scaling,
                merged,
                method,
                dora_magnitude,
            } => {
                if *merged {
                    return base.forward(x);
                }
                if matches!(method, PeftMethod::Dora | PeftMethod::Qdora) {
                    let base_w = base.weight(x.device())?;
                    let delta = lora_b
                        .weight
                        .matmul(&lora_a.weight)?
                        .broadcast_mul(&tensor::scalar(*scaling, x.device())?)?;
                    let adapted = base_w.broadcast_add(&delta)?;
                    let direction = tensor::l2_normalize_rows(&adapted, 1e-6)?;
                    let mag = dora_magnitude.as_ref().ok_or_else(|| {
                        ApexError::Model("DoRA layer missing dora_magnitude".to_string())
                    })?;
                    let dora_w = direction.broadcast_mul(mag)?;
                    tensor::linear(x, &dora_w, base.bias())
                } else {
                    let result = base.forward(x)?;
                    let update = lora_b
                        .forward(&lora_a.forward(x)?)?
                        .broadcast_mul(&tensor::scalar(*scaling, x.device())?)?;
                    Ok(result.broadcast_add(&update)?)
                }
            }
        }
    }

    /// Returns true when this layer carries a LoRA-family adapter.
    pub fn is_lora(&self) -> bool {
        matches!(self, Self::Lora { .. })
    }

    /// Returns true when the adapter uses a quantized base layer.
    pub fn is_quantized_adapter(&self) -> bool {
        matches!(
            self,
            Self::Lora {
                base: BaseLinear::Quantized(_),
                ..
            }
        )
    }

    /// Returns true when this layer uses DoRA-style magnitudes.
    pub fn is_dora(&self) -> bool {
        matches!(
            self,
            Self::Lora {
                method: PeftMethod::Dora | PeftMethod::Qdora,
                ..
            }
        )
    }

    /// Merges adapter weights into a plain base layer and removes adapter state.
    pub fn merge_and_unload(&mut self) -> Result<()> {
        let replacement = match self.clone() {
            Self::Plain(_) => return Ok(()),
            Self::Lora {
                base,
                lora_a,
                lora_b,
                scaling,
                method,
                dora_magnitude,
                ..
            } => {
                let device = lora_a.weight.device().clone();
                let mut plain = base.into_plain(&device)?;
                if matches!(method, PeftMethod::Dora | PeftMethod::Qdora) {
                    let delta = lora_b
                        .weight
                        .matmul(&lora_a.weight)?
                        .broadcast_mul(&tensor::scalar(scaling, &device)?)?;
                    let adapted = plain.weight.broadcast_add(&delta)?;
                    let direction = tensor::l2_normalize_rows(&adapted, 1e-6)?;
                    let mag = dora_magnitude.ok_or_else(|| {
                        ApexError::Model("DoRA layer missing dora_magnitude".to_string())
                    })?;
                    plain.weight = direction.broadcast_mul(&mag)?;
                } else {
                    let delta = lora_b
                        .weight
                        .matmul(&lora_a.weight)?
                        .broadcast_mul(&tensor::scalar(scaling, &device)?)?;
                    plain.weight = plain.weight.broadcast_add(&delta)?;
                }
                Self::Plain(plain)
            }
        };
        *self = replacement;
        Ok(())
    }

    /// Returns total represented parameters including frozen base parameters.
    pub fn parameters(&self) -> usize {
        match self {
            Self::Plain(p) => p.parameters(),
            Self::Lora {
                base,
                lora_a,
                lora_b,
                dora_magnitude,
                ..
            } => {
                let base_params = match base {
                    BaseLinear::Plain(p) => p.parameters(),
                    BaseLinear::Quantized(q) => q.weight.shape.0 * q.weight.shape.1,
                };
                base_params
                    + lora_a.parameters()
                    + lora_b.parameters()
                    + dora_magnitude.as_ref().map(Tensor::elem_count).unwrap_or(0)
            }
        }
    }

    /// Returns parameters that are trainable under the current adapter policy.
    pub fn trainable_parameters(&self) -> usize {
        match self {
            Self::Plain(p) => {
                if p.trainable {
                    p.parameters()
                } else {
                    0
                }
            }
            Self::Lora {
                lora_a,
                lora_b,
                dora_magnitude,
                ..
            } => {
                lora_a.parameters()
                    + lora_b.parameters()
                    + dora_magnitude.as_ref().map(Tensor::elem_count).unwrap_or(0)
            }
        }
    }

    /// Appends full layer tensors to a named checkpoint list.
    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) -> Result<()> {
        match self {
            Self::Plain(p) => p.named_tensors(prefix, out),
            Self::Lora {
                base,
                lora_a,
                lora_b,
                dora_magnitude,
                ..
            } => {
                match base {
                    BaseLinear::Plain(p) => p.named_tensors(&format!("{prefix}.base"), out),
                    BaseLinear::Quantized(q) => {
                        out.push((
                            format!("{prefix}.base.weight"),
                            q.dequantize(lora_a.weight.device())?,
                        ));
                        if let Some(bias) = &q.bias {
                            out.push((format!("{prefix}.base.bias"), bias.clone()));
                        }
                    }
                }
                lora_a.named_tensors(&format!("{prefix}.lora_A"), out);
                lora_b.named_tensors(&format!("{prefix}.lora_B"), out);
                if let Some(mag) = dora_magnitude {
                    out.push((format!("{prefix}.dora_magnitude"), mag.clone()));
                }
            }
        }
        Ok(())
    }

    /// Appends only adapter tensors to a named checkpoint list.
    pub fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        if let Self::Lora {
            lora_a,
            lora_b,
            dora_magnitude,
            ..
        } = self
        {
            lora_a.named_tensors(&format!("{prefix}.lora_A"), out);
            lora_b.named_tensors(&format!("{prefix}.lora_B"), out);
            if let Some(mag) = dora_magnitude {
                out.push((format!("{prefix}.dora_magnitude"), mag.clone()));
            }
        }
    }
}

fn matches_target(name: &str, targets: &[String]) -> bool {
    targets
        .iter()
        .any(|target| name == target || name.ends_with(target) || name.contains(target))
}
