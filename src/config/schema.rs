use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ApexError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub d_head: usize,
    pub d_kv_compressed: usize,
    pub d_q_compressed: usize,
    pub d_head_rope: usize,
    pub d_ffn: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f64,
    pub rope_scaling: f64,
    pub dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            n_layers: 12,
            n_heads_q: 8,
            n_heads_kv: 2,
            d_head: 64,
            d_kv_compressed: 64,
            d_q_compressed: 96,
            d_head_rope: 32,
            d_ffn: 1376,
            vocab_size: 151_643,
            max_seq_len: 8192,
            rope_base: 10_000.0,
            rope_scaling: 1.0,
            dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AttentionConfig {
    pub global_layer_freq: usize,
    pub local_window: usize,
    pub flash: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            global_layer_freq: 6,
            local_window: 512,
            flash: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MoeConfig {
    pub enabled: bool,
    pub n_experts: usize,
    pub n_active: usize,
    pub n_shared: usize,
    pub moe_layer_freq: usize,
    pub balancer_alpha: f64,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            n_experts: 8,
            n_active: 2,
            n_shared: 1,
            moe_layer_freq: 2,
            balancer_alpha: 0.001,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SkipGateConfig {
    pub enabled: bool,
    pub hidden_dim: usize,
    pub threshold: f64,
}

impl Default for SkipGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hidden_dim: 64,
            threshold: 0.15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MultiTokenHeadConfig {
    pub enabled: bool,
    pub n_predict: usize,
    pub lambda_spec: f64,
}

impl Default for MultiTokenHeadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            n_predict: 4,
            lambda_spec: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ThinkingConfig {
    pub enabled: bool,
    pub max_thinking_tokens: usize,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_thinking_tokens: 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionConfig {
    pub enabled: bool,
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub encoder_type: String,
    pub d_vision: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub mlp_ratio: f64,
    pub dropout: f64,
    pub projector_type: String,
    pub n_visual_tokens: usize,
    pub projector_hidden_dim: usize,
    pub projector_layers: usize,
    pub image_token_id: u32,
    pub freeze_vision_encoder: bool,
    pub freeze_language_model: bool,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            encoder_type: "native_vit".to_string(),
            d_vision: 512,
            n_layers: 6,
            n_heads: 8,
            mlp_ratio: 4.0,
            dropout: 0.0,
            projector_type: "perceiver".to_string(),
            n_visual_tokens: 64,
            projector_hidden_dim: 1024,
            projector_layers: 2,
            image_token_id: 8,
            freeze_vision_encoder: false,
            freeze_language_model: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PeftMethod {
    #[default]
    Lora,
    Qlora,
    Dora,
    Qdora,
}

impl std::fmt::Display for PeftMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Lora => "lora",
            Self::Qlora => "qlora",
            Self::Dora => "dora",
            Self::Qdora => "qdora",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PeftConfig {
    pub enabled: bool,
    pub method: PeftMethod,
    pub r: usize,
    pub alpha: usize,
    pub dropout: f64,
    pub freeze_base_model: bool,
    pub quantization_bits: usize,
    pub quant_type: String,
    pub double_quant: bool,
    pub compute_dtype: String,
    pub target_modules: Vec<String>,
    pub modules_to_save: Vec<String>,
    pub bias: String,
}

impl Default for PeftConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: PeftMethod::Lora,
            r: 8,
            alpha: 16,
            dropout: 0.05,
            freeze_base_model: true,
            quantization_bits: 4,
            quant_type: "nf4".to_string(),
            double_quant: true,
            compute_dtype: "float32".to_string(),
            target_modules: [
                "W_Q", "W_K", "W_V", "W_O", "W_DKV", "W_UK", "W_UV", "W_DQ", "W_UQ", "W_KR",
                "W_QR", "W_gate", "W_up", "W_down", "router",
            ]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
            modules_to_save: Vec::new(),
            bias: "none".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AdapterDpoConfig {
    pub enabled: bool,
    pub beta: f64,
    pub label_smoothing: f64,
    pub reference_free: bool,
    pub length_normalize: bool,
    pub max_prompt_len: usize,
    pub max_response_len: usize,
    pub save_every_steps: usize,
}

impl Default for AdapterDpoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            beta: 0.1,
            label_smoothing: 0.0,
            reference_free: false,
            length_normalize: false,
            max_prompt_len: 128,
            max_response_len: 128,
            save_every_steps: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub peak_lr: f64,
    pub min_lr_ratio: f64,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub grad_clip: f64,
    pub weight_decay: f64,
    pub optimizer: String,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub gradient_accumulation_steps: usize,
    pub mixed_precision: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            seq_len: 2048,
            peak_lr: 3e-4,
            min_lr_ratio: 0.1,
            warmup_steps: 1000,
            max_steps: 100_000,
            grad_clip: 1.0,
            weight_decay: 0.1,
            optimizer: "adamw".to_string(),
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            gradient_accumulation_steps: 1,
            mixed_precision: "fp16".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GrpoConfig {
    #[serde(rename = "G")]
    pub g: usize,
    pub beta: f64,
    pub lambda_prm: f64,
    pub lambda_cai: f64,
    pub clip_eps: f64,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            g: 8,
            beta: 0.04,
            lambda_prm: 0.3,
            lambda_cai: 0.3,
            clip_eps: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ApexConfig {
    pub model: ModelConfig,
    pub attention: AttentionConfig,
    pub moe: MoeConfig,
    pub skip_gate: SkipGateConfig,
    pub multi_token_head: MultiTokenHeadConfig,
    pub thinking: ThinkingConfig,
    pub vision: VisionConfig,
    pub peft: PeftConfig,
    pub training: TrainingConfig,
    pub grpo: GrpoConfig,
    pub adapter_dpo: AdapterDpoConfig,
}

impl ApexConfig {
    pub fn from_yaml(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let cfg: Self = serde_yaml::from_str(&text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_yaml(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_yaml::to_string(self)?)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        let m = &self.model;
        if m.n_heads_kv == 0 || !m.n_heads_q.is_multiple_of(m.n_heads_kv) {
            return Err(ApexError::Config(format!(
                "n_heads_q ({}) must be divisible by n_heads_kv ({})",
                m.n_heads_q, m.n_heads_kv
            )));
        }
        if self.attention.global_layer_freq == 0
            || !m.n_layers.is_multiple_of(self.attention.global_layer_freq)
        {
            return Err(ApexError::Config(format!(
                "n_layers ({}) should be divisible by global_layer_freq ({})",
                m.n_layers, self.attention.global_layer_freq
            )));
        }
        if m.d_model != m.n_heads_q * m.d_head {
            return Err(ApexError::Config(format!(
                "d_model ({}) must equal n_heads_q ({}) * d_head ({})",
                m.d_model, m.n_heads_q, m.d_head
            )));
        }
        if self.moe.enabled && self.moe.n_active > self.moe.n_experts {
            return Err(ApexError::Config(
                "n_active cannot exceed n_experts".to_string(),
            ));
        }
        if self.vision.enabled {
            let v = &self.vision;
            if v.image_size == 0 || v.patch_size == 0 || !v.image_size.is_multiple_of(v.patch_size)
            {
                return Err(ApexError::Config(
                    "vision.image_size must be divisible by positive vision.patch_size".to_string(),
                ));
            }
            if v.n_heads == 0 || !v.d_vision.is_multiple_of(v.n_heads) {
                return Err(ApexError::Config(
                    "vision.d_vision must be divisible by vision.n_heads".to_string(),
                ));
            }
            if v.n_visual_tokens == 0 {
                return Err(ApexError::Config(
                    "vision.n_visual_tokens must be positive".to_string(),
                ));
            }
            if v.projector_type != "perceiver" && v.projector_type != "mlp" {
                return Err(ApexError::Config(
                    "vision.projector_type must be 'perceiver' or 'mlp'".to_string(),
                ));
            }
            if v.encoder_type != "native_vit" {
                return Err(ApexError::Config(
                    "only vision.encoder_type='native_vit' is implemented".to_string(),
                ));
            }
            if self.training.seq_len + v.n_visual_tokens > m.max_seq_len {
                return Err(ApexError::Config(
                    "training.seq_len + vision.n_visual_tokens exceeds model.max_seq_len"
                        .to_string(),
                ));
            }
        }
        if self.peft.enabled {
            let p = &self.peft;
            if p.r == 0 || p.alpha == 0 {
                return Err(ApexError::Config(
                    "peft.r and peft.alpha must be positive".to_string(),
                ));
            }
            if !(0.0..1.0).contains(&p.dropout) {
                return Err(ApexError::Config(
                    "peft.dropout must be in [0, 1)".to_string(),
                ));
            }
            if !matches!(p.bias.as_str(), "none" | "all" | "lora_only") {
                return Err(ApexError::Config(
                    "peft.bias must be one of none, all, lora_only".to_string(),
                ));
            }
            if matches!(p.method, PeftMethod::Qlora | PeftMethod::Qdora) {
                if p.quantization_bits != 4 {
                    return Err(ApexError::Config(
                        "educational QLoRA/QDoRA supports only 4-bit quantization".to_string(),
                    ));
                }
                if p.quant_type != "nf4" && p.quant_type != "fp4" {
                    return Err(ApexError::Config(
                        "peft.quant_type must be nf4 or fp4".to_string(),
                    ));
                }
            }
            if p.target_modules.is_empty() {
                return Err(ApexError::Config(
                    "peft.target_modules must contain at least one module".to_string(),
                ));
            }
        }
        if self.adapter_dpo.enabled {
            if !self.peft.enabled {
                return Err(ApexError::Config(
                    "adapter_dpo.enabled requires peft.enabled".to_string(),
                ));
            }
            if self.adapter_dpo.beta <= 0.0 {
                return Err(ApexError::Config(
                    "adapter_dpo.beta must be positive".to_string(),
                ));
            }
            if !(0.0..0.5).contains(&self.adapter_dpo.label_smoothing) {
                return Err(ApexError::Config(
                    "adapter_dpo.label_smoothing must be in [0, 0.5)".to_string(),
                ));
            }
        }
        Ok(())
    }
}
