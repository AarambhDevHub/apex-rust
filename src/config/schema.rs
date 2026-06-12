use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ApexError, Result};

/// Transformer dimensions and sequence-level model settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    /// Hidden size used by token embeddings and transformer blocks.
    pub d_model: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads_q: usize,
    /// Number of key/value heads used by grouped-query attention.
    pub n_heads_kv: usize,
    /// Content head dimension.
    pub d_head: usize,
    /// Compressed key/value latent width used by MLA blocks.
    pub d_kv_compressed: usize,
    /// Compressed query latent width used by MLA blocks.
    pub d_q_compressed: usize,
    /// RoPE-only head dimension used by MLA blocks.
    pub d_head_rope: usize,
    /// Feed-forward hidden width.
    pub d_ffn: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length supported by RoPE caches.
    pub max_seq_len: usize,
    /// Base frequency for rotary position embeddings.
    pub rope_base: f64,
    /// YaRN/RoPE scaling factor for long-context variants.
    pub rope_scaling: f64,
    /// Dropout probability kept in the schema for training compatibility.
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

/// Attention routing settings for alternating global MLA and local GQA blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AttentionConfig {
    /// Every Nth block is a global MLA block.
    pub global_layer_freq: usize,
    /// Sliding-window size used by local GQA blocks.
    pub local_window: usize,
    /// Whether optimized attention kernels are requested when available.
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

/// Mixture-of-experts feed-forward configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MoeConfig {
    /// Enables routed expert layers.
    pub enabled: bool,
    /// Number of routed experts per MoE layer.
    pub n_experts: usize,
    /// Number of routed experts selected per token.
    pub n_active: usize,
    /// Number of shared experts always applied.
    pub n_shared: usize,
    /// Frequency that controls which blocks use dense FFN versus MoE FFN.
    pub moe_layer_freq: usize,
    /// Expert-bias update strength for load balancing.
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

/// Token-level gate that can skip feed-forward computation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SkipGateConfig {
    /// Enables the skip gate inside transformer blocks.
    pub enabled: bool,
    /// Hidden width of the two-layer gate MLP.
    pub hidden_dim: usize,
    /// Gate probability threshold below which the FFN contribution is dropped.
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

/// Auxiliary heads for predicting multiple future tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MultiTokenHeadConfig {
    /// Enables speculative auxiliary heads.
    pub enabled: bool,
    /// Number of future-token heads attached to the final hidden state.
    pub n_predict: usize,
    /// Weight applied to speculative losses during pretraining.
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

/// Settings for optional thinking-token generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ThinkingConfig {
    /// Allows generation to enter a thinking-token phase.
    pub enabled: bool,
    /// Maximum number of tokens allowed inside the thinking phase.
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

/// Runtime generation defaults used by inference commands.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GenerationRuntimeConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Default sampling temperature.
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    pub top_p: f64,
    /// Top-k sampling cutoff; zero disables top-k.
    pub top_k: usize,
    /// Repetition penalty applied to generated token logits.
    pub repetition_penalty: f64,
    /// Enables thinking-token controls without requiring a CLI flag.
    pub enable_thinking: bool,
    /// Maximum tokens allowed inside the thinking phase.
    pub max_thinking_tokens: usize,
    /// Sampling temperature inside the thinking phase.
    pub thinking_temperature: f64,
    /// Sampling temperature after the thinking phase.
    pub output_temperature: f64,
    /// Enables speculative decoding when multi-token heads exist.
    pub use_speculative: bool,
}

impl Default for GenerationRuntimeConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            enable_thinking: false,
            max_thinking_tokens: 1024,
            thinking_temperature: 0.6,
            output_temperature: 0.3,
            use_speculative: false,
        }
    }
}

/// Vision encoder and projector settings for multimodal runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionConfig {
    /// Enables the image pathway and visual-token insertion.
    pub enabled: bool,
    /// Square input image size expected by the native ViT encoder.
    pub image_size: usize,
    /// Square patch size used to flatten image patches.
    pub patch_size: usize,
    /// Number of image channels.
    pub in_channels: usize,
    /// Vision encoder implementation name.
    pub encoder_type: String,
    /// Vision token hidden size before projection into text space.
    pub d_vision: usize,
    /// Number of vision encoder layers reserved by the config.
    pub n_layers: usize,
    /// Number of vision attention heads reserved by the config.
    pub n_heads: usize,
    /// Vision MLP expansion ratio reserved by the config.
    pub mlp_ratio: f64,
    /// Vision dropout probability kept for compatibility.
    pub dropout: f64,
    /// Projector implementation, currently `perceiver` or `mlp`.
    pub projector_type: String,
    /// Number of projected visual tokens inserted for each image token.
    pub n_visual_tokens: usize,
    /// Hidden width of the text projector.
    pub projector_hidden_dim: usize,
    /// Number of projector MLP layers.
    pub projector_layers: usize,
    /// Token ID that is replaced by projected visual embeddings.
    pub image_token_id: u32,
    /// Marks the vision encoder as frozen for training policies.
    pub freeze_vision_encoder: bool,
    /// Marks the language model as frozen for vision-only tuning policies.
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

/// Supported parameter-efficient fine-tuning methods.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PeftMethod {
    /// Low-rank adapters over selected linear layers.
    #[default]
    Lora,
    /// LoRA adapters with a 4-bit quantized base linear layer.
    Qlora,
    /// Weight-decomposed LoRA with trainable row magnitudes.
    Dora,
    /// DoRA adapters with a 4-bit quantized base linear layer.
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

/// Parameter-efficient fine-tuning configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PeftConfig {
    /// Enables adapter insertion during model construction.
    pub enabled: bool,
    /// PEFT method used for matching target modules.
    pub method: PeftMethod,
    /// Adapter rank.
    pub r: usize,
    /// Adapter scaling numerator.
    pub alpha: usize,
    /// Adapter dropout probability kept for compatibility.
    pub dropout: f64,
    /// Freezes non-adapter base weights when enabled.
    pub freeze_base_model: bool,
    /// Quantization bit width for QLoRA/QDoRA.
    pub quantization_bits: usize,
    /// Four-bit codebook type, usually `nf4` or `fp4`.
    pub quant_type: String,
    /// Quantizes per-row scales when using QLoRA/QDoRA.
    pub double_quant: bool,
    /// Compute dtype label used by configuration and reports.
    pub compute_dtype: String,
    /// Linear module name fragments that receive adapters.
    pub target_modules: Vec<String>,
    /// Extra modules that should be saved with adapters.
    pub modules_to_save: Vec<String>,
    /// Bias handling policy: `none`, `all`, or `lora_only`.
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

/// Adapter-DPO alignment settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AdapterDpoConfig {
    /// Enables adapter-DPO training mode.
    pub enabled: bool,
    /// DPO reward temperature.
    pub beta: f64,
    /// Probability mass assigned to the negative DPO label.
    pub label_smoothing: f64,
    /// Uses zero reference scores instead of a reference model.
    pub reference_free: bool,
    /// Normalizes sequence log-probability by response length.
    pub length_normalize: bool,
    /// Maximum prompt tokens used by preference loaders.
    pub max_prompt_len: usize,
    /// Maximum response tokens used by preference loaders.
    pub max_response_len: usize,
    /// Optional checkpoint cadence for adapter-DPO loops.
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

/// Common single-process training hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrainingConfig {
    /// Number of examples per optimization step.
    pub batch_size: usize,
    /// Training sequence length before optional visual-token expansion.
    pub seq_len: usize,
    /// Peak learning rate after warmup.
    pub peak_lr: f64,
    /// Fraction of peak LR used as the cosine floor.
    pub min_lr_ratio: f64,
    /// Number of warmup steps.
    pub warmup_steps: usize,
    /// Total scheduled optimization steps.
    pub max_steps: usize,
    /// Gradient clipping threshold.
    pub grad_clip: f64,
    /// AdamW weight decay coefficient.
    pub weight_decay: f64,
    /// Optimizer name for CLI/config compatibility.
    pub optimizer: String,
    /// Adam beta1 value.
    pub beta1: f64,
    /// Adam beta2 value.
    pub beta2: f64,
    /// Adam epsilon value.
    pub eps: f64,
    /// Number of batches accumulated per update.
    pub gradient_accumulation_steps: usize,
    /// Mixed-precision mode label.
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

/// Group relative policy optimization settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GrpoConfig {
    /// Number of generated samples per prompt.
    #[serde(rename = "G")]
    pub g: usize,
    /// KL penalty coefficient.
    pub beta: f64,
    /// Process reward model loss weight.
    pub lambda_prm: f64,
    /// Constitutional AI loss weight.
    pub lambda_cai: f64,
    /// PPO-style clipping range.
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

/// Complete APEX runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ApexConfig {
    /// Transformer dimensions and vocabulary settings.
    pub model: ModelConfig,
    /// Attention pattern and local-window settings.
    pub attention: AttentionConfig,
    /// Mixture-of-experts settings.
    pub moe: MoeConfig,
    /// Feed-forward skip-gate settings.
    pub skip_gate: SkipGateConfig,
    /// Speculative multi-token-head settings.
    pub multi_token_head: MultiTokenHeadConfig,
    /// Thinking-token generation settings.
    pub thinking: ThinkingConfig,
    /// Inference-time generation defaults.
    pub generation: GenerationRuntimeConfig,
    /// Multimodal vision settings.
    pub vision: VisionConfig,
    /// Adapter/PEFT settings.
    pub peft: PeftConfig,
    /// Training hyperparameters.
    pub training: TrainingConfig,
    /// GRPO hyperparameters.
    pub grpo: GrpoConfig,
    /// Adapter-DPO hyperparameters.
    pub adapter_dpo: AdapterDpoConfig,
}

impl ApexConfig {
    /// Loads a YAML config file and validates cross-field constraints.
    pub fn from_yaml(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let cfg: Self = serde_yaml::from_str(&text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Serializes the config as YAML, creating parent directories if needed.
    pub fn to_yaml(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_yaml::to_string(self)?)?;
        Ok(())
    }

    /// Validates dimensions, feature dependencies, and supported option values.
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
        let generation = &self.generation;
        if generation.max_new_tokens == 0 {
            return Err(ApexError::Config(
                "generation.max_new_tokens must be positive".to_string(),
            ));
        }
        if generation.temperature < 0.0
            || generation.thinking_temperature < 0.0
            || generation.output_temperature < 0.0
        {
            return Err(ApexError::Config(
                "generation temperatures must be non-negative".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&generation.top_p) {
            return Err(ApexError::Config(
                "generation.top_p must be in [0, 1]".to_string(),
            ));
        }
        if generation.repetition_penalty <= 0.0 {
            return Err(ApexError::Config(
                "generation.repetition_penalty must be positive".to_string(),
            ));
        }
        if generation.max_thinking_tokens == 0 {
            return Err(ApexError::Config(
                "generation.max_thinking_tokens must be positive".to_string(),
            ));
        }
        Ok(())
    }
}
