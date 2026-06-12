use super::schema::*;

/// Builds the smallest complete configuration used by tests and examples.
pub fn get_tiny_config() -> ApexConfig {
    ApexConfig {
        model: ModelConfig {
            d_model: 64,
            n_layers: 6,
            n_heads_q: 4,
            n_heads_kv: 2,
            d_head: 16,
            d_kv_compressed: 16,
            d_q_compressed: 24,
            d_head_rope: 8,
            d_ffn: 128,
            vocab_size: 1000,
            max_seq_len: 256,
            rope_base: 10_000.0,
            rope_scaling: 1.0,
            dropout: 0.0,
        },
        attention: AttentionConfig {
            global_layer_freq: 6,
            local_window: 64,
            flash: false,
        },
        moe: MoeConfig {
            n_experts: 4,
            n_active: 2,
            ..MoeConfig::default()
        },
        skip_gate: SkipGateConfig {
            hidden_dim: 16,
            ..SkipGateConfig::default()
        },
        thinking: ThinkingConfig {
            max_thinking_tokens: 64,
            ..ThinkingConfig::default()
        },
        vision: VisionConfig {
            image_size: 32,
            patch_size: 16,
            d_vision: 32,
            n_layers: 1,
            n_heads: 4,
            n_visual_tokens: 4,
            projector_hidden_dim: 64,
            projector_layers: 2,
            ..VisionConfig::default()
        },
        training: TrainingConfig {
            batch_size: 2,
            seq_len: 64,
            warmup_steps: 10,
            max_steps: 100,
            ..TrainingConfig::default()
        },
        grpo: GrpoConfig {
            g: 4,
            ..GrpoConfig::default()
        },
        ..ApexConfig::default()
    }
}

/// Builds a tiny LoRA configuration for adapter smoke tests.
pub fn get_tiny_lora_config() -> ApexConfig {
    let mut cfg = get_tiny_config();
    cfg.peft.enabled = true;
    cfg.peft.method = PeftMethod::Lora;
    cfg.peft.r = 4;
    cfg.peft.alpha = 8;
    cfg.peft.dropout = 0.0;
    cfg.training.peak_lr = 1e-4;
    cfg.training.max_steps = 20;
    cfg
}

/// Builds a tiny QLoRA configuration with 4-bit base layers.
pub fn get_tiny_qlora_config() -> ApexConfig {
    let mut cfg = get_tiny_lora_config();
    cfg.peft.method = PeftMethod::Qlora;
    cfg
}

/// Builds a tiny DoRA configuration with trainable row magnitudes.
pub fn get_tiny_dora_config() -> ApexConfig {
    let mut cfg = get_tiny_lora_config();
    cfg.peft.method = PeftMethod::Dora;
    cfg
}

/// Builds a tiny QDoRA configuration with quantized DoRA base layers.
pub fn get_tiny_qdora_config() -> ApexConfig {
    let mut cfg = get_tiny_lora_config();
    cfg.peft.method = PeftMethod::Qdora;
    cfg
}

/// Builds a tiny adapter-DPO configuration for the selected PEFT method.
pub fn get_tiny_adapter_dpo_config(method: PeftMethod) -> ApexConfig {
    let mut cfg = match method {
        PeftMethod::Lora => get_tiny_lora_config(),
        PeftMethod::Qlora => get_tiny_qlora_config(),
        PeftMethod::Dora => get_tiny_dora_config(),
        PeftMethod::Qdora => get_tiny_qdora_config(),
    };
    cfg.adapter_dpo.enabled = true;
    cfg.adapter_dpo.max_prompt_len = 64;
    cfg.adapter_dpo.max_response_len = 64;
    cfg.training.batch_size = 1;
    cfg
}

/// Builds a tiny vision-enabled configuration for multimodal tests.
pub fn get_tiny_vision_config() -> ApexConfig {
    let mut cfg = get_tiny_config();
    cfg.vision.enabled = true;
    cfg
}

/// Returns true when a layer index should use global MLA attention.
pub fn is_global_layer(layer_idx: usize, global_layer_freq: usize) -> bool {
    layer_idx % global_layer_freq == global_layer_freq - 1
}

/// Returns true when a layer index should use the MoE feed-forward path.
pub fn is_moe_layer(layer_idx: usize, moe: &MoeConfig) -> bool {
    moe.enabled && moe.moe_layer_freq != 0 && !layer_idx.is_multiple_of(moe.moe_layer_freq)
}
