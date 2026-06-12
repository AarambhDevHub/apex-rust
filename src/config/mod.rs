//! Configuration schema, preset builders, and layer-placement helpers.

mod presets;
mod schema;

pub use presets::{
    get_tiny_adapter_dpo_config, get_tiny_adapter_inference_config, get_tiny_config,
    get_tiny_dora_config, get_tiny_dora_inference_config, get_tiny_lora_config,
    get_tiny_lora_inference_config, get_tiny_qdora_config, get_tiny_qdora_inference_config,
    get_tiny_qlora_config, get_tiny_qlora_inference_config, get_tiny_vision_config,
    is_global_layer, is_moe_layer,
};
pub use schema::{
    AdapterDpoConfig, ApexConfig, AttentionConfig, GenerationRuntimeConfig, GrpoConfig,
    ModelConfig, MoeConfig, MultiTokenHeadConfig, PeftConfig, PeftMethod, SkipGateConfig,
    ThinkingConfig, TrainingConfig, VisionConfig,
};
