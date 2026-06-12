//! Training helpers: losses, cosine scheduler, checkpoints, and dry-run steps.

mod checkpoint;
mod losses;
mod optimizer;
mod runner;
mod scheduler;

pub use checkpoint::{
    load_adapter_safetensors, load_model_safetensors, save_adapter_safetensors,
    save_checkpoint_metadata, save_model_safetensors, CheckpointMetadata,
};
pub use losses::{
    compute_pretrain_loss, compute_sft_loss, compute_vision_sft_loss,
    expand_labels_for_visual_tokens, LossMetrics,
};
pub use optimizer::{
    adamw_update, clip_gradients, global_grad_norm, AdamWConfig, AdamWOptimizer, AdamWState,
    OptimizerStepStats, TrainingState,
};
pub use runner::dry_run_pretrain_step;
pub use scheduler::get_lr;
