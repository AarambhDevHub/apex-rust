//! Training helpers: losses, cosine scheduler, checkpoints, and optimizer-backed loops.

mod autograd;
mod checkpoint;
mod losses;
mod optimizer;
mod runner;
mod scheduler;
mod variables;

pub use autograd::{
    adapter_dpo_loss_tensor, clip_grad_store, dpo_loss_tensor, grpo_clipped_loss_tensor,
    pretrain_loss_tensor, sequence_logprob_tensor, sft_loss_tensor, vision_sft_loss_tensor,
    AutogradLoss, AutogradStepStats,
};
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
pub use runner::{
    dry_run_pretrain_step, train_adapter_dpo_steps, train_grpo_steps, train_pretrain_steps,
    train_sft_steps, TrainingReport,
};
pub use scheduler::get_lr;
pub use variables::{scope_for_model, TrainableScope, TrainableTensor, TrainableVariables};
