//! Alignment utilities for DPO, GRPO, reward models, PRM, and constitutional AI.

mod combined_reward;
mod constitutional;
mod dpo;
mod grpo;
mod prm;
mod reward;

pub use combined_reward::{
    combined_reward, extract_thinking_text, score_combined_reward, validate_weights,
    RewardBreakdown, RewardWeights,
};
pub use constitutional::{
    default_constitution, ConstitutionalAI, CritiqueResult, NoopTextGenerator, RevisionResult,
    TextGenerator, DEFAULT_CONSTITUTION,
};
pub use dpo::{adapter_dpo_step, dpo_loss, sequence_logprob, DpoMetrics};
pub use grpo::{clipped_policy_loss, grpo_advantages};
pub use prm::ProcessRewardModel;
pub use reward::{last_non_padded_indices, reward_model_loss, RewardModel};
