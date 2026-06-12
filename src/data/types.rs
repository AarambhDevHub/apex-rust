use std::path::PathBuf;

/// Packed fixed-length language-model pretraining sample.
#[derive(Debug, Clone)]
pub struct PretrainSample {
    /// Token IDs padded to the configured sequence length.
    pub input_ids: Vec<u32>,
    /// One for real tokens and zero for padding.
    pub attention_mask: Vec<u8>,
}

/// Supervised fine-tuning sample with role-aware token labels.
#[derive(Debug, Clone)]
pub struct SftSample {
    /// Token IDs padded or truncated to the configured sequence length.
    pub input_ids: Vec<u32>,
    /// Role IDs derived from special chat markers; assistant tokens use value `2`.
    pub token_types: Vec<u8>,
}

/// Preference-training sample containing one chosen and one rejected completion.
#[derive(Debug, Clone)]
pub struct PreferenceSample {
    /// Prompt-only token IDs after optional chat formatting.
    pub prompt_ids: Vec<u32>,
    /// Prompt plus chosen response token IDs.
    pub chosen_ids: Vec<u32>,
    /// Prompt plus rejected response token IDs.
    pub rejected_ids: Vec<u32>,
    /// Prompt length used to locate the response for DPO log-probability.
    pub prompt_len: usize,
}

/// Vision instruction example containing an image path and text target.
#[derive(Debug, Clone)]
pub struct VisionInstructionSample {
    /// Path to the image file after resolving the dataset image root.
    pub image: PathBuf,
    /// User prompt associated with the image.
    pub prompt: String,
    /// Assistant response associated with the image.
    pub response: String,
}
