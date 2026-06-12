//! Dataset record types and JSONL readers for pretraining, SFT, preferences, and vision.

mod batching;
mod readers;
mod types;

pub use batching::{
    batch_preference_samples, batch_pretrain_samples, batch_sft_samples, collate_vision_text_batch,
    format_vision_instruction, pad_i64_rows, pad_u32_rows, paths_from_iter, BatchOptions,
    PreferenceBatch, PretrainBatch, SftBatch, StreamingPretrainDataset, StreamingPretrainIter,
    VisionTextBatch, VisionTextSample, IGNORE_INDEX,
};
pub use readers::{
    format_preference_example, pack_tokens, read_preference_jsonl, read_pretrain_jsonl,
    read_sft_jsonl, read_vision_jsonl,
};
pub use types::{PreferenceSample, PretrainSample, SftSample, VisionInstructionSample};
