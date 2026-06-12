use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct PretrainSample {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SftSample {
    pub input_ids: Vec<u32>,
    pub token_types: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PreferenceSample {
    pub prompt_ids: Vec<u32>,
    pub chosen_ids: Vec<u32>,
    pub rejected_ids: Vec<u32>,
    pub prompt_len: usize,
}

#[derive(Debug, Clone)]
pub struct VisionInstructionSample {
    pub image: PathBuf,
    pub prompt: String,
    pub response: String,
}
