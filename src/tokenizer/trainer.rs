use std::path::Path;

use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder};

use crate::error::{ApexError, Result};

use super::constants::SPECIAL_TOKENS;

pub fn train_bpe_tokenizer(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    vocab_size: usize,
) -> Result<()> {
    let input = input.as_ref();
    let output = output.as_ref();
    let specials = SPECIAL_TOKENS
        .iter()
        .map(|(_, token)| AddedToken::from((*token).to_string(), true))
        .chain(std::iter::once(AddedToken::from(
            "<|unk|>".to_string(),
            true,
        )))
        .collect::<Vec<_>>();
    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .show_progress(false)
        .special_tokens(specials)
        .build();
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()
        .map_err(|e| ApexError::Tokenizer(format!("failed to build tokenizer: {e}")))?;
    tokenizer
        .train_from_files(&mut trainer, vec![input.to_string_lossy().to_string()])
        .map_err(|e| ApexError::Tokenizer(format!("tokenizer training failed: {e}")))?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    tokenizer
        .save(output, false)
        .map_err(|e| ApexError::Tokenizer(format!("failed to save tokenizer: {e}")))?;
    Ok(())
}
