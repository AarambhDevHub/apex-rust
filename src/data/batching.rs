//! Batch construction and streaming pretraining utilities.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tracing::warn;

use crate::error::{ApexError, Result};
use crate::tokenizer::{special, ApexTokenizer};

use super::types::{PreferenceSample, PretrainSample, SftSample, VisionInstructionSample};

/// Label value ignored by vision SFT cross-entropy.
pub const IGNORE_INDEX: i64 = -100;

/// Common options used by vector-backed batch builders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchOptions {
    /// Number of examples per batch.
    pub batch_size: usize,
    /// Whether to shuffle example order before batching.
    pub shuffle: bool,
    /// Whether to drop an incomplete final batch.
    pub drop_last: bool,
    /// Deterministic shuffle seed.
    pub seed: u64,
}

impl BatchOptions {
    /// Creates batch options after validating `batch_size`.
    pub fn new(batch_size: usize, shuffle: bool, drop_last: bool, seed: u64) -> Result<Self> {
        if batch_size == 0 {
            return Err(ApexError::Data(
                "batch_size must be greater than zero".into(),
            ));
        }
        Ok(Self {
            batch_size,
            shuffle,
            drop_last,
            seed,
        })
    }

    /// Python-style defaults for pretraining loaders.
    pub fn pretrain(batch_size: usize) -> Result<Self> {
        Self::new(batch_size, true, true, 42)
    }

    /// Python-style defaults for SFT loaders.
    pub fn sft(batch_size: usize) -> Result<Self> {
        Self::new(batch_size, true, true, 42)
    }

    /// Python-style defaults for preference loaders.
    pub fn preference(batch_size: usize) -> Result<Self> {
        Self::new(batch_size, true, true, 42)
    }
}

/// Batched pretraining tensors represented as nested token vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PretrainBatch {
    /// Token IDs with shape `[B, S]`.
    pub input_ids: Vec<Vec<u32>>,
    /// Attention mask with shape `[B, S]`.
    pub attention_mask: Vec<Vec<u8>>,
}

/// Batched SFT tensors represented as nested token vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftBatch {
    /// Token IDs with shape `[B, S]`.
    pub input_ids: Vec<Vec<u32>>,
    /// Role-token type labels with shape `[B, S]`.
    pub token_types: Vec<Vec<u8>>,
}

/// Batched preference tensors for DPO-style training.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreferenceBatch {
    /// Padded prompt token IDs.
    pub prompt_ids: Vec<Vec<u32>>,
    /// Padded chosen prompt+response token IDs.
    pub chosen_ids: Vec<Vec<u32>>,
    /// Padded rejected prompt+response token IDs.
    pub rejected_ids: Vec<Vec<u32>>,
    /// Attention mask for chosen IDs.
    pub chosen_attention_mask: Vec<Vec<u8>>,
    /// Attention mask for rejected IDs.
    pub rejected_attention_mask: Vec<Vec<u8>>,
    /// Unpadded prompt lengths.
    pub prompt_lens: Vec<usize>,
}

/// Tokenized vision-language instruction item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionTextSample {
    /// Token IDs containing the image placeholder token.
    pub token_ids: Vec<u32>,
    /// Labels where prompt and padding positions are ignored.
    pub labels: Vec<i64>,
    /// Original resolved image path.
    pub image_path: PathBuf,
    /// User prompt text.
    pub prompt: String,
    /// Assistant response text.
    pub response: String,
}

/// Batched vision-language instruction metadata and token labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionTextBatch {
    /// Padded token IDs.
    pub token_ids: Vec<Vec<u32>>,
    /// Padded labels using `IGNORE_INDEX` for ignored positions.
    pub labels: Vec<Vec<i64>>,
    /// Resolved image paths in batch order.
    pub image_paths: Vec<PathBuf>,
    /// Prompt texts in batch order.
    pub prompts: Vec<String>,
    /// Response texts in batch order.
    pub responses: Vec<String>,
}

/// Builds pretraining batches from already packed samples.
pub fn batch_pretrain_samples(
    samples: &[PretrainSample],
    options: BatchOptions,
) -> Vec<PretrainBatch> {
    ordered_chunks(samples.len(), options)
        .into_iter()
        .map(|indices| PretrainBatch {
            input_ids: indices
                .iter()
                .map(|&idx| samples[idx].input_ids.clone())
                .collect(),
            attention_mask: indices
                .iter()
                .map(|&idx| samples[idx].attention_mask.clone())
                .collect(),
        })
        .collect()
}

/// Builds SFT batches from fixed-length samples.
pub fn batch_sft_samples(samples: &[SftSample], options: BatchOptions) -> Vec<SftBatch> {
    ordered_chunks(samples.len(), options)
        .into_iter()
        .map(|indices| SftBatch {
            input_ids: indices
                .iter()
                .map(|&idx| samples[idx].input_ids.clone())
                .collect(),
            token_types: indices
                .iter()
                .map(|&idx| samples[idx].token_types.clone())
                .collect(),
        })
        .collect()
}

/// Builds padded preference batches for DPO-style losses.
pub fn batch_preference_samples(
    samples: &[PreferenceSample],
    options: BatchOptions,
    pad_token_id: u32,
    max_seq_len: usize,
) -> Vec<PreferenceBatch> {
    ordered_chunks(samples.len(), options)
        .into_iter()
        .map(|indices| {
            let prompts = indices
                .iter()
                .map(|&idx| truncate(&samples[idx].prompt_ids, max_seq_len))
                .collect::<Vec<_>>();
            let chosen = indices
                .iter()
                .map(|&idx| truncate(&samples[idx].chosen_ids, max_seq_len))
                .collect::<Vec<_>>();
            let rejected = indices
                .iter()
                .map(|&idx| truncate(&samples[idx].rejected_ids, max_seq_len))
                .collect::<Vec<_>>();
            let (prompt_ids, _) = pad_u32_rows(&prompts, pad_token_id, None);
            let (chosen_ids, chosen_attention_mask) = pad_u32_rows(&chosen, pad_token_id, None);
            let (rejected_ids, rejected_attention_mask) =
                pad_u32_rows(&rejected, pad_token_id, None);
            PreferenceBatch {
                prompt_ids,
                chosen_ids,
                rejected_ids,
                chosen_attention_mask,
                rejected_attention_mask,
                prompt_lens: indices.iter().map(|&idx| samples[idx].prompt_len).collect(),
            }
        })
        .collect()
}

/// Formats one vision-language instruction into token IDs and labels.
pub fn format_vision_instruction(
    sample: &VisionInstructionSample,
    tokenizer: &ApexTokenizer,
    max_length: usize,
) -> Result<VisionTextSample> {
    let prompt_text = format!(
        "{}{}\n{}\n{}\n{}\n",
        special("bos"),
        special("user"),
        special("img"),
        sample.prompt,
        special("assistant")
    );
    let answer_text = format!("{}{}", sample.response, special("eos"));
    let prompt_ids = tokenizer.encode(&prompt_text, false)?;
    let answer_ids = tokenizer.encode(&answer_text, false)?;
    let mut token_ids = prompt_ids
        .iter()
        .copied()
        .chain(answer_ids.iter().copied())
        .take(max_length)
        .collect::<Vec<_>>();
    let prompt_label_count = prompt_ids.len().min(token_ids.len());
    let mut labels = vec![IGNORE_INDEX; prompt_label_count];
    let remaining = token_ids.len().saturating_sub(labels.len());
    labels.extend(answer_ids.iter().take(remaining).map(|&id| i64::from(id)));
    token_ids.truncate(max_length);
    labels.truncate(max_length);
    Ok(VisionTextSample {
        token_ids,
        labels,
        image_path: sample.image.clone(),
        prompt: sample.prompt.clone(),
        response: sample.response.clone(),
    })
}

/// Builds a padded vision-language batch from instruction samples.
pub fn collate_vision_text_batch(
    samples: &[VisionInstructionSample],
    tokenizer: &ApexTokenizer,
    max_length: usize,
    pad_token_id: u32,
    ignore_index: i64,
) -> Result<VisionTextBatch> {
    let formatted = samples
        .iter()
        .map(|sample| format_vision_instruction(sample, tokenizer, max_length))
        .collect::<Result<Vec<_>>>()?;
    let token_rows = formatted
        .iter()
        .map(|sample| sample.token_ids.clone())
        .collect::<Vec<_>>();
    let label_rows = formatted
        .iter()
        .map(|sample| sample.labels.clone())
        .collect::<Vec<_>>();
    let (token_ids, _) = pad_u32_rows(&token_rows, pad_token_id, None);
    let labels = pad_i64_rows(&label_rows, ignore_index, None);
    Ok(VisionTextBatch {
        token_ids,
        labels,
        image_paths: formatted
            .iter()
            .map(|sample| sample.image_path.clone())
            .collect(),
        prompts: formatted
            .iter()
            .map(|sample| sample.prompt.clone())
            .collect(),
        responses: formatted
            .iter()
            .map(|sample| sample.response.clone())
            .collect(),
    })
}

/// Pads token rows to a common width and returns an attention mask.
pub fn pad_u32_rows(
    rows: &[Vec<u32>],
    pad_token_id: u32,
    max_len: Option<usize>,
) -> (Vec<Vec<u32>>, Vec<Vec<u8>>) {
    let width = max_len.unwrap_or_else(|| rows.iter().map(Vec::len).max().unwrap_or(0));
    let mut padded = Vec::with_capacity(rows.len());
    let mut masks = Vec::with_capacity(rows.len());
    for row in rows {
        let mut values = row.iter().copied().take(width).collect::<Vec<_>>();
        let real_len = values.len();
        values.resize(width, pad_token_id);
        let mut mask = vec![1u8; real_len];
        mask.resize(width, 0);
        padded.push(values);
        masks.push(mask);
    }
    (padded, masks)
}

/// Pads signed label rows to a common width.
pub fn pad_i64_rows(rows: &[Vec<i64>], pad_value: i64, max_len: Option<usize>) -> Vec<Vec<i64>> {
    let width = max_len.unwrap_or_else(|| rows.iter().map(Vec::len).max().unwrap_or(0));
    rows.iter()
        .map(|row| {
            let mut values = row.iter().copied().take(width).collect::<Vec<_>>();
            values.resize(width, pad_value);
            values
        })
        .collect()
}

/// Lazy pretraining dataset that tokenizes text files line by line.
pub struct StreamingPretrainDataset<'a> {
    /// Text file paths to stream.
    pub file_paths: Vec<PathBuf>,
    /// Tokenizer used for each line.
    pub tokenizer: &'a ApexTokenizer,
    /// Fixed output sequence length.
    pub seq_len: usize,
    /// Whether file order is shuffled per iterator construction.
    pub shuffle_files: bool,
    /// Deterministic file-shuffle seed.
    pub seed: u64,
}

impl<'a> StreamingPretrainDataset<'a> {
    /// Creates a streaming pretraining dataset.
    pub fn new(
        file_paths: Vec<PathBuf>,
        tokenizer: &'a ApexTokenizer,
        seq_len: usize,
        shuffle_files: bool,
        seed: u64,
    ) -> Result<Self> {
        if seq_len == 0 {
            return Err(ApexError::Data("seq_len must be greater than zero".into()));
        }
        Ok(Self {
            file_paths,
            tokenizer,
            seq_len,
            shuffle_files,
            seed,
        })
    }

    /// Creates a lazy iterator over packed pretraining samples.
    pub fn iter(&'a self) -> StreamingPretrainIter<'a> {
        let mut files = self.file_paths.clone();
        if self.shuffle_files {
            let mut rng = StdRng::seed_from_u64(self.seed);
            files.shuffle(&mut rng);
        }
        StreamingPretrainIter {
            files,
            tokenizer: self.tokenizer,
            seq_len: self.seq_len,
            pad_token_id: self.tokenizer.pad_token_id(),
            file_index: 0,
            reader: None,
            buffer: Vec::new(),
            final_emitted: false,
        }
    }

    /// Collects up to `limit` streamed samples for smoke tests and small jobs.
    pub fn collect_samples(&'a self, limit: Option<usize>) -> Result<Vec<PretrainSample>> {
        let mut out = Vec::new();
        for item in self.iter() {
            out.push(item?);
            if limit.is_some_and(|limit| out.len() >= limit) {
                break;
            }
        }
        Ok(out)
    }
}

/// Iterator returned by `StreamingPretrainDataset`.
pub struct StreamingPretrainIter<'a> {
    files: Vec<PathBuf>,
    tokenizer: &'a ApexTokenizer,
    seq_len: usize,
    pad_token_id: u32,
    file_index: usize,
    reader: Option<BufReader<File>>,
    buffer: Vec<u32>,
    final_emitted: bool,
}

impl Iterator for StreamingPretrainIter<'_> {
    type Item = Result<PretrainSample>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.buffer.len() >= self.seq_len {
                let ids = self.buffer.drain(..self.seq_len).collect::<Vec<_>>();
                return Some(Ok(PretrainSample {
                    input_ids: ids,
                    attention_mask: vec![1; self.seq_len],
                }));
            }

            if let Some(reader) = self.reader.as_mut() {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        self.reader = None;
                        continue;
                    }
                    Ok(_) => match self.tokenizer.encode(line.trim(), false) {
                        Ok(tokens) => {
                            self.buffer.extend(tokens);
                            continue;
                        }
                        Err(err) => return Some(Err(err)),
                    },
                    Err(err) => return Some(Err(err.into())),
                }
            }

            if self.file_index < self.files.len() {
                let path = &self.files[self.file_index];
                self.file_index += 1;
                if !path.exists() {
                    warn!("skipping missing streaming data file: {}", path.display());
                    continue;
                }
                match File::open(path) {
                    Ok(file) => {
                        self.reader = Some(BufReader::new(file));
                        continue;
                    }
                    Err(err) => return Some(Err(err.into())),
                }
            }

            if !self.final_emitted && self.buffer.len() >= self.seq_len / 2 {
                self.final_emitted = true;
                let real_len = self.buffer.len();
                let mut ids = std::mem::take(&mut self.buffer);
                ids.resize(self.seq_len, self.pad_token_id);
                let mut mask = vec![1u8; real_len];
                mask.resize(self.seq_len, 0);
                return Some(Ok(PretrainSample {
                    input_ids: ids,
                    attention_mask: mask,
                }));
            }
            return None;
        }
    }
}

fn ordered_chunks(len: usize, options: BatchOptions) -> Vec<Vec<usize>> {
    let mut indices = (0..len).collect::<Vec<_>>();
    if options.shuffle {
        let mut rng = StdRng::seed_from_u64(options.seed);
        indices.shuffle(&mut rng);
    }
    indices
        .chunks(options.batch_size)
        .filter(|chunk| !options.drop_last || chunk.len() == options.batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn truncate(values: &[u32], max_len: usize) -> Vec<u32> {
    if max_len == 0 {
        values.to_vec()
    } else {
        values.iter().copied().take(max_len).collect()
    }
}

/// Converts a path-like list into owned path buffers.
pub fn paths_from_iter<I, P>(paths: I) -> Vec<PathBuf>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    paths
        .into_iter()
        .map(|path| path.as_ref().to_path_buf())
        .collect()
}
