use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

use crate::error::{ApexError, Result};
use crate::tokenizer::{special, ApexTokenizer, ChatMessage};

use super::types::{PreferenceSample, PretrainSample, SftSample, VisionInstructionSample};

/// Raw pretraining JSONL row with a single `text` field.
#[derive(Debug, Deserialize)]
struct TextRow {
    text: String,
}

/// Raw chat message row used by SFT and vision-message datasets.
#[derive(Debug, Deserialize)]
struct MessageRow {
    role: String,
    content: String,
}

/// Raw SFT JSONL row containing a list of chat messages.
#[derive(Debug, Deserialize)]
struct SftRow {
    messages: Vec<MessageRow>,
}

/// Raw preference row supporting common alias names for prompt and responses.
#[derive(Debug, Deserialize)]
struct PreferenceRow {
    prompt: Option<String>,
    instruction: Option<String>,
    question: Option<String>,
    chosen: Option<String>,
    accepted: Option<String>,
    preferred: Option<String>,
    rejected: Option<String>,
    rejected_response: Option<String>,
    dispreferred: Option<String>,
}

/// Raw vision instruction row supporting prompt/response or chat-message schema.
#[derive(Debug, Deserialize)]
struct VisionRow {
    image: String,
    prompt: Option<String>,
    response: Option<String>,
    messages: Option<Vec<MessageRow>>,
}

/// Reads pretraining JSONL and packs all text into fixed-length token blocks.
pub fn read_pretrain_jsonl(
    path: impl AsRef<Path>,
    tokenizer: &ApexTokenizer,
    seq_len: usize,
) -> Result<Vec<PretrainSample>> {
    let mut tokens = Vec::new();
    for line in read_jsonl_lines(path)? {
        let row: TextRow = serde_json::from_str(&line)?;
        tokens.extend(tokenizer.encode(&row.text, false)?);
    }
    Ok(pack_tokens(&tokens, seq_len, tokenizer.pad_token_id()))
}

/// Packs a token stream into padded fixed-length pretraining samples.
pub fn pack_tokens(tokens: &[u32], seq_len: usize, pad_id: u32) -> Vec<PretrainSample> {
    if seq_len == 0 {
        return Vec::new();
    }
    let mut samples = Vec::new();
    let mut start = 0usize;
    while start < tokens.len() {
        let end = (start + seq_len).min(tokens.len());
        let mut ids = tokens[start..end].to_vec();
        let mut mask = vec![1u8; ids.len()];
        if ids.len() < seq_len {
            if ids.len() < seq_len / 2 && !samples.is_empty() {
                break;
            }
            ids.resize(seq_len, pad_id);
            mask.resize(seq_len, 0);
        }
        samples.push(PretrainSample {
            input_ids: ids,
            attention_mask: mask,
        });
        start += seq_len;
    }
    samples
}

/// Reads SFT JSONL with `messages` rows and produces role-masked token samples.
pub fn read_sft_jsonl(
    path: impl AsRef<Path>,
    tokenizer: &ApexTokenizer,
    max_seq_len: usize,
) -> Result<Vec<SftSample>> {
    let mut out = Vec::new();
    for line in read_jsonl_lines(path)? {
        let row: SftRow = serde_json::from_str(&line)?;
        let messages: Vec<ChatMessage> = row
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        let mut ids = tokenizer.encode_chat(&messages, false, false)?;
        ids.truncate(max_seq_len);
        let mut token_types = tokenizer.get_token_types(&ids);
        ids.resize(max_seq_len, tokenizer.pad_token_id());
        token_types.resize(max_seq_len, 0);
        out.push(SftSample {
            input_ids: ids,
            token_types,
        });
    }
    Ok(out)
}

/// Reads preference JSONL using accepted alias fields for prompt/chosen/rejected.
pub fn read_preference_jsonl(
    path: impl AsRef<Path>,
    tokenizer: &ApexTokenizer,
    max_prompt_len: usize,
    max_response_len: usize,
) -> Result<Vec<PreferenceSample>> {
    let mut out = Vec::new();
    for line in read_jsonl_lines(path)? {
        let row: PreferenceRow = serde_json::from_str(&line)?;
        let prompt = pick(row.prompt, row.instruction, row.question, "prompt")?;
        let chosen = pick(row.chosen, row.accepted, row.preferred, "chosen")?;
        let rejected = pick(
            row.rejected,
            row.rejected_response,
            row.dispreferred,
            "rejected",
        )?;
        out.push(format_preference_example(
            tokenizer,
            &prompt,
            &chosen,
            &rejected,
            max_prompt_len,
            max_response_len,
            true,
        )?);
    }
    Ok(out)
}

/// Formats one preference example and returns prompt/chosen/rejected token sequences.
pub fn format_preference_example(
    tokenizer: &ApexTokenizer,
    prompt: &str,
    chosen: &str,
    rejected: &str,
    max_prompt_len: usize,
    max_response_len: usize,
    add_chat_template: bool,
) -> Result<PreferenceSample> {
    let mut prompt_ids = if add_chat_template {
        let prompt_text = tokenizer.format_chat(
            &[ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            true,
            false,
        );
        tokenizer.encode(&prompt_text, false)?
    } else {
        tokenizer.encode(prompt, true)?
    };
    if prompt_ids.len() > max_prompt_len {
        prompt_ids = prompt_ids[prompt_ids.len() - max_prompt_len..].to_vec();
    }
    let mut chosen_ids = tokenizer.encode(chosen, false)?;
    chosen_ids.truncate(max_response_len);
    chosen_ids.push(tokenizer.eos_token_id());
    let mut rejected_ids = tokenizer.encode(rejected, false)?;
    rejected_ids.truncate(max_response_len);
    rejected_ids.push(tokenizer.eos_token_id());
    let prompt_len = prompt_ids.len();
    let mut full_chosen = prompt_ids.clone();
    full_chosen.extend(chosen_ids);
    let mut full_rejected = prompt_ids.clone();
    full_rejected.extend(rejected_ids);
    if full_chosen.len() < 2 || full_rejected.len() < 2 {
        return Err(ApexError::Data(
            "preference example produced too few tokens".to_string(),
        ));
    }
    Ok(PreferenceSample {
        prompt_ids,
        chosen_ids: full_chosen,
        rejected_ids: full_rejected,
        prompt_len,
    })
}

/// Reads vision JSONL rows and resolves image paths relative to an optional root.
pub fn read_vision_jsonl(
    path: impl AsRef<Path>,
    image_root: Option<impl AsRef<Path>>,
) -> Result<Vec<VisionInstructionSample>> {
    let path = path.as_ref();
    let root = image_root
        .as_ref()
        .map(|p| p.as_ref().to_path_buf())
        .unwrap_or_else(|| {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    let mut out = Vec::new();
    for line in read_jsonl_lines(path)? {
        let row: VisionRow = serde_json::from_str(&line)?;
        let (prompt, response) = match (row.prompt, row.response, row.messages) {
            (Some(prompt), Some(response), _) => (prompt, response),
            (_, _, Some(messages)) => {
                let mut prompt = None;
                let mut response = None;
                for msg in messages {
                    if msg.role == "user" {
                        prompt = Some(msg.content.replace(special("img"), "").trim().to_string());
                    } else if msg.role == "assistant" {
                        response = Some(msg.content);
                    }
                }
                (
                    prompt.ok_or_else(|| {
                        ApexError::Data("vision row missing user prompt".to_string())
                    })?,
                    response.ok_or_else(|| {
                        ApexError::Data("vision row missing assistant response".to_string())
                    })?,
                )
            }
            _ => {
                return Err(ApexError::Data(
                    "vision rows need prompt/response or messages".to_string(),
                ))
            }
        };
        out.push(VisionInstructionSample {
            image: root.join(row.image),
            prompt,
            response,
        });
    }
    Ok(out)
}

fn read_jsonl_lines(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let mut lines = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        let line = line.trim();
        if !line.is_empty() {
            lines.push(line.to_string());
        }
    }
    Ok(lines)
}

fn pick(a: Option<String>, b: Option<String>, c: Option<String>, label: &str) -> Result<String> {
    a.or(b)
        .or(c)
        .ok_or_else(|| ApexError::Data(format!("preference row missing {label}")))
}
