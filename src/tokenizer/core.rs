use std::collections::{HashMap, HashSet};
use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{ApexError, Result};

use super::constants::{special, SPECIAL_TOKENS};

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct ApexTokenizer {
    tokenizer: Option<Tokenizer>,
    special_ids: HashMap<String, u32>,
    simple_vocab: HashMap<String, u32>,
    simple_rev: HashMap<u32, String>,
}

impl std::fmt::Debug for ApexTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApexTokenizer")
            .field("has_tokenizer_json", &self.tokenizer.is_some())
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

impl ApexTokenizer {
    pub fn new(path: Option<impl AsRef<Path>>) -> Result<Self> {
        let tokenizer = if let Some(path) = path {
            let path = path.as_ref();
            if path.exists() {
                Some(Tokenizer::from_file(path).map_err(|e| {
                    ApexError::Tokenizer(format!(
                        "failed to load tokenizer {}: {e}",
                        path.display()
                    ))
                })?)
            } else {
                None
            }
        } else {
            None
        };

        let mut special_ids = HashMap::new();
        if let Some(tok) = tokenizer.as_ref() {
            for (fallback, (name, text)) in SPECIAL_TOKENS.iter().enumerate() {
                special_ids.insert(
                    (*name).to_string(),
                    tok.token_to_id(text).unwrap_or(fallback as u32),
                );
            }
        } else {
            for (i, (name, _)) in SPECIAL_TOKENS.iter().enumerate() {
                special_ids.insert((*name).to_string(), i as u32);
            }
        }

        let mut simple_vocab = HashMap::new();
        let mut simple_rev = HashMap::new();
        for (name, text) in SPECIAL_TOKENS {
            let id = *special_ids.get(name).unwrap_or(&0);
            simple_vocab.insert(text.to_string(), id);
            simple_rev.insert(id, text.to_string());
        }
        let common = [
            " ", "\n", "\t", ".", ",", "!", "?", ":", ";", "the", "a", "an", "is", "are", "was",
            "were", "in", "on", "at", "to", "for", "of", "and", "The", "I", "you", "he", "she",
            "it", "we", "they", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "def", "class",
            "import", "return", "if", "else", "function", "var", "let", "const",
        ];
        let mut next = 9u32;
        for token in common {
            simple_vocab.entry(token.to_string()).or_insert_with(|| {
                let id = next;
                next += 1;
                simple_rev.insert(id, token.to_string());
                id
            });
        }

        Ok(Self {
            tokenizer,
            special_ids,
            simple_vocab,
            simple_rev,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer
            .as_ref()
            .map(|t| t.get_vocab_size(false))
            .unwrap_or_else(|| self.simple_vocab.len())
    }

    pub fn pad_token_id(&self) -> u32 {
        self.id("pad", 0)
    }
    pub fn bos_token_id(&self) -> u32 {
        self.id("bos", 1)
    }
    pub fn eos_token_id(&self) -> u32 {
        self.id("eos", 2)
    }
    pub fn system_token_id(&self) -> u32 {
        self.id("system", 3)
    }
    pub fn user_token_id(&self) -> u32 {
        self.id("user", 4)
    }
    pub fn assistant_token_id(&self) -> u32 {
        self.id("assistant", 5)
    }
    pub fn thinking_start_id(&self) -> u32 {
        self.id("thinking_start", 6)
    }
    pub fn thinking_end_id(&self) -> u32 {
        self.id("thinking_end", 7)
    }
    pub fn img_token_id(&self) -> u32 {
        self.id("img", 8)
    }

    fn id(&self, name: &str, fallback: u32) -> u32 {
        *self.special_ids.get(name).unwrap_or(&fallback)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut ids = if let Some(tok) = self.tokenizer.as_ref() {
            tok.encode(text, false)
                .map_err(|e| ApexError::Tokenizer(format!("encode failed: {e}")))?
                .get_ids()
                .to_vec()
        } else {
            self.simple_encode(text)
        };
        if add_special_tokens {
            ids.insert(0, self.bos_token_id());
            ids.push(self.eos_token_id());
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        if let Some(tok) = self.tokenizer.as_ref() {
            return tok
                .decode(ids, skip_special_tokens)
                .map_err(|e| ApexError::Tokenizer(format!("decode failed: {e}")));
        }
        let special: HashSet<u32> = self.special_ids.values().copied().collect();
        let mut out = String::new();
        for &id in ids {
            if skip_special_tokens && special.contains(&id) {
                continue;
            }
            if let Some(token) = self.simple_rev.get(&id) {
                out.push_str(token);
            } else if (100..356).contains(&id) {
                out.push(char::from_u32(id - 100).unwrap_or('�'));
            } else {
                out.push_str(&format!("<{}>", id));
            }
        }
        Ok(out)
    }

    pub fn format_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> String {
        let mut parts = vec![special("bos").to_string()];
        for msg in messages {
            match msg.role.as_str() {
                "system" => parts.push(format!("{}\n{}\n", special("system"), msg.content)),
                "user" => parts.push(format!("{}\n{}\n", special("user"), msg.content)),
                "assistant" => {
                    if enable_thinking && !msg.content.contains(special("thinking_start")) {
                        parts.push(format!(
                            "{}\n{}\n{}\n{}\n",
                            special("assistant"),
                            special("thinking_start"),
                            msg.content,
                            special("thinking_end")
                        ));
                    } else {
                        parts.push(format!("{}\n{}\n", special("assistant"), msg.content));
                    }
                }
                _ => {}
            }
        }
        if add_generation_prompt {
            parts.push(format!("{}\n", special("assistant")));
            if enable_thinking {
                parts.push(format!("{}\n", special("thinking_start")));
            }
        }
        parts.concat()
    }

    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<Vec<u32>> {
        let text = self.format_chat(messages, add_generation_prompt, enable_thinking);
        self.encode(&text, false)
    }

    pub fn get_token_types(&self, token_ids: &[u32]) -> Vec<u8> {
        let mut current = 0u8;
        let mut out = Vec::with_capacity(token_ids.len());
        for &tid in token_ids {
            if tid == self.system_token_id() {
                current = 0;
            } else if tid == self.user_token_id() {
                current = 1;
            } else if tid == self.assistant_token_id()
                || tid == self.thinking_start_id()
                || tid == self.thinking_end_id()
            {
                current = 2;
            }
            out.push(current);
        }
        out
    }

    fn simple_encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            if let Some((_, token)) = SPECIAL_TOKENS
                .iter()
                .find(|(_, token)| remaining.starts_with(*token))
            {
                ids.push(*self.simple_vocab.get(*token).unwrap_or(&0));
                remaining = &remaining[token.len()..];
                continue;
            }
            let ch = remaining.chars().next().unwrap_or_default();
            let byte_id = 100 + (ch as u32).min(255);
            ids.push(byte_id);
            remaining = &remaining[ch.len_utf8()..];
        }
        ids
    }
}
