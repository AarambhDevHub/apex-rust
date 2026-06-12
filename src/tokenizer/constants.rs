//! Tokenizer constants and special-token lookup helpers.

/// Default vocabulary size used by full-size configuration presets.
pub const VOCAB_SIZE: usize = 151_643;

/// Ordered special-token table used by the fallback tokenizer.
pub const SPECIAL_TOKENS: [(&str, &str); 9] = [
    ("pad", "<|pad|>"),
    ("bos", "<|begin_of_text|>"),
    ("eos", "<|end_of_text|>"),
    ("system", "<|system|>"),
    ("user", "<|user|>"),
    ("assistant", "<|assistant|>"),
    ("thinking_start", "<|thinking|>"),
    ("thinking_end", "<|/thinking|>"),
    ("img", "<|img|>"),
];

/// Returns the special-token text for a symbolic token name.
pub fn special(name: &str) -> &'static str {
    SPECIAL_TOKENS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, token)| *token)
        .unwrap_or("")
}
