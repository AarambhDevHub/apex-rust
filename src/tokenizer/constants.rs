pub const VOCAB_SIZE: usize = 151_643;

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

pub fn special(name: &str) -> &'static str {
    SPECIAL_TOKENS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, token)| *token)
        .unwrap_or("")
}
