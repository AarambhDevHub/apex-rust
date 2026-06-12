use thiserror::Error;

/// Convenience result alias used by all public APIs.
pub type Result<T> = std::result::Result<T, ApexError>;

/// Error categories produced by configuration, data, model, and IO operations.
#[derive(Debug, Error)]
pub enum ApexError {
    /// Invalid configuration or unsupported configuration combination.
    #[error("config error: {0}")]
    Config(String),
    /// Dataset, token, or user input problem.
    #[error("data error: {0}")]
    Data(String),
    /// Model construction, forward pass, or parameter error.
    #[error("model error: {0}")]
    Model(String),
    /// Tokenizer load, encode, decode, or training error.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// Checkpoint metadata or safetensors write error.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    /// Tensor shape mismatch.
    #[error("invalid shape: {0}")]
    Shape(String),
    /// Wrapped filesystem error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Wrapped JSON serialization error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Wrapped YAML serialization error.
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
    /// Wrapped Candle tensor error.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    /// Wrapped general-purpose error.
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}
