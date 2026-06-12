use thiserror::Error;

pub type Result<T> = std::result::Result<T, ApexError>;

#[derive(Debug, Error)]
pub enum ApexError {
    #[error("config error: {0}")]
    Config(String),
    #[error("data error: {0}")]
    Data(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    #[error("invalid shape: {0}")]
    Shape(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}
