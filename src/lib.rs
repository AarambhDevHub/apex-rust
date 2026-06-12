//! Public library surface for the APEX Rust Candle runtime.
//!
//! The crate exposes configuration, tokenization, data loading, model,
//! generation, training loss, checkpoint, vision, and inspection utilities.

/// Alignment losses and preference-training metrics.
pub mod alignment;
/// YAML configuration schema and preset builders.
pub mod config;
/// JSONL dataset readers and sample records.
pub mod data;
/// Error type shared across the crate.
pub mod error;
/// Evaluation metrics and benchmark helpers.
pub mod eval;
/// Autoregressive generation and sampling utilities.
pub mod generation;
/// Candle model layers and the APEX transformer.
pub mod model;
/// Shared tensor helper functions built on Candle.
pub mod tensor;
/// Tokenizer loading, fallback encoding, chat formatting, and tokenizer training.
pub mod tokenizer;
/// Training losses, schedulers, and checkpoint writers.
pub mod train;
/// Model inspection and reporting utilities.
pub mod utils;
/// Vision encoder, projector, and multimodal wrapper.
pub mod vision;

/// Main YAML-backed configuration type.
pub use config::ApexConfig;
/// Shared crate error and result aliases.
pub use error::{ApexError, Result};
