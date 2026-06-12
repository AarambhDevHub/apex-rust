//! Evaluation metrics for token accuracy, perplexity, generation quality, and timing.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::{ApexError, Result};
use crate::model::ApexModel;
use crate::train;

/// Accuracy summary for next-token predictions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// Number of non-ignored labels.
    pub total: usize,
    /// Number of predictions equal to the label.
    pub correct: usize,
    /// `correct / total`, using zero-safe division.
    pub accuracy: f64,
}

/// Computes argmax next-token accuracy over `[B, S, V]` logits.
pub fn next_token_accuracy(
    logits: &candle_core::Tensor,
    labels: &[Vec<i64>],
    ignore_index: i64,
) -> Result<ClassificationMetrics> {
    let dims = logits.dims();
    if dims.len() != 3 {
        return Err(ApexError::Shape(format!(
            "logits must be [B,S,V], got {dims:?}"
        )));
    }
    let values = logits.to_vec3::<f32>()?;
    let mut total = 0usize;
    let mut correct = 0usize;
    for (batch, label_row) in labels.iter().enumerate() {
        for (pos, &label) in label_row.iter().enumerate() {
            if label == ignore_index {
                continue;
            }
            total += 1;
            let pred = values[batch][pos]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0);
            if pred == label {
                correct += 1;
            }
        }
    }
    Ok(ClassificationMetrics {
        total,
        correct,
        accuracy: correct as f64 / total.max(1) as f64,
    })
}

/// Perplexity calculation result over one or more batches.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PerplexityResult {
    /// Mean cross-entropy loss.
    pub loss: f64,
    /// Exponentiated loss with a clamp for numerical safety.
    pub perplexity: f64,
    /// Number of valid next-token labels.
    pub token_count: usize,
    /// Number of batches evaluated.
    pub batch_count: usize,
}

/// Computes language-model perplexity using pretraining loss.
pub fn compute_perplexity(
    model: &mut ApexModel,
    batches: &[Vec<Vec<u32>>],
) -> Result<PerplexityResult> {
    let mut loss_sum = 0.0;
    let mut tokens = 0usize;
    for batch in batches {
        let out = model.forward(batch, None, 0, None, false)?;
        let m = train::compute_pretrain_loss(&out.logits, None, batch, 0.0)?;
        loss_sum += m.loss_total * m.valid_tokens as f64;
        tokens += m.valid_tokens;
    }
    let loss = loss_sum / tokens.max(1) as f64;
    Ok(PerplexityResult {
        loss,
        perplexity: loss.min(100.0).exp(),
        token_count: tokens,
        batch_count: batches.len(),
    })
}

/// Forward-pass benchmark summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Number of sequences in the benchmark input.
    pub batch_size: usize,
    /// Tokens per sequence.
    pub seq_len: usize,
    /// Number of repeated forward passes.
    pub repeats: usize,
    /// Mean latency in milliseconds.
    pub mean_ms: f64,
    /// Minimum latency in milliseconds.
    pub min_ms: f64,
    /// Maximum latency in milliseconds.
    pub max_ms: f64,
    /// Throughput measured from mean latency.
    pub tokens_per_second: f64,
    /// Shape of the final logits tensor.
    pub logits_shape: Vec<usize>,
}

impl BenchmarkResult {
    /// Formats benchmark metrics as a Markdown table.
    pub fn to_markdown(&self) -> String {
        format!(
            "| Metric | Value |\n|---|---:|\n| Batch size | {} |\n| Sequence length | {} |\n| Repeats | {} |\n| Mean forward time | {:.3} ms |\n| Tokens / second | {:.2} |\n| Logits shape | {:?} |",
            self.batch_size,
            self.seq_len,
            self.repeats,
            self.mean_ms,
            self.tokens_per_second,
            self.logits_shape
        )
    }
}

/// Runs repeated model forward passes and measures latency.
pub fn run_forward_benchmark(
    model: &mut ApexModel,
    input_ids: &[Vec<u32>],
    repeats: usize,
) -> Result<BenchmarkResult> {
    let repeats = repeats.max(1);
    let mut timings = Vec::with_capacity(repeats);
    let mut shape = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        let out = model.forward(input_ids, None, 0, None, false)?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
        shape = out.logits.dims().to_vec();
    }
    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    Ok(BenchmarkResult {
        batch_size: input_ids.len(),
        seq_len: input_ids.first().map(Vec::len).unwrap_or(0),
        repeats,
        mean_ms: mean,
        min_ms: timings.iter().copied().fold(f64::INFINITY, f64::min),
        max_ms: timings.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        tokens_per_second: input_ids.iter().map(Vec::len).sum::<usize>() as f64 / (mean / 1000.0),
        logits_shape: shape,
    })
}

/// Lightweight generation-quality summary over decoded strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationQualityReport {
    /// Number of generated strings.
    pub count: usize,
    /// Mean whitespace-token length.
    pub average_length: f64,
    /// Distinct unigram ratio.
    pub distinct_1: f64,
    /// Distinct bigram ratio.
    pub distinct_2: f64,
    /// Fraction of repeated whitespace tokens.
    pub repetition_rate: f64,
}

/// Evaluates decoded strings with simple diversity and repetition metrics.
pub fn evaluate_generated_texts(texts: &[String]) -> GenerationQualityReport {
    GenerationQualityReport {
        count: texts.len(),
        average_length: average_length(texts),
        distinct_1: distinct_n(texts, 1),
        distinct_2: distinct_n(texts, 2),
        repetition_rate: repetition_rate(texts),
    }
}

/// Computes mean whitespace-token length.
pub fn average_length(texts: &[String]) -> f64 {
    if texts.is_empty() {
        return 0.0;
    }
    texts
        .iter()
        .map(|t| t.split_whitespace().count())
        .sum::<usize>() as f64
        / texts.len() as f64
}

/// Computes distinct-n ratio over whitespace tokens.
pub fn distinct_n(texts: &[String], n: usize) -> f64 {
    let mut total = 0usize;
    let mut unique = std::collections::HashSet::new();
    for text in texts {
        let toks: Vec<&str> = text.split_whitespace().collect();
        if toks.len() < n {
            continue;
        }
        for window in toks.windows(n) {
            unique.insert(window.join("\u{1f}"));
            total += 1;
        }
    }
    unique.len() as f64 / total.max(1) as f64
}

/// Computes the fraction of tokens that are repeats within each string.
pub fn repetition_rate(texts: &[String]) -> f64 {
    let mut repeated = 0usize;
    let mut total = 0usize;
    for text in texts {
        let mut counts = std::collections::HashMap::new();
        for tok in text.split_whitespace() {
            *counts.entry(tok).or_insert(0usize) += 1;
            total += 1;
        }
        repeated += counts.values().map(|c| c.saturating_sub(1)).sum::<usize>();
    }
    repeated as f64 / total.max(1) as f64
}
