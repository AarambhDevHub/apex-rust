//! Runtime device and precision selection helpers.

use candle_core::{DType, Device};

use crate::error::{ApexError, Result};

/// Resolved execution settings for CLI commands.
#[derive(Debug, Clone)]
pub struct RuntimeBackend {
    /// Device selected for tensors and model weights.
    pub device: Device,
    /// User-facing device label.
    pub device_label: String,
    /// Requested mixed-precision mode.
    pub precision_label: String,
    /// Effective tensor dtype used by CPU-first runtime paths.
    pub compute_dtype: DType,
    /// Explanation when a requested mode falls back to CPU fp32.
    pub fallback_reason: Option<String>,
}

/// Resolves `cpu`, `auto`, or `cuda[:index]` into an executable backend.
pub fn resolve_runtime(device: &str, mixed_precision: Option<&str>) -> Result<RuntimeBackend> {
    let requested = device.trim().to_ascii_lowercase();
    let precision = mixed_precision
        .unwrap_or("fp32")
        .trim()
        .to_ascii_lowercase();
    let (device, label) = match requested.as_str() {
        "" | "auto" | "cpu" => (Device::Cpu, "cpu".to_string()),
        name if name == "cuda" || name.starts_with("cuda:") => {
            return Err(ApexError::Config(
                "CUDA was requested, but this build keeps Candle CUDA disabled until it is locally validated. Use --device cpu or --device auto.".to_string(),
            ));
        }
        other => {
            return Err(ApexError::Config(format!(
                "unsupported device '{other}', expected cpu, auto, cuda, or cuda:<index>"
            )));
        }
    };
    let (compute_dtype, fallback_reason) = match precision.as_str() {
        "" | "none" | "fp32" | "float32" => (DType::F32, None),
        "fp16" | "float16" | "bf16" | "bfloat16" => (
            DType::F32,
            Some(format!(
                "requested mixed_precision={precision}, using fp32 because the active backend is CPU"
            )),
        ),
        other => {
            return Err(ApexError::Config(format!(
                "unsupported mixed precision '{other}', expected fp32, fp16, or bf16"
            )));
        }
    };
    Ok(RuntimeBackend {
        device,
        device_label: label,
        precision_label: precision,
        compute_dtype,
        fallback_reason,
    })
}
