use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::Tensor;
use safetensors::{serialize_to_file, Dtype, SafeTensors, View};
use serde::{Deserialize, Serialize};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};
use crate::model::{load_adapter_tensors, load_full_tensors, ApexModel, TensorLoadReport};

/// JSON sidecar metadata for a saved checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Metadata format marker.
    pub format: String,
    /// Crate version that wrote the checkpoint.
    pub version: String,
    /// Training step at save time.
    pub step: usize,
    /// Epoch number at save time.
    pub epoch: usize,
    /// Last reported loss value.
    pub loss: f64,
    /// Optional full training/model config.
    pub config: Option<ApexConfig>,
    /// Human-readable note about the payload files.
    pub note: String,
}

/// Writes checkpoint metadata as pretty JSON.
pub fn save_checkpoint_metadata(
    path: impl AsRef<Path>,
    step: usize,
    epoch: usize,
    loss: f64,
    config: Option<&ApexConfig>,
) -> Result<PathBuf> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let metadata = CheckpointMetadata {
        format: "apex_rust_checkpoint_metadata".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        step,
        epoch,
        loss,
        config: config.cloned(),
        note: "Tensor payloads are written as safetensors with JSON metadata sidecars.".to_string(),
    };
    fs::write(path, serde_json::to_string_pretty(&metadata)?)?;
    Ok(path.to_path_buf())
}

/// Owned safetensors view backed by little-endian f32 bytes.
#[derive(Clone)]
struct OwnedSafeTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for OwnedSafeTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Saves all model tensors into a safetensors file.
pub fn save_model_safetensors(path: impl AsRef<Path>, model: &ApexModel) -> Result<PathBuf> {
    save_named_tensors(path, model.named_tensors()?, "full_model")
}

/// Saves only adapter tensors into a safetensors file.
pub fn save_adapter_safetensors(path: impl AsRef<Path>, model: &ApexModel) -> Result<PathBuf> {
    save_named_tensors(path, model.adapter_tensors(), "adapter")
}

/// Loads a full model safetensors file into an existing model.
pub fn load_model_safetensors(
    path: impl AsRef<Path>,
    model: &mut ApexModel,
    strict: bool,
) -> Result<TensorLoadReport> {
    let tensors = read_safetensors(path, &model.device)?;
    let report = load_full_tensors(model, &tensors, strict)?;
    enforce_strict("model", &report, strict)?;
    Ok(report)
}

/// Loads an adapter-only safetensors file into an adapter-enabled model.
pub fn load_adapter_safetensors(
    path: impl AsRef<Path>,
    model: &mut ApexModel,
    strict: bool,
) -> Result<TensorLoadReport> {
    let tensors = read_safetensors(path, &model.device)?;
    let report = load_adapter_tensors(model, &tensors, strict)?;
    enforce_strict("adapter", &report, strict)?;
    Ok(report)
}

fn save_named_tensors(
    path: impl AsRef<Path>,
    tensors: Vec<(String, candle_core::Tensor)>,
    kind: &str,
) -> Result<PathBuf> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut views = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let shape = tensor.dims().to_vec();
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        let mut data = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        views.push((
            name,
            OwnedSafeTensor {
                dtype: Dtype::F32,
                shape,
                data,
            },
        ));
    }
    let metadata = HashMap::from([
        ("format".to_string(), "apex_rust_safetensors".to_string()),
        ("kind".to_string(), kind.to_string()),
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
    ]);
    serialize_to_file(views, Some(metadata), path)
        .map_err(|e| ApexError::Checkpoint(format!("failed to write safetensors: {e}")))?;
    Ok(path.to_path_buf())
}

fn read_safetensors(
    path: impl AsRef<Path>,
    device: &candle_core::Device,
) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let safe = SafeTensors::deserialize(&bytes)
        .map_err(|e| ApexError::Checkpoint(format!("failed to read safetensors: {e}")))?;
    let mut tensors = HashMap::new();
    for name in safe.names() {
        let view = safe
            .tensor(name)
            .map_err(|e| ApexError::Checkpoint(format!("failed to read tensor {name}: {e}")))?;
        if view.dtype() != Dtype::F32 {
            return Err(ApexError::Checkpoint(format!(
                "tensor {name} has dtype {:?}; only F32 is supported",
                view.dtype()
            )));
        }
        let data = view.data();
        if data.len() % std::mem::size_of::<f32>() != 0 {
            return Err(ApexError::Checkpoint(format!(
                "tensor {name} byte length {} is not divisible by 4",
                data.len()
            )));
        }
        let values = data
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        tensors.insert(
            name.to_string(),
            Tensor::from_vec(values, view.shape(), device)?,
        );
    }
    Ok(tensors)
}

fn enforce_strict(kind: &str, report: &TensorLoadReport, strict: bool) -> Result<()> {
    if strict && (!report.missing.is_empty() || !report.unexpected.is_empty()) {
        return Err(ApexError::Checkpoint(format!(
            "{kind} load mismatch: missing={} unexpected={}",
            summarize_names(&report.missing),
            summarize_names(&report.unexpected)
        )));
    }
    Ok(())
}

fn summarize_names(names: &[String]) -> String {
    const MAX_NAMES: usize = 8;
    if names.is_empty() {
        return "[]".to_string();
    }
    let shown = names
        .iter()
        .take(MAX_NAMES)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > MAX_NAMES {
        format!("[{shown}, ... +{} more]", names.len() - MAX_NAMES)
    } else {
        format!("[{shown}]")
    }
}
