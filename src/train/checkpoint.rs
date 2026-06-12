use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::{serialize_to_file, Dtype, View};
use serde::{Deserialize, Serialize};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};
use crate::model::ApexModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub format: String,
    pub version: String,
    pub step: usize,
    pub epoch: usize,
    pub loss: f64,
    pub config: Option<ApexConfig>,
    pub note: String,
}

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

pub fn save_model_safetensors(path: impl AsRef<Path>, model: &ApexModel) -> Result<PathBuf> {
    save_named_tensors(path, model.named_tensors()?, "full_model")
}

pub fn save_adapter_safetensors(path: impl AsRef<Path>, model: &ApexModel) -> Result<PathBuf> {
    save_named_tensors(path, model.adapter_tensors(), "adapter")
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
