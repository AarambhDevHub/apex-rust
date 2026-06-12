use candle_core::{Device, Tensor};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};

/// CHW image buffer normalized or ready to normalize.
#[derive(Debug, Clone)]
pub struct ImageTensor {
    /// Flat pixel values in channel-major order.
    pub pixels: Vec<f32>,
    /// Number of channels.
    pub channels: usize,
    /// Image height.
    pub height: usize,
    /// Image width.
    pub width: usize,
}

impl ImageTensor {
    /// Creates an image tensor after validating buffer length.
    pub fn new(pixels: Vec<f32>, channels: usize, height: usize, width: usize) -> Result<Self> {
        if pixels.len() != channels * height * width {
            return Err(ApexError::Shape(format!(
                "image pixels length {} does not match CxHxW {}",
                pixels.len(),
                channels * height * width
            )));
        }
        Ok(Self {
            pixels,
            channels,
            height,
            width,
        })
    }

    /// Converts the image buffer to a Candle tensor with shape `[1,C,H,W]`.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self.pixels.clone(),
            (1, self.channels, self.height, self.width),
            device,
        )?)
    }
}

/// Image preprocessing configuration for CHW tensors.
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    /// Expected square image size.
    pub image_size: usize,
    /// Expected channel count.
    pub channels: usize,
    /// Per-channel normalization mean.
    pub mean: [f32; 3],
    /// Per-channel normalization standard deviation.
    pub std: [f32; 3],
}

impl ImagePreprocessor {
    /// Creates a preprocessor from the vision config.
    pub fn new(cfg: &ApexConfig) -> Self {
        Self {
            image_size: cfg.vision.image_size,
            channels: cfg.vision.in_channels,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Normalizes a CHW image with ImageNet-style mean and standard deviation.
    pub fn normalize_chw(&self, image: &ImageTensor) -> Result<ImageTensor> {
        if image.channels != self.channels {
            return Err(ApexError::Shape(format!(
                "expected {} image channels, got {}",
                self.channels, image.channels
            )));
        }
        if image.height != self.image_size || image.width != self.image_size {
            return Err(ApexError::Shape(format!(
                "expected {}x{} image, got {}x{}; resize before calling normalize_chw",
                self.image_size, self.image_size, image.height, image.width
            )));
        }
        let plane = image.height * image.width;
        let mut pixels = image.pixels.clone();
        for channel in 0..image.channels {
            let mean = self.mean[channel.min(2)];
            let std = self.std[channel.min(2)];
            for pixel in pixels
                .iter_mut()
                .take((channel + 1) * plane)
                .skip(channel * plane)
            {
                *pixel = (*pixel - mean) / std;
            }
        }
        ImageTensor::new(pixels, image.channels, image.height, image.width)
    }
}
