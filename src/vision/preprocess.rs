use candle_core::{Device, Tensor};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};

#[derive(Debug, Clone)]
pub struct ImageTensor {
    pub pixels: Vec<f32>,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl ImageTensor {
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

    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self.pixels.clone(),
            (1, self.channels, self.height, self.width),
            device,
        )?)
    }
}

#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    pub image_size: usize,
    pub channels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImagePreprocessor {
    pub fn new(cfg: &ApexConfig) -> Self {
        Self {
            image_size: cfg.vision.image_size,
            channels: cfg.vision.in_channels,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

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
