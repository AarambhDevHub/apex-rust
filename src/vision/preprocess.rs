use candle_core::{Device, Tensor};
#[cfg(feature = "image")]
use std::path::Path;

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

    /// Creates an image tensor from CHW pixels.
    pub fn from_chw(
        pixels: Vec<f32>,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        Self::new(pixels, channels, height, width)
    }

    /// Creates an image tensor from HWC pixels and converts them to CHW layout.
    pub fn from_hwc(
        pixels: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        if pixels.len() != height * width * channels {
            return Err(ApexError::Shape(format!(
                "HWC image pixels length {} does not match HxWxC {}",
                pixels.len(),
                height * width * channels
            )));
        }
        let mut chw = vec![0.0_f32; pixels.len()];
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let src = (y * width + x) * channels + c;
                    let dst = (c * height + y) * width + x;
                    chw[dst] = pixels[src];
                }
            }
        }
        Self::new(chw, channels, height, width)
    }

    /// Creates an image tensor from HWC bytes in the `[0, 255]` range.
    pub fn from_hwc_u8(
        pixels: Vec<u8>,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        let floats = pixels
            .into_iter()
            .map(|v| f32::from(v) / 255.0)
            .collect::<Vec<_>>();
        Self::from_hwc(floats, height, width, channels)
    }

    /// Returns a copy clamped to `[0, 1]`, dividing by 255 when values look byte-scaled.
    pub fn to_unit_range(&self) -> Result<Self> {
        let max = self
            .pixels
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let scale = if max > 2.0 { 255.0 } else { 1.0 };
        let pixels = self
            .pixels
            .iter()
            .map(|v| (*v / scale).clamp(0.0, 1.0))
            .collect();
        Self::new(pixels, self.channels, self.height, self.width)
    }

    /// Converts between one-channel grayscale and three-channel RGB when needed.
    pub fn to_channels(&self, channels: usize) -> Result<Self> {
        if self.channels == channels {
            return Ok(self.clone());
        }
        match (self.channels, channels) {
            (1, 3) => {
                let plane = self.height * self.width;
                let mut pixels = Vec::with_capacity(3 * plane);
                for _ in 0..3 {
                    pixels.extend_from_slice(&self.pixels[..plane]);
                }
                Self::new(pixels, 3, self.height, self.width)
            }
            (3, 1) => {
                let plane = self.height * self.width;
                let mut pixels = Vec::with_capacity(plane);
                for idx in 0..plane {
                    let r = self.pixels[idx];
                    let g = self.pixels[plane + idx];
                    let b = self.pixels[2 * plane + idx];
                    pixels.push(0.299 * r + 0.587 * g + 0.114 * b);
                }
                Self::new(pixels, 1, self.height, self.width)
            }
            _ => Err(ApexError::Shape(format!(
                "cannot convert image from {} to {channels} channels",
                self.channels
            ))),
        }
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
            mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
        }
    }

    /// Loads an image file, converts it to RGB, resizes, and normalizes it.
    #[cfg(feature = "image")]
    pub fn load_path(&self, path: impl AsRef<Path>) -> Result<ImageTensor> {
        let path = path.as_ref();
        let image = image::ImageReader::open(path)?
            .decode()
            .map_err(|e| {
                ApexError::Data(format!("failed to decode image {}: {e}", path.display()))
            })?
            .to_rgb8();
        let (width, height) = image.dimensions();
        let raw = image.into_raw();
        let image = ImageTensor::from_hwc_u8(raw, height as usize, width as usize, 3)?;
        self.preprocess(&image)
    }

    /// Loads an image file and returns a `[1,C,H,W]` tensor.
    #[cfg(feature = "image")]
    pub fn load_path_to_tensor(&self, path: impl AsRef<Path>, device: &Device) -> Result<Tensor> {
        self.load_path(path)?.to_tensor(device)
    }

    /// Converts an arbitrary CHW image buffer into the model-ready image tensor.
    pub fn preprocess(&self, image: &ImageTensor) -> Result<ImageTensor> {
        let image = image.to_unit_range()?.to_channels(self.channels)?;
        let image = self.resize_chw(&image)?;
        self.normalize_chw(&image)
    }

    /// Converts an image buffer into a `[1,C,H,W]` tensor.
    pub fn preprocess_to_tensor(&self, image: &ImageTensor, device: &Device) -> Result<Tensor> {
        self.preprocess(image)?.to_tensor(device)
    }

    /// Converts a batch of image buffers into a `[B,C,H,W]` tensor.
    pub fn batch_to_tensor(&self, images: &[ImageTensor], device: &Device) -> Result<Tensor> {
        if images.is_empty() {
            return Err(ApexError::Data("images must not be empty".to_string()));
        }
        let mut values = Vec::new();
        for image in images {
            let processed = self.preprocess(image)?;
            values.extend_from_slice(&processed.pixels);
        }
        Ok(Tensor::from_vec(
            values,
            (
                images.len(),
                self.channels,
                self.image_size,
                self.image_size,
            ),
            device,
        )?)
    }

    /// Resizes a CHW image to the configured square image size using bilinear sampling.
    pub fn resize_chw(&self, image: &ImageTensor) -> Result<ImageTensor> {
        resize_bilinear_chw(image, self.image_size, self.image_size)
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

fn resize_bilinear_chw(
    image: &ImageTensor,
    out_height: usize,
    out_width: usize,
) -> Result<ImageTensor> {
    if out_height == 0 || out_width == 0 {
        return Err(ApexError::Shape(
            "resize target height and width must be positive".to_string(),
        ));
    }
    if image.height == 0 || image.width == 0 {
        return Err(ApexError::Shape(
            "source image height and width must be positive".to_string(),
        ));
    }
    if image.height == out_height && image.width == out_width {
        return Ok(image.clone());
    }

    let mut out = vec![0.0_f32; image.channels * out_height * out_width];
    let y_scale = if out_height == 1 {
        0.0
    } else {
        (image.height - 1) as f32 / (out_height - 1) as f32
    };
    let x_scale = if out_width == 1 {
        0.0
    } else {
        (image.width - 1) as f32 / (out_width - 1) as f32
    };

    for c in 0..image.channels {
        for y in 0..out_height {
            let src_y = y as f32 * y_scale;
            let y0 = src_y.floor() as usize;
            let y1 = (y0 + 1).min(image.height - 1);
            let wy = src_y - y0 as f32;
            for x in 0..out_width {
                let src_x = x as f32 * x_scale;
                let x0 = src_x.floor() as usize;
                let x1 = (x0 + 1).min(image.width - 1);
                let wx = src_x - x0 as f32;

                let top =
                    sample_chw(image, c, y0, x0) * (1.0 - wx) + sample_chw(image, c, y0, x1) * wx;
                let bottom =
                    sample_chw(image, c, y1, x0) * (1.0 - wx) + sample_chw(image, c, y1, x1) * wx;
                let value = top * (1.0 - wy) + bottom * wy;
                out[(c * out_height + y) * out_width + x] = value;
            }
        }
    }
    ImageTensor::new(out, image.channels, out_height, out_width)
}

fn sample_chw(image: &ImageTensor, channel: usize, y: usize, x: usize) -> f32 {
    image.pixels[(channel * image.height + y) * image.width + x]
}
