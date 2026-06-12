use apex_rust::config::get_tiny_vision_config;
use apex_rust::vision::ApexVisionModel;
use candle_core::{Device, Tensor};

fn main() -> apex_rust::Result<()> {
    let mut cfg = get_tiny_vision_config();
    cfg.model.n_layers = 2;
    cfg.attention.global_layer_freq = 2;
    cfg.moe.enabled = false;
    cfg.skip_gate.enabled = false;
    cfg.multi_token_head.enabled = false;

    let image_values =
        vec![0.5_f32; cfg.vision.in_channels * cfg.vision.image_size * cfg.vision.image_size];
    let image = Tensor::from_vec(
        image_values,
        (
            1,
            cfg.vision.in_channels,
            cfg.vision.image_size,
            cfg.vision.image_size,
        ),
        &Device::Cpu,
    )?;
    let input_ids = vec![vec![1, cfg.vision.image_token_id, 2, 3]];
    let mut model = ApexVisionModel::new(cfg.clone(), Device::Cpu)?;
    let output = model.forward(&input_ids, Some(&image), 0, true)?;

    println!("logits_shape={:?}", output.logits.dims());
    println!("visual_tokens={}", cfg.vision.n_visual_tokens);
    println!("kv_cache_layers={}", output.kv_caches.len());
    Ok(())
}
