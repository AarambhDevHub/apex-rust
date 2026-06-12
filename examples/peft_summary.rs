use apex_rust::config::get_tiny_qdora_config;
use apex_rust::model::ApexModel;
use candle_core::Device;

fn main() -> apex_rust::Result<()> {
    let mut cfg = get_tiny_qdora_config();
    cfg.model.n_layers = 2;
    cfg.attention.global_layer_freq = 2;
    cfg.moe.enabled = false;
    cfg.skip_gate.enabled = false;

    let model = ApexModel::new(cfg, Device::Cpu)?;
    println!("total_parameters={}", model.total_parameters());
    println!("trainable_parameters={}", model.trainable_parameters());
    println!("lora_modules={}", model.count_lora_modules());
    println!("qlora_modules={}", model.count_qlora_modules());
    println!("dora_modules={}", model.count_dora_modules());
    Ok(())
}
