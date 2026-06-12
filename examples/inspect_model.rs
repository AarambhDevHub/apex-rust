use apex_rust::config::get_tiny_config;
use apex_rust::model::ApexModel;
use apex_rust::utils::{architecture_text, ModelInspection};
use candle_core::Device;

fn main() -> apex_rust::Result<()> {
    let model = ApexModel::new(get_tiny_config(), Device::Cpu)?;
    let inspection = ModelInspection::from_model(&model);
    println!("{}", architecture_text(&model));
    println!(
        "total={} active={} trainable={}",
        inspection.parameters.total_parameters,
        inspection.parameters.active_parameters,
        inspection.parameters.trainable_parameters
    );
    Ok(())
}
