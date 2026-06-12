use apex_rust::config::get_tiny_config;
use apex_rust::model::ApexModel;
use apex_rust::train::compute_pretrain_loss;
use candle_core::Device;

fn main() -> apex_rust::Result<()> {
    let cfg = get_tiny_config();
    let mut model = ApexModel::new(cfg.clone(), Device::Cpu)?;
    let tokens = vec![(0..16).map(|i| 9 + i as u32).collect::<Vec<_>>()];
    let output = model.forward(&tokens, None, 0, None, false)?;
    let metrics = compute_pretrain_loss(
        &output.logits,
        output.spec_logits.as_deref(),
        &tokens,
        cfg.multi_token_head.lambda_spec,
    )?;
    println!(
        "logits={:?} loss={:.4}",
        output.logits.dims(),
        metrics.loss_total
    );
    Ok(())
}
