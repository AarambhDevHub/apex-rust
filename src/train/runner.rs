use crate::error::Result;
use crate::model::ApexModel;

use super::losses::{compute_pretrain_loss, LossMetrics};

pub fn dry_run_pretrain_step(model: &mut ApexModel, tokens: &[Vec<u32>]) -> Result<LossMetrics> {
    let out = model.forward(tokens, None, 0, None, false)?;
    compute_pretrain_loss(
        &out.logits,
        out.spec_logits.as_deref(),
        tokens,
        model.config.multi_token_head.lambda_spec,
    )
}
