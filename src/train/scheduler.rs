//! Learning-rate schedules used by training commands.

/// Cosine-decay schedule with linear warmup and a minimum learning-rate ratio.
pub fn get_lr(
    step: usize,
    warmup_steps: usize,
    max_steps: usize,
    peak_lr: f64,
    min_lr_ratio: f64,
) -> f64 {
    if step < warmup_steps {
        return peak_lr * (step as f64 / warmup_steps.max(1) as f64);
    }
    if step >= max_steps {
        return peak_lr * min_lr_ratio;
    }
    let progress = (step - warmup_steps) as f64 / (max_steps - warmup_steps).max(1) as f64;
    let cosine = 0.5 * (1.0 + std::f64::consts::PI.mul_add(progress, 0.0).cos());
    peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
}
