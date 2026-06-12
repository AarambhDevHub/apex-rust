//! CPU-safe optimizer math and training-state helpers.

use serde::{Deserialize, Serialize};

use crate::config::TrainingConfig;
use crate::error::{ApexError, Result};

/// AdamW hyperparameters used by vector-backed optimizer utilities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AdamWConfig {
    /// Learning rate for each update.
    pub lr: f64,
    /// Adam first-moment decay.
    pub beta1: f64,
    /// Adam second-moment decay.
    pub beta2: f64,
    /// Numerical stability epsilon.
    pub eps: f64,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f64,
    /// Optional global gradient norm clipping threshold.
    pub grad_clip: f64,
}

impl AdamWConfig {
    /// Creates AdamW settings from the top-level training config.
    pub fn from_training(config: &TrainingConfig) -> Self {
        Self {
            lr: config.peak_lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
            grad_clip: config.grad_clip,
        }
    }

    /// Validates AdamW hyperparameters.
    pub fn validate(&self) -> Result<()> {
        if self.lr < 0.0 {
            return Err(ApexError::Config("AdamW lr must be non-negative".into()));
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(ApexError::Config(
                "AdamW beta1 and beta2 must be in [0, 1)".into(),
            ));
        }
        if self.eps <= 0.0 {
            return Err(ApexError::Config("AdamW eps must be positive".into()));
        }
        if self.weight_decay < 0.0 || self.grad_clip < 0.0 {
            return Err(ApexError::Config(
                "AdamW weight_decay and grad_clip must be non-negative".into(),
            ));
        }
        Ok(())
    }
}

/// Per-parameter AdamW moment state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdamWState {
    /// Number of optimizer updates applied to this parameter.
    pub step: usize,
    /// First moment estimate.
    pub exp_avg: Vec<f32>,
    /// Second moment estimate.
    pub exp_avg_sq: Vec<f32>,
}

impl AdamWState {
    /// Creates zeroed AdamW state for a flat parameter vector.
    pub fn new(parameter_len: usize) -> Self {
        Self {
            step: 0,
            exp_avg: vec![0.0; parameter_len],
            exp_avg_sq: vec![0.0; parameter_len],
        }
    }
}

/// Vector-backed AdamW optimizer for tests and future CPU training loops.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdamWOptimizer {
    /// Optimizer hyperparameters.
    pub config: AdamWConfig,
    /// Per-parameter moment states.
    pub states: Vec<AdamWState>,
}

impl AdamWOptimizer {
    /// Creates an AdamW optimizer for flat parameter vector lengths.
    pub fn new(parameter_lengths: &[usize], config: AdamWConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            states: parameter_lengths
                .iter()
                .map(|&len| AdamWState::new(len))
                .collect(),
        })
    }

    /// Applies one AdamW update to all parameter vectors.
    pub fn step(
        &mut self,
        params: &mut [Vec<f32>],
        grads: &[Vec<f32>],
    ) -> Result<OptimizerStepStats> {
        if params.len() != grads.len() || params.len() != self.states.len() {
            return Err(ApexError::Shape(
                "params, grads, and optimizer states lengths must match".into(),
            ));
        }
        let mut clipped_grads = grads.to_vec();
        let grad_norm = clip_gradients(&mut clipped_grads, self.config.grad_clip)?;
        for ((param, grad), state) in params.iter_mut().zip(&clipped_grads).zip(&mut self.states) {
            adamw_update(param, grad, state, &self.config)?;
        }
        Ok(OptimizerStepStats {
            grad_norm,
            clipped: self.config.grad_clip > 0.0 && grad_norm > self.config.grad_clip,
            step: self.states.first().map(|state| state.step).unwrap_or(0),
        })
    }
}

/// Scalar stats returned after one optimizer step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OptimizerStepStats {
    /// Global gradient norm before clipping.
    pub grad_norm: f64,
    /// Whether gradients were scaled by clipping.
    pub clipped: bool,
    /// Optimizer step after the update.
    pub step: usize,
}

/// Applies one AdamW update to a flat parameter vector.
pub fn adamw_update(
    param: &mut [f32],
    grad: &[f32],
    state: &mut AdamWState,
    config: &AdamWConfig,
) -> Result<()> {
    config.validate()?;
    if param.len() != grad.len()
        || param.len() != state.exp_avg.len()
        || param.len() != state.exp_avg_sq.len()
    {
        return Err(ApexError::Shape(
            "param, grad, exp_avg, and exp_avg_sq lengths must match".into(),
        ));
    }
    state.step += 1;
    let beta1_correction = 1.0 - config.beta1.powi(state.step as i32);
    let beta2_correction = 1.0 - config.beta2.powi(state.step as i32);
    for (((value, &grad), exp_avg), exp_avg_sq) in param
        .iter_mut()
        .zip(grad)
        .zip(&mut state.exp_avg)
        .zip(&mut state.exp_avg_sq)
    {
        if config.weight_decay > 0.0 {
            *value *= (1.0 - config.lr * config.weight_decay) as f32;
        }
        *exp_avg = (config.beta1 as f32).mul_add(*exp_avg, (1.0 - config.beta1) as f32 * grad);
        *exp_avg_sq =
            (config.beta2 as f32).mul_add(*exp_avg_sq, (1.0 - config.beta2) as f32 * grad * grad);
        let m_hat = f64::from(*exp_avg) / beta1_correction.max(1e-16);
        let v_hat = f64::from(*exp_avg_sq) / beta2_correction.max(1e-16);
        *value -= (config.lr * m_hat / (v_hat.sqrt() + config.eps)) as f32;
    }
    Ok(())
}

/// Computes the global L2 norm across gradient vectors.
pub fn global_grad_norm(grads: &[Vec<f32>]) -> f64 {
    grads
        .iter()
        .flat_map(|grad| grad.iter())
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Clips gradients in place and returns the pre-clip global norm.
pub fn clip_gradients(grads: &mut [Vec<f32>], max_norm: f64) -> Result<f64> {
    if max_norm < 0.0 {
        return Err(ApexError::Config("max_norm must be non-negative".into()));
    }
    let norm = global_grad_norm(grads);
    if max_norm > 0.0 && norm > max_norm {
        let scale = (max_norm / (norm + 1e-12)) as f32;
        for value in grads.iter_mut().flat_map(|grad| grad.iter_mut()) {
            *value *= scale;
        }
    }
    Ok(norm)
}

/// Tracks high-level training progress across dry or real loops.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TrainingState {
    /// Global micro-step count.
    pub global_step: usize,
    /// Epoch counter.
    pub epoch: usize,
    /// Best validation loss seen so far.
    pub best_val_loss: f64,
    /// Number of micro-steps accumulated per optimizer update.
    pub gradient_accumulation_steps: usize,
}

impl TrainingState {
    /// Creates initial training state.
    pub fn new(gradient_accumulation_steps: usize) -> Self {
        Self {
            global_step: 0,
            epoch: 0,
            best_val_loss: f64::INFINITY,
            gradient_accumulation_steps: gradient_accumulation_steps.max(1),
        }
    }

    /// Advances one micro-step and returns true when an optimizer step is due.
    pub fn advance_micro_step(&mut self) -> bool {
        self.global_step += 1;
        self.global_step
            .is_multiple_of(self.gradient_accumulation_steps)
    }

    /// Updates best validation loss and returns true when it improved.
    pub fn update_best_val_loss(&mut self, loss: f64) -> bool {
        if loss < self.best_val_loss {
            self.best_val_loss = loss;
            true
        } else {
            false
        }
    }
}
