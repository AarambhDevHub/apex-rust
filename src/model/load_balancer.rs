/// Tracks MoE expert usage and updates router bias for load balancing.
#[derive(Clone)]
pub struct LoadBalancer {
    /// Number of routed experts.
    pub n_experts: usize,
    /// Ideal usage rate per expert.
    pub target_rate: f32,
    /// Bias update strength.
    pub alpha: f32,
    /// Current additive router bias per expert.
    pub bias: Vec<f32>,
    /// Number of update calls processed.
    pub total_updates: usize,
    /// Cumulative selected-token counts per expert.
    pub cumulative_counts: Vec<usize>,
}

impl LoadBalancer {
    /// Creates a new load balancer with zero expert bias.
    pub fn new(n_experts: usize, alpha: f64) -> Self {
        Self {
            n_experts,
            target_rate: 1.0 / n_experts as f32,
            alpha: alpha as f32,
            bias: vec![0.0; n_experts],
            total_updates: 0,
            cumulative_counts: vec![0; n_experts],
        }
    }

    /// Updates expert usage counts and returns load-balance statistics.
    pub fn update(&mut self, top_k_idx: &[Vec<usize>]) -> LoadBalancerStats {
        let mut counts = vec![0usize; self.n_experts];
        let mut total = 0usize;
        for row in top_k_idx {
            for &idx in row {
                if idx < self.n_experts {
                    counts[idx] += 1;
                    total += 1;
                }
            }
        }
        let mut rates = vec![0.0_f32; self.n_experts];
        for i in 0..self.n_experts {
            rates[i] = counts[i] as f32 / total.max(1) as f32;
            let delta = self.target_rate - rates[i];
            self.bias[i] = (self.bias[i] + self.alpha * delta.signum()).clamp(-1.0, 1.0);
            self.cumulative_counts[i] += counts[i];
        }
        self.total_updates += 1;
        let max_load = rates.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_load = rates.iter().copied().fold(f32::INFINITY, f32::min);
        let mean = rates.iter().sum::<f32>() / self.n_experts.max(1) as f32;
        let load_std = (rates.iter().map(|r| (*r - mean).powi(2)).sum::<f32>()
            / self.n_experts.max(1) as f32)
            .sqrt();
        LoadBalancerStats {
            max_load,
            min_load,
            load_std,
            bias_min: self.bias.iter().copied().fold(f32::INFINITY, f32::min),
            bias_max: self.bias.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        }
    }
}

/// Summary of current expert usage and bias range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadBalancerStats {
    /// Highest expert selection rate in the update batch.
    pub max_load: f32,
    /// Lowest expert selection rate in the update batch.
    pub min_load: f32,
    /// Standard deviation of expert selection rates.
    pub load_std: f32,
    /// Minimum router bias after the update.
    pub bias_min: f32,
    /// Maximum router bias after the update.
    pub bias_max: f32,
}
