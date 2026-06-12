#[derive(Clone)]
pub struct LoadBalancer {
    pub n_experts: usize,
    pub target_rate: f32,
    pub alpha: f32,
    pub bias: Vec<f32>,
    pub total_updates: usize,
    pub cumulative_counts: Vec<usize>,
}

impl LoadBalancer {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadBalancerStats {
    pub max_load: f32,
    pub min_load: f32,
    pub load_std: f32,
    pub bias_min: f32,
    pub bias_max: f32,
}
