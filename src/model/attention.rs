use candle_core::{Device, IndexOp, Tensor, D};

use crate::config::ApexConfig;
use crate::error::{ApexError, Result};
use crate::tensor;

use super::linear::LinearLayer;
use super::mask::additive_mask;
use super::rope::{apply_rope_pair, apply_rope_single};

#[derive(Clone)]
pub enum KvCache {
    Mla { c_kv: Tensor, k_rope: Tensor },
    Gqa { k: Tensor, v: Tensor },
}

#[derive(Clone)]
pub struct MlaAttention {
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub d_head: usize,
    pub d_head_rope: usize,
    pub d_kv_compressed: usize,
    pub w_dkv: LinearLayer,
    pub w_uk: LinearLayer,
    pub w_uv: LinearLayer,
    pub w_dq: LinearLayer,
    pub w_uq: LinearLayer,
    pub w_kr: LinearLayer,
    pub w_qr: LinearLayer,
    pub w_o: LinearLayer,
}

impl MlaAttention {
    pub fn new(cfg: &ApexConfig, prefix: &str, device: &Device) -> Result<Self> {
        let m = &cfg.model;
        Ok(Self {
            n_heads_q: m.n_heads_q,
            n_heads_kv: m.n_heads_kv,
            d_head: m.d_head,
            d_head_rope: m.d_head_rope,
            d_kv_compressed: m.d_kv_compressed,
            w_dkv: LinearLayer::new(
                &format!("{prefix}.W_DKV"),
                m.d_model,
                m.d_kv_compressed,
                false,
                cfg,
                device,
            )?,
            w_uk: LinearLayer::new(
                &format!("{prefix}.W_UK"),
                m.d_kv_compressed,
                m.n_heads_kv * m.d_head,
                false,
                cfg,
                device,
            )?,
            w_uv: LinearLayer::new(
                &format!("{prefix}.W_UV"),
                m.d_kv_compressed,
                m.n_heads_kv * m.d_head,
                false,
                cfg,
                device,
            )?,
            w_dq: LinearLayer::new(
                &format!("{prefix}.W_DQ"),
                m.d_model,
                m.d_q_compressed,
                false,
                cfg,
                device,
            )?,
            w_uq: LinearLayer::new(
                &format!("{prefix}.W_UQ"),
                m.d_q_compressed,
                m.n_heads_q * m.d_head,
                false,
                cfg,
                device,
            )?,
            w_kr: LinearLayer::new(
                &format!("{prefix}.W_KR"),
                m.d_model,
                m.n_heads_kv * m.d_head_rope,
                false,
                cfg,
                device,
            )?,
            w_qr: LinearLayer::new(
                &format!("{prefix}.W_QR"),
                m.d_model,
                m.n_heads_q * m.d_head_rope,
                false,
                cfg,
                device,
            )?,
            w_o: LinearLayer::new(
                &format!("{prefix}.W_O"),
                m.n_heads_q * m.d_head,
                m.d_model,
                false,
                cfg,
                device,
            )?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &[usize],
        mask: Option<&[Vec<bool>]>,
        cache: Option<&KvCache>,
    ) -> Result<(Tensor, KvCache)> {
        let (b, s, _) = dims3(x, "MLA input")?;
        let c_kv_new = self.w_dkv.forward(x)?;
        let k_rope_new = self
            .w_kr
            .forward(x)?
            .reshape((b, s, self.n_heads_kv, self.d_head_rope))?
            .transpose(1, 2)?;
        let k_rope_new = apply_rope_single(&k_rope_new, cos, sin, positions)?;
        let (c_kv_full, k_rope_full) = match cache {
            Some(KvCache::Mla { c_kv, k_rope }) => (
                Tensor::cat(&[c_kv, &c_kv_new], 1)?,
                Tensor::cat(&[k_rope, &k_rope_new], 2)?,
            ),
            Some(_) => return Err(ApexError::Model("expected MLA cache".to_string())),
            None => (c_kv_new, k_rope_new),
        };
        let full = c_kv_full.dim(1)?;
        let k_content = self
            .w_uk
            .forward(&c_kv_full)?
            .reshape((b, full, self.n_heads_kv, self.d_head))?
            .transpose(1, 2)?;
        let v = self
            .w_uv
            .forward(&c_kv_full)?
            .reshape((b, full, self.n_heads_kv, self.d_head))?
            .transpose(1, 2)?;
        let q_content = self
            .w_uq
            .forward(&self.w_dq.forward(x)?)?
            .reshape((b, s, self.n_heads_q, self.d_head))?
            .transpose(1, 2)?;
        let q_rope = self
            .w_qr
            .forward(x)?
            .reshape((b, s, self.n_heads_q, self.d_head_rope))?
            .transpose(1, 2)?;
        let q_rope = apply_rope_single(&q_rope, cos, sin, positions)?;
        let q = Tensor::cat(&[&q_content, &q_rope], D::Minus1)?;
        let k = Tensor::cat(&[&k_content, &k_rope_full], D::Minus1)?;
        let groups = self.n_heads_q / self.n_heads_kv;
        let k = tensor::repeat_kv_heads(&k, groups)?;
        let v = tensor::repeat_kv_heads(&v, groups)?;
        let out = scaled_attention(&q, &k, &v, mask, self.d_head + self.d_head_rope)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads_q * self.d_head))?;
        Ok((
            self.w_o.forward(&out)?,
            KvCache::Mla {
                c_kv: c_kv_full,
                k_rope: k_rope_full,
            },
        ))
    }

    pub fn parameters(&self) -> usize {
        [
            &self.w_dkv,
            &self.w_uk,
            &self.w_uv,
            &self.w_dq,
            &self.w_uq,
            &self.w_kr,
            &self.w_qr,
            &self.w_o,
        ]
        .iter()
        .map(|l| l.parameters())
        .sum()
    }

    pub fn trainable_parameters(&self) -> usize {
        [
            &self.w_dkv,
            &self.w_uk,
            &self.w_uv,
            &self.w_dq,
            &self.w_uq,
            &self.w_kr,
            &self.w_qr,
            &self.w_o,
        ]
        .iter()
        .map(|l| l.trainable_parameters())
        .sum()
    }

    pub fn merge_and_unload(&mut self) -> Result<()> {
        for layer in [
            &mut self.w_dkv,
            &mut self.w_uk,
            &mut self.w_uv,
            &mut self.w_dq,
            &mut self.w_uq,
            &mut self.w_kr,
            &mut self.w_qr,
            &mut self.w_o,
        ] {
            layer.merge_and_unload()?;
        }
        Ok(())
    }

    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) -> Result<()> {
        self.w_dkv.named_tensors(&format!("{prefix}.W_DKV"), out)?;
        self.w_uk.named_tensors(&format!("{prefix}.W_UK"), out)?;
        self.w_uv.named_tensors(&format!("{prefix}.W_UV"), out)?;
        self.w_dq.named_tensors(&format!("{prefix}.W_DQ"), out)?;
        self.w_uq.named_tensors(&format!("{prefix}.W_UQ"), out)?;
        self.w_kr.named_tensors(&format!("{prefix}.W_KR"), out)?;
        self.w_qr.named_tensors(&format!("{prefix}.W_QR"), out)?;
        self.w_o.named_tensors(&format!("{prefix}.W_O"), out)?;
        Ok(())
    }

    pub fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        self.w_dkv.adapter_tensors(&format!("{prefix}.W_DKV"), out);
        self.w_uk.adapter_tensors(&format!("{prefix}.W_UK"), out);
        self.w_uv.adapter_tensors(&format!("{prefix}.W_UV"), out);
        self.w_dq.adapter_tensors(&format!("{prefix}.W_DQ"), out);
        self.w_uq.adapter_tensors(&format!("{prefix}.W_UQ"), out);
        self.w_kr.adapter_tensors(&format!("{prefix}.W_KR"), out);
        self.w_qr.adapter_tensors(&format!("{prefix}.W_QR"), out);
        self.w_o.adapter_tensors(&format!("{prefix}.W_O"), out);
    }
}

#[derive(Clone)]
pub struct GqaAttention {
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub d_head: usize,
    pub local_window: usize,
    pub w_q: LinearLayer,
    pub w_k: LinearLayer,
    pub w_v: LinearLayer,
    pub w_o: LinearLayer,
}

impl GqaAttention {
    pub fn new(cfg: &ApexConfig, prefix: &str, device: &Device) -> Result<Self> {
        let m = &cfg.model;
        Ok(Self {
            n_heads_q: m.n_heads_q,
            n_heads_kv: m.n_heads_kv,
            d_head: m.d_head,
            local_window: cfg.attention.local_window,
            w_q: LinearLayer::new(
                &format!("{prefix}.W_Q"),
                m.d_model,
                m.n_heads_q * m.d_head,
                false,
                cfg,
                device,
            )?,
            w_k: LinearLayer::new(
                &format!("{prefix}.W_K"),
                m.d_model,
                m.n_heads_kv * m.d_head,
                false,
                cfg,
                device,
            )?,
            w_v: LinearLayer::new(
                &format!("{prefix}.W_V"),
                m.d_model,
                m.n_heads_kv * m.d_head,
                false,
                cfg,
                device,
            )?,
            w_o: LinearLayer::new(
                &format!("{prefix}.W_O"),
                m.n_heads_q * m.d_head,
                m.d_model,
                false,
                cfg,
                device,
            )?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &[usize],
        mask: Option<&[Vec<bool>]>,
        cache: Option<&KvCache>,
    ) -> Result<(Tensor, KvCache)> {
        let (b, s, _) = dims3(x, "GQA input")?;
        let mut q = self
            .w_q
            .forward(x)?
            .reshape((b, s, self.n_heads_q, self.d_head))?
            .transpose(1, 2)?;
        let mut k = self
            .w_k
            .forward(x)?
            .reshape((b, s, self.n_heads_kv, self.d_head))?
            .transpose(1, 2)?;
        let mut v = self
            .w_v
            .forward(x)?
            .reshape((b, s, self.n_heads_kv, self.d_head))?
            .transpose(1, 2)?;
        let (qr, kr) = apply_rope_pair(&q, &k, cos, sin, positions)?;
        q = qr;
        k = kr;
        if let Some(KvCache::Gqa {
            k: prev_k,
            v: prev_v,
        }) = cache
        {
            k = Tensor::cat(&[prev_k, &k], 2)?;
            v = Tensor::cat(&[prev_v, &v], 2)?;
        } else if cache.is_some() {
            return Err(ApexError::Model("expected GQA cache".to_string()));
        }
        let kv_len = k.dim(2)?;
        if kv_len > self.local_window {
            k = k.i((.., .., kv_len - self.local_window.., ..))?;
            v = v.i((.., .., kv_len - self.local_window.., ..))?;
        }
        let cache = KvCache::Gqa {
            k: k.clone(),
            v: v.clone(),
        };
        let groups = self.n_heads_q / self.n_heads_kv;
        let k = tensor::repeat_kv_heads(&k, groups)?;
        let v = tensor::repeat_kv_heads(&v, groups)?;
        let out = scaled_attention(&q, &k, &v, mask, self.d_head)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads_q * self.d_head))?;
        Ok((self.w_o.forward(&out)?, cache))
    }

    pub fn parameters(&self) -> usize {
        [&self.w_q, &self.w_k, &self.w_v, &self.w_o]
            .iter()
            .map(|l| l.parameters())
            .sum()
    }

    pub fn trainable_parameters(&self) -> usize {
        [&self.w_q, &self.w_k, &self.w_v, &self.w_o]
            .iter()
            .map(|l| l.trainable_parameters())
            .sum()
    }

    pub fn merge_and_unload(&mut self) -> Result<()> {
        for layer in [&mut self.w_q, &mut self.w_k, &mut self.w_v, &mut self.w_o] {
            layer.merge_and_unload()?;
        }
        Ok(())
    }

    pub fn named_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) -> Result<()> {
        self.w_q.named_tensors(&format!("{prefix}.W_Q"), out)?;
        self.w_k.named_tensors(&format!("{prefix}.W_K"), out)?;
        self.w_v.named_tensors(&format!("{prefix}.W_V"), out)?;
        self.w_o.named_tensors(&format!("{prefix}.W_O"), out)?;
        Ok(())
    }

    pub fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        self.w_q.adapter_tensors(&format!("{prefix}.W_Q"), out);
        self.w_k.adapter_tensors(&format!("{prefix}.W_K"), out);
        self.w_v.adapter_tensors(&format!("{prefix}.W_V"), out);
        self.w_o.adapter_tensors(&format!("{prefix}.W_O"), out);
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum AttentionKind {
    Mla(MlaAttention),
    Gqa(GqaAttention),
}

impl AttentionKind {
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &[usize],
        mask: Option<&[Vec<bool>]>,
        cache: Option<&KvCache>,
    ) -> Result<(Tensor, KvCache)> {
        match self {
            Self::Mla(a) => a.forward(x, cos, sin, positions, mask, cache),
            Self::Gqa(a) => a.forward(x, cos, sin, positions, mask, cache),
        }
    }

    pub(crate) fn parameters(&self) -> usize {
        match self {
            Self::Mla(a) => a.parameters(),
            Self::Gqa(a) => a.parameters(),
        }
    }

    pub(crate) fn trainable_parameters(&self) -> usize {
        match self {
            Self::Mla(a) => a.trainable_parameters(),
            Self::Gqa(a) => a.trainable_parameters(),
        }
    }

    pub(crate) fn merge_and_unload(&mut self) -> Result<()> {
        match self {
            Self::Mla(a) => a.merge_and_unload(),
            Self::Gqa(a) => a.merge_and_unload(),
        }
    }

    pub(crate) fn named_tensors(
        &self,
        prefix: &str,
        out: &mut Vec<(String, Tensor)>,
    ) -> Result<()> {
        match self {
            Self::Mla(a) => a.named_tensors(prefix, out),
            Self::Gqa(a) => a.named_tensors(prefix, out),
        }
    }

    pub(crate) fn adapter_tensors(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        match self {
            Self::Mla(a) => a.adapter_tensors(prefix, out),
            Self::Gqa(a) => a.adapter_tensors(prefix, out),
        }
    }
}

fn dims3(t: &Tensor, label: &str) -> Result<(usize, usize, usize)> {
    let dims = t.dims();
    if dims.len() != 3 {
        return Err(ApexError::Shape(format!(
            "{label} expects rank 3, got {dims:?}"
        )));
    }
    Ok((dims[0], dims[1], dims[2]))
}

fn scaled_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&[Vec<bool>]>,
    scale_dim: usize,
) -> Result<Tensor> {
    let mut scores = q.matmul(&k.transpose(2, 3)?)?;
    scores = scores.broadcast_div(&tensor::scalar((scale_dim as f64).sqrt(), q.device())?)?;
    if let Some(mask) = mask {
        let q_len = q.dim(2)?;
        let kv_len = k.dim(2)?;
        let add = additive_mask(mask, q_len, kv_len, q.device())?;
        scores = scores.broadcast_add(&add)?;
    }
    let weights = tensor::softmax_last(&scores)?;
    Ok(weights.matmul(v)?)
}
