//! MiniLM（BERT）の前向き計算と平均プーリング。
//!
//! `candle-transformers` を使わず自前で持っているのは二つ理由がある。あちらは
//! 使わないモデルを大量に抱えていて exe を膨らませること、そして既定で C 実装の
//! `onig` を引き込み Windows へのクロスビルドを面倒にすることの二つ。必要なのは
//! BERT の encoder 一本だけなので、そこだけ書く。
//!
//! 形の定数は可能な限り重みの次元から導く。設定値を別途持つと、モデルを差し替えた
//! ときに設定だけが古いまま残り、無言で誤った出力を出す。

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module};
use std::collections::HashMap;

/// 重みから導けない値だけを持つ。
#[derive(Debug, Clone, Copy)]
pub struct BertParams {
    pub num_attention_heads: usize,
    pub layer_norm_eps: f64,
}

impl Default for BertParams {
    /// MiniLM-L6 (hidden 384 / 12 heads / eps 1e-12)。
    fn default() -> Self {
        Self {
            num_attention_heads: 12,
            layer_norm_eps: 1e-12,
        }
    }
}

struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    fn get(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .with_context(|| format!("重みが足りない: {name}"))
    }

    fn linear(&self, prefix: &str) -> Result<Linear> {
        Ok(Linear::new(
            self.get(&format!("{prefix}.weight"))?,
            Some(self.get(&format!("{prefix}.bias"))?),
        ))
    }

    fn layer_norm(&self, prefix: &str, eps: f64) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.get(&format!("{prefix}.weight"))?,
            self.get(&format!("{prefix}.bias"))?,
            eps,
        ))
    }
}

struct Embeddings {
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    norm: LayerNorm,
}

impl Embeddings {
    fn forward(&self, ids: &Tensor, device: &Device) -> Result<Tensor> {
        let (_batch, seq_len) = ids.dims2()?;
        let words = self.word.forward(ids)?;

        let positions = Tensor::arange(0u32, seq_len as u32, device)?;
        let positions = self.position.forward(&positions)?;

        // 単一の文しか扱わないので token_type は常に 0。行列を作らず 0 行だけ引く。
        let token_types = self
            .token_type
            .forward(&Tensor::zeros(1, DType::U32, device)?)?;

        let sum = words
            .broadcast_add(&positions)?
            .broadcast_add(&token_types)?;
        Ok(self.norm.forward(&sum)?)
    }
}

struct Layer {
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    attn_norm: LayerNorm,
    intermediate: Linear,
    output: Linear,
    output_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl Layer {
    /// `mask` は加算済みのバイアス形式 `[batch, 1, 1, seq]`。実トークンで 0、
    /// 詰め物で大きな負値が入っている。
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq, _hidden) = x.dims3()?;

        let split = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((batch, seq, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = split(self.query.forward(x)?)?;
        let k = split(self.key.forward(x)?)?;
        let v = split(self.value.forward(x)?)?;

        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;

        let context = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq,
            self.num_heads * self.head_dim,
        ))?;

        // 自己注意の残差接続。
        let x = self
            .attn_norm
            .forward(&(self.attn_out.forward(&context)? + x)?)?;

        // 全結合部。HF の "gelu" は誤差関数版なので、tanh 近似の `gelu()` ではなく
        // `gelu_erf()` を使う。ここを取り違えると出力が微妙にずれる。
        let hidden = self.intermediate.forward(&x)?.gelu_erf()?;
        Ok(self
            .output_norm
            .forward(&(self.output.forward(&hidden)? + x)?)?)
    }
}

pub struct Bert {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    hidden_size: usize,
    max_position: usize,
    device: Device,
}

impl Bert {
    pub fn new(tensors: HashMap<String, Tensor>, params: BertParams) -> Result<Self> {
        let w = Weights { tensors };
        let device = w.get("embeddings.word_embeddings.weight")?.device().clone();

        // 形はすべて重みから読む。
        let (_vocab, hidden_size) = w.get("embeddings.word_embeddings.weight")?.dims2()?;
        let (max_position, _) = w.get("embeddings.position_embeddings.weight")?.dims2()?;
        anyhow::ensure!(
            hidden_size % params.num_attention_heads == 0,
            "hidden_size {hidden_size} が注意ヘッド数 {} で割り切れない",
            params.num_attention_heads
        );
        let head_dim = hidden_size / params.num_attention_heads;

        let embeddings = Embeddings {
            word: Embedding::new(w.get("embeddings.word_embeddings.weight")?, hidden_size),
            position: Embedding::new(w.get("embeddings.position_embeddings.weight")?, hidden_size),
            token_type: Embedding::new(
                w.get("embeddings.token_type_embeddings.weight")?,
                hidden_size,
            ),
            norm: w.layer_norm("embeddings.LayerNorm", params.layer_norm_eps)?,
        };

        // 層数も数えて決める。
        let num_layers = (0..)
            .take_while(|i| {
                w.tensors
                    .contains_key(&format!("encoder.layer.{i}.attention.self.query.weight"))
            })
            .count();
        anyhow::ensure!(num_layers > 0, "encoder の層が一つも見つからない");

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let p = format!("encoder.layer.{i}");
            layers.push(Layer {
                query: w.linear(&format!("{p}.attention.self.query"))?,
                key: w.linear(&format!("{p}.attention.self.key"))?,
                value: w.linear(&format!("{p}.attention.self.value"))?,
                attn_out: w.linear(&format!("{p}.attention.output.dense"))?,
                attn_norm: w.layer_norm(
                    &format!("{p}.attention.output.LayerNorm"),
                    params.layer_norm_eps,
                )?,
                intermediate: w.linear(&format!("{p}.intermediate.dense"))?,
                output: w.linear(&format!("{p}.output.dense"))?,
                output_norm: w
                    .layer_norm(&format!("{p}.output.LayerNorm"), params.layer_norm_eps)?,
                num_heads: params.num_attention_heads,
                head_dim,
            });
        }

        Ok(Self {
            embeddings,
            layers,
            hidden_size,
            max_position,
            device,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// 位置埋め込みが持てる最大長。これを越える入力は呼び出し側で切り詰める。
    pub fn max_position(&self) -> usize {
        self.max_position
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `ids` と `attention_mask` は `[batch, seq]`。詰め物の位置は mask が 0。
    /// 戻り値は平均プーリング後に L2 正規化した `[batch, hidden]`。
    ///
    /// 正規化まで済ませるのは、後段が必ずコサイン類似度を取るため。ここで一度だけ
    /// 済ませておけば、類似度は単なる内積になる。
    pub fn embed(&self, ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden = self.forward(ids, attention_mask)?;
        let pooled = mean_pool(&hidden, attention_mask)?;
        l2_normalize(&pooled)
    }

    fn forward(&self, ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (batch, seq) = ids.dims2()?;
        anyhow::ensure!(
            seq <= self.max_position,
            "系列長 {seq} が位置埋め込みの上限 {} を越えている",
            self.max_position
        );

        // 加算バイアス形式のマスクへ変換する。詰め物には -inf ではなく大きな負値を
        // 使う。全要素が詰め物の行があると -inf では softmax が NaN になるため。
        let bias =
            ((attention_mask.to_dtype(DType::F32)? - 1.0)? * 1e4)?.reshape((batch, 1, 1, seq))?;

        let mut x = self.embeddings.forward(ids, &self.device)?;
        for layer in &self.layers {
            x = layer.forward(&x, &bias)?;
        }
        Ok(x)
    }
}

/// 詰め物を除いた平均。sentence-transformers の MiniLM の既定のプーリング。
fn mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
    let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
    // 空の系列でも 0 除算しない。
    let counts = mask.sum(1)?.clamp(1e-9, f32::INFINITY)?;
    Ok(summed.broadcast_div(&counts)?)
}

fn l2_normalize(t: &Tensor) -> Result<Tensor> {
    let norm = t.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(t.broadcast_div(&norm.clamp(1e-12, f32::INFINITY)?)?)
}
