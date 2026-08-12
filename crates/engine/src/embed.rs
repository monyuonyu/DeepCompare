//! 行から埋め込みベクトルを作るところ。
//!
//! 推論は行数に比例して重くなるので、ここで無駄な推論を落としきる。効くのは二つ。
//!
//! - **重複除去**: ソースコードには `}` や空行のように同一内容の行が大量に出る。
//!   同じ文字列は一度しか通さない。
//! - **長さで並べてから詰める**: 一括処理では最長の行に合わせて詰め物が入るので、
//!   長さがばらばらのまま束ねると、短い行のために無駄な計算をすることになる。

use crate::bert::{Bert, BertParams};
use crate::weights;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use tokenizers::{Tokenizer, TruncationParams};

/// 一度に処理する行数。大きいほど行列積の効率は上がるが、詰め物の無駄と
/// 一時領域も増える。長さ順に並べた上での値なので、これくらいで頭打ちになる。
const BATCH_SIZE: usize = 64;

pub struct Embedder {
    tokenizer: Tokenizer,
    bert: Bert,
}

impl Embedder {
    /// 埋め込まれた重みとトークナイザから組み立てる。
    ///
    /// バイト列を受け取る形にしているのは、`include_bytes!` で exe に埋める判断を
    /// 実行ファイル側に持たせるため。エンジン自身が抱えると、試験のたびに 22 MiB を
    /// 読み込むことになる。
    pub fn from_bytes(weights_blob: &[u8], tokenizer_json: &[u8]) -> Result<Self> {
        let device = Device::Cpu;
        let tensors = weights::load(weights_blob, &device).context("重みを読めない")?;
        let bert = Bert::new(tensors, BertParams::default())?;

        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("トークナイザを読めない: {e}"))?;
        // 位置埋め込みの上限を越えた入力は前向き計算で落ちる。極端に長い 1 行
        // （minify されたファイルなど）でも比較そのものは続けたいので切り詰める。
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: bert.max_position(),
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("切り詰めの設定に失敗: {e}"))?;

        Ok(Self { tokenizer, bert })
    }

    pub fn hidden_size(&self) -> usize {
        self.bert.hidden_size()
    }

    /// 各行の埋め込みを、入力と同じ順序で返す。すべて L2 正規化済みなので、
    /// 類似度は内積で取れる。
    pub fn embed_lines(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>> {
        if lines.is_empty() {
            return Ok(Vec::new());
        }

        // 同じ文字列は一度だけ通す。
        let mut unique: Vec<&str> = Vec::new();
        let mut index_of: HashMap<&str, usize> = HashMap::new();
        let mut assignment = Vec::with_capacity(lines.len());
        for line in lines {
            let idx = *index_of.entry(line).or_insert_with(|| {
                unique.push(line);
                unique.len() - 1
            });
            assignment.push(idx);
        }

        let encodings = self
            .tokenizer
            .encode_batch(unique.clone(), true)
            .map_err(|e| anyhow::anyhow!("トークナイズに失敗: {e}"))?;

        // 長さの近いものを束ねて、詰め物を減らす。
        let mut order: Vec<usize> = (0..unique.len()).collect();
        order.sort_by_key(|&i| encodings[i].get_ids().len());

        let mut embeddings = vec![Vec::new(); unique.len()];
        for chunk in order.chunks(BATCH_SIZE) {
            let max_len = chunk
                .iter()
                .map(|&i| encodings[i].get_ids().len())
                .max()
                .unwrap_or(0)
                .max(1);

            let mut ids = Vec::with_capacity(chunk.len() * max_len);
            let mut mask = Vec::with_capacity(chunk.len() * max_len);
            for &i in chunk {
                let src = encodings[i].get_ids();
                ids.extend_from_slice(src);
                mask.extend(std::iter::repeat_n(1u32, src.len()));
                // 詰め物。id は何でもよいが、mask が 0 なので出力には効かない。
                ids.extend(std::iter::repeat_n(0u32, max_len - src.len()));
                mask.extend(std::iter::repeat_n(0u32, max_len - src.len()));
            }

            let device = self.bert.device();
            let shape = (chunk.len(), max_len);
            let ids = Tensor::from_vec(ids, shape, device)?;
            let mask = Tensor::from_vec(mask, shape, device)?;

            let pooled = self.bert.embed(&ids, &mask)?.to_vec2::<f32>()?;
            for (&i, vector) in chunk.iter().zip(pooled) {
                embeddings[i] = vector;
            }
        }

        Ok(assignment
            .into_iter()
            .map(|i| embeddings[i].clone())
            .collect())
    }
}

/// 正規化済みベクトル同士のコサイン類似度。内積そのもの。
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_unit_vectors_is_one() {
        let v = vec![0.6, 0.8];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_unit_vectors_is_zero() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }
}
