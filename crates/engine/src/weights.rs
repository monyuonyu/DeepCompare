//! exe に埋め込む重みの容器。**読み書き両方の権威をこのファイルに置く**。
//!
//! 目的は exe を小さくすることだけで、推論を速くすることではない。よって行ごとの
//! 対称 int8 で保存し、読み込み時に f32 へ戻す。推論の演算は f32 のままなので、
//! 量子化 BERT を書き起こす必要がない。
//!
//! 書き出しは `deepcompare-model-prep`、読み込みはこの中の [`load`]。片方だけを
//! 直すと無言で壊れるので、定義を分割しないこと。

use anyhow::{bail, ensure, Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

pub const MAGIC: &[u8; 4] = b"DCM1";

/// 重みひとつの保存形式。
pub mod kind {
    /// f32 をそのまま。バイアスや LayerNorm など、小さくて精度が効く物に使う。
    pub const F32: u8 = 0;
    /// 行ごとの対称 int8。`value = q as f32 * scale[row]`。
    pub const Q8_PER_ROW: u8 = 1;
}

/// int8 化する下限。これを下回る物は f32 のままでも合計サイズにほぼ効かず、
/// 量子化誤差だけが乗るので触らない。
pub const QUANTIZE_MIN_ELEMS: usize = 4096;

// ---------------------------------------------------------------- 書き出し側

pub fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn write_str(out: &mut Vec<u8>, s: &str) {
    write_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

pub fn write_shape(out: &mut Vec<u8>, shape: &[usize]) {
    out.push(shape.len() as u8);
    for &d in shape {
        write_u32(out, d as u32);
    }
}

// ---------------------------------------------------------------- 読み込み側

/// 先頭から順に読むだけの薄いカーソル。長さ確認を一箇所に集約して、
/// 壊れた入力で添字が飛ぶのを防ぐ。
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("長さが範囲外")?;
        ensure!(end <= self.data.len(), "重みの終端を越えて読もうとした");
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        Ok(String::from_utf8(self.take(len)?.to_vec())?)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let bytes = self.take(n * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// 埋め込まれた重みを名前つきのテンソルへ展開する。
pub fn load(data: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut r = Reader { data, pos: 0 };
    ensure!(r.take(4)? == MAGIC, "重みの形式が違う（MAGIC 不一致）");
    let count = r.u32()? as usize;

    let mut out = HashMap::with_capacity(count);
    for _ in 0..count {
        let name = r.string()?;
        let kind = r.u8()?;
        let rank = r.u8()? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(r.u32()? as usize);
        }
        let numel: usize = shape.iter().product();

        let values = match kind {
            kind::F32 => r.f32s(numel)?,
            kind::Q8_PER_ROW => {
                ensure!(rank == 2, "{name}: 行ごと量子化は 2 次元のみ");
                let (rows, cols) = (shape[0], shape[1]);
                let scales = r.f32s(rows)?;
                let quantized = r.take(numel)?;
                let mut values = Vec::with_capacity(numel);
                for row in 0..rows {
                    let scale = scales[row];
                    for &q in &quantized[row * cols..(row + 1) * cols] {
                        values.push(q as i8 as f32 * scale);
                    }
                }
                values
            }
            other => bail!("{name}: 未知の保存形式 {other}"),
        };

        let tensor = Tensor::from_vec(values, shape, device)
            .with_context(|| format!("{name}: テンソルを作れない"))?;
        out.insert(name, tensor);
    }
    ensure!(r.pos == data.len(), "重みの末尾に余分なバイトがある");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 書き出しと読み込みが同じ形式を指していることの担保。
    /// ここが割れると、モデルは読めるのに出力だけが静かに壊れる。
    #[test]
    fn round_trip_preserves_shape_and_values() {
        let rows = 3usize;
        let cols = 2usize;
        let values: Vec<f32> = vec![1.0, -0.5, 0.25, 0.125, -2.0, 2.0];

        let mut body = Vec::new();
        write_str(&mut body, "w");
        body.push(kind::Q8_PER_ROW);
        write_shape(&mut body, &[rows, cols]);
        let mut scales = Vec::new();
        for r in 0..rows {
            let max_abs = values[r * cols..(r + 1) * cols]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            scales.push(max_abs / 127.0);
        }
        for s in &scales {
            body.extend_from_slice(&s.to_le_bytes());
        }
        for r in 0..rows {
            for &v in &values[r * cols..(r + 1) * cols] {
                body.push(((v / scales[r]).round() as i8) as u8);
            }
        }

        let mut blob = MAGIC.to_vec();
        write_u32(&mut blob, 1);
        blob.extend_from_slice(&body);

        let loaded = load(&blob, &Device::Cpu).expect("読み込める");
        let t = &loaded["w"];
        assert_eq!(t.dims(), &[rows, cols]);
        let got = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (g, e) in got.iter().zip(&values) {
            // 各行の最大値で正規化した int8 なので、誤差は行の最大値の 1/254 以内。
            assert!((g - e).abs() < 0.01, "got {g} expected {e}");
        }
    }

    #[test]
    fn wrong_magic_is_rejected() {
        let mut blob = b"XXXX".to_vec();
        write_u32(&mut blob, 0);
        assert!(load(&blob, &Device::Cpu).is_err());
    }

    #[test]
    fn truncated_input_errors_instead_of_panicking() {
        // 埋め込みが途中で切れていたら、添字外アクセスではなくエラーで落ちること。
        let mut blob = MAGIC.to_vec();
        write_u32(&mut blob, 1);
        write_str(&mut blob, "w");
        blob.push(kind::F32);
        write_shape(&mut blob, &[100]);
        // 本体を書かずに終える。
        assert!(load(&blob, &Device::Cpu).is_err());
    }

    #[test]
    fn trailing_garbage_is_rejected() {
        let mut blob = MAGIC.to_vec();
        write_u32(&mut blob, 0);
        blob.push(0xFF);
        assert!(load(&blob, &Device::Cpu).is_err());
    }
}
