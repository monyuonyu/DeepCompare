//! MiniLM の safetensors を int8 量子化して、exe に埋め込む形式へ変換する開発用ツール。
//! 配布物には含めない（publish = false）。
//!
//! 使い方:
//!   model-prep assets/src/model.safetensors assets/minilm.dcm [--no-quantize]
//!
//! `--no-quantize` は検証用。同じ経路を f32 のまま通した物を作れるので、
//! 「自前の BERT が間違っている」のか「int8 で精度が落ちた」のかを切り分けられる。

use anyhow::{Context, Result};
use deepcompare_engine::weights::{self as format, kind, QUANTIZE_MIN_ELEMS};
use safetensors::{Dtype, SafeTensors};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut quantize = true;
    for arg in args {
        if arg == "--no-quantize" {
            quantize = false;
        } else {
            positional.push(arg.into());
        }
    }
    let mut positional = positional.into_iter();
    let input: PathBuf = positional.next().context("入力 safetensors のパスが必要")?;
    let output: PathBuf = positional.next().context("出力先のパスが必要")?;

    let raw =
        std::fs::read(&input).with_context(|| format!("読み込めない: {}", input.display()))?;
    let tensors = SafeTensors::deserialize(&raw).context("safetensors として解釈できない")?;

    let mut names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    names.sort();

    let mut body = Vec::new();
    let mut written = 0u32;
    let mut report = Report::default();

    for name in &names {
        // 推論に使わない物は落とす。ここで捨てた分がそのまま exe の削減になる。
        if should_skip(name) {
            report.skipped.push(name.to_string());
            continue;
        }
        let view = tensors.tensor(name)?;
        let shape = view.shape().to_vec();
        let values = to_f32(view.data(), view.dtype())
            .with_context(|| format!("未対応の dtype: {name} ({:?})", view.dtype()))?;

        format::write_str(&mut body, name);
        if quantize && shape.len() == 2 && values.len() >= QUANTIZE_MIN_ELEMS {
            body.push(kind::Q8_PER_ROW);
            format::write_shape(&mut body, &shape);
            let stats = write_q8_per_row(&mut body, &values, shape[0], shape[1]);
            report.record_quantized(name, &values, stats);
        } else {
            body.push(kind::F32);
            format::write_shape(&mut body, &shape);
            for v in &values {
                body.extend_from_slice(&v.to_le_bytes());
            }
            report.kept_f32_bytes += values.len() * 4;
        }
        written += 1;
    }

    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(format::MAGIC);
    format::write_u32(&mut out, written);
    out.extend_from_slice(&body);

    std::fs::write(&output, &out).with_context(|| format!("書き込めない: {}", output.display()))?;
    report.print(raw.len(), out.len(), written, &output);
    Ok(())
}

/// 文埋め込みでは使わない重み。
///
/// - `pooler`: sentence-transformers の MiniLM は平均プーリングを使うので出番が無い。
/// - `cls` / `*.predictions.*`: 事前学習のマスク言語モデル用ヘッド。語彙サイズの
///   行列を抱えているので、残すと exe が無駄に膨らむ。
fn should_skip(name: &str) -> bool {
    name.starts_with("pooler.")
        || name.contains(".pooler.")
        || name.starts_with("cls.")
        || name.contains(".predictions.")
        || name.ends_with(".position_ids")
}

fn to_f32(bytes: &[u8], dtype: Dtype) -> Option<Vec<f32>> {
    match dtype {
        Dtype::F32 => Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Dtype::F16 => Some(
            bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        ),
        Dtype::BF16 => Some(
            bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        ),
        _ => None,
    }
}

#[derive(Default, Clone, Copy)]
struct QuantStats {
    max_abs_err: f32,
    sum_sq_err: f64,
    sum_sq_ref: f64,
}

/// 行ごとに `scale = max|w| / 127` を取り、`round(w / scale)` を i8 で書く。
///
/// 行ごとにするのは、BERT の線形層は行（出力チャネル）ごとに大きさの桁が違うため。
/// 行列全体で 1 つのスケールにすると、小さい行がまるごと 0 に潰れる。
fn write_q8_per_row(out: &mut Vec<u8>, values: &[f32], rows: usize, cols: usize) -> QuantStats {
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let max_abs = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // 全要素 0 の行でも 0 除算にしない。
        scales.push(if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 });
    }
    for s in &scales {
        out.extend_from_slice(&s.to_le_bytes());
    }

    let mut stats = QuantStats::default();
    for r in 0..rows {
        let scale = scales[r];
        for &v in &values[r * cols..(r + 1) * cols] {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
            let err = (q as f32 * scale) - v;
            stats.max_abs_err = stats.max_abs_err.max(err.abs());
            stats.sum_sq_err += (err as f64) * (err as f64);
            stats.sum_sq_ref += (v as f64) * (v as f64);
        }
    }
    stats
}

#[derive(Default)]
struct Report {
    skipped: Vec<String>,
    quantized: usize,
    quantized_bytes: usize,
    kept_f32_bytes: usize,
    worst: Option<(String, f32)>,
    total_sq_err: f64,
    total_sq_ref: f64,
}

impl Report {
    fn record_quantized(&mut self, name: &str, values: &[f32], stats: QuantStats) {
        self.quantized += 1;
        self.quantized_bytes += values.len();
        self.total_sq_err += stats.sum_sq_err;
        self.total_sq_ref += stats.sum_sq_ref;
        let worse = self
            .worst
            .as_ref()
            .is_none_or(|(_, e)| stats.max_abs_err > *e);
        if worse {
            self.worst = Some((name.to_string(), stats.max_abs_err));
        }
    }

    fn print(
        &self,
        input_bytes: usize,
        output_bytes: usize,
        written: u32,
        output: &std::path::Path,
    ) {
        let mib = |b: usize| b as f64 / 1024.0 / 1024.0;
        println!("出力: {}", output.display());
        println!(
            "  {:.1} MiB -> {:.1} MiB ({:.1}%)",
            mib(input_bytes),
            mib(output_bytes),
            output_bytes as f64 / input_bytes as f64 * 100.0
        );
        println!(
            "  テンソル {written} 本 (int8 {} 本 / f32 のまま {:.2} MiB)",
            self.quantized,
            mib(self.kept_f32_bytes)
        );
        if !self.skipped.is_empty() {
            println!(
                "  除外 {} 本: {}",
                self.skipped.len(),
                self.skipped.join(", ")
            );
        }
        // 相対誤差。埋め込みのコサイン類似度への影響は別途、実文で測る。
        let rel = (self.total_sq_err / self.total_sq_ref.max(f64::MIN_POSITIVE)).sqrt();
        println!("  量子化の相対二乗誤差: {:.4}%", rel * 100.0);
        if let Some((name, err)) = &self.worst {
            println!("  最大絶対誤差: {err:.6} ({name})");
        }
    }
}
