//! 自前の BERT 実装を、独立実装（ONNX Runtime）の出力と突き合わせる。
//!
//! `candle-transformers` を使わず前向き計算を手で書いた以上、ここが合っていることを
//! 示せないと、以降の類似度も対応付けも意味を持たない。GELU の種類、LayerNorm を
//! 残差の前に置くか後に置くか、注意マスクの向き、プーリングの分母——どれを取り違えても
//! 「それらしい数字」は出てしまい、テストが無ければ気付けない。
//!
//! 参照値は `tools/reference_embeddings.py` が作る。資材が無い環境では黙って飛ばす。

use deepcompare_engine::embed::{cosine_similarity, Embedder};
use std::path::{Path, PathBuf};

fn asset(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(name)
}

struct Reference {
    texts: Vec<String>,
    vectors: Vec<Vec<f32>>,
}

/// 参照値と資材が揃っていれば読み込む。無ければ `None`。
fn load_reference(weights: &str) -> Option<(Embedder, Reference)> {
    let weights_path = asset(weights);
    let tokenizer_path = asset("assets/tokenizer.json");
    let reference_path = asset("tests/reference_embeddings.json");
    if !weights_path.exists() || !tokenizer_path.exists() || !reference_path.exists() {
        eprintln!(
            "資材が無いので飛ばす（{} / tools/reference_embeddings.py を先に実行）",
            weights_path.display()
        );
        return None;
    }

    let raw = std::fs::read_to_string(&reference_path).expect("参照値を読める");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("参照値が JSON");
    let texts: Vec<String> = json["texts"]
        .as_array()
        .expect("texts")
        .iter()
        .map(|v| v.as_str().expect("文字列").to_string())
        .collect();
    let vectors: Vec<Vec<f32>> = json["vectors"]
        .as_array()
        .expect("vectors")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("行")
                .iter()
                .map(|v| v.as_f64().expect("数値") as f32)
                .collect()
        })
        .collect();

    let embedder = Embedder::from_bytes(
        &std::fs::read(&weights_path).expect("重みを読める"),
        &std::fs::read(&tokenizer_path).expect("トークナイザを読める"),
    )
    .expect("埋め込み器を作れる");

    Some((embedder, Reference { texts, vectors }))
}

/// 参照値との一致度を返す（最小値, 平均値）。
fn agreement(weights: &str) -> Option<(f32, f32)> {
    let (embedder, reference) = load_reference(weights)?;
    let inputs: Vec<&str> = reference.texts.iter().map(String::as_str).collect();
    let ours = embedder.embed_lines(&inputs).expect("埋め込める");
    assert_eq!(ours.len(), reference.vectors.len());

    let mut min = f32::INFINITY;
    let mut sum = 0.0f32;
    for (i, (got, want)) in ours.iter().zip(&reference.vectors).enumerate() {
        assert_eq!(got.len(), want.len(), "次元が違う");
        let similarity = cosine_similarity(got, want);
        if similarity < min {
            min = similarity;
        }
        sum += similarity;
        println!("{similarity:.6}  {:?}", truncate(&reference.texts[i]));
    }
    Some((min, sum / ours.len() as f32))
}

fn truncate(s: &str) -> String {
    if s.chars().count() > 40 {
        format!("{}…", s.chars().take(40).collect::<String>())
    } else {
        s.to_string()
    }
}

/// まず量子化を挟まない f32 で、前向き計算そのものが正しいことを示す。
/// ここがずれていれば実装の誤りであって、量子化のせいではない。
#[test]
fn f32_weights_match_the_independent_implementation() {
    let Some((min, mean)) = agreement("assets/minilm-f32.dcm") else {
        return;
    };
    println!("f32: 最小 {min:.6} / 平均 {mean:.6}");
    assert!(
        min > 0.9999,
        "独立実装と一致しない（最小 {min:.6}）。前向き計算のどこかが違う"
    );
}

/// その上で、int8 量子化がどれだけ埋め込みを動かすかを測る。
#[test]
fn int8_weights_stay_close_to_the_reference() {
    let Some((min, mean)) = agreement("assets/minilm.dcm") else {
        return;
    };
    println!("int8: 最小 {min:.6} / 平均 {mean:.6}");
    // 行の対応付けに使う類似度は 0.5 前後を境に判断するので、0.99 を保てば
    // 判断が変わることはまず無い。
    assert!(
        min > 0.99,
        "int8 化で埋め込みが動きすぎている（最小 {min:.6}）"
    );
}
