//! 二段構えのアライメントが、素朴な総当たりに対してどれだけ効くかを測る。
//!
//! 素朴な実装（＝旧 Python 版と同じ考え方）は、両ファイルの全行を埋め込み、
//! 全行×全行の DP を回す。ここではそれを Rust で同じ土俵に載せて比べる。
//! つまり以下の数字は「Python が遅かった分」ではなく、**手順の違いだけ**の差。
//!
//! 実行:
//!   cargo test -p deepcompare-engine --release --test performance -- --ignored --nocapture

use deepcompare_engine::align::{self, DEFAULT_PAIR_THRESHOLD};
use deepcompare_engine::compare::{compare, CompareOptions, Phase};
use deepcompare_engine::embed::{cosine_similarity, Embedder};
use deepcompare_engine::text::{DecodedText, Encoding, LineEnding};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn asset(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(name)
}

fn load_embedder() -> Option<Embedder> {
    let weights = asset("assets/minilm.dcm");
    let tokenizer = asset("assets/tokenizer.json");
    if !weights.exists() || !tokenizer.exists() {
        eprintln!("資材が無いので飛ばす");
        return None;
    }
    Some(
        Embedder::from_bytes(
            &std::fs::read(weights).expect("重み"),
            &std::fs::read(tokenizer).expect("トークナイザ"),
        )
        .expect("埋め込み器"),
    )
}

/// それらしいソースコードを作る。実際のファイルと同じく、同じ行の繰り返しと
/// 空行が多く含まれる形にする。
fn synthetic_source(lines: usize) -> Vec<String> {
    let templates = [
        "fn handle_{i}(input: &str) -> Result<()> {{",
        "    let value = parse(input)?;",
        "    if value.is_empty() {{",
        "        return Err(Error::Empty);",
        "    }}",
        "    log::debug!(\"handling {{}}\", value);",
        "    Ok(())",
        "}}",
        "",
    ];
    (0..lines)
        .map(|i| templates[i % templates.len()].replace("{i}", &i.to_string()))
        .collect()
}

fn decoded(lines: Vec<String>) -> DecodedText {
    DecodedText {
        lines,
        encoding: Encoding::Utf8,
        line_ending: LineEnding::Lf,
    }
}

/// 素朴な実装。全行を埋め込み、全体を 1 つの DP にかける。
fn naive_compare(left: &DecodedText, right: &DecodedText, embedder: &Embedder) -> (usize, usize) {
    let all: Vec<&str> = left
        .lines
        .iter()
        .chain(&right.lines)
        .map(String::as_str)
        .collect();
    let vectors = embedder.embed_lines(&all).expect("埋め込み");
    let split = left.lines.len();
    let pairs = align::needleman_wunsch(
        left.lines.len(),
        right.lines.len(),
        DEFAULT_PAIR_THRESHOLD,
        |i, j| cosine_similarity(&vectors[i], &vectors[split + j]),
    );
    (all.len(), pairs.len())
}

#[test]
#[ignore = "モデルを読むので明示的に実行する"]
fn two_stage_alignment_beats_the_naive_approach() {
    let Some(embedder) = load_embedder() else {
        return;
    };

    // 実際の改修に近い形。数千行のうち数 % だけが変わる。
    const TOTAL: usize = 3000;
    const CHANGED_EVERY: usize = 50;

    let left_lines = synthetic_source(TOTAL);
    let right_lines: Vec<String> = left_lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            if i % CHANGED_EVERY == 0 && !line.is_empty() {
                format!("{line} // 変更")
            } else {
                line.clone()
            }
        })
        .collect();
    let changed = TOTAL / CHANGED_EVERY;

    let left = decoded(left_lines);
    let right = decoded(right_lines);

    let started = Instant::now();
    let result = compare(
        &left,
        &right,
        &embedder,
        CompareOptions::default(),
        &|_: Phase| {},
    )
    .expect("比較できる");
    let staged = started.elapsed();

    println!("--- 二段構え ---");
    println!("  所要 {:.3} 秒", staged.as_secs_f64());
    println!(
        "  埋め込んだ行 {} / 全 {} 行（一致で畳めた行 {}）",
        result.stats.embedded_lines,
        TOTAL * 2,
        result.stats.identical_lines
    );

    let started = Instant::now();
    let (naive_embedded, _) = naive_compare(&left, &right, &embedder);
    let naive = started.elapsed();

    println!("--- 素朴（全行を埋め込み、全面 DP）---");
    println!("  所要 {:.3} 秒", naive.as_secs_f64());
    println!("  埋め込んだ行 {naive_embedded}");
    println!(
        "--- 差 --- {:.1} 倍速い / 埋め込みは {:.1}% で済んだ",
        naive.as_secs_f64() / staged.as_secs_f64().max(1e-9),
        result.stats.embedded_lines as f64 / naive_embedded as f64 * 100.0
    );

    // 変更のある塊の行しか埋め込まないので、変更行数の数倍で収まるはず。
    assert!(
        result.stats.embedded_lines <= changed * 4,
        "変更していない行まで埋め込んでいる: {} 行",
        result.stats.embedded_lines
    );
    assert!(
        staged < naive,
        "二段構えが素朴な実装より速くない（{staged:?} vs {naive:?}）"
    );
    // 行が消えたり増えたりしていないこと。
    let lefts: Vec<usize> = result.rows.iter().filter_map(|r| r.left).collect();
    let rights: Vec<usize> = result.rows.iter().filter_map(|r| r.right).collect();
    assert_eq!(lefts, (0..TOTAL).collect::<Vec<_>>());
    assert_eq!(rights, (0..TOTAL).collect::<Vec<_>>());
}

#[test]
#[ignore = "モデルを読むので明示的に実行する"]
fn identical_files_are_effectively_free() {
    let Some(embedder) = load_embedder() else {
        return;
    };
    let text = decoded(synthetic_source(5000));

    let started = Instant::now();
    let result = compare(
        &text,
        &text,
        &embedder,
        CompareOptions::default(),
        &|_: Phase| {},
    )
    .expect("比較できる");
    let elapsed = started.elapsed();

    println!("完全一致 5000 行: {:.3} 秒", elapsed.as_secs_f64());
    // 一致するファイル同士でモデルを動かす必要はまったく無い。
    assert_eq!(result.stats.embedded_lines, 0);
    assert_eq!(result.stats.identical_lines, 5000);
}
