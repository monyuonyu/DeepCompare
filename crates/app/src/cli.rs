//! 画面を開かずに使える経路。
//!
//! GUI しか出口が無いと、動作確認が「人が画面を見る」ことでしか行えない。実際に
//! Windows での検証がそこで詰まった。ここでは比較結果をテキストとして出せるように
//! して、遠隔からでも Linux 版と機械的に突き合わせられるようにする。
//!
//! フォントの診断も同じ理由で入れてある。「日本語が豆腐になっていないか」は本来
//! 目で見るしかないが、書体ファイルを直接読んでグリフの有無を調べれば、描画せずに
//! 判定できる。egui と同じ fontations（skrifa）で読むので、egui が読めるかどうかを
//! そのまま反映する。

use anyhow::{Context, Result};
use deepcompare_engine::compare::{compare, CompareOptions, Phase};
use deepcompare_engine::inline::SpanKind;
use deepcompare_engine::{decode_file, Embedder};
use std::io::Write;
use std::path::{Path, PathBuf};

/// 日本語が出せているかを確かめる文字。ひらがな・カタカナ・漢字・全角記号を
/// それぞれ 1 文字ずつ。どれか欠けると差分表示のどこかが豆腐になる。
const SAMPLE_CHARS: &[(char, &str)] = &[
    ('設', "漢字"),
    ('あ', "ひらがな"),
    ('ア', "カタカナ"),
    ('、', "全角句読点"),
    ('　', "全角空白"),
    ('①', "丸数字"),
];

pub struct CliOptions {
    pub left: PathBuf,
    pub right: PathBuf,
    pub output: Option<PathBuf>,
    pub pair_threshold: f32,
}

/// 比較を実行し、行ごとの結果をテキストで書き出す。
pub fn run_compare(options: CliOptions, weights: &[u8], tokenizer: &[u8]) -> Result<()> {
    let left = read(&options.left)?;
    let right = read(&options.right)?;
    let embedder = Embedder::from_bytes(weights, tokenizer).context("モデルを読み込めない")?;

    let started = std::time::Instant::now();
    let result = compare(
        &left,
        &right,
        &embedder,
        CompareOptions {
            pair_threshold: options.pair_threshold,
            ..CompareOptions::default()
        },
        &|_: Phase| {},
    )?;
    let elapsed = started.elapsed();

    let mut out = String::new();
    out.push_str(&format!(
        "left  {} encoding={} line_ending={} lines={}\n",
        options.left.display(),
        left.encoding,
        left.line_ending,
        left.lines.len()
    ));
    out.push_str(&format!(
        "right {} encoding={} line_ending={} lines={}\n",
        options.right.display(),
        right.encoding,
        right.line_ending,
        right.lines.len()
    ));
    out.push_str(&format!(
        "stats rows={} identical={} embedded={} skipped_blocks={} elapsed_ms={}\n",
        result.stats.rows,
        result.stats.identical_lines,
        result.stats.embedded_lines,
        result.stats.skipped_blocks,
        elapsed.as_millis()
    ));
    out.push_str(&format!("threshold {:.2}\n", options.pair_threshold));
    out.push_str("---\n");

    // 1 行 1 レコード。環境をまたいで diff で比べられるよう、桁を固定する。
    // 種別は = 一致 / ~ 変更あり / - 左のみ / + 右のみ。
    for row in &result.rows {
        let kind = match (row.left, row.right) {
            (Some(_), Some(_)) if row.is_unchanged() => '=',
            (Some(_), Some(_)) => '~',
            (Some(_), None) => '-',
            (None, Some(_)) => '+',
            (None, None) => '?',
        };
        let num = |v: Option<usize>| v.map(|i| (i + 1).to_string()).unwrap_or_default();
        let score = row
            .score
            .map(|s| format!("{s:.4}"))
            .unwrap_or_else(|| "-".to_owned());
        // 行内で変わった部分の数。ここが環境で変われば差分の取り方が違っている。
        let inline_changes = row
            .left_spans
            .iter()
            .chain(&row.right_spans)
            .filter(|s| s.kind == SpanKind::Changed)
            .count();
        out.push_str(&format!(
            "{kind} {:>6} {:>6} {score:>6} {inline_changes:>2}  {}\n",
            num(row.left),
            num(row.right),
            row.left
                .map(|i| left.lines[i].as_str())
                .or_else(|| row.right.map(|j| right.lines[j].as_str()))
                .unwrap_or("")
        ));
    }

    emit(&out, options.output.as_deref())
}

/// 書体の診断。どの書体が見つかり、日本語のグリフを持っているかを報告する。
pub fn run_font_check(output: Option<&Path>) -> Result<()> {
    let mut out = String::new();
    out.push_str(&format!("platform {}\n", std::env::consts::OS));
    out.push_str("---\n");

    let mut found_any = false;
    for path in crate::fonts::candidates() {
        let exists = Path::new(path).exists();
        if !exists {
            out.push_str(&format!("MISS    {path}\n"));
            continue;
        }
        let data = match std::fs::read(path) {
            Ok(data) => data,
            Err(error) => {
                out.push_str(&format!("UNREAD  {path}  ({error})\n"));
                continue;
            }
        };
        out.push_str(&format!("FOUND   {path}  {} バイト\n", data.len()));

        // egui が実際に読むのと同じ経路で開く。ここで失敗するなら、ファイルが
        // 存在していても画面には出ない。
        match skrifa::FontRef::from_index(&data, crate::fonts::FONT_INDEX) {
            Err(error) => {
                out.push_str(&format!(
                    "        読めない（index {}）: {error}\n",
                    crate::fonts::FONT_INDEX
                ));
            }
            Ok(font) => {
                use skrifa::MetadataProvider;
                let charmap = font.charmap();
                let mut missing = Vec::new();
                for (c, label) in SAMPLE_CHARS {
                    if charmap.map(*c).is_none() {
                        missing.push(format!("{c}({label})"));
                    }
                }
                if missing.is_empty() {
                    out.push_str("        日本語グリフ: 全て有り → 豆腐にならない\n");
                    found_any = true;
                } else {
                    out.push_str(&format!(
                        "        日本語グリフ: 欠落 {} → 豆腐になる\n",
                        missing.join(" ")
                    ));
                }
            }
        }
    }

    out.push_str("---\n");
    out.push_str(if found_any {
        "結果: 日本語を表示できる書体が見つかった\n"
    } else {
        "結果: 表示できる書体が無い。日本語は豆腐になる\n"
    });

    emit(&out, output)
}

fn read(path: &Path) -> Result<deepcompare_engine::DecodedText> {
    let bytes =
        std::fs::read(path).with_context(|| format!("ファイルを開けない: {}", path.display()))?;
    Ok(decode_file(&bytes))
}

/// 出力先。
///
/// `-o` を用意しているのは Windows の都合。GUI サブシステムで作った exe には
/// 標準出力が繋がらないので、コンソールから実行しても何も見えない。ファイルへ
/// 書ければ、遠隔からでも確実に結果を回収できる。
fn emit(text: &str, output: Option<&Path>) -> Result<()> {
    match output {
        Some(path) => {
            std::fs::write(path, text)
                .with_context(|| format!("書き込めない: {}", path.display()))?;
        }
        None => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(text.as_bytes())?;
            stdout.flush()?;
        }
    }
    Ok(())
}
