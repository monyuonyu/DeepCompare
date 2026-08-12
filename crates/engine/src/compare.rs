//! 比較全体の組み立て。GUI が呼ぶのはここ一つ。
//!
//! 手順は、まず文字列一致で動かない区間を畳み、残った塊の行だけをまとめて埋め込み、
//! 塊ごとに Needleman-Wunsch をかけ、最後に対応した行の行内差分を取る、の順。
//!
//! 埋め込みを「塊ごと」ではなく**全塊分まとめて一度**呼ぶのは意図的で、そうしないと
//! 重複除去と長さ揃えが塊の内側でしか効かなくなる。

use crate::align::{self, Block, Pair, Segment, EXACT_SCORE};
use crate::embed::{cosine_similarity, Embedder};
use crate::inline::{inline_diff, Span, SpanKind};
use crate::text::DecodedText;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct CompareOptions {
    pub pair_threshold: f32,
    /// 意味的アライメントに回す 1 塊の上限（行数の積）。
    pub max_block_area: usize,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self {
            pair_threshold: align::DEFAULT_PAIR_THRESHOLD,
            max_block_area: align::MAX_BLOCK_AREA,
        }
    }
}

/// 表示 1 行分。
#[derive(Debug, Clone)]
pub struct Row {
    pub left: Option<usize>,
    pub right: Option<usize>,
    /// 両側が埋まっている場合のみ入る類似度。
    pub score: Option<f32>,
    pub left_spans: Vec<Span>,
    pub right_spans: Vec<Span>,
}

impl Row {
    /// 左右が揃っていて内容も完全に同じか。表示の色分けはこれで決める。
    pub fn is_unchanged(&self) -> bool {
        self.left.is_some()
            && self.right.is_some()
            && self.left_spans.iter().all(|s| s.kind == SpanKind::Equal)
            && self.right_spans.iter().all(|s| s.kind == SpanKind::Equal)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub rows: usize,
    /// 文字列一致で畳めた行数。モデルを通していない。
    pub identical_lines: usize,
    /// 実際に埋め込みを計算した行数（重複除去前）。
    pub embedded_lines: usize,
    /// 大きすぎて意味的アライメントを諦めた塊の数。
    pub skipped_blocks: usize,
}

#[derive(Debug, Clone)]
pub struct Comparison {
    pub rows: Vec<Row>,
    pub stats: Stats,
}

/// 進捗の通知。埋め込みが支配的なので、その前後だけ知らせれば十分。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Segmenting,
    Embedding,
    Aligning,
    Done,
}

pub fn compare(
    left: &DecodedText,
    right: &DecodedText,
    embedder: &Embedder,
    options: CompareOptions,
    // 塊ごとの並列処理の内側からは呼ばない（呼ぶと通知の順序が保証できない）ので、
    // `Sync` は求めない。求めると呼び出し側が送信口を Mutex で包む羽目になる。
    progress: &dyn Fn(Phase),
) -> Result<Comparison> {
    progress(Phase::Segmenting);
    let segments = align::segment(&left.lines, &right.lines);

    // 意味的アライメントにかける塊だけを選ぶ。
    let mut scored_blocks: Vec<&Block> = Vec::new();
    let mut skipped_blocks = 0usize;
    for segment in &segments {
        if let Segment::Changed(block) = segment {
            if block.area() > options.max_block_area {
                skipped_blocks += 1;
            } else {
                scored_blocks.push(block);
            }
        }
    }

    // 埋め込みが要る行を集めて、一度に通す。
    progress(Phase::Embedding);
    let mut wanted: Vec<&str> = Vec::new();
    let mut left_slot: HashMap<usize, usize> = HashMap::new();
    let mut right_slot: HashMap<usize, usize> = HashMap::new();
    for block in &scored_blocks {
        for i in block.left.clone() {
            left_slot.entry(i).or_insert_with(|| {
                wanted.push(left.lines[i].as_str());
                wanted.len() - 1
            });
        }
        for j in block.right.clone() {
            right_slot.entry(j).or_insert_with(|| {
                wanted.push(right.lines[j].as_str());
                wanted.len() - 1
            });
        }
    }
    let embedded_lines = wanted.len();
    let vectors = embedder.embed_lines(&wanted)?;

    progress(Phase::Aligning);
    // 塊どうしは独立なので並列に処理し、あとで元の順序へ戻す。
    let aligned: Vec<(usize, Vec<Pair>)> = segments
        .par_iter()
        .enumerate()
        .filter_map(|(index, segment)| {
            let Segment::Changed(block) = segment else {
                return None;
            };
            let pairs = if block.area() > options.max_block_area {
                align::align_without_scoring(block)
            } else {
                let left_base = block.left.start;
                let right_base = block.right.start;
                let local = align::needleman_wunsch(
                    block.left.len(),
                    block.right.len(),
                    options.pair_threshold,
                    |i, j| {
                        let a = &vectors[left_slot[&(left_base + i)]];
                        let b = &vectors[right_slot[&(right_base + j)]];
                        cosine_similarity(a, b)
                    },
                );
                // 塊の中の相対位置を絶対行番号へ戻す。
                local
                    .into_iter()
                    .map(|p| Pair {
                        left: p.left.map(|i| left_base + i),
                        right: p.right.map(|j| right_base + j),
                        score: p.score,
                    })
                    .collect()
            };
            Some((index, pairs))
        })
        .collect();

    let mut aligned: HashMap<usize, Vec<Pair>> = aligned.into_iter().collect();

    // 順序どおりに並べ直して行を作る。
    let mut rows = Vec::new();
    let mut identical_lines = 0usize;
    for (index, segment) in segments.iter().enumerate() {
        match segment {
            Segment::Identical {
                left_start,
                right_start,
                len,
            } => {
                identical_lines += len;
                for k in 0..*len {
                    let l = left_start + k;
                    let r = right_start + k;
                    let span = whole_equal(&left.lines[l]);
                    rows.push(Row {
                        left: Some(l),
                        right: Some(r),
                        score: Some(EXACT_SCORE),
                        left_spans: span.clone(),
                        right_spans: span,
                    });
                }
            }
            Segment::Changed(_) => {
                for pair in aligned.remove(&index).unwrap_or_default() {
                    rows.push(build_row(pair, left, right));
                }
            }
        }
    }

    progress(Phase::Done);
    Ok(Comparison {
        stats: Stats {
            rows: rows.len(),
            identical_lines,
            embedded_lines,
            skipped_blocks,
        },
        rows,
    })
}

fn whole_equal(line: &str) -> Vec<Span> {
    if line.is_empty() {
        Vec::new()
    } else {
        vec![Span {
            kind: SpanKind::Equal,
            range: 0..line.len(),
        }]
    }
}

fn build_row(pair: Pair, left: &DecodedText, right: &DecodedText) -> Row {
    let (left_spans, right_spans) = match (pair.left, pair.right) {
        (Some(l), Some(r)) => inline_diff(&left.lines[l], &right.lines[r]),
        (Some(l), None) => (
            vec![Span {
                kind: SpanKind::Changed,
                range: 0..left.lines[l].len(),
            }],
            Vec::new(),
        ),
        (None, Some(r)) => (
            Vec::new(),
            vec![Span {
                kind: SpanKind::Changed,
                range: 0..right.lines[r].len(),
            }],
        ),
        (None, None) => (Vec::new(), Vec::new()),
    };
    Row {
        left: pair.left,
        right: pair.right,
        score: pair.score,
        left_spans,
        right_spans,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{Encoding, LineEnding};

    fn decoded(text: &str) -> DecodedText {
        DecodedText {
            lines: text.lines().map(String::from).collect(),
            encoding: Encoding::Utf8,
            line_ending: LineEnding::Lf,
        }
    }

    /// 埋め込みを伴わない部分だけを検査する。モデルの読み込みは重いので、
    /// ここでは一致区間の畳み込みと行の並びに絞る。
    #[test]
    fn identical_files_need_no_embedding_at_all() {
        let a = decoded("fn main() {\n    println!(\"hi\");\n}\n");
        let segments = align::segment(&a.lines, &a.lines);
        assert!(
            segments
                .iter()
                .all(|s| matches!(s, Segment::Identical { .. })),
            "完全一致なら塊は一つも出ないはず"
        );
    }

    #[test]
    fn unchanged_row_is_reported_as_unchanged() {
        let row = Row {
            left: Some(0),
            right: Some(0),
            score: Some(EXACT_SCORE),
            left_spans: whole_equal("same"),
            right_spans: whole_equal("same"),
        };
        assert!(row.is_unchanged());
    }

    #[test]
    fn row_with_inline_change_is_not_unchanged() {
        let left = decoded("let x = 1;");
        let right = decoded("let x = 2;");
        let row = build_row(Pair::both(0, 0, 0.9), &left, &right);
        assert!(!row.is_unchanged());
    }

    #[test]
    fn gap_row_carries_spans_on_one_side_only() {
        let left = decoded("removed");
        let right = decoded("");
        let row = build_row(Pair::left_only(0), &left, &right);
        assert!(!row.left_spans.is_empty());
        assert!(row.right_spans.is_empty());
        assert_eq!(row.score, None);
    }
}
