//! 行内の差分。どこが変わったかを構造として返す。
//!
//! 旧 Python 実装はここで HTML 文字列を組み立て、それを `QLabel` にリッチテキスト
//! として渡していた。差分部分を `<span style=...>` で囲むだけで本文のエスケープを
//! しておらず、`<`、`&`、`>` を含む行——つまり C++ のテンプレート、HTML、XML、
//! ジェネリクスを含むあらゆるコード——は表示が壊れるか、内容が丸ごと消えていた。
//!
//! 文字列ではなく範囲の列を返せば、書式付けの責任は描画側に移り、この種の取り違えは
//! 起こりようがなくなる。

use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    Equal,
    Changed,
}

/// 行の一部を指す。`range` は元の行へのバイト範囲。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub kind: SpanKind,
    pub range: Range<usize>,
}

/// 行内差分をあきらめる長さ。
///
/// 文字単位の差分は O(n·m) なので、minify されたファイルの 1 行のような極端な入力で
/// 固まる。その場合は行全体を「変更あり」として扱う。行の対応付け自体は済んでいるので、
/// 失われるのは行内のどこが変わったかという情報だけ。
pub const MAX_INLINE_DIFF_BYTES: usize = 8192;

/// 左右の行を突き合わせ、それぞれの範囲の列を返す。
pub fn inline_diff(left: &str, right: &str) -> (Vec<Span>, Vec<Span>) {
    if left == right {
        return (whole(left, SpanKind::Equal), whole(right, SpanKind::Equal));
    }
    if left.len() > MAX_INLINE_DIFF_BYTES || right.len() > MAX_INLINE_DIFF_BYTES {
        return (
            whole(left, SpanKind::Changed),
            whole(right, SpanKind::Changed),
        );
    }

    use similar::{ChangeTag, TextDiff};

    let diff = TextDiff::from_chars(left, right);
    let mut left_spans = SpanBuilder::default();
    let mut right_spans = SpanBuilder::default();

    for change in diff.iter_all_changes() {
        let len = change.value().len();
        match change.tag() {
            ChangeTag::Equal => {
                left_spans.push(SpanKind::Equal, len);
                right_spans.push(SpanKind::Equal, len);
            }
            ChangeTag::Delete => left_spans.push(SpanKind::Changed, len),
            ChangeTag::Insert => right_spans.push(SpanKind::Changed, len),
        }
    }
    (left_spans.finish(), right_spans.finish())
}

fn whole(s: &str, kind: SpanKind) -> Vec<Span> {
    if s.is_empty() {
        return Vec::new();
    }
    vec![Span {
        kind,
        range: 0..s.len(),
    }]
}

/// 同じ種別が続く範囲は 1 つにまとめる。文字単位の差分をそのまま出すと
/// 1 文字ずつの範囲が並び、描画側が無駄に重くなる。
#[derive(Default)]
struct SpanBuilder {
    spans: Vec<Span>,
    cursor: usize,
}

impl SpanBuilder {
    fn push(&mut self, kind: SpanKind, len: usize) {
        if len == 0 {
            return;
        }
        let end = self.cursor + len;
        match self.spans.last_mut() {
            Some(last) if last.kind == kind && last.range.end == self.cursor => {
                last.range.end = end;
            }
            _ => self.spans.push(Span {
                kind,
                range: self.cursor..end,
            }),
        }
        self.cursor = end;
    }

    fn finish(self) -> Vec<Span> {
        self.spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts<'a>(s: &'a str, spans: &[Span]) -> Vec<(SpanKind, &'a str)> {
        spans
            .iter()
            .map(|p| (p.kind, &s[p.range.clone()]))
            .collect()
    }

    #[test]
    fn identical_lines_have_no_changed_span() {
        let (l, r) = inline_diff("let x = 1;", "let x = 1;");
        assert!(l.iter().all(|s| s.kind == SpanKind::Equal));
        assert!(r.iter().all(|s| s.kind == SpanKind::Equal));
    }

    #[test]
    fn only_the_differing_part_is_marked() {
        let (l, r) = inline_diff("let x = 1;", "let x = 2;");
        assert_eq!(
            texts("let x = 1;", &l),
            vec![
                (SpanKind::Equal, "let x = "),
                (SpanKind::Changed, "1"),
                (SpanKind::Equal, ";")
            ]
        );
        assert_eq!(
            texts("let x = 2;", &r),
            vec![
                (SpanKind::Equal, "let x = "),
                (SpanKind::Changed, "2"),
                (SpanKind::Equal, ";")
            ]
        );
    }

    #[test]
    fn angle_brackets_and_ampersands_survive_intact() {
        // 旧実装が壊していた入力。差分を HTML として組み立てていたため、
        // `<int>` がタグとして解釈され表示から消えていた。
        let left = "std::vector<int>& xs = a && b;";
        let right = "std::vector<long>& xs = a && b;";
        let (l, r) = inline_diff(left, right);

        // 範囲を連結すれば元の行に戻る。つまり 1 文字も失われていない。
        let rebuilt: String = l.iter().map(|s| &left[s.range.clone()]).collect();
        assert_eq!(rebuilt, left);
        let rebuilt: String = r.iter().map(|s| &right[s.range.clone()]).collect();
        assert_eq!(rebuilt, right);
    }

    #[test]
    fn spans_are_contiguous_and_cover_the_whole_line() {
        let left = "aaa bbb ccc";
        let right = "aaa xxx ccc";
        let (l, r) = inline_diff(left, right);
        for (line, spans) in [(left, &l), (right, &r)] {
            let mut cursor = 0;
            for span in spans {
                assert_eq!(span.range.start, cursor, "範囲に隙間がある");
                cursor = span.range.end;
            }
            assert_eq!(cursor, line.len(), "行末まで覆えていない");
        }
    }

    #[test]
    fn multibyte_ranges_land_on_character_boundaries() {
        // 範囲がバイト境界を割ると、描画側で添字が panic する。
        let left = "// 設定を読む";
        let right = "// 設定を書く";
        let (l, r) = inline_diff(left, right);
        for (line, spans) in [(left, &l), (right, &r)] {
            for span in spans {
                assert!(line.is_char_boundary(span.range.start));
                assert!(line.is_char_boundary(span.range.end));
            }
        }
    }

    #[test]
    fn empty_line_against_content_marks_everything_changed() {
        let (l, r) = inline_diff("", "added");
        assert!(l.is_empty());
        assert_eq!(texts("added", &r), vec![(SpanKind::Changed, "added")]);
    }

    #[test]
    fn very_long_lines_skip_inline_diff_instead_of_hanging() {
        let left = "x".repeat(MAX_INLINE_DIFF_BYTES + 1);
        let right = "y".repeat(MAX_INLINE_DIFF_BYTES + 1);
        let (l, r) = inline_diff(&left, &right);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].kind, SpanKind::Changed);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn adjacent_changes_are_merged_into_one_span() {
        // 文字単位の差分をそのまま出すと 1 文字ずつの範囲が並ぶ。
        let (l, _) = inline_diff("abcd", "aXYd");
        assert_eq!(
            texts("abcd", &l),
            vec![
                (SpanKind::Equal, "a"),
                (SpanKind::Changed, "bc"),
                (SpanKind::Equal, "d")
            ]
        );
    }
}
