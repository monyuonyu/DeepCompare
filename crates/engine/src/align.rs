//! 行の対応付け。
//!
//! 旧 Python 実装は、両ファイルの全行の総当たりで類似度行列を作り、その上で
//! Needleman-Wunsch を回していた。1000 行同士なら 100 万マスの純 Python DP に
//! 100 万回のモデル推論が乗り、実用的なファイルでは待てない。
//!
//! ここでは二段に分ける。まず文字列一致で確実に対応する区間を畳み、残った「変化
//! した塊」に対してだけ意味的アライメントを行う。典型的な差分では変化は全体の
//! 数 % なので、推論も DP もその数 % にしかかからない。

/// 対応付けられた 1 行分。左右どちらかが空くことがある。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pair {
    pub left: Option<usize>,
    pub right: Option<usize>,
    /// 両側が埋まっている場合のみ入る意味的類似度。文字列一致で畳んだ区間は 1.0。
    pub score: Option<f32>,
}

impl Pair {
    pub fn both(left: usize, right: usize, score: f32) -> Self {
        Self {
            left: Some(left),
            right: Some(right),
            score: Some(score),
        }
    }

    pub fn left_only(left: usize) -> Self {
        Self {
            left: Some(left),
            right: None,
            score: None,
        }
    }

    pub fn right_only(right: usize) -> Self {
        Self {
            left: None,
            right: Some(right),
            score: None,
        }
    }

    /// 左右が揃っていて、かつ内容が完全一致していたか。
    pub fn is_exact(&self) -> bool {
        self.score == Some(EXACT_SCORE)
    }
}

/// 文字列一致で畳んだ行に与える類似度。モデルを通していないことを示すため、
/// 推論結果が理論上取り得る値ではなくちょうど 1.0 を使う。
pub const EXACT_SCORE: f32 = 1.0;

/// 対応付ける価値があるとみなす類似度の下限。これを下回る組は、対にせず
/// 左右それぞれを空きとして並べる。
///
/// 旧実装は「空き 1 つにつき -0.5」という罰則を置き、類似度はそのまま加算していた。
/// だが類似度が 0..1 に収まる以上、対にすれば 0 以上、空き 2 つなら -1.0 なので、
/// **どれだけ無関係な行同士でも必ず対にされ、空きが選ばれることが原理的に無かった**。
/// たとえば左が `[A, B]`、右が `[B', C]` で B だけが対応する場合でも、
/// A↔B' と B↔C という誤った対が作られる。
///
/// ここでは対角のスコアを `類似度 - この閾値` とし、空きを 0 とする。こうすると
/// 「類似度がこの値を上回るときだけ対にする」がそのまま式の意味になる。
pub const DEFAULT_PAIR_THRESHOLD: f32 = 0.5;

/// 意味的アライメントに回す 1 塊の上限（左右の行数の積）。
///
/// DP は O(n·m) の時間と領域を使うので、極端に大きな塊は畳まずに切り離す。
/// 400 万マスで f32 の表が約 16 MiB、手元の実測で 1 秒未満に収まる。
pub const MAX_BLOCK_AREA: usize = 4_000_000;

/// 変化した塊。両側の行番号の半開区間で表す。
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub left: std::ops::Range<usize>,
    pub right: std::ops::Range<usize>,
}

impl Block {
    pub fn area(&self) -> usize {
        self.left.len().saturating_mul(self.right.len())
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty() && self.right.is_empty()
    }
}

/// 文字列一致だけで、動かない区間と変化した塊に切り分ける。
///
/// 戻り値は入力の順序どおりに並んだ「確定した対応」と「未確定の塊」の列。
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    /// 内容が完全に一致する区間。左右とも同じ長さ。
    Identical {
        left_start: usize,
        right_start: usize,
        len: usize,
    },
    /// 変化した塊。意味的アライメントにかける対象。
    Changed(Block),
}

/// 左右の行から、一致区間と変化区間の列を作る。
pub fn segment(left: &[String], right: &[String]) -> Vec<Segment> {
    use similar::{capture_diff_slices, Algorithm, DiffOp};

    let ops = capture_diff_slices(Algorithm::Myers, left, right);
    let mut segments: Vec<Segment> = Vec::new();

    // Delete と Insert が隣接している場合は 1 つの塊として扱う。別々に流すと
    // 「消えた行」と「増えた行」の対応を意味的に取る機会そのものが無くなる。
    let mut pending: Option<Block> = None;
    let flush = |pending: &mut Option<Block>, segments: &mut Vec<Segment>| {
        if let Some(block) = pending.take() {
            if !block.is_empty() {
                segments.push(Segment::Changed(block));
            }
        }
    };

    for op in ops {
        match op {
            DiffOp::Equal {
                old_index,
                new_index,
                len,
            } => {
                flush(&mut pending, &mut segments);
                segments.push(Segment::Identical {
                    left_start: old_index,
                    right_start: new_index,
                    len,
                });
            }
            DiffOp::Delete {
                old_index,
                old_len,
                new_index,
            } => {
                extend(
                    &mut pending,
                    old_index..old_index + old_len,
                    new_index..new_index,
                );
            }
            DiffOp::Insert {
                old_index,
                new_index,
                new_len,
            } => {
                extend(
                    &mut pending,
                    old_index..old_index,
                    new_index..new_index + new_len,
                );
            }
            DiffOp::Replace {
                old_index,
                old_len,
                new_index,
                new_len,
            } => {
                extend(
                    &mut pending,
                    old_index..old_index + old_len,
                    new_index..new_index + new_len,
                );
            }
        }
    }
    flush(&mut pending, &mut segments);
    segments
}

fn extend(
    pending: &mut Option<Block>,
    left: std::ops::Range<usize>,
    right: std::ops::Range<usize>,
) {
    match pending {
        Some(block) => {
            if !left.is_empty() {
                if block.left.is_empty() {
                    block.left = left;
                } else {
                    block.left.end = left.end;
                }
            }
            if !right.is_empty() {
                if block.right.is_empty() {
                    block.right = right;
                } else {
                    block.right.end = right.end;
                }
            }
        }
        None => *pending = Some(Block { left, right }),
    }
}

/// Needleman-Wunsch。`sim(i, j)` は塊の中での相対位置に対する類似度を返す。
///
/// 対角のスコアは `sim - pair_threshold`、空きは 0。同点なら対角を選ぶ。
/// 行番号は塊の先頭からの相対値で、呼び出し側が絶対行番号へ戻す。
/// `Pair::score` に入るのは閾値を引く前の生の類似度。
pub fn needleman_wunsch<F>(rows: usize, cols: usize, pair_threshold: f32, sim: F) -> Vec<Pair>
where
    F: Fn(usize, usize) -> f32,
{
    if rows == 0 {
        return (0..cols).map(Pair::right_only).collect();
    }
    if cols == 0 {
        return (0..rows).map(Pair::left_only).collect();
    }

    let width = cols + 1;
    let mut dp = vec![0.0f32; (rows + 1) * width];
    // 経路。0 = 斜め, 1 = 上（左のみ）, 2 = 左（右のみ）。
    let mut from = vec![0u8; (rows + 1) * width];

    // 端の列と行は空きが並ぶだけ。空きのスコアは 0 なので dp は 0 のまま。
    for i in 1..=rows {
        from[i * width] = 1;
    }
    for slot in from[1..=cols].iter_mut() {
        *slot = 2;
    }

    for i in 1..=rows {
        for j in 1..=cols {
            let diag = dp[(i - 1) * width + (j - 1)] + (sim(i - 1, j - 1) - pair_threshold);
            let up = dp[(i - 1) * width + j];
            let left = dp[i * width + (j - 1)];

            // 同点なら斜めを優先する。左右を空きにするより対応付けた方が読みやすい。
            //
            // 空き同士が同点の場合は「右だけ」を選ぶ。経路は逆向きに辿ってから
            // 反転するので、ここで右を選ぶと最終的な並びでは左（削除）が先に来る。
            // 差分ツールの慣習は削除→追加の順で、逆にすると読み違えやすい。
            let (best, dir) = if diag >= up && diag >= left {
                (diag, 0u8)
            } else if left >= up {
                (left, 2u8)
            } else {
                (up, 1u8)
            };
            dp[i * width + j] = best;
            from[i * width + j] = dir;
        }
    }

    let mut pairs = Vec::with_capacity(rows.max(cols));
    let (mut i, mut j) = (rows, cols);
    while i > 0 || j > 0 {
        // 端に張り付いたら残りは一方向にしか進めない。
        if i == 0 {
            j -= 1;
            pairs.push(Pair::right_only(j));
            continue;
        }
        if j == 0 {
            i -= 1;
            pairs.push(Pair::left_only(i));
            continue;
        }
        match from[i * width + j] {
            0 => {
                i -= 1;
                j -= 1;
                pairs.push(Pair::both(i, j, sim(i, j)));
            }
            1 => {
                i -= 1;
                pairs.push(Pair::left_only(i));
            }
            _ => {
                j -= 1;
                pairs.push(Pair::right_only(j));
            }
        }
    }
    pairs.reverse();
    pairs
}

/// 意味的アライメントを行わずに、左右をそのまま並べる。
/// 塊が大きすぎて DP に載せられない場合の退避先。
pub fn align_without_scoring(block: &Block) -> Vec<Pair> {
    let mut pairs = Vec::with_capacity(block.left.len() + block.right.len());
    pairs.extend(block.left.clone().map(Pair::left_only));
    pairs.extend(block.right.clone().map(Pair::right_only));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<String> {
        text.lines().map(String::from).collect()
    }

    #[test]
    fn identical_files_produce_a_single_identical_segment() {
        let a = lines("one\ntwo\nthree");
        let segments = segment(&a, &a);
        assert_eq!(
            segments,
            vec![Segment::Identical {
                left_start: 0,
                right_start: 0,
                len: 3
            }]
        );
    }

    #[test]
    fn only_the_changed_region_becomes_a_block() {
        // ここが二段構えの肝。1000 行のうち 1 行だけ違うなら、モデルに渡すのも
        // DP にかけるのもその 1 行分だけで済まなければ意味がない。
        let a = lines("a\nb\nc\nd\ne");
        let b = lines("a\nb\nX\nd\ne");
        let blocks: Vec<Block> = segment(&a, &b)
            .into_iter()
            .filter_map(|s| match s {
                Segment::Changed(b) => Some(b),
                _ => None,
            })
            .collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].left, 2..3);
        assert_eq!(blocks[0].right, 2..3);
    }

    #[test]
    fn adjacent_delete_and_insert_merge_into_one_block() {
        // 分けて流すと、消えた行と増えた行を対応付ける機会が失われる。
        let a = lines("keep\nold1\nold2\ntail");
        let b = lines("keep\nnew1\ntail");
        let blocks: Vec<Block> = segment(&a, &b)
            .into_iter()
            .filter_map(|s| match s {
                Segment::Changed(b) => Some(b),
                _ => None,
            })
            .collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].left, 1..3);
        assert_eq!(blocks[0].right, 1..2);
    }

    #[test]
    fn segments_cover_every_line_exactly_once() {
        let a = lines("a\nb\nc\nd\ne\nf");
        let b = lines("a\nX\nc\nY\nZ\nf");
        let mut left_seen = vec![0usize; a.len()];
        let mut right_seen = vec![0usize; b.len()];
        for s in segment(&a, &b) {
            match s {
                Segment::Identical {
                    left_start,
                    right_start,
                    len,
                } => {
                    for k in 0..len {
                        left_seen[left_start + k] += 1;
                        right_seen[right_start + k] += 1;
                    }
                }
                Segment::Changed(block) => {
                    for i in block.left.clone() {
                        left_seen[i] += 1;
                    }
                    for j in block.right.clone() {
                        right_seen[j] += 1;
                    }
                }
            }
        }
        assert!(left_seen.iter().all(|&c| c == 1), "{left_seen:?}");
        assert!(right_seen.iter().all(|&c| c == 1), "{right_seen:?}");
    }

    #[test]
    fn empty_side_yields_only_gaps() {
        let pairs = needleman_wunsch(0, 3, DEFAULT_PAIR_THRESHOLD, |_, _| 0.0);
        assert_eq!(
            pairs,
            vec![
                Pair::right_only(0),
                Pair::right_only(1),
                Pair::right_only(2)
            ]
        );
        let pairs = needleman_wunsch(2, 0, DEFAULT_PAIR_THRESHOLD, |_, _| 0.0);
        assert_eq!(pairs, vec![Pair::left_only(0), Pair::left_only(1)]);
    }

    #[test]
    fn high_similarity_pairs_are_matched_diagonally() {
        // 対角だけ似ている場合、全行が 1 対 1 で対応するはず。
        let pairs = needleman_wunsch(
            3,
            3,
            DEFAULT_PAIR_THRESHOLD,
            |i, j| {
                if i == j {
                    1.0
                } else {
                    0.0
                }
            },
        );
        assert_eq!(pairs.len(), 3);
        for (k, p) in pairs.iter().enumerate() {
            assert_eq!(p.left, Some(k));
            assert_eq!(p.right, Some(k));
        }
    }

    #[test]
    fn dissimilar_lines_become_gaps_rather_than_forced_pairs() {
        // 何も似ていないなら、無理に対応させず左右を空きで並べる方が読める。
        let pairs = needleman_wunsch(2, 2, DEFAULT_PAIR_THRESHOLD, |_, _| 0.0);
        assert!(pairs.iter().all(|p| p.left.is_none() || p.right.is_none()));
        assert_eq!(pairs.iter().filter(|p| p.left.is_some()).count(), 2);
        assert_eq!(pairs.iter().filter(|p| p.right.is_some()).count(), 2);
    }

    #[test]
    fn an_inserted_line_shifts_the_rest_instead_of_misaligning_it() {
        // 右に 1 行挿入された形。左 0,1,2 が右 0,2,3 に対応し、右 1 が空きになる。
        let pairs = needleman_wunsch(3, 4, DEFAULT_PAIR_THRESHOLD, |i, j| {
            let mapping = [0usize, 2, 3];
            if mapping[i] == j {
                1.0
            } else {
                0.0
            }
        });
        let matched: Vec<(usize, usize)> = pairs
            .iter()
            .filter_map(|p| Some((p.left?, p.right?)))
            .collect();
        assert_eq!(matched, vec![(0, 0), (1, 2), (2, 3)]);
        assert_eq!(
            pairs.iter().filter(|p| p.left.is_none()).count(),
            1,
            "挿入された行が空き 1 つとして出ること"
        );
    }

    #[test]
    fn a_single_corresponding_line_is_not_dragged_out_of_place() {
        // 左 [A, B]、右 [B', C] で、対応するのは B と B' だけという形。
        // 旧実装の罰則の置き方では、長さが揃っている限り必ず対にされるため
        // A↔B' と B↔C という誤った並びになっていた。
        // 正しくは A を空き、B↔B'、C を空きとして並べる。
        let sim = |i: usize, j: usize| match (i, j) {
            (1, 0) => 0.95, // B と B'
            _ => 0.1,
        };
        let pairs = needleman_wunsch(2, 2, DEFAULT_PAIR_THRESHOLD, sim);
        let matched: Vec<(usize, usize)> = pairs
            .iter()
            .filter_map(|p| Some((p.left?, p.right?)))
            .collect();
        assert_eq!(matched, vec![(1, 0)], "対応するのは B と B' の 1 組だけ");
        assert_eq!(pairs.iter().filter(|p| p.right.is_none()).count(), 1);
        assert_eq!(pairs.iter().filter(|p| p.left.is_none()).count(), 1);
    }

    #[test]
    fn a_removed_line_is_listed_before_the_added_one() {
        // 対にするほど似ていない 1 行同士は、左右それぞれの空きとして並ぶ。
        // その並び順は削除が先。逆だと「増えてから消えた」ように読めてしまう。
        let pairs = needleman_wunsch(1, 1, DEFAULT_PAIR_THRESHOLD, |_, _| 0.0);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], Pair::left_only(0), "削除が先に来ること");
        assert_eq!(pairs[1], Pair::right_only(0), "追加が後に来ること");
    }

    #[test]
    fn pairs_report_the_raw_similarity_not_the_thresholded_one() {
        let pairs = needleman_wunsch(1, 1, DEFAULT_PAIR_THRESHOLD, |_, _| 0.8);
        assert_eq!(pairs[0].score, Some(0.8));
    }

    #[test]
    fn every_line_appears_exactly_once_and_in_order() {
        // 行が消えたり重複したりしないことの担保。旧実装は backtrack が None に
        // 当たると break していて、経路が途切れると残りの行が黙って消えていた。
        let pairs = needleman_wunsch(5, 4, DEFAULT_PAIR_THRESHOLD, |i, j| {
            ((i + j) % 3) as f32 * 0.4
        });
        let lefts: Vec<usize> = pairs.iter().filter_map(|p| p.left).collect();
        let rights: Vec<usize> = pairs.iter().filter_map(|p| p.right).collect();
        assert_eq!(lefts, vec![0, 1, 2, 3, 4]);
        assert_eq!(rights, vec![0, 1, 2, 3]);
    }

    #[test]
    fn oversized_block_falls_back_to_plain_listing() {
        let block = Block {
            left: 0..2,
            right: 0..3,
        };
        let pairs = align_without_scoring(&block);
        assert_eq!(pairs.iter().filter_map(|p| p.left).count(), 2);
        assert_eq!(pairs.iter().filter_map(|p| p.right).count(), 3);
    }
}
