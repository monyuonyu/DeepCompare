namespace DeepCompare.Engine;

/// <summary>ファイルの中で位置が変わった一続きの行。</summary>
public sealed record MovedBlock(
    int LeftStart,
    int RightStart,
    int Length,
    /// <summary>行の内容まで完全に同じか。</summary>
    bool Exact);

/// <summary>
/// 移動したブロックの検出。
///
/// 関数をファイル内で動かすと、行の対応付けは「その場所から削除」「別の場所へ追加」に
/// 分かれる。差分としては正しいが、読む側が知りたいのは「動いただけ」であって、
/// 2 箇所を見比べて同じ内容だと確かめる作業ではない。
///
/// 対象は片側にしか出てこなかった行だけ。対応の付いた行は既に並んでいるので触らない。
/// </summary>
public static class MovedBlockDetector
{
    /// <summary>
    /// これより短いものは移動とみなさない。
    ///
    /// `}` や空行のような 1 行は、どのファイルにも何度も現れる。短い一致を拾うと
    /// 「移動」だらけになって、本当に動いたものが埋もれる。
    /// </summary>
    public const int DefaultMinimumLines = 3;

    public static List<MovedBlock> Detect(
        Comparison comparison,
        DecodedText left,
        DecodedText right,
        int minimumLines = DefaultMinimumLines)
    {
        // 片側にしか出なかった行を、連続する塊として集める。
        var leftRuns = Runs(comparison.Rows, takeLeft: true);
        var rightRuns = Runs(comparison.Rows, takeLeft: false);
        if (leftRuns.Count == 0 || rightRuns.Count == 0)
        {
            return [];
        }

        var moved = new List<MovedBlock>();
        var usedRight = new List<(int Start, int End)>();

        foreach (var leftRun in leftRuns)
        {
            foreach (var rightRun in rightRuns)
            {
                var match = LongestCommonRun(
                    left.Lines, leftRun, right.Lines, rightRun, minimumLines, usedRight);
                if (match is not { } found)
                {
                    continue;
                }

                moved.Add(found);
                usedRight.Add((found.RightStart, found.RightStart + found.Length));
                break;
            }
        }

        return moved;
    }

    /// <summary>片側だけの行が続く区間を集める。</summary>
    private static List<(int Start, int End)> Runs(IReadOnlyList<Row> rows, bool takeLeft)
    {
        var runs = new List<(int Start, int End)>();
        int? start = null;
        int? previous = null;

        foreach (var row in rows)
        {
            var line = takeLeft
                ? (row.Right is null ? row.Left : null)
                : (row.Left is null ? row.Right : null);

            if (line is { } value)
            {
                if (start is null)
                {
                    start = value;
                }
                else if (previous is { } last && value != last + 1)
                {
                    // 行番号が飛んだら別の塊。
                    runs.Add((start.Value, last + 1));
                    start = value;
                }
                previous = value;
                continue;
            }

            if (start is { } from && previous is { } to)
            {
                runs.Add((from, to + 1));
            }
            start = null;
            previous = null;
        }

        if (start is { } lastFrom && previous is { } lastTo)
        {
            runs.Add((lastFrom, lastTo + 1));
        }
        return runs;
    }

    /// <summary>
    /// 2 つの区間の中から、内容の一致する最長の連続部分を探す。
    /// 既に別の移動として使った右側の範囲は避ける。
    /// </summary>
    private static MovedBlock? LongestCommonRun(
        IReadOnlyList<string> leftLines, (int Start, int End) leftRun,
        IReadOnlyList<string> rightLines, (int Start, int End) rightRun,
        int minimumLines,
        List<(int Start, int End)> usedRight)
    {
        MovedBlock? best = null;

        for (var i = leftRun.Start; i < leftRun.End; i++)
        {
            for (var j = rightRun.Start; j < rightRun.End; j++)
            {
                var length = 0;
                while (i + length < leftRun.End
                       && j + length < rightRun.End
                       && string.Equals(
                           leftLines[i + length], rightLines[j + length], StringComparison.Ordinal))
                {
                    length++;
                }

                if (length < minimumLines || length <= (best?.Length ?? 0))
                {
                    continue;
                }
                if (usedRight.Any(used => j < used.End && used.Start < j + length))
                {
                    continue;
                }
                best = new MovedBlock(i, j, length, Exact: true);
            }
        }

        return best;
    }
}
