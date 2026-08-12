namespace DeepCompare.Engine;

/// <summary>
/// 差分のひと塊。表示行の連続した範囲と、それが左右のどの行に当たるかを持つ。
///
/// <see cref="LeftStart"/> / <see cref="RightStart"/> は、その側に行が 1 つも無いとき
/// （片側だけの追加や削除）は<b>挿入位置</b>を指す。0 行の塊でも「どこへ入れるか」が
/// 決まらないと反映できないため、常に意味のある値を入れてある。
/// </summary>
public sealed record DiffBlock(
    int RowStart,
    int RowCount,
    int LeftStart,
    int LeftCount,
    int RightStart,
    int RightCount)
{
    public bool LeftIsEmpty => LeftCount == 0;
    public bool RightIsEmpty => RightCount == 0;
}

/// <summary>
/// 差分の反映。
///
/// 比較結果から「ひと塊」を切り出し、片側の内容をもう片側へ写す。ここが無いと
/// 見るだけの道具で終わるので、比較機能と同じだけ重要。
///
/// 文字列の並びを受け取って新しい並びを返すだけにしてあり、ファイルにも UI にも
/// 触らない。反映の正しさを画面なしで固定できるようにするため。
/// </summary>
public static class Merge
{
    /// <summary>
    /// 比較結果を差分の塊に切る。一致した行は含めない。
    ///
    /// 連続した「一致していない行」を 1 つの塊にまとめる。1 行ずつ反映させると
    /// 「1 行消して 1 行足す」形の変更を 2 回操作することになり、間で不整合な
    /// 状態が見えてしまう。
    /// </summary>
    public static List<DiffBlock> Blocks(Comparison comparison)
    {
        var rows = comparison.Rows;
        var blocks = new List<DiffBlock>();

        // 片側に行が無い塊のために、直前までに出てきた行番号の続きを覚えておく。
        var nextLeft = 0;
        var nextRight = 0;

        var index = 0;
        while (index < rows.Count)
        {
            if (rows[index].IsUnchanged)
            {
                if (rows[index].Left is { } ul)
                {
                    nextLeft = ul + 1;
                }
                if (rows[index].Right is { } ur)
                {
                    nextRight = ur + 1;
                }
                index++;
                continue;
            }

            var start = index;
            int? firstLeft = null;
            int? firstRight = null;
            var leftCount = 0;
            var rightCount = 0;

            while (index < rows.Count && !rows[index].IsUnchanged)
            {
                if (rows[index].Left is { } l)
                {
                    firstLeft ??= l;
                    leftCount++;
                    nextLeft = l + 1;
                }
                if (rows[index].Right is { } r)
                {
                    firstRight ??= r;
                    rightCount++;
                    nextRight = r + 1;
                }
                index++;
            }

            blocks.Add(new DiffBlock(
                start, index - start,
                firstLeft ?? nextLeft, leftCount,
                firstRight ?? nextRight, rightCount));
        }

        return blocks;
    }

    /// <summary>
    /// 塊の範囲について、写す側の行で写される側の行を置き換えた結果を返す。
    /// 元の並びは変更しない。
    /// </summary>
    /// <param name="target">写される側の全行。</param>
    /// <param name="targetStart">置き換えを始める行。</param>
    /// <param name="targetCount">置き換える行数。0 なら挿入になる。</param>
    /// <param name="source">写す側の全行。</param>
    /// <param name="sourceStart">写し始める行。</param>
    /// <param name="sourceCount">写す行数。0 なら削除になる。</param>
    public static List<string> Replace(
        IReadOnlyList<string> target, int targetStart, int targetCount,
        IReadOnlyList<string> source, int sourceStart, int sourceCount)
    {
        if (targetStart < 0 || targetCount < 0 || targetStart + targetCount > target.Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(targetStart),
                $"置き換える範囲が並びの外にある: 開始 {targetStart} 数 {targetCount} 全体 {target.Count}");
        }
        if (sourceStart < 0 || sourceCount < 0 || sourceStart + sourceCount > source.Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sourceStart),
                $"写す範囲が並びの外にある: 開始 {sourceStart} 数 {sourceCount} 全体 {source.Count}");
        }

        var result = new List<string>(target.Count - targetCount + sourceCount);
        for (var i = 0; i < targetStart; i++)
        {
            result.Add(target[i]);
        }
        for (var i = 0; i < sourceCount; i++)
        {
            result.Add(source[sourceStart + i]);
        }
        for (var i = targetStart + targetCount; i < target.Count; i++)
        {
            result.Add(target[i]);
        }
        return result;
    }

    /// <summary>塊 1 つを左から右へ写す。</summary>
    public static List<string> CopyToRight(
        DiffBlock block, IReadOnlyList<string> leftLines, IReadOnlyList<string> rightLines)
        => Replace(
            rightLines, block.RightStart, block.RightCount,
            leftLines, block.LeftStart, block.LeftCount);

    /// <summary>塊 1 つを右から左へ写す。</summary>
    public static List<string> CopyToLeft(
        DiffBlock block, IReadOnlyList<string> leftLines, IReadOnlyList<string> rightLines)
        => Replace(
            leftLines, block.LeftStart, block.LeftCount,
            rightLines, block.RightStart, block.RightCount);
}
