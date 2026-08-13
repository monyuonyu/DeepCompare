namespace DeepCompare.Engine;

/// <summary>
/// 対応付けの結果の中から、元の行番号で位置を探す。
///
/// **二段階描画のために要る。** 段階 2 で行数が変わる（2000 行の 5% 変更で
/// 2098 → 2027 行）ので、差し替えたときに読んでいた場所が飛ぶ。
/// 行の添字は差し替えで動くが、元のファイルの行番号は動かない。
/// </summary>
public static class RowAnchor
{
    /// <summary>
    /// その行番号を持つ行の位置。見つからなければ -1。
    ///
    /// **左を先に全部見てから右を見る。** 「どちらかが合えばよい」と
    /// 1 回の走査で決めると、左がもっと近くにあるのに遠くの右に当たって飛ぶ。
    /// </summary>
    public static int Find(IReadOnlyList<Row> rows, (int? Left, int? Right) anchor)
    {
        if (anchor is { Left: null, Right: null })
        {
            return -1;
        }

        if (anchor.Left is { } left)
        {
            for (var i = 0; i < rows.Count; i++)
            {
                if (rows[i].Left == left)
                {
                    return i;
                }
            }
        }

        // 左が消えている場合（段階 2 で対応の付き方が変わった）だけ、右で探す。
        if (anchor.Right is { } right)
        {
            for (var i = 0; i < rows.Count; i++)
            {
                if (rows[i].Right == right)
                {
                    return i;
                }
            }
        }
        return -1;
    }
}
