// クラス Merge とメソッド ThreeWayMerge.Merge が同名で衝突するので別名を付ける。
using MergeOps = DeepCompare.Engine.Merge;

namespace DeepCompare.Engine;

/// <summary>まとまり 1 つがどこから来たか。</summary>
public enum MergeSource
{
    /// <summary>どちらも触っていない。共通祖先のまま。</summary>
    Unchanged,

    /// <summary>左だけが変えた。</summary>
    Left,

    /// <summary>右だけが変えた。</summary>
    Right,

    /// <summary>左右が同じ変更をした。どちらを採っても同じ。</summary>
    Both,

    /// <summary>左右が違う変更をした。人が決める必要がある。</summary>
    Conflict,
}

/// <summary>
/// マージ結果のまとまり 1 つ。
///
/// 競合のときは <see cref="Lines"/> が空で、三者の内容を個別に持つ。どちらかを
/// 勝手に選んで埋めると、選んだこと自体が見えなくなる。
/// </summary>
public sealed record MergeRegion(
    MergeSource Source,
    IReadOnlyList<string> Lines,
    IReadOnlyList<string> BaseLines,
    IReadOnlyList<string> LeftLines,
    IReadOnlyList<string> RightLines);

public sealed record ThreeWayResult(IReadOnlyList<MergeRegion> Regions)
{
    public int ConflictCount => Regions.Count(r => r.Source == MergeSource.Conflict);

    public bool HasConflicts => ConflictCount > 0;

    /// <summary>
    /// 1 本の行の並びにする。
    /// </summary>
    /// <param name="markConflicts">
    /// 競合を git 風の印（<c>&lt;&lt;&lt;&lt;&lt;&lt;&lt;</c> 等）で囲む。false なら
    /// 競合部分は左を採る。**印を消して黙って左を採ると、競合があったこと自体が
    /// 消える**ので、既定は印を付ける。
    /// </param>
    public List<string> ToLines(bool markConflicts = true, string leftLabel = "左", string rightLabel = "右")
    {
        var result = new List<string>();
        foreach (var region in Regions)
        {
            if (region.Source != MergeSource.Conflict)
            {
                result.AddRange(region.Lines);
                continue;
            }

            if (!markConflicts)
            {
                result.AddRange(region.LeftLines);
                continue;
            }

            result.Add($"<<<<<<< {leftLabel}");
            result.AddRange(region.LeftLines);
            result.Add("=======");
            result.AddRange(region.RightLines);
            result.Add($">>>>>>> {rightLabel}");
        }
        return result;
    }
}

/// <summary>
/// 3 方向マージ。共通祖先を挟んで左右の変更を突き合わせる。
///
/// 手順は、祖先に対する左の変更と右の変更をそれぞれ求め、祖先の座標の上で重ね合わせる。
/// 片側だけが触った範囲はそのまま採り、両側が触った範囲は結果が同じかどうかで
/// 採用と競合を分ける。
///
/// **意味的な対応付けがそのまま効く。** 祖先との差分に <see cref="Embedder"/> を使えば、
/// 名前を変えただけの行が「削除＋追加」ではなく「変更」として拾えるので、
/// 競合になる範囲がその分だけ狭くなる。
/// </summary>
public static class ThreeWayMerge
{
    /// <summary>祖先の座標で見た、片側の変更 1 つ。</summary>
    private readonly record struct SideChange(int Start, int Count, string[] Replacement);

    public static ThreeWayResult Merge(
        DecodedText baseText,
        DecodedText left,
        DecodedText right,
        Embedder? embedder = null,
        CompareOptions? options = null)
    {
        var leftChanges = ChangesAgainstBase(baseText, left, embedder, options);
        var rightChanges = ChangesAgainstBase(baseText, right, embedder, options);

        var regions = new List<MergeRegion>();
        var at = 0;
        var i = 0;
        var j = 0;

        while (i < leftChanges.Count || j < rightChanges.Count)
        {
            // 次に現れる変更の位置。
            var nextLeft = i < leftChanges.Count ? leftChanges[i].Start : int.MaxValue;
            var nextRight = j < rightChanges.Count ? rightChanges[j].Start : int.MaxValue;
            var start = Math.Min(nextLeft, nextRight);

            if (at < start)
            {
                regions.Add(Unchanged(baseText, at, start - at));
                at = start;
            }

            // 重なり合う、または隣り合う変更をまとめて 1 つのまとまりにする。
            //
            // **接しているだけの変更も束ねる**（判定が <= なのは意図的）。
            // 例: 祖先 x/y/z に対し、左が y を、右が z を変えた場合。間に一致行が
            // 無いので、これは競合として扱う。
            //
            // 素直に考えれば x/Y/Z へ自動マージできそうだが、`diff3 -m` も
            // `git merge-file` も**どちらも競合と判定する**。git のマージツールとして
            // 使う以上、競合の判定が git と食い違うのは危険で、自動マージした結果が
            // `git merge` と変わってしまう。余計な競合が出る手間より、黙って違う
            // 結果を出す方が損害が大きい。
            //
            // 幅 0 の変更（純粋な挿入）どうしも、この判定で同じ位置なら束ねられる。
            var end = start;
            var takenLeft = new List<SideChange>();
            var takenRight = new List<SideChange>();
            bool grew;
            do
            {
                grew = false;
                while (i < leftChanges.Count && leftChanges[i].Start <= end)
                {
                    end = Math.Max(end, leftChanges[i].Start + leftChanges[i].Count);
                    takenLeft.Add(leftChanges[i]);
                    i++;
                    grew = true;
                }
                while (j < rightChanges.Count && rightChanges[j].Start <= end)
                {
                    end = Math.Max(end, rightChanges[j].Start + rightChanges[j].Count);
                    takenRight.Add(rightChanges[j]);
                    j++;
                    grew = true;
                }
            }
            while (grew);

            var baseLines = Slice(baseText.Lines, start, end - start);
            var leftLines = Apply(baseLines, start, takenLeft);
            var rightLines = Apply(baseLines, start, takenRight);

            var source = (takenLeft.Count > 0, takenRight.Count > 0) switch
            {
                (true, false) => MergeSource.Left,
                (false, true) => MergeSource.Right,
                // 両側が同じ結果に行き着いたなら競合ではない。同じ修正を
                // 別々に入れた場合がこれに当たる。
                (true, true) when leftLines.SequenceEqual(rightLines, StringComparer.Ordinal)
                    => MergeSource.Both,
                _ => MergeSource.Conflict,
            };

            regions.Add(new MergeRegion(
                source,
                source == MergeSource.Conflict
                    ? []
                    : source == MergeSource.Right ? rightLines : leftLines,
                baseLines,
                leftLines,
                rightLines));

            at = end;
        }

        if (at < baseText.Lines.Count)
        {
            regions.Add(Unchanged(baseText, at, baseText.Lines.Count - at));
        }

        return new ThreeWayResult(regions);
    }

    private static MergeRegion Unchanged(DecodedText baseText, int start, int count)
    {
        var lines = Slice(baseText.Lines, start, count);
        return new MergeRegion(MergeSource.Unchanged, lines, lines, lines, lines);
    }

    /// <summary>祖先に対する片側の変更を、祖先の座標で並べる。</summary>
    private static List<SideChange> ChangesAgainstBase(
        DecodedText baseText, DecodedText side, Embedder? embedder, CompareOptions? options)
    {
        var comparison = DiffComparer.Compare(baseText, side, embedder, options);
        var changes = new List<SideChange>();
        foreach (var block in MergeOps.Blocks(comparison))
        {
            var replacement = new string[block.RightCount];
            for (var k = 0; k < block.RightCount; k++)
            {
                replacement[k] = side.Lines[block.RightStart + k];
            }
            changes.Add(new SideChange(block.LeftStart, block.LeftCount, replacement));
        }
        return changes;
    }

    /// <summary>祖先の一部分に、その範囲へかかる変更を当てた結果。</summary>
    private static List<string> Apply(
        IReadOnlyList<string> baseLines, int baseStart, List<SideChange> changes)
    {
        if (changes.Count == 0)
        {
            return [.. baseLines];
        }

        var result = new List<string>();
        var at = 0;
        foreach (var change in changes)
        {
            var from = change.Start - baseStart;
            while (at < from)
            {
                result.Add(baseLines[at]);
                at++;
            }
            result.AddRange(change.Replacement);
            at = from + change.Count;
        }
        while (at < baseLines.Count)
        {
            result.Add(baseLines[at]);
            at++;
        }
        return result;
    }

    private static List<string> Slice(IReadOnlyList<string> lines, int start, int count)
    {
        var result = new List<string>(count);
        for (var k = 0; k < count; k++)
        {
            result.Add(lines[start + k]);
        }
        return result;
    }
}
