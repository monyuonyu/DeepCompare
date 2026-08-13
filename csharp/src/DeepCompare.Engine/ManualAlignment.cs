namespace DeepCompare.Engine;

/// <summary>
/// 人が指定した対応付け。
///
/// **自動の対応付けは必ず外す場面がある。** 意味的な類似度でも、
/// 「同じ役割だが書き方を全部変えた行」までは拾えない。逆に、たまたま
/// 似ているだけの無関係な行が対にされることもある。そこを直せるようにする。
/// </summary>
public sealed record ManualAlignment
{
    /// <summary>
    /// 対にすると決めた組（左の行番号 → 右の行番号）。0 始まり。
    /// </summary>
    public IReadOnlyDictionary<int, int> Linked { get; init; }
        = new Dictionary<int, int>();

    /// <summary>
    /// 対にしないと決めた組。自動で対にされたものを割るのに使う。
    /// </summary>
    public IReadOnlySet<(int Left, int Right)> Unlinked { get; init; }
        = new HashSet<(int, int)>();

    public bool IsEmpty => Linked.Count == 0 && Unlinked.Count == 0;

    /// <summary>組を 1 つ足したもの。**元は変えない**（記録として積める）。</summary>
    public ManualAlignment Link(int leftLine, int rightLine)
    {
        var linked = new Dictionary<int, int>(Linked) { [leftLine] = rightLine };
        // 同じ右行を 2 つの左行に割り当てない。**後から言われた方を採る。**
        foreach (var key in linked.Where(p => p.Value == rightLine && p.Key != leftLine)
                     .Select(p => p.Key).ToList())
        {
            linked.Remove(key);
        }
        var unlinked = new HashSet<(int, int)>(Unlinked);
        unlinked.Remove((leftLine, rightLine));
        return this with { Linked = linked, Unlinked = unlinked };
    }

    public ManualAlignment Unlink(int leftLine, int rightLine)
    {
        var linked = new Dictionary<int, int>(Linked);
        if (linked.TryGetValue(leftLine, out var current) && current == rightLine)
        {
            linked.Remove(leftLine);
        }
        return this with
        {
            Linked = linked,
            Unlinked = new HashSet<(int, int)>(Unlinked) { (leftLine, rightLine) },
        };
    }

    /// <summary>
    /// 対応付けの結果に、人の指定を当てる。
    ///
    /// **行の順序は動かさない。** 左は左の順、右は右の順を保つ。それを崩すと
    /// もはや「同じファイルの別の見せ方」ではなくなる。
    /// 順序の制約で対にできない指定は、**黙って捨てずに諦める**
    /// （できないことをできたように見せない）。
    /// </summary>
    public static IReadOnlyList<Row> Apply(
        IReadOnlyList<Row> rows, ManualAlignment alignment,
        DecodedText left, DecodedText right, Language? language = null)
    {
        if (alignment.IsEmpty)
        {
            return rows;
        }

        var result = new List<Row>(rows);

        // 1. 「対にしない」を先に処理する。**割ってから繋ぐ。**
        //    逆にすると、繋いだそばから割られることがある。
        for (var i = 0; i < result.Count; i++)
        {
            if (result[i] is { Left: { } l, Right: { } r }
                && alignment.Unlinked.Contains((l, r)))
            {
                result[i] = OneSided(l, null, left, right);
                result.Insert(i + 1, OneSided(null, r, left, right));
                i++;
            }
        }

        // 2. 「対にする」を処理する。
        foreach (var (leftLine, rightLine) in alignment.Linked.OrderBy(p => p.Key))
        {
            TryLink(result, leftLine, rightLine, left, right, language);
        }

        return result;
    }

    private static void TryLink(
        List<Row> rows, int leftLine, int rightLine,
        DecodedText left, DecodedText right, Language? language)
    {
        var leftAt = IndexOfLeft(rows, leftLine);
        var rightAt = IndexOfRight(rows, rightLine);
        if (leftAt < 0 || rightAt < 0)
        {
            return;
        }

        // 既に対になっているなら何もしない。
        if (leftAt == rightAt)
        {
            return;
        }

        // まず、それぞれが今持っている相手を外す。
        Detach(rows, leftAt, keepLeft: true, left, right);
        rightAt = IndexOfRight(rows, rightLine);   // 挿入で位置が動きうる
        Detach(rows, rightAt, keepLeft: false, left, right);

        leftAt = IndexOfLeft(rows, leftLine);
        rightAt = IndexOfRight(rows, rightLine);
        if (leftAt < 0 || rightAt < 0)
        {
            return;
        }

        // **間に挟まる行が順序の制約を破らないか確かめる。**
        // 左行より後ろに右行がある場合、その間の左行は右行より前に来ることになり、
        // 右の順序が崩れる。そういう指定は諦める。
        var (from, to) = leftAt < rightAt ? (leftAt, rightAt) : (rightAt, leftAt);
        for (var i = from + 1; i < to; i++)
        {
            var row = rows[i];
            if (leftAt < rightAt && row.Right is not null)
            {
                return;    // 間に右行がある → 繋ぐと右の順序が崩れる
            }
            if (rightAt < leftAt && row.Left is not null)
            {
                return;    // 間に左行がある → 繋ぐと左の順序が崩れる
            }
        }

        // 対にする。片方の行を消して、もう片方を両側持ちにする。
        var (leftSpans, rightSpans) =
            InlineDiff.Compute(left.Lines[leftLine], right.Lines[rightLine], language);
        // **類似度は付けない。** 人が繋いだのであって、モデルが「似ている」と
        // 判断したのではない。1.0 を入れると、モデルが完全一致と見たように読める。
        var merged = new Row(leftLine, rightLine, null, leftSpans, rightSpans);

        rows[from] = merged;
        rows.RemoveAt(to);
    }

    /// <summary>その行が持っている相手を外し、別の行として分ける。</summary>
    private static void Detach(
        List<Row> rows, int at, bool keepLeft, DecodedText left, DecodedText right)
    {
        if (at < 0 || rows[at] is not { Left: { } l, Right: { } r })
        {
            return;
        }
        if (keepLeft)
        {
            rows[at] = OneSided(l, null, left, right);
            rows.Insert(at + 1, OneSided(null, r, left, right));
        }
        else
        {
            rows[at] = OneSided(null, r, left, right);
            rows.Insert(at, OneSided(l, null, left, right));
        }
    }

    private static int IndexOfLeft(List<Row> rows, int line)
        => rows.FindIndex(r => r.Left == line);

    private static int IndexOfRight(List<Row> rows, int line)
        => rows.FindIndex(r => r.Right == line);

    private static Row OneSided(int? leftLine, int? rightLine, DecodedText left, DecodedText right)
        => new(leftLine, rightLine, null,
            leftLine is { } l ? Whole(left.Lines[l]) : [],
            rightLine is { } r ? Whole(right.Lines[r]) : []);

    private static IReadOnlyList<Span> Whole(string line)
        => line.Length == 0 ? [] : [new Span(SpanKind.Changed, 0, line.Length)];
}
