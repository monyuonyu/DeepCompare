namespace DeepCompare.Engine;

public sealed record CompareOptions(
    float PairThreshold = Aligner.DefaultPairThreshold,
    long MaxBlockArea = Aligner.MaxBlockArea,

    /// <summary>
    /// 埋め込みに通す行数の上限。超える分の塊は構造的な対応付けのまま残す。
    ///
    /// 1 行あたり 13ms 前後（AOT + AVX2、定格clock）かかる。上限が無いと総入れ替えの
    /// ような差分で 1 分近くかかり、段階 2 の差し替えがいつまでも来ない。
    /// 750 行なら 10 秒程度に収まる。
    ///
    /// 小さい塊から順に予算を割り当てる。小さい塊ほど「1 行が 1 行に書き換わった」に
    /// 近く、意味的な対応付けの効きが大きい。逆に巨大な塊は事実上の書き直しで、
    /// 意味的に対応付けても得るものが少ない。現実的な差分（2000 行の 5% 変更）で
    /// 必要なのは 196 行なので、この上限に当たること自体が稀。
    /// </summary>
    int MaxEmbeddedLines = 750,

    /// <summary>何を「重要でない差分」とみなすか。null なら何も無視しない。</summary>
    Importance? Importance = null,

    /// <summary>
    /// 行内差分の単位をトークンにするための言語。null なら文字単位。
    /// <see cref="Lexer.ForPath"/> で拡張子から決められる。
    /// </summary>
    Language? Language = null,

    /// <summary>
    /// 人が指定した対応付け。null なら自動のまま。
    ///
    /// **自動の対応付けは必ず外す場面がある。** そこを直せるようにする。
    /// </summary>
    ManualAlignment? Manual = null)
{
    /// <summary>null を既定値に潰したもの。呼ぶ側で毎回 null 判定をしなくて済む。</summary>
    public Importance Ignoring => Importance ?? Engine.Importance.Default;
}

/// <summary>表示 1 行分。</summary>
public sealed record Row(
    int? Left,
    int? Right,
    float? Score,
    IReadOnlyList<Span> LeftSpans,
    IReadOnlyList<Span> RightSpans)
{
    /// <summary>
    /// 左右が揃っていて、重要な違いが無いか。表示の色分けと絞り込みはこれで決める。
    /// 「重要でない」と定義された違いは、ここでは無いものとして扱う。
    /// </summary>
    public bool IsUnchanged =>
        Left is not null
        && Right is not null
        && LeftSpans.All(s => s.Kind != SpanKind.Changed)
        && RightSpans.All(s => s.Kind != SpanKind.Changed);

    /// <summary>重要でない違いを含むか。一致扱いだが完全に同一ではない、の印。</summary>
    public bool HasUnimportantDifferences =>
        LeftSpans.Any(s => s.Kind == SpanKind.Unimportant)
        || RightSpans.Any(s => s.Kind == SpanKind.Unimportant);
}

public sealed record CompareStats(
    int Rows,
    int IdenticalLines,
    int EmbeddedLines,
    int SkippedBlocks,
    int UnimportantRows = 0);

public sealed record Comparison(IReadOnlyList<Row> Rows, CompareStats Stats);

/// <summary>進捗の通知。埋め込みが支配的なので、その前後だけ知らせれば十分。</summary>
public enum Phase
{
    Segmenting,
    Embedding,
    Aligning,
    Done,
}

/// <summary>
/// 比較全体の組み立て。
///
/// 手順は、まず文字列一致で動かない区間を畳み、残った塊の行だけをまとめて埋め込み、
/// 塊ごとに Needleman-Wunsch をかけ、最後に対応した行の行内差分を取る、の順。
///
/// 埋め込みを「塊ごと」ではなく全塊分まとめて一度呼ぶのは意図的で、そうしないと
/// 重複除去と長さ揃えが塊の内側でしか効かなくなる。
///
/// <b>二段階で使う。</b><see cref="Compare"/> の embedder に null を渡すと Myers の
/// 結果だけで組み立てて即座に返る（2000 行で 20ms 程度）。UI はまずこれを描き、
/// 続けて embedder 付きで呼び直した結果に差し替える。
///
/// 段階を分ける理由は、埋め込みが比較時間のほぼ全部を占めるから。同じ 2000 行で
/// GNU diff が 2ms、こちらは 2.7 秒かかる。乗り換えを狙う道具として、その差を
/// そのまま待ち時間にはできない。構造的な答えを先に出し、意味的な対応付けは
/// 後から差し替える。
/// </summary>
public static class DiffComparer
{
    /// <param name="embedder">
    /// null なら埋め込みを使わず、変更ブロックは構造的な対応付け（左を並べてから右）
    /// のままにする。段階 1 用。
    /// </param>
    public static Comparison Compare(
        DecodedText left,
        DecodedText right,
        Embedder? embedder,
        CompareOptions? options = null,
        Action<Phase>? progress = null)
    {
        options ??= new CompareOptions();
        var ignoring = options.Ignoring;

        progress?.Invoke(Phase.Segmenting);

        // 重要でない差分を落とした形で対応付ける。空白だけが違う行はここで一致に畳まれ、
        // 埋め込みにも回らなくなる。表示には元の行を使う。
        var leftKeys = ignoring.NormalizeAll(left.Lines);
        var rightKeys = ignoring.NormalizeAll(right.Lines);
        var segments = Aligner.Split(leftKeys, rightKeys);

        // どの塊を埋め込みで精緻化するかを先に決める。小さい塊から予算を使う。
        var refined = new HashSet<int>();
        var changedBlocks = 0;
        for (var index = 0; index < segments.Count; index++)
        {
            if (segments[index] is Segment.Changed)
            {
                changedBlocks++;
            }
        }

        if (embedder is not null)
        {
            var budget = options.MaxEmbeddedLines;
            var candidates = new List<(int Index, int Cost)>();
            for (var index = 0; index < segments.Count; index++)
            {
                if (segments[index] is Segment.Changed c && c.Block.Area <= options.MaxBlockArea)
                {
                    candidates.Add((index, c.Block.LeftLength + c.Block.RightLength));
                }
            }
            candidates.Sort((a, b) => a.Cost.CompareTo(b.Cost));
            foreach (var (index, cost) in candidates)
            {
                // 昇順なので、予算に入らなくなったら以降も入らない。
                if (cost > budget)
                {
                    break;
                }
                budget -= cost;
                refined.Add(index);
            }
        }

        var skippedBlocks = changedBlocks - refined.Count;

        // 埋め込みが要る行を集めて、一度に通す。
        progress?.Invoke(Phase.Embedding);
        var wanted = new List<string>();
        var leftSlot = new Dictionary<int, int>();
        var rightSlot = new Dictionary<int, int>();

        foreach (var index in refined)
        {
            var block = ((Segment.Changed)segments[index]).Block;
            for (var i = 0; i < block.LeftLength; i++)
            {
                var line = block.LeftStart + i;
                if (!leftSlot.ContainsKey(line))
                {
                    leftSlot[line] = wanted.Count;
                    // 正規化後を通す。無視すると決めた箇所に埋め込みの表現力を
                    // 使わせない。
                    wanted.Add(leftKeys[line]);
                }
            }
            for (var j = 0; j < block.RightLength; j++)
            {
                var line = block.RightStart + j;
                if (!rightSlot.ContainsKey(line))
                {
                    rightSlot[line] = wanted.Count;
                    wanted.Add(rightKeys[line]);
                }
            }
        }

        var vectors = wanted.Count == 0
            ? []
            : embedder!.EmbedLines(wanted);

        progress?.Invoke(Phase.Aligning);
        var rows = new List<Row>();
        var identicalLines = 0;

        for (var index = 0; index < segments.Count; index++)
        {
            switch (segments[index])
            {
                case Segment.Identical identical:
                    identicalLines += identical.Length;
                    for (var k = 0; k < identical.Length; k++)
                    {
                        var l = identical.LeftStart + k;
                        var r = identical.RightStart + k;
                        rows.Add(EqualRow(l, r, left, right, options.Language));
                    }
                    break;

                case Segment.Changed changed:
                    var block = changed.Block;
                    List<Pair> pairs;
                    if (!refined.Contains(index))
                    {
                        pairs = Aligner.WithoutScoring(block);
                    }
                    else
                    {
                        var local = Aligner.NeedlemanWunsch(
                            block.LeftLength,
                            block.RightLength,
                            options.PairThreshold,
                            (i, j) => Embedder.CosineSimilarity(
                                vectors[leftSlot[block.LeftStart + i]],
                                vectors[rightSlot[block.RightStart + j]]));
                        // 塊の中の相対位置を絶対行番号へ戻す。
                        pairs = local
                            .Select(p => new Pair(
                                p.Left is { } li ? block.LeftStart + li : null,
                                p.Right is { } ri ? block.RightStart + ri : null,
                                p.Score))
                            .ToList();
                    }
                    foreach (var pair in pairs)
                    {
                        // 塊の中でも、正規化後が同じなら「重要でない差分」として扱う。
                        // Myers は連続した区間しか畳まないので、こちらへ落ちてくることがある。
                        if (pair.Left is { } pl && pair.Right is { } pr
                            && !ignoring.IgnoresNothing
                            && string.Equals(leftKeys[pl], rightKeys[pr], StringComparison.Ordinal))
                        {
                            rows.Add(EqualRow(pl, pr, left, right, options.Language));
                        }
                        else
                        {
                            rows.Add(BuildRow(pair, left, right, options.Language));
                        }
                    }
                    break;
            }
        }

        // 末尾の改行だけが違う場合。行の内容は同じなので差分が 1 つも立たず、
        // バイトが違うのに「一致」と報告してしまう。GNU diff は最終行を変更として
        // 出すので、それに合わせる（unified 形式で適用できる形にもなる）。
        if (left.EndsWithNewline != right.EndsWithNewline
            && rows.Count > 0
            && rows[^1] is { Left: not null, Right: not null } lastRow
            && lastRow.IsUnchanged)
        {
            rows[^1] = lastRow with
            {
                LeftSpans = WholeChanged(left.Lines[lastRow.Left.Value]),
                RightSpans = WholeChanged(right.Lines[lastRow.Right.Value]),
            };
            identicalLines--;
        }

        // 人が指定した対応付けを当てる。**最後に当てる**ので、埋め込みの
        // 経路には手が入らない（指定を変えても再計算は行の並べ替えだけ）。
        var finalRows = options.Manual is { IsEmpty: false } manual
            ? ManualAlignment.Apply(rows, manual, left, right, options.Language)
            : rows;

        progress?.Invoke(Phase.Done);

        var unimportant = 0;
        foreach (var row in finalRows)
        {
            if (row.HasUnimportantDifferences)
            {
                unimportant++;
            }
        }

        return new Comparison(
            finalRows,
            new CompareStats(finalRows.Count, identicalLines, wanted.Count, skippedBlocks, unimportant));
    }

    private static IReadOnlyList<Span> WholeEqual(string line)
        => line.Length == 0 ? [] : [new Span(SpanKind.Equal, 0, line.Length)];

    private static IReadOnlyList<Span> WholeChanged(string line)
        => line.Length == 0 ? [] : [new Span(SpanKind.Changed, 0, line.Length)];

    /// <summary>
    /// 対応付けの上では一致している 1 行。元の文字列まで同じならそのまま、違うなら
    /// その違いは「重要でない」ものなので、行内差分を取ったうえで印だけ弱める。
    /// </summary>
    private static Row EqualRow(
        int l, int r, DecodedText left, DecodedText right, Language? language)
    {
        var leftText = left.Lines[l];
        var rightText = right.Lines[r];
        if (string.Equals(leftText, rightText, StringComparison.Ordinal))
        {
            var span = WholeEqual(leftText);
            return new Row(l, r, Aligner.ExactScore, span, span);
        }

        var (a, b) = InlineDiff.Compute(leftText, rightText, language);
        return new Row(l, r, Aligner.ExactScore, Soften(a), Soften(b));
    }

    /// <summary>Changed の印を Unimportant に落とす。</summary>
    private static IReadOnlyList<Span> Soften(List<Span> spans)
    {
        for (var i = 0; i < spans.Count; i++)
        {
            if (spans[i].Kind == SpanKind.Changed)
            {
                spans[i] = spans[i] with { Kind = SpanKind.Unimportant };
            }
        }
        return spans;
    }

    private static Row BuildRow(
        Pair pair, DecodedText left, DecodedText right, Language? language)
    {
        IReadOnlyList<Span> leftSpans;
        IReadOnlyList<Span> rightSpans;

        if (pair.Left is { } l && pair.Right is { } r)
        {
            var (a, b) = InlineDiff.Compute(left.Lines[l], right.Lines[r], language);
            leftSpans = a;
            rightSpans = b;
        }
        else if (pair.Left is { } only)
        {
            var text = left.Lines[only];
            leftSpans = text.Length == 0 ? [] : [new Span(SpanKind.Changed, 0, text.Length)];
            rightSpans = [];
        }
        else if (pair.Right is { } added)
        {
            var text = right.Lines[added];
            leftSpans = [];
            rightSpans = text.Length == 0 ? [] : [new Span(SpanKind.Changed, 0, text.Length)];
        }
        else
        {
            leftSpans = [];
            rightSpans = [];
        }

        return new Row(pair.Left, pair.Right, pair.Score, leftSpans, rightSpans);
    }
}
