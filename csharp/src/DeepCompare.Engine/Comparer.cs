namespace DeepCompare.Engine;

public sealed record CompareOptions(
    float PairThreshold = Aligner.DefaultPairThreshold,
    long MaxBlockArea = Aligner.MaxBlockArea);

/// <summary>表示 1 行分。</summary>
public sealed record Row(
    int? Left,
    int? Right,
    float? Score,
    IReadOnlyList<Span> LeftSpans,
    IReadOnlyList<Span> RightSpans)
{
    /// <summary>左右が揃っていて内容も完全に同じか。表示の色分けはこれで決める。</summary>
    public bool IsUnchanged =>
        Left is not null
        && Right is not null
        && LeftSpans.All(s => s.Kind == SpanKind.Equal)
        && RightSpans.All(s => s.Kind == SpanKind.Equal);
}

public sealed record CompareStats(int Rows, int IdenticalLines, int EmbeddedLines, int SkippedBlocks);

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
/// 比較全体の組み立て。UI が呼ぶのはここ一つ。
///
/// 手順は、まず文字列一致で動かない区間を畳み、残った塊の行だけをまとめて埋め込み、
/// 塊ごとに Needleman-Wunsch をかけ、最後に対応した行の行内差分を取る、の順。
///
/// 埋め込みを「塊ごと」ではなく全塊分まとめて一度呼ぶのは意図的で、そうしないと
/// 重複除去と長さ揃えが塊の内側でしか効かなくなる。
/// </summary>
public static class DiffComparer
{
    public static Comparison Compare(
        DecodedText left,
        DecodedText right,
        Embedder embedder,
        CompareOptions? options = null,
        Action<Phase>? progress = null)
    {
        options ??= new CompareOptions();
        progress?.Invoke(Phase.Segmenting);
        var segments = Aligner.Split(left.Lines, right.Lines);

        // 埋め込みが要る行を集めて、一度に通す。
        progress?.Invoke(Phase.Embedding);
        var wanted = new List<string>();
        var leftSlot = new Dictionary<int, int>();
        var rightSlot = new Dictionary<int, int>();
        var skippedBlocks = 0;

        foreach (var segment in segments)
        {
            if (segment is not Segment.Changed changed)
            {
                continue;
            }
            if (changed.Block.Area > options.MaxBlockArea)
            {
                skippedBlocks++;
                continue;
            }
            for (var i = 0; i < changed.Block.LeftLength; i++)
            {
                var line = changed.Block.LeftStart + i;
                if (!leftSlot.ContainsKey(line))
                {
                    leftSlot[line] = wanted.Count;
                    wanted.Add(left.Lines[line]);
                }
            }
            for (var j = 0; j < changed.Block.RightLength; j++)
            {
                var line = changed.Block.RightStart + j;
                if (!rightSlot.ContainsKey(line))
                {
                    rightSlot[line] = wanted.Count;
                    wanted.Add(right.Lines[line]);
                }
            }
        }

        var vectors = embedder.EmbedLines(wanted);

        progress?.Invoke(Phase.Aligning);
        var rows = new List<Row>();
        var identicalLines = 0;

        foreach (var segment in segments)
        {
            switch (segment)
            {
                case Segment.Identical identical:
                    identicalLines += identical.Length;
                    for (var k = 0; k < identical.Length; k++)
                    {
                        var l = identical.LeftStart + k;
                        var r = identical.RightStart + k;
                        var span = WholeEqual(left.Lines[l]);
                        rows.Add(new Row(l, r, Aligner.ExactScore, span, span));
                    }
                    break;

                case Segment.Changed changed:
                    var block = changed.Block;
                    List<Pair> pairs;
                    if (block.Area > options.MaxBlockArea)
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
                        rows.Add(BuildRow(pair, left, right));
                    }
                    break;
            }
        }

        progress?.Invoke(Phase.Done);
        return new Comparison(
            rows,
            new CompareStats(rows.Count, identicalLines, wanted.Count, skippedBlocks));
    }

    private static IReadOnlyList<Span> WholeEqual(string line)
        => line.Length == 0 ? [] : [new Span(SpanKind.Equal, 0, line.Length)];

    private static Row BuildRow(Pair pair, DecodedText left, DecodedText right)
    {
        IReadOnlyList<Span> leftSpans;
        IReadOnlyList<Span> rightSpans;

        if (pair.Left is { } l && pair.Right is { } r)
        {
            var (a, b) = InlineDiff.Compute(left.Lines[l], right.Lines[r]);
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
