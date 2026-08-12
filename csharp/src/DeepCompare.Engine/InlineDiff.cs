namespace DeepCompare.Engine;

public enum SpanKind
{
    Equal,
    Changed,
}

/// <summary>行の一部を指す。Start/Length は元の行への文字単位の範囲。</summary>
public readonly record struct Span(SpanKind Kind, int Start, int Length)
{
    public int End => Start + Length;
}

/// <summary>
/// 行内の差分。どこが変わったかを構造として返す。
///
/// 旧 Python 実装はここで HTML 文字列を組み立て、それをリッチテキストとして描画側に
/// 渡していた。差分部分をタグで囲むだけで本文のエスケープをしておらず、&lt;, &amp;, &gt; を
/// 含む行——つまり C++ のテンプレート、HTML、XML、ジェネリクスを含むあらゆるコード——は
/// 表示が壊れるか、内容が丸ごと消えていた。
///
/// 文字列ではなく範囲の列を返せば、書式付けの責任は描画側に移り、この種の取り違えは
/// 起こりようがなくなる。
/// </summary>
public static class InlineDiff
{
    /// <summary>
    /// 行内差分をあきらめる長さ。
    ///
    /// 文字単位の差分は O(n·m) なので、minify されたファイルの 1 行のような極端な入力で
    /// 固まる。その場合は行全体を「変更あり」として扱う。行の対応付け自体は済んでいるので、
    /// 失われるのは行内のどこが変わったかという情報だけ。
    /// </summary>
    public const int MaxInlineDiffChars = 8192;

    public static (List<Span> Left, List<Span> Right) Compute(string left, string right)
    {
        if (left == right)
        {
            return (Whole(left, SpanKind.Equal), Whole(right, SpanKind.Equal));
        }
        if (left.Length > MaxInlineDiffChars || right.Length > MaxInlineDiffChars)
        {
            return (Whole(left, SpanKind.Changed), Whole(right, SpanKind.Changed));
        }

        var leftBuilder = new SpanBuilder();
        var rightBuilder = new SpanBuilder();

        // 文字を 1 要素とみなして、行の差分と同じ経路を使う。
        var leftChars = ToUnits(left);
        var rightChars = ToUnits(right);
        foreach (var op in Myers.Compute(leftChars, rightChars))
        {
            switch (op.Kind)
            {
                case DiffKind.Equal:
                    leftBuilder.Push(SpanKind.Equal, UnitLength(leftChars, op.OldStart, op.OldLength));
                    rightBuilder.Push(SpanKind.Equal, UnitLength(rightChars, op.NewStart, op.NewLength));
                    break;
                case DiffKind.Delete:
                    leftBuilder.Push(SpanKind.Changed, UnitLength(leftChars, op.OldStart, op.OldLength));
                    break;
                case DiffKind.Insert:
                    rightBuilder.Push(SpanKind.Changed, UnitLength(rightChars, op.NewStart, op.NewLength));
                    break;
                case DiffKind.Replace:
                    leftBuilder.Push(SpanKind.Changed, UnitLength(leftChars, op.OldStart, op.OldLength));
                    rightBuilder.Push(SpanKind.Changed, UnitLength(rightChars, op.NewStart, op.NewLength));
                    break;
            }
        }
        return (leftBuilder.Finish(), rightBuilder.Finish());
    }

    /// <summary>
    /// 差分の単位。サロゲートペアと結合文字は 1 単位にまとめる。
    /// 途中で切ると、描画時に文字が壊れる。
    /// </summary>
    private static List<string> ToUnits(string text)
    {
        var units = new List<string>(text.Length);
        var enumerator = System.Globalization.StringInfo.GetTextElementEnumerator(text);
        while (enumerator.MoveNext())
        {
            units.Add((string)enumerator.Current);
        }
        return units;
    }

    private static int UnitLength(List<string> units, int start, int length)
    {
        var total = 0;
        for (var i = start; i < start + length; i++)
        {
            total += units[i].Length;
        }
        return total;
    }

    private static List<Span> Whole(string text, SpanKind kind)
        => text.Length == 0 ? [] : [new Span(kind, 0, text.Length)];

    /// <summary>
    /// 同じ種別が続く範囲は 1 つにまとめる。文字単位の差分をそのまま出すと
    /// 1 文字ずつの範囲が並び、描画側が無駄に重くなる。
    /// </summary>
    private sealed class SpanBuilder
    {
        private readonly List<Span> _spans = [];
        private int _cursor;

        public void Push(SpanKind kind, int length)
        {
            if (length == 0)
            {
                return;
            }
            if (_spans.Count > 0)
            {
                var last = _spans[^1];
                if (last.Kind == kind && last.End == _cursor)
                {
                    _spans[^1] = last with { Length = last.Length + length };
                    _cursor += length;
                    return;
                }
            }
            _spans.Add(new Span(kind, _cursor, length));
            _cursor += length;
        }

        public List<Span> Finish() => _spans;
    }
}
