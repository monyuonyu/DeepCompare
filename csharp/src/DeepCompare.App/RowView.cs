using Avalonia.Controls.Documents;
using Avalonia.Media;
using DeepCompare.Engine;
// Avalonia にも Span があるので、どちらを指すか明示する。
using EngineSpan = DeepCompare.Engine.Span;

namespace DeepCompare.App;

/// <summary>
/// 表示 1 行分。
///
/// 文字列ではなく <see cref="InlineCollection"/> を組み立てているのが要点。旧 Python 実装は
/// ここで HTML を作っていたため、`&lt;` や `&amp;` を含む行が壊れていた。書式を持つ要素として
/// 渡せば、本文は一切加工されないのでその種の取り違えが起こりようがない。
/// </summary>
public sealed class RowView
{
    // 色は App.axaml のテーマ辞書から引く（Palette）。ここで値を持たないので、
    // 明暗の切り替えに勝手に付いてくる。
    private static readonly IBrush Transparent = Brushes.Transparent;

    /// <summary>元の比較結果。移動の印を後から付けるために保持する。</summary>
    public Row Row { get; }

    /// <summary>
    /// 移動したブロックの相手側の行番号（1 始まり）。移動でないなら null。
    /// </summary>
    public int? MovedToLine
    {
        get;
        set
        {
            field = value;
            MovedText = value is { } line ? $"⇄{line}" : string.Empty;
        }
    }

    /// <summary>画面に出す移動の印。</summary>
    public string MovedText { get; private set; } = string.Empty;

    /// <summary>属する差分の塊。どれにも属さないなら -1。</summary>
    public int BlockIndex { get; set; } = -1;

    /// <summary>塊の先頭行か。コピーボタンはここにだけ出す。</summary>
    public bool IsBlockStart { get; set; }

    /// <summary>検索に使う素の本文。表示は Inlines 側が持つ。</summary>
    public string LeftText { get; }
    public string RightText { get; }

    public string LeftNumber { get; }
    public string RightNumber { get; }
    public string ScoreText { get; }
    public IBrush Background { get; }
    public InlineCollection LeftInlines { get; }
    public InlineCollection RightInlines { get; }

    public RowView(Row row, DecodedText left, DecodedText right,
        Language? language = null, LexState leftState = default, LexState rightState = default)
    {
        Row = row;
        LeftNumber = row.Left is { } l ? (l + 1).ToString() : string.Empty;
        RightNumber = row.Right is { } r ? (r + 1).ToString() : string.Empty;

        var unchanged = row.IsUnchanged;
        ScoreText = row.Score switch
        {
            // 対応付けの上で一致した行。数値を出しても意味がないので記号にする。
            // 重要でない違いを含む場合は区別できるようにする。
            not null when unchanged => row.HasUnimportantDifferences ? "≈" : "=",
            { } score => score.ToString("F2"),
            null => string.Empty,
        };

        Background = (row.Left, row.Right) switch
        {
            (not null, not null) when unchanged => Transparent,
            (not null, not null) => Palette.Brush("BgChanged"),
            (not null, null) => Palette.Brush("BgRemoved"),
            (null, not null) => Palette.Brush("BgAdded"),
            _ => Transparent,
        };

        LeftText = row.Left is { } li ? left.Lines[li] : string.Empty;
        RightText = row.Right is { } ri ? right.Lines[ri] : string.Empty;
        LeftInlines = Build(row.Left is null ? null : LeftText, row.LeftSpans, language, leftState);
        RightInlines = Build(row.Right is null ? null : RightText, row.RightSpans, language, rightState);
    }

    /// <summary>
    /// 差分の範囲と構文の範囲を重ねて描く。
    ///
    /// 2 つの範囲の切れ目は一致しないので、両方の境界で切り直す。差分が「変更あり」の
    /// ところは差分の色を使い、それ以外を構文の色にする。**どこが変わったかは
    /// 構文の色より優先する。** 色分けが綺麗でも変更点を見失っては本末転倒。
    /// </summary>
    private static InlineCollection Build(
        string? text, IReadOnlyList<EngineSpan> spans, Language? language, LexState state)
    {
        var inlines = new InlineCollection();
        if (text is null || text.Length == 0)
        {
            return inlines;
        }
        if (spans.Count == 0)
        {
            inlines.Add(new Run(text) { Foreground = Palette.Brush("FgNormal") });
            return inlines;
        }

        var tokens = language is null ? null : Lexer.Tokenize(text, language, ref state);

        foreach (var span in spans)
        {
            var diffBrush = span.Kind switch
            {
                SpanKind.Changed => Palette.Brush("FgInline"),
                SpanKind.Unimportant => Palette.Brush("FgUnimportant"),
                _ => (IBrush?)null,
            };

            if (tokens is null || diffBrush is not null)
            {
                // 変更部分は構文で塗り分けない。1 つの塊として見せる。
                inlines.Add(new Run(text.Substring(span.Start, span.Length))
                {
                    Foreground = diffBrush ?? Palette.Brush("FgNormal"),
                });
                continue;
            }

            // 一致部分は構文の色で塗る。範囲を跨ぐトークンは切り出す。
            var at = span.Start;
            var end = span.Start + span.Length;
            foreach (var token in tokens)
            {
                var from = Math.Max(at, token.Start);
                var to = Math.Min(end, token.Start + token.Length);
                if (to <= from)
                {
                    continue;
                }
                inlines.Add(new Run(text[from..to]) { Foreground = Colour(token.Kind) });
                at = to;
            }
            if (at < end)
            {
                inlines.Add(new Run(text[at..end]) { Foreground = Palette.Brush("FgNormal") });
            }
        }
        return inlines;
    }

    private static IBrush Colour(TokenKind kind) => kind switch
    {
        TokenKind.Keyword => Palette.Brush("FgKeyword"),
        TokenKind.String => Palette.Brush("FgString"),
        TokenKind.Comment => Palette.Brush("FgComment"),
        TokenKind.Number => Palette.Brush("FgNumber"),
        TokenKind.Punctuation => Palette.Brush("FgPunctuation"),
        _ => Palette.Brush("FgNormal"),
    };
}
