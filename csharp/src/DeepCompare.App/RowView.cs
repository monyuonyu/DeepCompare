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
    private static readonly IBrush BgChanged = new SolidColorBrush(Color.Parse("#2E3350"));
    private static readonly IBrush BgRemoved = new SolidColorBrush(Color.Parse("#40262A"));
    private static readonly IBrush BgAdded = new SolidColorBrush(Color.Parse("#223A2A"));
    private static readonly IBrush FgNormal = new SolidColorBrush(Color.Parse("#D6D8DE"));
    private static readonly IBrush FgInline = new SolidColorBrush(Color.Parse("#FFA500"));
    private static readonly IBrush Transparent = Brushes.Transparent;

    public string LeftNumber { get; }
    public string RightNumber { get; }
    public string ScoreText { get; }
    public IBrush Background { get; }
    public InlineCollection LeftInlines { get; }
    public InlineCollection RightInlines { get; }

    public RowView(Row row, DecodedText left, DecodedText right)
    {
        LeftNumber = row.Left is { } l ? (l + 1).ToString() : string.Empty;
        RightNumber = row.Right is { } r ? (r + 1).ToString() : string.Empty;

        var unchanged = row.IsUnchanged;
        ScoreText = row.Score switch
        {
            // 文字列として完全一致した行。数値を出しても意味がないので記号にする。
            not null when unchanged => "=",
            { } score => score.ToString("F2"),
            null => string.Empty,
        };

        Background = (row.Left, row.Right) switch
        {
            (not null, not null) when unchanged => Transparent,
            (not null, not null) => BgChanged,
            (not null, null) => BgRemoved,
            (null, not null) => BgAdded,
            _ => Transparent,
        };

        LeftInlines = Build(row.Left is { } li ? left.Lines[li] : null, row.LeftSpans);
        RightInlines = Build(row.Right is { } ri ? right.Lines[ri] : null, row.RightSpans);
    }

    private static InlineCollection Build(string? text, IReadOnlyList<EngineSpan> spans)
    {
        var inlines = new InlineCollection();
        if (text is null || text.Length == 0)
        {
            return inlines;
        }
        if (spans.Count == 0)
        {
            inlines.Add(new Run(text) { Foreground = FgNormal });
            return inlines;
        }
        foreach (var span in spans)
        {
            inlines.Add(new Run(text.Substring(span.Start, span.Length))
            {
                Foreground = span.Kind == SpanKind.Changed ? FgInline : FgNormal,
            });
        }
        return inlines;
    }
}
