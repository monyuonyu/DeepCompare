using Avalonia;
using Avalonia.Media;
using AvaloniaEdit.Document;
using AvaloniaEdit.Rendering;
using DeepCompare.Engine;
using EngineSpan = DeepCompare.Engine.Span;

namespace DeepCompare.App;

/// <summary>
/// 揃えた本文の 1 行ぶんの素性。
///
/// **エディタへ流す本文は、左右で行数を揃えたもの。** 片側にしか無い行の
/// 位置には詰め物の空行を入れる。そうしないと、同じ内容の行が上下に
/// ずれて並び、目で追えない（Beyond Compare も同じく揃える）。
/// </summary>
public sealed record AlignedLine(
    int? SourceLine,
    bool IsFiller,
    bool IsChanged,
    bool IsOnlyHere,
    bool IsEdited,
    IReadOnlyList<EngineSpan> Spans)
{
    /// <summary>
    /// 差分の塊の何番目か。塊でなければ -1。
    ///
    /// **写す単位は塊。** 1 行ずつ写させると、5 行まとめて書き換えた
    /// 場所で 5 回押すことになる。
    /// </summary>
    public int BlockIndex { get; init; } = -1;

    /// <summary>塊の先頭の行か。**ここにだけ矢印を出す。**</summary>
    public bool IsBlockStart { get; init; }

    /// <summary>
    /// 対応の近さ（0〜1）。対になっていなければ null。
    ///
    /// **この道具の持ち味なので、数字で見せる。** 「なぜこの 2 行が
    /// 並んでいるのか」の根拠がこれ。完全一致は「＝」で示す。
    /// </summary>
    public float? Score { get; init; }
}

/// <summary>
/// 行の地を塗る。
///
/// **文字の後ろではなく、行いっぱいに塗る。** 行末までの余白が塗られて
/// いないと、どこまでがその行なのか分からない。
/// </summary>
public sealed class DiffBackgroundRenderer : IBackgroundRenderer
{
    private IReadOnlyList<AlignedLine> _lines = [];

    public void Update(IReadOnlyList<AlignedLine> lines) => _lines = lines;

    public KnownLayer Layer => KnownLayer.Background;

    private static void DrawHatch(DrawingContext context, Rect area)
        => Hatching.Draw(context, area);

    public void Draw(TextView view, DrawingContext context)
    {
        if (view.VisualLinesValid is false)
        {
            return;
        }

        foreach (var visual in view.VisualLines)
        {
            var number = visual.FirstDocumentLine.LineNumber;
            if (number < 1 || number > _lines.Count)
            {
                continue;
            }
            var line = _lines[number - 1];

            var top = visual.GetTextLineVisualYPosition(
                visual.TextLines[0], VisualYPosition.TextTop) - view.VerticalOffset;
            var area = new Rect(0, top, view.Bounds.Width, visual.Height);

            if (line.IsFiller)
            {
                // **斜線で「ここには行が無い」を示す。**
                // 色を塗るだけだと「空行がある」と読まれる。相手側にしか
                // 行が無いことを、地の模様で言う（Beyond Compare も同じ）。
                DrawHatch(context, area);
                continue;
            }

            var brush = line switch
            {
                { IsChanged: true } => Palette.Brush("BgChanged"),
                { IsOnlyHere: true } => Palette.Brush("BgAdded"),
                _ => null,
            };
            if (brush is null)
            {
                continue;
            }

            context.FillRectangle(brush, area);

            // 自分が直した行の印。**左端に細い柱。**
            if (line.IsEdited)
            {
                context.FillRectangle(
                    Palette.Brush("EditedMark"), new Rect(0, top, 3, visual.Height));
            }
        }
    }
}

/// <summary>
/// 斜線を引く。**行が無いことを示す模様。**
///
/// 一定の間隔で 45 度の線を引くだけ。模様の画像を作って敷く手もあるが、
/// 行の高さが変わるたびに作り直すことになる。
/// </summary>
file static class Hatching
{
    public static void Draw(DrawingContext context, Rect area)
    {
        const double gap = 7;
        var pen = new Pen(Palette.Brush("GapLine"), 1);
        using (context.PushClip(area))
        {
            // 左上から右下へ。**高さのぶんだけ左へずらして始める** —
            // そうしないと左端の三角が塗り残る。
            for (var x = area.X - area.Height; x < area.Right; x += gap)
            {
                context.DrawLine(
                    pen,
                    new Point(x, area.Bottom),
                    new Point(x + area.Height, area.Y));
            }
        }
    }
}

/// <summary>
/// 行の中で変わった文字を塗る。
///
/// **色ではなく地で示す。** 文字の色だけだと、1 文字違いや、もともと
/// 色の付いた場所（文字列・注記）で気づけない。
/// </summary>
public sealed class InlineDiffColorizer : DocumentColorizingTransformer
{
    private IReadOnlyList<AlignedLine> _lines = [];
    private Language? _language;

    public void Update(IReadOnlyList<AlignedLine> lines, Language? language = null)
    {
        _lines = lines;
        _language = language;
    }

    protected override void ColorizeLine(DocumentLine line)
    {
        var number = line.LineNumber;
        if (number < 1 || number > _lines.Count)
        {
            return;
        }

        var info = _lines[number - 1];
        var length = line.Length;

        // **構文の色を先に置く。** 差分の地はこの後に重ねるので、
        // 「どこが変わったか」が構文の色に負けない。
        if (_language is { } language && length > 0)
        {
            var text = CurrentContext.Document.GetText(line.Offset, length);
            var state = default(LexState);
            foreach (var token in Lexer.Tokenize(text, language, ref state))
            {
                var brush = TokenBrush(token.Kind);
                if (brush is null || token.Length <= 0)
                {
                    continue;
                }
                var from = Math.Min(token.Start, length);
                var to = Math.Min(token.Start + token.Length, length);
                if (to <= from)
                {
                    continue;
                }
                ChangeLinePart(line.Offset + from, line.Offset + to, element =>
                    element.TextRunProperties.SetForegroundBrush(brush));
            }
        }
        foreach (var span in info.Spans)
        {
            if (span.Kind != SpanKind.Changed || span.Length <= 0)
            {
                continue;
            }
            // **行の長さを超えない。** 編集の途中では、素性と本文が
            // 一瞬ずれる（打った直後、まだ比べ直していない）。
            var start = Math.Min(span.Start, length);
            var end = Math.Min(span.Start + span.Length, length);
            if (end <= start)
            {
                continue;
            }
            ChangeLinePart(line.Offset + start, line.Offset + end, element =>
            {
                element.TextRunProperties.SetBackgroundBrush(Palette.Brush("BgInline"));
            });
        }
    }

    /// <summary>構文の色。**行の中で意味の違うものだけ分ける。**</summary>
    private static IBrush? TokenBrush(TokenKind kind) => kind switch
    {
        TokenKind.Keyword => Palette.Brush("FgKeyword"),
        TokenKind.String => Palette.Brush("FgString"),
        TokenKind.Comment => Palette.Brush("FgComment"),
        TokenKind.Number => Palette.Brush("FgNumber"),
        TokenKind.Punctuation => Palette.Brush("FgPunctuation"),
        _ => null,
    };
}
