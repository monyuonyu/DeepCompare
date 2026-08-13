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

            var brush = line switch
            {
                { IsFiller: true } => Palette.Brush("GapBg"),
                { IsChanged: true } => Palette.Brush("BgChanged"),
                { IsOnlyHere: true } => Palette.Brush("BgAdded"),
                _ => null,
            };
            if (brush is null)
            {
                continue;
            }

            foreach (var rect in BackgroundGeometryBuilder.GetRectsFromVisualSegment(
                view, visual, 0, 1000))
            {
                context.FillRectangle(
                    brush,
                    new Rect(0, rect.Y, view.Bounds.Width, rect.Height));
            }

            // 自分が直した行の印。**左端に細い柱。**
            if (line.IsEdited)
            {
                foreach (var rect in BackgroundGeometryBuilder.GetRectsFromVisualSegment(
                    view, visual, 0, 1))
                {
                    context.FillRectangle(
                        Palette.Brush("EditedMark"),
                        new Rect(0, rect.Y, 3, rect.Height));
                }
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

    public void Update(IReadOnlyList<AlignedLine> lines) => _lines = lines;

    protected override void ColorizeLine(DocumentLine line)
    {
        var number = line.LineNumber;
        if (number < 1 || number > _lines.Count)
        {
            return;
        }

        var info = _lines[number - 1];
        var length = line.Length;
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
}
