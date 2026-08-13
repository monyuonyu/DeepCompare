using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using AvaloniaEdit.Rendering;

namespace DeepCompare.App;

/// <summary>
/// 対応の近さを行ごとに出す列。
///
/// **この道具の持ち味なので、数字で見せる。** 「なぜこの 2 行が並んで
/// いるのか」の根拠がこれ。文字列としては別物でも 0.89 で対になった、
/// と分かることに意味がある。
///
/// 完全一致は「＝」。数字を並べても読み飛ばすだけで、**違うものだけが
/// 目に入る**方がよい。
/// </summary>
public sealed class ScoreColumn : Control
{
    private IReadOnlyList<AlignedLine> _lines = [];
    private TextView? _view;

    public ScoreColumn()
    {
        Width = 46;
    }

    public void Attach(TextView view)
    {
        _view = view;
        view.ScrollOffsetChanged += (_, _) => InvalidateVisual();
        view.VisualLinesChanged += (_, _) => InvalidateVisual();
    }

    public void Update(IReadOnlyList<AlignedLine> lines)
    {
        _lines = lines;
        InvalidateVisual();
    }

    public override void Render(DrawingContext context)
    {
        if (_view is not { VisualLinesValid: true } view)
        {
            return;
        }

        var typeface = new Typeface(FontFamily.Default);
        var dim = Palette.Brush("FgDim");
        var normal = Palette.Brush("FgNormal");

        foreach (var visual in view.VisualLines)
        {
            var number = visual.FirstDocumentLine.LineNumber;
            if (number < 1 || number > _lines.Count)
            {
                continue;
            }
            var line = _lines[number - 1];
            if (line.IsFiller || line.Score is not { } score)
            {
                continue;
            }

            // **完全一致は記号ひとつ。** 1.00 が並ぶと、目が滑って
            // 「近いが同じではない」行を見落とす。
            var (text, brush) = score >= 1f
                ? ("=", dim)
                : ($"{score:0.00}", normal);

            var formatted = new FormattedText(
                text, System.Globalization.CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight, typeface, 11, brush);

            var y = visual.GetTextLineVisualYPosition(
                visual.TextLines[0], VisualYPosition.TextTop) - view.VerticalOffset;
            context.DrawText(formatted, new Point(Bounds.Width - formatted.Width - 6, y));
        }
    }
}
