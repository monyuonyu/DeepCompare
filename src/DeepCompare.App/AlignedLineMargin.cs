using Avalonia;
using Avalonia.Media;
using AvaloniaEdit.Editing;
using AvaloniaEdit.Rendering;

namespace DeepCompare.App;

/// <summary>
/// 行番号の余白。
///
/// **エディタの既定の番号は使えない。** 揃えた本文には詰め物の空行が
/// 入っているので、そのまま数えると元のファイルの行番号とずれる。
/// ここでは素性（<see cref="AlignedLine.SourceLine"/>）の番号を描き、
/// 詰め物には何も描かない。
/// </summary>
public sealed class AlignedLineMargin : AbstractMargin
{
    private IReadOnlyList<AlignedLine> _lines = [];
    private int _widest = 3;

    public void Update(IReadOnlyList<AlignedLine> lines)
    {
        _lines = lines;
        // **一番大きい番号で幅を決める。** 行が増えるたびに幅が変わると、
        // 本文の左端が動いて読みづらい。
        var max = 0;
        foreach (var line in lines)
        {
            if (line.SourceLine is { } at && at + 1 > max)
            {
                max = at + 1;
            }
        }
        _widest = Math.Max(3, max.ToString().Length);
        InvalidateMeasure();
        InvalidateVisual();
    }

    protected override Size MeasureOverride(Size availableSize)
    {
        var typeface = new Typeface(FontFamily.Default);
        var sample = new FormattedText(
            new string('9', _widest), System.Globalization.CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight, typeface, 11, Brushes.Black);
        return new Size(sample.Width + 12, 0);
    }

    public override void Render(DrawingContext context)
    {
        if (TextView is not { VisualLinesValid: true } view)
        {
            return;
        }

        var brush = Palette.Brush("FgDim");
        var typeface = new Typeface(FontFamily.Default);

        foreach (var visual in view.VisualLines)
        {
            var number = visual.FirstDocumentLine.LineNumber;
            if (number < 1 || number > _lines.Count)
            {
                continue;
            }
            if (_lines[number - 1].SourceLine is not { } source)
            {
                // 詰め物には番号が無い。**空欄のままにする** —
                // ここに何か描くと、元のファイルに行があるように見える。
                continue;
            }

            var text = new FormattedText(
                (source + 1).ToString(), System.Globalization.CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight, typeface, 11, brush);

            var y = visual.GetTextLineVisualYPosition(
                visual.TextLines[0], VisualYPosition.TextTop) - view.VerticalOffset;
            context.DrawText(text, new Point(Bounds.Width - text.Width - 6, y));
        }
    }
}
