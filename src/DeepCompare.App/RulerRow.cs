using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using AvaloniaEdit.Rendering;

namespace DeepCompare.App;

/// <summary>
/// 桁の目盛り（BC の Ruler）。
///
/// **固定長のデータで要る。** 「この項目は 10 桁目から 8 桁」と決まって
/// いる形式では、どの桁がずれたのかを数えられないと直せない。
/// 本文の上に貼り付き、横スクロールにも追従する。
/// </summary>
public sealed class RulerRow : Control
{
    private TextView? _view;

    public RulerRow()
    {
        Height = 14;
    }

    public void Attach(TextView view)
    {
        _view = view;
        view.ScrollOffsetChanged += (_, _) => InvalidateVisual();
        view.VisualLinesChanged += (_, _) => InvalidateVisual();
    }

    public override void Render(DrawingContext context)
    {
        if (_view is not { } view)
        {
            return;
        }

        context.FillRectangle(
            Palette.Brush("PanelBg"), new Rect(0, 0, Bounds.Width, Bounds.Height));

        // 1 文字の幅。**等幅の前提で測る。**
        var unit = view.WideSpaceWidth;
        if (unit <= 0)
        {
            return;
        }

        var typeface = new Typeface(FontFamily.Default);
        var dim = Palette.Brush("FgDim");
        var pen = new Pen(Palette.Brush("Divider"), 1);

        var first = (int)(view.HorizontalOffset / unit);
        var last = first + (int)(Bounds.Width / unit) + 2;

        for (var column = first; column <= last; column++)
        {
            if (column <= 0)
            {
                continue;
            }
            var x = column * unit - view.HorizontalOffset;

            // **10 桁ごとに数字、5 桁ごとに長い印、あとは短い印。**
            // 全部に数字を振ると読めない。
            if (column % 10 == 0)
            {
                var text = new FormattedText(
                    column.ToString(), System.Globalization.CultureInfo.InvariantCulture,
                    FlowDirection.LeftToRight, typeface, 9, dim);
                context.DrawText(text, new Point(x - text.Width / 2, 0));
            }
            else if (column % 5 == 0)
            {
                context.DrawLine(pen, new Point(x, 6), new Point(x, Bounds.Height));
            }
            else
            {
                context.DrawLine(pen, new Point(x, 10), new Point(x, Bounds.Height));
            }
        }
    }
}
