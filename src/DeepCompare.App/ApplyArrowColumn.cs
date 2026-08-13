using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using AvaloniaEdit.Rendering;

namespace DeepCompare.App;

/// <summary>
/// 差分の塊を左右へ写す矢印の列。
///
/// **行に貼り付いて動く。** エディタの行の位置を毎回見て、そこへ描く。
/// 別の入れ物として横に置くと、スクロールのたびにずれる。
///
/// 押す単位は**塊**。1 行ずつ写させると、5 行まとめて書き換えた場所で
/// 5 回押すことになる。
/// </summary>
public sealed class ApplyArrowColumn : Control
{
    private IReadOnlyList<AlignedLine> _lines = [];
    private TextView? _view;
    private int _hovered = -1;

    public ApplyArrowColumn()
    {
        Width = 22;
        Cursor = new Cursor(StandardCursorType.Hand);
    }

    /// <summary>押されたときに呼ぶ。引数は塊の番号。</summary>
    public Action<int>? Apply { get; set; }

    /// <summary>矢印の向き。true なら右へ写す（左のエディタの脇）。</summary>
    public bool ToRight { get; set; } = true;

    /// <summary>位置を測るのに使うエディタ。**行の位置はここから取る。**</summary>
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

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        var at = BlockAt(e.GetPosition(this).Y);
        if (at != _hovered)
        {
            _hovered = at;
            InvalidateVisual();
        }
    }

    protected override void OnPointerExited(PointerEventArgs e)
    {
        base.OnPointerExited(e);
        _hovered = -1;
        InvalidateVisual();
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (BlockAt(e.GetPosition(this).Y) is var block && block >= 0)
        {
            Apply?.Invoke(block);
            e.Handled = true;
        }
    }

    /// <summary>その高さにある塊。無ければ -1。</summary>
    private int BlockAt(double y)
    {
        if (_view is not { VisualLinesValid: true } view)
        {
            return -1;
        }
        foreach (var visual in view.VisualLines)
        {
            var number = visual.FirstDocumentLine.LineNumber;
            if (number < 1 || number > _lines.Count)
            {
                continue;
            }
            var top = visual.GetTextLineVisualYPosition(
                visual.TextLines[0], VisualYPosition.TextTop) - view.VerticalOffset;
            if (y >= top && y <= top + visual.Height)
            {
                return _lines[number - 1].BlockIndex;
            }
        }
        return -1;
    }

    public override void Render(DrawingContext context)
    {
        context.FillRectangle(Palette.Brush("PanelBg"), new Rect(0, 0, Bounds.Width, Bounds.Height));

        if (_view is not { VisualLinesValid: true } view)
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
            if (!line.IsBlockStart || line.BlockIndex < 0)
            {
                continue;
            }

            var top = visual.GetTextLineVisualYPosition(
                visual.TextLines[0], VisualYPosition.TextTop) - view.VerticalOffset;
            var middle = top + visual.Height / 2;

            var hot = line.BlockIndex == _hovered;
            var fill = hot ? Palette.Brush("ApplyArrowHover") : Palette.Brush("ApplyArrowFill");
            var mark = hot ? Palette.Brush("ApplyArrowOn") : Palette.Brush("ApplyArrow");

            // 枠。**押せる場所だと形で分かるようにする。**
            var box = new Rect(2, middle - 9, Bounds.Width - 4, 18);
            context.FillRectangle(fill, box, 3);
            context.DrawRectangle(
                null, new Pen(Palette.Brush("ApplyArrowBorder"), 1), box, 3, 3);

            // 矢印。中心を合わせる。
            var cx = Bounds.Width / 2;
            var arrow = new PathGeometry
            {
                Figures =
                [
                    new PathFigure
                    {
                        StartPoint = ToRight
                            ? new Point(cx - 5, middle - 2.5)
                            : new Point(cx + 5, middle - 2.5),
                        IsClosed = true,
                        IsFilled = true,
                        Segments = ToRight
                            ?
                            [
                                new LineSegment { Point = new Point(cx + 1, middle - 2.5) },
                                new LineSegment { Point = new Point(cx + 1, middle - 5) },
                                new LineSegment { Point = new Point(cx + 6, middle) },
                                new LineSegment { Point = new Point(cx + 1, middle + 5) },
                                new LineSegment { Point = new Point(cx + 1, middle + 2.5) },
                                new LineSegment { Point = new Point(cx - 5, middle + 2.5) },
                            ]
                            :
                            [
                                new LineSegment { Point = new Point(cx - 1, middle - 2.5) },
                                new LineSegment { Point = new Point(cx - 1, middle - 5) },
                                new LineSegment { Point = new Point(cx - 6, middle) },
                                new LineSegment { Point = new Point(cx - 1, middle + 5) },
                                new LineSegment { Point = new Point(cx - 1, middle + 2.5) },
                                new LineSegment { Point = new Point(cx + 5, middle + 2.5) },
                            ],
                    },
                ],
            };
            context.DrawGeometry(mark, null, arrow);
        }
    }
}
