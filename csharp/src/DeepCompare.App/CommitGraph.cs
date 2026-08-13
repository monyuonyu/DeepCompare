using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// 履歴 1 行分のグラフを描く。
///
/// **行ごとに部品を作らず、1 行 1 つの描画で済ませる。** 履歴は数百行あるので、
/// 線や丸を要素として並べると、作るだけで重くなる（差分の地図と同じ理由）。
///
/// SourceTree の Graph 列に当たる。
/// </summary>
public sealed class CommitGraph : Control
{
    /// <summary>この行のグラフ。</summary>
    public static readonly StyledProperty<GraphRow?> RowProperty =
        AvaloniaProperty.Register<CommitGraph, GraphRow?>(nameof(Row));

    /// <summary>次の行（1 つ古い側）のグラフ。線を繋ぐのに要る。</summary>
    public static readonly StyledProperty<GraphRow?> NextProperty =
        AvaloniaProperty.Register<CommitGraph, GraphRow?>(nameof(Next));

    static CommitGraph()
    {
        AffectsRender<CommitGraph>(RowProperty, NextProperty);
        AffectsMeasure<CommitGraph>(RowProperty);
    }

    public GraphRow? Row
    {
        get => GetValue(RowProperty);
        set => SetValue(RowProperty, value);
    }

    public GraphRow? Next
    {
        get => GetValue(NextProperty);
        set => SetValue(NextProperty, value);
    }

    /// <summary>列の間隔。狭いと線が重なり、広いと説明の場所が減る。</summary>
    private const double LaneWidth = 14;
    private const double Radius = 3.5;

    /// <summary>
    /// 枝ごとに色を変える。**同じ枝はどの行でも同じ色**になるので、目で追える。
    ///
    /// 差分の色（緑・赤・青）とは別の並びにする。同じ色を別の意味で使うと、
    /// 画面全体で「この色は何か」が決まらなくなる。
    /// </summary>
    private static readonly string[] LaneColours =
    [
        "#5B8DEF", "#E0A030", "#6FB86F", "#C169C1", "#4FB3C9", "#D9736B",
        "#8E86D9", "#B08A4A", "#5FA88C", "#C77FA8", "#7A97C4", "#A0A0A0",
    ];

    private static IBrush Colour(int lane)
        => new SolidColorBrush(Color.Parse(LaneColours[Math.Abs(lane) % LaneColours.Length]));

    protected override Size MeasureOverride(Size availableSize)
        => new(Math.Max(1, (Row?.Width ?? 1)) * LaneWidth, 0);

    public override void Render(DrawingContext context)
    {
        base.Render(context);

        if (Row is not { } row)
        {
            return;
        }

        var height = Bounds.Height;
        if (height <= 0)
        {
            return;
        }

        var middle = height / 2;

        // 素通りする線。**上から下まで通す。**
        foreach (var lane in row.Passing)
        {
            var x = X(lane);
            context.DrawLine(new Pen(Colour(lane), 1.6),
                new Point(x, 0), new Point(x, height));
        }

        // 自分の列の上半分（1 つ新しい側から降りてくる線）。
        // 先頭の行にも引く。**そこで切れていると、上に続きがあることが伝わらない。**
        var own = X(row.Lane);
        context.DrawLine(new Pen(Colour(row.Lane), 1.6),
            new Point(own, 0), new Point(own, middle));

        // 親へ伸びる線。列が変わるなら、真ん中で曲げて次の行へ渡す。
        foreach (var edge in row.Edges)
        {
            var to = X(edge.ToLane);
            var brush = Colour(edge.ToLane);

            if (Math.Abs(to - own) < 0.5)
            {
                context.DrawLine(new Pen(brush, 1.6),
                    new Point(own, middle), new Point(own, height));
                continue;
            }

            // 斜めではなく、いったん横へ出してから下ろす。**規則が単純な方が
            // 目で追える**（斜め線は交差したときにどれがどれか分からなくなる）。
            var figure = new PathFigure
            {
                StartPoint = new Point(own, middle),
                IsClosed = false,
                Segments =
                [
                    new QuadraticBezierSegment
                    {
                        Point1 = new Point(to, middle),
                        Point2 = new Point(to, middle + (height - middle) / 2),
                    },
                    new LineSegment { Point = new Point(to, height) },
                ],
            };
            context.DrawGeometry(null, new Pen(brush, 1.6),
                new PathGeometry { Figures = [figure] });
        }

        // 丸。**マージは大きく描く。** 合流点が目に留まると、履歴の形が読める。
        var radius = row.IsMerge ? Radius + 1.5 : Radius;
        context.DrawEllipse(
            Palette.Brush("WindowBg"), new Pen(Colour(row.Lane), 2),
            new Point(own, middle), radius, radius);
    }

    private static double X(int lane) => lane * LaneWidth + LaneWidth / 2;
}
