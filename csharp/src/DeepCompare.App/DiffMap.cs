using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;

namespace DeepCompare.App;

/// <summary>
/// 差分の地図。ファイル全体のどこに差分があるかを 1 本の帯で示し、押すとそこへ飛ぶ。
///
/// **これが無いと「全体像」が分からない。** 差分ビューは画面に入る 40 行ほどしか
/// 見せないので、2000 行のファイルでは今どのあたりに居るのか、残りに差分がどれだけ
/// あるのかが掴めない。Beyond Compare も同じ位置に同じものを置いている。
///
/// 行ごとに図形を作らず、<see cref="Render"/> で直に描く。数千行あるので、
/// 図形を並べると作るだけで重くなる。
/// </summary>
public sealed class DiffMap : Control
{
    public static readonly StyledProperty<System.Collections.IEnumerable?> RowsProperty =
        AvaloniaProperty.Register<DiffMap, System.Collections.IEnumerable?>(nameof(Rows));

    /// <summary>いま画面に映っている行の範囲（先頭の位置と割合）。四角で示す。</summary>
    public static readonly StyledProperty<double> ViewStartProperty =
        AvaloniaProperty.Register<DiffMap, double>(nameof(ViewStart));

    public static readonly StyledProperty<double> ViewSizeProperty =
        AvaloniaProperty.Register<DiffMap, double>(nameof(ViewSize), 1.0);

    /// <summary>押された位置（0〜1）。呼ぶ側がそこへスクロールする。</summary>
    public static readonly StyledProperty<double> ClickedPositionProperty =
        AvaloniaProperty.Register<DiffMap, double>(
            nameof(ClickedPosition), defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);

    static DiffMap()
    {
        AffectsRender<DiffMap>(RowsProperty, ViewStartProperty, ViewSizeProperty);
    }

    public System.Collections.IEnumerable? Rows
    {
        get => GetValue(RowsProperty);
        set => SetValue(RowsProperty, value);
    }

    public double ViewStart
    {
        get => GetValue(ViewStartProperty);
        set => SetValue(ViewStartProperty, value);
    }

    public double ViewSize
    {
        get => GetValue(ViewSizeProperty);
        set => SetValue(ViewSizeProperty, value);
    }

    public double ClickedPosition
    {
        get => GetValue(ClickedPositionProperty);
        set => SetValue(ClickedPositionProperty, value);
    }

    public DiffMap()
    {
        // 細いと 1 行の差分が線にしか見えず、位置は分かっても量が分からない。
        Width = 26;
        Cursor = new Cursor(StandardCursorType.Hand);
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        Move(e.GetPosition(this).Y);
        e.Pointer.Capture(this);
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        // 押したまま動かすと、なぞった先へ追いかける。スクロールバーと同じ感覚にする。
        if (ReferenceEquals(e.Pointer.Captured, this))
        {
            Move(e.GetPosition(this).Y);
        }
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        e.Pointer.Capture(null);
    }

    private void Move(double y)
    {
        if (Bounds.Height <= 0)
        {
            return;
        }
        var position = Math.Clamp(y / Bounds.Height, 0, 1);

        // 同じ値を入れても知らせが飛ばないので、必ず変わるようにしてから戻す。
        // なぞって同じ位置に戻ったときに反応しないのを避ける。
        ClickedPosition = -1;
        ClickedPosition = position;
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);

        var height = Bounds.Height;
        var width = Bounds.Width;
        if (height <= 0 || width <= 0)
        {
            return;
        }

        context.FillRectangle(Palette.Brush("PanelBg"), new Rect(0, 0, width, height));

        if (Rows is not System.Collections.ICollection collection || collection.Count == 0)
        {
            return;
        }

        var count = collection.Count;
        var perRow = height / count;

        // **1 行が 1px を切っても、差分は必ず見えるようにする。**
        // 高さそのままで描くと、5000 行のファイルで 1 行の差分が消える。
        var minimum = 2.0;

        var index = 0;
        foreach (var item in collection)
        {
            if (item is RowView row)
            {
                var brush = Kind(row);
                if (brush is not null)
                {
                    var y = index * perRow;
                    var h = Math.Max(minimum, perRow);
                    // はみ出さないように下端で止める。
                    if (y + h > height)
                    {
                        y = Math.Max(0, height - h);
                    }
                    context.FillRectangle(brush, new Rect(3, y, width - 6, h));
                }
            }
            index++;
        }

        // いま見えている範囲。
        //
        // **薄く塗ったうえで枠を描く。** 枠だけだと、差分の色が濃い場所では
        // 線が紛れて見つけられない。塗りだけだと下の色が読めなくなる。
        var viewY = Math.Clamp(ViewStart, 0, 1) * height;
        var viewH = Math.Max(10, Math.Clamp(ViewSize, 0, 1) * height);
        var top = Math.Min(viewY, Math.Max(0, height - viewH));
        var area = new Rect(0, top, width, viewH);

        context.FillRectangle(new SolidColorBrush(Colors.Gray, 0.25), area);

        // 枠はアクセント色。**差分の緑や赤の上でも見分けられる色**でないと、
        // 一番濃い場所で見失う。上下の辺は太くして、範囲の切れ目を分かりやすく。
        var accent = Palette.Brush("Accent");
        context.DrawRectangle(null, new Pen(accent, 1.5),
            new Rect(0.75, top + 0.75, width - 1.5, viewH - 1.5));
        context.DrawLine(new Pen(accent, 2.5), new Point(0, top + 1), new Point(width, top + 1));
        context.DrawLine(new Pen(accent, 2.5),
            new Point(0, top + viewH - 1), new Point(width, top + viewH - 1));
    }

    /// <summary>その行を地図に出すか、出すなら何色か。一致した行は出さない。</summary>
    private static IBrush? Kind(RowView row) => (row.Row.Left, row.Row.Right) switch
    {
        (not null, not null) when row.Row.IsUnchanged => null,
        (not null, not null) => Palette.Brush("StatusDifferent"),
        (not null, null) => Palette.Brush("StatusLeftOnly"),
        (null, not null) => Palette.Brush("StatusRightOnly"),
        _ => null,
    };
}
