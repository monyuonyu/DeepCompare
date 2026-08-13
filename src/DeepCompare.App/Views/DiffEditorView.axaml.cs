using Avalonia;
using Avalonia.Controls;
using Avalonia.VisualTree;
using Avalonia.Markup.Xaml;

namespace DeepCompare.App.Views;

/// <summary>
/// 左右のエディタを並べ、縦だけ同期させる。
/// </summary>
public partial class DiffEditorView : UserControl
{
    private bool _syncing;

    public DiffEditorView()
    {
        // **生成される初期化を呼ぶ。** AvaloniaXamlLoader.Load を直に
        // 呼ぶと x:Name のフィールドが埋まらず、直後の組み立てで落ちる。
        InitializeComponent();

        // **中のスクロール領域を直に掴む。**
        // TextView の ScrollOffsetChanged では追従しなかった（実機で
        // 左が 1 行目・右が 20 行目のまま動かない、という形で出た）。
        // 部品が内側に持っている ScrollViewer の Offset を、
        // そのまま相手へ写す。
        AttachedToVisualTree += (_, _) => HookScroll();

        // 初めの配置を組む。**XAML の列定義と食い違わないよう、
        // ここで一度作り直す。**
        ApplyLayout();

        // 近さは右のペインだけに出す。
        LeftPane.ShowScores = false;
        RightPane.ShowScores = true;

        // 写しの矢印。左のものは右へ、右のものは左へ写す。
        LeftPane.ArrowColumn.ToRight = true;
        LeftPane.ArrowColumn.Apply = block => ApplyBlock?.Invoke(block, true);
        RightPane.ArrowColumn.ToRight = false;
        RightPane.ArrowColumn.Apply = block => ApplyBlock?.Invoke(block, false);
    }

    /// <summary>塊を写す。引数は（塊の番号, 右へ写すか）。</summary>
    public Action<int, bool>? ApplyBlock { get; set; }

    /// <summary>いる行が変わった（揃えた本文での行番号、0 から）。</summary>
    public event EventHandler<int>? CaretLineChanged
    {
        add { LeftPane.CaretLineChanged += value; RightPane.CaretLineChanged += value; }
        remove { LeftPane.CaretLineChanged -= value; RightPane.CaretLineChanged -= value; }
    }

    /// <summary>見えている範囲が変わった。**左を代表にする**（縦は同期している）。</summary>
    public event EventHandler<(double Start, double Size)>? ViewportChanged
    {
        add => LeftPane.ViewportChanged += value;
        remove => LeftPane.ViewportChanged -= value;
    }

    /// <summary>地図から飛ばす。左右そろって動く。</summary>
    public void ScrollToFraction(double fraction)
    {
        LeftPane.ScrollToFraction(fraction);
        RightPane.ScrollToFraction(fraction);
    }

    private bool _overUnder;

    /// <summary>
    /// 上下に並べるか（BC の Over-Under Layout）。
    ///
    /// **横に長い本文を読むときはこちら。** 左右に割ると 1 行あたりの
    /// 幅が半分になり、長い行はどちらも切れる。
    /// </summary>
    public bool OverUnder
    {
        get => _overUnder;
        set
        {
            if (_overUnder == value)
            {
                return;
            }
            _overUnder = value;
            ApplyLayout();
        }
    }

    private void ApplyLayout()
    {
        Layout.ColumnDefinitions.Clear();
        Layout.RowDefinitions.Clear();

        if (_overUnder)
        {
            Layout.RowDefinitions.Add(new RowDefinition(1, GridUnitType.Star) { MinHeight = 60 });
            Layout.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
            Layout.RowDefinitions.Add(new RowDefinition(1, GridUnitType.Star) { MinHeight = 60 });

            Set(LeftPane, row: 0, column: 0);
            Set(Divider, row: 1, column: 0);
            Set(RightPane, row: 2, column: 0);

            Divider.Height = 4;
            Divider.Width = double.NaN;
            Divider.ResizeDirection = GridResizeDirection.Rows;
        }
        else
        {
            Layout.ColumnDefinitions.Add(
                new ColumnDefinition(1, GridUnitType.Star) { MinWidth = 120 });
            Layout.ColumnDefinitions.Add(new ColumnDefinition(GridLength.Auto));
            Layout.ColumnDefinitions.Add(
                new ColumnDefinition(1, GridUnitType.Star) { MinWidth = 120 });

            Set(LeftPane, row: 0, column: 0);
            Set(Divider, row: 0, column: 1);
            Set(RightPane, row: 0, column: 2);

            Divider.Width = 4;
            Divider.Height = double.NaN;
            Divider.ResizeDirection = GridResizeDirection.Columns;
        }

        static void Set(Control control, int row, int column)
        {
            Grid.SetRow(control, row);
            Grid.SetColumn(control, column);
        }
    }

    /// <summary>
    /// 長い行を折り返すか。**左右そろえる。**
    /// 片方だけ折り返すと、同じ行が上下にずれて並ぶ。
    /// </summary>
    public bool WordWrap
    {
        get => LeftPane.WordWrap;
        set
        {
            LeftPane.WordWrap = value;
            RightPane.WordWrap = value;
        }
    }

    /// <summary>一致行を畳む／開く。左右そろって。</summary>
    public void SetFolded(bool folded)
    {
        LeftPane.SetFolded(folded);
        RightPane.SetFolded(folded);
    }

    /// <summary>その行へ移す。</summary>
    public void GoToLine(int index)
    {
        LeftPane.GoToLine(index);
        RightPane.GoToLine(index);
    }

    /// <summary>打たれたことを外へ伝える。左右どちらかは呼ぶ側が見る。</summary>
    public event EventHandler? LeftChanged
    {
        add => LeftPane.Changed += value;
        remove => LeftPane.Changed -= value;
    }

    public event EventHandler? RightChanged
    {
        add => RightPane.Changed += value;
        remove => RightPane.Changed -= value;
    }

    public void Fill(
        AlignedDocument left, AlignedDocument right,
        bool leftReadOnly, bool rightReadOnly,
        DeepCompare.Engine.Language? language = null)
    {
        LeftPane.Fill(left, leftReadOnly, language);
        RightPane.Fill(right, rightReadOnly, language);

        // **入れ直した直後に揃える。** 片方だけ位置が残っていると、
        // 開いた瞬間から左右が別の場所を向いている。
        Mirror(_leftScroll, _rightScroll);
    }

    public IReadOnlyList<string> LeftLines() => LeftPane.CurrentLines();
    public IReadOnlyList<string> RightLines() => RightPane.CurrentLines();

    private ScrollViewer? _leftScroll;
    private ScrollViewer? _rightScroll;

    /// <summary>
    /// 中のスクロール領域を探して繋ぐ。
    ///
    /// **見つかるまで探し続ける。** 部品の中身は木に付いた後に組まれる。
    /// </summary>
    private void HookScroll()
    {
        if (_leftScroll is not null && _rightScroll is not null)
        {
            return;
        }

        _leftScroll ??= LeftPane.Editor.GetVisualDescendants()
            .OfType<ScrollViewer>().FirstOrDefault();
        _rightScroll ??= RightPane.Editor.GetVisualDescendants()
            .OfType<ScrollViewer>().FirstOrDefault();

        if (_leftScroll is null || _rightScroll is null)
        {
            // まだ組まれていない。次の描画でもう一度。
            Avalonia.Threading.Dispatcher.UIThread.Post(HookScroll);
            return;
        }

        _leftScroll.ScrollChanged += (_, _) => Mirror(_leftScroll, _rightScroll);
        _rightScroll.ScrollChanged += (_, _) => Mirror(_rightScroll, _leftScroll);
    }

    /// <summary>片方の位置をもう片方へ写す。縦も横も。</summary>
    private void Mirror(ScrollViewer? from, ScrollViewer? to)
    {
        if (_syncing || from is null || to is null)
        {
            return;
        }
        _syncing = true;
        try
        {
            var wanted = from.Offset;
            if (Math.Abs(to.Offset.Y - wanted.Y) > 0.5
                || Math.Abs(to.Offset.X - wanted.X) > 0.5)
            {
                to.Offset = new Vector(wanted.X, wanted.Y);
            }
        }
        finally
        {
            _syncing = false;
        }
    }

    private void Sync(DiffEditorPane from, DiffEditorPane to)
    {
        // **戻ってこないようにする。** 片方を動かすと相手が動き、
        // その相手がまた自分を動かす。
        if (_syncing)
        {
            return;
        }
        _syncing = true;
        try
        {
            var source = from.Editor.TextArea.TextView;
            var target = to.Editor.TextArea.TextView;

            if (Math.Abs(target.VerticalOffset - source.VerticalOffset) > 0.5)
            {
                to.Editor.ScrollToVerticalOffset(source.VerticalOffset);
            }
            if (Math.Abs(target.HorizontalOffset - source.HorizontalOffset) > 0.5)
            {
                to.Editor.ScrollToHorizontalOffset(source.HorizontalOffset);
            }
        }
        finally
        {
            _syncing = false;
        }
    }
}
