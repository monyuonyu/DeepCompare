using Avalonia.Controls;
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

        // **縦も横も同じだけ動かす。**
        // 同じ行を見比べるのが目的なので、縦が揃っていないと意味が無い。
        // 横も、片方だけ動くと同じ桁を見比べられない（Beyond Compare も
        // 両方そろえる）。
        //
        // **スクロールバーの位置で合わせる。** TextView の ScrollOffset は
        // 読むだけの値で、代入しても動かない。
        LeftPane.Editor.TextArea.TextView.ScrollOffsetChanged +=
            (_, _) => Sync(from: LeftPane, to: RightPane);
        RightPane.Editor.TextArea.TextView.ScrollOffsetChanged +=
            (_, _) => Sync(from: RightPane, to: LeftPane);

        // 写しの矢印。**位置は本文から測る。**
        _leftArrows.ToRight = true;
        _leftArrows.Attach(LeftPane.Editor.TextArea.TextView);
        _leftArrows.Apply = block => ApplyBlock?.Invoke(block, true);
        LeftArrows.Content = _leftArrows;

        _rightArrows.ToRight = false;
        _rightArrows.Attach(RightPane.Editor.TextArea.TextView);
        _rightArrows.Apply = block => ApplyBlock?.Invoke(block, false);
        RightArrows.Content = _rightArrows;
    }

    private readonly ApplyArrowColumn _leftArrows = new();
    private readonly ApplyArrowColumn _rightArrows = new();

    /// <summary>塊を写す。引数は（塊の番号, 右へ写すか）。</summary>
    public Action<int, bool>? ApplyBlock { get; set; }

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
        bool leftReadOnly, bool rightReadOnly)
    {
        LeftPane.Fill(left, leftReadOnly);
        RightPane.Fill(right, rightReadOnly);
        _leftArrows.Update(left.Lines);
        _rightArrows.Update(right.Lines);

        // **入れ直した直後に揃える。** 片方だけ位置が残っていると、
        // 開いた瞬間から左右が別の場所を向いている。
        Sync(from: LeftPane, to: RightPane);
    }

    public IReadOnlyList<string> LeftLines() => LeftPane.CurrentLines();
    public IReadOnlyList<string> RightLines() => RightPane.CurrentLines();

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
