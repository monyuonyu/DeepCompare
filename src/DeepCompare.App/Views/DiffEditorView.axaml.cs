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

        // **縦だけ同期する。** 横は片方だけ長い行があるので、
        // まとめて動かすと読みたい側が動かせない。
        LeftPane.Editor.TextArea.TextView.ScrollOffsetChanged +=
            (_, _) => Sync(from: LeftPane, to: RightPane);
        RightPane.Editor.TextArea.TextView.ScrollOffsetChanged +=
            (_, _) => Sync(from: RightPane, to: LeftPane);
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
        bool leftReadOnly, bool rightReadOnly)
    {
        LeftPane.Fill(left, leftReadOnly);
        RightPane.Fill(right, rightReadOnly);
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
            var y = from.Editor.TextArea.TextView.VerticalOffset;
            if (Math.Abs(to.Editor.TextArea.TextView.VerticalOffset - y) > 0.5)
            {
                to.Editor.ScrollToVerticalOffset(y);
            }
        }
        finally
        {
            _syncing = false;
        }
    }
}
