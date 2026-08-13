using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Markup.Xaml;

namespace DeepCompare.App.Views;

public partial class FolderCompareView : UserControl
{
    public FolderCompareView()
    {
        AvaloniaXamlLoader.Load(this);
        // 一覧の行を開いてテキスト比較へ移る。ダブルクリックはこの手の一覧での
        // 標準的な操作なので、ボタンを増やさずここで拾う。
        AddHandler(DoubleTappedEvent, OnDoubleTapped, handledEventsToo: true);

        // 書き込み先は表示側にしかない。ViewModel から画面に触らせず、
        // ここで差し込む。
        DataContextChanged += (_, _) =>
        {
            if (DataContext is FolderCompareViewModel model)
            {
                model.Clipboard = text =>
                {
                    var clipboard = TopLevel.GetTopLevel(this)?.Clipboard;
                    if (clipboard is not null)
                    {
                        // Avalonia 12 で API が変わった。DataFormat で種類を明示する。
                        _ = clipboard.SetValueAsync(DataFormat.Text, text);
                    }
                };
            }
        };
    }

    private void OnDoubleTapped(object? sender, TappedEventArgs e)
    {
        if (DataContext is FolderCompareViewModel model)
        {
            model.OpenSelected();
        }
    }
}
