using Avalonia.Controls;
using Avalonia.Input;
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
    }

    private void OnDoubleTapped(object? sender, TappedEventArgs e)
    {
        if (DataContext is FolderCompareViewModel model)
        {
            model.OpenSelected();
        }
    }
}
