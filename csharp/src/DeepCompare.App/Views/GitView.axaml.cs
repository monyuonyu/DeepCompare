using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Markup.Xaml;

namespace DeepCompare.App.Views;

public partial class GitView : UserControl
{
    public GitView()
    {
        AvaloniaXamlLoader.Load(this);

        // 書き込み先は表示側にしかない。ViewModel から画面に触らせず、
        // ここで差し込む（フォルダー比較の画面と同じ流儀）。
        DataContextChanged += (_, _) =>
        {
            if (DataContext is GitViewModel model)
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
}
