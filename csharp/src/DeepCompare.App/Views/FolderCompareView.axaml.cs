using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
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
                model.Confirm = ConfirmAsync;
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

    /// <summary>
    /// 上書きの確認。
    ///
    /// **戻せない操作の前にだけ出す。** 何でも訊くと読まずに押すようになり、
    /// 本当に危ないときの歯止めにならない。
    /// </summary>
    private async Task<bool> ConfirmAsync(string message)
    {
        var owner = TopLevel.GetTopLevel(this) as Window;
        if (owner is null)
        {
            return false;
        }

        var answer = false;
        // **押す前に何が起きるか分かる文言にする。** 「はい」だけだと、
        // 読まずに押したときの被害が大きい。
        var label = message.Contains("消します") ? "消す" : "続ける";
        var yes = new Button { Content = label, Classes = { "accent" }, MinWidth = 90 };
        var no = new Button { Content = "やめる", MinWidth = 90 };

        var dialog = new Window
        {
            Title = "確認",
            SizeToContent = SizeToContent.WidthAndHeight,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            CanResize = false,
            Content = new StackPanel
            {
                Margin = new Thickness(20),
                Spacing = 16,
                Children =
                {
                    new TextBlock { Text = message, MaxWidth = 380, TextWrapping = TextWrapping.Wrap },
                    new StackPanel
                    {
                        Orientation = Avalonia.Layout.Orientation.Horizontal,
                        Spacing = 8,
                        HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Right,
                        Children = { no, yes },
                    },
                },
            },
        };

        yes.Click += (_, _) => { answer = true; dialog.Close(); };
        no.Click += (_, _) => dialog.Close();

        await dialog.ShowDialog(owner);
        return answer;
    }

    private void OnDoubleTapped(object? sender, TappedEventArgs e)
    {
        if (DataContext is FolderCompareViewModel model)
        {
            model.OpenSelected();
        }
    }
}
