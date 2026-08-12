using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;

namespace DeepCompare.App;

public partial class MainWindow : Window
{
    private readonly ShellViewModel _shell;

    public MainWindow() : this([])
    {
    }

    public MainWindow(string[] startupPaths)
    {
        InitializeComponent();
        _shell = new ShellViewModel(PickPathAsync, PickSavePathAsync);
        DataContext = _shell;

        AddHandler(DragDrop.DropEvent, OnDrop);
        AddHandler(DragDrop.DragOverEvent, OnDragOver);

        // 旧実装と同じく、引数 2 つで比較対象を渡せる。
        // フォルダーが渡されたらフォルダー比較へ、ファイルならテキスト比較へ。
        if (startupPaths.Length >= 2)
        {
            var left = startupPaths[0];
            var right = startupPaths[1];
            Opened += (_, _) =>
            {
                if (Directory.Exists(left) && Directory.Exists(right))
                {
                    _shell.ShowFolders(left, right);
                }
                else
                {
                    _shell.ShowText(left, right);
                }
            };
        }
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);

    // Avalonia 12 でドラッグ＆ドロップの API が変わっている。
    // 旧: e.Data.Contains(DataFormats.Files) / e.Data.GetFiles()
    // 新: e.DataTransfer.Contains(DataFormat.File) / e.DataTransfer.TryGetFiles()
    private static void OnDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = e.DataTransfer.Contains(DataFormat.File)
            ? DragDropEffects.Copy
            : DragDropEffects.None;
    }

    private void OnDrop(object? sender, DragEventArgs e)
    {
        var files = e.DataTransfer.TryGetFiles();
        if (files is null)
        {
            return;
        }
        var paths = files
            .Select(f => f.TryGetLocalPath())
            .Where(p => !string.IsNullOrEmpty(p))
            .Select(p => p!)
            .ToList();
        if (paths.Count == 0)
        {
            return;
        }

        // 落とし先はいま出ている画面。起動画面ならファイルでもフォルダーでも受ける。
        switch (_shell.Current)
        {
            case HomeViewModel home:
                home.AcceptDropped(paths);
                break;
            case TextCompareViewModel text:
                text.AcceptDroppedFiles(paths);
                break;
        }
    }

    /// <summary>書き出し先を選ばせる。既存ファイルの上書き確認は OS 側が出す。</summary>
    private async Task<string?> PickSavePathAsync(string title, string suggestedName)
    {
        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = title,
            SuggestedFileName = suggestedName,
        });
        return file?.TryGetLocalPath();
    }

    /// <summary>ファイルとフォルダーのどちらを選ばせるかは呼び出し側が決める。</summary>
    private async Task<string?> PickPathAsync(string title, bool isFolder)
    {
        if (isFolder)
        {
            var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
            {
                Title = title,
                AllowMultiple = false,
            });
            return folders.Count > 0 ? folders[0].TryGetLocalPath() : null;
        }

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
        });
        return files.Count > 0 ? files[0].TryGetLocalPath() : null;
    }
}
