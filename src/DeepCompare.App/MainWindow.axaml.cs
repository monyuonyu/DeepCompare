using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;

namespace DeepCompare.App;

public partial class MainWindow : Window
{
    private readonly ShellViewModel _shell;

    public MainWindow() : this([], false, false, false, false, false, false, "", "")
    {
    }

    public MainWindow(
        string[] startupPaths, bool structured = false, bool git = false,
        bool merge = false, bool version = false, bool snapshot = false,
        bool overUnder = false, string keys = "", string ignoredColumns = "")
    {
        InitializeComponent();
        _shell = new ShellViewModel(PickPathAsync, PickSavePathAsync)
        {
            DefaultOverUnder = overUnder,
        };
        DataContext = _shell;

        RestoreWindow();
        Closing += (_, _) => RememberWindow();

        AddHandler(DragDrop.DropEvent, OnDrop);
        AddHandler(DragDrop.DragOverEvent, OnDragOver);

        // 3 方向マージは 祖先 左 右 の 3 つ。
        if (merge && startupPaths.Length >= 3)
        {
            var (ancestor, ours, theirs) = (startupPaths[0], startupPaths[1], startupPaths[2]);
            Opened += (_, _) => _shell.ShowMerge(ancestor, ours, theirs);
            return;
        }

        // 写しは フォルダー 1 つ（写しを添えればそのまま比べる）。
        if (snapshot)
        {
            var folder = startupPaths.Length > 0 ? startupPaths[0] : string.Empty;
            var file = startupPaths.Length > 1 ? startupPaths[1] : string.Empty;
            Opened += (_, _) => _shell.ShowSnapshot(folder, file);
            return;
        }

        // Git は場所 1 つで開ける。引数が無ければ今いる場所。
        if (git)
        {
            var where = startupPaths.Length > 0 ? startupPaths[0] : Environment.CurrentDirectory;
            Opened += (_, _) => _shell.ShowGit(where);
            return;
        }

        // **1 つだけ渡されたら、それを開いて待つ。** 以前はホーム画面のままで、
        // 渡したものが読めているのかすら分からなかった。
        if (startupPaths.Length == 1 && !structured && !version)
        {
            var only = startupPaths[0];
            Opened += (_, _) =>
            {
                if (Directory.Exists(only) || DeepCompare.Engine.RemoteLocation.IsRemote(only))
                {
                    _shell.ShowFolders(only, string.Empty);
                }
                else
                {
                    _shell.ShowText(only, string.Empty);
                }
            };
            return;
        }

        // 旧実装と同じく、引数 2 つで比較対象を渡せる。
        // フォルダーが渡されたらフォルダー比較へ、ファイルならテキスト比較へ。
        if (startupPaths.Length >= 2)
        {
            var left = startupPaths[0];
            var right = startupPaths[1];
            Opened += (_, _) =>
            {
                if (structured)
                {
                    _shell.ShowStructured(left, right);
                }
                else if (version)
                {
                    _shell.ShowVersionInfo(left, right);
                }
                // **リモートはフォルダーとして扱う。** Directory.Exists は
                // リモートに対して必ず false なので、これが無いと
                // s3:// や ftp:// がテキスト比較で開かれる。
                else if ((Directory.Exists(left) || DeepCompare.Engine.RemoteLocation.IsRemote(left))
                         && (Directory.Exists(right) || DeepCompare.Engine.RemoteLocation.IsRemote(right)))
                {
                    _shell.ShowFolders(left, right);
                }
                // 両方が画像なら画像比較へ。**フラグを増やさない。** 画像を
                // 行で突き合わせたい場面はまず無いので、既定をこちらにしてよい
                // （テキストとして見たければ画面から切り替えられる）。
                else if (DeepCompare.Engine.ImageCompare.LooksLikeImage(left)
                         && DeepCompare.Engine.ImageCompare.LooksLikeImage(right))
                {
                    _shell.ShowImage(left, right);
                }
                // ノートブックはセル単位の画面へ。**画像と同じ考え方で、
                // フラグを増やさない。** .ipynb を行で突き合わせたい場面は
                // まず無い（実体は JSON で、出力の base64 が数千行動く）。
                else if (DeepCompare.Engine.Notebook.LooksLikeNotebook(left)
                         && DeepCompare.Engine.Notebook.LooksLikeNotebook(right))
                {
                    _shell.ShowNotebook(left, right);
                }
                // CSV / TSV は列単位の画面へ。行で突き合わせると、
                // 1 行挿入されただけで以降が全部ずれる。
                else if (LooksLikeTable(left) && LooksLikeTable(right))
                {
                    _shell.ShowTable(left, right, keys, ignoredColumns);
                }
                // Office は**本文を取り出して**テキスト比較へ。
                // 中身をそのまま行で比べても読めない（zip + XML）。
                else if (ShellViewModel.CanExtractText(left) && ShellViewModel.CanExtractText(right))
                {
                    _shell.ShowExtractedText(left, right);
                }
                else
                {
                    _shell.ShowText(left, right);
                }
            };
        }
    }

    private void OnMinimize(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        => WindowState = WindowState.Minimized;

    /// <summary>最大化と復元を行き来する。**帯の二度押しと同じ動き。**</summary>
    private void OnMaximize(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
        => WindowState = WindowState == WindowState.Maximized
            ? WindowState.Normal
            : WindowState.Maximized;

    private void OnClose(object? sender, Avalonia.Interactivity.RoutedEventArgs e) => Close();

    /// <summary>値が来たときだけ何かする観測者。</summary>
    private sealed class AnonymousObserver<T>(Action<T> onNext) : IObserver<T>
    {
        public void OnCompleted() { }

        public void OnError(Exception error) { }

        public void OnNext(T value) => onNext(value);
    }

    private readonly DeepCompare.Engine.SessionStore _windowStore =
        DeepCompare.Engine.SessionStore.Default;

    /// <summary>
    /// 前に閉じたときの大きさに戻す。
    ///
    /// **画面より大きくは戻さない。** 大きな画面で最大化したものを
    /// そのまま小さな画面で開くと、閉じるボタンが画面の外に出る。
    /// </summary>
    private void RestoreWindow()
    {
        var saved = _windowStore.LoadFile();
        if (saved.WindowMaximized)
        {
            WindowState = WindowState.Maximized;
            return;
        }
        if (saved.WindowWidth < MinWidth || saved.WindowHeight < MinHeight)
        {
            return;
        }

        var screen = Screens.Primary?.WorkingArea;
        var maxWidth = screen?.Width ?? int.MaxValue;
        var maxHeight = screen?.Height ?? int.MaxValue;
        Width = Math.Min(saved.WindowWidth, maxWidth);
        Height = Math.Min(saved.WindowHeight, maxHeight);
    }

    /// <summary>
    /// 閉じるときに覚える。
    ///
    /// **最大化中は Width/Height を使わない。** 最大化した窓の
    /// Width は画面いっぱいの値になり、戻したときの大きさが失われる。
    /// </summary>
    private void RememberWindow()
    {
        var maximized = WindowState == WindowState.Maximized;
        var width = maximized ? 0 : Width;
        var height = maximized ? 0 : Height;
        _windowStore.SaveWindow(width, height, maximized);
    }

    /// <summary>
    /// 表として開くべき名前か。
    ///
    /// **拡張子だけで決める。** 中身を読んで判定すると、大きなファイルで
    /// 起動が遅くなるうえ、区切り文字を含む普通のテキストを表と誤る。
    /// 違ったら画面から「テキストとして比べ直す」で戻れる。
    /// </summary>
    private static bool LooksLikeTable(string path)
    {
        var extension = Path.GetExtension(path).ToLowerInvariant();
        return extension is ".csv" or ".tsv";
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
        switch (_shell.Selected?.Content)
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
