using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;

namespace DeepCompare.App;

public partial class App : Application
{
    /// <summary>起動引数で渡された比較対象。旧実装と同じく 2 つ受け取れる。</summary>
    public static string[] StartupFiles { get; set; } = [];

    /// <summary>起動と同時に構造比較を開くか（<c>--structured</c>）。</summary>
    public static bool StartStructured { get; set; }

    /// <summary>起動と同時に Git の画面を開くか（<c>--git</c>）。</summary>
    public static bool StartGit { get; set; }

    /// <summary>起動と同時に 3 方向マージの画面を開くか（<c>--merge-view</c>）。</summary>
    public static bool StartMerge { get; set; }

    /// <summary>版の比較を出して起動する。</summary>
    public static bool StartVersion { get; set; }

    /// <summary>写しの画面を出して起動する。</summary>
    public static bool StartSnapshot { get; set; }

    /// <summary>上下に並べた状態で開く（`--over-under`）。</summary>
    public static bool StartOverUnder { get; set; }

    /// <summary>表として開くときのキー列と見ない列（`--key` / `--ignore-column`）。</summary>
    public static string StartKeys { get; set; } = string.Empty;
    public static string StartIgnoredColumns { get; set; } = string.Empty;

    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.MainWindow = new MainWindow(
                StartupFiles, StartStructured, StartGit, StartMerge, StartVersion,
                StartSnapshot, StartOverUnder, StartKeys, StartIgnoredColumns);
        }
        base.OnFrameworkInitializationCompleted();
    }
}
