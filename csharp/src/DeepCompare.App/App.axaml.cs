using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;

namespace DeepCompare.App;

public partial class App : Application
{
    /// <summary>起動引数で渡された比較対象。旧実装と同じく 2 つ受け取れる。</summary>
    public static string[] StartupFiles { get; set; } = [];

    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.MainWindow = new MainWindow(StartupFiles);
        }
        base.OnFrameworkInitializationCompleted();
    }
}
