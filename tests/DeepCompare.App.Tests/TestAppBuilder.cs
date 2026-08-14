using Avalonia;
using Avalonia.Headless;
using DeepCompare.App.Tests;

[assembly: AvaloniaTestApplication(typeof(TestAppBuilder))]

namespace DeepCompare.App.Tests;

/// <summary>
/// 仮想 platform 上で App を組み立てる。実画面は要らないが、
/// スタイルとフォントは本番と同じものを読ませる（スタイルの二重定義で
/// 起動しなくなる類の不具合は、ここを本番と揃えないと出てこない）。
/// </summary>
public static class TestAppBuilder
{
    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<global::DeepCompare.App.App>()
            .UseHeadless(new AvaloniaHeadlessPlatformOptions());
}
