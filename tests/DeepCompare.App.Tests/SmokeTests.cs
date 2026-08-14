using Avalonia.Headless.XUnit;
using Xunit;

namespace DeepCompare.App.Tests;

/// <summary>
/// 土台が立っているかだけを見る。ここが通らなければ他は全部無意味。
/// </summary>
public class SmokeTests
{
    [AvaloniaFact]
    public void 仮想画面でウィンドウが開く()
    {
        var window = new MainWindow([], false, false, false, false, false, false, "", "");
        window.Show();

        Assert.True(window.IsVisible);
    }
}
