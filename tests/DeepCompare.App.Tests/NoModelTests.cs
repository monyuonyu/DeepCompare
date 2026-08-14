using Avalonia.Headless.XUnit;
using DeepCompare.Engine;
using Xunit;

namespace DeepCompare.App.Tests;

/// <summary>
/// **モデルを配布物に含めないので、無い状態が普通。**
/// そのとき Myers の答えを捨てて空にしてはいけない
/// （直前まではそうなっていて、初回起動が壊れて見えた）。
///
/// 環境変数で「無い場所」を指して再現する。
/// </summary>
public class NoModelTests : IDisposable
{
    private readonly string? _saved;

    public NoModelTests()
    {
        _saved = Environment.GetEnvironmentVariable(Embedder.ModelEnvironmentVariable);
        Environment.SetEnvironmentVariable(
            Embedder.ModelEnvironmentVariable,
            Path.Combine(Path.GetTempPath(), $"no-model-{Guid.NewGuid():N}.dcm"));
    }

    public void Dispose()
        => Environment.SetEnvironmentVariable(Embedder.ModelEnvironmentVariable, _saved);

    [Fact]
    public void モデルが無ければnullを返す()
        => Assert.Null(Embedder.CreateFromDefaultAssetsOrNull());

    /// <summary>
    /// **名指しで無い物を指されたら投げる。** 「このモデルで比べろ」と
    /// 言われて黙って別のやり方の答えを返すのは、間違った結果を黙って
    /// 出すのと変わらない。
    /// </summary>
    [Fact]
    public void 名指しされたモデルが無ければ投げる()
        => Assert.Throws<FileNotFoundException>(
            () => Embedder.CreateFromDefaultAssetsOrNull("居ない.dcm"));

    [AvaloniaFact]
    public async Task モデルが無くても比較の結果は残る()
    {
        using var left = new TempFile("あ\nい\nう\n");
        using var right = new TempFile("あ\nZ\nう\n");
        var shell = TestShell.Create();
        shell.FastMode = false;   // 段階 2 まで進ませる

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        // **ここが肝。** 以前はモデルが無いと例外が出て、
        // VisibleRows を空にし Placeholder をエラー文言に差し替えていた。
        Assert.NotEmpty(model.VisibleRows);
        Assert.True(model.HasDifferences);
        Assert.False(model.IsBusy);
    }

    [AvaloniaFact]
    public async Task モデルが無いことを黙らない()
    {
        using var left = new TempFile("あ\nい\nう\n");
        using var right = new TempFile("あ\nZ\nう\n");
        var shell = TestShell.Create();
        shell.FastMode = false;

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        Assert.True(model.HasModelWarning);
        Assert.Contains("モデル", model.ModelWarning);
    }

    /// <summary>
    /// 反映のたびに再比較が走る。**そこでも結果を消さない。**
    /// </summary>
    [AvaloniaFact]
    public async Task モデルが無くても塊を写せる()
    {
        using var left = new TempFile("あ\nい\nう\n");
        using var right = new TempFile("あ\nZ\nう\n");
        var shell = TestShell.Create();
        shell.FastMode = false;

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();
        var block = model.VisibleRows.First(row => row.IsBlockStart);
        await model.ApplyBlockAsync(block, toRight: true);

        Assert.NotEmpty(model.VisibleRows);
        Assert.False(model.HasDifferences);
        Assert.True(model.RightModified);
    }
}
