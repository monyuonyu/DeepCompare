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
[Collection("モデルの置き場所")]
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

/// <summary>
/// **落として置いた物が、名前が違うだけで無視されないこと。**
///
/// 同梱をやめたので、利用者が置くモデルの名前は既定（minilm.dcm）とは
/// 限らない。実行ファイルの隣を直に触る挙動なので、環境変数ではなく
/// 出力先そのものにファイルを置いて確かめる。
/// </summary>
[Collection("モデルの置き場所")]
public class ModelDiscoveryTests : IDisposable
{
    private readonly string _placed;
    private readonly string? _savedEnvironment;

    public ModelDiscoveryTests()
    {
        // 既定名の物が在ると分岐に入らないので、環境変数は空にしておく。
        _savedEnvironment = Environment.GetEnvironmentVariable(Embedder.ModelEnvironmentVariable);
        Environment.SetEnvironmentVariable(Embedder.ModelEnvironmentVariable, null);

        _placed = Path.Combine(AppContext.BaseDirectory, "zz-置いただけ.dcm");
        File.WriteAllBytes(_placed, [0x00]);
    }

    public void Dispose()
    {
        File.Delete(_placed);
        Environment.SetEnvironmentVariable(Embedder.ModelEnvironmentVariable, _savedEnvironment);
    }

    [Fact]
    public void 隣に置いたモデルが一覧に出る()
        => Assert.Contains("zz-置いただけ.dcm", Embedder.AvailableModels());

    /// <summary>
    /// 既定名の物が在る間は、そちらが優先される（従来どおり）。
    /// **名前順で先頭を拾う実装なので、ここが崩れると既存の置き方が壊れる。**
    /// </summary>
    [Fact]
    public void 既定の名前が在ればそちらを使う()
    {
        var beside = Path.Combine(AppContext.BaseDirectory, Embedder.DefaultWeightsFileName);
        if (!File.Exists(beside))
        {
            return;   // 開発の出力先に既定モデルが無い環境では確かめようがない
        }
        Assert.Equal(beside, Embedder.ResolveModelPath());
    }

    /// <summary>
    /// 隣に置いた物が、既定の名前でなくても選ばれること。
    /// **既定名の物を一時的に退けて確かめる** — そうしないと
    /// 「既定が在るから既定が選ばれた」だけを見ることになる。
    /// </summary>
    [Fact]
    public void 既定の名前が無ければ隣の物を使う()
    {
        var beside = Path.Combine(AppContext.BaseDirectory, Embedder.DefaultWeightsFileName);
        var hidden = beside + ".hidden";
        var moved = File.Exists(beside);
        if (moved)
        {
            File.Move(beside, hidden);
        }
        try
        {
            // **どれが選ばれるかは、隣に何が在るかで変わる**（名前順の先頭）。
            // 出力先に何が置かれても崩れないよう、確かめるのは 2 点だけ:
            // 在る物を指すこと（以前は無い既定名を返していた）と、
            // 退けた既定名ではないこと。
            var resolved = Embedder.ResolveModelPath();
            Assert.True(File.Exists(resolved), $"{resolved} が無い");
            Assert.NotEqual(
                Embedder.DefaultWeightsFileName, Path.GetFileName(resolved));
        }
        finally
        {
            if (moved)
            {
                File.Move(hidden, beside);
            }
        }
    }
}

/// <summary>
/// 設定画面のモデル欄。**状態を黙らないこと**を見る。
/// </summary>
[Collection("モデルの置き場所")]
public class SettingsModelTests
{
    [AvaloniaFact]
    public void モデルが在れば名前と大きさを出す()
    {
        var settings = new SettingsViewModel(TestShell.Create());

        Assert.True(settings.HasModel);
        Assert.Contains(".dcm", settings.ModelStatus);
        Assert.Contains("MB", settings.ModelStatus);
    }

    [AvaloniaFact]
    public void 置き場所を出す()
        => Assert.NotEmpty(new SettingsViewModel(TestShell.Create()).ModelLocation);

    /// <summary>
    /// **置いた直後に反映されること。** 起動時に 1 回数えるだけだと、
    /// 置いた人に「まだ無い」と言い続ける。
    /// </summary>
    [AvaloniaFact]
    public void 置き場所を見直すと一覧に増える()
    {
        var settings = new SettingsViewModel(TestShell.Create());
        var before = settings.AvailableModels.Count;

        var placed = Path.Combine(AppContext.BaseDirectory, "zz-見直し用.dcm");
        File.WriteAllBytes(placed, [0x00]);
        try
        {
            settings.RefreshModelsCommand.Execute(null);
            Assert.Equal(before + 1, settings.AvailableModels.Count);
            Assert.Contains("zz-見直し用.dcm", settings.AvailableModels);
        }
        finally
        {
            File.Delete(placed);
            settings.RefreshModelsCommand.Execute(null);
        }
    }
}

/// <summary>
/// **設定画面を実際に描く。**
///
/// ビルドが通っても開かないことがある（スタイルの二重定義、存在しない
/// 部品名、束縛の型違い）。ViewModel の性質を見るだけでは出てこないので、
/// 仮想画面へ載せて描かせる。
/// </summary>
[Collection("モデルの置き場所")]
public class SettingsViewRenderTests
{
    [AvaloniaFact]
    public void 設定画面が開く()
    {
        var view = new Views.SettingsView
        {
            DataContext = new SettingsViewModel(TestShell.Create()),
        };
        var window = new Avalonia.Controls.Window { Content = view };

        window.Show();

        Assert.True(window.IsVisible);
        Assert.NotNull(view.DataContext);
    }
}
