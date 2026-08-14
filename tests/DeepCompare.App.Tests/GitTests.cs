using System.Diagnostics;
using Avalonia.Headless.XUnit;
using Xunit;

namespace DeepCompare.App.Tests;

/// <summary>
/// Git は `git` コマンドを呼ぶ方式（ROADMAP 5.1 の決定）。
/// **本物のリポジトリを一時フォルダーに作って回す。**
/// </summary>
public class GitTests
{
    /// <summary>git が入っていない環境ではこの一群を飛ばす。</summary>
    private static bool GitAvailable => Run(Path.GetTempPath(), "--version").ok;

    private static (bool ok, string output) Run(string cwd, params string[] args)
    {
        var info = new ProcessStartInfo("git")
        {
            WorkingDirectory = cwd,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var arg in args)
        {
            info.ArgumentList.Add(arg);
        }
        try
        {
            using var process = Process.Start(info);
            if (process is null)
            {
                return (false, string.Empty);
            }
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            return (process.ExitCode == 0, output);
        }
        catch (Exception)
        {
            return (false, string.Empty);
        }
    }

    /// <summary>コミットが 1 つある、まっさらなリポジトリを作る。</summary>
    private static TempFolder NewRepository()
    {
        var folder = new TempFolder();
        Run(folder.Path, "init", "-q");
        // **その場限りの名前を使う。** 走らせた人の設定に依存させない。
        Run(folder.Path, "config", "user.email", "test@example.invalid");
        Run(folder.Path, "config", "user.name", "試験");
        File.WriteAllText(Path.Combine(folder.Path, "はじめ.txt"), "あ\n");
        Run(folder.Path, "add", ".");
        Run(folder.Path, "commit", "-q", "-m", "最初");
        return folder;
    }

    [AvaloniaFact]
    public async Task リポジトリでない場所ならそう言う()
    {
        using var folder = new TempFolder();
        var model = new GitViewModel(TestShell.Create(), folder.Path);

        await model.RefreshAsync();

        // git が無い環境でも「無い」と言うので、どちらかの知らせが出ていればよい。
        Assert.NotEmpty(model.Message);
        Assert.Empty(model.Commits);
    }

    [AvaloniaFact]
    public async Task 履歴と枝が読める()
    {
        if (!GitAvailable)
        {
            return;
        }
        using var repository = NewRepository();
        var model = new GitViewModel(TestShell.Create(), repository.Path);

        await model.RefreshAsync();

        Assert.NotEmpty(model.Commits);
        Assert.NotEmpty(model.Branch);
        Assert.Empty(model.Message);
    }

    [AvaloniaFact]
    public async Task 直したファイルが未ステージとして出る()
    {
        if (!GitAvailable)
        {
            return;
        }
        using var repository = NewRepository();
        File.WriteAllText(Path.Combine(repository.Path, "はじめ.txt"), "あ\nい\n");
        var model = new GitViewModel(TestShell.Create(), repository.Path);

        await model.RefreshAsync();

        Assert.Contains(model.UnstagedFiles, f => f.FileName == "はじめ.txt");
        Assert.DoesNotContain(model.StagedFiles, f => f.FileName == "はじめ.txt");
    }

    [AvaloniaFact]
    public async Task ステージするとstaged側へ移る()
    {
        if (!GitAvailable)
        {
            return;
        }
        using var repository = NewRepository();
        File.WriteAllText(Path.Combine(repository.Path, "はじめ.txt"), "あ\nい\n");
        var model = new GitViewModel(TestShell.Create(), repository.Path);
        await model.RefreshAsync();

        var row = model.UnstagedFiles.Single(f => f.FileName == "はじめ.txt");
        await model.StageAsync(row, stage: true);

        Assert.Contains(model.StagedFiles, f => f.FileName == "はじめ.txt");
        Assert.DoesNotContain(model.UnstagedFiles, f => f.FileName == "はじめ.txt");
    }

    [AvaloniaFact]
    public async Task 降ろすと未ステージへ戻る()
    {
        if (!GitAvailable)
        {
            return;
        }
        using var repository = NewRepository();
        File.WriteAllText(Path.Combine(repository.Path, "はじめ.txt"), "あ\nい\n");
        var model = new GitViewModel(TestShell.Create(), repository.Path);
        await model.RefreshAsync();

        var row = model.UnstagedFiles.Single(f => f.FileName == "はじめ.txt");
        await model.StageAsync(row, stage: true);
        var staged = model.StagedFiles.Single(f => f.FileName == "はじめ.txt");
        await model.StageAsync(staged, stage: false);

        Assert.Contains(model.UnstagedFiles, f => f.FileName == "はじめ.txt");
        Assert.DoesNotContain(model.StagedFiles, f => f.FileName == "はじめ.txt");
    }

    /// <summary>
    /// **空のメッセージでコミットさせない。** 押せる見た目で何も起きないより、
    /// 押せないほうがよい。
    /// </summary>
    [AvaloniaFact]
    public async Task メッセージが空ならコミットできない()
    {
        if (!GitAvailable)
        {
            return;
        }
        using var repository = NewRepository();
        File.WriteAllText(Path.Combine(repository.Path, "はじめ.txt"), "あ\nい\n");
        var model = new GitViewModel(TestShell.Create(), repository.Path);
        await model.RefreshAsync();
        var row = model.UnstagedFiles.Single(f => f.FileName == "はじめ.txt");
        await model.StageAsync(row, stage: true);

        model.CommitMessage = "   ";

        Assert.False(model.CanCommit);
    }

    [AvaloniaFact]
    public async Task コミットすると履歴が増え作業ツリーが片付く()
    {
        if (!GitAvailable)
        {
            return;
        }
        using var repository = NewRepository();
        File.WriteAllText(Path.Combine(repository.Path, "はじめ.txt"), "あ\nい\n");
        var model = new GitViewModel(TestShell.Create(), repository.Path);
        await model.RefreshAsync();
        var before = model.Commits.Count;

        var row = model.UnstagedFiles.Single(f => f.FileName == "はじめ.txt");
        await model.StageAsync(row, stage: true);
        model.CommitMessage = "二つ目";
        Assert.True(model.CanCommit);
        await model.CommitAsync(amend: false);

        Assert.Equal(before + 1, model.Commits.Count);
        Assert.Empty(model.StagedFiles);
        Assert.Empty(model.UnstagedFiles);
        Assert.Empty(model.CommitMessage);
    }

    [AvaloniaFact]
    public async Task 新しいファイルも未ステージとして出る()
    {
        if (!GitAvailable)
        {
            return;
        }
        using var repository = NewRepository();
        File.WriteAllText(Path.Combine(repository.Path, "あたらしい.txt"), "新\n");
        var model = new GitViewModel(TestShell.Create(), repository.Path);

        await model.RefreshAsync();

        Assert.Contains(model.UnstagedFiles, f => f.FileName == "あたらしい.txt");
    }
}
