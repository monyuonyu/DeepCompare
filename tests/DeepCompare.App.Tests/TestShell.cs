namespace DeepCompare.App.Tests;

/// <summary>
/// 画面試験の下ごしらえ。
///
/// **人に選ばせる経路は塞ぐ。** ファイル選択の枠は試験からは開けない
/// （開いたら誰も閉じないので固まる）ので、null を返す関数を渡しておく。
/// </summary>
internal static class TestShell
{
    /// <param name="savePath">
    /// 「どこへ書き出すか」を尋ねられたときに返す場所。
    /// 既定は null＝人が取り消した扱い。
    /// </param>
    public static ShellViewModel Create(string? savePath = null)
        => new(
            (_, _) => Task.FromResult<string?>(null),
            (_, _) => Task.FromResult(savePath))
        {
            // 埋め込みを読むと 1 件ごとに数秒かかるうえ、対応付けが
            // モデル任せになって答えが揺れる。**構造のみで確定させる。**
            FastMode = true,
        };
}

/// <summary>使い終わったら消える一時ファイル。</summary>
internal sealed class TempFile : IDisposable
{
    public TempFile(string contents, string extension = ".txt")
    {
        Path = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            $"deepcompare-test-{Guid.NewGuid():N}{extension}");
        File.WriteAllText(Path, contents);
    }

    public string Path { get; }

    public string Read() => File.ReadAllText(Path);

    public void Dispose()
    {
        try
        {
            File.Delete(Path);
        }
        catch (IOException)
        {
            // 消せなくても試験の結果は変わらない。
        }
    }
}

/// <summary>使い終わったら消える一時フォルダー。</summary>
internal sealed class TempFolder : IDisposable
{
    public TempFolder()
    {
        Path = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), $"deepcompare-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(Path);
    }

    public string Path { get; }

    public void Dispose()
    {
        try
        {
            Directory.Delete(Path, recursive: true);
        }
        catch (IOException)
        {
        }
    }
}
