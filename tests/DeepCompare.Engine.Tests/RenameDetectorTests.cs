using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// リネームの検出。
///
/// 誤検出（無関係なファイルを同じものだと言う）の方が、見逃しより害が大きい。
/// 見逃せば従来どおり「左のみ／右のみ」に出るだけだが、誤検出は嘘の対応を見せる。
/// </summary>
public sealed class RenameDetectorTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "dc-rename-" + Guid.NewGuid().ToString("N"));

    private string Left => Path.Combine(_root, "left");
    private string Right => Path.Combine(_root, "right");

    public RenameDetectorTests()
    {
        Directory.CreateDirectory(Left);
        Directory.CreateDirectory(Right);
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch
        {
            // 後片付けの失敗は結果に関係ない。
        }
    }

    private void Write(string side, string relative, string content)
    {
        var path = Path.Combine(side == "left" ? Left : Right, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }

    private List<RenamePair> Detect(float threshold = RenameDetector.DefaultThreshold)
    {
        var comparison = FolderComparer.Compare(Left, Right);
        return RenameDetector.Detect(comparison, Left, Right, threshold);
    }

    private static string Body(int lines, string marker = "")
        => string.Join('\n', Enumerable.Range(0, lines).Select(i => $"line {i} {marker}"));

    [Fact]
    public void IdenticalContentUnderANewNameIsAMove()
    {
        Write("left", "old.cs", Body(20));
        Write("right", "new.cs", Body(20));

        var pair = Assert.Single(Detect());

        Assert.Equal("old.cs", pair.LeftPath);
        Assert.Equal("new.cs", pair.RightPath);
        Assert.True(pair.IdenticalContent);
        Assert.Equal(1f, pair.Similarity);
    }

    [Fact]
    public void MovingToAnotherFolderIsDetected()
    {
        Write("left", "a/thing.cs", Body(20));
        Write("right", "b/thing.cs", Body(20));

        var pair = Assert.Single(Detect());

        Assert.Equal("a/thing.cs", pair.LeftPath);
        Assert.Equal("b/thing.cs", pair.RightPath);
    }

    /// <summary>名前が変わり、中身も少し変わった場合。</summary>
    [Fact]
    public void ARenameWithSmallEditsIsStillDetected()
    {
        Write("left", "old.cs", Body(20));
        Write("right", "new.cs", Body(20) + "\nline 20 追加");

        var pair = Assert.Single(Detect());

        Assert.False(pair.IdenticalContent);
        Assert.True(pair.Similarity > 0.9f, $"重なり {pair.Similarity}");
    }

    /// <summary>
    /// 無関係なファイル同士を組にしないこと。見逃しより誤検出の方が害が大きい。
    /// </summary>
    [Fact]
    public void UnrelatedFilesAreNotPaired()
    {
        Write("left", "alpha.cs", Body(20, "alpha"));
        Write("right", "beta.cs", Body(20, "beta"));

        Assert.Empty(Detect());
    }

    [Fact]
    public void FilesPresentOnBothSidesAreNotConsidered()
    {
        Write("left", "same.cs", Body(20));
        Write("right", "same.cs", Body(20));

        Assert.Empty(Detect());
    }

    /// <summary>候補が複数あるときは最も重なる相手を選ぶこと。</summary>
    [Fact]
    public void TheBestMatchWins()
    {
        Write("left", "source.cs", Body(20));
        Write("right", "close.cs", Body(20) + "\n少しだけ違う");
        Write("right", "far.cs", Body(5));

        var pair = Assert.Single(Detect(), p => p.LeftPath == "source.cs");

        Assert.Equal("close.cs", pair.RightPath);
    }

    /// <summary>1 つの相手を 2 つの元に割り当てないこと。</summary>
    [Fact]
    public void OneTargetIsUsedOnlyOnce()
    {
        Write("left", "a.cs", Body(20));
        Write("left", "b.cs", Body(20));
        Write("right", "moved.cs", Body(20));

        var pairs = Detect();

        Assert.Single(pairs);
        Assert.Equal("moved.cs", pairs[0].RightPath);
    }

    [Fact]
    public void AHigherThresholdRejectsWeakMatches()
    {
        Write("left", "old.cs", Body(20));
        Write("right", "new.cs", Body(10) + "\n" + Body(10, "違う"));

        Assert.NotEmpty(Detect(0.2f));
        Assert.Empty(Detect(0.95f));
    }

    /// <summary>空行だけのファイルで誤検出しないこと。共通しすぎて根拠にならない。</summary>
    [Fact]
    public void BlankOnlyFilesDoNotMatchEachOther()
    {
        Write("left", "a.txt", "\n\n\n");
        Write("right", "b.txt", "\n\n");

        Assert.Empty(Detect());
    }

    [Fact]
    public void NothingToDoWhenOneSideHasNoUniqueFiles()
    {
        Write("left", "only.cs", Body(20));

        Assert.Empty(Detect());
    }
}
