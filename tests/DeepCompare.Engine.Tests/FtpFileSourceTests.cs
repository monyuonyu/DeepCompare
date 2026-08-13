using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// FTP の一覧の読み取り。
///
/// **接続そのものは本物のサーバーで確かめてある**（pyftpdlib を立てて
/// curl + diff と突き合わせ）。ここでは、応答の形を読む部分を固定する。
/// </summary>
public sealed class FtpListingTests
{
    // --- MLSD（機械が読む形） ---

    [Fact]
    public void MLSDを読む()
    {
        var text =
            "type=cdir;sizd=4096;modify=20260813050000; .\r\n"
            + "type=pdir;sizd=4096;modify=20260813050000; ..\r\n"
            + "type=file;size=42;modify=20260813051500; a.txt\r\n"
            + "type=dir;sizd=4096;modify=20260813052000; 下\r\n";

        var entries = FtpFileSource.ParseMlsd(text, string.Empty);

        // **自分自身と親は落とす。** 残すと木が循環する。
        Assert.Equal(2, entries.Count);

        var file = entries.Single(e => !e.IsDirectory);
        Assert.Equal("a.txt", file.RelativePath);
        Assert.Equal(42, file.Size);
        // modify は規約で UTC と決まっている。
        Assert.Equal(new DateTimeOffset(2026, 8, 13, 5, 15, 0, TimeSpan.Zero), file.Modified);

        Assert.Equal("下", entries.Single(e => e.IsDirectory).RelativePath);
    }

    [Fact]
    public void MLSDでも根からの相対にする()
    {
        // 深い段の項目が「その段からの相対」になると、次に読むときに見つからない。
        var text = "type=file;size=5; b.txt\r\n";

        var entries = FtpFileSource.ParseMlsd(text, "下");

        Assert.Equal("下/b.txt", Assert.Single(entries).RelativePath);
    }

    [Fact]
    public void MLSDで名前に空白が入っても切らない()
    {
        var text = "type=file;size=10; 空白 の 入る 名前.txt\r\n";

        Assert.Equal("空白 の 入る 名前.txt",
            Assert.Single(FtpFileSource.ParseMlsd(text, string.Empty)).RelativePath);
    }

    // --- LIST（人が読む形。MLSD が使えない相手向け） ---

    [Fact]
    public void LISTを読む()
    {
        var text =
            "drwxr-xr-x   2 owner group     4096 Aug 13 05:00 下\r\n"
            + "-rw-r--r--   1 owner group       42 Aug 13 05:15 a.txt\r\n";

        var entries = FtpFileSource.ParseList(text, string.Empty);

        Assert.Equal(2, entries.Count);
        Assert.True(entries.Single(e => e.RelativePath == "下").IsDirectory);

        var file = entries.Single(e => e.RelativePath == "a.txt");
        Assert.False(file.IsDirectory);
        Assert.Equal(42, file.Size);
        // **時刻は諦める。** LIST の日付は年が落ちていたり、書式が実装ごとに
        // 違ったりする。中身と大きさで比べる方が確か。
        Assert.Null(file.Modified);
    }

    [Fact]
    public void LISTで名前に空白が入っても切らない()
    {
        // **8 個目までを飛ばして、残り全部を名前にする。**
        // 空白で切ると「空白 入り.txt」が「空白」になる。
        var text = "-rw-r--r--   1 owner group       33 Aug 13 05:15 空白 入り.txt\r\n";

        Assert.Equal("空白 入り.txt",
            Assert.Single(FtpFileSource.ParseList(text, string.Empty)).RelativePath);
    }

    [Fact]
    public void LISTの自分自身と親は落とす()
    {
        var text =
            "drwxr-xr-x   2 owner group     4096 Aug 13 05:00 .\r\n"
            + "drwxr-xr-x   3 owner group     4096 Aug 13 05:00 ..\r\n"
            + "-rw-r--r--   1 owner group       10 Aug 13 05:15 a.txt\r\n";

        Assert.Equal("a.txt",
            Assert.Single(FtpFileSource.ParseList(text, string.Empty)).RelativePath);
    }

    [Fact]
    public void 読めない行は飛ばす()
    {
        // 相手が何を返すか分からない。**そこで丸ごと失敗しない。**
        var text = "total 12\r\nよく分からない行\r\n"
            + "-rw-r--r--   1 owner group       10 Aug 13 05:15 a.txt\r\n";

        Assert.Equal("a.txt",
            Assert.Single(FtpFileSource.ParseList(text, string.Empty)).RelativePath);
    }
}

public sealed class FtpLocationTests
{
    [Theory]
    [InlineData("ftp://主機/場所", true)]
    [InlineData("ftps://主機/場所", true)]
    [InlineData("/手元", false)]
    public void リモートかどうかを見分ける(string location, bool expected)
        => Assert.Equal(expected, RemoteLocation.IsRemote(location));

    [Fact]
    public void 合言葉を伏せる()
        => Assert.Equal("ftp://利用者:***@主機/場所",
            RemoteLocation.Redact("ftp://利用者:ひみつ@主機/場所"));
}
