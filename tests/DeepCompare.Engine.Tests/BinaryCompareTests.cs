using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class BinaryCompareTests
{
    private static byte[] Bytes(params int[] values) => [.. values.Select(v => (byte)v)];

    private static byte[] Sequence(int count, int start = 0)
        => [.. Enumerable.Range(start, count).Select(i => (byte)(i % 256))];

    [Fact]
    public void 同じ内容なら差分は無い()
    {
        var data = Sequence(64);

        var result = BinaryCompare.Compare(data, data);

        Assert.True(result.Identical);
        Assert.Equal(0, result.DifferentRows);
        Assert.Equal(4, result.Rows.Count);   // 64 バイト ÷ 16
    }

    [Fact]
    public void 一バイト違えばその行だけが差分になる()
    {
        var left = Sequence(64);
        var right = Sequence(64);
        right[20] = 0xFF;   // 2 行目（16〜31）の中

        var result = BinaryCompare.Compare(left, right);

        Assert.Equal(1, result.DifferentRows);
        var row = Assert.Single(result.Rows, r => !r.IsUnchanged);
        Assert.Equal(16, row.LeftOffset);
        Assert.Equal([4], row.ChangedColumns);   // 行の中の 4 バイト目
    }

    [Fact]
    public void 末尾に足しただけなら前の行は一致したまま()
    {
        var left = Sequence(64);
        var right = Sequence(80);

        var result = BinaryCompare.Compare(left, right);

        // 先頭 4 行はそのまま。増えた 1 行だけが右のみになる。
        Assert.Equal(4, result.Rows.Count(r => r.IsUnchanged));
        Assert.Single(result.Rows, r => r.LeftOffset is null);
    }

    [Fact]
    public void 先頭に一バイト挿入すると以降が全部ずれる()
    {
        // **16 進表示の性質としてこうなる。** 1 行 16 バイトの固定区切りなので、
        // 先頭が 1 バイト動くと以降のすべての行で中身が変わる。
        //
        // バイト単位で対応付ければ「1 バイト挿入」と表せるが、そうすると
        // **オフセットが 16 の倍数で揃わなくなり、16 進表示の利点（アドレスが
        // 目で追える）が失われる。** アドレスを見るための表示なので、
        // 固定区切りを採る。Beyond Compare の 16 進表示も同じ。
        var left = Sequence(160);
        var right = new List<byte>(left);
        right.Insert(0, 0xAA);

        var result = BinaryCompare.Compare(left, right.ToArray());

        Assert.False(result.Identical);
        // 一致する行は無い（末尾の半端な行を除いて全部ずれる）。
        Assert.Equal(0, result.Rows.Count(r => r.IsUnchanged));
    }

    [Fact]
    public void 片側にしか無い部分を出す()
    {
        var left = Sequence(32);
        var right = Sequence(64);

        var result = BinaryCompare.Compare(left, right);

        Assert.Contains(result.Rows, r => r.LeftOffset is null && r.RightOffset is not null);
        Assert.False(result.Identical);
    }

    [Fact]
    public void 空のファイルを扱える()
    {
        var result = BinaryCompare.Compare([], []);

        Assert.True(result.Identical);
        Assert.Empty(result.Rows);
    }

    [Fact]
    public void 片方だけ空なら全部が片側のみになる()
    {
        var result = BinaryCompare.Compare([], Sequence(32));

        Assert.Equal(2, result.Rows.Count);
        Assert.All(result.Rows, r => Assert.Null(r.LeftOffset));
    }

    [Fact]
    public void 長さが同じでも中身が違えば同一とは言わない()
    {
        Assert.False(BinaryCompare.Compare(Bytes(1, 2, 3), Bytes(1, 2, 4)).Identical);
    }

    // --- 表示 ---

    [Fact]
    public void 十六進を二桁ずつ空白で区切る()
    {
        var result = BinaryCompare.Compare(Bytes(0x00, 0x0F, 0xFF), []);

        Assert.Equal("00 0F FF", result.Rows[0].Hex(left: true));
    }

    [Fact]
    public void 八バイトごとに区切りを広げる()
    {
        var result = BinaryCompare.Compare(Sequence(16), []);

        // 8 バイト目の前だけ空白が 2 つになる。数えやすさが段違いに変わる。
        Assert.Contains("07  08", result.Rows[0].Hex(left: true));
    }

    [Fact]
    public void 読めるバイトだけ文字にする()
    {
        var result = BinaryCompare.Compare(Encoding.ASCII.GetBytes("Hi\0\n!"), []);

        Assert.Equal("Hi..!", result.Rows[0].Ascii(left: true));
    }

    [Fact]
    public void 無い側は空文字を返す()
    {
        var result = BinaryCompare.Compare([], Sequence(8));

        Assert.Equal(string.Empty, result.Rows[0].Hex(left: true));
        Assert.Equal(string.Empty, result.Rows[0].Ascii(left: true));
    }

    // --- バイナリらしさの判定 ---

    [Fact]
    public void NULを含むファイルをバイナリとみなす()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, Bytes(0x41, 0x00, 0x42));
            Assert.True(BinaryCompare.LooksBinary(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void 日本語のテキストをバイナリとみなさない()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "日本語のテキストです\n", new UTF8Encoding(false));
            Assert.False(BinaryCompare.LooksBinary(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void UTF16のテキストをバイナリとみなさない()
    {
        // UTF-16 は NUL を含むが読めるテキスト。**BOM を先に見る。**
        // ここを間違えると、UTF-16 のファイルが 16 進でしか開けなくなる。
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "abc", new UnicodeEncoding(false, true));
            Assert.False(BinaryCompare.LooksBinary(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void 空のファイルをバイナリとみなさない()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, []);
            Assert.False(BinaryCompare.LooksBinary(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    // --- ファイルから ---

    [Fact]
    public void 上限を超えたら切り捨ててその旨を返す()
    {
        var left = Path.GetTempFileName();
        var right = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(left, Sequence(200));
            File.WriteAllBytes(right, Sequence(200));

            var (comparison, truncated) = BinaryCompare.CompareFiles(left, right, maximumBytes: 64);

            Assert.True(truncated);
            Assert.Equal(64, comparison.LeftLength);
        }
        finally
        {
            File.Delete(left);
            File.Delete(right);
        }
    }

    [Fact]
    public void 要約を出す()
    {
        var text = BinaryCompare.Format(BinaryCompare.Compare(Bytes(1, 2), Bytes(1, 3)));

        Assert.Contains("1 行が違います", text);
    }

    [Fact]
    public void 同じなら同じと言う()
    {
        var text = BinaryCompare.Format(BinaryCompare.Compare(Bytes(1, 2), Bytes(1, 2)));

        Assert.Contains("同じ内容です", text);
    }
}
