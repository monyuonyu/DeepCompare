using System.Buffers.Binary;
using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// PE のバージョン情報を読む試験。
///
/// **実物の exe をリポジトリに置かない。** 300KB のバイナリが増えるうえ、
/// 中身を目で確かめられない。ここでは最小の PE を組み立てる。境界の揃え方や
/// 鍵の長さが奇数の場合など、**狙った条件を作れる**利点の方が大きい。
///
/// 実物との一致は別に取ってある（pefile と 13 個で照合、食い違い 0 件）。
/// </summary>
public sealed class VersionInfoTests
{
    // --- PE を組み立てる ---

    private sealed class Builder
    {
        private readonly List<byte> _bytes = [];

        public int Length => _bytes.Count;

        public void U16(int value) => _bytes.AddRange(BitConverter.GetBytes((ushort)value));
        public void U32(uint value) => _bytes.AddRange(BitConverter.GetBytes(value));
        public void Bytes(ReadOnlySpan<byte> value) => _bytes.AddRange(value.ToArray());
        public void Zero(int count) => _bytes.AddRange(new byte[count]);

        /// <summary>UTF-16 の鍵。終端の 0 も入れる。</summary>
        public void Key(string text)
        {
            Bytes(Encoding.Unicode.GetBytes(text));
            U16(0);
        }

        /// <summary>4 バイト境界へ揃える。**ここを外すと以降が全部化ける。**</summary>
        public void Align()
        {
            while (_bytes.Count % 4 != 0)
            {
                _bytes.Add(0);
            }
        }

        public void PatchU16(int at, int value)
            => BinaryPrimitives.WriteUInt16LittleEndian(
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_bytes)[at..], (ushort)value);

        public void PatchU32(int at, uint value)
            => BinaryPrimitives.WriteUInt32LittleEndian(
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_bytes)[at..], value);

        public byte[] ToArray() => [.. _bytes];
    }

    /// <summary>VS_VERSIONINFO を組み立てる。</summary>
    private static byte[] BuildVersionResource(
        string fileVersion, string productVersion,
        IEnumerable<(string Key, string Value)> strings)
    {
        var b = new Builder();

        var rootAt = b.Length;
        b.U16(0);              // 長さ（後で埋める）
        b.U16(52);             // 値の長さ = VS_FIXEDFILEINFO
        b.U16(0);              // 種別 = 数値
        b.Key("VS_VERSION_INFO");
        b.Align();

        var (fm, fl) = Split(fileVersion);
        var (pm, pl) = Split(productVersion);
        b.U32(0xFEEF04BD);     // 署名
        b.U32(0x00010000);     // 構造体の版
        b.U32(fm); b.U32(fl);
        b.U32(pm); b.U32(pl);
        b.U32(0x3F); b.U32(0); // フラグの覆いと値
        b.U32(0x40004); b.U32(1); b.U32(0);   // OS / 種類 / 副種類
        b.U32(0); b.U32(0);    // 日付
        b.Align();

        var stringFileInfoAt = b.Length;
        b.U16(0);
        b.U16(0);
        b.U16(1);              // 種別 = 文字列
        b.Key("StringFileInfo");
        b.Align();

        var tableAt = b.Length;
        b.U16(0);
        b.U16(0);
        b.U16(1);
        b.Key("040904B0");     // 英語（米国）／Unicode
        b.Align();

        foreach (var (key, value) in strings)
        {
            var itemAt = b.Length;
            b.U16(0);
            // **値の長さは文字数**（バイト数ではない）。終端の 0 も 1 文字。
            b.U16(value.Length + 1);
            b.U16(1);
            b.Key(key);
            b.Align();
            b.Key(value);
            b.Align();
            b.PatchU16(itemAt, b.Length - itemAt);
        }

        b.PatchU16(tableAt, b.Length - tableAt);
        b.PatchU16(stringFileInfoAt, b.Length - stringFileInfoAt);
        b.PatchU16(rootAt, b.Length - rootAt);
        return b.ToArray();
    }

    private static (uint Most, uint Least) Split(string version)
    {
        var parts = version.Split('.');
        uint At(int i) => i < parts.Length && uint.TryParse(parts[i], out var v) ? v : 0;
        return ((At(0) << 16) | At(1), (At(2) << 16) | At(3));
    }

    /// <summary>バージョンリソースだけを持つ最小の PE。</summary>
    private static byte[] BuildExecutable(
        byte[] versionResource, ushort machine = 0x8664, uint timestamp = 0,
        uint certificateSize = 0, int[]? languages = null, byte[][]? perLanguage = null)
    {
        const int PeAt = 0x80;
        const int SectionRva = 0x1000;

        var b = new Builder();
        b.Bytes("MZ"u8);
        b.Zero(0x3C - 2);
        b.U32(PeAt);
        b.Zero(PeAt - 0x40);

        b.U32(0x00004550);     // "PE\0\0"
        b.U16(machine);
        b.U16(1);              // セクション数
        b.U32(timestamp);
        b.U32(0); b.U32(0);    // 記号表
        b.U16(240);            // Optional の大きさ（PE32+）
        b.U16(0x22);

        var optionalAt = b.Length;
        b.U16(0x20B);          // PE32+
        b.Zero(110);           // データディレクトリの手前まで

        // データディレクトリ 16 個。3 番目がリソース、5 番目が証明書。
        var directoriesAt = b.Length;
        for (var i = 0; i < 16; i++)
        {
            b.U32(0); b.U32(0);
        }
        b.PatchU32(directoriesAt + 2 * 8, SectionRva);                    // リソースの RVA
        b.PatchU32(directoriesAt + 2 * 8 + 4, 0x1000);
        b.PatchU32(directoriesAt + 4 * 8 + 4, certificateSize);           // 証明書の大きさ

        Assert.Equal(240, b.Length - optionalAt);

        // リソースの中身（木 + データ）を先に組んで、大きさを確かめる。
        var resource = languages is null
            ? BuildResourceSection(versionResource, SectionRva)
            : BuildMultiLanguageSection(languages, perLanguage!, SectionRva);

        var sectionAt = b.Length;
        b.Bytes(".rsrc\0\0\0"u8);
        b.U32((uint)resource.Length);   // 仮想の大きさ
        b.U32(SectionRva);
        b.U32((uint)resource.Length);   // 生の大きさ
        b.U32(0);                       // 生の位置（後で埋める）
        b.Zero(16);

        b.Align();
        var rawAt = b.Length;
        b.PatchU32(sectionAt + 20, (uint)rawAt);
        b.Bytes(resource);

        return b.ToArray();
    }

    /// <summary>種類 → 名前 → 言語の 3 段と、データ本体。</summary>
    private static byte[] BuildResourceSection(byte[] versionResource, uint sectionRva)
    {
        var b = new Builder();

        void Directory(int idOrOffset, int target, bool isDirectory)
        {
            b.U32(0); b.U32(0);        // 特性と時刻
            b.U16(0); b.U16(0);        // 版
            b.U16(0); b.U16(1);        // 名前つき 0 / 番号つき 1
            b.U32((uint)idOrOffset);
            // 最上位ビットが立っていれば「次もディレクトリ」の意味。
            b.U32(isDirectory ? (uint)target | 0x80000000u : (uint)target);
        }

        // 3 段のディレクトリは各 24 バイト（ヘッダ 16 + 項目 8）。
        Directory(16, 24, isDirectory: true);      // RT_VERSION → 次の段
        Directory(1, 48, isDirectory: true);       // 名前 1 → 次の段
        Directory(1033, 72, isDirectory: false);   // 言語 → データ項目

        var dataEntryAt = b.Length;
        Assert.Equal(72, dataEntryAt);
        b.U32(0);                                  // データの RVA（後で埋める）
        b.U32((uint)versionResource.Length);
        b.U32(0); b.U32(0);                        // 符号ページと予約

        b.Align();
        var dataAt = b.Length;
        b.PatchU32(dataEntryAt, sectionRva + (uint)dataAt);
        b.Bytes(versionResource);
        return b.ToArray();
    }

    /// <summary>言語ごとに別のリソースを持つ木。**実物の exe はこの形。**</summary>
    private static byte[] BuildMultiLanguageSection(
        int[] languages, byte[][] blocks, uint sectionRva)
    {
        var b = new Builder();

        void Directory(int count)
        {
            b.U32(0); b.U32(0);
            b.U16(0); b.U16(0);
            b.U16(0); b.U16(count);
        }

        // 種類 → 名前 の 2 段は 1 項目ずつ。
        Directory(1);
        b.U32(16);                                   // RT_VERSION
        b.U32(24u | 0x80000000u);
        Directory(1);
        b.U32(1);
        b.U32(48u | 0x80000000u);

        // 言語の段。項目数ぶん並ぶ。
        var languageAt = b.Length;
        Directory(languages.Length);
        var entriesAt = b.Length;
        for (var i = 0; i < languages.Length; i++)
        {
            b.U32((uint)languages[i]);
            b.U32(0);                                // データ項目の位置（後で埋める）
        }

        // データ項目とデータ本体。
        var dataEntries = new int[languages.Length];
        for (var i = 0; i < languages.Length; i++)
        {
            b.Align();
            dataEntries[i] = b.Length;
            b.PatchU32(entriesAt + i * 8 + 4, (uint)dataEntries[i]);
            b.U32(0);
            b.U32((uint)blocks[i].Length);
            b.U32(0); b.U32(0);
        }
        for (var i = 0; i < languages.Length; i++)
        {
            b.Align();
            b.PatchU32(dataEntries[i], sectionRva + (uint)b.Length);
            b.Bytes(blocks[i]);
        }

        Assert.Equal(48, languageAt);
        return b.ToArray();
    }

    private static byte[] Sample(
        string fileVersion = "1.2.3.4",
        string productVersion = "1.2.0.0",
        ushort machine = 0x8664,
        uint timestamp = 0,
        uint certificateSize = 0,
        params (string Key, string Value)[] strings)
    {
        if (strings.Length == 0)
        {
            strings =
            [
                ("CompanyName", "ほげ株式会社"),
                ("FileDescription", "試験用"),
                ("FileVersion", fileVersion),
                ("ProductVersion", "1.2.0-beta"),
            ];
        }
        return BuildExecutable(
            BuildVersionResource(fileVersion, productVersion, strings),
            machine, timestamp, certificateSize);
    }

    // --- 試験 ---

    [Fact]
    public void 版と文字列を読む()
    {
        var version = VersionInfo.Read(Sample());

        Assert.Equal("1.2.3.4", version.FileVersion);
        Assert.Equal("1.2.0.0", version.ProductVersion);
        Assert.Equal("x64", version.Machine);
        Assert.Equal("ほげ株式会社", version.Get("CompanyName"));
        Assert.Equal("試験用", version.Get("FileDescription"));
        Assert.Equal("1.2.0-beta", version.Get("ProductVersion"));
    }

    [Fact]
    public void 表示用の版と数値の版が食い違っても両方持つ()
    {
        // **これが両方持つ理由。** 表示は "1.2.0-beta" でも、数値は 1.2.0.0。
        var version = VersionInfo.Read(Sample());

        Assert.Equal("1.2.0.0", version.ProductVersion);
        Assert.Equal("1.2.0-beta", version.Get("ProductVersion"));
    }

    [Theory]
    [InlineData((ushort)0x014C, "x86")]
    [InlineData((ushort)0x8664, "x64")]
    [InlineData((ushort)0xAA64, "ARM64")]
    [InlineData((ushort)0x1234, "0x1234")]
    public void アーキテクチャを読む(ushort machine, string expected)
        => Assert.Equal(expected, VersionInfo.Read(Sample(machine: machine)).Machine);

    [Fact]
    public void 時刻が0なら時刻なしとする()
    {
        // 再現ビルドでは意図的に 0 や固定値が入る。**1970 年と表示しない。**
        Assert.Null(VersionInfo.Read(Sample(timestamp: 0)).BuiltAt);

        // epoch は手で書かずに換算する。**手で書くと桁を間違える**
        // （実際、最初にこれを書いたとき 1 日ずれた値を入れていた）。
        var expected = new DateTimeOffset(2026, 8, 13, 5, 0, 0, TimeSpan.Zero);
        Assert.Equal(expected,
            VersionInfo.Read(Sample(timestamp: (uint)expected.ToUnixTimeSeconds())).BuiltAt);
    }

    [Fact]
    public void 署名の有無を見る()
    {
        Assert.False(VersionInfo.Read(Sample()).HasSignature);
        Assert.True(VersionInfo.Read(Sample(certificateSize: 4096)).HasSignature);
    }

    [Fact]
    public void 鍵の長さが奇数でも次の項目がずれない()
    {
        // **4 バイト境界の揃えを外すと、ここで以降が全部化ける。**
        var version = VersionInfo.Read(Sample(strings:
        [
            ("Abc", "1"),          // 鍵 3 文字
            ("Abcd", "2"),         // 鍵 4 文字
            ("Abcde", "3"),        // 鍵 5 文字
            ("CompanyName", "最後まで読めている"),
        ]));

        Assert.Equal("1", version.Get("Abc"));
        Assert.Equal("2", version.Get("Abcd"));
        Assert.Equal("3", version.Get("Abcde"));
        Assert.Equal("最後まで読めている", version.Get("CompanyName"));
    }

    [Fact]
    public void サロゲートペアを一文字として読む()
    {
        // 実物で pefile と食い違ったのがここ（あちらは \uXXXX に逃がしていた）。
        var version = VersionInfo.Read(Sample(strings: [("CompanyName", "emoji 🐟 社")]));

        Assert.Equal("emoji 🐟 社", version.Get("CompanyName"));
    }

    [Fact]
    public void 表示言語に合うリソースを選ぶ()
    {
        // **実物で気づいた。** Windows の notepad.exe は FileDescription を
        // 英語と日本語の両方持っており、Windows 自身は「メモ帳」と言うのに
        // 最初のものを採ると "Notepad" になる。**説明文こそ人が読む部分。**
        const int English = 0x0409;
        const int Japanese = 0x0411;

        var pe = BuildExecutable(
            [],
            languages: [English, Japanese],
            perLanguage:
            [
                BuildVersionResource("1.0.0.0", "1.0.0.0", [("FileDescription", "Notepad")]),
                BuildVersionResource("1.0.0.0", "1.0.0.0", [("FileDescription", "メモ帳")]),
            ]);

        Assert.Equal("メモ帳", VersionInfo.Read(pe, Japanese).Get("FileDescription"));
        Assert.Equal("Notepad", VersionInfo.Read(pe, English).Get("FileDescription"));

        // 持っていない言語を頼まれたら、最初のものに落とす。**読めないより良い。**
        Assert.Equal("Notepad", VersionInfo.Read(pe, 0x040C).Get("FileDescription"));
    }

    [Fact]
    public void 隣の言語フォルダの資源を使う()
    {
        // **実機で気づいた。** Windows のシステムファイルは exe 自身に英語だけを
        // 置き、各言語を `<場所>\<言語>\<名前>.mui` に分ける。notepad.exe を
        // 読むと "Notepad" だが、Windows 自身は「メモ帳」と言う。
        var root = Path.Combine(Path.GetTempPath(), "dc-mui-" + Guid.NewGuid().ToString("N")[..8]);

        // **言語を固定する。** 実装は CurrentUICulture を見るので、走らせる
        // 環境の言語のままだと、CI（英語）では作ったフォルダーと探す場所が
        // 食い違って落ちる。インバリアントだと名前が空になり、
        // Path.Combine が root をそのまま返すのでフォルダーすらできない。
        var original = System.Globalization.CultureInfo.CurrentUICulture;
        System.Globalization.CultureInfo.CurrentUICulture =
            new System.Globalization.CultureInfo("ja-JP");

        var language = Path.Combine(root, "ja-JP");
        Directory.CreateDirectory(language);
        try
        {
            var exe = Path.Combine(root, "app.exe");
            File.WriteAllBytes(exe, BuildExecutable(BuildVersionResource("9.9.9.9", "9.9.9.9",
                [("FileDescription", "English name"), ("CompanyName", "Only in exe")])));

            // .mui は資源だけを持つ入れ物。版は 0 にしておく。
            File.WriteAllBytes(Path.Combine(language, "app.exe.mui"),
                BuildExecutable(BuildVersionResource("0.0.0.0", "0.0.0.0",
                    [("FileDescription", "日本語の名前")])));

            var version = VersionInfo.Read(exe);

            Assert.Equal("日本語の名前", version.Get("FileDescription"));
            // **数値の版は元のファイルのもの。** .mui の 0.0.0.0 で上書きしない。
            Assert.Equal("9.9.9.9", version.FileVersion);
            // .mui に無い項目は元のファイルのものが残る。
            Assert.Equal("Only in exe", version.Get("CompanyName"));
        }
        finally
        {
            System.Globalization.CultureInfo.CurrentUICulture = original;
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void muiが無ければ元のファイルだけを見る()
    {
        var root = Path.Combine(Path.GetTempPath(), "dc-mui-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            var exe = Path.Combine(root, "app.exe");
            File.WriteAllBytes(exe, BuildExecutable(BuildVersionResource("1.0.0.0", "1.0.0.0",
                [("FileDescription", "そのまま")])));

            Assert.Null(VersionInfo.FindMui(exe));
            Assert.Equal("そのまま", VersionInfo.Read(exe).Get("FileDescription"));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void 実行ファイルでなければ断る()
    {
        Assert.Throws<InvalidDataException>(() => VersionInfo.Read("これは PE ではない"u8.ToArray()));
        Assert.False(VersionInfo.LooksLikeExecutable("PK\x03\x04"u8.ToArray()));
        Assert.True(VersionInfo.LooksLikeExecutable("MZ\x90\0"u8.ToArray()));
    }

    [Fact]
    public void バージョン情報が無くても落ちない()
    {
        // リソースが空の PE。**版が読めないことと、壊れていることは別。**
        var pe = BuildExecutable(BuildVersionResource("0.0.0.0", "0.0.0.0", []));
        var version = VersionInfo.Read(pe);

        Assert.Equal("0.0.0.0", version.FileVersion);
        Assert.Empty(version.Strings);
    }

    [Fact]
    public void 違いだけを並べる()
    {
        var left = VersionInfo.Read(Sample("1.2.3.4", strings:
            [("CompanyName", "ほげ"), ("FileDescription", "同じ")]));
        var right = VersionInfo.Read(Sample("1.2.3.5", strings:
            [("CompanyName", "ふが"), ("FileDescription", "同じ")]));

        var differences = VersionInfo.Compare(left, right);

        var fileVersion = differences.Single(d => d.Key == "FileVersion（数値）");
        Assert.False(fileVersion.IsSame);
        Assert.Equal("1.2.3.4", fileVersion.Left);
        Assert.Equal("1.2.3.5", fileVersion.Right);

        Assert.True(differences.Single(d => d.Key == "FileDescription").IsSame);
        Assert.False(differences.Single(d => d.Key == "CompanyName").IsSame);

        // 並びは決めてある。ファイルごとに順が変わると左右が見比べられない。
        Assert.Equal("FileVersion（数値）", differences[0].Key);
        Assert.Equal("ProductVersion（数値）", differences[1].Key);
    }

    [Fact]
    public void 片方にしか無い項目も出す()
    {
        var left = VersionInfo.Read(Sample(strings: [("Comments", "左にだけある")]));
        var right = VersionInfo.Read(Sample(strings: [("CompanyName", "右にだけある")]));

        var differences = VersionInfo.Compare(left, right);

        var comments = differences.Single(d => d.Key == "Comments");
        Assert.Equal("左にだけある", comments.Left);
        Assert.Null(comments.Right);
        Assert.False(comments.IsSame);
    }
}
