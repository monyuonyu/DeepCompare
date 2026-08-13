using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace DeepCompare.Engine;

/// <summary>実行ファイルから読み取ったバージョン情報。</summary>
public sealed record ExecutableVersion(
    /// <summary>ファイル版（数値）。<c>1.2.3.4</c> の形。</summary>
    string FileVersion,

    /// <summary>製品版（数値）。</summary>
    string ProductVersion,

    /// <summary>
    /// 文字列の表。CompanyName、FileDescription など。
    ///
    /// **数値の版と別に持つ。** 表示用の版（"1.2.3-beta"）と数値の版
    /// （1.2.3.0）は食い違うことがあり、どちらも見たい場面がある。
    /// </summary>
    IReadOnlyDictionary<string, string> Strings)
{
    /// <summary>32 ビットか 64 ビットか。**入れ替わっていたら真っ先に気づきたい。**</summary>
    public string Machine { get; init; } = string.Empty;

    /// <summary>ビルド時刻（PE のタイムスタンプ）。再現ビルドだと固定値が入る。</summary>
    public DateTimeOffset? BuiltAt { get; init; }

    /// <summary>署名の情報が入っているか。中身までは検証しない。</summary>
    public bool HasSignature { get; init; }

    public string? Get(string key) => Strings.GetValueOrDefault(key);
}

/// <summary>1 項目の違い。</summary>
public sealed record VersionDifference(string Key, string? Left, string? Right)
{
    public bool IsSame => string.Equals(Left, Right, StringComparison.Ordinal);
}

/// <summary>
/// 実行ファイルのバージョン情報を読み、比べる（BC の Version Compare に当たる）。
///
/// **何のためにあるか。** 「入れ替えたはずの DLL が古いまま」「配った exe と
/// 手元のが同じ物か」は、16 進で比べても分からない（ビルドのたびに全部変わる）。
/// 版と会社名と説明を並べれば一目で済む。
///
/// PE を自前で読む。**Windows の API に頼らない。** 頼ると、Linux で作った
/// 成果物を Linux 上で確かめられなくなる（CI がまさにその形）。
/// </summary>
public static class VersionInfo
{
    /// <summary>読める形か。中身を見て決める（拡張子は当てにならない）。</summary>
    public static bool LooksLikeExecutable(byte[] head)
        => head.Length >= 2 && head[0] == (byte)'M' && head[1] == (byte)'Z';

    public static bool LooksLikeExecutable(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            Span<byte> head = stackalloc byte[2];
            return stream.ReadAtLeast(head, 2, throwOnEndOfStream: false) == 2
                && head[0] == (byte)'M' && head[1] == (byte)'Z';
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    public static ExecutableVersion Read(string path) => Read(File.ReadAllBytes(path));

    public static ExecutableVersion Read(byte[] file)
    {
        if (!LooksLikeExecutable(file))
        {
            throw new InvalidDataException("Windows の実行ファイル（PE）ではありません。");
        }

        // DOS ヘッダの 0x3C に PE ヘッダの位置がある。
        var peOffset = ReadInt32(file, 0x3C);
        if (peOffset <= 0 || peOffset + 24 > file.Length
            || ReadUInt32(file, peOffset) != 0x00004550)   // "PE\0\0"
        {
            throw new InvalidDataException("PE ヘッダが見つかりません。");
        }

        var coff = peOffset + 4;
        var machine = ReadUInt16(file, coff);
        var sectionCount = ReadUInt16(file, coff + 2);
        var timestamp = ReadUInt32(file, coff + 4);
        var optionalSize = ReadUInt16(file, coff + 16);
        var optional = coff + 20;

        // PE32 と PE32+ でデータディレクトリの位置がずれる。
        var magic = ReadUInt16(file, optional);
        var directories = magic == 0x20B ? optional + 112 : optional + 96;

        // リソースは 3 番目、証明書は 5 番目のディレクトリ。
        var resourceRva = ReadUInt32(file, directories + 2 * 8);
        var certificateSize = ReadUInt32(file, directories + 4 * 8 + 4);

        var sections = optional + optionalSize;
        var strings = new Dictionary<string, string>(StringComparer.Ordinal);
        var fileVersion = string.Empty;
        var productVersion = string.Empty;

        if (resourceRva != 0
            && Locate(file, sections, sectionCount, resourceRva) is { } resourceOffset)
        {
            var block = FindVersionResource(file, resourceOffset, resourceRva, sections, sectionCount);
            if (block is { } range)
            {
                (fileVersion, productVersion) = ParseVersionInfo(file, range.Offset, range.Length, strings);
            }
        }

        return new ExecutableVersion(fileVersion, productVersion, strings)
        {
            Machine = machine switch
            {
                0x014C => "x86",
                0x8664 => "x64",
                0xAA64 => "ARM64",
                0x01C4 => "ARM",
                _ => $"0x{machine:X4}",
            },
            // **0 は「時刻なし」。** 再現ビルドでは意図的に 0 や固定値が入る。
            BuiltAt = timestamp == 0
                ? null
                : DateTimeOffset.FromUnixTimeSeconds(timestamp),
            HasSignature = certificateSize > 0,
        };
    }

    /// <summary>RVA をファイル内の位置に直す。セクションの外なら null。</summary>
    private static int? Locate(byte[] file, int sections, int count, uint rva)
    {
        for (var i = 0; i < count; i++)
        {
            var entry = sections + i * 40;
            if (entry + 40 > file.Length)
            {
                return null;
            }
            var virtualSize = ReadUInt32(file, entry + 8);
            var virtualAddress = ReadUInt32(file, entry + 12);
            var rawSize = ReadUInt32(file, entry + 16);
            var rawOffset = ReadUInt32(file, entry + 20);

            // **仮想の大きさと生の大きさの小さい方で判定する。** 仮想の方が
            // 大きい（0 で埋める分）ことがあり、そこを指すとファイルの外へ出る。
            var extent = Math.Min(virtualSize == 0 ? rawSize : virtualSize, rawSize);
            if (rva >= virtualAddress && rva < virtualAddress + extent)
            {
                return (int)(rawOffset + (rva - virtualAddress));
            }
        }
        return null;
    }

    private const int TypeVersion = 16;   // RT_VERSION

    /// <summary>
    /// リソースの木を辿って VS_VERSIONINFO の場所を探す。
    ///
    /// 木は「種類 → 名前 → 言語」の 3 段。**言語は最初に見つかったものを使う。**
    /// 多言語のファイルでは版だけ複数入っていることがあるが、数値は同じで、
    /// 違うのは説明文だけ。選ばせる意味が薄い。
    /// </summary>
    private static (int Offset, int Length)? FindVersionResource(
        byte[] file, int root, uint rootRva, int sections, int sectionCount)
    {
        var typeEntry = FindEntry(file, root, root, TypeVersion);
        if (typeEntry is not { IsDirectory: true } type)
        {
            return null;
        }

        var nameEntry = FirstEntry(file, root + type.Offset);
        if (nameEntry is not { IsDirectory: true } name)
        {
            return null;
        }

        var languageEntry = FirstEntry(file, root + name.Offset);
        if (languageEntry is not { IsDirectory: false } language)
        {
            return null;
        }

        // 葉は data entry を指す。RVA と大きさが入っている。
        var dataEntry = root + language.Offset;
        if (dataEntry + 8 > file.Length)
        {
            return null;
        }
        var dataRva = ReadUInt32(file, dataEntry);
        var size = (int)ReadUInt32(file, dataEntry + 4);

        var offset = Locate(file, sections, sectionCount, dataRva);
        return offset is { } start && start + size <= file.Length ? (start, size) : null;
    }

    private readonly record struct ResourceEntry(int Offset, bool IsDirectory);

    /// <summary>その番号の項目を探す。</summary>
    private static ResourceEntry? FindEntry(byte[] file, int directory, int root, int id)
    {
        if (directory + 16 > file.Length)
        {
            return null;
        }
        var named = ReadUInt16(file, directory + 12);
        var numbered = ReadUInt16(file, directory + 14);
        var first = directory + 16;

        // 名前つきの項目を飛ばして番号つきだけを見る。
        for (var i = 0; i < numbered; i++)
        {
            var entry = first + (named + i) * 8;
            if (entry + 8 > file.Length)
            {
                return null;
            }
            if (ReadUInt32(file, entry) == (uint)id)
            {
                var value = ReadUInt32(file, entry + 4);
                return new ResourceEntry((int)(value & 0x7FFFFFFF), (value & 0x80000000) != 0);
            }
        }
        return null;
    }

    private static ResourceEntry? FirstEntry(byte[] file, int directory)
    {
        if (directory + 16 > file.Length)
        {
            return null;
        }
        var total = ReadUInt16(file, directory + 12) + ReadUInt16(file, directory + 14);
        if (total == 0)
        {
            return null;
        }
        var entry = directory + 16;
        if (entry + 8 > file.Length)
        {
            return null;
        }
        var value = ReadUInt32(file, entry + 4);
        return new ResourceEntry((int)(value & 0x7FFFFFFF), (value & 0x80000000) != 0);
    }

    /// <summary>
    /// VS_VERSIONINFO を読む。
    ///
    /// 構造は「長さ・値の長さ・種別・鍵（UTF-16）・詰め物・値・子」の入れ子。
    /// **どの段でも 4 バイト境界に揃える。** ここを守らないと、鍵の長さが
    /// 奇数のときに次の項目の頭がずれ、以降が全部化ける。
    /// </summary>
    private static (string FileVersion, string ProductVersion) ParseVersionInfo(
        byte[] file, int start, int length, Dictionary<string, string> strings)
    {
        var end = Math.Min(start + length, file.Length);
        var fileVersion = string.Empty;
        var productVersion = string.Empty;

        var (valueLength, key, valueAt, next) = ReadNode(file, start, end);
        if (key != "VS_VERSION_INFO")
        {
            return (fileVersion, productVersion);
        }

        // VS_FIXEDFILEINFO。署名 0xFEEF04BD で始まる。
        if (valueLength >= 52 && valueAt + 52 <= end
            && ReadUInt32(file, valueAt) == 0xFEEF04BD)
        {
            fileVersion = Version(file, valueAt + 8);
            productVersion = Version(file, valueAt + 16);
        }

        // 子を辿る。StringFileInfo の中に「言語 → 鍵と値」が入っている。
        var child = Align(valueAt + valueLength);
        while (child + 6 < next && child + 6 < end)
        {
            var (_, childKey, _, childNext) = ReadNode(file, child, end);
            if (childNext <= child)
            {
                break;    // 長さ 0。**進まない場合は必ず抜ける**（無限に回る）
            }
            if (childKey == "StringFileInfo")
            {
                ReadStringTables(file, child, childNext, strings);
            }
            child = Align(childNext);
        }

        return (fileVersion, productVersion);
    }

    private static void ReadStringTables(
        byte[] file, int start, int end, Dictionary<string, string> strings)
    {
        var (valueLength, _, valueAt, _) = ReadNode(file, start, end);
        var table = Align(valueAt + valueLength);

        while (table + 6 < end)
        {
            var (_, _, tableValueAt, tableNext) = ReadNode(file, table, end);
            if (tableNext <= table)
            {
                break;
            }

            var item = Align(tableValueAt);
            while (item + 6 < tableNext)
            {
                var (itemValueLength, itemKey, itemValueAt, itemNext) = ReadNode(file, item, tableNext);
                if (itemNext <= item)
                {
                    break;
                }
                if (itemKey.Length > 0 && itemValueLength > 0)
                {
                    // 値の長さは**文字数**（バイト数ではない）。
                    var bytes = Math.Min(itemValueLength * 2, end - itemValueAt);
                    if (bytes > 0)
                    {
                        strings[itemKey] = Encoding.Unicode
                            .GetString(file, itemValueAt, bytes).TrimEnd('\0');
                    }
                }
                item = Align(itemNext);
            }
            table = Align(tableNext);
        }
    }

    private static (int ValueLength, string Key, int ValueAt, int Next) ReadNode(
        byte[] file, int at, int end)
    {
        if (at + 6 > end)
        {
            return (0, string.Empty, at, at);
        }
        var length = ReadUInt16(file, at);
        var valueLength = ReadUInt16(file, at + 2);

        var keyAt = at + 6;
        var keyEnd = keyAt;
        while (keyEnd + 1 < end && ReadUInt16(file, keyEnd) != 0)
        {
            keyEnd += 2;
        }
        var key = Encoding.Unicode.GetString(file, keyAt, keyEnd - keyAt);

        // 鍵の終端（2 バイト）を越えてから 4 バイト境界へ揃える。
        var valueAt = Align(keyEnd + 2);
        return (valueLength, key, valueAt, length == 0 ? at : at + length);
    }

    private static int Align(int offset) => (offset + 3) & ~3;

    /// <summary>2 つの 32 ビットを <c>a.b.c.d</c> にする。上位が先。</summary>
    private static string Version(byte[] file, int at)
    {
        var most = ReadUInt32(file, at);
        var least = ReadUInt32(file, at + 4);
        return string.Create(CultureInfo.InvariantCulture,
            $"{most >> 16}.{most & 0xFFFF}.{least >> 16}.{least & 0xFFFF}");
    }

    private static ushort ReadUInt16(byte[] file, int at)
        => at + 2 <= file.Length ? BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(at)) : (ushort)0;

    private static uint ReadUInt32(byte[] file, int at)
        => at + 4 <= file.Length ? BinaryPrimitives.ReadUInt32LittleEndian(file.AsSpan(at)) : 0;

    private static int ReadInt32(byte[] file, int at) => (int)ReadUInt32(file, at);

    /// <summary>
    /// 出す項目と、その順序。
    ///
    /// **並べる順を決めておく。** 辞書の順にすると、ファイルごとに項目の順が
    /// 変わって左右が見比べられない。
    /// </summary>
    private static readonly string[] Order =
    [
        "FileVersion（数値）", "ProductVersion（数値）", "アーキテクチャ", "ビルド時刻", "署名",
        "FileVersion", "ProductVersion", "FileDescription", "ProductName",
        "CompanyName", "LegalCopyright", "OriginalFilename", "InternalName", "Comments",
    ];

    public static IReadOnlyList<VersionDifference> Compare(
        ExecutableVersion left, ExecutableVersion right)
    {
        var result = new List<VersionDifference>();

        void Add(string key, string? a, string? b)
        {
            if (a is not null || b is not null)
            {
                result.Add(new VersionDifference(key, a, b));
            }
        }

        Add("FileVersion（数値）", Blank(left.FileVersion), Blank(right.FileVersion));
        Add("ProductVersion（数値）", Blank(left.ProductVersion), Blank(right.ProductVersion));
        Add("アーキテクチャ", Blank(left.Machine), Blank(right.Machine));
        // **UTC と明示する。** PE のタイムスタンプは UTC で入っており、
        // 手元の時刻に直すとビルド機の時計と食い違って見える。
        // なお決定論的ビルドでは、ここに時刻ではなく内容のハッシュが入る。
        Add("ビルド時刻",
            left.BuiltAt?.ToString("yyyy-MM-dd HH:mm:ss 'UTC'"),
            right.BuiltAt?.ToString("yyyy-MM-dd HH:mm:ss 'UTC'"));
        Add("署名", left.HasSignature ? "あり" : "なし", right.HasSignature ? "あり" : "なし");

        // 決めた順で並べ、残りは名前順で後ろに付ける。
        var seen = new HashSet<string>(Order, StringComparer.Ordinal);
        foreach (var key in Order.Where(k => !k.EndsWith("）", StringComparison.Ordinal)
                                             && k is not ("アーキテクチャ" or "ビルド時刻" or "署名")))
        {
            Add(key, left.Get(key), right.Get(key));
        }
        foreach (var key in left.Strings.Keys.Concat(right.Strings.Keys)
                     .Where(k => !seen.Contains(k)).Distinct().Order(StringComparer.Ordinal))
        {
            Add(key, left.Get(key), right.Get(key));
        }

        return result;
    }

    private static string? Blank(string value) => value.Length == 0 ? null : value;

    public static string Format(IReadOnlyList<VersionDifference> differences, bool differencesOnly)
    {
        var text = new StringBuilder();
        var differing = differences.Count(d => !d.IsSame);

        foreach (var difference in differences)
        {
            if (differencesOnly && difference.IsSame)
            {
                continue;
            }
            var mark = difference.IsSame ? ' ' : '~';
            text.AppendLine($"{mark} {difference.Key,-24} {difference.Left ?? "（無し）"}"
                + $"  |  {difference.Right ?? "（無し）"}");
        }

        text.AppendLine("---");
        text.AppendLine(differing == 0
            ? "バージョン情報は同じです。"
            : $"{differing} 項目が違います。");
        return text.ToString();
    }
}
