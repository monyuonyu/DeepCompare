using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace DeepCompare.Engine;

/// <summary>写し取ったファイル 1 件。</summary>
public sealed record SnapshotEntry(
    string RelativePath,
    bool IsDirectory,
    long Size,
    DateTime Modified)
{
    /// <summary>
    /// 中身の指紋。取らなかった場合は null。
    ///
    /// **時刻と大きさだけでは足りない。** 中身を変えて時刻を戻す（あるいは
    /// 同じ秒の内に書き換える）と、両方とも同じままになる。指紋があれば
    /// そこを見分けられる。
    /// </summary>
    public string? Hash { get; init; }
}

/// <summary>ある時点のフォルダーの姿。</summary>
public sealed record Snapshot(
    string Root,
    DateTimeOffset TakenAt,
    IReadOnlyList<SnapshotEntry> Entries)
{
    /// <summary>指紋を取ってあるか。取っていなければ中身の変化は見分けられない。</summary>
    public bool HasHashes => Entries.Any(e => e.Hash is not null);

    public int FileCount => Entries.Count(e => !e.IsDirectory);
    public int DirectoryCount => Entries.Count(e => e.IsDirectory);
}

/// <summary>
/// フォルダーの状態を写し取り、後の時点と比べる（BC の Snapshot に当たる）。
///
/// **何のためにあるか。** インストーラーが何を置いたか、ビルドが何を作ったか、
/// 実行が何を書き換えたか — どれも「前」と「後」を比べたいのに、「前」は
/// もう手元に無い。写しを取っておけば後から比べられる。
///
/// 中身は保存しない。**保存すると容量がフォルダーと同じだけ要る。**
/// 保存するのは名前・大きさ・時刻・指紋だけで、それで「何が変わったか」までは
/// 分かる（「どう変わったか」は元のファイルが要る）。
/// </summary>
public static class Snapshots
{
    /// <summary>写しの書式の版。読むときに食い違いを弾く。</summary>
    private const string Header = "deepcompare-snapshot\tv1";

    /// <summary>
    /// 写しを取る。
    ///
    /// <paramref name="withHashes"/> を立てると中身を読んで指紋を取る。
    /// **既定は取らない。** 数万ファイルを全部読むと時間がかかるうえ、
    /// 用途によっては大きさと時刻で足りる。
    /// </summary>
    public static Snapshot Take(
        string root,
        bool withHashes = false,
        NameFilter? filter = null,
        CancellationToken cancellationToken = default)
    {
        var entries = new List<SnapshotEntry>();
        var full = Path.GetFullPath(root);
        Walk(full, full, entries, withHashes, filter, cancellationToken);

        // 並びを決めておく。**取るたびに順序が変わると、書き出した写しの
        // 差分がその都度騒がしくなる**（写し自体をバージョン管理に入れる人がいる）。
        entries.Sort((a, b) => string.CompareOrdinal(a.RelativePath, b.RelativePath));

        return new Snapshot(full, DateTimeOffset.Now, entries);
    }

    private static void Walk(
        string root, string directory, List<SnapshotEntry> into,
        bool withHashes, NameFilter? filter, CancellationToken cancellationToken)
    {
        IEnumerable<FileSystemInfo> children;
        try
        {
            children = new DirectoryInfo(directory).EnumerateFileSystemInfos();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // 読めない場所は飛ばす。**そこで止めない。** 1 か所の権限で
            // 写し全体が取れなくなるのは割に合わない。
            return;
        }

        foreach (var info in children)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // シンボリックリンクは辿らない。辿ると循環して終わらなくなる。
            if (info.LinkTarget is not null)
            {
                continue;
            }

            var isDirectory = info.Attributes.HasFlag(FileAttributes.Directory);
            if (filter is not null && !filter.Allows(info.Name, isDirectory))
            {
                continue;
            }

            var relative = Path.GetRelativePath(root, info.FullName).Replace('\\', '/');

            if (isDirectory)
            {
                into.Add(new SnapshotEntry(relative, true, 0, info.LastWriteTimeUtc));
                Walk(root, info.FullName, into, withHashes, filter, cancellationToken);
                continue;
            }

            var file = (FileInfo)info;
            string? hash = null;
            if (withHashes)
            {
                try
                {
                    using var stream = file.OpenRead();
                    hash = Convert.ToHexString(SHA256.HashData(stream));
                }
                catch (Exception error) when (error is IOException or UnauthorizedAccessException)
                {
                    // 読めなかった。**「指紋なし」として残す。** 項目ごと落とすと
                    // 後で比べたときに「消えた」ことになってしまう。
                }
            }

            into.Add(new SnapshotEntry(relative, false, file.Length, file.LastWriteTimeUtc)
            {
                Hash = hash,
            });
        }
    }

    /// <summary>
    /// 写しを文字列にする。
    ///
    /// **人が読める行の並びにする。** 独自の詰め方にすると、道具が無いと中身を
    /// 確かめられない。タブ区切りなら grep も diff もそのまま効く。
    /// </summary>
    public static string Save(Snapshot snapshot)
    {
        var text = new StringBuilder();
        text.Append(Header).Append('\n');
        text.Append("root\t").Append(snapshot.Root).Append('\n');
        text.Append("taken\t").Append(snapshot.TakenAt.ToString("o", CultureInfo.InvariantCulture)).Append('\n');
        text.Append("---\n");

        foreach (var entry in snapshot.Entries)
        {
            text.Append(entry.IsDirectory ? 'd' : 'f').Append('\t')
                .Append(entry.Size.ToString(CultureInfo.InvariantCulture)).Append('\t')
                .Append(entry.Modified.ToString("o", CultureInfo.InvariantCulture)).Append('\t')
                .Append(entry.Hash ?? "-").Append('\t')
                // **名前を最後に置く。** タブを含む名前があっても、後ろを
                // 全部名前として読めば壊れない。
                .Append(entry.RelativePath).Append('\n');
        }
        return text.ToString();
    }

    public static Snapshot Load(string text)
    {
        var lines = text.Split('\n');
        if (lines.Length == 0 || lines[0].TrimEnd('\r') != Header)
        {
            throw new InvalidDataException("DeepCompare の写しではありません。");
        }

        var root = string.Empty;
        var taken = default(DateTimeOffset);
        var index = 1;

        for (; index < lines.Length; index++)
        {
            var line = lines[index].TrimEnd('\r');
            if (line == "---")
            {
                index++;
                break;
            }
            var tab = line.IndexOf('\t');
            if (tab < 0)
            {
                continue;
            }
            switch (line[..tab])
            {
                case "root": root = line[(tab + 1)..]; break;
                case "taken":
                    DateTimeOffset.TryParse(line[(tab + 1)..], CultureInfo.InvariantCulture,
                        DateTimeStyles.RoundtripKind, out taken);
                    break;
            }
        }

        var entries = new List<SnapshotEntry>();
        for (; index < lines.Length; index++)
        {
            var line = lines[index].TrimEnd('\r');
            if (line.Length == 0)
            {
                continue;
            }
            // 名前を最後に置いてあるので、区切りは 4 つまでで止める。
            var fields = line.Split('\t', 5);
            if (fields.Length < 5)
            {
                continue;
            }

            entries.Add(new SnapshotEntry(
                fields[4],
                fields[0] == "d",
                long.TryParse(fields[1], CultureInfo.InvariantCulture, out var size) ? size : 0,
                DateTime.TryParse(fields[2], CultureInfo.InvariantCulture,
                    DateTimeStyles.RoundtripKind, out var modified) ? modified : default)
            {
                Hash = fields[3] == "-" ? null : fields[3],
            });
        }

        return new Snapshot(root, taken, entries);
    }

    /// <summary>
    /// 2 つの写しを比べる。
    ///
    /// 結果はフォルダー比較と同じ型で返す。**画面もレポートもそのまま使える。**
    /// 「写し同士の比較」に別の見せ方を用意すると、覚えることが増える。
    /// </summary>
    public static FolderComparison Compare(
        Snapshot before, Snapshot after, FolderCompareOptions? options = null)
    {
        options ??= new FolderCompareOptions();

        var left = before.Entries.ToDictionary(e => options.Matching.Key(e.RelativePath), e => e);
        var right = after.Entries.ToDictionary(e => options.Matching.Key(e.RelativePath), e => e);

        var keys = new List<string>(left.Keys);
        foreach (var key in right.Keys)
        {
            if (!left.ContainsKey(key))
            {
                keys.Add(key);
            }
        }
        keys.Sort(StringComparer.Ordinal);

        var entries = new List<FolderEntry>();
        var identical = 0;
        var different = 0;
        var leftOnly = 0;
        var rightOnly = 0;
        var directories = 0;

        foreach (var key in keys)
        {
            var a = left.GetValueOrDefault(key);
            var b = right.GetValueOrDefault(key);
            var sample = a ?? b!;
            var path = sample.RelativePath;
            var name = path.Contains('/') ? path[(path.LastIndexOf('/') + 1)..] : path;
            var depth = path.Count(c => c == '/');

            if (sample.IsDirectory)
            {
                directories++;
            }

            EntryStatus status;
            if (a is null)
            {
                status = EntryStatus.RightOnly;
                rightOnly++;
            }
            else if (b is null)
            {
                status = EntryStatus.LeftOnly;
                leftOnly++;
            }
            else if (sample.IsDirectory)
            {
                status = EntryStatus.Identical;
                identical++;
            }
            else if (SameContent(a, b, options))
            {
                status = EntryStatus.Identical;
                identical++;
            }
            else
            {
                status = EntryStatus.Different;
                different++;
            }

            entries.Add(new FolderEntry(
                path, name, depth, sample.IsDirectory, status,
                a?.Size, b?.Size, a?.Modified, b?.Modified));
        }

        return new FolderComparison(entries,
            new FolderStats(identical, different, leftOnly, rightOnly, directories, 0));
    }

    /// <summary>
    /// 中身が同じか。
    ///
    /// **指紋が両方にあれば、それだけで決める。** 大きさと時刻が一致していても
    /// 中身が違うことはある（同じ秒の内の書き換え、時刻を戻す操作）。
    /// 指紋が無い側があるときだけ、大きさと時刻に落とす。
    /// </summary>
    private static bool SameContent(
        SnapshotEntry a, SnapshotEntry b, FolderCompareOptions options)
    {
        if (a.Hash is not null && b.Hash is not null)
        {
            return string.Equals(a.Hash, b.Hash, StringComparison.Ordinal);
        }
        if (a.Size != b.Size)
        {
            return false;
        }
        var difference = Math.Abs((a.Modified - b.Modified).TotalSeconds);
        return difference <= options.TimestampToleranceSeconds
            || (options.IgnoreDaylightSavingOffset && Math.Abs(difference - 3600) <= options.TimestampToleranceSeconds);
    }
}
