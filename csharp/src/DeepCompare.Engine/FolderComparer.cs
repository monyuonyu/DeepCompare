using System.Text.RegularExpressions;

namespace DeepCompare.Engine;

public enum EntryStatus
{
    /// <summary>両側にあり、内容も同じ。</summary>
    Identical,
    /// <summary>両側にあるが内容が違う。</summary>
    Different,
    LeftOnly,
    RightOnly,
}

/// <summary>フォルダー比較の 1 行。ディレクトリ自身も 1 行として出る。</summary>
public sealed record FolderEntry(
    string RelativePath,
    string Name,
    int Depth,
    bool IsDirectory,
    EntryStatus Status,
    long? LeftSize,
    long? RightSize,
    DateTime? LeftModified,
    DateTime? RightModified)
{
    /// <summary>読み取れなかった場合の理由。権限やリンク切れなど。</summary>
    public string? Error { get; init; }
}

public sealed record FolderStats(
    int Identical,
    int Different,
    int LeftOnly,
    int RightOnly,
    int Directories,
    int Errors);

public sealed record FolderComparison(IReadOnlyList<FolderEntry> Entries, FolderStats Stats);

/// <summary>フォルダー比較で「同じ」をどう決めるか。</summary>
public enum FolderComparisonMode
{
    /// <summary>中身を読んで比べる。正確だが、数が多いと時間がかかる。</summary>
    Content,

    /// <summary>
    /// 大きさと更新時刻だけで比べる。中身は読まない。数万個を相手にするときや、
    /// 低速な媒体を跨ぐときはこちらでないと終わらない。
    /// </summary>
    SizeAndTimestamp,
}

/// <summary>
/// 名前による絞り込み。ワイルドカードは <c>*</c> と <c>?</c> のみ。
///
/// <see cref="Include"/> は<b>ファイルにだけ</b>効く。ディレクトリにも効かせると
/// <c>*.cs</c> のような指定で全ディレクトリが弾かれ、再帰が入口で止まる。
/// <see cref="Exclude"/> は両方に効く。
/// </summary>
public sealed record NameFilter(
    IReadOnlyList<string>? Include = null,
    IReadOnlyList<string>? Exclude = null)
{
    public static readonly NameFilter Any = new();

    private readonly Regex? _include = Build(Include);
    private readonly Regex? _exclude = Build(Exclude);

    public bool FiltersNothing => _include is null && _exclude is null;

    /// <param name="isDirectory">Include を適用しない側かどうか。</param>
    public bool Allows(string name, bool isDirectory)
    {
        if (_exclude is not null && _exclude.IsMatch(name))
        {
            return false;
        }
        if (isDirectory || _include is null)
        {
            return true;
        }
        return _include.IsMatch(name);
    }

    private static Regex? Build(IReadOnlyList<string>? patterns)
    {
        if (patterns is null || patterns.Count == 0)
        {
            return null;
        }

        var parts = patterns.Select(p =>
            "(?:^" + Regex.Escape(p).Replace("\\*", ".*").Replace("\\?", ".") + "$)");
        return new Regex(
            string.Join('|', parts), RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
    }
}

public sealed record FolderCompareOptions
{
    /// <summary>
    /// 走査から外す名前。生成物や版管理の内部を比べても意味が無く、数だけ膨れて
    /// 本当に見たい差分が埋もれる。
    /// </summary>
    public IReadOnlySet<string> ExcludedNames { get; init; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        ".git", ".svn", ".hg", "node_modules", "target", "bin", "obj",
        "__pycache__", ".venv", "venv", ".idea", ".vs", "dist", "build",
    };

    public bool Recursive { get; init; } = true;

    /// <summary>同じ内容の項目を結果に含めるか。既定では含めて、表示側で絞る。</summary>
    public bool IncludeIdentical { get; init; } = true;

    /// <summary>名前による絞り込み。既定は素通し。</summary>
    public NameFilter Filter { get; init; } = NameFilter.Any;

    /// <summary>同じかどうかの判定方法。</summary>
    public FolderComparisonMode Mode { get; init; } = FolderComparisonMode.Content;

    /// <summary>
    /// ファイル名の突き合わせ方。
    ///
    /// 既定は大小文字を無視する（従来どおり）。**Unicode 正規化は既定では揃えない。**
    /// macOS を経由したファイルを比べるときだけ NormalizeUnicode を立てる。
    /// 常に揃えると、正規化だけが違う 2 つのファイルが同じ場所に共存している状態を
    /// 見落とす（Linux では実際に共存できる）。
    /// </summary>
    public NameMatching Matching { get; init; } = new NameMatching(IgnoreCase: true);

    /// <summary>
    /// 更新時刻の差をどこまで同じとみなすか（秒）。
    ///
    /// FAT は 2 秒刻みでしか時刻を持てず、NTFS との間でコピーすると 1〜2 秒ずれる。
    /// 媒体を跨ぐ比較では、この程度を許さないと全件が「違う」になる。
    /// </summary>
    public double TimestampToleranceSeconds { get; init; }

    /// <summary>
    /// ちょうど 1 時間のずれを同じとみなす。夏時間の切り替えを挟むと、同じファイルの
    /// 時刻が 1 時間ずれて記録される環境がある。
    /// </summary>
    public bool IgnoreDaylightSavingOffset { get; init; }

    /// <summary>指定した大きさ未満のファイルを対象から外す。0 なら制限しない。</summary>
    public long MinimumSize { get; init; }

    /// <summary>指定した大きさを超えるファイルを対象から外す。0 なら制限しない。</summary>
    public long MaximumSize { get; init; }
}

/// <summary>
/// フォルダーどうしの比較。
///
/// 相対パスで対応付け、内容が同じかどうかだけをここで決める。どこがどう違うかは
/// 行単位の比較（<see cref="DiffComparer"/>）の仕事なので、ここでは踏み込まない。
/// フォルダー全体に対して意味的な比較を走らせると、見たい一覧が出るまでに
/// 何分もかかることになる。
/// </summary>
public static class FolderComparer
{
    /// <summary>内容比較で一度に読む量。</summary>
    private const int ChunkSize = 64 * 1024;

    public static FolderComparison Compare(
        string leftRoot,
        string rightRoot,
        FolderCompareOptions? options = null,
        Action<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new FolderCompareOptions();
        var entries = new List<FolderEntry>();
        var counters = new Counters();

        Walk(leftRoot, rightRoot, string.Empty, 0, options, entries, counters, progress, cancellationToken);

        return new FolderComparison(
            entries,
            new FolderStats(
                counters.Identical, counters.Different, counters.LeftOnly,
                counters.RightOnly, counters.Directories, counters.Errors));
    }

    private sealed class Counters
    {
        public int Identical, Different, LeftOnly, RightOnly, Directories, Errors;
    }

    private static void Walk(
        string? leftDir,
        string? rightDir,
        string relative,
        int depth,
        FolderCompareOptions options,
        List<FolderEntry> entries,
        Counters counters,
        Action<string>? progress,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (relative.Length > 0)
        {
            progress?.Invoke(relative);
        }

        var leftNames = ListNames(leftDir, options, out var leftError);
        var rightNames = ListNames(rightDir, options, out var rightError);
        if (leftError is not null || rightError is not null)
        {
            counters.Errors++;
            entries.Add(new FolderEntry(
                relative, Path.GetFileName(relative), depth, true,
                EntryStatus.Different, null, null, null, null)
            {
                Error = leftError ?? rightError,
            });
            return;
        }

        // 名前順に並べる。左右の一覧を突き合わせるので、順序が安定していないと読めない。
        // 突き合わせの鍵と表示する名前は別。鍵は正規化した形、表示は実際の名前。
        var allKeys = leftNames.Keys.Union(rightNames.Keys, StringComparer.Ordinal)
            .OrderBy(n => n, StringComparer.OrdinalIgnoreCase);

        foreach (var key in allKeys)
        {
            cancellationToken.ThrowIfCancellationRequested();
            leftNames.TryGetValue(key, out var left);
            rightNames.TryGetValue(key, out var right);
            // 表示は実際のファイル名を使う。鍵をそのまま出すと、大小文字を無視した
            // ときに片側の綴りだけが出てしまい、どちらの名前なのか分からなくなる。
            var name = left?.Name ?? right?.Name ?? key;
            var childRelative = relative.Length == 0 ? name : $"{relative}/{name}";

            var leftIsDir = left?.Attributes.HasFlag(FileAttributes.Directory) ?? false;
            var rightIsDir = right?.Attributes.HasFlag(FileAttributes.Directory) ?? false;

            if (leftIsDir || rightIsDir)
            {
                counters.Directories++;
                var status = (left, right) switch
                {
                    (not null, not null) => EntryStatus.Identical,
                    (not null, null) => EntryStatus.LeftOnly,
                    (null, not null) => EntryStatus.RightOnly,
                    _ => EntryStatus.Identical,
                };
                entries.Add(new FolderEntry(
                    childRelative, name, depth, true, status,
                    null, null, left?.LastWriteTime, right?.LastWriteTime));

                if (options.Recursive)
                {
                    Walk(
                        leftIsDir ? left!.FullName : null,
                        rightIsDir ? right!.FullName : null,
                        childRelative, depth + 1, options, entries, counters, progress, cancellationToken);
                }
                continue;
            }

            var leftFile = left as FileInfo;
            var rightFile = right as FileInfo;
            if (leftFile is not null && rightFile is not null)
            {
                string? error = null;
                bool same;
                try
                {
                    same = options.Mode == FolderComparisonMode.SizeAndTimestamp
                        ? leftFile.Length == rightFile.Length
                          && SameTimestamp(leftFile.LastWriteTime, rightFile.LastWriteTime, options)
                        : SameContent(leftFile, rightFile, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    // 読めないファイルで比較全体を止めない。
                    same = false;
                    error = ex.Message;
                    counters.Errors++;
                }

                if (same)
                {
                    counters.Identical++;
                    if (!options.IncludeIdentical)
                    {
                        continue;
                    }
                }
                else if (error is null)
                {
                    counters.Different++;
                }

                entries.Add(new FolderEntry(
                    childRelative, name, depth, false,
                    same ? EntryStatus.Identical : EntryStatus.Different,
                    leftFile.Length, rightFile.Length,
                    leftFile.LastWriteTime, rightFile.LastWriteTime)
                {
                    Error = error,
                });
            }
            else if (leftFile is not null)
            {
                counters.LeftOnly++;
                entries.Add(new FolderEntry(
                    childRelative, name, depth, false, EntryStatus.LeftOnly,
                    leftFile.Length, null, leftFile.LastWriteTime, null));
            }
            else if (rightFile is not null)
            {
                counters.RightOnly++;
                entries.Add(new FolderEntry(
                    childRelative, name, depth, false, EntryStatus.RightOnly,
                    null, rightFile.Length, null, rightFile.LastWriteTime));
            }
        }
    }

    private static bool WithinSizeLimits(long length, FolderCompareOptions options)
        => (options.MinimumSize <= 0 || length >= options.MinimumSize)
           && (options.MaximumSize <= 0 || length <= options.MaximumSize);

    /// <summary>
    /// 更新時刻が同じとみなせるか。許容誤差と、夏時間ぶんの 1 時間を見る。
    /// </summary>
    internal static bool SameTimestamp(DateTime left, DateTime right, FolderCompareOptions options)
    {
        var difference = Math.Abs((left - right).TotalSeconds);
        if (difference <= options.TimestampToleranceSeconds)
        {
            return true;
        }
        // ちょうど 1 時間ずれている場合も、許容誤差の範囲で同じとみなす。
        return options.IgnoreDaylightSavingOffset
               && Math.Abs(difference - 3600) <= options.TimestampToleranceSeconds;
    }

    private static Dictionary<string, FileSystemInfo> ListNames(
        string? directory, FolderCompareOptions options, out string? error)
    {
        error = null;
        // 鍵は options.Matching で作る。辞書の比較子で大小文字を吸収すると、
        // Unicode 正規化のような別の揃え方を足せない。
        var result = new Dictionary<string, FileSystemInfo>(StringComparer.Ordinal);
        if (directory is null)
        {
            return result;
        }
        try
        {
            foreach (var info in new DirectoryInfo(directory).EnumerateFileSystemInfos())
            {
                // シンボリックリンクは辿らない。辿ると循環して走査が終わらなくなる。
                if (info.LinkTarget is not null)
                {
                    continue;
                }
                if (options.ExcludedNames.Contains(info.Name))
                {
                    continue;
                }

                var isDirectory = info.Attributes.HasFlag(FileAttributes.Directory);
                if (!options.Filter.Allows(info.Name, isDirectory))
                {
                    continue;
                }
                if (!isDirectory && info is FileInfo file && !WithinSizeLimits(file.Length, options))
                {
                    continue;
                }

                result[options.Matching.Key(info.Name)] = info;
            }
        }
        catch (Exception ex)
        {
            error = ex.Message;
        }
        return result;
    }

    /// <summary>
    /// 内容が同じかどうか。サイズが違えば読まずに決まる。同サイズのときだけ
    /// 流し読みして、違いが出た時点で打ち切る。
    /// </summary>
    private static bool SameContent(FileInfo left, FileInfo right, CancellationToken cancellationToken)
    {
        if (left.Length != right.Length)
        {
            return false;
        }
        if (left.Length == 0)
        {
            return true;
        }

        using var leftStream = left.OpenRead();
        using var rightStream = right.OpenRead();
        var leftBuffer = new byte[ChunkSize];
        var rightBuffer = new byte[ChunkSize];

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var leftRead = leftStream.ReadAtLeast(leftBuffer, ChunkSize, throwOnEndOfStream: false);
            var rightRead = rightStream.ReadAtLeast(rightBuffer, ChunkSize, throwOnEndOfStream: false);
            if (leftRead != rightRead)
            {
                return false;
            }
            if (leftRead == 0)
            {
                return true;
            }
            if (!leftBuffer.AsSpan(0, leftRead).SequenceEqual(rightBuffer.AsSpan(0, rightRead)))
            {
                return false;
            }
        }
    }
}
