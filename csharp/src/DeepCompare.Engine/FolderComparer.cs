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
        var allNames = leftNames.Keys.Union(rightNames.Keys, StringComparer.OrdinalIgnoreCase)
            .OrderBy(n => n, StringComparer.OrdinalIgnoreCase);

        foreach (var name in allNames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            leftNames.TryGetValue(name, out var left);
            rightNames.TryGetValue(name, out var right);
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
                    same = SameContent(leftFile, rightFile, cancellationToken);
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

    private static Dictionary<string, FileSystemInfo> ListNames(
        string? directory, FolderCompareOptions options, out string? error)
    {
        error = null;
        var result = new Dictionary<string, FileSystemInfo>(StringComparer.OrdinalIgnoreCase);
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
                result[info.Name] = info;
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
