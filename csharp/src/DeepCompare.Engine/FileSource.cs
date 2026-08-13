namespace DeepCompare.Engine;

/// <summary>置いてあるもの 1 つ。</summary>
public sealed record RemoteEntry(
    /// <summary>根からの相対。区切りは常に <c>/</c>。</summary>
    string RelativePath,
    bool IsDirectory,
    long Size,
    DateTimeOffset? Modified)
{
    public string Name
    {
        get
        {
            var slash = RelativePath.LastIndexOf('/');
            return slash < 0 ? RelativePath : RelativePath[(slash + 1)..];
        }
    }
}

/// <summary>
/// 比較の対象を取ってくる先。
///
/// **なぜ抽象を挟むか。** 手元のフォルダー・書庫の中・SFTP・WebDAV・S3 を
/// 同じ形で扱えないと、比較の側がそれぞれを知ることになる。
///
/// **決めごと**:
/// - パスは根からの相対で、区切りは常に <c>/</c>。Windows の <c>\</c> は入口で直す
/// - <see cref="List"/> は再帰しない。1 段ずつ辿る（深い木で全部取りに行かない）
/// - 書き込みを持たない実装があってよい（読み取り専用の置き場は普通にある）
/// </summary>
public interface IFileSource : IDisposable
{
    /// <summary>人に見せる名前。画面の見出しに使う。</summary>
    string Display { get; }

    /// <summary>書き込めるか。**できないことを押せる状態で置かない**ための判断に使う。</summary>
    bool CanWrite { get; }

    /// <summary>その場所の直下にあるものを並べる。再帰はしない。</summary>
    IReadOnlyList<RemoteEntry> List(string relativePath, CancellationToken cancellationToken = default);

    byte[] Read(string relativePath, CancellationToken cancellationToken = default);

    void Write(string relativePath, byte[] content, CancellationToken cancellationToken = default);

    void Delete(string relativePath, CancellationToken cancellationToken = default);

    bool Exists(string relativePath, CancellationToken cancellationToken = default);
}

/// <summary>
/// 手元のフォルダー。
///
/// **既存の経路と同じ振る舞いにする。** ここが違うと、抽象を挟んだ時点で
/// 手元の比較の結果が変わってしまう。
/// </summary>
public sealed class LocalFileSource(string root) : IFileSource
{
    /// <summary>根。絶対パスに直しておく。</summary>
    public string Root { get; } = Path.GetFullPath(root);

    public string Display => Root;

    public bool CanWrite => true;

    public IReadOnlyList<RemoteEntry> List(
        string relativePath, CancellationToken cancellationToken = default)
    {
        var directory = Resolve(relativePath);
        if (!Directory.Exists(directory))
        {
            return [];
        }

        var result = new List<RemoteEntry>();
        foreach (var info in new DirectoryInfo(directory).EnumerateFileSystemInfos())
        {
            cancellationToken.ThrowIfCancellationRequested();

            // シンボリックリンクは辿らない。辿ると循環して終わらなくなる。
            if (info.LinkTarget is not null)
            {
                continue;
            }

            var isDirectory = info.Attributes.HasFlag(FileAttributes.Directory);
            result.Add(new RemoteEntry(
                Join(relativePath, info.Name),
                isDirectory,
                isDirectory ? 0 : ((FileInfo)info).Length,
                info.LastWriteTimeUtc));
        }
        return result;
    }

    public byte[] Read(string relativePath, CancellationToken cancellationToken = default)
        => File.ReadAllBytes(Resolve(relativePath));

    public void Write(string relativePath, byte[] content, CancellationToken cancellationToken = default)
    {
        var path = Resolve(relativePath);
        var directory = Path.GetDirectoryName(path);
        if (directory is { Length: > 0 })
        {
            Directory.CreateDirectory(directory);
        }
        File.WriteAllBytes(path, content);
    }

    public void Delete(string relativePath, CancellationToken cancellationToken = default)
    {
        var path = Resolve(relativePath);
        if (Directory.Exists(path))
        {
            Directory.Delete(path, recursive: true);
        }
        else if (File.Exists(path))
        {
            File.Delete(path);
        }
    }

    public bool Exists(string relativePath, CancellationToken cancellationToken = default)
    {
        var path = Resolve(relativePath);
        return File.Exists(path) || Directory.Exists(path);
    }

    public void Dispose()
    {
        // 手元のフォルダーには後始末が要らない。
    }

    /// <summary>
    /// 相対パスを実体の場所に直す。
    ///
    /// **根の外を指せないようにする。** <c>../</c> を含むパスをそのまま繋ぐと、
    /// 同期や削除が根の外へ届く。リモートから来た名前を信じてはいけない。
    /// </summary>
    private string Resolve(string relativePath)
    {
        var cleaned = relativePath.Replace('\\', '/').Trim('/');
        var combined = Path.GetFullPath(cleaned.Length == 0
            ? Root
            : Path.Combine(Root, cleaned.Replace('/', Path.DirectorySeparatorChar)));

        if (!combined.StartsWith(Root, StringComparison.Ordinal))
        {
            throw new UnauthorizedAccessException($"根の外を指しています: {relativePath}");
        }
        return combined;
    }

    internal static string Join(string parent, string name)
        => parent.Length == 0 ? name : $"{parent.TrimEnd('/')}/{name}";
}
