using System.Formats.Tar;
using System.IO.Compression;

namespace DeepCompare.Engine;

/// <summary>
/// 比較の対象となるフォルダー。実体のフォルダーか、書庫を展開した一時領域。
///
/// <see cref="IDisposable"/> なのは、展開した一時領域を確実に消すため。使い終わりが
/// 呼ぶ側から見えないと、比較のたびに一時領域が積もる。
/// </summary>
public sealed class ArchiveSource : IDisposable
{
    private readonly string? _temporary;

    private ArchiveSource(string path, string? temporary)
    {
        Path = path;
        _temporary = temporary;
    }

    /// <summary>走査に使うフォルダーのパス。</summary>
    public string Path { get; }

    /// <summary>書庫を展開したものか。</summary>
    public bool IsExtracted => _temporary is not null;

    /// <summary>
    /// リモートから取ってきたもの。**取ってきた結果を知らせるために持つ**
    /// （上限で切ったことを黙らないため）。
    /// </summary>
    public RemoteMirror? Mirror { get; private init; }

    /// <summary>人に見せる場所。リモートなら元の場所（一時領域の名前ではなく）。</summary>
    public string Display => Mirror?.Display ?? Path;

    /// <summary>拡張子から書庫と判断できるか。</summary>
    public static bool LooksLikeArchive(string path)
    {
        var name = path.ToLowerInvariant();
        return name.EndsWith(".zip")
            || name.EndsWith(".tar")
            || name.EndsWith(".tar.gz")
            || name.EndsWith(".tgz");
    }

    /// <summary>
    /// フォルダーならそのまま、書庫なら一時領域へ展開して返す。
    /// </summary>
    /// <param name="mirrorOptions">リモートのときの取り方。null なら既定。</param>
    /// <param name="progress">リモートから取っている最中の知らせ。</param>
    public static ArchiveSource Open(
        string path, MirrorOptions? mirrorOptions = null, Action<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // リモートは一時領域へ取ってきて、**普通のフォルダーとして**扱う。
        // 比較の側に「リモートかどうか」を知らせない。
        if (RemoteLocation.IsRemote(path))
        {
            using var source = RemoteLocation.Open(path);
            var mirror = RemoteMirror.Fetch(source, mirrorOptions, progress, cancellationToken);
            return new ArchiveSource(mirror.Path, null) { Mirror = mirror };
        }

        if (Directory.Exists(path))
        {
            return new ArchiveSource(path, null);
        }
        if (!File.Exists(path))
        {
            throw new DirectoryNotFoundException($"見つからない: {path}");
        }
        if (!LooksLikeArchive(path))
        {
            throw new NotSupportedException(
                $"対応していない形式: {path}（zip / tar / tar.gz / tgz）");
        }

        var destination = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "dc-archive-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(destination);

        try
        {
            Extract(path, destination);
        }
        catch
        {
            // 途中で失敗したら残さない。
            TryDelete(destination);
            throw;
        }

        return new ArchiveSource(destination, destination);
    }

    private static void Extract(string archive, string destination)
    {
        var name = archive.ToLowerInvariant();
        if (name.EndsWith(".zip"))
        {
            // 標準の展開は書庫内の "../" による外への書き出しを弾く。自前で
            // 展開先を組み立てないのは、その保護を捨てないため。
            ZipFile.ExtractToDirectory(archive, destination, overwriteFiles: true);
            return;
        }
        if (name.EndsWith(".tar"))
        {
            TarFile.ExtractToDirectory(archive, destination, overwriteFiles: true);
            return;
        }

        // .tar.gz / .tgz
        using var file = File.OpenRead(archive);
        using var gzip = new GZipStream(file, CompressionMode.Decompress);
        TarFile.ExtractToDirectory(gzip, destination, overwriteFiles: true);
    }

    public void Dispose()
    {
        Mirror?.Dispose();
        if (_temporary is not null)
        {
            TryDelete(_temporary);
        }
    }

    private static void TryDelete(string directory)
    {
        try
        {
            Directory.Delete(directory, recursive: true);
        }
        catch (Exception)
        {
            // 消せなくても比較の結果には関係ない。一時領域なので、いずれ環境側が片付ける。
        }
    }
}
