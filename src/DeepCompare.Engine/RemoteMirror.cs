namespace DeepCompare.Engine;

/// <summary>取ってきた結果の要約。</summary>
public sealed record MirrorResult(int Files, long Bytes, IReadOnlyList<string> Skipped)
{
    /// <summary>取ってこなかったものがあるか。**あれば必ず言う。**</summary>
    public bool HasSkipped => Skipped.Count > 0;
}

public sealed record MirrorOptions
{
    /// <summary>名前による絞り込み。取ってくる量はここで決まる。</summary>
    public NameFilter Filter { get; init; } = NameFilter.Any;

    /// <summary>
    /// 1 ファイルの上限。超えるものは取ってこない。
    ///
    /// **既定を無制限にしない。** リモートに 2GB の記録が 1 つあるだけで、
    /// 比較を始める前に回線と時間を使い切る。
    /// </summary>
    public long MaximumFileSize { get; init; } = 32 * 1024 * 1024;

    /// <summary>
    /// 全体の上限。
    ///
    /// **一時領域を埋め尽くさない。** 相手の大きさは事前に分からないので、
    /// ここで止められないと、途中で disk full になって何も残らない。
    /// </summary>
    public long MaximumTotalSize { get; init; } = 512 * 1024 * 1024;

    /// <summary>辿る深さの上限。循環する置き場を相手にしても終わる。</summary>
    public int MaximumDepth { get; init; } = 32;
}

/// <summary>
/// リモートの中身を一時領域へ取ってきて、**普通のフォルダーとして**扱えるようにする。
///
/// **比較の側を書き換えない。** 書庫を扱う <see cref="ArchiveSource"/> と同じ
/// 考え方で、既存の比較・同期・レポート・画面がそのまま効く。
/// 比較エンジンに「リモートかどうか」を知らせると、そこら中に分岐が増える。
///
/// 引き換えに、対象を全部取ってくる。**だから絞り込みと上限を必ず持つ。**
/// 実際の用途（サーバー上の設定ファイルを手元と比べる）では、数十〜数百
/// ファイルが相手なので、これで足りる。
/// </summary>
public sealed class RemoteMirror : IDisposable
{
    private readonly string _temporary;

    private RemoteMirror(string path, string display, MirrorResult result)
    {
        _temporary = path;
        Path = path;
        Display = display;
        Result = result;
    }

    /// <summary>走査に使うフォルダーのパス。</summary>
    public string Path { get; }

    /// <summary>元の場所。画面の見出しには**こちらを出す**（一時領域の名前ではなく）。</summary>
    public string Display { get; }

    public MirrorResult Result { get; }

    /// <summary>
    /// 取ってくる。
    ///
    /// <paramref name="progress"/> には、いま取っているものの相対パスを渡す。
    /// **黙って何分も待たせない。**
    /// </summary>
    public static RemoteMirror Fetch(
        IFileSource source,
        MirrorOptions? options = null,
        Action<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new MirrorOptions();

        var temporary = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "dc-remote-" + Guid.NewGuid().ToString("N")[..12]);
        Directory.CreateDirectory(temporary);

        var files = 0;
        long bytes = 0;
        var skipped = new List<string>();

        try
        {
            Walk(source, temporary, string.Empty, 0, options, progress,
                ref files, ref bytes, skipped, cancellationToken);
        }
        catch
        {
            // 途中で失敗したら一時領域を残さない。**中途半端な写しを
            // 「リモートの姿」として比べさせない。**
            TryDelete(temporary);
            throw;
        }

        return new RemoteMirror(temporary, source.Display,
            new MirrorResult(files, bytes, skipped));
    }

    private static void Walk(
        IFileSource source, string root, string relativePath, int depth,
        MirrorOptions options, Action<string>? progress,
        ref int files, ref long bytes, List<string> skipped,
        CancellationToken cancellationToken)
    {
        if (depth > options.MaximumDepth)
        {
            skipped.Add($"{relativePath}（深すぎます）");
            return;
        }

        foreach (var entry in source.List(relativePath, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (!options.Filter.Allows(entry.Name, entry.IsDirectory))
            {
                continue;
            }

            var target = System.IO.Path.Combine(
                root, entry.RelativePath.Replace('/', System.IO.Path.DirectorySeparatorChar));

            if (entry.IsDirectory)
            {
                Directory.CreateDirectory(target);
                Walk(source, root, entry.RelativePath, depth + 1, options, progress,
                    ref files, ref bytes, skipped, cancellationToken);
                continue;
            }

            if (entry.Size > options.MaximumFileSize)
            {
                skipped.Add($"{entry.RelativePath}（{entry.Size:N0} バイト。1 件の上限を超えます）");
                continue;
            }
            if (bytes + entry.Size > options.MaximumTotalSize)
            {
                skipped.Add($"{entry.RelativePath}（全体の上限に達しました）");
                continue;
            }

            progress?.Invoke(entry.RelativePath);

            var directory = System.IO.Path.GetDirectoryName(target);
            if (directory is { Length: > 0 })
            {
                Directory.CreateDirectory(directory);
            }

            var content = source.Read(entry.RelativePath, cancellationToken);
            File.WriteAllBytes(target, content);

            // **時刻を合わせる。** 合わせないと、時刻で比べる経路が
            // 「全部違う」としか言わなくなる。
            if (entry.Modified is { } when)
            {
                try
                {
                    File.SetLastWriteTimeUtc(target, when.UtcDateTime);
                }
                catch (Exception error) when (error is IOException or ArgumentOutOfRangeException)
                {
                    // 時刻を置けなくても、中身は取れている。そこで止めない。
                }
            }

            files++;
            bytes += content.Length;
        }
    }

    /// <summary>取ってきた形を人に伝える。**上限で切ったことを黙らない。**</summary>
    public string Describe()
    {
        var text = $"{Display} から {Result.Files:N0} ファイル"
            + $"（{Result.Bytes / 1024.0 / 1024.0:F1} MB）を取ってきました";
        if (Result.HasSkipped)
        {
            text += $"。**{Result.Skipped.Count} 件は取っていません**："
                + string.Join(" / ", Result.Skipped.Take(5))
                + (Result.Skipped.Count > 5 ? " ほか" : string.Empty);
        }
        return text;
    }

    public void Dispose() => TryDelete(_temporary);

    private static void TryDelete(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // 消せなくても、比較の結果には関係しない。
        }
    }
}
