namespace DeepCompare.Engine;

/// <summary>名前が変わったと判断した組。</summary>
public sealed record RenamePair(
    string LeftPath,
    string RightPath,
    float Similarity,
    /// <summary>内容が完全に同じか。同じなら移動、違うなら移動＋変更。</summary>
    bool IdenticalContent);

/// <summary>
/// リネームの検出。
///
/// フォルダー比較は名前で対応付けるので、ファイル名が変わると「左のみ」と「右のみ」に
/// 分かれて出る。中身がほぼ同じでも別物として並ぶため、名前を変えた変更が読めない。
/// BC は名前一致しか見ないので、ここは**このツールの差になる部分**。
///
/// 判定は 2 段階。まず内容のハッシュが一致すれば確実な移動。残りは行の集合の
/// 重なり具合で見る。**行を読むのは片側だけの項目に限る**ので、一致しているファイルの
/// 山を相手にしても費用は増えない。
/// </summary>
public static class RenameDetector
{
    /// <summary>これを下回る重なりは別物として扱う。</summary>
    public const float DefaultThreshold = 0.5f;

    /// <summary>
    /// 内容を読む上限。これを超えるファイルはハッシュ一致だけで判定する。
    /// 大きなバイナリを行に割って集合を作っても意味が無く、時間だけかかる。
    /// </summary>
    public const long MaxContentBytes = 4 * 1024 * 1024;

    /// <summary>
    /// 「左のみ」「右のみ」の項目から、名前が変わっただけの組を見つける。
    /// </summary>
    /// <param name="threshold">対応とみなす重なりの下限（0〜1）。</param>
    public static List<RenamePair> Detect(
        FolderComparison comparison,
        string leftRoot,
        string rightRoot,
        float threshold = DefaultThreshold,
        CancellationToken cancellationToken = default)
    {
        var left = comparison.Entries
            .Where(e => !e.IsDirectory && e.Status == EntryStatus.LeftOnly && e.Error is null)
            .ToList();
        var right = comparison.Entries
            .Where(e => !e.IsDirectory && e.Status == EntryStatus.RightOnly && e.Error is null)
            .ToList();

        if (left.Count == 0 || right.Count == 0)
        {
            return [];
        }

        var leftInfo = left.Select(e => Read(leftRoot, e, cancellationToken)).ToList();
        var rightInfo = right.Select(e => Read(rightRoot, e, cancellationToken)).ToList();

        var pairs = new List<RenamePair>();
        var usedRight = new bool[right.Count];

        // 1 段階目: 中身が完全に同じもの。確実な移動なので、ここは重なりを見るまでもない。
        for (var i = 0; i < leftInfo.Count; i++)
        {
            for (var j = 0; j < rightInfo.Count; j++)
            {
                if (usedRight[j] || leftInfo[i].Hash is null)
                {
                    continue;
                }
                if (leftInfo[i].Hash == rightInfo[j].Hash)
                {
                    usedRight[j] = true;
                    pairs.Add(new RenamePair(
                        left[i].RelativePath, right[j].RelativePath, 1f, IdenticalContent: true));
                    leftInfo[i] = leftInfo[i] with { Done = true };
                    break;
                }
            }
        }

        // 2 段階目: 行の集合の重なり。最も重なる相手を選ぶ。
        for (var i = 0; i < leftInfo.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (leftInfo[i].Done || leftInfo[i].Lines is null)
            {
                continue;
            }

            var best = -1;
            var bestScore = threshold;
            for (var j = 0; j < rightInfo.Count; j++)
            {
                if (usedRight[j] || rightInfo[j].Lines is null)
                {
                    continue;
                }
                var score = Overlap(leftInfo[i].Lines!, rightInfo[j].Lines!);
                if (score > bestScore)
                {
                    bestScore = score;
                    best = j;
                }
            }

            if (best >= 0)
            {
                usedRight[best] = true;
                pairs.Add(new RenamePair(
                    left[i].RelativePath, right[best].RelativePath, bestScore,
                    IdenticalContent: false));
            }
        }

        return pairs;
    }

    private readonly record struct FileInfoLite(string? Hash, HashSet<string>? Lines, bool Done);

    private static FileInfoLite Read(
        string root, FolderEntry entry, CancellationToken cancellationToken)
    {
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var path = Path.Combine(root, entry.RelativePath.Replace('/', Path.DirectorySeparatorChar));
            var bytes = File.ReadAllBytes(path);
            var hash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes));

            if (bytes.Length > MaxContentBytes)
            {
                return new FileInfoLite(hash, null, false);
            }

            var text = TextDecoder.Decode(bytes);
            // 空行は共通しすぎて重なりの根拠にならない。落としておく。
            var lines = new HashSet<string>(StringComparer.Ordinal);
            foreach (var line in text.Lines)
            {
                if (line.Trim().Length > 0)
                {
                    lines.Add(line);
                }
            }
            return new FileInfoLite(hash, lines, false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception)
        {
            // 読めないものは対象外。比較全体は止めない。
            return new FileInfoLite(null, null, false);
        }
    }

    /// <summary>
    /// 行の集合の重なり（Jaccard）。片方だけが大きく育った場合に値が落ちるので、
    /// 「名前を変えて大幅に書き足した」は別物と判断される。それで良い。
    /// </summary>
    private static float Overlap(HashSet<string> left, HashSet<string> right)
    {
        if (left.Count == 0 || right.Count == 0)
        {
            return 0f;
        }
        var shared = 0;
        // 小さい方を回す。大きい方を回すと無駄に引きが増える。
        var (small, large) = left.Count <= right.Count ? (left, right) : (right, left);
        foreach (var line in small)
        {
            if (large.Contains(line))
            {
                shared++;
            }
        }
        var union = left.Count + right.Count - shared;
        return union == 0 ? 0f : (float)shared / union;
    }
}
