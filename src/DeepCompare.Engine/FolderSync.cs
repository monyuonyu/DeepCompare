using System.Text;

namespace DeepCompare.Engine;

/// <summary>同期の向き。</summary>
public enum SyncDirection
{
    /// <summary>左を正として右を合わせる。</summary>
    ToRight,

    /// <summary>右を正として左を合わせる。</summary>
    ToLeft,

    /// <summary>新しい方を正として両側を合わせる。</summary>
    Both,
}

/// <summary>何をするか。</summary>
public enum SyncAction
{
    /// <summary>何もしない。</summary>
    None,
    CopyToRight,
    CopyToLeft,
    DeleteLeft,
    DeleteRight,
}

/// <summary>1 件分の予定。</summary>
public sealed record SyncStep(
    string RelativePath,
    SyncAction Action,
    /// <summary>なぜそうするか。**予定を人が読んで確かめられるようにする。**</summary>
    string Reason)
{
    public string Describe() => Action switch
    {
        SyncAction.CopyToRight => $"→ {RelativePath}（{Reason}）",
        SyncAction.CopyToLeft => $"← {RelativePath}（{Reason}）",
        SyncAction.DeleteLeft => $"✕左 {RelativePath}（{Reason}）",
        SyncAction.DeleteRight => $"✕右 {RelativePath}（{Reason}）",
        _ => RelativePath,
    };
}

/// <summary>同期の設定。</summary>
public sealed record SyncOptions
{
    public SyncDirection Direction { get; init; } = SyncDirection.ToRight;

    /// <summary>
    /// 片側にしか無いものを消すか。
    ///
    /// **既定は消さない。** 同期の事故はほとんどこれで起きる。「揃える」の
    /// つもりで押したら、片側にしか無い作りかけが消えた、という形。
    /// </summary>
    public bool DeleteOrphans { get; init; }

    /// <summary>
    /// 新しい方を正とするときの、時刻の許容誤差（秒）。
    ///
    /// FAT と NTFS の間では 1〜2 秒ずれる。ここを 0 にすると、中身が同じでも
    /// 毎回どちらかを写すことになる。
    /// </summary>
    public double ToleranceSeconds { get; init; } = 2;
}

public sealed record SyncPlan(IReadOnlyList<SyncStep> Steps)
{
    public int Copies => Steps.Count(s => s.Action is SyncAction.CopyToLeft or SyncAction.CopyToRight);
    public int Deletions => Steps.Count(s => s.Action is SyncAction.DeleteLeft or SyncAction.DeleteRight);
    public bool IsEmpty => Steps.Count == 0;
}

/// <summary>実行の結果。</summary>
public sealed record SyncResult(int Done, IReadOnlyList<string> Errors)
{
    public bool AllSucceeded => Errors.Count == 0;
}

/// <summary>
/// フォルダーの同期（BC の Folder Sync）。
///
/// **必ず「予定」を作ってから実行する。** 何が起きるかを見ずに走らせる同期は、
/// 取り返しのつかない事故を生む。予定は人が読める形で、理由まで添える。
///
/// 比較の結果をそのまま使う。同期のために走査をやり直さない
/// （その間にファイルが変わると、見たものと違うことが起きる）。
/// </summary>
public static class FolderSync
{
    /// <summary>比較の結果から予定を組み立てる。**ここでは何も書き換えない。**</summary>
    public static SyncPlan Plan(FolderComparison comparison, SyncOptions options)
    {
        var steps = new List<SyncStep>();

        foreach (var entry in comparison.Entries)
        {
            if (entry.IsDirectory || entry.Error is not null)
            {
                continue;
            }

            switch (entry.Status)
            {
                case EntryStatus.Identical:
                    break;

                case EntryStatus.Different:
                    steps.Add(Different(entry, options));
                    break;

                case EntryStatus.LeftOnly:
                    steps.Add(options.Direction switch
                    {
                        SyncDirection.ToLeft when options.DeleteOrphans
                            => new SyncStep(entry.RelativePath, SyncAction.DeleteLeft, "右に無い"),
                        SyncDirection.ToLeft
                            => new SyncStep(entry.RelativePath, SyncAction.None, "右に無い（消さない設定）"),
                        _ => new SyncStep(entry.RelativePath, SyncAction.CopyToRight, "左にしか無い"),
                    });
                    break;

                case EntryStatus.RightOnly:
                    steps.Add(options.Direction switch
                    {
                        SyncDirection.ToRight when options.DeleteOrphans
                            => new SyncStep(entry.RelativePath, SyncAction.DeleteRight, "左に無い"),
                        SyncDirection.ToRight
                            => new SyncStep(entry.RelativePath, SyncAction.None, "左に無い（消さない設定）"),
                        _ => new SyncStep(entry.RelativePath, SyncAction.CopyToLeft, "右にしか無い"),
                    });
                    break;
            }
        }

        return new SyncPlan([.. steps.Where(s => s.Action != SyncAction.None)]);
    }

    private static SyncStep Different(FolderEntry entry, SyncOptions options)
    {
        if (options.Direction == SyncDirection.ToRight)
        {
            return new SyncStep(entry.RelativePath, SyncAction.CopyToRight, "中身が違う");
        }
        if (options.Direction == SyncDirection.ToLeft)
        {
            return new SyncStep(entry.RelativePath, SyncAction.CopyToLeft, "中身が違う");
        }

        // 両方向のときは新しい方を採る。
        var left = entry.LeftModified;
        var right = entry.RightModified;
        if (left is null || right is null)
        {
            return new SyncStep(entry.RelativePath, SyncAction.None, "時刻が分からない");
        }

        var difference = (right.Value - left.Value).TotalSeconds;
        if (Math.Abs(difference) <= options.ToleranceSeconds)
        {
            // **時刻が同じで中身が違う。** どちらが新しいか決められないので触らない。
            // 勝手にどちらかを採ると、片方の変更が黙って消える。
            return new SyncStep(entry.RelativePath, SyncAction.None,
                "中身が違うが時刻が同じ（人が決める）");
        }

        return difference > 0
            ? new SyncStep(entry.RelativePath, SyncAction.CopyToLeft, "右の方が新しい")
            : new SyncStep(entry.RelativePath, SyncAction.CopyToRight, "左の方が新しい");
    }

    /// <summary>
    /// 予定を実行する。
    ///
    /// **1 件失敗しても続ける。** 途中で止めると、どこまで進んだのか分からない
    /// 中途半端な状態が残る。失敗は集めて最後に返す。
    /// </summary>
    public static SyncResult Apply(
        SyncPlan plan, string leftRoot, string rightRoot,
        Action<SyncStep>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var done = 0;
        var errors = new List<string>();

        foreach (var step in plan.Steps)
        {
            cancellationToken.ThrowIfCancellationRequested();
            progress?.Invoke(step);

            var relative = step.RelativePath.Replace('/', Path.DirectorySeparatorChar);
            var left = Path.Combine(leftRoot, relative);
            var right = Path.Combine(rightRoot, relative);

            try
            {
                switch (step.Action)
                {
                    case SyncAction.CopyToRight:
                        Copy(left, right);
                        break;
                    case SyncAction.CopyToLeft:
                        Copy(right, left);
                        break;
                    case SyncAction.DeleteLeft:
                        File.Delete(left);
                        break;
                    case SyncAction.DeleteRight:
                        File.Delete(right);
                        break;
                }
                done++;
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException)
            {
                errors.Add($"{step.RelativePath}: {error.Message}");
            }
        }

        return new SyncResult(done, errors);
    }

    private static void Copy(string from, string to)
    {
        var directory = Path.GetDirectoryName(to);
        if (directory is { Length: > 0 })
        {
            Directory.CreateDirectory(directory);
        }
        File.Copy(from, to, overwrite: true);

        // **時刻も合わせる。** 合わせないと、次に時刻で比べたときに
        // また「違う」と出て、写し続けることになる。
        File.SetLastWriteTimeUtc(to, File.GetLastWriteTimeUtc(from));
    }

    /// <summary>予定を人が読む形に整える。</summary>
    public static string Format(SyncPlan plan)
    {
        if (plan.IsEmpty)
        {
            return "することはありません。" + Environment.NewLine;
        }

        var text = new StringBuilder();
        foreach (var step in plan.Steps)
        {
            text.AppendLine(step.Describe());
        }
        text.AppendLine();
        text.AppendLine($"写す {plan.Copies} 件 / 消す {plan.Deletions} 件");
        if (plan.Deletions > 0)
        {
            text.AppendLine("**消す操作が含まれています。** 実行すると元に戻せません。");
        }
        return text.ToString();
    }
}
