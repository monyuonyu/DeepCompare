namespace DeepCompare.Engine;

/// <summary>操作の結果。**成否と、人に見せる一言をまとめて返す。**</summary>
public sealed record FileOperationResult(bool Ok, string Message)
{
    public static FileOperationResult Failed(string message) => new(false, message);
    public static FileOperationResult Done(string message) => new(true, message);
}

/// <summary>
/// フォルダー比較から行うファイル操作（BC の Actions）。
///
/// **画面から切り離す。** どれも戻せない操作なので、条件の判定を
/// 画面のコードに埋めると試験できない。ここに集めて、画面は
/// 「確認を取ってこれを呼ぶ」だけにする。
///
/// **この層は確認を取らない。** 訊くかどうかは画面が決める。
/// ここで訊くと、CLI や試験から呼べなくなる。
/// </summary>
public static class FolderOperations
{
    /// <summary>
    /// 名前として使えるか。使えないなら理由。
    ///
    /// **区切り文字を断る。** 「名前を変える」つもりで別のフォルダーへ
    /// 移してしまうのを防ぐ（移すなら Move を使う）。
    /// </summary>
    public static string? WhyInvalidName(string name)
    {
        if (name.Trim().Length == 0)
        {
            return "名前が空です。";
        }
        if (name.Contains('/') || name.Contains('\\'))
        {
            return "名前にフォルダーの区切りは使えません。"
                + "移すなら「反対側へ移す」を使ってください。";
        }
        if (name is "." or "..")
        {
            return $"{name} は名前に使えません。";
        }
        // Windows で使えない文字。**Linux でも断る** — 往復すると壊れるので、
        // 作れてしまう方が後で困る。
        foreach (var c in new[] { ':', '*', '?', '"', '<', '>', '|' })
        {
            if (name.Contains(c))
            {
                return $"名前に {c} は使えません（Windows で開けなくなります）。";
            }
        }
        return null;
    }

    /// <summary>名前を変える。</summary>
    public static FileOperationResult Rename(string path, string newName, bool isDirectory)
    {
        if (WhyInvalidName(newName) is { } reason)
        {
            return FileOperationResult.Failed(reason);
        }

        var oldName = Path.GetFileName(path);
        if (newName == oldName)
        {
            // **同じ名前は「何もしなかった」として成功にしない。**
            // 「変えました」と出ると、変わったと思われる。
            return FileOperationResult.Failed("名前が変わっていません。");
        }

        var directory = Path.GetDirectoryName(path);
        if (string.IsNullOrEmpty(directory))
        {
            return FileOperationResult.Failed("場所が分かりません。");
        }

        var destination = Path.Combine(directory, newName);

        // **大小文字だけの変更を通す。** Windows では同じ名前と判定されるが、
        // それを「既にある」で断ると、大小文字を直せなくなる。
        var sameTarget = string.Equals(path, destination,
            StringComparison.OrdinalIgnoreCase);

        if (!sameTarget && (File.Exists(destination) || Directory.Exists(destination)))
        {
            return FileOperationResult.Failed($"{newName} は既にあります。");
        }

        try
        {
            if (isDirectory)
            {
                Directory.Move(path, destination);
            }
            else
            {
                File.Move(path, destination, overwrite: false);
            }
            return FileOperationResult.Done($"{oldName} を {newName} にしました。");
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return FileOperationResult.Failed($"名前を変えられません: {error.Message}");
        }
    }

    /// <summary>
    /// 反対側へ移す。**元からは消える。**
    ///
    /// 呼ぶ前に確認を取ること。ここでは訊かない。
    /// </summary>
    public static FileOperationResult Move(string source, string destination, bool isDirectory)
    {
        if (!File.Exists(source) && !Directory.Exists(source))
        {
            return FileOperationResult.Failed("移すものがありません。");
        }

        // **自分の中へ移そうとしていないか。** L/dir を L/dir/sub へ移すと、
        // 無限に潜るか、途中で失敗して中身が壊れる。
        if (isDirectory && IsInside(destination, source))
        {
            return FileOperationResult.Failed("自分の中へは移せません。");
        }

        var name = Path.GetFileName(source);
        try
        {
            var parent = Path.GetDirectoryName(destination);
            if (!string.IsNullOrEmpty(parent))
            {
                Directory.CreateDirectory(parent);
            }

            if (isDirectory)
            {
                if (Directory.Exists(destination))
                {
                    Directory.Delete(destination, recursive: true);
                }
                Directory.Move(source, destination);
            }
            else
            {
                File.Move(source, destination, overwrite: true);
            }
            return FileOperationResult.Done($"{name} を移しました。");
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return FileOperationResult.Failed($"移せません: {error.Message}");
        }
    }

    /// <summary>フォルダーを作る。</summary>
    public static FileOperationResult NewFolder(string root, string name)
    {
        if (WhyInvalidName(name) is { } reason)
        {
            return FileOperationResult.Failed(reason);
        }
        if (root.Length == 0 || !Directory.Exists(root))
        {
            return FileOperationResult.Failed("作る場所がありません。");
        }

        var destination = Path.Combine(root, name);
        if (Directory.Exists(destination) || File.Exists(destination))
        {
            return FileOperationResult.Failed($"{name} は既にあります。");
        }

        try
        {
            Directory.CreateDirectory(destination);
            return FileOperationResult.Done($"{name} を作りました。");
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return FileOperationResult.Failed($"作れません: {error.Message}");
        }
    }

    /// <summary>
    /// <paramref name="path"/> が <paramref name="root"/> の中にあるか。
    ///
    /// **末尾に区切りを足してから比べる。** そうしないと
    /// <c>/a/bc</c> が <c>/a/b</c> の中にあると判定される。
    /// </summary>
    internal static bool IsInside(string path, string root)
    {
        var normalizedRoot = Path.GetFullPath(root)
            .TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        var normalizedPath = Path.GetFullPath(path);

        return normalizedPath.StartsWith(normalizedRoot,
            OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal);
    }
}
