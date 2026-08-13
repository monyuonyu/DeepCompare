using Renci.SshNet;
using Renci.SshNet.Common;

namespace DeepCompare.Engine;

/// <summary>SFTP の接続設定。</summary>
public sealed record SftpSettings(string Host, string User)
{
    public int Port { get; init; } = 22;

    /// <summary>合言葉。鍵で入るなら空でよい。</summary>
    public string? Password { get; init; }

    /// <summary>秘密鍵のファイル。null なら既定の場所を順に探す。</summary>
    public string? PrivateKeyPath { get; init; }

    /// <summary>秘密鍵の合言葉。</summary>
    public string? PrivateKeyPassphrase { get; init; }

    public string Root { get; init; } = ".";
}

/// <summary>
/// SFTP の置き場。
///
/// **ここだけライブラリを使う。** SSH は鍵交換・暗号・多重化を持つ規約で、
/// 自前で書くのは現実的でない（FTP や WebDAV とは規模が違う）。
/// 暗号を自分で実装するのは、間違えたときの害が大きい領域でもある。
///
/// **鍵を優先する。** 合言葉より鍵の方が普通に使われており、
/// `~/.ssh/id_ed25519` などに既に置いてある。
/// </summary>
public sealed class SftpFileSource : IFileSource
{
    private readonly SftpClient _client;
    private readonly SftpSettings _settings;

    public SftpFileSource(SftpSettings settings)
    {
        _settings = settings;
        _client = new SftpClient(BuildConnection(settings));
        _client.Connect();
    }

    /// <summary>
    /// 入り方を組み立てる。**鍵を先に試す。**
    ///
    /// 鍵と合言葉の両方を渡しておくと、相手が受け付ける方で入れる。
    /// 片方だけに決め打つと、鍵しか通さない置き場でも合言葉しか通さない
    /// 置き場でも、どちらかで詰まる。
    /// </summary>
    private static ConnectionInfo BuildConnection(SftpSettings settings)
    {
        var methods = new List<AuthenticationMethod>();

        foreach (var keyPath in KeyCandidates(settings.PrivateKeyPath))
        {
            try
            {
                var key = settings.PrivateKeyPassphrase is { Length: > 0 } passphrase
                    ? new PrivateKeyFile(keyPath, passphrase)
                    : new PrivateKeyFile(keyPath);
                methods.Add(new PrivateKeyAuthenticationMethod(settings.User, key));
            }
            catch (Exception error) when (error is SshException or IOException
                                            or UnauthorizedAccessException)
            {
                // 読めない鍵、合言葉つきで合言葉が違う鍵。**そこで止めない** —
                // 別の鍵か合言葉で入れることがある。
            }
        }

        if (settings.Password is { Length: > 0 })
        {
            methods.Add(new PasswordAuthenticationMethod(settings.User, settings.Password));
        }

        if (methods.Count == 0)
        {
            throw new ArgumentException(
                "SFTP の入り方がありません。合言葉を書くか、鍵を "
                + "~/.ssh/id_ed25519 などに置いてください。");
        }

        return new ConnectionInfo(settings.Host, settings.Port, settings.User, [.. methods]);
    }

    /// <summary>
    /// 使う秘密鍵の候補。**新しい形式を先に見る。**
    /// RSA を先に試すと、Ed25519 しか受けない置き場で無駄な往復が増える。
    /// </summary>
    private static IEnumerable<string> KeyCandidates(string? explicitPath)
    {
        if (explicitPath is { Length: > 0 })
        {
            if (File.Exists(explicitPath))
            {
                yield return explicitPath;
            }
            yield break;
        }

        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        foreach (var name in new[] { "id_ed25519", "id_ecdsa", "id_rsa" })
        {
            var path = Path.Combine(home, ".ssh", name);
            if (File.Exists(path))
            {
                yield return path;
            }
        }
    }

    public string Display => $"sftp://{_settings.Host}/{_settings.Root.TrimStart('.', '/')}".TrimEnd('/');

    public bool CanWrite { get; init; } = true;

    private string PathFor(string relativePath)
    {
        var root = _settings.Root.TrimEnd('/');
        var cleaned = relativePath.Replace('\\', '/').Trim('/');
        return cleaned.Length == 0 ? (root.Length == 0 ? "." : root) : $"{root}/{cleaned}";
    }

    public IReadOnlyList<RemoteEntry> List(
        string relativePath, CancellationToken cancellationToken = default)
    {
        var result = new List<RemoteEntry>();
        var prefix = relativePath.Trim('/');

        foreach (var item in _client.ListDirectory(PathFor(relativePath)))
        {
            cancellationToken.ThrowIfCancellationRequested();

            // **自分自身と親は落とす。** 残すと木が循環する。
            if (item.Name is "." or "..")
            {
                continue;
            }
            // シンボリックリンクは辿らない。辿ると循環して終わらなくなる。
            if (item.IsSymbolicLink)
            {
                continue;
            }

            result.Add(new RemoteEntry(
                prefix.Length == 0 ? item.Name : $"{prefix}/{item.Name}",
                item.IsDirectory,
                item.IsDirectory ? 0 : item.Length,
                new DateTimeOffset(item.LastWriteTimeUtc, TimeSpan.Zero)));
        }
        return result;
    }

    public byte[] Read(string relativePath, CancellationToken cancellationToken = default)
    {
        using var memory = new MemoryStream();
        _client.DownloadFile(PathFor(relativePath), memory);
        return memory.ToArray();
    }

    public void Write(string relativePath, byte[] content, CancellationToken cancellationToken = default)
    {
        Ensure(CanWrite);
        using var memory = new MemoryStream(content);
        _client.UploadFile(memory, PathFor(relativePath), canOverride: true);
    }

    public void Delete(string relativePath, CancellationToken cancellationToken = default)
    {
        Ensure(CanWrite);
        _client.DeleteFile(PathFor(relativePath));
    }

    public bool Exists(string relativePath, CancellationToken cancellationToken = default)
        => _client.Exists(PathFor(relativePath));

    private void Ensure(bool allowed)
    {
        if (!allowed)
        {
            throw new InvalidOperationException($"{Display} は読み取り専用です。");
        }
    }

    public void Dispose()
    {
        if (_client.IsConnected)
        {
            _client.Disconnect();
        }
        _client.Dispose();
    }
}
