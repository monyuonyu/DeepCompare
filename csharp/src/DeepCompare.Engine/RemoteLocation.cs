using System.Net;

namespace DeepCompare.Engine;

/// <summary>
/// 場所の指定を読む。
///
/// **パスを書く場所ならどこでもリモートを書けるようにする。** 別の入力欄を
/// 作ると、比較・同期・CLI の全部に「リモート用の経路」が生えることになる。
/// BC も <c>ftp://</c> の形で同じことをしている。
///
/// 受ける形:
/// <code>
/// /手元/の/場所
/// sftp://利用者@主機/場所          （鍵で入る。合言葉は省ける）
/// dav://利用者:合言葉@例.com/remote.php/dav/files/利用者/
/// davs://例.com/dav/            （https）
/// s3://鍵:秘密@入口/バケツ/接頭辞
/// </code>
/// </summary>
public static class RemoteLocation
{
    /// <summary>リモートの指定か。手元のパスなら false。</summary>
    public static bool IsRemote(string location)
        => location.StartsWith("dav://", StringComparison.OrdinalIgnoreCase)
        || location.StartsWith("davs://", StringComparison.OrdinalIgnoreCase)
        || location.StartsWith("s3://", StringComparison.OrdinalIgnoreCase)
        || location.StartsWith("ftp://", StringComparison.OrdinalIgnoreCase)
        || location.StartsWith("ftps://", StringComparison.OrdinalIgnoreCase)
        || location.StartsWith("sftp://", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// 場所から接続を作る。手元のパスなら <see cref="LocalFileSource"/>。
    ///
    /// **合言葉を場所に書けるようにするのは、そう書きたい人がいるから。**
    /// ただし画面に出るので、履歴に残す前に伏せる（<see cref="Redact"/>）。
    /// </summary>
    public static IFileSource Open(string location)
    {
        if (location.StartsWith("s3://", StringComparison.OrdinalIgnoreCase))
        {
            return OpenS3(location);
        }
        if (location.StartsWith("dav://", StringComparison.OrdinalIgnoreCase)
            || location.StartsWith("davs://", StringComparison.OrdinalIgnoreCase))
        {
            return OpenWebDav(location);
        }
        if (location.StartsWith("ftp://", StringComparison.OrdinalIgnoreCase)
            || location.StartsWith("ftps://", StringComparison.OrdinalIgnoreCase))
        {
            return OpenFtp(location);
        }
        if (location.StartsWith("sftp://", StringComparison.OrdinalIgnoreCase))
        {
            return OpenSftp(location);
        }
        return new LocalFileSource(location);
    }

    private static IFileSource OpenWebDav(string location)
    {
        var secure = location.StartsWith("davs://", StringComparison.OrdinalIgnoreCase);
        var rest = location[(location.IndexOf("://", StringComparison.Ordinal) + 3)..];

        NetworkCredential? credentials = null;
        var at = rest.LastIndexOf('@');
        if (at >= 0)
        {
            var userInfo = rest[..at];
            rest = rest[(at + 1)..];
            var colon = userInfo.IndexOf(':');
            credentials = colon >= 0
                ? new NetworkCredential(
                    Uri.UnescapeDataString(userInfo[..colon]),
                    Uri.UnescapeDataString(userInfo[(colon + 1)..]))
                : new NetworkCredential(Uri.UnescapeDataString(userInfo), string.Empty);
        }

        return new WebDavFileSource((secure ? "https://" : "http://") + rest, credentials);
    }

    private static IFileSource OpenSftp(string location)
    {
        var rest = location["sftp://".Length..];

        // 利用者名は省ける（今のログイン名を使う）。**合言葉は省くのが普通** —
        // 鍵で入るのが既定の使い方。
        var user = Environment.UserName;
        string? password = null;

        var at = rest.LastIndexOf('@');
        if (at >= 0)
        {
            var userInfo = rest[..at];
            rest = rest[(at + 1)..];
            var colon = userInfo.IndexOf(':');
            if (colon >= 0)
            {
                user = Uri.UnescapeDataString(userInfo[..colon]);
                password = Uri.UnescapeDataString(userInfo[(colon + 1)..]);
            }
            else
            {
                user = Uri.UnescapeDataString(userInfo);
            }
        }

        var slash = rest.IndexOf('/');
        var authority = slash < 0 ? rest : rest[..slash];
        // 場所を省いたら、入った先の既定の場所（ふつうは home）。
        var root = slash < 0 ? "." : rest[slash..];

        var port = 22;
        var colonAt = authority.LastIndexOf(':');
        if (colonAt >= 0 && int.TryParse(authority[(colonAt + 1)..], out var parsed))
        {
            port = parsed;
            authority = authority[..colonAt];
        }

        return new SftpFileSource(new SftpSettings(authority, user)
        {
            Port = port,
            Password = password,
            Root = root,
            PrivateKeyPath = Environment.GetEnvironmentVariable("DEEPCOMPARE_SSH_KEY"),
            PrivateKeyPassphrase = Environment.GetEnvironmentVariable("DEEPCOMPARE_SSH_PASSPHRASE"),
        });
    }

    private static IFileSource OpenFtp(string location)
    {
        var secure = location.StartsWith("ftps://", StringComparison.OrdinalIgnoreCase);
        var rest = location[(location.IndexOf("://", StringComparison.Ordinal) + 3)..];

        var user = "anonymous";
        // 匿名の置き場は「連絡先を合言葉に」という慣習。**空にしない**
        // （空だと断る実装がある）。
        var password = "deepcompare@example.com";

        var at = rest.LastIndexOf('@');
        if (at >= 0)
        {
            var userInfo = rest[..at];
            rest = rest[(at + 1)..];
            var colon = userInfo.IndexOf(':');
            if (colon >= 0)
            {
                user = Uri.UnescapeDataString(userInfo[..colon]);
                password = Uri.UnescapeDataString(userInfo[(colon + 1)..]);
            }
            else
            {
                user = Uri.UnescapeDataString(userInfo);
            }
        }

        // 主機[:番号]/場所 に分ける。
        var slash = rest.IndexOf('/');
        var authority = slash < 0 ? rest : rest[..slash];
        var root = slash < 0 ? "/" : rest[slash..];

        var port = secure ? 21 : 21;
        var colonAt = authority.LastIndexOf(':');
        if (colonAt >= 0 && int.TryParse(authority[(colonAt + 1)..], out var parsed))
        {
            port = parsed;
            authority = authority[..colonAt];
        }

        return new FtpFileSource(new FtpSettings(authority, user, password)
        {
            Port = port,
            UseTls = secure,
            Root = root,
            // 自己署名の置き場を相手にすることがある。**明示的に切れるようにする**
            // （既定は検証したまま）。
            ValidateCertificate =
                Environment.GetEnvironmentVariable("DEEPCOMPARE_FTP_INSECURE") != "1",
        });
    }

    private static IFileSource OpenS3(string location)
    {
        var rest = location["s3://".Length..];

        string? accessKey = null;
        string? secretKey = null;
        var at = rest.LastIndexOf('@');
        if (at >= 0)
        {
            var userInfo = rest[..at];
            rest = rest[(at + 1)..];
            var colon = userInfo.IndexOf(':');
            if (colon >= 0)
            {
                accessKey = Uri.UnescapeDataString(userInfo[..colon]);
                secretKey = Uri.UnescapeDataString(userInfo[(colon + 1)..]);
            }
        }

        // 鍵を場所に書かない使い方もある。**環境変数から拾う。**
        accessKey ??= Environment.GetEnvironmentVariable("AWS_ACCESS_KEY_ID");
        secretKey ??= Environment.GetEnvironmentVariable("AWS_SECRET_ACCESS_KEY");

        if (accessKey is null || secretKey is null)
        {
            throw new ArgumentException(
                "S3 の鍵がありません。s3://鍵:秘密@入口/バケツ の形で書くか、"
                + "AWS_ACCESS_KEY_ID と AWS_SECRET_ACCESS_KEY を設定してください。");
        }

        // **入口に scheme が書かれていたら、先に外す。** そうしないと
        // `http://主機:番号/バケツ` を `/` で切ったときに壊れる
        // （MinIO や試験用の置き場はこの形で書くことが多い）。
        var scheme = "https://";
        foreach (var candidate in new[] { "http://", "https://" })
        {
            if (rest.StartsWith(candidate, StringComparison.OrdinalIgnoreCase))
            {
                scheme = candidate;
                rest = rest[candidate.Length..];
                break;
            }
        }

        // 入口 / バケツ / 接頭辞 に分ける。
        var parts = rest.Split('/', 3);
        if (parts.Length < 2 || parts[0].Length == 0 || parts[1].Length == 0)
        {
            throw new ArgumentException(
                $"S3 の場所は s3://入口/バケツ[/接頭辞] の形で書きます: {Redact(location)}");
        }

        var endpoint = scheme + parts[0];

        return new S3FileSource(new S3Settings(endpoint, parts[1], accessKey, secretKey)
        {
            Prefix = parts.Length > 2 ? parts[2] : string.Empty,
            Region = Environment.GetEnvironmentVariable("AWS_REGION") ?? "us-east-1",
        });
    }

    /// <summary>
    /// 合言葉を伏せた形。
    ///
    /// **画面・記録・レポートに出す前に必ず通す。** 場所の文字列はそのまま
    /// 履歴やセッションに残るので、そこに合言葉が入ると後から漏れる。
    /// </summary>
    public static string Redact(string location)
    {
        var scheme = location.IndexOf("://", StringComparison.Ordinal);
        if (scheme < 0)
        {
            return location;
        }

        var rest = location[(scheme + 3)..];
        var at = rest.LastIndexOf('@');
        if (at < 0)
        {
            return location;
        }

        var userInfo = rest[..at];
        var colon = userInfo.IndexOf(':');

        // **合言葉が書かれていなければ伏せない。** 鍵で入る SFTP は
        // `sftp://利用者@主機` の形なので、`:***@` を足すと
        // 「合言葉を使っている」という嘘の印象になる。
        if (colon < 0)
        {
            return location;
        }

        return location[..(scheme + 3)] + userInfo[..colon] + ":***@" + rest[(at + 1)..];
    }
}
