using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Xml;

namespace DeepCompare.Engine;

/// <summary>
/// WebDAV の置き場。
///
/// **依存を増やさない。** WebDAV は HTTP に動詞を足しただけの規約なので、
/// <see cref="HttpClient"/> だけで書ける。専用のライブラリを入れると、
/// NativeAOT での発行と単一 exe の方針に影響が出る。
///
/// Nextcloud・ownCloud・IIS・Apache の mod_dav が同じ経路で扱える。
/// </summary>
public sealed class WebDavFileSource : IFileSource
{
    private readonly HttpClient _client;
    private readonly bool _ownsClient;
    private readonly Uri _root;

    /// <param name="root">根の URL。末尾の <c>/</c> は有っても無くてもよい。</param>
    /// <param name="credentials">要らなければ null（公開の置き場）。</param>
    public WebDavFileSource(string root, NetworkCredential? credentials = null)
        : this(NormalizeRoot(root), CreateClient(credentials), ownsClient: true)
    {
    }

    /// <summary>試験で HTTP のやり取りを差し替えるための入口。</summary>
    public WebDavFileSource(Uri root, HttpClient client, bool ownsClient = false)
    {
        _root = root;
        _client = client;
        _ownsClient = ownsClient;
    }

    private static Uri NormalizeRoot(string root)
        => new(root.EndsWith('/') ? root : root + "/");

    private static HttpClient CreateClient(NetworkCredential? credentials)
    {
        var handler = new HttpClientHandler();
        if (credentials is not null)
        {
            handler.Credentials = credentials;
            // **先に送る。** 401 を受けてから送り直す形だと、往復が倍になる。
            handler.PreAuthenticate = true;
        }
        return new HttpClient(handler)
        {
            // 待ち続けない。相手が黙っていると画面が固まる。
            Timeout = TimeSpan.FromSeconds(60),
        };
    }

    public string Display => _root.ToString();

    public bool CanWrite { get; init; } = true;

    private Uri UriFor(string relativePath)
    {
        var cleaned = relativePath.Replace('\\', '/').Trim('/');
        if (cleaned.Length == 0)
        {
            return _root;
        }
        // 各段を個別に符号化する。**丸ごと符号化すると `/` まで潰れる。**
        var encoded = string.Join('/', cleaned.Split('/').Select(Uri.EscapeDataString));
        return new Uri(_root, encoded);
    }

    public IReadOnlyList<RemoteEntry> List(
        string relativePath, CancellationToken cancellationToken = default)
    {
        var request = new HttpRequestMessage(new HttpMethod("PROPFIND"), UriFor(relativePath));
        // **深さは 1。** 既定（infinity）だと、根に向けた 1 回で全部取りに行く。
        request.Headers.Add("Depth", "1");
        request.Content = new StringContent("""
            <?xml version="1.0" encoding="utf-8"?>
            <D:propfind xmlns:D="DAV:"><D:prop>
              <D:resourcetype/><D:getcontentlength/><D:getlastmodified/>
            </D:prop></D:propfind>
            """, Encoding.UTF8, "application/xml");

        using var response = Send(request, cancellationToken);
        var body = response.Content.ReadAsStringAsync(cancellationToken).GetAwaiter().GetResult();
        return ParseListing(body, relativePath, _root);
    }

    /// <summary>
    /// PROPFIND の応答を読む。
    ///
    /// **自分自身が最初の項目として返る。** それを落とさないと、
    /// フォルダーが自分の中に入っている木ができる。
    /// </summary>
    internal static IReadOnlyList<RemoteEntry> ParseListing(
        string xml, string relativePath, Uri root)
    {
        var result = new List<RemoteEntry>();
        var self = root.AbsolutePath.TrimEnd('/')
            + (relativePath.Trim('/').Length > 0 ? "/" + relativePath.Trim('/') : string.Empty);

        using var reader = XmlReader.Create(new StringReader(xml), new XmlReaderSettings
        {
            DtdProcessing = DtdProcessing.Prohibit,
            XmlResolver = null,
        });

        string? href = null;
        var isDirectory = false;
        long size = 0;
        DateTimeOffset? modified = null;

        reader.Read();
        while (!reader.EOF)
        {
            if (reader.NodeType == XmlNodeType.Element)
            {
                switch (reader.LocalName)
                {
                    case "response":
                        href = null;
                        isDirectory = false;
                        size = 0;
                        modified = null;
                        break;
                    case "href":
                        href = reader.ReadElementContentAsString();
                        continue;
                    case "collection":
                        isDirectory = true;
                        break;
                    case "getcontentlength":
                        long.TryParse(reader.ReadElementContentAsString(),
                            NumberStyles.Integer, CultureInfo.InvariantCulture, out size);
                        continue;
                    case "getlastmodified":
                        modified = ParseHttpDate(reader.ReadElementContentAsString());
                        continue;
                }
            }
            else if (reader.NodeType == XmlNodeType.EndElement && reader.LocalName == "response")
            {
                if (href is { Length: > 0 })
                {
                    var path = Uri.UnescapeDataString(new Uri(root, href).AbsolutePath).TrimEnd('/');
                    var selfPath = Uri.UnescapeDataString(self).TrimEnd('/');

                    // 自分自身は落とす。
                    if (!string.Equals(path, selfPath, StringComparison.Ordinal))
                    {
                        var rootPath = Uri.UnescapeDataString(root.AbsolutePath).TrimEnd('/');
                        var relative = path.StartsWith(rootPath, StringComparison.Ordinal)
                            ? path[rootPath.Length..].TrimStart('/')
                            : path.TrimStart('/');
                        result.Add(new RemoteEntry(relative, isDirectory, size, modified));
                    }
                }
            }
            reader.Read();
        }
        return result;
    }

    /// <summary>
    /// HTTP の日付を読む。
    ///
    /// **曜日は当てにしない。** 実装によっては実際とずれた曜日を返すが、
    /// .NET の TryParse は曜日の整合を見るので、そこで丸ごと読めなくなる
    /// （時刻が消えると「大きさと時刻で比べる」経路が黙って効かなくなる）。
    /// </summary>
    internal static DateTimeOffset? ParseHttpDate(string value)
    {
        var text = value.Trim();
        var comma = text.IndexOf(',');
        if (comma >= 0)
        {
            text = text[(comma + 1)..].Trim();
        }

        string[] formats =
        [
            "dd MMM yyyy HH:mm:ss 'GMT'",
            "dd MMM yyyy HH:mm:ss zzz",
            "dd-MMM-yy HH:mm:ss 'GMT'",      // RFC 850
            "MMM d HH:mm:ss yyyy",             // asctime
        ];

        if (DateTimeOffset.TryParseExact(text, formats, CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var exact))
        {
            return exact;
        }
        // ISO 8601 で返す実装もある。
        return DateTimeOffset.TryParse(text, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var loose)
            ? loose : null;
    }

    public byte[] Read(string relativePath, CancellationToken cancellationToken = default)
    {
        using var response = Send(
            new HttpRequestMessage(HttpMethod.Get, UriFor(relativePath)), cancellationToken);
        return response.Content.ReadAsByteArrayAsync(cancellationToken).GetAwaiter().GetResult();
    }

    public void Write(string relativePath, byte[] content, CancellationToken cancellationToken = default)
    {
        Ensure(CanWrite);
        var request = new HttpRequestMessage(HttpMethod.Put, UriFor(relativePath))
        {
            Content = new ByteArrayContent(content),
        };
        request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        Send(request, cancellationToken).Dispose();
    }

    public void Delete(string relativePath, CancellationToken cancellationToken = default)
    {
        Ensure(CanWrite);
        Send(new HttpRequestMessage(HttpMethod.Delete, UriFor(relativePath)),
            cancellationToken).Dispose();
    }

    public bool Exists(string relativePath, CancellationToken cancellationToken = default)
    {
        using var response = _client.Send(
            new HttpRequestMessage(HttpMethod.Head, UriFor(relativePath)), cancellationToken);
        return response.IsSuccessStatusCode;
    }

    private void Ensure(bool allowed)
    {
        if (!allowed)
        {
            throw new InvalidOperationException($"{Display} は読み取り専用です。");
        }
    }

    private HttpResponseMessage Send(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var response = _client.Send(request, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var status = response.StatusCode;
            response.Dispose();
            // **理由をそのまま出す。** 401 と 404 と 507 では、人がやることが違う。
            throw new IOException(status switch
            {
                HttpStatusCode.Unauthorized => $"認証が必要です（{(int)status}）: {request.RequestUri}",
                HttpStatusCode.Forbidden => $"権限がありません（{(int)status}）: {request.RequestUri}",
                HttpStatusCode.NotFound => $"見つかりません（{(int)status}）: {request.RequestUri}",
                HttpStatusCode.InsufficientStorage => $"置き場が一杯です（{(int)status}）",
                _ => $"WebDAV が {(int)status} {status} を返しました: {request.RequestUri}",
            });
        }
        return response;
    }

    public void Dispose()
    {
        if (_ownsClient)
        {
            _client.Dispose();
        }
    }
}
