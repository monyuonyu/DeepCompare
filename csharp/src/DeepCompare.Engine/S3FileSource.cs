using System.Globalization;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Xml;

namespace DeepCompare.Engine;

/// <summary>S3 互換の置き場への接続設定。</summary>
public sealed record S3Settings(
    /// <summary>入口。AWS なら <c>https://s3.ap-northeast-1.amazonaws.com</c>、
    /// MinIO なら <c>http://localhost:9000</c> など。</summary>
    string Endpoint,
    string Bucket,
    string AccessKey,
    string SecretKey)
{
    public string Region { get; init; } = "us-east-1";

    /// <summary>
    /// バケットを名前で指す形（<c>bucket.example.com</c>）にするか。
    ///
    /// **既定は path 形式。** MinIO や古い互換実装は仮想ホスト形式に対応して
    /// いないことがあり、そちらに倒すと繋がらない置き場が出る。
    /// </summary>
    public bool UseVirtualHost { get; init; }

    /// <summary>接頭辞。バケットの一部だけを比較の対象にするときに使う。</summary>
    public string Prefix { get; init; } = string.Empty;
}

/// <summary>
/// S3 互換ストレージ。
///
/// **AWS SDK を入れない。** 使うのは 4 つの操作（一覧・取得・置く・消す）だけで、
/// そのために数 MB の依存と、NativeAOT での発行に関わる面倒を持ち込む価値がない。
/// 署名（SigV4）は仕様どおりに書けば 60 行ほどで済む。
///
/// MinIO・Cloudflare R2・Backblaze B2 の S3 互換入口も同じ経路で扱える。
/// </summary>
public sealed class S3FileSource : IFileSource
{
    private readonly HttpClient _client;
    private readonly bool _ownsClient;
    private readonly S3Settings _settings;

    /// <summary>いまの時刻。**試験で固定できるようにする**（署名は時刻を含む）。</summary>
    private readonly Func<DateTimeOffset> _now;

    public S3FileSource(S3Settings settings, HttpClient? client = null,
        Func<DateTimeOffset>? now = null)
    {
        // **鍵が ASCII でなければ、ここで断る。** HTTP のヘッダは ASCII しか
        // 載らないので、そのまま進むと送る直前に
        // 「Request headers must contain only ASCII characters」で落ちる。
        // その文言からは、鍵が原因だと分からない。
        if (!System.Text.Ascii.IsValid(settings.AccessKey)
            || !System.Text.Ascii.IsValid(settings.SecretKey))
        {
            throw new ArgumentException(
                "S3 の鍵は ASCII の文字だけで書きます（HTTP のヘッダに載るため）。");
        }

        _settings = settings;
        _ownsClient = client is null;
        _client = client ?? new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
        _now = now ?? (() => DateTimeOffset.UtcNow);
    }

    public string Display => $"s3://{_settings.Bucket}/{_settings.Prefix}".TrimEnd('/');

    public bool CanWrite { get; init; } = true;

    // --- 署名 ---

    private const string Algorithm = "AWS4-HMAC-SHA256";

    /// <summary>
    /// SigV4 で署名する。
    ///
    /// **手順を仕様の順に書く。** 途中を入れ替えると通らないうえ、
    /// 相手は「署名が違う」としか言わないので、どこが違うのか分からなくなる。
    /// </summary>
    internal void Sign(HttpRequestMessage request, byte[] payload)
    {
        var now = _now().UtcDateTime;
        var stamp = now.ToString("yyyyMMdd'T'HHmmss'Z'", CultureInfo.InvariantCulture);
        var date = now.ToString("yyyyMMdd", CultureInfo.InvariantCulture);
        var payloadHash = Hex(SHA256.HashData(payload));

        var uri = request.RequestUri!;
        request.Headers.Host = uri.IsDefaultPort ? uri.Host : $"{uri.Host}:{uri.Port}";
        request.Headers.TryAddWithoutValidation("x-amz-date", stamp);
        request.Headers.TryAddWithoutValidation("x-amz-content-sha256", payloadHash);

        // 1. 正準リクエスト
        var canonical = CanonicalRequest(
            request.Method.Method, uri, request.Headers.Host!, stamp, payloadHash);

        // 2. 署名の対象
        var scope = $"{date}/{_settings.Region}/s3/aws4_request";
        var toSign = string.Join('\n',
            Algorithm, stamp, scope, Hex(SHA256.HashData(Encoding.UTF8.GetBytes(canonical))));

        // 3. 鍵を日付・地域・役務の順に畳む
        var key = Hmac(Encoding.UTF8.GetBytes("AWS4" + _settings.SecretKey), date);
        key = Hmac(key, _settings.Region);
        key = Hmac(key, "s3");
        key = Hmac(key, "aws4_request");
        var signature = Hex(Hmac(key, toSign));

        request.Headers.TryAddWithoutValidation("Authorization",
            $"{Algorithm} Credential={_settings.AccessKey}/{scope}, "
            + $"SignedHeaders={SignedHeaders}, Signature={signature}");
    }

    internal const string SignedHeaders = "host;x-amz-content-sha256;x-amz-date";

    /// <summary>
    /// 正準リクエストを組み立てる。
    ///
    /// **順序も改行も仕様どおりに。** ヘッダの並びの後には空行が 1 つ入る
    /// （各ヘッダ行が改行で終わり、その後にさらに区切りの改行が来るため）。
    /// </summary>
    internal static string CanonicalRequest(
        string method, Uri uri, string host, string stamp, string payloadHash)
    {
        var headers = $"host:{host}\n"
            + $"x-amz-content-sha256:{payloadHash}\n"
            + $"x-amz-date:{stamp}\n";

        return string.Join('\n',
            method,
            // **既に符号化された形をそのまま使う。** S3 はここで再符号化しない
            // （汎用の SigV4 は掛ける。参照実装を選び違えると %E6 が %25E6 になり、
            // 日本語のファイル名でだけ署名が合わなくなる）。
            uri.AbsolutePath.Length == 0 ? "/" : uri.AbsolutePath,
            CanonicalQuery(uri.Query),
            headers,
            SignedHeaders,
            payloadHash);
    }

    /// <summary>問い合わせの並びを名前順にする。**順序が違うと署名が合わない。**</summary>
    internal static string CanonicalQuery(string query)
    {
        if (query.Length <= 1)
        {
            return string.Empty;
        }
        var pairs = query.TrimStart('?').Split('&', StringSplitOptions.RemoveEmptyEntries)
            .Select(part =>
            {
                var equals = part.IndexOf('=');
                return equals < 0 ? (Key: part, Value: string.Empty)
                    : (Key: part[..equals], Value: part[(equals + 1)..]);
            })
            .OrderBy(p => p.Key, StringComparer.Ordinal);
        return string.Join('&', pairs.Select(p => $"{p.Key}={p.Value}"));
    }

    private static byte[] Hmac(byte[] key, string data)
        => HMACSHA256.HashData(key, Encoding.UTF8.GetBytes(data));

    private static string Hex(byte[] bytes) => Convert.ToHexString(bytes).ToLowerInvariant();

    // --- 場所の組み立て ---

    private string KeyFor(string relativePath)
    {
        var cleaned = relativePath.Replace('\\', '/').Trim('/');
        var prefix = _settings.Prefix.Trim('/');
        return prefix.Length == 0 ? cleaned
            : cleaned.Length == 0 ? prefix
            : $"{prefix}/{cleaned}";
    }

    private Uri UriFor(string key, string query = "")
    {
        var endpoint = _settings.Endpoint.TrimEnd('/');
        // **鍵の各段を個別に符号化する。** 丸ごとだと `/` が潰れて別の場所を指す。
        var encoded = string.Join('/', key.Split('/').Select(Uri.EscapeDataString));

        var path = _settings.UseVirtualHost
            ? $"{endpoint}/{encoded}"
            : $"{endpoint}/{Uri.EscapeDataString(_settings.Bucket)}/{encoded}";
        return new Uri(path.TrimEnd('/') + query);
    }

    // --- 操作 ---

    public IReadOnlyList<RemoteEntry> List(
        string relativePath, CancellationToken cancellationToken = default)
    {
        var prefix = KeyFor(relativePath);
        if (prefix.Length > 0)
        {
            prefix += "/";
        }

        var result = new List<RemoteEntry>();
        string? token = null;

        do
        {
            // **区切りを指定する。** 指定しないと、深い木の全部が 1 回で返る。
            var query = $"?list-type=2&delimiter=%2F&prefix={Uri.EscapeDataString(prefix)}"
                + (token is not null ? $"&continuation-token={Uri.EscapeDataString(token)}" : string.Empty);

            var request = new HttpRequestMessage(HttpMethod.Get,
                _settings.UseVirtualHost
                    ? new Uri(_settings.Endpoint.TrimEnd('/') + "/" + query)
                    : new Uri($"{_settings.Endpoint.TrimEnd('/')}/{Uri.EscapeDataString(_settings.Bucket)}/{query}"));

            Sign(request, []);
            using var response = Send(request, cancellationToken);
            var body = response.Content.ReadAsStringAsync(cancellationToken).GetAwaiter().GetResult();

            token = ParseListing(body, prefix, result, relativePath);
        }
        // **続きがある限り取りに行く。** 1 回の応答は 1000 件までなので、
        // ここを省くと 1001 件目から静かに消える。
        while (token is not null);

        return result;
    }

    /// <summary>
    /// 一覧の応答を読む。続きがあれば継続の印を返す。
    ///
    /// <paramref name="basePath"/> は**根からの相対**の起点。これを前に付けないと、
    /// 深い段の項目が「その段からの相対」になり、次に読むときに見つからない
    /// （`下/b.txt` が `b.txt` になって 404 になる）。
    /// </summary>
    internal static string? ParseListing(
        string xml, string prefix, List<RemoteEntry> into, string basePath = "")
    {
        var baseTrimmed = basePath.Trim('/');
        string Join(string name)
            => baseTrimmed.Length == 0 ? name : $"{baseTrimmed}/{name}";

        using var reader = XmlReader.Create(new StringReader(xml), new XmlReaderSettings
        {
            DtdProcessing = DtdProcessing.Prohibit,
            XmlResolver = null,
        });

        string? continuation = null;
        var truncated = false;
        string? key = null;
        long size = 0;
        DateTimeOffset? modified = null;
        var inContents = false;

        reader.Read();
        while (!reader.EOF)
        {
            if (reader.NodeType == XmlNodeType.Element)
            {
                switch (reader.LocalName)
                {
                    case "Contents":
                        inContents = true;
                        key = null; size = 0; modified = null;
                        break;
                    case "Key" when inContents:
                        key = reader.ReadElementContentAsString();
                        continue;
                    case "Size" when inContents:
                        long.TryParse(reader.ReadElementContentAsString(),
                            NumberStyles.Integer, CultureInfo.InvariantCulture, out size);
                        continue;
                    case "LastModified" when inContents:
                        modified = DateTimeOffset.TryParse(reader.ReadElementContentAsString(),
                            CultureInfo.InvariantCulture, DateTimeStyles.None, out var when)
                            ? when : null;
                        continue;
                    case "Prefix" when !inContents:
                    {
                        // CommonPrefixes の中の Prefix はフォルダーに当たる。
                        var value = reader.ReadElementContentAsString();
                        if (value.Length > prefix.Length)
                        {
                            into.Add(new RemoteEntry(
                                Join(value[prefix.Length..].TrimEnd('/')), true, 0, null));
                        }
                        continue;
                    }
                    case "IsTruncated":
                        truncated = reader.ReadElementContentAsString()
                            .Equals("true", StringComparison.OrdinalIgnoreCase);
                        continue;
                    case "NextContinuationToken":
                        continuation = reader.ReadElementContentAsString();
                        continue;
                }
            }
            else if (reader.NodeType == XmlNodeType.EndElement && reader.LocalName == "Contents")
            {
                inContents = false;
                if (key is { Length: > 0 } && key.Length > prefix.Length)
                {
                    into.Add(new RemoteEntry(Join(key[prefix.Length..]), false, size, modified));
                }
            }
            reader.Read();
        }

        return truncated ? continuation : null;
    }

    public byte[] Read(string relativePath, CancellationToken cancellationToken = default)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, UriFor(KeyFor(relativePath)));
        Sign(request, []);
        using var response = Send(request, cancellationToken);
        return response.Content.ReadAsByteArrayAsync(cancellationToken).GetAwaiter().GetResult();
    }

    public void Write(string relativePath, byte[] content, CancellationToken cancellationToken = default)
    {
        Ensure(CanWrite);
        var request = new HttpRequestMessage(HttpMethod.Put, UriFor(KeyFor(relativePath)))
        {
            Content = new ByteArrayContent(content),
        };
        request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        // **本文の指紋を先に取る。** 署名に入るので、後から中身を変えられない。
        Sign(request, content);
        Send(request, cancellationToken).Dispose();
    }

    public void Delete(string relativePath, CancellationToken cancellationToken = default)
    {
        Ensure(CanWrite);
        var request = new HttpRequestMessage(HttpMethod.Delete, UriFor(KeyFor(relativePath)));
        Sign(request, []);
        Send(request, cancellationToken).Dispose();
    }

    public bool Exists(string relativePath, CancellationToken cancellationToken = default)
    {
        var request = new HttpRequestMessage(HttpMethod.Head, UriFor(KeyFor(relativePath)));
        Sign(request, []);
        using var response = _client.Send(request, cancellationToken);
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
            var status = (int)response.StatusCode;
            var body = response.Content.ReadAsStringAsync(cancellationToken).GetAwaiter().GetResult();
            response.Dispose();

            // **相手が返した理由をそのまま出す。** S3 の応答には
            // <Code>SignatureDoesNotMatch</Code> のような手掛かりが入っている。
            var code = Extract(body, "Code");
            var message = Extract(body, "Message");
            throw new IOException(
                $"S3 が {status} を返しました"
                + (code is not null ? $"（{code}）" : string.Empty)
                + (message is not null ? $": {message}" : $": {request.RequestUri}"));
        }
        return response;
    }

    private static string? Extract(string xml, string element)
    {
        var open = $"<{element}>";
        var close = $"</{element}>";
        var start = xml.IndexOf(open, StringComparison.Ordinal);
        var end = xml.IndexOf(close, StringComparison.Ordinal);
        return start >= 0 && end > start ? xml[(start + open.Length)..end] : null;
    }

    public void Dispose()
    {
        if (_ownsClient)
        {
            _client.Dispose();
        }
    }
}
