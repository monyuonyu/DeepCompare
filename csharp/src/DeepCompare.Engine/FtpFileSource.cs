using System.Globalization;
using System.Net;
using System.Net.Security;
using System.Net.Sockets;
using System.Security.Authentication;
using System.Text;

namespace DeepCompare.Engine;

/// <summary>FTP の接続設定。</summary>
public sealed record FtpSettings(string Host, string User, string Password)
{
    public int Port { get; init; } = 21;

    /// <summary>
    /// TLS で包むか（明示的 FTPS、<c>AUTH TLS</c>）。
    ///
    /// **平文の FTP は合言葉がそのまま流れる。** 相手が対応しているなら
    /// こちらを使う。既定を切ってあるのは、古い置き場が対応していないため。
    /// </summary>
    public bool UseTls { get; init; }

    /// <summary>
    /// 証明書を検証するか。**既定は検証する。**
    ///
    /// 自己署名の置き場を相手にするときだけ切る。既定で切ると、
    /// 「TLS にした」という安心だけが残って中身が守られない。
    /// </summary>
    public bool ValidateCertificate { get; init; } = true;

    public string Root { get; init; } = "/";
}

/// <summary>
/// FTP / FTPS の置き場。
///
/// **依存を増やさない。** FTP は平文の規約（RFC 959）なので、TcpClient で
/// 書ける。.NET の FtpWebRequest は廃止済みで、置き換えの標準実装は無い。
///
/// **決めごと**:
/// - 受動モード（PASV）だけを使う。能動モードは相手からこちらへ繋ぎ返す形で、
///   ファイアウォールと NAT の内側ではまず通らない
/// - 一覧は MLSD を優先する。LIST は人が読む形で、実装ごとに違う
/// - 転送は常にバイナリ（TYPE I）。テキストモードは改行を勝手に書き換える
/// </summary>
public sealed class FtpFileSource : IFileSource
{
    private readonly FtpSettings _settings;
    private TcpClient? _control;
    private Stream? _stream;
    private StreamReader? _reader;

    public FtpFileSource(FtpSettings settings)
    {
        _settings = settings;
        Connect();
    }

    public string Display =>
        $"{(_settings.UseTls ? "ftps" : "ftp")}://{_settings.Host}{_settings.Root}";

    public bool CanWrite { get; init; } = true;

    // --- 接続 ---

    private void Connect()
    {
        _control = new TcpClient();
        _control.Connect(_settings.Host, _settings.Port);
        _stream = _control.GetStream();
        _reader = new StreamReader(_stream, Encoding.UTF8, leaveOpen: true);

        Expect(Read(), 220);

        if (_settings.UseTls)
        {
            // **認証の前に包む。** 後にすると合言葉が平文で流れる。
            Expect(Command("AUTH TLS"), 234);
            var ssl = new SslStream(_stream, leaveInnerStreamOpen: false,
                _settings.ValidateCertificate
                    ? null
                    : (_, _, _, _) => true);
            ssl.AuthenticateAsClient(new SslClientAuthenticationOptions
            {
                TargetHost = _settings.Host,
                EnabledSslProtocols = SslProtocols.Tls12 | SslProtocols.Tls13,
            });
            _stream = ssl;
            _reader = new StreamReader(_stream, Encoding.UTF8, leaveOpen: true);
        }

        var user = Command($"USER {_settings.User}");
        // 331 は「合言葉をどうぞ」。230 なら合言葉が要らなかった。
        if (Code(user) == 331)
        {
            Expect(Command($"PASS {_settings.Password}"), 230);
        }
        else
        {
            Expect(user, 230);
        }

        if (_settings.UseTls)
        {
            // データ接続も包む。**ここを忘れると中身だけ平文で流れる。**
            Expect(Command("PBSZ 0"), 200);
            Expect(Command("PROT P"), 200);
        }

        // **常にバイナリ。** テキストモードは改行を勝手に書き換えるので、
        // 比較の対象としては別物になる。
        Expect(Command("TYPE I"), 200);

        // 名前を UTF-8 で送れるなら、そう伝える。
        Command("OPTS UTF8 ON");
    }

    private string Read()
    {
        var line = _reader!.ReadLine()
            ?? throw new IOException("FTP の相手が黙って切りました。");

        // 複数行の応答は「符号-」で始まり、「符号 」で終わる。
        if (line.Length >= 4 && line[3] == '-')
        {
            var code = line[..3];
            string next;
            do
            {
                next = _reader.ReadLine()
                    ?? throw new IOException("FTP の応答が途中で切れました。");
            }
            while (!(next.Length >= 4 && next.StartsWith(code, StringComparison.Ordinal)
                     && next[3] == ' '));
            return next;
        }
        return line;
    }

    private string Command(string command)
    {
        var bytes = Encoding.UTF8.GetBytes(command + "\r\n");
        _stream!.Write(bytes);
        _stream.Flush();
        return Read();
    }

    private static int Code(string response)
        => response.Length >= 3 && int.TryParse(response[..3], out var code) ? code : 0;

    private static void Expect(string response, params int[] codes)
    {
        if (!codes.Contains(Code(response)))
        {
            // **相手が言った理由をそのまま出す。** 「失敗しました」だけだと、
            // 合言葉が違うのか、場所が無いのか、容量が無いのかが分からない。
            throw new IOException($"FTP: {response}");
        }
    }

    /// <summary>受動モードでデータ接続を開く。</summary>
    private TcpClient OpenData()
    {
        var response = Command("PASV");
        Expect(response, 227);

        // 227 Entering Passive Mode (127,0,0,1,201,110)
        var open = response.IndexOf('(');
        var close = response.IndexOf(')');
        if (open < 0 || close < open)
        {
            throw new IOException($"PASV の応答を読めません: {response}");
        }

        var parts = response[(open + 1)..close].Split(',');
        if (parts.Length < 6
            || !int.TryParse(parts[4], out var high) || !int.TryParse(parts[5], out var low))
        {
            throw new IOException($"PASV の応答を読めません: {response}");
        }

        var port = high * 256 + low;
        // **主機は元の接続先を使う。** 応答に書かれた住所は、NAT の内側の
        // 届かない値であることがある。
        var data = new TcpClient();
        data.Connect(_settings.Host, port);
        return data;
    }

    private Stream WrapData(TcpClient data)
    {
        Stream stream = data.GetStream();
        if (!_settings.UseTls)
        {
            return stream;
        }
        var ssl = new SslStream(stream, leaveInnerStreamOpen: false,
            _settings.ValidateCertificate ? null : (_, _, _, _) => true);
        ssl.AuthenticateAsClient(new SslClientAuthenticationOptions
        {
            TargetHost = _settings.Host,
            EnabledSslProtocols = SslProtocols.Tls12 | SslProtocols.Tls13,
        });
        return ssl;
    }

    // --- 場所 ---

    private string PathFor(string relativePath)
    {
        var root = _settings.Root.TrimEnd('/');
        var cleaned = relativePath.Replace('\\', '/').Trim('/');
        return cleaned.Length == 0 ? (root.Length == 0 ? "/" : root) : $"{root}/{cleaned}";
    }

    // --- 操作 ---

    public IReadOnlyList<RemoteEntry> List(
        string relativePath, CancellationToken cancellationToken = default)
    {
        var path = PathFor(relativePath);

        using var data = OpenData();
        // MLSD は機械が読む形で返る。**LIST は人が読む形で、実装ごとに違う。**
        var response = Command($"MLSD {path}");
        var useMlsd = Code(response) is 125 or 150;

        if (!useMlsd)
        {
            data.Dispose();
            using var fallback = OpenData();
            Expect(Command($"LIST {path}"), 125, 150);
            var listText = ReadAll(fallback);
            Expect(Read(), 226, 250);
            return ParseList(listText, relativePath);
        }

        var text = ReadAll(data);
        Expect(Read(), 226, 250);
        return ParseMlsd(text, relativePath);
    }

    private string ReadAll(TcpClient data)
    {
        using var stream = WrapData(data);
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return reader.ReadToEnd();
    }

    /// <summary>
    /// MLSD の応答を読む。<c>type=file;size=12;modify=20260813050000; 名前</c> の形。
    /// </summary>
    internal static IReadOnlyList<RemoteEntry> ParseMlsd(string text, string basePath)
    {
        var result = new List<RemoteEntry>();
        var prefix = basePath.Trim('/');

        foreach (var line in text.Split('\n'))
        {
            var trimmed = line.TrimEnd('\r');
            var space = trimmed.IndexOf(' ');
            if (space < 0)
            {
                continue;
            }

            var name = trimmed[(space + 1)..];
            // **自分自身と親は落とす。** 残すと木が循環する。
            if (name is "." or ".." || name.Length == 0)
            {
                continue;
            }

            var facts = trimmed[..space].Split(';', StringSplitOptions.RemoveEmptyEntries);
            var isDirectory = false;
            long size = 0;
            DateTimeOffset? modified = null;

            foreach (var fact in facts)
            {
                var equals = fact.IndexOf('=');
                if (equals < 0)
                {
                    continue;
                }
                var key = fact[..equals];
                var value = fact[(equals + 1)..];

                if (key.Equals("type", StringComparison.OrdinalIgnoreCase))
                {
                    if (value is "cdir" or "pdir")
                    {
                        isDirectory = true;
                        name = string.Empty;   // 自分自身・親は落とす
                    }
                    isDirectory |= value.Equals("dir", StringComparison.OrdinalIgnoreCase);
                }
                else if (key.Equals("size", StringComparison.OrdinalIgnoreCase))
                {
                    long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out size);
                }
                else if (key.Equals("modify", StringComparison.OrdinalIgnoreCase))
                {
                    // YYYYMMDDHHMMSS。**UTC で来る**（規約で決まっている）。
                    if (DateTime.TryParseExact(value[..Math.Min(14, value.Length)],
                            "yyyyMMddHHmmss", CultureInfo.InvariantCulture,
                            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
                            out var when))
                    {
                        modified = new DateTimeOffset(when, TimeSpan.Zero);
                    }
                }
            }

            if (name.Length == 0)
            {
                continue;
            }
            result.Add(new RemoteEntry(
                prefix.Length == 0 ? name : $"{prefix}/{name}", isDirectory, size, modified));
        }
        return result;
    }

    /// <summary>
    /// LIST の応答を読む（MLSD が使えない相手向け）。
    ///
    /// **Unix 風の形だけを見る。** 実装ごとに違ううえ、時刻は年が落ちて
    /// いたりする。ここで取れなければ時刻は null にして、大きさと中身で比べる。
    /// </summary>
    internal static IReadOnlyList<RemoteEntry> ParseList(string text, string basePath)
    {
        var result = new List<RemoteEntry>();
        var prefix = basePath.Trim('/');

        foreach (var line in text.Split('\n'))
        {
            var trimmed = line.TrimEnd('\r');
            if (trimmed.Length < 10)
            {
                continue;
            }

            // drwxr-xr-x 2 owner group 4096 Aug 13 05:00 名前
            var parts = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 9)
            {
                continue;
            }

            // **名前に空白が入りうる。** 8 個目までを飛ばして、残り全部を名前にする。
            var nameStart = 0;
            var seen = 0;
            for (var i = 0; i < trimmed.Length && seen < 8; i++)
            {
                if (trimmed[i] != ' ')
                {
                    continue;
                }
                while (i < trimmed.Length && trimmed[i] == ' ')
                {
                    i++;
                }
                seen++;
                nameStart = i;
                i--;
            }

            var name = trimmed[nameStart..].Trim();
            if (name is "." or ".." || name.Length == 0)
            {
                continue;
            }

            var isDirectory = trimmed[0] == 'd';
            long.TryParse(parts[4], NumberStyles.Integer, CultureInfo.InvariantCulture, out var size);

            result.Add(new RemoteEntry(
                prefix.Length == 0 ? name : $"{prefix}/{name}",
                isDirectory, isDirectory ? 0 : size, null));
        }
        return result;
    }

    public byte[] Read(string relativePath, CancellationToken cancellationToken = default)
    {
        using var data = OpenData();
        Expect(Command($"RETR {PathFor(relativePath)}"), 125, 150);

        using var stream = WrapData(data);
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        // **先に読み切ってから応答を待つ。** 逆にすると、相手は書き込みで
        // 止まり、こちらは応答待ちで止まる。
        Expect(Read(), 226, 250);
        return memory.ToArray();
    }

    public void Write(string relativePath, byte[] content, CancellationToken cancellationToken = default)
    {
        Ensure(CanWrite);
        using var data = OpenData();
        Expect(Command($"STOR {PathFor(relativePath)}"), 125, 150);

        using (var stream = WrapData(data))
        {
            stream.Write(content);
            stream.Flush();
        }
        Expect(Read(), 226, 250);
    }

    public void Delete(string relativePath, CancellationToken cancellationToken = default)
    {
        Ensure(CanWrite);
        Expect(Command($"DELE {PathFor(relativePath)}"), 250);
    }

    public bool Exists(string relativePath, CancellationToken cancellationToken = default)
        => Code(Command($"SIZE {PathFor(relativePath)}")) == 213;

    private void Ensure(bool allowed)
    {
        if (!allowed)
        {
            throw new InvalidOperationException($"{Display} は読み取り専用です。");
        }
    }

    public void Dispose()
    {
        try
        {
            if (_stream is not null)
            {
                Command("QUIT");
            }
        }
        catch (Exception error) when (error is IOException or SocketException
                                        or ObjectDisposedException)
        {
            // 切れている相手に別れを言えなくても、こちらの後始末は続ける。
        }

        _reader?.Dispose();
        _stream?.Dispose();
        _control?.Dispose();
    }
}
