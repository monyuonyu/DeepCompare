using System.Net.Http.Headers;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Unicode;

namespace DeepCompare.Assist;

/// <summary>やり取りの 1 通。</summary>
public sealed record ChatMessage(string Role, string Content)
{
    public static ChatMessage System(string content) => new("system", content);
    public static ChatMessage User(string content) => new("user", content);
}

/// <summary>繋がらなかった、あるいは相手が変な物を返した。</summary>
public sealed class AssistException(string message, Exception? inner = null)
    : Exception(message, inner);

/// <summary>
/// OpenAI 互換のエンドポイントへ話しかける。
///
/// **外部 SDK を使わない。** 叩くのは <c>/chat/completions</c> ひとつで、
/// HttpClient と System.Text.Json で足りる。Ollama・LM Studio・llama.cpp が
/// どれも同じ形を話すので、これで 3 つとも繋がる。
/// </summary>
public sealed class ChatClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly AssistSettings _settings;
    private readonly bool _ownsHttp;

    public ChatClient(AssistSettings settings, HttpClient? http = null)
    {
        if (!settings.IsConfigured)
        {
            throw new ArgumentException(
                "接続先とモデルの名前が要ります。設定していないなら、そもそも"
                + "この機能を出さないでください。", nameof(settings));
        }

        _settings = settings;
        _ownsHttp = http is null;
        _http = http ?? new HttpClient();
        _http.Timeout = settings.Timeout;

        if (settings.ApiKey is { Length: > 0 } key)
        {
            // **非 ASCII の鍵をここで断る。** そのまま渡すと HttpClient が
            // 「Request headers must contain only ASCII characters」で落ち、
            // それだけ見ても原因が分からない（繋がらないのだと読んでしまう）。
            if (!key.All(char.IsAscii))
            {
                throw new ArgumentException(
                    "鍵に ASCII 以外の文字が入っています。貼り付けのときに"
                    + "全角文字や日本語が混じっていないか確かめてください。",
                    nameof(settings));
            }

            // **鍵はここでしか触らない。** 記録にも表示にも回さない。
            _http.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", key);
        }
    }

    /// <summary>
    /// 繋がるかを確かめる。**短い時限で諦める。**
    /// 設定画面の「試す」でしか呼ばない。比較の経路からは呼ばない。
    /// </summary>
    public async Task<bool> ProbeAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            using var timed = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timed.CancelAfter(_settings.ProbeTimeout);

            using var response = await _http.GetAsync(
                _settings.UrlFor("models"), timed.Token);
            return response.IsSuccessStatusCode;
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException)
        {
            // **繋がらないことは異常ではない。** 相手を起動していないだけ。
            return false;
        }
    }

    /// <summary>
    /// answer を丸ごと受け取る。
    /// </summary>
    /// <param name="schema">
    /// 返させたい形（JSON Schema）。null なら自由な文。
    /// **形を指定できるのはオフラインでも同じ。** llama.cpp / Ollama は
    /// 文法による制約付き復号を持つので、安全性の設計が外部 API 固有の
    /// 機能に依存しない。
    /// </param>
    public async Task<string> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        JsonNode? schema = null,
        CancellationToken cancellationToken = default)
    {
        var body = BuildRequest(messages, schema, stream: false);
        using var response = await SendAsync(body, cancellationToken);
        var text = await response.Content.ReadAsStringAsync(cancellationToken);

        try
        {
            var json = JsonNode.Parse(text);
            var content = json?["choices"]?[0]?["message"]?["content"]?.GetValue<string>();
            return content
                ?? throw new AssistException("返事に本文がありません。");
        }
        catch (JsonException error)
        {
            throw new AssistException("返事を読めませんでした。", error);
        }
    }

    /// <summary>
    /// 少しずつ受け取る（SSE）。
    ///
    /// **ローカルのモデルは生成が遅い。** 出来上がるまで黙っていると、
    /// 止まっているのか考えているのかが分からない。
    /// </summary>
    public async IAsyncEnumerable<string> StreamAsync(
        IReadOnlyList<ChatMessage> messages,
        JsonNode? schema = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation]
        CancellationToken cancellationToken = default)
    {
        var body = BuildRequest(messages, schema, stream: true);
        using var request = new HttpRequestMessage(
            HttpMethod.Post, _settings.UrlFor("chat/completions"))
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json"),
        };

        HttpResponseMessage response;
        try
        {
            response = await _http.SendAsync(
                request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException)
        {
            throw new AssistException($"{_settings.Endpoint} へ繋がりませんでした。", error);
        }

        using (response)
        {
            await EnsureSuccessAsync(response, cancellationToken);

            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream, Encoding.UTF8);

            while (await reader.ReadLineAsync(cancellationToken) is { } line)
            {
                if (!line.StartsWith("data:", StringComparison.Ordinal))
                {
                    continue;
                }

                var payload = line[5..].Trim();

                // 終わりの合図。**中身を読もうとしない**（JSON ではない）。
                if (payload is "[DONE]")
                {
                    yield break;
                }
                if (payload.Length == 0)
                {
                    continue;
                }

                string? piece = null;
                try
                {
                    piece = JsonNode.Parse(payload)?["choices"]?[0]?["delta"]?["content"]
                        ?.GetValue<string>();
                }
                catch (JsonException)
                {
                    // **1 かけら読めなくても止めない。** 途中で切れた行や
                    // 相手側の余計な出力で、そこまでの結果を捨てるのは惜しい。
                    continue;
                }

                if (piece is { Length: > 0 })
                {
                    yield return piece;
                }
            }
        }
    }

    private async Task<HttpResponseMessage> SendAsync(
        string body, CancellationToken cancellationToken)
    {
        try
        {
            var response = await _http.PostAsync(
                _settings.UrlFor("chat/completions"),
                new StringContent(body, Encoding.UTF8, "application/json"),
                cancellationToken);
            await EnsureSuccessAsync(response, cancellationToken);
            return response;
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException)
        {
            throw new AssistException($"{_settings.Endpoint} へ繋がりませんでした。", error);
        }
    }

    /// <summary>
    /// 失敗したら、**相手が言っている内容を添える。**
    /// 「400 でした」だけでは、モデル名の綴り違いなのか形式の問題なのか分からない。
    /// </summary>
    private static async Task EnsureSuccessAsync(
        HttpResponseMessage response, CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var detail = string.Empty;
        try
        {
            var text = await response.Content.ReadAsStringAsync(cancellationToken);
            detail = text.Length > 400 ? text[..400] + "…" : text;
        }
        catch (Exception error) when (error is IOException or HttpRequestException)
        {
            // 中身が読めなくても、状態だけは伝える。
        }

        throw new AssistException(
            $"応答が {(int)response.StatusCode} でした。"
            + (detail.Length > 0 ? $" {detail}" : string.Empty));
    }

    private string BuildRequest(
        IReadOnlyList<ChatMessage> messages, JsonNode? schema, bool stream)
    {
        var array = new JsonArray();
        foreach (var message in messages)
        {
            array.Add((JsonNode)new JsonObject
            {
                ["role"] = message.Role,
                ["content"] = message.Content,
            });
        }

        var request = new JsonObject
        {
            ["model"] = _settings.Model,
            ["messages"] = array,
            ["stream"] = stream,
            // **低くする。** ここでやらせるのは要約・分類・言い換えで、
            // 発想の広さは要らない。振れ幅は再現しにくい不具合になる。
            ["temperature"] = 0.2,
        };

        if (schema is not null)
        {
            request["response_format"] = new JsonObject
            {
                ["type"] = "json_schema",
                ["json_schema"] = new JsonObject
                {
                    ["name"] = "assist",
                    ["strict"] = true,
                    ["schema"] = schema.DeepClone(),
                },
            };
        }

        return request.ToJsonString(JsonOptions);
    }

    /// <summary>
    /// 書き出しの設定。
    ///
    /// **日本語をエスケープしない。** 既定だと「説明」が `\u8AAC\u660E` になり、
    /// 1 文字 6 バイトへ膨らむ。前置きも問いも日本語なので、送る量が 3 倍近くになり、
    /// ローカルのモデルではその分だけ待つ時間が伸びる。
    ///
    /// **全部を素通しにはしない。** `<` `>` `&` は逃がしたままにする
    /// （UnsafeRelaxedJsonEscaping はそこも通す）。
    /// </summary>
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        Encoder = JavaScriptEncoder.Create(UnicodeRanges.All),
    };

    public void Dispose()
    {
        if (_ownsHttp)
        {
            _http.Dispose();
        }
    }
}
