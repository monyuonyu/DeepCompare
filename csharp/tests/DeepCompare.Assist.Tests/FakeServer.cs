using System.Net;
using System.Text;

namespace DeepCompare.Assist.Tests;

/// <summary>
/// OpenAI 互換の相手を演じる、本物の HTTP サーバー。
///
/// **HttpClient を差し替えて済ませない。** それだと確かめられるのは
/// 「私が思っている形」だけで、実際に線の上を流れるバイト列は見ない。
/// SSE の区切り方や、途中で切れた行の扱いは、本物の通信でしか出ない。
/// </summary>
public sealed class FakeServer : IDisposable
{
    private readonly HttpListener _listener = new();
    private readonly CancellationTokenSource _stopping = new();
    private readonly Task _loop;

    /// <summary>受け取った本文。**何を送ったかを確かめるために残す。**</summary>
    public List<string> ReceivedBodies { get; } = [];

    /// <summary>受け取った経路。</summary>
    public List<string> ReceivedPaths { get; } = [];

    /// <summary>受け取った認証ヘッダ。</summary>
    public List<string?> ReceivedAuth { get; } = [];

    /// <summary>返す中身を決める。</summary>
    public Func<string, string, (int Status, string Body, string ContentType)> Handler { get; set; }
        = (_, _) => (200, "{}", "application/json");

    /// <summary>返事を遅らせる。時限の試験に使う。</summary>
    public TimeSpan Delay { get; set; } = TimeSpan.Zero;

    public string BaseUrl { get; }

    public FakeServer()
    {
        // **空いている口を borrow する。** 決め打ちの番号だと、
        // 試験を並べて走らせたときにぶつかる。
        var port = FreePort();
        BaseUrl = $"http://127.0.0.1:{port}/v1";
        _listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        _listener.Start();
        _loop = Task.Run(LoopAsync);
    }

    private static int FreePort()
    {
        using var probe = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0);
        probe.Start();
        var port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();
        return port;
    }

    private async Task LoopAsync()
    {
        while (!_stopping.IsCancellationRequested)
        {
            HttpListenerContext context;
            try
            {
                context = await _listener.GetContextAsync();
            }
            catch (Exception error) when (error is HttpListenerException or ObjectDisposedException)
            {
                return;   // 止めた。
            }

            try
            {
                await RespondAsync(context);
            }
            catch (Exception error) when (error is HttpListenerException or IOException)
            {
                // 相手が先に切った。試験としては珍しくない（時限の確認など）。
            }
        }
    }

    private async Task RespondAsync(HttpListenerContext context)
    {
        var path = context.Request.Url?.AbsolutePath ?? string.Empty;
        var body = string.Empty;

        if (context.Request.HasEntityBody)
        {
            using var reader = new StreamReader(context.Request.InputStream, Encoding.UTF8);
            body = await reader.ReadToEndAsync();
        }

        lock (ReceivedBodies)
        {
            ReceivedPaths.Add(path);
            ReceivedBodies.Add(body);
            ReceivedAuth.Add(context.Request.Headers["Authorization"]);
        }

        if (Delay > TimeSpan.Zero)
        {
            await Task.Delay(Delay, _stopping.Token);
        }

        var (status, text, contentType) = Handler(path, body);
        context.Response.StatusCode = status;
        context.Response.ContentType = contentType;

        var bytes = Encoding.UTF8.GetBytes(text);
        context.Response.ContentLength64 = bytes.Length;
        await context.Response.OutputStream.WriteAsync(bytes);
        context.Response.Close();
    }

    /// <summary>
    /// 1 通の返事を、OpenAI の形に包む。
    ///
    /// **手で JSON を書かない。** 括弧の数を目で数える羽目になり、
    /// 壊れていても「相手の返事が読めない」としか出ないので原因が遠い。
    /// </summary>
    public static string ChatResponse(string content)
    {
        var response = new System.Text.Json.Nodes.JsonObject
        {
            ["id"] = "x",
            ["object"] = "chat.completion",
            ["choices"] = new System.Text.Json.Nodes.JsonArray
            {
                new System.Text.Json.Nodes.JsonObject
                {
                    ["index"] = 0,
                    ["message"] = new System.Text.Json.Nodes.JsonObject
                    {
                        ["role"] = "assistant",
                        ["content"] = content,
                    },
                    ["finish_reason"] = "stop",
                },
            },
        };
        return response.ToJsonString();
    }

    /// <summary>少しずつ返す形（SSE）に包む。</summary>
    public static string StreamResponse(params string[] pieces)
    {
        var text = new StringBuilder();
        foreach (var piece in pieces)
        {
            var escaped = System.Text.Json.JsonSerializer.Serialize(piece);
            text.Append($"data: {{\"choices\":[{{\"delta\":{{\"content\":{escaped}}}}}]}}\n\n");
        }
        text.Append("data: [DONE]\n\n");
        return text.ToString();
    }

    public void Dispose()
    {
        _stopping.Cancel();
        _listener.Close();
        try
        {
            _loop.Wait(TimeSpan.FromSeconds(2));
        }
        catch (AggregateException)
        {
            // 止めるときの例外は握る。
        }
        _stopping.Dispose();
    }
}
