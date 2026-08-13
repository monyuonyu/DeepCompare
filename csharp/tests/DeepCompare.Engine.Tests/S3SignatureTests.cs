using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// SigV4 の署名を、本家（botocore）の計算結果と突き合わせる。
///
/// **署名は「合っているか」しか分からない。** 相手は「違う」としか言わないので、
/// 自分の試験だけでは、仕様を取り違えたまま固定してしまう。
/// 期待値は botocore で計算したもので、**手で書いたものではない。**
/// </summary>
public sealed class S3SignatureTests
{
    private static readonly S3Settings Settings = new(
        "https://s3.ap-northeast-1.amazonaws.com", "my-bucket", "AKIAIOSFODNN7EXAMPLE",
        "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
    {
        Region = "ap-northeast-1",
    };

    /// <summary>時刻を固定する。署名は時刻を含むので、これが無いと比べられない。</summary>
    private static readonly DateTimeOffset Fixed =
        new(2026, 8, 13, 12, 34, 56, TimeSpan.Zero);

    private static string SignatureOf(HttpMethod method, string url, byte[] payload)
    {
        using var client = new HttpClient();
        using var source = new S3FileSource(Settings, client, () => Fixed);

        var request = new HttpRequestMessage(method, url);
        source.Sign(request, payload);

        var header = request.Headers.GetValues("Authorization").Single();
        const string Marker = "Signature=";
        return header[(header.IndexOf(Marker, StringComparison.Ordinal) + Marker.Length)..];
    }

    [Theory]
    [InlineData("GET", "https://s3.ap-northeast-1.amazonaws.com/my-bucket/a.txt", "",
        "4f86984e47a7b71a2c370a6cec9429fe321ba2a6a9c1a0bd3a00ab808a7b74fc")]
    [InlineData("PUT", "https://s3.ap-northeast-1.amazonaws.com/my-bucket/deep/b.txt", "hello",
        "169198cb094c325075f33bd8438084fdf8f76c3ef7fda389ace1318f0e1a233c")]
    [InlineData("DELETE", "https://s3.ap-northeast-1.amazonaws.com/my-bucket/c.txt", "",
        "1676a875d5fcdd3b793e184433879706672e5b5fa49a21cebcc44e34a08061b7")]
    // 日本語の鍵。**符号化した形をそのまま署名に使う。** S3 はここで再符号化
    // しない（汎用の SigV4 は掛ける）。期待値は S3SigV4Auth で計算したもの。
    [InlineData("GET",
        "https://s3.ap-northeast-1.amazonaws.com/my-bucket/%E6%97%A5%E6%9C%AC%E8%AA%9E.txt", "",
        "bb22dd0d22dc65b569f0a41fdd422f6ca1e2fb4f90f6e2fef2833d60b1032944")]
    public void 本家と同じ署名になる(string method, string url, string body, string expected)
    {
        Assert.Equal(expected,
            SignatureOf(new HttpMethod(method), url, Encoding.UTF8.GetBytes(body)));
    }

    [Fact]
    public void 正準リクエストが本家と一致する()
    {
        // 署名が食い違ったときに**どこが違うか分かる**ように、途中の形も固定する。
        const string Hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        var canonical = S3FileSource.CanonicalRequest(
            "GET", new Uri("https://s3.ap-northeast-1.amazonaws.com/my-bucket/a.txt"),
            "s3.ap-northeast-1.amazonaws.com", "20260813T123456Z", Hash);

        // ヘッダの並びの後に**空行が 1 つ**入る（各行が改行で終わり、
        // その後に区切りの改行が来るため）。ここを外すと通らない。
        Assert.Equal(
            "GET\n/my-bucket/a.txt\n\n"
            + "host:s3.ap-northeast-1.amazonaws.com\n"
            + $"x-amz-content-sha256:{Hash}\n"
            + "x-amz-date:20260813T123456Z\n\n"
            + "host;x-amz-content-sha256;x-amz-date\n"
            + Hash,
            canonical);
    }

    [Fact]
    public void 鍵がASCIIでなければ断る()
    {
        // HTTP のヘッダは ASCII しか載らない。そのまま進むと送る直前に
        // 「Request headers must contain only ASCII characters」で落ち、
        // **その文言からは鍵が原因だと分からない。**
        var error = Assert.Throws<ArgumentException>(() => new S3FileSource(
            new S3Settings("https://例", "bucket", "日本語の鍵", "秘密")));

        Assert.Contains("ASCII", error.Message);
    }

    [Fact]
    public void 本文が変われば署名も変わる()
    {
        // 本文の指紋が署名に入る。**後から中身を差し替えられない。**
        var url = "https://s3.ap-northeast-1.amazonaws.com/my-bucket/a.txt";
        Assert.NotEqual(
            SignatureOf(HttpMethod.Put, url, "あ"u8.ToArray()),
            SignatureOf(HttpMethod.Put, url, "い"u8.ToArray()));
    }

    [Fact]
    public void 署名に要るヘッダが全部付く()
    {
        using var client = new HttpClient();
        using var source = new S3FileSource(Settings, client, () => Fixed);
        var request = new HttpRequestMessage(HttpMethod.Get,
            "https://s3.ap-northeast-1.amazonaws.com/my-bucket/a.txt");

        source.Sign(request, []);

        Assert.Equal("20260813T123456Z", request.Headers.GetValues("x-amz-date").Single());
        // 空の本文の SHA-256。相手はこれを見て本文の完全性を確かめる。
        Assert.Equal("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            request.Headers.GetValues("x-amz-content-sha256").Single());
        Assert.Equal("s3.ap-northeast-1.amazonaws.com", request.Headers.Host);
    }

    [Fact]
    public void 問い合わせの並びを名前順にする()
    {
        // **順序が違うと署名が合わない。**
        Assert.Equal("a=1&b=2&c=3", S3FileSource.CanonicalQuery("?c=3&a=1&b=2"));
        Assert.Equal("delimiter=%2F&list-type=2",
            S3FileSource.CanonicalQuery("?list-type=2&delimiter=%2F"));
        Assert.Equal(string.Empty, S3FileSource.CanonicalQuery(""));
        Assert.Equal(string.Empty, S3FileSource.CanonicalQuery("?"));
        // 値の無い問い合わせも形を保つ。
        Assert.Equal("acl=", S3FileSource.CanonicalQuery("?acl"));
    }
}
