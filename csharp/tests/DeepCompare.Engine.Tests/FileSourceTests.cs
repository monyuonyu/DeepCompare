using System.Net;
using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class LocalFileSourceTests : IDisposable
{
    private readonly string _root =
        Path.Combine(Path.GetTempPath(), "dc-src-" + Guid.NewGuid().ToString("N")[..8]);

    public LocalFileSourceTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
        }
    }

    [Fact]
    public void 一段ずつ辿る()
    {
        // **再帰しない。** 深い木で全部取りに行くと、リモートでは往復が爆発する。
        Directory.CreateDirectory(Path.Combine(_root, "下", "さらに下"));
        File.WriteAllText(Path.Combine(_root, "a.txt"), "あ");
        File.WriteAllText(Path.Combine(_root, "下", "b.txt"), "い");

        using var source = new LocalFileSource(_root);

        var top = source.List("");
        Assert.Equal(2, top.Count);
        Assert.Contains(top, e => e.RelativePath == "a.txt" && !e.IsDirectory);
        Assert.Contains(top, e => e.RelativePath == "下" && e.IsDirectory);

        var inner = source.List("下");
        Assert.Contains(inner, e => e.RelativePath == "下/b.txt");
    }

    [Fact]
    public void 読み書きと削除ができる()
    {
        using var source = new LocalFileSource(_root);

        source.Write("深い/場所/c.txt", "内容"u8.ToArray());
        Assert.True(source.Exists("深い/場所/c.txt"));
        Assert.Equal("内容", Encoding.UTF8.GetString(source.Read("深い/場所/c.txt")));

        source.Delete("深い/場所/c.txt");
        Assert.False(source.Exists("深い/場所/c.txt"));
    }

    [Fact]
    public void 根の外は指せない()
    {
        // **リモートから来た名前を信じてはいけない。** `../` を含むパスを
        // そのまま繋ぐと、同期や削除が根の外へ届く。
        using var source = new LocalFileSource(_root);

        Assert.Throws<UnauthorizedAccessException>(() => source.Read("../外.txt"));
        Assert.Throws<UnauthorizedAccessException>(() => source.Delete("下/../../外"));
        Assert.Throws<UnauthorizedAccessException>(
            () => source.Write("../../外.txt", "x"u8.ToArray()));
    }

    [Fact]
    public void 区切りはどちらでも受ける()
    {
        using var source = new LocalFileSource(_root);
        source.Write("下/d.txt", "x"u8.ToArray());

        Assert.True(source.Exists("下/d.txt"));
        Assert.True(source.Exists(@"下\d.txt"));
    }

    [Fact]
    public void 無い場所を並べても落ちない()
    {
        using var source = new LocalFileSource(_root);
        Assert.Empty(source.List("無い場所"));
    }
}

public sealed class WebDavFileSourceTests
{
    /// <summary>やり取りを差し替える。**本物のサーバーを立てずに経路を確かめる。**</summary>
    private sealed class FakeHandler(
        Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
    {
        public List<HttpRequestMessage> Requests { get; } = [];

        protected override HttpResponseMessage Send(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add(request);
            return respond(request);
        }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(Send(request, cancellationToken));
    }

    private const string Listing = """
        <?xml version="1.0"?>
        <D:multistatus xmlns:D="DAV:">
          <D:response>
            <D:href>/dav/</D:href>
            <D:propstat><D:prop><D:resourcetype><D:collection/></D:resourcetype></D:prop></D:propstat>
          </D:response>
          <D:response>
            <D:href>/dav/a.txt</D:href>
            <D:propstat><D:prop>
              <D:resourcetype/>
              <D:getcontentlength>12</D:getcontentlength>
              <D:getlastmodified>Wed, 13 Aug 2026 05:00:00 GMT</D:getlastmodified>
            </D:prop></D:propstat>
          </D:response>
          <D:response>
            <D:href>/dav/%E6%97%A5%E6%9C%AC%E8%AA%9E/</D:href>
            <D:propstat><D:prop><D:resourcetype><D:collection/></D:resourcetype></D:prop></D:propstat>
          </D:response>
        </D:multistatus>
        """;

    [Fact]
    public void 一覧を読む()
    {
        var entries = WebDavFileSource.ParseListing(Listing, "", new Uri("https://例/dav/"));

        // **自分自身は落とす。** 落とさないと、フォルダーが自分の中に入る。
        Assert.Equal(2, entries.Count);

        var file = entries.Single(e => !e.IsDirectory);
        Assert.Equal("a.txt", file.RelativePath);
        Assert.Equal(12, file.Size);
        Assert.Equal(new DateTimeOffset(2026, 8, 13, 5, 0, 0, TimeSpan.Zero), file.Modified);

        // 符号化された名前は戻す。**%E6%97%A5 のまま出しても読めない。**
        Assert.Equal("日本語", entries.Single(e => e.IsDirectory).RelativePath);
    }

    [Theory]
    // **曜日は当てにしない。** 下の 2 つ目は曜日が実際とずれているが、
    // 実装によってはこれを返す。.NET の TryParse は曜日の整合を見るので、
    // そのまま渡すと時刻が丸ごと読めなくなる。
    [InlineData("Thu, 13 Aug 2026 05:00:00 GMT")]
    [InlineData("Wed, 13 Aug 2026 05:00:00 GMT")]
    [InlineData("13 Aug 2026 05:00:00 GMT")]
    [InlineData("2026-08-13T05:00:00Z")]
    public void 日時のいろいろな書き方を読む(string text)
    {
        Assert.Equal(new DateTimeOffset(2026, 8, 13, 5, 0, 0, TimeSpan.Zero),
            WebDavFileSource.ParseHttpDate(text));
    }

    [Fact]
    public void 読めない日時は諦める()
    {
        Assert.Null(WebDavFileSource.ParseHttpDate("いつか"));
        Assert.Null(WebDavFileSource.ParseHttpDate(""));
    }

    [Fact]
    public void 深さ一を指定する()
    {
        // **指定しないと、根に向けた 1 回で全部取りに行く。**
        var handler = new FakeHandler(_ => new HttpResponseMessage(HttpStatusCode.MultiStatus)
        {
            Content = new StringContent(Listing, Encoding.UTF8, "application/xml"),
        });
        using var client = new HttpClient(handler);
        using var source = new WebDavFileSource(new Uri("https://例/dav/"), client);

        source.List("");

        Assert.Equal("1", handler.Requests[0].Headers.GetValues("Depth").Single());
        Assert.Equal("PROPFIND", handler.Requests[0].Method.Method);
    }

    [Fact]
    public void 段ごとに符号化する()
    {
        // 丸ごと符号化すると `/` まで潰れて、別の場所を指す。
        var handler = new FakeHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new ByteArrayContent("中身"u8.ToArray()),
        });
        using var client = new HttpClient(handler);
        using var source = new WebDavFileSource(new Uri("https://例/dav/"), client);

        source.Read("下/日本語 の名前.txt");

        var uri = handler.Requests[0].RequestUri!;
        Assert.Contains("/dav/", uri.AbsoluteUri);
        // 段の区切りは残る（/dav/ で 2 つ、下/ で 1 つ）。
        Assert.Equal(3, uri.AbsolutePath.Count(c => c == '/'));
    }

    [Theory]
    [InlineData(HttpStatusCode.Unauthorized, "認証")]
    [InlineData(HttpStatusCode.Forbidden, "権限")]
    [InlineData(HttpStatusCode.NotFound, "見つかりません")]
    [InlineData(HttpStatusCode.InsufficientStorage, "一杯")]
    public void 失敗の理由をそのまま出す(HttpStatusCode status, string expected)
    {
        // **401 と 404 と 507 では、人がやることが違う。**
        var handler = new FakeHandler(_ => new HttpResponseMessage(status)
        {
            Content = new StringContent(string.Empty),
        });
        using var client = new HttpClient(handler);
        using var source = new WebDavFileSource(new Uri("https://例/dav/"), client);

        var error = Assert.Throws<IOException>(() => source.Read("a.txt"));
        Assert.Contains(expected, error.Message);
    }

    [Fact]
    public void 読み取り専用なら書けない()
    {
        var handler = new FakeHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
        using var client = new HttpClient(handler);
        using var source = new WebDavFileSource(new Uri("https://例/dav/"), client)
        {
            CanWrite = false,
        };

        Assert.Throws<InvalidOperationException>(() => source.Write("a.txt", []));
        Assert.Throws<InvalidOperationException>(() => source.Delete("a.txt"));
        Assert.Empty(handler.Requests);   // **送る前に止める。**
    }
}

public sealed class S3ListingTests
{
    [Fact]
    public void 一覧とフォルダーを読む()
    {
        const string Xml = """
            <?xml version="1.0"?>
            <ListBucketResult>
              <IsTruncated>false</IsTruncated>
              <Contents><Key>prefix/a.txt</Key><Size>10</Size>
                <LastModified>2026-08-13T05:00:00.000Z</LastModified></Contents>
              <Contents><Key>prefix/日本語.txt</Key><Size>20</Size>
                <LastModified>2026-08-13T06:00:00.000Z</LastModified></Contents>
              <CommonPrefixes><Prefix>prefix/下/</Prefix></CommonPrefixes>
            </ListBucketResult>
            """;

        var entries = new List<RemoteEntry>();
        var token = S3ListingTests.Parse(Xml, "prefix/", entries);

        Assert.Null(token);
        Assert.Equal(3, entries.Count);
        Assert.Contains(entries, e => e.RelativePath == "a.txt" && e.Size == 10 && !e.IsDirectory);
        Assert.Contains(entries, e => e.RelativePath == "日本語.txt" && e.Size == 20);
        // CommonPrefixes はフォルダーに当たる。
        Assert.Contains(entries, e => e.RelativePath == "下" && e.IsDirectory);
    }

    [Fact]
    public void 続きがあれば印を返す()
    {
        // **1 回の応答は 1000 件まで。** ここを見落とすと 1001 件目から静かに消える。
        const string Xml = """
            <?xml version="1.0"?>
            <ListBucketResult>
              <IsTruncated>true</IsTruncated>
              <NextContinuationToken>つづき</NextContinuationToken>
              <Contents><Key>p/a.txt</Key><Size>1</Size></Contents>
            </ListBucketResult>
            """;

        var entries = new List<RemoteEntry>();
        Assert.Equal("つづき", Parse(Xml, "p/", entries));
    }

    [Fact]
    public void 続きが無ければ印を返さない()
    {
        const string Xml = """
            <?xml version="1.0"?>
            <ListBucketResult>
              <IsTruncated>false</IsTruncated>
              <NextContinuationToken>使わない</NextContinuationToken>
            </ListBucketResult>
            """;

        Assert.Null(Parse(Xml, "p/", []));
    }

    private static string? Parse(string xml, string prefix, List<RemoteEntry> into)
        => S3FileSource.ParseListing(xml, prefix, into);
}
