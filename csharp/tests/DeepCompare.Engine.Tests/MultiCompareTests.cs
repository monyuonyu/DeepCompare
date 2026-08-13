using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 3 つ以上を並べた比較。
///
/// **知りたいのは「どれが仲間外れか」。** 2 つずつ比べても、その関係は見えない。
/// </summary>
public sealed class MultiCompareTests
{
    private static MultiComparison Compare(params string[] yaml)
        => MultiCompare.Compare(
            [.. yaml.Select((_, i) => $"env{i}")],
            [.. yaml.Select(YamlReader.Parse)]);

    [Fact]
    public void 全部同じなら差分は無い()
    {
        var result = Compare("a: 1\nb: 2", "b: 2\na: 1", "a: 1\nb: 2");

        Assert.Equal(0, result.Differences);
        Assert.All(result.Rows, r => Assert.Equal(MultiStatus.Same, r.Status));
    }

    [Fact]
    public void 一つだけ違うものを名指しする()
    {
        // **設定の間違いはたいていこの形で現れる。**
        var result = Compare("port: 8080", "port: 8080", "port: 9090");

        var row = Assert.Single(result.Rows);
        Assert.Equal(MultiStatus.OneDiffers, row.Status);
        Assert.Equal(2, row.Odd);   // 3 つ目が仲間外れ
    }

    [Fact]
    public void 全部違えばそう言う()
    {
        // 接続先のように、環境ごとに違って当然のもの。
        var result = Compare("host: a", "host: b", "host: c");

        Assert.Equal(MultiStatus.AllDiffer, Assert.Single(result.Rows).Status);
    }

    [Fact]
    public void 一部に無い項目を見つける()
    {
        // **これが一番危ない。** 本番にだけ設定が無い、という形で事故になる。
        var result = Compare("a: 1\nb: 2", "a: 1\nb: 2", "a: 1");

        var row = Assert.Single(result.Rows, r => r.Path == "$.b");
        Assert.Equal(MultiStatus.Missing, row.Status);
        Assert.Null(row.Values[2]);
    }

    [Fact]
    public void どれか一つにしか無い位置も拾う()
    {
        // 片方を基準にすると、基準に無い項目が最初から見えなくなる。
        var result = Compare("a: 1", "a: 1", "a: 1\nextra: 9");

        Assert.Contains(result.Rows, r => r.Path == "$.extra");
    }

    [Fact]
    public void 二つでも使える()
    {
        // 2 つのときは「1 つだけ違う」に意味が無いので、全部違う扱い。
        var result = Compare("a: 1", "a: 2");

        Assert.Equal(MultiStatus.AllDiffer, Assert.Single(result.Rows).Status);
    }

    [Fact]
    public void 入れ子の位置を辿れる()
    {
        var result = Compare(
            "server:\n  tls:\n    enabled: true",
            "server:\n  tls:\n    enabled: true",
            "server:\n  tls:\n    enabled: false");

        var row = Assert.Single(result.Rows);
        Assert.Equal("$.server.tls.enabled", row.Path);
        Assert.Equal(MultiStatus.OneDiffers, row.Status);
    }

    [Fact]
    public void 並びの中も位置で揃える()
    {
        var result = Compare(
            "ports:\n  - 80\n  - 443",
            "ports:\n  - 80\n  - 443",
            "ports:\n  - 80\n  - 8443");

        var row = Assert.Single(result.Rows, r => r.Path == "$.ports[1]");
        Assert.Equal(MultiStatus.OneDiffers, row.Status);
    }

    [Fact]
    public void 数は値で比べる()
    {
        Assert.Equal(0, Compare("n: 1", "n: 1.0", "n: 1.00").Differences);
    }

    [Fact]
    public void 仲間外れに印を付けて出す()
    {
        var text = MultiCompare.Format(Compare("port: 80", "port: 80", "port: 99"));

        // **そこが見たいものなので、目に入る形にする。**
        Assert.Contains("[99]", text);
        Assert.Contains("1 つだけ違う 1", text);
    }

    [Fact]
    public void 同じものは既定で出さない()
    {
        var text = MultiCompare.Format(Compare("a: 1\nb: 2", "a: 1\nb: 3", "a: 1\nb: 4"));

        Assert.DoesNotContain("$.a", text);
        Assert.Contains("$.b", text);
    }

    [Fact]
    public void 全部同じならそう言う()
    {
        Assert.Contains("全部同じです", MultiCompare.Format(Compare("a: 1", "a: 1", "a: 1")));
    }

    [Fact]
    public void 一つでは比べられない()
    {
        Assert.Throws<ArgumentException>(() =>
            MultiCompare.Compare(["only"], [YamlReader.Parse("a: 1")]));
    }

    [Fact]
    public void 実際の設定の形で使える()
    {
        var dev = YamlReader.Parse("""
            database:
              host: localhost
              pool: 5
            features:
              newUi: true
              beta: true
            logLevel: debug
            """);
        var staging = YamlReader.Parse("""
            database:
              host: staging.db
              pool: 5
            features:
              newUi: true
              beta: true
            logLevel: info
            """);
        var prod = YamlReader.Parse("""
            database:
              host: prod.db
              pool: 20
            features:
              newUi: true
            logLevel: warn
            """);

        var result = MultiCompare.Compare(["dev", "staging", "prod"], [dev, staging, prod]);

        // 本番にだけ beta が無い。**これが一番拾いたい形。**
        var missing = Assert.Single(result.Rows, r => r.Status == MultiStatus.Missing);
        Assert.Equal("$.features.beta", missing.Path);

        // 本番だけ pool が違う。
        var odd = Assert.Single(result.Rows,
            r => r.Status == MultiStatus.OneDiffers && r.Path == "$.database.pool");
        Assert.Equal(2, odd.Odd);
    }
}
