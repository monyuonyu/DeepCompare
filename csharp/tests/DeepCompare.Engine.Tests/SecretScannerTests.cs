using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 秘密の混入検出。
///
/// **試験に本物の秘密は書かない。** 形だけを真似た値を使う。
/// 本物を書くと、この試験ファイルごと秘密が漏れる。
/// </summary>
public sealed class SecretScannerTests
{
    private static IReadOnlyList<SecretFinding> Scan(params string[] lines)
        => SecretScanner.Scan(lines);

    // --- 形が決まっているもの（ほぼ確実） ---

    [Theory]
    [InlineData("AKIAIOSFODNN7EXAMPLE", "AWS のアクセスキー")]
    [InlineData("ghp_1234567890abcdefghijklmnopqrstuvwxyz", "GitHub のトークン")]
    [InlineData("AIzaSyD1234567890abcdefghijklmnopqrstuv", "Google の API キー")]
    [InlineData("sk_live_1234567890abcdefghij", "Stripe の鍵")]
    [InlineData("xoxb-123456789012-abcdefghijkl", "Slack のトークン")]
    [InlineData("npm_abcdefghijklmnopqrstuvwxyz0123456789", "npm のトークン")]
    public void 発行元の形が決まっているものを見つける(string value, string kind)
    {
        var finding = Assert.Single(Scan($"const key = \"{value}\";"),
            f => f.Confidence == SecretConfidence.High);

        Assert.Equal(kind, finding.Kind);
    }

    [Fact]
    public void 秘密鍵の始まりを見つける()
    {
        var finding = Assert.Single(Scan("-----BEGIN RSA PRIVATE KEY-----"));

        Assert.Equal("秘密鍵", finding.Kind);
        Assert.Equal(SecretConfidence.High, finding.Confidence);
    }

    [Fact]
    public void URLに埋めた資格情報を見つける()
    {
        Assert.Contains(Scan("https://admin:hunter2@example.com/api"),
            f => f.Kind == "URL に埋めた資格情報");
    }

    [Fact]
    public void 接続文字列のパスワードを見つける()
    {
        Assert.Contains(Scan("Server=db;User Id=app;Password=s3cr3tvalue;"),
            f => f.Kind == "接続文字列のパスワード");
    }

    // --- 名前で拾うもの（たぶん） ---

    [Fact]
    public void 秘密らしい名前に即値が付いていれば拾う()
    {
        var finding = Assert.Single(Scan("api_key = \"a1b2c3d4e5f6g7h8\""));

        Assert.Equal(SecretConfidence.Medium, finding.Confidence);
    }

    [Fact]
    public void 参照は拾わない()
    {
        // **ここが要点。** 環境変数から読む書き方まで拾うと、正しく書いた
        // コードほど警告が増えることになる。
        Assert.Empty(Scan(
            "api_key = os.environ[\"API_KEY\"]",
            "token := os.Getenv(\"TOKEN\")",
            "password: process.env.DB_PASSWORD"));
    }

    [Theory]
    [InlineData("your_api_key_here")]
    [InlineData("YOUR-TOKEN")]
    [InlineData("xxxxxxxxxxxx")]
    [InlineData("changeme")]
    [InlineData("<your-key>")]
    [InlineData("${API_KEY}")]
    [InlineData("example_value")]
    [InlineData("dummy_secret")]
    public void 明らかに本物でない値は拾わない(string value)
    {
        // **ここを外すと、説明書やテストで毎回騒ぐ。** 読まれない警告は
        // 無いのと同じなので、置き換え用の値は必ず除く。
        Assert.Empty(Scan($"api_key = \"{value}\""));
    }

    // --- 乱雑さで拾うもの（かもしれない） ---

    [Fact]
    public void 乱数のような長い文字列を弱い印で拾う()
    {
        var finding = Assert.Single(Scan("value = aB3xK9mP2qR7tV1wY5zC8nF4hJ6L"));

        Assert.Equal(SecretConfidence.Low, finding.Confidence);
    }

    [Fact]
    public void 英字だけの長い文字列は拾わない()
    {
        // 英単語の連なりや長い識別子を毎回拾うと使いものにならない。
        Assert.Empty(Scan("public void VeryLongMethodNameForSomething()"));
    }

    [Fact]
    public void 数字だけの長い並びは拾わない()
    {
        Assert.Empty(Scan("const timestamp = 17234567890123456789012345;"));
    }

    [Fact]
    public void 普通のコードでは何も出ない()
    {
        Assert.Empty(Scan(
            "public sealed record Pair(int? Left, int? Right);",
            "    var result = Compare(left, right, embedder);",
            "// 意味的な類似度で行を対応付ける",
            "if (string.IsNullOrEmpty(text)) { return; }"));
    }

    // --- 重なりの扱い ---

    [Fact]
    public void 同じ場所を二重に数えない()
    {
        // 形でも名前でも当たる書き方。**1 件として出す。**
        var findings = Scan("api_key = \"AKIAIOSFODNN7EXAMPLE\"");

        Assert.Single(findings);
        Assert.Equal(SecretConfidence.High, findings[0].Confidence);
    }

    // --- 伏せ字 ---

    [Fact]
    public void 見つけた値をそのまま出さない()
    {
        // **警告に秘密を書くと、秘密が別の場所へ増えるだけ。**
        const string secret = "ghp_1234567890abcdefghijklmnopqrstuvwxyz";

        var finding = Assert.Single(Scan($"token = \"{secret}\""));

        Assert.DoesNotContain(secret, finding.Masked);
        Assert.DoesNotContain(secret, finding.Describe());
        Assert.Contains("…", finding.Masked);
    }

    [Fact]
    public void 短い値は全部伏せる()
    {
        var finding = Assert.Single(Scan("password=abc123x;"));

        Assert.DoesNotContain("abc123x", finding.Masked);
    }

    // --- 差分の増えた側だけ ---

    [Fact]
    public void 増えた行だけを調べる()
    {
        // **既にあるものを毎回言われても直しようがない。**
        // これから外へ出る分に絞る方が、警告の数が減って読んでもらえる。
        var left = new DecodedText(
            ["const old = \"AKIAIOSFODNN7EXAMPLE\";", "var keep = 1;"],
            TextEncoding.Utf8, LineEnding.Lf);
        var right = new DecodedText(
            ["const old = \"AKIAIOSFODNN7EXAMPLE\";", "var keep = 1;",
             "const added = \"ghp_1234567890abcdefghijklmnopqrstuvwxyz\";"],
            TextEncoding.Utf8, LineEnding.Lf);

        var comparison = DiffComparer.Compare(left, right, embedder: null);
        var findings = SecretScanner.ScanAdded(comparison, right);

        var finding = Assert.Single(findings);
        Assert.Equal("GitHub のトークン", finding.Kind);
        Assert.Equal(3, finding.Line);   // 元のファイルでの行番号
    }

    [Fact]
    public void 増えた行が無ければ何も出ない()
    {
        var text = new DecodedText(
            ["const key = \"AKIAIOSFODNN7EXAMPLE\";"], TextEncoding.Utf8, LineEnding.Lf);

        var comparison = DiffComparer.Compare(text, text, embedder: null);

        Assert.Empty(SecretScanner.ScanAdded(comparison, text));
    }

    // --- 乱雑さの計算 ---

    [Fact]
    public void 同じ文字だけなら乱雑さは零()
    {
        Assert.Equal(0, SecretScanner.Entropy("aaaaaaaa"));
    }

    [Fact]
    public void 種類が多いほど乱雑さが上がる()
    {
        Assert.True(SecretScanner.Entropy("aB3xK9mP2qR7") > SecretScanner.Entropy("aaaabbbbcccc"));
    }

    [Fact]
    public void 要約を出す()
    {
        var text = SecretScanner.Format(Scan(
            "a = \"AKIAIOSFODNN7EXAMPLE\"",
            "b = \"realvalue123456\""));

        Assert.Contains("ほぼ確実: 1 件", text);
    }

    [Fact]
    public void 何も無ければそう言う()
    {
        Assert.Contains("見つかりませんでした", SecretScanner.Format(Scan("var x = 1;")));
    }
}
