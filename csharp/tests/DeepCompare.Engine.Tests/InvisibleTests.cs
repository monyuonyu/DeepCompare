using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 見えない差分。
///
/// **試験の中でも文字は必ずエスケープで書く。** 見えない文字をソースに直接
/// 置くと、編集や貼り付けで静かに消え、試験が何も試さなくなっても気づけない。
/// </summary>
public sealed class InvisibleTests
{
    private static DecodedText Text(params string[] lines)
        => new(lines, TextEncoding.Utf8, LineEnding.Lf);

    private static IReadOnlyList<InvisibleFinding> Scan(params string[] lines)
        => InvisibleScanner.Scan(Text(lines));

    // --- 見えない文字 ---

    [Fact]
    public void ゼロ幅空白を見つける()
    {
        var findings = Scan("abc\u200bdef");

        var finding = Assert.Single(findings);
        Assert.Equal(InvisibleKind.ZeroWidth, finding.Kind);
        Assert.Equal(1, finding.Line);
        Assert.Equal(4, finding.Column);
        Assert.Contains("U+200B", finding.Detail);
    }

    [Fact]
    public void 行の途中のBOMを見つける()
    {
        // ファイル先頭の BOM は剥がされるが、途中に紛れ込んだものは残る。
        Assert.Equal(InvisibleKind.ZeroWidth, Assert.Single(Scan("a\ufeffb")).Kind);
    }

    [Fact]
    public void 先頭のU_FEFFはBOMとして扱い行の中の文字として数えない()
    {
        // 参照実装との照合でここを踏んだ。ファイル先頭の U+FEFF は BOM なので
        // TextDecoder が剥がす。行の中の「幅の無い文字」として数えると、
        // 以降の桁が 1 つずれる。
        var text = TextDecoder.Decode(Encoding.UTF8.GetBytes("\ufeffa\u3000b"));

        var findings = InvisibleScanner.Scan(text);

        Assert.Contains(findings, f => f.Kind == InvisibleKind.ByteOrderMark);
        var space = Assert.Single(findings, f => f.Kind == InvisibleKind.IdeographicSpace);
        Assert.Equal(2, space.Column);   // BOM を数えていれば 3 になる
    }

    [Fact]
    public void 書字方向の上書きを見つける()
    {
        // 見た目の順序を変えられる。ファイル名の偽装に使われる手口でもある。
        Assert.Equal(InvisibleKind.ZeroWidth, Assert.Single(Scan("a\u202eb")).Kind);
    }

    [Fact]
    public void 全角空白を見つける()
    {
        var finding = Assert.Single(Scan("if\u3000(x)"));

        Assert.Equal(InvisibleKind.IdeographicSpace, finding.Kind);
        Assert.Equal(3, finding.Column);
    }

    [Fact]
    public void ノーブレークスペースを見つける()
    {
        // 幅も見た目も普通の空白と同じ。目では絶対に気づけない。
        var finding = Assert.Single(Scan("a\u00a0b"));

        Assert.Equal(InvisibleKind.LookalikeSpace, finding.Kind);
        Assert.Contains("U+00A0", finding.Detail);
    }

    [Fact]
    public void 普通の空白は挙げない()
    {
        Assert.Empty(Scan("a b c"));
    }

    [Fact]
    public void 見つけた位置を一文字ずつ正しく示す()
    {
        var findings = Scan("\u200ba\u3000b\u00a0");

        Assert.Equal(4, findings.Count);   // 3 個 ＋ 行末の空白
        Assert.Equal(1, findings[0].Column);
        Assert.Equal(3, findings[1].Column);
        Assert.Equal(5, findings[2].Column);
    }

    // --- 正規化 ---

    [Fact]
    public void NFDの文字列を挙げる()
    {
        // 「が」を「か」＋濁点で書いたもの。表示は同じ。
        var nfd = "か\u3099";

        var finding = Assert.Single(Scan(nfd));

        Assert.Equal(InvisibleKind.NotNormalized, finding.Kind);
    }

    [Fact]
    public void NFCの文字列は挙げない()
    {
        Assert.Empty(Scan("が"));   // 合成済みの「が」
    }

    [Fact]
    public void 見た目が同じでも別物であることを確かめられる()
    {
        // この 2 つは画面上で区別が付かないが、文字列としては違う。
        // 差分ツールが「なぜか一致しない」と言う典型。
        const string composed = "が";
        const string decomposed = "か\u3099";

        Assert.NotEqual(composed, decomposed);
        Assert.Equal(composed, decomposed.Normalize(NormalizationForm.FormC));
    }

    // --- 空白と字下げ ---

    [Fact]
    public void 行末の空白を見つける()
    {
        var finding = Assert.Single(Scan("abc   "));

        Assert.Equal(InvisibleKind.TrailingWhitespace, finding.Kind);
        Assert.Equal(4, finding.Column);
        Assert.Contains("3", finding.Detail);
    }

    [Fact]
    public void 空行を行末の空白として挙げない()
    {
        Assert.Empty(Scan(""));
    }

    [Fact]
    public void 字下げのタブと空白の混在を見つける()
    {
        var findings = Scan("\t    x = 1");

        Assert.Contains(findings, f => f.Kind == InvisibleKind.MixedIndent);
    }

    [Fact]
    public void 字下げがタブだけなら挙げない()
    {
        Assert.DoesNotContain(Scan("\t\tx = 1"), f => f.Kind == InvisibleKind.MixedIndent);
    }

    [Fact]
    public void 本文の途中のタブは字下げの混在と見なさない()
    {
        // 表の整形などで使う。字下げの話ではない。
        Assert.DoesNotContain(Scan("a\tb"), f => f.Kind == InvisibleKind.MixedIndent);
    }

    // --- ファイル全体 ---

    [Fact]
    public void 改行の混在を挙げる()
    {
        var text = new DecodedText(["a", "b"], TextEncoding.Utf8, LineEnding.Mixed);

        Assert.Contains(InvisibleScanner.Scan(text), f => f.Kind == InvisibleKind.MixedLineEnding);
    }

    [Fact]
    public void BOMを挙げる()
    {
        var text = new DecodedText(["a"], TextEncoding.Utf8Bom, LineEnding.Lf);

        Assert.Contains(InvisibleScanner.Scan(text), f => f.Kind == InvisibleKind.ByteOrderMark);
    }

    [Fact]
    public void 最終行に改行が無いことを挙げる()
    {
        var text = new DecodedText(["a"], TextEncoding.Utf8, LineEnding.Lf) { EndsWithNewline = false };

        Assert.Contains(InvisibleScanner.Scan(text), f => f.Kind == InvisibleKind.NoFinalNewline);
    }

    [Fact]
    public void 何も無ければ何も言わない()
    {
        Assert.Empty(Scan("普通の行です", "if (x) {", "}"));
    }

    // --- 見える形に直す ---

    [Fact]
    public void 見えない文字を印に置き換える()
    {
        var revealed = InvisibleScanner.Reveal("a\u200bb\u3000c\td");

        Assert.Contains("<U+200B>", revealed);
        Assert.Contains('␣', revealed);   // 全角空白の印
        Assert.Contains('→', revealed);   // タブの印
    }

    [Fact]
    public void 消さずに置き換える()
    {
        // 消すと「そこに何かあった」ことまで消える。元の問題が別の形で残るだけ。
        var revealed = InvisibleScanner.Reveal("a\u200bb");

        Assert.Contains("a", revealed);
        Assert.Contains("b", revealed);
        Assert.DoesNotContain('\u200b', revealed);
    }

    [Fact]
    public void 普通の行はそのまま返す()
    {
        Assert.Equal("普通の行", InvisibleScanner.Reveal("普通の行"));
    }

    // --- ファイル名の突き合わせ ---

    [Fact]
    public void 既定では名前を厳密に比べる()
    {
        Assert.True(NameMatching.Exact.IsExact);
        Assert.False(NameMatching.Exact.Same("README.md", "readme.md"));
        Assert.False(NameMatching.Exact.Same("が", "か\u3099"));
    }

    [Fact]
    public void 正規化を揃えれば同じ名前とみなせる()
    {
        // macOS が作る日本語のファイル名は NFD、Windows と Linux は NFC。
        var matching = new NameMatching(NormalizeUnicode: true);

        Assert.True(matching.Same("が.txt", "か\u3099.txt"));
        // 大小文字までは無視しない。
        Assert.False(matching.Same("A.txt", "a.txt"));
    }

    [Fact]
    public void 大小文字を無視できる()
    {
        var matching = new NameMatching(IgnoreCase: true);

        Assert.True(matching.Same("README.md", "readme.md"));
    }

    [Fact]
    public void 両方を吸収する設定がある()
    {
        Assert.True(NameMatching.Lenient.Same("がA.txt", "か\u3099a.txt"));
    }

    [Fact]
    public void 惜しい組を理由付きで挙げる()
    {
        var misses = NameMatching.FindNearMisses(
            ["が.txt", "README.md", "同じ.txt"],
            ["か\u3099.txt", "readme.md", "同じ.txt"]);

        Assert.Equal(2, misses.Count);
        Assert.Contains(misses, m => m.Reason.Contains("正規化"));
        Assert.Contains(misses, m => m.Reason.Contains("大小文字"));
        // 完全に一致する名前は挙げない。
        Assert.DoesNotContain(misses, m => m.Left == "同じ.txt");
    }

    [Fact]
    public void 惜しい組が無ければ空を返す()
    {
        Assert.Empty(NameMatching.FindNearMisses(["a.txt"], ["b.txt"]));
    }

    // --- 実際に困る形 ---

    [Fact]
    public void なぜか一致しない二行の理由を示せる()
    {
        // 画面では完全に同じに見える 2 行。
        const string left = "const x = 1;";
        const string right = "const\u00a0x = 1;";

        Assert.NotEqual(left, right);
        Assert.Empty(Scan(left));

        var finding = Assert.Single(Scan(right));
        Assert.Equal(InvisibleKind.LookalikeSpace, finding.Kind);
        Assert.Equal(6, finding.Column);
    }

    [Fact]
    public void 要約を出す()
    {
        var summary = InvisibleScanner.Format(Scan("a\u200bb", "c\u200bd  "));

        Assert.Contains("幅の無い文字: 2 件", summary);
        Assert.Contains("行末の空白: 1 件", summary);
    }

    [Fact]
    public void 何も無ければそう言う()
    {
        Assert.Contains("見つかりませんでした", InvisibleScanner.Format(Scan("普通")));
    }
}
