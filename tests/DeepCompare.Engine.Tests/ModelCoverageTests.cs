using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class ModelCoverageTests
{
    [Fact]
    public void 日本語の割合を数える()
    {
        Assert.Equal(1.0, ModelCoverage.JapaneseRatio(["日本語だけの行"]), 3);
        Assert.Equal(0.0, ModelCoverage.JapaneseRatio(["only english here"]), 3);

        // 半分ずつ。
        Assert.Equal(0.5, ModelCoverage.JapaneseRatio(["abcd", "あいうえ"]), 3);
    }

    [Fact]
    public void 記号は分母に入れない()
    {
        // **コードは括弧と記号が多い。** 入れると日本語だけの行でも
        // 割合が下がり、判定が効かなくなる。
        Assert.Equal(1.0,
            ModelCoverage.JapaneseRatio(["  設定 = { \"値\": [1, 2, 3] };"]), 3);
    }

    [Fact]
    public void 日本語が多ければ知らせる()
    {
        var warning = ModelCoverage.Warn(["設定を読み込む"], ["設定の読み込み"]);

        Assert.NotNull(warning);
        // **何が起きているかと、どうすれば直るかを言う。**
        // 濁点の例まで書いていたが、状態バーに常時居座る文としては長すぎた。
        Assert.Contains("文字の重なり", warning);
        Assert.Contains("多言語モデル", warning);
    }

    [Fact]
    public void コメントが数行あるだけでは知らせない()
    {
        // 低くしすぎると、そのたびに警告が出て読まれなくなる。
        var lines = Enumerable.Repeat("public void Method(int value) { return; }", 20)
            .Append("// 日本語のコメント").ToList();

        Assert.Null(ModelCoverage.Warn(lines, lines));
    }

    [Fact]
    public void 空でも落ちない()
    {
        Assert.Equal(0, ModelCoverage.JapaneseRatio([]));
        Assert.Null(ModelCoverage.Warn([], []));
        Assert.Null(ModelCoverage.Warn(["   "], ["!!!"]));
    }

    [Theory]
    [InlineData('あ', true)]
    [InlineData('ア', true)]
    [InlineData('漢', true)]
    [InlineData('ｱ', true)]
    [InlineData('a', false)]
    [InlineData('1', false)]
    public void 日本語の文字を見分ける(char c, bool expected)
        => Assert.Equal(expected, ModelCoverage.IsJapanese(c));
}
