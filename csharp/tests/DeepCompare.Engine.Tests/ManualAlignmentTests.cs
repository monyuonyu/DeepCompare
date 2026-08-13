using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class ManualAlignmentTests
{
    private static DecodedText Text(params string[] lines)
        => TextDecoder.Decode(new UTF8Encoding(false).GetBytes(string.Join('\n', lines) + "\n"));

    private static Comparison Compare(
        DecodedText left, DecodedText right, ManualAlignment? manual = null)
        // 埋め込みは使わない。**手動の指定が効くかを見たいので、自動の
        // 対応付けは決まった形（文字列一致）にしておく。**
        => DiffComparer.Compare(left, right, null, new CompareOptions(Manual: manual));

    /// <summary>行の対応を (左, 右) の並びで書き出す。null は空き。</summary>
    private static List<(int?, int?)> Pairs(Comparison comparison)
        => [.. comparison.Rows.Select(r => (r.Left, r.Right))];

    [Fact]
    public void 指定が無ければ何も変わらない()
    {
        var left = Text("a", "b", "c");
        var right = Text("a", "x", "c");

        var without = Pairs(Compare(left, right));
        var withEmpty = Pairs(Compare(left, right, new ManualAlignment()));

        Assert.Equal(without, withEmpty);
    }

    [Fact]
    public void 対にしないと言えば割れる()
    {
        // **対になっている状態から始める。** 文字列一致だけの対応付けでは
        // 1 行対 1 行の置き換えも対にならないので、ここは Apply を直に試す。
        var left = Text("a", "b", "c");
        var right = Text("a", "x", "c");

        var paired = new List<Row>
        {
            new(0, 0, 1f, [], []),
            new(1, 1, 0.8f, [], []),
            new(2, 2, 1f, [], []),
        };

        var result = ManualAlignment.Apply(
            paired, new ManualAlignment().Unlink(1, 1), left, right);

        Assert.Equal([(0, 0), (1, null), (null, 1), (2, 2)],
            result.Select(r => (r.Left, r.Right)));
    }

    [Fact]
    public void 割った行は片側だけになる()
    {
        var left = Text("あ");
        var right = Text("い");

        var result = ManualAlignment.Apply(
            [new Row(0, 0, 0.7f, [], [])], new ManualAlignment().Unlink(0, 0), left, right);

        Assert.Equal(2, result.Count);
        Assert.Null(result[0].Right);
        Assert.Null(result[1].Left);
        // 片側だけの行は「違う」として塗る。**空白のまま残さない。**
        Assert.Contains(result[0].LeftSpans, s => s.Kind == SpanKind.Changed);
        Assert.Contains(result[1].RightSpans, s => s.Kind == SpanKind.Changed);
    }

    [Fact]
    public void 対にすると言えば繋がる()
    {
        // 自動では a↔a のあと、b と c は別物として並ぶ。
        var left = Text("共通", "まったく違う左の行");
        var right = Text("共通", "こちらも全然ちがう");

        var manual = new ManualAlignment().Link(1, 1);
        var pairs = Pairs(Compare(left, right, manual));

        Assert.Equal([(0, 0), (1, 1)], pairs);
    }

    [Fact]
    public void 離れた行同士も繋げる()
    {
        // 左の 1 行目と、右の 2 行目を対にする。
        var left = Text("共通", "動かした行");
        var right = Text("共通", "増えた行", "動かした行");

        var before = Pairs(Compare(left, right));
        Assert.Equal(3, before.Count);

        var manual = new ManualAlignment().Link(1, 1);
        var pairs = Pairs(Compare(left, right, manual));

        // 左 1 と右 1 が対になり、右 2 が残る。
        Assert.Contains(pairs, p => p == ((int?)1, (int?)1));
        Assert.Contains(pairs, p => p == ((int?)null, (int?)2));
    }

    [Fact]
    public void 順序を崩す指定は諦める()
    {
        // 左 [a, b] 右 [b, a] で、左 0（a）と右 1（a）を対にするのは可能。
        // だが左 1（b）と右 0（b）も同時に対にすると、順序が交差する。
        var left = Text("a", "b");
        var right = Text("b", "a");

        var manual = new ManualAlignment().Link(0, 1).Link(1, 0);
        var pairs = Pairs(Compare(left, right, manual));

        // **できないものはできない。** 片方は諦める。行が消えたり増えたり
        // しないことだけは必ず保つ。
        Assert.Equal(2, pairs.Count(p => p.Item1 is not null));
        Assert.Equal(2, pairs.Count(p => p.Item2 is not null));
        Assert.All(pairs, p => Assert.True(p.Item1 is not null || p.Item2 is not null));
    }

    [Fact]
    public void 繋いでも行は増えも減りもしない()
    {
        var left = Text("あ", "い", "う", "え");
        var right = Text("あ", "ぜんぜん違う", "う", "お");

        foreach (var (l, r) in new[] { (1, 1), (3, 3), (1, 3), (3, 1) })
        {
            var pairs = Pairs(Compare(left, right, new ManualAlignment().Link(l, r)));

            // 左右それぞれの行が、ちょうど 1 回ずつ出る。
            Assert.Equal([0, 1, 2, 3], pairs.Where(p => p.Item1 is not null)
                .Select(p => p.Item1!.Value).Order());
            Assert.Equal([0, 1, 2, 3], pairs.Where(p => p.Item2 is not null)
                .Select(p => p.Item2!.Value).Order());
        }
    }

    [Fact]
    public void 繋いだ行は行内差分を取り直す()
    {
        var left = Text("値 = 100");
        var right = Text("値 = 200");

        var manual = new ManualAlignment().Link(0, 0);
        var row = Compare(left, right, manual).Rows.Single();

        // 違う部分だけが Changed になる。**丸ごと違う扱いにしない。**
        Assert.Contains(row.LeftSpans, s => s.Kind == SpanKind.Equal);
        Assert.Contains(row.LeftSpans, s => s.Kind == SpanKind.Changed);
    }

    [Fact]
    public void 同じ右行を二つの左行に割り当てない()
    {
        // 後から言われた方を採る。**両方に割り当てると行が増える。**
        var manual = new ManualAlignment().Link(0, 5).Link(3, 5);

        Assert.False(manual.Linked.ContainsKey(0));
        Assert.Equal(5, manual.Linked[3]);
    }

    [Fact]
    public void 繋ぐと言った後に割ると言えば割れる()
    {
        var manual = new ManualAlignment().Link(1, 1).Unlink(1, 1);

        Assert.Empty(manual.Linked);
        Assert.Contains((1, 1), manual.Unlinked);
    }

    [Fact]
    public void 割ると言った後に繋ぐと言えば繋がる()
    {
        var manual = new ManualAlignment().Unlink(1, 1).Link(1, 1);

        Assert.Equal(1, manual.Linked[1]);
        Assert.DoesNotContain((1, 1), manual.Unlinked);
    }

    [Fact]
    public void 存在しない行を指定しても落ちない()
    {
        var left = Text("a");
        var right = Text("b");

        var manual = new ManualAlignment().Link(99, 99).Unlink(50, 50);
        var pairs = Pairs(Compare(left, right, manual));

        Assert.Equal(1, pairs.Count(p => p.Item1 is not null));
        Assert.Equal(1, pairs.Count(p => p.Item2 is not null));
    }
}
