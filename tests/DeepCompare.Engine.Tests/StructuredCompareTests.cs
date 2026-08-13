using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class StructuredCompareTests
{
    private static IReadOnlyList<StructuralChange> Diff(
        string left, string right, StructuredCompareOptions? options = null)
        => StructuredCompare.CompareJson(left, right, options);

    // --- ここが 7.1 を作った理由そのもの ---

    [Fact]
    public void キーの順序が違うだけなら差分にしない()
    {
        var changes = Diff(
            """{"a": 1, "b": 2, "c": 3}""",
            """{"c": 3, "b": 2, "a": 1}""");

        Assert.Empty(changes);
    }

    [Fact]
    public void 整形が違うだけなら差分にしない()
    {
        var changes = Diff(
            """{"a":1,"b":[1,2]}""",
            """
            {
                "a": 1,
                "b": [
                    1,
                    2
                ]
            }
            """);

        Assert.Empty(changes);
    }

    [Fact]
    public void 大量のキーの並べ替えの中から変わった一個だけを見つける()
    {
        // テキスト差分なら全行が動く。構造比較なら 1 個だけ出る。
        var left = "{" + string.Join(",", Enumerable.Range(0, 50).Select(i => $"\"k{i}\":{i}")) + "}";
        var right = "{" + string.Join(",",
            Enumerable.Range(0, 50).Reverse().Select(i => $"\"k{i}\":{(i == 7 ? 999 : i)}")) + "}";

        var changes = Diff(left, right);

        var change = Assert.Single(changes);
        Assert.Equal(StructuralChangeKind.Changed, change.Kind);
        Assert.Equal("$.k7", change.Path);
        Assert.Equal("7", change.Left!.Value);
        Assert.Equal("999", change.Right!.Value);
    }

    // --- 型の変化 ---

    [Fact]
    public void 文字列と数値の違いを型の変化として出す()
    {
        var changes = Diff("""{"port": "8080"}""", """{"port": 8080}""");

        var change = Assert.Single(changes);
        Assert.Equal(StructuralChangeKind.TypeChanged, change.Kind);
        Assert.Contains("文字列", change.Describe());
        Assert.Contains("数値", change.Describe());
    }

    [Fact]
    public void 真偽値が文字列になった場合も型の変化()
    {
        var changes = Diff("""{"debug": true}""", """{"debug": "true"}""");

        Assert.Equal(StructuralChangeKind.TypeChanged, Assert.Single(changes).Kind);
    }

    [Fact]
    public void 種類も値も違えば普通の変更として出す()
    {
        var changes = Diff("""{"x": 1}""", """{"x": "abc"}""");

        Assert.Equal(StructuralChangeKind.Changed, Assert.Single(changes).Kind);
    }

    [Fact]
    public void nullと不在は違う()
    {
        // {"a": null} は「a は無い」ではない。設定ファイルでは意味が変わる。
        var nulled = Diff("""{"a": 1}""", """{"a": null}""");
        Assert.Equal(StructuralChangeKind.Changed, Assert.Single(nulled).Kind);

        var missing = Diff("""{"a": 1}""", """{}""");
        Assert.Equal(StructuralChangeKind.Removed, Assert.Single(missing).Kind);
    }

    // --- 数値の書式 ---

    [Fact]
    public void 数値は既定では値で比べる()
    {
        Assert.Empty(Diff("""{"a": 1.0, "b": 1e3}""", """{"a": 1, "b": 1000}"""));
    }

    [Fact]
    public void 数値を書式ごと比べる指定もできる()
    {
        var changes = Diff(
            """{"a": 1.0}""", """{"a": 1}""",
            new StructuredCompareOptions { NumbersByValue = false });

        Assert.Single(changes);
    }

    [Fact]
    public void 倍精度で表せない桁数は文字列として比べる()
    {
        // 識別子や金額が入っていることがある。丸めて同じ扱いにすると壊す。
        var changes = Diff(
            """{"id": 123456789012345678901}""",
            """{"id": 123456789012345678902}""");

        Assert.Single(changes);
    }

    // --- 配列 ---

    [Fact]
    public void 配列はキーで対応付けるので並べ替えだけなら中身の差分は出ない()
    {
        var changes = Diff(
            """[{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]""",
            """[{"id": 2, "v": "b"}, {"id": 1, "v": "a"}]""");

        // 中身の変更は無い。位置の変化だけが出る。
        Assert.All(changes, c => Assert.Equal(StructuralChangeKind.Moved, c.Kind));
    }

    [Fact]
    public void 位置の変化を報告しない指定もできる()
    {
        var changes = Diff(
            """[{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]""",
            """[{"id": 2, "v": "b"}, {"id": 1, "v": "a"}]""",
            new StructuredCompareOptions { ReportMoves = false });

        Assert.Empty(changes);
    }

    [Fact]
    public void 並べ替えたうえで一個だけ変えた場合その一個だけを変更として出す()
    {
        var changes = Diff(
            """[{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]""",
            """[{"id": 2, "v": "ちがう"}, {"id": 1, "v": "a"}]""",
            new StructuredCompareOptions { ReportMoves = false });

        var change = Assert.Single(changes);
        Assert.Equal(StructuralChangeKind.Changed, change.Kind);
        Assert.Equal("$[id=2].v", change.Path);
    }

    [Fact]
    public void キーが重複していたら位置で対応付ける()
    {
        // id が重複していると、どれと組にすべきか決められない。
        // 勝手に選ぶより、位置で組にする方が予測できる。
        var changes = Diff(
            """[{"id": 1, "v": "a"}, {"id": 1, "v": "b"}]""",
            """[{"id": 1, "v": "a"}, {"id": 1, "v": "c"}]""");

        var change = Assert.Single(changes);
        Assert.Equal("$[1].v", change.Path);
    }

    [Fact]
    public void 一部の要素にキーが無ければ位置で対応付ける()
    {
        var changes = Diff(
            """[{"id": 1, "v": "a"}, {"v": "b"}]""",
            """[{"id": 1, "v": "a"}, {"v": "c"}]""");

        Assert.Equal("$[1].v", Assert.Single(changes).Path);
    }

    [Fact]
    public void 葉だけの配列は位置で比べる()
    {
        var changes = Diff("""[1, 2, 3]""", """[1, 9, 3]""");

        Assert.Equal("$[1]", Assert.Single(changes).Path);
    }

    [Fact]
    public void 配列の増減を出す()
    {
        var changes = Diff("""[1, 2]""", """[1, 2, 3]""");

        var change = Assert.Single(changes);
        Assert.Equal(StructuralChangeKind.Added, change.Kind);
        Assert.Equal("$[2]", change.Path);
    }

    [Fact]
    public void キーのある配列の増減はキーで示す()
    {
        var changes = Diff(
            """[{"name": "a"}]""",
            """[{"name": "a"}, {"name": "b"}]""");

        var change = Assert.Single(changes);
        Assert.Equal(StructuralChangeKind.Added, change.Kind);
        Assert.Equal("$[name=b]", change.Path);
    }

    [Fact]
    public void 対応付けに使う名前を指定できる()
    {
        var changes = Diff(
            """[{"sku": "x", "n": 1}, {"sku": "y", "n": 2}]""",
            """[{"sku": "y", "n": 2}, {"sku": "x", "n": 1}]""",
            new StructuredCompareOptions { ArrayKeys = ["sku"], ReportMoves = false });

        Assert.Empty(changes);
    }

    // --- パス ---

    [Fact]
    public void 入れ子の位置を構造上のパスで示す()
    {
        var changes = Diff(
            """{"spec": {"containers": [{"image": "nginx:1"}]}}""",
            """{"spec": {"containers": [{"image": "nginx:2"}]}}""");

        Assert.Equal("$.spec.containers[0].image", Assert.Single(changes).Path);
    }

    [Fact]
    public void 素直でない名前は括弧書きにする()
    {
        var changes = Diff("""{"a b": 1}""", """{"a b": 2}""");

        Assert.Equal("""$["a b"]""", Assert.Single(changes).Path);
    }

    [Fact]
    public void 引用符を含む名前を逃がす()
    {
        var changes = Diff("""{"a\"b": 1}""", """{"a\"b": 2}""");

        Assert.Equal("""$["a\"b"]""", Assert.Single(changes).Path);
    }

    // --- 無視する位置 ---

    [Fact]
    public void 指定した位置を無視する()
    {
        var changes = Diff(
            """{"data": 1, "generated_at": "2026-01-01"}""",
            """{"data": 1, "generated_at": "2026-08-13"}""",
            new StructuredCompareOptions { IgnoredPaths = ["$.generated_at"] });

        Assert.Empty(changes);
    }

    [Fact]
    public void 親を無視すれば子も無視される()
    {
        var changes = Diff(
            """{"meta": {"a": 1, "b": 2}, "data": 1}""",
            """{"meta": {"a": 9, "b": 9}, "data": 1}""",
            new StructuredCompareOptions { IgnoredPaths = ["$.meta"] });

        Assert.Empty(changes);
    }

    [Fact]
    public void 無視する位置に星印を使える()
    {
        // ノートブックの実行回数のように、繰り返しの中の一箇所を狙う。
        var changes = Diff(
            """{"cells": [{"execution_count": 1, "src": "a"}, {"execution_count": 2, "src": "b"}]}""",
            """{"cells": [{"execution_count": 9, "src": "a"}, {"execution_count": 8, "src": "b"}]}""",
            new StructuredCompareOptions { IgnoredPaths = ["$.cells[*].execution_count"] });

        Assert.Empty(changes);
    }

    [Fact]
    public void 無視の指定は消したい所だけに効く()
    {
        var changes = Diff(
            """{"cells": [{"execution_count": 1, "src": "a"}]}""",
            """{"cells": [{"execution_count": 9, "src": "ちがう"}]}""",
            new StructuredCompareOptions { IgnoredPaths = ["$.cells[*].execution_count"] });

        Assert.Equal("$.cells[0].src", Assert.Single(changes).Path);
    }

    [Fact]
    public void 追加や削除にも無視の指定が効く()
    {
        var changes = Diff(
            """{"data": 1, "tmp": 2}""",
            """{"data": 1}""",
            new StructuredCompareOptions { IgnoredPaths = ["$.tmp"] });

        Assert.Empty(changes);
    }

    // --- 読み取り ---

    [Fact]
    public void 末尾カンマを許す()
    {
        Assert.Empty(Diff("""{"a": 1,}""", """{"a": 1}"""));
    }

    [Fact]
    public void 注釈を許す()
    {
        Assert.Empty(Diff("""{"a": 1 /* めも */}""", """{"a": 1}"""));
    }

    [Fact]
    public void 読めない入力は場所を添えて知らせる()
    {
        var e = Assert.Throws<StructuredParseException>(() => Diff("""{"a": }""", "{}"));

        Assert.Contains("行目", e.Message);
    }

    // --- 実際の形 ---

    [Fact]
    public void ロックファイル風の入力で上がった依存だけを出す()
    {
        var left = """
            {
              "lockfileVersion": 3,
              "packages": {
                "node_modules/a": {"version": "1.0.0", "resolved": "https://x/a-1.0.0.tgz"},
                "node_modules/b": {"version": "2.0.0", "resolved": "https://x/b-2.0.0.tgz"}
              }
            }
            """;
        var right = """
            {
              "lockfileVersion": 3,
              "packages": {
                "node_modules/b": {"version": "2.0.0", "resolved": "https://x/b-2.0.0.tgz"},
                "node_modules/a": {"version": "1.1.0", "resolved": "https://x/a-1.1.0.tgz"}
              }
            }
            """;

        var changes = Diff(left, right);

        // テキスト差分なら全体が動く。ここでは a の 2 項目だけが出る。
        Assert.Equal(2, changes.Count);
        Assert.All(changes, c => Assert.Contains("node_modules/a", c.Path));
    }

    [Fact]
    public void 要約を出す()
    {
        var changes = Diff(
            """{"a": 1, "b": 2, "gone": 3}""",
            """{"a": 9, "b": "2", "added": 4}""");

        var summary = StructuredCompare.Summarize(changes);

        Assert.Contains("追加 1", summary);
        Assert.Contains("削除 1", summary);
        Assert.Contains("変更 1", summary);
        Assert.Contains("型の変化 1", summary);
    }

    [Fact]
    public void 同じなら同じと言う()
    {
        Assert.Equal(
            "構造としては同じです。" + Environment.NewLine,
            StructuredCompare.Format(Diff("""{"a":1}""", """{"a":1}""")));
    }

    [Fact]
    public void 深い入れ子でも落ちない()
    {
        var deep = new string('[', 100) + "1" + new string(']', 100);

        Assert.Empty(Diff(deep, deep));
    }

    [Fact]
    public void 根の種類が違えば一個の差分として出す()
    {
        var changes = Diff("""{"a": 1}""", """[1]""");

        var change = Assert.Single(changes);
        Assert.Equal("$", change.Path);
        Assert.Equal(StructuralChangeKind.Changed, change.Kind);
    }
}
