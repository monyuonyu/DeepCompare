using System.Text.Json;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class NotebookTests
{
    /// <summary>ノートブックを組み立てる。セルは (種類, 本文, 実行回数, 出力)。</summary>
    private static string Build(
        params (string Kind, string Source, int? Count, string? Output)[] cells)
        => Build("python", "python3", cells);

    private static string Build(
        string language, string kernel,
        (string Kind, string Source, int? Count, string? Output)[] cells)
    {
        var items = cells.Select(c =>
        {
            var cell = new Dictionary<string, object?>
            {
                ["cell_type"] = c.Kind,
                ["metadata"] = new Dictionary<string, object?>(),
                // **本文は行の配列。** nbformat 4 系の普通の形。
                ["source"] = c.Source.Split('\n'),
            };
            if (c.Kind == "code")
            {
                cell["execution_count"] = c.Count;
                cell["outputs"] = c.Output is null
                    ? Array.Empty<object>()
                    : new object[]
                    {
                        new Dictionary<string, object?>
                        {
                            ["output_type"] = "stream",
                            ["name"] = "stdout",
                            ["text"] = new[] { c.Output },
                        },
                    };
            }
            return cell;
        }).ToArray();

        var root = new Dictionary<string, object?>
        {
            ["cells"] = items,
            ["metadata"] = new Dictionary<string, object?>
            {
                ["kernelspec"] = new Dictionary<string, object?>
                {
                    ["name"] = kernel,
                    ["language"] = language,
                },
                ["language_info"] = new Dictionary<string, object?> { ["name"] = language },
            },
            ["nbformat"] = 4,
            ["nbformat_minor"] = 5,
        };
        return JsonSerializer.Serialize(root, new JsonSerializerOptions { WriteIndented = true });
    }

    [Fact]
    public void セルを読む()
    {
        var document = Notebook.Read(Build(
            ("markdown", "# 見出し", null, null),
            ("code", "print(1)", 3, "1\n")));

        Assert.Equal(2, document.Cells.Count);
        Assert.Equal(CellKind.Markdown, document.Cells[0].Kind);
        Assert.Equal("# 見出し", document.Cells[0].Source);
        Assert.Equal(CellKind.Code, document.Cells[1].Kind);
        Assert.Equal(3, document.Cells[1].ExecutionCount);
        Assert.Equal("1\n", document.Cells[1].Outputs);
        Assert.Equal("python", document.Language);
        Assert.Equal(1, document.CodeCells);
    }

    [Fact]
    public void 実行しただけなら同じとみなす()
    {
        // **これがこの機能の理由。** 1 度実行し直すだけで、実行回数と出力が
        // 全部動く。行で比べると差分だらけになって、本当の変更が埋もれる。
        var before = Notebook.Read(Build(
            ("code", "print(1)", 1, "1\n"),
            ("code", "print(2)", 2, "2\n")));
        var after = Notebook.Read(Build(
            ("code", "print(1)", 7, "1\n"),
            ("code", "print(2)", 8, "2\n")));

        var result = Notebook.Compare(before, after);

        Assert.False(result.HasSourceChanges);
        Assert.Equal(0, result.OutputOnly);   // 既定では実行回数を見ない
        Assert.All(result.Cells, c => Assert.Equal(CellChange.Unchanged, c.Change));
    }

    [Fact]
    public void 出力だけの違いは頼めば拾う()
    {
        var before = Notebook.Read(Build(("code", "print(x)", 1, "1\n")));
        var after = Notebook.Read(Build(("code", "print(x)", 1, "999\n")));

        Assert.Equal(0, Notebook.Compare(before, after).OutputOnly);

        var withOutputs = Notebook.Compare(before, after,
            new NotebookCompareOptions { CompareOutputs = true });
        Assert.Equal(1, withOutputs.OutputOnly);
        // **出力だけの違いは「本文の変化」に数えない。**
        Assert.False(withOutputs.HasSourceChanges);
    }

    [Fact]
    public void 実行回数だけの違いも頼めば拾う()
    {
        var before = Notebook.Read(Build(("code", "print(x)", 1, "1\n")));
        var after = Notebook.Read(Build(("code", "print(x)", 5, "1\n")));

        Assert.Equal(1, Notebook.Compare(before, after,
            new NotebookCompareOptions { CompareExecutionCount = true }).OutputOnly);
    }

    [Fact]
    public void 本文を直したセルは一行にまとめる()
    {
        // **消えた・増えたの 2 行にしない。** 1 文字直しただけで 2 行に
        // 見えると、何が起きたか読み取りにくい。
        var before = Notebook.Read(Build(
            ("code", "print(1)", 1, null),
            ("code", "print(2)", 2, null)));
        var after = Notebook.Read(Build(
            ("code", "print(1)", 1, null),
            ("code", "print(22)", 2, null)));

        var result = Notebook.Compare(before, after);

        Assert.Equal(1, result.SourceChanged);
        Assert.Equal(0, result.Added);
        Assert.Equal(0, result.Removed);
        Assert.True(result.HasSourceChanges);

        var changed = result.Cells.Single(c => c.Change == CellChange.SourceChanged);
        Assert.Equal("print(2)", changed.Left!.Source);
        Assert.Equal("print(22)", changed.Right!.Source);
    }

    [Fact]
    public void 種類が変わったら消えたと増えたにする()
    {
        // コードが markdown になったのを「変わった」で片付けると、
        // **種類が変わったこと自体が消える。**
        var before = Notebook.Read(Build(("code", "x", 1, null)));
        var after = Notebook.Read(Build(("markdown", "x", null, null)));

        var result = Notebook.Compare(before, after);

        Assert.Equal(1, result.Removed);
        Assert.Equal(1, result.Added);
        Assert.Equal(0, result.SourceChanged);
    }

    [Fact]
    public void セルの増減を見分ける()
    {
        var before = Notebook.Read(Build(
            ("code", "a", 1, null),
            ("code", "c", 2, null)));
        var after = Notebook.Read(Build(
            ("code", "a", 1, null),
            ("code", "b", 2, null),
            ("code", "c", 3, null)));

        var result = Notebook.Compare(before, after);

        Assert.Equal(1, result.Added);
        Assert.Equal(0, result.Removed);
        Assert.Equal(0, result.SourceChanged);
        Assert.Equal("b", result.Cells.Single(c => c.Change == CellChange.Added).Right!.Source);
    }

    [Fact]
    public void カーネルの付け替えを別に出す()
    {
        var before = Notebook.Read(Build("python", "python3", [("code", "x", 1, null)]));
        var after = Notebook.Read(Build("julia", "julia-1.10", [("code", "x", 1, null)]));

        var result = Notebook.Compare(before, after);

        Assert.NotNull(result.MetadataChange);
        Assert.Contains("python", result.MetadataChange);
        Assert.Contains("julia", result.MetadataChange);
        // 本文は変わっていない。**そこを混ぜない。**
        Assert.False(result.HasSourceChanges);
    }

    [Fact]
    public void 本文が文字列一本でも読める()
    {
        // nbformat 4 系は配列だが、書き出す道具によっては文字列 1 本になる。
        // 片方だけを見ていると、そちらが丸ごと空になる。
        var text = """
            {"cells":[{"cell_type":"code","source":"print(1)\n","outputs":[],
              "execution_count":null,"metadata":{}}],"metadata":{},"nbformat":4,"nbformat_minor":5}
            """;

        Assert.Equal("print(1)\n", Notebook.Read(text).Cells[0].Source);
    }

    [Fact]
    public void 画像の中身は持たない()
    {
        // base64 が数千行になる。比べても人には何も分からない。
        var text = """
            {"cells":[{"cell_type":"code","source":["plot()"],"execution_count":1,"metadata":{},
              "outputs":[{"output_type":"display_data","metadata":{},
                "data":{"image/png":"iVBORw0KGgoAAAANSUhEUg...","text/plain":["<Figure>"]}}]}],
             "metadata":{},"nbformat":4,"nbformat_minor":5}
            """;

        var cell = Notebook.Read(text).Cells[0];

        Assert.True(cell.HasImageOutput);
        Assert.Equal("<Figure>", cell.Outputs);
        Assert.DoesNotContain("iVBOR", cell.Outputs);
    }

    [Fact]
    public void 例外は名前と本文だけ拾う()
    {
        // 追跡（traceback）には色の指示が混ざっていて読めない。
        var text = """
            {"cells":[{"cell_type":"code","source":["1/0"],"execution_count":1,"metadata":{},
              "outputs":[{"output_type":"error","ename":"ZeroDivisionError",
                "evalue":"division by zero","traceback":["\u001b[31m---\u001b[0m","..."]}]}],
             "metadata":{},"nbformat":4,"nbformat_minor":5}
            """;

        var cell = Notebook.Read(text).Cells[0];

        Assert.Equal("ZeroDivisionError: division by zero", cell.Outputs);
        // **序数比較を明示する。** xUnit の DoesNotContain は既定で文化依存
        // 比較を使い、ICU の照合では制御文字が無視される。そのため ESC は
        // 「位置 0 で見つかった」ことになり、含まれていないのに失敗した。
        Assert.DoesNotContain(((char)0x1B).ToString(), cell.Outputs, StringComparison.Ordinal);
        Assert.DoesNotContain("---", cell.Outputs, StringComparison.Ordinal);
    }

    [Fact]
    public void 出力と実行回数を落とせる()
    {
        var stripped = Notebook.Strip(Build(("code", "print(1)", 42, "1\n")));

        using var document = JsonDocument.Parse(stripped);
        var cell = document.RootElement.GetProperty("cells")[0];

        Assert.Equal(JsonValueKind.Null, cell.GetProperty("execution_count").ValueKind);
        Assert.Equal(0, cell.GetProperty("outputs").GetArrayLength());
        // 本文は残る。**落とすのは実行の跡だけ。**
        Assert.Equal("print(1)", cell.GetProperty("source")[0].GetString());
    }

    [Fact]
    public void 落とした後は実行の違いが消える()
    {
        // nbstripout と同じ考え方。**通した結果同士は文字列として一致する。**
        var before = Build(("code", "print(1)", 1, "1\n"), ("markdown", "説明", null, null));
        var after = Build(("code", "print(1)", 99, "違う出力\n"), ("markdown", "説明", null, null));

        Assert.NotEqual(before, after);
        Assert.Equal(Notebook.Strip(before), Notebook.Strip(after));
    }

    [Fact]
    public void 日本語を逃がさずに書き出す()
    {
        // \uXXXX に逃がすと、日本語のノートブックが読めない形になり、
        // 差分もそこら中に出る。
        var stripped = Notebook.Strip(Build(("markdown", "# 日本語の見出し", null, null)));

        Assert.Contains("日本語の見出し", stripped);
        Assert.DoesNotContain("\\u65e5", stripped);
    }

    [Fact]
    public void 拡張子で見分ける()
    {
        Assert.True(Notebook.LooksLikeNotebook("a.ipynb"));
        Assert.True(Notebook.LooksLikeNotebook("/tmp/解析.IPYNB"));
        Assert.False(Notebook.LooksLikeNotebook("a.json"));
    }
}
