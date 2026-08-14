using Avalonia.Headless.XUnit;
using Xunit;

namespace DeepCompare.App.Tests;

public class NotebookCompareTests
{
    /// <summary>最小の .ipynb を組み立てる。</summary>
    private static string Notebook(params string[] cells)
    {
        var body = string.Join(",\n", cells);
        return $$"""
            {
              "cells": [{{body}}],
              "metadata": {},
              "nbformat": 4,
              "nbformat_minor": 5
            }
            """;
    }

    private static string Code(string source, string? output = null)
    {
        var outputs = output is null
            ? "[]"
            : $$"""
                [{"output_type": "stream", "name": "stdout", "text": ["{{output}}"]}]
                """;
        return $$"""
            {
              "cell_type": "code",
              "execution_count": 1,
              "metadata": {},
              "source": ["{{source}}"],
              "outputs": {{outputs}}
            }
            """;
    }

    private static string Markdown(string source) => $$"""
        {
          "cell_type": "markdown",
          "metadata": {},
          "source": ["{{source}}"]
        }
        """;

    private static NotebookCompareViewModel Open(TempFile left, TempFile right)
        => new(TestShell.Create()) { LeftPath = left.Path, RightPath = right.Path };

    [AvaloniaFact]
    public async Task 中身が同じなら差分は出ない()
    {
        using var left = new TempFile(Notebook(Code("print(1)")), ".ipynb");
        using var right = new TempFile(Notebook(Code("print(1)")), ".ipynb");

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.Empty(model.Cells);
        // **「同じ」と言い切らない。** 出力と実行回数は見ていない。
        Assert.Contains("違いはありません", model.Message);
    }

    [AvaloniaFact]
    public async Task 変わったセルが出る()
    {
        using var left = new TempFile(Notebook(Code("print(1)")), ".ipynb");
        using var right = new TempFile(Notebook(Code("print(2)")), ".ipynb");

        var model = Open(left, right);
        await model.CompareAsync();

        var cell = Assert.Single(model.Cells);
        Assert.Contains("print(1)", cell.LeftText);
        Assert.Contains("print(2)", cell.RightText);
    }

    /// <summary>
    /// **出力は既定で見ない。** 走らせ直すたびに変わるものを差分に出すと、
    /// 本文の変更が埋もれる。
    /// </summary>
    [AvaloniaFact]
    public async Task 出力の違いは既定で無視され見る設定にすれば出る()
    {
        using var left = new TempFile(Notebook(Code("print(1)", "1")), ".ipynb");
        using var right = new TempFile(Notebook(Code("print(1)", "999")), ".ipynb");

        var model = Open(left, right);
        await model.CompareAsync();
        var ignored = model.Cells.Count;

        model.CompareOutputs = true;
        await model.CompareAsync();
        var seen = model.Cells.Count;

        Assert.Equal(0, ignored);
        Assert.Equal(1, seen);
    }

    [AvaloniaFact]
    public async Task 増えたセルは片側だけとして出る()
    {
        using var left = new TempFile(Notebook(Code("print(1)")), ".ipynb");
        using var right = new TempFile(
            Notebook(Code("print(1)"), Markdown("説明")), ".ipynb");

        var model = Open(left, right);
        await model.CompareAsync();

        var cell = Assert.Single(model.Cells);
        Assert.False(cell.HasLeft);
        Assert.True(cell.HasRight);
    }

    [AvaloniaFact]
    public async Task 同じセルも出す設定にすれば現れる()
    {
        using var left = new TempFile(Notebook(Code("print(1)"), Code("print(2)")), ".ipynb");
        using var right = new TempFile(Notebook(Code("print(1)"), Code("print(9)")), ".ipynb");

        var model = Open(left, right);
        model.ShowUnchanged = true;
        await model.CompareAsync();

        Assert.Equal(2, model.Cells.Count);
    }

    [AvaloniaFact]
    public async Task 壊れたノートブックは理由を出して止まる()
    {
        using var left = new TempFile(Notebook(Code("print(1)")), ".ipynb");
        using var right = new TempFile("{これは JSON ではない", ".ipynb");

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.NotEmpty(model.Message);
        Assert.Empty(model.Cells);
        Assert.False(model.Busy);
    }
}
