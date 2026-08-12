using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 表形式の比較。
///
/// 読み取りは引用符の扱いを間違えると列がずれ、以降の比較がすべて無意味になる。
/// 比較はキー列での照合が要点で、これが効かないと 1 行挿入しただけで全行が
/// 差異として出る。
/// </summary>
public sealed class TableCompareTests
{
    private static Table Csv(string text) => TableCompare.Parse(text, TableFormat.Csv);

    // ---- 読み取り ----

    [Fact]
    public void ReadsHeaderAndRows()
    {
        var table = Csv("id,name\n1,alice\n2,bob\n");

        Assert.Equal(["id", "name"], table.Header);
        Assert.Equal(2, table.Rows.Count);
        Assert.Equal(["1", "alice"], table.Rows[0].Cells);
    }

    [Fact]
    public void WithoutAHeaderEveryLineIsARow()
    {
        var table = TableCompare.Parse("1,a\n2,b\n", TableFormat.Csv with { HasHeader = false });

        Assert.Empty(table.Header);
        Assert.Equal(2, table.Rows.Count);
    }

    /// <summary>引用符の中では区切り文字をそのまま扱うこと。</summary>
    [Fact]
    public void QuotedDelimitersStayInsideTheCell()
    {
        var table = Csv("a,b\n\"x,y\",z\n");

        Assert.Equal(["x,y", "z"], table.Rows[0].Cells);
    }

    /// <summary>引用符の中の引用符は 2 つ重ねる決まり。</summary>
    [Fact]
    public void DoubledQuotesBecomeOne()
    {
        var table = Csv("a\n\"he said \"\"hi\"\"\"\n");

        Assert.Equal(["he said \"hi\""], table.Rows[0].Cells);
    }

    /// <summary>引用符の中の改行はセルの一部。ここを取り違えると行数が狂う。</summary>
    [Fact]
    public void NewlinesInsideQuotesDoNotEndTheRow()
    {
        var table = Csv("a,b\n\"one\ntwo\",z\n");

        Assert.Single(table.Rows);
        Assert.Equal("one\ntwo", table.Rows[0].Cells[0]);
    }

    [Fact]
    public void TabSeparatedValuesUseTheTabDelimiter()
    {
        var table = TableCompare.Parse("id\tname\n1\talice\n", TableFormat.Tsv);

        Assert.Equal(["id", "name"], table.Header);
        Assert.Equal(["1", "alice"], table.Rows[0].Cells);
    }

    [Fact]
    public void CrLfIsHandledLikeLf()
    {
        var table = Csv("a,b\r\n1,2\r\n");

        Assert.Single(table.Rows);
        Assert.Equal(["1", "2"], table.Rows[0].Cells);
    }

    [Fact]
    public void ARowWithoutATrailingNewlineIsStillRead()
    {
        var table = Csv("a,b\n1,2");

        Assert.Single(table.Rows);
    }

    [Fact]
    public void EmptyInputHasNoRows()
    {
        var table = Csv(string.Empty);

        Assert.Empty(table.Header);
        Assert.Empty(table.Rows);
    }

    [Fact]
    public void ColumnsAreFoundByName()
    {
        var table = Csv("id,Name,age\n1,a,2\n");

        Assert.Equal(1, table.IndexOfColumn("name"));
        Assert.Equal(-1, table.IndexOfColumn("missing"));
    }

    // ---- 比較 ----

    [Fact]
    public void IdenticalTablesHaveNoDifferences()
    {
        var table = Csv("id,name\n1,alice\n");
        var result = TableCompare.Compare(table, Csv("id,name\n1,alice\n"));

        Assert.Equal(0, result.Different);
        Assert.All(result.Rows, r => Assert.True(r.IsUnchanged));
    }

    [Fact]
    public void ChangedCellsAreReportedByColumn()
    {
        var result = TableCompare.Compare(
            Csv("id,name,age\n1,alice,30\n"),
            Csv("id,name,age\n1,ALICE,30\n"));

        var row = Assert.Single(result.Rows);
        Assert.Equal([1], row.ChangedColumns);
    }

    [Fact]
    public void IgnoredColumnsAreNotCompared()
    {
        var result = TableCompare.Compare(
            Csv("id,name,updated\n1,alice,2024-01-01\n"),
            Csv("id,name,updated\n1,alice,2026-08-13\n"),
            ignoredColumns: [2]);

        Assert.Equal(0, result.Different);
    }

    /// <summary>
    /// キー列を指定すれば並び順が違っても照合できること。ここが表比較の要点で、
    /// 並び順で突き合わせると全行が差異になる。
    /// </summary>
    [Fact]
    public void KeyColumnsMatchRowsRegardlessOfOrder()
    {
        var result = TableCompare.Compare(
            Csv("id,name\n1,alice\n2,bob\n3,carol\n"),
            Csv("id,name\n3,carol\n1,alice\n2,BOB\n"),
            keyColumns: [0]);

        Assert.Equal(1, result.Different);
        Assert.Equal(0, result.LeftOnly);
        Assert.Equal(0, result.RightOnly);

        var changed = Assert.Single(result.Rows, r => r.Left is not null && !r.IsUnchanged);
        Assert.Equal([1], changed.ChangedColumns);
    }

    [Fact]
    public void RowsMissingOnEitherSideAreReported()
    {
        var result = TableCompare.Compare(
            Csv("id,name\n1,alice\n2,bob\n"),
            Csv("id,name\n1,alice\n3,carol\n"),
            keyColumns: [0]);

        Assert.Equal(1, result.LeftOnly);
        Assert.Equal(1, result.RightOnly);
    }

    /// <summary>同じキーが複数あるときは現れた順に組にする。</summary>
    [Fact]
    public void DuplicateKeysArePairedInOrder()
    {
        var result = TableCompare.Compare(
            Csv("k,v\na,1\na,2\n"),
            Csv("k,v\na,1\na,3\n"),
            keyColumns: [0]);

        Assert.Equal(0, result.LeftOnly);
        Assert.Equal(0, result.RightOnly);
        Assert.Equal(1, result.Different);
    }

    /// <summary>
    /// キー列が無い場合でも、1 行挿入されただけで以降が全部ずれないこと。
    /// 行を素直に上から突き合わせる実装だとここで落ちる。
    /// </summary>
    [Fact]
    public void WithoutKeyColumnsAnInsertedRowDoesNotShiftEverything()
    {
        var result = TableCompare.Compare(
            Csv("id,name\n1,a\n2,b\n3,c\n"),
            Csv("id,name\n1,a\n9,new\n2,b\n3,c\n"));

        Assert.Equal(0, result.Different);
        Assert.Equal(1, result.RightOnly);
        Assert.Equal(0, result.LeftOnly);
    }

    [Fact]
    public void MultipleKeyColumnsAreCombined()
    {
        var result = TableCompare.Compare(
            Csv("a,b,v\nx,1,first\nx,2,second\n"),
            Csv("a,b,v\nx,2,SECOND\nx,1,first\n"),
            keyColumns: [0, 1]);

        Assert.Equal(1, result.Different);
        Assert.Equal(0, result.LeftOnly);
    }
}
