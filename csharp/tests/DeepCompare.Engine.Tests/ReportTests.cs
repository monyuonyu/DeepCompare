using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 書き出し。unified 形式は `patch` が食えることが要件なので、
/// 「差分を左へ適用すると右になる」ことを軸に固定する。行番号や件数がずれると、
/// 読めば正しそうなのに patch が拒否する、という形でしか現れない。
/// </summary>
public sealed class ReportTests
{
    private static DecodedText Text(string content)
        => TextDecoder.Decode(System.Text.Encoding.UTF8.GetBytes(content));

    private static (string Diff, DecodedText Left, DecodedText Right) Diff(
        string leftContent, string rightContent, int context = 3)
    {
        var left = Text(leftContent);
        var right = Text(rightContent);
        var comparison = DiffComparer.Compare(left, right, embedder: null);
        return (Report.UnifiedDiff(comparison, left, right, "a", "b", context), left, right);
    }

    /// <summary>
    /// 差分を左へ適用して右になること。`patch` がやることと同じ手順をここで再現する。
    /// 差分はハンクの範囲しか含まないので、範囲の外は左をそのまま運ぶ。
    ///
    /// 見た目の突き合わせではなく適用を試すのは、行番号や件数のずれが
    /// 「読めば正しそうなのに patch が拒否する」形でしか現れないため。
    /// </summary>
    private static void AssertPatchApplies(string leftContent, string rightContent)
    {
        var (diff, left, right) = Diff(leftContent, rightContent);
        Assert.Equal(right.Lines, ApplyUnified(left.Lines, diff));
    }

    private static List<string> ApplyUnified(IReadOnlyList<string> source, string diff)
    {
        var result = new List<string>();
        var at = 0;

        foreach (var chunk in diff.Split('\n'))
        {
            if (chunk.StartsWith("---") || chunk.StartsWith("+++") || chunk.Length == 0
                || chunk.StartsWith('\\'))
            {
                continue;
            }

            if (chunk.StartsWith("@@"))
            {
                // "@@ -12,7 +12,6 @@" の -側の開始位置。件数が 1 のときは省略される。
                var minus = chunk.Split(' ')[1][1..];
                var start = int.Parse(minus.Split(',')[0]);
                var count = minus.Contains(',') ? int.Parse(minus.Split(',')[1]) : 1;
                // 件数 0 のときだけ「直前の行」を指すので、1 を足さない。
                var from = count == 0 ? start : start - 1;
                while (at < from)
                {
                    result.Add(source[at]);
                    at++;
                }
                continue;
            }

            switch (chunk[0])
            {
                case ' ':
                    result.Add(source[at]);
                    at++;
                    break;
                case '-':
                    at++;
                    break;
                case '+':
                    result.Add(chunk[1..]);
                    break;
            }
        }

        while (at < source.Count)
        {
            result.Add(source[at]);
            at++;
        }
        return result;
    }

    [Fact]
    public void IdenticalFilesProduceNoDiff()
    {
        Assert.Equal(string.Empty, Diff("a\nb\n", "a\nb\n").Diff);
    }

    [Theory]
    [InlineData("a\nb\nc\n", "a\nB\nc\n")]
    [InlineData("a\nb\n", "a\n")]
    [InlineData("a\n", "a\nb\n")]
    [InlineData("x\n", "y\n")]
    [InlineData("a\nb\nc\nd\ne\nf\ng\nh\n", "a\nb\nc\nD\ne\nf\ng\nh\n")]
    public void ApplyingTheDiffToTheLeftProducesTheRight(string left, string right)
    {
        AssertPatchApplies(left, right);
    }

    /// <summary>離れた変更が別々のハンクに分かれること。</summary>
    [Fact]
    public void DistantChangesBecomeSeparateHunks()
    {
        var lines = string.Join('\n', Enumerable.Range(0, 40).Select(i => $"line{i}")) + "\n";
        var changed = lines.Replace("line1\n", "LINE1\n").Replace("line38\n", "LINE38\n");

        var (diff, _, _) = Diff(lines, changed);

        Assert.Equal(2, diff.Split('\n').Count(l => l.StartsWith("@@")));
        AssertPatchApplies(lines, changed);
    }

    /// <summary>近い変更は 1 つのハンクにまとまること。分けると同じ行が 2 回出る。</summary>
    [Fact]
    public void NearbyChangesMergeIntoOneHunk()
    {
        var lines = string.Join('\n', Enumerable.Range(0, 20).Select(i => $"line{i}")) + "\n";
        var changed = lines.Replace("line5\n", "LINE5\n").Replace("line7\n", "LINE7\n");

        var (diff, _, _) = Diff(lines, changed);

        Assert.Equal(1, diff.Split('\n').Count(l => l.StartsWith("@@")));
    }

    /// <summary>
    /// 末尾に改行が無い側には印を出すこと。これが無いと patch が
    /// 「最終行は改行付き」と解釈して失敗する。
    /// </summary>
    [Fact]
    public void MissingTrailingNewlineIsMarked()
    {
        var (diff, _, _) = Diff("a\nb", "a\nB");

        Assert.Contains("\\ No newline at end of file", diff);
    }

    /// <summary>
    /// 末尾の改行だけが違う場合も差分として出ること。行の内容は同じなので
    /// 見落としやすいが、バイトは違う。
    /// </summary>
    [Fact]
    public void ATrailingNewlineOnlyChangeIsStillADifference()
    {
        var left = Text("a\nb\n");
        var right = Text("a\nb");
        var comparison = DiffComparer.Compare(left, right, embedder: null);

        Assert.Contains(comparison.Rows, row => !row.IsUnchanged);
        Assert.NotEqual(string.Empty, Report.UnifiedDiff(comparison, left, right));
    }

    [Fact]
    public void HtmlEscapesMarkupInTheContent()
    {
        var left = Text("<script>alert(1)</script>\n");
        var right = Text("<b>&amp;</b>\n");
        var comparison = DiffComparer.Compare(left, right, embedder: null);

        var html = Report.Html(comparison, left, right);

        Assert.DoesNotContain("<script>", html);
        Assert.Contains("&lt;script&gt;", html);
        Assert.Contains("&amp;amp;", html);
    }

    [Fact]
    public void FolderCsvQuotesFieldsContainingSeparators()
    {
        var comparison = new FolderComparison(
            [
                new FolderEntry("a,b.txt", "a,b.txt", 0, false, EntryStatus.Different, 1, 2, null, null),
                new FolderEntry("q\"uote.txt", "q\"uote.txt", 0, false, EntryStatus.LeftOnly, 3, null, null, null),
            ],
            new FolderStats(0, 1, 1, 0, 0, 0));

        var csv = Report.FolderCsv(comparison);

        Assert.Contains("\"a,b.txt\"", csv);
        Assert.Contains("\"q\"\"uote.txt\"", csv);
    }
}
