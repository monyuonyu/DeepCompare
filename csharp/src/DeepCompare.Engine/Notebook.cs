using System.Text;

namespace DeepCompare.Engine;

/// <summary>セルの種類。</summary>
public enum CellKind
{
    Code,
    Markdown,
    Raw,
}

/// <summary>ノートブックのセル 1 つ。</summary>
public sealed record NotebookCell(
    CellKind Kind,
    /// <summary>本文。行の配列を 1 本に繋いだもの。</summary>
    string Source)
{
    /// <summary>実行回数。**比べない。** 実行しただけで動く。</summary>
    public int? ExecutionCount { get; init; }

    /// <summary>出力を人が読める形にしたもの。既定では比べない。</summary>
    public string Outputs { get; init; } = string.Empty;

    /// <summary>出力の中に画像が含まれるか。**中身は持たない**（base64 が巨大）。</summary>
    public bool HasImageOutput { get; init; }

    public string Id { get; init; } = string.Empty;

    public bool HasOutputs => Outputs.Length > 0 || HasImageOutput;
}

public sealed record NotebookDocument(
    IReadOnlyList<NotebookCell> Cells,
    string Language,
    /// <summary>カーネルの名前など。付け替えただけで全体が動くので、別に持つ。</summary>
    string Kernel)
{
    public int CodeCells => Cells.Count(c => c.Kind == CellKind.Code);
}

/// <summary>1 つのセルに起きたこと。</summary>
public enum CellChange
{
    Unchanged,

    /// <summary>本文が変わった。**これが唯一「意味のある変化」。**</summary>
    SourceChanged,

    /// <summary>本文は同じで、出力か実行回数だけが変わった。</summary>
    OutputOnly,

    Added,
    Removed,
}

public sealed record NotebookCellDiff(
    CellChange Change,
    NotebookCell? Left,
    NotebookCell? Right,
    int LeftIndex,
    int RightIndex);

public sealed record NotebookComparison(IReadOnlyList<NotebookCellDiff> Cells)
{
    public int SourceChanged => Cells.Count(c => c.Change == CellChange.SourceChanged);
    public int OutputOnly => Cells.Count(c => c.Change == CellChange.OutputOnly);
    public int Added => Cells.Count(c => c.Change == CellChange.Added);
    public int Removed => Cells.Count(c => c.Change == CellChange.Removed);

    /// <summary>本文に変化があるか。**これが「実質的に変わったか」の答え。**</summary>
    public bool HasSourceChanges => SourceChanged > 0 || Added > 0 || Removed > 0;

    /// <summary>言語やカーネルの変化。付け替えただけで全体が動くので別に出す。</summary>
    public string? MetadataChange { get; init; }
}

public sealed record NotebookCompareOptions
{
    /// <summary>
    /// 出力も比べるか。
    ///
    /// **既定は比べない。** ノートブックの差分がほとんど読めないのは、
    /// 実行しただけで出力と実行回数が全部動くから。まず本文だけを見せて、
    /// 出力を見たいときに開く、という順序にする。
    /// </summary>
    public bool CompareOutputs { get; init; }

    /// <summary>実行回数を比べるか。**既定は比べない**（実行順を変えるだけで動く）。</summary>
    public bool CompareExecutionCount { get; init; }
}

/// <summary>
/// Jupyter ノートブック（<c>.ipynb</c>）を**セル単位で**比べる。
///
/// **行で比べても読めない。** 実体は JSON で、本文・出力・実行回数が
/// 混ざっている。1 文字直して実行し直すと、出力の base64 が数千行動き、
/// 直した 1 行はその中に埋もれる。
///
/// 読み取りは 7.1 の構造比較と同じ <see cref="JsonNode"/> を使う。
/// **新しい解析器を書かない。**
/// </summary>
public static class Notebook
{
    public static bool LooksLikeNotebook(string path)
        => Path.GetExtension(path).Equals(".ipynb", StringComparison.OrdinalIgnoreCase);

    public static NotebookDocument Read(string text) => Read(JsonReader.Parse(text));

    public static NotebookDocument Read(JsonNode root)
    {
        var cells = new List<NotebookCell>();
        var array = root.Member("cells");

        if (array is { Kind: JsonKind.Array })
        {
            foreach (var cell in array.Items)
            {
                cells.Add(ReadCell(cell));
            }
        }

        var metadata = root.Member("metadata");
        var language = metadata?.Member("language_info")?.Member("name")?.Value
            ?? metadata?.Member("kernelspec")?.Member("language")?.Value
            ?? string.Empty;
        var kernel = metadata?.Member("kernelspec")?.Member("name")?.Value ?? string.Empty;

        return new NotebookDocument(cells, language, kernel);
    }

    private static NotebookCell ReadCell(JsonNode cell)
    {
        var kind = cell.Member("cell_type")?.Value switch
        {
            "code" => CellKind.Code,
            "markdown" => CellKind.Markdown,
            _ => CellKind.Raw,
        };

        var source = JoinText(cell.Member("source"));
        var executionCount = cell.Member("execution_count") is { Kind: JsonKind.Number } count
            && int.TryParse(count.Value, out var n) ? n : (int?)null;

        var outputs = new StringBuilder();
        var hasImage = false;

        if (cell.Member("outputs") is { Kind: JsonKind.Array } list)
        {
            foreach (var output in list.Items)
            {
                // 種類ごとに、人が読める部分だけを取る。
                var text = output.Member("text");
                if (text is not null)
                {
                    outputs.Append(JoinText(text));
                }

                if (output.Member("data") is { Kind: JsonKind.Object } data)
                {
                    foreach (var (mime, value) in data.Members)
                    {
                        if (mime.StartsWith("image/", StringComparison.Ordinal))
                        {
                            // **画像の中身は持たない。** base64 が数千行になり、
                            // それを比べても人には何も分からない。
                            hasImage = true;
                            continue;
                        }
                        if (mime is "text/plain" or "text/html" or "text/markdown")
                        {
                            outputs.Append(JoinText(value));
                        }
                    }
                }

                // 例外は名前と本文だけ。追跡（traceback）は色の指示が混ざっていて読めない。
                if (output.Member("ename")?.Value is { Length: > 0 } name)
                {
                    outputs.Append(name).Append(": ").Append(output.Member("evalue")?.Value ?? string.Empty);
                }
            }
        }

        return new NotebookCell(kind, source)
        {
            ExecutionCount = executionCount,
            Outputs = outputs.ToString(),
            HasImageOutput = hasImage,
            Id = cell.Member("id")?.Value ?? string.Empty,
        };
    }

    /// <summary>
    /// <c>source</c> は「行の配列」か「1 本の文字列」のどちらか。
    ///
    /// **両方に対応する。** nbformat 4 系は配列だが、書き出す道具によっては
    /// 文字列 1 本になる。片方だけを見ていると、そちらが丸ごと空になる。
    /// </summary>
    private static string JoinText(JsonNode? node)
    {
        if (node is null)
        {
            return string.Empty;
        }
        if (node.Kind == JsonKind.String)
        {
            return node.Value;
        }
        if (node.Kind != JsonKind.Array)
        {
            return string.Empty;
        }

        var text = new StringBuilder();
        foreach (var item in node.Items)
        {
            if (item.Kind == JsonKind.String)
            {
                text.Append(item.Value);
            }
        }
        return text.ToString();
    }

    /// <summary>
    /// セルを対応付けて比べる。
    ///
    /// 対応付けは**本文の一致**で行う（Myers）。id で対応付ける手もあるが、
    /// id は nbformat 4.5 以降にしか無く、道具によっては書き出すたびに変わる。
    /// </summary>
    public static NotebookComparison Compare(
        NotebookDocument left, NotebookDocument right, NotebookCompareOptions? options = null)
    {
        options ??= new NotebookCompareOptions();

        // 本文と種類で鍵を作る。出力と実行回数は入れない
        // （入れると、実行しただけで全部「別のセル」になる）。
        var leftKeys = left.Cells.Select(Key).ToList();
        var rightKeys = right.Cells.Select(Key).ToList();

        var result = new List<NotebookCellDiff>();
        foreach (var op in Myers.Compute(leftKeys, rightKeys))
        {
            switch (op.Kind)
            {
                case DiffKind.Equal:
                    for (var i = 0; i < op.OldLength; i++)
                    {
                        var l = left.Cells[op.OldStart + i];
                        var r = right.Cells[op.NewStart + i];
                        result.Add(new NotebookCellDiff(
                            Classify(l, r, options), l, r,
                            op.OldStart + i, op.NewStart + i));
                    }
                    break;

                // Replace は両側に長さを持つ。**同じ位置の対を「本文が変わった」に
                // する**。片方ずつ消えた・増えたにすると、1 文字直しただけで
                // 2 行に見える。
                case DiffKind.Replace:
                {
                    var shared = Math.Min(op.OldLength, op.NewLength);
                    for (var i = 0; i < shared; i++)
                    {
                        var l = left.Cells[op.OldStart + i];
                        var r = right.Cells[op.NewStart + i];
                        result.Add(new NotebookCellDiff(
                            l.Kind == r.Kind ? CellChange.SourceChanged : CellChange.Removed,
                            l, r, op.OldStart + i, op.NewStart + i));
                        // 種類が変わったなら、消えた／増えたの 2 行にする。
                        if (l.Kind != r.Kind)
                        {
                            result[^1] = new NotebookCellDiff(
                                CellChange.Removed, l, null, op.OldStart + i, -1);
                            result.Add(new NotebookCellDiff(
                                CellChange.Added, null, r, -1, op.NewStart + i));
                        }
                    }
                    for (var i = shared; i < op.OldLength; i++)
                    {
                        result.Add(new NotebookCellDiff(
                            CellChange.Removed, left.Cells[op.OldStart + i], null,
                            op.OldStart + i, -1));
                    }
                    for (var i = shared; i < op.NewLength; i++)
                    {
                        result.Add(new NotebookCellDiff(
                            CellChange.Added, null, right.Cells[op.NewStart + i],
                            -1, op.NewStart + i));
                    }
                    break;
                }

                case DiffKind.Delete:
                    for (var i = 0; i < op.OldLength; i++)
                    {
                        result.Add(new NotebookCellDiff(
                            CellChange.Removed, left.Cells[op.OldStart + i], null,
                            op.OldStart + i, -1));
                    }
                    break;

                case DiffKind.Insert:
                    for (var i = 0; i < op.NewLength; i++)
                    {
                        result.Add(new NotebookCellDiff(
                            CellChange.Added, null, right.Cells[op.NewStart + i],
                            -1, op.NewStart + i));
                    }
                    break;
            }
        }

        string? metadata = null;
        if (left.Language != right.Language || left.Kernel != right.Kernel)
        {
            metadata = $"言語 {Blank(left.Language)} → {Blank(right.Language)}"
                + $" / カーネル {Blank(left.Kernel)} → {Blank(right.Kernel)}";
        }

        return new NotebookComparison(result) { MetadataChange = metadata };
    }

    private static string Blank(string value) => value.Length == 0 ? "（無し）" : value;

    private static string Key(NotebookCell cell) => $"{cell.Kind}{cell.Source}";

    private static CellChange Classify(
        NotebookCell left, NotebookCell right, NotebookCompareOptions options)
    {
        // ここへ来る時点で本文は一致している（鍵が同じ）。
        var outputsDiffer = options.CompareOutputs
            && (!string.Equals(left.Outputs, right.Outputs, StringComparison.Ordinal)
                || left.HasImageOutput != right.HasImageOutput);
        var countDiffers = options.CompareExecutionCount
            && left.ExecutionCount != right.ExecutionCount;

        return outputsDiffer || countDiffers ? CellChange.OutputOnly : CellChange.Unchanged;
    }

    /// <summary>
    /// 出力と実行回数を落とした形で書き出す。
    ///
    /// **git に入れる前に通す用途。** これを通しておけば、実行しただけで
    /// 差分が出ることがなくなる（nbstripout と同じ考え方）。
    ///
    /// 書き換えには System.Text.Json.Nodes を使う。比較用の <see cref="JsonNode"/> は
    /// **不変で、書き出しも持たない**（比べるためだけの形）。
    /// </summary>
    public static string Strip(string text)
    {
        var root = System.Text.Json.Nodes.JsonNode.Parse(text)
            ?? throw new InvalidDataException("JSON として読めません。");
        Scrub(root);
        return root.ToJsonString(new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true,
            // **非 ASCII を \uXXXX に逃がさない。** 逃がすと日本語のノートブックが
            // 読めない形になり、差分もそこら中に出る。
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        });
    }

    private static void Scrub(System.Text.Json.Nodes.JsonNode node)
    {
        if (node is System.Text.Json.Nodes.JsonObject obj)
        {
            if (obj.ContainsKey("cell_type"))
            {
                if (obj.ContainsKey("execution_count"))
                {
                    obj["execution_count"] = null;
                }
                if (obj.ContainsKey("outputs"))
                {
                    obj["outputs"] = new System.Text.Json.Nodes.JsonArray();
                }
            }
            foreach (var (_, value) in obj.ToList())
            {
                if (value is not null)
                {
                    Scrub(value);
                }
            }
        }
        else if (node is System.Text.Json.Nodes.JsonArray array)
        {
            foreach (var item in array)
            {
                if (item is not null)
                {
                    Scrub(item);
                }
            }
        }
    }

    public static string Format(NotebookComparison comparison, bool showUnchanged = false)
    {
        var text = new StringBuilder();

        if (comparison.MetadataChange is { } metadata)
        {
            text.AppendLine($"! {metadata}");
        }

        foreach (var cell in comparison.Cells)
        {
            if (cell.Change == CellChange.Unchanged && !showUnchanged)
            {
                continue;
            }

            var mark = cell.Change switch
            {
                CellChange.SourceChanged => '~',
                CellChange.OutputOnly => 'o',
                CellChange.Added => '+',
                CellChange.Removed => '-',
                _ => '=',
            };
            var sample = cell.Right ?? cell.Left!;
            var index = cell.RightIndex >= 0 ? cell.RightIndex : cell.LeftIndex;
            var first = FirstLine(sample.Source);

            text.AppendLine($"{mark} [{index}] {sample.Kind}: {first}");
        }

        text.AppendLine("---");
        text.AppendLine(comparison.HasSourceChanges
            ? $"本文が変わった {comparison.SourceChanged}"
              + $" / 増えた {comparison.Added} / 消えた {comparison.Removed}"
              + $"（出力だけ {comparison.OutputOnly}）"
            : comparison.OutputOnly > 0
                ? $"**本文は同じです。** 出力か実行回数だけが違います（{comparison.OutputOnly} セル）。"
                : "同じです。");
        return text.ToString();
    }

    private static string FirstLine(string source)
    {
        var line = source.Split('\n')[0].TrimEnd('\r');
        return line.Length > 70 ? line[..70] + "…" : line;
    }
}
