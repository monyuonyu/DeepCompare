using System.Text;

namespace DeepCompare.Engine;

/// <summary>区切り文字と引用符の決まり。</summary>
public sealed record TableFormat(
    char Delimiter = ',',
    char Quote = '"',
    bool HasHeader = true)
{
    public static readonly TableFormat Csv = new();
    public static readonly TableFormat Tsv = new('\t');

    /// <summary>拡張子から推定する。分からなければ CSV とみなす。</summary>
    public static TableFormat ForPath(string path)
        => Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".tsv" or ".tab" => Tsv,
            _ => Csv,
        };
}

/// <summary>表の 1 行。区切って引用符を外した後の値。</summary>
public sealed record TableRow(IReadOnlyList<string> Cells)
{
    public string Cell(int index) => index < Cells.Count ? Cells[index] : string.Empty;
}

public sealed record Table(IReadOnlyList<string> Header, IReadOnlyList<TableRow> Rows)
{
    public int ColumnCount => Math.Max(
        Header.Count,
        Rows.Count == 0 ? 0 : Rows.Max(r => r.Cells.Count));

    /// <summary>見出しから列の位置を引く。無ければ -1。</summary>
    public int IndexOfColumn(string name)
    {
        for (var i = 0; i < Header.Count; i++)
        {
            if (string.Equals(Header[i], name, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }
        return -1;
    }
}

/// <summary>表の 1 行分の比較結果。</summary>
public sealed record TableRowComparison(
    int? Left,
    int? Right,
    /// <summary>違いのある列の位置。左右が揃っているときだけ意味を持つ。</summary>
    IReadOnlyList<int> ChangedColumns)
{
    public bool IsUnchanged => Left is not null && Right is not null && ChangedColumns.Count == 0;
}

public sealed record TableComparison(
    Table Left,
    Table Right,
    IReadOnlyList<TableRowComparison> Rows)
{
    public int Different => Rows.Count(r => r.Left is not null && r.Right is not null && !r.IsUnchanged);
    public int LeftOnly => Rows.Count(r => r.Right is null);
    public int RightOnly => Rows.Count(r => r.Left is null);
}

/// <summary>
/// 表形式の比較。BC の Table Compare に相当する。
///
/// 行を素直に上から突き合わせると、1 行挿入されただけで以降が全部ずれる。**キー列を
/// 指定すれば並び順が違っても照合できる**ので、そちらを主に据える。キー列が無い場合は
/// 行全体を 1 つの文字列として既存の対応付け（<see cref="DiffComparer"/>）に流す。
/// 埋め込みを渡せば、キーが無くても内容の近い行どうしが対応する。
/// </summary>
public static class TableCompare
{
    /// <summary>
    /// 1 つの表として読む。引用符の中では区切り文字と改行をそのまま扱う。
    /// </summary>
    public static Table Parse(string text, TableFormat format)
    {
        var rows = new List<TableRow>();
        var cells = new List<string>();
        var cell = new StringBuilder();
        var inQuotes = false;
        var sawAnything = false;

        void EndCell()
        {
            cells.Add(cell.ToString());
            cell.Clear();
        }

        void EndRow()
        {
            EndCell();
            rows.Add(new TableRow(cells.ToArray()));
            cells.Clear();
        }

        for (var i = 0; i < text.Length; i++)
        {
            var c = text[i];
            sawAnything = true;

            if (inQuotes)
            {
                if (c == format.Quote)
                {
                    // 引用符の中の引用符は 2 つ重ねる決まり。
                    if (i + 1 < text.Length && text[i + 1] == format.Quote)
                    {
                        cell.Append(format.Quote);
                        i++;
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    cell.Append(c);
                }
                continue;
            }

            if (c == format.Quote && cell.Length == 0)
            {
                inQuotes = true;
            }
            else if (c == format.Delimiter)
            {
                EndCell();
            }
            else if (c is '\n')
            {
                EndRow();
            }
            else if (c is '\r')
            {
                // CRLF の CR は捨てる。単独の CR は改行として扱う。
                if (i + 1 < text.Length && text[i + 1] == '\n')
                {
                    continue;
                }
                EndRow();
            }
            else
            {
                cell.Append(c);
            }
        }

        // 末尾に改行が無い場合の最後の行。空の入力では行を作らない。
        if (sawAnything && (cell.Length > 0 || cells.Count > 0))
        {
            EndRow();
        }

        if (format.HasHeader && rows.Count > 0)
        {
            return new Table(rows[0].Cells, rows.Skip(1).ToList());
        }
        return new Table([], rows);
    }

    /// <summary>
    /// 2 つの表を比べる。
    /// </summary>
    /// <param name="keyColumns">
    /// 行を対応付ける列の位置。空なら並び順で対応付ける。
    /// </param>
    /// <param name="ignoredColumns">比較から外す列の位置。</param>
    /// <param name="embedder">
    /// キー列が無いときの対応付けに使う。null なら文字列一致だけ。
    /// </param>
    public static TableComparison Compare(
        Table left,
        Table right,
        IReadOnlyList<int>? keyColumns = null,
        IReadOnlyList<int>? ignoredColumns = null,
        Embedder? embedder = null)
    {
        var ignored = ignoredColumns is null ? [] : new HashSet<int>(ignoredColumns);
        var rows = keyColumns is { Count: > 0 }
            ? PairByKey(left, right, keyColumns, ignored)
            : PairByOrder(left, right, ignored, embedder);
        return new TableComparison(left, right, rows);
    }

    /// <summary>
    /// キー列で照合する。並び順が違っても対応が取れる。同じキーが複数あるときは
    /// 現れた順に組にする。
    /// </summary>
    private static List<TableRowComparison> PairByKey(
        Table left, Table right, IReadOnlyList<int> keyColumns, HashSet<int> ignored)
    {
        static string Key(TableRow row, IReadOnlyList<int> columns)
            => string.Join('', columns.Select(row.Cell));

        var rightByKey = new Dictionary<string, Queue<int>>(StringComparer.Ordinal);
        for (var j = 0; j < right.Rows.Count; j++)
        {
            var key = Key(right.Rows[j], keyColumns);
            if (!rightByKey.TryGetValue(key, out var queue))
            {
                rightByKey[key] = queue = new Queue<int>();
            }
            queue.Enqueue(j);
        }

        var result = new List<TableRowComparison>();
        var usedRight = new bool[right.Rows.Count];

        for (var i = 0; i < left.Rows.Count; i++)
        {
            var key = Key(left.Rows[i], keyColumns);
            if (rightByKey.TryGetValue(key, out var queue) && queue.Count > 0)
            {
                var j = queue.Dequeue();
                usedRight[j] = true;
                result.Add(new TableRowComparison(
                    i, j, ChangedColumns(left.Rows[i], right.Rows[j], ignored)));
            }
            else
            {
                result.Add(new TableRowComparison(i, null, []));
            }
        }

        // 左に相手が居なかった右の行。キーで照合する以上、位置は末尾にまとめる。
        for (var j = 0; j < right.Rows.Count; j++)
        {
            if (!usedRight[j])
            {
                result.Add(new TableRowComparison(null, j, []));
            }
        }
        return result;
    }

    /// <summary>
    /// 並び順で対応付ける。行を 1 つの文字列に潰して既存の対応付けに流すので、
    /// 挿入や削除があってもずれない。無視する列は潰す前に落とす。
    /// </summary>
    private static List<TableRowComparison> PairByOrder(
        Table left, Table right, HashSet<int> ignored, Embedder? embedder)
    {
        static DecodedText AsText(Table table, HashSet<int> ignored)
        {
            var lines = table.Rows
                .Select(row => string.Join(
                    '',
                    row.Cells.Where((_, index) => !ignored.Contains(index))))
                .ToList();
            return new DecodedText(lines, TextEncoding.Utf8, LineEnding.Lf);
        }

        var comparison = DiffComparer.Compare(
            AsText(left, ignored), AsText(right, ignored), embedder);

        var result = new List<TableRowComparison>();
        var pendingLeft = new List<int>();
        var pendingRight = new List<int>();

        // 対応が付かなかった行は、変更の塊の中で位置順に組にする。
        //
        // 表では「1 行消えて 1 行増えた」より「この行のこの列が変わった」の方が
        // 読みやすい。行の比較ではそう畳まない（別物かもしれない）が、表は列が
        // 揃っているぶん、位置で組にする根拠がある。
        void Flush()
        {
            var pairs = Math.Min(pendingLeft.Count, pendingRight.Count);
            for (var k = 0; k < pairs; k++)
            {
                var l = pendingLeft[k];
                var r = pendingRight[k];
                result.Add(new TableRowComparison(
                    l, r, ChangedColumns(left.Rows[l], right.Rows[r], ignored)));
            }
            for (var k = pairs; k < pendingLeft.Count; k++)
            {
                result.Add(new TableRowComparison(pendingLeft[k], null, []));
            }
            for (var k = pairs; k < pendingRight.Count; k++)
            {
                result.Add(new TableRowComparison(null, pendingRight[k], []));
            }
            pendingLeft.Clear();
            pendingRight.Clear();
        }

        foreach (var row in comparison.Rows)
        {
            if (row.Left is { } pairedLeft && row.Right is { } pairedRight)
            {
                Flush();
                result.Add(new TableRowComparison(
                    pairedLeft, pairedRight,
                    ChangedColumns(left.Rows[pairedLeft], right.Rows[pairedRight], ignored)));
                continue;
            }
            if (row.Left is { } onlyLeft)
            {
                pendingLeft.Add(onlyLeft);
            }
            if (row.Right is { } onlyRight)
            {
                pendingRight.Add(onlyRight);
            }
        }
        Flush();

        return result;
    }

    private static List<int> ChangedColumns(TableRow left, TableRow right, HashSet<int> ignored)
    {
        var columns = Math.Max(left.Cells.Count, right.Cells.Count);
        var changed = new List<int>();
        for (var c = 0; c < columns; c++)
        {
            if (ignored.Contains(c))
            {
                continue;
            }
            if (!string.Equals(left.Cell(c), right.Cell(c), StringComparison.Ordinal))
            {
                changed.Add(c);
            }
        }
        return changed;
    }
}
