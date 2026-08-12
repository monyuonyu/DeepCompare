using System.Globalization;
using System.Text;

namespace DeepCompare.Engine;

/// <summary>
/// 比較結果の書き出し。
///
/// 画面で見るだけでなく、外へ渡せる形にする。特に unified 形式は `patch` や
/// `git apply` がそのまま食える必要があるので、行番号と件数の数え方を厳密に合わせる。
/// ここがずれると「見た目は正しいのに適用できない」ものが出る。
/// </summary>
public static class Report
{
    /// <summary>
    /// unified diff。<paramref name="context"/> は変更の前後に付ける文脈の行数。
    ///
    /// 意味的な対応付けで「1 行が 1 行に変わった」と判定していても、unified 形式に
    /// そういう概念は無いので削除と追加として書く。適用できることを優先する。
    /// </summary>
    public static string UnifiedDiff(
        Comparison comparison,
        DecodedText left,
        DecodedText right,
        string leftLabel = "a",
        string rightLabel = "b",
        int context = 3)
    {
        var blocks = Merge.Blocks(comparison);
        var text = new StringBuilder();
        if (blocks.Count == 0)
        {
            return string.Empty;
        }

        text.Append("--- ").Append(leftLabel).Append('\n');
        text.Append("+++ ").Append(rightLabel).Append('\n');

        var index = 0;
        while (index < blocks.Count)
        {
            // 文脈が重なる塊は 1 つのまとまりにする。分けて出すと、同じ行が
            // 2 回現れて適用できない。
            var last = index;
            while (last + 1 < blocks.Count
                   && blocks[last + 1].LeftStart - (blocks[last].LeftStart + blocks[last].LeftCount)
                      <= context * 2)
            {
                last++;
            }

            var first = blocks[index];
            var final = blocks[last];
            var leftFrom = Math.Max(0, first.LeftStart - context);
            var rightFrom = Math.Max(0, first.RightStart - context);
            var leftTo = Math.Min(left.Lines.Count, final.LeftStart + final.LeftCount + context);
            var rightTo = Math.Min(right.Lines.Count, final.RightStart + final.RightCount + context);

            text.Append("@@ -")
                .Append(Range(leftFrom, leftTo - leftFrom))
                .Append(" +")
                .Append(Range(rightFrom, rightTo - rightFrom))
                .Append(" @@\n");

            var leftAt = leftFrom;
            var rightAt = rightFrom;
            for (var b = index; b <= last; b++)
            {
                var block = blocks[b];
                // 塊の手前の一致部分。左右とも同じだけ進む。
                while (leftAt < block.LeftStart)
                {
                    text.Append(' ').Append(left.Lines[leftAt]).Append('\n');
                    AppendNoNewlineMarker(text, left, leftAt);
                    leftAt++;
                    rightAt++;
                }
                for (var i = 0; i < block.LeftCount; i++)
                {
                    var line = block.LeftStart + i;
                    text.Append('-').Append(left.Lines[line]).Append('\n');
                    AppendNoNewlineMarker(text, left, line);
                }
                for (var j = 0; j < block.RightCount; j++)
                {
                    var line = block.RightStart + j;
                    text.Append('+').Append(right.Lines[line]).Append('\n');
                    AppendNoNewlineMarker(text, right, line);
                }
                leftAt = block.LeftStart + block.LeftCount;
                rightAt = block.RightStart + block.RightCount;
            }

            // 末尾の文脈。
            while (leftAt < leftTo)
            {
                text.Append(' ').Append(left.Lines[leftAt]).Append('\n');
                AppendNoNewlineMarker(text, left, leftAt);
                leftAt++;
                rightAt++;
            }

            index = last + 1;
        }

        return text.ToString();
    }

    /// <summary>
    /// 最終行で、かつ元のファイルが改行で終わっていないなら印を出す。
    /// これが無いと patch は「最終行は改行付き」と解釈し、適用に失敗する。
    /// </summary>
    private static void AppendNoNewlineMarker(StringBuilder text, DecodedText source, int line)
    {
        if (!source.EndsWithNewline && line == source.Lines.Count - 1)
        {
            text.Append("\\ No newline at end of file\n");
        }
    }

    /// <summary>unified の範囲表記。1 行だけのときは件数を省くのが慣例。</summary>
    private static string Range(int startZeroBased, int count)
        => count == 1
            ? (startZeroBased + 1).ToString(CultureInfo.InvariantCulture)
            : $"{(count == 0 ? startZeroBased : startZeroBased + 1)},{count}";

    /// <summary>左右を並べた HTML。色分けは画面と揃える。</summary>
    public static string Html(
        Comparison comparison, DecodedText left, DecodedText right,
        string leftLabel = "左", string rightLabel = "右")
    {
        var text = new StringBuilder();
        text.Append("""
            <!DOCTYPE html>
            <html lang="ja"><head><meta charset="utf-8">
            <title>DeepCompare</title>
            <style>
            body{background:#141414;color:#d6d8de;font-family:system-ui,sans-serif;margin:0;padding:16px}
            table{border-collapse:collapse;width:100%;font-family:Consolas,"DejaVu Sans Mono",monospace;font-size:13px}
            th{text-align:left;padding:6px;border-bottom:1px solid #444;font-family:system-ui,sans-serif}
            td{padding:1px 6px;vertical-align:top;white-space:pre-wrap;word-break:break-all}
            td.n{color:#6e7488;text-align:right;width:1%;user-select:none}
            tr.changed{background:#2e3350}tr.removed{background:#40262a}tr.added{background:#223a2a}
            span.i{color:#ffa500}span.u{color:#6e7488}
            </style></head><body>
            <table><thead><tr><th colspan="2">
            """);
        text.Append(Escape(leftLabel)).Append("</th><th colspan=\"2\">")
            .Append(Escape(rightLabel)).Append("</th></tr></thead><tbody>\n");

        foreach (var row in comparison.Rows)
        {
            var cssClass = (row.Left, row.Right) switch
            {
                (not null, not null) when row.IsUnchanged => string.Empty,
                (not null, not null) => " class=\"changed\"",
                (not null, null) => " class=\"removed\"",
                (null, not null) => " class=\"added\"",
                _ => string.Empty,
            };
            text.Append("<tr").Append(cssClass).Append('>');
            text.Append("<td class=\"n\">")
                .Append(row.Left is { } l ? (l + 1).ToString(CultureInfo.InvariantCulture) : string.Empty)
                .Append("</td><td>")
                .Append(Spans(row.Left is { } li ? left.Lines[li] : null, row.LeftSpans))
                .Append("</td>");
            text.Append("<td class=\"n\">")
                .Append(row.Right is { } r ? (r + 1).ToString(CultureInfo.InvariantCulture) : string.Empty)
                .Append("</td><td>")
                .Append(Spans(row.Right is { } ri ? right.Lines[ri] : null, row.RightSpans))
                .Append("</td>");
            text.Append("</tr>\n");
        }

        text.Append("</tbody></table></body></html>\n");
        return text.ToString();
    }

    /// <summary>フォルダー比較の一覧。表計算で開いて絞り込める形。</summary>
    public static string FolderCsv(FolderComparison comparison)
    {
        var text = new StringBuilder();
        text.Append("状態,種別,パス,左のサイズ,右のサイズ,左の更新日時,右の更新日時,エラー\n");
        foreach (var entry in comparison.Entries)
        {
            var status = entry.Status switch
            {
                EntryStatus.Identical => "一致",
                EntryStatus.Different => "差異",
                EntryStatus.LeftOnly => "左のみ",
                EntryStatus.RightOnly => "右のみ",
                _ => "?",
            };
            text.Append(Csv(status)).Append(',')
                .Append(Csv(entry.IsDirectory ? "ディレクトリ" : "ファイル")).Append(',')
                .Append(Csv(entry.RelativePath)).Append(',')
                .Append(entry.LeftSize?.ToString(CultureInfo.InvariantCulture) ?? string.Empty).Append(',')
                .Append(entry.RightSize?.ToString(CultureInfo.InvariantCulture) ?? string.Empty).Append(',')
                .Append(Csv(Stamp(entry.LeftModified))).Append(',')
                .Append(Csv(Stamp(entry.RightModified))).Append(',')
                .Append(Csv(entry.Error ?? string.Empty)).Append('\n');
        }
        return text.ToString();
    }

    private static string Stamp(DateTime? value)
        => value?.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture) ?? string.Empty;

    /// <summary>
    /// CSV の 1 項目。区切り・引用符・改行を含むものは必ず囲む。囲まないと、
    /// パスに読点が入っているだけで列がずれる。
    /// </summary>
    private static string Csv(string value)
        => value.AsSpan().IndexOfAny(",\"\n\r") >= 0
            ? '"' + value.Replace("\"", "\"\"") + '"'
            : value;

    private static string Spans(string? text, IReadOnlyList<Span> spans)
    {
        if (text is null || text.Length == 0)
        {
            return string.Empty;
        }
        if (spans.Count == 0)
        {
            return Escape(text);
        }

        var builder = new StringBuilder();
        foreach (var span in spans)
        {
            var piece = Escape(text.Substring(span.Start, span.Length));
            builder.Append(span.Kind switch
            {
                SpanKind.Changed => $"<span class=\"i\">{piece}</span>",
                SpanKind.Unimportant => $"<span class=\"u\">{piece}</span>",
                _ => piece,
            });
        }
        return builder.ToString();
    }

    /// <summary>
    /// HTML への埋め込み。旧 Python 実装はここを素通しにしていたため、
    /// `&lt;` を含む行が壊れていた。
    /// </summary>
    private static string Escape(string value)
        => value.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");
}
