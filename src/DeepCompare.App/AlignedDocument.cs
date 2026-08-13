using System.Text;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// エディタへ流す、行数を揃えた本文。
///
/// **片側にしか無い行の位置には空行を入れる。** そうしないと、同じ内容の
/// 行が上下にずれて並び、目で追えない（Beyond Compare も同じく揃える）。
/// 入れた空行は「詰め物」で、元のファイルには無い。保存するときは取り除き、
/// 編集するときは触らせない。
/// </summary>
public sealed record AlignedDocument(string Text, IReadOnlyList<AlignedLine> Lines)
{
    public static readonly AlignedDocument Empty = new(string.Empty, []);

    /// <summary>
    /// 比較の結果から、左右の揃えた本文を作る。
    ///
    /// **行の順序は比較の結果に従う。** 元のファイルの順序ではない
    /// （移動として扱われた行は、相手に合わせた位置へ来る）。
    /// </summary>
    public static (AlignedDocument Left, AlignedDocument Right) Build(
        IReadOnlyList<Row> rows,
        IReadOnlyList<string> leftLines,
        IReadOnlyList<string> rightLines,
        IReadOnlySet<int>? leftEdited = null,
        IReadOnlySet<int>? rightEdited = null)
    {
        var leftText = new StringBuilder();
        var rightText = new StringBuilder();
        var leftInfo = new List<AlignedLine>(rows.Count);
        var rightInfo = new List<AlignedLine>(rows.Count);

        for (var i = 0; i < rows.Count; i++)
        {
            var row = rows[i];
            var changed = row is { Left: not null, Right: not null } && !row.IsUnchanged;

            AppendSide(
                leftText, leftInfo, row.Left, leftLines, row.LeftSpans,
                changed, onlyHere: row.Left is not null && row.Right is null,
                edited: row.Left is { } l && leftEdited?.Contains(l) == true,
                last: i == rows.Count - 1);

            AppendSide(
                rightText, rightInfo, row.Right, rightLines, row.RightSpans,
                changed, onlyHere: row.Right is not null && row.Left is null,
                edited: row.Right is { } r && rightEdited?.Contains(r) == true,
                last: i == rows.Count - 1);
        }

        return (new AlignedDocument(leftText.ToString(), leftInfo),
                new AlignedDocument(rightText.ToString(), rightInfo));
    }

    private static void AppendSide(
        StringBuilder text,
        List<AlignedLine> info,
        int? source,
        IReadOnlyList<string> lines,
        IReadOnlyList<Span> spans,
        bool changed,
        bool onlyHere,
        bool edited,
        bool last)
    {
        if (source is { } at && at < lines.Count)
        {
            text.Append(lines[at]);
            info.Add(new AlignedLine(at, false, changed, onlyHere, edited, spans));
        }
        else
        {
            // 詰め物。**中身は空。** 何か文字を入れると、選んで写したときに
            // 元のファイルに無いものが混ざる。
            info.Add(new AlignedLine(null, true, false, false, false, []));
        }

        if (!last)
        {
            text.Append('\n');
        }
    }

    /// <summary>
    /// 揃えた本文から、詰め物を除いた元の行の並びへ戻す。
    ///
    /// **保存するときに必ず通す。** 詰め物をそのまま書くと、
    /// 相手にしか無かった場所に空行が増えたファイルができる。
    /// </summary>
    public IReadOnlyList<string> WithoutFillers(string edited)
    {
        var result = new List<string>();
        var lines = edited.Replace("\r\n", "\n").Split('\n');
        for (var i = 0; i < lines.Length; i++)
        {
            // **素性より本文が長いことがある。** 打っている途中は、
            // まだ比べ直していないので行が増えている。増えた分は残す。
            if (i < Lines.Count && Lines[i].IsFiller && lines[i].Length == 0)
            {
                continue;
            }
            result.Add(lines[i]);
        }
        return result;
    }
}
