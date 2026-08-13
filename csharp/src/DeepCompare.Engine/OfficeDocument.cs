using System.IO.Compression;
using System.Text;
using System.Xml;

namespace DeepCompare.Engine;

/// <summary>Office 文書の種類。</summary>
public enum OfficeKind
{
    Word,
    Excel,
    PowerPoint,
}

/// <summary>文書から取り出した本文の一区切り。</summary>
public sealed record OfficePart(
    /// <summary>どこにあるか。「段落 3」「売上!B2」など、**人が探せる形**。</summary>
    string Where,
    string Text);

public sealed record OfficeContent(OfficeKind Kind, IReadOnlyList<OfficePart> Parts)
{
    /// <summary>比較に渡す行の並び。位置と本文を 1 行にする。</summary>
    public IReadOnlyList<string> ToLines()
        => [.. Parts.Select(p => p.Where.Length > 0
            ? $"{p.Where}\t{Escape(p.Text)}"
            : Escape(p.Text))];

    /// <summary>
    /// 本文の改行とタブを見える形に置き換える。
    ///
    /// **1 つの区切りが 1 行に収まらないと、位置がずれる。** 段落の中の
    /// 改行（Word の Shift+Enter、セルの中の改行）をそのまま出すと、
    /// 「段落 3」の続きが位置の書かれていない行になり、行で比べたときに
    /// 対応が狂う。可逆な形にしておく。
    /// </summary>
    public static string Escape(string text)
        => text.Replace("\\", "\\\\").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
}

/// <summary>
/// Office 文書（<c>.docx</c> / <c>.xlsx</c> / <c>.pptx</c>）から本文を取り出す。
///
/// **実体は zip + XML。** 中身をそのまま比べると、書式や ID の書き換えで
/// 本文と関係のない差分が大量に出る。開いて保存し直しただけでも動く。
/// ここでは**本文だけ**を取り出して、読める形にする。
///
/// 書式は落とす。**そこまで見たいなら Word の変更履歴を使う方が早い。**
/// この道具の役目は「文言がどう変わったか」を出すこと。
/// </summary>
public static class OfficeDocument
{
    public static bool LooksLikeOffice(string path) => KindOf(path) is not null;

    public static OfficeKind? KindOf(string path)
        => Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".docx" or ".docm" => OfficeKind.Word,
            ".xlsx" or ".xlsm" => OfficeKind.Excel,
            ".pptx" or ".pptm" => OfficeKind.PowerPoint,
            _ => null,
        };

    public static OfficeContent Read(string path)
    {
        var kind = KindOf(path)
            ?? throw new NotSupportedException($"Office 文書ではありません: {path}");

        using var stream = File.OpenRead(path);
        return Read(stream, kind);
    }

    public static OfficeContent Read(Stream stream, OfficeKind kind)
    {
        using var zip = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
        return kind switch
        {
            OfficeKind.Word => new OfficeContent(kind, ReadWord(zip)),
            OfficeKind.Excel => new OfficeContent(kind, ReadExcel(zip)),
            _ => new OfficeContent(kind, ReadPowerPoint(zip)),
        };
    }

    // --- Word ---

    private static List<OfficePart> ReadWord(ZipArchive zip)
    {
        var parts = new List<OfficePart>();
        var entry = zip.GetEntry("word/document.xml");
        if (entry is null)
        {
            return parts;
        }

        using var reader = Open(entry);
        var paragraph = new StringBuilder();
        var number = 0;

        // **Read() を無条件に呼ばない。** ReadElementContentAsString は要素を
        // 消費して次のノードまで進めるので、その後にもう一度 Read() を呼ぶと
        // 直後の要素を飛ばす。実際、それで `<w:tab/>` が消えていた。
        reader.Read();
        while (!reader.EOF)
        {
            if (reader.NodeType == XmlNodeType.Element)
            {
                switch (reader.LocalName)
                {
                    case "t":
                        paragraph.Append(reader.ReadElementContentAsString());
                        continue;

                    // **改行とタブは字として残す。** 落とすと単語が繋がる。
                    case "br":
                    case "cr":
                        paragraph.Append('\n');
                        break;
                    case "tab":
                        paragraph.Append('\t');
                        break;

                    // **`<w:p/>` は空の段落。** 自己完結タグなので EndElement が
                    // 来ない。ここで閉じないと段落を 1 つ取りこぼし、以降の
                    // 段落番号が全部ずれる。
                    case "p" when reader.IsEmptyElement:
                        parts.Add(new OfficePart($"段落 {++number}", string.Empty));
                        break;
                }
            }
            else if (reader.NodeType == XmlNodeType.EndElement && reader.LocalName == "p")
            {
                parts.Add(new OfficePart($"段落 {++number}", paragraph.ToString()));
                paragraph.Clear();
            }
            reader.Read();
        }
        return parts;
    }

    // --- Excel ---

    private static List<OfficePart> ReadExcel(ZipArchive zip)
    {
        // 文字列は共有表に集められ、セルには番号だけが入っている。
        // **先に表を読まないと、本文が全部数字に見える。**
        var shared = ReadSharedStrings(zip);
        var sheets = ReadSheetNames(zip);

        var parts = new List<OfficePart>();
        foreach (var (name, path) in sheets)
        {
            if (zip.GetEntry(path) is { } entry)
            {
                ReadSheet(entry, name, shared, parts);
            }
        }
        return parts;
    }

    private static List<string> ReadSharedStrings(ZipArchive zip)
    {
        var result = new List<string>();
        var entry = zip.GetEntry("xl/sharedStrings.xml");
        if (entry is null)
        {
            return result;
        }

        using var reader = Open(entry);
        var text = new StringBuilder();
        var inItem = false;

        reader.Read();
        while (!reader.EOF)
        {
            if (reader.NodeType == XmlNodeType.Element)
            {
                if (reader.LocalName == "si")
                {
                    inItem = true;
                    text.Clear();
                }
                else if (reader.LocalName == "t" && inItem)
                {
                    text.Append(reader.ReadElementContentAsString());
                    continue;
                }
            }
            else if (reader.NodeType == XmlNodeType.EndElement && reader.LocalName == "si")
            {
                result.Add(text.ToString());
                inItem = false;
            }
            reader.Read();
        }
        return result;
    }

    /// <summary>
    /// シートの名前と、その中身がどのファイルかの対応。
    ///
    /// **名前で出したい。** 「sheet2.xml の B3」では、どのシートか分からない。
    /// </summary>
    private static List<(string Name, string Path)> ReadSheetNames(ZipArchive zip)
    {
        const string RelationshipNamespace =
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

        var relations = new Dictionary<string, string>(StringComparer.Ordinal);
        if (zip.GetEntry("xl/_rels/workbook.xml.rels") is { } rels)
        {
            using var reader = Open(rels);
            while (reader.Read())
            {
                if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "Relationship")
                {
                    var id = reader.GetAttribute("Id");
                    var target = reader.GetAttribute("Target");
                    if (id is not null && target is not null)
                    {
                        relations[id] = target.StartsWith('/')
                            ? target.TrimStart('/')
                            : "xl/" + target;
                    }
                }
            }
        }

        var result = new List<(string, string)>();
        if (zip.GetEntry("xl/workbook.xml") is { } workbook)
        {
            using var reader = Open(workbook);
            var index = 0;
            while (reader.Read())
            {
                if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "sheet")
                {
                    index++;
                    var name = reader.GetAttribute("name") ?? $"Sheet{index}";
                    var id = reader.GetAttribute("id", RelationshipNamespace);
                    var path = id is not null && relations.TryGetValue(id, out var target)
                        ? target
                        : $"xl/worksheets/sheet{index}.xml";
                    result.Add((name, path));
                }
            }
        }
        return result;
    }

    private static void ReadSheet(
        ZipArchiveEntry entry, string sheet, List<string> shared, List<OfficePart> parts)
    {
        using var reader = Open(entry);
        string? reference = null;
        string? type = null;

        reader.Read();
        while (!reader.EOF)
        {
            if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "c")
            {
                reference = reader.GetAttribute("r");
                type = reader.GetAttribute("t");
            }
            else if (reader.NodeType == XmlNodeType.Element && reader.LocalName is "v" or "t")
            {
                var raw = reader.ReadElementContentAsString();

                // 型が `s` なら、値は共有表への番号。**そのまま出すと数字に見える。**
                var value = type == "s" && int.TryParse(raw, out var index)
                    && index >= 0 && index < shared.Count
                    ? shared[index]
                    : raw;

                if (value.Length > 0)
                {
                    parts.Add(new OfficePart($"{sheet}!{reference ?? "?"}", value));
                }
                continue;
            }
            reader.Read();
        }
    }

    // --- PowerPoint ---

    private static List<OfficePart> ReadPowerPoint(ZipArchive zip)
    {
        // スライドは番号順に並べる。**辞書順だと 10 が 2 より前に来る。**
        var slides = zip.Entries
            .Where(e => e.FullName.StartsWith("ppt/slides/slide", StringComparison.Ordinal)
                        && e.FullName.EndsWith(".xml", StringComparison.Ordinal))
            .Select(e => (Entry: e, Number: SlideNumber(e.FullName)))
            .OrderBy(x => x.Number)
            .ToList();

        var parts = new List<OfficePart>();
        foreach (var (entry, number) in slides)
        {
            using var reader = Open(entry);
            var shape = new StringBuilder();
            var paragraph = new StringBuilder();
            var shapeNumber = 0;

            // **図形（テキストボックス）ごとに 1 つにまとめる。**
            // PowerPoint では改行が段落（a:p）の区切りになるので、段落ごとに
            // 出すと「改行を含む 1 つの文」が 2 つに割れる。利用者から見た
            // 単位はテキストボックスの中身なので、そこで区切る。
            void FlushParagraph()
            {
                if (paragraph.Length == 0)
                {
                    return;
                }
                if (shape.Length > 0)
                {
                    shape.Append('\n');
                }
                shape.Append(paragraph);
                paragraph.Clear();
            }

            void FlushShape()
            {
                FlushParagraph();
                // 空の図形は落とす。スライドの雛形には空の枠が必ず付いており、
                // 残すと**本文より枠の方が多くなる**。
                if (shape.Length > 0)
                {
                    shapeNumber++;
                    parts.Add(new OfficePart(
                        shapeNumber == 1 ? $"スライド {number}" : $"スライド {number} 枠 {shapeNumber}",
                        shape.ToString()));
                }
                shape.Clear();
            }

            reader.Read();
            while (!reader.EOF)
            {
                if (reader.NodeType == XmlNodeType.Element)
                {
                    if (reader.LocalName == "t")
                    {
                        paragraph.Append(reader.ReadElementContentAsString());
                        continue;
                    }
                    if (reader.LocalName == "br")
                    {
                        paragraph.Append('\n');
                    }
                }
                else if (reader.NodeType == XmlNodeType.EndElement)
                {
                    if (reader.LocalName == "p")
                    {
                        FlushParagraph();
                    }
                    else if (reader.LocalName is "sp" or "graphicFrame" or "pic")
                    {
                        FlushShape();
                    }
                }
                reader.Read();
            }
            FlushShape();   // 閉じ忘れの保険
        }
        return parts;
    }

    private static int SlideNumber(string name)
    {
        var digits = new string([.. Path.GetFileNameWithoutExtension(name).Where(char.IsDigit)]);
        return int.TryParse(digits, out var value) ? value : int.MaxValue;
    }

    private static XmlReader Open(ZipArchiveEntry entry)
        => XmlReader.Create(entry.Open(), new XmlReaderSettings
        {
            // **外部参照を辿らない。** 開いただけで外へ通信する形は作らない。
            DtdProcessing = DtdProcessing.Prohibit,
            XmlResolver = null,
            IgnoreWhitespace = false,
        });

    /// <summary>人が読む形に整える。</summary>
    public static string Format(OfficeContent content)
    {
        var text = new StringBuilder();
        foreach (var line in content.ToLines())
        {
            text.AppendLine(line);
        }
        return text.ToString();
    }
}
