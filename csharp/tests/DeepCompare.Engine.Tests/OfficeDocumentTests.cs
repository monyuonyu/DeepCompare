using System.IO.Compression;
using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// Office 文書から本文を取り出す試験。
///
/// **実物のファイルはリポジトリに置かない。** 中身を目で確かめられないうえ、
/// 数十 KB のバイナリが増える。ここでは zip + XML を組み立てる。
/// 実物との一致は別に取ってある（python-docx / openpyxl / python-pptx と
/// 40 文書で照合、食い違い 0 件）。
/// </summary>
public sealed class OfficeDocumentTests
{
    private static byte[] Zip(params (string Name, string Content)[] entries)
    {
        using var memory = new MemoryStream();
        using (var zip = new ZipArchive(memory, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var (name, content) in entries)
            {
                using var writer = new StreamWriter(zip.CreateEntry(name).Open(), new UTF8Encoding(false));
                writer.Write(content);
            }
        }
        memory.Position = 0;
        return memory.ToArray();
    }

    private static OfficeContent Read(byte[] zip, OfficeKind kind)
    {
        using var stream = new MemoryStream(zip);
        return OfficeDocument.Read(stream, kind);
    }

    private const string WordNamespace =
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"";
    private const string SheetNamespace =
        "xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"";
    private const string SlideNamespace =
        "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
        + "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"";

    // --- Word ---

    [Fact]
    public void 段落を順に取り出す()
    {
        var content = Read(Zip(("word/document.xml", $"""
            <w:document {WordNamespace}><w:body>
              <w:p><w:r><w:t>最初の段落</w:t></w:r></w:p>
              <w:p><w:r><w:t>二つ目</w:t></w:r></w:p>
            </w:body></w:document>
            """)), OfficeKind.Word);

        Assert.Equal(2, content.Parts.Count);
        Assert.Equal("段落 1", content.Parts[0].Where);
        Assert.Equal("最初の段落", content.Parts[0].Text);
        Assert.Equal("二つ目", content.Parts[1].Text);
    }

    [Fact]
    public void 空の段落も残す()
    {
        // **`<w:p/>` は自己完結タグで EndElement が来ない。** ここで閉じないと
        // 段落を 1 つ取りこぼし、以降の段落番号が全部ずれる。
        var content = Read(Zip(("word/document.xml", $"""
            <w:document {WordNamespace}><w:body>
              <w:p><w:r><w:t>あ</w:t></w:r></w:p>
              <w:p/>
              <w:p><w:r><w:t>い</w:t></w:r></w:p>
            </w:body></w:document>
            """)), OfficeKind.Word);

        Assert.Equal(3, content.Parts.Count);
        Assert.Equal(string.Empty, content.Parts[1].Text);
        Assert.Equal("段落 3", content.Parts[2].Where);
        Assert.Equal("い", content.Parts[2].Text);
    }

    [Fact]
    public void タブと改行を字として残す()
    {
        // **`<w:tab/>` は ReadElementContentAsString の後に Read() を呼ぶと
        // 飛ばされる。** 実物で実際に消えていた。
        var content = Read(Zip(("word/document.xml", $"""
            <w:document {WordNamespace}><w:body>
              <w:p><w:r><w:t>前</w:t><w:tab/><w:t>後</w:t><w:br/><w:t>次の行</w:t></w:r></w:p>
            </w:body></w:document>
            """)), OfficeKind.Word);

        Assert.Equal("前\t後\n次の行", Assert.Single(content.Parts).Text);
    }

    // --- Excel ---

    private static byte[] Workbook(string sharedStrings, string sheet)
        => Zip(
            ("xl/workbook.xml",
             $"""
              <workbook {SheetNamespace}
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                <sheets><sheet name="売上" sheetId="1" r:id="rId1"/></sheets>
              </workbook>
              """),
            ("xl/_rels/workbook.xml.rels",
             """
             <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
               <Relationship Id="rId1" Target="worksheets/sheet1.xml"
                 Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
             </Relationships>
             """),
            ("xl/sharedStrings.xml", sharedStrings),
            ("xl/worksheets/sheet1.xml", sheet));

    [Fact]
    public void 共有文字列を解いてセルの中身を出す()
    {
        // **先に共有表を読まないと、本文が全部数字に見える。**
        var content = Read(Workbook(
            $"""
             <sst {SheetNamespace} count="2" uniqueCount="2">
               <si><t>見出し</t></si><si><t>日本語の値</t></si>
             </sst>
             """,
            $"""
             <worksheet {SheetNamespace}><sheetData>
               <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
               <row r="2"><c r="A2"><v>123</v></c></row>
             </sheetData></worksheet>
             """), OfficeKind.Excel);

        Assert.Equal(3, content.Parts.Count);
        // **シート名で位置を出す。** 「sheet1.xml の A1」では、どのシートか分からない。
        Assert.Equal("売上!A1", content.Parts[0].Where);
        Assert.Equal("見出し", content.Parts[0].Text);
        Assert.Equal("日本語の値", content.Parts[1].Text);
        Assert.Equal("123", content.Parts[2].Text);
    }

    [Fact]
    public void 共有表に無い番号でも落ちない()
    {
        var content = Read(Workbook(
            $"""<sst {SheetNamespace}><si><t>あ</t></si></sst>""",
            $"""
             <worksheet {SheetNamespace}><sheetData>
               <row r="1"><c r="A1" t="s"><v>99</v></c></row>
             </sheetData></worksheet>
             """), OfficeKind.Excel);

        // 番号のまま出す。**例外にして丸ごと読めなくするより良い。**
        Assert.Equal("99", Assert.Single(content.Parts).Text);
    }

    // --- PowerPoint ---

    private static string Slide(params string[] paragraphs)
    {
        var body = string.Join("", paragraphs.Select(p => $"<a:p><a:r><a:t>{p}</a:t></a:r></a:p>"));
        return $"""
            <p:sld {SlideNamespace}><p:cSld><p:spTree>
              <p:sp><p:txBody>{body}</p:txBody></p:sp>
            </p:spTree></p:cSld></p:sld>
            """;
    }

    [Fact]
    public void スライドを番号順に並べる()
    {
        // **辞書順だと 10 が 2 より前に来る。**
        var content = Read(Zip(
            ("ppt/slides/slide1.xml", Slide("一枚目")),
            ("ppt/slides/slide10.xml", Slide("十枚目")),
            ("ppt/slides/slide2.xml", Slide("二枚目"))), OfficeKind.PowerPoint);

        Assert.Equal(["スライド 1", "スライド 2", "スライド 10"],
            content.Parts.Select(p => p.Where));
        Assert.Equal("十枚目", content.Parts[2].Text);
    }

    [Fact]
    public void 図形の中の段落はまとめる()
    {
        // PowerPoint では改行が段落の区切りになる。段落ごとに出すと
        // 「改行を含む 1 つの文」が 2 つに割れる。**利用者から見た単位は
        // テキストボックスの中身。**
        var content = Read(Zip(
            ("ppt/slides/slide1.xml", Slide("一行目", "二行目"))), OfficeKind.PowerPoint);

        Assert.Equal("一行目\n二行目", Assert.Single(content.Parts).Text);
    }

    [Fact]
    public void 空の枠は落とす()
    {
        // スライドの雛形には空の枠が必ず付いており、残すと本文より枠の方が多くなる。
        var content = Read(Zip(("ppt/slides/slide1.xml", $"""
            <p:sld {SlideNamespace}><p:cSld><p:spTree>
              <p:sp><p:txBody><a:p><a:r><a:t>本文</a:t></a:r></a:p></p:txBody></p:sp>
              <p:sp><p:txBody><a:p/></p:txBody></p:sp>
            </p:spTree></p:cSld></p:sld>
            """)), OfficeKind.PowerPoint);

        Assert.Equal("本文", Assert.Single(content.Parts).Text);
    }

    [Fact]
    public void 同じスライドに枠が複数あれば番号を振る()
    {
        var content = Read(Zip(("ppt/slides/slide3.xml", $"""
            <p:sld {SlideNamespace}><p:cSld><p:spTree>
              <p:sp><p:txBody><a:p><a:r><a:t>見出し</a:t></a:r></a:p></p:txBody></p:sp>
              <p:sp><p:txBody><a:p><a:r><a:t>本文</a:t></a:r></a:p></p:txBody></p:sp>
            </p:spTree></p:cSld></p:sld>
            """)), OfficeKind.PowerPoint);

        Assert.Equal(["スライド 3", "スライド 3 枠 2"], content.Parts.Select(p => p.Where));
    }

    // --- 共通 ---

    [Fact]
    public void 本文の改行は行に収まる形に逃がす()
    {
        // **1 つの区切りが 1 行に収まらないと、位置がずれる。**
        // 「段落 3」の続きが位置の書かれていない行になり、対応が狂う。
        var content = Read(Zip(("word/document.xml", $"""
            <w:document {WordNamespace}><w:body>
              <w:p><w:r><w:t>前</w:t><w:br/><w:t>後</w:t></w:r></w:p>
            </w:body></w:document>
            """)), OfficeKind.Word);

        var line = Assert.Single(content.ToLines());
        Assert.Equal("段落 1\t前\\n後", line);
        Assert.DoesNotContain('\n', line);
    }

    [Fact]
    public void 逃がし方は元に戻せる()
    {
        Assert.Equal("a\\\\b\\nc\\td", OfficeContent.Escape("a\\b\nc\td"));
    }

    [Fact]
    public void 拡張子で見分ける()
    {
        Assert.Equal(OfficeKind.Word, OfficeDocument.KindOf("a.docx"));
        Assert.Equal(OfficeKind.Excel, OfficeDocument.KindOf("/tmp/売上.XLSX"));
        Assert.Equal(OfficeKind.PowerPoint, OfficeDocument.KindOf("a.pptm"));
        Assert.Null(OfficeDocument.KindOf("a.zip"));
        Assert.False(OfficeDocument.LooksLikeOffice("a.txt"));
    }

    [Fact]
    public void 中身が無くても落ちない()
    {
        // 壊れた文書でも、読めないことと落ちることは別。
        Assert.Empty(Read(Zip(("なにか.xml", "<a/>")), OfficeKind.Word).Parts);
        Assert.Empty(Read(Zip(("なにか.xml", "<a/>")), OfficeKind.Excel).Parts);
        Assert.Empty(Read(Zip(("なにか.xml", "<a/>")), OfficeKind.PowerPoint).Parts);
    }
}
