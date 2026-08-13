using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using AvaloniaEdit;
using AvaloniaEdit.Editing;

namespace DeepCompare.App.Views;

/// <summary>
/// 片側ぶんのエディタ。
///
/// **本文そのものはエディタ部品に任せる。** 行を並べたリストとして
/// 描いていたが、文字単位の選択も、行をまたぐ選択も、自然なキー操作も
/// 作り込めなかった。差分の色は、地を塗る仕掛けと文字を染める仕掛けを
/// 差し込んで実現する（本文の中身には手を入れない）。
/// </summary>
public partial class DiffEditorPane : UserControl
{
    private readonly DiffBackgroundRenderer _background = new();
    private readonly InlineDiffColorizer _colorizer = new();
    private readonly AlignedLineMargin _numbers = new();

    private AlignedDocument _document = AlignedDocument.Empty;
    private bool _filling;

    public DiffEditorPane()
    {
        // **生成される初期化を呼ぶ。** AvaloniaXamlLoader.Load を直に
        // 呼ぶと x:Name のフィールドが埋まらず、直後の組み立てで落ちる。
        InitializeComponent();

        Editor.TextArea.TextView.BackgroundRenderers.Add(_background);
        Editor.TextArea.TextView.LineTransformers.Add(_colorizer);
        Editor.TextArea.LeftMargins.Insert(0, _numbers);

        ArrowColumn.Attach(Editor.TextArea.TextView);
        Arrows.Content = ArrowColumn;

        // いる行を外へ伝える。**下の帯と地図がこれで追う。**
        Editor.TextArea.Caret.PositionChanged += (_, _) =>
            CaretLineChanged?.Invoke(this, Editor.TextArea.Caret.Line - 1);

        // 見えている範囲。地図の枠がこれで動く。
        Editor.TextArea.TextView.ScrollOffsetChanged += (_, _) => RaiseViewport();

        // **打った内容を外へ伝える。** 詰め物を除いた形で渡す。
        Editor.TextChanged += (_, _) =>
        {
            if (!_filling)
            {
                Changed?.Invoke(this, EventArgs.Empty);
            }
        };
    }

    /// <summary>人が打ったときに上がる。読み込みで入れ直したときは上がらない。</summary>
    public event EventHandler? Changed;

    /// <summary>
    /// いる行が変わったときに上がる（揃えた本文での行番号、0 から）。
    /// **下の帯と地図はこれで追う。**
    /// </summary>
    public event EventHandler<int>? CaretLineChanged;

    /// <summary>見えている範囲が変わったときに上がる（先頭の割合, 見えている割合）。</summary>
    public event EventHandler<(double Start, double Size)>? ViewportChanged;

    /// <summary>
    /// 見えている範囲を知らせる。**割合で渡す。**
    /// </summary>
    private void RaiseViewport()
    {
        var view = Editor.TextArea.TextView;
        var total = view.DocumentHeight;
        if (total <= 0)
        {
            return;
        }
        var start = view.VerticalOffset / total;
        var size = Math.Min(1, Bounds.Height / total);
        ViewportChanged?.Invoke(this, (start, size));
    }

    /// <summary>地図から飛ばされたとき、その割合の位置へ動かす。</summary>
    public void ScrollToFraction(double fraction)
    {
        var view = Editor.TextArea.TextView;
        var total = view.DocumentHeight;
        if (total <= 0)
        {
            return;
        }
        Editor.ScrollToVerticalOffset(Math.Clamp(fraction, 0, 1) * total);
    }

    /// <summary>その行へ移す（揃えた本文での行番号、0 から）。</summary>
    public void GoToLine(int index)
    {
        if (index < 0 || Editor.Document is null || index >= Editor.Document.LineCount)
        {
            return;
        }
        Editor.TextArea.Caret.Line = index + 1;
        Editor.TextArea.Caret.Column = 1;
        Editor.ScrollToLine(index + 1);
    }

    /// <summary>いまの中身を、詰め物を除いた行の並びで返す。</summary>
    public IReadOnlyList<string> CurrentLines() => _document.WithoutFillers(Editor.Text);

    /// <summary>
    /// 本文と素性を入れ直す。
    ///
    /// **見ていた場所を保つ。** 入れ直しのたびに先頭へ戻ると、
    /// 反映のたびに現場まで戻ることになる。
    /// </summary>
    /// <summary>写しの矢印。**このペインの左端に置く。**</summary>
    public ApplyArrowColumn ArrowColumn { get; } = new();

    public void Fill(AlignedDocument document, bool readOnly)
    {
        _document = document;
        ArrowColumn.Update(document.Lines);
        _background.Update(document.Lines);
        _colorizer.Update(document.Lines);
        _numbers.Update(document.Lines);

        var caret = Editor.CaretOffset;
        var scroll = Editor.TextArea.TextView.VerticalOffset;

        _filling = true;
        Editor.IsReadOnly = readOnly;
        Editor.Text = document.Text;
        _filling = false;

        Editor.CaretOffset = Math.Min(caret, Editor.Text.Length);

        // **読んでいた場所へ戻す。** ScrollOffset は読むだけの値なので、
        // 代入しても何も起きない（そこに気づかず、入れ直すたびに
        // 左右がばらばらの位置を向いていた）。
        Editor.ScrollToVerticalOffset(scroll);
        Editor.TextArea.TextView.InvalidateVisual();
    }

    /// <summary>色だけ塗り直す。**本文には触らない**（打っている最中に呼ぶ）。</summary>
    public void Repaint(IReadOnlyList<AlignedLine> lines)
    {
        _document = _document with { Lines = lines };
        _background.Update(lines);
        _colorizer.Update(lines);
        _numbers.Update(lines);
        Editor.TextArea.TextView.Redraw();
    }
}
