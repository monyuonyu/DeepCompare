using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using AvaloniaEdit;
using AvaloniaEdit.Editing;
using AvaloniaEdit.Folding;
using DeepCompare.Engine;

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
    private FoldingManager? _folding;

    /// <summary>
    /// 一致行を既定で畳むか。
    /// **「違う行だけ出す」に繋ぐ。** 別々に持つと、押したのに畳まれない。
    /// </summary>
    private bool _foldByDefault;

    /// <summary>
    /// 長い行を折り返すか。
    ///
    /// **折り返すと行の高さが変わる。** 左右で折り返し方が違うと縦が
    /// ずれるので、左右そろえて切り替える（呼ぶ側の仕事）。
    /// </summary>
    public bool WordWrap
    {
        get => Editor.WordWrap;
        set
        {
            Editor.WordWrap = value;
            // 折り返しの有無で行の位置が全部変わる。脇の列も描き直す。
            ArrowColumn.InvalidateVisual();
            ScoreColumn.InvalidateVisual();
        }
    }

    /// <summary>畳む／開くをまとめて切り替える。</summary>
    public void SetFolded(bool folded)
    {
        _foldByDefault = folded;
        if (_folding is null)
        {
            return;
        }
        foreach (var section in _folding.AllFoldings)
        {
            section.IsFolded = folded;
        }
    }

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

        ScoreColumn.Attach(Editor.TextArea.TextView);
        Scores.Content = ScoreColumn;

        _ruler.Attach(Editor.TextArea.TextView);
        Ruler.Content = _ruler;

        // 折りたたみ。**差分から離れた一致行の連なりだけを畳む。**
        _folding = FoldingManager.Install(Editor.TextArea);

        // いる行を外へ伝える。**下の帯と地図がこれで追う。**
        Editor.TextArea.Caret.PositionChanged += (_, _) =>
            CaretLineChanged?.Invoke(this, Editor.TextArea.Caret.Line - 1);

        // 見えている範囲。地図の枠がこれで動く。
        //
        // **行が組まれたときにも知らせる。** スクロールしたときだけだと、
        // 開いた直後は初期値（全体）のままで、少し動かすまで枠が縮まらない。
        Editor.TextArea.TextView.ScrollOffsetChanged += (_, _) => RaiseViewport();
        Editor.TextArea.TextView.VisualLinesChanged += (_, _) => RaiseViewport();

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
    /// 折りたたみを組み直す。
    ///
    /// **差分の周りは畳まない。** 前後 3 行を残す。畳んだ場所は
    /// 「N 行」と出るので、隠したことは黙っていない。
    /// 短い連なりは畳まない（開け閉めの手間の方が大きい）。
    /// </summary>
    private void UpdateFoldings()
    {
        if (_folding is null || Editor.Document is null)
        {
            return;
        }

        const int context = 3;
        const int least = 6;

        var lines = _document.Lines;
        var keep = new bool[lines.Count];
        for (var i = 0; i < lines.Count; i++)
        {
            if (lines[i].BlockIndex < 0)
            {
                continue;
            }
            for (var k = Math.Max(0, i - context); k <= Math.Min(lines.Count - 1, i + context); k++)
            {
                keep[k] = true;
            }
        }

        var foldings = new List<NewFolding>();
        var start = -1;
        for (var i = 0; i <= lines.Count; i++)
        {
            var plain = i < lines.Count && !keep[i];
            if (plain && start < 0)
            {
                start = i;
            }
            else if (!plain && start >= 0)
            {
                var count = i - start;
                if (count >= least)
                {
                    var from = Editor.Document.GetLineByNumber(start + 1).Offset;
                    var to = Editor.Document.GetLineByNumber(i).EndOffset;
                    // **既定では開いておく。** いきなり畳まれていると、
                    // 何が隠れているのか分からないまま読み始めることになる。
                    // 畳みたい人は印を押す（まとめて畳むのは下のボタン）。
                    foldings.Add(new NewFolding(from, to)
                    {
                        Name = $"… {count} 行",
                        DefaultClosed = _foldByDefault,
                    });
                }
                start = -1;
            }
        }

        _folding.UpdateFoldings(foldings, -1);
    }

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
        // **見えている高さは本文の入れ物から取る。** このコントロール全体には
        // 矢印の列も含まれるが、高さは同じなのでどちらでもよい。
        var visible = view.Bounds.Height > 0 ? view.Bounds.Height : Bounds.Height;
        var size = Math.Min(1, visible / total);
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

    /// <summary>対応の近さ。**このペインの右端に置く。**</summary>
    public ScoreColumn ScoreColumn { get; } = new();

    private readonly RulerRow _ruler = new();

    /// <summary>
    /// 桁の目盛りを出すか。
    /// **固定長のデータで要る** — どの桁がずれたのかを数えられないと直せない。
    /// </summary>
    public bool ShowRuler
    {
        get => Ruler.IsVisible;
        set => Ruler.IsVisible = value;
    }

    /// <summary>
    /// 近さを出すか。**右のペインだけ。**
    /// 左にも出すと、画面の真ん中に数字が並んで本文の邪魔になる
    /// （数字は左右で同じ値なので、片方で足りる）。
    /// </summary>
    public bool ShowScores
    {
        get => Scores.IsVisible;
        set => Scores.IsVisible = value;
    }

    public void Fill(AlignedDocument document, bool readOnly, Language? language = null)
    {
        _document = document;
        ArrowColumn.Update(document.Lines);
        ScoreColumn.Update(document.Lines);
        _background.Update(document.Lines);
        _colorizer.Update(document.Lines, language);
        _numbers.Update(document.Lines);

        var caret = Editor.CaretOffset;
        var scroll = Editor.TextArea.TextView.VerticalOffset;

        _filling = true;
        Editor.IsReadOnly = readOnly;
        Editor.Text = document.Text;
        _filling = false;

        UpdateFoldings();
        Editor.CaretOffset = Math.Min(caret, Editor.Text.Length);

        // **入れ直した直後にも知らせる。** ここではまだ高さが決まって
        // いないことがあるので、組み終わってからもう一度。
        Avalonia.Threading.Dispatcher.UIThread.Post(RaiseViewport);

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
