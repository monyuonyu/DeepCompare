using Avalonia.Controls.Documents;
using Avalonia.Media;
using DeepCompare.Engine;
// Avalonia にも Span があるので、どちらを指すか明示する。
using EngineSpan = DeepCompare.Engine.Span;

namespace DeepCompare.App;

/// <summary>
/// 表示 1 行分。
///
/// 文字列ではなく <see cref="InlineCollection"/> を組み立てているのが要点。旧 Python 実装は
/// ここで HTML を作っていたため、`&lt;` や `&amp;` を含む行が壊れていた。書式を持つ要素として
/// 渡せば、本文は一切加工されないのでその種の取り違えが起こりようがない。
/// </summary>
/// <summary>折りたたみの縦線の位置。</summary>
public enum OutlineMark
{
    None,

    /// <summary>範囲の先頭。ここに畳む箱を出す。</summary>
    Head,

    /// <summary>範囲の途中。</summary>
    Body,

    /// <summary>範囲の末尾。ここで線を止める。</summary>
    Tail,
}

public sealed class RowView : ViewModelBase
{
    // 色は App.axaml のテーマ辞書から引く（Palette）。ここで値を持たないので、
    // 明暗の切り替えに勝手に付いてくる。
    private static readonly IBrush Transparent = Brushes.Transparent;

    /// <summary>元の比較結果。移動の印を後から付けるために保持する。</summary>
    public Row Row { get; }

    /// <summary>
    /// 移動したブロックの相手側の行番号（1 始まり）。移動でないなら null。
    /// </summary>
    public int? MovedToLine
    {
        get;
        set
        {
            field = value;
            MovedText = value is { } line ? $"⇄{line}" : string.Empty;
        }
    }

    /// <summary>画面に出す移動の印。</summary>
    public string MovedText { get; private set; } = string.Empty;

    /// <summary>属する差分の塊。どれにも属さないなら -1。</summary>
    public int BlockIndex { get; set; } = -1;

    /// <summary>塊の先頭行か。コピーボタンはここにだけ出す。</summary>
    public bool IsBlockStart { get; set; }

    /// <summary>
    /// 反映の矢印を出すか。**書き込めない側へは出さない。**
    ///
    /// 出しておくと、押せる見た目なのに何も起きない（あるいは書き戻せない
    /// 相手に書こうとする）。git のある時点の中身や、ノートブック・Office から
    /// 取り出した本文がこれに当たる。
    /// </summary>
    public bool CanApplyToRight { get; set; }
    public bool CanApplyToLeft { get; set; }

    private bool _isEditing;

    /// <summary>
    /// この行を本文の中で直接編集しているか（BC の Character Mode）。
    ///
    /// **選んだ 1 行だけを入力欄にする。** 全行を入力欄にすると、仮想化して
    /// いても数千個の入力欄を作ることになり、開いた時点で重くなる。
    /// </summary>
    public bool IsEditing
    {
        get => _isEditing;
        set
        {
            if (Set(ref _isEditing, value) && value)
            {
                // 入る時点の中身を写す。取り消せるように元は残す。
                EditLeft = LeftText;
                EditRight = RightText;
            }
        }
    }

    private string _editLeft = string.Empty;
    public string EditLeft
    {
        get => _editLeft;
        set { if (Set(ref _editLeft, value)) { RefreshWhileEditing(); } }
    }

    private string _editRight = string.Empty;
    public string EditRight
    {
        get => _editRight;
        set { if (Set(ref _editRight, value)) { RefreshWhileEditing(); } }
    }

    private Language? _language;
    private LexState _leftState;
    private LexState _rightState;

    /// <summary>
    /// 打っている最中も、相手側の色を更新する。
    ///
    /// **編集中の行そのものには色を付けられない。** 入力欄は書式付きの
    /// 文字を扱えないため。だが**相手側は入力欄ではない**ので、そちらを
    /// 打つたびに塗り直せば「いまどこが違うか」が見える。
    ///
    /// 行の並びは触らない。**打つたびに組み直すと、書いている途中で
    /// 行が動いてカーソルを見失う。**
    /// </summary>
    private void RefreshWhileEditing()
    {
        if (!IsEditing)
        {
            return;
        }

        var left = CanEditLeft ? EditLeft : LeftText;
        var right = CanEditRight ? EditRight : RightText;
        if (!CanEditLeft || !CanEditRight)
        {
            return;
        }

        var (leftSpans, rightSpans) = InlineDiff.Compute(left, right, _language);
        LeftInlines = Build(left, leftSpans, _language, _leftState);
        RightInlines = Build(right, rightSpans, _language, _rightState);
        OnPropertyChanged(nameof(LeftInlines));
        OnPropertyChanged(nameof(RightInlines));
    }

    /// <summary>その側に行があるときだけ直せる。無い行は作れない。</summary>
    public bool CanEditLeft => Row.Left is not null;
    public bool CanEditRight => Row.Right is not null;

    /// <summary>
    /// 「ここに N 行畳んである」の帯か。
    ///
    /// **隠したことを黙っていない。** 差分だけを出すと一致行は消えるが、
    /// どれだけ消えたのかが分からないと、見落としたのか元から無いのかを
    /// 区別できない。Beyond Compare も「36 FILTERED LINES」の帯を出す。
    /// </summary>
    public bool IsFoldBand { get; init; }

    /// <summary>畳んである行数。</summary>
    public int FoldedCount { get; init; }

    /// <summary>畳んである範囲（元の行の添字）。押すと開くのに使う。</summary>
    public int FoldStart { get; init; }

    /// <summary>帯に出す数。**言葉は入れない。** 数と形だけで足りる。</summary>
    public string FoldCountText => FoldedCount.ToString("N0");

    /// <summary>
    /// 折りたたみの縦線の形（Excel のアウトラインと同じ考え方）。
    ///
    /// **畳める範囲がどこからどこまでかを、開いた状態でも見せる。**
    /// 帯は「畳んだ場所」しか示さないので、開いている間は範囲が分からない。
    /// </summary>
    public OutlineMark Outline { get; set; }

    /// <summary>その範囲の先頭と行数（元の行の添字）。畳むときに使う。</summary>
    public int OutlineStart { get; set; }
    public int OutlineCount { get; set; }

    /// <summary>
    /// 差分の塊の範囲を示す線（折りたたみと同じ形）。
    ///
    /// **矢印がどこまでを写すのかを見せる。** 塊の先頭にしか矢印が出ないので、
    /// 1 行だけなのか 20 行なのかが、押すまで分からなかった。
    /// </summary>
    public OutlineMark BlockOutline { get; set; }

    public bool BlockLineAbove => BlockOutline is OutlineMark.Body or OutlineMark.Tail;
    public bool BlockLineBelow => BlockOutline is OutlineMark.Head or OutlineMark.Body;
    public bool BlockFoot => BlockOutline == OutlineMark.Tail;

    public bool IsOutlineHead => Outline == OutlineMark.Head;
    public bool HasOutline => Outline != OutlineMark.None;

    /// <summary>縦線を出すか。先頭の行は箱を出すので線は下半分だけ。</summary>
    public bool OutlineLineAbove => Outline is OutlineMark.Body or OutlineMark.Tail;
    public bool OutlineLineBelow => Outline is OutlineMark.Head or OutlineMark.Body;
    public bool OutlineFoot => Outline == OutlineMark.Tail;

    /// <summary>帯を作る。中身は持たないので、比較の行は借りるだけ。</summary>
    public static RowView Band(Row anchor, DecodedText left, DecodedText right,
        int start, int count)
        => new(anchor, left, right) { IsFoldBand = true, FoldStart = start, FoldedCount = count };

    /// <summary>検索に使う素の本文。表示は Inlines 側が持つ。</summary>
    public string LeftText { get; }
    public string RightText { get; }

    public string LeftNumber { get; }
    public string RightNumber { get; }
    public string ScoreText { get; }
    public IBrush Background { get; }

    /// <summary>本文のセルごとの背景。対応が無い側は斜線になる。</summary>
    public IBrush LeftBackground { get; }
    public IBrush RightBackground { get; }
    /// <summary>
    /// 描く中身。**差し替えられるようにしてある**（打っている最中に
    /// 相手側を塗り直すため）。
    /// </summary>
    public InlineCollection LeftInlines { get; private set; }
    public InlineCollection RightInlines { get; private set; }

    /// <summary>
    /// 空白を記号で見せるか。
    ///
    /// **1 文字を 1 文字に置き換える。** 「&lt;U+200B&gt;」のように長さの変わる
    /// 置き換えをすると、差分の範囲（何文字目から何文字目か）とずれて、
    /// 色の付く位置が狂う。
    /// </summary>
    public static bool ShowWhitespace { get; set; }

    public RowView(Row row, DecodedText left, DecodedText right,
        Language? language = null, LexState leftState = default, LexState rightState = default)
    {
        Row = row;
        LeftNumber = row.Left is { } l ? (l + 1).ToString() : string.Empty;
        RightNumber = row.Right is { } r ? (r + 1).ToString() : string.Empty;

        var unchanged = row.IsUnchanged;
        ScoreText = row.Score switch
        {
            // 対応付けの上で一致した行。数値を出しても意味がないので記号にする。
            // 重要でない違いを含む場合は区別できるようにする。
            not null when unchanged => row.HasUnimportantDifferences ? "≈" : "=",
            { } score => score.ToString("F2"),
            null => string.Empty,
        };

        // **左右で背景を分ける。** 以前は行全体を 1 色で塗っていたので、
        // 片側にしか行が無いとき、空いている側まで同じ色になっていた。
        // 空いている側は斜線にして「対応が無い」ことを示す（BC と同じ）。
        (LeftBackground, RightBackground) = (row.Left, row.Right) switch
        {
            // 重要でない違い（空白だけ、無視する指定に当たった箇所）は青。
            // BC も「重要な差異は赤、重要でない差異は青」で分けている。
            (not null, not null) when unchanged && row.HasUnimportantDifferences
                => (Palette.Brush("BgUnimportant"), Palette.Brush("BgUnimportant")),
            (not null, not null) when unchanged => (Transparent, Transparent),
            (not null, not null) => (Palette.Brush("BgChanged"), Palette.Brush("BgChanged")),
            (not null, null) => (Palette.Brush("BgRemoved"), Palette.Gap()),
            (null, not null) => (Palette.Gap(), Palette.Brush("BgAdded")),
            _ => (Transparent, Transparent),
        };

        // 行全体の色は行番号の列などに効く。片側だけのときは、その側の色に寄せる。
        Background = (row.Left, row.Right) switch
        {
            (not null, not null) when unchanged && row.HasUnimportantDifferences
                => Palette.Brush("BgUnimportant"),
            (not null, not null) when unchanged => Transparent,
            (not null, not null) => Palette.Brush("BgChanged"),
            (not null, null) => Palette.Brush("BgRemoved"),
            (null, not null) => Palette.Brush("BgAdded"),
            _ => Transparent,
        };

        LeftText = row.Left is { } li ? left.Lines[li] : string.Empty;
        RightText = row.Right is { } ri ? right.Lines[ri] : string.Empty;
        LeftInlines = Build(row.Left is null ? null : LeftText, row.LeftSpans, language, leftState);
        RightInlines = Build(row.Right is null ? null : RightText, row.RightSpans, language, rightState);

        // 打っている最中に塗り直すために覚えておく。
        _language = language;
        _leftState = leftState;
        _rightState = rightState;
    }

    /// <summary>
    /// 差分の範囲と構文の範囲を重ねて描く。
    ///
    /// 2 つの範囲の切れ目は一致しないので、両方の境界で切り直す。差分が「変更あり」の
    /// ところは差分の色を使い、それ以外を構文の色にする。**どこが変わったかは
    /// 構文の色より優先する。** 色分けが綺麗でも変更点を見失っては本末転倒。
    /// </summary>
    private static InlineCollection Build(
        string? text, IReadOnlyList<EngineSpan> spans, Language? language, LexState state)
    {
        var inlines = new InlineCollection();
        if (text is null || text.Length == 0)
        {
            return inlines;
        }
        if (spans.Count == 0)
        {
            AddRun(inlines, text, Palette.Brush("FgNormal"));
            return inlines;
        }

        var tokens = language is null ? null : Lexer.Tokenize(text, language, ref state);

        foreach (var span in spans)
        {
            var diffBrush = span.Kind switch
            {
                SpanKind.Changed => Palette.Brush("FgInline"),
                SpanKind.Unimportant => Palette.Brush("FgUnimportant"),
                _ => (IBrush?)null,
            };

            if (tokens is null || diffBrush is not null)
            {
                // 変更部分は構文で塗り分けない。1 つの塊として見せる。
                //
                // **変わった文字は背景で塗る。** 文字の色を変えるだけだと、
                // 1 文字違いや、もともと色の付いた場所（文字列・注記）では
                // 差分だと気づけない。Beyond Compare も背景で示している。
                AddRun(inlines, text.Substring(span.Start, span.Length),
                    diffBrush ?? Palette.Brush("FgNormal"),
                    span.Kind == SpanKind.Changed ? Palette.Brush("BgInline") : null);
                continue;
            }

            // 一致部分は構文の色で塗る。範囲を跨ぐトークンは切り出す。
            var at = span.Start;
            var end = span.Start + span.Length;
            foreach (var token in tokens)
            {
                var from = Math.Max(at, token.Start);
                var to = Math.Min(end, token.Start + token.Length);
                if (to <= from)
                {
                    continue;
                }
                AddRun(inlines, text[from..to], Colour(token.Kind));
                at = to;
            }
            if (at < end)
            {
                AddRun(inlines, text[at..end], Palette.Brush("FgNormal"));
            }
        }
        return inlines;
    }

    /// <summary>
    /// 文字を足す。空白の可視化が入っているときは、空白だけ別の色の記号にする。
    ///
    /// **記号は必ず 1 文字に置き換える。** 長さが変わると、差分の範囲（何文字目
    /// から何文字目か）とずれて色の付く位置が狂う。
    /// </summary>
    private static void AddRun(
        InlineCollection inlines, string text, IBrush brush, IBrush? background = null)
    {
        if (!ShowWhitespace || text.Length == 0)
        {
            inlines.Add(new Run(text) { Foreground = brush, Background = background });
            return;
        }

        var faint = Palette.Brush("FgUnimportant");
        var start = 0;
        for (var i = 0; i < text.Length; i++)
        {
            var mark = text[i] switch
            {
                ' ' => '\u00b7',      // 半角空白 → 中点
                '\t' => '\u2192',     // タブ → 矢印
                '\u3000' => '\u25a1', // 全角空白 → 四角
                '\u00a0' => '\u00b0', // ノーブレークスペース → 度記号（普通の空白と区別する）
                _ => '\0',
            };
            if (mark == '\0')
            {
                continue;
            }

            if (i > start)
            {
                inlines.Add(new Run(text[start..i])
                    { Foreground = brush, Background = background });
            }
            inlines.Add(new Run(mark.ToString())
                { Foreground = faint, Background = background });
            start = i + 1;
        }
        if (start < text.Length)
        {
            inlines.Add(new Run(text[start..])
                { Foreground = brush, Background = background });
        }
    }

    private static IBrush Colour(TokenKind kind) => kind switch
    {
        TokenKind.Keyword => Palette.Brush("FgKeyword"),
        TokenKind.String => Palette.Brush("FgString"),
        TokenKind.Comment => Palette.Brush("FgComment"),
        TokenKind.Number => Palette.Brush("FgNumber"),
        TokenKind.Punctuation => Palette.Brush("FgPunctuation"),
        _ => Palette.Brush("FgNormal"),
    };
}
