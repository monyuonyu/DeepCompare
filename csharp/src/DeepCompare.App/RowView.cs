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
        set => Set(ref _editLeft, value);
    }

    private string _editRight = string.Empty;
    public string EditRight
    {
        get => _editRight;
        set => Set(ref _editRight, value);
    }

    /// <summary>その側に行があるときだけ直せる。無い行は作れない。</summary>
    public bool CanEditLeft => Row.Left is not null;
    public bool CanEditRight => Row.Right is not null;

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
    public InlineCollection LeftInlines { get; }
    public InlineCollection RightInlines { get; }

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
                AddRun(inlines, text.Substring(span.Start, span.Length),
                    diffBrush ?? Palette.Brush("FgNormal"));
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
    private static void AddRun(InlineCollection inlines, string text, IBrush brush)
    {
        if (!ShowWhitespace || text.Length == 0)
        {
            inlines.Add(new Run(text) { Foreground = brush });
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
                inlines.Add(new Run(text[start..i]) { Foreground = brush });
            }
            inlines.Add(new Run(mark.ToString()) { Foreground = faint });
            start = i + 1;
        }
        if (start < text.Length)
        {
            inlines.Add(new Run(text[start..]) { Foreground = brush });
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
