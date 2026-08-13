using System.Collections.ObjectModel;
using System.Windows.Input;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// テキスト比較の画面。
///
/// 比較は必ず別スレッドで行う。旧 Python 実装は GUI スレッドで直接走らせていたため、
/// 比較中はウィンドウが固まり、起動引数つきで開くとモデル読み込み前に走り出して
/// スプラッシュ画面ごと固まっていた。
/// </summary>
public sealed class TextCompareViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    /// <summary>自分が乗っているタブ。見出しを比較の中身に合わせて書き換える。</summary>
    public CompareTab? Tab { get; set; }
    private Comparison? _comparison;
    private List<RowView> _allRows = [];
    private int _compareGeneration;

    /// <summary>
    /// 中身の取り出し方。既定はファイルから読む。
    ///
    /// git のある時点の中身を比べるときに差し替える。**バイト列で受け取る**ので、
    /// 符号化の判定は普段と同じ経路（<see cref="TextDecoder"/>）を通る。
    /// パスは表示にも使うので、<c>HEAD:src/A.cs</c> のように拡張子が残る形で渡すと
    /// 構文強調もそのまま効く。
    /// </summary>
    public Func<string, byte[]>? ContentLoader { get; set; }

    /// <summary>
    /// 書き戻せない側。git のある時点の中身は書き換えられない。
    ///
    /// 保存できないものを保存できるように見せると、押した後に失敗する。
    /// 最初から押せないようにする。
    /// </summary>
    public bool LeftReadOnly { get; set; }

    /// <summary>
    /// 保存の受け取り手。null ならファイルへ書く。
    ///
    /// **git の索引へ書き戻すために要る。** 索引はファイルではないので、
    /// パスへ書く経路では届かない。hunk 単位の stage がこれで成り立つ。
    /// </summary>
    public Func<IReadOnlyList<string>, DecodedText, Task>? LeftSaveHandler { get; set; }
    public Func<IReadOnlyList<string>, DecodedText, Task>? RightSaveHandler { get; set; }

    /// <summary>保存のボタンに出す言葉。用途で変わる。</summary>
    public string LeftSaveLabel { get; set; } = "左を保存";
    public string RightSaveLabel { get; set; } = "右を保存";

    public bool RightReadOnly { get; set; }

    // 編集の対象。読み込んだ時点の符号化と改行を保つため、DecodedText も持っておく。
    private DecodedText? _leftSource;
    private DecodedText? _rightSource;
    private EditableDocument? _leftDocument;
    private EditableDocument? _rightDocument;
    private List<DiffBlock> _blocks = [];

    // どちら側を編集したかの順序。取り消しは「最後に触った側」から戻す必要がある。
    // 片側ずつ独立した履歴だけでは、左→右→左と操作したときに順序が復元できない。
    private readonly Stack<bool> _undoSides = new();
    private readonly Stack<bool> _redoSides = new();

    private string _leftPath = string.Empty;
    private string _rightPath = string.Empty;
    private string _statusText = string.Empty;
    private string _placeholder = "比較する 2 つのファイルを指定してください（ドラッグ＆ドロップ可）";
    private bool _isBusy;
    private bool _changesOnly;
    private double _pairThreshold = Aligner.DefaultPairThreshold;
    private double _textWidth = 460;
    private double _viewportWidth;
    private double _contentWidth = 300;
    private int _selectedRowIndex = -1;
    private double _contextLines;
    private string _searchText = string.Empty;
    private bool _searchUseRegex;
    private bool _searchMatchCase;
    private bool _searchWholeWord;
    private string _searchStatus = string.Empty;
    private int _whitespaceMode;
    private bool _ignoreCase;
    private string _ignoredPatterns = string.Empty;
    private string _importanceError = string.Empty;

    public TextCompareViewModel(ShellViewModel shell)
    {
        _shell = shell;
        // 行の文字色は C# 側で焼き込むので、テーマが変わったら作り直す。
        _shell.ThemeChanged += RebuildForTheme;
        BrowseLeftCommand = new RelayCommand(() => PickAsync(left: true));
        BrowseRightCommand = new RelayCommand(() => PickAsync(left: false));
        CompareCommand = new RelayCommand(RunCompareAsync);
        CopyToRightCommand = new RelayCommand<RowView>(row => ApplyBlockAsync(row, toRight: true));
        CopyToLeftCommand = new RelayCommand<RowView>(row => ApplyBlockAsync(row, toRight: false));
        UndoCommand = new RelayCommand(UndoAsync, () => _undoSides.Count > 0);
        RedoCommand = new RelayCommand(RedoAsync, () => _redoSides.Count > 0);
        SaveLeftCommand = new RelayCommand(
            () => SaveAsync(left: true), () => LeftModified && !LeftReadOnly);
        SaveRightCommand = new RelayCommand(
            () => SaveAsync(left: false), () => RightModified && !RightReadOnly);
        ApplyImportanceCommand = new RelayCommand(RecompareAsync);
        ExportUnifiedCommand = new RelayCommand(() => ExportAsync(unified: true));
        ExportHtmlCommand = new RelayCommand(() => ExportAsync(unified: false));
        NextDifferenceCommand = new RelayCommand(
            () => { MoveToDifference(forward: true); return Task.CompletedTask; },
            () => HasDifferences);
        PreviousDifferenceCommand = new RelayCommand(
            () => { MoveToDifference(forward: false); return Task.CompletedTask; },
            () => HasDifferences);
        FindNextCommand = new RelayCommand(() => { FindFrom(forward: true); return Task.CompletedTask; });
        FindPreviousCommand = new RelayCommand(() => { FindFrom(forward: false); return Task.CompletedTask; });
        CopyLineCommand = new RelayCommand<RowView>(row => CopyTextAsync(row, both: false));
        CopyBothLinesCommand = new RelayCommand<RowView>(row => CopyTextAsync(row, both: true));
        CompareClipboardCommand = new RelayCommand(CompareClipboardAsync);
        CollapseOutlineCommand = new RelayCommand<RowView>(
            row => { CollapseOutline(row); return Task.CompletedTask; }, row => row.IsOutlineHead);
        ExpandFoldCommand = new RelayCommand<RowView>(
            band => { ExpandFold(band); return Task.CompletedTask; }, band => band.IsFoldBand);
        DeleteSelectedCommand = new RelayCommand(DeleteSelectedAsync);
        PasteOverSelectionCommand = new RelayCommand(PasteOverSelectionAsync);
        HideDetailsCommand = new RelayCommand(
            () => { ShowDetails = false; return Task.CompletedTask; });
        ShowDetailsCommand = new RelayCommand(
            () => { ShowDetails = true; return Task.CompletedTask; });
        SelectBlockCommand = new RelayCommand<RowView>(
            row => { SelectBlock(row); return Task.CompletedTask; }, row => row.BlockIndex >= 0);

        MarkForLinkCommand = new RelayCommand<RowView>(
            row => { MarkForLink(row); return Task.CompletedTask; },
            row => row.Row.Left is not null || row.Row.Right is not null);
        LinkRowsCommand = new RelayCommand<RowView>(
            row => LinkAsync(row), row => CanLinkWith(row));
        UnlinkRowCommand = new RelayCommand<RowView>(
            row => UnlinkAsync(row), row => row.Row is { Left: not null, Right: not null });
        ClearManualCommand = new RelayCommand(
            ClearManualAsync, () => !Manual.IsEmpty);
        CloseSearchCommand = new RelayCommand(
            () => { IsSearchOpen = false; return Task.CompletedTask; });
        OpenSearchCommand = new RelayCommand(
            () => { IsSearchOpen = true; return Task.CompletedTask; });
        OpenReplaceCommand = new RelayCommand(
            () => { IsReplaceOpen = true; return Task.CompletedTask; });
        ReplaceAllCommand = new RelayCommand(
            ReplaceAllAsync, () => SearchText.Length > 0 && (!LeftReadOnly || !RightReadOnly));

        CopySelectedLeftCommand = new RelayCommand(
            () => CopySelectionAsync(both: false), () => SelectedRows.Count > 0);
        CopySelectedBothCommand = new RelayCommand(
            () => CopySelectionAsync(both: true), () => SelectedRows.Count > 0);
    }

    public ObservableCollection<RowView> VisibleRows { get; } = [];
    public ICommand BrowseLeftCommand { get; }
    public ICommand BrowseRightCommand { get; }
    public ICommand CompareCommand { get; }

    /// <summary>テーマの切り替えなど、画面をまたぐ操作。</summary>
    public ShellViewModel Shell => _shell;
    public RelayCommand<RowView> CopyToRightCommand { get; }
    public RelayCommand<RowView> CopyToLeftCommand { get; }
    public RelayCommand<RowView> CopyLineCommand { get; }
    public RelayCommand<RowView> CopyBothLinesCommand { get; }
    public RelayCommand<RowView> SelectBlockCommand { get; }
    public RelayCommand<RowView> MarkForLinkCommand { get; }
    public RelayCommand<RowView> LinkRowsCommand { get; }
    public RelayCommand<RowView> UnlinkRowCommand { get; }
    public RelayCommand ClearManualCommand { get; }

    /// <summary>
    /// 人が指定した対応付け。**比較をやり直しても残す。**
    /// 直したそばから自動の判断に戻されたら、直す意味が無い。
    /// </summary>
    public ManualAlignment Manual { get; private set; } = new();

    /// <summary>「この行と繋ぐ」の起点。片方を選んでからもう片方を指す。</summary>
    private RowView? _linkAnchor;

    private string _manualNote = string.Empty;

    /// <summary>手動の指定について、いま何が起きているか。</summary>
    public string ManualNote
    {
        get => _manualNote;
        private set => Set(ref _manualNote, value);
    }

    public bool HasManual => !Manual.IsEmpty;

    private void MarkForLink(RowView row)
    {
        _linkAnchor = row;
        var side = row.Row.Left is not null ? "左" : "右";
        var line = (row.Row.Left ?? row.Row.Right ?? 0) + 1;
        ManualNote = $"{side} {line} 行目を起点にしました。繋ぐ相手を選んでください。";
        LinkRowsCommand.Raise();
    }

    /// <summary>
    /// 起点と繋げるか。**同じ側どうしは繋げない**（左は左、右は右の順序を保つ）。
    /// </summary>
    private bool CanLinkWith(RowView row)
    {
        if (_linkAnchor is not { } anchor || ReferenceEquals(anchor, row))
        {
            return false;
        }
        return (anchor.Row.Left is not null && row.Row.Right is not null)
            || (anchor.Row.Right is not null && row.Row.Left is not null);
    }

    private async Task LinkAsync(RowView row)
    {
        if (_linkAnchor is not { } anchor)
        {
            return;
        }

        var leftLine = anchor.Row.Left ?? row.Row.Left;
        var rightLine = anchor.Row.Right ?? row.Row.Right;
        if (leftLine is null || rightLine is null)
        {
            return;
        }

        Manual = Manual.Link(leftLine.Value, rightLine.Value);
        _linkAnchor = null;
        ManualNote = $"左 {leftLine + 1} 行目と右 {rightLine + 1} 行目を対応させました。";
        await RecompareAsync();
        RaiseManualState();
    }

    private async Task UnlinkAsync(RowView row)
    {
        if (row.Row is not { Left: { } left, Right: { } right })
        {
            return;
        }
        Manual = Manual.Unlink(left, right);
        ManualNote = $"左 {left + 1} 行目と右 {right + 1} 行目の対応を外しました。";
        await RecompareAsync();
        RaiseManualState();
    }

    private async Task ClearManualAsync()
    {
        Manual = new ManualAlignment();
        _linkAnchor = null;
        ManualNote = "手で付けた対応を全部取り消しました。";
        await RecompareAsync();
        RaiseManualState();
    }

    private void RaiseManualState()
    {
        OnPropertyChanged(nameof(HasManual));
        ClearManualCommand.Raise();
        LinkRowsCommand.Raise();
    }
    public RelayCommand<RowView> ExpandFoldCommand { get; }
    public RelayCommand<RowView> CollapseOutlineCommand { get; }
    public RelayCommand CompareClipboardCommand { get; }

    /// <summary>クリップボードの中身を読む。表示側から差し込む。</summary>
    public Func<Task<string?>>? ReadClipboard { get; set; }

    /// <summary>
    /// 左のファイルと、クリップボードの中身を比べる。
    ///
    /// **貼り付けてもらったコードを手元と突き合わせる**のによく使う。
    /// いちいち一時ファイルへ保存してから開く手間が消える。
    /// </summary>
    private async Task CompareClipboardAsync()
    {
        if (ReadClipboard is null)
        {
            return;
        }
        var text = await ReadClipboard();
        if (string.IsNullOrEmpty(text))
        {
            StatusText = "クリップボードに文字がありません。";
            return;
        }
        _shell.ShowTextAgainstClipboard(LeftPath.Trim(), text);
    }

    private string _modelWarning = string.Empty;

    /// <summary>
    /// モデルがこの本文を扱えないときの知らせ。
    ///
    /// **効いていないことを黙らない。** 日本語はいまのモデルの語彙にほとんど
    /// 無く、意味的な対応付けは効いていない。
    /// </summary>
    public string ModelWarning
    {
        get => _modelWarning;
        private set
        {
            if (Set(ref _modelWarning, value))
            {
                OnPropertyChanged(nameof(HasModelWarning));
            }
        }
    }

    public bool HasModelWarning => ModelWarning.Length > 0;

    /// <summary>
    /// その添字の行が見えるところまでスクロールする。表示側から差し込む。
    ///
    /// **段階 2 で行数が変わる**（2000 行の 5% 変更で 2098 → 2027 行）ので、
    /// 差し替えたときに読んでいた場所が飛ぶ。それを戻すために要る。
    /// </summary>
    public Action<int>? ScrollToRow { get; set; }

    /// <summary>
    /// いま画面の上に見えている行が、元のファイルの何行目か。
    ///
    /// **行の添字ではなく元の行番号を鍵にする。** 添字は差し替えで動くが、
    /// 元の行番号は動かない。
    /// </summary>
    private (int? Left, int? Right) CurrentAnchor()
    {
        if (VisibleRows.Count == 0)
        {
            return (null, null);
        }
        var at = Math.Clamp((int)(VisibleRows.Count * MapViewStart), 0, VisibleRows.Count - 1);
        var row = VisibleRows[at].Row;
        return (row.Left, row.Right);
    }

    /// <summary>覚えておいた場所へ戻す。見つからなければ何もしない。</summary>
    private void RestoreAnchor((int? Left, int? Right) anchor)
    {
        if (ScrollToRow is null)
        {
            return;
        }
        var at = RowAnchor.Find([.. VisibleRows.Select(v => v.Row)], anchor);
        if (at >= 0)
        {
            ScrollToRow(at);
        }
    }

    /// <summary>書き込み先。表示側から差し込む（ViewModel から画面に触らない）。</summary>
    public Action<string>? Clipboard { get; set; }

    /// <summary>
    /// その行を写す。
    ///
    /// <paramref name="both"/> が真なら左右をタブで区切って写す。表計算や
    /// 別の道具へ持っていくときに使う。
    /// </summary>
    private Task CopyTextAsync(RowView row, bool both)
    {
        Clipboard?.Invoke(both ? $"{row.LeftText}\t{row.RightText}" : row.LeftText);
        return Task.CompletedTask;
    }

    private bool _isSearchOpen;

    /// <summary>
    /// 検索の枠を出しているか。
    ///
    /// **常に出しておかない。** 検索欄と置換欄をツールバーに並べると、
    /// 使っていないときも幅を取り、他の操作が押し出される。
    /// VS Code と同じで、Ctrl+F で開き Esc で閉じる。
    /// </summary>
    public bool IsSearchOpen
    {
        get => _isSearchOpen;
        set
        {
            if (Set(ref _isSearchOpen, value) && !value)
            {
                // 閉じたら置換も畳む。**次に開いたとき置換が出ていると驚く。**
                IsReplaceOpen = false;
            }
        }
    }

    private bool _isReplaceOpen;

    /// <summary>置換の行も出しているか。**検索の中に畳んである。**</summary>
    public bool IsReplaceOpen
    {
        get => _isReplaceOpen;
        set
        {
            if (Set(ref _isReplaceOpen, value) && value)
            {
                // 置換を開くなら検索も開いている必要がある。
                IsSearchOpen = true;
            }
        }
    }

    private string _replaceText = string.Empty;

    /// <summary>置き換える文字列。**空でも通す**（消す操作になる）。</summary>
    public string ReplaceText
    {
        get => _replaceText;
        set => Set(ref _replaceText, value);
    }

    public RelayCommand ReplaceAllCommand { get; }
    public RelayCommand CloseSearchCommand { get; }
    public RelayCommand OpenSearchCommand { get; }
    public RelayCommand OpenReplaceCommand { get; }

    /// <summary>
    /// 検索に当たるところを全部置き換える。
    ///
    /// **読み取り専用の側は触らない。** git の索引や取り出した本文は
    /// 書き戻せないので、そこを書き換えると保存で失敗する。
    ///
    /// **取り消しで戻せる。** 一括置換は当たり所を読み違えると被害が大きい。
    /// </summary>
    private async Task ReplaceAllAsync()
    {
        if (_leftDocument is null || _rightDocument is null || SearchText.Length == 0)
        {
            return;
        }

        var query = new SearchQuery(SearchText, SearchUseRegex, SearchMatchCase, SearchWholeWord);
        var total = 0;

        // 0 = 両方 / 1 = 左だけ / 2 = 右だけ（検索の Sides と同じ意味）。
        if (SearchSide != 2 && !LeftReadOnly)
        {
            var replaced = TextSearch.ReplaceAll(
                _leftDocument.Lines, query, ReplaceText, out var count);
            if (count > 0)
            {
                _leftDocument.Replace(0, _leftDocument.Lines.Count, replaced);
                _undoSides.Push(false);
                total += count;
            }
        }
        if (SearchSide != 1 && !RightReadOnly)
        {
            var replaced = TextSearch.ReplaceAll(
                _rightDocument.Lines, query, ReplaceText, out var count);
            if (count > 0)
            {
                _rightDocument.Replace(0, _rightDocument.Lines.Count, replaced);
                _undoSides.Push(true);
                total += count;
            }
        }

        if (total == 0)
        {
            // **「置き換えました」と出さない。** 当たらなかったのか、
            // 読み取り専用で触れなかったのかが分かるようにする。
            StatusText = LeftReadOnly && RightReadOnly
                ? "どちらも読み取り専用なので置き換えられません。"
                : $"「{SearchText}」は見つかりませんでした。";
            return;
        }

        _redoSides.Clear();
        await RecompareAsync();
        StatusText = $"{total} か所を置き換えました（取り消しで戻せます）。";
    }

    /// <summary>
    /// いま選んでいる行。**画面側が入れる**（ListBox の複数選択）。
    /// 1 行だけのときも入るので、写しの操作はこちらに寄せられる。
    /// </summary>
    public System.Collections.IList SelectedRows { get; } = new List<object>();

    public RelayCommand DeleteSelectedCommand { get; private set; } = null!;
    public RelayCommand PasteOverSelectionCommand { get; private set; } = null!;

    /// <summary>
    /// いま触っている側。
    ///
    /// **左右のどちらに効かせるかは、最後に押した側で決める。**
    /// 一括の削除や貼り付けは「どちらの本文を書き換えるか」を必ず要求するが、
    /// そのたびに選ばせるのは煩わしい。Beyond Compare も同じく、
    /// 触った側が対象になる。
    /// </summary>
    public bool ActiveIsLeft { get; set; } = true;

    /// <summary>
    /// 選んだ行をまとめて消す。
    ///
    /// **読み取り専用の側は触らない。** 取り消しで戻せる。
    /// </summary>
    public async Task DeleteSelectedAsync()
    {
        await ReplaceSelectedAsync([]);
    }

    /// <summary>
    /// 選んだ行を、クリップボードの中身で置き換える。
    ///
    /// **行の数が違ってもよい。** 1 行を選んで 100 行貼れば 100 行になる。
    /// これができないと「全部選んで貼り替える」ができない。
    /// </summary>
    public async Task PasteOverSelectionAsync()
    {
        if (ReadClipboard is null)
        {
            return;
        }
        var text = await ReadClipboard();
        if (string.IsNullOrEmpty(text))
        {
            StatusText = "クリップボードに文字がありません。";
            return;
        }
        // 末尾の改行で空行が 1 つ増えるのを避ける。
        var lines = text.Replace("\r\n", "\n").TrimEnd('\n').Split('\n');
        await ReplaceSelectedAsync(lines);
    }

    /// <summary>選んだ行を差し替える土台。消すのも貼るのもここを通る。</summary>
    private async Task ReplaceSelectedAsync(IReadOnlyList<string> replacement)
    {
        var document = ActiveIsLeft ? _leftDocument : _rightDocument;
        var readOnly = ActiveIsLeft ? LeftReadOnly : RightReadOnly;
        if (document is null || readOnly || SelectedRows.Count == 0)
        {
            StatusText = readOnly
                ? "この側は読み取り専用です。"
                : "行を選んでください。";
            return;
        }

        // 選んだ行のうち、その側に実体のあるものだけを集める。
        // **片側にしか無い行を選んでいることがある**（相手側は空欄）。
        var indexes = new List<int>();
        foreach (var item in SelectedRows)
        {
            if (item is not RowView row)
            {
                continue;
            }
            var at = ActiveIsLeft ? row.Row.Left : row.Row.Right;
            if (at is { } value)
            {
                indexes.Add(value);
            }
        }
        if (indexes.Count == 0)
        {
            StatusText = "選んだ行は、この側にはありません。";
            return;
        }

        indexes.Sort();
        var start = indexes[0];
        var count = indexes[^1] - start + 1;

        document.Replace(start, count, replacement);
        _undoSides.Push(!ActiveIsLeft);
        _redoSides.Clear();
        RaiseEditState();
        await RecompareAsync();

        StatusText = replacement.Count == 0
            ? $"{count} 行を消しました（取り消しで戻せます）。"
            : $"{count} 行を {replacement.Count} 行に置き換えました（取り消しで戻せます）。";
    }

    public RelayCommand CopySelectedLeftCommand { get; }
    public RelayCommand CopySelectedBothCommand { get; }

    /// <summary>選択が変わったことを画面から知らせてもらう。</summary>
    public void SelectionChanged()
    {
        CopySelectedLeftCommand.Raise();
        CopySelectedBothCommand.Raise();
        OnPropertyChanged(nameof(SelectionSummary));
        OnPropertyChanged(nameof(HasSelection));
    }

    /// <summary>選んでいる行数。**0 行と 1 行は出さない** — 常に出ると読まなくなる。</summary>
    public string SelectionSummary
        => SelectedRows.Count > 1 ? $"{SelectedRows.Count} 行を選択中" : string.Empty;

    public bool HasSelection => SelectedRows.Count > 1;

    /// <summary>
    /// 選んだ行をまとめて写す。
    ///
    /// **畳んだ帯は飛ばす。** 帯には本文が無いので、写すと空行が混ざる。
    /// **画面の順序で写す。** 選んだ順ではない（Ctrl を押しながら飛び飛びに
    /// 選ぶと、貼り付けたとき元の順序でないと読めない）。
    /// </summary>
    private Task CopySelectionAsync(bool both)
    {
        var chosen = SelectedRows.OfType<RowView>().ToHashSet();
        if (chosen.Count == 0)
        {
            return Task.CompletedTask;
        }

        var text = new System.Text.StringBuilder();
        foreach (var row in VisibleRows)
        {
            if (!chosen.Contains(row) || row.IsFoldBand)
            {
                continue;
            }
            text.AppendLine(both ? $"{row.LeftText}\t{row.RightText}" : row.LeftText);
        }

        Clipboard?.Invoke(text.ToString());
        StatusText = $"{chosen.Count} 行を写しました。";
        return Task.CompletedTask;
    }

    /// <summary>その塊の先頭へ移る。塊の途中で右クリックしたときに使う。</summary>
    private void SelectBlock(RowView row)
    {
        for (var i = 0; i < VisibleRows.Count; i++)
        {
            if (VisibleRows[i].BlockIndex == row.BlockIndex && VisibleRows[i].IsBlockStart)
            {
                SelectedRowIndex = i;
                return;
            }
        }
    }
    public RelayCommand UndoCommand { get; }
    public RelayCommand RedoCommand { get; }
    public RelayCommand SaveLeftCommand { get; }
    public RelayCommand SaveRightCommand { get; }

    /// <summary>
    /// 空白の扱い。<see cref="WhitespaceMode"/> の並び順に対応する。
    /// 画面では選択肢の位置で持ち、比較へ渡すときに列挙型へ戻す。
    /// </summary>
    public int WhitespaceModeIndex
    {
        get => _whitespaceMode;
        set { if (Set(ref _whitespaceMode, value)) { _ = RecompareAsync(); } }
    }

    public bool IgnoreCase
    {
        get => _ignoreCase;
        set { if (Set(ref _ignoreCase, value)) { _ = RecompareAsync(); } }
    }

    /// <summary>無視する正規表現。1 行 1 つ。</summary>
    public string IgnoredPatterns
    {
        get => _ignoredPatterns;
        set => Set(ref _ignoredPatterns, value);
    }

    /// <summary>正規表現が不正だったときの説明。</summary>
    public string ImportanceError
    {
        get => _importanceError;
        private set => Set(ref _importanceError, value);
    }

    public ICommand ApplyImportanceCommand { get; }
    public ICommand ExportUnifiedCommand { get; }
    public ICommand ExportHtmlCommand { get; }
    public RelayCommand NextDifferenceCommand { get; }
    public RelayCommand PreviousDifferenceCommand { get; }
    public ICommand FindNextCommand { get; }
    public ICommand FindPreviousCommand { get; }

    /// <summary>一覧で選んでいる行。差分間の移動と検索はここを動かす。</summary>
    public int SelectedRowIndex
    {
        get => _selectedRowIndex;
        set
        {
            if (Set(ref _selectedRowIndex, value))
            {
                // 選んだ行を下の帯へ移す。ここで直せる。
                ShowDetailFor(value >= 0 && value < VisibleRows.Count ? VisibleRows[value] : null);
                UpdateEditingRow();
            }
        }
    }

    /// <summary>畳んだときに変更の前後へ残す行数。</summary>
    public double ContextLines
    {
        get => _contextLines;
        set
        {
            if (Set(ref _contextLines, value))
            {
                ResetFolds();
                RebuildVisibleRows();
            }
        }
    }

    public string SearchText
    {
        get => _searchText;
        set
        {
            if (Set(ref _searchText, value))
            {
                ReplaceAllCommand.Raise();
                OnPropertyChanged(nameof(SearchStatus));
            }
        }
    }

    public bool SearchUseRegex
    {
        get => _searchUseRegex;
        set => Set(ref _searchUseRegex, value);
    }

    public bool SearchMatchCase
    {
        get => _searchMatchCase;
        set => Set(ref _searchMatchCase, value);
    }

    public bool SearchWholeWord
    {
        get => _searchWholeWord;
        set => Set(ref _searchWholeWord, value);
    }

    public string SearchStatus
    {
        get => _searchStatus;
        private set => Set(ref _searchStatus, value);
    }

    public bool LeftModified => _leftDocument?.IsModified ?? false;
    public bool RightModified => _rightDocument?.IsModified ?? false;

    /// <summary>保存していない変更があるか。画面を離れる前の確認に使う。</summary>
    public bool HasUnsavedChanges => LeftModified || RightModified;

    /// <summary>戻り先がフォルダー一覧なら、そう分かる文言にする。</summary>

    public string LeftPath
    {
        get => _leftPath;
        set => Set(ref _leftPath, value);
    }

    public string RightPath
    {
        get => _rightPath;
        set => Set(ref _rightPath, value);
    }

    public string StatusText
    {
        get => _statusText;
        private set => Set(ref _statusText, value);
    }

    private string _diffCountText = string.Empty;

    /// <summary>
    /// 差分が何か所あるか。
    ///
    /// **行数ではなく塊の数を数える。** 20 行まとめて書き換えた場所は
    /// 「1 か所」。直す作業の単位はそちらで、行数は作業量の目安にならない。
    /// </summary>
    public string DiffCountText
    {
        get => _diffCountText;
        private set => Set(ref _diffCountText, value);
    }

    // ---- 符号化 ----

    /// <summary>
    /// 選べる符号化。
    ///
    /// **推定が外れたときの逃げ道。** 日本語がわずかしか含まれない
    /// Shift_JIS のファイルは UTF-8 として読めてしまうことがあり、
    /// そのときは化けた本文を前に、直しようが無かった。
    /// </summary>
    public IReadOnlyList<EncodingChoice> Encodings { get; } =
    [
        new(TextEncoding.Utf8, "UTF-8"),
        new(TextEncoding.Utf8Bom, "UTF-8 (BOM)"),
        new(TextEncoding.ShiftJis, "Shift_JIS"),
        new(TextEncoding.EucJp, "EUC-JP"),
        new(TextEncoding.Utf16Le, "UTF-16 LE"),
        new(TextEncoding.Utf16Be, "UTF-16 BE"),
    ];

    private TextEncoding? _forcedLeftEncoding;
    private TextEncoding? _forcedRightEncoding;

    private EncodingChoice? _leftEncoding;

    /// <summary>左の符号化。**変えると読み直す。**</summary>
    public EncodingChoice? LeftEncoding
    {
        get => _leftEncoding;
        set
        {
            var before = _leftEncoding;
            if (!Set(ref _leftEncoding, value) || value is null || before is null)
            {
                return;
            }
            // 表示のために入れ直しただけなら読み直さない。
            if (value.Value == _leftSource?.Encoding)
            {
                return;
            }
            _forcedLeftEncoding = value.Value;
            _ = ReloadAsync();
        }
    }

    private EncodingChoice? _rightEncoding;
    public EncodingChoice? RightEncoding
    {
        get => _rightEncoding;
        set
        {
            var before = _rightEncoding;
            if (!Set(ref _rightEncoding, value) || value is null || before is null)
            {
                return;
            }
            if (value.Value == _rightSource?.Encoding)
            {
                return;
            }
            _forcedRightEncoding = value.Value;
            _ = ReloadAsync();
        }
    }

    private string _leftEnding = string.Empty;
    public string LeftEnding
    {
        get => _leftEnding;
        private set => Set(ref _leftEnding, value);
    }

    private string _rightEnding = string.Empty;
    public string RightEnding
    {
        get => _rightEnding;
        private set => Set(ref _rightEnding, value);
    }

    /// <summary>
    /// 符号化を選び直したので読み直す。
    ///
    /// **直していたら断る。** 読み直せば編集は消える。捨ててよいかは
    /// 人にしか決められない。
    /// </summary>
    private async Task ReloadAsync()
    {
        if (LeftModified || RightModified)
        {
            StatusText = "直した内容があるので読み直せません。保存するか、取り消してください。";
            return;
        }
        await RunCompareAsync();
    }

    public string Placeholder
    {
        get => _placeholder;
        private set => Set(ref _placeholder, value);
    }

    public bool ShowPlaceholder => VisibleRows.Count == 0;

    private string _invisibleWarning = string.Empty;

    /// <summary>
    /// 見えない差分の知らせ。
    ///
    /// **聞かれてから答えるのでは遅い。** 「なぜか一致しない」と悩む前に、
    /// ゼロ幅文字や全角空白が混じっていることを出す。何も無ければ空。
    /// </summary>
    public string InvisibleWarning
    {
        get => _invisibleWarning;
        private set
        {
            if (Set(ref _invisibleWarning, value))
            {
                OnPropertyChanged(nameof(HasInvisibleWarning));
            }
        }
    }

    public bool HasInvisibleWarning => _invisibleWarning.Length > 0;

    private void UpdateInvisibleWarning(DecodedText left, DecodedText right)
    {
        // 行末の空白と最終行の改行は、ここで騒ぐほどではない（前者は
        // 「重要でない差分」で畳めるし、後者は既に差分として出る）。
        static IEnumerable<InvisibleFinding> Notable(DecodedText text)
            => InvisibleScanner.Scan(text).Where(f =>
                f.Kind is not (InvisibleKind.TrailingWhitespace or InvisibleKind.NoFinalNewline));

        var findings = Notable(left).Concat(Notable(right)).ToList();
        if (findings.Count == 0)
        {
            InvisibleWarning = string.Empty;
            return;
        }

        var parts = findings
            .GroupBy(f => f.Kind)
            .OrderByDescending(g => g.Count())
            .Select(g => $"{InvisibleFinding.Label(g.Key)} {g.Count()}");
        InvisibleWarning = "見えない差分: " + string.Join("、", parts);
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set => Set(ref _isBusy, value);
    }

    /// <summary>完全一致の行を畳んで、変更点だけを見る。</summary>
    public bool ChangesOnly
    {
        get => _changesOnly;
        set
        {
            if (Set(ref _changesOnly, value))
            {
                ResetFolds();
                RebuildVisibleRows();
            }
        }
    }

    /// <summary>
    /// 対応付けると判断する類似度の下限。
    ///
    /// 既定の 0.5 を動かせるようにしているのは、短いコード行では類似度が伸びにくく、
    /// `self.config = config` と `self.settings = settings` のように明らかに対応する行でも
    /// 境界付近に落ちることがあるため。適切な値は比較するコードの性質によって変わる。
    /// </summary>
    public double PairThreshold
    {
        get => _pairThreshold;
        set => Set(ref _pairThreshold, value);
    }

    /// <summary>
    /// 本文列の幅。
    ///
    /// 最長行に合わせるだけだと、行の短いファイルで右側が大きく余る。かといって
    /// 常に画面幅で割ると長い行が読めない。**内容に必要な幅と、画面を左右で
    /// 分け合った幅の、大きい方**を採る。
    /// </summary>
    public double TextWidth
    {
        get => _textWidth;
        private set => Set(ref _textWidth, value);
    }

    /// <summary>一覧に使える横幅。画面側から知らせてもらう。</summary>
    public double ViewportWidth
    {
        get => _viewportWidth;
        set
        {
            if (Set(ref _viewportWidth, value))
            {
                UpdateTextWidth();
            }
        }
    }

    private bool _showWhitespace;

    /// <summary>
    /// 空白を記号で見せるか（BC の Visible Whitespace）。
    ///
    /// **「なぜか一致しない」の原因が空白のとき、これが唯一の手がかりになる。**
    /// 下の帯の知らせは「どこかにある」ことしか言わないが、これは位置まで示す。
    /// </summary>
    public bool ShowWhitespace
    {
        get => _showWhitespace;
        set
        {
            if (Set(ref _showWhitespace, value))
            {
                RowView.ShowWhitespace = value;
                RebuildForTheme();   // 行を作り直す（色と同じ理由）
            }
        }
    }

    // --- 差分の地図 ---

    private double _mapViewStart;
    /// <summary>地図に出す「いま見えている範囲」の先頭（0〜1）。</summary>
    public double MapViewStart
    {
        get => _mapViewStart;
        set => Set(ref _mapViewStart, value);
    }

    private double _mapViewSize = 1;
    public double MapViewSize
    {
        get => _mapViewSize;
        set => Set(ref _mapViewSize, value);
    }

    private double _mapClicked;
    /// <summary>
    /// 地図が押された位置（0〜1）。
    ///
    /// 選択している行を動かすことでそこへ飛ぶ。ListBox は
    /// AutoScrollToSelectedItem なので、選択を変えれば付いてくる。
    /// </summary>
    public double MapClicked
    {
        get => _mapClicked;
        set
        {
            if (!Set(ref _mapClicked, value) || value < 0 || VisibleRows.Count == 0)
            {
                return;
            }
            var index = (int)Math.Round(value * (VisibleRows.Count - 1));
            SelectedRowIndex = Math.Clamp(index, 0, VisibleRows.Count - 1);
        }
    }

    /// <summary>本文以外の列（行番号 3 つ・コピーボタン・移動の印）が使う幅。</summary>
    // 折りたたみの柱 ＋ 行番号 3 つ ＋ 矢印 2 つ ＋ 移動の印
    /// <summary>
    /// 本文以外が使う幅。
    ///
    /// 内訳: 差分の地図 26 ＋ 折りたたみの柱 18 ＋ 行番号と類似度 52×3
    /// ＋ 反映の矢印 22×2 ＋ 移動の印 52 ＋ 縦スクロールバー 18。
    ///
    /// **地図とスクロールバーを数え忘れていた。** その分だけ本文が広く取られ、
    /// 右端の類似度が画面の外へ押し出されていた（既定の窓の大きさで、
    /// この道具の売りである類似度が見えない状態だった）。
    /// </summary>
    private const double GutterWidth = 26 + 18 + 52 * 3 + 22 * 2 + 52 + 18;

    private void UpdateTextWidth()
    {
        // 画面幅が分からないうちは内容だけで決める。起動直後に一度だけ通る。
        if (_viewportWidth <= 0)
        {
            TextWidth = _contentWidth;
            return;
        }

        // **内容の幅で広げない。** 以前は「内容に要る幅」と「画面を分け合った幅」の
        // 大きい方を採っていたが、片側に長い行が 1 本あるだけで両側がその幅まで
        // 広がり、右のペインが画面の外へ押し出されていた。
        //
        // 左右は必ず半分ずつにする。長い行は横スクロールで見る（左右が同時に
        // 動くので、対応する行が画面から外れない）。
        // **上下に並べるときは分け合わない。** 1 行が全幅を使えるのが
        // この配置の目的で、半分にしたら横に並べるのと変わらない。
        TextWidth = OverUnder
            ? Math.Max(300, _viewportWidth - GutterWidth / 2)
            : Math.Max(300, (_viewportWidth - GutterWidth) / 2);
    }

    private async Task PickAsync(bool left)
    {
        var title = left ? "ファイル1を選択" : "ファイル2を選択";
        if (await _shell.PickPath(title, false) is { } path)
        {
            if (left)
            {
                LeftPath = path;
            }
            else
            {
                RightPath = path;
            }
        }
    }

    public void AcceptDroppedFiles(IReadOnlyList<string> paths)
    {
        if (paths.Count == 0)
        {
            return;
        }
        if (paths.Count >= 2)
        {
            LeftPath = paths[0];
            RightPath = paths[1];
            return;
        }
        if (string.IsNullOrWhiteSpace(LeftPath))
        {
            LeftPath = paths[0];
        }
        else if (string.IsNullOrWhiteSpace(RightPath))
        {
            RightPath = paths[0];
        }
        else
        {
            LeftPath = paths[0];
        }
    }

    private async Task RunCompareAsync()
    {
        // **フォルダーを渡されたらフォルダー比較へ移す。**
        // 以前はそのまま読みに行き、「Access to the path ... is denied」という
        // 何が悪いのか分からない知らせが出ていた（Windows で踏んだ）。
        var leftIsFolder = LeftPath.Trim() is { Length: > 0 } l && Directory.Exists(l);
        var rightIsFolder = RightPath.Trim() is { Length: > 0 } r && Directory.Exists(r);
        if (leftIsFolder && rightIsFolder)
        {
            _shell.ShowFolders(LeftPath.Trim(), RightPath.Trim());
            return;
        }
        if (leftIsFolder || rightIsFolder)
        {
            Placeholder = "片方がフォルダーです。"
                + "テキストとして比べるならファイルを、"
                + "中身を突き合わせるなら両方フォルダーを指定してください。";
            return;
        }

        // **片方だけでも中身を出す。** 「両方指定してください」とだけ言われて
        // 何も出ないと、指定したものが読めているのかすら分からない。
        if (string.IsNullOrWhiteSpace(LeftPath) != string.IsNullOrWhiteSpace(RightPath))
        {
            await ShowOneSideAsync();
            return;
        }
        if (string.IsNullOrWhiteSpace(LeftPath) && string.IsNullOrWhiteSpace(RightPath))
        {
            Placeholder = "比べるファイルを指定してください。";
            return;
        }

        // 後から始まった比較が先に終わることがある。世代を持たせて、古い結果が
        // 新しい結果を上書きしないようにする。
        var generation = ++_compareGeneration;

        IsBusy = true;
        StatusText = "解析しています…";
        var leftPath = LeftPath.Trim();
        var rightPath = RightPath.Trim();
        var compareOptions = BuildOptions();

        try
        {
            // 段階 1: 文字列一致だけで組む。2000 行で 20ms 程度なので、体感は即座。
            var first = await Task.Run(() =>
            {
                var load = ContentLoader ?? File.ReadAllBytes;
                // 人が指定した符号化があればそれで読む。無ければ推定に任せる。
                var left = _forcedLeftEncoding is { } fl
                    ? TextDecoder.Decode(load(leftPath), fl)
                    : TextDecoder.Decode(load(leftPath));
                var right = _forcedRightEncoding is { } fr
                    ? TextDecoder.Decode(load(rightPath), fr)
                    : TextDecoder.Decode(load(rightPath));
                var started = DateTime.UtcNow;
                var comparison = DiffComparer.Compare(
                    left, right, embedder: null, compareOptions);
                return (left, right, comparison, elapsed: DateTime.UtcNow - started);
            });

            if (generation != _compareGeneration)
            {
                return;
            }

            // 読み込み直しなので編集履歴は捨てる。別のファイルの履歴を引き継ぐと、
            // 取り消したときに何が起きるか説明できない。
            _leftSource = first.left;
            _rightSource = first.right;
            _leftDocument = new EditableDocument(first.left.Lines);
            _rightDocument = new EditableDocument(first.right.Lines);
            _undoSides.Clear();
            _redoSides.Clear();
            RaiseEditState();

            Apply(first.left, first.right, first.comparison, first.elapsed, refining: true);

            // ここで操作可能にする。以降は待たせない。
            IsBusy = false;


            // 高速モードなら段階 2 を走らせない。**段階 1 の答えで確定する。**
            if (_shell.FastMode)
            {
                StatusText += "（構造のみ）";
                return;
            }

            // 段階 2: 埋め込みで対応付けを取り直し、差し替える。
            var warning = string.Empty;
            var refinedResult = await Task.Run(() =>
            {
                // モデルは殻が保持していて、初回だけ読む。
                var embedder = _shell.GetEmbedder();

                // **モデルが扱えない本文なら知らせる。** 効いていないのに
                // 「意味的に比べました」と出すのは、黙って間違った結果を
                // 出すのに近い。**語彙の数で判断する** — 多言語モデルなら
                // 日本語でも効くので、そのときは黙る。
                warning = ModelCoverage.Warn(
                    first.left.Lines, first.right.Lines, embedder.VocabSize) ?? string.Empty;

                var started = DateTime.UtcNow;
                var comparison = DiffComparer.Compare(
                    first.left, first.right, embedder, compareOptions);
                return (comparison, elapsed: DateTime.UtcNow - started);
            });

            if (generation != _compareGeneration)
            {
                return;
            }

            ModelWarning = warning;

            // **読んでいた場所を保つ。** 段階 2 は行数を変えるので、
            // 何もしないとスクロールが飛ぶ（届く前に読み始めていると気になる）。
            var anchor = CurrentAnchor();
            Apply(first.left, first.right, refinedResult.comparison, refinedResult.elapsed, refining: false);
            RestoreAnchor(anchor);
        }
        catch (Exception error)
        {
            if (generation != _compareGeneration)
            {
                return;
            }
            _allRows = [];
            VisibleRows.Clear();
            Placeholder = $"エラー: {error.Message}";
            StatusText = string.Empty;
            OnPropertyChanged(nameof(ShowPlaceholder));
        }
        finally
        {
            if (generation == _compareGeneration)
            {
                IsBusy = false;
            }
        }
    }

    /// <summary>テーマが変わったときに、いまの結果のまま行だけ作り直す。</summary>
    private void RebuildForTheme()
    {
        if (_comparison is null || _leftSource is null || _rightSource is null)
        {
            return;
        }
        Apply(CurrentLeft, CurrentRight, _comparison, TimeSpan.Zero, refining: false, keepStatus: true);
    }

    /// <summary>比較結果を画面に反映する。段階 1 と段階 2 の両方から呼ぶ。</summary>
    private void Apply(
        DecodedText left, DecodedText right, Comparison comparison, TimeSpan elapsed,
        bool refining, bool keepStatus = false)
    {
        _comparison = comparison;
        _blocks = Merge.Blocks(comparison);

        // 色分けの状態は行をまたぐ（ブロックコメント）ので、左右それぞれ順に持ち回る。
        // 表示している行だけを見ると、コメントの途中の行が色付かない。
        var language = DetectLanguage();
        var leftState = LexState.Start;
        var rightState = LexState.Start;
        _allRows = new List<RowView>(comparison.Rows.Count);
        foreach (var r in comparison.Rows)
        {
            var beforeLeft = leftState;
            var beforeRight = rightState;
            if (language is not null)
            {
                if (r.Left is { } li)
                {
                    Lexer.Tokenize(left.Lines[li], language, ref leftState);
                }
                if (r.Right is { } ri)
                {
                    Lexer.Tokenize(right.Lines[ri], language, ref rightState);
                }
            }
            _allRows.Add(new RowView(r, left, right, language, beforeLeft, beforeRight));
        }

        // 移動したブロックに印を付ける。片側だけに出た行が、実は別の場所へ動いた
        // だけだと分かれば、2 箇所を見比べる手間が要らなくなる。
        var moved = MovedBlockDetector.Detect(comparison, left, right);
        if (moved.Count > 0)
        {
            var leftMoved = new Dictionary<int, int>();
            var rightMoved = new Dictionary<int, int>();
            for (var m = 0; m < moved.Count; m++)
            {
                for (var k = 0; k < moved[m].Length; k++)
                {
                    leftMoved[moved[m].LeftStart + k] = moved[m].RightStart + k;
                    rightMoved[moved[m].RightStart + k] = moved[m].LeftStart + k;
                }
            }
            foreach (var view in _allRows)
            {
                if (view.Row.Right is null && view.Row.Left is { } ml
                    && leftMoved.TryGetValue(ml, out var toRight))
                {
                    view.MovedToLine = toRight + 1;
                }
                else if (view.Row.Left is null && view.Row.Right is { } mr
                    && rightMoved.TryGetValue(mr, out var fromLeft))
                {
                    view.MovedToLine = fromLeft + 1;
                }
            }
        }

        // 塊の先頭行にだけコピーボタンを出す。全行に出すと、1 行ずつ反映できるように
        // 見えてしまうが、実際の単位は塊。
        for (var i = 0; i < _blocks.Count; i++)
        {
            var block = _blocks[i];
            for (var k = 0; k < block.RowCount; k++)
            {
                var row = _allRows[block.RowStart + k];
                row.BlockIndex = i;
                row.IsBlockStart = k == 0;
                // 矢印は**書き込める側へだけ**出す。
                row.CanApplyToRight = k == 0 && !RightReadOnly;
                row.CanApplyToLeft = k == 0 && !LeftReadOnly;
            }
        }

        // 最長行に合わせて本文列の幅を決める。等幅なので概算で足りる。
        var longest = left.Lines.Concat(right.Lines)
            .Select(l => l.Length).DefaultIfEmpty(0).Max();
        _contentWidth = Math.Max(300, Math.Min(longest * 7.6, 20000));
        UpdateTextWidth();

        var stats = comparison.Stats;
        if (keepStatus)
        {
            RebuildVisibleRows();
            return;
        }
        StatusText =
            $"{stats.Rows} 行 / 一致 {stats.IdenticalLines} 行 / "
            + (refining
                ? "意味的な対応付けを計算中…"
                : $"埋め込み {stats.EmbeddedLines} 行 / {elapsed.TotalSeconds:F2} 秒")
            + (!refining && stats.SkippedBlocks > 0
                ? $"    {stats.SkippedBlocks} 箇所は構造的な対応付けのまま"
                : string.Empty);

        // 差分の塊の数。**続いた差分行は 1 つと数える。**
        var blocks = 0;
        var inBlock = false;
        foreach (var row in VisibleRows)
        {
            var differs = !row.Row.IsUnchanged;
            if (differs && !inBlock)
            {
                blocks++;
            }
            inBlock = differs;
        }
        DiffCountText = blocks == 0 ? "差分なし" : $"差分 {blocks} か所";

        // **符号化と改行は左右それぞれの下に出す。** ひとまとめに並べていたが、
        // どちらの話なのかを読み取るのに文字を追う必要があった。
        // 「中身は同じなのに全行差分になる」の原因がここで一目で分かる。
        LeftEncoding = Encodings.FirstOrDefault(e => e.Value == left.Encoding);
        RightEncoding = Encodings.FirstOrDefault(e => e.Value == right.Encoding);
        LeftEnding = TextDecoder.Label(left.LineEnding);
        RightEnding = TextDecoder.Label(right.LineEnding);

        // 「同じに見えるのに一致しない」の原因を、聞かれる前に出す。
        // 符号化と改行を出しているのと同じ理由。
        UpdateInvisibleWarning(left, right);

        ResetFolds();
        RebuildVisibleRows();

        // **比較したら最初の差分を選ぶ。** 開いた直後にやることは、たいてい
        // 「最初の差分を見る」なので、その 1 手を省く。BC も同じ。
        if (!keepStatus && SelectedRowIndex < 0)
        {
            SelectFirstDifference();
        }
    }


    // ---- 差分の反映（ROADMAP 1.1）----

    /// <summary>
    /// 画面の設定から比較の指定を作る。正規表現が不正なら、無視の指定だけを外して
    /// 比較は続ける。**設定の誤りで比較そのものが止まる方が困る。**
    /// </summary>
    private CompareOptions BuildOptions()
    {
        var patterns = IgnoredPatterns
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();
        try
        {
            var importance = new Importance(
                (WhitespaceMode)WhitespaceModeIndex, IgnoreCase, patterns);
            ImportanceError = string.Empty;
            return new CompareOptions(
                (float)PairThreshold, Importance: importance, Language: DetectLanguage(),
                Manual: Manual);
        }
        catch (ArgumentException error)
        {
            ImportanceError = error.Message;
            return new CompareOptions(
                (float)PairThreshold,
                Importance: new Importance((WhitespaceMode)WhitespaceModeIndex, IgnoreCase),
                Language: DetectLanguage(),
                Manual: Manual);
        }
    }

    /// <summary>
    /// 拡張子から言語を決める。左を優先し、分からなければ右を見る。
    /// 拡張子の違うもの同士（例: .cs と .txt）を比べることがあるため。
    /// </summary>
    private Language? DetectLanguage()
        => Lexer.ForPath(LeftPath.Trim()) ?? Lexer.ForPath(RightPath.Trim());

    /// <summary>いま編集中の内容を、読み込んだときの符号化のまま表した形。</summary>
    private DecodedText CurrentLeft => _leftSource! with { Lines = _leftDocument!.Lines };
    private DecodedText CurrentRight => _rightSource! with { Lines = _rightDocument!.Lines };

    // --- 選んだ行をその場で直す（BC の Text Details）---

    private string _detailLeft = string.Empty;

    /// <summary>
    /// 選んだ行の中身（左）。**ここで直せる。**
    ///
    /// 本文の側を編集できるようにすると、仮想化した一覧の中に入力欄を置くことに
    /// なり、行の高さと選択の扱いが一気に難しくなる。BC も同じ理由で、
    /// 編集は下の帯（Text Details）で行う形にしている。
    ///
    /// **全幅で出るので、横に長い行も折り返さずに読める**という利点もある。
    /// </summary>
    /// <summary>
    /// 選んだ行の左。
    ///
    /// **打った内容はその場で本文へ入る。** 以前は「直す」ボタンを
    /// 押させていたが、押し忘れると黙って消えた。欄を離れた時点で
    /// 反映される（画面側で LostFocus にしてある）。
    /// </summary>
    public string DetailLeft
    {
        get => _detailLeft;
        set
        {
            if (Set(ref _detailLeft, value) && !_fillingDetail)
            {
                _ = ApplyDetailAsync();
            }
        }
    }

    private string _detailRight = string.Empty;
    public string DetailRight
    {
        get => _detailRight;
        set
        {
            if (Set(ref _detailRight, value) && !_fillingDetail)
            {
                _ = ApplyDetailAsync();
            }
        }
    }

    /// <summary>
    /// 選んだ行の中身を欄へ入れている最中か。
    ///
    /// **これが無いと、行を選び直しただけで本文を書き換えてしまう。**
    /// 欄への代入と、人が打ったのとを区別する術がこれしかない。
    /// </summary>
    private bool _fillingDetail;

    private int _searchSide;

    /// <summary>
    /// どちら側を探すか。0 = 両方 / 1 = 左だけ / 2 = 右だけ（BC の Sides）。
    /// </summary>
    public int SearchSide
    {
        get => _searchSide;
        set => Set(ref _searchSide, value);
    }

    private bool _searchWrap = true;

    /// <summary>端まで来たら反対側から続けるか（BC の Wrap search）。</summary>
    public bool SearchWrap
    {
        get => _searchWrap;
        set => Set(ref _searchWrap, value);
    }

    /// <summary>
    /// **既定で入れておく。** 「本文を直接直せない道具」だと思われる方が
    /// 損が大きい。切っていると気づける仕掛けが、押していないボタン 1 つ
    /// しか無かった。読み取り専用の側は、これが入っていても触れない。
    /// </summary>
    private bool _fullEdit = true;

    /// <summary>
    /// 本文のペインで直接編集するか（BC の Full Edit）。
    ///
    /// 切っているとき（BC の Line Mode）は、行を選んで下の帯で直す。
    /// 入れると、選んだ行がその場で入力欄になる。
    ///
    /// **選んだ 1 行だけを入力欄にする。** 全行を入力欄にすると、仮想化して
    /// いても数千個を作ることになり、開いた時点で重くなる。
    /// </summary>
    public bool FullEdit
    {
        get => _fullEdit;
        set
        {
            if (Set(ref _fullEdit, value))
            {
                UpdateEditingRow();
            }
        }
    }

    private RowView? _editingRow;

    /// <summary>いま入力欄になっている行を選択に合わせる。</summary>
    private void UpdateEditingRow()
    {
        var target = _fullEdit && _selectedRowIndex >= 0 && _selectedRowIndex < VisibleRows.Count
            ? VisibleRows[_selectedRowIndex]
            : null;

        if (ReferenceEquals(_editingRow, target))
        {
            return;
        }

        // 別の行へ移るときは、直した内容を先に本文へ戻す。
        // **黙って捨てない。** 打った内容が消えるのは一番腹が立つ。
        if (_editingRow is { IsEditing: true })
        {
            _ = CommitRowEditAsync(_editingRow);
            _editingRow.IsEditing = false;
        }

        _editingRow = target;
        if (target is not null)
        {
            target.IsEditing = true;
        }
    }

    /// <summary>本文の中で直した内容を戻す。塊のコピーと同じく履歴に積む。</summary>
    public async Task CommitRowEditAsync(RowView row)
    {
        if (_leftDocument is null || _rightDocument is null)
        {
            return;
        }

        var touchedLeft = row.Row.Left is not null && !LeftReadOnly && row.EditLeft != row.LeftText;
        var touchedRight = row.Row.Right is not null && !RightReadOnly && row.EditRight != row.RightText;
        if (!touchedLeft && !touchedRight)
        {
            return;
        }

        if (touchedLeft)
        {
            _leftDocument.Replace(row.Row.Left!.Value, 1, [row.EditLeft]);
            _undoSides.Push(false);
        }
        if (touchedRight)
        {
            _rightDocument.Replace(row.Row.Right!.Value, 1, [row.EditRight]);
            _undoSides.Push(true);
        }
        _redoSides.Clear();
        await RecompareAsync();
    }

    private bool _overUnder;

    /// <summary>
    /// 上下に並べるか（BC の over-under layout）。
    ///
    /// **横に長い行のときに効く。** 左右に並べると 1 行あたりの幅が半分に
    /// なるので、長い行は横スクロールしないと読めない。上下なら全幅を使える。
    /// </summary>
    public bool OverUnder
    {
        get => _overUnder;
        set
        {
            if (Set(ref _overUnder, value))
            {
                OnPropertyChanged(nameof(RowOrientation));
                OnPropertyChanged(nameof(RowHeight));
                // **幅の基準も変わる。** 上下なら 1 行が全幅を使える。
                UpdateTextWidth();
            }
        }
    }

    /// <summary>行の中で左右を並べる向き。</summary>
    public Avalonia.Layout.Orientation RowOrientation
        => OverUnder ? Avalonia.Layout.Orientation.Vertical
                     : Avalonia.Layout.Orientation.Horizontal;

    /// <summary>
    /// 1 行の高さ。**上下配置では 2 段ぶん要る。**
    /// 固定にしているのは、仮想化のために高さが先に決まっている必要があるため。
    /// </summary>
    public double RowHeight => OverUnder ? 40 : 20;

    private bool _detailAsHex;

    /// <summary>
    /// 下の帯を 16 進で見せるか（BC の Hex Details）。
    ///
    /// **「同じに見えるのに一致しない」の最後の拠り所。** 空白の可視化でも
    /// 分からないとき（結合文字、正規化、見えない制御文字）、バイトを見れば
    /// 必ず分かる。
    /// </summary>
    public bool DetailAsHex
    {
        get => _detailAsHex;
        set
        {
            if (Set(ref _detailAsHex, value))
            {
                OnPropertyChanged(nameof(DetailLeftHex));
                OnPropertyChanged(nameof(DetailRightHex));
                OnPropertyChanged(nameof(DetailShowsText));
                // **同時に出さない。** 下の帯は 1 つで、3 つの見せ方を切り替える。
                if (value)
                {
                    DetailAsAlignment = false;
                }
            }
        }
    }

    private bool _detailAsAlignment;

    /// <summary>
    /// 下の帯を「文字の対応」で見せるか（BC の Alignment Details）。
    ///
    /// **行内差分の色だけでは足りない場面がある。** 片方にしか無い文字が
    /// あると桁がずれ、どの文字がどれに対応しているのかが読み取れない。
    /// </summary>
    public bool DetailAsAlignment
    {
        get => _detailAsAlignment;
        set
        {
            if (Set(ref _detailAsAlignment, value))
            {
                OnPropertyChanged(nameof(DetailAlignment));
                OnPropertyChanged(nameof(DetailShowsText));
                if (value)
                {
                    DetailAsHex = false;
                }
            }
        }
    }

    /// <summary>下の帯が本文（直せる状態）を出しているか。</summary>
    public bool DetailShowsText => !DetailAsHex && !DetailAsAlignment;

    /// <summary>
    /// 選んだ行の、文字の対応（BC の Alignment Details）。
    ///
    /// **左右を上下に並べ、対応する位置を桁で揃える。** 行内差分は色で
    /// 「どこが違うか」を示すが、**片方にしか無い文字があると桁がずれる**ので、
    /// 「この文字がどれに対応しているか」までは読み取れない。
    /// 消えた側に空きを入れて、位置を合わせて見せる。
    ///
    /// 例（`abc` と `axbc`）:
    ///   左  a b c
    ///   右  a x b c
    ///          ^ 増えた
    /// </summary>
    public string DetailAlignment
    {
        get
        {
            if (_detailRow is null || DetailLeft.Length == 0 || DetailRight.Length == 0)
            {
                return string.Empty;
            }

            var (left, right) = InlineDiff.Compute(DetailLeft, DetailRight);

            var top = new System.Text.StringBuilder();
            var bottom = new System.Text.StringBuilder();
            var marks = new System.Text.StringBuilder();

            var li = 0;
            var ri = 0;

            // **両側の範囲を同時に進める。** 片方ずつ書くと桁が合わない。
            while (li < left.Count || ri < right.Count)
            {
                var l = li < left.Count ? left[li] : default;
                var r = ri < right.Count ? right[ri] : default;

                var hasL = li < left.Count;
                var hasR = ri < right.Count;

                // 両方が「同じ」なら、そのまま並べる。
                if (hasL && hasR && l.Kind == SpanKind.Equal && r.Kind == SpanKind.Equal)
                {
                    var text = DetailLeft.Substring(l.Start, l.Length);
                    Append(top, bottom, marks, text, text, same: true);
                    li++;
                    ri++;
                    continue;
                }

                // 片側だけ「同じ」なら、もう片方の変更を先に出す（増えた／消えた）。
                if (hasL && l.Kind != SpanKind.Equal)
                {
                    var text = DetailLeft.Substring(l.Start, l.Length);
                    Append(top, bottom, marks, text, new string(' ', text.Length), same: false);
                    li++;
                    continue;
                }
                if (hasR && r.Kind != SpanKind.Equal)
                {
                    var text = DetailRight.Substring(r.Start, r.Length);
                    Append(top, bottom, marks, new string(' ', text.Length), text, same: false);
                    ri++;
                    continue;
                }

                // ここへ来るのは片側が尽きたとき。残りをそのまま出す。
                if (hasL)
                {
                    var text = DetailLeft.Substring(l.Start, l.Length);
                    Append(top, bottom, marks, text, new string(' ', text.Length), same: false);
                    li++;
                }
                else if (hasR)
                {
                    var text = DetailRight.Substring(r.Start, r.Length);
                    Append(top, bottom, marks, new string(' ', text.Length), text, same: false);
                    ri++;
                }
            }

            return $"左 {top}{Environment.NewLine}   {marks}{Environment.NewLine}右 {bottom}";
        }
    }

    /// <summary>
    /// 対応の 1 区間を書き足す。
    ///
    /// **印は違うところにだけ付ける。** 全部に付けると、どこを見ればいいのか
    /// 分からなくなる。
    /// </summary>
    private static void Append(
        System.Text.StringBuilder top, System.Text.StringBuilder bottom,
        System.Text.StringBuilder marks, string left, string right, bool same)
    {
        var width = Math.Max(left.Length, right.Length);
        top.Append(left.PadRight(width));
        bottom.Append(right.PadRight(width));
        marks.Append(same ? new string(' ', width) : new string('^', width));
    }

    /// <summary>
    /// 選んだ行のバイト列。**元の符号化で出す。**
    /// UTF-8 に直してから見せると、Shift_JIS のファイルで実際のバイトと違う
    /// ものを見せることになり、この機能の意味が無くなる。
    /// </summary>
    public string DetailLeftHex => HexOf(DetailLeft, _leftSource);
    public string DetailRightHex => HexOf(DetailRight, _rightSource);

    private static string HexOf(string line, DecodedText? source)
    {
        if (line.Length == 0)
        {
            return string.Empty;
        }
        try
        {
            // 改行は付けない（行の中身だけを見たい）。
            var bytes = source is null
                ? System.Text.Encoding.UTF8.GetBytes(line)
                : TextEncoder.Encode([line], source.Encoding, LineEnding.None, endsWithNewline: false);

            var text = new System.Text.StringBuilder(bytes.Length * 3);
            for (var i = 0; i < bytes.Length; i++)
            {
                if (i > 0)
                {
                    text.Append(i % 8 == 0 ? "  " : " ");
                }
                text.Append(bytes[i].ToString("X2"));
            }
            return text.ToString();
        }
        catch (Exception)
        {
            // その符号化で表せない文字があるとき。比較そのものは続けたい。
            return "（この符号化では表せない文字があります）";
        }
    }

    private bool _showDetails = true;

    /// <summary>下の帯を出すか。狭い画面では畳みたいことがある。</summary>
    /// <summary>下の帯を閉じる／開く。**帯の上と状態バーの両方から。**</summary>
    public RelayCommand HideDetailsCommand { get; }
    public RelayCommand ShowDetailsCommand { get; }

    public bool ShowDetails
    {
        get => _showDetails;
        set => Set(ref _showDetails, value);
    }

    public string DetailLeftLabel => _detailRow is null
        ? "左（行を選ぶと、ここで直せます）"
        : _detailRow.LeftNumber is { Length: > 0 } l
            ? $"左 {l} 行目"
            : "左（この行に対応する行はありません）";

    public string DetailRightLabel => _detailRow is null
        ? "右"
        : _detailRow.RightNumber is { Length: > 0 } r
            ? $"右 {r} 行目"
            : "右（この行に対応する行はありません）";

    /// <summary>その側に行があるときだけ直せる。無い行は作れない（塊のコピーで足す）。</summary>
    public bool CanEditLeftDetail => _detailRow?.Row.Left is not null && !LeftReadOnly;
    public bool CanEditRightDetail => _detailRow?.Row.Right is not null && !RightReadOnly;

    private RowView? _detailRow;

    /// <summary>選んだ行を下の帯へ移す。</summary>
    /// <summary>
    /// 片側だけを並べる。
    ///
    /// **比較ではないので類似度は出さない。** 出すと「相手と比べた結果」に
    /// 見えるが、相手はまだ無い。行番号と本文だけを見せて、
    /// もう片方を指定すれば比べ始める、という状態にする。
    /// </summary>
    private async Task ShowOneSideAsync()
    {
        var left = !string.IsNullOrWhiteSpace(LeftPath);
        var path = (left ? LeftPath : RightPath).Trim();

        IsBusy = true;
        try
        {
            var text = await Task.Run(() =>
            {
                var load = ContentLoader ?? File.ReadAllBytes;
                return TextDecoder.Decode(load(path));
            });

            // 片側だけの「比較」を組む。相手は空。
            var empty = new DecodedText([], text.Encoding, text.LineEnding);
            var comparison = DiffComparer.Compare(
                left ? text : empty, left ? empty : text, embedder: null, BuildOptions());

            Apply(left ? text : empty, left ? empty : text, comparison, TimeSpan.Zero, refining: false);
            StatusText = $"{Path.GetFileName(path)} だけを開いています"
                + $"（{text.Lines.Count} 行）。"
                + "もう片方を指定すると比べ始めます。";
            Placeholder = string.Empty;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException
                                        or NotSupportedException)
        {
            Placeholder = ReadableError(error, path);
        }
        finally
        {
            IsBusy = false;
        }
    }

    /// <summary>
    /// 読めなかった理由を、何をすればいいか分かる形にする。
    ///
    /// **元の文言をそのまま出さない。** 「Access to the path ... is denied」は
    /// 英語なうえ、フォルダーを渡したのか権限が無いのかが読み取れない。
    /// </summary>
    private static string ReadableError(Exception error, string path)
    {
        var name = Path.GetFileName(path.TrimEnd('/', '\\'));
        return error switch
        {
            UnauthorizedAccessException when Directory.Exists(path)
                => $"{name} はフォルダーです。フォルダーどうしを比べるなら、"
                   + "両方にフォルダーを指定してください。",
            UnauthorizedAccessException
                => $"{name} を読む権限がありません。",
            FileNotFoundException or DirectoryNotFoundException
                => $"{name} が見つかりません。",
            _ => $"{name} を読めません: {error.Message}",
        };
    }

    private void ShowDetailFor(RowView? row)
    {
        _detailRow = row;
        _fillingDetail = true;
        DetailLeft = row?.LeftText ?? string.Empty;
        DetailRight = row?.RightText ?? string.Empty;
        _fillingDetail = false;
        OnPropertyChanged(nameof(DetailLeftHex));
        OnPropertyChanged(nameof(DetailRightHex));
        // **行を選び直したら対応も測り直す。** 前の行の対応が残ると、
        // いま選んでいる行のものだと読まれる。
        OnPropertyChanged(nameof(DetailAlignment));
        OnPropertyChanged(nameof(DetailLeftLabel));
        OnPropertyChanged(nameof(DetailRightLabel));
        OnPropertyChanged(nameof(CanEditLeftDetail));
        OnPropertyChanged(nameof(CanEditRightDetail));
    }

    /// <summary>
    /// 下の帯で直した内容を本文へ戻す。
    ///
    /// **1 行の置き換えとして履歴に積む。** 塊のコピーと同じ扱いにするので、
    /// 取り消しも同じ操作で戻せる。
    /// </summary>
    private async Task ApplyDetailAsync()
    {
        if (_detailRow is not { } row || _leftDocument is null || _rightDocument is null)
        {
            return;
        }

        var changed = false;
        if (row.Row.Left is { } leftIndex && !LeftReadOnly && DetailLeft != row.LeftText)
        {
            _leftDocument.Replace(leftIndex, 1, [DetailLeft]);
            changed = true;
        }
        if (row.Row.Right is { } rightIndex && !RightReadOnly && DetailRight != row.RightText)
        {
            _rightDocument.Replace(rightIndex, 1, [DetailRight]);
            changed = true;
        }

        if (!changed)
        {
            return;
        }

        // どちら側を触ったかは取り消しのときに要る。両方直したなら 2 回積む。
        if (row.Row.Left is not null && DetailLeft != row.LeftText)
        {
            _undoSides.Push(false);
        }
        if (row.Row.Right is not null && DetailRight != row.RightText)
        {
            _undoSides.Push(true);
        }
        _redoSides.Clear();
        await RecompareAsync();
    }

    private async Task ApplyBlockAsync(RowView row, bool toRight)
    {
        if (_leftDocument is null || _rightDocument is null
            || row.BlockIndex < 0 || row.BlockIndex >= _blocks.Count)
        {
            return;
        }

        var block = _blocks[row.BlockIndex];
        if (toRight)
        {
            _rightDocument.Replace(
                block.RightStart, block.RightCount,
                Slice(_leftDocument.Lines, block.LeftStart, block.LeftCount));
        }
        else
        {
            _leftDocument.Replace(
                block.LeftStart, block.LeftCount,
                Slice(_rightDocument.Lines, block.RightStart, block.RightCount));
        }

        _undoSides.Push(toRight);
        // 新しい操作をした時点で、やり直せる分は無くなる。
        _redoSides.Clear();
        await RecompareAsync();
    }

    private async Task UndoAsync()
    {
        if (_undoSides.Count == 0)
        {
            return;
        }
        var side = _undoSides.Pop();
        (side ? _rightDocument : _leftDocument)!.Undo();
        _redoSides.Push(side);
        await RecompareAsync();
    }

    private async Task RedoAsync()
    {
        if (_redoSides.Count == 0)
        {
            return;
        }
        var side = _redoSides.Pop();
        (side ? _rightDocument : _leftDocument)!.Redo();
        _undoSides.Push(side);
        await RecompareAsync();
    }

    private async Task SaveAsync(bool left)
    {
        var document = left ? _leftDocument : _rightDocument;
        var source = left ? _leftSource : _rightSource;
        var path = left ? LeftPath.Trim() : RightPath.Trim();
        if (document is null || source is null || string.IsNullOrEmpty(path))
        {
            return;
        }

        try
        {
            var handler = left ? LeftSaveHandler : RightSaveHandler;
            if (handler is not null)
            {
                await handler(document.Lines, source);
            }
            else
            {
                var bytes = TextEncoder.Encode(document.Lines, source);
                await File.WriteAllBytesAsync(path, bytes);
            }
            document.MarkSaved();
            RaiseEditState();
            StatusText = $"{path} を保存した（{document.Lines.Count} 行）";
        }
        catch (Exception error)
        {
            // 保存の失敗は黙って流せない。書けたつもりで作業を続ける方が損害が大きい。
            StatusText = $"保存できない: {error.Message}";
        }
    }

    private static List<string> Slice(IReadOnlyList<string> lines, int start, int count)
    {
        var result = new List<string>(count);
        for (var i = 0; i < count; i++)
        {
            result.Add(lines[start + i]);
        }
        return result;
    }

    /// <summary>
    /// 編集後に比較をやり直す。読み込みからではなく、いまの編集内容から組み直す。
    /// 段階 1 を即座に描いてから、段階 2 を背景で当てるのは初回と同じ。
    /// </summary>
    private async Task RecompareAsync()
    {
        if (_leftDocument is null || _rightDocument is null)
        {
            return;
        }

        var generation = ++_compareGeneration;
        var compareOptions = BuildOptions();
        var left = CurrentLeft;
        var right = CurrentRight;
        RaiseEditState();

        var first = await Task.Run(() =>
        {
            var started = DateTime.UtcNow;
            var comparison = DiffComparer.Compare(
                left, right, embedder: null, compareOptions);
            return (comparison, elapsed: DateTime.UtcNow - started);
        });

        if (generation != _compareGeneration)
        {
            return;
        }
        Apply(left, right, first.comparison, first.elapsed, refining: true);

        var refined = await Task.Run(() =>
        {
            var embedder = _shell.GetEmbedder();
            var started = DateTime.UtcNow;
            var comparison = DiffComparer.Compare(
                left, right, embedder, compareOptions);
            return (comparison, elapsed: DateTime.UtcNow - started);
        });

        if (generation != _compareGeneration)
        {
            return;
        }
        Apply(left, right, refined.comparison, refined.elapsed, refining: false);
    }

    /// <summary>比較結果を書き出す。</summary>
    private async Task ExportAsync(bool unified)
    {
        if (_comparison is null || _leftSource is null || _rightSource is null)
        {
            StatusText = "先に比較してください。";
            return;
        }

        var suggested = Path.GetFileNameWithoutExtension(LeftPath.Trim())
            + (unified ? ".diff" : ".html");
        if (await _shell.PickSavePath(unified ? "unified 差分を保存" : "HTML を保存", suggested)
            is not { } path)
        {
            return;
        }

        try
        {
            var left = CurrentLeft;
            var right = CurrentRight;
            var content = unified
                ? Report.UnifiedDiff(_comparison, left, right, LeftPath.Trim(), RightPath.Trim())
                : Report.Html(_comparison, left, right, LeftPath.Trim(), RightPath.Trim());

            if (unified && content.Length == 0)
            {
                StatusText = "差異が無いので、書き出す差分がありません。";
                return;
            }

            // 書き出しは常に UTF-8。patch も閲覧器もそれを前提にしている。
            await File.WriteAllTextAsync(path, content, new System.Text.UTF8Encoding(false));
            StatusText = $"{path} へ書き出した";
        }
        catch (Exception error)
        {
            StatusText = $"書き出せない: {error.Message}";
        }
    }

    /// <summary>編集にまつわる表示とボタンの有効・無効を更新する。</summary>
    private void RaiseEditState()
    {
        OnPropertyChanged(nameof(LeftModified));
        OnPropertyChanged(nameof(RightModified));
        OnPropertyChanged(nameof(HasUnsavedChanges));
        UndoCommand.Raise();
        RedoCommand.Raise();
        SaveLeftCommand.Raise();
        SaveRightCommand.Raise();
    }

    /// <summary>
    /// 絞り込みの結果を作り直す。結果が届いたときと、絞り込みを切り替えたときだけ。
    /// 毎フレーム絞り直すと、行数に比例した処理が描画のたびに走る。
    /// </summary>

    // ---- 差分間の移動と検索（ROADMAP 1.2）----

    /// <summary>
    /// 次（前）の差分の先頭へ選択を移す。表示している行の中だけを見るので、
    /// 「変更のある行だけ表示」で畳んでいるときも辻褄が合う。
    /// </summary>
    /// <summary>
    /// この向きにまだ差分があるか。
    ///
    /// **端で止まるのではなく回り込む**ので、差分が 1 つでもあれば真。
    /// 差分が 1 つも無いときだけ偽。押しても何も起きないボタンを
    /// 押せる状態で置かないため（BC は矢印の色を変えて示す）。
    /// </summary>
    public bool HasDifferences => VisibleRows.Any(r => r.IsBlockStart);

    private void MoveToDifference(bool forward)
    {
        if (VisibleRows.Count == 0)
        {
            return;
        }

        var starts = new List<int>();
        for (var i = 0; i < VisibleRows.Count; i++)
        {
            if (VisibleRows[i].IsBlockStart)
            {
                starts.Add(i);
            }
        }
        if (starts.Count == 0)
        {
            SearchStatus = "差分なし";
            return;
        }

        var current = SelectedRowIndex;
        int target;
        if (forward)
        {
            // 末尾まで行ったら先頭へ回る。端で止まると「壊れた」ように見える。
            target = starts.FirstOrDefault(i => i > current, starts[0]);
        }
        else
        {
            target = starts.LastOrDefault(i => i < current, starts[^1]);
        }

        SelectedRowIndex = target;
        var order = starts.IndexOf(target) + 1;
        SearchStatus = $"差分 {order}/{starts.Count}";
    }

    /// <summary>
    /// 検索。左右どちらの本文に当たっても、その行へ移る。
    /// 片側だけを対象にすると「見えているのに飛べない」ことになる。
    /// </summary>
    private void FindFrom(bool forward)
    {
        if (string.IsNullOrEmpty(SearchText) || VisibleRows.Count == 0)
        {
            SearchStatus = string.Empty;
            return;
        }

        var query = new SearchQuery(SearchText, SearchUseRegex, SearchMatchCase, SearchWholeWord);
        List<int> matches;
        try
        {
            matches = MatchingRows(query);
        }
        catch (ArgumentException error)
        {
            SearchStatus = error.Message;
            return;
        }

        if (matches.Count == 0)
        {
            SearchStatus = "見つからない";
            return;
        }

        var current = SelectedRowIndex;
        int target;
        if (forward)
        {
            var next = matches.FirstOrDefault(i => i > current, -1);
            if (next < 0)
            {
                // 端まで来た。**回り込むかどうかは選ばせる。**
                // 黙って先頭へ戻ると、2 周目に入ったことに気づかず
                // 同じ場所を何度も見ることになる。
                if (!SearchWrap)
                {
                    SearchStatus = $"{matches.Count} 件（末尾）";
                    return;
                }
                next = matches[0];
            }
            target = next;
        }
        else
        {
            var previous = matches.LastOrDefault(i => i < current, -1);
            if (previous < 0)
            {
                if (!SearchWrap)
                {
                    SearchStatus = $"{matches.Count} 件（先頭）";
                    return;
                }
                previous = matches[^1];
            }
            target = previous;
        }

        SelectedRowIndex = target;
        SearchStatus = $"{matches.IndexOf(target) + 1}/{matches.Count} 件目";
    }

    private List<int> MatchingRows(SearchQuery query)
    {
        var matches = new List<int>();
        for (var i = 0; i < VisibleRows.Count; i++)
        {
            var row = VisibleRows[i];

            // どちら側を探すか。**片側だけ探せると効く場面がある。**
            // 「右に入ったはずの文字列が本当に入っているか」を見たいとき、
            // 両側を探すと左の一致で止まってしまう。
            List<string> haystack = SearchSide switch
            {
                1 => [row.LeftText],
                2 => [row.RightText],
                _ => [row.LeftText, row.RightText],
            };

            if (TextSearch.Find(haystack, query).Count > 0)
            {
                matches.Add(i);
            }
        }
        return matches;
    }

    /// <summary>
    /// 変更から <see cref="ContextLines"/> 行以内にあるか。畳んだときに前後の文脈を
    /// 残すために使う。文脈が無いと、差分だけ並んでも何の中の変更か分からない。
    /// </summary>
    private bool NearAChange(int index)
    {
        if (_comparison is null || ContextLines <= 0)
        {
            return false;
        }
        var context = (int)ContextLines;
        var from = Math.Max(0, index - context);
        var to = Math.Min(_comparison.Rows.Count - 1, index + context);
        for (var i = from; i <= to; i++)
        {
            if (!_comparison.Rows[i].IsUnchanged)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>最初の差分の行を選ぶ。差分が無ければ先頭。</summary>
    private void SelectFirstDifference()
    {
        for (var i = 0; i < VisibleRows.Count; i++)
        {
            if (!VisibleRows[i].Row.IsUnchanged)
            {
                SelectedRowIndex = i;
                return;
            }
        }
        SelectedRowIndex = VisibleRows.Count > 0 ? 0 : -1;
    }

    /// <summary>
    /// いま畳んでいる行（元の行の添字）。
    ///
    /// **1 つの集合で持つ。** 「絞り込みで畳んだもの」と「手で畳んだもの」を
    /// 別々に持つと、両方に入っている行の扱いが決まらず、開いたのに閉じたままに
    /// 見える、といった食い違いが必ず出る。
    /// </summary>
    private readonly HashSet<int> _folded = [];

    /// <summary>絞り込みの切り替えに合わせて、畳む所を作り直す。</summary>
    private void ResetFolds()
    {
        _folded.Clear();
        if (!ChangesOnly || _comparison is null)
        {
            return;
        }
        for (var i = 0; i < _comparison.Rows.Count; i++)
        {
            if (_comparison.Rows[i].IsUnchanged && !NearAChange(i))
            {
                _folded.Add(i);
            }
        }
    }

    private void AddBand(int start, int count)
    {
        if (_comparison is null || count <= 0)
        {
            return;
        }
        VisibleRows.Add(RowView.Band(
            _comparison.Rows[start], _leftSource!, _rightSource!, start, count));
    }

    /// <summary>帯を押したら、その範囲を開く。</summary>
    public void ExpandFold(RowView band)
    {
        if (!band.IsFoldBand)
        {
            return;
        }
        for (var i = band.FoldStart; i < band.FoldStart + band.FoldedCount; i++)
        {
            _folded.Remove(i);
        }
        RebuildVisibleRows();
    }

    /// <summary>アウトラインの箱を押したら、その範囲を畳む。</summary>
    public void CollapseOutline(RowView row)
    {
        if (!row.IsOutlineHead)
        {
            return;
        }
        for (var i = row.OutlineStart; i < row.OutlineStart + row.OutlineCount; i++)
        {
            _folded.Add(i);
        }
        RebuildVisibleRows();
    }

    /// <summary>
    /// 畳める範囲に縦線を付ける（Excel のアウトラインと同じ考え方）。
    ///
    /// **開いた状態でも「ここは畳める」と分かる**ようにするのが目的。
    /// 帯は畳んだ場所しか示さないので、開いている間は範囲が見えなくなる。
    ///
    /// 一致した行が続く所を 1 つの範囲とする。**2 行以下は線を出さない。**
    /// 1 行畳んでも得るものが無く、線だけが増えて読みにくくなる。
    /// </summary>
    private void MarkOutlines()
    {
        const int minimum = 3;

        var start = -1;
        for (var i = 0; i <= VisibleRows.Count; i++)
        {
            var foldable = i < VisibleRows.Count
                && !VisibleRows[i].IsFoldBand
                && VisibleRows[i].Row.IsUnchanged;

            if (foldable)
            {
                if (start < 0)
                {
                    start = i;
                }
                continue;
            }

            if (start >= 0 && i - start >= minimum)
            {
                // 畳むときに要るのは元の行の添字。表示の添字とは違う。
                var origin = OriginOf(VisibleRows[start]);
                var count = i - start;
                for (var k = start; k < i; k++)
                {
                    VisibleRows[k].Outline = k == start ? OutlineMark.Head
                        : k == i - 1 ? OutlineMark.Tail
                        : OutlineMark.Body;
                    VisibleRows[k].OutlineStart = origin;
                    VisibleRows[k].OutlineCount = count;
                }
            }
            else if (start >= 0)
            {
                for (var k = start; k < i; k++)
                {
                    VisibleRows[k].Outline = OutlineMark.None;
                }
            }
            start = -1;
        }
    }

    /// <summary>
    /// 差分の塊の範囲に線を付ける。
    ///
    /// **矢印がどこまでを写すのかを見せる。** 塊の先頭にしか矢印が出ないので、
    /// 1 行だけなのか 20 行なのかが、押すまで分からなかった。折りたたみと
    /// 同じ形にして、線の意味を覚え直さなくて済むようにする。
    ///
    /// **1 行だけの塊には線を出さない。** 矢印だけで範囲は明らかで、
    /// 線を足しても情報が増えない。
    /// </summary>
    private void MarkBlockRanges()
    {
        var start = -1;
        var block = -1;

        for (var i = 0; i <= VisibleRows.Count; i++)
        {
            var current = i < VisibleRows.Count && !VisibleRows[i].IsFoldBand
                ? VisibleRows[i].BlockIndex
                : -1;

            if (current >= 0 && current == block)
            {
                continue;
            }

            // 塊が切り替わった。前の塊に線を引く。
            if (start >= 0 && block >= 0)
            {
                var length = i - start;
                for (var k = start; k < i; k++)
                {
                    VisibleRows[k].BlockOutline = length < 2 ? OutlineMark.None
                        : k == start ? OutlineMark.Head
                        : k == i - 1 ? OutlineMark.Tail
                        : OutlineMark.Body;
                }
            }

            block = current;
            start = current >= 0 ? i : -1;
        }
    }

    /// <summary>表示している行が、元の並びで何番目か。</summary>
    private int OriginOf(RowView row) => _allRows.IndexOf(row);

    private void RebuildVisibleRows()
    {
        VisibleRows.Clear();
        if (_comparison is null)
        {
            OnPropertyChanged(nameof(ShowPlaceholder));
            return;
        }
        // 畳んだ行は帯にまとめて出す。**隠したことを黙っていない。**
        // どれだけ消えたのか分からないと、見落としたのか元から無いのかを
        // 区別できない。
        var foldStart = -1;

        for (var i = 0; i < _allRows.Count; i++)
        {
            if (!_folded.Contains(i))
            {
                if (foldStart >= 0)
                {
                    AddBand(foldStart, i - foldStart);
                    foldStart = -1;
                }
                VisibleRows.Add(_allRows[i]);
            }
            else if (foldStart < 0)
            {
                foldStart = i;
            }
        }
        if (foldStart >= 0)
        {
            AddBand(foldStart, _allRows.Count - foldStart);
        }

        MarkOutlines();
        MarkBlockRanges();
        OnPropertyChanged(nameof(ShowPlaceholder));
        OnPropertyChanged(nameof(SearchStatus));
        OnPropertyChanged(nameof(HasDifferences));
        NextDifferenceCommand.Raise();
        PreviousDifferenceCommand.Raise();
    }
}
