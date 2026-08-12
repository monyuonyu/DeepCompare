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
    private readonly object? _backTarget;
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

    public TextCompareViewModel(ShellViewModel shell, object? backTarget)
    {
        _shell = shell;
        _backTarget = backTarget;
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
        NextDifferenceCommand = new RelayCommand(() => { MoveToDifference(forward: true); return Task.CompletedTask; });
        PreviousDifferenceCommand = new RelayCommand(() => { MoveToDifference(forward: false); return Task.CompletedTask; });
        FindNextCommand = new RelayCommand(() => { FindFrom(forward: true); return Task.CompletedTask; });
        FindPreviousCommand = new RelayCommand(() => { FindFrom(forward: false); return Task.CompletedTask; });
        BackCommand = new RelayCommand(() => { _shell.GoBack(_backTarget); return Task.CompletedTask; });
    }

    public ObservableCollection<RowView> VisibleRows { get; } = [];
    public ICommand BrowseLeftCommand { get; }
    public ICommand BrowseRightCommand { get; }
    public ICommand CompareCommand { get; }
    public ICommand BackCommand { get; }

    /// <summary>テーマの切り替えなど、画面をまたぐ操作。</summary>
    public ShellViewModel Shell => _shell;
    public RelayCommand<RowView> CopyToRightCommand { get; }
    public RelayCommand<RowView> CopyToLeftCommand { get; }
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
    public ICommand NextDifferenceCommand { get; }
    public ICommand PreviousDifferenceCommand { get; }
    public ICommand FindNextCommand { get; }
    public ICommand FindPreviousCommand { get; }

    /// <summary>一覧で選んでいる行。差分間の移動と検索はここを動かす。</summary>
    public int SelectedRowIndex
    {
        get => _selectedRowIndex;
        set => Set(ref _selectedRowIndex, value);
    }

    /// <summary>畳んだときに変更の前後へ残す行数。</summary>
    public double ContextLines
    {
        get => _contextLines;
        set
        {
            if (Set(ref _contextLines, value))
            {
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
    public string BackText => _backTarget is FolderCompareViewModel ? "一覧へ戻る" : "最初の画面へ";

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

    /// <summary>本文以外の列（行番号 3 つ・コピーボタン・移動の印）が使う幅。</summary>
    private const double GutterWidth = 52 * 3 + 40 + 52;

    private void UpdateTextWidth()
    {
        // 画面幅が分からないうちは内容だけで決める。起動直後に一度だけ通る。
        var share = _viewportWidth > 0
            ? Math.Max(300, (_viewportWidth - GutterWidth) / 2)
            : 0;
        TextWidth = Math.Max(_contentWidth, share);
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
        if (string.IsNullOrWhiteSpace(LeftPath) || string.IsNullOrWhiteSpace(RightPath))
        {
            Placeholder = "両方のファイルを指定してください。";
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
                var left = TextDecoder.Decode(load(leftPath));
                var right = TextDecoder.Decode(load(rightPath));
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

            // 段階 2: 埋め込みで対応付けを取り直し、差し替える。
            var refinedResult = await Task.Run(() =>
            {
                // モデルは殻が保持していて、初回だけ読む。
                var embedder = _shell.GetEmbedder();
                var started = DateTime.UtcNow;
                var comparison = DiffComparer.Compare(
                    first.left, first.right, embedder, compareOptions);
                return (comparison, elapsed: DateTime.UtcNow - started);
            });

            if (generation != _compareGeneration)
            {
                return;
            }

            Apply(first.left, first.right, refinedResult.comparison, refinedResult.elapsed, refining: false);
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
                ? "意味的な対応付けを計算中…    "
                : $"埋め込み {stats.EmbeddedLines} 行 / {elapsed.TotalSeconds:F2} 秒    ")
            // 符号化と改行コードを出すのは、「中身は同じなのに全行差分になる」
            // という混乱の原因がここで一目で分かるから。
            + $"左: {TextDecoder.Label(left.Encoding)} / {TextDecoder.Label(left.LineEnding)}    "
            + $"右: {TextDecoder.Label(right.Encoding)} / {TextDecoder.Label(right.LineEnding)}"
            + (!refining && stats.SkippedBlocks > 0
                ? $"    {stats.SkippedBlocks} 箇所は構造的な対応付けのまま"
                : string.Empty);

        // 「同じに見えるのに一致しない」の原因を、聞かれる前に出す。
        // 符号化と改行を出しているのと同じ理由。
        UpdateInvisibleWarning(left, right);

        RebuildVisibleRows();
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
                (float)PairThreshold, Importance: importance, Language: DetectLanguage());
        }
        catch (ArgumentException error)
        {
            ImportanceError = error.Message;
            return new CompareOptions(
                (float)PairThreshold,
                Importance: new Importance((WhitespaceMode)WhitespaceModeIndex, IgnoreCase),
                Language: DetectLanguage());
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
            var bytes = TextEncoder.Encode(document.Lines, source);
            await File.WriteAllBytesAsync(path, bytes);
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
        var target = forward
            ? matches.FirstOrDefault(i => i > current, matches[0])
            : matches.LastOrDefault(i => i < current, matches[^1]);

        SelectedRowIndex = target;
        SearchStatus = $"{matches.IndexOf(target) + 1}/{matches.Count} 件目";
    }

    private List<int> MatchingRows(SearchQuery query)
    {
        var matches = new List<int>();
        for (var i = 0; i < VisibleRows.Count; i++)
        {
            var row = VisibleRows[i];
            if (TextSearch.Find([row.LeftText, row.RightText], query).Count > 0)
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

    private void RebuildVisibleRows()
    {
        VisibleRows.Clear();
        if (_comparison is null)
        {
            OnPropertyChanged(nameof(ShowPlaceholder));
            return;
        }
        for (var i = 0; i < _allRows.Count; i++)
        {
            if (!ChangesOnly || !_comparison.Rows[i].IsUnchanged || NearAChange(i))
            {
                VisibleRows.Add(_allRows[i]);
            }
        }
        OnPropertyChanged(nameof(ShowPlaceholder));
        OnPropertyChanged(nameof(SearchStatus));
    }
}
