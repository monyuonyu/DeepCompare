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

    private string _leftPath = string.Empty;
    private string _rightPath = string.Empty;
    private string _statusText = string.Empty;
    private string _placeholder = "比較する 2 つのファイルを指定してください（ドラッグ＆ドロップ可）";
    private bool _isBusy;
    private bool _changesOnly;
    private double _pairThreshold = Aligner.DefaultPairThreshold;
    private double _textWidth = 460;

    public TextCompareViewModel(ShellViewModel shell, object? backTarget)
    {
        _shell = shell;
        _backTarget = backTarget;
        BrowseLeftCommand = new RelayCommand(() => PickAsync(left: true));
        BrowseRightCommand = new RelayCommand(() => PickAsync(left: false));
        CompareCommand = new RelayCommand(RunCompareAsync);
        BackCommand = new RelayCommand(() => { _shell.GoBack(_backTarget); return Task.CompletedTask; });
    }

    public ObservableCollection<RowView> VisibleRows { get; } = [];
    public ICommand BrowseLeftCommand { get; }
    public ICommand BrowseRightCommand { get; }
    public ICommand CompareCommand { get; }
    public ICommand BackCommand { get; }

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

    /// <summary>本文列の幅。最長行に合わせて決め、横スクロールできるようにする。</summary>
    public double TextWidth
    {
        get => _textWidth;
        private set => Set(ref _textWidth, value);
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

        IsBusy = true;
        StatusText = "解析しています…";
        var leftPath = LeftPath.Trim();
        var rightPath = RightPath.Trim();
        var threshold = (float)PairThreshold;

        try
        {
            var result = await Task.Run(() =>
            {
                // モデルは殻が保持していて、初回だけ読む。読み込みは数秒かかる。
                var embedder = _shell.GetEmbedder();
                var left = TextDecoder.Decode(File.ReadAllBytes(leftPath));
                var right = TextDecoder.Decode(File.ReadAllBytes(rightPath));
                var started = DateTime.UtcNow;
                var comparison = DiffComparer.Compare(
                    left, right, embedder, new CompareOptions(threshold));
                return (left, right, comparison, elapsed: DateTime.UtcNow - started);
            });

            _comparison = result.comparison;
            _allRows = result.comparison.Rows
                .Select(r => new RowView(r, result.left, result.right)).ToList();

            // 最長行に合わせて本文列の幅を決める。等幅なので概算で足りる。
            var longest = result.left.Lines.Concat(result.right.Lines)
                .Select(l => l.Length).DefaultIfEmpty(0).Max();
            TextWidth = Math.Max(300, Math.Min(longest * 7.6, 20000));

            var stats = result.comparison.Stats;
            StatusText =
                $"{stats.Rows} 行 / 一致 {stats.IdenticalLines} 行 / 埋め込み {stats.EmbeddedLines} 行 / "
                + $"{result.elapsed.TotalSeconds:F2} 秒    "
                // 符号化と改行コードを出すのは、「中身は同じなのに全行差分になる」
                // という混乱の原因がここで一目で分かるから。
                + $"左: {TextDecoder.Label(result.left.Encoding)} / {TextDecoder.Label(result.left.LineEnding)}    "
                + $"右: {TextDecoder.Label(result.right.Encoding)} / {TextDecoder.Label(result.right.LineEnding)}"
                + (stats.SkippedBlocks > 0
                    ? $"    {stats.SkippedBlocks} 箇所は大きすぎるため意味的な対応付けを省略"
                    : string.Empty);

            RebuildVisibleRows();
        }
        catch (Exception error)
        {
            _allRows = [];
            VisibleRows.Clear();
            Placeholder = $"エラー: {error.Message}";
            StatusText = string.Empty;
            OnPropertyChanged(nameof(ShowPlaceholder));
        }
        finally
        {
            IsBusy = false;
        }
    }

    /// <summary>
    /// 絞り込みの結果を作り直す。結果が届いたときと、絞り込みを切り替えたときだけ。
    /// 毎フレーム絞り直すと、行数に比例した処理が描画のたびに走る。
    /// </summary>
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
            if (!ChangesOnly || !_comparison.Rows[i].IsUnchanged)
            {
                VisibleRows.Add(_allRows[i]);
            }
        }
        OnPropertyChanged(nameof(ShowPlaceholder));
    }
}
