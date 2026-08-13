using System.Collections.ObjectModel;
using System.Windows.Input;
using Avalonia;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>フォルダー一覧の 1 行。</summary>
public sealed class FolderRowView(FolderEntry entry) : ViewModelBase
{

    public FolderEntry Entry { get; } = entry;

    /// <summary>階層を字下げで表す。</summary>
    public Thickness Indent { get; } = new(entry.Depth * 14, 0, 0, 0);

    /// <summary>下に何か入っているか。走査した後に決まる。</summary>
    public bool HasChildren { get; set; }

    private bool _isExpanded;

    /// <summary>
    /// 開いているか。**既定は閉じる。**
    ///
    /// 数千ファイルの木を最初から全部展開すると、目的の場所へ辿り着く前に
    /// 一覧が流れてしまう。Beyond Compare も既定では閉じた状態で出す。
    /// </summary>
    public bool IsExpanded
    {
        get => _isExpanded;
        set
        {
            if (Set(ref _isExpanded, value))
            {
                OnPropertyChanged(nameof(ToggleIcon));
            }
        }
    }

    /// <summary>開閉の三角。子が無ければ出さない。</summary>
    public Avalonia.Media.Geometry? ToggleIcon => HasChildren
        ? Icons.Get(_isExpanded ? "IconChevronDown" : "IconChevronRight")
        : null;

    public string Name { get; } = entry.IsDirectory ? entry.Name + "/" : entry.Name;

    /// <summary>フォルダーを太字にして、階層の切れ目を目で追えるようにする。</summary>
    public FontWeight NameWeight { get; } = entry.IsDirectory ? FontWeight.SemiBold : FontWeight.Normal;

    public string StatusText { get; } = entry.Error is not null
        ? "読めない"
        : entry.Status switch
        {
            EntryStatus.Identical => entry.IsDirectory ? string.Empty : "一致",
            EntryStatus.Different => "差異あり",
            EntryStatus.LeftOnly => "左のみ",
            EntryStatus.RightOnly => "右のみ",
            _ => string.Empty,
        };

    public IBrush StatusBrush { get; } = entry.Status switch
    {
        EntryStatus.Different => Palette.Brush("StatusDifferent"),
        EntryStatus.LeftOnly => Palette.Brush("StatusLeftOnly"),
        EntryStatus.RightOnly => Palette.Brush("StatusRightOnly"),
        _ => Palette.Brush("StatusSame"),
    };

    public string LeftSizeText { get; } = Format(entry.LeftSize);
    public string RightSizeText { get; } = Format(entry.RightSize);
    public string LeftModifiedText { get; } = entry.LeftModified?.ToString("yyyy-MM-dd HH:mm") ?? string.Empty;
    public string RightModifiedText { get; } = entry.RightModified?.ToString("yyyy-MM-dd HH:mm") ?? string.Empty;

    /// <summary>テキスト比較を開けるのは、両側に実体があり内容が違うファイルだけ。</summary>
    public bool CanOpenText { get; } =
        !entry.IsDirectory && entry.Status is EntryStatus.Different or EntryStatus.Identical;

    private static string Format(long? size) => size switch
    {
        null => string.Empty,
        < 1024 => $"{size} B",
        < 1024 * 1024 => $"{size / 1024.0:F1} KB",
        _ => $"{size / 1024.0 / 1024.0:F1} MB",
    };
}

/// <summary>
/// フォルダー比較の画面。
///
/// ここでは「同じかどうか」までしか出さない。どこがどう違うかは行を開いたときに
/// テキスト比較へ渡す。フォルダー全体に意味的な比較をかけると、一覧が出るまでに
/// 何分もかかることになる。
/// </summary>
public sealed class FolderCompareViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;
    private List<FolderRowView> _allRows = [];
    private FolderComparison? _comparison;
    private bool _isBusy;
    private bool _differencesOnly = true;
    private string _statusText = string.Empty;
    private FolderRowView? _selected;
    private string _include = string.Empty;
    private string _exclude = string.Empty;
    private bool _byTimestamp;
    private double _tolerance;
    private bool _ignoreDst;
    private bool _detectRenames;
    private string _renameText = string.Empty;

    public FolderCompareViewModel(ShellViewModel shell)
    {
        _shell = shell;
        _shell.ThemeChanged += RebuildForTheme;
        RefreshCommand = new RelayCommand(RunAsync);
        OpenSelectedCommand = new RelayCommand(() => { OpenSelected(); return Task.CompletedTask; });
        ExportCsvCommand = new RelayCommand(ExportCsvAsync);
        ToggleCommand = new RelayCommand<FolderRowView>(
            row => { Toggle(row); return Task.CompletedTask; });
        ExpandAllCommand = new RelayCommand(() => { ExpandAll(true); return Task.CompletedTask; });
        CollapseAllCommand = new RelayCommand(() => { ExpandAll(false); return Task.CompletedTask; });
        BrowseLeftCommand = new RelayCommand(() => PickAsync(left: true));
        BrowseRightCommand = new RelayCommand(() => PickAsync(left: false));
    }

    private string _leftRoot = string.Empty;
    private string _rightRoot = string.Empty;

    /// <summary>
    /// 比べる場所。**書き換えられる。**
    ///
    /// 以前はコンストラクタで固定していたが、画面をサイドバーで行き来する形に
    /// したので、同じ画面のまま別のフォルダーを指定できる必要がある。
    /// </summary>
    public string LeftRoot
    {
        get => _leftRoot;
        set => Set(ref _leftRoot, value);
    }

    public string RightRoot
    {
        get => _rightRoot;
        set => Set(ref _rightRoot, value);
    }

    public ObservableCollection<FolderRowView> Rows { get; } = [];
    public ICommand ToggleCommand { get; }
    public ICommand ExpandAllCommand { get; }
    public ICommand CollapseAllCommand { get; }
    public ICommand BrowseLeftCommand { get; }
    public ICommand BrowseRightCommand { get; }

    private async Task PickAsync(bool left)
    {
        var title = left ? "左のフォルダーを選択" : "右のフォルダーを選択";
        if (await _shell.PickPath(title, true) is { } path)
        {
            if (left)
            {
                LeftRoot = path;
            }
            else
            {
                RightRoot = path;
            }
        }
    }

    /// <summary>テーマの切り替えなど、画面をまたぐ操作。</summary>
    public ShellViewModel Shell => _shell;
    public ICommand RefreshCommand { get; }
    public ICommand OpenSelectedCommand { get; }
    public ICommand ExportCsvCommand { get; }

    /// <summary>一覧を CSV で書き出す。絞り込みではなく走査した全件を出す。</summary>
    private async Task ExportCsvAsync()
    {
        if (_comparison is null)
        {
            StatusText = "先に走査してください。";
            return;
        }
        if (await _shell.PickSavePath("一覧を CSV で保存", "folder-compare.csv") is not { } path)
        {
            return;
        }
        try
        {
            // BOM を付ける。付けないと Excel が UTF-8 と判定せず日本語が化ける。
            await File.WriteAllTextAsync(
                path, Report.FolderCsv(_comparison), new System.Text.UTF8Encoding(true));
            StatusText = $"{path} へ書き出した";
        }
        catch (Exception error)
        {
            StatusText = $"書き出せない: {error.Message}";
        }
    }

    /// <summary>対象にする名前。空白区切りで複数。ファイルにだけ効く。</summary>
    public string IncludeNames
    {
        get => _include;
        set => Set(ref _include, value);
    }

    /// <summary>除外する名前。空白区切りで複数。</summary>
    public string ExcludeNames
    {
        get => _exclude;
        set => Set(ref _exclude, value);
    }

    /// <summary>中身を読まず、大きさと更新時刻で比べる。</summary>
    public bool CompareByTimestamp
    {
        get => _byTimestamp;
        set => Set(ref _byTimestamp, value);
    }

    /// <summary>更新時刻の差をどこまで同じとみなすか（秒）。</summary>
    public double TimestampTolerance
    {
        get => _tolerance;
        set => Set(ref _tolerance, value);
    }

    /// <summary>ちょうど 1 時間のずれを同じとみなす（夏時間）。</summary>
    public bool IgnoreDaylightSaving
    {
        get => _ignoreDst;
        set => Set(ref _ignoreDst, value);
    }

    /// <summary>名前が変わっただけのファイルを探すか。片側だけの項目にしか触らない。</summary>
    public bool DetectRenames
    {
        get => _detectRenames;
        set => Set(ref _detectRenames, value);
    }

    /// <summary>見つかったリネームの一覧。</summary>
    public string RenameText
    {
        get => _renameText;
        private set
        {
            if (Set(ref _renameText, value))
            {
                OnPropertyChanged(nameof(HasRenames));
            }
        }
    }

    public bool HasRenames => RenameText.Length > 0;

    /// <summary>画面の設定から走査の指定を作る。</summary>
    private FolderCompareOptions BuildOptions()
    {
        static List<string> Split(string value)
            => [.. value.Split([' ', ',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];

        return new FolderCompareOptions
        {
            Filter = new NameFilter(Split(IncludeNames), Split(ExcludeNames)),
            Mode = CompareByTimestamp
                ? FolderComparisonMode.SizeAndTimestamp
                : FolderComparisonMode.Content,
            TimestampToleranceSeconds = TimestampTolerance,
            IgnoreDaylightSavingOffset = IgnoreDaylightSaving,
        };
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set => Set(ref _isBusy, value);
    }

    /// <summary>既定で差異のみ表示。一致まで並べると本当に見たい行が埋もれる。</summary>
    public bool DifferencesOnly
    {
        get => _differencesOnly;
        set
        {
            if (Set(ref _differencesOnly, value))
            {
                Rebuild();
            }
        }
    }

    public string StatusText
    {
        get => _statusText;
        private set => Set(ref _statusText, value);
    }

    public FolderRowView? Selected
    {
        get => _selected;
        set => Set(ref _selected, value);
    }

    private async Task RunAsync()
    {
        IsBusy = true;
        StatusText = "走査しています…";
        try
        {
            var left = LeftRoot;
            var right = RightRoot;
            var options = BuildOptions();
            var detectRenames = DetectRenames;
            var (result, renames) = await Task.Run(() =>
            {
                // 書庫なら一時領域へ展開する。走査が終わったら消す。
                using var leftSource = ArchiveSource.Open(left);
                using var rightSource = ArchiveSource.Open(right);
                var comparison = FolderComparer.Compare(leftSource.Path, rightSource.Path, options);
                var found = detectRenames
                    ? RenameDetector.Detect(comparison, leftSource.Path, rightSource.Path)
                    : [];
                return (comparison, found);
            });

            RenameText = renames.Count == 0
                ? string.Empty
                : "名前が変わったもの: " + string.Join(
                    " / ",
                    renames.Select(r => $"{r.LeftPath} → {r.RightPath}"
                        + (r.IdenticalContent ? string.Empty : $" ({r.Similarity:F2})")));
            _comparison = result;
            _allRows = result.Entries.Select(e => new FolderRowView(e)).ToList();

            var s = result.Stats;
            StatusText = $"差異 {s.Different} / 左のみ {s.LeftOnly} / 右のみ {s.RightOnly} / "
                + $"一致 {s.Identical} / フォルダー {s.Directories}"
                + (s.Errors > 0 ? $" / 読めなかったもの {s.Errors}" : string.Empty);
            Rebuild();
        }
        catch (Exception error)
        {
            _allRows = [];
            Rows.Clear();
            StatusText = $"エラー: {error.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    /// <summary>テーマが変わったときに、走査し直さずに行だけ作り直す。</summary>
    private void RebuildForTheme()
    {
        if (_comparison is null)
        {
            return;
        }
        _allRows = _comparison.Entries.Select(e => new FolderRowView(e)).ToList();

        // 下に何か入っているか。次の行の方が深ければ、そこが中身。
        for (var i = 0; i < _allRows.Count; i++)
        {
            _allRows[i].HasChildren = _allRows[i].Entry.IsDirectory
                && i + 1 < _allRows.Count
                && _allRows[i + 1].Entry.Depth > _allRows[i].Entry.Depth;
        }
        Rebuild();
    }

    private void Rebuild()
    {
        Rows.Clear();
        if (_comparison is null)
        {
            return;
        }

        // 閉じているフォルダーの中身は出さない。**深さで飛ばす。**
        // 親子の対応表を持たなくても、走査の順序が深さ優先なのでこれで足りる。
        var closedAt = int.MaxValue;

        foreach (var row in _allRows)
        {
            var depth = row.Entry.Depth;
            if (depth > closedAt)
            {
                continue;
            }
            closedAt = int.MaxValue;

            if (!DifferencesOnly || row.Entry.Status != EntryStatus.Identical)
            {
                Rows.Add(row);
            }

            if (row.Entry.IsDirectory && !row.IsExpanded)
            {
                closedAt = depth;
            }
        }
        OnPropertyChanged(nameof(IsEmpty));
    }

    /// <summary>その行を開閉する。</summary>
    public void Toggle(FolderRowView row)
    {
        if (!row.HasChildren)
        {
            return;
        }
        row.IsExpanded = !row.IsExpanded;
        Rebuild();
    }

    /// <summary>全部開く／全部閉じる。深い木で目的の場所を探すとき用。</summary>
    public void ExpandAll(bool expand)
    {
        foreach (var row in _allRows)
        {
            row.IsExpanded = expand;
        }
        Rebuild();
    }

    public bool IsEmpty => Rows.Count == 0;

    public void OpenSelected()
    {
        if (Selected is not { CanOpenText: true } row)
        {
            return;
        }
        var relative = row.Entry.RelativePath.Replace('/', Path.DirectorySeparatorChar);
        var left = Path.Combine(LeftRoot, relative);
        var right = Path.Combine(RightRoot, relative);
        if (File.Exists(left) && File.Exists(right))
        {
            // 戻り先としてこの画面自身を渡す。走査し直さずに一覧へ帰れる。
            _shell.ShowText(left, right);
        }
    }
}
