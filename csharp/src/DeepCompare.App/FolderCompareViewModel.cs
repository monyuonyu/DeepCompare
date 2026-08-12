using System.Collections.ObjectModel;
using System.Windows.Input;
using Avalonia;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>フォルダー一覧の 1 行。</summary>
public sealed class FolderRowView(FolderEntry entry)
{
    private static readonly IBrush Different = new SolidColorBrush(Color.Parse("#FFA500"));
    private static readonly IBrush LeftOnly = new SolidColorBrush(Color.Parse("#E06C75"));
    private static readonly IBrush RightOnly = new SolidColorBrush(Color.Parse("#8FBF6F"));
    private static readonly IBrush Same = new SolidColorBrush(Color.Parse("#7E838F"));

    public FolderEntry Entry { get; } = entry;

    /// <summary>階層を字下げで表す。木構造の畳み込みは持たず、一覧として素直に出す。</summary>
    public Thickness Indent { get; } = new(entry.Depth * 16, 0, 0, 0);

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
        EntryStatus.Different => Different,
        EntryStatus.LeftOnly => LeftOnly,
        EntryStatus.RightOnly => RightOnly,
        _ => Same,
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

    public FolderCompareViewModel(ShellViewModel shell, string leftRoot, string rightRoot)
    {
        _shell = shell;
        LeftRoot = leftRoot;
        RightRoot = rightRoot;
        BackCommand = new RelayCommand(() => { _shell.GoHome(); return Task.CompletedTask; });
        RefreshCommand = new RelayCommand(RunAsync);
        OpenSelectedCommand = new RelayCommand(() => { OpenSelected(); return Task.CompletedTask; });
        _ = RunAsync();
    }

    public string LeftRoot { get; }
    public string RightRoot { get; }
    public ObservableCollection<FolderRowView> Rows { get; } = [];
    public ICommand BackCommand { get; }
    public ICommand RefreshCommand { get; }
    public ICommand OpenSelectedCommand { get; }

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
            var result = await Task.Run(() => FolderComparer.Compare(left, right));
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

    private void Rebuild()
    {
        Rows.Clear();
        if (_comparison is null)
        {
            return;
        }
        foreach (var row in _allRows)
        {
            if (!DifferencesOnly || row.Entry.Status != EntryStatus.Identical)
            {
                Rows.Add(row);
            }
        }
        OnPropertyChanged(nameof(IsEmpty));
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
            _shell.ShowText(left, right, this);
        }
    }
}
