using System.Collections.ObjectModel;
using System.Windows.Input;
using Avalonia;
using Avalonia.Media;
using Avalonia.Threading;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// 一覧に出すものの絞り込み。
///
/// **「左だけにあるもの」「右だけにあるもの」を単独で見たい場面は多い。**
/// 移行漏れを探す、消し忘れを探す、といった作業がそれ。差異全部の中から
/// 目で拾うのは、数が増えると現実的でなくなる。
/// </summary>
public enum FolderFilter
{
    /// <summary>全部。</summary>
    All,

    /// <summary>差異のあるもの（違う・片側のみ）。</summary>
    Differences,

    /// <summary>左にしか無いもの。</summary>
    LeftOnly,

    /// <summary>右にしか無いもの。</summary>
    RightOnly,

    /// <summary>中身が同じもの。</summary>
    Identical,
}

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

    /// <summary>
    /// 左右それぞれに出す名前。**その側に無ければ空。**
    ///
    /// Beyond Compare と同じで、名前も左右に並べる。片側にしか無いファイルは
    /// 反対側が空欄になるので、一覧を眺めただけでどちらに寄っているかが分かる。
    /// 1 本の名前列に状態の文字を添える形だと、その判断に文字を読む必要がある。
    /// </summary>
    public string LeftName { get; } = entry.Status == EntryStatus.RightOnly
        ? string.Empty
        : (entry.IsDirectory ? entry.Name + "/" : entry.Name);

    public string RightName { get; } = entry.Status == EntryStatus.LeftOnly
        ? string.Empty
        : (entry.IsDirectory ? entry.Name + "/" : entry.Name);

    /// <summary>その側に実体があるか。無い側は斜線で埋める（テキスト比較と同じ）。</summary>
    public bool HasLeft { get; } = entry.Status != EntryStatus.RightOnly;
    public bool HasRight { get; } = entry.Status != EntryStatus.LeftOnly;

    public IBrush LeftBackground { get; } = entry.Status == EntryStatus.RightOnly
        ? Palette.Gap()
        : Brushes.Transparent;

    public IBrush RightBackground { get; } = entry.Status == EntryStatus.LeftOnly
        ? Palette.Gap()
        : Brushes.Transparent;

    /// <summary>
    /// 中央に置く印。
    ///
    /// **「中身が違う」ものにだけ出す（BC と同じ）。**
    /// 片側にしか無いものは、反対側が斜線で埋まっているので中央に印は要らない。
    /// 全部の行に何か出すと、本当に見たい「両方にあるが中身が違う」が埋もれる。
    /// </summary>
    public string CenterMark { get; } = entry.Status == EntryStatus.Different ? "≠" : string.Empty;

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

    /// <summary>自分が乗っているタブ。見出しを比較の中身に合わせて書き換える。</summary>
    public CompareTab? Tab { get; set; }
    private List<FolderRowView> _allRows = [];
    private FolderComparison? _comparison;
    private bool _isBusy;
    private string _statusText = string.Empty;
    private FolderRowView? _selected;
    private string _include = string.Empty;
    private string _exclude = string.Empty;
    private bool _byTimestamp;
    private double _tolerance;
    private bool _ignoreDst;
    private bool _detectRenames;
    private string _renameText = string.Empty;
    private string _mirrorNote = string.Empty;

    public FolderCompareViewModel(ShellViewModel shell)
    {
        _shell = shell;
        _shell.ThemeChanged += RebuildForTheme;
        RefreshCommand = new RelayCommand(RunAsync);
        OpenSelectedCommand = new RelayCommand(() => { OpenSelected(); return Task.CompletedTask; });
        ExportCsvCommand = new RelayCommand(ExportCsvAsync);
        OpenStructuredCommand = new RelayCommand<FolderRowView>(
            OpenStructuredAsync, row => row.CanOpenText);
        // BC の Actions メニューにある「Exclude」。設定を開いて除外欄に打ち込む
        // 近道で、実際いちばんよく使う操作の 1 つ。
        ExcludeCommand = new RelayCommand<FolderRowView>(
            row => { Exclude(row.Entry.Name); return Task.CompletedTask; });
        ExcludeTypeCommand = new RelayCommand<FolderRowView>(
            row => { Exclude("*" + System.IO.Path.GetExtension(row.Entry.Name)); return Task.CompletedTask; },
            row => System.IO.Path.GetExtension(row.Entry.Name).Length > 0);
        CopyToRightCommand = new RelayCommand<FolderRowView>(row => CopyFileAsync(row, toRight: true),
            row => !row.Entry.IsDirectory && row.HasLeft);
        CopyToLeftCommand = new RelayCommand<FolderRowView>(row => CopyFileAsync(row, toRight: false),
            row => !row.Entry.IsDirectory && row.HasRight);
        CancelCommand = new RelayCommand(
            () => { CancelScan(); return Task.CompletedTask; }, () => IsBusy);
        RenameLeftCommand = new RelayCommand<FolderRowView>(row => RenameAsync(row, left: true),
            row => row.HasLeft);
        RenameRightCommand = new RelayCommand<FolderRowView>(row => RenameAsync(row, left: false),
            row => row.HasRight);
        MoveToRightCommand = new RelayCommand<FolderRowView>(row => MoveAsync(row, toRight: true),
            row => row.HasLeft);
        MoveToLeftCommand = new RelayCommand<FolderRowView>(row => MoveAsync(row, toRight: false),
            row => row.HasRight);
        NewFolderLeftCommand = new RelayCommand(() => NewFolderAsync(left: true));
        NewFolderRightCommand = new RelayCommand(() => NewFolderAsync(left: false));
        // BC の Actions の残り。**どれも戻せないので必ず確認を出す。**
        DeleteLeftCommand = new RelayCommand<FolderRowView>(row => DeleteAsync(row, left: true),
            row => row.HasLeft);
        DeleteRightCommand = new RelayCommand<FolderRowView>(row => DeleteAsync(row, left: false),
            row => row.HasRight);
        TouchToLeftCommand = new RelayCommand<FolderRowView>(row => TouchAsync(row, toLeft: true),
            row => !row.Entry.IsDirectory && row.HasLeft && row.HasRight);
        TouchToRightCommand = new RelayCommand<FolderRowView>(row => TouchAsync(row, toLeft: false),
            row => !row.Entry.IsDirectory && row.HasLeft && row.HasRight);
        OpenBinaryCommand = new RelayCommand<FolderRowView>(row =>
        {
            var (left, right) = PathsOf(row);
            if (File.Exists(left) && File.Exists(right))
            {
                _shell.ShowBinary(left, right);
            }
            return Task.CompletedTask;
        }, row => !row.Entry.IsDirectory);
        OpenImageCommand = new RelayCommand<FolderRowView>(row =>
        {
            var (left, right) = PathsOf(row);
            if (File.Exists(left) && File.Exists(right))
            {
                _shell.ShowImage(left, right);
            }
            return Task.CompletedTask;
        // 名前で判断する。中身を覗いて判断してもよいが、押せるかどうかを
        // 決めるためだけに全行のファイルを開くことになる。
        }, row => !row.Entry.IsDirectory && ImageCompare.LooksLikeImage(row.Entry.RelativePath));
        OpenVersionCommand = new RelayCommand<FolderRowView>(row =>
        {
            var (left, right) = PathsOf(row);
            if (File.Exists(left) && File.Exists(right))
            {
                _shell.ShowVersionInfo(left, right);
            }
            return Task.CompletedTask;
        // 中身の先頭 2 バイトで判断する。**拡張子では決めない** —
        // .dll も .exe も .sys も .ocx も同じ形式で、名前の方が当てにならない。
        }, row => !row.Entry.IsDirectory && row.HasLeft && row.HasRight
                  && VersionInfo.LooksLikeExecutable(PathsOf(row).Left));
        OpenExtractedCommand = new RelayCommand<FolderRowView>(row =>
        {
            var (left, right) = PathsOf(row);
            if (File.Exists(left) && File.Exists(right))
            {
                _shell.ShowExtractedText(left, right);
            }
            return Task.CompletedTask;
        }, row => !row.Entry.IsDirectory
                  && ShellViewModel.CanExtractText(row.Entry.RelativePath));
        RevealLeftCommand = new RelayCommand<FolderRowView>(row => RevealAsync(row, left: true));
        RevealRightCommand = new RelayCommand<FolderRowView>(row => RevealAsync(row, left: false));
        CopyLeftPathCommand = new RelayCommand<FolderRowView>(row => CopyPathAsync(row, left: true));
        CopyRightPathCommand = new RelayCommand<FolderRowView>(row => CopyPathAsync(row, left: false));
        OpenRowCommand = new RelayCommand<FolderRowView>(row =>
        {
            Selected = row;
            OpenSelected();
            return Task.CompletedTask;
        }, row => row.CanOpenText);
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
    public ICommand OpenRowCommand { get; }
    public ICommand OpenStructuredCommand { get; }
    public ICommand OpenBinaryCommand { get; }
    public ICommand OpenImageCommand { get; }
    public ICommand OpenVersionCommand { get; }
    public ICommand OpenExtractedCommand { get; }
    public ICommand DeleteLeftCommand { get; }
    public ICommand DeleteRightCommand { get; }
    public ICommand TouchToLeftCommand { get; }
    public ICommand TouchToRightCommand { get; }
    public ICommand ExcludeCommand { get; }

    /// <summary>
    /// 片側を消す。
    ///
    /// **必ず確認を出す。** 上書きと違い、こちらは元の場所ごと消える。
    /// フォルダーは中身ごと消えるので、その旨も文に入れる。
    /// </summary>
    public RelayCommand<FolderRowView> RenameLeftCommand { get; }
    public RelayCommand<FolderRowView> RenameRightCommand { get; }
    public RelayCommand<FolderRowView> MoveToRightCommand { get; }
    public RelayCommand<FolderRowView> MoveToLeftCommand { get; }
    public RelayCommand NewFolderLeftCommand { get; }
    public RelayCommand NewFolderRightCommand { get; }

    /// <summary>
    /// 名前を変える（BC の Rename）。
    ///
    /// **片側だけ変える。** 両方まとめて変えると、名前が違うから別物として
    /// 並んでいるのか、揃えたから同じ行に来たのかが分からなくなる。
    /// </summary>
    private async Task RenameAsync(FolderRowView row, bool left)
    {
        if (Prompt is null)
        {
            return;
        }

        var (l, r) = PathsOf(row);
        var target = left ? l : r;
        var oldName = Path.GetFileName(target);

        var newName = await Prompt($"{oldName} の新しい名前", oldName);
        if (newName is null)
        {
            return;
        }

        var result = FolderOperations.Rename(target, newName, row.Entry.IsDirectory);
        StatusText = result.Message;
        if (result.Ok)
        {
            await RunAsync();
        }
    }

    /// <summary>
    /// 反対側へ移す（BC の Move）。**元からは消える。**
    ///
    /// 複製（Copy）と違い戻せないので、**必ず確認を出す。**
    /// </summary>
    private async Task MoveAsync(FolderRowView row, bool toRight)
    {
        if (Confirm is null)
        {
            return;
        }

        var (l, r) = PathsOf(row);
        var source = toRight ? l : r;
        var destination = toRight ? r : l;
        var name = Path.GetFileName(source);

        var exists = File.Exists(destination) || Directory.Exists(destination);
        var message = exists
            ? $"{name} を移します。移す先の同じ名前のものは上書きされ、元からは消えます。"
            : $"{name} を移します。元からは消えます。";
        if (!await Confirm(message))
        {
            return;
        }

        var result = FolderOperations.Move(source, destination, row.Entry.IsDirectory);
        StatusText = result.Message;
        if (result.Ok)
        {
            await RunAsync();
        }
    }

    /// <summary>
    /// フォルダーを作る（BC の New Folder）。
    ///
    /// **選んでいる行の中ではなく、比較している場所の直下に作る。**
    /// 選択によって作られる場所が変わると、どこにできたのか分からなくなる。
    /// </summary>
    private async Task NewFolderAsync(bool left)
    {
        if (Prompt is null)
        {
            return;
        }

        var root = left ? LeftRoot : RightRoot;
        var name = await Prompt(
            $"{(left ? "左" : "右")}に作るフォルダーの名前", "新しいフォルダー");
        if (name is null)
        {
            return;
        }

        var result = FolderOperations.NewFolder(root, name);
        StatusText = result.Message;
        if (result.Ok)
        {
            await RunAsync();
        }
    }

    internal async Task DeleteAsync(FolderRowView row, bool left)
    {
        var (l, r) = PathsOf(row);
        var target = left ? l : r;
        var name = Path.GetFileName(target);

        if (Confirm is null)
        {
            return;
        }
        var message = row.Entry.IsDirectory
            ? $"{name} を中身ごと消します。元に戻せません。"
            : $"{name} を消します。元に戻せません。";
        if (!await Confirm(message))
        {
            return;
        }

        try
        {
            if (row.Entry.IsDirectory)
            {
                Directory.Delete(target, recursive: true);
            }
            else
            {
                File.Delete(target);
            }
            await RunAsync();
            StatusText = $"{name} を消しました。";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            StatusText = $"消せません: {error.Message}";
        }
    }

    /// <summary>
    /// 更新時刻を反対側に合わせる（BC の Touch）。
    ///
    /// **中身が同じなのに時刻だけ違うとき**に使う。時刻で比べる設定
    /// （--by-timestamp）にしていると、これだけで「違う」と出てしまう。
    /// </summary>
    private async Task TouchAsync(FolderRowView row, bool toLeft)
    {
        var (l, r) = PathsOf(row);
        var (target, source) = toLeft ? (l, r) : (r, l);

        try
        {
            File.SetLastWriteTimeUtc(target, File.GetLastWriteTimeUtc(source));
            await RunAsync();
            StatusText = $"{Path.GetFileName(target)} の時刻を合わせました。";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            StatusText = $"時刻を変えられません: {error.Message}";
        }
    }
    public ICommand ExcludeTypeCommand { get; }
    public ICommand CopyToRightCommand { get; }
    public ICommand CopyToLeftCommand { get; }

    /// <summary>
    /// 除外の型に足して走査しなおす。
    ///
    /// **設定を開かずに 1 手で外せる**のが要点。「これは見なくていい」と
    /// 分かるのは一覧を眺めている最中なので、そこで外せないと手が止まる。
    /// </summary>
    private void Exclude(string pattern)
    {
        var current = ExcludeNames.Trim();
        if (current.Split(' ', StringSplitOptions.RemoveEmptyEntries).Contains(pattern))
        {
            return;
        }
        ExcludeNames = current.Length == 0 ? pattern : $"{current} {pattern}";
        _ = RunAsync();
    }

    /// <summary>
    /// ファイルを反対側へ写す。
    ///
    /// **上書きの確認は呼ぶ側（画面）が出す。** ViewModel から確認の窓を
    /// 出すと、試験ができなくなるうえ、画面のない経路（CLI）で使えなくなる。
    /// </summary>
    public Func<string, Task<bool>>? Confirm { get; set; }

    /// <summary>
    /// 名前を入力してもらう。取り消したら null。
    /// **画面側が差し込む** — ViewModel から窓を開かない。
    /// </summary>
    public Func<string, string, Task<string?>>? Prompt { get; set; }

    internal async Task CopyFileAsync(FolderRowView row, bool toRight)
    {
        var (left, right) = PathsOf(row);
        var (from, to) = toRight ? (left, right) : (right, left);

        if (!File.Exists(from))
        {
            StatusText = "写す元がありません。";
            return;
        }

        // **上書きになるときだけ訊く。** 何も無い所へ写すのは戻せる（消せばよい）
        // が、上書きは戻せない。
        if (File.Exists(to) && Confirm is not null
            && !await Confirm($"{Path.GetFileName(to)} を上書きします。よろしいですか。"))
        {
            return;
        }

        try
        {
            var directory = Path.GetDirectoryName(to);
            if (directory is not null)
            {
                Directory.CreateDirectory(directory);
            }
            File.Copy(from, to, overwrite: true);
            await RunAsync();
            StatusText = $"{Path.GetFileName(from)} を写しました。";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            StatusText = $"写せません: {error.Message}";
        }
    }
    public ICommand RevealLeftCommand { get; }
    public ICommand RevealRightCommand { get; }
    public ICommand CopyLeftPathCommand { get; }
    public ICommand CopyRightPathCommand { get; }
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

    /// <summary>
    /// リモートから取ってきたときの知らせ。
    ///
    /// **上限で切ったことを黙らない。** 取っていないものがあるのに
    /// 「差異なし」と出すのは嘘に近い。
    /// </summary>
    public string MirrorNote
    {
        get => _mirrorNote;
        private set
        {
            if (Set(ref _mirrorNote, value))
            {
                OnPropertyChanged(nameof(HasMirrorNote));
            }
        }
    }

    public bool HasMirrorNote => MirrorNote.Length > 0;

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
        private set
        {
            if (Set(ref _isBusy, value))
            {
                CancelCommand.Raise();
            }
        }
    }

    private FolderFilter _filter = FolderFilter.Differences;

    /// <summary>既定は差異のみ。一致まで並べると本当に見たい行が埋もれる。</summary>
    public FolderFilter Filter
    {
        get => _filter;
        set
        {
            if (Set(ref _filter, value))
            {
                ExpandForFilter();
                Rebuild();
                // 押している状態を出すため、どれが選ばれているかを全部知らせる。
                OnPropertyChanged(nameof(ShowAll));
                OnPropertyChanged(nameof(ShowDifferences));
                OnPropertyChanged(nameof(ShowLeftOnly));
                OnPropertyChanged(nameof(ShowRightOnly));
                OnPropertyChanged(nameof(ShowIdentical));
            }
        }
    }

    // ToggleButton は bool しか扱えないので、種類ごとに口を開ける。
    // 立てられたらその種類にする。**下ろす操作は受けない**（何も選ばれていない
    // 状態に落ちると、一覧が空になって理由が分からなくなる）。
    public bool ShowAll
    {
        get => _filter == FolderFilter.All;
        set { if (value) { Filter = FolderFilter.All; } else { OnPropertyChanged(); } }
    }

    public bool ShowDifferences
    {
        get => _filter == FolderFilter.Differences;
        set { if (value) { Filter = FolderFilter.Differences; } else { OnPropertyChanged(); } }
    }

    public bool ShowLeftOnly
    {
        get => _filter == FolderFilter.LeftOnly;
        set { if (value) { Filter = FolderFilter.LeftOnly; } else { OnPropertyChanged(); } }
    }

    public bool ShowRightOnly
    {
        get => _filter == FolderFilter.RightOnly;
        set { if (value) { Filter = FolderFilter.RightOnly; } else { OnPropertyChanged(); } }
    }

    public bool ShowIdentical
    {
        get => _filter == FolderFilter.Identical;
        set { if (value) { Filter = FolderFilter.Identical; } else { OnPropertyChanged(); } }
    }

    public string StatusText
    {
        get => _statusText;
        set => Set(ref _statusText, value);
    }

    public FolderRowView? Selected
    {
        get => _selected;
        set => Set(ref _selected, value);
    }

    private CancellationTokenSource? _scanning;

    /// <summary>
    /// 走査をやめる。
    ///
    /// **大きな木では要る。** 場所を選び間違えて `/` を走査し始めたとき、
    /// 終わるまで何もできないのは困る。
    /// </summary>
    public void CancelScan() => _scanning?.Cancel();

    /// <summary>やめるためのコマンド。**走っている間だけ押せる。**</summary>
    public RelayCommand CancelCommand { get; private set; } = null!;

    internal async Task RunAsync()
    {
        // 前の走査が残っていたら止める。**二重に走らせない。**
        _scanning?.Cancel();
        _scanning?.Dispose();
        _scanning = new CancellationTokenSource();
        var token = _scanning.Token;

        IsBusy = true;
        StatusText = "走査しています…";
        try
        {
            var left = LeftRoot;
            var right = RightRoot;
            var options = BuildOptions();
            var detectRenames = DetectRenames;
            // リモートは一時領域へ取ってくる。**絞り込みをそのまま渡す** —
            // 渡さないと、除外したはずのものまで回線を使って取ってくる。
            var mirrorOptions = new MirrorOptions { Filter = options.Filter };

            // **取っている最中を出す。** 黙って何分も待たせない。
            void Fetching(string name) => Dispatcher.UIThread.Post(
                () => StatusText = $"取ってきています… {name}");

            // **進捗を出す。** 数万ファイルの走査で、動いているのか
            // 止まっているのかが分からないのが一番困る。
            // 画面へ流し込む回数は抑える（1 件ごとに送ると描画で詰まる）。
            var seen = 0;
            void Progress(string path)
            {
                if (++seen % 200 == 0)
                {
                    Dispatcher.UIThread.Post(() => StatusText = $"走査しています… {seen:N0} 件");
                }
            }

            var (result, renames, notes) = await Task.Run(() =>
            {
                // 書庫なら一時領域へ展開し、リモートなら取ってくる。
                // 走査が終わったら消す。
                using var leftSource = ArchiveSource.Open(left, mirrorOptions, Fetching);
                using var rightSource = ArchiveSource.Open(right, mirrorOptions, Fetching);
                var comparison = FolderComparer.Compare(
                    leftSource.Path, rightSource.Path, options, Progress, token);
                var found = detectRenames
                    ? RenameDetector.Detect(comparison, leftSource.Path, rightSource.Path)
                    : [];

                // **上限で切ったことを黙らない。**
                var messages = new List<string>();
                foreach (var mirror in new[] { leftSource.Mirror, rightSource.Mirror })
                {
                    if (mirror is { Result.HasSkipped: true })
                    {
                        messages.Add(mirror.Describe());
                    }
                }
                return (comparison, found, messages);
            });

            MirrorNote = string.Join("  ", notes);

            RenameText = renames.Count == 0
                ? string.Empty
                : "名前が変わったもの: " + string.Join(
                    " / ",
                    renames.Select(r => $"{r.LeftPath} → {r.RightPath}"
                        + (r.IdenticalContent ? string.Empty : $" ({r.Similarity:F2})")));
            _comparison = result;
            _allRows = result.Entries.Select(e => new FolderRowView(e)).ToList();
            // **子の有無と展開はここでも通す。** 以前は「テーマが変わったとき」の
            // 経路にしか無く、比べ終わった直後の一覧では開閉の三角が出ず、
            // 絞り込んでいてもフォルダーが畳まれたままだった。
            MarkChildren();
            ExpandForFilter();

            var s = result.Stats;
            StatusText = $"差異 {s.Different} / 左のみ {s.LeftOnly} / 右のみ {s.RightOnly} / "
                + $"一致 {s.Identical} / フォルダー {s.Directories}"
                + (s.Errors > 0 ? $" / 読めなかったもの {s.Errors}" : string.Empty);
            Rebuild();
        }
        catch (OperationCanceledException)
        {
            // **やめたことを言う。** 一覧が空のまま黙って終わると、
            // 差異が無かったのか止めたのかが分からない。
            _allRows = [];
            Rows.Clear();
            StatusText = "走査をやめました。";
        }
        catch (Exception error)
        {
            _allRows = [];
            Rows.Clear();
            // **合言葉を出さない。** 場所を含む知らせがそのまま画面に残る。
            StatusText = $"エラー: {RemoteLocation.Redact(error.Message)}";
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
        MarkChildren();
        ExpandForFilter();
        Rebuild();
    }

    /// <summary>
    /// 下に何か入っているか。次の行の方が深ければ、そこが中身。
    ///
    /// **並びが深さ優先であることに頼る。** 親子の対応表を持たなくても、
    /// 隣を見るだけで足りる。
    /// </summary>
    private void MarkChildren()
    {
        for (var i = 0; i < _allRows.Count; i++)
        {
            _allRows[i].HasChildren = _allRows[i].Entry.IsDirectory
                && i + 1 < _allRows.Count
                && _allRows[i + 1].Entry.Depth > _allRows[i].Entry.Depth;
        }
    }

    /// <summary>
    /// 絞り込みに合う行を含むフォルダーを開く。
    ///
    /// **絞り込んでいるのに畳まれていると、探しているものが見えない。**
    /// 「差異あり」にしたのに `src/` が閉じたままだと、中の差異に辿り着くまで
    /// フォルダーを 1 つずつ開くことになる（それでは絞り込んだ意味が無い）。
    ///
    /// **「全部出す」では開かない。** そこでは畳んだ状態が出発点として正しく、
    /// 数千ファイルの木が一度に開くと目的の場所へ辿り着けない。
    /// </summary>
    private void ExpandForFilter()
    {
        if (_filter == FolderFilter.All || _allRows.Count == 0)
        {
            return;
        }

        // 深さごとに「その段に見せるものがあったか」を持つ。
        // **後ろから見る。** 並びは深さ優先なので、フォルダーに着いたときには
        // その子を見終わっている。
        var deepest = _allRows.Max(r => r.Entry.Depth);
        var hasMatch = new bool[deepest + 2];

        for (var i = _allRows.Count - 1; i >= 0; i--)
        {
            var row = _allRows[i];
            var depth = row.Entry.Depth;

            if (!row.Entry.IsDirectory)
            {
                if (Matches(row))
                {
                    hasMatch[depth] = true;
                }
                continue;
            }

            // 中に見せるものがあれば開き、親にもそれを伝える。
            row.IsExpanded = hasMatch[depth + 1];
            if (row.IsExpanded)
            {
                hasMatch[depth] = true;
            }

            // **より深い段の記録は捨てる。** 隣のフォルダーの中身が
            // 混ざると、空のフォルダーまで開いてしまう。
            for (var d = depth + 1; d < hasMatch.Length; d++)
            {
                hasMatch[d] = false;
            }
        }
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

            if (Matches(row))
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

    /// <summary>
    /// その行を出すか。
    ///
    /// **フォルダーの行は、絞り込んでいても出す。** 中身を隠すと、階層の
    /// どこにあるファイルなのかが分からなくなる。
    /// </summary>
    private bool Matches(FolderRowView row)
    {
        if (row.Entry.IsDirectory)
        {
            return true;
        }
        return Filter switch
        {
            FolderFilter.Differences => row.Entry.Status != EntryStatus.Identical,
            FolderFilter.LeftOnly => row.Entry.Status == EntryStatus.LeftOnly,
            FolderFilter.RightOnly => row.Entry.Status == EntryStatus.RightOnly,
            FolderFilter.Identical => row.Entry.Status == EntryStatus.Identical,
            _ => true,
        };
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

    /// <summary>
    /// → を押したとき。
    ///
    /// **ファイル管理ソフトと同じ動き。** 閉じたフォルダーなら開く。
    /// 開いているなら最初の子へ移る。ファイルなら何もしない。
    /// </summary>
    public void MoveRight()
    {
        if (Selected is not { } row)
        {
            return;
        }
        if (row.HasChildren && !row.IsExpanded)
        {
            Toggle(row);
            return;
        }
        if (row.HasChildren)
        {
            var index = Rows.IndexOf(row);
            if (index >= 0 && index + 1 < Rows.Count)
            {
                Selected = Rows[index + 1];
            }
        }
    }

    /// <summary>
    /// ← を押したとき。
    ///
    /// 開いているフォルダーなら閉じる。そうでなければ**親へ戻る**
    /// （深く潜ったところから上がるのに、これが無いと一段ずつ上に
    /// 押し続けることになる）。
    /// </summary>
    public void MoveLeft()
    {
        if (Selected is not { } row)
        {
            return;
        }
        if (row.HasChildren && row.IsExpanded)
        {
            Toggle(row);
            return;
        }

        var index = Rows.IndexOf(row);
        if (index < 0)
        {
            return;
        }
        var depth = row.Entry.Depth;
        for (var i = index - 1; i >= 0; i--)
        {
            if (Rows[i].Entry.Depth < depth)
            {
                Selected = Rows[i];
                return;
            }
        }
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

    /// <summary>その行の左右の実体のパス。片側に無ければ空。</summary>
    public (string Left, string Right) PathsOf(FolderRowView row)
    {
        var relative = row.Entry.RelativePath.Replace('/', Path.DirectorySeparatorChar);
        return (Path.Combine(LeftRoot, relative), Path.Combine(RightRoot, relative));
    }

    /// <summary>構造として比べる。JSON のときに効く。</summary>
    private Task OpenStructuredAsync(FolderRowView row)
    {
        var (left, right) = PathsOf(row);
        if (File.Exists(left) && File.Exists(right))
        {
            _shell.ShowStructured(left, right);
        }
        return Task.CompletedTask;
    }

    /// <summary>その場所をファイル管理ソフトで開く。</summary>
    private Task RevealAsync(FolderRowView row, bool left)
    {
        var (l, r) = PathsOf(row);
        var target = left ? l : r;
        var directory = row.Entry.IsDirectory ? target : Path.GetDirectoryName(target);
        if (directory is null || !Directory.Exists(directory))
        {
            return Task.CompletedTask;
        }
        try
        {
            // 環境ごとに開き方が違う。使えるものを順に試す。
            var opener = OperatingSystem.IsWindows() ? "explorer"
                : OperatingSystem.IsMacOS() ? "open" : "xdg-open";
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = opener,
                Arguments = $"\"{directory}\"",
                UseShellExecute = false,
            });
        }
        catch (Exception)
        {
            // 開けなくても比較は続けられる。黙って諦める。
        }
        return Task.CompletedTask;
    }

    /// <summary>パスを写す。別の道具へ渡すときに使う。</summary>
    private Task CopyPathAsync(FolderRowView row, bool left)
    {
        var (l, r) = PathsOf(row);
        Clipboard?.Invoke(left ? l : r);
        return Task.CompletedTask;
    }

    /// <summary>書き込み先。表示側から差し込む（ViewModel から画面に触らない）。</summary>
    public Action<string>? Clipboard { get; set; }

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
