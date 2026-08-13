using System.Collections.ObjectModel;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>版の表の 1 行。</summary>
public sealed class VersionRowView(VersionDifference difference)
{
    public string Key => difference.Key;
    public string Left => difference.Left ?? "（無し）";
    public string Right => difference.Right ?? "（無し）";
    public bool IsSame => difference.IsSame;

    /// <summary>違う行だけ背景を変える。**同じ行を塗ると、違いが埋もれる。**</summary>
    public IBrush Background => difference.IsSame
        ? Brushes.Transparent
        : Palette.Brush("BgChanged");

    /// <summary>片方に無い側は斜線で埋める（テキスト比較・フォルダー比較と同じ）。</summary>
    public IBrush LeftBackground => difference.Left is null ? Palette.Gap() : Brushes.Transparent;
    public IBrush RightBackground => difference.Right is null ? Palette.Gap() : Brushes.Transparent;

    public FontWeight Weight => difference.IsSame ? FontWeight.Normal : FontWeight.SemiBold;
}

/// <summary>
/// 実行ファイルのバージョン情報を並べる画面（BC の Version Compare に当たる）。
///
/// **読み取りは engine に置く。** ここは表にするだけ。
/// </summary>
public sealed class VersionCompareViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    /// <summary>自分が乗っているタブ。見出しを比較の中身に合わせて書き換える。</summary>
    public CompareTab? Tab { get; set; }

    public ShellViewModel Shell => _shell;

    public VersionCompareViewModel(ShellViewModel shell)
    {
        _shell = shell;
        CompareCommand = new RelayCommand(CompareAsync, () => !_busy);
        OpenAsBinaryCommand = new RelayCommand(() =>
        {
            _shell.ShowBinary(LeftPath, RightPath);
            return Task.CompletedTask;
        }, () => LeftPath.Length > 0 && RightPath.Length > 0);
    }

    public RelayCommand CompareCommand { get; }
    public RelayCommand OpenAsBinaryCommand { get; }

    public ObservableCollection<VersionRowView> Rows { get; } = [];

    private string _leftPath = string.Empty;
    public string LeftPath
    {
        get => _leftPath;
        set => Set(ref _leftPath, value);
    }

    private string _rightPath = string.Empty;
    public string RightPath
    {
        get => _rightPath;
        set => Set(ref _rightPath, value);
    }

    private bool _differencesOnly;

    /// <summary>
    /// 違う項目だけ出す。
    ///
    /// **既定は全部出す。** 項目は十数個しかないので、隠す利得より
    /// 「会社名も説明も同じだ」と目で確かめられる方が大きい。
    /// </summary>
    public bool DifferencesOnly
    {
        get => _differencesOnly;
        set { if (Set(ref _differencesOnly, value)) { Rebuild(); } }
    }

    private string _summary = string.Empty;
    public string Summary
    {
        get => _summary;
        private set => Set(ref _summary, value);
    }

    private string _message = string.Empty;
    public string Message
    {
        get => _message;
        private set => Set(ref _message, value);
    }

    private bool _busy;
    public bool Busy
    {
        get => _busy;
        private set { if (Set(ref _busy, value)) { CompareCommand.Raise(); } }
    }

    private IReadOnlyList<VersionDifference> _differences = [];

    private async Task CompareAsync()
    {
        if (LeftPath.Length == 0 || RightPath.Length == 0)
        {
            Message = "比べる 2 つの実行ファイルを指定してください。";
            return;
        }

        Busy = true;
        Message = string.Empty;
        Rows.Clear();

        try
        {
            var (leftPath, rightPath) = (LeftPath, RightPath);
            _differences = await Task.Run(() => VersionInfo.Compare(
                VersionInfo.Read(leftPath), VersionInfo.Read(rightPath)));

            Rebuild();

            var differing = _differences.Count(d => !d.IsSame);
            Summary = differing == 0
                ? "バージョン情報は同じです。"
                : $"{differing} 項目が違います。";

            if (Tab is { } tab)
            {
                tab.Title = System.IO.Path.GetFileName(leftPath) + "（版）";
            }
        }
        catch (Exception error) when (error is IOException or InvalidDataException
                                        or UnauthorizedAccessException)
        {
            Message = error.Message;
            _differences = [];
        }
        finally
        {
            Busy = false;
        }
    }

    private void Rebuild()
    {
        Rows.Clear();
        foreach (var difference in _differences)
        {
            if (DifferencesOnly && difference.IsSame)
            {
                continue;
            }
            Rows.Add(new VersionRowView(difference));
        }
    }
}
