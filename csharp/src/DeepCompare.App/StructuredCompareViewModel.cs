using System.Collections.ObjectModel;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>差分 1 件の表示用。色と印を持たせる。</summary>
public sealed class StructuralChangeRow(StructuralChange change)
{
    public StructuralChange Change { get; } = change;

    public string Path => Change.Path;

    public string Mark => Change.Kind switch
    {
        StructuralChangeKind.Added => "+",
        StructuralChangeKind.Removed => "-",
        StructuralChangeKind.Changed => "~",
        StructuralChangeKind.TypeChanged => "!",
        StructuralChangeKind.Moved => "→",
        _ => " ",
    };

    public string KindLabel => Change.Kind switch
    {
        StructuralChangeKind.Added => "追加",
        StructuralChangeKind.Removed => "削除",
        StructuralChangeKind.Changed => "変更",
        StructuralChangeKind.TypeChanged => "型の変化",
        StructuralChangeKind.Moved => "位置の変化",
        _ => string.Empty,
    };

    public string Left => Change.Left?.Display() ?? string.Empty;

    public string Right => Change.Right?.Display() ?? string.Empty;

    /// <summary>型の変化のときだけ、種類の名前を添える。ここが目で気づけない差の筆頭。</summary>
    public string TypeNote => Change.Kind == StructuralChangeKind.TypeChanged
        ? $"{Change.Left?.KindName()} → {Change.Right?.KindName()}"
        : string.Empty;

    public bool HasTypeNote => TypeNote.Length > 0;

    public IBrush Background => Change.Kind switch
    {
        StructuralChangeKind.Added => Palette.Brush("BgAdded"),
        StructuralChangeKind.Removed => Palette.Brush("BgRemoved"),
        // 型の変化は目で気づけない差の筆頭なので、変更とは別の色にしたい。
        // いまは専用の色を持っていないので、変更と同じ扱い。印（!）で見分ける。
        StructuralChangeKind.TypeChanged => Palette.Brush("BgChanged"),
        StructuralChangeKind.Moved => Palette.Brush("BgChanged"),
        _ => Palette.Brush("BgChanged"),
    };
}

/// <summary>
/// 構造化データの比較画面。
///
/// テキスト比較とは別の画面にする。**出すものが行ではなく「位置と変化」**なので、
/// 左右に並べる形が合わない。JSON のキーの順序が違えば、行の並びには意味が無い。
/// </summary>
public sealed class StructuredCompareViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;
    private readonly object? _back;

    public StructuredCompareViewModel(ShellViewModel shell, object? back = null)
    {
        _shell = shell;
        _back = back;
        CompareCommand = new RelayCommand(CompareAsync, () => !_busy);
        BackCommand = new RelayCommand(() => { _shell.GoBack(_back); return Task.CompletedTask; });
        SaveCommand = new RelayCommand(SaveAsync, () => Changes.Count > 0);
        OpenAsTextCommand = new RelayCommand(() =>
        {
            // 構造では説明しきれないときのために、生のテキスト比較へ移れるようにする。
            _shell.ShowText(LeftPath, RightPath, this);
            return Task.CompletedTask;
        }, () => LeftPath.Length > 0 && RightPath.Length > 0);
    }

    public ObservableCollection<StructuralChangeRow> Changes { get; } = [];

    public RelayCommand CompareCommand { get; }
    public RelayCommand BackCommand { get; }
    public RelayCommand SaveCommand { get; }
    public RelayCommand OpenAsTextCommand { get; }

    private string _leftPath = string.Empty;
    public string LeftPath
    {
        get => _leftPath;
        set { if (Set(ref _leftPath, value)) { OpenAsTextCommand.Raise(); } }
    }

    private string _rightPath = string.Empty;
    public string RightPath
    {
        get => _rightPath;
        set { if (Set(ref _rightPath, value)) { OpenAsTextCommand.Raise(); } }
    }

    private string _message = string.Empty;
    public string Message
    {
        get => _message;
        private set => Set(ref _message, value);
    }

    private string _summary = string.Empty;
    public string Summary
    {
        get => _summary;
        private set => Set(ref _summary, value);
    }

    private bool _busy;
    public bool Busy
    {
        get => _busy;
        private set
        {
            if (Set(ref _busy, value))
            {
                CompareCommand.Raise();
            }
        }
    }

    // --- 設定 ---

    private string _arrayKeys = "id, name, key, path";
    /// <summary>配列の要素を対応付ける名前。読点か空白で区切って複数指定できる。</summary>
    public string ArrayKeys
    {
        get => _arrayKeys;
        set => Set(ref _arrayKeys, value);
    }

    private string _ignoredPaths = string.Empty;
    /// <summary>比較しない位置。1 行に 1 つ書く。</summary>
    public string IgnoredPaths
    {
        get => _ignoredPaths;
        set => Set(ref _ignoredPaths, value);
    }

    private bool _reportMoves = true;
    public bool ReportMoves
    {
        get => _reportMoves;
        set => Set(ref _reportMoves, value);
    }

    private bool _strictNumbers;
    /// <summary>1.0 と 1 を別のものとして扱う。</summary>
    public bool StrictNumbers
    {
        get => _strictNumbers;
        set => Set(ref _strictNumbers, value);
    }

    private StructuredCompareOptions BuildOptions()
    {
        var keys = ArrayKeys
            .Split([',', '、', ' ', '\t'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var ignored = IgnoredPaths
            .Split(['\n', '\r'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        return new StructuredCompareOptions
        {
            ArrayKeys = keys.Length > 0 ? keys : new StructuredCompareOptions().ArrayKeys,
            IgnoredPaths = ignored,
            ReportMoves = ReportMoves,
            NumbersByValue = !StrictNumbers,
        };
    }

    private async Task CompareAsync()
    {
        if (LeftPath.Length == 0 || RightPath.Length == 0)
        {
            Message = "左右のファイルを指定してください。";
            return;
        }

        Busy = true;
        Message = string.Empty;
        Changes.Clear();
        Summary = string.Empty;

        try
        {
            var options = BuildOptions();
            var left = LeftPath;
            var right = RightPath;

            // 読み取りと比較は作業スレッドへ。大きいファイルで画面が固まらないように。
            var changes = await Task.Run(() => StructuredCompare.CompareJson(
                File.ReadAllText(left), File.ReadAllText(right), options));

            foreach (var change in changes)
            {
                Changes.Add(new StructuralChangeRow(change));
            }
            Summary = changes.Count == 0
                ? "構造としては同じです。"
                : StructuredCompare.Summarize(changes);
        }
        catch (StructuredParseException error)
        {
            // どこが悪いかまで出す。「読めません」だけでは直せない。
            Message = error.Message;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Message = error.Message;
        }
        finally
        {
            Busy = false;
            SaveCommand.Raise();
        }
    }

    private async Task SaveAsync()
    {
        var path = await _shell.PickSavePath("差分を書き出す", "structural-diff.txt");
        if (path is null)
        {
            return;
        }
        var text = StructuredCompare.Format([.. Changes.Select(c => c.Change)]);
        await File.WriteAllTextAsync(path, text, new System.Text.UTF8Encoding(false));
        Message = $"{path} へ書き出しました。";
    }
}
