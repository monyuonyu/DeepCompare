using System.Collections.ObjectModel;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>セル 1 組の表示用。</summary>
public sealed class NotebookCellRow(NotebookCellDiff diff, int number)
{
    public NotebookCellDiff Diff { get; } = diff;

    /// <summary>画面での通し番号。**元の位置ではない** — 片方にしか無いセルがあるため。</summary>
    public int Number { get; } = number;

    public string LeftText => Text(Diff.Left);
    public string RightText => Text(Diff.Right);

    private static string Text(NotebookCell? cell)
        => cell is null ? string.Empty : cell.Source.TrimEnd('\n');

    public bool HasLeft => Diff.Left is not null;
    public bool HasRight => Diff.Right is not null;

    /// <summary>
    /// セルの種類。**左右で違うことは無い** — 種類が変わったら
    /// 「消えた＋増えた」に分かれるので、同じ組に並ぶことがない。
    /// </summary>
    public string KindLabel => (Diff.Left ?? Diff.Right)?.Kind switch
    {
        CellKind.Code => "コード",
        CellKind.Markdown => "Markdown",
        CellKind.Raw => "生",
        _ => string.Empty,
    };

    public string ChangeLabel => Diff.Change switch
    {
        CellChange.SourceChanged => "本文が変わった",
        CellChange.OutputOnly => "出力だけが変わった",
        CellChange.Added => "増えた",
        CellChange.Removed => "消えた",
        _ => "変化なし",
    };

    /// <summary>
    /// 元の位置。**両方に出す。** セルが増減すると番号がずれるので、
    /// 「左の 5 番目と右の 7 番目」が分からないと元のファイルに戻れない。
    /// </summary>
    public string PositionLabel => (Diff.LeftIndex, Diff.RightIndex) switch
    {
        ( >= 0, >= 0) => $"左 {Diff.LeftIndex + 1} / 右 {Diff.RightIndex + 1}",
        ( >= 0, _) => $"左 {Diff.LeftIndex + 1}",
        (_, >= 0) => $"右 {Diff.RightIndex + 1}",
        _ => string.Empty,
    };

    /// <summary>
    /// 出力の有無。**中身は出さない。**
    /// base64 の画像を並べても読めないし、それが差分を埋もれさせる元凶だった。
    /// </summary>
    public string OutputLabel
    {
        get
        {
            var left = Diff.Left?.HasOutputs ?? false;
            var right = Diff.Right?.HasOutputs ?? false;
            if (!left && !right)
            {
                return string.Empty;
            }
            var image = (Diff.Left?.HasImageOutput ?? false)
                     || (Diff.Right?.HasImageOutput ?? false);
            return image ? "出力あり（画像を含む）" : "出力あり";
        }
    }

    public bool HasOutputLabel => OutputLabel.Length > 0;

    /// <summary>変化なしのセルは畳んでよい。</summary>
    public bool IsInteresting => Diff.Change != CellChange.Unchanged;

    public IBrush Background => Diff.Change switch
    {
        CellChange.SourceChanged => Palette.Brush("BgChanged"),
        CellChange.Added => Palette.Brush("BgAdded"),
        CellChange.Removed => Palette.Brush("BgRemoved"),
        // **出力だけの変化は目立たせない。** 既定では見ないものなので、
        // 色を付けると「変わった」と読まれてしまう。
        _ => Palette.Brush("CardBg"),
    };
}

/// <summary>
/// Jupyter ノートブックをセル単位で比べる画面。
///
/// **行で比べる画面と分ける。** 実体は JSON で、本文・出力・実行回数が
/// 混ざっている。1 文字直して実行し直すと出力の base64 が数千行動き、
/// 直した 1 行はその中に埋もれる。
///
/// 既定では**出力と実行回数を見ない**。見たいときだけ開く。
/// </summary>
public sealed class NotebookCompareViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    public NotebookCompareViewModel(ShellViewModel shell)
    {
        _shell = shell;
        CompareCommand = new RelayCommand(CompareAsync, () => !Busy);
        OpenAsTextCommand = new RelayCommand(
            OpenAsTextAsync, () => LeftPath.Length > 0 && RightPath.Length > 0);
    }

    public ShellViewModel Shell => _shell;
    public CompareTab? Tab { get; set; }

    public ObservableCollection<NotebookCellRow> Cells { get; } = [];

    public RelayCommand CompareCommand { get; }
    public RelayCommand OpenAsTextCommand { get; }

    private string _leftPath = string.Empty;
    public string LeftPath
    {
        get => _leftPath;
        set
        {
            if (Set(ref _leftPath, value))
            {
                OpenAsTextCommand.Raise();
            }
        }
    }

    private string _rightPath = string.Empty;
    public string RightPath
    {
        get => _rightPath;
        set
        {
            if (Set(ref _rightPath, value))
            {
                OpenAsTextCommand.Raise();
            }
        }
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

    private bool _compareOutputs;

    /// <summary>
    /// 出力も比べるか。**既定は比べない。**
    /// 実行しただけで全部動くので、既定で見ると本当の変更が埋もれる。
    /// </summary>
    public bool CompareOutputs
    {
        get => _compareOutputs;
        set
        {
            if (Set(ref _compareOutputs, value) && Cells.Count > 0)
            {
                // 切り替えたら測り直す。**古い結果を残さない。**
                CompareCommand.Execute(null);
            }
        }
    }

    private bool _showUnchanged;

    /// <summary>変化なしのセルも出すか。既定では出さない。</summary>
    public bool ShowUnchanged
    {
        get => _showUnchanged;
        set
        {
            if (Set(ref _showUnchanged, value) && Cells.Count > 0)
            {
                CompareCommand.Execute(null);
            }
        }
    }

    private string _metadataChange = string.Empty;

    /// <summary>言語やカーネルの変化。**別に出す** — 付け替えただけで全体が動く。</summary>
    public string MetadataChange
    {
        get => _metadataChange;
        private set
        {
            if (Set(ref _metadataChange, value))
            {
                OnPropertyChanged(nameof(HasMetadataChange));
            }
        }
    }

    public bool HasMetadataChange => MetadataChange.Length > 0;

    internal async Task CompareAsync()
    {
        if (LeftPath.Length == 0 || RightPath.Length == 0)
        {
            return;
        }

        Busy = true;
        Message = "読んでいます…";
        try
        {
            var options = new NotebookCompareOptions
            {
                CompareOutputs = CompareOutputs,
                CompareExecutionCount = false,
            };

            var (comparison, leftName, rightName) = await Task.Run(() =>
            {
                var left = Notebook.Read(File.ReadAllText(LeftPath));
                var right = Notebook.Read(File.ReadAllText(RightPath));
                return (Notebook.Compare(left, right, options),
                        Path.GetFileName(LeftPath), Path.GetFileName(RightPath));
            });

            Cells.Clear();
            var number = 0;
            foreach (var diff in comparison.Cells)
            {
                if (!ShowUnchanged && diff.Change == CellChange.Unchanged)
                {
                    continue;
                }
                Cells.Add(new NotebookCellRow(diff, ++number));
            }

            MetadataChange = comparison.MetadataChange ?? string.Empty;
            Summary = $"本文が変わった {comparison.SourceChanged}"
                + $" / 増えた {comparison.Added}"
                + $" / 消えた {comparison.Removed}"
                + $" / 出力だけ {comparison.OutputOnly}";

            Message = comparison.HasSourceChanges
                ? string.Empty
                // **「同じ」と言い切らない。** 出力は見ていない。
                : "本文に違いはありません（出力と実行回数は見ていません）。";

            if (Tab is { } tab)
            {
                tab.Title = $"{leftName} ↔ {rightName}（ノート）";
            }
        }
        // **壊れた .ipynb は JSON として読めない。** ここを捕まえないと
        // 例外がコマンドの catch-all まで抜け、画面には「読んでいます…」が
        // 出たまま残る（止まっているように見えない）。構造比較と同じく、
        // どこが悪いかまで出す。
        catch (StructuredParseException error)
        {
            Cells.Clear();
            Message = error.Message;
            Summary = string.Empty;
        }
        catch (Exception error) when (error is IOException or InvalidDataException
                                        or UnauthorizedAccessException)
        {
            Cells.Clear();
            Message = error.Message;
            Summary = string.Empty;
        }
        finally
        {
            Busy = false;
        }
    }

    /// <summary>
    /// 本文だけを取り出して、行単位の比較で開く。
    ///
    /// **セル単位で足りないときの逃げ道。** 行内の差分や、意味的な
    /// 対応付けはあちらの画面にしか無い。
    /// </summary>
    private Task OpenAsTextAsync()
    {
        _shell.ShowExtractedText(LeftPath, RightPath);
        return Task.CompletedTask;
    }
}
