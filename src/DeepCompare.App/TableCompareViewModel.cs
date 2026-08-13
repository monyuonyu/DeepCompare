using System.Collections.ObjectModel;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>表のセル 1 つ。</summary>
public sealed class TableCellView(string text, bool changed)
{
    public string Text { get; } = text;

    /// <summary>この列が変わったか。**列単位で塗る。**</summary>
    public bool Changed { get; } = changed;

    public IBrush Background => Changed
        ? Palette.Brush("BgChanged")
        : Brushes.Transparent;
}

/// <summary>表の行 1 組。</summary>
public sealed class TableRowView
{
    public TableRowView(TableRowComparison comparison, Table left, Table right)
    {
        Comparison = comparison;
        var changed = comparison.ChangedColumns.ToHashSet();

        if (comparison.Left is { } l && l < left.Rows.Count)
        {
            LeftCells = [.. left.Rows[l].Cells.Select(
                (cell, i) => new TableCellView(cell, changed.Contains(i)))];
            LeftNumber = l + 1;
        }
        if (comparison.Right is { } r && r < right.Rows.Count)
        {
            RightCells = [.. right.Rows[r].Cells.Select(
                (cell, i) => new TableCellView(cell, changed.Contains(i)))];
            RightNumber = r + 1;
        }
    }

    public TableRowComparison Comparison { get; }

    public IReadOnlyList<TableCellView> LeftCells { get; } = [];
    public IReadOnlyList<TableCellView> RightCells { get; } = [];

    public int LeftNumber { get; }
    public int RightNumber { get; }

    public bool HasLeft => Comparison.Left is not null;
    public bool HasRight => Comparison.Right is not null;

    /// <summary>行番号。**片方にしか無ければ空にする** — 0 と書くと位置に見える。</summary>
    public string LeftLabel => HasLeft ? LeftNumber.ToString() : string.Empty;
    public string RightLabel => HasRight ? RightNumber.ToString() : string.Empty;

    public string ChangeLabel => (HasLeft, HasRight) switch
    {
        (true, true) => Comparison.IsUnchanged ? "同じ" : "違う",
        (true, false) => "左だけ",
        _ => "右だけ",
    };

    public bool IsInteresting => !Comparison.IsUnchanged;

    /// <summary>
    /// 行の背景。
    ///
    /// **両方にある行は塗らない。** どの列が動いたかはセルの色で示すので、
    /// 行ごと同じ色で塗ると、その上に乗るセルの色が埋もれて見えなくなる
    /// （実際に画面で確かめて気づいた）。片方にしか無い行だけ、行として塗る。
    /// </summary>
    public IBrush Background => (HasLeft, HasRight) switch
    {
        (true, true) => Palette.Brush("CardBg"),
        (true, false) => Palette.Brush("BgRemoved"),
        _ => Palette.Brush("BgAdded"),
    };

    /// <summary>行をまとめて 1 本の文字列に。**画面が狭いときの逃げ道。**</summary>
    public string LeftText => string.Join(" | ", LeftCells.Select(c => c.Text));
    public string RightText => string.Join(" | ", RightCells.Select(c => c.Text));
}

/// <summary>
/// CSV / TSV を列単位で比べる画面。
///
/// **行を上から突き合わせない。** 1 行挿入されただけで以降が全部ずれる。
/// キー列を指定すれば並び順が違っても照合できる。
/// </summary>
public sealed class TableCompareViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    public TableCompareViewModel(ShellViewModel shell)
    {
        _shell = shell;
        CompareCommand = new RelayCommand(CompareAsync, () => !Busy);
        OpenAsTextCommand = new RelayCommand(
            OpenAsTextAsync, () => LeftPath.Length > 0 && RightPath.Length > 0);
    }

    public ShellViewModel Shell => _shell;
    public CompareTab? Tab { get; set; }

    public ObservableCollection<TableRowView> Rows { get; } = [];

    /// <summary>見出し。**左のものを使う。** 列が増減したら差分として出る。</summary>
    public ObservableCollection<string> Header { get; } = [];

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

    private string _keyColumns = string.Empty;

    /// <summary>
    /// 行を対応付けるキー列。列名をカンマで区切る。
    ///
    /// **空でも動く。** そのときは行全体を 1 つの文字列として、
    /// 通常の対応付けに流す（並び順の違いには弱くなる）。
    /// </summary>
    public string KeyColumns
    {
        get => _keyColumns;
        set => Set(ref _keyColumns, value);
    }

    private string _ignoredColumns = string.Empty;

    /// <summary>見ない列。更新時刻など、毎回動いて意味の無い列を外す。</summary>
    public string IgnoredColumns
    {
        get => _ignoredColumns;
        set => Set(ref _ignoredColumns, value);
    }

    private bool _showUnchanged;
    public bool ShowUnchanged
    {
        get => _showUnchanged;
        set
        {
            if (Set(ref _showUnchanged, value) && Rows.Count > 0)
            {
                CompareCommand.Execute(null);
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

    private async Task CompareAsync()
    {
        if (LeftPath.Length == 0 || RightPath.Length == 0)
        {
            return;
        }

        Busy = true;
        Message = "読んでいます…";
        try
        {
            var keyNames = Split(KeyColumns);
            var ignoredNames = Split(IgnoredColumns);

            var (comparison, leftName, rightName) = await Task.Run(() =>
            {
                var left = TableCompare.Parse(
                    File.ReadAllText(LeftPath), TableFormat.ForPath(LeftPath));
                var right = TableCompare.Parse(
                    File.ReadAllText(RightPath), TableFormat.ForPath(RightPath));

                var keys = Resolve(keyNames, left);
                var ignored = Resolve(ignoredNames, left);

                // **埋め込みは渡さない。** 表の比較で意味的な対応が効くのは
                // キーが無いときだけで、そのときはモデルの読み込み（数秒）が
                // 待ち時間として丸ごと乗る。要るなら CLI で指定できる。
                return (TableCompare.Compare(left, right, keys, ignored, embedder: null),
                        Path.GetFileName(LeftPath), Path.GetFileName(RightPath));
            });

            Header.Clear();
            foreach (var name in comparison.Left.Header)
            {
                Header.Add(name);
            }

            Rows.Clear();
            foreach (var row in comparison.Rows)
            {
                if (!ShowUnchanged && row.IsUnchanged)
                {
                    continue;
                }
                Rows.Add(new TableRowView(row, comparison.Left, comparison.Right));
            }

            Summary = $"違う {comparison.Different}"
                + $" / 左だけ {comparison.LeftOnly}"
                + $" / 右だけ {comparison.RightOnly}"
                + $"（左 {comparison.Left.Rows.Count} 行 / 右 {comparison.Right.Rows.Count} 行）";

            Message = comparison.Different == 0
                       && comparison.LeftOnly == 0 && comparison.RightOnly == 0
                ? "違いはありません。"
                : string.Empty;

            if (Tab is { } tab)
            {
                tab.Title = $"{leftName} ↔ {rightName}（表）";
            }
        }
        catch (Exception error) when (error is IOException or InvalidDataException
                                        or UnauthorizedAccessException
                                        or ArgumentException)
        {
            Rows.Clear();
            Header.Clear();
            Message = error.Message;
            Summary = string.Empty;
        }
        finally
        {
            Busy = false;
        }
    }

    /// <summary>カンマ区切りを列名の配列に。**空白は落とす。**</summary>
    private static string[] Split(string text)
        => text.Split(',', StringSplitOptions.RemoveEmptyEntries
                          | StringSplitOptions.TrimEntries);

    /// <summary>
    /// 列の指定を番号に直す。名前でも番号（1 始まり）でも受ける。
    ///
    /// **見つからないものは断る。** 黙って無視すると、キーを指定したつもりで
    /// 指定できておらず、並び順の違いで全行がずれた結果を見ることになる。
    /// </summary>
    private static List<int> Resolve(string[] names, Table table)
    {
        var result = new List<int>();
        foreach (var name in names)
        {
            if (int.TryParse(name, out var number))
            {
                result.Add(number - 1);
                continue;
            }
            var found = table.IndexOfColumn(name);
            if (found < 0)
            {
                throw new ArgumentException($"列が見つかりません: {name}");
            }
            result.Add(found);
        }
        return result;
    }

    private Task OpenAsTextAsync()
    {
        _shell.ShowText(LeftPath, RightPath);
        return Task.CompletedTask;
    }
}
