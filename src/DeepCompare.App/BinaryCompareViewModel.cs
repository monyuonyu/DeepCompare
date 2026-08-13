using System.Collections.ObjectModel;
using Avalonia.Controls.Documents;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>16 進表示の 1 行。</summary>
public sealed class HexRowView
{
    public HexRowView(HexRow row)
    {
        Row = row;
        Offset = (row.LeftOffset ?? row.RightOffset ?? 0).ToString("X8");

        LeftAscii = row.Ascii(left: true);
        RightAscii = row.Ascii(left: false);

        // 違うバイトだけ色を変える。行ごと塗ると、16 バイトのうち 1 つだけ
        // 違う場合に「どこが」が分からない。
        LeftHex = Build(row, left: true);
        RightHex = Build(row, left: false);

        (LeftBackground, RightBackground) = (row.LeftOffset, row.RightOffset) switch
        {
            (not null, not null) when row.IsUnchanged => (Brushes.Transparent, Brushes.Transparent),
            (not null, not null) => (Palette.Brush("BgChanged"), Palette.Brush("BgChanged")),
            (not null, null) => (Palette.Brush("BgRemoved"), Palette.Gap()),
            (null, not null) => (Palette.Gap(), Palette.Brush("BgAdded")),
            _ => (Brushes.Transparent, Brushes.Transparent),
        };
    }

    public HexRow Row { get; }
    public string Offset { get; }
    public string LeftAscii { get; }
    public string RightAscii { get; }
    public InlineCollection LeftHex { get; }
    public InlineCollection RightHex { get; }
    public IBrush LeftBackground { get; }
    public IBrush RightBackground { get; }

    private static InlineCollection Build(HexRow row, bool left)
    {
        var inlines = new InlineCollection();
        var bytes = left ? row.LeftBytes : row.RightBytes;
        if (bytes.Count == 0)
        {
            return inlines;
        }

        var normal = Palette.Brush("FgNormal");
        var changed = Palette.Brush("FgInline");

        for (var i = 0; i < bytes.Count; i++)
        {
            if (i > 0)
            {
                inlines.Add(new Run(i % 8 == 0 ? "  " : " ") { Foreground = normal });
            }
            inlines.Add(new Run(bytes[i].ToString("X2"))
            {
                Foreground = row.ChangedColumns.Contains(i) ? changed : normal,
                FontWeight = row.ChangedColumns.Contains(i) ? FontWeight.Bold : FontWeight.Normal,
            });
        }
        return inlines;
    }
}

/// <summary>
/// バイト列としての比較（16 進）。
///
/// **テキストとして読めないファイルのための出口。** 実行ファイル、画像、独自形式。
/// テキスト比較に掛けると符号化の推定が外れて意味の無い差分が出るか、そもそも
/// 読めない。
/// </summary>
public sealed class BinaryCompareViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    /// <summary>自分が乗っているタブ。</summary>
    public CompareTab? Tab { get; set; }

    public BinaryCompareViewModel(ShellViewModel shell)
    {
        _shell = shell;
        CompareCommand = new RelayCommand(CompareAsync, () => !_busy);
        OpenAsTextCommand = new RelayCommand(() =>
        {
            _shell.ShowText(LeftPath, RightPath);
            return Task.CompletedTask;
        }, () => LeftPath.Length > 0 && RightPath.Length > 0);
    }

    public ObservableCollection<HexRowView> Rows { get; } = [];

    public RelayCommand CompareCommand { get; }
    public RelayCommand OpenAsTextCommand { get; }

    /// <summary>テーマの切り替えなど、画面をまたぐ操作。</summary>
    public ShellViewModel Shell => _shell;

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
        private set { if (Set(ref _busy, value)) { CompareCommand.Raise(); } }
    }

    private bool _changesOnly = true;
    /// <summary>違う行だけ出す。既定で絞る。16 進は一致行まで並べると長すぎる。</summary>
    public bool ChangesOnly
    {
        get => _changesOnly;
        set { if (Set(ref _changesOnly, value)) { Rebuild(); } }
    }

    private BinaryComparison? _comparison;

    private async Task CompareAsync()
    {
        if (LeftPath.Length == 0 || RightPath.Length == 0)
        {
            Message = "左右のファイルを指定してください。";
            return;
        }

        Busy = true;
        Message = string.Empty;
        Rows.Clear();

        try
        {
            var (left, right) = (LeftPath, RightPath);
            var (comparison, truncated) = await Task.Run(
                () => BinaryCompare.CompareFiles(left, right));

            _comparison = comparison;
            Rebuild();

            Summary = comparison.Identical
                ? $"同じ内容です（{comparison.LeftLength:N0} バイト）"
                : $"{comparison.DifferentRows:N0} 行が違います"
                  + $"（左 {comparison.LeftLength:N0} / 右 {comparison.RightLength:N0} バイト）";

            if (truncated)
            {
                // **切り捨てたことは必ず出す。** 黙って途中までを比べると、
                // 「差分なし」が「先頭 64MB に差分なし」の意味になってしまう。
                Message = "大きいので先頭 64MB だけを比べています。";
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Message = error.Message;
        }
        finally
        {
            Busy = false;
        }
    }

    private void Rebuild()
    {
        Rows.Clear();
        if (_comparison is null)
        {
            return;
        }
        foreach (var row in _comparison.Rows)
        {
            if (!ChangesOnly || !row.IsUnchanged)
            {
                Rows.Add(new HexRowView(row));
            }
        }
    }
}
