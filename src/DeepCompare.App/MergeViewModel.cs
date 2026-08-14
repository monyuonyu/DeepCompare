using System.Collections.ObjectModel;
using System.Text;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>競合をどう決めたか。</summary>
public enum ConflictChoice
{
    /// <summary>まだ決めていない。</summary>
    Undecided,
    Left,
    Right,

    /// <summary>両方を並べて残す。順序は左が先。</summary>
    Both,

    /// <summary>
    /// LLM の提案を採る。
    ///
    /// **他と並ぶ 1 つの選択肢として置く。** 自動では選ばれず、
    /// 押さなければ何も起きない。承認の粒度が競合 1 つ単位になる。
    /// </summary>
    Assist,

    /// <summary>どちらも採らない（祖先へ戻す）。</summary>
    Neither,
}

/// <summary>
/// マージ結果のまとまり 1 つ。競合なら人が決めるまで未決のまま。
///
/// **勝手に片方を採らない。** 選んだこと自体が見えなくなると、後から
/// 「なぜこうなっているのか」を追えない。
/// </summary>
public sealed class MergeRegionRow(MergeRegion region, int index) : ViewModelBase
{
    public MergeRegion Region { get; } = region;

    /// <summary>競合の通し番号。競合でなければ 0。</summary>
    public int Number { get; } = index;

    public bool IsConflict => Region.Source == MergeSource.Conflict;

    private ConflictChoice _choice = ConflictChoice.Undecided;
    public ConflictChoice Choice
    {
        get => _choice;
        set
        {
            if (Set(ref _choice, value))
            {
                OnPropertyChanged(nameof(ChoiceLabel));
                OnPropertyChanged(nameof(IsDecided));
                OnPropertyChanged(nameof(Background));
                OnPropertyChanged(nameof(ResultText));
            }
        }
    }

    public bool IsDecided => !IsConflict || Choice != ConflictChoice.Undecided;

    public string ChoiceLabel => Choice switch
    {
        ConflictChoice.Left => "左を採った",
        ConflictChoice.Right => "右を採った",
        ConflictChoice.Both => "両方を残した",
        ConflictChoice.Neither => "どちらも採らない",
        // **提案だと分かるように書く。** 後から履歴を見た人が、
        // これを人が書いたものと思わないように。
        ConflictChoice.Assist => "提案を採った",
        _ => "未決",
    };

    private string _assistText = string.Empty;

    /// <summary>
    /// LLM が出した解決案。**空なら選択肢として現れない。**
    /// </summary>
    public string AssistText
    {
        get => _assistText;
        set
        {
            if (Set(ref _assistText, value))
            {
                OnPropertyChanged(nameof(HasAssist));
                OnPropertyChanged(nameof(ResultText));
            }
        }
    }

    public bool HasAssist => AssistText.Length > 0;

    public string SourceLabel => Region.Source switch
    {
        MergeSource.Unchanged => "変化なし",
        MergeSource.Left => "左の変更",
        MergeSource.Right => "右の変更",
        MergeSource.Both => "同じ変更",
        MergeSource.Conflict => $"競合 {Number}",
        _ => string.Empty,
    };

    public string BaseText => Join(Region.BaseLines);
    public string LeftText => Join(Region.LeftLines);
    public string RightText => Join(Region.RightLines);

    /// <summary>採用される行。競合で未決なら空。</summary>
    public IReadOnlyList<string> Resolved => Region.Source == MergeSource.Conflict
        ? Choice switch
        {
            ConflictChoice.Left => Region.LeftLines,
            ConflictChoice.Right => Region.RightLines,
            ConflictChoice.Both => [.. Region.LeftLines, .. Region.RightLines],
            ConflictChoice.Neither => Region.BaseLines,
            ConflictChoice.Assist => AssistLines,
            _ => [],
        }
        : Region.Lines;

    /// <summary>
    /// 提案を行に割る。
    ///
    /// **末尾の空行を落とす。** モデルは最後に改行を足しがちで、
    /// そのまま採ると空行が 1 つ増える（差分としては見えにくい変化）。
    /// </summary>
    private IReadOnlyList<string> AssistLines
    {
        get
        {
            if (AssistText.Length == 0)
            {
                return [];
            }
            var lines = AssistText.Replace("\r\n", "\n").Split('\n').ToList();
            while (lines.Count > 0 && lines[^1].Length == 0)
            {
                lines.RemoveAt(lines.Count - 1);
            }
            return lines;
        }
    }

    public string ResultText => Join(Resolved);

    public IBrush Background => Region.Source switch
    {
        MergeSource.Conflict => Choice == ConflictChoice.Undecided
            ? Palette.Brush("BgRemoved")
            : Palette.Brush("BgChanged"),
        MergeSource.Left or MergeSource.Right or MergeSource.Both => Palette.Brush("BgAdded"),
        _ => Palette.Brush("CardBg"),
    };

    /// <summary>変化なしのまとまりは畳んで良い。差分だけ見たいときに邪魔になる。</summary>
    public bool IsInteresting => Region.Source != MergeSource.Unchanged;

    private static string Join(IReadOnlyList<string> lines)
        => lines.Count == 0 ? "（空）" : string.Join('\n', lines);
}

/// <summary>
/// 3 方向マージの画面。
///
/// 競合は**人が決めるまで未決のまま置く**。既定で片方を採ると、決めたことが
/// 記録に残らない。書き出しは禁じないが、未決の競合は git 風の印を付けて残す
/// ので、決めていないことが結果から消えることはない。
/// </summary>
public sealed class MergeViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    /// <summary>自分が乗っているタブ。見出しを比較の中身に合わせて書き換える。</summary>
    public CompareTab? Tab { get; set; }

    /// <summary>起動画面へ戻るなど、画面をまたぐ操作。</summary>
    public ShellViewModel Shell => _shell;
    private DecodedText? _baseSource;

    public MergeViewModel(ShellViewModel shell)
    {
        _shell = shell;

        MergeCommand = new RelayCommand(MergeAsync, () => !_busy);
        SaveCommand = new RelayCommand(SaveAsync, () => Regions.Count > 0);
        TakeLeftCommand = new RelayCommand<MergeRegionRow>(row => Decide(row, ConflictChoice.Left));
        TakeRightCommand = new RelayCommand<MergeRegionRow>(row => Decide(row, ConflictChoice.Right));
        TakeBothCommand = new RelayCommand<MergeRegionRow>(row => Decide(row, ConflictChoice.Both));
        TakeNeitherCommand = new RelayCommand<MergeRegionRow>(row => Decide(row, ConflictChoice.Neither));
        TakeAllLeftCommand = new RelayCommand(() => DecideAll(ConflictChoice.Left));
        TakeAllRightCommand = new RelayCommand(() => DecideAll(ConflictChoice.Right));
        TakeAssistCommand = new RelayCommand<MergeRegionRow>(
            row => Decide(row, ConflictChoice.Assist), row => row.HasAssist);
        AskAssistCommand = new RelayCommand<MergeRegionRow>(
            AskAssistAsync, row => AssistAvailable && row.IsConflict && !_busy);
    }

    public ObservableCollection<MergeRegionRow> Regions { get; } = [];

    public RelayCommand MergeCommand { get; }
    public RelayCommand SaveCommand { get; }
    public RelayCommand<MergeRegionRow> TakeLeftCommand { get; }
    public RelayCommand<MergeRegionRow> TakeRightCommand { get; }
    public RelayCommand<MergeRegionRow> TakeBothCommand { get; }
    public RelayCommand<MergeRegionRow> TakeNeitherCommand { get; }
    public RelayCommand TakeAllLeftCommand { get; }
    public RelayCommand TakeAllRightCommand { get; }

    /// <summary>解決案を求める。**競合 1 つずつ。** まとめては聞かない。</summary>
    public RelayCommand<MergeRegionRow> AskAssistCommand { get; }

    /// <summary>提案を採る。**他の選択肢と同じ扱いで、押さなければ何も起きない。**</summary>
    public RelayCommand<MergeRegionRow> TakeAssistCommand { get; }

    private Assist.AssistSettings? _assistSettings;

    private Assist.AssistSettings AssistSettings => _assistSettings ??= LoadAssistSettings();

    private static Assist.AssistSettings LoadAssistSettings()
    {
        var saved = SessionStore.Default.LoadFile();
        return new Assist.AssistSettings
        {
            Endpoint = Environment.GetEnvironmentVariable(
                AssistCli.EndpointEnvironmentVariable) ?? saved.AssistEndpoint,
            Model = Environment.GetEnvironmentVariable(
                AssistCli.ModelEnvironmentVariable) ?? saved.AssistModel,
            ApiKey = Environment.GetEnvironmentVariable(AssistCli.ApiKeyEnvironmentVariable),
            AllowResolutionProposals = saved.AssistAllowResolution,
        };
    }

    /// <summary>
    /// 解決案を出せるか。
    ///
    /// **接続先だけでは足りない。** 解決案は意味を取り違えると害になる生成で、
    /// 説明や分類とは性質が違う。設定で明示的に許すまで出さない。
    /// </summary>
    public bool AssistAvailable
        => AssistSettings.IsConfigured && AssistSettings.AllowResolutionProposals;

    private async Task AskAssistAsync(MergeRegionRow row)
    {
        if (!AssistAvailable || !row.IsConflict)
        {
            return;
        }

        Busy = true;
        Message = $"競合 {row.Number} の案を聞いています…";
        try
        {
            using var client = new Assist.ChatClient(AssistSettings);
            var assistant = new Assist.GitAssistant(client);

            var proposal = await assistant.ProposeResolutionAsync(
                AssistSettings,
                Path.GetFileName(LeftPath),
                row.LeftText, row.RightText,
                row.BaseText is "（空）" ? null : row.BaseText);

            row.AssistText = proposal.Trim();

            // **勝手に選ばない。** 出しただけで、採るかどうかは人が決める。
            Message = row.HasAssist
                ? $"競合 {row.Number} の案が届きました。採るかどうかは選んでください。"
                : $"競合 {row.Number} について案は出ませんでした。";
            TakeAssistCommand.Raise();
        }
        catch (Exception error) when (error is Assist.AssistException
                                        or InvalidOperationException
                                        or ArgumentException)
        {
            // **マージの操作まで止めない。**
            Message = error.Message;
        }
        finally
        {
            Busy = false;
        }
    }

    private string _basePath = string.Empty;
    public string BasePath
    {
        get => _basePath;
        set => Set(ref _basePath, value);
    }

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
        private set { if (Set(ref _busy, value)) { MergeCommand.Raise(); } }
    }

    private bool _showUnchanged;
    /// <summary>変化なしのまとまりも出すか。既定は隠す。</summary>
    public bool ShowUnchanged
    {
        get => _showUnchanged;
        set { if (Set(ref _showUnchanged, value)) { Rebuild(); } }
    }

    /// <summary>未決の競合が残っているか。書き出すときに印を付けるかの判断に使う。</summary>
    public bool HasUndecided => Regions.Any(r => r.IsConflict && !r.IsDecided);

    public string DecisionText
    {
        get
        {
            var conflicts = Regions.Count(r => r.IsConflict);
            if (conflicts == 0)
            {
                return Regions.Count == 0 ? string.Empty : "競合はありません。";
            }
            var decided = Regions.Count(r => r.IsConflict && r.IsDecided);
            return $"競合 {conflicts} 件のうち {decided} 件を決めました。";
        }
    }

    private ThreeWayResult? _result;

    /// <summary>
    /// 中身の供給。null ならファイルとして読む。
    ///
    /// **git の競合を解くために要る。** 索引に積まれた 3 つの中身は
    /// ファイルとして存在しないので、一時ファイルへ書き出すことになる。
    /// そうすると後始末が要るうえ、画面にその名前が出てしまう。
    /// </summary>
    public Func<string, byte[]>? ContentLoader { get; set; }

    /// <summary>
    /// 書き出し先が決まっているときの受け取り手。null なら保存先を尋ねる。
    ///
    /// git の競合では、書き出す場所は競合しているファイルそのもので、
    /// 書いた後に索引へ載せるところまでが一続きになる。
    /// </summary>
    public Func<IReadOnlyList<string>, Task>? SaveHandler { get; set; }

    /// <summary>書き出しのボタンに出す言葉。用途で変わる。</summary>
    private string _saveLabel = "書き出す";
    public string SaveLabel
    {
        get => _saveLabel;
        set => Set(ref _saveLabel, value);
    }

    internal async Task MergeAsync()
    {
        if (BasePath.Length == 0 || LeftPath.Length == 0 || RightPath.Length == 0)
        {
            Message = "祖先・左・右の 3 つを指定してください。";
            return;
        }

        Busy = true;
        Message = string.Empty;
        Regions.Clear();

        try
        {
            var (basePath, leftPath, rightPath) = (BasePath, LeftPath, RightPath);
            var result = await Task.Run(() =>
            {
                var load = ContentLoader ?? File.ReadAllBytes;
                var ancestor = TextDecoder.Decode(load(basePath));
                var left = TextDecoder.Decode(load(leftPath));
                var right = TextDecoder.Decode(load(rightPath));
                // 埋め込みは使わない。マージの判定は git と揃えたいので、
                // 意味的な対応付けを持ち込むと結果が食い違う。
                return (ancestor, merged: ThreeWayMerge.Merge(ancestor, left, right));
            });

            _baseSource = result.ancestor;
            _result = result.merged;
            Rebuild();

            // **書き出しは禁じない。** git のマージ中に、印を付けたまま保存して
            // 後で続きをやりたいことがある。決めていないことが結果から消えない
            // ようにするのが目的なので、印を残せば足りる。
            Summary = _result.HasConflicts
                ? $"{_result.ConflictCount} 件の競合があります。決めていないものは印を付けて書き出します。"
                : "競合はありません。そのまま書き出せます。";
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

    private void Rebuild()
    {
        if (_result is null)
        {
            return;
        }

        // 決めた内容は作り直しても保つ。表示の切り替えで決定が消えると腹が立つ。
        var decisions = Regions
            .Where(r => r.IsConflict)
            .ToDictionary(r => r.Number, r => r.Choice);

        Regions.Clear();
        var conflictNumber = 0;
        foreach (var region in _result.Regions)
        {
            var number = region.Source == MergeSource.Conflict ? ++conflictNumber : 0;
            if (region.Source == MergeSource.Unchanged && !ShowUnchanged)
            {
                continue;
            }
            var row = new MergeRegionRow(region, number);
            if (number > 0 && decisions.TryGetValue(number, out var choice))
            {
                row.Choice = choice;
            }
            Regions.Add(row);
        }
        NotifyDecisionChanged();
    }

    private Task Decide(MergeRegionRow row, ConflictChoice choice)
    {
        row.Choice = choice;
        NotifyDecisionChanged();
        return Task.CompletedTask;
    }

    private Task DecideAll(ConflictChoice choice)
    {
        foreach (var row in Regions.Where(r => r.IsConflict))
        {
            row.Choice = choice;
        }
        NotifyDecisionChanged();
        return Task.CompletedTask;
    }

    private void NotifyDecisionChanged()
    {
        OnPropertyChanged(nameof(HasUndecided));
        OnPropertyChanged(nameof(DecisionText));
        SaveCommand.Raise();
    }

    internal async Task SaveAsync()
    {
        if (_result is null)
        {
            return;
        }

        // 未決が残っている場合は、印を付けた形でしか書き出さない。
        // 黙って片方を採ると、決めていないことが結果から消える。
        var undecided = Regions.Count(r => r.IsConflict && !r.IsDecided);

        // 書き出し先が決まっている場合（git の競合）は尋ねない。
        string? path = null;
        if (SaveHandler is null)
        {
            path = await _shell.PickSavePath("マージ結果を書き出す", "merged.txt");
            if (path is null)
            {
                return;
            }
        }

        var lines = new List<string>();
        var conflictNumber = 0;
        foreach (var region in _result.Regions)
        {
            if (region.Source != MergeSource.Conflict)
            {
                lines.AddRange(region.Lines);
                continue;
            }

            conflictNumber++;
            var row = Regions.FirstOrDefault(r => r.IsConflict && r.Number == conflictNumber);
            if (row is { IsDecided: true })
            {
                lines.AddRange(row.Resolved);
                continue;
            }

            // 決めていない競合は git 風の印で囲んで残す。
            lines.Add("<<<<<<< 左");
            lines.AddRange(region.LeftLines);
            lines.Add("=======");
            lines.AddRange(region.RightLines);
            lines.Add(">>>>>>> 右");
        }

        if (SaveHandler is { } handler)
        {
            // **未決が残ったまま索引へ載せない。** 印の付いた行がそのまま
            // コミットに入る事故は、git を使っていて一番起きやすい失敗。
            if (undecided > 0)
            {
                Message = $"**未決の競合が {undecided} 件あります。** 全部決めてから確定してください。";
                return;
            }
            await handler(lines);
            return;
        }

        // 符号化と改行は祖先に合わせる。読んだときの形を保つ。
        var bytes = _baseSource is { } source
            ? TextEncoder.Encode(lines, source)
            : new UTF8Encoding(false).GetBytes(string.Join('\n', lines));
        await File.WriteAllBytesAsync(path!, bytes);

        Message = undecided > 0
            ? $"{path} へ書き出しました。**未決の競合 {undecided} 件は印を付けたまま残しています。**"
            : $"{path} へ書き出しました。";
    }
}
