using System.Collections.ObjectModel;
using Avalonia.Controls;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>変更ファイル 1 件の表示用。</summary>
public sealed class GitFileRow(GitFileStatus status)
{
    public GitFileStatus Status { get; } = status;

    public string Path => Status.Path;

    public string Display => Status.OriginalPath is { Length: > 0 } original
        ? $"{original} → {Status.Path}"
        : Status.Path;

    /// <summary>
    /// ファイル名だけ。**先に目に入るのはこれ。**
    /// パスをそのまま並べていたので、末尾を読むまで何のファイルか
    /// 分からなかった（VS Code は名前が先、場所は薄く後ろ）。
    /// </summary>
    public string FileName => System.IO.Path.GetFileName(Status.Path);

    /// <summary>置き場所。名前の後ろに薄く出す。</summary>
    public string Directory
    {
        get
        {
            var directory = System.IO.Path.GetDirectoryName(Status.Path) ?? string.Empty;
            return directory.Replace('\\', '/');
        }
    }

    /// <summary>
    /// 状態の 1 文字。**git の 2 文字ではなく、人が読む 1 文字にする。**
    /// 「.M」と書かれても、点が何を指すのかは git を知らないと読めない。
    /// </summary>
    public string Mark => Status switch
    {
        { IsConflicted: true } => "!",
        { Index: GitStatusCode.Untracked } => "U",
        { Index: GitStatusCode.Added } => "A",
        { Index: GitStatusCode.Deleted } => "D",
        { WorkTree: GitStatusCode.Deleted } => "D",
        { Index: GitStatusCode.Renamed } => "R",
        _ => "M",
    };

    /// <summary>状態の色。VS Code と同じ考え方で、足した／消した／直したを分ける。</summary>
    public IBrush MarkBrush => Status switch
    {
        { IsConflicted: true } => Palette.Brush("FgWarning"),
        { Index: GitStatusCode.Untracked } => Palette.Brush("FgAdded"),
        { Index: GitStatusCode.Added } => Palette.Brush("FgAdded"),
        { Index: GitStatusCode.Deleted } => Palette.Brush("FgRemoved"),
        { WorkTree: GitStatusCode.Deleted } => Palette.Brush("FgRemoved"),
        _ => Palette.Brush("FgChanged"),
    };

    /// <summary>git の 2 文字表記。**ツールチップにだけ残す。**</summary>
    public string RawMark => Status.IsConflicted ? "UU" : $"{Code(Status.Index)}{Code(Status.WorkTree)}";

    public string KindLabel => Status switch
    {
        { IsConflicted: true } => "競合",
        { Index: GitStatusCode.Untracked } => "未追跡",
        { IsStaged: true, IsDirty: true } => "stage 済み＋変更あり",
        { IsStaged: true } => "stage 済み",
        _ => "未 stage",
    };

    public IBrush Background => Status switch
    {
        { IsConflicted: true } => Palette.Brush("BgRemoved"),
        { Index: GitStatusCode.Untracked } => Palette.Brush("BgAdded"),
        { IsStaged: true } => Palette.Brush("BgChanged"),
        _ => Palette.Brush("CardBg"),
    };

    /// <summary>削除されたファイルは開いても中身が無い。押せないようにする。</summary>
    public bool CanOpen => Status.Index != GitStatusCode.Deleted
        && Status.WorkTree != GitStatusCode.Deleted
        && Status.Index != GitStatusCode.Untracked;

    /// <summary>索引へ載せられるか。既に載っていて作業ツリーがきれいなら何も起きない。</summary>
    public bool CanStage => Status.IsDirty || Status.Index == GitStatusCode.Untracked;

    /// <summary>索引から降ろせるか。未追跡のものは索引に載っていない。</summary>
    public bool CanUnstage => Status.IsStaged;

    private static char Code(GitStatusCode code) => code switch
    {
        GitStatusCode.Unchanged => '.',
        GitStatusCode.Modified => 'M',
        GitStatusCode.Added => 'A',
        GitStatusCode.Deleted => 'D',
        GitStatusCode.Renamed => 'R',
        GitStatusCode.Copied => 'C',
        GitStatusCode.TypeChanged => 'T',
        GitStatusCode.Untracked => '?',
        GitStatusCode.Ignored => '!',
        GitStatusCode.Unmerged => 'U',
        _ => ' ',
    };
}

/// <summary>枝 1 本の表示用。</summary>
public sealed class GitBranchRow(GitBranch branch)
{
    public GitBranch Branch { get; } = branch;

    public string Name => Branch.Name;
    public bool IsCurrent => Branch.IsCurrent;

    /// <summary>追跡先との進み遅れ。**どちらも 0 なら何も出さない。**</summary>
    public string TrackText => (Branch.Ahead, Branch.Behind) switch
    {
        (0, 0) => string.Empty,
        (var a, 0) => $"↑{a}",
        (0, var b) => $"↓{b}",
        var (a, b) => $"↑{a} ↓{b}",
    };

    public bool HasTrack => TrackText.Length > 0;
}

/// <summary>枝やタグの札 1 つ。</summary>
public sealed class GitRefChip(GitRef reference)
{
    public string Text => reference.Name;

    /// <summary>種類ごとに色を変える。**文字だけだと、どれが遠隔か分からない。**</summary>
    public IBrush Background => Palette.Brush(reference.Kind switch
    {
        GitRefKind.Tag => "RefTagBg",
        GitRefKind.Remote => "RefRemoteBg",
        GitRefKind.Head => "RefHeadBg",
        _ => reference.IsCurrent ? "RefCurrentBg" : "RefLocalBg",
    });

    public IBrush Foreground => Palette.Brush(reference.Kind switch
    {
        GitRefKind.Tag => "RefTagFg",
        GitRefKind.Remote => "RefRemoteFg",
        GitRefKind.Head => "RefHeadFg",
        _ => reference.IsCurrent ? "RefCurrentFg" : "RefLocalFg",
    });

    /// <summary>いま居る枝だけ太字にする。</summary>
    public FontWeight Weight => reference.IsCurrent ? FontWeight.SemiBold : FontWeight.Normal;

    public Geometry? Icon => Icons.Get(reference.Kind switch
    {
        GitRefKind.Tag => "IconTag",
        GitRefKind.Remote => "IconCloud",
        _ => "IconBranch",
    });
}

/// <summary>
/// コミットで変わったファイル 1 件。
///
/// 作業ツリーの行（<see cref="GitFileRow"/>）とは別に持つ。あちらは索引と
/// 作業ツリーの 2 段を表すが、こちらに段は無い。**同じ型にすると
/// 「stage 済み」のような、ここでは意味を成さない札が出る。**
/// </summary>
public sealed class GitCommitFileRow(GitFileStatus status)
{
    public string Path => status.Path;

    /// <summary>
    /// ファイル名だけ。**これを先に、大きく出す。**
    ///
    /// パスをそのまま 1 本の文字列で出すと、幅が足りないときに末尾が落ちる。
    /// 落ちるのは肝心のファイル名の方で、残るのは全部同じ
    /// <c>csharp/src/DeepCompare.App/…</c> という、区別に使えない部分になる。
    /// </summary>
    public string Name => System.IO.Path.GetFileName(status.Path);

    /// <summary>置き場所。名前の後ろに薄く添える。切れても困らない。</summary>
    public string Folder
    {
        get
        {
            var directory = System.IO.Path.GetDirectoryName(status.Path)?.Replace('\\', '/') ?? string.Empty;
            if (status.OriginalPath is { Length: > 0 } original)
            {
                // 名前が変わったものは、どこから来たかを添える。
                return directory.Length > 0 ? $"{directory} ← {original}" : $"← {original}";
            }
            return directory;
        }
    }

    public bool HasFolder => Folder.Length > 0;

    public string Display => status.OriginalPath is { Length: > 0 } original
        ? $"{original} → {status.Path}"
        : status.Path;

    public string Mark => status.Index switch
    {
        GitStatusCode.Added => "A",
        GitStatusCode.Deleted => "D",
        GitStatusCode.Renamed => "R",
        GitStatusCode.Copied => "C",
        _ => "M",
    };

    public IBrush MarkColour => Palette.Brush(status.Index switch
    {
        GitStatusCode.Added => "GitAdded",
        GitStatusCode.Deleted => "GitRemoved",
        _ => "GitChanged",
    });
}

/// <summary>コミット 1 件の表示用。</summary>
public sealed class GitCommitRow(GitCommit commit)
{
    public GitCommit Commit { get; } = commit;

    public string ShortHash => Commit.ShortHash;
    public string Subject => Commit.Subject;
    public string Author => Commit.Author;
    public string When => Commit.When.ToLocalTime().ToString("yyyy-MM-dd HH:mm");
    public string MergeNote => Commit.IsMerge ? "マージ" : string.Empty;
    public bool IsMerge => Commit.IsMerge;

    /// <summary>この行のグラフ。<see cref="GitGraph"/> が決める。</summary>
    public GraphRow? Graph { get; set; }

    /// <summary>
    /// グラフ列の幅。**行ごとに変えない。** 変えると列が揃わない。
    ///
    /// <see cref="GridLength"/> で持つ。数値のまま <c>ColumnDefinition.Width</c> へ
    /// 束縛すると**静かに既定（<c>1*</c>）へ落ちて**、列が半分ずつに割れる。
    /// </summary>
    public GridLength GraphWidth { get; set; } = new(CommitGraph.LaneWidth);

    public IReadOnlyList<GitRefChip> Refs { get; } =
        [.. commit.Refs.Select(r => new GitRefChip(r))];

    public bool HasRefs => Refs.Count > 0;

    /// <summary>
    /// 詳細に出す相対時刻。**「3 日前」の方が、日付より早く分かる。**
    /// 1 か月を超えたら日付そのものを出す（「87 日前」は数え直しが要る）。
    /// </summary>
    public string Ago
    {
        get
        {
            var span = DateTimeOffset.Now - Commit.When;
            if (span < TimeSpan.Zero)
            {
                return "これから";     // 時計がずれている
            }
            if (span.TotalMinutes < 1)
            {
                return "たった今";
            }
            if (span.TotalHours < 1)
            {
                return $"{(int)span.TotalMinutes} 分前";
            }
            if (span.TotalDays < 1)
            {
                return $"{(int)span.TotalHours} 時間前";
            }
            if (span.TotalDays < 31)
            {
                return $"{(int)span.TotalDays} 日前";
            }
            return Commit.When.ToLocalTime().ToString("yyyy-MM-dd");
        }
    }
}

/// <summary>
/// Git の画面。作業ツリーと履歴。
///
/// **差分の表示は既存のテキスト比較画面に任せる。** git 自身の diff を出すのでは
/// なく、こちらの比較エンジンに掛けるので、意味的な行の対応付けがそのまま効く。
/// ここが SourceTree との差になる部分。
/// </summary>
public sealed class GitViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    /// <summary>自分が乗っているタブ。見出しを比較の中身に合わせて書き換える。</summary>
    public CompareTab? Tab { get; set; }

    /// <summary>起動画面へ戻るなど、画面をまたぐ操作。</summary>
    public ShellViewModel Shell => _shell;
    private GitRepository? _repository;

    public GitViewModel(ShellViewModel shell, string path)
    {
        _shell = shell;
        _path = path;

        RefreshCommand = new RelayCommand(RefreshAsync, () => !_busy);
        OpenFileCommand = new RelayCommand<GitFileRow>(
            row => { OpenFile(row); return Task.CompletedTask; }, row => row.CanOpen);
        // 押しても何も起きないボタンを押せる状態で置かない。
        StageCommand = new RelayCommand<GitFileRow>(
            row => StageAsync(row, stage: true), row => row.CanStage);
        UnstageCommand = new RelayCommand<GitFileRow>(
            row => StageAsync(row, stage: false), row => row.CanUnstage);
        CommitCommand = new RelayCommand(CommitAsync, () => CanCommit);

        // **リポジトリは遅延で渡す。** 画面を作る時点ではまだ開いていない。
        Assist = new GitAssistViewModel(() => _repository)
        {
            // 草案は入力欄へ入れるだけ。**そのまま記録はしない。**
            CommitDraftHandler = draft => CommitMessage = draft,
        };
        AmendCommand = new RelayCommand(() => CommitAsync(amend: true), () => Commits.Count > 0);
        FetchCommand = new RelayCommand(() => RemoteAsync("取得", r => r.Fetch()));
        PullCommand = new RelayCommand(() => RemoteAsync("取り込み", r => r.Pull()));
        PushCommand = new RelayCommand(() => RemoteAsync("送信", r => r.Push()));
        // いま居る枝へは切り替えられない（押しても何も起きない）。
        SwitchBranchCommand = new RelayCommand<GitBranchRow>(
            row => SwitchAsync(row.Name), row => !row.IsCurrent);
        CreateBranchCommand = new RelayCommand(CreateBranchAsync,
            () => NewBranchName.Trim().Length > 0);
        OpenCommitCommand = new RelayCommand<GitCommitRow>(
            row => { OpenCommit(row); return Task.CompletedTask; });
        OpenCommitFileCommand = new RelayCommand<GitCommitFileRow>(
            file => { OpenCommitFile(SelectedCommit, file); return Task.CompletedTask; });

        CheckoutCommand = new RelayCommand<GitCommitRow>(
            row => WriteAsync(
                $"{row.ShortHash} へ移りました。枝から離れた状態です",
                r => r.Checkout(row.Commit.Hash)));
        BranchHereCommand = new RelayCommand<GitCommitRow>(
            row => WriteAsync(
                $"{NewBranchName.Trim()} を {row.ShortHash} から作りました",
                r => r.CreateBranch(NewBranchName.Trim(), row.Commit.Hash)),
            _ => NewBranchName.Trim().Length > 0);
        RevertCommand = new RelayCommand<GitCommitRow>(
            row => WriteAsync(
                $"{row.ShortHash} を打ち消すコミットを作りました",
                r => r.Revert(row.Commit.Hash, row.IsMerge)));
        CherryPickCommand = new RelayCommand<GitCommitRow>(
            row => WriteAsync(
                $"{row.ShortHash} の変更をいまの枝に載せました",
                r => r.CherryPick(row.Commit.Hash)));
        StagePartlyCommand = new RelayCommand<GitFileRow>(
            row => { StagePartly(row); return Task.CompletedTask; },
            // 索引に載っていて、作業ツリーが汚れているものだけ。未追跡は
            // 「索引の中身」が無いので、塊で分ける意味が無い。
            row => row.Status.IsDirty && row.Status.Index != GitStatusCode.Untracked
                   && !row.Status.IsConflicted);
        ResolveCommand = new RelayCommand<GitFileRow>(
            row => { Resolve(row); return Task.CompletedTask; },
            row => row.Status.IsConflicted);
        CopyHashCommand = new RelayCommand<GitCommitRow>(row => Copy(row.Commit.Hash));
        CopySubjectCommand = new RelayCommand<GitCommitRow>(row => Copy(row.Subject));
    }

    /// <summary>書き込み先。表示側から差し込む（ViewModel から画面に触らない）。</summary>
    public Action<string>? Clipboard { get; set; }

    private Task Copy(string text)
    {
        Clipboard?.Invoke(text);
        Message = $"写しました: {text}";
        return Task.CompletedTask;
    }

    /// <summary>
    /// 書き換える操作をまとめて実行し、終わったら読み直す。
    ///
    /// **成功したときだけ「できた」と言う。** git が断ったら、その理由を
    /// そのまま出す。ここで握り潰すと、何も起きていないのに成功に見える。
    /// </summary>
    private async Task WriteAsync(string done, Action<GitRepository> operation)
    {
        if (_repository is not { } repository || Busy)
        {
            return;
        }

        Busy = true;
        try
        {
            await Task.Run(() => operation(repository));
            Message = done;
        }
        catch (GitException error)
        {
            Message = error.Message;
            return;
        }
        finally
        {
            Busy = false;
        }

        await RefreshAsync();
    }

    /// <summary>グラフの列の間隔。描く側と同じ値を使う（別々に持つとずれる）。</summary>
    internal const double GraphLaneWidth = CommitGraph.LaneWidth;

    private GridLength _graphColumnWidth = new(GraphLaneWidth);

    /// <summary>グラフ列の幅。見出しと本体で同じ値を使い、列を揃える。</summary>
    public GridLength GraphColumnWidth
    {
        get => _graphColumnWidth;
        private set => Set(ref _graphColumnWidth, value);
    }

    public ObservableCollection<GitFileRow> Files { get; } = [];

    /// <summary>
    /// 索引に載っているもの／載っていないもの。
    ///
    /// **2 つに分ける。** 1 つの一覧に混ぜて色で示していたが、
    /// 「次のコミットに入るのはどれか」が一目で分からなかった。
    /// VS Code と同じ並びにする（上が載っているもの）。
    ///
    /// **同じファイルが両方に出ることがある。** 一部だけ stage して
    /// その後さらに直した場合で、git の見え方をそのまま出す。
    /// </summary>
    public ObservableCollection<GitFileRow> StagedFiles { get; } = [];
    public ObservableCollection<GitFileRow> UnstagedFiles { get; } = [];

    public bool HasStagedFiles => StagedFiles.Count > 0;
    public bool HasUnstagedFiles => UnstagedFiles.Count > 0;

    /// <summary>見出しに出す件数。**0 のときは出さない。**</summary>
    public string StagedCount => StagedFiles.Count.ToString();
    public string UnstagedCount => UnstagedFiles.Count.ToString();

    private void SplitFiles()
    {
        StagedFiles.Clear();
        UnstagedFiles.Clear();
        foreach (var row in Files)
        {
            if (row.Status.IsStaged)
            {
                StagedFiles.Add(row);
            }
            // **stage 済みでも、そのあと直していれば下にも出す。**
            if (!row.Status.IsStaged || row.Status.IsDirty)
            {
                UnstagedFiles.Add(row);
            }
        }
        OnPropertyChanged(nameof(HasStagedFiles));
        OnPropertyChanged(nameof(HasUnstagedFiles));
        OnPropertyChanged(nameof(StagedCount));
        OnPropertyChanged(nameof(UnstagedCount));
    }

    /// <summary>
    /// 変更が 1 件も無い。**空の枠をそのまま見せない。**
    /// 何も無いのか、まだ読んでいないのか、失敗したのかが区別できない。
    /// </summary>
    public bool IsClean => HasRepository && Files.Count == 0;

    public ObservableCollection<GitCommitRow> Commits { get; } = [];

    public RelayCommand RefreshCommand { get; }
    public RelayCommand<GitFileRow> OpenFileCommand { get; }
    public RelayCommand<GitFileRow> StageCommand { get; }
    public RelayCommand<GitFileRow> UnstageCommand { get; }
    public RelayCommand<GitCommitRow> OpenCommitCommand { get; }
    public RelayCommand<GitCommitFileRow> OpenCommitFileCommand { get; }
    public RelayCommand<GitFileRow> ResolveCommand { get; }
    public RelayCommand<GitFileRow> StagePartlyCommand { get; }
    public RelayCommand<GitCommitRow> CheckoutCommand { get; }
    public RelayCommand<GitCommitRow> BranchHereCommand { get; }
    public RelayCommand<GitCommitRow> RevertCommand { get; }
    public RelayCommand<GitCommitRow> CherryPickCommand { get; }
    public RelayCommand<GitCommitRow> CopyHashCommand { get; }
    public RelayCommand<GitCommitRow> CopySubjectCommand { get; }
    public RelayCommand CommitCommand { get; }
    public RelayCommand AmendCommand { get; }
    public RelayCommand FetchCommand { get; }
    public RelayCommand PullCommand { get; }
    public RelayCommand PushCommand { get; }
    public RelayCommand<GitBranchRow> SwitchBranchCommand { get; }
    public RelayCommand CreateBranchCommand { get; }

    public ObservableCollection<GitBranchRow> Branches { get; } = [];

    /// <summary>
    /// LLM 支援。**接続先が無ければ画面に出ない。**
    /// リポジトリは遅延で渡す — 画面を開いた時点ではまだ開いていない。
    /// </summary>
    public GitAssistViewModel Assist { get; }

    private string _commitMessage = string.Empty;

    /// <summary>コミットの説明。</summary>
    public string CommitMessage
    {
        get => _commitMessage;
        set
        {
            if (Set(ref _commitMessage, value))
            {
                CommitCommand.Raise();
            }
        }
    }

    private string _newBranchName = string.Empty;
    public string NewBranchName
    {
        get => _newBranchName;
        set
        {
            if (Set(ref _newBranchName, value))
            {
                CreateBranchCommand.Raise();
            }
        }
    }

    /// <summary>
    /// コミットできるか。
    ///
    /// **索引に何も載っていなければ押せない。** 押しても git が断るだけなので、
    /// その前に分かる方がよい。
    /// </summary>
    public bool CanCommit => _hasStaged && CommitMessage.Trim().Length > 0;

    private bool _hasStaged;

    private async Task CommitAsync() => await CommitAsync(amend: false);

    private async Task CommitAsync(bool amend)
    {
        if (_repository is not { } repository)
        {
            return;
        }

        var message = CommitMessage.Trim();
        if (message.Length == 0 && amend)
        {
            // 書き直しで空なら、直前の説明をそのまま使う。
            message = repository.LastMessage();
        }

        try
        {
            await Task.Run(() => repository.Commit(message, amend));
            CommitMessage = string.Empty;
            await RefreshAsync();
            Message = amend ? "直前のコミットを書き直しました。" : "コミットしました。";
        }
        catch (GitException error)
        {
            Message = error.Message;
        }
    }

    /// <summary>
    /// 遠隔とのやりとり。
    ///
    /// **認証が要るので、待たずに失敗させる**（GIT_TERMINAL_PROMPT=0）。
    /// 画面が固まったまま戻らないより、理由を出して終わる方がよい。
    /// </summary>
    private async Task RemoteAsync(string label, Func<GitRepository, string> action)
    {
        if (_repository is not { } repository)
        {
            return;
        }
        Busy = true;
        Message = $"{label}中…";
        try
        {
            var output = await Task.Run(() => action(repository));
            await RefreshAsync();
            Message = output.Length > 0 ? output : $"{label}が終わりました。";
        }
        catch (GitException error)
        {
            Message = error.Message;
        }
        finally
        {
            Busy = false;
        }
    }

    private async Task SwitchAsync(string branch)
    {
        if (_repository is not { } repository)
        {
            return;
        }
        try
        {
            await Task.Run(() => repository.Switch(branch));
            await RefreshAsync();
        }
        catch (GitException error)
        {
            // 作業ツリーが汚れていると git が止める。**その理由をそのまま出す。**
            Message = error.Message;
        }
    }

    private async Task CreateBranchAsync()
    {
        if (_repository is not { } repository)
        {
            return;
        }
        try
        {
            var name = NewBranchName.Trim();
            await Task.Run(() => repository.CreateBranch(name));
            NewBranchName = string.Empty;
            await RefreshAsync();
            Message = $"{name} を作って切り替えました。";
        }
        catch (GitException error)
        {
            Message = error.Message;
        }
    }

    private string _path;
    public string Path
    {
        get => _path;
        set => Set(ref _path, value);
    }

    private string _branch = string.Empty;
    public string Branch
    {
        get => _branch;
        private set => Set(ref _branch, value);
    }

    private string _root = string.Empty;
    public string Root
    {
        get => _root;
        private set => Set(ref _root, value);
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
                RefreshCommand.Raise();
            }
        }
    }

    public bool HasRepository => _repository is not null;

    /// <summary>
    /// 読み直す。
    ///
    /// git が入っていないのは異常ではない。その場合はここで理由を出して終わり、
    /// 画面ごと落ちたりはしない。
    /// </summary>
    public async Task RefreshAsync()
    {
        Busy = true;
        Message = string.Empty;

        try
        {
            if (GitRepository.Version() is null)
            {
                Message = "git が見つかりません。Git 機能を使うには git を入れてください。";
                return;
            }

            var path = Path;
            var repository = await Task.Run(() => GitRepository.Discover(path));
            if (repository is null)
            {
                Message = $"{path} は git リポジトリの中にありません。";
                Files.Clear();
                SplitFiles();
                Commits.Clear();
                return;
            }

            _repository = repository;
            OnPropertyChanged(nameof(HasRepository));
            Root = repository.Root;

            // 状態・枝・履歴をまとめて 1 回の作業スレッドで取る。
            var (files, branch, commits, branches, staged) = await Task.Run(() => (
                repository.Status().Where(f => f.Index != GitStatusCode.Ignored).ToList(),
                repository.CurrentBranch(),
                // **すべての枝を含める。** いま居る枝だけだと線が 1 本しか無く、
                // グラフにする意味が無い。
                repository.Log(100, all: true),
                repository.Branches(),
                repository.HasStagedChanges()));

            _hasStaged = staged;
            OnPropertyChanged(nameof(CanCommit));
            CommitCommand.Raise();
            AmendCommand.Raise();

            Branches.Clear();
            foreach (var b in branches)
            {
                Branches.Add(new GitBranchRow(b));
            }

            Branch = branch ?? "（切り離された HEAD）";

            Files.Clear();
            foreach (var file in files.OrderBy(f => f.Path, StringComparer.Ordinal))
            {
                Files.Add(new GitFileRow(file));
            }
            SplitFiles();

            Commits.Clear();
            SelectedCommit = null;

            // グラフの列を決める。**全部まとめて計算する。** 1 行だけ見ても
            // どこから来てどこへ行くかは決まらない（前後の行に依存する）。
            var graph = GitGraph.Build(commits);
            var width = graph.Count == 0 ? 1 : graph.Max(g => g.Width);
            var graphWidth = new GridLength(width * GraphLaneWidth);
            GraphColumnWidth = graphWidth;

            for (var i = 0; i < commits.Count; i++)
            {
                Commits.Add(new GitCommitRow(commits[i])
                {
                    Graph = graph[i],
                    // 列の幅は全行で揃える。**行ごとに変えると説明の左端が
                    // 揃わず、目が行を追えなくなる。**
                    GraphWidth = graphWidth,
                });
            }

            // 先頭を選んでおく。詳細の枠が空のままだと、何をすればいいか伝わらない。
            if (Commits.Count > 0)
            {
                SelectedCommit = Commits[0];
            }

            OnPropertyChanged(nameof(IsClean));

            Summary = files.Count == 0
                ? "作業ツリーはきれいです。"
                : $"{files.Count} 件（stage 済み {files.Count(f => f.IsStaged)}"
                  + $" / 未 stage {files.Count(f => f.IsDirty && f.Index != GitStatusCode.Untracked)}"
                  + $" / 未追跡 {files.Count(f => f.Index == GitStatusCode.Untracked)}"
                  + $" / 競合 {files.Count(f => f.IsConflicted)}）";
        }
        catch (GitException error)
        {
            Message = error.Message;
        }
        finally
        {
            Busy = false;
        }
    }

    /// <summary>
    /// HEAD の中身と作業ツリーの中身を比べる。
    ///
    /// 一時ファイルへ書き出さず、<see cref="TextCompareViewModel.ContentLoader"/> で
    /// 直接渡す。一時ファイルを使うと後始末が要るうえ、画面にその名前が出てしまう。
    /// </summary>
    private void OpenFile(GitFileRow row)
    {
        if (_repository is not { } repository)
        {
            return;
        }

        var relative = row.Status.Path;
        var absolute = System.IO.Path.Combine(repository.Root, relative);

        // 左は HEAD の中身。拡張子が残る形にして、構文強調をそのまま効かせる。
        _shell.ShowTextWith(
            $"HEAD:{relative}", absolute,
            path => path.StartsWith("HEAD:", StringComparison.Ordinal)
                ? repository.Show("HEAD", path[5..])
                : File.ReadAllBytes(path),
            leftReadOnly: true, rightReadOnly: false);
    }

    /// <summary>
    /// 索引と作業ツリーを並べ、塊ごとに索引へ載せる（hunk 単位の stage）。
    ///
    /// **git add -p の代わり。** あちらは端末で 1 塊ずつ y/n を答える形で、
    /// 前後の文脈が見えない。並べて見ながら選べる方が間違えにくい。
    /// </summary>
    private void StagePartly(GitFileRow row)
    {
        if (_repository is not { } repository)
        {
            return;
        }
        // 索引へ載せたら一覧を読み直す。**載ったことが画面に出ないと、
        // 押したのかどうか分からない。**
        _shell.ShowIndexAgainstWorkTree(repository, row.Status.Path, () => _ = RefreshAsync());
    }

    /// <summary>
    /// 競合を解く画面を開く。
    ///
    /// 索引に積まれた 3 つ（祖先・こちら・むこう）を渡す。**作業ツリーの
    /// ファイルは使わない。** そこには git が書いた印（&lt;&lt;&lt;&lt;&lt;&lt;&lt;）が
    /// 混ざっており、それを 3 方向マージに掛けても意味がない。
    /// </summary>
    private void Resolve(GitFileRow row)
    {
        if (_repository is not { } repository)
        {
            return;
        }

        var relative = row.Status.Path;
        var absolute = System.IO.Path.Combine(repository.Root, relative);

        _shell.ShowGitConflict(
            relative,
            name => repository.ConflictStage(relative, name switch
            {
                var n when n.StartsWith("共通の祖先:", StringComparison.Ordinal) => 1,
                var n when n.StartsWith("こちら:", StringComparison.Ordinal) => 2,
                _ => 3,
            }),
            async lines =>
            {
                // 改行は「こちら」に合わせる。祖先が無い競合（両側で追加）もある。
                var reference = TextDecoder.Decode(repository.ConflictStage(relative, 2));
                await File.WriteAllBytesAsync(absolute, TextEncoder.Encode(lines, reference));

                // **書いたら索引へ載せるところまでやる。** git は索引に載って
                // 初めて「解決した」と見なす。書くだけで止めると、競合したまま
                // に見え続ける。
                await Task.Run(() => repository.Stage(relative));
                Message = $"{relative} を解決して索引へ載せました。";
                await RefreshAsync();
            });
    }

    /// <summary>
    /// そのコミットで、そのファイルが何に変わったかを見る。最初の親と比べる。
    ///
    /// **どのファイルかは、選んだコミットの変更一覧から来る。** 以前は作業ツリー
    /// 側で選ばれているファイルに合わせていたが、そのコミットで触っていない
    /// ファイルを指していると「差分なし」が出るだけで、意味が無かった。
    /// </summary>
    private void OpenCommit(GitCommitRow row) => OpenCommitFile(row, SelectedCommitFile);

    private void OpenCommitFile(GitCommitRow? row, GitCommitFileRow? file)
    {
        if (_repository is not { } repository || row is null)
        {
            return;
        }
        if (row.Commit.Parents.Count == 0)
        {
            Message = "最初のコミットには親が無いので、比べる相手がありません。";
            return;
        }
        if (file is null)
        {
            Message = "下の一覧でファイルを選ぶと、そのコミットでの変化を出します。";
            return;
        }

        var hash = row.Commit.Hash;
        var parent = row.Commit.Parents[0];
        var relative = file.Path;

        _shell.ShowTextWith(
            $"{parent[..7]}:{relative}", $"{hash[..7]}:{relative}",
            path =>
            {
                var separator = path.IndexOf(':');
                var revision = separator < 0 ? "HEAD" : path[..separator];
                var file = separator < 0 ? path : path[(separator + 1)..];
                // 片側に存在しないことは異常ではない（作られた／消された）。空として扱う。
                return repository.Exists(revision, file) ? repository.Show(revision, file) : [];
            },
            leftReadOnly: true, rightReadOnly: true);
    }

    private GitCommitRow? _selectedCommit;

    /// <summary>
    /// 履歴で選ばれているコミット。選ぶたびに、下の詳細を読み直す。
    ///
    /// **一覧を引くときにまとめて取らない。** 100 件それぞれの変更ファイルを
    /// 先に取ると、git を 100 回起動することになる。見るのは 1 件なので、
    /// 選ばれてから取る。
    /// </summary>
    public GitCommitRow? SelectedCommit
    {
        get => _selectedCommit;
        set
        {
            if (Set(ref _selectedCommit, value))
            {
                OnPropertyChanged(nameof(HasSelectedCommit));
                _ = LoadCommitDetailAsync(value);
            }
        }
    }

    public bool HasSelectedCommit => _selectedCommit is not null;

    /// <summary>選んだコミットの変更ファイル。</summary>
    public ObservableCollection<GitCommitFileRow> CommitFiles { get; } = [];

    private GitCommitFileRow? _selectedCommitFile;
    public GitCommitFileRow? SelectedCommitFile
    {
        get => _selectedCommitFile;
        set => Set(ref _selectedCommitFile, value);
    }

    private string _commitBody = string.Empty;

    /// <summary>
    /// 説明の 2 行目より後ろ。1 行目（件名）は上に出ているので落とす。
    ///
    /// **同じ文を 2 回出さない。** 1 行だけのコミットでは全文＝件名なので、
    /// そのまま出すと真下に同じ文が並び、続きがあるように見える。
    /// </summary>
    public string CommitBody
    {
        get => _commitBody;
        private set => Set(ref _commitBody, value);
    }

    public bool HasCommitBody => _commitBody.Length > 0;

    /// <summary>
    /// 説明の欄の幅。**説明が無ければ 0 にして、ファイル一覧を全幅にする。**
    /// 隠すだけだと列は残り、左半分が空いたままになる。
    /// </summary>
    public GridLength BodyWidth => HasCommitBody ? new GridLength(1, GridUnitType.Star)
        : new GridLength(0);

    private static string BodyAfterSubject(string full)
    {
        var breakAt = full.IndexOf('\n');
        return breakAt < 0 ? string.Empty : full[(breakAt + 1)..].Trim('\n', '\r', ' ');
    }

    private string _commitParents = string.Empty;

    /// <summary>親のハッシュ。マージなら 2 つ以上並ぶ。</summary>
    public string CommitParents
    {
        get => _commitParents;
        private set => Set(ref _commitParents, value);
    }

    /// <summary>詳細の読み込みを 1 つに絞る。素早く選び替えたときに古い結果を出さない。</summary>
    private int _detailGeneration;

    private async Task LoadCommitDetailAsync(GitCommitRow? row)
    {
        var generation = ++_detailGeneration;

        CommitFiles.Clear();
        SelectedCommitFile = null;
        CommitBody = string.Empty;
        OnPropertyChanged(nameof(HasCommitBody));
        OnPropertyChanged(nameof(BodyWidth));
        CommitParents = string.Empty;

        if (row is null || _repository is not { } repository)
        {
            return;
        }

        var hash = row.Commit.Hash;
        try
        {
            var (body, files) = await Task.Run(() => (
                repository.CommitBody(hash),
                repository.CommitFiles(hash)));

            // 待っている間に別のコミットが選ばれていたら、この結果は捨てる。
            if (generation != _detailGeneration)
            {
                return;
            }

            CommitBody = BodyAfterSubject(body);
            OnPropertyChanged(nameof(HasCommitBody));
        OnPropertyChanged(nameof(BodyWidth));
            CommitParents = string.Join("  ", row.Commit.Parents.Select(p => p[..Math.Min(7, p.Length)]));

            foreach (var file in files.OrderBy(f => f.Path, StringComparer.Ordinal))
            {
                CommitFiles.Add(new GitCommitFileRow(file));
            }

            // 1 件だけなら選んでおく。**そこを選ばせるためだけの一手間を省く。**
            if (CommitFiles.Count == 1)
            {
                SelectedCommitFile = CommitFiles[0];
            }
        }
        catch (GitException error)
        {
            if (generation == _detailGeneration)
            {
                Message = error.Message;
            }
        }
    }

    private GitFileRow? _selectedFile;
    public GitFileRow? SelectedFile
    {
        get => _selectedFile;
        set => Set(ref _selectedFile, value);
    }

    private async Task StageAsync(GitFileRow row, bool stage)
    {
        if (_repository is not { } repository)
        {
            return;
        }
        try
        {
            var path = row.Status.Path;
            await Task.Run(() =>
            {
                if (stage)
                {
                    repository.Stage(path);
                }
                else
                {
                    repository.Unstage(path);
                }
            });
            await RefreshAsync();
        }
        catch (GitException error)
        {
            Message = error.Message;
        }
    }
}
