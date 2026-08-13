using System.Collections.ObjectModel;
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

    /// <summary>索引と作業ツリーの状態を 2 文字で。git の表記に合わせる。</summary>
    public string Mark => Status.IsConflicted ? "UU" : $"{Code(Status.Index)}{Code(Status.WorkTree)}";

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
        OpenCommitCommand = new RelayCommand<GitCommitRow>(
            row => { OpenCommit(row); return Task.CompletedTask; });
    }

    public ObservableCollection<GitFileRow> Files { get; } = [];
    public ObservableCollection<GitCommitRow> Commits { get; } = [];

    public RelayCommand RefreshCommand { get; }
    public RelayCommand<GitFileRow> OpenFileCommand { get; }
    public RelayCommand<GitFileRow> StageCommand { get; }
    public RelayCommand<GitFileRow> UnstageCommand { get; }
    public RelayCommand<GitCommitRow> OpenCommitCommand { get; }

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
                Commits.Clear();
                return;
            }

            _repository = repository;
            OnPropertyChanged(nameof(HasRepository));
            Root = repository.Root;

            // 状態・枝・履歴をまとめて 1 回の作業スレッドで取る。
            var (files, branch, commits) = await Task.Run(() => (
                repository.Status().Where(f => f.Index != GitStatusCode.Ignored).ToList(),
                repository.CurrentBranch(),
                repository.Log(100)));

            Branch = branch ?? "（切り離された HEAD）";

            Files.Clear();
            foreach (var file in files.OrderBy(f => f.Path, StringComparer.Ordinal))
            {
                Files.Add(new GitFileRow(file));
            }

            Commits.Clear();
            foreach (var commit in commits)
            {
                Commits.Add(new GitCommitRow(commit));
            }

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

    /// <summary>そのコミットで何が変わったかを見る。親と比べる。</summary>
    private void OpenCommit(GitCommitRow row)
    {
        if (_repository is not { } repository)
        {
            return;
        }
        if (row.Commit.Parents.Count == 0)
        {
            Message = "最初のコミットには親が無いので、比べる相手がありません。";
            return;
        }

        // どのファイルを見るかは、いま選ばれている変更ファイルに合わせる。
        // 選ばれていなければ、何を出すか決められないので知らせる。
        if (SelectedFile is not { } selected)
        {
            Message = "先に左の一覧でファイルを選んでください。そのファイルの、このコミットでの変化を出します。";
            return;
        }

        var hash = row.Commit.Hash;
        var parent = row.Commit.Parents[0];
        var relative = selected.Status.Path;

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
