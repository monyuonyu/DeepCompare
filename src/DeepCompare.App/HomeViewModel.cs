using System.Collections.ObjectModel;
using System.Windows.Input;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>一覧に出す 1 件。表示用の文言まで作ってしまう。</summary>
public sealed record SessionEntry(Session Session)
{
    public string Name => Session.Name;
    public string Kind => Session.Kind == SessionKind.Folder ? "フォルダー" : "ファイル";
    public string Paths => $"{Session.LeftPath}  ↔  {Session.RightPath}";
}

/// <summary>
/// 保存した比較の一覧。
///
/// **比較の種類はサイドバーが持つ**ので、ここでは扱わない。この画面の役目は
/// 「よく使う組み合わせに名前を付けて残す」ことだけ。そのとき何を無視していたかも
/// 一緒に覚えるので、開き直しても設定を入れ直す必要がない。
/// </summary>
public sealed class HomeViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;
    private readonly SessionStore _sessions = new();
    private string _leftPath = string.Empty;
    private string _rightPath = string.Empty;
    private string _message = string.Empty;
    private string _sessionName = string.Empty;

    public HomeViewModel(ShellViewModel shell)
    {
        _shell = shell;
        BrowseLeftFileCommand = new RelayCommand(() => PickAsync(isFolder: false, left: true));
        BrowseRightFileCommand = new RelayCommand(() => PickAsync(isFolder: false, left: false));
        BrowseLeftFolderCommand = new RelayCommand(() => PickAsync(isFolder: true, left: true));
        BrowseRightFolderCommand = new RelayCommand(() => PickAsync(isFolder: true, left: false));
        UseRemoteLeftCommand = new RelayCommand(
            () => { UseRemote(left: true); return Task.CompletedTask; });
        UseRemoteRightCommand = new RelayCommand(
            () => { UseRemote(left: false); return Task.CompletedTask; });
        StartKindCommand = new RelayCommand<CompareKind>(
            kind => { StartKind(kind); return Task.CompletedTask; });
        SaveSessionCommand = new RelayCommand(() => { SaveSession(); return Task.CompletedTask; });
        OpenSessionCommand = new RelayCommand<SessionEntry>(entry => { OpenSession(entry); return Task.CompletedTask; });
        RemoveSessionCommand = new RelayCommand<SessionEntry>(entry => { RemoveSession(entry); return Task.CompletedTask; });
        ReloadSessions();
    }

    public ObservableCollection<SessionEntry> Sessions { get; } = [];

    public bool HasSessions => Sessions.Count > 0;

    /// <summary>保存するときの名前。空なら左右の名前から作る。</summary>
    public string SessionName
    {
        get => _sessionName;
        set => Set(ref _sessionName, value);
    }

    public ICommand StartKindCommand { get; }
    public ICommand SaveSessionCommand { get; }

    /// <summary>テーマの切り替えなど、画面をまたぐ操作。</summary>
    public ShellViewModel Shell => _shell;

    /// <summary>
    /// 種類を選んで始める。
    ///
    /// 左右が入っていればそのまま比較まで進み、空なら画面だけ開く。
    /// **空でも開く**のは、その画面にも指定欄があるから。ここで止めると
    /// 「どこに入れればいいのか」が分からなくなる。
    /// </summary>
    private void StartKind(CompareKind kind)
    {
        Message = string.Empty;
        var left = LeftPath.Trim();
        var right = RightPath.Trim();

        // **入っているものと選んだ種類が食い違うなら、入っているものを優先する。**
        // フォルダーを指定したまま「テキスト比較」を押せてしまい、
        // 開いた先で「Access to the path ... is denied」になっていた。
        var anyFolder = (left.Length > 0 && Directory.Exists(left))
                     || (right.Length > 0 && Directory.Exists(right));
        if (anyFolder && kind.Id is CompareKindId.Text or CompareKindId.Structured
                                 or CompareKindId.Notebook or CompareKindId.Table
                                 or CompareKindId.Image or CompareKindId.VersionInfo)
        {
            _shell.ShowFolders(left, right);
            return;
        }

        _shell.Open(kind, left, right);
    }
    public ICommand OpenSessionCommand { get; }
    public ICommand RemoveSessionCommand { get; }

    private void ReloadSessions()
    {
        Sessions.Clear();
        foreach (var session in _sessions.Load())
        {
            Sessions.Add(new SessionEntry(session));
        }
        OnPropertyChanged(nameof(HasSessions));
    }

    private void SaveSession()
    {
        var left = LeftPath.Trim();
        var right = RightPath.Trim();
        if (left.Length == 0 || right.Length == 0)
        {
            Message = "保存する前に左右の両方を指定してください。";
            return;
        }

        var isFolder = Directory.Exists(left) && Directory.Exists(right);
        var name = SessionName.Trim();
        if (name.Length == 0)
        {
            // 名前を毎回考えさせない。後から付け直せる。
            name = $"{Path.GetFileName(left.TrimEnd(Path.DirectorySeparatorChar))} ↔ "
                 + $"{Path.GetFileName(right.TrimEnd(Path.DirectorySeparatorChar))}";
        }

        _sessions.Upsert(new Session
        {
            Name = name,
            Kind = isFolder ? SessionKind.Folder : SessionKind.Text,
            LeftPath = left,
            RightPath = right,
        });
        SessionName = string.Empty;
        Message = $"「{name}」を保存しました。";
        ReloadSessions();
    }

    private void OpenSession(SessionEntry entry)
    {
        // 開いた時点で「最近使った」順を更新する。
        _sessions.Upsert(entry.Session);
        ReloadSessions();

        // **保存した設定をそのまま渡す。** ここが抜けていると、
        // 「そのとき何を無視していたか」まで覚える意味が無くなり、
        // 開くたびに空白の扱いや除外の指定を入れ直すことになる。
        var session = entry.Session;

        if (session.Kind == SessionKind.Folder)
        {
            var model = _shell.ShowFolders(session.LeftPath, session.RightPath, run: false);
            model.IncludeNames = string.Join(" ", session.IncludeNames);
            model.ExcludeNames = string.Join(" ", session.ExcludeNames);
            model.TimestampTolerance = session.TimestampToleranceSeconds;
            model.IgnoreDaylightSaving = session.IgnoreDaylightSavingOffset;
            model.RefreshCommand.Execute(null);
        }
        else
        {
            var model = _shell.ShowText(session.LeftPath, session.RightPath, run: false);
            model.PairThreshold = session.PairThreshold;
            model.WhitespaceModeIndex = (int)session.Whitespace;
            model.IgnoreCase = session.IgnoreCase;
            model.IgnoredPatterns = string.Join("\n", session.IgnoredPatterns);
            model.CompareCommand.Execute(null);
        }
    }

    private void RemoveSession(SessionEntry entry)
    {
        _sessions.Remove(entry.Session.Name);
        ReloadSessions();
    }

    public string LeftPath
    {
        get => _leftPath;
        set => Set(ref _leftPath, value);
    }

    public string RightPath
    {
        get => _rightPath;
        set => Set(ref _rightPath, value);
    }

    public string Message
    {
        get => _message;
        private set => Set(ref _message, value);
    }

    public ICommand BrowseLeftFileCommand { get; }
    public ICommand BrowseRightFileCommand { get; }
    public ICommand BrowseLeftFolderCommand { get; }
    public ICommand BrowseRightFolderCommand { get; }

    /// <summary>
    /// リモートを組み立てる小画面。**左右で 1 つを使い回す。**
    /// 開くたびに前の入力が残っていると、右に左の主機が入る。
    /// </summary>
    public RemoteBuilderViewModel Remote { get; } = new();

    public ICommand UseRemoteLeftCommand { get; }
    public ICommand UseRemoteRightCommand { get; }

    /// <summary>組み立てた場所を欄に入れて開く。</summary>
    private void UseRemote(bool left)
    {
        var location = Remote.Location;
        if (location.Length == 0)
        {
            return;
        }
        if (left)
        {
            LeftPath = location;
        }
        else
        {
            RightPath = location;
        }
        Remote.Reset();
        Open();
    }


    private async Task PickAsync(bool isFolder, bool left)
    {
        var title = (isFolder, left) switch
        {
            (true, true) => "左のフォルダーを選択",
            (true, false) => "右のフォルダーを選択",
            (false, true) => "左のファイルを選択",
            (false, false) => "右のファイルを選択",
        };
        if (await _shell.PickPath(title, isFolder) is { } path)
        {
            if (left)
            {
                LeftPath = path;
            }
            else
            {
                RightPath = path;
            }

            // **選んだ時点で開く。** 落としたときと同じ。
            Open();
        }
    }

    private bool Validate(out string left, out string right)
    {
        left = LeftPath.Trim();
        right = RightPath.Trim();
        if (left.Length == 0 && right.Length == 0)
        {
            Message = "比べるものを指定してください。";
            return false;
        }
        Message = string.Empty;
        return true;
    }

    /// <summary>
    /// 落とされたものを受け取る。ファイルかフォルダーかは中身を見て判断できるので、
    /// 利用者に選ばせる必要はない。
    /// </summary>
    public void AcceptDropped(IReadOnlyList<string> paths)
    {
        if (paths.Count == 0)
        {
            return;
        }
        if (paths.Count >= 2)
        {
            LeftPath = paths[0];
            RightPath = paths[1];
        }
        else if (LeftPath.Trim().Length == 0)
        {
            LeftPath = paths[0];
        }
        else if (RightPath.Trim().Length == 0)
        {
            RightPath = paths[0];
        }
        else
        {
            LeftPath = paths[0];
        }

        // **落としたら、その場で比較の画面へ移る。**
        // 以前は「揃いました。ボタンを押してください」と言うだけで、
        // 落としたのにもう一手間かかった。片方だけでも中身は出せる。
        Open();
    }

    /// <summary>
    /// いま入っている指定で画面を開く。
    ///
    /// **片方だけでも開く。** 開いた先でもう片方を指定できるので、
    /// ここで止める理由が無い。フォルダーかファイルかは中身で決める。
    /// </summary>
    public void Open()
    {
        var left = LeftPath.Trim();
        var right = RightPath.Trim();
        if (left.Length == 0 && right.Length == 0)
        {
            return;
        }

        // **どちらか一方でもフォルダーなら、フォルダーとして開く。**
        // 片方しか無いときは、それがフォルダーかどうかで決める。
        var anyFolder = (left.Length > 0 && Directory.Exists(left))
                     || (right.Length > 0 && Directory.Exists(right));
        var anyRemote = RemoteLocation.IsRemote(left) || RemoteLocation.IsRemote(right);

        if (anyFolder || anyRemote)
        {
            _shell.ShowFolders(left, right);
        }
        else
        {
            _shell.ShowText(left, right);
        }
        Message = string.Empty;
    }
}
