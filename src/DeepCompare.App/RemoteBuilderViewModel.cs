namespace DeepCompare.App;

/// <summary>繋ぎ方 1 つ。</summary>
public sealed record RemoteScheme(string Prefix, string Label, string Hint, bool NeedsUser);

/// <summary>
/// リモートの場所を組み立てる小画面。
///
/// **書き方を覚えていなくても届く。** これまでは起動画面に
/// 「s3://鍵:秘密@入口/バケツ」のような見本を並べていただけで、
/// 記号の順番を 1 つ間違えれば黙って普通のパスとして扱われた。
///
/// 埋めた欄から URI を組み立て、**その場に出す。** 覚えたい人は
/// 出来上がりを見て覚えられるし、覚えたくない人は見なくてよい。
/// </summary>
public sealed class RemoteBuilderViewModel : ViewModelBase
{
    public IReadOnlyList<RemoteScheme> Schemes { get; } =
    [
        new("sftp://", "SFTP", "ssh で入る。鍵があれば合言葉は要りません", true),
        new("ftps://", "FTPS", "暗号化した FTP", true),
        new("ftp://", "FTP", "**合言葉がそのまま流れます。** 相手が対応しているなら FTPS へ", true),
        new("davs://", "WebDAV (https)", "Nextcloud など", true),
        new("dav://", "WebDAV (http)", "暗号化されません", true),
        new("s3://", "S3", "鍵を空にすると AWS_ACCESS_KEY_ID などから拾います", false),
    ];

    private RemoteScheme _scheme;
    public RemoteBuilderViewModel()
    {
        _scheme = Schemes[0];
    }

    public RemoteScheme Scheme
    {
        get => _scheme;
        set { if (Set(ref _scheme, value)) { Changed(); } }
    }

    private string _host = string.Empty;

    /// <summary>主機。S3 なら入口（空なら amazonaws.com）。</summary>
    public string Host
    {
        get => _host;
        set { if (Set(ref _host, value)) { Changed(); } }
    }

    private string _user = string.Empty;

    /// <summary>利用者名。S3 では鍵。</summary>
    public string User
    {
        get => _user;
        set { if (Set(ref _user, value)) { Changed(); } }
    }

    private string _password = string.Empty;

    /// <summary>
    /// 合言葉。S3 では秘密鍵。
    ///
    /// **組み立てた URI に入るので、保存すると平文で残る。**
    /// 空のままにして、環境変数か鍵ファイルに任せるのが安全。
    /// </summary>
    public string Password
    {
        get => _password;
        set { if (Set(ref _password, value)) { Changed(); } }
    }

    private string _path = string.Empty;

    /// <summary>相手側の場所。S3 ならバケツ以下。</summary>
    public string Path
    {
        get => _path;
        set { if (Set(ref _path, value)) { Changed(); } }
    }

    private void Changed()
    {
        OnPropertyChanged(nameof(Location));
        OnPropertyChanged(nameof(CanUse));
        OnPropertyChanged(nameof(Warning));
        OnPropertyChanged(nameof(HasWarning));
        OnPropertyChanged(nameof(UserLabel));
        OnPropertyChanged(nameof(PasswordLabel));
        OnPropertyChanged(nameof(HostLabel));
    }

    // S3 だけ呼び名が変わる。**同じ欄に違う名前を付けるだけで、
    // 「ここに何を入れるのか」の迷いが消える。**
    public string HostLabel => Scheme.Prefix == "s3://" ? "入口" : "主機";
    public string UserLabel => Scheme.Prefix == "s3://" ? "鍵" : "利用者";
    public string PasswordLabel => Scheme.Prefix == "s3://" ? "秘密鍵" : "合言葉";

    /// <summary>組み立てた場所。**埋めながら育つのが見える。**</summary>
    public string Location
    {
        get
        {
            var host = Host.Trim();
            if (host.Length == 0)
            {
                return string.Empty;
            }

            var credential = string.Empty;
            var user = User.Trim();
            if (user.Length > 0)
            {
                // **記号を含む合言葉で壊れないようにする。** @ や / が
                // そのまま入ると、区切りとして読まれて別の場所を指す。
                credential = Uri.EscapeDataString(user);
                if (Password.Length > 0)
                {
                    credential += ":" + Uri.EscapeDataString(Password);
                }
                credential += "@";
            }

            var path = Path.Trim().TrimStart('/');
            return $"{Scheme.Prefix}{credential}{host}/{path}";
        }
    }

    public bool CanUse => Location.Length > 0;

    /// <summary>危ないときだけ言う。**常時出ていると読まれなくなる。**</summary>
    public string Warning
    {
        get
        {
            if (Scheme.Prefix is "ftp://" or "dav://")
            {
                return "暗号化されません。合言葉がそのまま流れます。";
            }
            if (Password.Length > 0)
            {
                return "合言葉は組み立てた場所に含まれます。保存すると平文で残ります。";
            }
            return string.Empty;
        }
    }

    public bool HasWarning => Warning.Length > 0;

    /// <summary>次に開くときのために、埋めたものを消しておく。</summary>
    public void Reset()
    {
        _host = _user = _password = _path = string.Empty;
        OnPropertyChanged(nameof(Host));
        OnPropertyChanged(nameof(User));
        OnPropertyChanged(nameof(Password));
        OnPropertyChanged(nameof(Path));
        Changed();
    }
}
