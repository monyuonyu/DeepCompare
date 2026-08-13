using System.Diagnostics;
using System.Text;

namespace DeepCompare.Engine;

/// <summary>git の呼び出しに失敗したときに投げる。標準エラーの中身を添える。</summary>
public sealed class GitException(string message) : Exception(message);

/// <summary>作業ツリーと索引での状態。git の 1 文字表記に対応する。</summary>
public enum GitStatusCode
{
    /// <summary>変化なし（`.`）。</summary>
    Unchanged,
    Modified,
    Added,
    Deleted,
    Renamed,
    Copied,
    TypeChanged,
    Untracked,
    Ignored,
    Unmerged,
}

/// <summary>
/// 1 ファイルの状態。
///
/// 索引と作業ツリーを分けて持つ。git の状態はここが 2 段になっているのが要で、
/// 「一部だけ stage した」状態を 1 つの符号に潰すと表現できない。
/// </summary>
public sealed record GitFileStatus(
    string Path,
    GitStatusCode Index,
    GitStatusCode WorkTree,
    /// <summary>名前が変わった／写された場合の元の名前。</summary>
    string? OriginalPath = null)
{
    public bool IsStaged => Index is not (GitStatusCode.Unchanged or GitStatusCode.Untracked
        or GitStatusCode.Ignored);

    public bool IsDirty => WorkTree is not GitStatusCode.Unchanged;

    public bool IsConflicted => Index is GitStatusCode.Unmerged || WorkTree is GitStatusCode.Unmerged;
}

public sealed record GitCommit(
    string Hash,
    string ShortHash,
    string Author,
    DateTimeOffset When,
    string Subject,
    IReadOnlyList<string> Parents)
{
    public bool IsMerge => Parents.Count > 1;
}

public sealed record GitBranch(
    string Name,
    bool IsCurrent,
    string? Upstream,
    int Ahead,
    int Behind);

/// <summary>
/// git リポジトリ。**git をコマンドとして呼ぶ。**
///
/// 自前で .git を読む道もあるが、そちらは pack・部分クローン・worktree・
/// submodule・フック・資格情報の受け渡しを全部自分で持つことになる。
/// コマンドを呼べば、利用者が既に設定した資格情報も改行の扱いもそのまま効く。
///
/// **決めごと**:
/// - 出力は必ず `-z` か制御文字区切りで受ける。ファイル名には改行も引用符も入りうる
/// - `core.quotepath=false` を常に渡す。既定では日本語のファイル名が
///   `\346\227\245` の形に化け、そのまま突き合わせると一致しなくなる
/// - `GIT_TERMINAL_PROMPT=0` を渡す。認証を待って**画面が固まる**のを防ぐ。
///   認証が要る操作は、待たずに失敗させて理由を出す方がよい
/// - 標準出力と標準エラーは同時に読む。片方だけ読むと、もう片方のパイプが
///   埋まった時点で相手が書き込みで止まり、こちらは読み終わらない
/// </summary>
public sealed class GitRepository
{
    /// <summary>作業ツリーの根。</summary>
    public string Root { get; }

    private readonly string _git;

    private GitRepository(string root, string git)
    {
        Root = root;
        _git = git;
    }

    /// <summary>
    /// 指定した場所を含むリポジトリを探す。見つからなければ null。
    ///
    /// **例外を投げない。** git が入っていないのは異常ではなく、ただ Git 機能が
    /// 使えないだけ。呼ぶ側が「無ければその欄を出さない」と書けるようにする。
    /// </summary>
    public static GitRepository? Discover(string path, string git = "git")
    {
        var start = Directory.Exists(path) ? path : Path.GetDirectoryName(path);
        if (string.IsNullOrEmpty(start))
        {
            return null;
        }

        try
        {
            var result = Execute(git, start, ["rev-parse", "--show-toplevel"]);
            if (result.ExitCode != 0)
            {
                return null;
            }
            var root = result.StandardOutput.Trim();
            return root.Length == 0 ? null : new GitRepository(root, git);
        }
        catch (Exception)
        {
            // git が入っていない、実行できない、など。使えないという結論は同じ。
            return null;
        }
    }

    /// <summary>git が使えるか。使えるならその版。</summary>
    public static string? Version(string git = "git")
    {
        try
        {
            var result = Execute(git, Environment.CurrentDirectory, ["--version"]);
            return result.ExitCode == 0 ? result.StandardOutput.Trim() : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>いまの枝の名前。切り離された HEAD なら null。</summary>
    public string? CurrentBranch()
    {
        var result = Run(["symbolic-ref", "--quiet", "--short", "HEAD"], allowFailure: true);
        var name = result.StandardOutput.Trim();
        return name.Length == 0 ? null : name;
    }

    /// <summary>
    /// 作業ツリーの状態。
    ///
    /// <c>--porcelain=v2 -z</c> で受ける。v1 は名前をエスケープするうえ、
    /// 索引と作業ツリーの区別が読みにくい。
    /// </summary>
    public IReadOnlyList<GitFileStatus> Status(bool includeIgnored = false)
    {
        string[] arguments = includeIgnored
            ? ["status", "--porcelain=v2", "-z", "--untracked-files=all", "--ignored=matching"]
            : ["status", "--porcelain=v2", "-z", "--untracked-files=all"];
        return ParseStatus(Run(arguments).StandardOutput);
    }

    /// <summary>
    /// 状態の解析。
    ///
    /// **単純に NUL で切ってはいけない。** 名前が変わった項目（`2` で始まる行）
    /// だけは、そのレコードの後ろにもう 1 つ NUL 区切りで元の名前が続く。
    /// 一律に切ると 1 個ずれ、以降が全部おかしくなる。
    /// </summary>
    internal static List<GitFileStatus> ParseStatus(string output)
    {
        var entries = output.Split('\0');
        var result = new List<GitFileStatus>();

        for (var i = 0; i < entries.Length; i++)
        {
            var entry = entries[i];
            if (entry.Length == 0)
            {
                continue;
            }

            switch (entry[0])
            {
                case '1':
                {
                    // 1 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <path>
                    var fields = entry.Split(' ', 9);
                    if (fields.Length < 9)
                    {
                        continue;
                    }
                    result.Add(new GitFileStatus(fields[8], Code(fields[1][0]), Code(fields[1][1])));
                    break;
                }
                case '2':
                {
                    // 2 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <X><score> <path>\0<origPath>
                    var fields = entry.Split(' ', 10);
                    if (fields.Length < 10)
                    {
                        continue;
                    }
                    // 元の名前は次のレコードとして入っている。読んだら i を進める。
                    var original = i + 1 < entries.Length ? entries[++i] : null;
                    result.Add(new GitFileStatus(
                        fields[9], Code(fields[1][0]), Code(fields[1][1]), original));
                    break;
                }
                case 'u':
                {
                    // u <XY> <sub> <m1> <m2> <m3> <mW> <h1> <h2> <h3> <path>
                    var fields = entry.Split(' ', 11);
                    if (fields.Length < 11)
                    {
                        continue;
                    }
                    result.Add(new GitFileStatus(
                        fields[10], GitStatusCode.Unmerged, GitStatusCode.Unmerged));
                    break;
                }
                case '?':
                    result.Add(new GitFileStatus(
                        entry[2..], GitStatusCode.Untracked, GitStatusCode.Untracked));
                    break;
                case '!':
                    result.Add(new GitFileStatus(
                        entry[2..], GitStatusCode.Ignored, GitStatusCode.Ignored));
                    break;
                default:
                    // 見出し行（`# branch.oid ...`）。読み飛ばす。
                    break;
            }
        }

        return result;
    }

    private static GitStatusCode Code(char c) => c switch
    {
        '.' => GitStatusCode.Unchanged,
        'M' => GitStatusCode.Modified,
        'A' => GitStatusCode.Added,
        'D' => GitStatusCode.Deleted,
        'R' => GitStatusCode.Renamed,
        'C' => GitStatusCode.Copied,
        'T' => GitStatusCode.TypeChanged,
        'U' => GitStatusCode.Unmerged,
        _ => GitStatusCode.Unchanged,
    };

    // 記録の区切り。ファイルの中身にはまず出てこない制御文字を使う。
    // 改行では切れない（コミットの題は改行を含まないが、著者名は保証がない）。
    private const char FieldSeparator = '\u001f';
    private const char RecordSeparator = '\u001e';

    /// <summary>
    /// 履歴。
    ///
    /// **1 回の起動で全部流し読む。** コミットごとに git を起動すると、
    /// 1000 件で 1000 回のプロセス生成になり、それだけで数秒かかる。
    /// </summary>
    public IReadOnlyList<GitCommit> Log(int limit = 200, string? revision = null, string? path = null)
    {
        var arguments = new List<string>
        {
            "log",
            $"--max-count={limit}",
            $"--pretty=format:%H{FieldSeparator}%h{FieldSeparator}%an{FieldSeparator}%aI"
                + $"{FieldSeparator}%P{FieldSeparator}%s{RecordSeparator}",
        };
        if (revision is { Length: > 0 })
        {
            arguments.Add(revision);
        }
        if (path is { Length: > 0 })
        {
            // 枝の名前とファイル名が同じときに git が迷う。`--` で切って明示する。
            arguments.Add("--");
            arguments.Add(path);
        }

        return ParseLog(Run([.. arguments]).StandardOutput);
    }

    internal static List<GitCommit> ParseLog(string output)
    {
        var result = new List<GitCommit>();
        foreach (var record in output.Split(RecordSeparator))
        {
            var trimmed = record.TrimStart('\n', '\r');
            if (trimmed.Length == 0)
            {
                continue;
            }
            var fields = trimmed.Split(FieldSeparator);
            if (fields.Length < 6)
            {
                continue;
            }
            result.Add(new GitCommit(
                fields[0],
                fields[1],
                fields[2],
                DateTimeOffset.TryParse(fields[3], out var when) ? when : default,
                fields[5],
                fields[4].Length == 0
                    ? []
                    : fields[4].Split(' ', StringSplitOptions.RemoveEmptyEntries)));
        }
        return result;
    }

    /// <summary>
    /// 枝の一覧。追跡先と、進み・遅れの数も取る。
    ///
    /// <c>for-each-ref</c> を使う。<c>branch -vv</c> は人が読む形で、
    /// 進み遅れが括弧書きの文の中に埋まっており、解析が壊れやすい。
    /// </summary>
    public IReadOnlyList<GitBranch> Branches()
    {
        var current = CurrentBranch();
        var result = Run([
            "for-each-ref",
            $"--format=%(refname:short){FieldSeparator}%(upstream:short){FieldSeparator}"
                + $"%(upstream:track){RecordSeparator}",
            "refs/heads",
        ]);

        var branches = new List<GitBranch>();
        foreach (var record in result.StandardOutput.Split(RecordSeparator))
        {
            var trimmed = record.Trim('\n', '\r');
            if (trimmed.Length == 0)
            {
                continue;
            }
            var fields = trimmed.Split(FieldSeparator);
            if (fields.Length < 3)
            {
                continue;
            }
            var (ahead, behind) = ParseTrack(fields[2]);
            branches.Add(new GitBranch(
                fields[0],
                string.Equals(fields[0], current, StringComparison.Ordinal),
                fields[1].Length == 0 ? null : fields[1],
                ahead,
                behind));
        }
        return branches;
    }

    /// <summary>`[ahead 2, behind 1]` の形から数を取る。追跡先が無ければ空。</summary>
    internal static (int Ahead, int Behind) ParseTrack(string track)
    {
        var ahead = 0;
        var behind = 0;
        var span = track.AsSpan();

        var a = span.IndexOf("ahead ", StringComparison.Ordinal);
        if (a >= 0)
        {
            _ = int.TryParse(Digits(span[(a + 6)..]), out ahead);
        }
        var b = span.IndexOf("behind ", StringComparison.Ordinal);
        if (b >= 0)
        {
            _ = int.TryParse(Digits(span[(b + 7)..]), out behind);
        }
        return (ahead, behind);

        static ReadOnlySpan<char> Digits(ReadOnlySpan<char> span)
        {
            var end = 0;
            while (end < span.Length && char.IsAsciiDigit(span[end]))
            {
                end++;
            }
            return span[..end];
        }
    }

    /// <summary>
    /// ある時点でのファイルの中身。
    ///
    /// バイト列で返す。**文字列にしない。** ここで符号化を決め打つと、
    /// Shift_JIS のファイルが化けたまま比較に渡る。判定は既存の
    /// <see cref="TextDecoder"/> に任せる。
    /// </summary>
    public byte[] Show(string revision, string path)
    {
        // パスは常に根からの相対で渡す。呼ぶ側が絶対パスを持っていても動くように直す。
        var relative = ToRelative(path);
        var result = RunRaw(["show", $"{revision}:{relative}"]);
        if (result.ExitCode != 0)
        {
            throw new GitException(
                $"{revision} に {relative} がありません: {result.StandardError.Trim()}");
        }
        return result.StandardOutputBytes;
    }

    /// <summary>その時点にファイルが存在したか。</summary>
    public bool Exists(string revision, string path)
        => RunRaw(["cat-file", "-e", $"{revision}:{ToRelative(path)}"]).ExitCode == 0;

    /// <summary>根からの相対に直す。既に相対ならそのまま。区切りは git に合わせて `/`。</summary>
    public string ToRelative(string path)
    {
        if (!Path.IsPathRooted(path))
        {
            return path.Replace('\\', '/');
        }
        var relative = Path.GetRelativePath(Root, path);
        return relative.Replace('\\', '/');
    }

    /// <summary>索引へ載せる。</summary>
    public void Stage(string path) => Run(["add", "--", ToRelative(path)]);

    /// <summary>索引から降ろす。中身は変えない。</summary>
    public void Unstage(string path) => Run(["restore", "--staged", "--", ToRelative(path)]);

    /// <summary>
    /// 3 方向マージの材料を取る。競合しているファイルについて、
    /// 祖先（:1）・こちら（:2）・あちら（:3）の中身を返す。
    ///
    /// 既に <see cref="ThreeWayMerge"/> があるので、材料さえ揃えば画面に出せる。
    /// </summary>
    public (byte[]? Base, byte[]? Ours, byte[]? Theirs) ConflictSources(string path)
    {
        var relative = ToRelative(path);
        return (AtStage(1), AtStage(2), AtStage(3));

        byte[]? AtStage(int number)
        {
            var result = RunRaw(["show", $":{number}:{relative}"]);
            // 片側で削除された競合では、その段が存在しない。無いことは異常ではない。
            return result.ExitCode == 0 ? result.StandardOutputBytes : null;
        }
    }

    /// <summary>
    /// コミットを作る。
    ///
    /// **索引に載っているものだけを対象にする**（`git commit` と同じ）。
    /// `-a` に当たる動きは持たない。何が入るかは stage した内容で決まる、
    /// という git の約束をそのまま保つ方が、後で驚かない。
    /// </summary>
    public void Commit(string message, bool amend = false)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            throw new GitException("コミットの説明が空です。");
        }

        // 説明は引数で渡す。一時ファイルを作ると後始末が要る。
        string[] arguments = amend
            ? ["commit", "--amend", "-m", message]
            : ["commit", "-m", message];
        Run(arguments);
    }

    /// <summary>直前のコミットの説明。書き直すときの初期値に使う。</summary>
    public string LastMessage()
        => Run(["log", "-1", "--pretty=%B"], allowFailure: true).StandardOutput.TrimEnd();

    /// <summary>枝を切り替える。作業ツリーが汚れていれば git 側が止める。</summary>
    public void Switch(string branch) => Run(["switch", branch]);

    /// <summary>枝を作って切り替える。</summary>
    public void CreateBranch(string name, string? from = null)
    {
        if (from is { Length: > 0 })
        {
            Run(["switch", "-c", name, from]);
        }
        else
        {
            Run(["switch", "-c", name]);
        }
    }

    /// <summary>枝を消す。**まだ取り込まれていない枝は git が止める。**</summary>
    public void DeleteBranch(string name, bool force = false)
        => Run(["branch", force ? "-D" : "-d", name]);

    /// <summary>
    /// 取ってくる・送る。
    ///
    /// **認証が要る操作なので、待たずに失敗させる。**
    /// GIT_TERMINAL_PROMPT=0 を渡してあるので、資格情報が無ければその場で
    /// 終わる。画面が固まったまま戻らないより、理由を出して終わる方がよい。
    /// </summary>
    public string Fetch() => Run(["fetch", "--prune"]).StandardError.Trim();

    public string Pull() => Run(["pull", "--ff-only"]).StandardError.Trim();

    public string Push() => Run(["push"]).StandardError.Trim();

    /// <summary>索引に載っているものがあるか。コミットできるかの判断に使う。</summary>
    public bool HasStagedChanges()
        => RunRaw(["diff", "--cached", "--quiet"]).ExitCode != 0;

    // --- 実行 ---

    private GitResult Run(string[] arguments, bool allowFailure = false)
    {
        var result = Execute(_git, Root, arguments);
        if (result.ExitCode != 0 && !allowFailure)
        {
            throw new GitException(
                $"git {string.Join(' ', arguments)} が失敗しました: {result.StandardError.Trim()}");
        }
        return result;
    }

    private GitResult RunRaw(string[] arguments) => Execute(_git, Root, arguments, binary: true);

    private sealed record GitResult(
        int ExitCode,
        string StandardOutput,
        string StandardError,
        byte[] StandardOutputBytes);

    private static GitResult Execute(
        string git, string workingDirectory, string[] arguments, bool binary = false)
    {
        var info = new ProcessStartInfo
        {
            FileName = git,
            WorkingDirectory = workingDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        // 既定では日本語のファイル名が \346\227\245 の形に化ける。常に切る。
        info.ArgumentList.Add("-c");
        info.ArgumentList.Add("core.quotepath=false");
        foreach (var argument in arguments)
        {
            info.ArgumentList.Add(argument);
        }

        // 認証を待って固まらせない。待つ相手がいない場面で画面ごと止まる方が困る。
        info.Environment["GIT_TERMINAL_PROMPT"] = "0";
        info.Environment["GIT_OPTIONAL_LOCKS"] = "0";

        using var process = new Process { StartInfo = info };
        process.Start();

        // **両方の流れを同時に読む。** 片方を読み切ってからもう片方へ回ると、
        // 先に埋まったパイプで相手が書き込みを待ち、こちらは読み終わらない。
        // 出力が大きいとき（git show の中身など）に確実に起きる。
        using var outputStream = new MemoryStream();
        var copying = process.StandardOutput.BaseStream.CopyToAsync(outputStream);
        var errorText = process.StandardError.ReadToEndAsync();

        copying.GetAwaiter().GetResult();
        var error = errorText.GetAwaiter().GetResult();
        process.WaitForExit();

        var bytes = outputStream.ToArray();
        return new GitResult(
            process.ExitCode,
            binary ? string.Empty : new UTF8Encoding(false).GetString(bytes),
            error,
            bytes);
    }
}
