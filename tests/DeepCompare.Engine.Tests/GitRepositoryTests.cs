using System.Diagnostics;
using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// git の呼び出し。
///
/// **本物の git に対して試す。** 出力を文字列で作って解析器だけ試すと、
/// 「私が思っている git の出力」しか固定できない。実際に踏むのは、
/// 版によって形が違う所と、名前に妙な文字が入った所。
///
/// git が入っていない環境ではこの一式を飛ばす。試験が動かないことと
/// 実装が壊れていることを混同しない。
/// </summary>
public sealed class GitRepositoryTests : IDisposable
{
    private readonly string _root;

    public GitRepositoryTests()
    {
        // **git が無ければ黙って通さない。** 「試験が動かない」と「実装が正しい」を
        // 混同すると、壊れていることに気づけない。落として理由を出す。
        if (GitRepository.Version() is null)
        {
            throw new InvalidOperationException(
                "この一式には git が要ります。git を入れてから走らせてください。");
        }
        _root = Path.Combine(Path.GetTempPath(), "dc-git-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(_root);

        Git("init", "--initial-branch=main");
        // 利用者の設定に依存させない。CI でも同じ結果にする。
        Git("config", "user.email", "t@example.com");
        Git("config", "user.name", "試験");
        Git("config", "commit.gpgsign", "false");
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_root))
            {
                Directory.Delete(_root, recursive: true);
            }
        }
        catch (IOException)
        {
            // 消せなくても試験の結果は変わらない。
        }
    }

    private void Git(params string[] arguments)
    {
        var info = new ProcessStartInfo("git")
        {
            WorkingDirectory = _root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in arguments)
        {
            info.ArgumentList.Add(argument);
        }
        using var process = Process.Start(info)!;
        var error = process.StandardError.ReadToEnd();
        process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"git {string.Join(' ', arguments)}: {error}");
        }
    }

    /// <summary>失敗しても続ける git。競合するマージのように、0 でない終了が正常な場合に使う。</summary>
    private void GitAllowFailure(params string[] arguments)
    {
        var info = new ProcessStartInfo("git")
        {
            WorkingDirectory = _root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in arguments)
        {
            info.ArgumentList.Add(argument);
        }
        using var process = Process.Start(info)!;
        process.StandardError.ReadToEnd();
        process.StandardOutput.ReadToEnd();
        process.WaitForExit();
    }

    private void WriteFile(string name, string content)
    {
        var path = Path.Combine(_root, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content, new UTF8Encoding(false));
    }

    private GitRepository Open()
    {
        var repository = GitRepository.Discover(_root);
        Assert.NotNull(repository);
        return repository;
    }

    // --- 解析だけの試験（git が無くても走る） ---

    [Fact]
    public void 名前が変わった項目で一個ずれない()
    {
        // porcelain v2 の -z では、リネームのレコードだけ後ろに元の名前が続く。
        // 一律に NUL で切ると 1 個ずれ、以降が全部おかしくなる。
        var output =
            "1 .M N... 100644 100644 100644 aaa bbb ふつう.txt\0"
            + "2 R. N... 100644 100644 100644 ccc ddd R100 あたらしい.txt\0ふるい.txt\0"
            + "1 M. N... 100644 100644 100644 eee fff つぎ.txt\0";

        var result = GitRepository.ParseStatus(output);

        Assert.Equal(3, result.Count);
        Assert.Equal("ふつう.txt", result[0].Path);
        Assert.Equal("あたらしい.txt", result[1].Path);
        Assert.Equal("ふるい.txt", result[1].OriginalPath);
        // ここがずれていないことが要点。元の名前を項目として数えていない。
        Assert.Equal("つぎ.txt", result[2].Path);
        Assert.Null(result[2].OriginalPath);
    }

    [Fact]
    public void 見出し行を項目として数えない()
    {
        var output =
            "# branch.oid aaaaaa\0# branch.head main\0"
            + "1 .M N... 100644 100644 100644 aaa bbb a.txt\0";

        Assert.Single(GitRepository.ParseStatus(output));
    }

    [Fact]
    public void 進みと遅れを読む()
    {
        Assert.Equal((2, 1), GitRepository.ParseTrack("[ahead 2, behind 1]"));
        Assert.Equal((3, 0), GitRepository.ParseTrack("[ahead 3]"));
        Assert.Equal((0, 5), GitRepository.ParseTrack("[behind 5]"));
        Assert.Equal((0, 0), GitRepository.ParseTrack(""));
        Assert.Equal((0, 0), GitRepository.ParseTrack("[gone]"));
    }

    [Fact]
    public void 親を持たないコミットを読める()
    {
        var record = "abc123abc著者2026-08-13T10:00:00+09:00最初";

        var log = GitRepository.ParseLog(record);

        var commit = Assert.Single(log);
        Assert.Empty(commit.Parents);
        Assert.Equal("最初", commit.Subject);
        Assert.False(commit.IsMerge);
    }

    [Fact]
    public void 親が二つならマージとみなす()
    {
        var record = "hha2026-08-13T10:00:00+09:00p1 p2まーじ";

        Assert.True(Assert.Single(GitRepository.ParseLog(record)).IsMerge);
    }

    [Fact]
    public void 枝とタグの名前を読み分ける()
    {
        // **タグには `tag: ` が前に付く。** full 形式でも付く。
        var refs = GitRepository.ParseRefs(
            "HEAD -> refs/heads/main, refs/remotes/origin/main, tag: refs/tags/v1.0");

        Assert.Equal(3, refs.Count);

        Assert.Equal("main", refs[0].Name);
        Assert.Equal(GitRefKind.Local, refs[0].Kind);
        Assert.True(refs[0].IsCurrent);

        Assert.Equal("origin/main", refs[1].Name);
        Assert.Equal(GitRefKind.Remote, refs[1].Kind);
        Assert.False(refs[1].IsCurrent);

        Assert.Equal("v1.0", refs[2].Name);
        Assert.Equal(GitRefKind.Tag, refs[2].Kind);
    }

    [Fact]
    public void 斜線を含む枝を遠隔と間違えない()
    {
        // `feature/x` は手元の枝。短い形だと `origin/x` と区別が付かないので、
        // **--decorate=full で受け取る**のが効いている場所。
        var refs = GitRepository.ParseRefs("refs/heads/feature/x, refs/remotes/origin/feature/x");

        Assert.Equal(GitRefKind.Local, refs[0].Kind);
        Assert.Equal("feature/x", refs[0].Name);
        Assert.Equal(GitRefKind.Remote, refs[1].Kind);
        Assert.Equal("origin/feature/x", refs[1].Name);
    }

    [Fact]
    public void 切り離されたHEADを読む()
    {
        var refs = GitRepository.ParseRefs("HEAD, tag: refs/tags/v2");

        Assert.Equal(GitRefKind.Head, refs[0].Kind);
        Assert.True(refs[0].IsCurrent);
        Assert.Equal(GitRefKind.Tag, refs[1].Kind);
    }

    [Fact]
    public void 名前でない飾りは落とす()
    {
        // 浅いクローンだと `grafted` が混ざる。札にすると意味が通らない。
        Assert.Empty(GitRepository.ParseRefs("grafted"));
        Assert.Empty(GitRepository.ParseRefs(""));
    }

    // --- 本物の git に対する試験 ---

    [Fact]
    public void リポジトリを見つける()
    {
        WriteFile("a.txt", "あ\n");

        var repository = Open();

        // /tmp が symlink の環境（macOS）では realpath が違う。名前で比べる。
        Assert.Equal(Path.GetFileName(_root), Path.GetFileName(repository.Root));
    }

    [Fact]
    public void 本物のgitから枝とタグの札を取る()
    {
        // **文字列を組み立てた試験だけでは足りない。** git が実際に出す形は
        // 版や設定で変わる（タグに `tag: ` が付くことを取りこぼしていた）。
        WriteFile("a.txt", "1\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");
        Git("tag", "v1.0");
        Git("switch", "-c", "feature/x");
        WriteFile("b.txt", "2\n");
        Git("add", "-A");
        Git("commit", "-m", "枝の方");

        var log = Open().Log(all: true);

        var tip = log[0];
        var local = Assert.Single(tip.Refs);
        Assert.Equal("feature/x", local.Name);
        Assert.Equal(GitRefKind.Local, local.Kind);
        Assert.True(local.IsCurrent);

        var root = log[^1];
        Assert.Contains(root.Refs, r => r.Kind == GitRefKind.Tag && r.Name == "v1.0");
        Assert.Contains(root.Refs, r => r.Kind == GitRefKind.Local && r.Name is "main" or "master");
    }

    [Fact]
    public void ファイル名だけでもリポジトリを見つける()
    {
        // **一番自然な呼び方**（`deepcompare --git-diff a.txt`）がこの形。
        // 親のディレクトリが空文字になるので、そこで諦めると常に失敗する。
        WriteFile("a.txt", "1\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");

        var previous = Directory.GetCurrentDirectory();
        try
        {
            Directory.SetCurrentDirectory(_root);
            Assert.NotNull(GitRepository.Discover("a.txt"));
        }
        finally
        {
            Directory.SetCurrentDirectory(previous);
        }
    }

    [Fact]
    public void 一部だけ索引へ載せると三つが別物になる()
    {
        // **これが hunk 単位の stage の要。** 索引・作業ツリー・HEAD が
        // それぞれ別の中身になる状態を作れないと、「一部だけ stage」は表現できない。
        WriteFile("a.txt", "1\n2\n3\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");

        // 作業ツリーで 2 か所直す。
        WriteFile("a.txt", "1 を直した\n2\n3 も直した\n");

        var repository = Open();

        // そのうち 1 か所だけを索引へ載せる。
        var partial = new UTF8Encoding(false).GetBytes("1 を直した\n2\n3\n");
        repository.StageContent("a.txt", partial);

        var head = new UTF8Encoding(false).GetString(repository.Show("HEAD", "a.txt"));
        var index = new UTF8Encoding(false).GetString(repository.IndexContent("a.txt"));
        var work = File.ReadAllText(Path.Combine(_root, "a.txt"));

        Assert.Equal("1\n2\n3\n", head);
        Assert.Equal("1 を直した\n2\n3\n", index);
        Assert.Equal("1 を直した\n2\n3 も直した\n", work);

        // git 自身も「索引に載った分」と「まだの分」の両方を認めるはず。
        var status = repository.Status().Single(f => f.Path == "a.txt");
        Assert.Equal(GitStatusCode.Modified, status.Index);
        Assert.Equal(GitStatusCode.Modified, status.WorkTree);
    }

    [Fact]
    public void 索引へ載せても作業ツリーは変わらない()
    {
        WriteFile("a.txt", "作業ツリーの中身\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");
        WriteFile("a.txt", "書き換えた\n");

        Open().StageContent("a.txt", new UTF8Encoding(false).GetBytes("索引だけの中身\n"));

        // **作業ツリーには触らない。** ここが崩れると、直しかけの内容が消える。
        Assert.Equal("書き換えた\n", File.ReadAllText(Path.Combine(_root, "a.txt")));
    }

    [Fact]
    public void 実行権限を保つ()
    {
        // 権限を決め打ちにすると、実行できたファイルが stage しただけで
        // 実行できなくなる。
        WriteFile("run.sh", "#!/bin/sh\necho a\n");
        Git("add", "-A");
        Git("update-index", "--chmod=+x", "run.sh");
        Git("commit", "-m", "最初");

        var repository = Open();
        Assert.Equal("100755", repository.IndexMode("run.sh"));

        repository.StageContent("run.sh", new UTF8Encoding(false).GetBytes("#!/bin/sh\necho b\n"));
        Assert.Equal("100755", repository.IndexMode("run.sh"));
    }

    [Fact]
    public void 索引に無いファイルは空を返す()
    {
        WriteFile("追跡していない.txt", "x\n");

        var repository = Open();
        Assert.Empty(repository.IndexContent("追跡していない.txt"));
        Assert.False(repository.IsInIndex("追跡していない.txt"));
        // 新しいファイルは普通の権限で作る。
        Assert.Equal("100644", repository.IndexMode("追跡していない.txt"));
    }

    [Fact]
    public void 同じ中身なら同じ名前になる()
    {
        WriteFile("a.txt", "x\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");

        var repository = Open();
        var content = new UTF8Encoding(false).GetBytes("同じ中身\n");

        // blob の名前は中身だけで決まる。**呼ぶたびに変わったら索引が壊れる。**
        Assert.Equal(repository.HashObject(content), repository.HashObject(content));
        Assert.NotEqual(
            repository.HashObject(content),
            repository.HashObject(new UTF8Encoding(false).GetBytes("違う中身\n")));
    }

    [Fact]
    public void 大きい中身を渡しても固まらない()
    {
        // 標準入力へ書いた後に閉じないと、相手は終わりを知れず、
        // こちらは相手の終了を待つ形で止まる。**1MB 程度で確実に起きる。**
        WriteFile("a.txt", "x\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");

        var big = new byte[1024 * 1024];
        Array.Fill(big, (byte)'a');

        var hash = Open().HashObject(big);
        Assert.Equal(40, hash.Length);
    }

    [Fact]
    public void 競合している三つの中身を取る()
    {
        WriteFile("a.txt", "祖先\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");

        Git("switch", "-c", "むこう");
        WriteFile("a.txt", "むこう\n");
        Git("add", "-A");
        Git("commit", "-m", "むこうの変更");

        Git("switch", "-");
        WriteFile("a.txt", "こちら\n");
        Git("add", "-A");
        Git("commit", "-m", "こちらの変更");

        GitAllowFailure("merge", "むこう");   // 競合して 0 以外で終わるのが正常

        var repository = Open();
        var text = (int stage) =>
            new UTF8Encoding(false).GetString(repository.ConflictStage("a.txt", stage));

        Assert.Equal("祖先\n", text(1));
        Assert.Equal("こちら\n", text(2));
        Assert.Equal("むこう\n", text(3));
    }

    [Fact]
    public void 祖先の無い競合では空を返す()
    {
        // 両側で別々に同じ名前のファイルを作った競合。**祖先が存在しない。**
        // ここで例外を投げると、その形の競合が画面から解けなくなる。
        WriteFile("土台.txt", "x\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");

        Git("switch", "-c", "むこう");
        WriteFile("新しい.txt", "むこうが作った\n");
        Git("add", "-A");
        Git("commit", "-m", "むこうで追加");

        Git("switch", "-");
        WriteFile("新しい.txt", "こちらが作った\n");
        Git("add", "-A");
        Git("commit", "-m", "こちらで追加");

        GitAllowFailure("merge", "むこう");

        var repository = Open();
        Assert.Empty(repository.ConflictStage("新しい.txt", 1));
        Assert.NotEmpty(repository.ConflictStage("新しい.txt", 2));
        Assert.NotEmpty(repository.ConflictStage("新しい.txt", 3));
    }

    [Fact]
    public void 競合していないファイルの段は空()
    {
        WriteFile("a.txt", "1\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");

        // 競合していなければ索引に段は無い。例外ではなく空。
        Assert.Empty(Open().ConflictStage("a.txt", 2));
    }

    [Fact]
    public void コミットで変わったファイルを取る()
    {
        WriteFile("a.txt", "1\n");
        WriteFile("消える.txt", "x\n");
        Git("add", "-A");
        Git("commit", "-m", "最初");

        WriteFile("a.txt", "2\n");
        WriteFile("増える.txt", "y\n");
        File.Delete(Path.Combine(_root, "消える.txt"));
        Git("add", "-A");
        Git("commit", "-m", "二つ目");

        var files = Open().CommitFiles("HEAD");

        Assert.Equal(3, files.Count);
        Assert.Equal(GitStatusCode.Modified, files.Single(f => f.Path == "a.txt").Index);
        Assert.Equal(GitStatusCode.Added, files.Single(f => f.Path == "増える.txt").Index);
        Assert.Equal(GitStatusCode.Deleted, files.Single(f => f.Path == "消える.txt").Index);
    }

    [Fact]
    public void リポジトリでない場所ではnullを返す()
    {
        var outside = Path.Combine(Path.GetTempPath(), "dc-plain-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(outside);
        try
        {
            // 一時領域そのものが git の中にあると、この試験は成り立たない。
            // その場合だけ確かめずに済ませる（起きないはずだが、黙って通さない）。
            if (GitRepository.Discover(Path.GetTempPath()) is not null)
            {
                Assert.Fail("一時領域が git リポジトリの中にあるため確かめられません");
            }

            Assert.Null(GitRepository.Discover(outside));
        }
        finally
        {
            Directory.Delete(outside, recursive: true);
        }
    }

    [Fact]
    public void 追跡されていないファイルと変更を見分ける()
    {
        WriteFile("committed.txt", "もと\n");
        Git("add", "."); Git("commit", "-m", "最初");
        WriteFile("committed.txt", "かえた\n");
        WriteFile("new.txt", "あたらしい\n");

        var status = Open().Status();

        var committed = Assert.Single(status, s => s.Path == "committed.txt");
        Assert.Equal(GitStatusCode.Modified, committed.WorkTree);
        Assert.Equal(GitStatusCode.Unchanged, committed.Index);
        Assert.False(committed.IsStaged);

        var added = Assert.Single(status, s => s.Path == "new.txt");
        Assert.Equal(GitStatusCode.Untracked, added.Index);
    }

    [Fact]
    public void 索引と作業ツリーを分けて持つ()
    {
        WriteFile("f.txt", "1\n");
        Git("add", "."); Git("commit", "-m", "最初");

        WriteFile("f.txt", "2\n");
        Git("add", "f.txt");
        WriteFile("f.txt", "3\n");   // stage した後にもう一度変える

        var status = Assert.Single(Open().Status(), s => s.Path == "f.txt");

        // 1 つの符号に潰すとこの状態は表現できない。
        Assert.Equal(GitStatusCode.Modified, status.Index);
        Assert.Equal(GitStatusCode.Modified, status.WorkTree);
        Assert.True(status.IsStaged);
        Assert.True(status.IsDirty);
    }

    [Fact]
    public void 日本語のファイル名が化けない()
    {
        // core.quotepath を切っていないと \346\227\245 の形で返り、突き合わせが壊れる。
        WriteFile("日本語のファイル.txt", "中身\n");

        var status = Open().Status();

        Assert.Single(status, s => s.Path == "日本語のファイル.txt");
    }

    [Fact]
    public void 空白を含むファイル名を扱える()
    {
        WriteFile("a b c.txt", "x\n");

        // path は最後の欄なので、空白で切りすぎると途中で落ちる。
        Assert.Single(Open().Status(), s => s.Path == "a b c.txt");
    }

    [Fact]
    public void 名前が変わったことを本物のgitでも読める()
    {
        WriteFile("もと.txt", string.Join("\n", Enumerable.Range(0, 40).Select(i => $"行 {i}")));
        Git("add", "."); Git("commit", "-m", "最初");
        Git("mv", "もと.txt", "さき.txt");
        WriteFile("べつ.txt", "べつのファイル\n");

        var status = Open().Status();

        var renamed = Assert.Single(status, s => s.Path == "さき.txt");
        Assert.Equal(GitStatusCode.Renamed, renamed.Index);
        Assert.Equal("もと.txt", renamed.OriginalPath);
        // ずれていなければ、後ろの項目も正しく読めている。
        Assert.Single(status, s => s.Path == "べつ.txt");
    }

    [Fact]
    public void 履歴を新しい順に返す()
    {
        WriteFile("a.txt", "1\n"); Git("add", "."); Git("commit", "-m", "ひとつめ");
        WriteFile("a.txt", "2\n"); Git("add", "."); Git("commit", "-m", "ふたつめ");

        var log = Open().Log();

        Assert.Equal(2, log.Count);
        Assert.Equal("ふたつめ", log[0].Subject);
        Assert.Equal("ひとつめ", log[1].Subject);
        Assert.Equal("試験", log[0].Author);
        Assert.Empty(log[1].Parents);
        Assert.Single(log[0].Parents);
    }

    [Fact]
    public void 件数を絞れる()
    {
        for (var i = 0; i < 5; i++)
        {
            WriteFile("a.txt", $"{i}\n");
            Git("add", "."); Git("commit", "-m", $"こみっと {i}");
        }

        Assert.Equal(2, Open().Log(limit: 2).Count);
    }

    [Fact]
    public void 題に制御文字が無くても改行を含む本文で崩れない()
    {
        WriteFile("a.txt", "1\n"); Git("add", ".");
        Git("commit", "-m", "題", "-m", "本文の 1 行目\n本文の 2 行目");

        var commit = Assert.Single(Open().Log());

        // %s は題だけを返す。本文の改行で記録が割れていないこと。
        Assert.Equal("題", commit.Subject);
    }

    [Fact]
    public void ファイルを指定した履歴を取れる()
    {
        WriteFile("a.txt", "1\n"); Git("add", "."); Git("commit", "-m", "a を作る");
        WriteFile("b.txt", "1\n"); Git("add", "."); Git("commit", "-m", "b を作る");

        var log = Open().Log(path: "a.txt");

        Assert.Equal("a を作る", Assert.Single(log).Subject);
    }

    [Fact]
    public void 枝の名前とどれが今かを返す()
    {
        WriteFile("a.txt", "1\n"); Git("add", "."); Git("commit", "-m", "最初");
        Git("branch", "べつの枝");

        var repository = Open();
        var branches = repository.Branches();

        Assert.Equal(2, branches.Count);
        Assert.True(Assert.Single(branches, b => b.Name == "main").IsCurrent);
        Assert.False(Assert.Single(branches, b => b.Name == "べつの枝").IsCurrent);
        Assert.Equal("main", repository.CurrentBranch());
    }

    [Fact]
    public void 切り離されたHEADでは枝の名前が無い()
    {
        WriteFile("a.txt", "1\n"); Git("add", "."); Git("commit", "-m", "最初");
        var repository = Open();
        Git("checkout", "--detach", "HEAD");

        Assert.Null(repository.CurrentBranch());
    }

    [Fact]
    public void ある時点の中身をバイト列で取れる()
    {
        WriteFile("a.txt", "むかし\n"); Git("add", "."); Git("commit", "-m", "1");
        WriteFile("a.txt", "いま\n"); Git("add", "."); Git("commit", "-m", "2");

        var repository = Open();

        Assert.Equal("いま\n", Encoding.UTF8.GetString(repository.Show("HEAD", "a.txt")));
        Assert.Equal("むかし\n", Encoding.UTF8.GetString(repository.Show("HEAD~1", "a.txt")));
    }

    [Fact]
    public void 中身を符号化で決め打たない()
    {
        // Shift_JIS のファイルを入れ、バイト列がそのまま返ることを見る。
        // ここで文字列に直していると、この時点で化けて戻せなくなる。
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var sjis = Encoding.GetEncoding(932);
        var bytes = sjis.GetBytes("日本語\n");
        File.WriteAllBytes(Path.Combine(_root, "sjis.txt"), bytes);
        Git("add", "."); Git("commit", "-m", "sjis");

        Assert.Equal(bytes, Open().Show("HEAD", "sjis.txt"));
    }

    [Fact]
    public void 大きい出力でも詰まらない()
    {
        // 標準出力と標準エラーを同時に読んでいないと、パイプが埋まった時点で
        // 相手が書き込みで止まり、こちらは読み終わらない。
        var big = string.Join("\n", Enumerable.Range(0, 200_000).Select(i => $"行 {i} の中身"));
        WriteFile("big.txt", big);
        Git("add", "."); Git("commit", "-m", "大きいファイル");

        var content = Open().Show("HEAD", "big.txt");

        Assert.True(content.Length > 2_000_000, $"実際の大きさ {content.Length}");
    }

    [Fact]
    public void 無いファイルを求めたら理由を添えて失敗する()
    {
        WriteFile("a.txt", "1\n"); Git("add", "."); Git("commit", "-m", "最初");
        var repository = Open();

        var error = Assert.Throws<GitException>(() => repository.Show("HEAD", "無い.txt"));

        Assert.Contains("無い.txt", error.Message);
        Assert.False(repository.Exists("HEAD", "無い.txt"));
        Assert.True(repository.Exists("HEAD", "a.txt"));
    }

    [Fact]
    public void 絶対パスでも根からの相対に直す()
    {
        WriteFile("nested/a.txt", "1\n"); Git("add", "."); Git("commit", "-m", "最初");
        var repository = Open();

        var content = repository.Show("HEAD", Path.Combine(repository.Root, "nested", "a.txt"));

        Assert.Equal("1\n", Encoding.UTF8.GetString(content));
    }

    [Fact]
    public void 索引へ載せたり降ろしたりできる()
    {
        WriteFile("a.txt", "1\n"); Git("add", "."); Git("commit", "-m", "最初");
        WriteFile("a.txt", "2\n");
        var repository = Open();

        repository.Stage("a.txt");
        Assert.True(Assert.Single(repository.Status(), s => s.Path == "a.txt").IsStaged);

        repository.Unstage("a.txt");
        var after = Assert.Single(repository.Status(), s => s.Path == "a.txt");
        Assert.False(after.IsStaged);
        // 中身は変えていない。降ろすだけ。
        Assert.Equal("2\n", File.ReadAllText(Path.Combine(_root, "a.txt")));
    }

    [Fact]
    public void 競合の三つの材料を取れる()
    {
        WriteFile("f.txt", "もと\n"); Git("add", "."); Git("commit", "-m", "祖先");
        Git("checkout", "-b", "あちら");
        WriteFile("f.txt", "あちらの変更\n"); Git("add", "."); Git("commit", "-m", "あちら");
        Git("checkout", "main");
        WriteFile("f.txt", "こちらの変更\n"); Git("add", "."); Git("commit", "-m", "こちら");

        var repository = Open();
        // 競合させる。失敗して当然なので、終了コードは見ない。
        try { Git("merge", "あちら"); } catch (InvalidOperationException) { }

        var status = Assert.Single(repository.Status(), s => s.Path == "f.txt");
        Assert.True(status.IsConflicted);

        var (ancestor, ours, theirs) = repository.ConflictSources("f.txt");
        Assert.Equal("もと\n", Encoding.UTF8.GetString(ancestor!));
        Assert.Equal("こちらの変更\n", Encoding.UTF8.GetString(ours!));
        Assert.Equal("あちらの変更\n", Encoding.UTF8.GetString(theirs!));
    }

    [Fact]
    public void 競合の材料をそのまま三方向マージへ渡せる()
    {
        // 5.2 と 2.1 が繋がることの確認。
        // **離れた行だけを変えると git は競合しない**（自動でマージが済み、
        // ステージ 1/2/3 が残らない）。競合する行と、競合しない行の両方を作る。
        WriteFile("f.txt", "1\n2\n3\n4\n5\n"); Git("add", "."); Git("commit", "-m", "祖先");
        Git("checkout", "-b", "あちら");
        WriteFile("f.txt", "1\n2\nあちらの 3\n4\nあちらの 5\n");
        Git("add", "."); Git("commit", "-m", "あちら");
        Git("checkout", "main");
        WriteFile("f.txt", "こちらの 1\n2\nこちらの 3\n4\n5\n");
        Git("add", "."); Git("commit", "-m", "こちら");

        var repository = Open();
        // 3 行目で競合する。失敗して当然なので終了コードは見ない。
        try { Git("merge", "あちら"); } catch (InvalidOperationException) { }

        var (ancestor, ours, theirs) = repository.ConflictSources("f.txt");
        Assert.NotNull(ancestor);
        Assert.NotNull(ours);
        Assert.NotNull(theirs);

        // バイト列をそのまま渡せる。符号化の判定は TextDecoder に任せたまま。
        var merged = ThreeWayMerge.Merge(
            TextDecoder.Decode(ancestor), TextDecoder.Decode(ours), TextDecoder.Decode(theirs));

        // git と同じく 3 行目を競合と見る。
        Assert.True(merged.HasConflicts);

        // 片側だけが変えた行は、競合の外で自動的に入る。
        var lines = merged.ToLines();
        Assert.Contains("こちらの 1", lines);
        Assert.Contains("あちらの 5", lines);
    }
}

/// <summary>
/// git を書き換える操作。**本物の git に対して試す。**
/// </summary>
public sealed class GitWriteTests : IDisposable
{
    private readonly string _root;

    public GitWriteTests()
    {
        if (GitRepository.Version() is null)
        {
            throw new InvalidOperationException("この一式には git が要ります。");
        }
        _root = Path.Combine(Path.GetTempPath(), "dc-gitw-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(_root);
        Git("init", "--initial-branch=main");
        Git("config", "user.email", "t@example.com");
        Git("config", "user.name", "試験");
        Git("config", "commit.gpgsign", "false");
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    private void Git(params string[] arguments)
    {
        var info = new System.Diagnostics.ProcessStartInfo("git")
        {
            WorkingDirectory = _root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in arguments)
        {
            info.ArgumentList.Add(argument);
        }
        using var process = System.Diagnostics.Process.Start(info)!;
        var error = process.StandardError.ReadToEnd();
        process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"git {string.Join(' ', arguments)}: {error}");
        }
    }

    private void Write(string name, string content)
        => File.WriteAllText(Path.Combine(_root, name), content, new UTF8Encoding(false));

    private GitRepository Open() => GitRepository.Discover(_root)!;

    [Fact]
    public void 索引に載せてからコミットする()
    {
        Write("a.txt", "1\n");
        var repository = Open();
        repository.Stage("a.txt");

        Assert.True(repository.HasStagedChanges());
        repository.Commit("最初のコミット");

        var commit = Assert.Single(repository.Log());
        Assert.Equal("最初のコミット", commit.Subject);
        Assert.False(repository.HasStagedChanges());
    }

    [Fact]
    public void 載せていないものはコミットに入らない()
    {
        // **索引に載っているものだけが対象**（git の約束をそのまま保つ）。
        Write("staged.txt", "1\n");
        Write("loose.txt", "2\n");
        var repository = Open();
        repository.Stage("staged.txt");
        repository.Commit("片方だけ");

        var status = repository.Status();
        Assert.Contains(status, s => s.Path == "loose.txt" && s.Index == GitStatusCode.Untracked);
    }

    [Fact]
    public void 説明が空ならコミットしない()
    {
        Write("a.txt", "1\n");
        var repository = Open();
        repository.Stage("a.txt");

        Assert.Throws<GitException>(() => repository.Commit("   "));
    }

    [Fact]
    public void 直前のコミットを書き直せる()
    {
        Write("a.txt", "1\n");
        var repository = Open();
        repository.Stage("a.txt");
        repository.Commit("まちがい");

        repository.Commit("なおした", amend: true);

        var commit = Assert.Single(repository.Log());
        Assert.Equal("なおした", commit.Subject);
    }

    [Fact]
    public void 直前の説明を読める()
    {
        Write("a.txt", "1\n");
        var repository = Open();
        repository.Stage("a.txt");
        repository.Commit("題\n\n本文");

        Assert.Contains("本文", repository.LastMessage());
    }

    [Fact]
    public void 枝を作って切り替える()
    {
        Write("a.txt", "1\n");
        var repository = Open();
        repository.Stage("a.txt");
        repository.Commit("最初");

        repository.CreateBranch("あたらしい枝");

        Assert.Equal("あたらしい枝", repository.CurrentBranch());
        Assert.Contains(repository.Branches(), b => b.Name == "あたらしい枝" && b.IsCurrent);
    }

    [Fact]
    public void 枝を切り替えて戻れる()
    {
        Write("a.txt", "1\n");
        var repository = Open();
        repository.Stage("a.txt");
        repository.Commit("最初");
        repository.CreateBranch("わき");

        repository.Switch("main");

        Assert.Equal("main", repository.CurrentBranch());
    }

    [Fact]
    public void 取り込んでいない枝は消せない()
    {
        // **git 側の歯止めをそのまま活かす。** 自前で判断すると、git の
        // 規則と食い違ったときにどちらが正しいか分からなくなる。
        Write("a.txt", "1\n");
        var repository = Open();
        repository.Stage("a.txt");
        repository.Commit("最初");
        repository.CreateBranch("わき");
        Write("b.txt", "2\n");
        repository.Stage("b.txt");
        repository.Commit("わきでの作業");
        repository.Switch("main");

        Assert.Throws<GitException>(() => repository.DeleteBranch("わき"));
        repository.DeleteBranch("わき", force: true);   // 強く言えば消せる
        Assert.DoesNotContain(repository.Branches(), b => b.Name == "わき");
    }
}
