using System.Text;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// 画面を開かずに使える経路。
///
/// GUI しか出口が無いと、動作確認が「人が画面を見る」ことでしか行えない。実際に
/// Windows での検証がそこで詰まった。比較結果をテキストとして出せるようにして、
/// 遠隔からでも別環境の出力と機械的に突き合わせられるようにする。
/// </summary>
internal static class Cli
{
    private static readonly string[] TakesValue =
    [
        "-o", "--threshold", "--ws", "--ignore-pattern",
        "--include", "--exclude", "--tolerance", "--min-size", "--max-size",
        "--merge", "--block", "--report", "--context",
        "--key", "--ignore-column", "--delimiter",
        "--array-key", "--ignore-path",
        "--limit", "--rev", "--path", "--secret-level", "--model", "--vocab",
        "--assist-endpoint", "--assist-model",
        "--link", "--unlink",
    ];

    /// <summary>画面を開かずに済む要求なら処理して終了コードを返す。GUI を開くなら null。</summary>
    public static int? TryRun(string[] args, string usage)
    {
        if (args.Contains("-h") || args.Contains("--help"))
        {
            Console.Write(usage);
            return 0;
        }

        var output = ValueOf(args, "-o");

        if (args.Contains("--font-check"))
        {
            return Report(() => RunFontCheck(output));
        }
        if (args.Contains("--sync"))
        {
            var folders = Positional(args);
            if (folders.Length < 2)
            {
                Console.Error.WriteLine("--sync には 2 つのフォルダーが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunSync(folders[0], folders[1], args, output);
        }
        if (args.Contains("--multi"))
        {
            var files = Positional(args);
            if (files.Length < 2)
            {
                Console.Error.WriteLine("--multi には 2 つ以上のファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            try
            {
                var result = MultiCompare.Compare(
                    [.. files.Select(f => Path.GetFileNameWithoutExtension(f) ?? f)],
                    [.. files.Select(StructuredReaders.ParseFile)]);

                Emit(MultiCompare.Format(result, !args.Contains("--all")), output);
                return result.Differences == 0 ? 0 : 1;
            }
            catch (Exception error) when (error is StructuredParseException or IOException)
            {
                Console.Error.WriteLine(error.Message);
                return 2;
            }
        }
        if (args.Contains("--deps"))
        {
            var files = Positional(args);
            if (files.Length < 2)
            {
                Console.Error.WriteLine("--deps には比較する 2 つのファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            try
            {
                var changes = DependencySummary.Compare(
                    StructuredReaders.ParseFile(files[0]),
                    StructuredReaders.ParseFile(files[1]));

                var text = new StringBuilder();
                text.AppendLine($"left  {files[0]}");
                text.AppendLine($"right {files[1]}");
                text.AppendLine("legend + 追加 / - 削除 / ↑ 上げた / ↓ 下げた / ~ その他");
                text.AppendLine("---");
                text.Append(DependencySummary.Format(changes));

                Emit(text.ToString(), output);
                return changes.Count == 0 ? 0 : 1;
            }
            catch (Exception error) when (error is StructuredParseException or IOException)
            {
                Console.Error.WriteLine(error.Message);
                return 2;
            }
        }
        if (args.Contains("--secrets"))
        {
            var files = Positional(args);
            if (files.Length < 1)
            {
                Console.Error.WriteLine("--secrets には調べるファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunSecrets(files, args, output);
        }
        if (args.Contains("--assist-status") || args.Contains("--assist-commit")
            || args.Contains("--assist-probe") || args.Contains("--assist-resolve"))
        {
            // **比較の経路とは別の入口。** 「比較のつもりが通信していた」を
            // 起こさないため、ここへ来るのは明示的に指定したときだけ。
            var where = Positional(args);
            var path = where.Length > 0 ? where[0] : Environment.CurrentDirectory;
            var settings = AssistCli.ResolveSettings(
                args, SessionStore.Default.LoadFile(), name => ValueOf(args, name));

            using var writer = output is null
                ? Console.Out
                : new StreamWriter(output, false, new UTF8Encoding(false));

            if (args.Contains("--assist-probe"))
            {
                return AssistCli.ProbeAsync(settings, writer).GetAwaiter().GetResult();
            }
            if (args.Contains("--assist-resolve"))
            {
                var files = Positional(args);
                if (files.Length < 3)
                {
                    Console.Error.WriteLine(
                        "--assist-resolve には ＜こちら＞ ＜あちら＞ ＜元＞ の 3 つが要ります");
                    return 2;
                }
                return AssistCli.ProposeResolutionAsync(
                    settings, files[0], files[1], files[2], writer).GetAwaiter().GetResult();
            }
            if (args.Contains("--assist-commit"))
            {
                return AssistCli.DraftCommitAsync(
                    path, settings, args.Contains("--staged"), writer).GetAwaiter().GetResult();
            }
            return AssistCli.ExplainStatusAsync(path, settings, writer).GetAwaiter().GetResult();
        }
        if (args.Contains("--print-embeddings"))
        {
            // **本家の埋め込みと突き合わせるための出口。**
            // トークン列が合っていても、重みの変換や推論がずれていれば
            // 数値は合わない。そこは別に確かめる必要がある。
            var files = Positional(args);
            if (files.Length < 1)
            {
                Console.Error.WriteLine("--print-embeddings には調べるファイルが必要です");
                return 2;
            }
            try
            {
                var embedder = Embedder.CreateFromDefaultAssets(ValueOf(args, "--model"));
                var lines = File.ReadAllLines(files[0]);
                var vectors = embedder.EmbedLines(lines);
                var text = new StringBuilder();
                foreach (var vector in vectors)
                {
                    text.AppendLine(string.Join(' ',
                        vector.Select(v => v.ToString("G9", System.Globalization.CultureInfo.InvariantCulture))));
                }
                Emit(text.ToString(), output);
                return 0;
            }
            catch (Exception error) when (error is IOException or InvalidDataException
                                            or FileNotFoundException)
            {
                Console.Error.WriteLine(error.Message);
                return 2;
            }
        }
        if (args.Contains("--tokenize"))
        {
            // **参照実装と突き合わせるための出口。** トークナイザーは
            // 「合っているか」しか分からないので、外と比べられる形が要る。
            var files = Positional(args);
            var vocabPath = ValueOf(args, "--vocab");
            if (files.Length < 1 || vocabPath is null)
            {
                Console.Error.WriteLine("--tokenize には調べるファイルと --vocab が必要です");
                return 2;
            }
            try
            {
                using var vocab = File.OpenRead(vocabPath);
                var tokenizer = UnigramTokenizer.FromVocab(vocab);
                var text = new StringBuilder();
                foreach (var line in File.ReadLines(files[0]))
                {
                    text.AppendLine(string.Join(' ', tokenizer.Tokenize(line)));
                }
                Emit(text.ToString(), output);
                return 0;
            }
            catch (Exception error) when (error is IOException or InvalidDataException)
            {
                Console.Error.WriteLine(error.Message);
                return 2;
            }
        }
        if (args.Contains("--print-office"))
        {
            var files = Positional(args);
            if (files.Length < 1)
            {
                Console.Error.WriteLine("--print-office には Office 文書が 1 つか 2 つ必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunOfficeCompare(files, args, output);
        }
        if (args.Contains("--print-notebook"))
        {
            var files = Positional(args);
            if (files.Length < 2)
            {
                Console.Error.WriteLine("--print-notebook には比較する 2 つの .ipynb が必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunNotebookCompare(files[0], files[1], args, output);
        }
        if (args.Contains("--strip-notebook"))
        {
            var files = Positional(args);
            if (files.Length < 1)
            {
                Console.Error.WriteLine("--strip-notebook には .ipynb が必要です");
                Console.Error.Write(usage);
                return 2;
            }
            try
            {
                var stripped = Notebook.Strip(File.ReadAllText(files[0]));
                if (args.Contains("--in-place"))
                {
                    File.WriteAllText(files[0], stripped);
                    Console.Error.WriteLine($"{files[0]} から実行の跡を落としました。");
                    return 0;
                }
                Emit(stripped, output);
                return 0;
            }
            catch (Exception error) when (error is IOException or StructuredParseException
                                            or InvalidDataException)
            {
                Console.Error.WriteLine(error.Message);
                return 2;
            }
        }
        if (args.Contains("--print-version-info"))
        {
            var files = Positional(args);
            if (files.Length < 1)
            {
                Console.Error.WriteLine("--print-version-info には実行ファイルが 1 つか 2 つ必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunVersionInfo(files, args, output);
        }
        if (args.Contains("--snapshot"))
        {
            var where = Positional(args);
            if (where.Length < 1)
            {
                Console.Error.WriteLine("--snapshot には写し取るフォルダーが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunSnapshot(where[0], args, output);
        }
        if (args.Contains("--snapshot-diff"))
        {
            var files = Positional(args);
            if (files.Length < 1)
            {
                Console.Error.WriteLine("--snapshot-diff には写しが 1 つか 2 つ必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunSnapshotDiff(files, args, output);
        }
        if (args.Contains("--print-image"))
        {
            var files = Positional(args);
            if (files.Length < 2)
            {
                Console.Error.WriteLine("--print-image には比較する 2 つの画像が必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunImageCompare(files[0], files[1], args, output);
        }
        if (args.Contains("--print-binary"))
        {
            var files = Positional(args);
            if (files.Length < 2)
            {
                Console.Error.WriteLine("--print-binary には比較する 2 つのファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            try
            {
                var (comparison, truncated) = BinaryCompare.CompareFiles(files[0], files[1]);
                var text = new StringBuilder();
                if (truncated)
                {
                    // 切り捨てたことは必ず出す。黙って途中までを比べると、
                    // 「差分なし」が「先頭 64MB に差分なし」の意味になってしまう。
                    text.AppendLine("注意: 大きいので先頭 64MB だけを比べています。");
                }
                text.Append(BinaryCompare.Format(comparison));
                Emit(text.ToString(), output);
                return comparison.Identical ? 0 : 1;
            }
            catch (IOException error)
            {
                Console.Error.WriteLine(error.Message);
                return 2;
            }
        }
        if (args.Contains("--invisible"))
        {
            var files = Positional(args);
            if (files.Length < 1)
            {
                Console.Error.WriteLine("--invisible には調べるファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunInvisible(files[0], output);
        }
        if (args.Contains("--git-status"))
        {
            var where = Positional(args);
            return RunGitStatus(where.Length > 0 ? where[0] : Environment.CurrentDirectory, args, output);
        }
        if (args.Contains("--git-log"))
        {
            var where = Positional(args);
            return RunGitLog(where.Length > 0 ? where[0] : Environment.CurrentDirectory, args, output);
        }
        if (args.Contains("--git-diff"))
        {
            var where = Positional(args);
            if (where.Length < 1)
            {
                Console.Error.WriteLine("--git-diff には比較するファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunGitDiff(where[0], args, output);
        }
        if (args.Contains("--git-resolve"))
        {
            var where = Positional(args);
            if (where.Length < 1)
            {
                Console.Error.WriteLine("--git-resolve には解決するファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunGitResolve(where[0], args, output);
        }
        if (args.Contains("--print-json"))
        {
            var files = Positional(args);
            if (files.Length < 2)
            {
                Console.Error.WriteLine("--print-json には比較する 2 つのファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunStructuredCompare(files[0], files[1], args, output);
        }
        if (args.Contains("--print-table"))
        {
            var files = Positional(args);
            if (files.Length < 2)
            {
                Console.Error.WriteLine("--print-table には比較する 2 つのファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return Report(() => RunTableCompare(files[0], files[1], args, output));
        }
        if (args.Contains("--merge3"))
        {
            var three = Positional(args);
            if (three.Length < 3)
            {
                Console.Error.WriteLine("--merge3 には 祖先 左 右 の 3 つが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return Report(() => RunThreeWay(three[0], three[1], three[2], args, output));
        }
        if (args.Contains("--merge"))
        {
            var files = Positional(args);
            if (files.Length < 2)
            {
                Console.Error.WriteLine("--merge には左右 2 つのファイルが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return Report(() => RunMerge(files[0], files[1], args, output));
        }
        if (args.Contains("--print-folder"))
        {
            var folders = Positional(args);
            if (folders.Length < 2)
            {
                Console.Error.WriteLine("--print-folder には比較する 2 つのフォルダーが必要です");
                Console.Error.Write(usage);
                return 2;
            }
            return RunFolderCompare(folders[0], folders[1], args, output);
        }
        if (!args.Contains("--print"))
        {
            return null;
        }

        var positional = Positional(args);
        if (positional.Length < 2)
        {
            Console.Error.WriteLine("--print には比較する 2 つのファイルが必要です");
            Console.Error.Write(usage);
            return 2;
        }

        var reportFormat = ValueOf(args, "--report");
        var threshold = float.TryParse(ValueOf(args, "--threshold"), out var parsed)
            ? parsed
            : Aligner.DefaultPairThreshold;
        Importance importance;
        try
        {
            importance = new Importance(
                ParseWhitespace(ValueOf(args, "--ws")),
                args.Contains("--ignore-case") || args.Contains("-i"),
                ValuesOf(args, "--ignore-pattern"),
                args.Contains("--normalize-unicode"));
        }
        catch (ArgumentException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        return Report(() => RunCompare(
            positional[0], positional[1], threshold, output,
            args.Contains("--structural"), importance, reportFormat,
            int.TryParse(ValueOf(args, "--context"), out var ctx) ? ctx : 3,
            ValueOf(args, "--model"), ManualFrom(args)));
    }

    /// <summary>
    /// <c>--link 2:3</c> の形で指定された対応付けを読む。**行番号は 1 始まり**
    /// （画面に出ている番号と同じにする。0 始まりだと必ず 1 つずれる）。
    /// </summary>
    private static ManualAlignment? ManualFrom(string[] args)
    {
        var manual = new ManualAlignment();
        var any = false;

        foreach (var (option, link) in new[] { ("--link", true), ("--unlink", false) })
        {
            foreach (var value in ValuesOf(args, option) ?? [])
            {
                var parts = value.Split(':');
                if (parts.Length != 2
                    || !int.TryParse(parts[0], out var l) || !int.TryParse(parts[1], out var r)
                    || l < 1 || r < 1)
                {
                    Console.Error.WriteLine($"{option} は 左行:右行 の形で（1 始まり）: {value}");
                    continue;
                }
                manual = link ? manual.Link(l - 1, r - 1) : manual.Unlink(l - 1, r - 1);
                any = true;
            }
        }
        return any ? manual : null;
    }

    /// <summary>オプションとその値を取り除いた、位置引数だけの列。</summary>
    public static string[] Positional(string[] args)
    {
        var result = new List<string>();
        var skipNext = false;
        foreach (var arg in args)
        {
            if (skipNext)
            {
                skipNext = false;
                continue;
            }
            if (TakesValue.Contains(arg))
            {
                skipNext = true;
            }
            else if (!arg.StartsWith('-'))
            {
                result.Add(arg);
            }
        }
        return result.ToArray();
    }

    /// <summary>
    /// フォルダーの同期。
    ///
    /// **既定では予定を出すだけで、何も書き換えない。** 実際に走らせるには
    /// --apply が要る。同期は取り返しがつかないので、見てから決められるようにする。
    /// </summary>
    private static int RunSync(string left, string right, string[] args, string? output)
    {
        var options = new SyncOptions
        {
            Direction = ValueOf(args, "--direction") switch
            {
                "to-left" => SyncDirection.ToLeft,
                "both" => SyncDirection.Both,
                _ => SyncDirection.ToRight,
            },
            DeleteOrphans = args.Contains("--delete-orphans"),
            ToleranceSeconds = TimestampTolerance(args),
        };

        FolderComparison comparison;
        try
        {
            comparison = FolderComparer.Compare(left, right, new FolderCompareOptions
            {
                Filter = new NameFilter(ValuesOf(args, "--include"), ValuesOf(args, "--exclude")),
            });
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        var plan = FolderSync.Plan(comparison, options);

        var text = new StringBuilder();
        text.AppendLine($"left  {left}");
        text.AppendLine($"right {right}");
        text.AppendLine("legend → 右へ写す / ← 左へ写す / ✕ 消す");
        text.AppendLine("---");
        text.Append(FolderSync.Format(plan));

        if (!args.Contains("--apply"))
        {
            text.AppendLine();
            text.AppendLine("（予定を出しただけです。実行するには --apply を付けてください）");
            Emit(text.ToString(), output);
            return plan.IsEmpty ? 0 : 1;
        }

        var result = FolderSync.Apply(plan, left, right);
        text.AppendLine();
        text.AppendLine($"{result.Done} 件を実行しました。");
        foreach (var error in result.Errors)
        {
            text.AppendLine($"失敗: {error}");
        }

        Emit(text.ToString(), output);
        return result.AllSucceeded ? 0 : 2;
    }

    private static double TimestampTolerance(string[] args)
        => double.TryParse(ValueOf(args, "--tolerance"), out var value) ? value : 2;

    /// <summary>
    /// 秘密が混ざっていないか調べる。
    ///
    /// ファイルを 1 つ渡せばその全体を、2 つ渡せば**増えた側だけ**を見る。
    /// 終了コードは 0 何も無い / 1 見つかった / 2 異常。CI で止められる。
    /// </summary>
    private static int RunSecrets(string[] files, string[] args, string? output)
    {
        IReadOnlyList<SecretFinding> findings;
        try
        {
            if (files.Length >= 2)
            {
                var left = TextDecoder.Decode(File.ReadAllBytes(files[0]));
                var right = TextDecoder.Decode(File.ReadAllBytes(files[1]));
                var comparison = DiffComparer.Compare(left, right, embedder: null);
                findings = SecretScanner.ScanAdded(comparison, right);
            }
            else
            {
                findings = SecretScanner.Scan(
                    TextDecoder.Decode(File.ReadAllBytes(files[0])).Lines);
            }
        }
        catch (IOException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        // 弱い印まで数えて止めると、CI が通らなくなって外される。
        // **どこから止めるかを選べるようにする。**
        var least = ValueOf(args, "--secret-level") switch
        {
            "low" => SecretConfidence.Low,
            "high" => SecretConfidence.High,
            _ => SecretConfidence.Medium,
        };
        var kept = findings.Where(f => f.Confidence <= least).ToList();

        var text = new StringBuilder();
        text.AppendLine(files.Length >= 2
            ? $"increase {files[0]} → {files[1]}（増えた行だけ）"
            : $"file {files[0]}");
        text.AppendLine("---");
        text.Append(SecretScanner.Format(kept));

        Emit(text.ToString(), output);
        return kept.Count == 0 ? 0 : 1;
    }

    /// <summary>
    /// 見えない差分を調べる。
    ///
    /// 終了コードは 0 何も無い / 1 見つかった / 2 異常。
    /// 「同じに見えるのに一致しない」の原因を出すのが役目。
    /// </summary>
    private static int RunInvisible(string path, string? output)
    {
        IReadOnlyList<InvisibleFinding> findings;
        try
        {
            findings = InvisibleScanner.Scan(TextDecoder.Decode(File.ReadAllBytes(path)));
        }
        catch (IOException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        var text = new StringBuilder();
        text.AppendLine($"file {path}");
        text.AppendLine("---");
        text.Append(InvisibleScanner.Format(findings));

        Emit(text.ToString(), output);
        return findings.Count == 0 ? 0 : 1;
    }

    /// <summary>git が使えなければ理由を出して 2 を返す。呼ぶ側の分岐をまとめる。</summary>
    private static GitRepository? OpenRepository(string path)
    {
        if (GitRepository.Version() is null)
        {
            Console.Error.WriteLine("git が見つかりません。Git 機能を使うには git を入れてください。");
            return null;
        }
        var repository = GitRepository.Discover(path);
        if (repository is null)
        {
            Console.Error.WriteLine($"{path} は git リポジトリの中にありません。");
        }
        return repository;
    }

    /// <summary>
    /// 作業ツリーの状態。
    ///
    /// 終了コードは 0 きれい / 1 変更あり / 2 異常。CI で「作業ツリーが汚れていないか」
    /// を見る使い方ができる。
    /// </summary>
    private static int RunGitStatus(string path, string[] args, string? output)
    {
        var repository = OpenRepository(path);
        if (repository is null)
        {
            return 2;
        }

        List<GitFileStatus> files;
        try
        {
            files = [.. repository.Status()];
        }
        catch (GitException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        if (args.Contains("--changes-only"))
        {
            files.RemoveAll(f => f.Index == GitStatusCode.Unchanged
                && f.WorkTree == GitStatusCode.Unchanged);
        }

        var text = new StringBuilder();
        text.AppendLine($"root   {repository.Root}");
        text.AppendLine($"branch {repository.CurrentBranch() ?? "(切り離された HEAD)"}");
        text.AppendLine("legend 索引 / 作業ツリー の順。. は変化なし");
        text.AppendLine("---");
        foreach (var file in files.OrderBy(f => f.Path, StringComparer.Ordinal))
        {
            var mark = file.IsConflicted ? "競合" : $"{Mark(file.Index)}{Mark(file.WorkTree)}";
            var name = file.OriginalPath is { Length: > 0 } original
                ? $"{original} -> {file.Path}"
                : file.Path;
            text.AppendLine($"{mark} {name}");
        }
        text.AppendLine("---");
        var untracked = files.Count(f => f.Index == GitStatusCode.Untracked);
        text.AppendLine($"合計 {files.Count} 件"
            + $"（stage 済み {files.Count(f => f.IsStaged)}"
            + $" / 未 stage {files.Count(f => f.IsDirty && f.Index != GitStatusCode.Untracked)}"
            + $" / 未追跡 {untracked}"
            + $" / 競合 {files.Count(f => f.IsConflicted)}）");

        Emit(text.ToString(), output);
        return files.Count == 0 ? 0 : 1;

        static string Mark(GitStatusCode code) => code switch
        {
            GitStatusCode.Unchanged => ".",
            GitStatusCode.Modified => "M",
            GitStatusCode.Added => "A",
            GitStatusCode.Deleted => "D",
            GitStatusCode.Renamed => "R",
            GitStatusCode.Copied => "C",
            GitStatusCode.TypeChanged => "T",
            GitStatusCode.Untracked => "?",
            GitStatusCode.Ignored => "!",
            GitStatusCode.Unmerged => "U",
            _ => " ",
        };
    }

    private static int RunGitLog(string path, string[] args, string? output)
    {
        var repository = OpenRepository(path);
        if (repository is null)
        {
            return 2;
        }

        try
        {
            var limit = int.TryParse(ValueOf(args, "--limit"), out var n) ? n : 50;
            var commits = repository.Log(limit, ValueOf(args, "--rev"), ValueOf(args, "--path"));

            var text = new StringBuilder();
            foreach (var commit in commits)
            {
                var merge = commit.IsMerge ? " [マージ]" : string.Empty;
                text.AppendLine(
                    $"{commit.ShortHash} {commit.When:yyyy-MM-dd HH:mm} {commit.Author}{merge}");
                text.AppendLine($"    {commit.Subject}");
            }
            text.AppendLine($"--- {commits.Count} 件");
            Emit(text.ToString(), output);
            return 0;
        }
        catch (GitException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// ある時点のファイルと、いまの中身を比べる。
    ///
    /// **git 本体の diff ではなく、こちらの比較エンジンに掛ける。** 意味的な行の
    /// 対応付けが効くので、名前を変えただけの行が並ぶ。ここが git だけでは出せない部分。
    /// </summary>
    private static int RunGitDiff(string path, string[] args, string? output)
    {
        var repository = OpenRepository(path);
        if (repository is null)
        {
            return 2;
        }

        try
        {
            var revision = ValueOf(args, "--rev") ?? "HEAD";
            // 中身はバイト列で受け、符号化の判定は普段と同じ経路に通す。
            var left = TextDecoder.Decode(repository.Show(revision, path));
            var right = TextDecoder.Decode(File.ReadAllBytes(path));

            var embedder = args.Contains("--structural") ? null : Embedder.CreateFromDefaultAssets(ValueOf(args, "--model"));
            var comparison = DiffComparer.Compare(left, right, embedder, new CompareOptions(
                float.TryParse(ValueOf(args, "--threshold"), out var t) ? t : Aligner.DefaultPairThreshold));

            var text = new StringBuilder();
            text.AppendLine($"left  {revision}:{repository.ToRelative(path)}");
            text.AppendLine($"right {path}（作業ツリー）");
            text.AppendLine("---");
            text.Append(DeepCompare.Engine.Report.UnifiedDiff(
                comparison, left, right,
                $"a/{repository.ToRelative(path)}",
                $"b/{repository.ToRelative(path)}",
                int.TryParse(ValueOf(args, "--context"), out var ctx) ? ctx : 3));

            Emit(text.ToString(), output);
            return comparison.Rows.Any(r => !r.IsUnchanged) ? 1 : 0;
        }
        catch (Exception error) when (error is GitException or IOException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// Office 文書の**本文**を比べる。
    ///
    /// 1 つなら取り出した本文を出す（何を見ているかを確かめられる）。
    /// 終了コードは 0 差異なし / 1 差異あり / 2 異常。
    /// </summary>
    private static int RunOfficeCompare(string[] files, string[] args, string? output)
    {
        try
        {
            var left = OfficeDocument.Read(files[0]);
            if (files.Length < 2)
            {
                Emit(OfficeDocument.Format(left), output);
                return 0;
            }

            var right = OfficeDocument.Read(files[1]);

            // **本文を行として比べる。** 位置（段落番号やセル番地）を頭に付けて
            // あるので、どこが変わったかがそのまま出る。
            var leftLines = left.ToLines();
            var rightLines = right.ToLines();

            var text = new StringBuilder();
            text.AppendLine($"left  {files[0]}");
            text.AppendLine($"right {files[1]}");
            text.AppendLine("---");

            var differences = 0;
            foreach (var op in Myers.Compute(leftLines, rightLines))
            {
                switch (op.Kind)
                {
                    case DiffKind.Equal:
                        if (args.Contains("--all"))
                        {
                            for (var i = 0; i < op.OldLength; i++)
                            {
                                text.AppendLine($"  {leftLines[op.OldStart + i]}");
                            }
                        }
                        break;
                    case DiffKind.Replace:
                    case DiffKind.Delete:
                        for (var i = 0; i < op.OldLength; i++)
                        {
                            text.AppendLine($"- {leftLines[op.OldStart + i]}");
                            differences++;
                        }
                        goto case DiffKind.Insert;
                    case DiffKind.Insert:
                        for (var i = 0; i < op.NewLength; i++)
                        {
                            text.AppendLine($"+ {rightLines[op.NewStart + i]}");
                            differences++;
                        }
                        break;
                }
            }

            text.AppendLine("---");
            text.AppendLine(differences == 0 ? "本文は同じです。" : $"{differences} 箇所が違います。");
            Emit(text.ToString(), output);
            return differences > 0 ? 1 : 0;
        }
        catch (Exception error) when (error is IOException or NotSupportedException
                                        or InvalidDataException or System.Xml.XmlException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// ノートブックをセル単位で比べる。
    ///
    /// 終了コードは 0 本文に変化なし / 1 本文が変わった / 2 異常。
    /// **出力だけの違いは 0 を返す。** 実行しただけで CI が赤くなるのを避ける。
    /// </summary>
    private static int RunNotebookCompare(
        string leftPath, string rightPath, string[] args, string? output)
    {
        try
        {
            var options = new NotebookCompareOptions
            {
                CompareOutputs = args.Contains("--with-outputs"),
                CompareExecutionCount = args.Contains("--with-execution-count"),
            };

            var comparison = Notebook.Compare(
                Notebook.Read(File.ReadAllText(leftPath)),
                Notebook.Read(File.ReadAllText(rightPath)),
                options);

            var text = new StringBuilder();
            text.AppendLine($"left  {leftPath}");
            text.AppendLine($"right {rightPath}");
            text.AppendLine("legend ~ 本文が変わった / + 増えた / - 消えた / o 出力だけ / ! メタデータ");
            text.AppendLine("---");
            text.Append(Notebook.Format(comparison, args.Contains("--all")));
            Emit(text.ToString(), output);

            return comparison.HasSourceChanges ? 1 : 0;
        }
        catch (Exception error) when (error is IOException or StructuredParseException
                                        or UnauthorizedAccessException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// 実行ファイルのバージョン情報。1 つなら中身を出し、2 つなら比べる。
    ///
    /// 終了コードは 0 同じ（または 1 つ指定）/ 1 違う / 2 異常。
    /// </summary>
    private static int RunVersionInfo(string[] files, string[] args, string? output)
    {
        try
        {
            var left = VersionInfo.Read(files[0]);
            if (files.Length < 2)
            {
                // 1 つだけなら、比較の形（左だけ埋まった表）で出す。
                // **見せ方を 2 通り持たない。**
                var single = VersionInfo.Compare(left, left)
                    .Select(d => d with { Right = null }).ToList();
                var text = new StringBuilder();
                text.AppendLine(files[0]);
                text.AppendLine("---");
                foreach (var item in single)
                {
                    text.AppendLine($"  {item.Key,-24} {item.Left ?? "（無し）"}");
                }
                Emit(text.ToString(), output);
                return 0;
            }

            var right = VersionInfo.Read(files[1]);
            var differences = VersionInfo.Compare(left, right);

            var report = new StringBuilder();
            report.AppendLine($"left  {files[0]}");
            report.AppendLine($"right {files[1]}");
            report.AppendLine("---");
            report.Append(VersionInfo.Format(differences, !args.Contains("--all")));
            Emit(report.ToString(), output);

            return differences.Any(d => !d.IsSame) ? 1 : 0;
        }
        catch (Exception error) when (error is IOException or InvalidDataException
                                        or UnauthorizedAccessException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// フォルダーの状態を写し取る。
    ///
    /// **書き出し先を指定しなければ標準出力へ出す。** 他の道具へ流し込めるように
    /// するのと、うっかりファイルが増えないようにするため。
    /// </summary>
    private static int RunSnapshot(string root, string[] args, string? output)
    {
        if (!Directory.Exists(root))
        {
            Console.Error.WriteLine($"{root} はフォルダーではありません。");
            return 2;
        }

        try
        {
            var filter = new NameFilter(ValuesOf(args, "--include"), ValuesOf(args, "--exclude"));

            var snapshot = Snapshots.Take(root, withHashes: args.Contains("--hash"), filter: filter);
            Emit(Snapshots.Save(snapshot), output);

            // 件数は標準エラーへ。標準出力は写しそのものなので、混ぜると
            // **そのまま読み直せなくなる。**
            Console.Error.WriteLine(
                $"{snapshot.FileCount} ファイル / {snapshot.DirectoryCount} フォルダー"
                + (snapshot.HasHashes ? "（指紋あり）" : "（指紋なし）"));
            return 0;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// 写し同士、または写しと今のフォルダーを比べる。
    ///
    /// 引数が 1 つなら、写しに書いてある元の場所を今の姿と比べる
    /// （**一番よくある使い方**。「あの時から何が変わったか」）。
    ///
    /// 終了コードは 0 変化なし / 1 変化あり / 2 異常。
    /// </summary>
    private static int RunSnapshotDiff(string[] files, string[] args, string? output)
    {
        try
        {
            var before = Snapshots.Load(File.ReadAllText(files[0]));

            Snapshot after;
            if (files.Length >= 2)
            {
                after = Snapshots.Load(File.ReadAllText(files[1]));
            }
            else
            {
                if (!Directory.Exists(before.Root))
                {
                    Console.Error.WriteLine(
                        $"写しに書かれた {before.Root} が今はありません。"
                        + "比べる相手の写しを 2 つ目に渡してください。");
                    return 2;
                }
                // **写しと同じ条件で取り直す。** 片方だけ指紋があると、
                // 中身の変化を見分けられたり見分けられなかったりして食い違う。
                after = Snapshots.Take(before.Root, withHashes: before.HasHashes);
            }

            var result = Snapshots.Compare(before, after);
            var stats = result.Stats;

            var text = new StringBuilder();
            text.AppendLine($"before {before.Root}  {before.TakenAt:yyyy-MM-dd HH:mm:ss}");
            text.AppendLine(files.Length >= 2
                ? $"after  {after.Root}  {after.TakenAt:yyyy-MM-dd HH:mm:ss}"
                : $"after  {after.Root}（今）");
            if (!before.HasHashes)
            {
                // **指紋が無いことは必ず言う。** 「変化なし」が「大きさと時刻に
                // 変化なし」の意味になっているのを黙っていると、嘘に近い。
                text.AppendLine("注意: 指紋なしの写しです。大きさと時刻でしか比べていません。");
            }
            text.AppendLine($"stats different={stats.Different} removed={stats.LeftOnly} "
                + $"added={stats.RightOnly} identical={stats.Identical}");
            text.AppendLine("legend ~ 変わった / - 消えた / + 増えた");
            text.AppendLine("---");

            foreach (var entry in result.Entries)
            {
                var kind = entry.Status switch
                {
                    EntryStatus.Different => '~',
                    EntryStatus.LeftOnly => '-',
                    EntryStatus.RightOnly => '+',
                    _ => '=',
                };
                if (kind == '=' && !args.Contains("--all"))
                {
                    continue;
                }
                text.AppendLine($"{kind} {entry.RelativePath}");
            }

            Emit(text.ToString(), output);
            return stats.Different > 0 || stats.LeftOnly > 0 || stats.RightOnly > 0 ? 1 : 0;
        }
        catch (Exception error) when (error is IOException or InvalidDataException
                                        or UnauthorizedAccessException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// 画像を画素で比べる。
    ///
    /// 終了コードは 0 同じ / 1 違う / 2 異常。
    /// **しきい値の内側の差だけなら 0 を返す。** JPEG を保存し直しただけの
    /// 違いで CI が赤くなるのは、ほとんどの場合ただの雑音になる。
    /// </summary>
    private static int RunImageCompare(string leftPath, string rightPath, string[] args, string? output)
    {
        try
        {
            var options = new ImageCompareOptions
            {
                Tolerance = int.TryParse(ValueOf(args, "--tolerance"), out var t) ? t : 8,
                CompareAlpha = !args.Contains("--ignore-alpha"),
            };

            var comparison = ImageCompare.Compare(
                ImageLoader.Load(leftPath), ImageLoader.Load(rightPath), options);

            var text = new StringBuilder();
            text.AppendLine($"left  {leftPath}");
            text.AppendLine($"right {rightPath}");
            text.AppendLine("---");
            text.Append(ImageCompare.Format(comparison));
            Emit(text.ToString(), output);

            return comparison.LooksSame ? 0 : 1;
        }
        catch (Exception error) when (error is IOException or InvalidOperationException
                                        or NotSupportedException or ArgumentException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// 競合しているファイルを 3 方向マージで解く。
    ///
    /// 索引に積まれた 3 つ（祖先・こちら・むこう）を使う。**作業ツリーの
    /// ファイルは読まない。** そこには git が書いた印が混ざっている。
    ///
    /// 終了コードは 0 解決した / 1 競合が残っている / 2 異常。
    /// </summary>
    private static int RunGitResolve(string path, string[] args, string? output)
    {
        var repository = OpenRepository(path);
        if (repository is null)
        {
            return 2;
        }

        try
        {
            // 根からの相対に直すために、まず絶対パスにする。相対のまま渡すと
            // 現在地がリポジトリの根でないときにずれる。
            var relative = repository.ToRelative(Path.GetFullPath(path));
            var ours = repository.ConflictStage(relative, 2);
            var theirs = repository.ConflictStage(relative, 3);
            if (ours.Length == 0 && theirs.Length == 0)
            {
                Console.Error.WriteLine($"{relative} は競合していません。");
                return 2;
            }

            var ancestorText = TextDecoder.Decode(repository.ConflictStage(relative, 1));
            var oursText = TextDecoder.Decode(ours);
            var theirsText = TextDecoder.Decode(theirs);

            var result = ThreeWayMerge.Merge(ancestorText, oursText, theirsText);

            // どちらかに寄せる指示があれば、そこだけ決める。
            var take = args.Contains("--take-ours") ? MergeSource.Left
                : args.Contains("--take-theirs") ? MergeSource.Right
                : (MergeSource?)null;

            if (result.HasConflicts && take is null)
            {
                var text = new StringBuilder();
                text.AppendLine($"{relative}: {result.ConflictCount} 件の競合");
                text.AppendLine("--take-ours か --take-theirs を付けると、その側に寄せて解決します。");
                text.AppendLine("---");
                var number = 0;
                foreach (var region in result.Regions.Where(r => r.Source == MergeSource.Conflict))
                {
                    text.AppendLine($"[{++number}] こちら:");
                    foreach (var line in region.LeftLines)
                    {
                        text.AppendLine($"  {line}");
                    }
                    text.AppendLine($"[{number}] むこう:");
                    foreach (var line in region.RightLines)
                    {
                        text.AppendLine($"  {line}");
                    }
                }
                Emit(text.ToString(), output);
                return 1;
            }

            var lines = new List<string>();
            foreach (var region in result.Regions)
            {
                lines.AddRange(region.Source == MergeSource.Conflict
                    ? (take == MergeSource.Left ? region.LeftLines : region.RightLines)
                    : region.Lines);
            }

            // 符号化と改行は「こちら」に合わせる。祖先が無い競合もあるので、
            // 祖先を基準にすると、その場合に形が変わってしまう。
            File.WriteAllBytes(path, TextEncoder.Encode(lines, oursText));
            repository.Stage(relative);

            Emit($"{relative} を解決して索引へ載せました"
                + (take is null ? "（競合はありませんでした）" : $"（{(take == MergeSource.Left ? "こちら" : "むこう")}に寄せました）")
                + Environment.NewLine, output);
            return 0;
        }
        catch (Exception error) when (error is GitException or IOException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>
    /// 構造化データの比較。
    ///
    /// 終了コードは --print-folder と揃える。0 差異なし / 1 差異あり / 2 異常。
    /// CI で「設定ファイルが意図せず変わっていないか」を見る使い方を想定している。
    /// </summary>
    private static int RunStructuredCompare(string leftPath, string rightPath, string[] args, string? output)
    {
        StructuredCompareOptions options;
        IReadOnlyList<StructuralChange> changes;
        try
        {
            var arrayKeys = ValuesOf(args, "--array-key");
            options = new StructuredCompareOptions
            {
                // 指定が無ければ既定の候補（id, name, key, path）を使う。
                ArrayKeys = arrayKeys.Count > 0 ? arrayKeys : new StructuredCompareOptions().ArrayKeys,
                IgnoredPaths = ValuesOf(args, "--ignore-path"),
                ReportMoves = !args.Contains("--ignore-order"),
                NumbersByValue = !args.Contains("--strict-numbers"),
            };
            // 形式は拡張子から決める。**左右で違っても構わない**
            // （YAML と JSON を突き合わせることは実際にある）。
            changes = StructuredCompare.Compare(
                StructuredReaders.ParseFile(leftPath),
                StructuredReaders.ParseFile(rightPath),
                options);
        }
        catch (Exception error) when (error is StructuredParseException or IOException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        var text = new StringBuilder();
        text.AppendLine($"left  {leftPath}");
        text.AppendLine($"right {rightPath}");
        text.AppendLine("legend + 右のみ / - 左のみ / ~ 変更 / ! 型の変化 / → 位置の変化");
        text.AppendLine("---");
        text.Append(StructuredCompare.Format(changes));

        Emit(text.ToString(), output);
        return changes.Count == 0 ? 0 : 1;
    }

    /// <summary>
    /// 表形式の比較。列の位置は 1 始まりで受け、内部の 0 始まりへ直す。
    /// 表計算の列番号と揃えないと、指定を間違えたことに気づきにくい。
    /// </summary>
    private static int RunTableCompare(string leftPath, string rightPath, string[] args, string? output)
    {
        var format = TableFormat.ForPath(leftPath) with
        {
            HasHeader = !args.Contains("--no-header"),
        };
        if (ValueOf(args, "--delimiter") is { Length: > 0 } delimiter)
        {
            format = format with
            {
                Delimiter = delimiter switch
                {
                    "tab" or "\\t" => '\t',
                    _ => delimiter[0],
                },
            };
        }

        var left = TableCompare.Parse(File.ReadAllText(leftPath), format);
        var right = TableCompare.Parse(File.ReadAllText(rightPath), format);

        static List<int> Columns(string[] args, string flag, Table table)
            => [.. ValuesOf(args, flag).Select(v =>
                int.TryParse(v, out var number)
                    ? number - 1
                    : table.IndexOfColumn(v) is var found && found >= 0
                        ? found
                        : throw new ArgumentException($"列が見つからない: {v}"))];

        var keys = Columns(args, "--key", left);
        var ignored = Columns(args, "--ignore-column", left);

        var embedder = args.Contains("--structural") ? null : Embedder.CreateFromDefaultAssets(ValueOf(args, "--model"));
        var result = TableCompare.Compare(left, right, keys, ignored, embedder);

        var text = new StringBuilder();
        text.AppendLine($"left  {leftPath} 行={left.Rows.Count} 列={left.ColumnCount}");
        text.AppendLine($"right {rightPath} 行={right.Rows.Count} 列={right.ColumnCount}");
        text.AppendLine($"stats different={result.Different} left_only={result.LeftOnly} "
            + $"right_only={result.RightOnly}"
            + (keys.Count > 0 ? $" key={string.Join(',', keys.Select(k => k + 1))}" : string.Empty));
        text.AppendLine("legend = 一致 / ~ 差異あり / - 左のみ / + 右のみ");
        text.AppendLine("---");

        foreach (var row in result.Rows)
        {
            var kind = (row.Left, row.Right) switch
            {
                (not null, not null) when row.IsUnchanged => '=',
                (not null, not null) => '~',
                (not null, null) => '-',
                _ => '+',
            };
            var leftNumber = row.Left is { } l ? (l + 1).ToString() : string.Empty;
            var rightNumber = row.Right is { } r ? (r + 1).ToString() : string.Empty;
            var columns = row.ChangedColumns.Count == 0
                ? string.Empty
                : "列 " + string.Join(',', row.ChangedColumns.Select(c =>
                    c < left.Header.Count ? left.Header[c] : (c + 1).ToString()));
            var content = row.Left is { } li
                ? string.Join(" | ", left.Rows[li].Cells)
                : row.Right is { } ri ? string.Join(" | ", right.Rows[ri].Cells) : string.Empty;
            text.AppendLine($"{kind} {leftNumber,6} {rightNumber,6}  {columns,-24}  {content}");
        }

        Emit(text.ToString(), output);
        return result.Different > 0 || result.LeftOnly > 0 || result.RightOnly > 0 ? 1 : 0;
    }

    /// <summary>
    /// 3 方向マージ。祖先を挟んで左右の変更を突き合わせる。
    /// 終了コードは 0 が競合なし、1 が競合あり。
    /// </summary>
    private static int RunThreeWay(
        string basePath, string leftPath, string rightPath, string[] args, string? output)
    {
        var baseText = TextDecoder.Decode(File.ReadAllBytes(basePath));
        var left = TextDecoder.Decode(File.ReadAllBytes(leftPath));
        var right = TextDecoder.Decode(File.ReadAllBytes(rightPath));

        var embedder = args.Contains("--structural") ? null : Embedder.CreateFromDefaultAssets(ValueOf(args, "--model"));
        var result = ThreeWayMerge.Merge(baseText, left, right, embedder);
        var lines = result.ToLines(
            markConflicts: !args.Contains("--take-left"), leftPath, rightPath);

        // 出力先が無ければ標準出力へ。マージ結果は他の道具へ渡すことが多い。
        if (output is not null)
        {
            File.WriteAllBytes(output, TextEncoder.Encode(lines, left));
        }
        else
        {
            Console.Out.Write(string.Join(TextEncoder.Newline(left.LineEnding), lines));
            if (left.EndsWithNewline && lines.Count > 0)
            {
                Console.Out.Write(TextEncoder.Newline(left.LineEnding));
            }
        }

        Console.Error.WriteLine(
            $"まとまり {result.Regions.Count} / 競合 {result.ConflictCount}");
        return result.HasConflicts ? 1 : 0;
    }

    /// <summary>
    /// 差分の反映。左右どちらかの内容をもう片方へ写して書き出す。
    ///
    /// 出力先を指定しない限り<b>何も書かない</b>。既定で上書きすると、確認のつもりで
    /// 実行した一回がファイルを壊す。
    /// </summary>
    private static int RunMerge(string leftPath, string rightPath, string[] args, string? output)
    {
        var direction = ValueOf(args, "--merge");
        var toRight = direction switch
        {
            "to-right" => true,
            "to-left" => false,
            _ => throw new ArgumentException($"--merge の値は to-right か to-left: {direction ?? "(無し)"}"),
        };

        var left = TextDecoder.Decode(File.ReadAllBytes(leftPath));
        var right = TextDecoder.Decode(File.ReadAllBytes(rightPath));

        // 反映は構造だけで決まる。埋め込みは対応付けの質を上げるので、既定では使う。
        var embedder = args.Contains("--structural") ? null : Embedder.CreateFromDefaultAssets(ValueOf(args, "--model"));
        var comparison = DiffComparer.Compare(left, right, embedder, new CompareOptions());
        var blocks = Merge.Blocks(comparison);

        var wanted = ValuesOf(args, "--block")
            .Select(v => int.TryParse(v, out var n) ? n : throw new ArgumentException($"--block の値が数値でない: {v}"))
            .ToHashSet();
        foreach (var number in wanted)
        {
            if (number < 1 || number > blocks.Count)
            {
                throw new ArgumentException($"--block {number} は範囲外（塊は {blocks.Count} 個）");
            }
        }

        var target = toRight ? right : left;
        var sourceLines = toRight ? left.Lines : right.Lines;
        var resultLines = (IReadOnlyList<string>)target.Lines;

        // 後ろから当てる。前から当てると、先に当てた分だけ後続の行番号がずれる。
        var applied = 0;
        for (var i = blocks.Count - 1; i >= 0; i--)
        {
            if (wanted.Count > 0 && !wanted.Contains(i + 1))
            {
                continue;
            }
            var block = blocks[i];
            resultLines = toRight
                ? Merge.Replace(resultLines, block.RightStart, block.RightCount,
                                sourceLines, block.LeftStart, block.LeftCount)
                : Merge.Replace(resultLines, block.LeftStart, block.LeftCount,
                                sourceLines, block.RightStart, block.RightCount);
            applied++;
        }

        var destination = toRight ? rightPath : leftPath;
        var bytes = TextEncoder.Encode(resultLines, target);

        if (args.Contains("--in-place"))
        {
            File.WriteAllBytes(destination, bytes);
            Console.Error.WriteLine(
                $"{destination} を書き換えた（塊 {applied}/{blocks.Count} を反映、"
                + $"{target.Lines.Count} 行 → {resultLines.Count} 行）");
        }
        else if (output is not null)
        {
            File.WriteAllBytes(output, bytes);
            Console.Error.WriteLine(
                $"{output} へ書いた（塊 {applied}/{blocks.Count} を反映、"
                + $"{target.Lines.Count} 行 → {resultLines.Count} 行）");
        }
        else
        {
            Console.Error.WriteLine(
                $"塊 {applied}/{blocks.Count} を反映すると {target.Lines.Count} 行 → "
                + $"{resultLines.Count} 行。書き出すには -o <パス> か --in-place を付ける");
            return 0;
        }

        return 0;
    }

    /// <summary>
    /// フォルダー比較。終了コードは 0 が差異なし、1 が差異あり、2 が異常。
    /// diff に倣ってある。CI から呼んで「差異が出たら落とす」を書けるようにするため。
    /// </summary>
    private static int RunFolderCompare(string leftRoot, string rightRoot, string[] args, string? output)
    {
        FolderComparison result;
        var options = new FolderCompareOptions
        {
            Filter = new NameFilter(ValuesOf(args, "--include"), ValuesOf(args, "--exclude")),
            Mode = args.Contains("--by-timestamp")
                ? FolderComparisonMode.SizeAndTimestamp
                : FolderComparisonMode.Content,
            TimestampToleranceSeconds =
                double.TryParse(ValueOf(args, "--tolerance"), out var tolerance) ? tolerance : 0,
            IgnoreDaylightSavingOffset = args.Contains("--ignore-dst"),
            MinimumSize = long.TryParse(ValueOf(args, "--min-size"), out var min) ? min : 0,
            MaximumSize = long.TryParse(ValueOf(args, "--max-size"), out var max) ? max : 0,
            Recursive = !args.Contains("--no-recurse"),
            IncludeIdentical = !args.Contains("--changes-only"),
            // macOS を経由したファイルは名前が NFD。揃えないと「片方にしか無い」が並ぶ。
            Matching = new NameMatching(
                NormalizeUnicode: args.Contains("--normalize-unicode"),
                IgnoreCase: !args.Contains("--case-sensitive-names")),
        };

        var started = DateTime.UtcNow;
        // 書庫は一時領域へ展開してから走査する。using で確実に片付ける。
        using var leftSource = OpenOrFail(leftRoot, out var leftError, args);
        using var rightSource = OpenOrFail(rightRoot, out var rightError, args);
        if (leftSource is null || rightSource is null)
        {
            Console.Error.WriteLine($"エラー: {leftError ?? rightError}");
            return 2;
        }
        try
        {
            result = FolderComparer.Compare(leftSource.Path, rightSource.Path, options);
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"エラー: {error.Message}");
            return 2;
        }
        var elapsed = DateTime.UtcNow - started;

        var stats = result.Stats;
        var text = new StringBuilder();
        // **合言葉を伏せる。** 結果はそのまま記録や課題に貼られる。
        text.AppendLine($"left  {RemoteLocation.Redact(leftRoot)}");
        text.AppendLine($"right {RemoteLocation.Redact(rightRoot)}");
        text.AppendLine($"stats identical={stats.Identical} different={stats.Different} "
            + $"left_only={stats.LeftOnly} right_only={stats.RightOnly} "
            + $"directories={stats.Directories} errors={stats.Errors} "
            + $"elapsed_ms={(long)elapsed.TotalMilliseconds}");
        text.AppendLine("legend = 一致 / ~ 内容が違う / - 左のみ / + 右のみ / D ディレクトリ / ! 読めない");
        text.AppendLine("---");

        foreach (var entry in result.Entries)
        {
            var kind = entry.Error is not null
                ? '!'
                : entry.Status switch
                {
                    EntryStatus.Identical => entry.IsDirectory ? 'D' : '=',
                    EntryStatus.Different => '~',
                    EntryStatus.LeftOnly => '-',
                    EntryStatus.RightOnly => '+',
                    _ => '?',
                };
            var leftSize = entry.LeftSize?.ToString() ?? string.Empty;
            var rightSize = entry.RightSize?.ToString() ?? string.Empty;
            text.AppendLine($"{kind} {leftSize,10} {rightSize,10}  {entry.RelativePath}"
                + (entry.Error is not null ? $"  ({entry.Error})" : string.Empty));
        }

        if (args.Contains("--csv"))
        {
            Emit(DeepCompare.Engine.Report.FolderCsv(result), output);
            var csvDiffers = stats.Different > 0 || stats.LeftOnly > 0 || stats.RightOnly > 0;
            return stats.Errors > 0 ? 2 : csvDiffers ? 1 : 0;
        }

        // 名前が変わっただけの組を後ろに付ける。名前で対応付ける以上、
        // 「左のみ」「右のみ」に分かれて出るのを補う。
        if (args.Contains("--detect-renames"))
        {
            var renames = RenameDetector.Detect(result, leftSource.Path, rightSource.Path);
            if (renames.Count > 0)
            {
                text.AppendLine("---");
                text.AppendLine($"renames {renames.Count}");
                foreach (var rename in renames)
                {
                    var mark = rename.IdenticalContent ? "移動" : $"移動+変更 {rename.Similarity:F2}";
                    text.AppendLine($"R {rename.LeftPath}  →  {rename.RightPath}  ({mark})");
                }
            }
        }

        Emit(text.ToString(), output);

        // 読めなかったものがあれば「差異なし」とは言えない。
        var differs = stats.Different > 0 || stats.LeftOnly > 0 || stats.RightOnly > 0;
        return stats.Errors > 0 ? 2 : differs ? 1 : 0;
    }

    /// <summary>
    /// フォルダー・書庫・リモートを開く。失敗しても例外を投げず、理由を返す。
    ///
    /// リモートは一時領域へ取ってくる。**絞り込みをそのまま渡す** — 渡さないと、
    /// 除外したはずのものまで回線を使って取ってくることになる。
    /// </summary>
    private static ArchiveSource? OpenOrFail(string path, out string? error, string[]? args = null)
    {
        try
        {
            error = null;
            var options = args is null ? null : new MirrorOptions
            {
                Filter = new NameFilter(ValuesOf(args, "--include"), ValuesOf(args, "--exclude")),
                MaximumFileSize = long.TryParse(ValueOf(args, "--max-size"), out var max)
                    ? max : 32 * 1024 * 1024,
            };

            // **黙って何分も待たせない。** 取っている最中を標準エラーへ出す
            // （標準出力は結果なので、混ぜると読み直せなくなる）。
            var remote = RemoteLocation.IsRemote(path);
            if (remote)
            {
                Console.Error.WriteLine($"{RemoteLocation.Redact(path)} から取ってきます…");
            }

            var source = ArchiveSource.Open(path, options,
                remote ? name => Console.Error.Write($"\r  {name}") : null);

            if (source.Mirror is { } mirror)
            {
                Console.Error.WriteLine();
                Console.Error.WriteLine(mirror.Describe());
            }
            return source;
        }
        catch (Exception failure)
        {
            error = failure.Message;
            return null;
        }
    }

    private static WhitespaceMode ParseWhitespace(string? value) => value switch
    {
        null or "respect" => WhitespaceMode.Respect,
        "trailing" => WhitespaceMode.IgnoreTrailing,
        "ends" => WhitespaceMode.IgnoreLeadingTrailing,
        "collapse" => WhitespaceMode.CollapseRuns,
        "all" => WhitespaceMode.IgnoreAll,
        _ => throw new ArgumentException(
            $"--ws の値が不正: {value}（respect / trailing / ends / collapse / all）"),
    };

    /// <summary>同じフラグが複数回現れることを許す。--ignore-pattern 用。</summary>
    private static List<string> ValuesOf(string[] args, string flag)
    {
        var values = new List<string>();
        for (var i = 0; i + 1 < args.Length; i++)
        {
            if (args[i] == flag)
            {
                values.Add(args[i + 1]);
            }
        }
        return values;
    }

    private static string? ValueOf(string[] args, string flag)
    {
        var index = Array.IndexOf(args, flag);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    /// <param name="structural">
    /// 埋め込みを使わず、文字列一致だけで組む。GUI の段階 1 と同じ経路で、
    /// 「即座に出る答え」の時間を計るために使う。
    /// </param>
    /// <returns>差異があれば 1、無ければ 0。diff に倣う。</returns>
    private static int RunCompare(
        string leftPath, string rightPath, float threshold, string? output,
        bool structural = false, Importance? importance = null,
        string? reportFormat = null, int context = 3, string? modelPath = null,
        ManualAlignment? manual = null)
    {
        var left = TextDecoder.Decode(File.ReadAllBytes(leftPath));
        var right = TextDecoder.Decode(File.ReadAllBytes(rightPath));
        var embedder = structural ? null : Embedder.CreateFromDefaultAssets(modelPath);

        // **モデルが扱えない本文なら知らせる。** 標準エラーへ出す
        // （標準出力は結果なので、混ぜると読み直せなくなる）。
        if (embedder is not null
            && ModelCoverage.Warn(left.Lines, right.Lines, embedder.VocabSize) is { } warning)
        {
            Console.Error.WriteLine(warning);
        }

        var started = DateTime.UtcNow;
        var result = DiffComparer.Compare(
            left, right, embedder,
            new CompareOptions(threshold, Importance: importance, Manual: manual));

        // **手で付けた対応があることは必ず出す。** 自動の結果だと思って
        // 読まれると、なぜそう並ぶのか分からなくなる。
        if (manual is { IsEmpty: false })
        {
            Console.Error.WriteLine(
                $"手で付けた対応: 繋いだ {manual.Linked.Count} 件 / 外した {manual.Unlinked.Count} 件");
        }
        var elapsed = DateTime.UtcNow - started;

        if (reportFormat is not null)
        {
            var report = reportFormat switch
            {
                "unified" => DeepCompare.Engine.Report.UnifiedDiff(
                    result, left, right, leftPath, rightPath, context),
                "html" => DeepCompare.Engine.Report.Html(result, left, right, leftPath, rightPath),
                _ => throw new ArgumentException($"--report の値は unified か html: {reportFormat}"),
            };
            Emit(report, output);
            return result.Rows.Any(row => !row.IsUnchanged) ? 1 : 0;
        }

        var text = new StringBuilder();
        text.AppendLine($"left  {leftPath} encoding={TextDecoder.Label(left.Encoding)} "
            + $"line_ending={TextDecoder.Label(left.LineEnding)} lines={left.Lines.Count}");
        text.AppendLine($"right {rightPath} encoding={TextDecoder.Label(right.Encoding)} "
            + $"line_ending={TextDecoder.Label(right.LineEnding)} lines={right.Lines.Count}");
        text.AppendLine($"stats rows={result.Stats.Rows} identical={result.Stats.IdenticalLines} "
            + $"embedded={result.Stats.EmbeddedLines} skipped_blocks={result.Stats.SkippedBlocks} "
            + $"unimportant={result.Stats.UnimportantRows} "
            + $"elapsed_ms={(long)elapsed.TotalMilliseconds}");
        text.AppendLine($"threshold {threshold:F2}");
        text.AppendLine("legend = 一致 / ≈ 重要でない違いのみ / ~ 変更あり / - 左のみ / + 右のみ");
        text.AppendLine("---");

        // 1 行 1 レコード。環境をまたいで diff で比べられるよう、桁を固定する。
        // 種別は = 一致 / ≈ 重要でない違いのみ / ~ 変更あり / - 左のみ / + 右のみ。
        foreach (var row in result.Rows)
        {
            var kind = (row.Left, row.Right) switch
            {
                (not null, not null) when row.IsUnchanged =>
                    row.HasUnimportantDifferences ? '≈' : '=',
                (not null, not null) => '~',
                (not null, null) => '-',
                (null, not null) => '+',
                _ => '?',
            };
            var leftNumber = row.Left is { } l ? (l + 1).ToString() : string.Empty;
            var rightNumber = row.Right is { } r ? (r + 1).ToString() : string.Empty;
            var score = row.Score is { } s ? s.ToString("F4") : "-";
            var inlineChanges = row.LeftSpans.Concat(row.RightSpans).Count(x => x.Kind == SpanKind.Changed);
            var content = row.Left is { } li ? left.Lines[li]
                : row.Right is { } ri ? right.Lines[ri]
                : string.Empty;
            text.AppendLine($"{kind} {leftNumber,6} {rightNumber,6} {score,6} {inlineChanges,2}  {content}");
        }

        Emit(text.ToString(), output);

        // 重要でないと定義した違いは差異に数えない。無視の指定が効いていれば
        // CI も通る、という筋を通す。
        var differs = result.Rows.Any(row => !row.IsUnchanged);
        return differs ? 1 : 0;
    }

    /// <summary>
    /// 日本語を表示できる書体があるかを調べる。
    ///
    /// 「豆腐になっていないか」は本来目で見るしかないが、書体ファイルの有無を
    /// 先に確かめられれば、遠隔でも当たりを付けられる。
    /// </summary>
    private static void RunFontCheck(string? output)
    {
        string[] candidates = OperatingSystem.IsWindows()
            ? [
                @"C:\Windows\Fonts\msgothic.ttc",
                @"C:\Windows\Fonts\YuGothM.ttc",
                @"C:\Windows\Fonts\meiryo.ttc",
                @"C:\Windows\Fonts\msmincho.ttc",
            ]
            : [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
            ];

        var text = new StringBuilder();
        text.AppendLine($"platform {(OperatingSystem.IsWindows() ? "windows" : "unix")}");
        text.AppendLine("---");
        var found = false;
        foreach (var path in candidates)
        {
            if (File.Exists(path))
            {
                text.AppendLine($"FOUND   {path}  {new FileInfo(path).Length} バイト");
                found = true;
            }
            else
            {
                text.AppendLine($"MISS    {path}");
            }
        }
        text.AppendLine("---");
        text.AppendLine(found
            ? "結果: 日本語を表示できる書体が見つかった"
            : "結果: 候補の書体が無い。表示は環境の既定に依存する");
        Emit(text.ToString(), output);
    }

    /// <summary>
    /// 出力先。`-o` を用意しているのは Windows の都合で、GUI サブシステムで作った exe には
    /// 標準出力が繋がらないため、コンソールから実行しても何も見えない。
    /// </summary>
    private static void Emit(string text, string? output)
    {
        if (output is not null)
        {
            File.WriteAllText(output, text, new UTF8Encoding(false));
        }
        else
        {
            Console.Out.Write(text);
            Console.Out.Flush();
        }
    }

    private static int Report(Action action) => Report(() => { action(); return 0; });

    /// <summary>
    /// 例外を終了コードに変える。異常は 2 とし、1 は「差異あり」のために空けてある。
    /// 両方を 1 にすると、CI から見て「差分が出た」と「壊れた」を区別できない。
    /// </summary>
    private static int Report(Func<int> action)
    {
        try
        {
            return action();
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"エラー: {error.Message}");
            return 2;
        }
    }
}
