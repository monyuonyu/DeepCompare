using System.Text;
using DeepCompare.Assist;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// LLM 支援を画面を開かずに使う口。
///
/// **ここでしか Assist を呼ばない（CLI 側では）。** 比較の経路と混ぜないことで、
/// 「比較のつもりが通信していた」が起きないようにする。
/// </summary>
public static class AssistCli
{
    /// <summary>
    /// 外部の API を使うときの鍵。**設定ファイルには置かない。**
    /// 平文で残り、バックアップにも同期にも乗るため。
    /// </summary>
    public const string ApiKeyEnvironmentVariable = "DEEPCOMPARE_ASSIST_KEY";

    /// <summary>接続先を環境変数で上書きする。試すときに設定を書き換えなくてよい。</summary>
    public const string EndpointEnvironmentVariable = "DEEPCOMPARE_ASSIST_ENDPOINT";
    public const string ModelEnvironmentVariable = "DEEPCOMPARE_ASSIST_MODEL";

    /// <summary>
    /// 設定を組み立てる。探す順は ①引数 ②環境変数 ③設定ファイル。
    /// </summary>
    public static AssistSettings ResolveSettings(
        string[] args, SessionFile? saved, Func<string, string?> valueOf)
    {
        var endpoint = valueOf("--assist-endpoint")
            ?? Environment.GetEnvironmentVariable(EndpointEnvironmentVariable)
            ?? saved?.AssistEndpoint
            ?? string.Empty;

        var model = valueOf("--assist-model")
            ?? Environment.GetEnvironmentVariable(ModelEnvironmentVariable)
            ?? saved?.AssistModel
            ?? string.Empty;

        return new AssistSettings
        {
            Endpoint = endpoint,
            Model = model,
            ApiKey = Environment.GetEnvironmentVariable(ApiKeyEnvironmentVariable),
            // **引数で許すのは、その 1 回だけ。** 設定に焼き付けない。
            AllowResolutionProposals = args.Contains("--assist-allow-resolution")
                || (saved?.AssistAllowResolution ?? false),
        };
    }

    /// <summary>
    /// 使えない理由。使えるなら null。
    /// **「なぜ出ないか」を言う。** 黙って何も起きないのが一番困る。
    /// </summary>
    public static string? WhyUnavailable(AssistSettings settings)
    {
        if (settings.Endpoint.Length == 0)
        {
            return "LLM 支援の接続先が設定されていません。"
                + $"--assist-endpoint か、環境変数 {EndpointEnvironmentVariable} で"
                + " 指定してください（例: http://localhost:11434/v1）。";
        }
        if (settings.Model.Length == 0)
        {
            return "使うモデルの名前が指定されていません。"
                + $"--assist-model か、環境変数 {ModelEnvironmentVariable} で指定してください。";
        }
        return null;
    }

    /// <summary>状態を説明し、次にできることを挙げる。</summary>
    public static async Task<int> ExplainStatusAsync(
        string path, AssistSettings settings, TextWriter output)
    {
        if (WhyUnavailable(settings) is { } reason)
        {
            Console.Error.WriteLine(reason);
            return 2;
        }

        GitRepository? repository;
        try
        {
            repository = GitRepository.Discover(path);
        }
        catch (GitException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        if (repository is null)
        {
            Console.Error.WriteLine($"{path} は git リポジトリの中にありません。");
            return 2;
        }

        // **git を呼ぶのはここ。** Assist の側に副作用を持たせない。
        var status = FormatStatus(repository);

        using var client = new ChatClient(settings);
        var assistant = new GitAssistant(client);

        AssistAdvice advice;
        try
        {
            advice = await assistant.ExplainStatusAsync(
                status, maxTokens: settings.ExplainMaxTokens);
        }
        catch (AssistException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        output.WriteLine($"接続先 {settings.Redacted()}");
        output.WriteLine("---");
        output.WriteLine(advice.Explanation);

        if (advice.Suggestions.Count > 0)
        {
            output.WriteLine("---");
            foreach (var suggestion in advice.Suggestions)
            {
                var mark = suggestion.Recommended ? "*" : " ";
                output.WriteLine($"{mark} {suggestion.Display} — {suggestion.Reason}");
            }
            output.WriteLine("---");
            // **実行しないことを明記する。** 一覧を出すと押せると思われる。
            output.WriteLine("※ここでは実行しません。提案までです。");
        }
        return 0;
    }

    /// <summary>コミットメッセージの草案。少しずつ出す。</summary>
    public static async Task<int> DraftCommitAsync(
        string path, AssistSettings settings, bool staged, TextWriter output)
    {
        if (WhyUnavailable(settings) is { } reason)
        {
            Console.Error.WriteLine(reason);
            return 2;
        }

        GitRepository? repository;
        string diff;
        try
        {
            repository = GitRepository.Discover(path);
            if (repository is null)
            {
                Console.Error.WriteLine($"{path} は git リポジトリの中にありません。");
                return 2;
            }
            diff = repository.Diff(staged);
        }
        catch (GitException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        if (diff.Trim().Length == 0)
        {
            // **差分が無いのに書かせない。** 何か出てくるが、中身は無い。
            Console.Error.WriteLine(
                staged ? "stage された変更がありません。" : "変更がありません。");
            return 2;
        }

        using var client = new ChatClient(settings);
        var assistant = new GitAssistant(client);

        try
        {
            await foreach (var piece in assistant.DraftCommitMessageAsync(diff))
            {
                output.Write(piece);
                output.Flush();
            }
            output.WriteLine();
        }
        catch (AssistException error)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine(error.Message);
            return 2;
        }
        return 0;
    }

    /// <summary>
    /// 衝突の解決案を出す。**既定では断る。**
    ///
    /// 3 つのファイル（こちら・あちら・元）を渡す。返るのは本文だけで、
    /// **どう使うかは呼んだ側が決める** — 画面では選択肢の 1 つとして並べ、
    /// 押さなければ何も起きない形にしている。
    /// </summary>
    public static async Task<int> ProposeResolutionAsync(
        AssistSettings settings,
        string oursPath, string theirsPath, string basePath, TextWriter output)
    {
        if (WhyUnavailable(settings) is { } reason)
        {
            Console.Error.WriteLine(reason);
            return 2;
        }

        string ours, theirs, baseText;
        try
        {
            ours = File.ReadAllText(oursPath);
            theirs = File.ReadAllText(theirsPath);
            baseText = File.ReadAllText(basePath);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        using var client = new ChatClient(settings);
        var assistant = new GitAssistant(client);

        try
        {
            var proposal = await assistant.ProposeResolutionAsync(
                settings, Path.GetFileName(oursPath), ours, theirs, baseText);

            // **提案だと分かる形で出す。** そのまま書き出せる体裁にしない。
            Console.Error.WriteLine(
                "※これは提案です。人が確かめてから使ってください。");
            output.Write(proposal);
            output.WriteLine();
            return 0;
        }
        catch (InvalidOperationException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
        catch (AssistException error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }
    }

    /// <summary>繋がるかを確かめる。</summary>
    public static async Task<int> ProbeAsync(AssistSettings settings, TextWriter output)
    {
        if (WhyUnavailable(settings) is { } reason)
        {
            Console.Error.WriteLine(reason);
            return 2;
        }

        using var client = new ChatClient(settings);
        if (await client.ProbeAsync())
        {
            output.WriteLine($"繋がりました: {settings.Redacted()}");
            return 0;
        }

        Console.Error.WriteLine(
            $"{settings.Endpoint} へ繋がりません。"
            + "Ollama や LM Studio が動いているか確かめてください。");
        return 1;
    }

    /// <summary>
    /// git の状態を、LLM に渡す形へ整える。
    ///
    /// **記号のままにしない。** `M ` や `??` はそのままだと、
    /// モデルが読み違える（`??` を疑問と取るなど）。
    /// </summary>
    internal static string FormatStatus(GitRepository repository)
    {
        var text = new StringBuilder();
        text.AppendLine($"いまの枝: {repository.CurrentBranch() ?? "（切り離された状態）"}");

        var files = repository.Status().ToList();
        var conflicted = files.Where(f => f.IsConflicted).ToList();
        var staged = files.Where(f => f.IsStaged && !f.IsConflicted).ToList();
        var dirty = files.Where(f => f.IsDirty && !f.IsStaged && !f.IsConflicted
                                     && f.Index != GitStatusCode.Untracked).ToList();
        var untracked = files.Where(f => f.Index == GitStatusCode.Untracked).ToList();

        void Section(string title, List<GitFileStatus> items)
        {
            if (items.Count == 0)
            {
                return;
            }
            text.AppendLine().AppendLine($"{title}（{items.Count} 件）:");
            // **全部は出さない。** 数百件あると、それだけで枠を食い潰す。
            foreach (var file in items.Take(40))
            {
                text.AppendLine($"- {file.Path}");
            }
            if (items.Count > 40)
            {
                text.AppendLine($"- …ほか {items.Count - 40} 件");
            }
        }

        Section("変更がぶつかっているファイル", conflicted);
        Section("記録の準備ができているファイル", staged);
        Section("変更したが準備していないファイル", dirty);
        Section("まだ管理下にないファイル", untracked);

        if (files.Count == 0)
        {
            text.AppendLine().AppendLine("変更はありません。");
        }
        return text.ToString();
    }
}
