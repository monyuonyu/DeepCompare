using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace DeepCompare.Assist;

/// <summary>提案 1 つ。</summary>
public sealed record AssistSuggestion(
    AssistAction Action,
    string Reason,
    bool Recommended)
{
    /// <summary>画面に出す一行。</summary>
    public string Display => AssistActions.Describe(Action);
}

/// <summary>状態の説明と、次にできることの一覧。</summary>
public sealed record AssistAdvice(
    string Explanation,
    IReadOnlyList<AssistSuggestion> Suggestions)
{
    public static AssistAdvice Empty { get; } = new(string.Empty, []);
}

/// <summary>
/// git の状態を人の言葉にし、次にできることを挙げる。
///
/// **副作用のあることをしない。** ここが返すのは文字列と列挙型だけで、
/// git を呼ぶ経路は無い。実行するかどうかは、これを受け取った側が決める。
///
/// 対象は「読むだけ」の 4 つ（ROADMAP 6.2 の上 4 つ）:
/// 状態の説明・次の一手・コミットメッセージの草案・衝突が何を意味するか。
/// **解決案の生成だけは別扱い**（設定で明示的に許すまで出さない）。
/// </summary>
public sealed class GitAssistant(ChatClient client)
{
    /// <summary>
    /// 前置き。**固定する。**
    ///
    /// llama.cpp / Ollama は前置きが一致すれば KV キャッシュを再利用するので、
    /// 続けて何度も聞く場面で前処理が丸ごと省ける。外部 API のキャッシュも
    /// 同じ「変わらない部分を先に」で効くので、設計は共通にできる。
    /// </summary>
    public const string SystemPrompt =
        """
        あなたはバージョン管理に不慣れな人を助ける案内役です。

        守ること:
        - 日本語で、短く、具体的に書く
        - 専門用語を使うときは、その場で言い換えを添える
        - 分からないことは分からないと言う。推測で断定しない
        - コマンドを書かない。操作は与えられた選択肢の中から選ぶ
        - 利用者のコードの中身を評価したり、書き換えを提案したりしない
        """;

    /// <summary>
    /// 返させる形。**列挙型の外を書けないようにする。**
    ///
    /// これが安全性の実体。自由な文字列を操作として受け取らないので、
    /// リポジトリの中身に何が書いてあっても命令にはならない。
    /// </summary>
    internal static JsonNode AdviceSchema()
    {
        var actions = new JsonArray();
        foreach (var name in AssistActions.Names)
        {
            // **JsonValue.Create を通す。** JsonArray.Add<T> は非プリミティブ型を
            // 実行時に組み立てる経路があり、NativeAOT で弾かれる。
            actions.Add((JsonNode?)JsonValue.Create(name));
        }

        return new JsonObject
        {
            ["type"] = "object",
            ["additionalProperties"] = false,
            ["required"] = StringArray("説明", "選択肢"),
            ["properties"] = new JsonObject
            {
                ["説明"] = new JsonObject { ["type"] = "string" },
                ["選択肢"] = new JsonObject
                {
                    ["type"] = "array",
                    ["items"] = new JsonObject
                    {
                        ["type"] = "object",
                        ["additionalProperties"] = false,
                        ["required"] = StringArray("操作", "理由", "推奨"),
                        ["properties"] = new JsonObject
                        {
                            ["操作"] = new JsonObject { ["enum"] = actions },
                            ["理由"] = new JsonObject { ["type"] = "string" },
                            ["推奨"] = new JsonObject { ["type"] = "boolean" },
                        },
                    },
                },
            },
        };
    }

    /// <summary>文字列だけの配列。**AOT で通る書き方に寄せる。**</summary>
    private static JsonArray StringArray(params string[] values)
    {
        var array = new JsonArray();
        foreach (var value in values)
        {
            array.Add((JsonNode?)JsonValue.Create(value));
        }
        return array;
    }

    /// <summary>
    /// いまの状態を説明し、次にできることを挙げる。
    /// </summary>
    /// <param name="status">
    /// git の状態を整えた文字列。**アプリ側が作る。**
    /// ここから git を呼ばないのは、この層に副作用を持たせないため。
    /// </param>
    public async Task<AssistAdvice> ExplainStatusAsync(
        string status, CancellationToken cancellationToken = default,
        int? maxTokens = null)
    {
        var answer = await client.CompleteAsync(
            [
                ChatMessage.System(SystemPrompt),
                ChatMessage.User(
                    $"""
                    いまのリポジトリの状態です。

                    {Clip(status)}

                    これが何を意味するかを 2〜3 文で説明し、
                    次にできることを挙げてください。
                    """),
            ],
            AdviceSchema(),
            cancellationToken,
            maxTokens);

        return ParseAdvice(answer);
    }

    /// <summary>
    /// 衝突が何を意味するかを説明する。**解決案は出さない。**
    ///
    /// 「何と何がぶつかったか」までなら 8B 級でも実用になる。
    /// 直し方の生成は性質が違う（<see cref="ProposeResolutionAsync"/>）。
    /// </summary>
    public async Task<string> ExplainConflictAsync(
        string path, string ours, string theirs, string? baseText = null,
        CancellationToken cancellationToken = default)
    {
        var prompt = new StringBuilder();
        prompt.AppendLine($"{path} で変更がぶつかりました。");
        if (baseText is { Length: > 0 })
        {
            prompt.AppendLine().AppendLine("＜元の内容＞").AppendLine(Clip(baseText));
        }
        prompt.AppendLine().AppendLine("＜こちらの変更＞").AppendLine(Clip(ours));
        prompt.AppendLine().AppendLine("＜相手の変更＞").AppendLine(Clip(theirs));
        prompt.AppendLine()
              .AppendLine("**何と何がぶつかっているか**を 2〜3 文で説明してください。")
              .AppendLine("どちらを採るべきかは書かないでください。");

        return await client.CompleteAsync(
            [ChatMessage.System(SystemPrompt), ChatMessage.User(prompt.ToString())],
            schema: null,
            cancellationToken);
    }

    /// <summary>
    /// コミットメッセージの草案。少しずつ返す。
    ///
    /// **草案であることを崩さない。** 出来たものをそのまま使うのではなく、
    /// 入力欄に入れて人が直す前提。
    /// </summary>
    public IAsyncEnumerable<string> DraftCommitMessageAsync(
        string diff, CancellationToken cancellationToken = default)
        => client.StreamAsync(
            [
                ChatMessage.System(SystemPrompt),
                ChatMessage.User(
                    $"""
                    次の差分に対するコミットメッセージの草案を書いてください。

                    - 1 行目は 50 文字以内の要約
                    - 空行を挟み、なぜそうしたかを箇条書きで
                    - 差分から読み取れないことは書かない

                    {Clip(diff, MaxDiffChars)}
                    """),
            ],
            schema: null,
            cancellationToken);

    /// <summary>
    /// 衝突の解決案を出す。**既定では呼べない。**
    ///
    /// 要約や分類と違い、これは意味を取り違えると害になる生成。
    /// 8B 級はもっともらしく間違え、ビルドが通るぶんだけ発見が遅れる。
    /// 設定で明示的に許したときだけ通る。
    ///
    /// 返すのは**本文だけ**。呼んだ側はこれを 3 方向マージの出力側へ
    /// 差分として流し込む。平文で見せない — 差分として目に入れば、
    /// 承認の粒度が hunk 単位になり「勝手に直された」が起きない。
    /// </summary>
    public async Task<string> ProposeResolutionAsync(
        AssistSettings settings,
        string path, string ours, string theirs, string? baseText = null,
        CancellationToken cancellationToken = default)
    {
        if (!settings.AllowResolutionProposals)
        {
            throw new InvalidOperationException(
                "解決案は既定で出しません。弱いモデルはもっともらしく間違え、"
                + "ビルドが通るぶんだけ発見が遅れます。設定で明示的に許してください。");
        }

        var prompt = new StringBuilder();
        prompt.AppendLine($"{path} の衝突を解いた結果の**本文だけ**を返してください。");
        prompt.AppendLine("説明や印（<<<<<<< など）を混ぜないでください。");
        if (baseText is { Length: > 0 })
        {
            prompt.AppendLine().AppendLine("＜元の内容＞").AppendLine(Clip(baseText));
        }
        prompt.AppendLine().AppendLine("＜こちらの変更＞").AppendLine(Clip(ours));
        prompt.AppendLine().AppendLine("＜相手の変更＞").AppendLine(Clip(theirs));

        return await client.CompleteAsync(
            [ChatMessage.System(SystemPrompt), ChatMessage.User(prompt.ToString())],
            schema: null,
            cancellationToken);
    }

    /// <summary>
    /// 返事が壊れていたときの文言。
    ///
    /// **モデルのせいだと分かるように書く。** 「失敗しました」だけだと、
    /// 設定を疑って接続先やモデル名を何度も直すことになる。
    /// </summary>
    public const string BrokenAnswerMessage =
        "モデルの返事が途中で切れました。小さいモデルは同じ文を繰り返して"
        + "終われなくなることがあります。もう一度試すか、大きいモデルに替えてください。";

    /// <summary>1 つの塊に入れる上限。</summary>
    internal const int MaxChars = 8_000;

    /// <summary>差分の上限。**説明より長く取る** — 短いと要約が的外れになる。</summary>
    internal const int MaxDiffChars = 24_000;

    /// <summary>
    /// 長すぎるものを切る。**切ったことを本文に書く。**
    /// 黙って切ると、モデルは「これで全部だ」と読んで的外れな要約を書く。
    /// </summary>
    internal static string Clip(string text, int limit = MaxChars)
        => text.Length <= limit
            ? text
            : text[..limit] + $"{Environment.NewLine}…（ここから先は長さの都合で省きました）";

    /// <summary>
    /// 返事を読む。**読めなくても落とさない。**
    ///
    /// 弱いモデルは形を指定しても崩すことがある。そこで例外を投げると、
    /// 支援が使えないどころか比較の操作まで止まる。
    /// </summary>
    internal static AssistAdvice ParseAdvice(string answer)
    {
        var extracted = ExtractJson(answer);

        JsonNode? json;
        try
        {
            json = JsonNode.Parse(extracted);
        }
        catch (JsonException)
        {
            // **壊れた JSON をそのまま見せない。** 途中で切れた出力を説明として
            // 出すと、生の `{"説明": ...` が画面に並ぶ（実測で起きた）。
            // 形になりかけている物は、読めなかったと言う方がまだ分かる。
            if (extracted.StartsWith('{'))
            {
                return new AssistAdvice(BrokenAnswerMessage, []);
            }

            // **平文で返ってきたら、説明として扱う。** 捨てるより惜しい。
            return new AssistAdvice(answer.Trim(), []);
        }

        var explanation = json?["説明"]?.GetValue<string>() ?? string.Empty;
        var suggestions = new List<AssistSuggestion>();
        var seen = new HashSet<AssistAction>();

        if (json?["選択肢"] is JsonArray array)
        {
            foreach (var item in array)
            {
                var action = AssistActions.Parse(item?["操作"]?.GetValue<string>());

                // **知らない操作は落とす。** None に倒した提案を並べても、
                // 「何もしない」が理由付きで何個も出るだけで読みにくい。
                if (action == AssistAction.None)
                {
                    continue;
                }

                // **同じ操作を二度出さない。** 小さいモデルは同じ提案を
                // 言い回しだけ変えて並べる（Qwen2.5 1.5B で pull が 2 回出た）。
                // 選ぶ側からは、違う選択肢が 2 つあるように見えてしまう。
                if (!seen.Add(action))
                {
                    continue;
                }

                suggestions.Add(new AssistSuggestion(
                    action,
                    item?["理由"]?.GetValue<string>() ?? string.Empty,
                    item?["推奨"]?.GetValue<bool>() ?? false));
            }
        }

        return new AssistAdvice(explanation.Trim(), suggestions);
    }

    /// <summary>
    /// 前後の飾りを剥がす。
    ///
    /// **``` で囲って返すモデルがある。** 形を指定しても起きるので、
    /// ここで吸収する。相手の行儀に期待しない。
    /// </summary>
    internal static string ExtractJson(string text)
    {
        var trimmed = text.Trim();

        if (trimmed.StartsWith("```", StringComparison.Ordinal))
        {
            var firstBreak = trimmed.IndexOf('\n');
            if (firstBreak > 0)
            {
                trimmed = trimmed[(firstBreak + 1)..];
            }
            var fence = trimmed.LastIndexOf("```", StringComparison.Ordinal);
            if (fence >= 0)
            {
                trimmed = trimmed[..fence];
            }
            trimmed = trimmed.Trim();
        }

        // 前後に一言添えてくる場合。**最初の { から最後の } まで**を採る。
        var start = trimmed.IndexOf('{');
        var end = trimmed.LastIndexOf('}');
        return start >= 0 && end > start ? trimmed[start..(end + 1)] : trimmed;
    }
}
