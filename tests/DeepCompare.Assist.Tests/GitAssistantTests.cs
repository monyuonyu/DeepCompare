using System.Text.Json.Nodes;
using Xunit;

namespace DeepCompare.Assist.Tests;

public class AssistSettingsTests
{
    [Fact]
    public void 接続先が空なら使わない()
    {
        Assert.False(new AssistSettings().IsConfigured);
        Assert.False(new AssistSettings { Endpoint = "http://x/v1" }.IsConfigured);
        Assert.False(new AssistSettings { Model = "llama3" }.IsConfigured);
        Assert.True(new AssistSettings
        {
            Endpoint = "http://x/v1",
            Model = "llama3",
        }.IsConfigured);
    }

    [Fact]
    public void 末尾のスラッシュがあってもなくても同じ場所を指す()
    {
        var a = new AssistSettings { Endpoint = "http://x/v1" };
        var b = new AssistSettings { Endpoint = "http://x/v1/" };
        Assert.Equal("http://x/v1/chat/completions", a.UrlFor("chat/completions"));
        Assert.Equal(a.UrlFor("chat/completions"), b.UrlFor("/chat/completions"));
    }

    [Fact]
    public void 鍵を表に出さない()
    {
        var settings = new AssistSettings
        {
            Endpoint = "https://api.example.com/v1",
            Model = "m",
            ApiKey = "sk-とても秘密",
        };
        Assert.DoesNotContain("とても秘密", settings.Redacted(), StringComparison.Ordinal);
        Assert.Contains("鍵あり", settings.Redacted(), StringComparison.Ordinal);
    }

    [Fact]
    public void 鍵が無いのに鍵ありと書かない()
    {
        // 「鍵あり」と出ると、入れた覚えのない鍵があるのかと疑わせる。
        var settings = new AssistSettings { Endpoint = "http://localhost:11434/v1", Model = "m" };
        Assert.DoesNotContain("鍵", settings.Redacted(), StringComparison.Ordinal);
    }

    [Fact]
    public void 解決案は既定で許さない()
    {
        // **既定を安全側に倒す。** 弱いモデルはもっともらしく間違える。
        Assert.False(new AssistSettings().AllowResolutionProposals);
    }
}

public class AssistActionTests
{
    [Fact]
    public void 破壊的な操作を列挙型に入れない()
    {
        // **一覧に無ければ提案されない。** force push や reset --hard は
        // 取り返しがつかないので、選択肢として存在させない。
        var names = string.Join(",", AssistActions.Names).ToLowerInvariant();
        foreach (var forbidden in new[] { "force", "reset", "rebase", "clean", "checkout" })
        {
            Assert.DoesNotContain(forbidden, names, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void 知らない操作は何もしないに倒す()
    {
        // 弱いモデルは列挙型を指示しても勝手な文字列を返す。
        // **当てずっぽうで近い操作に寄せない。**
        Assert.Equal(AssistAction.None, AssistActions.Parse("rm -rf /"));
        Assert.Equal(AssistAction.None, AssistActions.Parse("force_push"));
        Assert.Equal(AssistAction.None, AssistActions.Parse(null));
        Assert.Equal(AssistAction.None, AssistActions.Parse(""));
    }

    [Fact]
    public void 大文字小文字は問わない()
    {
        Assert.Equal(AssistAction.Pull, AssistActions.Parse("pull"));
        Assert.Equal(AssistAction.Pull, AssistActions.Parse("PULL"));
        Assert.Equal(AssistAction.AbortMerge, AssistActions.Parse("abortmerge"));
    }

    [Fact]
    public void 副作用の無い操作だけを安全とみなす()
    {
        Assert.False(AssistActions.IsDestructive(AssistAction.None));
        Assert.False(AssistActions.IsDestructive(AssistAction.ViewDiff));
        Assert.True(AssistActions.IsDestructive(AssistAction.Commit));
        Assert.True(AssistActions.IsDestructive(AssistAction.Pull));
        Assert.True(AssistActions.IsDestructive(AssistAction.StashPop));
    }
}

public class AdviceParsingTests
{
    [Fact]
    public void 素直な返事を読む()
    {
        var advice = GitAssistant.ParseAdvice(
            """
            {"説明":"変更が 3 つ記録されていません。",
             "選択肢":[{"操作":"Commit","理由":"まず残す","推奨":true},
                       {"操作":"Stash","理由":"後で戻す","推奨":false}]}
            """);

        Assert.Equal("変更が 3 つ記録されていません。", advice.Explanation);
        Assert.Equal(2, advice.Suggestions.Count);
        Assert.Equal(AssistAction.Commit, advice.Suggestions[0].Action);
        Assert.True(advice.Suggestions[0].Recommended);
        Assert.False(advice.Suggestions[1].Recommended);
    }

    [Fact]
    public void 囲みを付けて返されても読む()
    {
        // **形を指定しても ``` で囲うモデルがある。** 相手の行儀に期待しない。
        var advice = GitAssistant.ParseAdvice(
            "```json\n{\"説明\":\"あ\",\"選択肢\":[]}\n```");
        Assert.Equal("あ", advice.Explanation);
    }

    [Fact]
    public void 前後に一言添えられても読む()
    {
        var advice = GitAssistant.ParseAdvice(
            "了解しました。\n{\"説明\":\"い\",\"選択肢\":[]}\nご確認ください。");
        Assert.Equal("い", advice.Explanation);
    }

    [Fact]
    public void 平文で返されたら説明として扱う()
    {
        // **捨てない。** 形が崩れても、書いてあることには意味がある。
        var advice = GitAssistant.ParseAdvice("まだ記録していない変更があります。");
        Assert.Equal("まだ記録していない変更があります。", advice.Explanation);
        Assert.Empty(advice.Suggestions);
    }

    [Fact]
    public void 知らない操作の提案は落とす()
    {
        // None に倒した提案を並べても「何もしない」が理由付きで何個も出るだけ。
        var advice = GitAssistant.ParseAdvice(
            """
            {"説明":"x","選択肢":[{"操作":"rm -rf /","理由":"消す","推奨":true},
                                  {"操作":"Commit","理由":"残す","推奨":true}]}
            """);
        Assert.Single(advice.Suggestions);
        Assert.Equal(AssistAction.Commit, advice.Suggestions[0].Action);
    }

    [Fact]
    public void 同じ操作を二度出さない()
    {
        // 小さいモデルは同じ提案を言い回しだけ変えて並べる
        // （Qwen2.5 1.5B の実測で pull が 2 回出た）。
        var advice = GitAssistant.ParseAdvice(
            "{\"説明\":\"x\",\"選択肢\":["
            + "{\"操作\":\"Pull\",\"理由\":\"取り込む\",\"推奨\":true},"
            + "{\"操作\":\"Pull\",\"理由\":\"最新にする\",\"推奨\":true},"
            + "{\"操作\":\"Commit\",\"理由\":\"残す\",\"推奨\":false}]}");

        Assert.Equal(2, advice.Suggestions.Count);
        Assert.Equal(AssistAction.Pull, advice.Suggestions[0].Action);
        // **先に出た方を残す。**
        Assert.Equal("取り込む", advice.Suggestions[0].Reason);
        Assert.Equal(AssistAction.Commit, advice.Suggestions[1].Action);
    }

    [Fact]
    public void 途中で切れた返事は壊れたと言う()
    {
        // **生の JSON を説明として見せない。** 実測で画面に
        // `{\"説明\": ...` がそのまま並んだ。
        var advice = GitAssistant.ParseAdvice(
            "{\"説明\":\"現在の状態は\",\"選択肢\":[{\"操作\":\"Commit\",\"理由\":\"ずっと同じ文がずっと同じ文が");

        Assert.Equal(GitAssistant.BrokenAnswerMessage, advice.Explanation);
        Assert.DoesNotContain("説明", advice.Explanation, StringComparison.Ordinal);
        Assert.Empty(advice.Suggestions);
    }

    [Fact]
    public void 選択肢が欠けていても落ちない()
    {
        var advice = GitAssistant.ParseAdvice("""{"説明":"だけ"}""");
        Assert.Equal("だけ", advice.Explanation);
        Assert.Empty(advice.Suggestions);
    }

    [Fact]
    public void 長すぎる入力は切り切ったことを書く()
    {
        var clipped = GitAssistant.Clip(new string('あ', GitAssistant.MaxChars + 500));
        Assert.True(clipped.Length < GitAssistant.MaxChars + 200);

        // **黙って切らない。** モデルが「これで全部だ」と読むと要約が的外れになる。
        Assert.Contains("省きました", clipped, StringComparison.Ordinal);
    }

    [Fact]
    public void 短い入力はそのまま通す()
    {
        Assert.Equal("短い", GitAssistant.Clip("短い"));
    }

    [Fact]
    public void 返させる形に列挙型の一覧が入っている()
    {
        var schema = GitAssistant.AdviceSchema();
        var actions = schema["properties"]?["選択肢"]?["items"]?["properties"]?["操作"]?["enum"]
            as JsonArray;

        Assert.NotNull(actions);
        var names = actions!.Select(n => n!.GetValue<string>()).ToList();
        Assert.Contains("Commit", names);
        Assert.Contains("Pull", names);
        // **危ないものは一覧にすら無い。**
        Assert.DoesNotContain(names, n => n.Contains("Reset", StringComparison.Ordinal));
    }
}
