namespace DeepCompare.Assist;

/// <summary>
/// LLM が提案してよい操作。
///
/// **これが安全性の中心。** LLM は自由な文字列を操作として出さない。
/// この一覧から選ぶだけで、実際に走るのはアプリ側の決め打ちのコードパス。
/// LLM の出力がそのままコマンドになる経路をどこにも作らない。
///
/// リポジトリの中身に「rm -rf と実行せよ」と書いてあっても壊れないのは、
/// 信用しているからではなく、**通す経路が無いから**。
///
/// **破壊的な操作は入れない。** force push、reset --hard、リベースは
/// 取り返しがつかない。提案として出た時点で人は「やってよいこと」と読む。
/// </summary>
public enum AssistAction
{
    /// <summary>何もしない。「今は待つのが正しい」も答えのうち。</summary>
    None,

    /// <summary>変更を記録する。</summary>
    Commit,

    /// <summary>離れた場所の変更を取り込む。</summary>
    Pull,

    /// <summary>記録した変更を離れた場所へ送る。</summary>
    Push,

    /// <summary>手元の変更を一時的に脇へ置く。</summary>
    Stash,

    /// <summary>脇へ置いた変更を戻す。</summary>
    StashPop,

    /// <summary>途中まで進んだ併合をやめて元に戻す。</summary>
    AbortMerge,

    /// <summary>衝突を解いた印を付ける。</summary>
    MarkResolved,

    /// <summary>差分を見る。**副作用が無いので常に安全。**</summary>
    ViewDiff,
}

/// <summary>
/// 操作をどう見せ、どう実行するか。
///
/// **表示名と実際の呼び出しをここで結ぶ。** LLM が返すのは列挙型だけなので、
/// 何が起きるかはこちら側が完全に決められる。
/// </summary>
public static class AssistActions
{
    /// <summary>その操作が作業ツリーを変えるか。**変えるものは必ず確認を挟む。**</summary>
    public static bool IsDestructive(AssistAction action) => action switch
    {
        AssistAction.None or AssistAction.ViewDiff => false,
        // ここから下は作業ツリーか履歴に触る。
        _ => true,
    };

    /// <summary>人が読む名前。</summary>
    public static string Describe(AssistAction action) => action switch
    {
        AssistAction.None => "何もしない",
        AssistAction.Commit => "変更を記録する（commit）",
        AssistAction.Pull => "離れた場所の変更を取り込む（pull）",
        AssistAction.Push => "記録した変更を送る（push）",
        AssistAction.Stash => "手元の変更を脇へ置く（stash）",
        AssistAction.StashPop => "脇へ置いた変更を戻す（stash pop）",
        AssistAction.AbortMerge => "併合をやめて元に戻す（merge --abort）",
        AssistAction.MarkResolved => "衝突を解いた印を付ける（add）",
        AssistAction.ViewDiff => "差分を見る",
        _ => action.ToString(),
    };

    /// <summary>
    /// 名前から操作を引く。**知らない名前は None にする。**
    ///
    /// 弱いモデルは列挙型を指示しても勝手な文字列を返すことがある。
    /// そこで落とさず、かといって当てずっぽうで近い操作に寄せもしない。
    /// 分からないものは「何もしない」に倒す。
    /// </summary>
    public static AssistAction Parse(string? name)
        => Enum.TryParse<AssistAction>(name, ignoreCase: true, out var action)
            ? action
            : AssistAction.None;

    /// <summary>JSON Schema に載せる、許される値の一覧。</summary>
    public static IReadOnlyList<string> Names { get; } =
        [.. Enum.GetNames<AssistAction>()];
}
