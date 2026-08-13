namespace DeepCompare.Assist;

/// <summary>
/// LLM 支援の設定。
///
/// **既定で無効。** <see cref="Endpoint"/> が空なら機能そのものを出さない。
/// 起動時に接続を試みることもしない（居ない相手を待って固まるのが一番困る）。
/// </summary>
public sealed record AssistSettings
{
    /// <summary>
    /// OpenAI 互換のエンドポイント。Ollama なら
    /// <c>http://localhost:11434/v1</c>、LM Studio なら
    /// <c>http://localhost:1234/v1</c>。
    ///
    /// **既定は空。** 入れた人だけが使う。
    /// </summary>
    public string Endpoint { get; init; } = string.Empty;

    /// <summary>使うモデルの名前。</summary>
    public string Model { get; init; } = string.Empty;

    /// <summary>
    /// 鍵。**ローカルのサーバーには要らない。**
    /// 外部の API を使うときだけ入る。
    /// </summary>
    public string? ApiKey { get; init; }

    /// <summary>
    /// 待つ上限。
    ///
    /// **CPU で動かすローカルモデルに合わせる。** この機体で実測すると
    /// 1.5B が 15.9 トークン/秒、**7B は 2.2 トークン/秒**。7B に 400 トークン
    /// 書かせると 3 分かかる。60 秒だと、大きめのモデルでは必ず時限切れになり
    /// 「繋がりませんでした」と出る（実際に踏んだ）。
    ///
    /// 長くしても比較の邪魔にはならない。**支援は別の入口にあり、
    /// 待っている間も比較はそのまま使える。**
    /// </summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromMinutes(4);

    /// <summary>
    /// 繋がるかを確かめるときの上限。**さらに短く。**
    /// 設定画面で「試す」を押したときに何十秒も待たせない。
    /// </summary>
    public TimeSpan ProbeTimeout { get; init; } = TimeSpan.FromSeconds(5);

    /// <summary>
    /// 返事の長さの上限。
    ///
    /// **必ず入れる。** 入れずに形（JSON Schema）だけ指定すると、弱いモデルは
    /// 配列を延々と伸ばして終われなくなる。実測（Qwen2.5 1.5B、15.9 トークン/秒）
    /// では、上限なしで 797 トークン書いても終わらず時限切れ。上限 400 を付けると
    /// 同じ問いに 100 トークンほどで正しく答えた。
    ///
    /// **形の指定と対で要る。** 制約付き復号は「文法に沿うこと」しか保証せず、
    /// 「短く終わること」は保証しない。
    /// </summary>
    public int MaxOutputTokens { get; init; } = 600;

    /// <summary>
    /// 説明のときの上限。**短くする。**
    /// 2〜3 文と選択肢が数個なので 400 で足りる。CPU の 7B では
    /// 1 トークンが 0.45 秒なので、上限がそのまま待ち時間になる。
    /// </summary>
    public int ExplainMaxTokens { get; init; } = 400;

    /// <summary>
    /// 解決案（生成）まで出してよいか。
    ///
    /// **既定は false。** 説明や分類と違い、解決案は意味を取り違えると
    /// 害になる生成で、8B 級のモデルは**もっともらしく間違える**。
    /// ビルドが通るぶんだけ発見が遅れる。
    ///
    /// 信頼度を返させて閾値で切る手は取らない。**弱いモデルは自信も
    /// 較正されていない。** モデルの素性で決める方が確実。
    /// </summary>
    public bool AllowResolutionProposals { get; init; }

    /// <summary>使える状態か。**空なら機能を出さない。**</summary>
    public bool IsConfigured
        => Endpoint.Length > 0 && Model.Length > 0;

    /// <summary>
    /// 送り先の URL を組み立てる。末尾の <c>/</c> の有無に左右されないようにする。
    /// </summary>
    public string UrlFor(string path)
        => $"{Endpoint.TrimEnd('/')}/{path.TrimStart('/')}";

    /// <summary>
    /// 表に出してよい形。**鍵を伏せる。**
    /// 設定を書き出す場所すべてでこれを通す。
    /// </summary>
    public string Redacted()
        => ApiKey is { Length: > 0 }
            ? $"{Endpoint} ({Model}, 鍵あり)"
            : $"{Endpoint} ({Model})";
}
