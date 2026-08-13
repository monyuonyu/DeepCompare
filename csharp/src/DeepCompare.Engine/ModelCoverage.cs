namespace DeepCompare.Engine;

/// <summary>
/// いまのモデルが、その本文をどれだけ扱えるか。
///
/// **効いていないことを黙らない。** 意味的な対応付けはこの道具の売りだが、
/// 英語中心のモデルは日本語をほとんど語彙に持たない。それを言わずに
/// 「意味的に比べました」と出すのは、黙って間違った結果を出すのに近い。
///
/// 実測（2026-08-13、既定の minilm.dcm）:
/// - 語彙 30,522 のうち、かな 184 項目・漢字 486 項目（あわせて 2%）
/// - **濁点つきのかなは 1 つも無い。** BertNormalizer の strip_accents が
///   NFD 分解で結合文字（U+3099）を落とすため、学習時から存在しない
/// - その結果、「バグを直す」と「ハクを直す」の類似度が **1.0000** になる
/// </summary>
public static class ModelCoverage
{
    /// <summary>
    /// 日本語の文字の割合がこれを超えたら知らせる。
    ///
    /// **低くしすぎない。** コードの中に日本語のコメントが数行あるのは普通で、
    /// そのたびに警告が出ると読まなくなる。**半分を超えたら**「日本語の文書」
    /// として扱ってよい。
    /// </summary>
    public const double JapaneseThreshold = 0.5;

    /// <summary>その文字が日本語（かな・漢字）か。</summary>
    public static bool IsJapanese(char c)
        => c is >= '぀' and <= 'ゟ'      // ひらがな
        || c is >= '゠' and <= 'ヿ'      // カタカナ
        || c is >= '一' and <= '鿿'      // 漢字
        || c is >= 'ｦ' and <= 'ﾝ';     // 半角カタカナ

    /// <summary>
    /// 日本語の文字が占める割合。空白と記号は数えない。
    ///
    /// **記号を分母に入れない。** コードは括弧と記号が多いので、入れると
    /// 日本語だけの行でも割合が下がり、判定が効かなくなる。
    /// </summary>
    public static double JapaneseRatio(IEnumerable<string> lines)
    {
        var japanese = 0;
        var letters = 0;

        foreach (var line in lines)
        {
            foreach (var c in line)
            {
                if (IsJapanese(c))
                {
                    japanese++;
                    letters++;
                }
                else if (char.IsLetter(c))
                {
                    letters++;
                }
            }
        }
        return letters == 0 ? 0 : (double)japanese / letters;
    }

    /// <summary>
    /// 知らせるべきことがあれば、その文言。無ければ null。
    ///
    /// **何ができないかを具体的に言う。** 「精度が落ちます」では、
    /// 何を疑えばいいのか分からない。
    /// </summary>
    public static string? Warn(IEnumerable<string> left, IEnumerable<string> right)
    {
        var ratio = JapaneseRatio(left.Concat(right));
        return ratio > JapaneseThreshold
            ? $"日本語が {ratio:P0} を占めますが、いまのモデルは日本語をほとんど"
              + "語彙に持ちません（濁点も落ちるため「バグ」と「ハク」を同じと見ます）。"
              + "意味的な対応付けは効かず、文字の重なりで並んでいます。"
            : null;
    }
}
