namespace DeepCompare.Engine;

/// <summary>
/// いまのモデルが、その本文をどれだけ扱えるか。
///
/// **効いていないことを黙らない。** 意味的な対応付けはこの道具の売りだが、
/// 英語中心のモデルは日本語をほとんど語彙に持たない。それを言わずに
/// 「意味的に比べました」と出すのは、黙って間違った結果を出すのに近い。
///
/// 実測（2026-08-13）:
/// - 英語モデル（minilm.dcm、WordPiece 30,522 語）
///   かな 184 項目・漢字 486 項目（あわせて 2%）。**濁点つきのかなは 1 つも無い**
///   （BertNormalizer の strip_accents が NFD 分解で U+3099 を落とすため、
///   学習時から存在しない）。「バグを直す」と「ハクを直す」の類似度が **1.0000**
/// - 多言語モデル（multilingual.dcm、unigram 250,002 語）
///   濁点つきのかなが揃っており、「設定」「読み」「込む」のように語で切れる
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
    /// 日本語を扱えるモデルとみなす語彙の数。
    ///
    /// **語彙の数で見分ける。** 名前で判断すると、利用者が別名で置いた
    /// 多言語モデルを「日本語が苦手」と誤って言う。英語モデルは 30,522 語、
    /// 多言語モデルは 250,002 語で、桁が違う。
    /// </summary>
    public const int MultilingualVocabSize = 100_000;

    /// <summary>
    /// 知らせるべきことがあれば、その文言。無ければ null。
    ///
    /// **何ができないかを具体的に言う。** 「精度が落ちます」では、
    /// 何を疑えばいいのか分からない。
    /// </summary>
    /// <param name="vocabSize">
    /// いま使っているモデルの語彙の数。多言語モデルなら知らせない。
    /// 分からないときは null（英語モデルとみなして知らせる）。
    /// </param>
    public static string? Warn(
        IEnumerable<string> left, IEnumerable<string> right, int? vocabSize = null)
    {
        if (vocabSize >= MultilingualVocabSize)
        {
            return null;
        }

        var ratio = JapaneseRatio(left.Concat(right));
        return ratio > JapaneseThreshold
            // **短く言う。** 4 文にわたって説明していたが、状態バーの上に
            // 常時居座るうえ、読んでも何をすればよいのか分からなかった。
            // 「いま何が起きているか」と「どうすれば直るか」だけ残す。
            ? "日本語では意味的な対応付けが効きません（文字の重なりで並べています）。"
              + "多言語モデルを置くと効きます — README を参照。"
            : null;
    }
}
