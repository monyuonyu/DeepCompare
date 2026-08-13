namespace DeepCompare.Engine;

/// <summary>
/// 行を番号の並びに変える物。
///
/// **2 つある。** 英語中心のモデル（WordPiece）と多言語モデル（unigram）で
/// 切り方の規則がまったく違う。モデルを差し替えるときはトークナイザーも
/// 一緒に替わる必要があり、片方だけ替えると**無言で無意味な結果**が出る
/// （番号は付くが、モデルが学習した番号とは別物になる）。
/// </summary>
public interface ITokenizer
{
    /// <summary>
    /// 番号にする。特殊トークンは実装が付ける。
    /// <paramref name="maxTokens"/> には特殊トークンも数える。
    /// </summary>
    List<int> Encode(string text, int maxTokens);

    /// <summary>語彙の数。**重みと合っているかを確かめるのに使う。**</summary>
    int Count { get; }
}

/// <summary>語彙ファイルを読んで、正しい方のトークナイザーを作る。</summary>
public static class TokenizerLoader
{
    /// <summary>
    /// 中身を見て、どちらの方式かを決める。
    ///
    /// **見分け方はタブの有無。** unigram の語彙は各行が
    /// <c>トークン\tスコア</c> で、WordPiece はトークンだけ。
    /// 拡張子で決めないのは、モデルを差し替える人が名前を揃えてくれるとは
    /// 限らないから。**中身が正。**
    /// </summary>
    public static ITokenizer Load(Stream stream)
    {
        // 先頭を覗く。**巻き戻せる形に写してから読む** — ファイル以外の流れ
        // （埋め込み資材やネットワーク）は巻き戻せないことがある。
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        buffer.Position = 0;

        var isUnigram = LooksLikeUnigram(buffer);
        buffer.Position = 0;

        return isUnigram
            ? UnigramTokenizer.FromVocab(buffer)
            : WordPieceTokenizer.FromVocab(buffer);
    }

    private static bool LooksLikeUnigram(Stream stream)
    {
        using var reader = new StreamReader(
            stream, System.Text.Encoding.UTF8, true, 1024, leaveOpen: true);

        // 先頭の何行かを見る。**1 行目だけで決めない** — unigram の語彙は
        // <s> や <pad> のスコアが 0.0 で始まるが、そこが欠けている物もありうる。
        for (var i = 0; i < 8; i++)
        {
            if (reader.ReadLine() is not { } line)
            {
                break;
            }
            if (line.Contains('\t'))
            {
                return true;
            }
        }
        return false;
    }
}
