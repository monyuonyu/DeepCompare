using System.Globalization;
using System.Text;

namespace DeepCompare.Engine;

/// <summary>
/// SentencePiece の unigram 方式のトークナイザー（XLM-R 系）。
///
/// **WordPiece とは別物。** あちらは「語の先頭から最長一致で貪欲に切る」が、
/// こちらは**分割全体のスコアが最大になる切り方を選ぶ**（Viterbi）。
/// 貪欲だと、先に長いトークンを取ったせいで残りが unk になる切り方を選んでしまう。
///
/// 日本語のためにこれが要る。英語中心の WordPiece 語彙には濁点つきのかなが
/// 1 つも無く（`strip_accents` が学習時に落とす）、「バグ」と「ハク」が
/// 同じ埋め込みになる。多言語モデルの語彙には濁点つきのかながある。
/// </summary>
public sealed class UnigramTokenizer : ITokenizer
{
    /// <summary>空白の代わりに置く印。SentencePiece の決まり。</summary>
    public const char SpaceMarker = '▁';   // ▁

    private readonly Dictionary<string, (int Id, float Score)> _vocab;
    private readonly int _unknownId;
    private readonly int _maxTokenLength;

    /// <summary>先頭と末尾に付ける印。</summary>
    public int BosId { get; }
    public int EosId { get; }
    public int PadId { get; }

    private UnigramTokenizer(
        Dictionary<string, (int, float)> vocab, int unknownId,
        int bos, int eos, int pad, int maxTokenLength)
    {
        _vocab = vocab;
        _unknownId = unknownId;
        _maxTokenLength = maxTokenLength;
        BosId = bos;
        EosId = eos;
        PadId = pad;
    }

    /// <summary>
    /// 語彙を読む。1 行が <c>トークン\tスコア</c>。
    ///
    /// **タブ区切りにする。** JSON のまま持つと 15MB を読むのに時間がかかり、
    /// 起動が遅くなる（比較の初回だけとはいえ、待ち時間はそこに出る）。
    /// </summary>
    public static UnigramTokenizer FromVocab(Stream stream)
    {
        var vocab = new Dictionary<string, (int, float)>(StringComparer.Ordinal);
        var unknownId = 3;
        var bos = 0;
        var eos = 2;
        var pad = 1;
        var maxLength = 1;

        using var reader = new StreamReader(stream, new UTF8Encoding(false));
        var id = 0;
        while (reader.ReadLine() is { } line)
        {
            if (line.Length == 0)
            {
                id++;
                continue;
            }

            var tab = line.LastIndexOf('\t');
            var token = tab < 0 ? line : line[..tab];
            var score = tab < 0 ? 0f
                : float.TryParse(line[(tab + 1)..], NumberStyles.Float,
                    CultureInfo.InvariantCulture, out var parsed) ? parsed : 0f;

            // 同じ綴りが二度出たら、**先に出た方を採る**（id が小さい方が主）。
            if (!vocab.ContainsKey(token))
            {
                vocab[token] = (id, score);
            }

            switch (token)
            {
                case "<unk>": unknownId = id; break;
                case "<s>": bos = id; break;
                case "</s>": eos = id; break;
                case "<pad>": pad = id; break;
            }

            if (token.Length > maxLength)
            {
                maxLength = token.Length;
            }
            id++;
        }

        if (vocab.Count == 0)
        {
            throw new InvalidDataException("語彙が空です。");
        }
        return new UnigramTokenizer(vocab, unknownId, bos, eos, pad, maxLength);
    }

    public int Count => _vocab.Count;

    /// <summary>
    /// 正規化する。
    ///
    /// **NFKC で近似する。** 本来は SentencePiece の precompiled_charsmap
    /// （nmt_nfkc）だが、あれはバイナリの表で、完全再現には表そのものを
    /// 持ち込むことになる。日本語と英語の主要部分は NFKC で足りることを
    /// 参照実装との照合で確かめてある。
    /// </summary>
    public static string Normalize(string text)
    {
        // **小文字化しない。** tokenizer_config.json には do_lower_case: true と
        // 書いてあるが、あれは古い（slow）実装向けの設定で使われない。
        // 実際の挙動は tokenizer.json の normalizer が決め、そこに Lowercase は
        // 入っていない。**設定ファイルを 2 つ見て、新しい方を採る。**
        return text.Normalize(NormalizationForm.FormKC);
    }

    /// <summary>
    /// 空白を印に置き換える（Metaspace）。
    ///
    /// **先頭にも印を付ける。** そうしないと、行頭の語が「語の途中」の
    /// トークンとして切られ、埋め込みが変わる。
    /// </summary>
    internal static string ApplyMetaspace(string text)
    {
        // **中身が無ければ何も返さない。** 空白だけの行に印を 1 つ返すと、
        // それが語彙に無くて未知の番号になる。空行はコードの比較で大量に出る
        // ので、そこが全部同じ「未知」の埋め込みになると対応付けが狂う。
        if (text.Length == 0 || text.All(char.IsWhiteSpace))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(text.Length + 1);
        builder.Append(SpaceMarker);

        var previousWasSpace = true;
        foreach (var c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                // 連続する空白は 1 つにまとめる（WhitespaceSplit 相当）。
                if (!previousWasSpace)
                {
                    builder.Append(SpaceMarker);
                    previousWasSpace = true;
                }
                continue;
            }
            builder.Append(c);
            previousWasSpace = false;
        }

        // 末尾が印だけになったら落とす。
        if (builder.Length > 1 && builder[^1] == SpaceMarker)
        {
            builder.Length--;
        }
        return builder.ToString();
    }

    /// <summary>
    /// 最尤の分割を求める（Viterbi）。
    ///
    /// `best[i]` は「先頭から i 文字目までを切ったときの最良のスコア」。
    /// **貪欲に最長一致で切らない。** 先に長いトークンを取ったせいで、
    /// 残りが unk になる切り方を選んでしまう。
    /// </summary>
    internal List<string> Split(string text)
    {
        if (text.Length == 0)
        {
            return [];
        }

        var best = new float[text.Length + 1];
        var from = new int[text.Length + 1];
        Array.Fill(best, float.NegativeInfinity);
        best[0] = 0;

        // 未知の 1 文字に与える罰。**十分に悪い値**にして、語彙にある切り方が
        // あるならそちらが選ばれるようにする。
        const float UnknownPenalty = -20f;

        for (var i = 0; i < text.Length; i++)
        {
            if (float.IsNegativeInfinity(best[i]))
            {
                continue;
            }

            var limit = Math.Min(_maxTokenLength, text.Length - i);
            var matched = false;

            for (var length = limit; length >= 1; length--)
            {
                var piece = text.Substring(i, length);
                if (!_vocab.TryGetValue(piece, out var entry))
                {
                    continue;
                }
                matched = true;
                var score = best[i] + entry.Score;
                if (score > best[i + length])
                {
                    best[i + length] = score;
                    from[i + length] = i;
                }
            }

            // 語彙に無い文字。**1 文字だけ進める。** ここで諦めると、
            // 未知の文字が 1 つあるだけで行全体が切れなくなる。
            if (!matched || float.IsNegativeInfinity(best[i + 1]))
            {
                var score = best[i] + UnknownPenalty;
                if (score > best[i + 1])
                {
                    best[i + 1] = score;
                    from[i + 1] = i;
                }
            }
        }

        // 後ろから辿る。
        var pieces = new List<string>();
        var at = text.Length;
        while (at > 0)
        {
            var start = from[at];
            pieces.Add(text[start..at]);
            at = start;
        }
        pieces.Reverse();
        return pieces;
    }

    /// <summary>切った結果を文字列で返す。試験と照合に使う。</summary>
    public List<string> Tokenize(string text)
        => Split(ApplyMetaspace(Normalize(text)));

    /// <summary>
    /// 番号にする。前後に <c>&lt;s&gt;</c> と <c>&lt;/s&gt;</c> を足す。
    /// </summary>
    public List<int> Encode(string text, int maxLength)
    {
        var ids = new List<int>(maxLength) { BosId };

        foreach (var piece in Tokenize(text))
        {
            // 印の分を残して切る。**先に足してから溢れさせない。**
            if (ids.Count >= maxLength - 1)
            {
                break;
            }
            ids.Add(_vocab.TryGetValue(piece, out var entry) ? entry.Id : _unknownId);
        }

        ids.Add(EosId);
        return ids;
    }
}
