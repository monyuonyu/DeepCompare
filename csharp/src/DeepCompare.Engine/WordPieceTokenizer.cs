using System.Globalization;
using System.Text;

namespace DeepCompare.Engine;

/// <summary>
/// BERT の WordPiece トークナイザ。
///
/// Microsoft.ML.Tokenizers の BertTokenizer を使わず自前で持っている理由:
/// **あれは一部の記号を黙って捨てる**。`+`、`&lt;`、`&gt;` は語彙に存在する（1009 / 1026 / 1028）のに
/// 出力から消え、`x + 1` が `x 1` になっていた。コードの比較では `x + 1` と `x - 1` の
/// 区別が失われ、C++ のテンプレートやジェネリクスの差も見えなくなるため許容できない。
///
/// 規則は付属の tokenizer.json に書かれているものをそのまま実装している
/// （BertNormalizer: clean_text + handle_chinese_chars + strip_accents + lowercase、
/// BertPreTokenizer、WordPiece）。Python の tokenizers と Rust の実装もこれに従うので、
/// 同じ入力に対して同じトークン列が出る。
///
/// 注意: この正規化は非可逆で、日本語の濁点も落ちる（「だ」→「た」）。実測で確認した
/// モデル本来の挙動であり、一致させるためには再現する必要がある。
/// </summary>
public sealed class WordPieceTokenizer
{
    private const string ContinuationPrefix = "##";
    private const int MaxCharsPerWord = 100;

    private readonly Dictionary<string, int> _vocab;
    private readonly int _unknownId;
    private readonly int _clsId;
    private readonly int _sepId;

    private WordPieceTokenizer(Dictionary<string, int> vocab)
    {
        _vocab = vocab;
        _unknownId = vocab["[UNK]"];
        _clsId = vocab["[CLS]"];
        _sepId = vocab["[SEP]"];
    }

    /// <summary>1 行 1 語の vocab.txt から作る。行番号がそのまま ID。</summary>
    public static WordPieceTokenizer FromVocab(Stream stream)
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        using var reader = new StreamReader(stream, new UTF8Encoding(false));
        var id = 0;
        while (reader.ReadLine() is { } line)
        {
            // 語彙の行末には改行以外の空白は含まれない前提だが、CRLF だけは剥がす。
            vocab.TryAdd(line.TrimEnd('\r'), id);
            id++;
        }
        foreach (var required in new[] { "[UNK]", "[CLS]", "[SEP]" })
        {
            if (!vocab.ContainsKey(required))
            {
                throw new InvalidOperationException($"語彙に {required} が無い");
            }
        }
        return new WordPieceTokenizer(vocab);
    }

    /// <summary>[CLS] と [SEP] を付けた ID 列を返す。</summary>
    public List<int> Encode(string text, int maxTokens)
    {
        var ids = new List<int> { _clsId };
        foreach (var word in PreTokenize(Normalize(text)))
        {
            foreach (var id in EncodeWord(word))
            {
                // [SEP] の分を 1 つ残す。
                if (ids.Count >= maxTokens - 1)
                {
                    ids.Add(_sepId);
                    return ids;
                }
                ids.Add(id);
            }
        }
        ids.Add(_sepId);
        return ids;
    }

    /// <summary>BertNormalizer 相当。順序は clean_text → 漢字の分離 → 濁点等の除去 → 小文字化。</summary>
    internal string Normalize(string text)
    {
        var cleaned = new StringBuilder(text.Length + 16);
        foreach (var c in text)
        {
            if (c == '\0' || c == '�' || (char.IsControl(c) && c is not ('\t' or '\n' or '\r')))
            {
                continue;
            }
            if (char.IsWhiteSpace(c))
            {
                cleaned.Append(' ');
                continue;
            }
            // 漢字は 1 文字ずつ独立した語として扱う。前後に空白を入れて分離する。
            if (IsChineseChar(c))
            {
                cleaned.Append(' ').Append(c).Append(' ');
                continue;
            }
            cleaned.Append(c);
        }

        // NFD へ分解して結合文字を落とす。合成し直さないのが本来の挙動。
        var decomposed = cleaned.ToString().Normalize(NormalizationForm.FormD);
        var stripped = new StringBuilder(decomposed.Length);
        foreach (var c in decomposed)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)
            {
                stripped.Append(c);
            }
        }
        return stripped.ToString().ToLowerInvariant();
    }

    /// <summary>BertPreTokenizer 相当。空白で切り、記号は 1 文字ずつ独立させる。</summary>
    internal static List<string> PreTokenize(string text)
    {
        var words = new List<string>();
        var current = new StringBuilder();

        void Flush()
        {
            if (current.Length > 0)
            {
                words.Add(current.ToString());
                current.Clear();
            }
        }

        foreach (var c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                Flush();
            }
            else if (IsPunctuation(c))
            {
                Flush();
                words.Add(c.ToString());
            }
            else
            {
                current.Append(c);
            }
        }
        Flush();
        return words;
    }

    /// <summary>WordPiece。前から最長一致で切り、2 つ目以降には ## を付ける。</summary>
    private List<int> EncodeWord(string word)
    {
        if (word.Length > MaxCharsPerWord)
        {
            return [_unknownId];
        }

        var ids = new List<int>();
        var start = 0;
        while (start < word.Length)
        {
            var end = word.Length;
            var matched = -1;
            while (start < end)
            {
                var piece = start == 0 ? word[start..end] : ContinuationPrefix + word[start..end];
                if (_vocab.TryGetValue(piece, out var id))
                {
                    matched = id;
                    break;
                }
                end--;
            }
            if (matched < 0)
            {
                // 語のどこかが語彙に無ければ、語全体を未知語にする。
                return [_unknownId];
            }
            ids.Add(matched);
            start = end;
        }
        return ids;
    }

    /// <summary>
    /// BERT が「記号」とみなす範囲。ASCII の記号域は Unicode 分類では
    /// 記号(P)ではなく数学記号(S)などに入るものがあるため、明示的に含める。
    /// ここを落とすと `+` や `&lt;` が語に飲み込まれて未知語になる。
    /// </summary>
    internal static bool IsPunctuation(char c)
    {
        if (c is (>= '!' and <= '/') or (>= ':' and <= '@') or (>= '[' and <= '`') or (>= '{' and <= '~'))
        {
            return true;
        }
        return CharUnicodeInfo.GetUnicodeCategory(c) switch
        {
            UnicodeCategory.ConnectorPunctuation
                or UnicodeCategory.DashPunctuation
                or UnicodeCategory.OpenPunctuation
                or UnicodeCategory.ClosePunctuation
                or UnicodeCategory.InitialQuotePunctuation
                or UnicodeCategory.FinalQuotePunctuation
                or UnicodeCategory.OtherPunctuation => true,
            _ => false,
        };
    }

    /// <summary>漢字（CJK 統合漢字とその拡張）。ひらがな・カタカナは含まない。</summary>
    internal static bool IsChineseChar(char c)
        => c is (>= '一' and <= '鿿')
            or (>= '㐀' and <= '䶿')
            or (>= '豈' and <= '﫿');
}
