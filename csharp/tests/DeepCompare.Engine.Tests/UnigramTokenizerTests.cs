using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public class UnigramTokenizerTests
{
    /// <summary>
    /// 小さな語彙を組み立てる。**本物の 250k を試験に持ち込まない** —
    /// 試験は資材の有無に左右されず、どこでも同じように走るべき。
    /// 本物との照合は外部の参照実装（tokenizers）でやってある。
    /// </summary>
    private static UnigramTokenizer Build(params (string Token, float Score)[] entries)
    {
        var text = new StringBuilder();
        // 番号は行の順。特殊トークンの位置を本物（XLM-R）に合わせる。
        text.Append("<s>\t0.0\n<pad>\t0.0\n</s>\t0.0\n<unk>\t0.0\n");
        foreach (var (token, score) in entries)
        {
            text.Append($"{token}\t{score.ToString(System.Globalization.CultureInfo.InvariantCulture)}\n");
        }
        return UnigramTokenizer.FromVocab(
            new MemoryStream(new UTF8Encoding(false).GetBytes(text.ToString())));
    }

    [Fact]
    public void 特殊トークンの番号を語彙から読む()
    {
        var tokenizer = Build(("あ", -1f));
        Assert.Equal(0, tokenizer.BosId);
        Assert.Equal(1, tokenizer.PadId);
        Assert.Equal(2, tokenizer.EosId);
    }

    [Fact]
    public void 先頭に空白の印を付ける()
    {
        var tokenizer = Build(("▁あ", -1f), ("あ", -1f));
        Assert.Equal(["▁あ"], tokenizer.Tokenize("あ"));
    }

    [Fact]
    public void 語中の空白も印になる()
    {
        var tokenizer = Build(("▁a", -1f), ("▁b", -1f));
        Assert.Equal(["▁a", "▁b"], tokenizer.Tokenize("a b"));
    }

    [Fact]
    public void 連続した空白はひとつにまとめる()
    {
        var tokenizer = Build(("▁a", -1f), ("▁b", -1f));
        Assert.Equal(["▁a", "▁b"], tokenizer.Tokenize("a    b"));
    }

    [Fact]
    public void 末尾の空白は落とす()
    {
        var tokenizer = Build(("▁a", -1f));
        Assert.Equal(["▁a"], tokenizer.Tokenize("a   "));
    }

    [Fact]
    public void スコアの高い切り方を選ぶ()
    {
        // 「abc」は "ab"+"c" とも "a"+"bc" とも切れる。**合計が高い方**を選ぶ。
        var tokenizer = Build(
            ("▁", -1f), ("a", -5f), ("b", -5f), ("c", -5f),
            ("ab", -1f), ("bc", -9f));
        Assert.Equal(["▁", "ab", "c"], tokenizer.Tokenize("abc"));
    }

    [Fact]
    public void 長い一致より合計の良い切り方を優先する()
    {
        // **貪欲な最長一致だと "abcd" を取って残りが未知になる。**
        // Viterbi は全体を見るので、そちらを選ばない。
        var tokenizer = Build(
            ("▁", -1f), ("abcd", -1f), ("ab", -0.1f), ("cde", -0.1f));
        Assert.Equal(["▁", "ab", "cde"], tokenizer.Tokenize("abcde"));
    }

    [Fact]
    public void 語彙に無い文字があっても行全体を切れる()
    {
        var tokenizer = Build(("▁", -1f), ("a", -1f), ("b", -1f));
        var pieces = tokenizer.Tokenize("a☃b");
        Assert.Equal(["▁", "a", "☃", "b"], pieces);
    }

    [Fact]
    public void 語彙に無い文字は未知の番号になる()
    {
        var tokenizer = Build(("▁", -1f), ("a", -1f));
        var ids = tokenizer.Encode("a☃", 32);
        // <s> ▁ a <unk> </s>
        Assert.Equal([0, 4, 5, 3, 2], ids);
    }

    [Fact]
    public void 前後に開始と終了の印を付ける()
    {
        var tokenizer = Build(("▁a", -1f));
        var ids = tokenizer.Encode("a", 32);
        Assert.Equal(tokenizer.BosId, ids[0]);
        Assert.Equal(tokenizer.EosId, ids[^1]);
    }

    [Fact]
    public void 上限を超えても終了の印は必ず付く()
    {
        var tokenizer = Build(("▁", -1f), ("a", -1f));
        var ids = tokenizer.Encode(new string('a', 100), maxLength: 8);

        Assert.Equal(8, ids.Count);
        Assert.Equal(tokenizer.BosId, ids[0]);
        // **溢れても終わりの印を落とさない。** 落とすとモデルが
        // 「文が続いている」と見て、埋め込みが変わる。
        Assert.Equal(tokenizer.EosId, ids[^1]);
    }

    [Fact]
    public void 空の行でも印だけは返す()
    {
        var tokenizer = Build(("▁a", -1f));
        var ids = tokenizer.Encode("", 32);
        Assert.Equal([tokenizer.BosId, tokenizer.EosId], ids);
    }

    [Fact]
    public void 全角の記号は正規化でそろう()
    {
        // NFKC。**全角の英数字を別物として扱わない。**
        var tokenizer = Build(("▁", -1f), ("a", -1f), ("1", -1f));
        Assert.Equal(["▁", "a", "1"], tokenizer.Tokenize("ａ１"));
    }

    [Fact]
    public void 濁点つきのかなを落とさない()
    {
        // **これがこのトークナイザーを作った理由。** WordPiece 側は
        // strip_accents で濁点を落とし、「バグ」と「ハク」を同じにする。
        var tokenizer = Build(("▁", -1f), ("バ", -1f), ("グ", -1f), ("ハ", -1f), ("ク", -1f));
        Assert.Equal(["▁", "バ", "グ"], tokenizer.Tokenize("バグ"));
        Assert.Equal(["▁", "ハ", "ク"], tokenizer.Tokenize("ハク"));
        Assert.NotEqual(tokenizer.Encode("バグ", 32), tokenizer.Encode("ハク", 32));
    }

    [Fact]
    public void 大文字と小文字を区別する()
    {
        // tokenizer_config.json の do_lower_case は**使われない**設定。
        // 実際の normalizer には Lowercase が入っていない。
        var tokenizer = Build(("▁", -1f), ("A", -1f), ("a", -1f));
        Assert.Equal(["▁", "A"], tokenizer.Tokenize("A"));
    }

    [Fact]
    public void タブの無い語彙は語だけとして読む()
    {
        var text = "<s>\n<pad>\n</s>\n<unk>\n▁a\n";
        var tokenizer = UnigramTokenizer.FromVocab(
            new MemoryStream(new UTF8Encoding(false).GetBytes(text)));
        Assert.Equal(5, tokenizer.Count);
        Assert.Equal(["▁a"], tokenizer.Tokenize("a"));
    }

    [Fact]
    public void 同じ合計になる切り方は先に来た方を採る()
    {
        // 「MMM」は "M"+"MM" とも "MM"+"M" とも切れて**合計が同じ**。
        // 合計を float で取ると加算の順序で丸めが変わり、参照実装と
        // 違う方を選んでいた（本物の語彙で 155 行中 1 行だけ食い違った）。
        // double で足すと同点が真の同点になり、先に来た方が残る。
        var tokenizer = Build(("▁", -1f), ("M", -6.3547845f), ("MM", -9.1287651f));
        Assert.Equal(["▁", "M", "MM"], tokenizer.Tokenize("MMM"));
    }

    [Fact]
    public void 空の語彙は断る()
    {
        Assert.Throws<InvalidDataException>(
            () => UnigramTokenizer.FromVocab(new MemoryStream()));
    }
}

public class TokenizerLoaderTests
{
    private static Stream Of(string text)
        => new MemoryStream(new UTF8Encoding(false).GetBytes(text));

    [Fact]
    public void タブがあれば_unigram_として読む()
    {
        var tokenizer = TokenizerLoader.Load(Of("<s>\t0.0\n<pad>\t0.0\n</s>\t0.0\n<unk>\t0.0\n▁a\t-1\n"));
        Assert.IsType<UnigramTokenizer>(tokenizer);
    }

    [Fact]
    public void タブが無ければ_wordpiece_として読む()
    {
        var tokenizer = TokenizerLoader.Load(Of("[PAD]\n[UNK]\n[CLS]\n[SEP]\nabc\n"));
        Assert.IsType<WordPieceTokenizer>(tokenizer);
    }

    [Fact]
    public void 数行目にタブが出ても見つける()
    {
        // 先頭の特殊トークンにスコアが無い語彙もありうる。**1 行目で決めない。**
        var tokenizer = TokenizerLoader.Load(Of("<s>\n<pad>\n</s>\n<unk>\n▁a\t-1\n▁b\t-2\n"));
        Assert.IsType<UnigramTokenizer>(tokenizer);
    }

    [Fact]
    public void 巻き戻せない流れでも読める()
    {
        // 埋め込み資材やネットワークの流れは Seek できない。
        var text = "<s>\t0.0\n<pad>\t0.0\n</s>\t0.0\n<unk>\t0.0\n▁a\t-1\n";
        using var forwardOnly = new ForwardOnlyStream(
            new UTF8Encoding(false).GetBytes(text));
        var tokenizer = TokenizerLoader.Load(forwardOnly);
        Assert.Equal(5, tokenizer.Count);
    }

    /// <summary>前へしか進めない流れ。Seek を呼ぶと落ちる。</summary>
    private sealed class ForwardOnlyStream(byte[] data) : Stream
    {
        private int _position;

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            var take = Math.Min(count, data.Length - _position);
            Array.Copy(data, _position, buffer, offset, take);
            _position += take;
            return take;
        }

        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin)
            => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count)
            => throw new NotSupportedException();
    }
}
