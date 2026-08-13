namespace DeepCompare.Engine;

/// <summary>対応付けられた 1 行分。左右どちらかが空くことがある。</summary>
public readonly record struct Pair(int? Left, int? Right, float? Score)
{
    public static Pair Both(int left, int right, float score) => new(left, right, score);
    public static Pair LeftOnly(int left) => new(left, null, null);
    public static Pair RightOnly(int right) => new(null, right, null);
}

/// <summary>変化した塊。両側の行番号の半開区間で表す。</summary>
public readonly record struct Block(int LeftStart, int LeftLength, int RightStart, int RightLength)
{
    public long Area => (long)LeftLength * RightLength;
    public bool IsEmpty => LeftLength == 0 && RightLength == 0;
}

/// <summary>動かない区間か、意味的アライメントにかける塊か。</summary>
public abstract record Segment
{
    public sealed record Identical(int LeftStart, int RightStart, int Length) : Segment;
    public sealed record Changed(Block Block) : Segment;
}

/// <summary>
/// 行の対応付け。
///
/// 旧 Python 実装は両ファイルの全行の総当たりで類似度行列を作り、その上で
/// Needleman-Wunsch を回していた。1000 行同士なら 100 万マスの DP に 100 万回の
/// モデル推論が乗り、実用的なファイルでは待てない。
///
/// ここでは二段に分ける。まず文字列一致で確実に対応する区間を畳み、残った
/// 「変化した塊」に対してだけ意味的アライメントを行う。
/// </summary>
public static class Aligner
{
    /// <summary>文字列一致で畳んだ行に与える類似度。モデルを通していないことを示す。</summary>
    public const float ExactScore = 1.0f;

    /// <summary>
    /// 対応付ける価値があるとみなす類似度の下限。これを下回る組は、対にせず
    /// 左右それぞれを空きとして並べる。
    ///
    /// 旧実装は「空き 1 つにつき -0.5」という罰則を置き、類似度はそのまま加算していた。
    /// だが類似度が 0..1 に収まる以上、対にすれば 0 以上、空き 2 つなら -1.0 なので、
    /// どれだけ無関係な行同士でも必ず対にされ、空きが選ばれることが原理的に無かった。
    /// たとえば左が [A, B]、右が [B', C] で B だけが対応する場合でも、
    /// A↔B' と B↔C という誤った対が作られる。
    ///
    /// ここでは対角のスコアを「類似度 - この閾値」とし、空きを 0 とする。こうすると
    /// 「類似度がこの値を上回るときだけ対にする」がそのまま式の意味になる。
    /// </summary>
    public const float DefaultPairThreshold = 0.5f;

    /// <summary>意味的アライメントに回す 1 塊の上限（左右の行数の積）。</summary>
    public const long MaxBlockArea = 4_000_000;

    /// <summary>文字列一致だけで、動かない区間と変化した塊に切り分ける。</summary>
    public static List<Segment> Split(IReadOnlyList<string> left, IReadOnlyList<string> right)
    {
        var segments = new List<Segment>();
        foreach (var op in Myers.Compute(left, right))
        {
            switch (op.Kind)
            {
                case DiffKind.Equal:
                    segments.Add(new Segment.Identical(op.OldStart, op.NewStart, op.OldLength));
                    break;
                default:
                    var block = new Block(op.OldStart, op.OldLength, op.NewStart, op.NewLength);
                    if (!block.IsEmpty)
                    {
                        segments.Add(new Segment.Changed(block));
                    }
                    break;
            }
        }
        return segments;
    }

    /// <summary>
    /// Needleman-Wunsch。similarity(i, j) は塊の中での相対位置に対する類似度を返す。
    ///
    /// 対角のスコアは「類似度 - pairThreshold」、空きは 0。同点なら対角を選ぶ。
    /// 返す Score は閾値を引く前の生の類似度。
    /// </summary>
    public static List<Pair> NeedlemanWunsch(int rows, int cols, float pairThreshold, Func<int, int, float> similarity)
    {
        if (rows == 0)
        {
            return Enumerable.Range(0, cols).Select(Pair.RightOnly).ToList();
        }
        if (cols == 0)
        {
            return Enumerable.Range(0, rows).Select(Pair.LeftOnly).ToList();
        }

        var width = cols + 1;
        var dp = new float[(rows + 1) * width];
        // 経路。0 = 斜め, 1 = 上（左のみ）, 2 = 左（右のみ）。
        var from = new byte[(rows + 1) * width];

        // 端の列と行は空きが並ぶだけ。空きのスコアは 0 なので dp は 0 のまま。
        for (var i = 1; i <= rows; i++)
        {
            from[i * width] = 1;
        }
        for (var j = 1; j <= cols; j++)
        {
            from[j] = 2;
        }

        for (var i = 1; i <= rows; i++)
        {
            for (var j = 1; j <= cols; j++)
            {
                var diag = dp[(i - 1) * width + (j - 1)] + (similarity(i - 1, j - 1) - pairThreshold);
                var up = dp[(i - 1) * width + j];
                var leftScore = dp[i * width + (j - 1)];

                // 同点なら斜めを優先する。左右を空きにするより対応付けた方が読みやすい。
                //
                // 空き同士が同点の場合は「右だけ」を選ぶ。経路は逆向きに辿ってから
                // 反転するので、ここで右を選ぶと最終的な並びでは左（削除）が先に来る。
                // 差分ツールの慣習は削除→追加の順で、逆にすると読み違えやすい。
                float best;
                byte dir;
                if (diag >= up && diag >= leftScore)
                {
                    best = diag;
                    dir = 0;
                }
                else if (leftScore >= up)
                {
                    best = leftScore;
                    dir = 2;
                }
                else
                {
                    best = up;
                    dir = 1;
                }
                dp[i * width + j] = best;
                from[i * width + j] = dir;
            }
        }

        var pairs = new List<Pair>(Math.Max(rows, cols));
        var x = rows;
        var y = cols;
        while (x > 0 || y > 0)
        {
            // 端に張り付いたら残りは一方向にしか進めない。
            if (x == 0)
            {
                y--;
                pairs.Add(Pair.RightOnly(y));
                continue;
            }
            if (y == 0)
            {
                x--;
                pairs.Add(Pair.LeftOnly(x));
                continue;
            }
            switch (from[x * width + y])
            {
                case 0:
                    x--;
                    y--;
                    pairs.Add(Pair.Both(x, y, similarity(x, y)));
                    break;
                case 1:
                    x--;
                    pairs.Add(Pair.LeftOnly(x));
                    break;
                default:
                    y--;
                    pairs.Add(Pair.RightOnly(y));
                    break;
            }
        }
        pairs.Reverse();
        return pairs;
    }

    /// <summary>
    /// 意味的アライメントを行わずに左右をそのまま並べる。
    /// 塊が大きすぎて DP に載せられない場合の退避先。
    /// </summary>
    /// <summary>
    /// 文字の重なりで対応付ける。
    ///
    /// **埋め込みが使えないときの本命。** これまでは
    /// <see cref="WithoutScoring"/> で全部を削除＋追加として並べていたので、
    /// 1 文字違いの行すら対にならず、行の中のどこが変わったのかが
    /// 出せなかった（日本語ではモデルが効かないので、ほぼ常にこの経路）。
    ///
    /// 似ているかどうかは**2 文字の並びの重なり**で見る。単語で切らないので
    /// 日本語でも効き、語順の入れ替えにも強い。
    /// </summary>
    public static List<Pair> ByCharacters(
        Block block,
        IReadOnlyList<string> leftLines,
        IReadOnlyList<string> rightLines,
        float threshold = 0.35f)
    {
        // 各行の 2 文字の並びを先に作る。塊の中で何度も突き合わせるので、
        // 毎回作ると O(n*m*文字数) になる。
        var leftGrams = new HashSet<int>[block.LeftLength];
        var rightGrams = new HashSet<int>[block.RightLength];
        for (var i = 0; i < block.LeftLength; i++)
        {
            leftGrams[i] = Bigrams(leftLines[block.LeftStart + i]);
        }
        for (var j = 0; j < block.RightLength; j++)
        {
            rightGrams[j] = Bigrams(rightLines[block.RightStart + j]);
        }

        var local = NeedlemanWunsch(
            block.LeftLength, block.RightLength, threshold,
            (i, j) => Dice(leftGrams[i], rightGrams[j]));

        return [.. local.Select(p => new Pair(
            p.Left is { } li ? block.LeftStart + li : null,
            p.Right is { } ri ? block.RightStart + ri : null,
            p.Score))];
    }

    /// <summary>
    /// 2 文字の並びの集合。
    ///
    /// **空白は落とす。** 字下げだけが違う行を「別物」と見ないため。
    /// 1 文字の行は、その 1 文字を 1 つの並びとして持つ（空集合にすると
    /// どの行とも似ていないことになる）。
    /// </summary>
    private static HashSet<int> Bigrams(string text)
    {
        var set = new HashSet<int>();
        var previous = -1;
        foreach (var c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                continue;
            }
            if (previous >= 0)
            {
                set.Add((previous << 16) | c);
            }
            previous = c;
        }
        if (set.Count == 0 && previous >= 0)
        {
            set.Add(previous);
        }
        return set;
    }

    /// <summary>重なりの度合い（Dice 係数）。両方が空なら 1。</summary>
    private static float Dice(HashSet<int> a, HashSet<int> b)
    {
        if (a.Count == 0 && b.Count == 0)
        {
            return 1;
        }
        if (a.Count == 0 || b.Count == 0)
        {
            return 0;
        }
        var shared = 0;
        // 小さい方を回す。
        var (small, large) = a.Count <= b.Count ? (a, b) : (b, a);
        foreach (var gram in small)
        {
            if (large.Contains(gram))
            {
                shared++;
            }
        }
        return 2f * shared / (a.Count + b.Count);
    }

    public static List<Pair> WithoutScoring(Block block)
    {
        var pairs = new List<Pair>(block.LeftLength + block.RightLength);
        for (var i = 0; i < block.LeftLength; i++)
        {
            pairs.Add(Pair.LeftOnly(block.LeftStart + i));
        }
        for (var j = 0; j < block.RightLength; j++)
        {
            pairs.Add(Pair.RightOnly(block.RightStart + j));
        }
        return pairs;
    }
}
