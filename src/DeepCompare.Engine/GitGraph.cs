namespace DeepCompare.Engine;

/// <summary>グラフの線 1 本。ある行で、どの列からどの列へ繋がるか。</summary>
public readonly record struct GraphEdge(int FromLane, int ToLane);

/// <summary>
/// 履歴 1 行分のグラフ。
///
/// <paramref name="Lane"/> はこのコミットの丸を置く列。
/// <paramref name="Passing"/> はこの行を素通りする線の列（別の枝）。
/// <paramref name="Edges"/> はこの行から次の行（1 つ古い側）へ伸びる線。
/// </summary>
public sealed record GraphRow(
    int Lane,
    IReadOnlyList<int> Passing,
    IReadOnlyList<GraphEdge> Edges,
    int Width)
{
    /// <summary>親が 2 つ以上ある（マージ）。丸を大きくして目印にする。</summary>
    public bool IsMerge { get; init; }

    /// <summary>
    /// この行より上（新しい側）から、この列へ降りてくる線が無い。枝の先端。
    ///
    /// **上半分の線を引くかどうかがこれで決まる。** 先端にも引くと、画面の上に
    /// 続きがあるように見えて、そこが枝の始まりだと分からない。
    /// </summary>
    public bool IsTip { get; init; }
}

/// <summary>
/// コミットの並びに、枝の線（レーン）を割り当てる。
///
/// **SourceTree のグラフ列に当たるもの。** 一覧に並べるだけでは、どのコミットが
/// どの枝から来たのか、どこで分かれてどこで合流したのかが読み取れない。
///
/// 手順は素直に「上から順に、待っている親を列に置いていく」だけ。
/// **凝った見た目の最適化はしない。** 線が最短で結ばれるより、規則が単純で
/// 予測できる方が読みやすい（規則が分かれば目で追える）。
/// </summary>
public static class GitGraph
{
    /// <summary>
    /// 列を割り当てる。<paramref name="commits"/> は**新しい順**。
    ///
    /// 履歴が長いと列が増え続けるので、上限を持つ。超えた分は同じ列に重ねる。
    /// **列が 20 も並んだ時点で、目では追えない。**
    /// </summary>
    public static IReadOnlyList<GraphRow> Build(
        IReadOnlyList<GitCommit> commits, int maximumLanes = 12)
    {
        var rows = new List<GraphRow>(commits.Count);

        // 各列が「次に来るのを待っているコミット」のハッシュ。null なら空き。
        var lanes = new List<string?>();

        foreach (var commit in commits)
        {
            // この列で待たれているか。待たれていなければ新しい列に置く。
            var lane = lanes.IndexOf(commit.Hash);
            var tip = lane < 0;         // 誰も待っていない＝ここが枝の先端
            if (lane < 0)
            {
                lane = FreeLane(lanes, maximumLanes);
                if (lane >= lanes.Count)
                {
                    lanes.Add(commit.Hash);
                }
                else
                {
                    lanes[lane] = commit.Hash;
                }
            }

            // 同じコミットを待っている列がほかにもあれば、そこは合流して消える
            // （複数の枝が同じコミットに繋がる形）。
            for (var i = 0; i < lanes.Count; i++)
            {
                if (i != lane && lanes[i] == commit.Hash)
                {
                    lanes[i] = null;
                }
            }

            var edges = new List<GraphEdge>();

            // 親をどの列に置くか決める。**最初の親は自分の列を引き継ぐ。**
            // そうしないと、主要な枝が行ごとに横へ動いて追いにくくなる。
            for (var p = 0; p < commit.Parents.Count; p++)
            {
                var parent = commit.Parents[p];
                var existing = lanes.IndexOf(parent);

                int target;
                if (existing >= 0)
                {
                    target = existing;          // すでに待っている列がある（合流）
                }
                else if (p == 0)
                {
                    target = lane;              // 最初の親は自分の列
                    lanes[lane] = parent;
                }
                else
                {
                    target = FreeLane(lanes, maximumLanes);
                    if (target >= lanes.Count)
                    {
                        lanes.Add(parent);
                    }
                    else
                    {
                        lanes[target] = parent;
                    }
                }
                edges.Add(new GraphEdge(lane, target));
            }

            // **自分の列に自分がまだ残っていたら解放する。**
            // 最初の親が別の列で既に待たれている場合（分岐して合流する形）、
            // 自分の列を誰も引き継がない。放っておくと空かない列が増え続け、
            // 幅だけが伸びていく。
            if (lanes[lane] == commit.Hash)
            {
                lanes[lane] = null;
            }

            // この行を素通りする線（自分の列以外で、まだ待っている列）。
            var passing = new List<int>();
            for (var i = 0; i < lanes.Count; i++)
            {
                if (i != lane && lanes[i] is not null)
                {
                    passing.Add(i);
                }
            }

            // 末尾の空き列は詰める。放っておくと幅だけが伸びていく。
            while (lanes.Count > 0 && lanes[^1] is null)
            {
                lanes.RemoveAt(lanes.Count - 1);
            }

            rows.Add(new GraphRow(
                lane, passing, edges,
                Math.Max(lane + 1, lanes.Count))
            {
                IsMerge = commit.Parents.Count > 1,
                IsTip = tip,
            });
        }

        return rows;
    }

    /// <summary>
    /// 空いている列を探す。無ければ末尾に増やす。
    ///
    /// 上限に達したら**一番右の列を使い回す**。線は正しくなくなるが、
    /// 20 列も並べば目では追えないので、そこを守る意味が薄い。
    /// </summary>
    private static int FreeLane(List<string?> lanes, int maximum)
    {
        for (var i = 0; i < lanes.Count; i++)
        {
            if (lanes[i] is null)
            {
                return i;
            }
        }
        return lanes.Count < maximum ? lanes.Count : maximum - 1;
    }
}
