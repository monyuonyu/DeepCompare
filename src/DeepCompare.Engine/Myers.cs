namespace DeepCompare.Engine;

public enum DiffKind
{
    Equal,
    Delete,
    Insert,
    Replace,
}

/// <summary>差分の一区間。両側の半開区間で表す。</summary>
public readonly record struct DiffOp(DiffKind Kind, int OldStart, int OldLength, int NewStart, int NewLength);

/// <summary>
/// 行単位の差分（Myers のアルゴリズム）。
///
/// 自前で持っているのは、後段の区間分割が「どこが動かなかったか」を正確に必要とするため。
/// 既製品の出力形式に合わせて変換するより、必要な形をそのまま出す方が取り違えが起きない。
/// </summary>
public static class Myers
{
    /// <summary>
    /// 編集距離がこれを超えたら精密な差分を諦める。
    ///
    /// Myers は経路の記録に O(D * (N+M)) の領域を使うので、まったく似ていない
    /// 巨大なファイル同士では現実的でなくなる。その場合は「全体が 1 つの変更区間」として
    /// 返す。後段はそれを塊として扱えるので、結果は粗くなるが破綻はしない。
    /// </summary>
    public const int MaxEditDistance = 20_000;

    public static List<DiffOp> Compute(IReadOnlyList<string> left, IReadOnlyList<string> right)
    {
        // 前後の一致部分を先に削る。実際の改修では大半がここで落ちるので、
        // Myers に渡る問題が小さくなる。
        var prefix = 0;
        var maxPrefix = Math.Min(left.Count, right.Count);
        while (prefix < maxPrefix && left[prefix] == right[prefix])
        {
            prefix++;
        }

        var suffix = 0;
        var maxSuffix = Math.Min(left.Count, right.Count) - prefix;
        while (suffix < maxSuffix
               && left[left.Count - 1 - suffix] == right[right.Count - 1 - suffix])
        {
            suffix++;
        }

        var ops = new List<DiffOp>();
        if (prefix > 0)
        {
            ops.Add(new DiffOp(DiffKind.Equal, 0, prefix, 0, prefix));
        }

        var midLeftLength = left.Count - prefix - suffix;
        var midRightLength = right.Count - prefix - suffix;
        if (midLeftLength > 0 || midRightLength > 0)
        {
            ops.AddRange(DiffMiddle(left, right, prefix, midLeftLength, midRightLength));
        }

        if (suffix > 0)
        {
            ops.Add(new DiffOp(DiffKind.Equal, left.Count - suffix, suffix, right.Count - suffix, suffix));
        }
        return Coalesce(ops);
    }

    private static List<DiffOp> DiffMiddle(
        IReadOnlyList<string> left,
        IReadOnlyList<string> right,
        int offset,
        int n,
        int m)
    {
        // 片側が空なら差分は自明。
        if (n == 0)
        {
            return [new DiffOp(DiffKind.Insert, offset, 0, offset, m)];
        }
        if (m == 0)
        {
            return [new DiffOp(DiffKind.Delete, offset, n, offset, 0)];
        }

        var max = n + m;
        if (max > MaxEditDistance)
        {
            // 諦める場合も、両側を 1 つの変更区間として返せば後段は成立する。
            return [new DiffOp(DiffKind.Replace, offset, n, offset, m)];
        }

        var size = 2 * max + 1;
        var v = new int[size];
        var trace = new List<int[]>(Math.Min(max, 64));

        var found = -1;
        for (var d = 0; d <= max; d++)
        {
            trace.Add((int[])v.Clone());
            for (var k = -d; k <= d; k += 2)
            {
                int x;
                if (k == -d || (k != d && v[max + k - 1] < v[max + k + 1]))
                {
                    x = v[max + k + 1];
                }
                else
                {
                    x = v[max + k - 1] + 1;
                }
                var y = x - k;
                while (x < n && y < m && left[offset + x] == right[offset + y])
                {
                    x++;
                    y++;
                }
                v[max + k] = x;
                if (x >= n && y >= m)
                {
                    found = d;
                    break;
                }
            }
            if (found >= 0)
            {
                break;
            }
        }

        return Backtrack(trace, found, max, n, m, offset);
    }

    /// <summary>記録した経路を終点から辿り、区間の列を作る。</summary>
    private static List<DiffOp> Backtrack(List<int[]> trace, int found, int max, int n, int m, int offset)
    {
        var ops = new List<DiffOp>();
        var x = n;
        var y = m;

        for (var d = found; d > 0; d--)
        {
            var v = trace[d];
            var k = x - y;
            int prevK;
            if (k == -d || (k != d && v[max + k - 1] < v[max + k + 1]))
            {
                prevK = k + 1;
            }
            else
            {
                prevK = k - 1;
            }
            var prevX = v[max + prevK];
            var prevY = prevX - prevK;

            // 斜めに進んだぶん（＝一致していた行）を先に出す。
            while (x > prevX && y > prevY)
            {
                x--;
                y--;
                ops.Add(new DiffOp(DiffKind.Equal, offset + x, 1, offset + y, 1));
            }

            if (x > prevX)
            {
                x--;
                ops.Add(new DiffOp(DiffKind.Delete, offset + x, 1, offset + y, 0));
            }
            else if (y > prevY)
            {
                y--;
                ops.Add(new DiffOp(DiffKind.Insert, offset + x, 0, offset + y, 1));
            }
        }

        // 先頭に残った一致部分。
        while (x > 0 && y > 0)
        {
            x--;
            y--;
            ops.Add(new DiffOp(DiffKind.Equal, offset + x, 1, offset + y, 1));
        }

        ops.Reverse();
        return ops;
    }

    /// <summary>隣り合う同種の区間をまとめ、削除と挿入が接していれば Replace にする。</summary>
    private static List<DiffOp> Coalesce(List<DiffOp> ops)
    {
        var merged = new List<DiffOp>(ops.Count);
        foreach (var op in ops)
        {
            if (op.OldLength == 0 && op.NewLength == 0)
            {
                continue;
            }
            if (merged.Count > 0)
            {
                var last = merged[^1];
                if (last.Kind == op.Kind)
                {
                    merged[^1] = last with
                    {
                        OldLength = last.OldLength + op.OldLength,
                        NewLength = last.NewLength + op.NewLength,
                    };
                    continue;
                }
                // 削除の直後の挿入は、置換として 1 つにする。分けたままだと
                // 「消えた行」と「増えた行」を意味的に対応付ける機会が失われる。
                if (last.Kind == DiffKind.Delete && op.Kind == DiffKind.Insert)
                {
                    merged[^1] = new DiffOp(
                        DiffKind.Replace,
                        last.OldStart,
                        last.OldLength,
                        op.NewStart,
                        op.NewLength);
                    continue;
                }
                if (last.Kind == DiffKind.Replace && op.Kind is DiffKind.Delete or DiffKind.Insert)
                {
                    merged[^1] = last with
                    {
                        OldLength = last.OldLength + op.OldLength,
                        NewLength = last.NewLength + op.NewLength,
                    };
                    continue;
                }
            }
            merged.Add(op);
        }
        return merged;
    }
}
