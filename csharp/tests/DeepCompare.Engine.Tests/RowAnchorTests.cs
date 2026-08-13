using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class RowAnchorTests
{
    private static Row R(int? left, int? right) => new(left, right, null, [], []);

    [Fact]
    public void 左の行番号で位置を探す()
    {
        // 段階 2 で行数が変わっても、元の行番号は動かない。
        List<Row> rows = [R(0, 0), R(1, null), R(null, 1), R(2, 2)];

        Assert.Equal(3, RowAnchor.Find(rows, (2, null)));
        Assert.Equal(1, RowAnchor.Find(rows, (1, null)));
    }

    [Fact]
    public void 左が消えていれば右で探す()
    {
        // 段階 2 で対応の付き方が変わり、左の行が別の行と組んだ場合。
        List<Row> rows = [R(0, 0), R(null, 5), R(9, 9)];

        Assert.Equal(1, RowAnchor.Find(rows, (99, 5)));
    }

    [Fact]
    public void 左を先に全部見てから右を見る()
    {
        // **1 回の走査で「どちらかが合えばよい」とすると、左がもっと近くに
        // あるのに遠くの右に当たって飛ぶ。**
        List<Row> rows = [R(null, 7), R(3, 100)];

        // 左 3 は 1 番目、右 7 は 0 番目。左を優先するので 1 が返る。
        Assert.Equal(1, RowAnchor.Find(rows, (3, 7)));
    }

    [Fact]
    public void 見つからなければ諦める()
    {
        List<Row> rows = [R(0, 0)];

        Assert.Equal(-1, RowAnchor.Find(rows, (99, 99)));
        Assert.Equal(-1, RowAnchor.Find(rows, (null, null)));
        Assert.Equal(-1, RowAnchor.Find([], (0, 0)));
    }
}
