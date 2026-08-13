using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 語の埋め込みを int8 のまま持つ経路。
///
/// **黙って間違えないことを確かめる。** 展開したものと同じ値が出ること、
/// そして int8 のまま持っているのに f32 として読もうとしたら断ること。
/// 断らずに何かを返すと、桁が 100 倍違う値で計算が進む。
/// </summary>
public class TensorTests
{
    [Fact]
    public void f32_の行はそのまま取れる()
    {
        var tensor = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        Assert.Equal([3f, 4f], tensor.Row(1).ToArray());
        Assert.False(tensor.IsQuantized);
    }

    [Fact]
    public void int8_の行は_scale_を掛けて取れる()
    {
        // 行 0 の scale は 0.5、行 1 は 2.0。
        var tensor = new Tensor([2, 4, 3, -5], [0.5f, 2f], [2, 2]);
        Assert.True(tensor.IsQuantized);

        var row = new float[2];
        tensor.CopyRowTo(0, row);
        Assert.Equal([1f, 2f], row);

        tensor.CopyRowTo(1, row);
        Assert.Equal([6f, -10f], row);
    }

    [Fact]
    public void int8_のまま持つ物を_f32_として読もうとしたら断る()
    {
        var tensor = new Tensor([1, 2], [1f], [1, 2]);

        // **黙って別の物を返さない。** 量子化された生の値を f32 として
        // 返すと、桁が 100 倍違うまま計算が進み、結果だけが静かにおかしくなる。
        Assert.Throws<InvalidOperationException>(() => tensor.Row(0).ToArray());
        Assert.Throws<InvalidOperationException>(() => tensor.Data);
    }

    [Fact]
    public void どちらの持ち方でも_CopyRowTo_は同じ値を返す()
    {
        // 同じ中身を f32 と int8 の両方で作り、突き合わせる。
        var scales = new[] { 0.25f, 0.5f };
        sbyte[] quantized = [4, -8, 6, 10];
        var expanded = new float[4];
        for (var r = 0; r < 2; r++)
        {
            for (var c = 0; c < 2; c++)
            {
                expanded[r * 2 + c] = quantized[r * 2 + c] * scales[r];
            }
        }

        var asF32 = new Tensor(expanded, [2, 2]);
        var asInt8 = new Tensor(quantized, scales, [2, 2]);

        for (var r = 0; r < 2; r++)
        {
            var a = new float[2];
            var b = new float[2];
            asF32.CopyRowTo(r, a);
            asInt8.CopyRowTo(r, b);
            Assert.Equal(a, b);
        }
    }

    [Fact]
    public void 行と列の数を形から読む()
    {
        Assert.Equal(3, new Tensor(new float[6], [3, 2]).Rows);
        Assert.Equal(2, new Tensor(new float[6], [3, 2]).Cols);

        // 1 次元は 1 列として扱う（バイアスなど）。
        Assert.Equal(1, new Tensor(new float[4], [4]).Cols);
    }
}
