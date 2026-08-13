using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Media;

namespace DeepCompare.App;

/// <summary>描く文字のひとかたまり。色と地の色を持つ。</summary>
public sealed record RunSpec(string Text, IBrush Foreground, IBrush? Background = null);

/// <summary>
/// 行の中身を <see cref="TextBlock"/> へ流し込む。
///
/// **<c>Inlines</c> に直接バインドしてはいけない。**
/// <see cref="InlineCollection"/> は 1 つの TextBlock にしか属せないので、
/// 仮想化した一覧で入れ物が使い回されると、前の行から取り上げられて
/// **そこが空白になる**。実際、上下に動かした行が次々と白くなり、
/// 画面の外へ出して戻すと直る、という形で表面化した
/// （戻ってきたときに繋ぎ直されるため）。
///
/// ここでは「材料」だけを渡し、**流し込むたびに Run を作り直す。**
/// 材料は誰のものでもないので、使い回されても取り合いにならない。
/// </summary>
public static class TextRuns
{
    public static readonly AttachedProperty<IReadOnlyList<RunSpec>?> ValueProperty =
        AvaloniaProperty.RegisterAttached<TextBlock, IReadOnlyList<RunSpec>?>(
            "Value", typeof(TextRuns));

    static TextRuns()
    {
        ValueProperty.Changed.AddClassHandler<TextBlock>((block, args) =>
            Apply(block, args.NewValue as IReadOnlyList<RunSpec>));
    }

    public static void SetValue(TextBlock element, IReadOnlyList<RunSpec>? value)
        => element.SetValue(ValueProperty, value);

    public static IReadOnlyList<RunSpec>? GetValue(TextBlock element)
        => element.GetValue(ValueProperty);

    private static void Apply(TextBlock block, IReadOnlyList<RunSpec>? runs)
    {
        block.Inlines?.Clear();
        if (runs is null || runs.Count == 0)
        {
            // **空でも Text は触らない。** ここで空文字を入れると、
            // 幅の計算が Inlines と Text の両方に引きずられる。
            return;
        }

        block.Inlines ??= [];
        foreach (var run in runs)
        {
            block.Inlines.Add(new Run(run.Text)
            {
                Foreground = run.Foreground,
                Background = run.Background,
            });
        }
    }
}
