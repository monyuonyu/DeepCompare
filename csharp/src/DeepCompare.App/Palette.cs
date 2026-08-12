using Avalonia;
using Avalonia.Media;
using Avalonia.Styling;

namespace DeepCompare.App;

/// <summary>
/// 配色の引き当て。
///
/// 行の描画は C# 側で <see cref="Avalonia.Controls.Documents.Run"/> を組み立てるので、
/// XAML の <c>DynamicResource</c> が使えない。かといって色を C# にも書くと、
/// テーマ辞書と二重に持つことになり、片方だけ直し忘れる。
///
/// そこで**同じテーマ辞書を C# から引く**。値の持ち場は App.axaml 一箇所に保つ。
/// 引き当ては行ごとに何度も起きるので、テーマごとに覚えておく。
/// </summary>
public static class Palette
{
    private static readonly Dictionary<string, IBrush> Cache = new(StringComparer.Ordinal);
    private static ThemeVariant? _cachedFor;

    /// <summary>いま明るいテーマか。</summary>
    public static bool IsLight => Current == ThemeVariant.Light;

    private static ThemeVariant Current
        => Application.Current?.ActualThemeVariant ?? ThemeVariant.Light;

    /// <summary>テーマを切り替える。呼んだ側は行を作り直すこと。</summary>
    public static void Use(bool light)
    {
        if (Application.Current is { } application)
        {
            application.RequestedThemeVariant = light ? ThemeVariant.Light : ThemeVariant.Dark;
        }
        Invalidate();
    }

    /// <summary>覚えている色を捨てる。テーマが変わったときに呼ぶ。</summary>
    public static void Invalidate()
    {
        Cache.Clear();
        _cachedFor = null;
    }

    /// <summary>テーマ辞書から色を引く。見つからなければ目立つ色を返して気づけるようにする。</summary>
    public static IBrush Brush(string key)
    {
        var theme = Current;
        if (_cachedFor != theme)
        {
            Cache.Clear();
            _cachedFor = theme;
        }
        if (Cache.TryGetValue(key, out var cached))
        {
            return cached;
        }

        // 静かに既定色へ落とすと、辞書へ足し忘れた項目に気づけない。目立つ色にする。
        IBrush brush = new SolidColorBrush(Colors.Magenta);
        if (Application.Current?.Resources.TryGetResource(key, theme, out var value) == true
            && value is IBrush found)
        {
            brush = found;
        }

        Cache[key] = brush;
        return brush;
    }
}
