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
        _gap = null;
        _cachedFor = null;
    }

    private static IBrush? _gap;

    /// <summary>
    /// 「この側には対応する行が無い」ことを示す斜線。
    ///
    /// **空白で示さない。** 空白だと「中身が空の行」なのか「対応が無い」のかが
    /// 区別できない。Beyond Compare も同じ場所に斜線を敷いている。
    ///
    /// 敷き詰めなので、1 枚のタイルを作って繰り返す。行ごとに作ると数千行で
    /// 効いてくる。
    /// </summary>
    public static IBrush Gap()
    {
        if (_gap is not null)
        {
            return _gap;
        }

        var line = Brush("DividerStrong");
        var drawing = new GeometryDrawing
        {
            // 8×8 の枠に 45 度の線を 1 本。角を跨ぐ 2 本を足して、
            // タイルの継ぎ目で線が途切れないようにする。
            Geometry = StreamGeometry.Parse("M0,8 L8,0 M-2,2 L2,-2 M6,10 L10,6"),
            Pen = new Pen(line, 1),
        };
        _gap = new DrawingBrush(drawing)
        {
            TileMode = TileMode.Tile,
            SourceRect = new RelativeRect(0, 0, 8, 8, RelativeUnit.Absolute),
            DestinationRect = new RelativeRect(0, 0, 8, 8, RelativeUnit.Absolute),
            Stretch = Stretch.None,
            Opacity = 0.8,
        };
        return _gap;
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

/// <summary>
/// アイコンの図形を引く。
///
/// 図形は <c>Theme/Icons.axaml</c> が唯一の持ち場。C# 側で座標を書くと、
/// 同じ形が 2 箇所に増えて片方だけ直し忘れる。色と同じ扱いにする。
/// </summary>
public static class Icons
{
    private static readonly Dictionary<string, Geometry?> Cache = new(StringComparer.Ordinal);

    public static Geometry? Get(string key)
    {
        if (Cache.TryGetValue(key, out var cached))
        {
            return cached;
        }

        Geometry? geometry = null;
        if (Application.Current?.Resources.TryGetResource(key, null, out var value) == true)
        {
            geometry = value as Geometry;
        }

        // 見つからないときは null。図形が欠けても画面は出したい（色と違い、
        // 目立つ代替を出す方法が無いので、ここは静かに空ける）。
        Cache[key] = geometry;
        return geometry;
    }
}
