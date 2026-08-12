using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.VisualTree;

namespace DeepCompare.App.Views;

public partial class TextCompareView : UserControl
{
    private ScrollViewer? _scroll;

    public TextCompareView()
    {
        AvaloniaXamlLoader.Load(this);

        // 本文列の幅は「内容に必要な幅」と「画面を分け合った幅」の大きい方にする。
        // 画面幅は表示側にしか分からないので、変わるたびに知らせる。
        this.GetObservable(BoundsProperty).Subscribe(new AnonymousObserver<Rect>(bounds =>
        {
            if (DataContext is TextCompareViewModel model)
            {
                model.ViewportWidth = bounds.Width;
            }
        }));

        // 差分の地図に「いま見えている範囲」を出すために、スクロール位置を拾う。
        // ListBox の中の ScrollViewer は、テンプレートが当たるまで存在しない。
        AttachedToVisualTree += (_, _) => HookScroll();
    }

    private void HookScroll()
    {
        if (_scroll is not null)
        {
            return;
        }
        _scroll = this.GetVisualDescendants().OfType<ScrollViewer>().FirstOrDefault();
        if (_scroll is null)
        {
            return;
        }

        void Update()
        {
            if (DataContext is not TextCompareViewModel model)
            {
                return;
            }
            // Extent は中身の全体、Viewport は見えている分。差が 0 なら全部見えている。
            var scrollable = _scroll.Extent.Height - _scroll.Viewport.Height;
            model.MapViewStart = scrollable > 0 ? _scroll.Offset.Y / _scroll.Extent.Height : 0;
            model.MapViewSize = _scroll.Extent.Height > 0
                ? Math.Min(1, _scroll.Viewport.Height / _scroll.Extent.Height)
                : 1;
        }

        _scroll.GetObservable(ScrollViewer.OffsetProperty)
            .Subscribe(new AnonymousObserver<Vector>(_ => Update()));
        // 行が入れ替わると中身の高さが変わる。そのときも出し直す。
        _scroll.GetObservable(ScrollViewer.ExtentProperty)
            .Subscribe(new AnonymousObserver<Size>(_ => Update()));
    }

    /// <summary>
    /// 値が来たときだけ何かする観測者。Avalonia は System.Reactive を要求しないので、
    /// この程度は自前で持つ方が依存を増やさずに済む。
    /// </summary>
    private sealed class AnonymousObserver<T>(Action<T> onNext) : IObserver<T>
    {
        public void OnCompleted() { }

        public void OnError(Exception error) { }

        public void OnNext(T value) => onNext(value);
    }
}
