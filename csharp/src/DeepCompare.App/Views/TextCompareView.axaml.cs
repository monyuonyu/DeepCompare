using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace DeepCompare.App.Views;

public partial class TextCompareView : UserControl
{
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
