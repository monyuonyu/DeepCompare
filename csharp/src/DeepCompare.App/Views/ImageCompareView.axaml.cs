using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace DeepCompare.App.Views;

public partial class ImageCompareView : UserControl
{
    public ImageCompareView()
    {
        AvaloniaXamlLoader.Load(this);

        // **拡大しても画素をぼかさない。** 画素を見に来ているのに補間しては
        // 意味がない。Avalonia 12 では XAML の Setter から指定できなくなった
        // （RenderOptions が構造体にまとめられ、添付プロパティの
        // BitmapInterpolationModeProperty が無くなった）ので、ここで設定する。
        RenderOptions.SetBitmapInterpolationMode(this, BitmapInterpolationMode.None);
    }
}
