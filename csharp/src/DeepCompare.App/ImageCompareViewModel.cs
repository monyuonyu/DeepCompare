using Avalonia.Media.Imaging;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>画像の見せ方。</summary>
public enum ImageViewMode
{
    /// <summary>左右に並べる。**既定。** 何が写っているかがまず分かる。</summary>
    SideBySide,

    /// <summary>違うところを塗った 1 枚。どこが違うかを探すとき。</summary>
    Difference,

    /// <summary>重ねて、つまみで左右を行き来する。ずれを見るとき。</summary>
    Swipe,
}

/// <summary>
/// 画像比較の画面（BC の Picture Compare に当たる）。
///
/// **画素の比較そのものは engine に置く。** ここは読み込みと見せ方だけ。
/// </summary>
public sealed class ImageCompareViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    /// <summary>自分が乗っているタブ。見出しを比較の中身に合わせて書き換える。</summary>
    public CompareTab? Tab { get; set; }

    public ShellViewModel Shell => _shell;

    public ImageCompareViewModel(ShellViewModel shell)
    {
        _shell = shell;
        CompareCommand = new RelayCommand(CompareAsync, () => !_busy);
        OpenAsBinaryCommand = new RelayCommand(() =>
        {
            _shell.ShowBinary(LeftPath, RightPath);
            return Task.CompletedTask;
        }, () => LeftPath.Length > 0 && RightPath.Length > 0);

        SetSideBySideCommand = new RelayCommand(() => Switch(ImageViewMode.SideBySide));
        SetDifferenceCommand = new RelayCommand(() => Switch(ImageViewMode.Difference));
        SetSwipeCommand = new RelayCommand(() => Switch(ImageViewMode.Swipe));

        // **段階で拡大する。** 自由な倍率だと、画素の等倍（100%・200%）に
        // ぴったり合わせるのが難しく、画素を見に来た人が困る。
        ZoomInCommand = new RelayCommand(() => { Zoom = NextZoom(up: true); return Task.CompletedTask; });
        ZoomOutCommand = new RelayCommand(() => { Zoom = NextZoom(up: false); return Task.CompletedTask; });
        ZoomResetCommand = new RelayCommand(() => { Zoom = 1; return Task.CompletedTask; });
    }

    public RelayCommand CompareCommand { get; }
    public RelayCommand OpenAsBinaryCommand { get; }
    public RelayCommand SetSideBySideCommand { get; }
    public RelayCommand SetDifferenceCommand { get; }
    public RelayCommand SetSwipeCommand { get; }
    public RelayCommand ZoomInCommand { get; }
    public RelayCommand ZoomOutCommand { get; }
    public RelayCommand ZoomResetCommand { get; }

    private Task Switch(ImageViewMode mode)
    {
        Mode = mode;
        return Task.CompletedTask;
    }

    private static readonly double[] ZoomSteps =
        [0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16];

    private double NextZoom(bool up)
    {
        if (up)
        {
            foreach (var step in ZoomSteps)
            {
                if (step > Zoom + 0.001)
                {
                    return step;
                }
            }
            return ZoomSteps[^1];
        }
        for (var i = ZoomSteps.Length - 1; i >= 0; i--)
        {
            if (ZoomSteps[i] < Zoom - 0.001)
            {
                return ZoomSteps[i];
            }
        }
        return ZoomSteps[0];
    }

    private string _leftPath = string.Empty;
    public string LeftPath
    {
        get => _leftPath;
        set => Set(ref _leftPath, value);
    }

    private string _rightPath = string.Empty;
    public string RightPath
    {
        get => _rightPath;
        set => Set(ref _rightPath, value);
    }

    private Bitmap? _left;
    public Bitmap? Left
    {
        get => _left;
        private set => Set(ref _left, value);
    }

    private Bitmap? _right;
    public Bitmap? Right
    {
        get => _right;
        private set => Set(ref _right, value);
    }

    private Bitmap? _difference;
    public Bitmap? Difference
    {
        get => _difference;
        private set => Set(ref _difference, value);
    }

    private ImageViewMode _mode = ImageViewMode.SideBySide;
    public ImageViewMode Mode
    {
        get => _mode;
        set
        {
            if (Set(ref _mode, value))
            {
                OnPropertyChanged(nameof(IsSideBySide));
                OnPropertyChanged(nameof(IsDifference));
                OnPropertyChanged(nameof(IsSwipe));
            }
        }
    }

    public bool IsSideBySide => Mode == ImageViewMode.SideBySide;
    public bool IsDifference => Mode == ImageViewMode.Difference;
    public bool IsSwipe => Mode == ImageViewMode.Swipe;

    /// <summary>
    /// 重ね表示の境目（0〜1）。
    ///
    /// 左からこの割合までを左の画像、残りを右の画像で見せる。**ずれを見るのに
    /// 一番早い。** 並べて見比べても、数画素のずれは目で追えない。
    /// </summary>
    private double _swipe = 0.5;
    public double Swipe
    {
        get => _swipe;
        set
        {
            if (Set(ref _swipe, Math.Clamp(value, 0, 1)))
            {
                OnPropertyChanged(nameof(SwipeWidth));
                OnPropertyChanged(nameof(SwipeLineMargin));
            }
        }
    }

    /// <summary>重ね表示で、左の画像を見せる幅。</summary>
    public double SwipeWidth => ImageWidth * Swipe;

    /// <summary>境目の線の位置。左からの余白で置く。</summary>
    public Avalonia.Thickness SwipeLineMargin => new(SwipeWidth, 0, 0, 0);

    /// <summary>両方を覆う大きさ。重ね表示の台紙に使う。</summary>
    public double ImageWidth => _comparison?.Width ?? 0;
    public double ImageHeight => _comparison?.Height ?? 0;

    private double _zoom = 1;

    /// <summary>拡大率。**画素の等倍を保てるように、段階で持つ。**</summary>
    public double Zoom
    {
        get => _zoom;
        set
        {
            if (Set(ref _zoom, Math.Clamp(value, 0.1, 16)))
            {
                OnPropertyChanged(nameof(ZoomText));
            }
        }
    }

    public string ZoomText => $"{Zoom * 100:0}%";

    private int _tolerance = 8;

    /// <summary>
    /// これ以下の差は「同じ」とみなす。
    ///
    /// 0 にすると、JPEG を保存し直しただけで全画素が違うと出る。
    /// </summary>
    public int Tolerance
    {
        get => _tolerance;
        set
        {
            if (Set(ref _tolerance, Math.Clamp(value, 0, 255)) && _comparison is not null)
            {
                _ = CompareAsync();
            }
        }
    }

    private ImageComparison? _comparison;

    private string _summary = string.Empty;
    public string Summary
    {
        get => _summary;
        private set => Set(ref _summary, value);
    }

    private string _message = string.Empty;
    public string Message
    {
        get => _message;
        private set => Set(ref _message, value);
    }

    private string _sizeText = string.Empty;

    /// <summary>大きさ。違えばそれ自体が一番大きな差分なので、目立つ場所に出す。</summary>
    public string SizeText
    {
        get => _sizeText;
        private set => Set(ref _sizeText, value);
    }

    public bool SizesDiffer { get; private set; }

    /// <summary>大きさが違えば色を変える。それ自体が一番大きな差分。</summary>
    public Avalonia.Media.IBrush SizeBrush =>
        Palette.Brush(SizesDiffer ? "FgWarning" : "FgDim");

    private bool _busy;
    public bool Busy
    {
        get => _busy;
        private set { if (Set(ref _busy, value)) { CompareCommand.Raise(); } }
    }

    private async Task CompareAsync()
    {
        if (LeftPath.Length == 0 || RightPath.Length == 0)
        {
            Message = "比べる 2 つの画像を指定してください。";
            return;
        }

        Busy = true;
        Message = string.Empty;

        try
        {
            var (leftPath, rightPath, tolerance) = (LeftPath, RightPath, Tolerance);

            // 読み込みと比較は作業スレッドで。大きい画像だと画面が止まる。
            var work = await Task.Run(() =>
            {
                var left = ImageLoader.Load(leftPath);
                var right = ImageLoader.Load(rightPath);
                var comparison = ImageCompare.Compare(left, right,
                    new ImageCompareOptions { Tolerance = tolerance });
                return (left, right, comparison, highlight: ImageCompare.Highlight(left, comparison));
            });

            _comparison = work.comparison;

            // **画像は画面のスレッドで作る。** Bitmap は描画基盤に触るので、
            // 作業スレッドで作ると環境によって落ちる。
            Left = ImageLoader.ToBitmap(work.left);
            Right = ImageLoader.ToBitmap(work.right);
            Difference = ImageLoader.ToBitmap(work.highlight);

            SizesDiffer = work.left.Width != work.right.Width
                || work.left.Height != work.right.Height;
            OnPropertyChanged(nameof(SizesDiffer));
            OnPropertyChanged(nameof(SizeBrush));
            SizeText = SizesDiffer
                ? $"{work.left.Width}×{work.left.Height} と {work.right.Width}×{work.right.Height}"
                : $"{work.left.Width}×{work.left.Height}";

            OnPropertyChanged(nameof(ImageWidth));
            OnPropertyChanged(nameof(ImageHeight));
            OnPropertyChanged(nameof(SwipeWidth));
            OnPropertyChanged(nameof(SwipeLineMargin));

            Summary = work.comparison.IsIdentical
                ? "完全に同じです。"
                : work.comparison.LooksSame
                    ? $"見た目は同じです（しきい値 {tolerance} の内側の差だけ）。"
                    : $"違う画素 {work.comparison.DifferentCount:N0}"
                      + (work.comparison.MissingCount > 0
                          ? $" / 片方だけ {work.comparison.MissingCount:N0}"
                          : string.Empty)
                      + $"（{work.comparison.DifferenceRatio:P2}）";

            // 違いがあれば、そこを見せる側へ切り替える。**探させない。**
            if (!work.comparison.LooksSame && Mode == ImageViewMode.SideBySide)
            {
                Mode = ImageViewMode.Difference;
            }

            if (Tab is { } tab)
            {
                tab.Title = System.IO.Path.GetFileName(leftPath) + "（画像）";
            }
        }
        catch (Exception error) when (error is IOException or NotSupportedException
                                        or InvalidOperationException or UnauthorizedAccessException)
        {
            Message = error.Message;
            Left = Right = Difference = null;
            _comparison = null;
        }
        finally
        {
            Busy = false;
        }
    }
}
