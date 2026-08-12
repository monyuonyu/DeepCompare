using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// 画面全体の入れ物。どの画面を出すかだけを持ち、比較そのものには関わらない。
///
/// Beyond Compare や WinMerge と同じく、まず「何を比べるか」を選ぶ画面から入り、
/// フォルダー比較の一覧から個々のファイルのテキスト比較へ降りていく形にする。
/// </summary>
public sealed class ShellViewModel : ViewModelBase
{
    private object _current;

    /// <summary>
    /// モデルは読み込みに数秒かかるうえ 90MB 近い実体を持つので、
    /// 画面を移るたびに作り直さず、ここで一度だけ用意して使い回す。
    /// </summary>
    private Embedder? _embedder;

    public ShellViewModel(
        Func<string, bool, Task<string?>> pickPath,
        Func<string, string, Task<string?>> pickSavePath)
    {
        PickPath = pickPath;
        PickSavePath = pickSavePath;
        _lightTheme = _settings.LoadLightTheme();
        Palette.Use(_lightTheme);
        ToggleThemeCommand = new RelayCommand(() => { LightTheme = !LightTheme; return Task.CompletedTask; });
        Home = new HomeViewModel(this);
        _current = Home;
    }

    public Func<string, bool, Task<string?>> PickPath { get; }

    /// <summary>書き出し先を選ばせる。（題名、既定のファイル名）を受ける。</summary>
    public Func<string, string, Task<string?>> PickSavePath { get; }

    private readonly DeepCompare.Engine.SessionStore _settings = new();
    private bool _lightTheme;

    public System.Windows.Input.ICommand ToggleThemeCommand { get; }

    /// <summary>
    /// 明るいテーマを使うか。切り替えたら、色を持っている行を作り直す必要がある。
    /// C# 側で組み立てた文字色は、テーマが変わっても勝手には追随しない。
    /// </summary>
    public bool LightTheme
    {
        get => _lightTheme;
        set
        {
            if (!Set(ref _lightTheme, value))
            {
                return;
            }
            Palette.Use(value);
            _settings.SaveLightTheme(value);
            OnPropertyChanged(nameof(ThemeButtonText));
            ThemeChanged?.Invoke();
        }
    }

    public string ThemeButtonText => _lightTheme ? "ダークにする" : "ライトにする";

    /// <summary>テーマが変わったときに、開いている画面が行を作り直すための知らせ。</summary>
    public event Action? ThemeChanged;

    public HomeViewModel Home { get; }

    public object Current
    {
        get => _current;
        private set
        {
            if (Set(ref _current, value))
            {
                OnPropertyChanged(nameof(CanGoHome));
            }
        }
    }

    public bool CanGoHome => Current is not HomeViewModel;

    /// <summary>初回だけ読む。呼び出し側は必ず作業スレッドから呼ぶこと。</summary>
    public Embedder GetEmbedder() => _embedder ??= Embedder.CreateFromDefaultAssets();

    public void GoHome() => Current = Home;

    public void ShowFolders(string left, string right)
        => Current = new FolderCompareViewModel(this, left, right);

    /// <summary>テキスト比較を開く。フォルダー一覧からも起動画面からも呼ばれる。</summary>
    public void ShowText(string left, string right, object? back = null)
    {
        var model = new TextCompareViewModel(this, back);
        model.LeftPath = left;
        model.RightPath = right;
        Current = model;
        model.CompareCommand.Execute(null);
    }

    public void ShowEmptyText() => Current = new TextCompareViewModel(this, null);

    /// <summary>構造として比較する画面を開く。</summary>
    public void ShowStructured(string left, string right, object? back = null)
    {
        var model = new StructuredCompareViewModel(this, back)
        {
            LeftPath = left,
            RightPath = right,
        };
        Current = model;
        if (left.Length > 0 && right.Length > 0)
        {
            model.CompareCommand.Execute(null);
        }
    }

    /// <summary>フォルダー一覧など、元いた画面へ戻る。</summary>
    public void GoBack(object? target) => Current = target ?? Home;
}
