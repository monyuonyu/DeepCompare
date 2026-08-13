using System.Collections.ObjectModel;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// 起動画面に並べる「比較の種類」1 つ。
///
/// Beyond Compare の Home view と同じ形にする。大きな絵と短い名前を並べ、
/// そこへファイルを落とせばその種類で始まる。
/// </summary>
public sealed class CompareKind(string label, string hint, string iconKey, object content)
{
    public string Label { get; } = label;

    /// <summary>指したときに出す一言。</summary>
    public string Hint { get; } = hint;

    public Geometry? Icon => Icons.Get(iconKey);

    /// <summary>この種類が使う画面。**作り直さず持ち回る**ので、状態が残る。</summary>
    public object Content { get; } = content;
}

/// <summary>
/// 画面全体の入れ物。
///
/// **Beyond Compare と同じ形にする。** 起動画面で比較の種類を選び、
/// 各画面のツールバーにある家の印で起動画面へ戻る。画面の左端は
/// 比較そのもの（差分の地図）に使いたいので、種類の切り替えを左に置かない。
///
/// ただし**画面は作り直さず持ち回る**。戻ってからもう一度開いても、
/// 入力もスクロール位置も残る。BC も同じで、戻っても比較はやり直しにならない。
/// </summary>
public sealed class ShellViewModel : ViewModelBase
{
    private readonly SessionStore _settings = new();
    private bool _lightTheme;

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
        GoHomeCommand = new RelayCommand(() => { GoHome(); return Task.CompletedTask; });

        Text = new TextCompareViewModel(this);
        Folder = new FolderCompareViewModel(this);
        Structured = new StructuredCompareViewModel(this);
        Merge = new MergeViewModel(this);
        Git = new GitViewModel(this, Environment.CurrentDirectory);
        Home = new HomeViewModel(this);
        _current = Home;

        Kinds =
        [
            new CompareKind("テキスト比較", "2 つのファイルを行で突き合わせる", "IconText", Text),
            new CompareKind("フォルダー比較", "2 つのフォルダーを再帰的に比べる", "IconFolder", Folder),
            new CompareKind("構造として比較", "JSON を構造として比べる。キーの順序は差分にしない",
                "IconStructure", Structured),
            new CompareKind("3 方向マージ", "共通の元から分かれた 2 つの変更を合わせる", "IconMerge", Merge),
            new CompareKind("Git", "作業ツリーと履歴", "IconGit", Git),
        ];
    }

    public Func<string, bool, Task<string?>> PickPath { get; }

    /// <summary>書き出し先を選ばせる。（題名、既定のファイル名）を受ける。</summary>
    public Func<string, string, Task<string?>> PickSavePath { get; }

    public ObservableCollection<CompareKind> Kinds { get; }

    public TextCompareViewModel Text { get; }
    public FolderCompareViewModel Folder { get; }
    public StructuredCompareViewModel Structured { get; }
    public MergeViewModel Merge { get; }
    public GitViewModel Git { get; }
    public HomeViewModel Home { get; }

    private object _current;
    public object Current
    {
        get => _current;
        private set
        {
            if (Set(ref _current, value))
            {
                OnPropertyChanged(nameof(IsHome));
            }
        }
    }

    /// <summary>起動画面に居るか。家の印を出すかどうかに使う。</summary>
    public bool IsHome => ReferenceEquals(_current, Home);

    /// <summary>起動画面へ戻る。**状態は捨てない。**</summary>
    public void GoHome() => Current = Home;

    /// <summary>選んだ種類の画面を出す。起動画面の大きな絵から呼ばれる。</summary>
    public void Open(CompareKind kind) => Current = kind.Content;

    public System.Windows.Input.ICommand ToggleThemeCommand { get; }
    public System.Windows.Input.ICommand GoHomeCommand { get; }

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
            OnPropertyChanged(nameof(ThemeIcon));
            OnPropertyChanged(nameof(ThemeTooltip));
            ThemeChanged?.Invoke();
        }
    }

    /// <summary>いま押すと何になるか。押した先を示す方が、迷わない。</summary>
    public Geometry? ThemeIcon => Icons.Get(_lightTheme ? "IconMoon" : "IconSun");

    public string ThemeTooltip => _lightTheme ? "暗い配色にする" : "明るい配色にする";

    /// <summary>テーマが変わったときに、開いている画面が行を作り直すための知らせ。</summary>
    public event Action? ThemeChanged;

    /// <summary>初回だけ読む。呼び出し側は必ず作業スレッドから呼ぶこと。</summary>
    public Embedder GetEmbedder() => _embedder ??= Embedder.CreateFromDefaultAssets();

    private void Go(object content) => Current = content;

    // --- 画面を開く。指定があれば入れてから走らせる ---

    public void ShowText(string left, string right)
    {
        Text.LeftPath = left;
        Text.RightPath = right;
        Go(Text);
        Text.CompareCommand.Execute(null);
    }

    public void ShowFolders(string left, string right)
    {
        Folder.LeftRoot = left;
        Folder.RightRoot = right;
        Go(Folder);
        Folder.RefreshCommand.Execute(null);
    }

    public void ShowStructured(string left, string right)
    {
        Structured.LeftPath = left;
        Structured.RightPath = right;
        Go(Structured);
        if (left.Length > 0 && right.Length > 0)
        {
            Structured.CompareCommand.Execute(null);
        }
    }

    public void ShowMerge(string basePath, string left, string right)
    {
        Merge.BasePath = basePath;
        Merge.LeftPath = left;
        Merge.RightPath = right;
        Go(Merge);
        if (basePath.Length > 0 && left.Length > 0 && right.Length > 0)
        {
            Merge.MergeCommand.Execute(null);
        }
    }

    public void ShowGit(string path)
    {
        Git.Path = path;
        Go(Git);
        _ = Git.RefreshAsync();
    }

    public void ShowSaved() => Go(Home);

    /// <summary>
    /// 中身の取り出し方を差し替えたテキスト比較を出す。git のある時点の中身を
    /// 比べるときに使う。**同じテキスト比較画面を使い回す**ので、サイドバーの
    /// 「テキスト」に居ることが見た目からも分かる。
    /// </summary>
    public void ShowTextWith(
        string leftLabel, string rightLabel, Func<string, byte[]> loader,
        bool leftReadOnly, bool rightReadOnly)
    {
        Text.ContentLoader = loader;
        Text.LeftReadOnly = leftReadOnly;
        Text.RightReadOnly = rightReadOnly;
        Text.LeftPath = leftLabel;
        Text.RightPath = rightLabel;
        Go(Text);
        Text.CompareCommand.Execute(null);
    }

    /// <summary>普通のファイル比較へ戻す。読み込み方の差し替えを解く。</summary>
    public void ResetTextLoader()
    {
        Text.ContentLoader = null;
        Text.LeftReadOnly = false;
        Text.RightReadOnly = false;
    }
}
